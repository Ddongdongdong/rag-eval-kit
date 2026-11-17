"""v0.2 tests: hard QA set, ablation, LLM seam, answer_fn hook."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag.chunker import chunk_document
from rag.evaluator import evaluate_dataset
from rag.llm import LLMClient
from rag.retriever import TFIDFRetriever
from scripts.ablation import evaluate_all, load_docs, load_qa, write_report

BASE_DIR = Path(__file__).resolve().parent.parent
HARD_QA = BASE_DIR / "data" / "sample_qa_hard.jsonl"


def _load_hard() -> list[dict]:
    return [json.loads(line) for line in HARD_QA.read_text(encoding="utf-8").splitlines() if line.strip()]


def _heading_retriever() -> TFIDFRetriever:
    chunks: list[dict] = []
    for doc in load_docs():
        chunks.extend(chunk_document(doc["id"], doc["title"], doc["text"], strategy="heading"))
    return TFIDFRetriever(chunks)


def test_hard_qa_set_loads():
    qa = _load_hard()
    assert len(qa) == 12
    kinds = [item.get("kind") for item in qa]
    assert kinds.count("paraphrase") == 6
    assert kinds.count("cross-doc") == 6
    doc_ids = {p.stem for p in (BASE_DIR / "data" / "sample_docs").glob("*.md")}
    for item in qa:
        assert item["question"].strip()
        assert item["gold_answer"].strip()
        assert item["gold_doc_ids"]
        assert set(item["gold_doc_ids"]) <= doc_ids
        if item["kind"] == "cross-doc":
            assert len(item["gold_doc_ids"]) == 2


def test_hard_set_is_harder_at_k1_than_easy_set():
    retriever = _heading_retriever()
    easy = load_qa(BASE_DIR / "data" / "sample_qa.jsonl")
    hard = _load_hard()
    easy_r1 = evaluate_dataset(easy, retriever, top_k=1)["metrics"]["recall_at_k"]
    hard_r1 = evaluate_dataset(hard, retriever, top_k=1)["metrics"]["recall_at_k"]
    assert hard_r1 < easy_r1
    # ...but top-5 still saturates on this tiny corpus — the honest
    # signal is at k=1, not k=5 (see reports/ablation_report.md).
    assert evaluate_dataset(hard, retriever, top_k=5)["metrics"]["recall_at_k"] == 1.0


def test_ablation_produces_report(tmp_path):
    docs = load_docs()
    qa_sets = {
        "easy": load_qa(BASE_DIR / "data" / "sample_qa.jsonl"),
        "hard": _load_hard(),
    }
    data = evaluate_all(docs, qa_sets, ks=(1, 5))
    assert data["chunk_counts"]["heading"] == 22
    assert data["chunk_counts"]["fixed"] > 0
    assert ("heading", "hard", 1) in data["results"]
    assert ("fixed", "easy", 5) in data["results"]
    # write_report writes next to the repo reports/ dir; check content
    # by pointing it at the real path and reading it back.
    report = write_report(data, (1, 5), n_docs=len(docs))
    text = report.read_text(encoding="utf-8")
    assert "fixed" in text and "heading" in text
    assert "Recall@k curves" in text
    assert "Conclusion" in text


def test_llm_disabled_raises(monkeypatch):
    for var in ("RAG_EVAL_LLM", "RAG_EVAL_LLM_BASE_URL", "RAG_EVAL_LLM_API_KEY", "RAG_EVAL_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    client = LLMClient()
    assert client.enabled is False
    with pytest.raises(RuntimeError, match="RAG_EVAL_LLM=1"):
        client.generate("hello", context="some context")


def test_llm_enabled_without_config_raises(monkeypatch):
    monkeypatch.setenv("RAG_EVAL_LLM", "1")
    for var in ("RAG_EVAL_LLM_BASE_URL", "RAG_EVAL_LLM_API_KEY", "RAG_EVAL_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="missing environment variable"):
        LLMClient().generate("hello", context="some context")


def test_evaluate_dataset_with_answer_fn():
    docs = [
        ("returns", "Returns", "Customers can return a product within 30 days of delivery. Refunds are processed within 5 working days."),
        ("shipping", "Shipping", "Standard shipping takes 3 to 5 working days. Express shipping takes 1 to 2 working days."),
    ]
    chunks = []
    for doc_id, title, text in docs:
        chunks.extend(chunk_document(doc_id, title, text, strategy="fixed"))
    retriever = TFIDFRetriever(chunks)
    qa = [
        {"question": "How long does a refund take?", "gold_answer": "Refunds are processed within 5 working days.", "gold_doc_ids": ["returns"]},
    ]
    # A perfect answer_fn (e.g. an ideal LLM) must score coverage 1.0.
    result = evaluate_dataset(qa, retriever, top_k=5, answer_fn=lambda hits, q: "Refunds are processed within 5 working days.")
    assert result["metrics"]["answer_coverage"] == 1.0
    assert result["metrics"]["recall_at_k"] == 1.0


def test_run_demo_llm_mode_fails_without_key():
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_EVAL_LLM")}
    proc = subprocess.run(
        [sys.executable, "scripts/run_demo.py", "--mode", "llm"],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "RAG_EVAL_LLM" in (proc.stderr + proc.stdout)
    # Must fail loudly, never silently fall back to extractive mode.
    assert "report:" not in (proc.stderr + proc.stdout)
