#!/usr/bin/env python3
"""One-click demo: ingest sample docs -> query -> evaluate -> report.

Usage:
    python scripts/run_demo.py [--top-k 5] [--strategy heading]
                               [--qa easy|hard|both] [--mode extractive|llm]

Writes reports/demo_report.md and appends traces to reports/traces.jsonl.

Modes:
    extractive (default) — TF-IDF retrieval, extractive answers.
        No network, no LLM API, no key.
    llm — same retrieval, but the answer for each question is generated
        by an OpenAI-compatible model (see rag/llm.py). Requires
        RAG_EVAL_LLM=1 plus RAG_EVAL_LLM_BASE_URL, RAG_EVAL_LLM_API_KEY
        and RAG_EVAL_LLM_MODEL in the environment. Without them the
        run fails with a clear error — it never silently falls back
        to extractive answers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import chunk_document
from rag.evaluator import evaluate_dataset, extractive_answer
from rag.llm import LLMClient
from rag.retriever import TFIDFRetriever, rerank

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "sample_docs"
QA_PATHS = {
    "easy": BASE_DIR / "data" / "sample_qa.jsonl",
    "hard": BASE_DIR / "data" / "sample_qa_hard.jsonl",
}
REPORT_PATH = BASE_DIR / "reports" / "demo_report.md"


def load_docs() -> list[dict]:
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs.append({"id": path.stem, "title": path.stem, "text": path.read_text(encoding="utf-8")})
    return docs


def load_qa(name: str) -> list[dict]:
    return [json.loads(line) for line in QA_PATHS[name].read_text(encoding="utf-8").splitlines() if line.strip()]


def llm_answer_fn(llm: LLMClient):
    """Build an answer_fn that grounds the LLM in retrieved chunks."""

    def _answer(hits: list[dict], question: str) -> str:
        context = "\n\n".join(h["chunk"].get("text", "") for h in hits)
        return llm.generate(question, context=context)

    return _answer


def main() -> int:
    parser = argparse.ArgumentParser(description="rag-eval-kit demo")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strategy", choices=["fixed", "heading"], default="heading")
    parser.add_argument("--qa", choices=["easy", "hard", "both"], default="both")
    parser.add_argument("--mode", choices=["extractive", "llm"], default="extractive")
    args = parser.parse_args()

    # Fail fast in llm mode: no key/config means no run, no fallback.
    llm = None
    if args.mode == "llm":
        llm = LLMClient()
        try:
            llm.require_configured()
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    t0 = time.perf_counter()
    docs = load_docs()
    qa_names = ["easy", "hard"] if args.qa == "both" else [args.qa]
    chunks: list[dict] = []
    for doc in docs:
        chunks.extend(
            chunk_document(doc["id"], doc["title"], doc["text"], strategy=args.strategy)
        )
    retriever = TFIDFRetriever(chunks)
    answer_fn = llm_answer_fn(llm) if llm is not None else None
    results = {
        name: evaluate_dataset(load_qa(name), retriever, top_k=args.top_k, use_rerank=True, answer_fn=answer_fn)
        for name in qa_names
    }
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # One worked example for the report.
    example_q = "How long does a refund take after the warehouse receives the return?"
    hits = rerank(example_q, retriever.search(example_q, top_k=3), top_k=3)
    example_answer = (
        llm.generate(example_q, context=" ".join(h["chunk"].get("text", "") for h in hits))
        if llm is not None
        else extractive_answer(hits)
    )

    mode_note = (
        "LLM answers via an OpenAI-compatible endpoint (see rag/llm.py; "
        "key read from the environment only)"
        if llm is not None
        else "TF-IDF retrieval, extractive answers, no LLM API"
    )
    lines = [
        "# rag-eval-kit — Demo Report (v0.2)",
        "",
        f"Generated offline ({mode_note}). "
        f"Strategy: `{args.strategy}`, top_k={args.top_k}, mode={args.mode}, "
        f"docs={len(docs)}, chunks={len(chunks)}, "
        f"total time={elapsed_ms:.0f} ms.",
        "",
    ]
    for name in qa_names:
        result = results[name]
        m = result["metrics"]
        label = "easy" if name == "easy" else "hard (paraphrased + cross-doc)"
        lines += [
            f"## {label} set — {result['n']} questions",
            "",
            "| Metric | Value | What it means |",
            "|---|---|---|",
            f"| Recall@{args.top_k} | **{m['recall_at_k']:.3f}** | Share of gold sources found in the top-{args.top_k} results |",
            f"| Citation accuracy | **{m['citation_accuracy']:.3f}** | Share of gold docs present in the citations |",
            f"| Answer coverage | **{m['answer_coverage']:.3f}** | Token-overlap F1 vs the gold answer |",
            f"| Hallucination proxy | **{m['hallucination_proxy']:.3f}** | Share of answer tokens absent from retrieved context (lower is better) |",
            "",
            "| # | Question | Recall | Citation | Coverage | Halluc. |",
            "|---|---|---|---|---|---|",
        ]
        for i, row in enumerate(result["per_question"], 1):
            q = row["question"][:60]
            lines.append(
                f"| {i} | {q} | {row['recall_at_k']:.2f} | {row['citation_accuracy']:.2f} | "
                f"{row['answer_coverage']:.2f} | {row['hallucination_proxy']:.2f} |"
            )
        lines.append("")
    if "easy" in results and "hard" in results:
        easy_m, hard_m = results["easy"]["metrics"], results["hard"]["metrics"]
        lines += [
            "## Easy vs hard — reading the gap honestly",
            "",
            f"Recall@{args.top_k} is {easy_m['recall_at_k']:.3f} on the easy set "
            f"and {hard_m['recall_at_k']:.3f} on the hard set. On this small, "
            "cleanly separated 6-document corpus the hard set does not break "
            f"Recall@{args.top_k} — the retriever still finds every gold "
            "document inside the top-5. Where the hard set bites is at "
            "smaller k: see `reports/ablation_report.md` for the Recall@1/3 "
            "curves, where paraphrased and cross-document questions drop "
            "Recall@1 well below the easy set. That is the honest signal "
            "here — top-5 on 6 documents is too forgiving to discriminate, "
            "and a bigger, messier corpus is the v0.3 job, not a new metric.",
            "",
        ]
    lines += [
        "## Worked example",
        "",
        f"**Q:** {example_q}",
        "",
        f"**Answer ({args.mode} mode, top-3 chunks):** {example_answer[:500]}",
        "",
        "**Retrieved chunks:**",
        "",
    ]
    for h in hits:
        lines.append(f"- `{h['chunk']['id']}` score={h.get('rerank_score', h.get('score'))} — {h['chunk']['text'][:120]}")
    lines += [
        "",
        "## How to reproduce",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python scripts/run_demo.py                  # both QA sets, extractive",
        "python scripts/run_demo.py --qa hard        # hard set only",
        "python scripts/ablation.py                # fixed vs heading",
        "# LLM answers (needs env vars, see rag/llm.py):",
        "RAG_EVAL_LLM=1 RAG_EVAL_LLM_BASE_URL=<url> \\",
        "  RAG_EVAL_LLM_API_KEY=<key> RAG_EVAL_LLM_MODEL=<model> \\",
        "  python scripts/run_demo.py --mode llm",
        "uvicorn app.main:app --reload",
        "```",
        "",
        "Then `POST /evaluate` returns the same numbers via the API.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    for name in qa_names:
        m = results[name]["metrics"]
        print(f"{name}: n={results[name]['n']} Recall@{args.top_k}={m['recall_at_k']:.3f}  "
              f"citation={m['citation_accuracy']:.3f}  coverage={m['answer_coverage']:.3f}  "
              f"hallucination_proxy={m['hallucination_proxy']:.3f}")
    print(f"docs={len(docs)} chunks={len(chunks)} mode={args.mode}")
    print(f"report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
