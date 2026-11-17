#!/usr/bin/env python3
"""One-click demo: ingest sample docs -> query -> evaluate -> report.

Usage:
    python scripts/run_demo.py [--top-k 5] [--strategy heading]

Writes reports/demo_report.md and appends traces to reports/traces.jsonl.
No network, no LLM API — retrieval is TF-IDF, answers are extractive.
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
from rag.retriever import TFIDFRetriever, rerank

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "sample_docs"
QA_PATH = BASE_DIR / "data" / "sample_qa.jsonl"
REPORT_PATH = BASE_DIR / "reports" / "demo_report.md"


def load_docs() -> list[dict]:
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs.append({"id": path.stem, "title": path.stem, "text": path.read_text(encoding="utf-8")})
    return docs


def load_qa() -> list[dict]:
    return [json.loads(line) for line in QA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="rag-eval-kit demo")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strategy", choices=["fixed", "heading"], default="heading")
    args = parser.parse_args()

    t0 = time.perf_counter()
    docs = load_docs()
    qa = load_qa()
    chunks: list[dict] = []
    for doc in docs:
        chunks.extend(
            chunk_document(doc["id"], doc["title"], doc["text"], strategy=args.strategy)
        )
    retriever = TFIDFRetriever(chunks)
    result = evaluate_dataset(qa, retriever, top_k=args.top_k, use_rerank=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # One worked example for the report.
    example_q = "How long does a refund take after the warehouse receives the return?"
    hits = rerank(example_q, retriever.search(example_q, top_k=3), top_k=3)
    example_answer = extractive_answer(hits)

    m = result["metrics"]
    lines = [
        "# rag-eval-kit — Demo Report",
        "",
        f"Generated offline (TF-IDF retrieval, extractive answers, no LLM API). "
        f"Strategy: `{args.strategy}`, top_k={args.top_k}, "
        f"docs={len(docs)}, chunks={len(chunks)}, questions={result['n']}, "
        f"total time={elapsed_ms:.0f} ms.",
        "",
        "## Summary",
        "",
        "| Metric | Value | What it means |",
        "|---|---|---|",
        f"| Recall@{args.top_k} | **{m['recall_at_k']:.3f}** | Share of gold sources found in the top-{args.top_k} results |",
        f"| Citation accuracy | **{m['citation_accuracy']:.3f}** | Share of gold docs present in the citations |",
        f"| Answer coverage | **{m['answer_coverage']:.3f}** | Token-overlap F1 vs the gold answer |",
        f"| Hallucination proxy | **{m['hallucination_proxy']:.3f}** | Share of answer tokens absent from retrieved context (lower is better) |",
        "",
        "## Per-question results",
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
    lines += [
        "",
        "## Worked example",
        "",
        f"**Q:** {example_q}",
        "",
        f"**Extractive answer (top-3 chunks):** {example_answer[:500]}",
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
        "python scripts/run_demo.py",
        "uvicorn app.main:app --reload",
        "```",
        "",
        "Then `POST /evaluate` returns the same numbers via the API.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"docs={len(docs)} chunks={len(chunks)} questions={result['n']}")
    print(f"Recall@{args.top_k}={m['recall_at_k']:.3f}  citation={m['citation_accuracy']:.3f}  "
          f"coverage={m['answer_coverage']:.3f}  hallucination_proxy={m['hallucination_proxy']:.3f}")
    print(f"report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
