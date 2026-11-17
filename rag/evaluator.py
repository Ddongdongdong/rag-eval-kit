"""Evaluation metrics for offline RAG.

All metrics are deterministic and need no LLM:

- ``recall_at_k`` — fraction of gold chunk/doc ids found in the
  top-k retrieved ids.
- ``citation_accuracy`` — fraction of gold sources that appear in
  the predicted citations (per question, averaged).
- ``answer_coverage`` — token overlap (F1) between the generated
  (extractive) answer and the gold answer.
- ``hallucination proxy`` — share of answer tokens that do *not*
  appear in the retrieved context. For an extractive answer built
  from retrieved chunks this should be low; a high value means the
  answer asserts things the retriever never returned.

``evaluate_dataset`` runs a whole QA set through a retriever and
returns aggregate metrics plus one row per question.
"""

from __future__ import annotations

from collections.abc import Callable

from .retriever import TFIDFRetriever, rerank, tokenize


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float:
    """|gold ∩ retrieved[:k]| / |gold|; 0.0 when gold is empty."""
    if not gold_ids:
        return 0.0
    top = set(retrieved_ids[: max(k, 0)])
    return round(len(set(gold_ids) & top) / len(set(gold_ids)), 4)


def citation_accuracy(predicted_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    """Share of gold doc ids present in the predicted citations."""
    if not gold_doc_ids:
        return 0.0
    pred = set(predicted_doc_ids)
    return round(len(set(gold_doc_ids) & pred) / len(set(gold_doc_ids)), 4)


def _f1(pred_tokens: set[str], gold_tokens: set[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return 0.0
    inter = len(pred_tokens & gold_tokens)
    precision = inter / len(pred_tokens)
    recall = inter / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def answer_coverage(answer: str, gold_answer: str) -> float:
    """Token-overlap F1 between ``answer`` and ``gold_answer``."""
    return _f1(set(tokenize(answer)), set(tokenize(gold_answer)))


def hallucination_proxy(answer: str, context: str) -> float:
    """Share of answer tokens absent from ``context`` (0..1, lower is better).

    This is a *proxy*, not a judgement: an extractive answer copied
    from the context scores ~0, while an answer that invents numbers
    or terms scores high. Use it to compare systems, not to certify
    any single answer.
    """
    ans_tokens = set(tokenize(answer))
    if not ans_tokens:
        return 0.0
    ctx_tokens = set(tokenize(context))
    outside = ans_tokens - ctx_tokens
    return round(len(outside) / len(ans_tokens), 4)


def extractive_answer(retrieved: list[dict], max_chars: int = 600) -> str:
    """Build a naive extractive answer from retrieved chunks."""
    parts = []
    for item in retrieved:
        text = item["chunk"].get("text", "").strip()
        if text:
            parts.append(text)
    joined = " ".join(parts)
    return joined[:max_chars]


def evaluate_dataset(
    qa: list[dict],
    retriever: TFIDFRetriever,
    top_k: int = 5,
    use_rerank: bool = True,
    answer_fn: Callable[[list[dict], str], str] | None = None,
) -> dict:
    """Evaluate ``qa`` against ``retriever``.

    Each QA item: ``{"question": str, "gold_answer": str,
    "gold_doc_ids": [str], "gold_chunk_ids": [str] (optional)}``.
    ``answer_fn`` — when given — is called as ``answer_fn(hits, question)``
    to produce the answer for each question (e.g. an LLM answer built
    from the retrieved context); by default the extractive answer is
    used. Returns ``{"n": int, "top_k": int, "metrics": {...},
    "per_question": [...]}`` with mean Recall@k, citation accuracy,
    answer coverage and hallucination proxy.
    """
    rows: list[dict] = []
    for item in qa:
        question = item.get("question", "")
        gold_docs = list(item.get("gold_doc_ids", []))
        gold_chunks = list(item.get("gold_chunk_ids", []))
        # Prefer chunk-level gold when present, else doc-level gold
        # mapped through the retrieved chunk ids' doc ids.
        hits = retriever.search(question, top_k=top_k)
        if use_rerank:
            hits = rerank(question, hits, top_k=top_k)
        retrieved_ids = [h["chunk"]["id"] for h in hits]
        retrieved_docs = [h["chunk"]["doc_id"] for h in hits]
        if gold_chunks:
            rec = recall_at_k(retrieved_ids, gold_chunks, k=top_k)
        else:
            rec = recall_at_k(retrieved_docs, gold_docs, k=top_k)
        cit = citation_accuracy(retrieved_docs, gold_docs)
        context = " ".join(h["chunk"].get("text", "") for h in hits)
        answer = answer_fn(hits, question) if answer_fn else extractive_answer(hits)
        cov = answer_coverage(answer, item.get("gold_answer", ""))
        hal = hallucination_proxy(answer, context)
        rows.append(
            {
                "question": question,
                "gold_doc_ids": gold_docs,
                "retrieved_doc_ids": retrieved_docs,
                "retrieved_chunk_ids": retrieved_ids,
                "recall_at_k": rec,
                "citation_accuracy": cit,
                "answer_coverage": cov,
                "hallucination_proxy": hal,
            }
        )

    def _mean(key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else 0.0

    return {
        "n": len(rows),
        "top_k": top_k,
        "use_rerank": use_rerank,
        "metrics": {
            "recall_at_k": _mean("recall_at_k"),
            "citation_accuracy": _mean("citation_accuracy"),
            "answer_coverage": _mean("answer_coverage"),
            "hallucination_proxy": _mean("hallucination_proxy"),
        },
        "per_question": rows,
    }
