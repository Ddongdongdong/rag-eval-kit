from rag.chunker import chunk_document
from rag.evaluator import (
    answer_coverage,
    citation_accuracy,
    evaluate_dataset,
    hallucination_proxy,
    recall_at_k,
)
from rag.retriever import TFIDFRetriever


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0
    assert recall_at_k(["a", "b"], ["a", "c"], k=2) == 0.5
    assert recall_at_k([], ["a"], k=5) == 0.0
    assert recall_at_k(["a"], [], k=5) == 0.0


def test_citation_accuracy_and_coverage():
    assert citation_accuracy(["d1", "d2"], ["d1", "d3"]) == 0.5
    assert citation_accuracy([], ["d1"]) == 0.0
    assert answer_coverage("refund within 5 days", "refund within 5 days") == 1.0
    assert answer_coverage("cats and dogs", "refund within 5 days") == 0.0
    assert 0.0 < answer_coverage("refund within 5 working days", "refund within 5 days") < 1.0


def test_hallucination_proxy():
    ctx = "refunds are processed within 5 working days to the original payment method"
    assert hallucination_proxy("refunds are processed within 5 working days", ctx) == 0.0
    invented = "refunds are processed within 5 working days plus a free spaceship"
    assert hallucination_proxy(invented, ctx) > 0.0


def test_evaluate_dataset_end_to_end():
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
        {"question": "How long does standard shipping take?", "gold_answer": "Standard shipping takes 3 to 5 working days.", "gold_doc_ids": ["shipping"]},
    ]
    result = evaluate_dataset(qa, retriever, top_k=5)
    assert result["n"] == 2
    assert result["metrics"]["recall_at_k"] == 1.0
    assert result["metrics"]["citation_accuracy"] == 1.0
    assert len(result["per_question"]) == 2
