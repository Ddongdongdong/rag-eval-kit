from rag.chunker import chunk_document
from rag.retriever import TFIDFRetriever, rerank, tokenize


def _retriever():
    docs = [
        ("returns", "Returns", "Customers can return a product within 30 days of delivery for a refund."),
        ("shipping", "Shipping", "Standard shipping takes 3 to 5 working days and express takes 1 to 2 days."),
        ("pricing", "Pricing", "Annual billing gives a 15 percent discount compared with monthly billing."),
    ]
    chunks = []
    for doc_id, title, text in docs:
        chunks.extend(chunk_document(doc_id, title, text, strategy="fixed"))
    return TFIDFRetriever(chunks)


def test_tokenize_handles_english_and_cjk():
    assert "refund" in tokenize("Refund within 30 days")
    assert "退" in tokenize("退货 30 天")


def test_retriever_finds_relevant_doc():
    r = _retriever()
    hits = r.search("how do I return a product for a refund?", top_k=2)
    assert hits
    assert hits[0]["chunk"]["doc_id"] == "returns"
    assert hits[0]["score"] > 0


def test_retriever_empty_query_and_empty_index():
    assert TFIDFRetriever([]).search("anything") == []
    assert _retriever().search("") == []
    assert _retriever().search("   ") == []


def test_rerank_keeps_relevant_on_top():
    r = _retriever()
    hits = r.search("annual billing discount percent", top_k=3)
    ranked = rerank("annual billing discount percent", hits, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["chunk"]["doc_id"] == "pricing"
    assert "rerank_score" in ranked[0]
