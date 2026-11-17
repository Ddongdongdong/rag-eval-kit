from rag.trace import TraceLogger


def test_trace_logger_writes_jsonl(tmp_path):
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path)
    rec = logger.log(
        "How long is a refund?",
        [{"chunk": {"id": "returns#c0", "doc_id": "returns"}, "score": 0.9}],
        latency_ms=12.5,
        answer="Refunds take 5 days.",
    )
    assert rec["trace_id"]
    assert rec["query"] == "How long is a refund?"
    assert rec["retrieved"][0]["chunk_id"] == "returns#c0"
    assert len(logger) == 1
    assert len(logger.recent()) == 1
    assert path.exists()
    assert "refund" in path.read_text(encoding="utf-8")
