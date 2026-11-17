import pytest

from rag.chunker import chunk_by_heading, chunk_document, chunk_fixed


def test_fixed_chunker_splits_and_respects_size():
    text = "abcdefghij" * 30  # 300 chars
    chunks = chunk_fixed(text, chunk_size=100, overlap=10)
    assert len(chunks) >= 3
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0].startswith("abcdefghij")


def test_fixed_chunker_empty_and_invalid():
    assert chunk_fixed("") == []
    assert chunk_fixed("   ") == []
    assert chunk_fixed("short") == ["short"]
    with pytest.raises(ValueError):
        chunk_fixed("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_fixed("abc", chunk_size=10, overlap=10)


def test_heading_chunker_splits_by_heading():
    text = "# Title\n\nFirst section text.\n\n## Details\n\nSecond section text here."
    parts = chunk_by_heading(text)
    headings = [p["heading"] for p in parts]
    assert "Title" in headings
    assert "Details" in headings
    assert any("First section" in p["text"] for p in parts)
    assert any("Second section" in p["text"] for p in parts)


def test_chunk_document_strategies():
    text = "# A\n\nHello world. " * 20
    fixed = chunk_document("d1", "Doc", text, strategy="fixed", chunk_size=100, overlap=10)
    heading = chunk_document("d1", "Doc", text, strategy="heading")
    assert fixed and all(c["doc_id"] == "d1" for c in fixed)
    assert fixed[0]["id"] == "d1#c0"
    assert heading and heading[0]["heading"] == "A"
    with pytest.raises(ValueError):
        chunk_document("d1", "Doc", text, strategy="nope")
