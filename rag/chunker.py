"""Document chunking: fixed-length and heading/paragraph strategies.

Both strategies return the same shape so the retriever and evaluator
do not need to know which one was used::

    {"id": "doc1#c0", "doc_id": "doc1", "title": "...",
     "heading": "...", "text": "...", "chunk_index": 0}
"""

from __future__ import annotations

import re


def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split ``text`` into fixed-length windows with ``overlap`` chars.

    Raises ValueError on non-positive sizes or overlap >= chunk_size.
    An empty / whitespace-only text yields no chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def chunk_by_heading(text: str, max_chars: int = 1200) -> list[dict]:
    """Split by Markdown headings, then by paragraph within a section.

    Returns ``[{"heading": str, "text": str}]``. Text before the first
    heading is kept under heading ``""``. Long paragraphs are further
    split with :func:`chunk_fixed` so no chunk exceeds ``max_chars``
    by much (a single long paragraph is hard-split at ``max_chars``).
    """
    text = (text or "").strip()
    if not text:
        return []
    matches = list(_HEADING_RE.finditer(text))
    sections: list[dict] = []
    if not matches:
        sections.append({"heading": "", "text": text})
    else:
        if matches[0].start() > 0:
            lead = text[: matches[0].start()].strip()
            if lead:
                sections.append({"heading": "", "text": lead})
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.end() : end].strip()
            sections.append({"heading": m.group(2).strip(), "text": body})

    out: list[dict] = []
    for sec in sections:
        # Split the section body into paragraphs (blank-line separated).
        paras = [p.strip() for p in re.split(r"\n\s*\n", sec["text"]) if p.strip()]
        if not paras:
            continue
        buf = ""
        for para in paras:
            if len(para) > max_chars:
                if buf:
                    out.append({"heading": sec["heading"], "text": buf})
                    buf = ""
                for piece in chunk_fixed(para, chunk_size=max_chars, overlap=0):
                    out.append({"heading": sec["heading"], "text": piece})
                continue
            candidate = (buf + "\n\n" + para).strip() if buf else para
            if len(candidate) > max_chars and buf:
                out.append({"heading": sec["heading"], "text": buf})
                buf = para
            else:
                buf = candidate
        if buf:
            out.append({"heading": sec["heading"], "text": buf})
    return out


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    strategy: str = "fixed",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Chunk one document with ``strategy`` in {"fixed", "heading"}."""
    if strategy == "heading":
        parts = chunk_by_heading(text, max_chars=chunk_size * 2)
        chunks = []
        for i, part in enumerate(parts):
            chunks.append(
                {
                    "id": f"{doc_id}#c{i}",
                    "doc_id": doc_id,
                    "title": title,
                    "heading": part["heading"],
                    "text": part["text"],
                    "chunk_index": i,
                }
            )
        return chunks
    if strategy == "fixed":
        pieces = chunk_fixed(text, chunk_size=chunk_size, overlap=overlap)
        return [
            {
                "id": f"{doc_id}#c{i}",
                "doc_id": doc_id,
                "title": title,
                "heading": "",
                "text": piece,
                "chunk_index": i,
            }
            for i, piece in enumerate(pieces)
        ]
    raise ValueError(f"unknown chunking strategy: {strategy!r} (want 'fixed' or 'heading')")
