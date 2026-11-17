"""Offline retrieval: TF-IDF (standard library only) + simple rerank.

No embeddings, no network, no API key. Tokenisation handles English
words and CJK characters (each CJK character is one token) so the
same code works for English and Chinese corpora.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lower-case English words; keep each CJK character as a token."""
    return _TOKEN_RE.findall((text or "").lower())


class TFIDFRetriever:
    """TF-IDF retriever over a fixed list of chunks.

    ``chunks`` are dicts with at least ``id`` and ``text`` (the shape
    produced by :mod:`rag.chunker`). Call :meth:`fit` once, then
    :meth:`search` any number of times.
    """

    def __init__(self, chunks: list[dict] | None = None):
        self.chunks: list[dict] = list(chunks or [])
        self._idf: dict[str, float] = {}
        self._doc_vectors: list[dict[str, float]] = []
        self._doc_norms: list[float] = []
        if self.chunks:
            self.fit(self.chunks)

    def fit(self, chunks: list[dict]) -> "TFIDFRetriever":
        self.chunks = list(chunks)
        n = len(self.chunks)
        if n == 0:
            self._idf, self._doc_vectors, self._doc_norms = {}, [], []
            return self
        doc_freq: Counter = Counter()
        tokenised = [tokenize(c.get("text", "")) for c in self.chunks]
        for toks in tokenised:
            for tok in set(toks):
                doc_freq[tok] += 1
        # Smooth IDF: log((N + 1) / (df + 1)) + 1, always > 0.
        self._idf = {t: math.log((n + 1) / (df + 1)) + 1.0 for t, df in doc_freq.items()}
        self._doc_vectors = []
        self._doc_norms = []
        for toks in tokenised:
            counts = Counter(toks)
            total = max(len(toks), 1)
            vec = {t: (c / total) * self._idf.get(t, 1.0) for t, c in counts.items()}
            self._doc_vectors.append(vec)
            self._doc_norms.append(math.sqrt(sum(v * v for v in vec.values())) or 1.0)
        return self

    def _query_vector(self, query: str) -> tuple[dict[str, float], float]:
        toks = tokenize(query)
        if not toks:
            return {}, 1.0
        counts = Counter(toks)
        total = len(toks)
        vec = {t: (c / total) * self._idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return vec, norm

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k chunks as ``{"chunk": ..., "score": float}``.

        Cosine similarity on TF-IDF vectors. Empty query or empty
        index returns [].
        """
        if not self.chunks or not (query or "").strip():
            return []
        qvec, qnorm = self._query_vector(query)
        scored: list[tuple[float, dict]] = []
        for chunk, dvec, dnorm in zip(self.chunks, self._doc_vectors, self._doc_norms):
            dot = sum(qvec[t] * dvec.get(t, 0.0) for t in qvec)
            scored.append((dot / (qnorm * dnorm), chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"chunk": c, "score": round(s, 4)} for s, c in scored[: max(top_k, 0)]]


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Cheap rerank: TF-IDF score + exact query-term overlap bonus.

    ``candidates`` are ``{"chunk": ..., "score": ...}`` items from
    :meth:`TFIDFRetriever.search`. The overlap term rewards chunks
    that literally contain more distinct query terms, which fixes the
    common TF-IDF failure where a long chunk wins on length alone.
    Returns the top_k candidates, each with a ``rerank_score`` field.
    """
    qtokens = set(tokenize(query))
    out = []
    for item in candidates:
        text = item["chunk"].get("text", "")
        ctokens = set(tokenize(text))
        overlap = len(qtokens & ctokens) / max(len(qtokens), 1)
        # Keep the rerank score on a comparable scale to the TF-IDF score.
        rerank_score = round(0.7 * float(item.get("score", 0.0)) + 0.3 * overlap, 4)
        out.append({**item, "rerank_score": rerank_score})
    out.sort(key=lambda x: x["rerank_score"], reverse=True)
    return out[: max(top_k, 0)]
