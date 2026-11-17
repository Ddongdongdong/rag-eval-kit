"""FastAPI service for rag-eval-kit.

Endpoints:
    GET  /health    — liveness + index stats
    POST /ingest    — ingest documents (chunk + index)
    POST /query     — retrieve + extractive answer + trace
    POST /evaluate  — run the bundled QA set through the retriever
    GET  /traces    — recent query traces
    GET  /          — service info

On startup the bundled sample docs under ``data/sample_docs/`` are
loaded automatically so ``/query`` and ``/evaluate`` work with no
setup. ``POST /ingest`` replaces the index with the given documents.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.chunker import chunk_document
from rag.evaluator import evaluate_dataset, extractive_answer
from rag.retriever import TFIDFRetriever, rerank
from rag.trace import TraceLogger

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = BASE_DIR / "data" / "sample_docs"
SAMPLE_QA_PATHS = {
    "easy": BASE_DIR / "data" / "sample_qa.jsonl",
    "hard": BASE_DIR / "data" / "sample_qa_hard.jsonl",
}
SAMPLE_QA_PATH = SAMPLE_QA_PATHS["easy"]

app = FastAPI(title="rag-eval-kit", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- in-memory state (single-process demo service) ---
_chunks: list[dict] = []
_retriever = TFIDFRetriever()
_tracer = TraceLogger(BASE_DIR / "reports" / "traces.jsonl")


class DocumentIn(BaseModel):
    id: str
    title: str = ""
    text: str


class IngestIn(BaseModel):
    documents: list[DocumentIn]
    strategy: str = Field(default="heading", pattern="^(fixed|heading)$")
    chunk_size: int = 500
    overlap: int = 50


class QueryIn(BaseModel):
    query: str
    top_k: int = 5
    use_rerank: bool = True


class EvaluateIn(BaseModel):
    top_k: int = 5
    use_rerank: bool = True
    qa_set: str = Field(default="easy", pattern="^(easy|hard)$")


def _load_sample_docs() -> list[dict]:
    docs = []
    if not SAMPLE_DOCS_DIR.exists():
        return docs
    for path in sorted(SAMPLE_DOCS_DIR.glob("*.md")):
        docs.append(
            {"id": path.stem, "title": path.stem, "text": path.read_text(encoding="utf-8")}
        )
    return docs


def _index(documents: list[dict], strategy: str = "heading", chunk_size: int = 500, overlap: int = 50) -> int:
    global _chunks, _retriever
    chunks: list[dict] = []
    for doc in documents:
        chunks.extend(
            chunk_document(
                doc_id=doc["id"],
                title=doc.get("title", doc["id"]),
                text=doc.get("text", ""),
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    _chunks = chunks
    _retriever = TFIDFRetriever(chunks)
    return len(chunks)


def _load_sample_qa(qa_set: str = "easy") -> list[dict]:
    qa = []
    path = SAMPLE_QA_PATHS[qa_set]
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                qa.append(json.loads(line))
    return qa


@app.on_event("startup")
def _startup() -> None:
    docs = _load_sample_docs()
    if docs:
        _index(docs, strategy="heading")


@app.get("/")
def root() -> dict:
    return {
        "service": "rag-eval-kit",
        "version": "0.2.0",
        "docs": "/docs",
        "endpoints": ["GET /health", "POST /ingest", "POST /query", "POST /evaluate", "GET /traces"],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks": len(_chunks),
        "docs": len({c["doc_id"] for c in _chunks}),
        "traces": len(_tracer),
    }


@app.post("/ingest")
def ingest(payload: IngestIn) -> dict:
    if not payload.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")
    n = _index(
        [d.model_dump() for d in payload.documents],
        strategy=payload.strategy,
        chunk_size=payload.chunk_size,
        overlap=payload.overlap,
    )
    return {"status": "ok", "documents": len(payload.documents), "chunks": n, "strategy": payload.strategy}


@app.post("/query")
def query(payload: QueryIn) -> dict:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not _chunks:
        raise HTTPException(status_code=400, detail="index is empty, POST /ingest first")
    start = time.perf_counter()
    hits = _retriever.search(payload.query, top_k=payload.top_k)
    if payload.use_rerank:
        hits = rerank(payload.query, hits, top_k=payload.top_k)
    answer = extractive_answer(hits)
    latency_ms = (time.perf_counter() - start) * 1000
    record = _tracer.log(payload.query, hits, latency_ms, answer=answer)
    return {
        "trace_id": record["trace_id"],
        "query": payload.query,
        "answer": answer,
        "latency_ms": round(latency_ms, 2),
        "citations": [
            {
                "chunk_id": h["chunk"]["id"],
                "doc_id": h["chunk"]["doc_id"],
                "title": h["chunk"]["title"],
                "score": h.get("rerank_score", h.get("score")),
                "text": h["chunk"]["text"][:300],
            }
            for h in hits
        ],
    }


@app.post("/evaluate")
def evaluate(payload: EvaluateIn) -> dict:
    if not _chunks:
        raise HTTPException(status_code=400, detail="index is empty, POST /ingest first")
    qa = _load_sample_qa(payload.qa_set)
    if not qa:
        raise HTTPException(status_code=400, detail=f"no QA data found for qa_set={payload.qa_set!r}")
    return evaluate_dataset(qa, _retriever, top_k=payload.top_k, use_rerank=payload.use_rerank)


@app.get("/traces")
def traces(limit: int = 20) -> dict:
    return {"traces": _tracer.recent(limit=limit)}
