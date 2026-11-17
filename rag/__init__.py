"""RAG evaluation core: chunking, retrieval, evaluation, tracing."""

from .chunker import chunk_document, chunk_fixed, chunk_by_heading
from .retriever import TFIDFRetriever, rerank
from .evaluator import evaluate_dataset, recall_at_k, citation_accuracy, answer_coverage
from .trace import TraceLogger

__all__ = [
    "chunk_document",
    "chunk_fixed",
    "chunk_by_heading",
    "TFIDFRetriever",
    "rerank",
    "evaluate_dataset",
    "recall_at_k",
    "citation_accuracy",
    "answer_coverage",
    "TraceLogger",
]
