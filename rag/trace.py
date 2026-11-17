"""Query tracing: one JSONL record per query.

Each record carries a ``trace_id``, timestamp, the query, latency,
the retrieved chunk ids + scores, and any metric the caller computed.
Traces are appended to a JSONL file (default ``reports/traces.jsonl``)
and also kept in memory so the API can serve recent traces.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


class TraceLogger:
    def __init__(self, path: str | Path = "reports/traces.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory: list[dict] = []

    def log(
        self,
        query: str,
        retrieved: list[dict],
        latency_ms: float,
        metrics: dict | None = None,
        answer: str = "",
    ) -> dict:
        record = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "latency_ms": round(float(latency_ms), 2),
            "retrieved": [
                {
                    "chunk_id": item["chunk"]["id"],
                    "doc_id": item["chunk"]["doc_id"],
                    "score": item.get("rerank_score", item.get("score")),
                }
                for item in retrieved
            ],
            "answer_preview": (answer or "")[:200],
            "metrics": metrics or {},
        }
        self._memory.append(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def recent(self, limit: int = 20) -> list[dict]:
        return list(self._memory[-limit:])

    def __len__(self) -> int:
        return len(self._memory)
