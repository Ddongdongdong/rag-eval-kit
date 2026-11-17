"""Optional LLM interface — OFF by default.

The evaluation core never needs an LLM: retrieval is TF-IDF and the
answer is extractive. This module is the single seam where a real
model can be plugged in later, so swapping providers touches one
file, not the evaluator.

To enable it, set ``RAG_EVAL_LLM=1`` and implement ``generate()``
for your provider (OpenAI / DeepSeek / local model). Never put an
API key in code — read it from the environment at call time.
"""

from __future__ import annotations

import os


class LLMClient:
    """Stub LLM client. ``enabled`` is False unless env opts in."""

    def __init__(self, provider: str = "none"):
        self.provider = provider
        self.enabled = os.environ.get("RAG_EVAL_LLM", "") == "1"

    def generate(self, prompt: str, context: str = "") -> str:
        """Generate an answer. Raises unless explicitly enabled.

        Replace the body with a real provider call. Example shape
        for DeepSeek / OpenAI-compatible endpoints::

            import httpx
            key = os.environ["DEEPSEEK_API_KEY"]  # never hard-code
            ...
        """
        if not self.enabled:
            raise RuntimeError(
                "LLM is disabled (default). Set RAG_EVAL_LLM=1 and "
                "implement rag/llm.py:LLMClient.generate() to enable it."
            )
        raise NotImplementedError(
            "LLMClient.generate() is a stub. Implement it for your "
            f"provider ({self.provider}) — see the docstring in rag/llm.py."
        )
