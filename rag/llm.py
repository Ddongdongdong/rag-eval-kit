"""Optional LLM interface — OFF by default.

The evaluation core never needs an LLM: retrieval is TF-IDF and the
answer is extractive. This module is the single seam where a real
model can be plugged in, so swapping providers touches one file,
not the evaluator.

Enable it with environment variables only — never put a key in code,
in a report, or in a committed file::

    RAG_EVAL_LLM=1
    RAG_EVAL_LLM_BASE_URL=https://api.deepseek.com/v1   # any OpenAI-compatible endpoint
    RAG_EVAL_LLM_API_KEY=<your key>                      # read at call time, never logged
    RAG_EVAL_LLM_MODEL=deepseek-chat

``generate()`` raises a clear ``RuntimeError`` (never a silent
fallback to the extractive answer) when the LLM is disabled or when
any of the three variables above is missing.
"""

from __future__ import annotations

import os

ENV_FLAG = "RAG_EVAL_LLM"
ENV_BASE_URL = "RAG_EVAL_LLM_BASE_URL"
ENV_API_KEY = "RAG_EVAL_LLM_API_KEY"
ENV_MODEL = "RAG_EVAL_LLM_MODEL"

SYSTEM_PROMPT = (
    "Answer only using the provided context. If the context does not "
    "contain the answer, say so instead of guessing. Keep the answer "
    "short and stick to the facts in the context."
)


class LLMClient:
    """OpenAI-compatible chat client. Disabled unless env opts in."""

    def __init__(self, provider: str = "openai-compatible"):
        self.provider = provider
        self.enabled = os.environ.get(ENV_FLAG, "") == "1"
        # Read at construction time; the key is never stored anywhere
        # else and never appears in logs, reports or error messages.
        self.base_url = os.environ.get(ENV_BASE_URL, "").rstrip("/")
        self.api_key = os.environ.get(ENV_API_KEY, "")
        self.model = os.environ.get(ENV_MODEL, "")

    def require_configured(self) -> None:
        """Raise a clear error unless the LLM is enabled and configured."""
        if not self.enabled:
            raise RuntimeError(
                "LLM is disabled (default). To enable it, set RAG_EVAL_LLM=1 "
                f"and provide {ENV_BASE_URL}, {ENV_API_KEY} and {ENV_MODEL} "
                "as environment variables. No API key is needed for the "
                "default extractive mode."
            )
        missing = [
            name
            for name, value in (
                (ENV_BASE_URL, self.base_url),
                (ENV_API_KEY, self.api_key),
                (ENV_MODEL, self.model),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "LLM is enabled (RAG_EVAL_LLM=1) but missing environment "
                f"variable(s): {', '.join(missing)}. Set them in your shell "
                "before running with --mode llm — keys are read from the "
                "environment only and are never written to code or reports."
            )

    def generate(self, prompt: str, context: str = "") -> str:
        """Generate an answer grounded in ``context`` via chat completions.

        Raises RuntimeError when disabled, unconfigured, unreachable,
        or when the provider returns a non-200 status. The API key is
        sent only in the Authorization header and never echoed back.
        """
        self.require_configured()
        import httpx

        user = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:" if context else prompt
        try:
            # trust_env=False: ignore ambient HTTP(S)_PROXY variables so
            # the client talks directly to the configured base_url.
            with httpx.Client(timeout=60.0, trust_env=False) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0,
                    },
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"LLM request failed ({type(exc).__name__}): {exc}. "
                f"Check {ENV_BASE_URL} and your network, then retry."
            ) from exc
        if resp.status_code != 200:
            raise RuntimeError(
                f"LLM request failed with HTTP {resp.status_code}: "
                f"{resp.text[:300]}. Check {ENV_MODEL} and "
                f"{ENV_API_KEY} for endpoint {self.base_url}."
            )
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "LLM returned an unexpected response shape; expected "
                f"choices[0].message.content from {self.base_url}."
            ) from exc
        if not content or not str(content).strip():
            raise RuntimeError("LLM returned an empty answer.")
        return str(content).strip()
