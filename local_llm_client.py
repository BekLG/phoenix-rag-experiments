"""
local_llm_client.py
====================
Thin wrapper around a locally-hosted LLM served via Ollama, used only by
the non-optimized baseline pipeline (local_rag_pipeline.py, baseline_runner.py).

Mirrors MistralClient's .chat() shape so the baseline pipeline's code
reads almost identically to rag_pipeline.py -- the generation call site is
the only real difference between the optimized and baseline pipelines.

No rate limiting/backoff here (unlike mistral_client.py) -- a local model
has no API quota to respect, just local compute time.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger("phoenix_rag.local_llm_client")


class LocalLLMClient:
    """Minimal client for a local Ollama model.

    Requires Ollama running locally (`ollama serve`, default port 11434)
    with the target model already pulled, e.g.:
        ollama pull qwen2.5:7b-instruct
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        **_ignored,
    ) -> str:
        """Single chat completion via Ollama's /api/chat endpoint.

        **_ignored absorbs kwargs like response_format that MistralClient.chat()
        accepts but Ollama's basic chat endpoint doesn't, so this stays a
        drop-in-shaped replacement without erroring on extra keywords.
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"(`ollama serve`, and `ollama pull {model or self.model}` if not "
                "already pulled)"
            ) from exc

        data = response.json()
        return data["message"]["content"]