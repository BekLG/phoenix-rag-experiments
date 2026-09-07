"""
providers/mistral.py
====================
Thin, rate-limited wrapper around the Mistral AI SDK.

Every module that talks to Mistral (embeddings, question generation, answer
generation, Ragas judging) goes through this wrapper so retry logic, rate
limiting, and error handling live in exactly one place.

The two methods below -- :meth:`MistralClient.chat` and
:meth:`MistralClient.embed` -- are the whole surface the rest of the codebase
needs from a backend. That is what makes swapping in other providers a matter of
writing more classes shaped like this one rather than changing the callers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mistralai.client import Mistral

from phoenix_rag.config import MistralSettings
from phoenix_rag.providers.ratelimit import RateLimiter

logger = logging.getLogger("phoenix_rag.providers.mistral")


class MistralClient:
    """Rate-limited, retrying wrapper around the raw Mistral SDK client."""

    def __init__(self, settings: MistralSettings):
        if not settings.api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set. Export it or put it in a .env file "
                "(see .env.example). The config's provider blocks reference the "
                "variable by name; the value never lives in the YAML."
            )
        self.settings = settings
        self._client = Mistral(api_key=settings.api_key)
        self._limiter = RateLimiter(settings.requests_per_minute, period_seconds=60.0)

    # -- embeddings ---------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returns one vector per input text."""
        return self._with_retry(
            lambda: self._client.embeddings.create(
                model=self.settings.embedding_model, inputs=texts
            ),
            extract=lambda resp: [d.embedding for d in resp.data],
        )

    # -- chat / generation ----------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        response_format: dict | None = None,
    ) -> str:
        """Single chat completion, returns the text of the first choice."""

        def _call():
            kwargs: dict[str, Any] = dict(
                model=model or self.settings.generation_model,
                messages=messages,
                temperature=temperature,
            )
            if response_format:
                kwargs["response_format"] = response_format
            return self._client.chat.complete(**kwargs)

        return self._with_retry(_call, extract=lambda resp: resp.choices[0].message.content)

    # -- internals --------------------------------------------------------

    def _with_retry(self, call, extract):
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            self._limiter.acquire()
            try:
                response = call()
                return extract(response)
            except Exception as exc:  # noqa: BLE001 - SDK raises various error types
                last_error = exc
                backoff = self.settings.base_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Mistral call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    self.settings.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(
            f"Mistral API call failed after {self.settings.max_retries} attempts"
        ) from last_error
