"""
gemini_client.py
=================
Thin, rate-limited wrapper around Google's Gemini API (google-genai SDK),
providing the shared .chat()/.embed() interface used throughout the project.

NOTE ON SDK SHAPES: google-genai's exact response object attributes
(e.g. the embeddings response shape) have shifted across SDK releases in
the past. The shapes used below (`response.text`, `response.embeddings[i].values`)
match the SDK's documented shape as of this writing -- if you hit an
AttributeError here, print(dir(response)) once to confirm the exact
attribute names on your installed `google-genai` version and adjust
_with_retry's `extract` lambdas accordingly.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from google import genai
from google.genai import types as genai_types

from config import GeminiSettings

logger = logging.getLogger("phoenix_rag.gemini_client")


class RateLimiter:
    """Thread-safe paced limiter that avoids using the quota in a burst."""

    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._interval = period_seconds / max_calls
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_allowed)
            self._next_allowed = scheduled + self._interval
            wait_time = scheduled - now

        if wait_time > 0:
            logger.debug("Pacing Gemini request for %.2fs", wait_time)
            time.sleep(wait_time)


_LIMITERS: dict[tuple[str, str, int], RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_shared_rate_limiter(settings: GeminiSettings, model: str) -> RateLimiter:
    """Return the process-wide limiter for an API key/model/quota tuple."""
    rpm = settings.requests_per_minute_for(model)
    key = (settings.api_key, model, rpm)
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = RateLimiter(rpm, period_seconds=60.0)
            _LIMITERS[key] = limiter
        return limiter


class GeminiClient:
    """Rate-limited, retrying wrapper around the raw google-genai client."""

    def __init__(self, settings: GeminiSettings):
        if not settings.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Export it or put it in a .env file."
            )
        self.settings = settings
        self._client = genai.Client(api_key=settings.api_key)

    # -- embeddings ---------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returns one vector per input text."""
        return self._with_retry(
            lambda: self._client.models.embed_content(
                model=self.settings.embedding_model,
                contents=texts,
            ),
            extract=lambda resp: [e.values for e in resp.embeddings],
            model=self.settings.embedding_model,
        )

    # -- chat / generation ----------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        response_format: dict | None = None,
    ) -> str:
        """Single chat completion, returns the generated text.

        Converts common {"role", "content"} messages into
        Gemini's shape: system messages become `system_instruction`
        (Gemini has no "system" role in `contents`), "assistant" becomes
        "model" (Gemini's role name), everything else becomes "user".
        """
        system_instruction, contents = self._convert_messages(messages)

        def _call():
            config_kwargs: dict[str, Any] = {"temperature": temperature}
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            if response_format:
                # Gemini's structured-output knob; the exact config field
                # name has moved across SDK versions -- verify against
                # your installed google-genai version if you rely on this.
                config_kwargs["response_mime_type"] = "application/json"

            return self._client.models.generate_content(
                model=model or self.settings.generation_model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )

        selected_model = model or self.settings.generation_model
        return self._with_retry(
            _call,
            extract=lambda resp: resp.text,
            model=selected_model,
        )

    @staticmethod
    def _convert_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[dict]]:
        system_instruction: str | None = None
        contents: list[dict] = []
        for m in messages:
            role = m.get("role")
            text = m.get("content", "")
            if role == "system":
                system_instruction = (
                    f"{system_instruction}\n\n{text}" if system_instruction else text
                )
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})
        return system_instruction, contents

    # -- internals --------------------------------------------------------

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        for candidate in (
            getattr(exc, "status_code", None),
            getattr(exc, "code", None),
            getattr(getattr(exc, "response", None), "status_code", None),
        ):
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            value = headers.get("Retry-After") or headers.get("retry-after")
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                try:
                    retry_at = parsedate_to_datetime(value)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return max(
                        0.0,
                        (retry_at - datetime.now(timezone.utc)).total_seconds(),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
        return None

    @classmethod
    def _is_transient(cls, exc: Exception) -> bool:
        status = cls._status_code(exc)
        if status is not None:
            return status in {408, 429} or 500 <= status <= 599
        name = type(exc).__name__.lower()
        return isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(
            marker in name for marker in ("timeout", "connection", "transport")
        )

    def _with_retry(self, call, extract, model: str):
        last_error: Exception | None = None
        limiter = get_shared_rate_limiter(self.settings, model)
        for attempt in range(1, self.settings.max_retries + 1):
            limiter.acquire()
            try:
                response = call()
                return extract(response)
            except Exception as exc:  # noqa: BLE001 - SDK raises various error types
                last_error = exc
                if not self._is_transient(exc) or attempt == self.settings.max_retries:
                    raise
                backoff = self.settings.base_backoff_seconds * (2 ** (attempt - 1))
                backoff += random.uniform(0, self.settings.base_backoff_seconds)
                retry_after = self._retry_after_seconds(exc)
                if retry_after is not None:
                    backoff = max(backoff, retry_after)
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    self.settings.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError("Gemini API retry loop exited unexpectedly") from last_error
