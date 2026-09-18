"""
providers/ratelimit.py
======================
Sliding-window rate limiter, extracted from the Mistral client.

Nothing about pacing requests is Mistral-specific -- every hosted backend
publishes a per-minute quota and answers 429 when you exceed it -- so this lives
next to the provider implementations rather than inside one of them.

KNOWN LIMITATION (worth fixing when clients become injected)
------------------------------------------------------------
A limiter instance bounds only the calls made through it. Because each consumer
currently constructs its own client, the process holds several independent
limiters, and the effective ceiling is the sum of their budgets rather than the
one quota the API actually enforces. Sharing a single client -- and therefore a
single limiter -- per backend is what fixes it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

logger = logging.getLogger("phoenix_rag.providers.ratelimit")


class RateLimiter:
    """Allow at most `max_calls` acquisitions per `period_seconds`.

    Thread-safe: Ragas evaluates metrics with a worker pool, so judge calls can
    arrive concurrently.
    """

    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._evict(now)

            if len(self._calls) >= self.max_calls:
                wait_time = self.period_seconds - (now - self._calls[0])
                if wait_time > 0:
                    logger.debug("Rate limit hit, sleeping %.2fs", wait_time)
                    time.sleep(wait_time)
                self._evict(time.monotonic())

            self._calls.append(time.monotonic())

    def _evict(self, now: float) -> None:
        """Drop timestamps that have fallen out of the window."""
        while self._calls and now - self._calls[0] > self.period_seconds:
            self._calls.popleft()
