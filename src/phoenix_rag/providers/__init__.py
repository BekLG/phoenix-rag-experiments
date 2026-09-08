"""
phoenix_rag.providers
=====================
Backend clients: the things that actually call an embedding or chat API.

A **provider** binds one backend + model to one role. The role-agnostic surface
lives in :mod:`~phoenix_rag.providers.base` (:class:`ChatProvider`,
:data:`EmbeddingProvider`, and the :class:`Providers` bundle);
:func:`~phoenix_rag.providers.registry.build_providers` picks one per role from
an :class:`~phoenix_rag.config.AppConfig` and shares a rate limiter across roles
that hit the same API. ``mistral`` is the only backend wired up so far.

The shared machinery that is not specific to any one backend -- request pacing
and retry -- lives in :mod:`~phoenix_rag.providers.ratelimit` so the next backend
inherits it rather than reimplementing it.

Names resolve on first access (PEP 562): importing this package must not pull in
a vendor SDK for someone who only wanted the :class:`ChatProvider` type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY_EXPORTS = {
    "ChatProvider": "phoenix_rag.providers.base",
    "EmbeddingProvider": "phoenix_rag.providers.base",
    "Providers": "phoenix_rag.providers.base",
    "build_providers": "phoenix_rag.providers.registry",
    "MistralClient": "phoenix_rag.providers.mistral",
    "MistralChatProvider": "phoenix_rag.providers.mistral",
    "RateLimiter": "phoenix_rag.providers.ratelimit",
}

if TYPE_CHECKING:  # pragma: no cover - type checkers / IDEs only
    from phoenix_rag.providers.base import ChatProvider, EmbeddingProvider, Providers
    from phoenix_rag.providers.mistral import MistralChatProvider, MistralClient
    from phoenix_rag.providers.ratelimit import RateLimiter
    from phoenix_rag.providers.registry import build_providers


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # resolve once, then it is a normal attribute
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_EXPORTS])


__all__ = list(_LAZY_EXPORTS)
