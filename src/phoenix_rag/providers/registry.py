"""
providers/registry.py
======================
The composition root: turn an :class:`~phoenix_rag.config.AppConfig` into a
:class:`~phoenix_rag.providers.base.Providers` bundle -- one backend per role.

Two jobs live here and nowhere else:

1. **Backend dispatch.** Each role names a ``backend`` string ("mistral",
   "openai", "anthropic", "local"); the maps below turn that string into a
   concrete provider. A backend the config names but the code does not implement
   raises :class:`~phoenix_rag.config.UnsupportedBackendError` here, naming
   the role, rather than failing obscurely deep in a call stack.

2. **Shared rate limiting.** Every role that talks to the same API shares ONE
   :class:`~phoenix_rag.providers.ratelimit.RateLimiter`. That is the fix for
   the long-standing overshoot in which each consumer built its own client, so
   the process paced itself to the *sum* of N private budgets instead of the one
   quota the API actually enforces. Roles are grouped by backend identity
   (backend + credential variable + endpoint); the shared limiter is sized to
   the tightest ``requests_per_minute`` among the roles in the group.

Wired backends: ``mistral`` (chat + embedding), ``openai`` (chat + embedding),
``anthropic`` (chat), and ``local`` (chat via an OpenAI-compatible endpoint;
embedding via in-process sentence-transformers). Every backend except Mistral
loads its vendor package lazily, inside its builder in
:mod:`phoenix_rag.providers.langchain_backends` -- so importing this module stays
cheap and an install without a given backend's extra is only felt if a config
actually selects it.
"""

from __future__ import annotations

from typing import Callable

from phoenix_rag.config import AppConfig, MistralSettings, UnsupportedBackendError
from phoenix_rag.config.schema import ProviderConfig
from phoenix_rag.core.embeddings import MistralEmbeddings
from phoenix_rag.providers.base import ChatProvider, EmbeddingProvider, Providers
from phoenix_rag.providers.langchain_backends import (
    build_anthropic_chat,
    build_local_chat,
    build_local_embedding,
    build_openai_chat,
    build_openai_embedding,
)
from phoenix_rag.providers.mistral import MistralChatProvider, MistralClient
from phoenix_rag.providers.ratelimit import RateLimiter

# A backend-identity key: two roles share a limiter iff these match. base_url is
# part of it because the same "openai" backend pointed at api.openai.com and at a
# local Ollama endpoint are different quotas.
_Identity = tuple[str, str | None, str | None]


def _identity(provider: ProviderConfig) -> _Identity:
    return (provider.backend, provider.api_key_env, provider.base_url)


def _mistral_settings_for(provider: ProviderConfig) -> MistralSettings:
    """Project one role's :class:`ProviderConfig` onto the per-role
    :class:`MistralSettings` that :class:`MistralClient` consumes.

    All four model fields are set to this role's single model: chat providers
    pass their model explicitly on every call, and the embedding provider reads
    ``embedding_model`` -- so whichever field a given role actually uses holds
    the right value, and the others are harmless.
    """
    return MistralSettings(
        api_key=provider.resolve_api_key(),
        embedding_model=provider.model,
        generation_model=provider.model,
        optimizer_model=provider.model,
        judge_model=provider.model,
        requests_per_minute=provider.requests_per_minute,
        max_retries=provider.max_retries,
        base_backoff_seconds=provider.base_backoff_seconds,
    )


def _build_mistral_chat(provider: ProviderConfig, limiter: RateLimiter) -> ChatProvider:
    settings = _mistral_settings_for(provider)
    client = MistralClient(settings, limiter=limiter)
    return MistralChatProvider(client, model=provider.model, api_key=settings.api_key)


def _build_mistral_embedding(
    provider: ProviderConfig, limiter: RateLimiter
) -> EmbeddingProvider:
    settings = _mistral_settings_for(provider)
    client = MistralClient(settings, limiter=limiter)
    return MistralEmbeddings(settings, client=client)


# backend name -> builder. Split by surface because the two are not
# interchangeable: an embedding backend produces a LangChain Embeddings, a chat
# backend a ChatProvider. "local" chat is an OpenAI-compatible endpoint; "local"
# embedding is in-process sentence-transformers. Anthropic has no embedding model
# of its own, so it is chat-only here.
_CHAT_BACKENDS: dict[str, Callable[[ProviderConfig, RateLimiter], ChatProvider]] = {
    "mistral": _build_mistral_chat,
    "openai": build_openai_chat,
    "anthropic": build_anthropic_chat,
    "local": build_local_chat,
}
_EMBEDDING_BACKENDS: dict[
    str, Callable[[ProviderConfig, RateLimiter], EmbeddingProvider]
] = {
    "mistral": _build_mistral_embedding,
    "openai": build_openai_embedding,
    "local": build_local_embedding,
}


def _unsupported(role: str, provider: ProviderConfig, wired: list[str]) -> UnsupportedBackendError:
    return UnsupportedBackendError(
        f"The {role!r} role is configured with backend {provider.backend!r}, "
        f"which has no implementation yet. Wired backends for this role: "
        f"{', '.join(sorted(wired))}. Set the role to one of those, or add its "
        "backend to phoenix_rag.providers.registry."
    )


def _shared_limiters(role_configs: dict[str, ProviderConfig]) -> dict[_Identity, RateLimiter]:
    """One limiter per backend identity, sized to the tightest rpm in the group.

    Taking the minimum is the safe reading of a config that sets a different
    ``requests_per_minute`` on two roles that turn out to share an API: never
    pace above the lowest ceiling the operator asked for. For the all-Mistral
    default (45 everywhere) this is just 45, unchanged.
    """
    budgets: dict[_Identity, int] = {}
    for provider in role_configs.values():
        key = _identity(provider)
        budgets[key] = min(
            budgets.get(key, provider.requests_per_minute), provider.requests_per_minute
        )
    return {key: RateLimiter(rpm, period_seconds=60.0) for key, rpm in budgets.items()}


def build_providers(config: AppConfig) -> Providers:
    """Build the four role backends from ``config.providers``.

    Roles that hit the same API share one rate limiter. Raises
    :class:`UnsupportedBackendError` (naming the role) for a backend not wired up
    yet, and ``ValueError`` for a hosted backend whose API-key variable is unset.
    """
    role_configs = config.providers.as_map()
    limiters = _shared_limiters(role_configs)

    built: dict[str, object] = {}
    for role, provider in role_configs.items():
        limiter = limiters[_identity(provider)]
        if role == "embedding":
            builder = _EMBEDDING_BACKENDS.get(provider.backend)
            if builder is None:
                raise _unsupported(role, provider, list(_EMBEDDING_BACKENDS))
        else:
            builder = _CHAT_BACKENDS.get(provider.backend)
            if builder is None:
                raise _unsupported(role, provider, list(_CHAT_BACKENDS))
        built[role] = builder(provider, limiter)

    return Providers(**built)
