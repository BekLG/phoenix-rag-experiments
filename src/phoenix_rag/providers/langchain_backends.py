"""
providers/langchain_backends.py
===============================
The backends that reach their model through a LangChain integration package:
``openai``, ``anthropic``, ``deepseek``, and ``local`` (an OpenAI-compatible
endpoint served by an in-process runner such as Ollama, vLLM or LM Studio).

WHY ONE MODULE FOR SEVERAL BACKENDS
-----------------------------------
Mistral gets its own module because it wraps the raw ``mistralai`` SDK. These do
not need a raw SDK: LangChain already ships a chat model for each
(``ChatOpenAI``, ``ChatAnthropic``), and such an object is exactly what the judge
role must hand to Ragas. So one :class:`LangChainChatProvider` -- a
``BaseChatModel`` plus the shared rate limiter -- serves all of them, and the
per-backend code here is just a factory that builds the right ``BaseChatModel``.
``local`` and ``deepseek`` chat are both ``openai`` chat pointed at a different
``base_url``: the OpenAI client speaks to any server implementing the same wire
format. They differ only in whether a key is required and whether the endpoint
has a sensible default.

LAZY, PER-BACKEND IMPORTS
-------------------------
Each vendor package is imported *inside* its builder, never at module load. A
Mistral-only install (Mistral is the one hard dependency) therefore pays for no
SDK it will not use, and selecting a backend whose package is absent fails with
a ``pip install 'phoenix-rag[<extra>]'`` hint instead of an opaque ImportError
deep in a call stack. This module itself imports only ``langchain_core`` (already
a hard dependency), so :mod:`phoenix_rag.providers.registry` can import these
builders eagerly and still stay cheap.

RATE LIMITING
-------------
``chat`` text goes through the shared :class:`RateLimiter`, matching Mistral. The
judge's LangChain object (``as_langchain_chat_model``) does not -- it is a
separate object paced by Ragas's own RunConfig, exactly as the Mistral judge is.
In-process embeddings (``local``) have no API quota, so they are deliberately not
paced.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from phoenix_rag.config.schema import ProviderConfig
from phoenix_rag.providers.base import ChatProvider, EmbeddingProvider
from phoenix_rag.providers.ratelimit import RateLimiter

logger = logging.getLogger("phoenix_rag.providers.langchain")

# The role strings the callers use, mapped to LangChain's message classes.
_ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def _to_lc_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert the ``{"role", "content"}`` dicts the callers use to LangChain."""
    converted: list[BaseMessage] = []
    for message in messages:
        cls = _ROLE_TO_MESSAGE.get(message.get("role", "user"), HumanMessage)
        converted.append(cls(content=message["content"]))
    return converted


def _content_to_text(content: Any) -> str:
    """Flatten ``AIMessage.content`` to a string.

    For text prompts every backend returns a string, but the type also allows a
    list of content blocks; join the text of those so a caller that expects a
    string never gets a list back.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# temperature -> BaseChatModel
ModelFactory = Callable[[float], Any]


class LangChainChatProvider(ChatProvider):
    """A :class:`ChatProvider` over any LangChain ``BaseChatModel``.

    ``chat`` returns reply text, paced by the shared rate limiter and retried
    with exponential backoff -- the same guarantees :class:`MistralClient` gives,
    so generation / question-gen / summarizer / optimizer behave identically
    whichever backend serves them. ``as_langchain_chat_model`` returns a fresh
    ``BaseChatModel`` for the judge role.

    The provider is bound to one model; ``chat`` takes no model argument. Built
    models are cached by temperature so a repeated temperature reuses the
    underlying HTTP client rather than constructing a new one each call.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        limiter: RateLimiter,
        model: str,
        max_retries: int,
        base_backoff_seconds: float,
        label: str,
    ):
        self._model_factory = model_factory
        self._limiter = limiter
        self._model_name = model
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds
        self._label = label
        self._models: dict[float, Any] = {}

    def _model_at(self, temperature: float):
        model = self._models.get(temperature)
        if model is None:
            model = self._model_factory(temperature)
            self._models[temperature] = model
        return model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        response_format: dict | None = None,
    ) -> str:
        model = self._model_at(temperature)
        if response_format is not None:
            # Best-effort: OpenAI-compatible models accept this; one that does not
            # will raise its own error. No caller passes it today.
            model = model.bind(response_format=response_format)
        lc_messages = _to_lc_messages(messages)
        return self._with_retry(
            lambda: _content_to_text(model.invoke(lc_messages).content)
        )

    def as_langchain_chat_model(self, *, temperature: float = 0.0):
        return self._model_factory(temperature)

    def _with_retry(self, call):
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            self._limiter.acquire()
            try:
                return call()
            except Exception as exc:  # noqa: BLE001 - vendor SDKs raise various types
                last_error = exc
                backoff = self._base_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "%s call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    self._label, attempt, self._max_retries, exc, backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(
            f"{self._label} API call failed after {self._max_retries} attempts"
        ) from last_error


class RateLimitedEmbeddings(Embeddings):
    """Pace any LangChain ``Embeddings`` through a shared :class:`RateLimiter`.

    Mistral routes embeddings through the same limiter as its chat calls; this
    gives the OpenAI embedding backend that property too, so an all-OpenAI run
    paces embedding + chat against one shared budget rather than two independent
    ones.

    One ``acquire`` per public call. LangChain's embedding clients may split a
    large batch into several HTTP requests internally, so this is a coarser bound
    than Mistral's per-32-item pacing -- enough to respect the shared identity
    without reaching into vendor batching.
    """

    def __init__(self, inner: Embeddings, *, limiter: RateLimiter):
        self._inner = inner
        self._limiter = limiter

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._limiter.acquire()
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self._limiter.acquire()
        return self._inner.embed_query(text)


# --------------------------------------------------------------------------
# Lazy dependency loading
# --------------------------------------------------------------------------

def _require(module: str, *, backend: str, extra: str):
    """Import a vendor package, or raise a "pip install" hint if it is absent."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            f"The {backend!r} backend needs the {module!r} package, which is not "
            f"installed. Install it with:  pip install 'phoenix-rag[{extra}]'"
        ) from exc


def _missing_key_error(backend: str, provider: ProviderConfig) -> ValueError:
    return ValueError(
        f"No API key for the {backend!r} backend. Set the environment variable "
        f"named by this role's providers.<role>.api_key_env "
        f"({provider.api_key_env or 'unset'}); the value lives in .env, never in "
        "the YAML."
    )


# --------------------------------------------------------------------------
# OpenAI-compatible chat (serves both the "openai" and "local" backends)
# --------------------------------------------------------------------------

def _openai_compatible_chat(
    provider: ProviderConfig,
    limiter: RateLimiter,
    *,
    backend: str,
    extra: str,
    require_key: bool,
    default_base_url: str | None = None,
) -> ChatProvider:
    _require("langchain_openai", backend=backend, extra=extra)
    from langchain_openai import ChatOpenAI

    api_key = provider.resolve_api_key()
    if require_key and not api_key:
        raise _missing_key_error(backend, provider)

    # A hosted OpenAI-compatible vendor has one known endpoint, so the config
    # need not spell it out; an explicit base_url still wins, which is what lets
    # a role point at a proxy or a region-specific host.
    base_url = provider.base_url or default_base_url
    # The OpenAI client requires a non-empty key even when the endpoint (a local
    # server) ignores it, so a keyless local run still needs a placeholder.
    effective_key = api_key or "not-needed-for-local"

    def factory(temperature: float):
        kwargs: dict[str, Any] = dict(
            model=provider.model, api_key=effective_key, temperature=temperature
        )
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    return LangChainChatProvider(
        factory,
        limiter=limiter,
        model=provider.model,
        max_retries=provider.max_retries,
        base_backoff_seconds=provider.base_backoff_seconds,
        label=f"{backend}:{provider.model}",
    )


def build_openai_chat(provider: ProviderConfig, limiter: RateLimiter) -> ChatProvider:
    # Hosted OpenAI needs a key. If base_url is set, this "openai" backend is
    # pointed at a self-hosted OpenAI-compatible server, where the key is
    # optional -- so only require one when talking to the real API.
    return _openai_compatible_chat(
        provider,
        limiter,
        backend="openai",
        extra="openai",
        require_key=provider.base_url is None,
    )


def build_local_chat(provider: ProviderConfig, limiter: RateLimiter) -> ChatProvider:
    if not provider.base_url:
        raise ValueError(
            "The 'local' backend needs providers.<role>.base_url set to your "
            "OpenAI-compatible endpoint (e.g. http://localhost:11434/v1 for "
            "Ollama, or your vLLM / LM Studio URL)."
        )
    return _openai_compatible_chat(
        provider, limiter, backend="local", extra="local", require_key=False
    )


# --------------------------------------------------------------------------
# DeepSeek chat
# --------------------------------------------------------------------------

# DeepSeek serves the OpenAI wire format from this host, so it needs no vendor
# package of its own -- the [deepseek] extra just pulls langchain-openai.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def build_deepseek_chat(provider: ProviderConfig, limiter: RateLimiter) -> ChatProvider:
    """Chat over DeepSeek's OpenAI-compatible API.

    Chat-only: DeepSeek publishes no embeddings endpoint, so this backend is
    absent from the registry's embedding map and a config naming it for the
    embedding role is rejected there.

    The endpoint defaults to :data:`DEEPSEEK_BASE_URL`, so a role needs only
    ``backend``/``model``/``api_key_env``. A key is always required: unlike
    ``local``, this is a hosted API that authenticates every request.
    """
    return _openai_compatible_chat(
        provider,
        limiter,
        backend="deepseek",
        extra="deepseek",
        require_key=True,
        default_base_url=DEEPSEEK_BASE_URL,
    )


# --------------------------------------------------------------------------
# Anthropic chat
# --------------------------------------------------------------------------

def build_anthropic_chat(provider: ProviderConfig, limiter: RateLimiter) -> ChatProvider:
    _require("langchain_anthropic", backend="anthropic", extra="anthropic")
    from langchain_anthropic import ChatAnthropic

    api_key = provider.resolve_api_key()
    if not api_key:
        raise _missing_key_error("anthropic", provider)

    def factory(temperature: float):
        # max_tokens is the one knob not carried by ProviderConfig. Anthropic
        # requires an output cap and LangChain's default (1024) is low enough to
        # truncate a question-generation batch or an optimizer JSON proposal
        # mid-object; 4096 is a safe ceiling across the Claude models.
        return ChatAnthropic(
            model=provider.model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=4096,
        )

    return LangChainChatProvider(
        factory,
        limiter=limiter,
        model=provider.model,
        max_retries=provider.max_retries,
        base_backoff_seconds=provider.base_backoff_seconds,
        label=f"anthropic:{provider.model}",
    )


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def build_openai_embedding(
    provider: ProviderConfig, limiter: RateLimiter
) -> EmbeddingProvider:
    _require("langchain_openai", backend="openai", extra="openai")
    from langchain_openai import OpenAIEmbeddings

    api_key = provider.resolve_api_key()
    if provider.base_url is None and not api_key:
        raise _missing_key_error("openai", provider)
    effective_key = api_key or "not-needed-for-local"

    kwargs: dict[str, Any] = dict(model=provider.model, api_key=effective_key)
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    inner = OpenAIEmbeddings(**kwargs)
    return RateLimitedEmbeddings(inner, limiter=limiter)


def build_local_embedding(
    provider: ProviderConfig, limiter: RateLimiter
) -> EmbeddingProvider:
    _require("langchain_huggingface", backend="local", extra="local")
    from langchain_huggingface import HuggingFaceEmbeddings

    # Runs in-process via sentence-transformers, so there is no API quota to
    # pace: the shared limiter is intentionally not applied here.
    try:
        return HuggingFaceEmbeddings(model_name=provider.model)
    except ImportError as exc:  # pragma: no cover - only without sentence-transformers
        raise ImportError(
            "The 'local' embedding backend needs sentence-transformers. Install "
            "it with:  pip install 'phoenix-rag[local]'"
        ) from exc
