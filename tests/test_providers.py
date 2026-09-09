"""
test_providers.py
=================
Offline tests for the provider seam: the role-agnostic surface
(:mod:`phoenix_rag.providers.base`) and the composition root
(:mod:`phoenix_rag.providers.registry`).

Nothing here calls an API. Constructing the SDK/LangChain client objects does no
network I/O, so a fake key is enough to build the whole bundle; the actual
``chat``/``embed`` calls are validated against a live backend elsewhere.

These tests deliberately reach into a few private attributes (``_client``,
``_limiter``, ``_model``). The limiter-sharing invariant is the whole reason the
registry exists, and it is not observable from the public surface -- so the test
asserts it on the internals it is meant to guarantee.
"""

from __future__ import annotations

import os
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings

from phoenix_rag.config import AppConfig, UnsupportedBackendError
from phoenix_rag.providers import ChatProvider, build_providers
from phoenix_rag.providers.mistral import MistralChatProvider


class BuildProvidersTests(unittest.TestCase):
    def setUp(self) -> None:
        # A fake key so the Mistral roles build. patch.dict restores the real
        # environment (key or no key) after each test.
        self._env = patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key-not-real"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_each_role_gets_a_provider_of_the_right_shape(self):
        providers = build_providers(AppConfig())
        # The embedding role is a LangChain Embeddings (FAISS + Ragas want that);
        # the other three are ChatProviders.
        self.assertIsInstance(providers.embedding, Embeddings)
        self.assertIsInstance(providers.generation, ChatProvider)
        self.assertIsInstance(providers.optimizer, ChatProvider)
        self.assertIsInstance(providers.judge, ChatProvider)

    def test_chat_providers_are_bound_to_their_roles_model(self):
        # A provider is "the optimizer provider", already carrying its model --
        # the defaults name a different model per role, and each must land on the
        # provider for that role.
        providers = build_providers(AppConfig())
        self.assertEqual(providers.generation._model, "mistral-small-latest")
        self.assertEqual(providers.optimizer._model, "mistral-large-latest")
        self.assertEqual(providers.judge._model, "mistral-large-latest")
        self.assertEqual(providers.embedding.settings.embedding_model, "mistral-embed")

    def test_roles_on_one_backend_share_a_single_rate_limiter(self):
        # THE bug fix: with every role on Mistral, all four must pace against one
        # limiter, not four private ones that together triple the real quota.
        providers = build_providers(AppConfig())
        limiter_ids = {
            id(providers.embedding._client._limiter),
            id(providers.generation._client._limiter),
            id(providers.optimizer._client._limiter),
            id(providers.judge._client._limiter),
        }
        self.assertEqual(len(limiter_ids), 1)

    def test_the_shared_limiter_uses_the_tightest_configured_rpm(self):
        # Two roles on one API with different rpm: pace to the lower ceiling.
        config = AppConfig()
        config.providers.embedding.requests_per_minute = 30
        config.providers.generation.requests_per_minute = 90
        providers = build_providers(config)
        self.assertIs(
            providers.embedding._client._limiter,
            providers.generation._client._limiter,
        )
        self.assertEqual(providers.embedding._client._limiter.max_calls, 30)

    def test_an_unwired_backend_is_rejected_naming_the_role(self):
        # A backend with no builder at all -- openai/anthropic/local are wired
        # now, so this uses a name that genuinely has none.
        config = AppConfig()
        config.providers.optimizer.backend = "cohere"
        with self.assertRaises(UnsupportedBackendError) as caught:
            build_providers(config)
        message = str(caught.exception)
        self.assertIn("optimizer", message)
        self.assertIn("cohere", message)

    def test_a_missing_api_key_is_reported(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MISTRAL_API_KEY", None)
            with self.assertRaises(ValueError) as caught:
                build_providers(AppConfig())
        self.assertIn("MISTRAL_API_KEY", str(caught.exception))


class MistralChatProviderTests(unittest.TestCase):
    def test_chat_routes_the_bound_model_and_call_kwargs(self):
        # chat() takes no model -- the provider supplies its bound one, and
        # passes temperature/response_format straight through to the client.
        client = MagicMock()
        client.chat.return_value = "the answer"
        provider = MistralChatProvider(client, model="mistral-large-latest", api_key="k")

        messages = [{"role": "user", "content": "hi"}]
        out = provider.chat(messages, temperature=0.2)

        self.assertEqual(out, "the answer")
        client.chat.assert_called_once_with(
            messages, model="mistral-large-latest", temperature=0.2, response_format=None
        )

    def test_as_langchain_chat_model_returns_a_langchain_chat_model(self):
        # The judge role needs a LangChain object for Ragas; this must produce
        # one without the raw chat() client being involved.
        from langchain_core.language_models.chat_models import BaseChatModel

        provider = MistralChatProvider(MagicMock(), model="mistral-large-latest", api_key="k")
        llm = provider.as_langchain_chat_model()
        self.assertIsInstance(llm, BaseChatModel)


class LangChainChatProviderTests(unittest.TestCase):
    """The generic provider behind openai / anthropic / local chat.

    Driven by a fake ``BaseChatModel`` so nothing here needs a vendor SDK or a
    network. The provider's own job is message conversion, limiter pacing, retry,
    and handing back a model for the judge -- that is what these cover.
    """

    def _provider(self, fake_model, *, max_retries=3, backoff=0.0):
        from phoenix_rag.providers.langchain_backends import LangChainChatProvider
        from phoenix_rag.providers.ratelimit import RateLimiter

        self.temperatures: list[float] = []

        def factory(temperature):
            self.temperatures.append(temperature)
            return fake_model

        return LangChainChatProvider(
            factory,
            limiter=RateLimiter(1000),
            model="fake-model",
            max_retries=max_retries,
            base_backoff_seconds=backoff,
            label="fake:fake-model",
        )

    def test_chat_converts_messages_and_returns_text(self):
        from langchain_core.messages import HumanMessage, SystemMessage

        captured = {}

        class FakeModel:
            def invoke(self, messages):
                captured["messages"] = messages
                return SimpleNamespace(content="the answer")

        provider = self._provider(FakeModel())
        out = provider.chat(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            temperature=0.2,
        )

        self.assertEqual(out, "the answer")
        messages = captured["messages"]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(messages[0].content, "sys")
        self.assertEqual(messages[1].content, "hi")
        # The bound model is built at the call's temperature, not a fixed one.
        self.assertEqual(self.temperatures, [0.2])

    def test_repeated_temperature_reuses_one_model(self):
        class FakeModel:
            def invoke(self, messages):
                return SimpleNamespace(content="x")

        provider = self._provider(FakeModel())
        provider.chat([{"role": "user", "content": "a"}], temperature=0.2)
        provider.chat([{"role": "user", "content": "b"}], temperature=0.2)
        self.assertEqual(self.temperatures, [0.2])  # built once, then reused

    def test_content_list_is_flattened_to_text(self):
        class FakeModel:
            def invoke(self, messages):
                return SimpleNamespace(
                    content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
                )

        provider = self._provider(FakeModel())
        self.assertEqual(provider.chat([{"role": "user", "content": "hi"}]), "ab")

    def test_chat_retries_then_succeeds(self):
        class FlakyModel:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient")
                return SimpleNamespace(content="recovered")

        model = FlakyModel()
        provider = self._provider(model, max_retries=3, backoff=0.0)
        self.assertEqual(provider.chat([{"role": "user", "content": "hi"}]), "recovered")
        self.assertEqual(model.calls, 2)

    def test_chat_raises_after_max_retries_naming_the_backend(self):
        class DeadModel:
            def invoke(self, messages):
                raise RuntimeError("nope")

        provider = self._provider(DeadModel(), max_retries=2, backoff=0.0)
        with self.assertRaises(RuntimeError) as caught:
            provider.chat([{"role": "user", "content": "hi"}])
        self.assertIn("fake:fake-model", str(caught.exception))

    def test_as_langchain_chat_model_uses_the_factory(self):
        sentinel = object()
        provider = self._provider(sentinel)
        self.assertIs(provider.as_langchain_chat_model(temperature=0.0), sentinel)


class RateLimitedEmbeddingsTests(unittest.TestCase):
    def test_delegates_to_inner_and_paces_each_public_call(self):
        from phoenix_rag.providers.langchain_backends import RateLimitedEmbeddings

        inner = MagicMock()
        inner.embed_documents.return_value = [[1.0], [2.0]]
        inner.embed_query.return_value = [3.0]
        limiter = MagicMock()

        emb = RateLimitedEmbeddings(inner, limiter=limiter)
        self.assertEqual(emb.embed_documents(["a", "b"]), [[1.0], [2.0]])
        self.assertEqual(emb.embed_query("q"), [3.0])

        inner.embed_documents.assert_called_once_with(["a", "b"])
        inner.embed_query.assert_called_once_with("q")
        self.assertEqual(limiter.acquire.call_count, 2)  # one acquire per call


def _all_openai_config() -> AppConfig:
    """An AppConfig with every role on OpenAI (a valid all-OpenAI bundle)."""
    config = AppConfig()
    for role in ("embedding", "generation", "optimizer", "judge"):
        provider = getattr(config.providers, role)
        provider.backend = "openai"
        provider.api_key_env = "OPENAI_API_KEY"
        provider.model = (
            "text-embedding-3-small" if role == "embedding" else "gpt-4o-mini"
        )
    return config


@unittest.skipIf(
    find_spec("langchain_openai") is None,
    "langchain_openai is not installed",
)
class OpenAIBackendTests(unittest.TestCase):
    """Real construction of the OpenAI backend (no network: building the client
    objects does no I/O, same as the Mistral tests)."""

    def setUp(self) -> None:
        self._env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_all_openai_bundle_has_the_right_shapes(self):
        from phoenix_rag.providers.langchain_backends import (
            LangChainChatProvider,
            RateLimitedEmbeddings,
        )

        providers = build_providers(_all_openai_config())
        self.assertIsInstance(providers.embedding, RateLimitedEmbeddings)
        self.assertIsInstance(providers.generation, LangChainChatProvider)
        self.assertIsInstance(providers.optimizer, LangChainChatProvider)
        self.assertIsInstance(providers.judge, LangChainChatProvider)

    def test_judge_hands_out_a_langchain_chat_model(self):
        from langchain_core.language_models.chat_models import BaseChatModel

        providers = build_providers(_all_openai_config())
        self.assertIsInstance(providers.judge.as_langchain_chat_model(), BaseChatModel)

    def test_all_openai_roles_share_one_rate_limiter(self):
        # The limiter-sharing invariant holds for OpenAI just as for Mistral:
        # four roles, one API identity, one limiter.
        providers = build_providers(_all_openai_config())
        limiter_ids = {
            id(providers.embedding._limiter),
            id(providers.generation._limiter),
            id(providers.optimizer._limiter),
            id(providers.judge._limiter),
        }
        self.assertEqual(len(limiter_ids), 1)

    def test_missing_openai_key_is_reported(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertRaises(ValueError) as caught:
                build_providers(_all_openai_config())
        self.assertIn("OPENAI_API_KEY", str(caught.exception))


@unittest.skipIf(
    find_spec("langchain_openai") is None,
    "langchain_openai is not installed",
)
class OpenAICompatibleBuilderTests(unittest.TestCase):
    """Chat builders exercised directly, to cover the base_url / key
    permutations without needing a full four-role bundle."""

    def _provider(self, **overrides):
        from phoenix_rag.config.schema import ProviderConfig

        return ProviderConfig(**overrides)

    def _limiter(self):
        from phoenix_rag.providers.ratelimit import RateLimiter

        return RateLimiter(1000)

    def test_local_chat_requires_base_url(self):
        from phoenix_rag.providers.langchain_backends import build_local_chat

        with self.assertRaises(ValueError) as caught:
            build_local_chat(
                self._provider(backend="local", model="llama3"), self._limiter()
            )
        self.assertIn("base_url", str(caught.exception))

    def test_local_chat_builds_without_a_key(self):
        from langchain_core.language_models.chat_models import BaseChatModel
        from phoenix_rag.providers.langchain_backends import (
            LangChainChatProvider,
            build_local_chat,
        )

        provider = build_local_chat(
            self._provider(
                backend="local", model="llama3", base_url="http://localhost:11434/v1"
            ),
            self._limiter(),
        )
        self.assertIsInstance(provider, LangChainChatProvider)
        self.assertIsInstance(provider.as_langchain_chat_model(), BaseChatModel)

    def test_openai_chat_without_key_or_base_url_is_rejected(self):
        from phoenix_rag.providers.langchain_backends import build_openai_chat

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertRaises(ValueError):
                build_openai_chat(
                    self._provider(
                        backend="openai",
                        model="gpt-4o-mini",
                        api_key_env="OPENAI_API_KEY",
                    ),
                    self._limiter(),
                )

    def test_openai_chat_with_base_url_needs_no_key(self):
        # "openai" pointed at a self-hosted OpenAI-compatible server: valid keyless.
        from phoenix_rag.providers.langchain_backends import (
            LangChainChatProvider,
            build_openai_chat,
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            provider = build_openai_chat(
                self._provider(
                    backend="openai",
                    model="local-model",
                    base_url="http://localhost:8000/v1",
                ),
                self._limiter(),
            )
        self.assertIsInstance(provider, LangChainChatProvider)


class MissingBackendDependencyTests(unittest.TestCase):
    """A selected backend whose vendor package is absent must fail with a
    "pip install 'phoenix-rag[<extra>]'" hint, not an opaque ImportError."""

    def _provider(self, **overrides):
        from phoenix_rag.config.schema import ProviderConfig

        return ProviderConfig(**overrides)

    def _limiter(self):
        from phoenix_rag.providers.ratelimit import RateLimiter

        return RateLimiter(1000)

    @unittest.skipIf(
        find_spec("langchain_anthropic") is not None,
        "langchain_anthropic is installed; the install-hint path is unreachable",
    )
    def test_anthropic_without_its_package_points_at_the_extra(self):
        from phoenix_rag.providers.langchain_backends import build_anthropic_chat

        with self.assertRaises(ImportError) as caught:
            build_anthropic_chat(
                self._provider(
                    backend="anthropic",
                    model="claude-3-5-sonnet-latest",
                    api_key_env="ANTHROPIC_API_KEY",
                ),
                self._limiter(),
            )
        self.assertIn("phoenix-rag[anthropic]", str(caught.exception))

    @unittest.skipIf(
        find_spec("langchain_huggingface") is not None,
        "langchain_huggingface is installed; the install-hint path is unreachable",
    )
    def test_local_embedding_without_its_package_points_at_the_extra(self):
        from phoenix_rag.providers.langchain_backends import build_local_embedding

        with self.assertRaises(ImportError) as caught:
            build_local_embedding(
                self._provider(
                    backend="local", model="sentence-transformers/all-MiniLM-L6-v2"
                ),
                self._limiter(),
            )
        self.assertIn("phoenix-rag[local]", str(caught.exception))


class MissingApiKeysTests(unittest.TestCase):
    """ProvidersConfig.missing_api_keys: the backend-agnostic key check the UIs
    use instead of the Mistral-only AppConfig.mistral view."""

    def test_flags_every_role_whose_key_env_is_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MISTRAL_API_KEY", None)
            missing = AppConfig().providers.missing_api_keys()
        self.assertEqual(
            {role for role, _ in missing},
            {"embedding", "generation", "optimizer", "judge"},
        )
        self.assertTrue(all(env == "MISTRAL_API_KEY" for _, env in missing))

    def test_nothing_missing_when_the_key_is_set(self):
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "x"}):
            self.assertEqual(AppConfig().providers.missing_api_keys(), [])

    def test_a_local_role_with_no_key_env_is_not_flagged(self):
        config = AppConfig()
        config.providers.generation.backend = "local"
        config.providers.generation.api_key_env = None
        config.providers.generation.base_url = "http://localhost:11434/v1"
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "x"}):
            missing = config.providers.missing_api_keys()
        self.assertNotIn("generation", {role for role, _ in missing})


if __name__ == "__main__":
    unittest.main()
