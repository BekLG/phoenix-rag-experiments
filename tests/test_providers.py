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
        config = AppConfig()
        config.providers.optimizer.backend = "openai"
        with self.assertRaises(UnsupportedBackendError) as caught:
            build_providers(config)
        message = str(caught.exception)
        self.assertIn("optimizer", message)
        self.assertIn("openai", message)

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


if __name__ == "__main__":
    unittest.main()
