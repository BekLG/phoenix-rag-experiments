"""
embeddings.py
=============
LangChain-compatible embeddings class backed by Mistral's `mistral-embed`
model, routed through the shared rate-limited MistralClient.

This is the highest-frequency Mistral call site in the project -- every
FAISS index rebuild and every retrieval call goes through here -- so it's
also the one most likely to actually hit the per-minute rate limit in
practice. Routing through MistralClient (rather than a raw SDK client)
means those calls get rate-limited proactively and retried with backoff
on a 429, instead of raising and crashing the run.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from config import MistralSettings
from mistral_client import MistralClient

logger = logging.getLogger("phoenix_rag.embeddings")

# Mistral's embedding endpoint accepts a max batch size; chunk larger
# input lists to stay under it.
_MAX_BATCH = 32


class MistralEmbeddings(Embeddings):
    """Adapter so FAISS / LangChain can use Mistral embeddings directly."""

    def __init__(self, settings: MistralSettings | None = None):
        self.settings = settings or MistralSettings()
        self._client = MistralClient(self.settings)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _MAX_BATCH):
            batch = texts[i : i + _MAX_BATCH]
            logger.debug("Embedding batch %d-%d of %d", i, i + len(batch), len(texts))
            # MistralClient.embed() handles rate limiting + retry internally
            # and returns a plain list[list[float]] already in input order.
            vectors.extend(self._client.embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed([text])[0]