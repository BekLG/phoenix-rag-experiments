"""
embeddings.py
=============
LangChain-compatible embeddings class backed by Gemini's embedding model
(gemini-embedding-001), routed through the shared rate-limited GeminiClient.

This is the highest-frequency Gemini call site in the project -- every
FAISS index rebuild and every retrieval call goes through here -- so it's
also the one most likely to actually hit the per-minute rate limit in
practice.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from config import GeminiSettings
from gemini_client import GeminiClient

logger = logging.getLogger("phoenix_rag.embeddings")

# Batch size for embedding requests. Verify against Gemini's actual
# per-request batch limit for your model/tier if you hit a 400 here.
_MAX_BATCH = 32


class GeminiEmbeddings(Embeddings):
    """Adapter so FAISS / LangChain can use Gemini embeddings directly."""

    def __init__(self, settings: GeminiSettings | None = None):
        self.settings = settings or GeminiSettings()
        self._client = GeminiClient(self.settings)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _MAX_BATCH):
            batch = texts[i : i + _MAX_BATCH]
            logger.debug("Embedding batch %d-%d of %d", i, i + len(batch), len(texts))
            vectors.extend(self._client.embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed([text])[0]