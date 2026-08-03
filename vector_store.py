"""
vector_store.py
================
FAISS vector store construction and retriever creation.

The vector database is only ever used during retrieval, never during
question generation (see question_generator.py docstring for why).
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger("phoenix_rag.vector_store")


def build_vector_store(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed chunks and build a fresh in-memory FAISS index."""
    logger.info("Building FAISS index from %d chunks", len(chunks))
    return FAISS.from_documents(chunks, embeddings)


def save_vector_store(store: FAISS, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    logger.info("Saved FAISS index to %s", path)


def load_vector_store(path: str | Path, embeddings: Embeddings) -> FAISS:
    path = Path(path)
    logger.info("Loading FAISS index from %s", path)
    return FAISS.load_local(
        str(path), embeddings, allow_dangerous_deserialization=True
    )


def get_retriever(
    store: FAISS,
    top_k: int = 4,
    retriever_type: str = "similarity",
    similarity_threshold: float = 0.0,
) -> VectorStoreRetriever:
    """Build a retriever from a FAISS store using the given search strategy."""
    if retriever_type == "similarity":
        return store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    if retriever_type == "mmr":
        return store.as_retriever(
            search_type="mmr", search_kwargs={"k": top_k, "fetch_k": max(top_k * 4, 20)}
        )

    if retriever_type == "similarity_score_threshold":
        return store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": similarity_threshold},
        )

    raise ValueError(f"Unknown retriever_type: {retriever_type}")
