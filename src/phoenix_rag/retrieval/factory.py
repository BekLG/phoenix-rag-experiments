"""Construct retrieval strategies without making BM25 a dense dependency."""

from __future__ import annotations

from typing import TYPE_CHECKING

from phoenix_rag.retrieval.base import Retriever

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS
    from phoenix_rag.config import RetrievalConfig


def build_retriever(config: RetrievalConfig, vector_store: FAISS) -> Retriever:
    """Use one canonical store for every strategy; top_k is the final budget."""
    # Configs are mutable, so also validate at this public construction boundary.
    config.validate()
    if config.retrieval_strategy == "dense":
        from phoenix_rag.retrieval.dense import DenseRetriever

        return DenseRetriever(vector_store, config)

    from phoenix_rag.retrieval.bm25 import BM25Retriever

    if config.retrieval_strategy == "bm25":
        return BM25Retriever(vector_store, config)

    from phoenix_rag.retrieval.dense import DenseRetriever
    from phoenix_rag.retrieval.hybrid import HybridRetriever

    dense_k = config.hybrid_dense_k
    bm25_k = config.hybrid_bm25_k
    default_k = max(2 * config.top_k, 10)
    return HybridRetriever(
        dense=DenseRetriever(vector_store, config.copy_with(
            top_k=default_k if dense_k is None else dense_k,
        )),
        bm25=BM25Retriever(vector_store, config.copy_with(
            top_k=default_k if bm25_k is None else bm25_k,
        )),
        top_k=config.top_k,
        rrf_k=config.rrf_k,
    )
