"""FAISS retrieval adapted to the common result contract."""

from __future__ import annotations

import logging
from time import perf_counter

from langchain_community.vectorstores import FAISS

from phoenix_rag.config import RetrievalConfig
from phoenix_rag.core.vector_store import ensure_store_chunk_ids, get_retriever
from phoenix_rag.retrieval.base import RetrievalResult

logger = logging.getLogger("phoenix_rag.retrieval.dense")


class DenseRetriever:
    """Preserve the existing FAISS search parameters, filtering and hit order.

    ``retrieval_method`` is the effective dense search type. ``retrieval_score``:

    * similarity: raw FAISS score (squared L2 distance, lower is better, for the
      indexes this repo builds; a custom store may use another distance metric).
    * similarity_score_threshold: LangChain relevance score (higher is better),
      after the SAME relevance conversion and threshold filter as before.
    * mmr: None, since the existing MMR retriever returns only documents.

    Similarity's old Document-only API delegates to similarity_search_with_score;
    threshold retrieval already calls similarity_search_with_relevance_scores.
    Keeping their scores requires neither a second query nor a new ordering.
    """

    def __init__(self, vector_store: FAISS, retrieval_config: RetrievalConfig):
        self._store = vector_store
        # Also cover a legacy FAISS object loaded directly by an external caller.
        ensure_store_chunk_ids(vector_store)
        method = retrieval_config.retriever_type
        if method not in ("mmr", "similarity_score_threshold"):
            method = "similarity"  # Preserve RagPipeline's existing fallback.
        self._retriever = get_retriever(
            vector_store,
            top_k=retrieval_config.top_k,
            retriever_type=method,
            similarity_threshold=retrieval_config.similarity_threshold,
        )

    def retrieve(self, query: str) -> list[RetrievalResult]:
        started = perf_counter()
        method = self._retriever.search_type
        kwargs = self._retriever.search_kwargs
        if method == "similarity":
            scored = self._store.similarity_search_with_score(query, **kwargs)
        elif method == "similarity_score_threshold":
            scored = self._store.similarity_search_with_relevance_scores(query, **kwargs)
        else:
            scored = [(document, None) for document in self._retriever.invoke(query)]

        results = [
            RetrievalResult(
                chunk_id=document.metadata["chunk_id"],
                document_id=document.metadata["document_id"],
                text=document.page_content,
                metadata=dict(document.metadata),
                rank=rank,
                retrieval_score=float(score) if score is not None else None,
                retrieval_method=method,
            )
            for rank, (document, score) in enumerate(scored, start=1)
        ]
        logger.debug(
            "Retrieval method=%s results=%d latency_ms=%.3f "
            "strategy=dense candidates=%d final_chunk_ids=%s",
            method, len(results), (perf_counter() - started) * 1000,
            len(results), [result.chunk_id for result in results],
        )
        return results
