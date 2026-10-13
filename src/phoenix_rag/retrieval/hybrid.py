"""Reciprocal Rank Fusion of dense and lexical retrieval results."""

from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter

from phoenix_rag.retrieval.base import RetrievalResult, Retriever

logger = logging.getLogger("phoenix_rag.retrieval.hybrid")


class HybridRetriever:
    """Fuse by chunk identity and rank; raw method scores are never combined."""

    def __init__(self, dense: Retriever, bm25: Retriever, top_k: int, rrf_k: int = 60):
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if type(rrf_k) is not int or rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        self._dense = dense
        self._bm25 = bm25
        self._top_k = top_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str) -> list[RetrievalResult]:
        started = perf_counter()
        dense_results = self._dense.retrieve(query)
        bm25_results = self._bm25.retrieve(query)
        hits: dict[str, RetrievalResult] = {}
        ranks: dict[str, dict[str, int | None]] = {}
        for method, results in (("dense_rank", dense_results), ("bm25_rank", bm25_results)):
            for result in results:
                hits.setdefault(result.chunk_id, result)
                method_ranks = ranks.setdefault(
                    result.chunk_id, {"dense_rank": None, "bm25_rank": None},
                )
                previous = method_ranks[method]
                # Even a duplicate within a list contributes only its best rank.
                method_ranks[method] = min(previous, result.rank) if previous is not None else result.rank

        fused = []
        for chunk_id, result in hits.items():
            method_ranks = ranks[chunk_id]
            score = sum(1.0 / (self._rrf_k + rank)
                        for rank in method_ranks.values() if rank is not None)
            fused.append(replace(
                result, retrieval_method="hybrid", retrieval_score=score,
                extras={**method_ranks, "rrf_score": score},
            ))
        fused.sort(key=lambda result: (
            -result.retrieval_score,
            min(rank for rank in ranks[result.chunk_id].values() if rank is not None),
            result.chunk_id,
        ))
        results = [replace(result, rank=rank)
                   for rank, result in enumerate(fused[:self._top_k], start=1)]
        logger.debug(
            "Retrieval strategy=hybrid dense_candidates=%d bm25_candidates=%d "
            "fused_candidates=%d results=%d latency_ms=%.3f dense_top_ids=%s "
            "bm25_top_ids=%s fused_top_ids=%s final_chunk_ids=%s",
            len(dense_results), len(bm25_results), len(fused), len(results),
            (perf_counter() - started) * 1000,
            [result.chunk_id for result in dense_results],
            [result.chunk_id for result in bm25_results],
            [result.chunk_id for result in results],
            [result.chunk_id for result in results],
        )
        return results
