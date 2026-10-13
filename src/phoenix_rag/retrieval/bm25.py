"""BM25 over the canonical FAISS docstore, with a shared in-memory index.

Cache scope is the FAISS store (hence its corpus and embedding variant), then
chunk_size/chunk_overlap, just as for FAISS. Retrieval strategy, top_k and RRF
settings do not affect this index. A snapshot of chunk IDs and text detects
corpus extension, removal or replacement, including changes of equal size.
Rebuilds always use the entire docstore, never just the newly added document.

The cache is weakly owned by the store and is not persisted: a fresh store load
builds once on first BM25 construction, and pipelines sharing it reuse the index.
No BM25 pickle or new disk cache format is involved.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from weakref import WeakKeyDictionary

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from phoenix_rag.config import RetrievalConfig
from phoenix_rag.core.vector_store import ensure_store_chunk_ids
from phoenix_rag.retrieval.base import RetrievalResult

logger = logging.getLogger("phoenix_rag.retrieval.bm25")
_WORDS = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Deterministic Unicode word tokens, shared by corpus and query."""
    return _WORDS.findall(text.lower())


@dataclass
class _Index:
    signature: tuple[tuple[str, str], ...]
    chunks: list[Document]
    model: Any


_INDEXES: WeakKeyDictionary[FAISS, dict[tuple[int, int], _Index]] = WeakKeyDictionary()


def _build_index(chunks: list[Document], signature: tuple[tuple[str, str], ...]) -> _Index:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "BM25 and hybrid retrieval require rank_bm25. "
            "Install it with: pip install 'phoenix-rag[retrieval]'"
        ) from exc

    corpus = [tokenize(chunk.page_content) for chunk in chunks]
    # rank_bm25 divides by zero for an empty corpus or one with no word tokens.
    model = BM25Okapi(corpus) if any(corpus) else None
    return _Index(signature=signature, chunks=chunks, model=model)


def _get_index(store: FAISS, chunk_params: tuple[int, int]) -> _Index:
    ensure_store_chunk_ids(store)
    chunks = [
        store.docstore.search(docstore_id)
        for _, docstore_id in sorted(store.index_to_docstore_id.items())
    ]
    signature = tuple((chunk.metadata["chunk_id"], chunk.page_content) for chunk in chunks)
    variants = _INDEXES.setdefault(store, {})
    index = variants.get(chunk_params)
    if index is None or index.signature != signature:
        index = _build_index(chunks, signature)
        variants[chunk_params] = index
    return index


class BM25Retriever:
    """Lexical ranking; dense retriever_type and similarity_threshold are ignored.

    Zero-score hits are retained, with ties ordered by chunk_id. This also gives
    deterministic results for a query with no known words or no word tokens.
    """

    def __init__(self, vector_store: FAISS, retrieval_config: RetrievalConfig):
        self._store = vector_store
        self._top_k = retrieval_config.top_k
        self._chunk_params = (retrieval_config.chunk_size, retrieval_config.chunk_overlap)
        _get_index(self._store, self._chunk_params)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        started = perf_counter()
        index = _get_index(self._store, self._chunk_params)
        tokens = tokenize(query)
        scores = (index.model.get_scores(tokens) if index.model is not None
                  else [0.0] * len(index.chunks))
        ordered = sorted(
            zip(index.chunks, scores),
            key=lambda hit: (-float(hit[1]), hit[0].metadata["chunk_id"]),
        )[:self._top_k]
        results = [
            RetrievalResult(
                chunk_id=chunk.metadata["chunk_id"],
                document_id=chunk.metadata["document_id"],
                text=chunk.page_content,
                metadata=dict(chunk.metadata),
                rank=rank,
                retrieval_score=float(score),
                retrieval_method="bm25",
            )
            for rank, (chunk, score) in enumerate(ordered, start=1)
        ]
        logger.debug(
            "Retrieval strategy=bm25 candidates=%d results=%d latency_ms=%.3f "
            "final_chunk_ids=%s",
            len(index.chunks), len(results), (perf_counter() - started) * 1000,
            [result.chunk_id for result in results],
        )
        return results
