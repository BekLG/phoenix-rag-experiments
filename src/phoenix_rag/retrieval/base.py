"""The common retrieval result and backend-independent retriever contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RetrievalResult:
    """One hit, ranked from 1 in the retriever's original order.

    Score semantics belong to the retrieval method; scores from different
    methods need not be comparable. Metadata and extras are per-result dicts;
    freezing prevents field reassignment, not mutation inside those dicts.
    """

    chunk_id: str
    document_id: str
    text: str
    metadata: dict
    rank: int
    retrieval_score: float | None
    retrieval_method: str
    reranker_score: float | None = None
    extras: dict = field(default_factory=dict)


class Retriever(Protocol):
    """A configured retriever returning ordered hits for a query."""

    def retrieve(self, query: str) -> list[RetrievalResult]:
        ...
