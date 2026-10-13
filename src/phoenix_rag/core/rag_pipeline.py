"""
rag_pipeline.py
================
The actual RAG pipeline being optimized: retrieve -> build prompt -> generate.

Kept intentionally simple/stateless so it can be re-run cheaply for every
question in the benchmark, for every configuration the optimizer tries.

Generation calls go through an injected ChatProvider (providers/) rather than
a raw SDK client, so rate limiting and retry/backoff actually apply here.
Without this, a 429 mid-benchmark crashes the whole experiment run instead
of backing off and retrying -- this is not hypothetical, it's what actually
happened in a real run (rate limit hit on request 7/10 of an iteration,
unhandled, process died).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from phoenix_rag.config import RetrievalConfig
from phoenix_rag.providers.base import ChatProvider
from phoenix_rag.retrieval import build_retriever
from phoenix_rag.retrieval.base import RetrievalResult, Retriever

logger = logging.getLogger("phoenix_rag.rag_pipeline")


@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[str]
    # Chunk metadata for each retrieved context, in the same order as `contexts`.
    # Only `contexts` is used for evaluation -- this exists so an operator asking
    # the RAG interactively can see WHICH document each context came from, which
    # is the difference between a checkable answer and an unverifiable one once a
    # single index spans several documents (corpus.py stamps doc_id/doc_label).
    sources: list[dict] = field(default_factory=list)
    retrieval_results: list[RetrievalResult] | None = None


class RagPipeline:
    """Retrieve -> build prompt -> generate, parameterized by RetrievalConfig."""

    def __init__(
        self,
        vector_store: FAISS,
        generation: ChatProvider,
        retrieval_config: RetrievalConfig,
    ):
        self.vector_store = vector_store
        self.retrieval_config = retrieval_config

        # Rate-limited, retrying generation provider -- see module docstring.
        self._generation = generation

        self._retriever: Retriever = build_retriever(retrieval_config, vector_store)

    def retrieve(self, question: str) -> list[Document]:
        """Keep the existing Document-returning convenience API for callers."""
        return [
            Document(page_content=result.text, metadata=dict(result.metadata))
            for result in self._retriever.retrieve(question)
        ]

    def build_prompt(self, question: str, contexts: list[str]) -> str:
        joined_context = "\n\n".join(contexts)
        return self.retrieval_config.prompt_template.format(
            context=joined_context, question=question
        )

    def answer(self, question: str) -> RagResult:
        results = self._retriever.retrieve(question)
        contexts = [result.text for result in results]
        prompt = self.build_prompt(question, contexts)

        # The ChatProvider handles rate limiting + exponential-backoff retry
        # inside its backend client (see providers/), and returns the answer
        # text directly rather than a raw SDK response object.
        answer_text = self._generation.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return RagResult(
            question=question,
            answer=answer_text,
            contexts=contexts,
            sources=[dict(result.metadata) for result in results],
            retrieval_results=results,
        )

    def answer_many(self, questions: list[str]) -> list[RagResult]:
        results = []
        for i, q in enumerate(questions, start=1):
            logger.info("Answering question %d/%d", i, len(questions))
            results.append(self.answer(q))
        return results
