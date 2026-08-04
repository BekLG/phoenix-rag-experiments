"""
rag_pipeline.py
================
The actual RAG pipeline being optimized: retrieve -> build prompt -> generate.

Kept intentionally simple/stateless so it can be re-run cheaply for every
question in the benchmark, for every configuration the optimizer tries.

Generation calls go through MistralClient (mistral_client.py) rather than
a raw SDK client, so rate limiting and retry/backoff actually apply here.
Without this, a 429 mid-benchmark crashes the whole experiment run instead
of backing off and retrying -- this is not hypothetical, it's what actually
happened in a real run (rate limit hit on request 7/10 of an iteration,
unhandled, process died).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import MistralSettings, RetrievalConfig
from mistral_client import MistralClient

logger = logging.getLogger("phoenix_rag.rag_pipeline")


@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[str]


class RagPipeline:
    """Retrieve -> build prompt -> generate, parameterized by RetrievalConfig."""

    def __init__(
        self,
        vector_store: FAISS,
        mistral_settings: MistralSettings,
        retrieval_config: RetrievalConfig,
    ):
        self.vector_store = vector_store
        self.retrieval_config = retrieval_config

        # Rate-limited, retrying client -- see module docstring.
        self._client = MistralClient(mistral_settings)
        self._model = mistral_settings.generation_model

        # ---------------------------------------------------------
        # Retriever construction, including mmr support.
        # ---------------------------------------------------------
        if self.retrieval_config.retriever_type == "similarity_score_threshold":
            self._retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_config.top_k,
                    "score_threshold": self.retrieval_config.similarity_threshold,
                },
            )
        elif self.retrieval_config.retriever_type == "mmr":
            self._retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.retrieval_config.top_k,
                    "fetch_k": max(self.retrieval_config.top_k * 4, 20),
                },
            )
        else:
            self._retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.retrieval_config.top_k},
            )

    def retrieve(self, question: str) -> list[Document]:
        return self._retriever.invoke(question)

    def build_prompt(self, question: str, contexts: list[str]) -> str:
        joined_context = "\n\n".join(contexts)
        return self.retrieval_config.prompt_template.format(
            context=joined_context, question=question
        )

    def answer(self, question: str) -> RagResult:
        docs = self.retrieve(question)
        contexts = [d.page_content for d in docs]
        prompt = self.build_prompt(question, contexts)

        # MistralClient.chat() handles rate limiting + exponential-backoff
        # retry internally (see mistral_client.py), and returns the answer
        # text directly rather than a raw SDK response object.
        answer_text = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self._model,
            temperature=0.2,
        )
        return RagResult(question=question, answer=answer_text, contexts=contexts)

    def answer_many(self, questions: list[str]) -> list[RagResult]:
        results = []
        for i, q in enumerate(questions, start=1):
            logger.info("Answering question %d/%d", i, len(questions))
            results.append(self.answer(q))
        return results