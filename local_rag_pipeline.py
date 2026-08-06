"""
local_rag_pipeline.py
======================
Non-optimized baseline RAG pipeline: same retrieve -> build prompt ->
generate shape as rag_pipeline.RagPipeline, same RagResult output, but
generation goes to a locally-hosted LLM (via Ollama, e.g. qwen2.5)
instead of Mistral.

Structurally near-identical to RagPipeline on purpose -- the only
difference that matters for the comparison is the generation call site.
Retrieval logic, prompt formatting, and the output shape are unchanged,
so evaluator.py can score this pipeline's output the exact same way it
scores the optimized pipeline's output, and the two are directly
comparable.

This pipeline is NOT tunable by any optimizer -- it always runs with a
single fixed RetrievalConfig, once, over the benchmark.
"""

from __future__ import annotations

import logging

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import RetrievalConfig
from local_llm_client import LocalLLMClient
from rag_pipeline import RagResult  # reuse the same result shape as the optimized pipeline

logger = logging.getLogger("phoenix_rag.local_rag_pipeline")


class LocalRagPipeline:
    """Retrieve -> build prompt -> generate, using a local LLM for generation."""

    def __init__(
        self,
        vector_store: FAISS,
        retrieval_config: RetrievalConfig,
        local_model: str = "qwen2.5:7b-instruct",
        local_host: str = "http://localhost:11434",
    ):
        self.vector_store = vector_store
        self.retrieval_config = retrieval_config
        self._client = LocalLLMClient(model=local_model, host=local_host)

        # Same retriever construction as rag_pipeline.RagPipeline, including
        # mmr support, for parity between the two pipelines.
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

        answer_text = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return RagResult(question=question, answer=answer_text, contexts=contexts)

    def answer_many(self, questions: list[str]) -> list[RagResult]:
        results = []
        for i, q in enumerate(questions, start=1):
            logger.info("[baseline] Answering question %d/%d", i, len(questions))
            results.append(self.answer(q))
        return results