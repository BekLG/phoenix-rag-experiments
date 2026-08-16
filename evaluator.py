"""
evaluator.py
============
Runs Ragas metrics (faithfulness, context_recall, context_precision,
answer_relevancy) over a set of RAG results using Gemini as the judge LLM
and Gemini embeddings as the embedding backend.

NOTE ON langchain_google_genai's constructor kwarg: ChatGoogleGenerativeAI
takes `google_api_key` as of the versions documented when this was written.
If you get a validation error on construction, check
`ChatGoogleGenerativeAI.__init__`'s actual signature on your installed
langchain-google-genai version.

Ragas evaluates metrics as a batch of internally-concurrent async calls --
each metric can fire multiple sub-calls per sample (e.g. faithfulness
decomposes an answer into statements, then verifies each one separately),
so the real number of judge API calls is well above len(dataset) *
len(METRICS). The judge client here is built directly, NOT routed through
gemini_client.GeminiClient, so none of that module's rate limiting applies
to judge calls -- Ragas's default concurrency can burst well past Gemini's
actual per-minute limit and trigger a wall of 429s. run_evaluation() below
caps Ragas's own concurrency via RunConfig to keep it under the limit.
"""

from __future__ import annotations

import asyncio
import logging

from datasets import Dataset
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
try:
    from ragas.run_config import RunConfig
except ImportError:
    # Import path has moved between ragas versions; older/newer releases
    # sometimes expose it directly off the top-level package instead.
    from ragas import RunConfig
from ragas.metrics import (
    AnswerRelevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config import GeminiSettings
from embeddings import GeminiEmbeddings
from gemini_client import RateLimiter, get_shared_rate_limiter
from question_generator import BenchmarkQuestion
from rag_pipeline import RagResult

logger = logging.getLogger("phoenix_rag.evaluator")

METRICS = [faithfulness, context_recall, context_precision, AnswerRelevancy(strictness=1)]

# Caps how many judge calls Ragas fires concurrently. Lower this further
# if you still see 429 bursts (e.g. to 1 for a fully serialized, slowest
# but safest run); raise it if your Gemini plan has more headroom.
_RAGAS_MAX_WORKERS = 1


class _LangChainRateLimiter(BaseRateLimiter):
    """Adapter for LangChain's sync and async rate-limiter protocol."""

    def __init__(self, limiter: RateLimiter):
        self._limiter = limiter

    def acquire(self, *, blocking: bool = True) -> bool:
        if not blocking:
            # Ragas uses blocking acquisition. Avoid pretending that a
            # non-blocking check reserved quota when it did not.
            return False
        self._limiter.acquire()
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        if not blocking:
            return False
        await asyncio.to_thread(self._limiter.acquire)
        return True


def build_ragas_dataset(
    results: list[RagResult], questions: list[BenchmarkQuestion]
) -> Dataset:
    """Assemble the HuggingFace Dataset Ragas expects.

    `results` and `questions` must be aligned (same order, same question text)
    -- the caller (experiment_runner) is responsible for that pairing.
    """
    records = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "reference": [],
    }
    for result, bench_q in zip(results, questions):
        records["question"].append(result.question)
        records["answer"].append(result.answer)
        records["contexts"].append(result.contexts)
        records["ground_truth"].append(bench_q.reference_answer)
        records["reference"].append(bench_q.reference_answer)

    return Dataset.from_dict(records)


def run_evaluation(
    results: list[RagResult],
    questions: list[BenchmarkQuestion],
    gemini_settings: GeminiSettings,
) -> dict[str, float]:
    """Evaluate a batch of RAG results with Ragas, return mean metric scores."""
    dataset = build_ragas_dataset(results, questions)

    judge_limiter = get_shared_rate_limiter(
        gemini_settings,
        gemini_settings.judge_model,
    )
    judge_llm = ChatGoogleGenerativeAI(
        model=gemini_settings.judge_model,
        google_api_key=gemini_settings.api_key,
        temperature=0,
        rate_limiter=_LangChainRateLimiter(judge_limiter),
    )
    judge_embeddings = GeminiEmbeddings(gemini_settings)

    # Ragas's own concurrency, capped -- see module docstring.
    run_config = RunConfig(
        max_workers=_RAGAS_MAX_WORKERS,
        max_retries=10,
        max_wait=60,
    )

    logger.info(
        "Running Ragas evaluation over %d samples (max_workers=%d)",
        len(dataset), _RAGAS_MAX_WORKERS,
    )
    scored = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    df = scored.to_pandas()
    scores = {
        "faithfulness": float(df["faithfulness"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "response_relevancy": float(df["answer_relevancy"].mean())
        if "answer_relevancy" in df
        else float(df["response_relevancy"].mean()),
    }
    logger.info("Evaluation scores: %s", scores)
    return scores
