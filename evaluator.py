"""
evaluator.py
============
Runs Ragas metrics (faithfulness, context_recall, context_precision,
answer_relevancy) over a set of RAG results using Mistral Small as the
judge LLM and Mistral embeddings as the embedding backend.

Ragas evaluates metrics as a batch of internally-concurrent async calls --
each metric can fire multiple sub-calls per sample (e.g. faithfulness
decomposes an answer into statements, then verifies each one separately),
so the real number of judge API calls is well above len(dataset) *
len(METRICS). The judge client here (ChatMistralAI) is built directly,
NOT routed through mistral_client.MistralClient, so none of that module's
rate limiting applies to judge calls -- Ragas's default concurrency can
burst well past Mistral's actual per-minute limit and trigger a wall of
429s. run_evaluation() below caps Ragas's own concurrency via RunConfig
to keep it under the limit instead.
"""

from __future__ import annotations

import logging

from datasets import Dataset
from langchain_mistralai import ChatMistralAI

from ragas_compat import install_ragas_compat

install_ragas_compat()
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

from config import MistralSettings
from embeddings import MistralEmbeddings
from question_generator import BenchmarkQuestion
from rag_pipeline import RagResult

logger = logging.getLogger("phoenix_rag.evaluator")

METRICS = [faithfulness, context_recall, context_precision, AnswerRelevancy(strictness=1)]

# Caps how many judge calls Ragas fires concurrently. Lower this further
# if you still see 429 bursts (e.g. to 1 for a fully serialized, slowest
# but safest run); raise it if your Mistral plan has more headroom than
# the free tier's ~45-60 req/min.
_RAGAS_MAX_WORKERS = 3


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
    mistral_settings: MistralSettings,
) -> dict[str, float]:
    """Evaluate a batch of RAG results with Ragas, return mean metric scores."""
    dataset = build_ragas_dataset(results, questions)

    judge_llm = ChatMistralAI(
        model=mistral_settings.judge_model,
        api_key=mistral_settings.api_key,
        temperature=0,
    )
    judge_embeddings = MistralEmbeddings(mistral_settings)

    # Ragas's own concurrency, capped -- see module docstring. Also raises
    # Ragas's own max_wait/max_retries so a 429 that does slip through
    # gets backed off and retried by Ragas itself rather than surfacing
    # as noisy log spam (or, worst case, a failed evaluation row).
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
