"""
evaluator.py
============
Runs Ragas metrics (faithfulness, context_recall, context_precision,
answer_relevancy) over a set of RAG results using Mistral Small as the
judge LLM and Mistral embeddings as the embedding backend.
"""

from __future__ import annotations

import logging

from datasets import Dataset
from langchain_mistralai import ChatMistralAI
from ragas import evaluate
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

    logger.info("Running Ragas evaluation over %d samples", len(dataset))
    scored = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
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