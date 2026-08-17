"""
experiment_runner.py
=====================
Orchestrates the full self-optimization loop:

    1. Load + chunk the source document, build the FAISS index (once,
       rebuilt per-iteration only when chunk_size/overlap change).
    2. Generate (or load cached) the fixed benchmark question set from
       the FULL document.
    2b. Generate (or load cached) a document summary. Fed to the LLM
        optimizer on every iteration so it can tailor the prompt template
        it writes to what the source document actually is.
    3. For each iteration:
        a. Build a RAG pipeline with the current retrieval config.
        b. Answer every benchmark question.
        c. Score answers with Ragas.
        d. Persist config + scores.
        e. Track the best-performing configuration so far (with safety gates).
        f. Ask the LLM optimizer to propose the next configuration -- it
           sees the FULL iteration history plus the document summary, and
           proposes retrieval params AND the prompt template together.
        g. Stop early if targets are met or max_iterations is reached.

The rule-based optimizer (optimizer.propose_next_config) and the standalone
prompt refiner (prompt_refiner.refine_prompt) are no longer part of this
loop -- propose_next_config_llm is the only proposer used.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import AppConfig
from document_loader import load_document
from chunking import split_documents
from embeddings import GeminiEmbeddings
from vector_store import get_or_build_vector_store
from question_generator import get_or_create_benchmark
from document_summarizer import get_or_create_summary
from document_profile import get_or_create_profile
from rag_pipeline import RagPipeline
from evaluator import run_evaluation
from optimizer import meets_targets
from llm_optimizer import propose_next_config_llm
import storage

logger = logging.getLogger("phoenix_rag.experiment_runner")


def run_experiment(app_config: AppConfig) -> dict:
    """Run the full optimization loop. Returns the best result found."""

    embeddings = GeminiEmbeddings(app_config.gemini)

    # Benchmark is generated ONCE from the full document and never touched again.
    source_documents = load_document(app_config.source_document)
    full_text = "\n\n".join(document.page_content for document in source_documents)
    benchmark = get_or_create_benchmark(
        full_text=full_text,
        gemini_settings=app_config.gemini,
        qg_config=app_config.question_generation,
        benchmark_path=app_config.benchmark_path,
    )
    question_texts = [q.question for q in benchmark]
    logger.info("Benchmark ready: %d fixed evaluation questions", len(benchmark))

    document_summary = get_or_create_summary(
        full_text=full_text,
        gemini_settings=app_config.gemini,
        summary_path=app_config.summary_path,
    )
    logger.info("Document summary ready (%d chars)", len(document_summary))

    pages = (
        len(source_documents)
        if Path(app_config.source_document).suffix.lower() == ".pdf"
        else None
    )
    document_profile = get_or_create_profile(
        full_text=full_text,
        profile_path=app_config.profile_path,
        pages=pages,
    )
    logger.info("Document profile ready: %s", document_profile)

    current_config = app_config.retrieval
    best_score = -1.0
    best_result: dict | None = None

    # Full iteration history -- config + scores + what was tried each
    # round. Passed to the LLM optimizer so it can see trade-offs across
    # the whole run, not just the current iteration.
    history: list[dict] = []

    cached_chunk_params: tuple[int, int] | None = None
    vector_store = None

    for iteration in range(1, app_config.optimizer.max_iterations + 1):
        logger.info("=== Iteration %d/%d ===", iteration, app_config.optimizer.max_iterations)

        chunk_params = (current_config.chunk_size, current_config.chunk_overlap)
        if vector_store is None or chunk_params != cached_chunk_params:
            logger.info("Rebuilding FAISS index (chunk params changed)")
            chunks = split_documents(
                source_documents,
                chunk_size=current_config.chunk_size,
                chunk_overlap=current_config.chunk_overlap,
            )
            vector_store = get_or_build_vector_store(
                chunks=chunks,
                embeddings=embeddings,
                cache_root=app_config.faiss_index_path,
                source_document=app_config.source_document,
                embedding_model=app_config.gemini.embedding_model,
                chunk_size=current_config.chunk_size,
                chunk_overlap=current_config.chunk_overlap,
            )
            cached_chunk_params = chunk_params

        pipeline = RagPipeline(vector_store, app_config.gemini, current_config)
        results = pipeline.answer_many(question_texts)

        scores = run_evaluation(results, benchmark, app_config.gemini)

        storage.save_iteration_config(iteration, current_config)
        storage.append_experiment_result(iteration, current_config, scores)

        # ------------------------------------------------------------------
        # Weighted Score + Minimum Faithfulness Gate
        # ------------------------------------------------------------------
        weighted_score = (
            scores.get("faithfulness", 0.0) * 0.40 +
            scores.get("context_recall", 0.0) * 0.20 +
            scores.get("context_precision", 0.0) * 0.20 +
            scores.get("response_relevancy", 0.0) * 0.20
        )

        is_safe = scores.get("faithfulness", 0.0) >= 0.80

        if is_safe and weighted_score > best_score:
            best_score = weighted_score
            best_result = {"iteration": iteration, "config": current_config, "scores": scores}
            storage.save_best_configuration(iteration, current_config, scores)
            logger.info("New best configuration saved! (Weighted Score: %.4f)", best_score)

        elif not is_safe and weighted_score > best_score:
            logger.warning(
                "Iteration %d scored highest (%.4f) but failed the Faithfulness safety gate (%.4f). Discarded.",
                iteration, weighted_score, scores.get("faithfulness", 0.0)
            )
        # ------------------------------------------------------------------

        if meets_targets(scores, app_config.optimizer):
            logger.info("Targets met at iteration %d, stopping early", iteration)
            history.append({
                "iteration": iteration,
                "config": current_config.to_dict(),
                "scores": scores,
                "applied_rules": "all_targets_met",
            })
            storage.append_evaluation_scores(iteration, scores, applied_rules=["all_targets_met"])
            break

        next_config, applied_rules = propose_next_config_llm(
            current_config,
            scores,
            app_config.optimizer,
            app_config.gemini,
            history,
            document_summary,
            document_profile,
        )

        history.append({
            "iteration": iteration,
            "config": current_config.to_dict(),
            "scores": scores,
            "applied_rules": "; ".join(applied_rules),
        })
        storage.append_evaluation_scores(iteration, scores, applied_rules=applied_rules)

        current_config = next_config

        if not applied_rules:
            logger.info("No further tuning rules triggered, stopping")
            break

    logger.info("Experiment complete. Best weighted score: %.4f", best_score)
    return best_result or {}
