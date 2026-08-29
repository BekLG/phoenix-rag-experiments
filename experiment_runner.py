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
    2c. Compute (or load cached) the document profile, then derive iteration
        1's chunk_size/chunk_overlap/top_k from it (seed_config.py). The
        config's retrieval block supplies the non-derivable fields and is
        the fallback when optimizer.seed_from_profile is False.
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

SINGLE DOCUMENT vs CORPUS
-------------------------
Steps 1, 2, 2b and 2c differ depending on whether app_config.corpus_path is set,
and nothing else in the loop does. They are therefore gathered up front by
prepare_inputs() into an ExperimentInputs, which hands the loop a benchmark, a
summary, a profile, and a callable that produces an index for a given
chunk_size/overlap. The scoring, safety gate, history and proposal logic below
are identical in both modes and have no idea which one is running.

  * corpus_path unset (the default) -- exactly the original behaviour: one
    document, its own cached benchmark/summary/profile, and the content-addressed
    index from vector_store.get_or_build_vector_store.
  * corpus_path set -- the benchmark, summary and profile describe every document
    in the corpus (see corpus.py), and the index spans all of them. A document
    added since the last run is embedded into the existing index rather than
    triggering a rebuild.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import corpus
import storage
from chunking import split_documents
from config import AppConfig
from document_loader import load_document
from document_profile import DocumentProfile, get_or_create_profile
from document_summarizer import get_or_create_summary
from embeddings import MistralEmbeddings
from evaluator import run_evaluation
from llm_optimizer import propose_next_config_llm
from optimizer import meets_targets
from question_generator import BenchmarkQuestion, get_or_create_benchmark
from rag_pipeline import RagPipeline
from seed_config import propose_seed_config
from vector_store import get_or_build_vector_store

logger = logging.getLogger("phoenix_rag.experiment_runner")


@dataclass
class ExperimentInputs:
    """Everything the optimization loop needs that depends on WHAT is indexed.

    `build_store` takes (chunk_size, chunk_overlap) and returns a vector store
    for those parameters. The loop calls it only when the chunk parameters
    change, so both implementations are free to be expensive.

    `on_success` is invoked once, after the loop finishes, only if a best
    configuration was actually found and saved. The corpus uses it to record
    which documents that configuration was measured against.
    """

    benchmark: list[BenchmarkQuestion]
    document_summary: str
    document_profile: DocumentProfile
    build_store: Callable[[int, int], object]
    description: str
    on_success: Callable[[], None] | None = None


def _single_document_inputs(
    app_config: AppConfig, embeddings: MistralEmbeddings
) -> ExperimentInputs:
    """The original single-document path, moved here verbatim in behaviour."""
    source_documents = load_document(app_config.source_document)
    full_text = "\n\n".join(document.page_content for document in source_documents)

    benchmark = get_or_create_benchmark(
        full_text=full_text,
        mistral_settings=app_config.mistral,
        qg_config=app_config.question_generation,
        benchmark_path=app_config.benchmark_path,
    )
    document_summary = get_or_create_summary(
        full_text=full_text,
        mistral_settings=app_config.mistral,
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

    def build_store(chunk_size: int, chunk_overlap: int):
        chunks = split_documents(
            source_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return get_or_build_vector_store(
            chunks=chunks,
            embeddings=embeddings,
            cache_root=app_config.faiss_index_path,
            source_document=app_config.source_document,
            embedding_model=app_config.mistral.embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    return ExperimentInputs(
        benchmark=benchmark,
        document_summary=document_summary,
        document_profile=document_profile,
        build_store=build_store,
        description=f"single document: {Path(app_config.source_document).name}",
    )


def _corpus_inputs(
    app_config: AppConfig, embeddings: MistralEmbeddings
) -> ExperimentInputs:
    """The multi-document path: every artifact describes the whole corpus."""
    corpus_state = corpus.load(app_config.corpus_path)

    if corpus_state.is_empty:
        # An operator who turned corpus mode on without adding anything yet still
        # has a configured source_document; seeding from it (reusing its cached
        # artifacts) is much friendlier than refusing to run.
        corpus.bootstrap_from_single_document(corpus_state, app_config)
    if corpus_state.is_empty:
        raise RuntimeError(
            f"Corpus at {app_config.corpus_path} is empty and could not be "
            "bootstrapped. Add a document before optimizing."
        )

    for problem in corpus.verify_documents(corpus_state):
        logger.warning("Corpus integrity: %s", problem)

    benchmark = corpus.corpus_benchmark(corpus_state, app_config)
    document_summary = corpus.summary_text(corpus_state)
    document_profile = corpus.profile(corpus_state)

    logger.info(
        "Corpus summary ready (%d chars, %d document(s))",
        len(document_summary), len(corpus_state.documents),
    )
    logger.info("Aggregated corpus profile: %s", document_profile)
    logger.info(
        "Benchmark composition: %s",
        ", ".join(
            f"{d.label}={d.question_count}" for d in corpus_state.documents
        ) or "(unattributed)",
    )
    if corpus.optimization_status(corpus_state) == "stale":
        logger.info(
            "Corpus membership changed since the last optimization -- the previously "
            "saved best configuration does not describe this corpus, and scores "
            "below are not comparable to it (the benchmark grew too)."
        )

    def build_store(chunk_size: int, chunk_overlap: int):
        store, _report = corpus.sync_index(
            corpus_state,
            embeddings=embeddings,
            embedding_model=app_config.mistral.embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return store

    labels = ", ".join(document.label for document in corpus_state.documents)
    return ExperimentInputs(
        benchmark=benchmark,
        document_summary=document_summary,
        document_profile=document_profile,
        build_store=build_store,
        description=f"corpus of {len(corpus_state.documents)}: {labels}",
        on_success=lambda: corpus.mark_optimized(corpus_state),
    )


def prepare_inputs(
    app_config: AppConfig, embeddings: MistralEmbeddings
) -> ExperimentInputs:
    """Gather the loop's document-dependent inputs for whichever mode is active."""
    if app_config.corpus_path:
        return _corpus_inputs(app_config, embeddings)
    return _single_document_inputs(app_config, embeddings)


def run_experiment(app_config: AppConfig) -> dict:
    """Run the full optimization loop. Returns the best result found."""

    embeddings = MistralEmbeddings(app_config.mistral)

    # Benchmark is generated ONCE and never touched again during the run, so
    # every configuration is scored against exactly the same questions.
    inputs = prepare_inputs(app_config, embeddings)
    question_texts = [question.question for question in inputs.benchmark]
    logger.info(
        "Benchmark ready: %d fixed evaluation questions (%s)",
        len(inputs.benchmark), inputs.description,
    )

    document_summary = inputs.document_summary
    document_profile = inputs.document_profile

    # Iteration 1 starts from parameters DERIVED FROM THIS DOCUMENT, not from the
    # config's retrieval block. Without this the LLM optimizer anchors on a
    # document-agnostic starting value and spends the whole budget nudging around
    # it -- see the trajectory in seed_config.py's module docstring.
    current_config = app_config.retrieval
    seed_rules: list[str] = []
    if app_config.optimizer.seed_from_profile:
        seed = propose_seed_config(
            base_config=current_config,
            profile=document_profile,
            opt_config=app_config.optimizer,
        )
        current_config = seed.config
        seed_rules = [seed.rationale]
    else:
        logger.info(
            "Profile seeding disabled; starting from the configured retrieval block: %s",
            current_config.to_dict(),
        )

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
            logger.info(
                "Vector store needed for chunk_size=%d, chunk_overlap=%d",
                *chunk_params,
            )
            vector_store = inputs.build_store(*chunk_params)
            cached_chunk_params = chunk_params

        pipeline = RagPipeline(vector_store, app_config.mistral, current_config)
        results = pipeline.answer_many(question_texts)

        scores = run_evaluation(results, inputs.benchmark, app_config.mistral)

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
            applied_rules = [*seed_rules, "all_targets_met"]
            seed_rules = []
            history.append({
                "iteration": iteration,
                "config": current_config.to_dict(),
                "scores": scores,
                "applied_rules": "; ".join(applied_rules),
            })
            storage.append_evaluation_scores(iteration, scores, applied_rules=applied_rules)
            break

        next_config, proposal_rules = propose_next_config_llm(
            current_config,
            scores,
            app_config.optimizer,
            app_config.mistral,
            history,
            document_summary,
            document_profile,
        )

        # Iteration 1's row carries the seed rationale ahead of that iteration's
        # proposal, so both evaluation_scores.csv and the history the LLM reads on
        # later iterations record what the starting point was derived from,
        # instead of it looking like an arbitrary default. Consumed once.
        applied_rules = [*seed_rules, *proposal_rules]
        seed_rules = []

        history.append({
            "iteration": iteration,
            "config": current_config.to_dict(),
            "scores": scores,
            "applied_rules": "; ".join(applied_rules),
        })
        storage.append_evaluation_scores(iteration, scores, applied_rules=applied_rules)

        current_config = next_config

        if not proposal_rules:
            logger.info("No further tuning rules triggered, stopping")
            break

    logger.info("Experiment complete. Best weighted score: %.4f", best_score)

    # Only after a best configuration was actually found and saved -- a run where
    # every iteration failed the faithfulness gate has not established anything
    # about this corpus, so it must not be recorded as having optimized it.
    if best_result and inputs.on_success is not None:
        inputs.on_success()

    return best_result or {}
