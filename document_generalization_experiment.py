"""
document_generalization_experiment.py
=======================================
Validates Phoenix RAG's core value proposition: that self-optimization
removes the need for manual re-tuning when the source document changes.

Two conditions are run against the SAME new document, graded on the SAME
fixed benchmark, so the comparison isolates exactly one variable --
whether the config was re-optimized for this document or just reused:

  A. FROZEN:  the best RetrievalConfig previously found for the OLD
              document, applied unchanged (chunking, top_k, retriever
              type, similarity_threshold, prompt_template all frozen) --
              only the FAISS index is rebuilt, since that's mechanically
              required for a new document's content to be searchable at
              all.
  B. FRESH:   Phoenix RAG's full self-optimization loop run from scratch
              on the new document -- fresh benchmark generation, fresh
              document summary, fresh LLM-driven parameter tuning,
              starting from the default RetrievalConfig.

GOTCHAS THIS SCRIPT HANDLES FOR YOU (see module docstring section in the
project chat history for why these matter):
  1. Benchmark/summary caching is keyed by file path, not by which
     document it came from -- get_or_create_benchmark() has no idea
     "this benchmark was for document A." This script points both at
     document-B-specific paths so document A's cached benchmark is never
     silently reused against document B.
  2. storage.py writes results to fixed paths (results/best_configuration.json,
     results/evaluation_scores.csv, etc.) -- running a fresh optimization
     loop would silently overwrite document A's results. This script
     backs up the existing results/ directory before running the FRESH
     condition.

Usage:
    python document_generalization_experiment.py \\
        --old-best-config results/best_configuration.json \\
        --new-source data/new_document.pdf \\
        --label docB \\
        --max-iterations 6
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from chunking import split_documents
from config import (
    GENERATED_QUESTIONS_DIR,
    RESULTS_DIR,
    ROOT_DIR,
    AppConfig,
    RetrievalConfig,
    load_or_create_default_config,
)
from document_loader import load_document, load_full_text
from document_summarizer import get_or_create_summary
from embeddings import MistralEmbeddings
from evaluator import run_evaluation
from experiment_runner import run_experiment
from question_generator import get_or_create_benchmark
from rag_pipeline import RagPipeline
from vector_store import build_vector_store

logger = logging.getLogger("phoenix_rag.document_generalization_experiment")

METRICS = ["faithfulness", "context_recall", "context_precision", "response_relevancy"]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def _weighted(scores: dict) -> float:
    return (
        scores.get("faithfulness", 0.0) * 0.40
        + scores.get("context_recall", 0.0) * 0.20
        + scores.get("context_precision", 0.0) * 0.20
        + scores.get("response_relevancy", 0.0) * 0.20
    )


def _backup_results_dir() -> Path:
    """Move the existing results/ dir aside so the FRESH optimization
    condition (which runs the real experiment_runner loop, writing to
    the same fixed paths storage.py always uses) doesn't overwrite
    document A's original results.
    """
    if not RESULTS_DIR.exists() or not any(RESULTS_DIR.iterdir()):
        return RESULTS_DIR

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = ROOT_DIR / f"results_backup_{timestamp}"
    shutil.move(str(RESULTS_DIR), str(backup_path))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Backed up existing results/ to %s before running FRESH condition", backup_path)
    return backup_path


def run_frozen_condition(
    app_config: AppConfig,
    frozen_config: RetrievalConfig,
    benchmark,
) -> dict:
    """Condition A: apply the OLD document's best config, unchanged, to
    the NEW document. Only the FAISS index is rebuilt (mechanically
    required -- the old index was built from a different document's text
    and can't search the new one).
    """
    logger.info("=== FROZEN condition: reusing old config unchanged ===")
    logger.info("Frozen config: %s", frozen_config.to_dict())

    embeddings = MistralEmbeddings(app_config.mistral)
    raw_docs = load_document(app_config.source_document)
    chunks = split_documents(
        raw_docs,
        chunk_size=frozen_config.chunk_size,
        chunk_overlap=frozen_config.chunk_overlap,
    )
    vector_store = build_vector_store(chunks, embeddings)

    pipeline = RagPipeline(vector_store, app_config.mistral, frozen_config)
    question_texts = [q.question for q in benchmark]
    results = pipeline.answer_many(question_texts)

    scores = run_evaluation(results, benchmark, app_config.mistral)
    weighted = _weighted(scores)
    logger.info("FROZEN condition scores: %s (weighted: %.4f)", scores, weighted)

    return {
        "condition": "frozen",
        "config": frozen_config.to_dict(),
        "scores": scores,
        "weighted_score": weighted,
    }


def run_fresh_condition(app_config: AppConfig) -> dict:
    """Condition B: run Phoenix RAG's full self-optimization loop from
    scratch on the new document, starting from the default RetrievalConfig
    (NOT seeded from the old document's best config).
    """
    logger.info("=== FRESH condition: full self-optimization from scratch ===")

    fresh_app_config = AppConfig(
        mistral=app_config.mistral,
        retrieval=RetrievalConfig(),  # explicit fresh defaults, not seeded
        question_generation=app_config.question_generation,
        optimizer=app_config.optimizer,
        source_document=app_config.source_document,
        faiss_index_path=app_config.faiss_index_path,
        benchmark_path=app_config.benchmark_path,
        summary_path=app_config.summary_path,
    )

    _backup_results_dir()
    best_result = run_experiment(fresh_app_config)

    if not best_result:
        raise RuntimeError(
            "FRESH condition produced no valid result (every iteration may have "
            "failed the faithfulness safety gate) -- check logs above."
        )

    scores = best_result["scores"]
    weighted = _weighted(scores)
    logger.info("FRESH condition scores: %s (weighted: %.4f)", scores, weighted)

    return {
        "condition": "fresh",
        "config": best_result["config"].to_dict(),
        "scores": scores,
        "weighted_score": weighted,
        "iteration_found": best_result["iteration"],
    }


def print_comparison(frozen: dict, fresh: dict) -> None:
    print()
    print(f"{'Metric':<22} {'Frozen (old config)':<22} {'Fresh (re-optimized)':<22} {'Delta':<10}")
    print("-" * 78)
    for m in METRICS:
        f = frozen["scores"].get(m, 0.0)
        r = fresh["scores"].get(m, 0.0)
        print(f"{m:<22} {f:<22.4f} {r:<22.4f} {r - f:+.4f}")
    print("-" * 78)
    print(
        f"{'weighted_score':<22} {frozen['weighted_score']:<22.4f} "
        f"{fresh['weighted_score']:<22.4f} {fresh['weighted_score'] - frozen['weighted_score']:+.4f}"
    )
    print()
    if fresh["weighted_score"] > frozen["weighted_score"]:
        delta_pct = (
            (fresh["weighted_score"] - frozen["weighted_score"]) / max(frozen["weighted_score"], 1e-9)
        ) * 100
        print(
            f"Re-optimizing for the new document improved weighted score by "
            f"{delta_pct:.1f}% over reusing the old config unchanged."
        )
    else:
        print(
            "The frozen (old-document) config matched or outperformed re-optimization "
            "on this new document -- worth investigating why before claiming a "
            "generalization benefit."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a statically-reused config vs. fresh re-optimization on a new document"
    )
    parser.add_argument(
        "--old-best-config", type=str, required=True,
        help="Path to the old document's saved best_configuration.json",
    )
    parser.add_argument(
        "--new-source", type=str, required=True,
        help="Path to the new document to test generalization against",
    )
    parser.add_argument(
        "--label", type=str, required=True,
        help="Short label for this new document, used to namespace its benchmark/summary/output files (e.g. 'docB')",
    )
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if not Path(args.old_best_config).exists():
        logger.error("Old best-config file not found: %s", args.old_best_config)
        sys.exit(1)
    if not Path(args.new_source).exists():
        logger.error("New source document not found: %s", args.new_source)
        sys.exit(1)

    old_data = json.loads(Path(args.old_best_config).read_text(encoding="utf-8"))
    frozen_config = RetrievalConfig.from_dict(old_data["config"])

    app_config = load_or_create_default_config()
    app_config.source_document = args.new_source
    # Document-B-specific paths -- this is what prevents document A's
    # cached benchmark/summary from being silently reused (gotcha #1).
    app_config.benchmark_path = str(GENERATED_QUESTIONS_DIR / f"benchmark_{args.label}.json")
    app_config.summary_path = str(GENERATED_QUESTIONS_DIR / f"document_summary_{args.label}.txt")
    app_config.optimizer.max_iterations = args.max_iterations

    full_text = load_full_text(app_config.source_document)
    benchmark = get_or_create_benchmark(
        full_text=full_text,
        mistral_settings=app_config.mistral,
        qg_config=app_config.question_generation,
        benchmark_path=app_config.benchmark_path,
    )
    logger.info(
        "Benchmark for new document ready: %d questions (used for BOTH conditions)",
        len(benchmark),
    )
    # Warm the document summary cache too, so the FRESH condition's first
    # iteration doesn't pay for it mid-run.
    get_or_create_summary(
        full_text=full_text,
        mistral_settings=app_config.mistral,
        summary_path=app_config.summary_path,
    )

    frozen_result = run_frozen_condition(app_config, frozen_config, benchmark)
    fresh_result = run_fresh_condition(app_config)

    output_path = RESULTS_DIR / f"generalization_experiment_{args.label}.json"
    output_path.write_text(
        json.dumps({"frozen": frozen_result, "fresh": fresh_result}, indent=2),
        encoding="utf-8",
    )
    logger.info("Full experiment results saved to %s", output_path)

    print_comparison(frozen_result, fresh_result)


if __name__ == "__main__":
    main()