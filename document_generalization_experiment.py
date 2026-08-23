"""
document_generalization_experiment.py
=======================================
Validates Phoenix RAG's core value proposition: that self-optimization
removes the need for manual re-tuning when the source document changes.

Two conditions are run against the SAME new document, graded on the SAME
fixed benchmark, with prompt_template HELD CONSTANT across both arms, so
the comparison isolates the retrieval parameters and nothing else:

  B. FRESH:   Phoenix RAG's full self-optimization loop run from scratch
              on the new document -- fresh benchmark generation, fresh
              document summary, fresh LLM-driven parameter tuning,
              starting from the default RetrievalConfig. Runs FIRST,
              because its winning prompt_template is what condition A
              borrows (see below).
  A. FROZEN:  the retrieval parameters previously found best for the OLD
              document (chunk_size, chunk_overlap, top_k,
              similarity_threshold, retriever_type), applied unchanged to
              the new document -- but paired with the FRESH arm's
              prompt_template rather than the old document's. The FAISS
              index is rebuilt, since that's mechanically required for a
              new document's content to be searchable at all.

WHY prompt_template IS NOT COMPARED
-----------------------------------
prompt_template lives in RetrievalConfig next to chunk_size and top_k,
but it is not the same kind of thing. chunk_size=500 is document-agnostic
and transfers to any corpus; a prompt that says "You are a technical
assistant specializing in vector databases ... decline to answer
questions outside this scope" is document-scoped CONTENT, and applying it
to an unrelated document instructs the generator to refuse every
question.

Leaving it in the frozen arm therefore measures prompt non-portability,
not config generalization -- the two get bundled into a single number and
the retrieval question becomes unanswerable. Both arms' prompts were
tuned against their own document, so neither is a fair "frozen" value.
Holding the prompt fixed at the FRESH arm's value removes that confound:
whatever delta remains is attributable to the five retrieval parameters.

GOTCHAS THIS SCRIPT HANDLES FOR YOU (see module docstring section in the
project chat history for why these matter):
  1. Benchmark/summary/profile caching is keyed by file path, not by which
     document it came from -- get_or_create_benchmark() has no idea
     "this benchmark was for document A." This script points both at
     document-B-specific paths so document A's cached benchmark is never
     silently reused against document B.
  2. storage.py normally writes to fixed paths (results/best_configuration.json,
     results/evaluation_scores.csv, etc.). This script redirects those paths
     to results/generalization_experiment/<label>/, leaving normal optimization
     results untouched.

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
import sys
from pathlib import Path

import storage
from chunking import split_documents
from config import (
    GENERATED_QUESTIONS_DIR,
    RESULTS_DIR,
    AppConfig,
    RetrievalConfig,
    load_or_create_default_config,
)
from document_loader import load_document, load_full_text
from document_profile import get_or_create_profile
from document_summarizer import get_or_create_summary
from embeddings import MistralEmbeddings
from evaluator import run_evaluation
from experiment_runner import run_experiment
from question_generator import get_or_create_benchmark
from rag_pipeline import RagPipeline
from vector_store import get_or_build_vector_store

logger = logging.getLogger("phoenix_rag.document_generalization_experiment")

METRICS = ["faithfulness", "context_recall", "context_precision", "response_relevancy"]

# The RetrievalConfig fields the two arms are actually allowed to differ on.
# prompt_template is deliberately absent -- see "WHY prompt_template IS NOT
# COMPARED" in the module docstring.
COMPARED_DIMENSIONS = [
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "similarity_threshold",
    "retriever_type",
]


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


def _configure_results_dir(results_dir: Path) -> None:
    """Redirect storage.py outputs for this standalone experiment process."""
    results_dir.mkdir(parents=True, exist_ok=True)
    storage.CONFIGS_DIR = results_dir / "configs"
    storage.EXPERIMENT_RESULTS_CSV = results_dir / "experiment_results.csv"
    storage.EVALUATION_SCORES_CSV = results_dir / "evaluation_scores.csv"
    storage.BEST_CONFIG_PATH = results_dir / "best_configuration.json"
    storage.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generalization optimization results will be written to %s", results_dir)


def _share_prompt(
    frozen_config: RetrievalConfig, shared_prompt: str
) -> tuple[RetrievalConfig, str]:
    """Return the frozen config with prompt_template swapped for the shared one.

    Returns (config_to_run, discarded_prompt). The discarded old-document
    prompt is handed back so it can be recorded in comparison.json for
    provenance -- it is never executed.
    """
    discarded = frozen_config.prompt_template
    controlled = frozen_config.copy_with(prompt_template=shared_prompt)

    if discarded == shared_prompt:
        logger.info(
            "Frozen and fresh arms already share an identical prompt_template; "
            "no substitution needed."
        )
    else:
        logger.info(
            "Prompt control applied: frozen arm's own prompt_template (%d chars, "
            "tuned on the OLD document) discarded in favour of the FRESH arm's "
            "(%d chars). prompt_template is now identical across both arms and "
            "is NOT a compared dimension.",
            len(discarded), len(shared_prompt),
        )
    return controlled, discarded


def _differing_dimensions(frozen: RetrievalConfig, fresh: RetrievalConfig) -> dict:
    """Report which of COMPARED_DIMENSIONS actually differ between the arms."""
    frozen_d, fresh_d = frozen.to_dict(), fresh.to_dict()
    return {
        dim: {"frozen": frozen_d.get(dim), "fresh": fresh_d.get(dim)}
        for dim in COMPARED_DIMENSIONS
        if frozen_d.get(dim) != fresh_d.get(dim)
    }


def run_frozen_condition(
    app_config: AppConfig,
    frozen_config: RetrievalConfig,
    benchmark,
) -> dict:
    """Condition A: apply the OLD document's best RETRIEVAL PARAMETERS to
    the NEW document, unchanged.

    `frozen_config` is expected to have already had its prompt_template
    replaced with the FRESH arm's via _share_prompt(), so that
    prompt_template is held constant across both arms and the only things
    varying are COMPARED_DIMENSIONS. Only the FAISS index is rebuilt
    (mechanically required -- the old index was built from a different
    document's text and can't search the new one).
    """
    logger.info("=== FROZEN condition: reusing old retrieval params unchanged ===")
    logger.info("Frozen config: %s", frozen_config.to_dict())

    embeddings = MistralEmbeddings(app_config.mistral)
    raw_docs = load_document(app_config.source_document)
    chunks = split_documents(
        raw_docs,
        chunk_size=frozen_config.chunk_size,
        chunk_overlap=frozen_config.chunk_overlap,
    )
    vector_store = get_or_build_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        cache_root=app_config.faiss_index_path,
        source_document=app_config.source_document,
        embedding_model=app_config.mistral.embedding_model,
        chunk_size=frozen_config.chunk_size,
        chunk_overlap=frozen_config.chunk_overlap,
    )

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


def run_fresh_condition(app_config: AppConfig, results_dir: Path) -> dict:
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
        profile_path=app_config.profile_path,
    )

    _configure_results_dir(results_dir)
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


def print_comparison(frozen: dict, fresh: dict, differing: dict) -> None:
    print()
    print("prompt_template: HELD CONSTANT across both arms (fresh arm's value).")
    if differing:
        print("Retrieval parameters under comparison (only those that differ):")
        for dim, vals in differing.items():
            print(f"  {dim:<22} frozen={vals['frozen']!s:<18} fresh={vals['fresh']!s}")
    else:
        print(
            "Retrieval parameters under comparison: NONE DIFFER -- both arms are "
            "byte-identical configs. Any delta below is pure run-to-run noise, "
            "and is a useful measurement of that noise floor."
        )
    print()
    print(f"{'Metric':<22} {'Frozen (old params)':<22} {'Fresh (re-optimized)':<22} {'Delta':<10}")
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
    if not differing:
        print(
            "Both arms ran identical configs, so this delta is a noise estimate, "
            "not an effect. Treat any future delta smaller than this as unresolvable."
        )
        print()
        return
    if fresh["weighted_score"] > frozen["weighted_score"]:
        delta_pct = (
            (fresh["weighted_score"] - frozen["weighted_score"]) / max(frozen["weighted_score"], 1e-9)
        ) * 100
        print(
            f"With the prompt held constant, re-optimizing the retrieval parameters "
            f"for the new document improved weighted score by {delta_pct:.1f}% over "
            f"reusing the old document's parameters."
        )
    else:
        print(
            "With the prompt held constant, the old document's retrieval parameters "
            "matched or outperformed re-optimization on this new document -- i.e. no "
            "measurable parameter-level generalization benefit here. Worth reporting "
            "as-is rather than investigating until it flips."
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
    parser.add_argument(
        "--force-regenerate-questions",
        action="store_true",
        help="Regenerate the document benchmark even when a cached file exists",
    )
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
    app_config.profile_path = str(GENERATED_QUESTIONS_DIR / f"document_profile_{args.label}.json")
    app_config.optimizer.max_iterations = args.max_iterations

    full_text = load_full_text(app_config.source_document)
    benchmark = get_or_create_benchmark(
        full_text=full_text,
        mistral_settings=app_config.mistral,
        qg_config=app_config.question_generation,
        benchmark_path=app_config.benchmark_path,
        force_regenerate=args.force_regenerate_questions,
    )
    logger.info(
        "Benchmark for new document ready: %d questions (used for BOTH conditions)",
        len(benchmark),
    )
    # Warm the summary and profile caches too, so the FRESH condition's first
    # iteration doesn't pay for either artifact mid-run.
    get_or_create_summary(
        full_text=full_text,
        mistral_settings=app_config.mistral,
        summary_path=app_config.summary_path,
    )
    get_or_create_profile(
        full_text=full_text,
        profile_path=app_config.profile_path,
        pages=len(load_document(app_config.source_document))
        if Path(app_config.source_document).suffix.lower() == ".pdf"
        else None,
    )

    experiment_results_dir = RESULTS_DIR / "generalization_experiment" / args.label

    # ORDER MATTERS: fresh runs FIRST because the frozen arm borrows its
    # winning prompt_template. Holding the prompt constant is what turns
    # this into a clean test of the retrieval parameters -- see the module
    # docstring.
    fresh_result = run_fresh_condition(app_config, experiment_results_dir)

    fresh_config = RetrievalConfig.from_dict(fresh_result["config"])
    controlled_frozen_config, discarded_prompt = _share_prompt(
        frozen_config, fresh_config.prompt_template
    )
    differing = _differing_dimensions(controlled_frozen_config, fresh_config)
    if not differing:
        logger.warning(
            "Frozen and fresh arms are IDENTICAL on every compared dimension (%s). "
            "The run will still proceed -- the resulting delta is a run-to-run "
            "noise estimate, which is worth having, but it is not an effect.",
            ", ".join(COMPARED_DIMENSIONS),
        )
    else:
        logger.info("Compared dimensions that differ: %s", differing)

    frozen_result = run_frozen_condition(app_config, controlled_frozen_config, benchmark)

    output_path = experiment_results_dir / "comparison.json"
    output_path.write_text(
        json.dumps(
            {
                "control": {
                    "prompt_template_held_constant": True,
                    "shared_prompt_source": "fresh (re-optimized on this document)",
                    "shared_prompt_template": fresh_config.prompt_template,
                    "frozen_discarded_prompt_template": discarded_prompt,
                    "compared_dimensions": COMPARED_DIMENSIONS,
                    "differing_dimensions": differing,
                    "note": (
                        "prompt_template is document-scoped content, not a portable "
                        "hyperparameter -- each arm's prompt was tuned against its own "
                        "document, so neither is a fair frozen value. It is held "
                        "constant at the fresh arm's value and excluded from the "
                        "comparison. Deltas below are attributable to "
                        "compared_dimensions only."
                    ),
                },
                "frozen": frozen_result,
                "fresh": fresh_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Full experiment results saved to %s", output_path)

    print_comparison(frozen_result, fresh_result, differing)


if __name__ == "__main__":
    main()
