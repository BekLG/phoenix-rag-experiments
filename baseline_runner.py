"""
baseline_runner.py
===================
Runs the non-optimized baseline RAG pipeline ONCE: a local LLM (default:
qwen2.5 via Ollama) for generation, a single FIXED retrieval configuration
(no optimization loop, no iteration), and the SAME cached benchmark
question set and SAME Ragas judge (Mistral) as the optimized pipeline in
experiment_runner.py -- so the two results are directly comparable.

Embeddings are kept on Mistral (same as the optimized run) so retrieval
quality isn't a second confound alongside the generation-model swap --
this isolates "does the optimization loop help" as the variable under
test, not "does an entirely local stack perform differently."

Usage:
    python baseline_runner.py --source data/source.pdf
    python baseline_runner.py --source data/source.pdf --local-model qwen2.5:7b-instruct
    python baseline_runner.py --source data/source.pdf --local-host http://localhost:11434
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import RESULTS_DIR, load_or_create_default_config
from document_loader import load_document, load_full_text
from chunking import split_documents
from embeddings import MistralEmbeddings
from vector_store import build_vector_store
from question_generator import get_or_create_benchmark
from local_rag_pipeline import LocalRagPipeline
from evaluator import run_evaluation

logger = logging.getLogger("phoenix_rag.baseline_runner")

BASELINE_RESULT_PATH = RESULTS_DIR / "baseline_result.json"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def run_baseline(app_config, local_model: str, local_host: str) -> dict:
    """Run the fixed, non-optimized baseline pipeline once and score it.

    Uses app_config.retrieval AS-IS (whatever RetrievalConfig defaults are
    set in config.py / config/default_config.json) and never mutates it --
    unlike experiment_runner.py, there is no optimization loop here.
    """
    embeddings = MistralEmbeddings(app_config.mistral)

    full_text = load_full_text(app_config.source_document)
    benchmark = get_or_create_benchmark(
        full_text=full_text,
        mistral_settings=app_config.mistral,
        qg_config=app_config.question_generation,
        benchmark_path=app_config.benchmark_path,
    )
    question_texts = [q.question for q in benchmark]
    logger.info(
        "Benchmark ready: %d fixed evaluation questions (same set used by "
        "the optimized run, so scores are directly comparable)",
        len(benchmark),
    )

    raw_docs = load_document(app_config.source_document)
    chunks = split_documents(
        raw_docs,
        chunk_size=app_config.retrieval.chunk_size,
        chunk_overlap=app_config.retrieval.chunk_overlap,
    )
    vector_store = build_vector_store(chunks, embeddings)

    logger.info(
        "Running baseline: local_model=%s, fixed config=%s",
        local_model, app_config.retrieval.to_dict(),
    )
    pipeline = LocalRagPipeline(
        vector_store,
        app_config.retrieval,
        local_model=local_model,
        local_host=local_host,
    )
    results = pipeline.answer_many(question_texts)

    # Same Ragas judge (Mistral) as the optimized run -- only the pipeline
    # under test differs, so the scores mean the same thing on both sides.
    scores = run_evaluation(results, benchmark, app_config.mistral)

    weighted_score = (
        scores.get("faithfulness", 0.0) * 0.40
        + scores.get("context_recall", 0.0) * 0.20
        + scores.get("context_precision", 0.0) * 0.20
        + scores.get("response_relevancy", 0.0) * 0.20
    )

    payload = {
        "config": app_config.retrieval.to_dict(),
        "local_model": local_model,
        "scores": scores,
        "weighted_score": weighted_score,
    }

    BASELINE_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Baseline result saved to %s", BASELINE_RESULT_PATH)
    logger.info("Baseline scores: %s (weighted: %.4f)", scores, weighted_score)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the non-optimized local-LLM baseline RAG pipeline"
    )
    parser.add_argument(
        "--source", type=str, default=None, help="Path to the source PDF/text document"
    )
    parser.add_argument(
        "--local-model", type=str, default="qwen2.5:7b-instruct",
        help="Ollama model name for generation",
    )
    parser.add_argument(
        "--local-host", type=str, default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    app_config = load_or_create_default_config()
    if args.source:
        app_config.source_document = args.source

    if not Path(app_config.source_document).exists():
        logger.error("Source document not found: %s", app_config.source_document)
        sys.exit(1)

    run_baseline(app_config, args.local_model, args.local_host)


if __name__ == "__main__":
    main()