"""
app.py
======
CLI entry point for Phoenix RAG.

Usage:
    python app.py --source data/source.pdf
    python app.py --source data/source.pdf --force-regenerate-questions
    python app.py --source data/source.pdf --max-iterations 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import LOGS_DIR, load_or_create_default_config
from experiment_runner import run_experiment
from dotenv import load_dotenv

load_dotenv()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "phoenix_rag.log"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phoenix RAG: self-optimizing RAG system")
    parser.add_argument(
        "--source", type=str, default=None, help="Path to the source PDF/text document"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a saved AppConfig JSON file"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None, help="Override max optimization iterations"
    )
    parser.add_argument(
        "--force-regenerate-questions",
        action="store_true",
        help="Regenerate the benchmark question set even if a cached one exists",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)
    logger = logging.getLogger("phoenix_rag.app")

    app_config = (
        load_or_create_default_config() if not args.config else _load_config(args.config)
    )

    if args.source:
        app_config.source_document = args.source
    if args.max_iterations:
        app_config.optimizer.max_iterations = args.max_iterations
    if args.force_regenerate_questions:
        app_config.question_generation.regenerate_each_iteration = True

    if not Path(app_config.source_document).exists():
        logger.error(
            "Source document not found: %s\n"
            "Pass --source /path/to/document.pdf or place a file at that path.",
            app_config.source_document,
        )
        sys.exit(1)

    logger.info("Starting Phoenix RAG optimization run")
    logger.info("Source document: %s", app_config.source_document)

    best = run_experiment(app_config)

    if best:
        logger.info("=" * 60)
        logger.info("BEST CONFIGURATION FOUND (iteration %d)", best["iteration"])
        logger.info("Scores: %s", best["scores"])
        logger.info("Config: %s", best["config"].to_dict())
        logger.info("Full results saved under results/")
    else:
        logger.warning("No successful iterations completed")


def _load_config(path: str):
    from config import AppConfig

    return AppConfig.load(path)


if __name__ == "__main__":
    main()
