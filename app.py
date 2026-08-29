"""
app.py
======
CLI entry point for Phoenix RAG.

Usage:
    python app.py --source data/source.pdf
    python app.py --source data/source.pdf --force-regenerate-questions
    python app.py --source data/source.pdf --max-iterations 15
    python app.py --source data/source.pdf --no-profile-seed

    python app.py --menu                    # interactive terminal front-end
    python app.py --corpus                  # optimize the whole multi-doc corpus
    python app.py --no-corpus --source X    # ignore the corpus for this run

For the GUI: streamlit run streamlit_app.py

CORPUS MODE
-----------
--corpus opts this run into the multi-document corpus (data/corpus/), where the
benchmark, summary, profile and FAISS index describe every document that has been
added rather than one file. Documents are added through the menu or the GUI. When
the saved config already has corpus_path set, that is honoured without the flag;
--no-corpus overrides it for one run without editing the config.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import CORPUS_DIR, LOGS_DIR, load_or_create_default_config
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
    parser.add_argument(
        "--no-profile-seed",
        action="store_true",
        help=(
            "Start iteration 1 from the config's retrieval block instead of "
            "deriving chunk_size/chunk_overlap/top_k from the document profile"
        ),
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Launch the interactive terminal front-end instead of running an experiment",
    )
    parser.add_argument(
        "--corpus",
        action="store_true",
        help=(
            "Optimize against the multi-document corpus in data/corpus/ instead of a "
            "single document. Add documents with --menu (option 2) or the GUI."
        ),
    )
    parser.add_argument(
        "--no-corpus",
        action="store_true",
        help=(
            "Ignore the corpus for this run even if the saved config enables it, and "
            "optimize against --source alone"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.menu:
        # Deferred so the menu's own logging setup is the only one installed.
        import menu

        sys.exit(menu.main())

    _setup_logging(args.verbose)
    logger = logging.getLogger("phoenix_rag.app")

    if args.corpus and args.no_corpus:
        logger.error("--corpus and --no-corpus contradict each other; pick one")
        sys.exit(2)

    app_config = (
        load_or_create_default_config() if not args.config else _load_config(args.config)
    )

    if args.source:
        app_config.source_document = args.source
    if args.max_iterations:
        app_config.optimizer.max_iterations = args.max_iterations
    if args.force_regenerate_questions:
        app_config.question_generation.regenerate_each_iteration = True
    if args.no_profile_seed:
        app_config.optimizer.seed_from_profile = False
    if args.corpus:
        app_config.corpus_path = str(CORPUS_DIR)
    if args.no_corpus:
        app_config.corpus_path = None

    # The source document is only required when it is what gets indexed. In corpus
    # mode it is just the seed for an empty corpus, and an already-populated corpus
    # does not need it at all -- refusing to run because a path that will never be
    # read is missing would be wrong.
    if not app_config.corpus_path and not Path(app_config.source_document).exists():
        logger.error(
            "Source document not found: %s\n"
            "Pass --source /path/to/document.pdf or place a file at that path.",
            app_config.source_document,
        )
        sys.exit(1)

    logger.info("Starting Phoenix RAG optimization run")
    if app_config.corpus_path:
        logger.info("Corpus: %s", app_config.corpus_path)
    else:
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
