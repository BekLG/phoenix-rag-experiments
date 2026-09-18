"""
cli.py
======
Command-line entry point for Phoenix RAG. Installed as ``phoenix-rag``.

Usage:
    phoenix-rag --source data/source.pdf
    phoenix-rag --source data/source.pdf --force-regenerate-questions
    phoenix-rag --source data/source.pdf --max-iterations 15
    phoenix-rag --source data/source.pdf --no-profile-seed

    phoenix-rag --menu                    # interactive terminal front-end
    phoenix-rag --corpus                  # optimize the whole multi-doc corpus
    phoenix-rag --no-corpus --source X    # ignore the corpus for this run

    phoenix-rag --workspace ~/rag-runs    # read and write somewhere else

For the GUI: ``streamlit run src/phoenix_rag/ui/streamlit_app.py``

WORKSPACE
---------
Every input and output path is relative to a workspace: ``data/``, ``results/``,
``generated_questions/``, ``logs/`` and ``config/`` live under it. It defaults to
the current directory (or ``$PHOENIX_RAG_WORKSPACE``), so running from a checkout
behaves exactly as the old scripts did. This entry point is what creates those
directories -- importing the library does not.

CORPUS MODE
-----------
--corpus opts this run into the multi-document corpus (``<workspace>/data/corpus/``),
where the benchmark, summary, profile and FAISS index describe every document
that has been added rather than one file. Documents are added through the menu or
the GUI. When the saved config already has corpus_path set, that is honoured
without the flag; --no-corpus overrides it for one run without editing the config.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from phoenix_rag.config import load_config, load_env, load_or_create_default_config
from phoenix_rag.workspace import Workspace, active_workspace, use_workspace


def _setup_logging(verbose: bool, workspace: Workspace) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(workspace.log_file),
        ],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="phoenix-rag",
        description="Phoenix RAG: self-optimizing RAG system",
    )
    parser.add_argument(
        "--source", type=str, default=None, help="Path to the source PDF/text document"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config YAML file (default: <workspace>/config/config.yaml)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=(
            "Directory holding data/, results/, generated_questions/, logs/ and "
            "config/. Defaults to $PHOENIX_RAG_WORKSPACE or the current directory."
        ),
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
            "Optimize against the multi-document corpus in <workspace>/data/corpus/ "
            "instead of a single document. Add documents with --menu (option 2) or the GUI."
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve and create the workspace before anything wants a path from it.
    workspace = use_workspace(args.workspace) if args.workspace else active_workspace()
    workspace.ensure()
    load_env()

    if args.menu:
        # Deferred so the menu's own logging setup is the only one installed.
        from phoenix_rag.ui import menu

        sys.exit(menu.main())

    _setup_logging(args.verbose, workspace)
    logger = logging.getLogger("phoenix_rag.cli")

    if args.corpus and args.no_corpus:
        logger.error("--corpus and --no-corpus contradict each other; pick one")
        sys.exit(2)

    app_config = (
        load_config(args.config) if args.config else load_or_create_default_config(workspace)
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
        app_config.corpus_path = str(workspace.corpus_dir)
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
    logger.info("Workspace: %s", workspace.root)
    if app_config.corpus_path:
        logger.info("Corpus: %s", app_config.corpus_path)
    else:
        logger.info("Source document: %s", app_config.source_document)

    # Imported here rather than at module scope: this pulls in ragas, datasets and
    # faiss, which is several seconds we should not spend on `--help` or `--menu`.
    from phoenix_rag.optimization.runner import run_experiment

    best = run_experiment(app_config)

    if best:
        logger.info("=" * 60)
        logger.info("BEST CONFIGURATION FOUND (iteration %d)", best["iteration"])
        logger.info("Scores: %s", best["scores"])
        logger.info("Config: %s", best["config"].to_dict())
        logger.info("Full results saved under %s", workspace.results_dir)
    else:
        logger.warning("No successful iterations completed")


if __name__ == "__main__":
    main()
