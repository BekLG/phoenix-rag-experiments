#!/usr/bin/env python3
"""
Add a new document to an existing Phoenix RAG setup and re-optimize.

Meant to run AFTER run_optimization.py has already produced a working
single-document setup in ./optimization-run. This script:
  1. Adds the new document (auto-converts single-doc -> corpus mode)
  2. Re-runs the optimization loop against the WHOLE corpus

Every function/attribute here was checked directly against phoenix_rag's
source (operations.py: add_document, AddDocumentResult, corpus.py) before
being used.

USAGE
    python add_document_and_reoptimize.py data/second_document.pdf
    python add_document_and_reoptimize.py data/second_document.pdf --max-iterations 5

IMPORTANT CAVEAT (read before comparing scores)
    Adding a document appends NEW questions to the benchmark. Scores from
    before this add are NOT comparable to scores after it -- the benchmark
    itself changed. Treat the re-optimization run as a new baseline, not
    as "did it get better than before."
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import phoenix_rag as pr
from phoenix_rag.operations import add_document, resolve_active_retrieval


def _to_dict(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return asdict(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_document", help="Path to the document to add")
    parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Optimization iterations to run against the grown corpus (default: 5)",
    )
    parser.add_argument(
        "--workspace", default="./optimization-run",
        help="Workspace used by run_optimization.py (default: ./optimization-run)",
    )
    parser.add_argument(
        "--no-optimize", action="store_true",
        help="Only add the document; skip re-running the optimization loop",
    )
    args = parser.parse_args()

    if not Path(args.new_document).exists():
        print(f"Error: document not found at {args.new_document}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 0. Reattach to the SAME workspace run_optimization.py used, so this
    #    grows the existing setup instead of starting a fresh one.
    # ------------------------------------------------------------------
    workspace = pr.Workspace(args.workspace).ensure()
    pr.use_workspace(workspace)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(workspace.log_file)],
    )

    print(f"Workspace: {workspace.root}")
    config = pr.load_or_create_default_config()

    if not config.corpus_path:
        print("Currently in single-document mode "
              f"(source_document: {config.source_document}).")
        print("Adding a second document will convert this into a corpus "
              "automatically -- the existing document is carried over, not dropped.\n")
    else:
        print(f"Already in corpus mode (corpus_path: {config.corpus_path}).\n")

    active_before = resolve_active_retrieval(config)
    print(f"Active retrieval config before add: {active_before.provenance}")
    print(json.dumps(_to_dict(active_before.config), indent=2, default=str))

    input(f"\nPress Enter to add '{args.new_document}' "
          "(this will call your configured LLM provider to summarize it "
          "and generate questions from it)...")

    # ------------------------------------------------------------------
    # 1. Add the document. auto-bootstraps single-doc -> corpus if needed,
    #    generates questions from the new document, appends them to the
    #    benchmark, and (by default) embeds the new chunks into the index
    #    immediately using whichever retrieval config is currently active.
    # ------------------------------------------------------------------
    print("\n--- Adding document ---\n")
    result = add_document(config, args.new_document)

    if result is None:
        print(f"'{args.new_document}' is already in the corpus (same content) "
              "-- nothing was added.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("DOCUMENT ADDED")
    print("=" * 60)
    for line in result.describe():
        print(f"  - {line}")
    print(f"\nDocuments in corpus : {result.document_count}")
    print(f"Benchmark questions : {result.questions_before} -> {result.questions_after}")

    if args.no_optimize:
        print("\n--no-optimize passed: skipping re-optimization.")
        print("Run this script again without --no-optimize, or run "
              "run_optimization.py again pointed at this workspace, when ready.")
        return

    print(
        "\nNOTE: the benchmark just grew. Scores from any PREVIOUS optimization "
        "run are not comparable to what this run produces -- this establishes "
        "a new baseline against the new, larger benchmark."
    )
    input(f"\nPress Enter to re-optimize against the full corpus "
          f"({args.max_iterations} iterations)...")

    # ------------------------------------------------------------------
    # 2. Re-optimize against the whole (now multi-document) corpus.
    #    config.corpus_path is already set by add_document -> enable_corpus,
    #    so run_experiment automatically optimizes the corpus, not just the
    #    single original document.
    # ------------------------------------------------------------------
    config.optimizer.max_iterations = args.max_iterations
    print("\n--- Re-running optimization against the grown corpus ---\n")
    best = pr.run_experiment(config)

    print("\n" + "=" * 60)
    if not best:
        print("NO CONFIGURATION CLEARED THE FAITHFULNESS SAFETY GATE (>= 0.80)")
        print("=" * 60)
        print(
            "Check "
            f"{workspace.results_dir / 'evaluation_scores.csv'} to see each "
            "iteration's actual faithfulness score and why it was rejected."
        )
    else:
        print("NEW BEST CONFIGURATION (against the grown corpus)")
        print("=" * 60)
        print(f"Iteration: {best['iteration']}")
        print("\nConfig:")
        print(json.dumps(_to_dict(best["config"]), indent=2, default=str))
        print("\nScores:")
        for metric, value in best["scores"].items():
            print(f"  {metric:<20}: {value:.4f}")

    print("\n" + "=" * 60)
    print("Full results written to:")
    print(f"  {workspace.results_dir / 'experiment_results.csv'}")
    print(f"  {workspace.results_dir / 'evaluation_scores.csv'}")
    print(f"  {workspace.results_dir / 'best_configuration.json'}")
    print(f"  {workspace.data_dir / 'corpus' / 'manifest.json'}   <- documents + per-doc question counts")
    print("=" * 60)


if __name__ == "__main__":
    main()
