#!/usr/bin/env python3
"""
05_grow_to_corpus_and_reoptimize.py — add a document to an existing corpus
and re-optimize against all its documents.

Meant to run AFTER an optimization run has populated ./phoenix-demo (or the
workspace selected with --workspace). A corpus holds one file or many. This script:
  1. Adds the new document to that corpus
  2. Re-runs the optimization loop against the WHOLE corpus

Every function/attribute here was checked directly against phoenix_rag's
source (operations.py: add_document, AddDocumentResult, corpus.py) before
being used.

USAGE
    python scripts/05_grow_to_corpus_and_reoptimize.py data/second_document.pdf
    python scripts/05_grow_to_corpus_and_reoptimize.py data/second_document.pdf --max-iterations 5

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
from phoenix_rag.operations import add_document, corpus_root


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
        "--workspace", default="./phoenix-demo",
        help="Workspace containing the corpus to grow (default: ./phoenix-demo)",
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
    # 0. Reattach to the SAME workspace the optimization run used, so this
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

    print(f"Corpus: {corpus_root(config)}")

    input(f"\nPress Enter to add '{args.new_document}' "
          "(this will call your configured LLM provider to summarize it "
          "and generate questions from it)...")

    # ------------------------------------------------------------------
    # 1. Add the document. Seeds an empty corpus from source_document if needed,
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
        print("Call run_experiment(config) using this workspace when ready.")
        return

    print(
        "\nNOTE: the benchmark just grew. Scores from any PREVIOUS optimization "
        "run are not comparable to what this run produces -- this establishes "
        "a new baseline against the new, larger benchmark."
    )
    input(f"\nPress Enter to re-optimize against the full corpus "
          f"({args.max_iterations} iterations)...")

    # ------------------------------------------------------------------
    # 2. Re-optimize against every registered document in the corpus.
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
    print(f"  {corpus_root(config) / 'manifest.json'}   <- documents + per-doc question counts")
    print("=" * 60)


if __name__ == "__main__":
    main()
