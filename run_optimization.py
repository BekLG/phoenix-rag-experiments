#!/usr/bin/env python3
"""
Run a full 5-iteration Phoenix RAG optimization experiment.

Every function call in this script was checked directly against the
phoenix_rag source (schema.py, workspace.py, runner.py, seed_config.py,
document_profile.py) — not guessed from the README.

USAGE
    source .venv/bin/activate
    pip install -e '.[dev]'          # add a backend extra as needed, e.g. '[anthropic]', '[local]'
    cp .env.example .env             # set your API key(s)
    python run_optimization.py data/your_document.pdf

WHAT THIS ACTUALLY RUNS
    run_experiment() scores each iteration with:
        weighted_score = faithfulness*0.40 + context_recall*0.20
                        + context_precision*0.20 + response_relevancy*0.20
    ...and only accepts a new "best" if faithfulness >= 0.80 (a hard safety
    gate). If no iteration ever clears that gate, best_result is {} — this
    script handles that case explicitly rather than crashing on it.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import phoenix_rag as pr


def _to_dict(obj):
    """Best-effort conversion for printing: works for AppConfig sub-objects
    (which have .to_dict()) and for plain dataclasses (DocumentProfile,
    SeedProposal) which don't."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return asdict(obj)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_optimization.py path/to/document.pdf")
        sys.exit(1)

    source_document = sys.argv[1]
    if not Path(source_document).exists():
        print(f"Error: document not found at {source_document}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. An isolated workspace for this run, so it doesn't read/write
    #    on top of any other experiment you've run from this directory.
    # ------------------------------------------------------------------
    workspace = pr.Workspace("./optimization-run").ensure()
    pr.use_workspace(workspace)
    print(f"Workspace: {workspace.root}")

    # ------------------------------------------------------------------
    # 2. Load (or create) the config, point it at your document, cap it
    #    to 5 iterations. load_or_create_default_config() also loads
    #    .env for you, so your API key(s) are picked up automatically.
    # ------------------------------------------------------------------
    config = pr.load_or_create_default_config()
    config.source_document = source_document
    config.optimizer.max_iterations = 5

    print(f"\nSource document : {source_document}")
    print(f"Max iterations  : {config.optimizer.max_iterations}")
    for role in ("embedding", "generation", "optimizer", "judge"):
        pc = getattr(config.providers, role)
        print(f"  {role:<10}: backend={pc.backend:<10} model={pc.model}")

    # ------------------------------------------------------------------
    # 3. Free sanity check, no LLM calls: profile the document and see
    #    what the document-derived iteration-1 seed will look like
    #    BEFORE spending a single token.
    # ------------------------------------------------------------------
    docs = pr.load_document(source_document)
    full_text = "\n\n".join(d.page_content for d in docs)
    profile = pr.compute_profile(full_text, pages=len(docs))

    print("\n--- Document profile (no LLM calls used) ---")
    print(json.dumps(_to_dict(profile), indent=2, default=str))

    if config.optimizer.seed_from_profile:
        seed = pr.propose_seed_config(
            base_config=config.retrieval,
            profile=profile,
            opt_config=config.optimizer,
        )
        print("\n--- Iteration 1 seed, derived from the document (no LLM calls used) ---")
        print(json.dumps(_to_dict(seed.config), indent=2, default=str))
        print(f"Rationale: {seed.rationale}")
    else:
        print("\nProfile seeding is off (optimizer.seed_from_profile=False) — "
              "iteration 1 will start from config.retrieval as configured.")

    input("\nPress Enter to start the real 5-iteration run "
          "(this will call your configured LLM providers)...")

    # ------------------------------------------------------------------
    # 4. Run the actual optimization loop.
    # ------------------------------------------------------------------
    print("\n--- Running optimization ---\n")
    best = pr.run_experiment(config)

    # ------------------------------------------------------------------
    # 5. Handle both real outcomes honestly: run_experiment() returns {}
    #    if every iteration failed the faithfulness >= 0.80 safety gate —
    #    that is a real possible outcome, not a script bug.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if not best:
        print("NO CONFIGURATION CLEARED THE FAITHFULNESS SAFETY GATE (>= 0.80)")
        print("=" * 60)
        print(
            "Every iteration was discarded even though some may have scored\n"
            "highest on weighted average. Check "
            f"{workspace.results_dir / 'evaluation_scores.csv'} to see each\n"
            "iteration's actual faithfulness score and why it was rejected."
        )
    else:
        print("BEST CONFIGURATION FOUND")
        print("=" * 60)
        print(f"Iteration: {best['iteration']}")
        print("\nConfig:")
        print(json.dumps(_to_dict(best["config"]), indent=2, default=str))
        print("\nScores:")
        for metric, value in best["scores"].items():
            print(f"  {metric:<20}: {value:.4f}")

    # ------------------------------------------------------------------
    # 6. Point to the full results on disk, whichever outcome happened.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Full results written to:")
    print(f"  {workspace.results_dir / 'experiment_results.csv'}   <- all iterations, config + scores")
    print(f"  {workspace.results_dir / 'evaluation_scores.csv'}     <- scores + optimizer reasoning per iteration")
    print(f"  {workspace.results_dir / 'best_configuration.json'}   <- saved only if a best was found")
    print(f"  {workspace.generated_questions_dir / 'benchmark.json'} <- the fixed question set used to score every iteration")
    print("=" * 60)


if __name__ == "__main__":
    main()
