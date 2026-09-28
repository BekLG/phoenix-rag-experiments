#!/usr/bin/env python3
"""
01_quickstart.py — the simplest possible use of Phoenix RAG.

This is the whole library reduced to its essence: point it at a document,
tell it how many optimization rounds to run, and let it go. Everything else
(chunking, embedding, benchmark generation, Ragas scoring, the optimizer)
happens automatically inside run_experiment().

Every call here was checked directly against the phoenix_rag source
(schema.py, workspace.py, __init__.py) -- not guessed from the README.

USAGE
    source .venv/bin/activate
    pip install -e '.[dev]'
    cp .env.example .env      # set MISTRAL_API_KEY (or your backend's key)
    python 01_quickstart.py data/your_document.pdf
"""

import sys

import phoenix_rag as pr

# ---------------------------------------------------------------------------
# 1. A "workspace" is just a folder Phoenix RAG reads from / writes to --
#    config, the generated benchmark, results, everything. use_workspace()
#    makes it the active one for every call below.
# ---------------------------------------------------------------------------
workspace = pr.Workspace("./demo-01-quickstart").ensure()
pr.use_workspace(workspace)

# ---------------------------------------------------------------------------
# 2. Get a config. If this workspace has no config.yaml yet, this WRITES one
#    for you, using sensible built-in defaults (Mistral backend, mistral-embed
#    for embeddings, etc.) -- you don't have to hand-author a YAML file to
#    get started.
# ---------------------------------------------------------------------------
config = pr.load_or_create_default_config()

# ---------------------------------------------------------------------------
# 3. Point it at your document, and cap the optimization loop. Keep this
#    small (2-3) for a first try -- each iteration costs real API calls.
# ---------------------------------------------------------------------------
config.source_document = sys.argv[1] if len(sys.argv) > 1 else "data/source.pdf"
config.optimizer.max_iterations = 3

# ---------------------------------------------------------------------------
# 4. Run it. This single call:
#      - profiles the document and derives a sensible starting config
#      - generates a fixed benchmark of evaluation questions from it
#      - builds a FAISS index
#      - runs the benchmark through the RAG pipeline
#      - scores every answer with Ragas (an LLM as judge)
#      - has an LLM optimizer propose the next config + prompt
#      - repeats for max_iterations, tracking the best result
# ---------------------------------------------------------------------------
best = pr.run_experiment(config)

# ---------------------------------------------------------------------------
# 5. best is {} if no iteration ever cleared the internal faithfulness safety
#    gate (>= 0.80) -- a real possible outcome, not an error.
# ---------------------------------------------------------------------------
if not best:
    print("No configuration cleared the faithfulness safety gate (>= 0.80).")
    print(f"See {workspace.results_dir / 'evaluation_scores.csv'} for details.")
else:
    print(f"Best iteration: {best['iteration']}")
    print(f"Scores: {best['scores']}")
    print(f"Full results in: {workspace.results_dir}")
