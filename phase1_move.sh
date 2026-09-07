#!/usr/bin/env bash
#
# Phase 1 mechanical file moves. Disposable -- delete this script once it has run.
#
# Every import inside these files has ALREADY been rewritten to the
# phoenix_rag.* paths, so the tree is broken until the moves happen and correct
# immediately after. Nothing here edits file contents.
#
#   bash phase1_move.sh
#
set -euo pipefail

cd "$(dirname "$0")"

if [ "$(git branch --show-current)" != "feat-library-phase1" ]; then
    git checkout -b feat-library-phase1
fi

mkdir -p tests

# ---------------------------------------------------------------------------
# core/ -- loading, chunking, embedding, indexing, retrieval, profiling
# ---------------------------------------------------------------------------
git mv chunking.py document_loader.py document_profile.py embeddings.py \
       rag_pipeline.py seed_config.py vector_store.py corpus.py \
       src/phoenix_rag/core/

# ---------------------------------------------------------------------------
# benchmark/ -- the fixed question set and the document summary
# ---------------------------------------------------------------------------
git mv question_generator.py src/phoenix_rag/benchmark/
git mv document_summarizer.py src/phoenix_rag/benchmark/summarizer.py

# ---------------------------------------------------------------------------
# evaluation/ -- Ragas scoring
# ---------------------------------------------------------------------------
git mv evaluator.py ragas_compat.py src/phoenix_rag/evaluation/

# ---------------------------------------------------------------------------
# optimization/ -- proposals, targets, and the loop that drives them
# ---------------------------------------------------------------------------
git mv optimizer.py llm_optimizer.py src/phoenix_rag/optimization/
git mv experiment_runner.py src/phoenix_rag/optimization/runner.py

# ---------------------------------------------------------------------------
# Top level and front-ends
# ---------------------------------------------------------------------------
git mv operations.py src/phoenix_rag/
git mv menu.py streamlit_app.py src/phoenix_rag/ui/

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
git mv test_corpus.py test_document_profile.py test_llm_optimizer_profile.py \
       test_operations.py test_seed_config.py tests/

# ---------------------------------------------------------------------------
# Removals.
#
# The first four were rewritten in place under src/phoenix_rag/ (config.py ->
# config/schema.py + config/loader.py, mistral_client.py -> providers/mistral.py
# + providers/ratelimit.py, storage.py -> storage.py, app.py -> cli.py), so
# there is no rename for git to detect -- the destinations already exist with
# substantially different contents.
#
# document_generalization_experiment.py is the retired comparison arm.
# ---------------------------------------------------------------------------
git rm -q config.py mistral_client.py storage.py app.py
git rm -q document_generalization_experiment.py

echo
echo "Moves complete. Verify with:"
echo "  pip install -e '.[ui,dev]'"
echo "  python -c 'import phoenix_rag; print(phoenix_rag.__version__)'"
echo "  pytest"
echo "  phoenix-rag --help"
echo
echo "Then delete this script:  git clean -f phase1_move.sh  (or rm phase1_move.sh)"
