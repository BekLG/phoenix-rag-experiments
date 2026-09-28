#!/usr/bin/env python3
"""
04_ask_the_tuned_rag.py — use the OPTIMIZED system to actually answer
questions, after 03_full_optimization_run.py has produced a best config.

This is the payoff step: optimization isn't the end goal, a working,
tuned RAG system you can query is. AskSession automatically uses your saved
best_configuration.json if one exists -- you don't have to manually copy
chunk_size/top_k/etc. out of the results yourself.

Every call here was checked directly against phoenix_rag's operations.py.

USAGE
    # Run 03_full_optimization_run.py FIRST, against the same workspace,
    # then:
    python 04_ask_the_tuned_rag.py "What does Ragas measure?"
"""

import sys

import phoenix_rag as pr
from phoenix_rag.operations import AskSession

# ---------------------------------------------------------------------------
# Reattach to the SAME workspace the optimization run used -- this is where
# best_configuration.json lives.
# ---------------------------------------------------------------------------
workspace = pr.Workspace("./phoenix-demo").ensure()
pr.use_workspace(workspace)

config = pr.load_or_create_default_config()

# ---------------------------------------------------------------------------
# Building the session does the "expensive" part once: loading (or building)
# the FAISS index for whichever config is active. Reuse `session` for many
# questions rather than creating a new one per question.
# ---------------------------------------------------------------------------
session = AskSession(config)

# ---------------------------------------------------------------------------
# IMPORTANT: this is NOT necessarily config.retrieval as written in
# config.yaml. AskSession checks for a saved best_configuration.json FIRST
# and uses that if it exists -- so you automatically get the TUNED
# parameters, not the raw defaults, without doing anything extra.
# ---------------------------------------------------------------------------
print(f"Using retrieval config from: {session.provenance}")
print(session.retrieval_config.to_dict())
print()

question = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"

result = session.ask(question)

print(f"Q: {question}")
print(f"A: {result.answer}")
print("\nRetrieved context (what the answer is actually grounded in):")
for label in AskSession.attribute(result):
    print(f"  {label}")

# ---------------------------------------------------------------------------
# A session can answer many questions cheaply -- the index is already
# loaded. Uncomment to try a few in a row:
# ---------------------------------------------------------------------------
# for q in ["What metrics does it use?", "How is faithfulness measured?"]:
#     r = session.ask(q)
#     print(f"\nQ: {q}\nA: {r.answer}")
