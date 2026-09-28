#!/usr/bin/env python3
"""
02_inspect_before_optimizing.py — see what the optimizer will do BEFORE
spending a single token.

Everything in this script is free and instant: no LLM calls, no API keys
needed. This is the step worth showing a mentor to demonstrate that Phoenix
RAG's starting point is DERIVED from the document's actual structure, not a
generic one-size-fits-all default.

Every call here was checked directly against the phoenix_rag source
(document_profile.py, seed_config.py, config/schema.py).

USAGE
    python 02_inspect_before_optimizing.py data/your_document.pdf
"""

import json
import sys
from dataclasses import asdict

import phoenix_rag as pr

workspace = pr.Workspace("./demo-02-inspect").ensure()
pr.use_workspace(workspace)

config = pr.load_or_create_default_config()
source_document = sys.argv[1] if len(sys.argv) > 1 else "data/source.pdf"
config.source_document = source_document

# ---------------------------------------------------------------------------
# See which backend/model each of the four independent roles is using.
# (embedding, generation, optimizer, judge can each be a different provider --
#  Mistral, OpenAI, Anthropic, DeepSeek, or a fully local model.)
# ---------------------------------------------------------------------------
print("--- Configured providers (one backend per role) ---")
for role in ("embedding", "generation", "optimizer", "judge"):
    pc = getattr(config.providers, role)
    print(f"  {role:<10}: backend={pc.backend:<10} model={pc.model}")

# ---------------------------------------------------------------------------
# Load the document and compute its structural profile: page count, section
# density, characters-per-section, document type. This is pure computation --
# no LLM involved. This is what iteration 1's config gets derived FROM.
# ---------------------------------------------------------------------------
docs = pr.load_document(source_document)
full_text = "\n\n".join(d.page_content for d in docs)
profile = pr.compute_profile(full_text, pages=len(docs))

print("\n--- Document profile (no LLM calls) ---")
print(json.dumps(asdict(profile), indent=2, default=str))

# ---------------------------------------------------------------------------
# Derive iteration 1's chunk_size / chunk_overlap / top_k FROM that profile.
# A document-agnostic default once caused an optimizer to get stuck at the
# wrong chunk size for ten straight iterations on a document it never fit --
# this seeding step is the fix for that.
# ---------------------------------------------------------------------------
if config.optimizer.seed_from_profile:
    seed = pr.propose_seed_config(
        base_config=config.retrieval,
        profile=profile,
        opt_config=config.optimizer,
    )
    print("\n--- Iteration 1's seeded config (document-derived, no LLM calls) ---")
    print(json.dumps(seed.config.to_dict(), indent=2, default=str))
    print(f"\nWhy: {seed.rationale}")
else:
    print("\nProfile seeding is off -- iteration 1 will use config.retrieval as-is:")
    print(json.dumps(config.retrieval.to_dict(), indent=2, default=str))

# ---------------------------------------------------------------------------
# Chunk the document the same way the pipeline will, so you can see exactly
# how many chunks this config produces -- useful context for judging whether
# top_k is a reasonable fraction of the total.
# ---------------------------------------------------------------------------
chunks = pr.split_documents(
    docs,
    chunk_size=seed.config.chunk_size if config.optimizer.seed_from_profile else config.retrieval.chunk_size,
    chunk_overlap=seed.config.chunk_overlap if config.optimizer.seed_from_profile else config.retrieval.chunk_overlap,
)
print(f"\nThis produces {len(chunks)} chunks from the document.")
