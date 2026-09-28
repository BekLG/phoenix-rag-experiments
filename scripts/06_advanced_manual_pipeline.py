#!/usr/bin/env python3
"""
06_advanced_manual_pipeline.py — bypass run_experiment() entirely and drive
every stage of Phoenix RAG by hand.

WHY THIS SCRIPT EXISTS
    01-05 use run_experiment() and the operations.py helpers -- the intended,
    easy path. This script shows the library is NOT a black box: every stage
    run_experiment() calls internally is itself a public function you can
    call directly. Good for a mentor demo that wants to see "under the hood,"
    or for building something run_experiment() doesn't support yet (e.g. a
    custom stopping rule, or scoring only some of the metrics).

Every function/class signature here was checked directly against the
phoenix_rag source: providers/registry.py, core/vector_store.py,
core/rag_pipeline.py, evaluation/evaluator.py, optimization/seed_config.py.

USAGE
    python 06_advanced_manual_pipeline.py data/your_document.pdf
"""

import sys

import phoenix_rag as pr
from phoenix_rag.providers.registry import build_providers
from phoenix_rag.core.vector_store import get_or_build_vector_store
from phoenix_rag.core.rag_pipeline import RagPipeline
from phoenix_rag.evaluation.evaluator import run_evaluation

workspace = pr.Workspace("./demo-06-manual").ensure()
pr.use_workspace(workspace)

config = pr.load_or_create_default_config()
source_document = sys.argv[1] if len(sys.argv) > 1 else "data/source.pdf"
config.source_document = source_document

# ---------------------------------------------------------------------------
# STAGE 1 — build the four role backends yourself.
#
# This is the SAME call run_experiment() makes internally. It builds one
# backend per role (embedding/generation/optimizer/judge) from config.providers,
# sharing a rate limiter across any roles that hit the same API. Doing this
# yourself means you could, for example, swap in a different provider for
# just one role without touching the rest of the script.
# ---------------------------------------------------------------------------
providers = build_providers(config)
print(f"Generation backend : {config.providers.generation.backend}/{config.providers.generation.model}")
print(f"Judge backend      : {config.providers.judge.backend}/{config.providers.judge.model}")

# ---------------------------------------------------------------------------
# STAGE 2 — profile the document and derive a starting retrieval config.
#
# Pure computation, no LLM calls. Same as 02_inspect_before_optimizing.py.
# ---------------------------------------------------------------------------
docs = pr.load_document(source_document)
full_text = "\n\n".join(d.page_content for d in docs)
profile = pr.compute_profile(full_text, pages=len(docs))
seed = pr.propose_seed_config(config.retrieval, profile, config.optimizer)
retrieval_config = seed.config
print(f"\nDerived starting config: chunk_size={retrieval_config.chunk_size}, "
      f"top_k={retrieval_config.top_k}")

# ---------------------------------------------------------------------------
# STAGE 3 — chunk the document and build (or load) the FAISS index yourself.
#
# get_or_build_vector_store is CONTENT-ADDRESSED: it hashes the document's
# raw bytes + embedding model + chunk_size + chunk_overlap into a cache key.
# Call it again with the exact same four inputs and it reuses the existing
# index instead of re-embedding -- no extra code needed for that caching,
# it's built into this one function.
# ---------------------------------------------------------------------------
chunks = pr.split_documents(docs, retrieval_config.chunk_size, retrieval_config.chunk_overlap)
vector_store = get_or_build_vector_store(
    chunks=chunks,
    embeddings=providers.embedding,
    cache_root=workspace.data_dir / "faiss_index",
    source_document=source_document,
    embedding_model=config.providers.embedding.model,
    chunk_size=retrieval_config.chunk_size,
    chunk_overlap=retrieval_config.chunk_overlap,
)
print(f"Index built from {len(chunks)} chunks.")

# ---------------------------------------------------------------------------
# STAGE 4 — generate the fixed benchmark question set yourself.
#
# This is what get_or_create_benchmark does internally, called directly. It's
# generated from the FULL document text, never from retrieved chunks, so
# every retrieval config gets scored against exactly the same questions.
# ---------------------------------------------------------------------------
benchmark = pr.get_or_create_benchmark(
    full_text=full_text,
    generation=providers.generation,
    qg_config=config.question_generation,
    benchmark_path=config.benchmark_path,
)
print(f"Generated {len(benchmark)} benchmark questions.")

# ---------------------------------------------------------------------------
# STAGE 5 — build a RagPipeline by hand and answer every benchmark question.
#
# RagPipeline is the retrieve -> build prompt -> generate step. Nothing here
# is hidden: it's just a vector store, a generation provider, and a retrieval
# config (chunk settings, top_k, retriever_type, the prompt template).
# ---------------------------------------------------------------------------
pipeline = RagPipeline(
    vector_store=vector_store,
    generation=providers.generation,
    retrieval_config=retrieval_config,
)

question_texts = [q.question for q in benchmark]
results = pipeline.answer_many(question_texts)
print(f"Answered {len(results)} questions.")

# You can also ask ONE question directly, outside any benchmark/evaluation:
# single_result = pipeline.answer("What is this document about?")
# print(single_result.answer)

# ---------------------------------------------------------------------------
# STAGE 6 — score the results with Ragas yourself.
#
# This is what the optimizer loop calls after every iteration. It needs a
# judge model (as a LangChain chat model) and the SAME embedding model used
# to build the index, since some Ragas metrics compare embeddings directly.
# ---------------------------------------------------------------------------
scores = run_evaluation(
    results=results,
    questions=benchmark,
    judge=providers.judge,
    judge_embeddings=providers.embedding,
)

print("\n--- Ragas scores (computed manually, no optimizer loop involved) ---")
for metric, value in scores.items():
    print(f"  {metric:<20}: {value:.4f}")

# ---------------------------------------------------------------------------
# From here, you could call phoenix_rag.optimization.llm_optimizer's
# propose_next_config_llm(...) yourself to get ONE optimizer proposal for the
# next iteration, and loop stages 3-6 manually -- which is literally what
# run_experiment() does for you automatically in the simpler scripts (01, 03).
# ---------------------------------------------------------------------------
print("\nThis is exactly what run_experiment() automates across multiple "
      "iterations -- everything above is the same machinery, called by hand.")
