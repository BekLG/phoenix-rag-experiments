# Phoenix RAG

A self-optimizing Retrieval-Augmented Generation (RAG) system built around a
**corpus of one or more documents**. It evaluates answers against a corpus
benchmark with Ragas, tunes retrieval parameters and the answer prompt, and
repeats until the best-performing configuration is found or its targets are hit.
Every run uses the corpus manifest, benchmark, and indexes, including when the
corpus contains just one file. There is no separate single-file optimization path.

It is **provider-agnostic**: each model role (embedding, generation, optimizer,
judge) can run on a different backend — Mistral, OpenAI, Anthropic, DeepSeek, or
a local model — so you can use whichever API keys or hardware you already have.

## Supported providers

| Backend | Chat | Embeddings | Install extra | Notes |
|---|---|---|---|---|
| `mistral` | ✓ | ✓ | (built in) | Default |
| `openai` | ✓ | ✓ | `[openai]` | |
| `anthropic` | ✓ | — | `[anthropic]` | No embeddings API; pair with another embedding backend |
| `deepseek` | ✓ | — | `[deepseek]` | No embeddings API; pair with another embedding backend |
| `local` | ✓ | ✓ | `[local]` | Ollama / vLLM for chat, sentence-transformers for embeddings |

Roles mix freely, e.g. local generation and embeddings (free) with a cloud
model as optimizer and judge. See [Choosing providers](#choosing-providers).

## Architecture

```
Corpus (one or more registered documents)
      │
  ┌───┴────────────────────┐
  ▼                        ▼
Canonical chunks       Full document texts
  │                        │
  ▼                        ▼
FAISS vectors          LLM: Generate Evaluation Questions
and docstore               │
  │                        ▼
  │                    Corpus benchmark (fixed within a run)
  │                        │
  ▼                        ▼
Retriever ──► RAG Pipeline ──► Generated Answer
                                    │
                                    ▼
                        Ragas + judge LLM
                                    │
                                    ▼
              Faithfulness / Context Recall / Context Precision /
                          Response Relevancy
                                    │
                                    ▼
                        Optimization Engine
                                    │
                                    ▼
                  Tune Retrieval Parameters → Next Iteration
```

**Key design rule:** evaluation questions come from the *complete text* of
registered documents, never from retrieved chunks. Registration normally adds
each document's questions to the corpus benchmark; if the corpus has no
benchmark, one is generated from its full text. The benchmark stays fixed
throughout an optimization run, so every configuration is scored against the
same questions. Adding documents can extend it before the next run.

## Project layout

Phoenix RAG is a `src/`-layout Python package. The optimizer, the pipeline, and
both front-ends are importable, so the library can be driven from your own code
as well as from the CLI.

```
pyproject.toml            Package metadata, dependencies, extras, console script
configs/default.yaml      Committed, commented example configuration
src/phoenix_rag/
    cli.py                  Console entry point (`phoenix-rag`)
    workspace.py            Resolves where data/results/logs/config live
    storage.py              Persists configs / results / scores / best config
    operations.py           Operator actions shared by both front-ends
    config/
        schema.py             Configuration dataclasses (pure data, no I/O)
        loader.py             YAML persistence + `.env` handling
    providers/
        (one rate-limited, retrying wrapper per backend)
        ratelimit.py          Sliding-window rate limiter
    core/
        document_loader.py    PDF / text ingestion
        chunking.py           RecursiveCharacterTextSplitter wrapper
        embeddings.py         LangChain Embeddings adapter for the embedding backend
        vector_store.py       FAISS index build/save/load + retriever factory
        corpus.py             Corpus of one or more documents: manifest + incremental indexing
        document_profile.py   Deterministic document characteristics for tuning
        seed_config.py        Derives iteration 1's chunk/top_k from that profile
        rag_pipeline.py       Retrieve → prompt → generate
    benchmark/
        question_generator.py Generates + caches the fixed benchmark question set
        summarizer.py         Generates + caches the document summary
    evaluation/
        evaluator.py          Ragas evaluation using the configured judge model
        ragas_compat.py       Shims for optional integrations Ragas imports eagerly
    optimization/
        optimizer.py          Bounds clamping + target checks
        llm_optimizer.py      LLM-driven retrieval parameter proposals
        runner.py             Orchestrates the full optimization loop
tests/                    Offline test suite
```

Everything the system *writes* lives in a **workspace** directory, separate from
the installed package — by default the current working directory. You can
override this in code by configuring a `Workspace` before running operations:

```python
from phoenix_rag.workspace import Workspace, use_workspace
use_workspace(Workspace("~/rag-runs/experiment-a"))
```

```
config/config.yaml        The live configuration the app reads and rewrites
data/                     Input documents; indexes live under the corpus root
data/corpus/              Corpus manifest, benchmark, and per-variant indexes
results/                  Per-iteration configs, CSV results, best config
generated_questions/      Legacy artifacts adopted when initializing a corpus
logs/                     Run logs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'         # drop the extra for a runtime-only install

# add the extras for the providers you want:
pip install -e '.[openai]'      # or [anthropic], [deepseek], [local]

cp .env.example .env
# edit .env and set the API key(s) for the providers you use
```

`requirements.txt` remains as the exact pinned set the results in this repo were
produced with; `pyproject.toml` carries the ranges a fresh install resolves.

### Configuration and secrets

Configuration is YAML. `configs/default.yaml` is the commented example; the live
copy is `config/config.yaml` in your workspace, written on first run and rewritten
whenever you save from a front-end.

**No API key ever goes in the YAML.** Each model role names the *environment
variable* to read instead:

```yaml
providers:
  judge:
    backend: mistral
    model: mistral-large-latest
    api_key_env: MISTRAL_API_KEY    # the NAME, resolved from .env
```

That keeps `config/config.yaml` safe to share or diff while the values stay in
`.env`, which is gitignored. There are four roles — `embedding`, `generation`,
`optimizer`, and `judge` — and each one picks its own `backend`, so they can
point at different providers.

Set each role's `requests_per_minute` to match the quota for your account
(local backends can be set high). Generation, embedding, and optimization calls
use a rate-limited client; Ragas applies its own conservative concurrency limit.
FAISS indexes are cached below `corpus_path/indexes/` by embedding model, chunk
size, and overlap, with document membership tracked in the manifest. Recurring
configurations reuse existing vectors; switching the embedding model builds a
new index.

### Choosing providers

**Cloud-only, mixed vendors** — Anthropic as optimizer and judge, OpenAI
embeddings (Anthropic has no embeddings API):

```yaml
providers:
  embedding:
    backend: openai
    model: text-embedding-3-small
    api_key_env: OPENAI_API_KEY
  generation:
    backend: openai
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
  optimizer:
    backend: anthropic
    model: claude-sonnet-5-5          # any model your key can access
    api_key_env: ANTHROPIC_API_KEY
  judge:
    backend: anthropic
    model: claude-sonnet-5-5
    api_key_env: ANTHROPIC_API_KEY
```

**Local generation and embeddings, cloud judge** — a local Qwen served by
Ollama, a local sentence-transformers embedder, and Claude for the roles where
scoring quality matters most:

```yaml
providers:
  embedding:
    backend: local
    model: BAAI/bge-small-en-v1.5
  generation:
    backend: local
    model: qwen2.5:7b
    base_url: http://localhost:11434
  optimizer:
    backend: anthropic
    model: claude-sonnet-5-5
    api_key_env: ANTHROPIC_API_KEY
  judge:
    backend: anthropic
    model: claude-sonnet-5-5
    api_key_env: ANTHROPIC_API_KEY
```

Tips:

- Use a strong model as the **judge** — Ragas scores are only as trustworthy as
  the model that produces them.
- Local inference without a GPU is slow; start with `optimizer.max_iterations = 2`.
- To add a backend: install its extra, set `backend` + `model` + `api_key_env`
  for the role in `config.yaml`, and put the key in `.env`.

Place the files you want to register (PDF, `.txt`, or `.md`) somewhere accessible,
such as `data/`. The corpus records their paths; putting files in that directory
alone does not register them.

## Usage

```python
from phoenix_rag import load_or_create_default_config, run_experiment

config = load_or_create_default_config()
config.source_document = "data/handbook.pdf"  # First member if the corpus is empty.
config.optimizer.max_iterations = 5
best = run_experiment(config)
```

This creates or reuses the workspace's corpus at `data/corpus/` and optimizes
over all its registered documents. A corpus containing only `handbook.pdf` uses
the same pipeline as one containing many files. To grow it, use `add_document()`
as shown in [The corpus: one or many documents](#the-corpus-one-or-many-documents).


## Retrieval strategies

Dense retrieval remains the default. To enable lexical BM25 or dense + BM25
hybrid retrieval, install the optional extra (also included in `[all]`):

```bash
pip install -e '.[retrieval]'
```

Set the flat retrieval fields in your configuration:

```yaml
retrieval:
  retrieval_strategy: hybrid  # dense | bm25 | hybrid
  top_k: 5                   # final chunks passed to the LLM
  hybrid_dense_k: null        # null -> max(2 * top_k, 10)
  hybrid_bm25_k: null         # null -> max(2 * top_k, 10)
  rrf_k: 60
```

`retriever_type` (`similarity`, `mmr`, or `similarity_score_threshold`) and
`similarity_threshold` still configure dense search, including the dense half
of hybrid. BM25 ignores them. Candidate budgets apply only to hybrid; all
strategies pass at most `top_k` chunks to generation. Budgets and `rrf_k` must be
positive integers. Older configs load with the dense defaults.

BM25 reads the same canonical chunks and IDs from the FAISS docstore. It uses
lowercase Unicode regex word tokens for both indexing and queries. Its
in-memory cache belongs to the FAISS store and its chunk size/overlap variant;
pipelines sharing that store reuse the index. A chunk ID/text snapshot detects
corpus growth or replacement and triggers a rebuild from **all** chunks.
Loading a new store builds BM25 once when first requested. No BM25 data is
pickled or persisted, and dense retrieval never imports `rank_bm25`.

Hybrid merges by `chunk_id` and scores each chunk with
`sum(1 / (rrf_k + rank))`, using 1-based ranks from the candidate lists. Raw
dense and BM25 scores are not combined. Equal fused scores sort by best
individual rank, then chunk ID. BM25 score ties also sort by chunk ID, including
zero-score matches. Results expose `dense_rank`, `bm25_rank`, and `rrf_score`
in `extras`; a missing method rank is `None`. DEBUG retrieval logs include
candidate counts, latency, final chunk IDs, and hybrid's component rankings.

FAISS index keys still depend only on embedding model and chunk size/overlap,
with document membership tracked separately. There is no retrieval-result or
evaluation-result cache to invalidate. Per-iteration and best-config JSON
already save the complete config, including these new fields; CSV columns are
unchanged. The optimizer is unchanged, and `copy_with()` preserves the new
fields while allowing its existing `top_k` tuning.

## The corpus: one or many documents

The corpus is the unit of optimization. `run_experiment(config)` loads its
manifest and uses its benchmark, summary, aggregated profile, and index variants
whether it has one member or many.

To register another file in the corpus used above and optimize the expanded set:

```python
from phoenix_rag.operations import add_document

add_document(config, "data/appendix.pdf", label="appendix")
best = run_experiment(config)
```

Call `add_document()` for each additional file; there is no separate mode to
enable when growing from one document to several.

| Setting or operation | Meaning |
|----------------------|---------|
| `corpus_path: null` | Resolve the active workspace's `data/corpus/` on first use and persist that location. |
| Explicit `corpus_path` | Select a corpus root and its registered document set. |
| `source_document` | Seed an **empty** corpus, with a label derived from the file's name. It does not replace or filter a populated corpus. |
| `add_document(config, path)` | Register a file, generate its artifacts, and sync its chunks into the corpus index by default. |

To change which documents participate, add or remove corpus members, or select
another corpus root. Changing `source_document` on a populated corpus does not
change the documents searched. Use a separate workspace when you also want
separate configs and optimization results.

The same on-disk layout is used for one or many files:

```
data/corpus/
    manifest.json         documents, per-variant index membership, per-doc
                          summaries and profiles
    benchmark.json        the corpus benchmark — grows as documents are added
    corpus_summary.txt    the corpus summary the optimizer sees
    corpus_profile.json   the aggregated profile
    indexes/<key>/        one FAISS index per (embedding model, chunk_size, overlap)
```

**Adding is incremental.** A document is added to the *existing* index rather
than replacing it: only the new document's chunks are embedded, and every vector
already in the index is reused. The log line to look for is

```
sync_index: extended the existing index in place: 41 -> 58 vectors,
embedding only 17 new chunk(s) from vectordbs
```

Index identity is `(embedding_model, chunk_size, chunk_overlap)` — deliberately
*not* including the documents — and membership is tracked per variant in the
manifest. That is what allows a variant to gain a document without changing
identity. New `chunk_size` or `chunk_overlap` settings require a separate index
variant because the chunks differ. Existing variants stay in place; switching
back reuses their vectors and embeds only any corpus members they are missing.

**The optimizer sees every document.** Each document's summary and profile are
kept in the manifest. With one member, the corpus summary and profile use that
document's values. With several members, summaries become one briefing that
names each document, and profiles are aggregated into one `DocumentProfile`.
`core/seed_config.py` therefore sizes iteration 1 for the corpus that will
actually be searched.

Documents are identified by a digest of their **contents**, so re-adding the same
file (even renamed) is a no-op rather than a duplicate, and a file edited in place
is reported by the status view. Removing a document invalidates the index
variants that contained it — FAISS has no cheap per-vector removal — so removal
costs a rebuild where adding does not.

### When corpus membership changes

Adding a document generates questions from it and appends them to the benchmark,
because the alternative is quietly broken: with questions only from the older
documents, the new document's chunks are pure retrieval noise, and re-optimizing
would tune the configuration to avoid retrieving it.

The cost is that **scores from before an add are not comparable to scores after
it** — the benchmark itself changed. So the manifest records per-document
question counts, every run logs the benchmark size and composition at the start,
and the saved best configuration is marked `STALE` as soon as membership changes.
Re-run the optimization after adding a document; do not read the previous best
score as if it described the new corpus.

### Where iteration 1 starts

Iteration 1's `chunk_size`, `chunk_overlap`, and `top_k` are derived from the
**aggregated corpus profile** by `core/seed_config.py`, rather than taken from
`config/config.yaml`. A one-document corpus naturally uses that document's
profile. As documents are added, the aggregate reflects the corpus's content
and structure, so the initial settings describe the evidence actually searched.

`retriever_type`, `similarity_threshold`, and `prompt_template` still come from
the config — choosing those needs measured scores, which do not exist yet at
iteration 1. Set `optimizer.seed_from_profile` to `false` to start from the
configured retrieval values instead. When seeding is enabled, its rationale
is recorded in iteration 1's `applied_rules`.

### Migrating an older workspace

Older saved configs can still supply a seed document and cached artifacts.
When the corpus is empty, `adopt_legacy_single_document_artifacts()` reads
`source_document` and any usable `profile_path`, `summary_path`, and
`benchmark_path` caches, normally under `generated_questions/`. It registers
the seed in the corpus and generates only missing artifacts. The original
cache files remain untouched. If there is no usable seed and no registered
member, optimization reports an empty-corpus error; add a document first.

These legacy artifact fields support migration; they do not select an
alternative optimization path. Old indexes under `faiss_index_path` are unused
and are not migrated. Vectors are built in `corpus_path/indexes/<key>/`, then
reused by subsequent corpus runs. Old index folders remain untouched.

## Outputs

- `data/corpus/manifest.json` — documents, their artifacts, and index membership
- `data/corpus/benchmark.json` — the fixed evaluation question set for each run
- `data/corpus/corpus_summary.txt` — the summary supplied to the optimizer
- `data/corpus/corpus_profile.json` — the aggregated document facts
- `data/corpus/indexes/<key>/` — FAISS indexes per embedding and chunking variant
- `results/configs/iteration_NNN.json` — retrieval config used each iteration
- `results/evaluation_scores.csv` — Ragas scores per iteration + which
  optimization rules fired
- `results/experiment_results.csv` — existing parameter columns + scores per
  iteration; the full config, including retrieval strategy, is in the JSON
- `results/best_configuration.json` — the best config found so far, updated
  whenever a new best is found

An explicit `corpus_path` overrides `data/corpus/`. Per-iteration results go to
`results/`, so `results/best_configuration.json` describes whatever was optimized
most recently.

All of these are relative to the active workspace, so two `Workspace`
directories keep entirely separate results.

## Tests

```bash
pytest
```

The suite is fully offline. `tests/test_corpus.py` patches out summary and
question generation and supplies a fake embedder that counts how many texts it
was asked to embed — that counter is what actually proves an add extends the
index instead of rebuilding it. The FAISS-backed tests skip themselves if
`faiss-cpu` is not installed.

`tests/test_runner.py` covers first-run corpus creation from `source_document`
alone and automatic adoption of cached artifacts from saved legacy configs.

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`

`ragas` unconditionally imports `ChatVertexAI` from
`langchain_community.chat_models.vertexai` at import time — even though this
project never uses Google VertexAI. That module was removed from recent
`langchain-community` releases (VertexAI support now lives in the separate
`langchain-google-vertexai` package), so `from ragas import evaluate` would fail
before you could even run the app.

`evaluation/ragas_compat.py` handles this: it registers a stub module in
`sys.modules` before Ragas is imported, so no manual patching of your virtualenv
is needed. If you see this error anyway, something imported `ragas` before
`phoenix_rag.evaluation.evaluator` — import the evaluator first, or call
`install_ragas_compat()` yourself:

```python
from phoenix_rag.evaluation.ragas_compat import install_ragas_compat

install_ragas_compat()
```
