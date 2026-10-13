# Phoenix RAG

A self-optimizing Retrieval-Augmented Generation (RAG) system. It generates
evaluation questions from the documents in a corpus, evaluates a RAG pipeline with
Ragas, tunes retrieval parameters based on the results, and repeats until
the best-performing configuration is found (or the target scores are hit).

**Single-file mode has been removed. A corpus is used for every run, whether
you have one file or many.** The first file initializes the corpus; adding more
files extends that same corpus and its benchmark.

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
Corpus (one or more documents)
      │
      ▼
Recursive Text Splitter
      │
  ┌───┴────┐
  ▼        ▼
FAISS   Full Text of Each Document
  │        │
  │   LLM: Generate Evaluation Questions
  │        │
  │        ▼
  │   Corpus Benchmark Dataset (fixed within each run)
  │        │
  ▼        ▼
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

**Key design rule:** evaluation questions are generated once from each
*complete* source document (batched only for LLM context / rate-limit
reasons), never from retrieved chunks. This keeps the benchmark independent
of the retrieval pipeline being tuned — every configuration within a run is
scored against exactly the same questions. Adding a document extends the
benchmark for subsequent runs.

## Project layout

Phoenix RAG is a `src/`-layout Python package. Use the library from your own
Python code or run the example scripts in this repository.

```
pyproject.toml            Package metadata, dependencies, and provider extras
configs/default.yaml      Committed, commented example configuration
data/                     Source documents and local corpus data
scripts/                  Quickstart, query, corpus-growth, and validation examples
run_optimization.py       Interactive five-iteration corpus optimization example
add_document_and_reoptimize.py  Add a document and optimize the expanded corpus
src/phoenix_rag/
    workspace.py            Resolves where data/results/logs/config live
    storage.py              Persists configs / results / scores / best config
    operations.py           Corpus management, queries, and configuration operations
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
        corpus.py             Corpus for one or many documents + incremental indexing
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
docs/                     Detailed explanation of the system
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
data/                     Source documents
data/corpus/              Corpus manifest, benchmark, and per-variant indexes
results/                  Per-iteration configs, CSV results, best config
generated_questions/      Legacy artifacts adopted when initializing a corpus
logs/                     Run logs
```

`results/`, `logs/`, legacy `generated_questions/` caches, example workspaces,
and Python/test caches are generated locally and ignored by Git. They are not
required in a fresh checkout; runtime directories are created as needed.
Keep source files and any corpus you want to reuse under `data/`.

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

Dependencies and provider extras are defined in `pyproject.toml`.

### Configuration and secrets

Configuration is YAML. `configs/default.yaml` is the commented example; the live
copy is `config/config.yaml` in your workspace, written on first run and rewritten
when configuration is saved by the library or an example script.

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

Place your source documents (PDF or .txt/.md) under `data/`, e.g.
`data/source.pdf`.

## Usage

```python
from phoenix_rag import load_or_create_default_config, run_experiment

config = load_or_create_default_config()
config.source_document = "data/source.pdf"  # Seeds an empty corpus.
config.optimizer.max_iterations = 5
best = run_experiment(config)
```

The first run creates a one-document corpus. Later runs reuse its manifest,
benchmark, and cached indexes. Query the same corpus with the saved best
retrieval configuration:

```python
from phoenix_rag.operations import AskSession

session = AskSession(config)
print(session.ask("What are the main findings?").answer)
```

For an interactive example from the repository root:

```bash
python run_optimization.py data/source.pdf
python add_document_and_reoptimize.py data/second_document.pdf
```

Both scripts default to the `optimization-run/` workspace, so the second command
adds to the first command's corpus. The examples under `scripts/` declare their
own workspace paths; use the same workspace when optimizing, adding, and querying.

## Corpus workflow: one file or many

Every document, including the first, uses the same corpus machinery. There is
no separate mode to enable when moving from one file to multiple files.
To extend the corpus created above:

```python
from phoenix_rag import load_or_create_default_config, run_experiment
from phoenix_rag.operations import add_document

config = load_or_create_default_config()  # Use the same workspace as the first run.
for path in ("data/second_document.pdf", "data/third_document.pdf"):
    add_document(config, path)
best = run_experiment(config)
```

`corpus_path=None` (or `corpus_path: null` in YAML) means that the default corpus
location has not yet been saved; it does not select a single-file mode.
On first use, it is initialized to the active workspace's
`data/corpus/` and persisted in `config/config.yaml`. An explicit `corpus_path`
still selects a custom root. `source_document` seeds an empty corpus with a label
derived from the filename stem; later documents join through `add_document()`.
Once the corpus has members, changing `source_document` does not replace or add
documents; use `add_document()` to register each additional file.
The benchmark, summary, profile, and FAISS index describe every registered
document:

```
data/corpus/
    manifest.json         documents, per-variant index membership, per-doc
                          summaries and profiles
    benchmark.json        the corpus benchmark — grows as documents are added
    corpus_summary.txt    the rendered multi-document summary the optimizer sees
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
identity. Changing `chunk_size` or `chunk_overlap` still forces a full re-embed
into a new variant directory, because differently-sized chunks are different
vectors; the old variant is left in place, so switching back is free.

**The optimizer sees every document.** Each document's own summary and profile
are kept in the manifest and combined for the optimizer: the summaries are
rendered into one multi-document briefing (`corpus_summary.txt`) that names each
document and warns the model not to scope its prompt template to a single
subject, and the profiles are aggregated into one `DocumentProfile` so
`core/seed_config.py` sizes iteration 1 for the corpus that will actually be
searched.

**Existing configurations upgrade automatically.** A saved config with
`source_document` set and `corpus_path: null` needs no manual migration. On first
use, `adopt_legacy_single_document_artifacts()` reads existing `profile_path`,
`summary_path`, and `benchmark_path` caches (normally under
`generated_questions/`) and adopts them into the corpus. Only missing artifacts
are generated, so cached summaries and questions incur no new generation API
cost. The original cache files remain untouched. If the source file is missing
and the corpus is empty, optimization reports an empty-corpus error; restore the
file or add a document before running again.

**Legacy FAISS indexes are rebuilt once.** Old single-document indexes in
content-hash-named folders under `faiss_index_path` are no longer read or migrated.
The corpus builds its vectors under `data/corpus/indexes/<key>/` (or your explicit
corpus root). Those old index folders become orphaned and are safe to delete
manually; Phoenix RAG does not delete them automatically. Profile, summary, and
benchmark caches are adopted as described above; only the FAISS vectors need
rebuilding.

Documents are identified by a digest of their **contents**, so re-adding the same
file (even renamed) is a no-op rather than a duplicate, and a file edited in place
is reported by the status view. Removing a document invalidates the index
variants that contained it — FAISS has no cheap per-vector removal — so removal
costs a rebuild where adding does not.

### Benchmark changes

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
document profile by `core/seed_config.py`, not read from `config/config.yaml`.
A document-agnostic starting point (previously 300/50/1 for every document) acts
as an anchor the LLM optimizer nudges around: on a 12-page paper needing 800–1200
character chunks, ten consecutive iterations never left the 300–500 band. Seeding
puts iteration 1 in the regime the document's own `doc_type`,
`median_chars_per_page`, and section length imply, so the budget goes on refining
rather than travelling.

`retriever_type`, `similarity_threshold`, and `prompt_template` still come from
the config — choosing those needs measured scores, which do not exist yet at
iteration 1. Set `optimizer.seed_from_profile` to `false` to restore the old unseeded behaviour; the seed rationale is recorded in
iteration 1's `applied_rules` either way.

## Outputs

- `data/corpus/manifest.json` — documents, their artifacts, and index membership
- `data/corpus/benchmark.json` — the fixed evaluation question set for each run
- `data/corpus/corpus_summary.txt` — the summary supplied to the optimizer
- `data/corpus/corpus_profile.json` — the aggregated document facts
- `data/corpus/indexes/<key>/` — FAISS indexes per embedding and chunking variant
- `results/configs/iteration_NNN.json` — retrieval config used each iteration
- `results/evaluation_scores.csv` — Ragas scores per iteration + which
  optimization rules fired
- `results/experiment_results.csv` — full config + scores per iteration, one
  row each, convenient for plotting/analysis
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
