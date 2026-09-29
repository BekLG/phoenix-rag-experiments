# Phoenix RAG

A self-optimizing Retrieval-Augmented Generation (RAG) system. It generates
evaluation questions from a source document, evaluates a RAG pipeline with
Ragas, tunes retrieval parameters based on the results, and repeats until
the best-performing configuration is found (or the target scores are hit).

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
Source Document
      │
      ▼
Recursive Text Splitter
      │
  ┌───┴────┐
  ▼        ▼
FAISS   Full Document Chunks
  │        │
  │   LLM: Generate Evaluation Questions
  │        │
  │        ▼
  │   Evaluation Benchmark Dataset (fixed, generated once)
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

**Key design rule:** evaluation questions are generated once from the
*complete* source document (batched only for LLM context / rate-limit
reasons), never from retrieved chunks. This keeps the benchmark independent
of the retrieval pipeline being tuned — every configuration is scored
against exactly the same questions.

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
        corpus.py             Multi-document corpus: manifest + incremental indexing
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
data/                     Source documents + FAISS index
data/corpus/              Corpus manifest, benchmark, and per-variant indexes
results/                  Per-iteration configs, CSV results, best config
generated_questions/      Cached benchmark question set
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
FAISS indexes are cached below `faiss_index_path` using the document contents,
embedding model, chunk size, and overlap, so recurring configurations do not
consume embedding quota again. Switching the embedding model builds a new index.

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

Place your source document (PDF or .txt/.md) somewhere under `data/`, e.g.
`data/source.pdf`.

## Usage

```python
from phoenix_rag import load_or_create_default_config, run_experiment

config = load_or_create_default_config()
config.optimizer.max_iterations = 5
run_experiment(config)
```


## Multiple documents: the corpus

By default the system indexes exactly one document. Adding a second one switches it into **corpus mode** (`AppConfig.corpus_path`,
rooted at `data/corpus/`), where the benchmark, summary, profile, and FAISS index
all describe every document that has been added:

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

**Enabling corpus mode is opt-in and reversible.** With `corpus_path` unset,
every code path behaves exactly as it did before.

Documents are identified by a digest of their **contents**, so re-adding the same
file (even renamed) is a no-op rather than a duplicate, and a file edited in place
is reported by the status view. Removing a document invalidates the index
variants that contained it — FAISS has no cheap per-vector removal — so removal
costs a rebuild where adding does not.

### One honest caveat

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

- `generated_questions/benchmark.json` — the fixed evaluation question set
- `generated_questions/document_profile.json` — deterministic document facts
  supplied to the optimizer alongside the generated summary
- `results/configs/iteration_NNN.json` — retrieval config used each iteration
- `results/evaluation_scores.csv` — Ragas scores per iteration + which
  optimization rules fired
- `results/experiment_results.csv` — full config + scores per iteration, one
  row each, convenient for plotting/analysis
- `results/best_configuration.json` — the best config found so far, updated
  whenever a new best is found

In corpus mode the equivalents live under `data/corpus/` — `benchmark.json`,
`corpus_summary.txt`, and `corpus_profile.json` — and the per-iteration results
still go to `results/`, so `results/best_configuration.json` always describes
whatever was optimized most recently.

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