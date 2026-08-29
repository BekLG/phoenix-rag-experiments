# Phoenix RAG

A self-optimizing Retrieval-Augmented Generation (RAG) system. It generates
evaluation questions from a source document, evaluates a RAG pipeline with
Ragas, tunes retrieval parameters based on the results, and repeats until
the best-performing configuration is found (or the target scores are hit).

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
  │   Mistral: Generate Evaluation Questions
  │        │
  │        ▼
  │   Evaluation Benchmark Dataset (fixed, generated once)
  │        │
  ▼        ▼
Retriever ──► RAG Pipeline ──► Generated Answer
                                    │
                                    ▼
                        Ragas + Mistral (judge)
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

```
app.py                  CLI entry point
menu.py                   Terminal front-end (stdlib only)
streamlit_app.py          GUI front-end (streamlit run streamlit_app.py)
operations.py             Operator actions shared by both front-ends
config.py                Dataclasses for all configuration (Mistral, retrieval,
                          question generation, optimizer)
mistral_client.py         Rate-limited, retrying wrapper around the Mistral SDK
document_loader.py       PDF / text ingestion
chunking.py               RecursiveCharacterTextSplitter wrapper
embeddings.py             LangChain Embeddings adapter for Mistral embeddings
vector_store.py           FAISS index build/save/load + retriever factory
corpus.py                 Multi-document corpus: manifest + incremental indexing
question_generator.py    Generates + caches the fixed benchmark question set
document_profile.py       Deterministic document characteristics for tuning
seed_config.py            Derives iteration 1's chunk/top_k from that profile
rag_pipeline.py           Retrieve → prompt → generate
evaluator.py              Ragas evaluation using Mistral as judge
optimizer.py              Rule-based retrieval parameter tuning
storage.py                Persists configs / results / scores / best config
experiment_runner.py      Orchestrates the full optimization loop
config/                   Default saved AppConfig JSON
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
pip install -r requirements.txt

cp .env.example .env
# edit .env and set MISTRAL_API_KEY
```

Set `MistralSettings.requests_per_minute` to match the quota for your Mistral
account. Generation, embedding, and optimization calls use the shared
rate-limited client; Ragas applies its own conservative concurrency limit.
FAISS indexes are cached below `faiss_index_path` using the document contents,
embedding model, chunk size, and overlap, so recurring configurations do not
consume embedding quota again.

Place your source document (PDF or .txt/.md) somewhere under `data/`, e.g.
`data/source.pdf`.

## Usage

### Front-ends

Both front-ends expose the same five operations — optimize, add a document, ask
the RAG, compare old parameters against re-optimized ones, and edit the config —
and both call the same functions in `operations.py`, so neither can drift from
the other.

```bash
python menu.py            # terminal menu (standard library only)
python app.py --menu      # the same thing

streamlit run streamlit_app.py    # GUI (needs the streamlit dependency)
```

```
Phoenix RAG
== corpus: 2 doc(s) | 13 question(s) | best config STALE ==
  1) Optimize RAG
  2) Add document to existing FAISS index
  3) Ask the RAG
  4) Compare old parameters vs re-optimized (new document)
  5) Modify configuration
  6) Show corpus / status
  0) Exit
```

Option 5 edits every field of `config/default_config.json` — including
`question_generation.questions_per_batch` and `batch_size_chars`, which together
set the benchmark size, and `optimizer.max_iterations` — with type coercion and
validation, so a `chunk_overlap` above `chunk_size` or a prompt template missing
`{question}` is rejected at the point of editing rather than mid-run.

### Direct CLI

```bash
# Run with defaults (looks for data/source.pdf)
python app.py

# Point at a specific document
python app.py --source data/my_document.pdf

# Cap the optimization loop
python app.py --source data/my_document.pdf --max-iterations 5

# Force the benchmark question set to regenerate even if a cached one exists
python app.py --source data/my_document.pdf --force-regenerate-questions

# Start iteration 1 from config/default_config.json instead of the document profile
python app.py --source data/my_document.pdf --no-profile-seed

# Optimize against the whole multi-document corpus instead of one file
python app.py --corpus

# Ignore the corpus for one run, even if the saved config enables it
python app.py --no-corpus --source data/my_document.pdf

# Verbose logging
python app.py --source data/my_document.pdf --verbose
```

## Multiple documents: the corpus

By default the system indexes exactly one document. Adding a second one through
the menu or the GUI switches it into **corpus mode** (`AppConfig.corpus_path`,
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
`seed_config.py` sizes iteration 1 for the corpus that will actually be searched.

**Enabling corpus mode is opt-in and reversible.** With `corpus_path` unset,
every code path behaves exactly as it did before. `python app.py --no-corpus`
ignores the corpus for a single run, and the generalization experiment always
clears it — that comparison is only meaningful against one new document.

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
document profile by `seed_config.py`, not read from `config/default_config.json`.
A document-agnostic starting point (previously 300/50/1 for every document) acts
as an anchor the LLM optimizer nudges around: on a 12-page paper needing 800–1200
character chunks, ten consecutive iterations never left the 300–500 band. Seeding
puts iteration 1 in the regime the document's own `doc_type`,
`median_chars_per_page`, and section length imply, so the budget goes on refining
rather than travelling.

`retriever_type`, `similarity_threshold`, and `prompt_template` still come from
the config — choosing those needs measured scores, which do not exist yet at
iteration 1. Pass `--no-profile-seed` (or set `optimizer.seed_from_profile` to
`false`) to restore the old unseeded behaviour; the seed rationale is recorded in
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

The generalization experiment writes its optimization artifacts and comparison
under `results/generalization_experiment/<label>/`, leaving normal run results
untouched.

In corpus mode the equivalents live under `data/corpus/` — `benchmark.json`,
`corpus_summary.txt`, and `corpus_profile.json` — and the per-iteration results
still go to `results/`, so `results/best_configuration.json` always describes
whatever was optimized most recently. The status view (menu option 6, or the GUI
sidebar) says which that was.

## Tests

```bash
python -m unittest discover -p 'test_*.py'
```

The suite is fully offline. `test_corpus.py` patches out summary and question
generation and supplies a fake embedder that counts how many texts it was asked
to embed — that counter is what actually proves an add extends the index instead
of rebuilding it. The FAISS-backed tests skip themselves if `faiss-cpu` is not
installed.

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`

`ragas` unconditionally imports `ChatVertexAI` from
`langchain_community.chat_models.vertexai` at import time — even though
this project never uses Google VertexAI. That module was removed from
recent `langchain-community` releases (VertexAI support now lives in the
separate `langchain-google-vertexai` package), so `from ragas import
evaluate` fails before you can even run the app.

Fix: restore a stub module so the import succeeds, without pulling in the
full VertexAI/GCP SDK just to satisfy an unused import:

```bash
mkdir -p .venv/Lib/site-packages/langchain_community/chat_models
cat > .venv/Lib/site-packages/langchain_community/chat_models/vertexai.py << 'EOF'
class ChatVertexAI:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "ChatVertexAI requires the 'langchain-google-vertexai' package. "
            "Install it with: pip install langchain-google-vertexai"
        )
EOF
```

> On macOS/Linux, the path is `.venv/lib/python3.x/site-packages/...`
> instead of `.venv/Lib/site-packages/...`.

**This stub lives inside `.venv/` and is not tracked by pip**, so it will
be silently wiped out any time you recreate the virtual environment or run
`pip install --upgrade langchain-community` / `pip install -r
requirements.txt` from a clean env. If the error resurfaces, just re-run
the two commands above.
