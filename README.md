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
config.py                Dataclasses for all configuration (Mistral, retrieval,
                          question generation, optimizer)
mistral_client.py         Rate-limited, retrying wrapper around the Mistral SDK
document_loader.py       PDF / text ingestion
chunking.py               RecursiveCharacterTextSplitter wrapper
embeddings.py             LangChain Embeddings adapter for Mistral embeddings
vector_store.py           FAISS index build/save/load + retriever factory
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

# Verbose logging
python app.py --source data/my_document.pdf --verbose
```

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
