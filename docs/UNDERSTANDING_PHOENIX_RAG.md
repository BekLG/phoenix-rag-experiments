# Understanding Phoenix RAG

A guide to what this library is, how it's put together, and the ideas you need
to hold in your head to read the code confidently. It assumes you know Python
well, so it spends its time on **design decisions and the "why"**, not on syntax.

Read it top to bottom once; after that it works as a reference — each section is
self-contained.

---

## 1. What the library actually does

Phoenix RAG is a **self-optimizing RAG system** built around a **corpus of one or
more registered documents**. Every run uses that corpus, including when it has
only one file. There is no separate single-file optimization path. It repeatedly:

1. builds a retrieval-augmented-generation pipeline with some set of parameters,
2. answers a fixed set of benchmark questions with it,
3. has an LLM *judge* score those answers, and
4. asks an LLM *optimizer* to propose a better set of parameters for the next round.

It keeps the best-scoring configuration and stops when it hits its targets or runs
out of iterations. The output is **a tuned retrieval config plus a tuned answer
prompt** — the settings that produced the best measured answers for the corpus.

The corpus manifest records which documents participate. `source_document` can
seed an empty corpus; additional files are registered with
`phoenix_rag.operations.add_document()`. Placing files in a folder does not
automatically register them. Once the corpus has members, optimization uses
all of them, regardless of the value of `source_document`.

The mental model in one sentence:

> It's a search loop over RAG hyperparameters, where the objective function is
> "how good are the answers" as measured by an LLM judge, and the thing proposing
> the next point to try is another LLM.

Everything in the codebase is in service of that loop being *cheap, reproducible,
and swappable across model backends* (Mistral, OpenAI, Anthropic, DeepSeek, or a
local model).

### What "RAG parameters" means here

A RAG pipeline has knobs that dramatically change answer quality:

- **`chunk_size` / `chunk_overlap`** — how each corpus document is sliced before embedding.
  Too small and each chunk lacks context; too big and retrieval returns noise.
- **`top_k`** — the maximum number of final chunks passed to the LLM per question.
- **`retriever_type`** — the dense search method: `similarity`, `mmr` (maximal
  marginal relevance, trades some relevance for diversity), or
  `similarity_score_threshold`.
- **`similarity_threshold`** — the cutoff for dense threshold retrieval.
- **`prompt_template`** — the instructions wrapped around the retrieved context
  before the model answers.

These form the **search space**. The optimizer moves through it; the judge tells
it whether it's getting warmer.

Phase 2 also adds **`retrieval_strategy`**: `dense` (the default), `bm25`, or
`hybrid`. You select this in config; the optimizer's tuning logic is unchanged.
Hybrid has separate candidate budgets, while `top_k` remains the final context
budget. See [§11](#11-phases-1-and-2-retrieval-contracts-and-hybrid-search) for
what both retrieval phases changed and why they matter.

---

## 2. The single most important idea: the provider seam

If you only understand one part of this codebase, make it this one. Everything
else hangs off it.

### The problem it solves

The loop needs to call LLMs for four different jobs, and you want to be able to
run each job on a *different* backend — a cheap local model for the high-volume
work, a strong cloud model for the work that needs to be smart. The naive version
of this scatters `if backend == "openai": ...` branches through the whole
codebase. The seam confines all of that to one package: `phoenix_rag.providers`.

### The four roles

```
ROLES = ("embedding", "generation", "optimizer", "judge")
```

| Role         | What it does                                              | Call volume |
|--------------|-----------------------------------------------------------|-------------|
| `embedding`  | embeds chunks + queries; feeds FAISS and Ragas            | **highest** |
| `generation` | answers benchmark questions, writes them, summarizes      | high        |
| `optimizer`  | proposes the next retrieval config (one call/iteration)   | lowest      |
| `judge`      | scores answers with Ragas (many sub-calls per sample)     | **high**    |

The whole point of splitting these into roles is that they don't have to share a
backend. The shipped config runs `embedding` and `generation` **locally** (no
per-call cost) and `optimizer` and `judge` in the **cloud** (the two that most
need a strong model). See §4.

### Which backends can play which role

| Backend     | Chat (generation / optimizer / judge) | Embeddings                    |
|-------------|----------------------------------------|-------------------------------|
| `mistral`   | ✓                                      | ✓                             |
| `openai`    | ✓                                      | ✓                             |
| `anthropic` | ✓                                      | — (no embeddings API)         |
| `deepseek`  | ✓                                      | — (no embeddings API)         |
| `local`     | ✓ (Ollama / vLLM / LM Studio)          | ✓ (sentence-transformers)     |

So if you use Anthropic or DeepSeek for chat roles, the `embedding` role must
point at a different backend (`local`, `mistral`, or `openai`).

### The two surfaces — `providers/base.py`

This is the subtle bit. There isn't one universal "LLM" interface, because the two
*consumers* of chat models want two different things:

```python
class ChatProvider(ABC):
    @abstractmethod
    def chat(self, messages, *, temperature=0.3, response_format=None) -> str:
        """Send one chat completion; return the reply TEXT."""

    @abstractmethod
    def as_langchain_chat_model(self, *, temperature=0.0) -> "BaseChatModel":
        """Return a LangChain chat model object."""
```

- **`chat(...) -> str`** — generation, question generation, summarization, and the
  optimizer all just send messages and want the reply *text*. They should never
  have to know a LangChain object exists.
- **`as_langchain_chat_model(...) -> BaseChatModel`** — the **judge** feeds
  [Ragas](https://docs.ragas.io/), whose `evaluate(llm=..., embeddings=...)` only
  accepts LangChain-compatible objects. So any backend that can serve as judge
  must be able to *hand out* a LangChain chat model on top of answering `chat`.

Both surfaces live on the same ABC because a single backend (Mistral, say) can
play any role — it just exposes whichever surface that role needs.

#### Embeddings need no wrapper at all

```python
EmbeddingProvider = Embeddings   # a plain alias to langchain_core.embeddings.Embeddings
```

This is a deliberate non-abstraction. A LangChain `Embeddings` is *already* the
one interface that **both** FAISS (index build + retrieval) **and** Ragas
(`evaluate(embeddings=...)`) accept. Wrapping it would only be a layer to unwrap
again at the call site, so `EmbeddingProvider` is that type directly. The alias
exists purely so the code can say `EmbeddingProvider` and read symmetrically next
to `ChatProvider`.

> **Concept worth naming:** *don't introduce an abstraction whose only job is to
> be unwrapped.* The type you'd wrap is already the type both consumers want.

### The bundle — `Providers`

```python
@dataclass
class Providers:
    embedding: EmbeddingProvider
    generation: ChatProvider
    optimizer:  ChatProvider
    judge:      ChatProvider
```

One per role, built together, passed around as a unit. The loop takes a
`Providers` and never thinks about backends again.

### The composition root — `providers/registry.py`

`build_providers(app_config) -> Providers` is the **one place** that reads the
config and constructs concrete backends. This is the classic *composition root*
pattern: object graph wiring happens once, at the edge, and the rest of the code
receives already-wired objects.

Two things happen here that are worth understanding:

**(a) Rate limiters are shared by backend identity, not by role.** The historical
bug this fixes: if you give each of the four roles its own rate limiter sized to
the API's quota, and three of them hit the *same* API, you've authorized 3× the
real quota. The fix keys a limiter on an *identity*:

```python
_Identity = (backend, api_key_env, base_url)
```

All roles that resolve to the same identity share **one** `RateLimiter`, sized to
`min(requests_per_minute)` across those roles. So the process respects the one
quota the API actually enforces.

**(b) Vendor SDKs are imported lazily, inside each builder.** The Mistral client is
a hard dependency (it's the default and the registry imports it eagerly), but
OpenAI / Anthropic / DeepSeek / local backends import their SDKs *inside* the
builder function that needs them. That's what makes the optional extras work
(see the patterns table in §8 and the install steps in §9.2): you only need
`langchain-openai` installed if a config actually selects the openai backend.

### The generic LangChain adapter — `providers/langchain_backends.py`

Rather than a hand-written class per backend, one generic class wraps any
LangChain `BaseChatModel`:

```python
class LangChainChatProvider(ChatProvider):
    # holds a BaseChatModel + a shared RateLimiter,
    # parameterized by model_factory: Callable[[float], BaseChatModel]
```

`build_openai_chat`, `build_anthropic_chat`, `build_deepseek_chat`, and
`build_local_chat` all produce
one of these — they differ only in *which* `BaseChatModel` they construct.
`RateLimitedEmbeddings` does the same decorator trick for any embeddings backend.
The **local** chat backend is just `ChatOpenAI` pointed at an OpenAI-compatible
`base_url` (Ollama / vLLM / LM Studio); local embeddings are
`HuggingFaceEmbeddings` (sentence-transformers) running in-process, with no rate
limiter because there's no remote quota to respect.

### PEP 562 lazy package — `providers/__init__.py`

The package's `__getattr__` defers importing the heavy submodules until a name is
actually accessed, so `import phoenix_rag.providers` doesn't drag in every vendor
SDK. This is [PEP 562 module-level `__getattr__`](https://peps.python.org/pep-0562/)
— the module-scope equivalent of a lazy attribute.

> **The payoff of this whole section:** to add a new backend you write one builder
> in `langchain_backends.py`, register it in `registry.py`, and add an extra in
> `pyproject.toml`. Nothing else in the codebase changes.

---

## 3. From flat scripts to a `src/` package

This library started life as a folder of top-level scripts (`app.py`,
`rag_pipeline.py`, `evaluator.py`, …). The conversion was structural:

- **`src/` layout.** The package lives in `src/phoenix_rag/`, not at the repo root.
  This prevents "it imports because the CWD happens to be on `sys.path`" accidents
  and forces you to install (or set `pythonpath`) to import it — the same way a
  user would. (`pyproject.toml` sets `pythonpath = ["src"]` so the test suite runs
  straight from a checkout without an install.)
- **Subpackages by responsibility**, not by file type:

```
phoenix_rag/
├── cli.py                 # console entry point (`phoenix-rag`)
├── workspace.py           # resolves data/, results/, logs/, config/ roots
├── operations.py          # higher-level operations shared by the front-ends
├── storage.py             # writes iteration configs, scores, best-config to disk
│
├── config/
│   ├── schema.py          # config as DATA: dataclasses + validation, no I/O
│   └── loader.py          # config as I/O: YAML/env read+write, migration
│
├── providers/             # THE SEAM (see §2)
│   ├── base.py            # ChatProvider ABC, EmbeddingProvider alias, Providers
│   ├── registry.py        # build_providers() composition root
│   ├── ratelimit.py       # sliding-window RateLimiter
│   ├── langchain_backends.py  # generic LangChain adapters + per-backend builders
│   └── mistral.py         # the default backend
│
├── core/                  # the RAG mechanics
│   ├── document_loader.py # PDF/text → LangChain Documents
│   ├── chunking.py        # split_documents(chunk_size, overlap)
│   ├── chunk_identity.py  # deterministic document_id and chunk_id
│   ├── embeddings.py      # embedding helpers
│   ├── vector_store.py    # FAISS build/load/save and incremental additions
│   ├── rag_pipeline.py    # RagPipeline: retrieve → prompt → generate
│   ├── document_profile.py# analyze a document (length, structure, …)
│   ├── seed_config.py     # derive iteration-1 params from the profile
│   └── corpus.py          # all documents: manifest + incremental indexing
│
├── retrieval/             # interchangeable retrieval strategies (see §11)
│   ├── base.py            # Retriever protocol and RetrievalResult
│   ├── factory.py         # build_retriever(config, vector_store)
│   ├── dense.py           # existing FAISS search behind the common contract
│   ├── bm25.py            # lexical search over the same canonical chunks
│   └── hybrid.py          # merge dense + BM25 rankings with RRF
│
├── benchmark/
│   ├── question_generator.py  # generate_benchmark (fixed question set)
│   └── summarizer.py          # generate_summary (fed to the optimizer)
│
├── evaluation/
│   ├── evaluator.py       # run_evaluation → Ragas scores
│   └── ragas_compat.py    # shims around ragas/langchain version churn
│
└── optimization/
    ├── runner.py          # THE LOOP: run_experiment (see §6)
    ├── llm_optimizer.py   # propose_next_config_llm (the LLM proposer)
    └── optimizer.py       # meets_targets + rule-based helpers
```

Outside the package, at the repo root:

```
run_optimization.py             # script entry point for a run
add_document_and_reoptimize.py  # add a document to the corpus, then re-run
scripts/                        # helpers, e.g. validate_split_run.py (see §9.7)
configs/default.yaml            # committed, commented example configuration
tests/                          # offline test suite
```

The organizing principle: **`core/` knows nothing about optimization**, and
**`optimization/` knows nothing about which backend is behind a provider**. Each
layer depends only on the interfaces below it.

---

## 4. The cloud/local split

The shipped `config/config.yaml` assigns roles like this:

```yaml
providers:
  embedding:   { backend: local,  model: sentence-transformers/all-MiniLM-L6-v2 }
  generation:  { backend: local,  model: qwen2.5:3b-instruct, base_url: http://localhost:11434/v1 }
  optimizer:   { backend: mistral, model: mistral-medium-latest, api_key_env: MISTRAL_API_KEY }
  judge:       { backend: mistral, model: mistral-medium-latest, api_key_env: MISTRAL_API_KEY }
```

The reasoning is a cost/quality trade sorted by call volume:

- The two **highest-volume** roles (embedding, generation) run **locally**, where
  each call is free. Embedding fires on every chunk and every query; generation
  answers every benchmark question every iteration.
- The two roles that most need a **strong, consistent** model (optimizer, judge)
  run in the **cloud**. The judge in particular has to be trustworthy — it's the
  objective function.

You can collapse this to all-Mistral (no local model needed) by pointing every
role at `backend: mistral`; the config file documents that fallback inline. Or
move a cloud role to OpenAI/Anthropic/DeepSeek by changing `backend` + `model` +
`api_key_env` and installing that backend's extra. Remember the embedding
constraint from §2: Anthropic and DeepSeek can serve chat roles only, so keep
`embedding` on `local`, `mistral`, or `openai`.

Changing the embedding model selects a different FAISS index variant. If that
variant does not exist, the next run embeds every corpus document into it; if
it already exists, corpus syncing reuses it and adds any missing members
(see corpus caching in §7).

---

## 5. Config: data vs. behavior, and the credentials rule

The config layer is split into two modules on purpose.

### `config/schema.py` — config *as data*

Plain dataclasses (`AppConfig`, `ProvidersConfig`, `ProviderConfig`,
`RetrievalConfig`, `QuestionGenerationConfig`, `OptimizerConfig`) with validation
and `to_dict()`. **No I/O.** This is the in-memory shape of a configuration and
the rules for what's valid. A `ProviderConfig` looks like:

```python
backend="mistral"      # which backend serves this role
model=""               # the model name
api_key_env=None       # the NAME of the env var holding the key — never the key
base_url=None          # for local/OpenAI-compatible endpoints
requests_per_minute=45
max_retries=5
base_backoff_seconds=2.0
```

### `config/loader.py` — config *as I/O*

Reads and writes YAML, loads `.env`, and migrates legacy configs. `load_env()`
loads the `.env` file once; `load_or_create_default_config()` resolves in order:
`config/config.yaml` → a migrated legacy `default_config.json` → built-in defaults.

### The credentials rule (a hard constraint)

> **API keys never appear in YAML.** A role's config names the *environment
> variable* that holds its key (`api_key_env: MISTRAL_API_KEY`), and the value
> lives in `.env`, which is gitignored. `resolve_api_key()` does the
> `os.getenv(api_key_env)` lookup at build time.

This is enforced structurally: the schema has no field for a key *value*, and the
loader's legacy-migration path **deliberately drops** any inline `api_key` it finds
in an old config (warning you to move it to `.env`). So there is no code path that
persists a secret to a committable file. Local roles need no key at all
(`api_key_env: null`).

---

## 6. The optimization loop — `optimization/runner.py`

`run_experiment(app_config)` is the whole thing. Read it once with this map:

```
build_providers(app_config)          # §2 — wire backends ONCE for the run
        │
prepare_inputs(...)                   # initialize/load the corpus:
        │                             #   benchmark, corpus summary/profile, build_store()
        │
seed iteration 1 from corpus profile  # §7 — seed_config.propose_seed_config
        │
for iteration in 1..max_iterations:
    ├─ (re)build the FAISS store IF chunk_size/overlap changed   # cached otherwise
    ├─ RagPipeline(store, generation, config).answer_many(qs)    # generation role
    ├─ run_evaluation(results, benchmark, judge, embedding)      # judge role (Ragas)
    ├─ storage.save_iteration_config / append_experiment_result  # persist
    ├─ weighted_score = 0.40·faithfulness + 0.20·(recall+precision+relevancy)
    ├─ faithfulness ≥ 0.80 gate → maybe record as new best       # SAFETY GATE
    ├─ if meets_targets(...): stop early
    └─ propose_next_config_llm(config, scores, ..., history, summary, profile)
        → next config (+ prompt) for the next iteration
```

Details that matter:

- **The corpus benchmark is fixed within a run.** Every configuration is scored
  against the *identical* question set, or the comparison across iterations would
  be meaningless. Questions come from registered documents' *complete* text,
  never from retrieved chunks. Registration normally extends the saved benchmark;
  if none exists, it is generated from the full corpus text. Adding documents
  can change the benchmark before the next run.
- **The LLM optimizer sees the full history.** `propose_next_config_llm` gets the
  entire iteration history plus the corpus summary and aggregated profile, so it reasons
  about trade-offs across the whole run — not just the last score — and proposes
  the retrieval params **and** the prompt template *together*.
- **The safety gate.** The objective is a weighted score, but an answer that
  scores high while being unfaithful to the source (hallucinating) is worthless.
  So a configuration is only eligible to become "best" if `faithfulness ≥ 0.80`.
  A high score that fails the gate is logged and discarded.
- **The store is rebuilt only when it has to be.** The FAISS index depends only on
  the corpus, embedding model, and chunking parameters. Within a run, the loop
  caches the store and rebuilds it when `chunk_size`/`chunk_overlap` changes.
  Changing `top_k`, the prompt, or `retrieval_strategy` does not require new vectors.

### What a run writes

Everything lands in the workspace (§7):

- `data/corpus/manifest.json` — documents, their artifacts, and index membership
- `data/corpus/benchmark.json` — the fixed question set for each run
- `data/corpus/corpus_summary.txt` — the summary supplied to the optimizer
- `data/corpus/corpus_profile.json` — aggregated document facts
- `data/corpus/indexes/<key>/` — one FAISS index per embedding/chunking variant
- `results/configs/iteration_NNN.json` — the config used each iteration
- `results/evaluation_scores.csv` — Ragas scores per iteration + which rules fired
- `results/experiment_results.csv` — existing parameter columns + scores, one row
  per iteration; the complete config, including strategy fields, is in the JSON
- `results/best_configuration.json` — best config so far, updated on each new best

---

## 7. Cross-cutting ideas you'll keep meeting

**Corpus caching.** `corpus.sync_index()` keys each FAISS index on
`(embedding_model, chunk_size, chunk_overlap)` and tracks document membership in
the manifest. An unchanged variant is reused; new documents extend it, and
different embedding or chunking parameters build a separate variant. The corpus
also persists its benchmark and each document's summary and profile. This lets
repeated runs reuse those artifacts while rebuilding only the index when, for
example, switching from Mistral embeddings to MiniLM.

**Seeding from the corpus profile.** `document_profile.py` measures each document,
and `corpus.profile()` aggregates those measurements. A one-document corpus uses
that document's profile; a larger corpus combines its members' characteristics.
`seed_config.propose_seed_config` derives iteration 1's `chunk_size` /
`chunk_overlap` / `top_k` from this aggregate, so the starting configuration
reflects the corpus being searched instead of an arbitrary default.
`retriever_type`, `similarity_threshold`, and `prompt_template` still come from
the config, because choosing those needs measured scores that don't exist yet at
iteration 1. Set `optimizer.seed_from_profile: false` to start from the config's
`retrieval` block instead. When seeding is enabled, its rationale is recorded
in iteration 1's `applied_rules`.

**One corpus path from the first document.** `corpus_path: null` resolves to the
workspace's `data/corpus/`; an explicit path selects another corpus root.
`prepare_inputs()` always loads that corpus and hands the loop its benchmark,
summary, profile, and a `build_store(chunk_size, overlap)` callable.
`source_document` registers the first member only when the corpus is empty.
Changing it afterward does not replace or filter existing members. Use
`add_document()` or `remove_document()` to change membership, and a separate
workspace when you want separate configs and results as well.

Older configs may supply cached profile, summary, and benchmark files that are
adopted during first registration. This is migration into the same corpus path;
legacy FAISS directories are unused. See
[Migrating an older workspace](../README.md#migrating-an-older-workspace) for
those compatibility details. Corpus behavior follows these rules:

- **Adding is incremental.** A new document is added to the *existing* FAISS index;
  only its new chunks are embedded and every vector already in the index is reused.
  Index identity is `(embedding_model, chunk_size, chunk_overlap)` — deliberately
  *not* including the documents — with per-variant membership tracked in
  `data/corpus/manifest.json`. New chunking parameters require another variant;
  returning to an existing variant reuses its vectors and adds missing members.
- **Documents are identified by content digest**, so re-adding the same file (even
  renamed) is a no-op. Removing a document invalidates the index variants that
  contained it, so removal costs a rebuild where adding does not.
- **The optimizer sees every document** through `corpus_summary.txt` and one
  aggregated `DocumentProfile`. With one member, these use that document's
  values; with several, the summary describes all members and the profile
  combines their characteristics.
- **Adding a document changes the benchmark.** New questions are appended, so scores
  from before an add are **not comparable** to scores after it, and the saved best
  config is marked `STALE` as soon as membership changes. Re-run the optimization
  after an add (`add_document_and_reoptimize.py` does both steps).

**The workspace.** `workspace.py` resolves where `data/`, `results/`,
`generated_questions/`, `logs/`, and `config/` live — the current directory by
default, or `$PHOENIX_RAG_WORKSPACE`. These directories are created on demand
when data is written, which keeps the package import side-effect-free. Two
workspaces keep entirely separate results.

**Rate limiting.** `ratelimit.py` is a thread-safe sliding-window limiter (a
`deque` of timestamps guarded by a `threading.Lock`; `acquire()` evicts entries
older than the window and blocks if the window is full). It's thread-safe because
Ragas evaluates with a worker pool, so multiple threads call the judge and
embedding providers at once.

**Ragas compatibility shim.** `ragas` unconditionally imports
`ChatVertexAI` from a `langchain_community` module that recent releases removed.
`evaluation/ragas_compat.py` registers a stub in `sys.modules` before Ragas is
imported, so nothing needs patching in your virtualenv. If you ever see that
`ModuleNotFoundError`, something imported `ragas` before
`phoenix_rag.evaluation.evaluator`.

---

## 8. Python patterns worth naming

You know Python, so here's the shortlist of patterns this codebase leans on, with
where to see each:

| Pattern | Where | Why |
|---|---|---|
| **ABC with two abstract methods** | `ChatProvider` (base.py) | Two consumer surfaces on one contract |
| **Type alias as a stand-in for an interface** | `EmbeddingProvider = Embeddings` | Avoid a pass-through wrapper |
| **`@dataclass` as a typed bundle** | `Providers`, all of `config/schema.py` | Structure without boilerplate |
| **Composition root** | `build_providers` (registry.py) | Wire the object graph once, at the edge |
| **Callable injection** | `ExperimentInputs.build_store`, `model_factory` | Erase a branch; defer expense |
| **PEP 562 module `__getattr__`** | `providers/__init__.py` | Lazy submodule import |
| **Lazy in-function imports** | each builder in langchain_backends.py | Optional deps |
| **Decorator/wrapper class** | `RateLimitedEmbeddings`, `LangChainChatProvider` | Add rate limiting without touching the wrapped object |
| **Sliding-window rate limiter** | `ratelimit.py` | Respect an API quota across threads |
| **Persistent corpus cache** | `corpus.sync_index`, corpus manifest and benchmark | Cheap, reproducible reruns |
| **Optional extras** | `pyproject.toml` `[openai]`/`[anthropic]`/`[deepseek]`/`[local]`/`[all]` | Only install the backend you use |

---

## 9. Running it: prerequisites and a step-by-step guide

This section is the practical one. It tells you **what must be in place before you
run**, **how to add each piece**, and then **the exact steps** to a working run.

### 9.0 What you need depends on which backends your config uses

There is no single fixed prerequisite list — you only need the pieces for the
backends the four roles actually point at. Two common setups:

| Setup | Roles | You need | You do NOT need |
|---|---|---|---|
| **Shipped split** (default) | embedding + generation → local; optimizer + judge → Mistral | Python, the `[local]` extra, a Mistral key, Ollama + the model | — |
| **All-cloud (simplest)** | all four → Mistral | Python, a Mistral key | Ollama, the `[local]` extra, sentence-transformers |

If you just want the shortest path to a run and don't mind paying for cloud calls,
jump to **§9.8 (the all-cloud shortcut)**. Otherwise follow §9.1–9.6 for the
shipped local/cloud split.

### 9.1 Prerequisite checklist (shipped split)

| # | Prerequisite | Why | Check it | Section |
|---|---|---|---|---|
| 1 | **Python ≥ 3.10** | the library targets 3.10–3.12 | `python3 --version` | 9.2 |
| 2 | **`uv`** (or plain pip) | creates the venv / installs (a uv-made venv has **no** pip inside) | `uv --version` | 9.2 |
| 3 | **Library installed with `[local]`** | default embedding + generation are local | `.venv/bin/python -c "import phoenix_rag"` | 9.2 |
| 4 | **Mistral API key in `.env`** | optimizer + judge are Mistral | `grep MISTRAL_API_KEY .env` | 9.3 |
| 5 | **Ollama running + model pulled** | local generation calls it | `ollama list` and `curl -s localhost:11434/api/tags` | 9.4 |
| 6 | **A corpus with at least one document, or a file to seed it** | supplies the evidence and benchmark for optimization | inspect the corpus with `operations.status(config)` | 9.5 |

### 9.2 Steps 1–3: Python, the environment, and installing the library

If your venv is created by **`uv`**, there is **no `pip` inside it** — use
`uv pip`. On a fresh machine:

```bash
# (only if uv isn't installed yet)
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the repo root:
uv venv                                              # creates .venv
uv pip install -e '.[local]' --python .venv/bin/python
```

`-e` installs in *editable* mode (your source edits take effect without
reinstalling). `[local]` pulls the local-backend dependencies: `langchain-openai`
(used for the OpenAI-compatible local chat), `langchain-huggingface`, and
`sentence-transformers`.

Pick the extra that matches your config's backends:

```bash
uv pip install -e .               --python .venv/bin/python   # core + Mistral only (all-cloud)
uv pip install -e '.[local]'      --python .venv/bin/python   # + local chat & embeddings
uv pip install -e '.[openai]'     --python .venv/bin/python   # + OpenAI backend
uv pip install -e '.[anthropic]'  --python .venv/bin/python   # + Anthropic backend
uv pip install -e '.[deepseek]'   --python .venv/bin/python   # + DeepSeek backend
uv pip install -e '.[all]'        --python .venv/bin/python   # everything
```

Add `dev` (e.g. `'.[dev,local]'`) if you want the test tooling.

Then **activate the venv** so you can run scripts and use the library:

```bash
source .venv/bin/activate
# or, without activating, run your scripts with: .venv/bin/python ...
```

> **Prefer plain pip?** It works too — just create a normal venv instead:
> `python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[local]'`.
> The uv note only matters if your venv was made with uv.

### 9.3 Step 4: the Mistral API key (`.env`)

The optimizer and judge roles are Mistral, so they need a key. **The key value goes
in `.env`, never in the YAML** — the YAML only names the environment variable.
There's a template to copy:

```bash
cp .env.example .env
# then edit .env and set the real value:
#   MISTRAL_API_KEY=sk-...your-key...
```

Get a key from the Mistral console (https://console.mistral.ai/). `.env` is
gitignored, so it never gets committed. A missing key is **not** an import-time
crash — it surfaces as an error from the first call that needs it (i.e. once the
judge or optimizer runs).

To move a cloud role to OpenAI, Anthropic, or DeepSeek instead: add
`OPENAI_API_KEY=`, `ANTHROPIC_API_KEY=`, or `DEEPSEEK_API_KEY=` (whatever name you
give `api_key_env`) to `.env`, install that extra (§9.2), and point the role's
`backend`/`model`/`api_key_env` at it in the YAML (§9.5).

### 9.4 Step 5: Ollama for local generation

The shipped `generation` role calls an OpenAI-compatible endpoint at
`http://localhost:11434/v1` — that's Ollama. You need it installed, running, and
holding the model named in the config.

```bash
# install (Linux/macOS); see https://ollama.com/download for other platforms
curl -fsSL https://ollama.com/install.sh | sh

ollama serve                       # start the server (often already running as a service)
ollama pull qwen2.5:3b-instruct    # pull the model the config names (~1.9 GB)

# verify:
ollama list                        # qwen2.5:3b-instruct should be listed
curl -s http://localhost:11434/api/tags   # server responds with JSON
```

The **local embedding** model (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB) is
*not* an Ollama model — it downloads automatically from Hugging Face into an
in-process cache the first time you run (so the first run needs internet and is a
bit slower). No server, no key.

> Any OpenAI-compatible server works here, not just Ollama — vLLM or LM Studio too.
> Just set `generation.base_url` to that server and `generation.model` to a model it
> serves. Without a GPU, local generation is slow; start with a small
> `max_iterations`.

### 9.5 Step 6: the config file and corpus

Create your live config from the committable example, then confirm which backend
serves each role:

```bash
cp configs/default.yaml config/config.yaml
```

Open `config/config.yaml` and check the `providers:` block. The shipped values are:

```yaml
providers:
  embedding:  { backend: local,   model: sentence-transformers/all-MiniLM-L6-v2, api_key_env: null }
  generation: { backend: local,   model: qwen2.5:3b-instruct, base_url: http://localhost:11434/v1, api_key_env: null }
  optimizer:  { backend: mistral,  model: mistral-medium-latest, api_key_env: MISTRAL_API_KEY }
  judge:      { backend: mistral,  model: mistral-medium-latest, api_key_env: MISTRAL_API_KEY }
```

- `model` must match a model your Ollama actually has (see `ollama list`).
- `api_key_env` names an environment variable — it must exist in `.env` for cloud
  roles, and be `null` for local roles.
- Set each cloud role's `requests_per_minute` to your account's quota.

For a new corpus, choose its first document (PDF, `.txt`, or `.md`) and set these
top-level fields in the same config:

```yaml
corpus_path: null                    # defaults to the workspace's data/corpus/
source_document: data/handbook.pdf    # seeds the corpus only when it is empty
```

Ensure that file exists before the first run. For a populated corpus, select
its `corpus_path`; its manifest determines which documents are used, and
`source_document` is ignored. Register additional files with `add_document()`
as shown below. The corpus directory stores the manifest and generated
artifacts; it does not automatically import files placed there.

`config/config.yaml` is **gitignored** (the app rewrites it as it saves best
configs). The committable template is `configs/default.yaml` — edit that if you
want your defaults tracked in git.

### 9.6 Run it

With the venv activated, optimize the configured corpus from Python. This is
the same call for one or many registered files:

```python
from phoenix_rag import load_or_create_default_config, run_experiment

config = load_or_create_default_config()
config.optimizer.max_iterations = 5

best = run_experiment(config)
```

To grow that corpus and re-optimize over both the existing and new documents:

```python
from phoenix_rag.operations import add_document

add_document(config, "data/appendix.pdf", label="appendix")
best = run_experiment(config)
```

On an empty corpus, either operation first attempts to register the configured
seed. Once populated, the corpus retains its membership across runs. An empty
corpus without a usable seed must have a document added before optimization.

Results — per-iteration configs, scores, and the best configuration found — land
under `results/` in the workspace (the current directory by default, or
`$PHOENIX_RAG_WORKSPACE`). Logs go to `logs/`.

### 9.7 Verify each piece on its own (troubleshooting)

If a full run misbehaves, prove the parts independently before debugging the loop:

```bash
# The whole split, one iteration, against a cached benchmark (every role fires):
.venv/bin/python scripts/validate_split_run.py

# The unit tests (fully offline, no API calls; proves the wiring):
.venv/bin/python -m pytest -q

# Local generation only:
curl -s http://localhost:11434/api/chat -d \
  '{"model":"qwen2.5:3b-instruct","messages":[{"role":"user","content":"say pong"}],"stream":false}'

# Is the Mistral key loaded?
.venv/bin/python -c "from phoenix_rag.config import load_env; import os; load_env(); \
  print('MISTRAL_API_KEY set:', bool(os.getenv('MISTRAL_API_KEY')))"
```

Common failures and their cause:

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langchain_huggingface` (or `_openai`, `_anthropic`) | ran without the matching extra | `uv pip install -e '.[local]' --python .venv/bin/python` (or the right extra) |
| `ModuleNotFoundError: ...chat_models.vertexai` | something imported `ragas` before the evaluator | import `phoenix_rag.evaluation.evaluator` first, or call `install_ragas_compat()` (§7) |
| error from the judge/optimizer about a missing key | `MISTRAL_API_KEY` not in `.env` | add it to `.env` (§9.3) |
| connection refused to `localhost:11434` | Ollama not running | `ollama serve` |
| `model 'qwen2.5:3b-instruct' not found` | model not pulled, or name mismatch | `ollama pull qwen2.5:3b-instruct`, or edit `generation.model` |
| first run hangs on "Loading weights" | MiniLM downloading from HF | wait once; it's cached afterward |
| best config shows `STALE` after adding a document | benchmark changed (§7) | re-run the optimization |

### 9.8 The all-cloud shortcut (no Ollama, no local extra)

Don't want to run a local model at all? Point every role at Mistral. Then the only
prerequisites are Python, the base install, and a Mistral key:

```bash
uv pip install -e . --python .venv/bin/python   # no [local] needed
cp .env.example .env                            # set MISTRAL_API_KEY
```

Edit `config/config.yaml` so all four roles read:

```yaml
providers:
  embedding:  { backend: mistral, model: mistral-embed,          api_key_env: MISTRAL_API_KEY }
  generation: { backend: mistral, model: mistral-small-latest,   api_key_env: MISTRAL_API_KEY }
  optimizer:  { backend: mistral, model: mistral-large-latest,   api_key_env: MISTRAL_API_KEY }
  judge:      { backend: mistral, model: mistral-large-latest,   api_key_env: MISTRAL_API_KEY }
```

Every call now goes to the cloud (simpler to set up, but you pay per call and the
high-volume embedding/generation roles are the bulk of that cost).

---

## 10. A reading order for the source

If you want to read the code rather than this summary, this order builds
understanding without backtracking:

1. `providers/base.py` — the contracts (§2).
2. `providers/registry.py` — how they're built (§2).
3. `retrieval/base.py` then `retrieval/factory.py` — retrieval results and strategy
   construction (§11); follow into `dense.py`, `bm25.py`, or `hybrid.py` as needed.
4. `core/rag_pipeline.py` — the thing being optimized.
5. `core/corpus.py` — document membership, shared artifacts, and index syncing (§7).
6. `optimization/runner.py` — the corpus optimization loop (§6).
7. `config/schema.py` then `config/loader.py` — the knobs and how they load (§5).
8. `optimization/llm_optimizer.py` — how the next config is proposed.

Everything else (`benchmark/*`, `evaluation/*`) is a
leaf you can read on demand once the spine above makes sense.

---

## 11. Phases 1 and 2: retrieval contracts and hybrid search

These phases let Phoenix retrieve evidence in more than one way while keeping
the same retrieve → prompt → generate pipeline.

| Phase | What changed | Main benefit |
|-------|--------------|--------------|
| **1: retrieval contract and identities** | Added `Retriever`, `RetrievalResult`, `DenseRetriever`, and stable document/chunk IDs. | Every retrieval method can describe and identify its hits consistently. |
| **2: BM25 and hybrid retrieval** | Added lexical retrieval, Reciprocal Rank Fusion (RRF), and a config-driven factory. | You can combine semantic similarity with word matching while controlling the final context size. |

### Phase 1: give every retriever the same contract

Previously, the pipeline worked directly with LangChain's FAISS retriever and its
returned `Document` objects. Phase 1 introduced a small protocol in
[retrieval/base.py](../src/phoenix_rag/retrieval/base.py):

```python
class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievalResult]:
        ...
```

A `RetrievalResult` carries the chunk's text and metadata, its `document_id` and
`chunk_id`, a 1-based rank, a retrieval method, and the method's score. It also
has an `extras` dictionary for method-specific details. The reserved
`reranker_score` field stays `None`; neither phase runs a reranker.

[DenseRetriever](../src/phoenix_rag/retrieval/dense.py) adapts the existing FAISS
search to that contract. It preserves the previous search parameters, ordering,
and filtering. Score meanings remain explicit:

- `similarity`: the raw FAISS score; for the indexes Phoenix builds, this is
  squared L2 distance, where lower is better.
- `similarity_score_threshold`: LangChain's relevance score after threshold
  filtering, where higher is better.
- `mmr`: `None`, because that search returns documents without scores.

The pipeline builds the same prompt from the result texts. `answer()` exposes
the structured hits through `RagResult.retrieval_results`, while the existing
`pipeline.retrieve()` convenience API still returns LangChain documents. Ragas
continues receiving the context strings it already consumed.

**Benefit:** retrieval implementations can change without rewriting prompt
construction or evaluation. A caller can also inspect which chunks were used,
their order, and the meaning of their scores. Phase 1 itself preserves dense
retrieval behavior; its benefit is a consistent interface and better traceability.

### Phase 1: identify a chunk independently of its rank

Two retrievers may return the same passage at different ranks. To merge those
hits, Phoenix needs an identity that does not depend on the query, retrieval
order, or FAISS's generated docstore keys.

[core/chunk_identity.py](../src/phoenix_rag/core/chunk_identity.py) assigns:

- **`document_id`** from the corpus's existing content digest, or derives a
  deterministic identity when one is missing. For a file-backed corpus document,
  renaming the file preserves its identity; changing its contents changes it.
- **`chunk_id`** by hashing `[document_id, exact chunk text, occurrence]`.
  The occurrence counts identical passages within that document, so repeated
  text gets distinct IDs. Adding an unrelated document does not change existing
  chunks' identities.

IDs are assigned during canonical chunking. Legacy FAISS stores missing these
fields are backfilled from their full docstore in insertion order, without
embedding the text again. If the original document digest and source file are
both unavailable, a deterministic text fallback provides an identity but cannot
recover the original file digest.

**Benefit:** hybrid retrieval can recognize the same chunk in both lists and
include it once. Stable IDs also make comparisons across queries, logs, and
corpus extensions easier to follow, while preserving source attribution.

### Phase 2: retrieve by meaning, words, or both

[build_retriever(config, vector_store)](../src/phoenix_rag/retrieval/factory.py)
selects the implementation. `RagPipeline` calls this factory, so normal answers
use the selected strategy automatically.

| `retrieval_strategy` | How it finds evidence | When it can help |
|----------------------|-----------------------|------------------|
| `dense` | Searches embedding vectors through FAISS. | Questions phrased differently from the source, where meaning matters more than shared words. |
| `bm25` | Ranks chunks by query-token matches, accounting for term frequency, rarity, and chunk length. | Distinctive names, technical terms, and identifiers that appear in the source. |
| `hybrid` | Gets candidates from dense and BM25, then fuses their ranks. | Questions that benefit from both semantic similarity and specific word matches. |

For example, a question about "reducing model overconfidence" may benefit from
semantic matches to passages about calibration. A question naming "BM25Okapi"
gives lexical search a distinctive token to find. Hybrid gives both methods a
chance to supply evidence. Whether that improves answer quality for your corpus
still needs to be measured against its benchmark.

BM25 uses one deterministic tokenizer for both chunks and questions: lowercase
text, then Unicode-aware regex word tokens (`\w+`). It has no stemming or heavy
NLP dependency. Its chunks come directly from the **same FAISS docstore** used
by dense retrieval; there is no separate lexical chunking pass.

**Benefit:** both methods search the same units of evidence. Differences in
their results come from ranking, and their IDs can be merged reliably. BM25
queries make no embedding call, although the current pipeline still prepares
the canonical FAISS store through its usual embedding/index setup.

### How hybrid combines results: a worked RRF example

Raw dense distances and BM25 scores have different scales and meanings. Hybrid
instead uses each hit's 1-based position in its candidate list:

```text
rrf_score(chunk) = sum(1 / (rrf_k + rank))
                  over the lists that returned that chunk
```

Suppose dense returns `[A, B, C]` and BM25 returns `[B, D, A]`. With `rrf_k=60`:

| Chunk | Dense rank | BM25 rank | RRF score |
|-------|------------|-----------|-----------|
| B | 2 | 1 | `1/62 + 1/61 ≈ 0.03252` |
| A | 1 | 3 | `1/61 + 1/63 ≈ 0.03227` |
| D | — | 2 | `1/62 ≈ 0.01613` |
| C | 3 | — | `1/63 ≈ 0.01587` |

The fused order is `[B, A, D, C]`. B and A each appear once despite being found
by both methods. With `top_k=3`, the LLM receives B, A, and D, with fresh ranks
1, 2, and 3. Equal fused scores sort by the best individual rank, then by
`chunk_id`, making tie-breaking deterministic.

Hybrid results set `retrieval_method="hybrid"` and use the fused score as
`retrieval_score`. Their `extras` contain `dense_rank`, `bm25_rank`, and
`rrf_score`; a missing method's rank is `None`.

**Benefit:** highly ranked evidence from either method can survive, and agreement
between methods adds weight without needing to normalize incompatible raw
scores. The recorded component ranks also explain how a fused result was ranked.

### Selecting a strategy and controlling context size

Dense remains the default and needs no new dependency. BM25 and hybrid require
the optional extra, also included in `[all]`:

```bash
pip install -e '.[retrieval]'
```

For example, in the existing YAML retrieval block:

```yaml
retrieval:
  retrieval_strategy: hybrid
  top_k: 5
  hybrid_dense_k: null
  hybrid_bm25_k: null
  rrf_k: 60
  retriever_type: similarity
```

Each `null` candidate budget resolves to `max(2 * top_k, 10)`, so this example
asks each method for up to 10 candidates and passes at most 5 fused chunks to
generation. Explicit `hybrid_dense_k` and `hybrid_bm25_k` values override those
candidate counts. The candidate fields and `rrf_k` apply only to hybrid.
`retriever_type` and `similarity_threshold` configure dense search, including
the dense half of hybrid; BM25 ignores them.

Unknown strategies and non-positive budgets or `rrf_k` raise clear errors.
Old configs load with `retrieval_strategy="dense"`, unset candidate budgets,
and `rrf_k=60`. The BM25 builder imports `rank_bm25` only when needed and gives
an installation hint if it is missing.

**Benefit:** you can broaden the pool of candidate evidence while keeping a
single final context budget, `top_k`. The existing optimizer still tunes that
budget, and `copy_with()` and config serialization preserve the new fields.
These phases do not add optimizer logic for selecting strategies or tuning RRF.

### Reusing indexes, growing the corpus, and inspecting results

BM25 builds an in-memory index once for a FAISS store and chunk size/overlap
combination. Queries and newly constructed pipelines sharing that unchanged
store reuse it. A snapshot of chunk IDs and text detects corpus changes; when
the snapshot changes, BM25 rebuilds from **all** stored chunks. This matters
when adding documents: BM25 covers both old and new corpus members. Loading a
fresh FAISS store builds a fresh BM25 object when requested. BM25 is not
persisted or pickled.

FAISS disk keys still depend on embedding model and chunk size/overlap, with
document membership tracked separately. Switching retrieval strategies does
not invalidate the vectors. There is no retrieval-result or evaluation-result
cache to re-key. Per-iteration and best-config JSON record the full config;
existing CSV columns stay unchanged.

At DEBUG level, retrieval logs show the strategy, candidate counts, latency,
and final chunk IDs. Hybrid also logs dense, BM25, and fused top IDs.

**Benefit:** switching strategies reuses existing vectors, growing the corpus
keeps lexical retrieval current, and logs expose the evidence selected before
the LLM generates an answer. Phases 1 and 2 establish this retrieval path;
reranking and changes to CLI, UI, optimizer, or evaluation schemas remain for
later work.
