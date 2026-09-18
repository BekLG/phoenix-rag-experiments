# Understanding Phoenix RAG

A guide to what this library is, how it's put together, and the ideas you need
to hold in your head to read the code confidently. It assumes you know Python
well, so it spends its time on **design decisions and the "why"**, not on syntax.

Read it top to bottom once; after that it works as a reference — each section is
self-contained.

---

## 1. What the library actually does

Phoenix RAG is a **self-optimizing RAG system**. You point it at a document (or a
folder of documents), and it repeatedly:

1. builds a retrieval-augmented-generation pipeline with some set of parameters,
2. answers a fixed set of benchmark questions with it,
3. has an LLM *judge* score those answers, and
4. asks an LLM *optimizer* to propose a better set of parameters for the next round.

It keeps the best-scoring configuration and stops when it hits its targets or runs
out of iterations. The output is **a tuned retrieval config plus a tuned answer
prompt** — the settings that made this particular document answer best.

The mental model in one sentence:

> It's a search loop over RAG hyperparameters, where the objective function is
> "how good are the answers" as measured by an LLM judge, and the thing proposing
> the next point to try is another LLM.

Everything in the codebase is in service of that loop being *cheap, reproducible,
and swappable across model backends*.

### What "RAG parameters" means here

A RAG pipeline has knobs that dramatically change answer quality:

- **`chunk_size` / `chunk_overlap`** — how the document is sliced before embedding.
  Too small and each chunk lacks context; too big and retrieval returns noise.
- **`top_k`** — how many chunks to retrieve per question.
- **`retriever_type`** — `similarity`, `mmr` (maximal marginal relevance, trades
  some relevance for diversity), or `similarity_score_threshold`.
- **`similarity_threshold`** — the cutoff for the threshold retriever.
- **`prompt_template`** — the instructions wrapped around the retrieved context
  before the model answers.

These form the **search space**. The optimizer moves through it; the judge tells
it whether it's getting warmer.

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
OpenAI / Anthropic / local backends import their SDKs *inside* the builder
function that needs them. That's what makes the optional extras work (§7): you
only need `langchain-openai` installed if a config actually selects the openai
backend.

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
├── cli.py                 # `phoenix-rag` entry point (argparse → run_experiment)
├── workspace.py           # resolves data/, results/, logs/, config/ roots
├── operations.py          # higher-level operations used by the UIs
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
│   ├── embeddings.py       # embedding helpers
│   ├── vector_store.py    # get_or_build_vector_store (content-addressed FAISS)
│   ├── rag_pipeline.py    # RagPipeline: retrieve → prompt → generate
│   ├── document_profile.py# analyze a document (length, structure, …)
│   ├── seed_config.py     # derive iteration-1 params from the profile
│   └── corpus.py          # multi-document mode
│
├── benchmark/
│   ├── question_generator.py  # get_or_create_benchmark (fixed question set)
│   └── summarizer.py          # get_or_create_summary (fed to the optimizer)
│
├── evaluation/
│   ├── evaluator.py       # run_evaluation → Ragas scores
│   └── ragas_compat.py    # shims around ragas/langchain version churn
│
├── optimization/
│   ├── runner.py          # THE LOOP: run_experiment (see §5)
│   ├── llm_optimizer.py   # propose_next_config_llm (the LLM proposer)
│   └── optimizer.py       # meets_targets + rule-based helpers
│
└── ui/
    ├── menu.py            # interactive terminal front-end
    └── streamlit_app.py   # the GUI
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
move a cloud role to OpenAI/Anthropic by changing `backend` + `model` +
`api_key_env` and installing that backend's extra.

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
prepare_inputs(...)                   # gather document-dependent inputs:
        │                             #   benchmark, summary, profile, build_store()
        │
seed iteration 1 from the profile     # §7 — seed_config.propose_seed_config
        │
for iteration in 1..max_iterations:
    ├─ (re)build the FAISS store IF chunk_size/overlap changed   # cached otherwise
    ├─ RagPipeline(store, generation, config).answer_many(qs)    # LOCAL generation
    ├─ run_evaluation(results, benchmark, judge, embedding)      # CLOUD judge (Ragas)
    ├─ storage.save_iteration_config / append_experiment_result  # persist
    ├─ weighted_score = 0.40·faithfulness + 0.20·(recall+precision+relevancy)
    ├─ faithfulness ≥ 0.80 gate → maybe record as new best       # SAFETY GATE
    ├─ if meets_targets(...): stop early
    └─ propose_next_config_llm(config, scores, ..., history, summary, profile)
        → next config (+ prompt) for the next iteration
```

Details that matter:

- **The benchmark is generated once and frozen.** Every configuration is scored
  against the *identical* question set, or the comparison across iterations would
  be meaningless.
- **The LLM optimizer sees the full history.** `propose_next_config_llm` gets the
  entire iteration history plus the document summary and profile, so it reasons
  about trade-offs across the whole run — not just the last score — and proposes
  the retrieval params **and** the prompt template *together*.
- **The safety gate.** The objective is a weighted score, but an answer that
  scores high while being unfaithful to the source (hallucinating) is worthless.
  So a configuration is only eligible to become "best" if `faithfulness ≥ 0.80`.
  A high score that fails the gate is logged and discarded.
- **The store is rebuilt only when it has to be.** The FAISS index depends only on
  `chunk_size`/`chunk_overlap`. The loop caches the store and rebuilds it only when
  those change — an iteration that only tweaks `top_k` or the prompt reuses it.

---

## 7. Cross-cutting ideas you'll keep meeting

**Content-addressed caching.** `get_or_build_vector_store` keys the FAISS index on
`(source, embedding_model, chunk_size, chunk_overlap)`. Same inputs → the cached
index is reused; change any of them → a fresh build. The benchmark, summary, and
profile are cached the same way (`get_or_create_*`). This is what makes repeated
runs cheap and makes the `validate_split_run.py` script able to reuse a cached
benchmark and only rebuild the index (because MiniLM's 384-dim vectors don't match
the cached 1024-dim Mistral ones).

**Seeding from the profile.** `document_profile.py` measures the document; then
`seed_config.propose_seed_config` derives iteration 1's `chunk_size` /
`chunk_overlap` / `top_k` *from that measurement* rather than from a
document-agnostic default. Without this, the LLM optimizer anchors on an arbitrary
starting point and burns its whole budget nudging around it. (`--no-profile-seed`
turns this off and starts from the config's `retrieval` block instead.)

**Corpus mode.** Everything above describes single-document mode. Set
`corpus_path` (or pass `--corpus`) and the benchmark/summary/profile/index describe
*every* document in a folder instead of one file. The clever part of `runner.py` is
that this difference is confined to `prepare_inputs()`: it hands the loop a
`benchmark`, a `summary`, a `profile`, and a `build_store(chunk_size, overlap)`
*callable* — and the scoring, safety gate, history, and proposal logic downstream
have **no idea** which mode is running. That's dependency injection used to erase a
branch from the hot path.

**The workspace.** `workspace.py` resolves where `data/`, `results/`,
`generated_questions/`, `logs/`, and `config/` live — the current directory by
default, or `$PHOENIX_RAG_WORKSPACE`. The CLI creates these dirs; *importing the
library does not*, which keeps the package import side-effect-free.

**Rate limiting.** `ratelimit.py` is a thread-safe sliding-window limiter (a
`deque` of timestamps guarded by a `threading.Lock`; `acquire()` evicts entries
older than the window and blocks if the window is full). It's thread-safe because
Ragas evaluates with a worker pool, so multiple threads call the judge and
embedding providers at once.

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
| **Lazy in-function imports** | each builder in langchain_backends.py; ragas in cli.py | Optional deps + fast `--help` |
| **Decorator/wrapper class** | `RateLimitedEmbeddings`, `LangChainChatProvider` | Add rate limiting without touching the wrapped object |
| **Sliding-window rate limiter** | `ratelimit.py` | Respect an API quota across threads |
| **Content-addressed cache** | `get_or_build_vector_store`, `get_or_create_*` | Cheap, reproducible reruns |
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
jump to **§9.6 (the all-cloud shortcut)**. Otherwise follow §9.1–9.5 for the
shipped local/cloud split.

### 9.1 Prerequisite checklist (shipped split)

| # | Prerequisite | Why | Check it | Section |
|---|---|---|---|---|
| 1 | **Python ≥ 3.10** | the library targets 3.10–3.12 | `python3 --version` | 9.2 |
| 2 | **`uv`** (or plain pip) | creates the venv / installs (this repo's venv is uv-made and has **no** pip) | `uv --version` | 9.2 |
| 3 | **Library installed with `[local]`** | default embedding + generation are local | `.venv/bin/python -c "import phoenix_rag"` | 9.2 |
| 4 | **Mistral API key in `.env`** | optimizer + judge are Mistral | `grep MISTRAL_API_KEY .env` | 9.3 |
| 5 | **Ollama running + model pulled** | local generation calls it | `ollama list` and `curl -s localhost:11434/api/tags` | 9.4 |
| 6 | **A source document** | the thing being optimized | `ls data/*.pdf` | 9.5 |

> On *this* machine, 1, 2, 5, and 6 are already satisfied (Python 3.12 venv via uv
> 0.12.3; Ollama 0.32.6 with `qwen2.5:3b-instruct` pulled and the server up;
> `data/` has several PDFs). You'd mainly be checking 3 and 4.

### 9.2 Steps 1–3: Python, the environment, and installing the library

The venv here is created by **`uv`**, which means **there is no `pip` inside it** —
use `uv pip`. On a fresh machine:

```bash
# (only if uv isn't installed yet)
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the repo root:
uv venv                                              # creates .venv (already exists here)
uv pip install -e '.[local]' --python .venv/bin/python
```

`-e` installs in *editable* mode (your source edits take effect without
reinstalling). `[local]` pulls the local-backend dependencies: `langchain-openai`
(used for the OpenAI-compatible local chat), `langchain-huggingface`, and
`sentence-transformers`.

Pick the extra that matches your config's backends:

```bash
uv pip install -e .            --python .venv/bin/python   # core + Mistral only (all-cloud)
uv pip install -e '.[local]'   --python .venv/bin/python   # + local chat & embeddings
uv pip install -e '.[openai]'  --python .venv/bin/python   # + OpenAI backend
uv pip install -e '.[all]'     --python .venv/bin/python   # everything
```

Then **activate the venv** so the `phoenix-rag` command is on your PATH (installing
the package created it in `.venv/bin/`):

```bash
source .venv/bin/activate      # now `phoenix-rag` works directly
# or, without activating, call it by path: .venv/bin/phoenix-rag ...
```

> **Prefer plain pip?** It works too — just create a normal venv instead:
> `python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[local]'`.
> The uv note only matters because *this* repo's existing venv was made with uv.

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

To move a cloud role to OpenAI or Anthropic instead: add `OPENAI_API_KEY=` or
`ANTHROPIC_API_KEY=` to `.env`, install that extra (§9.2), and point the role's
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
> serves.

### 9.5 Step 6: the config file and a document

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

Put a document at `data/source.pdf` (or point `--source` at any PDF/text file —
this repo already ships a few under `data/`).

`config/config.yaml` is **gitignored** (the app rewrites it as it saves best
configs). The committable template is `configs/default.yaml` — edit that if you
want your defaults tracked in git.

### 9.6 Run it

With the venv activated:

```bash
phoenix-rag --source data/source.pdf                 # single document, default 10 iterations
phoenix-rag --source data/Harvey_Abilene_Paradox.pdf --max-iterations 15
phoenix-rag --no-profile-seed --source data/x.pdf    # start from the config, not the profile
phoenix-rag --corpus                                 # optimize the whole data/corpus/ folder
phoenix-rag --menu                                   # interactive terminal front-end
phoenix-rag --help                                   # all flags
```

For the GUI:

```bash
streamlit run src/phoenix_rag/ui/streamlit_app.py    # needs: uv pip install -e '.[ui]' ...
```

Results — per-iteration configs, scores, and the best configuration found — land
under `results/` in the workspace (the current directory by default, or
`$PHOENIX_RAG_WORKSPACE`). Logs go to `logs/`.

### 9.7 Verify each piece on its own (troubleshooting)

If a full run misbehaves, prove the parts independently before debugging the loop:

```bash
# The whole split, one iteration, against a cached benchmark (every role fires):
.venv/bin/python scripts/validate_split_run.py

# The unit tests (no API calls; proves the wiring):
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
| `ModuleNotFoundError: langchain_huggingface` (or `_openai`) | ran without the matching extra | `uv pip install -e '.[local]' --python .venv/bin/python` |
| error from the judge/optimizer about a missing key | `MISTRAL_API_KEY` not in `.env` | add it to `.env` (§9.3) |
| connection refused to `localhost:11434` | Ollama not running | `ollama serve` |
| `model 'qwen2.5:3b-instruct' not found` | model not pulled, or name mismatch | `ollama pull qwen2.5:3b-instruct`, or edit `generation.model` |
| `phoenix-rag: command not found` | venv not activated | `source .venv/bin/activate` or call `.venv/bin/phoenix-rag` |
| first run hangs on "Loading weights" | MiniLM downloading from HF | wait once; it's cached afterward |

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
3. `core/rag_pipeline.py` — the thing being optimized.
4. `optimization/runner.py` — the loop that ties it together (§6).
5. `config/schema.py` then `config/loader.py` — the knobs and how they load (§5).
6. `optimization/llm_optimizer.py` — how the next config is proposed.

Everything else (`core/corpus.py`, `benchmark/*`, `evaluation/*`, `ui/*`) is a
leaf you can read on demand once the spine above makes sense.
