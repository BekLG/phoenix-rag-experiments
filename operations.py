"""
operations.py
=============
The operator-facing operations, with no user interface attached.

Both front-ends -- menu.py (terminal) and streamlit_app.py (GUI) -- are thin
presentation layers over this module. Everything that involves a decision
(which retrieval config is "current"? does the corpus need bootstrapping? is
this edit to default_config.json valid?) lives here exactly once, so the two
UIs cannot drift apart in behaviour. The UIs are responsible only for reading
input and formatting output.

The five operations the menu exposes map onto this module as:

    optimize RAG           -> experiment_runner.run_experiment, after ensure_corpus
    add document           -> add_document
    ask the RAG            -> AskSession
    compare old vs new     -> document_generalization_experiment.run_generalization_experiment
    modify configuration   -> editable_fields / apply_field / save_config

Two of those are just re-exports of existing entry points, and that is the
point: this module adds the operator plumbing around them, not a second
implementation of them.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import corpus
import storage
from chunking import split_documents
from config import (
    CORPUS_DIR,
    DATA_DIR,
    DEFAULT_CONFIG_PATH,
    AppConfig,
    RetrievalConfig,
    load_or_create_default_config,
)
from document_loader import load_document
from embeddings import MistralEmbeddings
from llm_optimizer import validate_prompt_template
from rag_pipeline import RagPipeline, RagResult
from vector_store import get_or_build_vector_store

logger = logging.getLogger("phoenix_rag.operations")


# =====================================================================
# Config loading / saving
# =====================================================================

def load_config() -> AppConfig:
    """The live configuration, creating config/default_config.json if absent."""
    return load_or_create_default_config()


def save_config(app_config: AppConfig) -> Path:
    """Persist edits back to the same file load_config() reads."""
    app_config.save(DEFAULT_CONFIG_PATH)
    logger.info("Configuration saved to %s", DEFAULT_CONFIG_PATH)
    return DEFAULT_CONFIG_PATH


# =====================================================================
# Corpus access
# =====================================================================

def corpus_root(app_config: AppConfig) -> Path:
    """Where this configuration's corpus lives, whether or not it is enabled."""
    return Path(app_config.corpus_path) if app_config.corpus_path else CORPUS_DIR


def enable_corpus(app_config: AppConfig, save: bool = True) -> AppConfig:
    """Turn on multi-document mode, persisting the switch.

    Called by add_document rather than asked of the operator: adding a second
    document IS the request to run in corpus mode, and a corpus that only the
    current process knows about would be silently ignored by the next
    `python app.py` run.
    """
    if not app_config.corpus_path:
        app_config.corpus_path = str(CORPUS_DIR)
        logger.info("Multi-document corpus mode enabled (root: %s)", CORPUS_DIR)
        if save:
            save_config(app_config)
    return app_config


def corpus_state(app_config: AppConfig) -> corpus.Corpus:
    """Load the corpus manifest for this configuration (empty if none exists)."""
    return corpus.load(corpus_root(app_config))


# =====================================================================
# Status
# =====================================================================

@dataclass
class StatusReport:
    """Everything a UI needs to tell the operator where the system stands."""

    corpus_enabled: bool
    corpus_root: Path
    documents: list[corpus.CorpusDocument]
    question_count: int
    optimization_status: str  # "never" | "stale" | "current"
    integrity_problems: list[str]
    source_document: str
    source_document_exists: bool
    best: dict | None
    active_retrieval_source: str
    index_variants: list[corpus.CorpusIndex]

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def headline(self) -> str:
        """One line for a menu header or a GUI caption."""
        if not self.corpus_enabled:
            name = Path(self.source_document).name
            state = {"never": "not optimized", "stale": "STALE", "current": "optimized"}
            return f"single document: {name} | {state.get(self.optimization_status, '')}"
        state = {
            "never": "never optimized",
            "stale": "best config STALE",
            "current": "best config current",
        }
        return (
            f"corpus: {self.document_count} doc(s) | "
            f"{self.question_count} question(s) | "
            f"{state.get(self.optimization_status, self.optimization_status)}"
        )


def status(app_config: AppConfig) -> StatusReport:
    """Describe the current state of the corpus, config and saved best result."""
    best = storage.load_best_configuration()
    enabled = bool(app_config.corpus_path)

    if enabled:
        state = corpus_state(app_config)
        return StatusReport(
            corpus_enabled=True,
            corpus_root=state.root,
            documents=state.documents,
            question_count=_benchmark_size(state),
            optimization_status=corpus.optimization_status(state),
            integrity_problems=corpus.verify_documents(state),
            source_document=app_config.source_document,
            source_document_exists=Path(app_config.source_document).exists(),
            best=best,
            active_retrieval_source=resolve_active_retrieval(app_config).provenance,
            index_variants=list(state.indexes.values()),
        )

    return StatusReport(
        corpus_enabled=False,
        corpus_root=corpus_root(app_config),
        documents=[],
        question_count=_single_benchmark_size(app_config),
        optimization_status="current" if best else "never",
        integrity_problems=[],
        source_document=app_config.source_document,
        source_document_exists=Path(app_config.source_document).exists(),
        best=best,
        active_retrieval_source=resolve_active_retrieval(app_config).provenance,
        index_variants=[],
    )


def _benchmark_size(state: corpus.Corpus) -> int:
    """Questions actually on disk, which is the number that will be scored."""
    return state.total_questions()


def _single_benchmark_size(app_config: AppConfig) -> int:
    from question_generator import load_benchmark

    path = Path(app_config.benchmark_path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return len(load_benchmark(path))
    except Exception:  # noqa: BLE001 -- a status view must never be what crashes
        logger.exception("Could not read the benchmark for the status view")
        return 0


# =====================================================================
# Add a document
# =====================================================================

@dataclass
class AddDocumentResult:
    document: corpus.CorpusDocument
    sync_report: corpus.SyncReport | None
    bootstrapped: corpus.CorpusDocument | None
    questions_before: int
    questions_after: int
    document_count: int

    def describe(self) -> list[str]:
        lines = []
        if self.bootstrapped is not None:
            lines.append(
                f"corpus seeded with the existing document "
                f"'{self.bootstrapped.label}' (cached artifacts reused, no API cost)"
            )
        lines.append(
            f"added '{self.document.label}' ({self.document.characters:,} chars"
            + (f", {self.document.pages} pages" if self.document.pages else "")
            + f", doc_type={self.document.document_profile.doc_type})"
        )
        lines.append(
            f"benchmark: {self.questions_before} -> {self.questions_after} question(s)"
        )
        if self.sync_report is not None:
            lines.append(f"index: {self.sync_report.describe()}")
        lines.append(f"corpus now holds {self.document_count} document(s)")
        return lines


def add_document(
    app_config: AppConfig,
    path: str | Path,
    label: str | None = None,
    generate_questions: bool = True,
    sync: bool = True,
) -> AddDocumentResult | None:
    """Add a document to the corpus and grow the existing index in place.

    Returns None if this file's content is already in the corpus -- adding the
    same document twice is a no-op, not an error, because the corpus identifies
    documents by content digest.

    `sync=True` embeds the new document's chunks into the index for the CURRENT
    retrieval parameters right away, so the operator sees the incremental add
    happen (and can ask the RAG about the new document immediately) instead of
    paying for it at the start of the next optimization run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    enable_corpus(app_config)
    state = corpus_state(app_config)

    # First add converts the single-document setup into a corpus. Doing this here
    # rather than making the operator do it explicitly is what keeps the existing
    # document from being silently dropped out of the index.
    bootstrapped = corpus.bootstrap_from_single_document(state, app_config)

    questions_before = _benchmark_size(state)
    document = corpus.register_document(
        state,
        app_config,
        path=path,
        label=label or path.stem,
        generate_questions=generate_questions,
    )
    if document is None:
        logger.info("%s is already in the corpus; nothing to do", path.name)
        return None
    questions_after = _benchmark_size(state)

    sync_report = None
    if sync:
        active = resolve_active_retrieval(app_config)
        embeddings = MistralEmbeddings(app_config.mistral)
        _store, sync_report = corpus.sync_index(
            state,
            embeddings=embeddings,
            embedding_model=app_config.mistral.embedding_model,
            chunk_size=active.config.chunk_size,
            chunk_overlap=active.config.chunk_overlap,
        )

    return AddDocumentResult(
        document=document,
        sync_report=sync_report,
        bootstrapped=bootstrapped,
        questions_before=questions_before,
        questions_after=questions_after,
        document_count=len(state.documents),
    )


def remove_document(app_config: AppConfig, doc_id_or_label: str):
    """Drop a document from the corpus by doc_id or label."""
    state = corpus_state(app_config)
    document = state.find(doc_id_or_label) or state.find_by_label(doc_id_or_label)
    if document is None:
        return None
    return corpus.remove_document(state, document.doc_id)


# =====================================================================
# Which retrieval config is "current"?
# =====================================================================

@dataclass
class ActiveRetrieval:
    config: RetrievalConfig
    provenance: str
    from_best: bool


def resolve_active_retrieval(app_config: AppConfig) -> ActiveRetrieval:
    """The retrieval config to serve queries with, and where it came from.

    The saved best configuration wins when there is one -- that is the whole
    output of an optimization run, and answering with the untuned default block
    instead would make the tuning invisible. The provenance string is returned
    alongside so the UI can always say which one is in use; an operator who
    cannot tell whether they are querying tuned or untuned parameters cannot
    interpret the answer.
    """
    best = storage.load_best_configuration()
    if best and isinstance(best.get("config"), dict):
        try:
            config = RetrievalConfig.from_dict(best["config"])
        except (TypeError, KeyError):
            logger.exception(
                "Saved best configuration at %s is unusable; falling back to the "
                "configured retrieval block",
                storage.BEST_CONFIG_PATH,
            )
        else:
            return ActiveRetrieval(
                config=config,
                provenance=(
                    f"{storage.BEST_CONFIG_PATH.name} "
                    f"(iteration {best.get('iteration', '?')})"
                ),
                from_best=True,
            )
    return ActiveRetrieval(
        config=app_config.retrieval,
        provenance=f"{DEFAULT_CONFIG_PATH.name} retrieval block (not yet optimized)",
        from_best=False,
    )


# =====================================================================
# Ask the RAG
# =====================================================================

class AskSession:
    """A loaded index plus a pipeline, reusable across questions.

    Construction is the expensive part (loading or extending the index), so a UI
    builds one session and asks many questions through it. Streamlit keeps it in
    session_state; the terminal menu keeps it for the duration of the ask loop.
    """

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        active = resolve_active_retrieval(app_config)
        self.retrieval_config = active.config
        self.provenance = active.provenance

        embeddings = MistralEmbeddings(app_config.mistral)

        if app_config.corpus_path:
            state = corpus_state(app_config)
            if state.is_empty:
                corpus.bootstrap_from_single_document(state, app_config)
            if state.is_empty:
                raise RuntimeError(
                    "The corpus is empty and could not be seeded from "
                    f"{app_config.source_document}. Add a document first."
                )
            store, report = corpus.sync_index(
                state,
                embeddings=embeddings,
                embedding_model=app_config.mistral.embedding_model,
                chunk_size=self.retrieval_config.chunk_size,
                chunk_overlap=self.retrieval_config.chunk_overlap,
            )
            self.scope = (
                f"corpus of {len(state.documents)}: "
                + ", ".join(d.label for d in state.documents)
            )
            self.index_note = report.describe()
            self.labels = [d.label for d in state.documents]
        else:
            source = Path(app_config.source_document)
            if not source.exists():
                raise FileNotFoundError(f"Source document not found: {source}")
            documents = load_document(source)
            chunks = split_documents(
                documents,
                chunk_size=self.retrieval_config.chunk_size,
                chunk_overlap=self.retrieval_config.chunk_overlap,
            )
            store = get_or_build_vector_store(
                chunks=chunks,
                embeddings=embeddings,
                cache_root=app_config.faiss_index_path,
                source_document=str(source),
                embedding_model=app_config.mistral.embedding_model,
                chunk_size=self.retrieval_config.chunk_size,
                chunk_overlap=self.retrieval_config.chunk_overlap,
            )
            self.scope = f"single document: {source.name}"
            self.index_note = f"index ready ({corpus.vector_count(store)} vectors)"
            self.labels = [source.stem]

        self.store = store
        self.pipeline = RagPipeline(store, app_config.mistral, self.retrieval_config)

    def ask(self, question: str) -> RagResult:
        return self.pipeline.answer(question)

    @staticmethod
    def attribute(result: RagResult) -> list[str]:
        """One short provenance label per retrieved context, in order.

        Falls back to the raw `source` path for chunks embedded before corpus
        provenance stamping existed, so an older index still produces something
        readable rather than a column of "unknown".
        """
        labels = []
        for index, metadata in enumerate(result.sources, start=1):
            label = metadata.get("doc_label")
            if not label:
                source = metadata.get("source")
                label = Path(source).name if source else "unknown"
            page = metadata.get("page")
            suffix = f", page {page + 1}" if isinstance(page, int) else ""
            labels.append(f"[{index}] {label}{suffix}")
        # An index built before `sources` existed yields no metadata at all;
        # still emit one row per context so the numbering lines up.
        while len(labels) < len(result.contexts):
            labels.append(f"[{len(labels) + 1}] unknown")
        return labels


# =====================================================================
# Configuration editor
# =====================================================================

class ConfigEditError(ValueError):
    """A rejected configuration edit, with a message meant for the operator."""


# Fields whose runtime value does not reveal the intended type. Everything else
# is classified from the value itself, which is both simpler and more reliable
# than parsing annotations -- config.py uses `from __future__ import annotations`,
# so dataclasses.fields() hands back strings like "tuple", not real types.
_FLOAT_FIELDS = {
    "retrieval.similarity_threshold",
    "mistral.base_backoff_seconds",
    "question_generation.dedup_similarity_threshold",
    "optimizer.target_faithfulness",
    "optimizer.target_context_recall",
    "optimizer.target_context_precision",
    "optimizer.target_response_relevancy",
    "optimizer.similarity_threshold_step",
}
_FLOAT_PAIR_FIELDS = {"optimizer.similarity_threshold_bounds"}
_TEXT_FIELDS = {"retrieval.prompt_template"}
_SECRET_FIELDS = {"mistral.api_key"}
_CHOICE_FIELDS = {
    "retrieval.retriever_type": ("similarity", "mmr", "similarity_score_threshold"),
}
# Set by add_document / enable_corpus, and pointing it at a stale root silently
# changes which documents are searched. Editable through the corpus operations,
# not through the free-text field editor.
_HIDDEN_FIELDS = {"corpus_path"}

_HELP = {
    "question_generation.questions_per_batch": (
        "Questions generated per text batch -- the 'questions per chunk' knob. "
        "Multiplied by the number of batches to give the benchmark size."
    ),
    "question_generation.batch_size_chars": (
        "Characters of document text per question-generation batch. Smaller "
        "batches mean more batches, so more questions in total."
    ),
    "question_generation.dedup_similarity_threshold": (
        "Cosine similarity above which two generated questions count as duplicates."
    ),
    "optimizer.max_iterations": "How many configurations the optimizer will try.",
    "optimizer.seed_from_profile": (
        "Derive iteration 1's chunk_size/overlap/top_k from the document profile "
        "instead of using the retrieval block below."
    ),
    "retrieval.chunk_size": "Characters per chunk. Changing this forces a full re-embed.",
    "retrieval.chunk_overlap": "Characters shared between neighbouring chunks; must be < chunk_size.",
    "retrieval.top_k": "Chunks retrieved per question.",
    "retrieval.prompt_template": (
        "Answer prompt. Must contain exactly one {context} and one {question}."
    ),
    "mistral.requests_per_minute": "Client-side rate limit shared by every Mistral call.",
    "source_document": "The document used when corpus mode is off, and the seed for a new corpus.",
}


@dataclass
class ConfigField:
    """One editable leaf of the AppConfig tree."""

    path: str
    value: object
    kind: str  # bool | int | float | str | text | choice | int_pair | float_pair | str_list
    choices: tuple[str, ...] | None
    secret: bool
    help: str

    @property
    def display_value(self) -> str:
        if self.secret:
            text = str(self.value or "")
            if not text:
                return "(unset)"
            return f"{text[:4]}...{text[-4:]} ({len(text)} chars)" if len(text) > 12 else "(set)"
        if self.kind == "text":
            text = str(self.value).replace("\n", " ")
            return f"{text[:57]}..." if len(text) > 60 else text
        if self.kind in {"int_pair", "float_pair", "str_list"}:
            return ", ".join(str(v) for v in self.value)
        return str(self.value)


def _classify(path: str, value: object) -> tuple[str, tuple[str, ...] | None]:
    if path in _CHOICE_FIELDS:
        return "choice", _CHOICE_FIELDS[path]
    if path in _TEXT_FIELDS:
        return "text", None
    if path in _FLOAT_PAIR_FIELDS:
        return "float_pair", None
    if path in _FLOAT_FIELDS:
        return "float", None
    if isinstance(value, bool):  # before int -- bool IS an int in Python
        return "bool", None
    if isinstance(value, int):
        return "int", None
    if isinstance(value, float):
        return "float", None
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            return ("float_pair" if any(isinstance(v, float) for v in value) else "int_pair"), None
        return "str_list", None
    return "str", None


def editable_fields(app_config: AppConfig) -> list[ConfigField]:
    """Flatten the AppConfig dataclass tree into an ordered list of leaves.

    Generic on purpose: a field added to any of the config dataclasses later
    becomes editable in both front-ends without touching either of them.
    """
    result: list[ConfigField] = []

    def walk(obj, prefix: str) -> None:
        for field_info in fields(obj):
            path = f"{prefix}{field_info.name}"
            if path in _HIDDEN_FIELDS:
                continue
            value = getattr(obj, field_info.name)
            if is_dataclass(value):
                walk(value, f"{path}.")
                continue
            kind, choices = _classify(path, value)
            result.append(
                ConfigField(
                    path=path,
                    value=value,
                    kind=kind,
                    choices=choices,
                    secret=path in _SECRET_FIELDS,
                    help=_HELP.get(path, ""),
                )
            )

    walk(app_config, "")
    return result


def find_field(app_config: AppConfig, path: str) -> ConfigField:
    for field_ in editable_fields(app_config):
        if field_.path == path:
            return field_
    raise ConfigEditError(f"No such configuration field: {path}")


_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}


def coerce(field_: ConfigField, raw):
    """Turn operator input into the value the dataclass field expects."""
    if field_.kind == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ConfigEditError(f"Expected true/false, got {raw!r}")

    if field_.kind == "int":
        try:
            return int(str(raw).strip())
        except ValueError:
            raise ConfigEditError(f"Expected a whole number, got {raw!r}") from None

    if field_.kind == "float":
        try:
            return float(str(raw).strip())
        except ValueError:
            raise ConfigEditError(f"Expected a number, got {raw!r}") from None

    if field_.kind == "choice":
        text = str(raw).strip()
        if text not in (field_.choices or ()):
            raise ConfigEditError(
                f"Expected one of {', '.join(field_.choices or ())}, got {raw!r}"
            )
        return text

    if field_.kind in {"int_pair", "float_pair"}:
        parts = [p for p in str(raw).replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ConfigEditError(
                f"Expected two values (low high), got {len(parts)}: {raw!r}"
            )
        cast = int if field_.kind == "int_pair" else float
        try:
            low, high = cast(parts[0]), cast(parts[1])
        except ValueError:
            raise ConfigEditError(f"Expected two numbers, got {raw!r}") from None
        if low >= high:
            raise ConfigEditError(
                f"Bounds must be ascending: {low} is not below {high}"
            )
        return (low, high)

    if field_.kind == "str_list":
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if not parts:
            raise ConfigEditError("Expected at least one comma-separated value")
        return tuple(parts)

    return str(raw)


def _set_path(app_config: AppConfig, path: str, value) -> None:
    target = app_config
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def validate(app_config: AppConfig) -> list[str]:
    """Cross-field checks, as operator-readable problem descriptions.

    Run against a candidate copy before an edit is committed, so a rejected edit
    leaves the live configuration untouched. These are the constraints that
    would otherwise fail deep inside a run: a chunk_overlap that the splitter
    rejects, bounds the optimizer clamps against, a prompt template that
    .format() raises on halfway through the benchmark.
    """
    problems: list[str] = []
    retrieval = app_config.retrieval
    optimizer = app_config.optimizer
    qg = app_config.question_generation
    mistral = app_config.mistral

    if retrieval.chunk_size <= 0:
        problems.append("retrieval.chunk_size must be positive")
    if retrieval.chunk_overlap < 0:
        problems.append("retrieval.chunk_overlap cannot be negative")
    if retrieval.chunk_overlap >= retrieval.chunk_size:
        problems.append(
            f"retrieval.chunk_overlap ({retrieval.chunk_overlap}) must be below "
            f"retrieval.chunk_size ({retrieval.chunk_size})"
        )
    if retrieval.top_k <= 0:
        problems.append("retrieval.top_k must be positive")
    if not 0.0 <= retrieval.similarity_threshold <= 1.0:
        problems.append("retrieval.similarity_threshold must be between 0 and 1")
    if not validate_prompt_template(retrieval.prompt_template):
        problems.append(
            "retrieval.prompt_template must contain {context} exactly once and "
            "{question} exactly once -- str.format() breaks otherwise"
        )

    if optimizer.max_iterations <= 0:
        problems.append("optimizer.max_iterations must be positive")
    for name in ("top_k_bounds", "chunk_size_bounds", "similarity_threshold_bounds"):
        bounds = getattr(optimizer, name)
        if len(bounds) != 2:
            problems.append(f"optimizer.{name} must have exactly two values")
        elif bounds[0] >= bounds[1]:
            problems.append(
                f"optimizer.{name} must be ascending, got {bounds[0]} >= {bounds[1]}"
            )

    if qg.batch_size_chars <= 0:
        problems.append("question_generation.batch_size_chars must be positive")
    if qg.questions_per_batch <= 0:
        problems.append("question_generation.questions_per_batch must be positive")
    if not 0.0 < qg.dedup_similarity_threshold <= 1.0:
        problems.append(
            "question_generation.dedup_similarity_threshold must be in (0, 1]"
        )

    if mistral.requests_per_minute <= 0:
        problems.append("mistral.requests_per_minute must be positive")
    if mistral.max_retries < 0:
        problems.append("mistral.max_retries cannot be negative")

    return problems


def apply_fields(app_config: AppConfig, updates: dict[str, object]) -> dict[str, object]:
    """Coerce, validate and commit several edits together. Returns stored values.

    Validated as a SET, on a deep copy, and committed only if the whole set is
    valid. Applying edits one at a time would reject legitimate batches: raising
    chunk_overlap from 50 to 400 while chunk_size is still 300 fails the
    overlap < chunk_size check even when chunk_size is being raised to 1000 in the
    same submission. Validating on a copy also means a rejected batch leaves the
    live configuration untouched, so a later save cannot persist half of it.
    """
    coerced: dict[str, object] = {}
    for path, raw in updates.items():
        field_ = find_field(app_config, path)
        coerced[path] = coerce(field_, raw)

    candidate = copy.deepcopy(app_config)
    for path, value in coerced.items():
        _set_path(candidate, path, value)
    problems = validate(candidate)
    if problems:
        raise ConfigEditError("; ".join(problems))

    for path, value in coerced.items():
        _set_path(app_config, path, value)
        logger.info("Configuration field %s set to %r", path, value)
    return coerced


def apply_field(app_config: AppConfig, path: str, raw) -> object:
    """Coerce, validate and commit one edit. Returns the stored value."""
    return apply_fields(app_config, {path: raw})[path]


def stage_document(uploaded_name: str, data: bytes) -> Path:
    """Write an uploaded file into data/ and return its path.

    The GUI receives bytes, but every document operation works on a path, and
    the manifest records that path so later chunk-parameter changes can re-read
    the file. Uploads therefore have to land somewhere durable, not a temp dir.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / Path(uploaded_name).name
    if target.exists() and target.read_bytes() == data:
        logger.info("%s is already staged at %s", uploaded_name, target)
        return target
    stem, suffix = target.stem, target.suffix
    counter = 1
    while target.exists() and target.read_bytes() != data:
        target = DATA_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    target.write_bytes(data)
    logger.info("Staged upload %s at %s (%d bytes)", uploaded_name, target, len(data))
    return target


class ListLogHandler(logging.Handler):
    """Collect log records into a list so a GUI can render a running log.

    The optimize and compare operations are long, and their only progress signal
    is the logging they already emit. Rather than adding a parallel callback
    mechanism through five modules, both front-ends attach one of these.
    """

    def __init__(self, limit: int = 2000):
        super().__init__()
        self.records: list[str] = []
        self.limit = limit
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:  # noqa: BLE001 -- logging must never break the run
            return
        if len(self.records) > self.limit:
            del self.records[: len(self.records) - self.limit]

    def text(self) -> str:
        return "\n".join(self.records)
