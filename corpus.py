"""
corpus.py
=========
A growable, multi-document corpus for Phoenix RAG.

WHY THIS EXISTS
---------------
Everything else in this project assumes exactly one source document, and that
assumption is baked into four independent caches:

  * vector_store.get_or_build_vector_store keys its FAISS directory on
    sha256(source_document bytes) + embedding_model + chunk_size + chunk_overlap.
    A second document hashes differently, so it lands in a DIFFERENT directory --
    there is no way to search both documents at once.
  * AppConfig.summary_path / profile_path / benchmark_path are each one file
    describing one document.
  * llm_optimizer.propose_next_config_llm receives one summary string and one
    DocumentProfile, so the optimizer can only reason about one document.

Consequently, pointing the system at a new document meant discarding the old one.
This module makes the opposite possible: ADD a document to the index that already
exists, and grow the summary, profile and benchmark alongside it, so the
optimizer tunes for a corpus rather than for whichever document was loaded last.

WHAT "INCREMENTAL" DOES AND DOES NOT MEAN
----------------------------------------
A FAISS index is specific to (embedding_model, chunk_size, chunk_overlap). Adding
a document at the CURRENT chunk parameters re-uses every vector already in the
index and embeds only the new document's chunks -- that is the cheap path, and it
is what "add to the existing index" means here.

When the optimizer later CHANGES chunk_size, the corpus has to be re-chunked and
re-embedded from scratch. That is not a limitation of this module; a chunk of a
different size is a different vector. sync_index() handles both cases: it keeps
one index directory per parameter variant, tracks which documents each variant
contains, and only ever embeds what a variant is missing.

LAYOUT
------
    data/corpus/
        manifest.json         documents, per-variant index membership, and the
                              per-document summaries and profiles
        benchmark.json        the corpus benchmark -- grows as documents are added
        corpus_summary.txt    rendered multi-document summary (what the optimizer
                              is shown), for human inspection
        corpus_profile.json   the aggregated profile, for human inspection
        indexes/<key>/        one FAISS index per (model, chunk_size, overlap)

The manifest is the source of truth. The two rendered files are derived and safe
to delete. The single-document artifacts under generated_questions/ are never
written to by this module, so single-document runs stay reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from chunking import split_documents
from config import AppConfig
from document_loader import load_document
from document_profile import (
    DocumentProfile,
    aggregate_profiles,
    compute_profile,
    load_profile,
)
from document_summarizer import generate_summary
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from question_generator import (
    BenchmarkQuestion,
    _dedupe,
    generate_benchmark,
    load_benchmark,
    save_benchmark,
)
from storage import _atomic_write_json
from vector_store import (
    add_chunks_to_store,
    build_vector_store,
    load_vector_store,
    save_vector_store,
)

logger = logging.getLogger("phoenix_rag.corpus")

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
INDEXES_DIRNAME = "indexes"
# Same convention as vector_store.get_or_build_vector_store: an index directory
# is only trustworthy once this marker exists, so a crash mid-save leaves a
# directory that is rebuilt rather than loaded as if it were complete.
COMPLETE_MARKER = ".complete"


# --------------------------------------------------------------------------
# Manifest data model
# --------------------------------------------------------------------------

@dataclass
class CorpusDocument:
    """One member document, plus the artifacts derived from it.

    `doc_id` is a digest of the file's CONTENT, not its path, so re-adding the
    same file under a different name is recognised as a duplicate, and editing a
    file in place is detectable (see verify_documents).
    """

    doc_id: str
    label: str
    path: str
    pages: int | None
    characters: int
    added_at: str
    profile: dict
    summary: str
    question_count: int = 0

    @property
    def document_profile(self) -> DocumentProfile:
        return DocumentProfile(**self.profile)

    @property
    def filename(self) -> str:
        return Path(self.path).name

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CorpusDocument":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CorpusIndex:
    """One FAISS index variant and the documents already embedded into it."""

    index_key: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    members: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CorpusIndex":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Corpus:
    root: Path
    documents: list[CorpusDocument] = field(default_factory=list)
    indexes: dict[str, CorpusIndex] = field(default_factory=dict)
    last_optimized_members: list[str] = field(default_factory=list)

    # -- derived paths -------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def benchmark_path(self) -> Path:
        return self.root / "benchmark.json"

    @property
    def summary_text_path(self) -> Path:
        return self.root / "corpus_summary.txt"

    @property
    def profile_path(self) -> Path:
        return self.root / "corpus_profile.json"

    def index_path(self, index_key: str) -> Path:
        return self.root / INDEXES_DIRNAME / index_key

    # -- lookups -------------------------------------------------------

    @property
    def document_ids(self) -> list[str]:
        return [document.doc_id for document in self.documents]

    @property
    def is_empty(self) -> bool:
        return not self.documents

    def find(self, doc_id: str) -> CorpusDocument | None:
        return next((d for d in self.documents if d.doc_id == doc_id), None)

    def find_by_label(self, label: str) -> CorpusDocument | None:
        return next((d for d in self.documents if d.label == label), None)

    def total_characters(self) -> int:
        return sum(document.characters for document in self.documents)

    def total_questions(self) -> int:
        """Questions actually on disk, not the sum of per-document counts.

        The two can differ: dedupe drops near-identical questions across
        documents, so the file is authoritative and the per-document counts
        record only what each document contributed.
        """
        if not self.benchmark_path.exists():
            return 0
        try:
            return len(load_benchmark(self.benchmark_path))
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.warning("Corpus benchmark at %s is unreadable", self.benchmark_path)
            return 0


# --------------------------------------------------------------------------
# Identity helpers
# --------------------------------------------------------------------------

def document_digest(path: str | Path) -> str:
    """Content digest of a source file -- the document's identity in a corpus."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def index_key(embedding_model: str, chunk_size: int, chunk_overlap: int) -> str:
    """Stable key for one index variant.

    Note what is NOT in here, unlike get_or_build_vector_store's key: the source
    document's digest. Membership is tracked in the manifest instead, which is
    precisely what lets a variant gain a document without changing identity.
    """
    payload = json.dumps(
        {
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


# --------------------------------------------------------------------------
# Load / save
# --------------------------------------------------------------------------

def load(root: str | Path) -> Corpus:
    """Load the manifest at `root`, or return an empty corpus if there is none."""
    root = Path(root)
    manifest_file = root / MANIFEST_NAME

    if not manifest_file.exists() or manifest_file.stat().st_size == 0:
        logger.info("No corpus manifest at %s; starting an empty corpus", manifest_file)
        return Corpus(root=root)

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    version = data.get("version")
    if version != MANIFEST_VERSION:
        logger.warning(
            "Corpus manifest version is %s, this build writes %d; reading it anyway",
            version, MANIFEST_VERSION,
        )

    corpus = Corpus(
        root=root,
        documents=[CorpusDocument.from_dict(d) for d in data.get("documents", [])],
        indexes={
            key: CorpusIndex.from_dict(value)
            for key, value in data.get("indexes", {}).items()
        },
        last_optimized_members=list(data.get("last_optimized_members", [])),
    )
    logger.info(
        "Loaded corpus from %s: %d document(s), %d index variant(s)",
        manifest_file, len(corpus.documents), len(corpus.indexes),
    )
    return corpus


def save(corpus: Corpus) -> None:
    """Persist the manifest atomically (same helper storage.py uses)."""
    corpus.root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        corpus.manifest_path,
        {
            "version": MANIFEST_VERSION,
            "documents": [document.to_dict() for document in corpus.documents],
            "indexes": {key: value.to_dict() for key, value in corpus.indexes.items()},
            "last_optimized_members": corpus.last_optimized_members,
        },
    )
    logger.debug("Saved corpus manifest to %s", corpus.manifest_path)


# --------------------------------------------------------------------------
# Registering documents
# --------------------------------------------------------------------------

def _unique_label(corpus: Corpus, label: str) -> str:
    """Make `label` unique within the corpus by suffixing, never by rejecting.

    Labels are display and namespacing sugar; the doc_id is identity. Refusing
    an add over a name collision would be a worse trade than quietly renaming
    and saying so.
    """
    candidate = (label or "document").strip().replace(" ", "_")
    if corpus.find_by_label(candidate) is None:
        return candidate
    suffix = 2
    while corpus.find_by_label(f"{candidate}_{suffix}") is not None:
        suffix += 1
    renamed = f"{candidate}_{suffix}"
    logger.warning("Label %r is already used in this corpus; using %r", candidate, renamed)
    return renamed


def _page_count(path: Path, loaded: list[Document]) -> int | None:
    """Pages, but only for formats where a page is a real thing.

    Mirrors the guard in experiment_runner.run_experiment: for a .txt or .md the
    loader returns one Document for the whole file, and calling that "1 page"
    would feed document_profile a median_chars_per_page equal to the entire
    document, which then drives the sizing regime off a fiction.
    """
    return len(loaded) if path.suffix.lower() == ".pdf" else None


def _extend_benchmark(
    corpus: Corpus,
    new_questions: list[BenchmarkQuestion],
) -> int:
    """Append `new_questions` to the corpus benchmark, deduped. Returns the delta.

    Deliberately additive: the questions already on disk were generated from
    documents that are still in the corpus and still need to be answerable, so
    they stay. Dedupe runs over the merged set because two documents on related
    subjects do produce overlapping questions.
    """
    existing: list[BenchmarkQuestion] = []
    if corpus.benchmark_path.exists() and corpus.benchmark_path.stat().st_size > 0:
        try:
            existing = load_benchmark(corpus.benchmark_path)
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.exception(
                "Corpus benchmark at %s is unreadable; treating it as empty rather "
                "than discarding the new questions",
                corpus.benchmark_path,
            )

    merged = _dedupe([*existing, *new_questions])
    delta = len(merged) - len(existing)
    save_benchmark(merged, corpus.benchmark_path)
    logger.info(
        "Corpus benchmark: %d question(s) -> %d (%+d from this document)",
        len(existing), len(merged), delta,
    )
    return max(0, delta)


def register_document(
    corpus: Corpus,
    app_config: AppConfig,
    path: str | Path,
    label: str,
    generate_questions: bool = True,
) -> CorpusDocument | None:
    """Add one document to the manifest, generating its derived artifacts.

    Computes the profile deterministically, generates the summary, and (unless
    `generate_questions` is False) generates questions from the FULL new document
    and appends them to the corpus benchmark.

    Extending the benchmark is the default because the alternative is quietly
    broken: with questions only from the older documents, the new document's
    chunks are pure noise in every retrieval, and re-optimizing would tune the
    configuration to avoid retrieving it. The cost is that scores from before an
    add are not comparable to scores after it -- the benchmark itself changed --
    which is why per-document counts are recorded and the saved best config is
    marked stale.

    Returns None if this exact file content is already a member (idempotent), so
    a repeated add is a no-op rather than a duplicate or an error.

    Does NOT touch any FAISS index; call sync_index() for that.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    doc_id = document_digest(path)
    already = corpus.find(doc_id)
    if already is not None:
        logger.info(
            "Document %s is already in this corpus as %r (identical content); "
            "nothing to register",
            path.name, already.label,
        )
        return None

    loaded = load_document(path)
    full_text = "\n\n".join(document.page_content for document in loaded)
    pages = _page_count(path, loaded)

    profile = compute_profile(full_text, pages=pages)
    logger.info("Profile for %s: %s", path.name, profile)

    summary = generate_summary(full_text, app_config.mistral)

    document = CorpusDocument(
        doc_id=doc_id,
        label=_unique_label(corpus, label),
        path=str(path),
        pages=pages,
        characters=profile.characters,
        added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        profile=asdict(profile),
        summary=summary,
        question_count=0,
    )
    corpus.documents.append(document)

    if generate_questions:
        questions = generate_benchmark(
            full_text=full_text,
            mistral_settings=app_config.mistral,
            qg_config=app_config.question_generation,
        )
        document.question_count = _extend_benchmark(corpus, questions)
    else:
        logger.warning(
            "Question generation skipped for %r. Until the benchmark covers this "
            "document, optimization is scored only on the others -- this "
            "document's chunks act as retrieval noise.",
            document.label,
        )

    save(corpus)
    write_rendered_artifacts(corpus)
    logger.info(
        "Registered %r (doc_id=%s, %d chars, %s pages, +%d questions)",
        document.label, doc_id, document.characters, document.pages,
        document.question_count,
    )
    return document


def bootstrap_from_single_document(
    corpus: Corpus, app_config: AppConfig
) -> CorpusDocument | None:
    """Seed an empty corpus with app_config.source_document, reusing its caches.

    This is the single-document -> corpus conversion, and it is written to spend
    nothing: the profile, summary and benchmark generated by previous runs are
    ADOPTED from generated_questions/ rather than regenerated. Only a missing
    artifact costs an API call.

    No-op on a corpus that already has documents.
    """
    if not corpus.is_empty:
        return None

    source = Path(app_config.source_document)
    if not source.exists():
        logger.warning(
            "Cannot bootstrap the corpus: configured source_document %s does not "
            "exist. The corpus will start from whatever is added next.",
            source,
        )
        return None

    logger.info("Bootstrapping corpus from the configured source document %s", source.name)

    loaded = load_document(source)
    full_text = "\n\n".join(document.page_content for document in loaded)
    pages = _page_count(source, loaded)

    cached_profile = Path(app_config.profile_path)
    if cached_profile.exists() and cached_profile.stat().st_size > 0:
        logger.info("Adopting the cached document profile at %s", cached_profile)
        profile = load_profile(cached_profile)
    else:
        profile = compute_profile(full_text, pages=pages)

    cached_summary = Path(app_config.summary_path)
    if cached_summary.exists() and cached_summary.stat().st_size > 0:
        logger.info("Adopting the cached document summary at %s", cached_summary)
        summary = cached_summary.read_text(encoding="utf-8")
    else:
        summary = generate_summary(full_text, app_config.mistral)

    document = CorpusDocument(
        doc_id=document_digest(source),
        label=_unique_label(corpus, source.stem),
        path=str(source.resolve()),
        pages=pages,
        characters=profile.characters,
        added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        profile=asdict(profile),
        summary=summary,
        question_count=0,
    )
    corpus.documents.append(document)

    cached_benchmark = Path(app_config.benchmark_path)
    corpus_has_benchmark = (
        corpus.benchmark_path.exists() and corpus.benchmark_path.stat().st_size > 0
    )
    if not corpus_has_benchmark and cached_benchmark.exists():
        logger.info("Adopting the cached benchmark at %s", cached_benchmark)
        adopted = load_benchmark(cached_benchmark)
        save_benchmark(adopted, corpus.benchmark_path)
        document.question_count = len(adopted)
    elif not corpus_has_benchmark:
        questions = generate_benchmark(
            full_text=full_text,
            mistral_settings=app_config.mistral,
            qg_config=app_config.question_generation,
        )
        document.question_count = _extend_benchmark(corpus, questions)

    save(corpus)
    write_rendered_artifacts(corpus)
    logger.info("Corpus bootstrapped with %r", document.label)
    return document


def remove_document(corpus: Corpus, doc_id: str) -> CorpusDocument | None:
    """Drop a document from the manifest and from every index's membership.

    The vectors themselves are NOT deleted -- FAISS has no cheap removal for the
    docstore-backed index used here -- so each affected variant's directory is
    invalidated and will be rebuilt on the next sync_index(). Stated plainly
    because it means removal costs a full re-embed, whereas adding does not.
    """
    document = corpus.find(doc_id)
    if document is None:
        return None

    corpus.documents = [d for d in corpus.documents if d.doc_id != doc_id]
    for key, entry in list(corpus.indexes.items()):
        if doc_id in entry.members:
            marker = corpus.index_path(key) / COMPLETE_MARKER
            marker.unlink(missing_ok=True)
            del corpus.indexes[key]
            logger.info(
                "Invalidated index variant %s (chunk_size=%d, overlap=%d); it will "
                "be rebuilt without %r on the next sync",
                key, entry.chunk_size, entry.chunk_overlap, document.label,
            )

    save(corpus)
    write_rendered_artifacts(corpus)
    return document


# --------------------------------------------------------------------------
# Index maintenance -- the incremental add
# --------------------------------------------------------------------------

@dataclass
class SyncReport:
    """What sync_index actually did, for logging and for the UIs to report."""

    index_key: str
    index_path: Path
    action: str  # "built" | "extended" | "loaded"
    added_labels: list[str]
    added_chunks: int
    vectors_before: int
    vectors_after: int

    def describe(self) -> str:
        if self.action == "built":
            return (
                f"built a new index ({self.vectors_after} vectors) over "
                f"{len(self.added_labels)} document(s): {', '.join(self.added_labels)}"
            )
        if self.action == "extended":
            return (
                f"extended the existing index in place: {self.vectors_before} -> "
                f"{self.vectors_after} vectors, embedding only {self.added_chunks} new "
                f"chunk(s) from {', '.join(self.added_labels)}"
            )
        return f"reused the existing index unchanged ({self.vectors_after} vectors)"


def vector_count(store) -> int:
    """FAISS vector count, defensively -- it is only ever used for reporting."""
    return int(getattr(getattr(store, "index", None), "ntotal", 0) or 0)


# Kept under the original private name for this module's own call sites.
_vector_count = vector_count


def _chunks_for_document(
    document: CorpusDocument, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Load, split, and stamp corpus provenance onto one document's chunks.

    The stamped doc_id/doc_label are what let "ask the RAG" tell the operator
    WHICH document a retrieved context came from -- with several documents in one
    index, an answer with no attribution is much harder to sanity-check.
    """
    path = Path(document.path)
    if not path.exists():
        raise FileNotFoundError(
            f"Document {document.label!r} is registered in the corpus but its file "
            f"is missing: {path}. Restore it, or remove the document from the corpus."
        )

    chunks = split_documents(
        load_document(path), chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    for chunk in chunks:
        chunk.metadata["doc_id"] = document.doc_id
        chunk.metadata["doc_label"] = document.label
        chunk.metadata.setdefault("source", str(path))
    return chunks


def sync_index(
    corpus: Corpus,
    embeddings: Embeddings,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[object, SyncReport]:
    """Return a FAISS index over the whole corpus at these chunk parameters.

    Three paths, and which one runs is the whole point of this module:

      * The variant exists and already contains every document -> LOAD it.
      * The variant exists but is missing documents -> LOAD it and embed ONLY the
        missing documents' chunks into it. Existing vectors are untouched. This
        is what "add a document to the existing index" means.
      * The variant does not exist -> BUILD it over every member. Reached on a
        first run, and whenever the optimizer changes chunk_size/chunk_overlap,
        since chunks of a different size are different vectors and cannot be
        reused.

    Membership is recorded per variant, so a document added while chunk_size=800
    was current is picked up automatically the next time a chunk_size=500 variant
    is synced.
    """
    if corpus.is_empty:
        raise ValueError(
            "Cannot build an index for an empty corpus -- add a document first."
        )

    key = index_key(embedding_model, chunk_size, chunk_overlap)
    path = corpus.index_path(key)
    marker = path / COMPLETE_MARKER
    entry = corpus.indexes.get(key)

    store = None
    if marker.exists() and entry is not None:
        try:
            store = load_vector_store(path, embeddings)
        except Exception:
            logger.exception("Index variant at %s is unreadable; rebuilding it", path)
            marker.unlink(missing_ok=True)
            store = None
            entry = None
            corpus.indexes.pop(key, None)

    # ---- build from scratch --------------------------------------------
    if store is None:
        entry = CorpusIndex(
            index_key=key,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            members=[],
        )
        logger.info(
            "Building a new index variant (chunk_size=%d, overlap=%d) over %d "
            "document(s)",
            chunk_size, chunk_overlap, len(corpus.documents),
        )
        all_chunks: list[Document] = []
        for document in corpus.documents:
            all_chunks.extend(_chunks_for_document(document, chunk_size, chunk_overlap))

        store = build_vector_store(all_chunks, embeddings)
        save_vector_store(store, path)
        marker.write_text("ok\n", encoding="ascii")

        entry.members = corpus.document_ids
        corpus.indexes[key] = entry
        save(corpus)

        report = SyncReport(
            index_key=key,
            index_path=path,
            action="built",
            added_labels=[d.label for d in corpus.documents],
            added_chunks=len(all_chunks),
            vectors_before=0,
            vectors_after=_vector_count(store),
        )
        logger.info("sync_index: %s", report.describe())
        return store, report

    # ---- extend in place, or reuse untouched ---------------------------
    missing = [d for d in corpus.documents if d.doc_id not in entry.members]
    vectors_before = _vector_count(store)

    if not missing:
        report = SyncReport(
            index_key=key,
            index_path=path,
            action="loaded",
            added_labels=[],
            added_chunks=0,
            vectors_before=vectors_before,
            vectors_after=vectors_before,
        )
        logger.info("sync_index: %s", report.describe())
        return store, report

    logger.info(
        "Index variant %s is missing %d document(s): %s. Embedding only those.",
        key, len(missing), ", ".join(d.label for d in missing),
    )
    new_chunks: list[Document] = []
    for document in missing:
        new_chunks.extend(_chunks_for_document(document, chunk_size, chunk_overlap))

    add_chunks_to_store(store, new_chunks)
    save_vector_store(store, path)
    marker.write_text("ok\n", encoding="ascii")

    entry.members = [*entry.members, *(d.doc_id for d in missing)]
    corpus.indexes[key] = entry
    save(corpus)

    report = SyncReport(
        index_key=key,
        index_path=path,
        action="extended",
        added_labels=[d.label for d in missing],
        added_chunks=len(new_chunks),
        vectors_before=vectors_before,
        vectors_after=_vector_count(store),
    )
    logger.info("sync_index: %s", report.describe())
    return store, report


# --------------------------------------------------------------------------
# Aggregated artifacts fed to the optimizer
# --------------------------------------------------------------------------

def profile(corpus: Corpus) -> DocumentProfile:
    """The corpus as one DocumentProfile, for seed_config and the optimizer."""
    return aggregate_profiles([d.document_profile for d in corpus.documents])


def _composition_line(corpus: Corpus) -> str:
    total = corpus.total_characters() or 1
    shares: dict[str, int] = {}
    for document in corpus.documents:
        doc_type = document.document_profile.doc_type
        shares[doc_type] = shares.get(doc_type, 0) + document.characters
    ordered = sorted(shares.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(
        f"{doc_type} ({chars} chars, {chars / total:.0%})" for doc_type, chars in ordered
    )


def summary_text(corpus: Corpus) -> str:
    """Render every member's summary into the one string the optimizer accepts.

    llm_optimizer.propose_next_config_llm takes `document_summary: str`. Rendering
    the whole corpus into that string is what lets the optimizer reason about all
    the documents' concepts at once -- no optimizer signature change, and the
    prompt template it writes can account for every subject in the corpus rather
    than only the one that happened to be loaded.

    The leading paragraph is not decoration: without it the model reads a
    multi-subject summary as one incoherent document and writes a prompt template
    scoped to whichever subject came first.
    """
    if corpus.is_empty:
        return "(the corpus is empty)"

    count = len(corpus.documents)
    if count == 1:
        # A one-document corpus should look exactly like the single-document case,
        # so behaviour does not change the moment a corpus is created.
        return corpus.documents[0].summary

    aggregate = profile(corpus)
    header = (
        f"THIS IS A {count}-DOCUMENT CORPUS. A single FAISS index spans all "
        f"{count} documents, and any benchmark question may be answerable from "
        "any one of them. The retrieval configuration and the prompt template you "
        "propose must work across EVERY document described below -- do not scope "
        "the prompt template to one subject, or questions about the others will "
        "be refused.\n\n"
        f"CORPUS TOTALS: {aggregate.characters} characters, "
        f"pages={aggregate.pages}, sections={aggregate.sections}, "
        f"dominant doc_type={aggregate.doc_type}.\n"
        f"COMPOSITION BY doc_type: {_composition_line(corpus)}."
    )

    blocks = [header]
    for position, document in enumerate(corpus.documents, start=1):
        member = document.document_profile
        blocks.append(
            f"=== DOCUMENT {position} of {count}: {document.label} "
            f"({document.filename}) ===\n"
            f"profile: pages={member.pages}, characters={member.characters}, "
            f"sections={member.sections}, "
            f"median_chars_per_page={member.median_chars_per_page}, "
            f"doc_type={member.doc_type}, table_heavy={member.table_heavy}, "
            f"list_heavy={member.list_heavy}, "
            f"benchmark_questions_contributed={document.question_count}\n"
            f"{document.summary}"
        )
    return "\n\n".join(blocks)


def write_rendered_artifacts(corpus: Corpus) -> None:
    """Mirror the derived summary text and aggregate profile to disk.

    Purely for inspection -- nothing reads these back. Note what is NOT written:
    app_config.summary_path and profile_path, which describe a single document
    and must stay untouched so single-document runs remain reproducible.
    """
    if corpus.is_empty:
        return
    corpus.root.mkdir(parents=True, exist_ok=True)
    corpus.summary_text_path.write_text(summary_text(corpus), encoding="utf-8")
    _atomic_write_json(corpus.profile_path, asdict(profile(corpus)))


def corpus_benchmark(
    corpus: Corpus, app_config: AppConfig
) -> list[BenchmarkQuestion]:
    """The fixed benchmark every configuration in a corpus run is scored against.

    Normally just loads what register_document accumulated. The fallback covers a
    corpus assembled with question generation turned off: rather than optimize
    against nothing, questions are generated once over the concatenated corpus
    text and cached, keeping the "benchmark is generated once and then fixed"
    guarantee from question_generator.py intact.
    """
    if corpus.benchmark_path.exists() and corpus.benchmark_path.stat().st_size > 0:
        questions = load_benchmark(corpus.benchmark_path)
        if questions:
            logger.info(
                "Corpus benchmark: %d question(s) across %d document(s)",
                len(questions), len(corpus.documents),
            )
            return questions

    logger.warning(
        "The corpus has no benchmark yet; generating one over all %d document(s). "
        "This happens when documents were added with question generation disabled.",
        len(corpus.documents),
    )
    joined = "\n\n".join(
        "\n\n".join(part.page_content for part in load_document(document.path))
        for document in corpus.documents
    )
    questions = generate_benchmark(
        full_text=joined,
        mistral_settings=app_config.mistral,
        qg_config=app_config.question_generation,
    )
    save_benchmark(questions, corpus.benchmark_path)
    return questions


# --------------------------------------------------------------------------
# Staleness and integrity
# --------------------------------------------------------------------------

def optimization_status(corpus: Corpus) -> str:
    """Whether the saved best configuration still describes this corpus.

    Returns "never", "stale", or "current". Membership changes invalidate a saved
    best config in two ways at once: the index it was measured against has grown,
    and the benchmark it was scored on has grown too. Reporting that is cheaper
    than letting an operator trust a number that no longer applies.
    """
    if not corpus.last_optimized_members:
        return "never"
    return (
        "current"
        if set(corpus.last_optimized_members) == set(corpus.document_ids)
        else "stale"
    )


def mark_optimized(corpus: Corpus) -> None:
    """Record that the current membership is what the saved best config reflects."""
    corpus.last_optimized_members = corpus.document_ids
    save(corpus)


def verify_documents(corpus: Corpus) -> list[str]:
    """Report members whose file is gone or whose content changed on disk.

    An edited-in-place file is the quiet failure this catches: its doc_id no
    longer matches its content, so the index still holds vectors for the OLD
    text while the summary and profile describe it as current. Detected and
    reported rather than silently re-embedded, because the fix (re-add, or
    rebuild) is the operator's call.
    """
    problems: list[str] = []
    for document in corpus.documents:
        path = Path(document.path)
        if not path.exists():
            problems.append(f"{document.label}: file is missing ({path})")
            continue
        if document_digest(path) != document.doc_id:
            problems.append(
                f"{document.label}: file has changed on disk since it was added "
                f"({path}) -- the index still holds the old text. Remove and re-add it."
            )
    return problems
