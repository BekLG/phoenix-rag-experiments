"""Content identities shared by fresh chunks and legacy FAISS docstores."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document


def document_digest(path: str | Path) -> str:
    """The corpus manifest's existing, 16-hex SHA-256 file content digest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def assign_document_ids(documents: list[Document]) -> None:
    """Stamp IDs in place, preferring existing IDs to reading source files.

    A legacy corpus chunk already has ``doc_id``, so even a moved/missing source
    retains its manifest identity. With no saved digest and no readable file,
    use the available text as a deterministic fallback: all sections sharing a
    source, or the individual text for a source-less Document. Such a fallback
    cannot recover the original file digest from a lossy cached representation.
    """
    by_source: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        source = document.metadata.get("source")
        if source:
            by_source[str(source)].append(document.page_content)

    source_ids: dict[str, str] = {}
    for document in documents:
        metadata = document.metadata
        if metadata.get("document_id"):
            continue
        if metadata.get("doc_id"):
            metadata["document_id"] = metadata["doc_id"]
            continue
        source = metadata.get("source")
        if source:
            source = str(source)
            if source not in source_ids:
                try:
                    source_ids[source] = document_digest(source)
                except OSError:
                    payload = json.dumps(by_source[source], ensure_ascii=False)
                    source_ids[source] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
            metadata["document_id"] = source_ids[source]
        else:
            metadata["document_id"] = hashlib.sha256(
                document.page_content.encode("utf-8")
            ).hexdigest()[:16]


def assign_chunk_ids(chunks: list[Document], *, overwrite: bool = False) -> None:
    """Stamp IDs in document order without changing text or existing provenance.

    ``chunk_id`` is SHA-256 of JSON [document_id, exact text, occurrence], where
    occurrence is zero-based among identical texts in that document. Repeated
    passages get distinct IDs; unrelated documents and query ordering do not
    affect them. Identical chunks can share identity across chunking variants.

    Legacy indexes must pass ALL stored chunks in vector insertion order, never
    just the query hits: a hit's retrieval rank is not a stable occurrence index.
    Existing chunk IDs survive backfilling (including mixed old/new indexes).
    """
    assign_document_ids(chunks)
    occurrences: dict[tuple[str, str], int] = defaultdict(int)
    for chunk in chunks:
        document_id = chunk.metadata["document_id"]
        key = (document_id, chunk.page_content)
        occurrence = occurrences[key]
        occurrences[key] += 1
        if overwrite or not chunk.metadata.get("chunk_id"):
            payload = json.dumps(
                [document_id, chunk.page_content, occurrence], ensure_ascii=False
            )
            chunk.metadata["chunk_id"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
