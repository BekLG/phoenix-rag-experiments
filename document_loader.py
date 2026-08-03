"""
document_loader.py
===================
Modular document ingestion. Supports PDF and plain text out of the box;
new loaders can be added by registering a new entry in `LOADER_REGISTRY`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger("phoenix_rag.document_loader")


LOADER_REGISTRY: dict[str, Callable[[str], list[Document]]] = {
    ".pdf": lambda path: PyPDFLoader(path).load(),
    ".txt": lambda path: TextLoader(path, encoding="utf-8").load(),
    ".md": lambda path: UnstructuredMarkdownLoader(path).load(),
}


def load_document(path: str | Path) -> list[Document]:
    """Load a single document into a list of LangChain Document objects.

    Raises ValueError for unsupported extensions so failures are loud and
    early rather than silently returning nothing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source document not found: {path}")

    suffix = path.suffix.lower()
    loader_fn = LOADER_REGISTRY.get(suffix)
    if loader_fn is None:
        raise ValueError(
            f"Unsupported document type '{suffix}'. "
            f"Supported types: {list(LOADER_REGISTRY)}"
        )

    logger.info("Loading document %s (%s)", path.name, suffix)
    docs = loader_fn(str(path))
    logger.info("Loaded %d page(s)/section(s) from %s", len(docs), path.name)
    return docs


def load_full_text(path: str | Path) -> str:
    """Convenience helper: load a document and return it as one joined string.

    Used by question generation, which needs the *complete* source document
    rather than page-level Document objects.
    """
    docs = load_document(path)
    return "\n\n".join(d.page_content for d in docs)
