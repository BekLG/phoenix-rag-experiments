"""
chunking.py
===========
Splits loaded documents into chunks using LangChain's
RecursiveCharacterTextSplitter. Chunk size / overlap are configurable
so the optimizer can mutate them between iterations.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from phoenix_rag.core.chunk_identity import assign_chunk_ids, assign_document_ids

logger = logging.getLogger("phoenix_rag.chunking")


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split documents into chunks for embedding/retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    # Stamp the full source/pages before splitting, so document identity does
    # not depend on chunk parameters. Keep the caller's metadata untouched.
    identified = [
        document.model_copy(update={"metadata": dict(document.metadata)})
        for document in documents
    ]
    assign_document_ids(identified)
    chunks = splitter.split_documents(identified)
    assign_chunk_ids(chunks, overwrite=True)
    logger.info(
        "Split %d document(s) into %d chunks (chunk_size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
