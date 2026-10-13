"""Retrieval contracts and implementations.

The factory loads implementations lazily; importing contracts alone does not
import the vector store or optional retrieval dependencies.
"""

from phoenix_rag.retrieval.factory import build_retriever

__all__ = ["build_retriever"]
