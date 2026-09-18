"""
providers/base.py
=================
The backend-agnostic provider interfaces, and the bundle that holds one provider
per role for a run.

Two roles need two different surfaces, and the split here is not cosmetic -- it
is forced by the two consumers:

- **chat text.** Generation, question generation, summarization and the
  optimizer proposer all send messages and want the reply *text* back. They have
  no use for a LangChain object and should not have to know one exists.
- **a LangChain chat model.** The judge role feeds Ragas, whose
  ``evaluate(llm=..., embeddings=...)`` accepts LangChain-compatible objects and
  nothing else. So a provider that can serve as judge must be able to hand out a
  LangChain chat model in addition to answering :meth:`ChatProvider.chat`.

Embeddings need no wrapper at all: a LangChain :class:`Embeddings` is already the
one interface both FAISS (index build + retrieval) and Ragas
(``evaluate(embeddings=...)``) accept, so :data:`EmbeddingProvider` is that type
directly rather than something around it.

Concrete implementations live in one module per backend (``mistral.py`` so far);
:mod:`phoenix_rag.providers.registry` picks one per role from an
:class:`~phoenix_rag.config.AppConfig`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:  # pragma: no cover - type checkers / IDEs only
    from langchain_core.language_models.chat_models import BaseChatModel

# A role's embedding backend is just a LangChain Embeddings. There is nothing to
# add on top of it: FAISS and Ragas already consume that interface, so wrapping
# it would only be a layer to unwrap again at the call site. Named here so the
# rest of the code can say `EmbeddingProvider` where it means "the embedding
# role's backend" and read symmetrically with `ChatProvider`.
EmbeddingProvider = Embeddings


class ChatProvider(ABC):
    """One chat/generation role, bound to a single backend and model.

    A provider instance already knows *which* model it speaks for -- it is "the
    optimizer provider" or "the judge provider", not a generic client you pass a
    model name to on every call. That is why :meth:`chat` takes no ``model``
    argument: per-role models (and, later, per-role *backends*) are settled when
    the provider is built, not at each call site.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        response_format: dict | None = None,
    ) -> str:
        """Send one chat completion; return the reply text of the first choice."""

    @abstractmethod
    def as_langchain_chat_model(self, *, temperature: float = 0.0) -> "BaseChatModel":
        """Return a LangChain chat model backed by this provider's model.

        Only the judge role needs this -- Ragas's ``evaluate(llm=...)`` requires
        a LangChain object -- but any chat backend intended to serve as judge has
        to be able to produce one, so it is part of the interface rather than a
        Mistral-only extra.
        """


@dataclass
class Providers:
    """The backends for one run: one per role, built together.

    Constructed once by :func:`phoenix_rag.providers.registry.build_providers`,
    which is also where every role that talks to the same API is made to share a
    single rate limiter -- so the process respects the one quota the API
    enforces rather than the sum of each consumer's private budget.
    """

    embedding: EmbeddingProvider
    generation: ChatProvider
    optimizer: ChatProvider
    judge: ChatProvider
