"""
config/schema.py
================
The configuration objects. Pure data: no filesystem access, no environment
reads at import, no I/O. Loading and saving live in :mod:`phoenix_rag.config.loader`.

WHAT CHANGED FROM THE FLAT config.py
------------------------------------
1. The path constants and their import-time ``mkdir`` loop are gone. Paths now
   come from a :class:`phoenix_rag.workspace.Workspace`; see that module for why.
2. Model settings are described per *role* (:class:`ProviderConfig`), not as one
   flat Mistral block. There are four roles -- embedding, generation, optimizer,
   judge -- and the old config already named a separate model for each of them
   (``embedding_model``, ``generation_model``, ``optimizer_model``,
   ``judge_model``), so this makes an existing distinction addressable rather
   than inventing one.

Each role's backend is built by :mod:`phoenix_rag.providers.registry`. The wired
backends are ``mistral`` and ``openai`` (chat + embedding), ``anthropic`` (chat),
and ``local`` (an OpenAI-compatible chat endpoint plus in-process embeddings).
Declaring a backend that has no builder raises :class:`UnsupportedBackendError`
at build time rather than failing obscurely somewhere deep in a call stack.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Literal

from phoenix_rag.workspace import active_workspace

# The four distinct jobs an LLM/embedding backend does in this system. Each can
# be pointed at a different provider and model.
ROLES = ("embedding", "generation", "optimizer", "judge")


class UnsupportedBackendError(RuntimeError):
    """A configured provider backend has no implementation yet."""


def _filtered(cls, data: dict) -> dict:
    """Keep only keys that are real fields of `cls`.

    Configs outlive the code that wrote them: a key removed from a dataclass
    (``use_llm_optimizer`` was one) must not turn every previously-saved config
    file into a TypeError on load.
    """
    return {k: v for k, v in data.items() if k in cls.__dataclass_fields__}


# --------------------------------------------------------------------------
# Per-role provider settings
# --------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Which backend and model serve one role, and how hard we may push it.

    The API key is referenced by *environment variable name*, never stored
    inline. That is what lets the YAML file be committed: it records that the
    optimizer role authenticates via ``OPENAI_API_KEY``, while the value of
    that variable stays in ``.env``, which is gitignored.
    """

    backend: str = "mistral"
    model: str = ""
    api_key_env: str | None = None

    # For self-hosted / OpenAI-compatible endpoints (Ollama, vLLM, LM Studio).
    base_url: str | None = None

    # Request pacing. Meaningful for hosted APIs with a published per-minute
    # quota; a local backend can leave requests_per_minute high.
    requests_per_minute: int = 45
    max_retries: int = 5
    base_backoff_seconds: float = 2.0

    def resolve_api_key(self) -> str:
        """Read the key from the environment. Empty string when unset.

        Returning empty rather than raising keeps *validation* separate from
        *lookup*: a local backend legitimately has no key, and the component
        that actually needs one reports the specific missing variable.
        """
        if not self.api_key_env:
            return ""
        return os.getenv(self.api_key_env, "")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderConfig":
        return cls(**_filtered(cls, data))


def _mistral(model: str) -> ProviderConfig:
    """A Mistral role default, matching the models the flat config shipped."""
    return ProviderConfig(backend="mistral", model=model, api_key_env="MISTRAL_API_KEY")


@dataclass
class ProvidersConfig:
    """One :class:`ProviderConfig` per role.

    The defaults reproduce the previous Mistral-everywhere setup exactly, so a
    run that does not mention providers behaves as it always did.
    """

    embedding: ProviderConfig = field(default_factory=lambda: _mistral("mistral-embed"))
    generation: ProviderConfig = field(
        default_factory=lambda: _mistral("mistral-small-latest")
    )
    optimizer: ProviderConfig = field(
        default_factory=lambda: _mistral("mistral-large-latest")
    )
    judge: ProviderConfig = field(default_factory=lambda: _mistral("mistral-large-latest"))

    def as_map(self) -> dict[str, ProviderConfig]:
        return {role: getattr(self, role) for role in ROLES}

    def backends(self) -> set[str]:
        return {provider.backend for provider in self.as_map().values()}

    def missing_api_keys(self) -> list[tuple[str, str]]:
        """Roles whose configured API-key variable is unset in the environment.

        Backend-agnostic: a role is flagged only if it *names* an ``api_key_env``
        (hosted backends do; a local backend leaves it ``None``) and that
        variable resolves empty. Returns ``(role, env_var)`` pairs so a UI can
        name exactly which variable to set, for whichever backends are in play --
        without going through the Mistral-only :attr:`AppConfig.mistral` view,
        which raises for any non-Mistral config.
        """
        return [
            (role, provider.api_key_env)
            for role, provider in self.as_map().items()
            if provider.api_key_env and not provider.resolve_api_key()
        ]

    def to_dict(self) -> dict:
        return {role: provider.to_dict() for role, provider in self.as_map().items()}

    @classmethod
    def from_dict(cls, data: dict) -> "ProvidersConfig":
        defaults = cls()
        resolved = {}
        for role in ROLES:
            block = data.get(role)
            if not block:
                resolved[role] = getattr(defaults, role)
                continue
            # Merge onto the role's default so a YAML block may specify only
            # what it overrides (e.g. just `model:`) and still get a usable key
            # env and pacing.
            merged = getattr(defaults, role).to_dict()
            merged.update(block)
            resolved[role] = ProviderConfig.from_dict(merged)
        return cls(**resolved)


# --------------------------------------------------------------------------
# Mistral compatibility view
# --------------------------------------------------------------------------

@dataclass
class MistralSettings:
    """Per-role settings consumed by the Mistral client.

    :class:`~phoenix_rag.providers.mistral.MistralClient` and
    :class:`~phoenix_rag.core.embeddings.MistralEmbeddings` each take one of
    these. The registry builds one per Mistral role from that role's
    :class:`ProviderConfig` (see ``providers.registry._mistral_settings_for``),
    so the four ``*_model`` fields just carry that role's model -- whichever
    field the role actually reads holds the right value, and the rest are inert.
    """

    api_key: str = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))

    embedding_model: str = "mistral-embed"
    generation_model: str = "mistral-small-latest"
    optimizer_model: str = "mistral-large-latest"
    judge_model: str = "mistral-large-latest"

    # Free-tier rate limiting (requests per minute). Adjust to your plan.
    requests_per_minute: int = 45
    max_retries: int = 5
    base_backoff_seconds: float = 2.0


# --------------------------------------------------------------------------
# Retrieval configuration (the part the optimizer is allowed to mutate)
# --------------------------------------------------------------------------

RetrieverType = Literal["similarity", "mmr", "similarity_score_threshold"]

RETRIEVER_TYPES: tuple[str, ...] = (
    "similarity",
    "mmr",
    "similarity_score_threshold",
)


@dataclass
class RetrievalConfig:
    """A single point-in-search-space configuration for the RAG pipeline.

    This is the object the optimizer mutates between iterations. Keep it
    flat and serializable.
    """

    chunk_size: int = 300
    chunk_overlap: int = 50

    top_k: int = 1
    similarity_threshold: float = 0.0  # only used by "similarity_score_threshold"
    retriever_type: RetrieverType = "similarity"

    prompt_template: str = (
        "You are a helpful assistant answering questions using ONLY the "
        "provided context. If the answer is not contained in the context, "
        "say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalConfig":
        return cls(**_filtered(cls, data))

    def copy_with(self, **overrides) -> "RetrievalConfig":
        data = self.to_dict()
        data.update(overrides)
        return RetrievalConfig.from_dict(data)


# --------------------------------------------------------------------------
# Question generation configuration
# --------------------------------------------------------------------------

@dataclass
class QuestionGenerationConfig:
    batch_size_chars: int = 6000  # characters per batch sent to the LLM
    questions_per_batch: int = 1
    question_types: tuple = (
        "factual",
        "reasoning",
        "comparison",
        "why",
        "multi-hop",
        "summarization",
        "definition",
        "relationship",
        "edge-case",
    )
    dedup_similarity_threshold: float = 0.9  # cosine similarity for near-dup removal
    regenerate_each_iteration: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "QuestionGenerationConfig":
        data = dict(_filtered(cls, data))
        if "question_types" in data:
            data["question_types"] = tuple(data["question_types"])
        return cls(**data)


# --------------------------------------------------------------------------
# Optimization loop configuration
# --------------------------------------------------------------------------

_BOUNDS_FIELDS = ("top_k_bounds", "chunk_size_bounds", "similarity_threshold_bounds")


@dataclass
class OptimizerConfig:
    """Targets, budget, and the bounds the proposer must stay inside."""

    max_iterations: int = 10

    # Iteration 1's chunk_size/chunk_overlap/top_k are derived from the document
    # profile (see core/seed_config.py) rather than taken from this config's
    # retrieval block. Set False to start from the retrieval block instead --
    # kept so the unseeded arm stays reproducible for comparison.
    seed_from_profile: bool = True

    target_faithfulness: float = 1.0
    target_context_recall: float = 9.5
    target_context_precision: float = 9.0
    target_response_relevancy: float = 9.5

    top_k_step: int = 2
    chunk_size_step: int = 200
    similarity_threshold_step: float = 0.05

    top_k_bounds: tuple = (2, 10)
    chunk_size_bounds: tuple = (300, 1500)
    similarity_threshold_bounds: tuple = (0.1, 0.75)

    @classmethod
    def from_dict(cls, data: dict) -> "OptimizerConfig":
        data = dict(_filtered(cls, data))
        # YAML and JSON both round-trip tuples as lists. Restore them so the
        # loaded object is indistinguishable from a constructed one.
        for name in _BOUNDS_FIELDS:
            if name in data and data[name] is not None:
                data[name] = tuple(data[name])
        return cls(**data)


# --------------------------------------------------------------------------
# Top level app config
# --------------------------------------------------------------------------

@dataclass
class AppConfig:
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    question_generation: QuestionGenerationConfig = field(
        default_factory=QuestionGenerationConfig
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Resolved against the active workspace at construction time, so a bare
    # AppConfig() still has usable paths without this module owning any.
    source_document: str = field(
        default_factory=lambda: str(active_workspace().data_dir / "source.pdf")
    )
    faiss_index_path: str = field(
        default_factory=lambda: str(active_workspace().data_dir / "faiss_index")
    )
    benchmark_path: str = field(
        default_factory=lambda: str(
            active_workspace().generated_questions_dir / "benchmark.json"
        )
    )
    summary_path: str = field(
        default_factory=lambda: str(
            active_workspace().generated_questions_dir / "document_summary.txt"
        )
    )
    profile_path: str = field(
        default_factory=lambda: str(
            active_workspace().generated_questions_dir / "document_profile.json"
        )
    )

    # Opt-in multi-document mode. None (the default) keeps every path above
    # describing a single document. When set, the experiment runner sources its
    # benchmark, summary, profile and FAISS index from the corpus manifest under
    # this root instead -- see core/corpus.py.
    corpus_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "providers": self.providers.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "question_generation": _tuples_to_lists(asdict(self.question_generation)),
            "optimizer": _tuples_to_lists(asdict(self.optimizer)),
            "source_document": self.source_document,
            "faiss_index_path": self.faiss_index_path,
            "benchmark_path": self.benchmark_path,
            "summary_path": self.summary_path,
            "profile_path": self.profile_path,
            "corpus_path": self.corpus_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        defaults = cls()
        return cls(
            providers=ProvidersConfig.from_dict(data.get("providers", {})),
            retrieval=RetrievalConfig.from_dict(data.get("retrieval", {})),
            question_generation=QuestionGenerationConfig.from_dict(
                data.get("question_generation", {})
            ),
            optimizer=OptimizerConfig.from_dict(data.get("optimizer", {})),
            source_document=data.get("source_document") or defaults.source_document,
            faiss_index_path=data.get("faiss_index_path") or defaults.faiss_index_path,
            benchmark_path=data.get("benchmark_path") or defaults.benchmark_path,
            summary_path=data.get("summary_path") or defaults.summary_path,
            profile_path=data.get("profile_path") or defaults.profile_path,
            corpus_path=data.get("corpus_path"),
        )


def _tuples_to_lists(value):
    """Tuples -> lists, recursively, so YAML/JSON output is predictable."""
    if isinstance(value, tuple):
        return [_tuples_to_lists(v) for v in value]
    if isinstance(value, list):
        return [_tuples_to_lists(v) for v in value]
    if isinstance(value, dict):
        return {k: _tuples_to_lists(v) for k, v in value.items()}
    return value
