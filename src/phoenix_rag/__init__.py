"""
Phoenix RAG
===========
A self-optimizing RAG system: it tunes retrieval parameters *and* the answer
prompt against a fixed, LLM-judged benchmark generated from your own documents.

Quick start
-----------
    import phoenix_rag as pr

    workspace = pr.Workspace("./my-project").ensure()
    pr.use_workspace(workspace)

    config = pr.load_or_create_default_config()
    config.source_document = "docs/handbook.pdf"

    best = pr.run_experiment(config)
    print(best["scores"], best["config"].to_dict())

Importing this package reads no environment and touches no disk. Credentials are
loaded when you ask for them (:func:`load_or_create_default_config` does it, or
call :func:`load_env` yourself); directories are created only by
:meth:`Workspace.ensure`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from phoenix_rag.config import (
    AppConfig,
    MistralSettings,
    OptimizerConfig,
    ProviderConfig,
    ProvidersConfig,
    QuestionGenerationConfig,
    RetrievalConfig,
    UnsupportedBackendError,
    load_config,
    load_env,
    load_or_create_default_config,
    save_config,
)
from phoenix_rag.workspace import (
    Workspace,
    active_workspace,
    reset_workspace,
    use_workspace,
)

__version__ = "0.1.0"

# --------------------------------------------------------------------------
# Lazily-exposed heavy API
# --------------------------------------------------------------------------
# The optimization loop pulls in ragas, datasets, torch and faiss transitively.
# That is several seconds of import time, so it must not be paid by someone who
# imported this package only to build an AppConfig -- which is exactly what a
# front-end or a test does. These names resolve on first attribute access
# instead (PEP 562).

_LAZY_EXPORTS = {
    "run_experiment": "phoenix_rag.optimization.runner",
    "prepare_inputs": "phoenix_rag.optimization.runner",
    "propose_next_config_llm": "phoenix_rag.optimization.llm_optimizer",
    "propose_seed_config": "phoenix_rag.core.seed_config",
    "RagPipeline": "phoenix_rag.core.rag_pipeline",
    "RagResult": "phoenix_rag.core.rag_pipeline",
    "MistralEmbeddings": "phoenix_rag.core.embeddings",
    "DocumentProfile": "phoenix_rag.core.document_profile",
    "compute_profile": "phoenix_rag.core.document_profile",
    "load_document": "phoenix_rag.core.document_loader",
    "split_documents": "phoenix_rag.core.chunking",
    "run_evaluation": "phoenix_rag.evaluation.evaluator",
    "BenchmarkQuestion": "phoenix_rag.benchmark.question_generator",
    "get_or_create_benchmark": "phoenix_rag.benchmark.question_generator",
}

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from phoenix_rag.benchmark.question_generator import (
        BenchmarkQuestion,
        get_or_create_benchmark,
    )
    from phoenix_rag.core.chunking import split_documents
    from phoenix_rag.core.document_loader import load_document
    from phoenix_rag.core.document_profile import DocumentProfile, compute_profile
    from phoenix_rag.core.embeddings import MistralEmbeddings
    from phoenix_rag.core.rag_pipeline import RagPipeline, RagResult
    from phoenix_rag.core.seed_config import propose_seed_config
    from phoenix_rag.evaluation.evaluator import run_evaluation
    from phoenix_rag.optimization.llm_optimizer import propose_next_config_llm
    from phoenix_rag.optimization.runner import prepare_inputs, run_experiment


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # resolve once, then it is a normal attribute
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_EXPORTS])


__all__ = [
    "__version__",
    # configuration
    "AppConfig",
    "MistralSettings",
    "OptimizerConfig",
    "ProviderConfig",
    "ProvidersConfig",
    "QuestionGenerationConfig",
    "RetrievalConfig",
    "UnsupportedBackendError",
    "load_config",
    "load_env",
    "load_or_create_default_config",
    "save_config",
    # workspace
    "Workspace",
    "active_workspace",
    "reset_workspace",
    "use_workspace",
    # pipeline / loop
    *_LAZY_EXPORTS,
]
