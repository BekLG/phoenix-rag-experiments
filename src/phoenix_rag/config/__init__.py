"""
phoenix_rag.config
==================
Configuration objects and their YAML persistence.

Split so that :mod:`~phoenix_rag.config.schema` stays pure data -- importable
with no I/O and no environment reads -- while :mod:`~phoenix_rag.config.loader`
owns the filesystem and ``.env`` handling. Everything is re-exported here, so
consumers import from ``phoenix_rag.config`` and do not care which half a name
came from.
"""

from phoenix_rag.config.loader import (
    load_config,
    load_env,
    load_or_create_default_config,
    save_config,
)
from phoenix_rag.config.schema import (
    ROLES,
    RETRIEVER_TYPES,
    AppConfig,
    MistralSettings,
    OptimizerConfig,
    ProviderConfig,
    ProvidersConfig,
    QuestionGenerationConfig,
    RetrievalConfig,
    RetrieverType,
    UnsupportedBackendError,
)

__all__ = [
    "ROLES",
    "RETRIEVER_TYPES",
    "AppConfig",
    "MistralSettings",
    "OptimizerConfig",
    "ProviderConfig",
    "ProvidersConfig",
    "QuestionGenerationConfig",
    "RetrievalConfig",
    "RetrieverType",
    "UnsupportedBackendError",
    "load_config",
    "load_env",
    "load_or_create_default_config",
    "save_config",
]
