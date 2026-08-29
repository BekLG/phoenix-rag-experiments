"""
config.py
=========
Central configuration for Phoenix RAG.

Everything that is "tunable" during the optimization loop (chunking,
retrieval, prompt template, etc.) lives here as a single dataclass so it
can be trivially serialized to/from JSON and passed around the pipeline
as one object.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
GENERATED_QUESTIONS_DIR = ROOT_DIR / "generated_questions"
LOGS_DIR = ROOT_DIR / "logs"
CONFIG_DIR = ROOT_DIR / "config"

for _dir in (DATA_DIR, RESULTS_DIR, GENERATED_QUESTIONS_DIR, LOGS_DIR, CONFIG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# API / model settings
# --------------------------------------------------------------------------

@dataclass
class MistralSettings:
    """Mistral AI credentials and model names."""

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


@dataclass
class RetrievalConfig:
    """A single point-in-search-space configuration for the RAG pipeline.

    This is the object the optimizer mutates between iterations. Keep it
    flat and JSON-serializable.
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
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

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
    questions_per_batch: int =  1
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


# --------------------------------------------------------------------------
# Optimization loop configuration
# --------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    """Note: this is now only ever consumed by the LLM-driven optimizer
    (llm_optimizer.propose_next_config_llm) -- experiment_runner.py no
    longer has a rule-based path to switch between, so there is no
    use_llm_optimizer flag here anymore.
    """

    max_iterations: int = 10

    # Iteration 1's chunk_size/chunk_overlap/top_k are derived from the document
    # profile (see seed_config.py) rather than taken from this config's retrieval
    # block. Set False to start from the retrieval block as before -- kept so the
    # unseeded arm stays reproducible for comparison.
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
        """Filters out unknown keys (e.g. a stale 'use_llm_optimizer' from
        a default_config.json saved before that flag was removed) instead
        of letting them crash the constructor with a TypeError.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------
# Top level app config
# --------------------------------------------------------------------------

@dataclass
class AppConfig:
    mistral: MistralSettings = field(default_factory=MistralSettings)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    question_generation: QuestionGenerationConfig = field(
        default_factory=QuestionGenerationConfig
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    source_document: str = str(DATA_DIR / "source.pdf")
    faiss_index_path: str = str(DATA_DIR / "faiss_index")
    benchmark_path: str = str(GENERATED_QUESTIONS_DIR / "benchmark.json")
    summary_path: str = str(GENERATED_QUESTIONS_DIR / "document_summary.txt")
    profile_path: str = str(GENERATED_QUESTIONS_DIR / "document_profile.json")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(_dataclass_to_json_safe(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        mistral_data = data.get("mistral", {})
        mistral_data = {
            key: value for key, value in mistral_data.items()
            if key in MistralSettings.__dataclass_fields__
        }
        return cls(
            mistral=MistralSettings(**mistral_data),
            retrieval=RetrievalConfig.from_dict(data.get("retrieval", {})),
            question_generation=QuestionGenerationConfig(
                **data.get("question_generation", {})
            ),
            optimizer=OptimizerConfig.from_dict(data.get("optimizer", {})),
            source_document=data.get("source_document", str(DATA_DIR / "source.pdf")),
            faiss_index_path=data.get("faiss_index_path", str(DATA_DIR / "faiss_index")),
            benchmark_path=data.get(
                "benchmark_path", str(GENERATED_QUESTIONS_DIR / "benchmark.json")
            ),
            summary_path=data.get(
                "summary_path", str(GENERATED_QUESTIONS_DIR / "document_summary.txt")
            ),
            profile_path=data.get(
                "profile_path", str(GENERATED_QUESTIONS_DIR / "document_profile.json")
            ),
        )


def _dataclass_to_json_safe(obj) -> dict:
    """asdict() but tuples -> lists so json.dumps behaves predictably."""
    raw = asdict(obj)

    def _convert(x):
        if isinstance(x, tuple):
            return list(x)
        if isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        return x

    return {k: _convert(v) for k, v in raw.items()}


DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"


def load_or_create_default_config() -> AppConfig:
    if DEFAULT_CONFIG_PATH.exists():
        return AppConfig.load(DEFAULT_CONFIG_PATH)
    cfg = AppConfig()
    cfg.save(DEFAULT_CONFIG_PATH)
    return cfg
