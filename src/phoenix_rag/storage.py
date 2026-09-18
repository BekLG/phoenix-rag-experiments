"""
storage.py
==========
Persistence layer. Every iteration writes:
    - configurations     -> JSON (results/configs/iteration_N.json)
    - experiment_results -> CSV  (results/experiment_results.csv, appended)
    - evaluation_scores  -> CSV  (results/evaluation_scores.csv, appended)
    - best_configuration -> JSON (results/best_configuration.json, overwritten
      whenever a new best is found)

PATHS ARE RESOLVED LAZILY
-------------------------
These used to be module-level constants computed from a ``RESULTS_DIR`` global,
with a ``CONFIGS_DIR.mkdir()`` at import. That made ``import storage`` create a
directory before any caller had asked for one, in whatever directory the process
started in.

The destinations are now computed on access, from the active workspace (or from
whatever :func:`configure_results_dir` last pointed at). The uppercase names are
kept as lazily-resolved module attributes via PEP 562 ``__getattr__``, because
``storage.BEST_CONFIG_PATH`` is read in several places in the front-ends and it
reads better there than a function call would. Each access returns a fresh
snapshot, which is what the redirect semantics below already implied.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path

from phoenix_rag.config import RetrievalConfig
from phoenix_rag.workspace import active_workspace

logger = logging.getLogger("phoenix_rag.storage")

_SCORE_FIELDS = [
    "faithfulness",
    "context_recall",
    "context_precision",
    "response_relevancy",
]

# Set by configure_results_dir. None means "use the active workspace".
_results_dir_override: Path | None = None


# --------------------------------------------------------------------------
# Destinations
# --------------------------------------------------------------------------

def results_dir() -> Path:
    if _results_dir_override is not None:
        return _results_dir_override
    return active_workspace().results_dir


def configs_dir() -> Path:
    return results_dir() / "configs"


def experiment_results_csv() -> Path:
    return results_dir() / "experiment_results.csv"


def evaluation_scores_csv() -> Path:
    return results_dir() / "evaluation_scores.csv"


def best_config_path() -> Path:
    return results_dir() / "best_configuration.json"


_LAZY_PATHS = {
    "RESULTS_DIR": results_dir,
    "CONFIGS_DIR": configs_dir,
    "EXPERIMENT_RESULTS_CSV": experiment_results_csv,
    "EVALUATION_SCORES_CSV": evaluation_scores_csv,
    "BEST_CONFIG_PATH": best_config_path,
}


def __getattr__(name: str):
    """Resolve the legacy uppercase path names on access (PEP 562)."""
    resolver = _LAZY_PATHS.get(name)
    if resolver is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return resolver()


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_PATHS])


def configure_results_dir(target: str | Path | None) -> Path:
    """Point this module's output at `target` for the current process.

    Callers that run a nested experiment use this to keep their output out of the
    top-level ``results/`` directory instead of overwriting a previous run's
    ``best_configuration.json``.

    Process-wide by design: it is a redirect, not a scope. Pass ``None`` (or call
    :func:`reset_results_dir`) to go back to the active workspace's ``results/``.
    """
    global _results_dir_override

    if target is None:
        _results_dir_override = None
        resolved = results_dir()
    else:
        _results_dir_override = Path(target)
        resolved = _results_dir_override

    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "configs").mkdir(parents=True, exist_ok=True)
    logger.info("Results for this run will be written to %s", resolved)
    return resolved


def reset_results_dir() -> None:
    """Drop any redirect, so destinations come from the active workspace again."""
    global _results_dir_override
    _results_dir_override = None


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def _atomic_write_json(path: str | Path, data) -> None:
    """Write JSON beside its target, then atomically replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, ensure_ascii=False)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(serialized)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def save_iteration_config(iteration: int, config: RetrievalConfig) -> Path:
    """Persist an iteration; the runner ignores the returned convenience path."""
    path = configs_dir() / f"iteration_{iteration:03d}.json"
    _atomic_write_json(path, config.to_dict())
    return path


def _append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    # Appending is intentionally non-atomic: replacing the file atomically would
    # require rewriting the full CSV for every iteration. JSON checkpoints are
    # atomic; switch this path to an atomic rewrite if partial rows are observed.
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_evaluation_scores(
    iteration: int, scores: dict[str, float], applied_rules: list[str]
) -> None:
    row = {"iteration": iteration, **{k: scores.get(k) for k in _SCORE_FIELDS}}
    row["applied_rules"] = "; ".join(applied_rules) if applied_rules else ""
    fieldnames = ["iteration", *_SCORE_FIELDS, "applied_rules"]
    _append_csv_row(evaluation_scores_csv(), row, fieldnames)


def append_experiment_result(
    iteration: int, config: RetrievalConfig, scores: dict[str, float]
) -> None:
    row = {
        "iteration": iteration,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "similarity_threshold": config.similarity_threshold,
        "retriever_type": config.retriever_type,
        **{k: scores.get(k) for k in _SCORE_FIELDS},
    }
    fieldnames = [
        "iteration",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "similarity_threshold",
        "retriever_type",
        *_SCORE_FIELDS,
    ]
    _append_csv_row(experiment_results_csv(), row, fieldnames)


def save_best_configuration(
    iteration: int, config: RetrievalConfig, scores: dict[str, float]
) -> None:
    payload = {
        "iteration": iteration,
        "config": config.to_dict(),
        "scores": scores,
    }
    _atomic_write_json(best_config_path(), payload)
    logger.info("New best configuration saved (iteration %d): %s", iteration, scores)


def load_best_configuration() -> dict | None:
    path = best_config_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def average_score(scores: dict[str, float]) -> float:
    values = [scores.get(k, 0.0) for k in _SCORE_FIELDS]
    return sum(values) / len(values) if values else 0.0
