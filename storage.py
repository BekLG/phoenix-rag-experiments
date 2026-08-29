"""
storage.py
==========
Persistence layer. Every iteration writes:
    - configurations   -> JSON  (results/configs/iteration_N.json)
    - experiment_results -> CSV (results/experiment_results.csv, appended)
    - evaluation_scores -> CSV  (results/evaluation_scores.csv, appended)
    - best_configuration -> JSON (results/best_configuration.json, overwritten
      whenever a new best is found)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path

from config import RESULTS_DIR, RetrievalConfig

logger = logging.getLogger("phoenix_rag.storage")

CONFIGS_DIR = RESULTS_DIR / "configs"
EXPERIMENT_RESULTS_CSV = RESULTS_DIR / "experiment_results.csv"
EVALUATION_SCORES_CSV = RESULTS_DIR / "evaluation_scores.csv"
BEST_CONFIG_PATH = RESULTS_DIR / "best_configuration.json"

CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

_SCORE_FIELDS = [
    "faithfulness",
    "context_recall",
    "context_precision",
    "response_relevancy",
]


def configure_results_dir(results_dir: str | Path) -> Path:
    """Point this module's output paths at `results_dir` for the current process.

    Every writer below reads its destination from a module-level global, so
    rebinding those globals redirects the whole persistence layer. Callers that
    run a nested experiment -- the generalization comparison, or a per-corpus
    optimization -- use this to keep their output out of the top-level results/
    directory instead of overwriting a previous run's best_configuration.json.

    Process-wide by design: it is a redirect, not a scope. A caller that needs
    the original paths back must call it again with RESULTS_DIR.
    """
    global CONFIGS_DIR, EXPERIMENT_RESULTS_CSV, EVALUATION_SCORES_CSV, BEST_CONFIG_PATH

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    CONFIGS_DIR = results_dir / "configs"
    EXPERIMENT_RESULTS_CSV = results_dir / "experiment_results.csv"
    EVALUATION_SCORES_CSV = results_dir / "evaluation_scores.csv"
    BEST_CONFIG_PATH = results_dir / "best_configuration.json"
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Results for this run will be written to %s", results_dir)
    return results_dir


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
    path = CONFIGS_DIR / f"iteration_{iteration:03d}.json"
    _atomic_write_json(path, config.to_dict())
    return path


def _append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    # Appending is intentionally non-atomic: replacing the file atomically would
    # require rewriting the full CSV for every iteration. JSON checkpoints are
    # atomic; switch this path to an atomic rewrite if partial rows are observed.
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
    _append_csv_row(EVALUATION_SCORES_CSV, row, fieldnames)


def append_experiment_result(iteration: int, config: RetrievalConfig, scores: dict[str, float]) -> None:
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
    _append_csv_row(EXPERIMENT_RESULTS_CSV, row, fieldnames)


def save_best_configuration(
    iteration: int, config: RetrievalConfig, scores: dict[str, float]
) -> None:
    payload = {
        "iteration": iteration,
        "config": config.to_dict(),
        "scores": scores,
    }
    _atomic_write_json(BEST_CONFIG_PATH, payload)
    logger.info("New best configuration saved (iteration %d): %s", iteration, scores)


def load_best_configuration() -> dict | None:
    if not BEST_CONFIG_PATH.exists():
        return None
    return json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))


def average_score(scores: dict[str, float]) -> float:
    values = [scores.get(k, 0.0) for k in _SCORE_FIELDS]
    return sum(values) / len(values) if values else 0.0
