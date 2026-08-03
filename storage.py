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


def save_iteration_config(iteration: int, config: RetrievalConfig) -> Path:
    path = CONFIGS_DIR / f"iteration_{iteration:03d}.json"
    path.write_text(json.dumps(config.to_dict(), indent=2))
    return path


def _append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="") as f:
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
    BEST_CONFIG_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("New best configuration saved (iteration %d): %s", iteration, scores)


def load_best_configuration() -> dict | None:
    if not BEST_CONFIG_PATH.exists():
        return None
    return json.loads(BEST_CONFIG_PATH.read_text())


def average_score(scores: dict[str, float]) -> float:
    values = [scores.get(k, 0.0) for k in _SCORE_FIELDS]
    return sum(values) / len(values) if values else 0.0
