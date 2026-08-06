"""
compare_results.py
===================
Loads the optimized run's best_configuration.json and the non-optimized
baseline's baseline_result.json, and prints a side-by-side comparison so
you can see whether optimization actually helped on this document, and
by how much.

Usage:
    python compare_results.py
"""

from __future__ import annotations

import json

from config import RESULTS_DIR

BEST_CONFIG_PATH = RESULTS_DIR / "best_configuration.json"
BASELINE_RESULT_PATH = RESULTS_DIR / "baseline_result.json"

METRICS = ["faithfulness", "context_recall", "context_precision", "response_relevancy"]


def _weighted(scores: dict) -> float:
    return (
        scores.get("faithfulness", 0.0) * 0.40
        + scores.get("context_recall", 0.0) * 0.20
        + scores.get("context_precision", 0.0) * 0.20
        + scores.get("response_relevancy", 0.0) * 0.20
    )


def main() -> None:
    if not BEST_CONFIG_PATH.exists():
        print(f"No optimized result found at {BEST_CONFIG_PATH} -- run app.py first.")
        return
    if not BASELINE_RESULT_PATH.exists():
        print(f"No baseline result found at {BASELINE_RESULT_PATH} -- run baseline_runner.py first.")
        return

    optimized = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_RESULT_PATH.read_text(encoding="utf-8"))

    opt_scores = optimized["scores"]
    base_scores = baseline["scores"]
    opt_weighted = _weighted(opt_scores)
    base_weighted = baseline.get("weighted_score", _weighted(base_scores))

    print(f"{'Metric':<22} {'Baseline (local, fixed)':<26} {'Optimized (Mistral, LLM-tuned)':<32} {'Delta':<10}")
    print("-" * 92)
    for m in METRICS:
        b = base_scores.get(m, 0.0)
        o = opt_scores.get(m, 0.0)
        print(f"{m:<22} {b:<26.4f} {o:<32.4f} {o - b:+.4f}")
    print("-" * 92)
    print(f"{'weighted_score':<22} {base_weighted:<26.4f} {opt_weighted:<32.4f} {opt_weighted - base_weighted:+.4f}")

    winner = "Optimized" if opt_weighted > base_weighted else "Baseline"
    print(f"\nHigher weighted score: {winner}")


if __name__ == "__main__":
    main()