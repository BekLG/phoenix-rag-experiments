"""
optimizer.py
What's left here are the two generic, config-shape-agnostic helpers that
llm_optimizer.py still depends on:
  - _clamp: bounds-clamp a numeric value.
  - meets_targets: check a scores dict against OptimizerConfig's targets.
"""

from __future__ import annotations

from config import OptimizerConfig

_EPS = 1e-4


def _clamp(value: float | int, bounds: tuple[float, float]) -> float | int:
    low, high = bounds
    return max(low, min(high, value))


def _at_least(value: float, target: float) -> bool:
    return value >= target - _EPS


def meets_targets(scores: dict[str, float], opt_config: OptimizerConfig) -> bool:
    return (
        _at_least(scores.get("faithfulness", 0.0), opt_config.target_faithfulness)
        and _at_least(scores.get("context_recall", 0.0), opt_config.target_context_recall)
        and _at_least(scores.get("context_precision", 0.0), opt_config.target_context_precision)
        and _at_least(scores.get("response_relevancy", 0.0), opt_config.target_response_relevancy)
    )