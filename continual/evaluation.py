"""Deterministic repeated evaluation for continual boundary matrices."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def run_repeated_evaluation(
    evaluator: Any,
    *,
    policy_params: Any,
    repeats: int,
    num_eval_envs: int,
) -> dict[str, float | int]:
    """Average several fixed evaluator batches and report population std.

    The same evaluator instance is reused, so its compiled rollout is reused
    while its deterministic PRNG stream advances once per repeat.
    """
    if repeats < 1:
        raise ValueError("continual_eval_repeats must be positive")
    if num_eval_envs < 1:
        raise ValueError("num_eval_envs must be positive")

    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        metrics = evaluator.run_evaluation(
            policy_params=policy_params,
            training_metrics={},
        )
        for key, value in metrics.items():
            array = np.asarray(value)
            if array.shape == () and np.isfinite(array).item():
                samples[key].append(float(array))

    aggregated: dict[str, float | int] = {
        "eval/repeats": repeats,
        "eval/num_episodes": repeats * num_eval_envs,
    }
    for key, values in samples.items():
        # Evaluator walltime is cumulative across calls, so the last value is
        # the meaningful total. Throughput and episode metrics are averaged.
        aggregated[key] = (
            values[-1]
            if key == "eval/walltime"
            else float(np.mean(values))
        )
        if key.startswith("eval/episode_") or key == "eval/avg_episode_length":
            aggregated[f"{key}_std"] = float(np.std(values, ddof=0))
    return aggregated
