"""Streaming, dependency-free continual learning-curve AUC helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OnlineNormalizedAUC:
    """Trapezoidal AUC divided by the current x-axis extent.

    The implicit origin is (0, 0). This matches the offline reporting
    convention used for BuilderBench success-rate AUC against environment
    steps and keeps the metric in the same [0, 1] units as success rate.
    """

    previous_step: float = 0.0
    previous_value: float = 0.0
    area: float = 0.0

    def update(self, step: float, value: float) -> float:
        step = float(step)
        value = float(value)
        if not math.isfinite(step) or not math.isfinite(value):
            raise ValueError("AUC observations must be finite")
        if step <= self.previous_step:
            raise ValueError(
                "AUC steps must increase strictly: "
                f"{step} <= {self.previous_step}"
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"AUC value must be in [0, 1], got {value}")
        self.area += (
            (step - self.previous_step)
            * (self.previous_value + value)
            / 2.0
        )
        self.previous_step = step
        self.previous_value = value
        return self.area / step
