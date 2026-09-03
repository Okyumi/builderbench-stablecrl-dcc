"""Streaming, dependency-free continual learning-curve AUC helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OnlineNormalizedAUC:
    """Trapezoidal AUC divided by the current x-axis extent.

    ``initial_value`` is the measured success before task training begins.
    It defaults to zero for compatibility with older single-task runs.  A
    continual run supplies its actual post-transfer, pre-training success so
    the area measures adaptation from the policy that really enters the task.
    """

    initial_value: float = 0.0
    previous_step: float = 0.0
    previous_value: float | None = None
    area: float = 0.0

    def __post_init__(self) -> None:
        self.initial_value = float(self.initial_value)
        if not math.isfinite(self.initial_value):
            raise ValueError("AUC initial value must be finite")
        if not 0.0 <= self.initial_value <= 1.0:
            raise ValueError(
                "AUC initial value must be in [0, 1], got "
                f"{self.initial_value}"
            )
        if self.previous_value is None:
            self.previous_value = self.initial_value

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
            * (float(self.previous_value) + value)
            / 2.0
        )
        self.previous_step = step
        self.previous_value = value
        return self.area / step
