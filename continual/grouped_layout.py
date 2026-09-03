"""Padding-only layout for controlled StableCRL comparisons.

Unlike :mod:`continual.semantic_layout`, this layout does not reorder the raw
BuilderBench observation into per-cube records and does not quantize the
previous selector into a one-hot feature.  It pads each existing raw feature
group to ``max_cubes`` and appends a validity mask.  This makes it useful as a
diagnostic control: the environment dynamics and upstream feature ordering
are unchanged, while all tasks expose fixed observation and goal sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from continual.semantic_layout import (
    ANGULAR_VELOCITY_DIM,
    LINEAR_VELOCITY_DIM,
    POSITION_DIM,
    QUATERNION_DIM,
    RAW_CUBE_FEATURE_DIM,
)


_GROUP_WIDTHS = (
    POSITION_DIM,
    QUATERNION_DIM,
    LINEAR_VELOCITY_DIM,
    ANGULAR_VELOCITY_DIM,
)


@dataclass(frozen=True)
class GroupedPadLayout:
    """Fixed-size layout that preserves upstream grouped feature ordering."""

    max_cubes: int = 8

    def __post_init__(self) -> None:
        if self.max_cubes < 1:
            raise ValueError("max_cubes must be positive")

    @property
    def observation_size(self) -> int:
        # Four padded raw groups, the original scalar selector, and a mask.
        return RAW_CUBE_FEATURE_DIM * self.max_cubes + 1 + self.max_cubes

    @property
    def goal_size(self) -> int:
        return POSITION_DIM * self.max_cubes + self.max_cubes

    def validate_num_cubes(self, num_cubes: int) -> None:
        if not 1 <= num_cubes <= self.max_cubes:
            raise ValueError(
                f"num_cubes={num_cubes} is outside [1, {self.max_cubes}]"
            )

    def pack_observation(self, raw: Any, num_cubes: int, *, xp=np) -> Any:
        self.validate_num_cubes(num_cubes)
        expected = RAW_CUBE_FEATURE_DIM * num_cubes + 1
        if raw.shape[-1] != expected:
            raise ValueError(
                f"expected raw observation dim {expected} for {num_cubes} "
                f"cubes, got {raw.shape[-1]}; delta_control layouts are not "
                "supported"
            )

        cursor = 0
        groups = []
        for width in _GROUP_WIDTHS:
            size = width * num_cubes
            groups.append(raw[..., cursor:cursor + size])
            groups.append(xp.zeros(
                raw.shape[:-1] + (width * (self.max_cubes - num_cubes),),
                dtype=raw.dtype,
            ))
            cursor += size
        selector = raw[..., cursor:cursor + 1]
        mask = xp.concatenate(
            [
                xp.ones(raw.shape[:-1] + (num_cubes,), dtype=raw.dtype),
                xp.zeros(
                    raw.shape[:-1] + (self.max_cubes - num_cubes,),
                    dtype=raw.dtype,
                ),
            ],
            axis=-1,
        )
        return xp.concatenate([*groups, selector, mask], axis=-1)

    def unpack_observation(self, packed: Any, num_cubes: int, *, xp=np) -> Any:
        self.validate_num_cubes(num_cubes)
        if packed.shape[-1] != self.observation_size:
            raise ValueError(
                f"expected packed observation dim {self.observation_size}, "
                f"got {packed.shape[-1]}"
            )

        cursor = 0
        groups = []
        for width in _GROUP_WIDTHS:
            padded_size = width * self.max_cubes
            groups.append(packed[..., cursor:cursor + width * num_cubes])
            cursor += padded_size
        selector = packed[..., cursor:cursor + 1]
        return xp.concatenate([*groups, selector], axis=-1)

    def pack_goal(
        self, raw: Any, num_cubes: int, *, mask: Any | None = None, xp=np
    ) -> Any:
        self.validate_num_cubes(num_cubes)
        expected = POSITION_DIM * num_cubes
        if raw.shape[-1] != expected:
            raise ValueError(
                f"expected raw goal dim {expected}, got {raw.shape[-1]}"
            )
        if mask is None:
            valid = xp.ones(
                raw.shape[:-1] + (num_cubes,), dtype=raw.dtype
            )
        else:
            if mask.shape[-1] != num_cubes:
                raise ValueError(
                    f"expected goal mask dim {num_cubes}, got "
                    f"{mask.shape[-1]}"
                )
            valid = xp.broadcast_to(
                mask, raw.shape[:-1] + (num_cubes,)
            ).astype(raw.dtype)
        positions = raw.reshape(
            raw.shape[:-1] + (num_cubes, POSITION_DIM)
        )
        masked_raw = (positions * valid[..., None]).reshape(
            raw.shape[:-1] + (-1,)
        )
        padding = xp.zeros(
            raw.shape[:-1] + (POSITION_DIM * (self.max_cubes - num_cubes),),
            dtype=raw.dtype,
        )
        padded_mask = xp.concatenate(
            [
                valid,
                xp.zeros(
                    raw.shape[:-1] + (self.max_cubes - num_cubes,),
                    dtype=raw.dtype,
                ),
            ],
            axis=-1,
        )
        return xp.concatenate([masked_raw, padding, padded_mask], axis=-1)

    def unpack_goal(self, packed: Any, num_cubes: int) -> Any:
        self.validate_num_cubes(num_cubes)
        if packed.shape[-1] != self.goal_size:
            raise ValueError(
                f"expected packed goal dim {self.goal_size}, "
                f"got {packed.shape[-1]}"
            )
        return packed[..., :POSITION_DIM * num_cubes]
