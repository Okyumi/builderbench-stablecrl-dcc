"""Fixed-size semantic layout for variable-cube BuilderBench tasks.

Padding by itself gives every slot a stable *shape* but not a stable meaning.
This module preserves BuilderBench object order, attaches the previous
selection to its object record, and adds an explicit validity mask. Masked-set
consumers may be permutation invariant; flat residual consumers instead rely
on the environment contract that object index has stable meaning across the
selected curriculum.

The David-Yan MJX creative environment emits grouped observations::

    [positions (3N), quaternions (4N), linear velocity (3N),
     angular velocity (3N), select_action (1)]

We convert that to::

    [cube_0(14), ..., cube_M(14), valid_mask(M)]

The fourteenth per-cube feature is a one-hot flag for the previously selected
cube. Keeping selection attached to the cube avoids changing the meaning of
the raw index-valued scalar when the number of valid cubes changes.

Goals use ``[positions(3M), valid_mask(M)]``. ``delta_control=True`` adds a
per-cube control field to the upstream observation and is deliberately
rejected until it has a separately versioned layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


POSITION_DIM = 3
QUATERNION_DIM = 4
LINEAR_VELOCITY_DIM = 3
ANGULAR_VELOCITY_DIM = 3
RAW_CUBE_FEATURE_DIM = (
    POSITION_DIM + QUATERNION_DIM + LINEAR_VELOCITY_DIM
    + ANGULAR_VELOCITY_DIM
)
CUBE_FEATURE_DIM = RAW_CUBE_FEATURE_DIM + 1


@dataclass(frozen=True)
class SemanticLayout:
    """Shape contract shared by every task in a continual run."""

    max_cubes: int = 8

    def __post_init__(self) -> None:
        if self.max_cubes < 1:
            raise ValueError("max_cubes must be positive")

    @property
    def observation_size(self) -> int:
        return self.max_cubes * CUBE_FEATURE_DIM + self.max_cubes

    @property
    def goal_size(self) -> int:
        return self.max_cubes * POSITION_DIM + self.max_cubes

    @property
    def cube_feature_size(self) -> int:
        return CUBE_FEATURE_DIM

    def validate_num_cubes(self, num_cubes: int) -> None:
        if not 1 <= num_cubes <= self.max_cubes:
            raise ValueError(
                f"num_cubes={num_cubes} is outside [1, {self.max_cubes}]"
            )

    def pack_observation(
        self, raw: Any, num_cubes: int, *, xp=np
    ) -> Any:
        """Convert an upstream grouped observation to masked cube slots."""
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
        for width in (
            POSITION_DIM,
            QUATERNION_DIM,
            LINEAR_VELOCITY_DIM,
            ANGULAR_VELOCITY_DIM,
        ):
            size = width * num_cubes
            groups.append(raw[..., cursor:cursor + size].reshape(
                raw.shape[:-1] + (num_cubes, width)
            ))
            cursor += size
        cubes = xp.concatenate(groups, axis=-1)
        select_action = raw[..., cursor:cursor + 1]
        centers = (
            (2.0 * xp.arange(num_cubes, dtype=raw.dtype) + 1.0)
            / float(num_cubes)
            - 1.0
        )
        selected_index = xp.argmin(
            xp.square(select_action - centers), axis=-1
        )
        selected = (
            selected_index[..., None] == xp.arange(num_cubes)
        ).astype(raw.dtype)
        cubes = xp.concatenate([cubes, selected[..., None]], axis=-1)
        pad = xp.zeros(
            raw.shape[:-1] + (self.max_cubes - num_cubes, CUBE_FEATURE_DIM),
            dtype=raw.dtype,
        )
        cubes = xp.concatenate([cubes, pad], axis=-2)
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
        return xp.concatenate(
            [cubes.reshape(raw.shape[:-1] + (-1,)), mask],
            axis=-1,
        )

    def unpack_observation(
        self, packed: Any, num_cubes: int, *, xp=np
    ) -> Any:
        """Invert :meth:`pack_observation` for calls into the raw env."""
        self.validate_num_cubes(num_cubes)
        if packed.shape[-1] != self.observation_size:
            raise ValueError(
                f"expected packed observation dim {self.observation_size}, "
                f"got {packed.shape[-1]}"
            )
        cube_end = self.max_cubes * CUBE_FEATURE_DIM
        cubes = packed[..., :cube_end].reshape(
            packed.shape[:-1] + (self.max_cubes, CUBE_FEATURE_DIM)
        )[..., :num_cubes, :]
        raw_cubes = cubes[..., :RAW_CUBE_FEATURE_DIM]
        offsets = np.cumsum((0, POSITION_DIM, QUATERNION_DIM,
                             LINEAR_VELOCITY_DIM))
        widths = (POSITION_DIM, QUATERNION_DIM,
                  LINEAR_VELOCITY_DIM, ANGULAR_VELOCITY_DIM)
        groups = [
            raw_cubes[..., offset:offset + width].reshape(
                packed.shape[:-1] + (-1,)
            )
            for offset, width in zip(offsets, widths)
        ]
        centers = (
            (2.0 * xp.arange(num_cubes, dtype=packed.dtype) + 1.0)
            / float(num_cubes)
            - 1.0
        )
        selected = cubes[..., -1]
        select_action = xp.sum(selected * centers, axis=-1, keepdims=True)
        return xp.concatenate([*groups, select_action], axis=-1)

    def pack_goal(self, raw: Any, num_cubes: int, *, xp=np) -> Any:
        self.validate_num_cubes(num_cubes)
        expected = POSITION_DIM * num_cubes
        if raw.shape[-1] != expected:
            raise ValueError(
                f"expected raw goal dim {expected}, got {raw.shape[-1]}"
            )
        pad = xp.zeros(
            raw.shape[:-1] + (POSITION_DIM * (self.max_cubes - num_cubes),),
            dtype=raw.dtype,
        )
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
        return xp.concatenate([raw, pad, mask], axis=-1)

    def unpack_goal(self, packed: Any, num_cubes: int) -> Any:
        self.validate_num_cubes(num_cubes)
        if packed.shape[-1] != self.goal_size:
            raise ValueError(
                f"expected packed goal dim {self.goal_size}, "
                f"got {packed.shape[-1]}"
            )
        return packed[..., :POSITION_DIM * num_cubes]

    def split_observation(self, packed: Any) -> Tuple[Any, Any]:
        """Return ``(cube_features, mask)``."""
        cube_end = self.max_cubes * CUBE_FEATURE_DIM
        mask_end = cube_end + self.max_cubes
        cubes = packed[..., :cube_end].reshape(
            packed.shape[:-1] + (self.max_cubes, CUBE_FEATURE_DIM)
        )
        return cubes, packed[..., cube_end:mask_end]

    def split_goal(self, packed: Any) -> Tuple[Any, Any]:
        """Return ``(cube_positions, mask)``."""
        position_end = self.max_cubes * POSITION_DIM
        positions = packed[..., :position_end].reshape(
            packed.shape[:-1] + (self.max_cubes, POSITION_DIM)
        )
        return positions, packed[..., position_end:]
