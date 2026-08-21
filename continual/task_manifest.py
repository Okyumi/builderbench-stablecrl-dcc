"""Stable task identities for continual BuilderBench experiments."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


_CREATIVE_ID = re.compile(
    r"^(?:(?:sparse|truncated-reward)-)?creative-(\d+)-task(\d+)$"
)


def canonical_goal(goal: np.ndarray, grid_size: float = 0.04) -> np.ndarray:
    """Canonicalize modulo cube permutation and horizontal translation.

    Height is intentionally anchored to the ground plane: translating in Z
    would incorrectly make a one-cube ``place`` and ``pick`` task identical.
    """
    points = np.asarray(goal, dtype=np.float64).reshape(-1, 3)
    quantized = np.rint(points / grid_size).astype(np.int64)
    order = np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))
    quantized = quantized[order]
    horizontal_origin = np.array(
        [quantized[0, 0], quantized[0, 1], 0], dtype=np.int64
    )
    quantized = quantized - horizontal_origin
    order = np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))
    return quantized[order]


def canonical_goal_hash(goal: np.ndarray, grid_size: float = 0.04) -> str:
    canonical = canonical_goal(goal, grid_size=grid_size)
    return hashlib.sha256(canonical.tobytes()).hexdigest()[:16]


@dataclass(frozen=True)
class TaskRecord:
    env_id: str
    num_cubes: int
    local_task_index: int
    goal_hash: str
    global_id: str
    canonical_goal: tuple[tuple[int, int, int], ...]
    task_data_version: str


def task_record(
    env_id: str,
    goal: np.ndarray,
    *,
    task_data_version: str,
    grid_size: float = 0.04,
) -> TaskRecord:
    match = _CREATIVE_ID.fullmatch(env_id)
    if match is None:
        raise ValueError(f"unsupported continual task id: {env_id!r}")
    num_cubes, local_task_index = map(int, match.groups())
    canonical = canonical_goal(goal, grid_size=grid_size)
    if len(canonical) != num_cubes:
        raise ValueError(
            f"{env_id} declares {num_cubes} cubes but goal has "
            f"{len(canonical)}"
        )
    digest = canonical_goal_hash(goal, grid_size=grid_size)
    global_id = (
        f"{task_data_version}:{num_cubes}:{local_task_index}:{digest}"
    )
    return TaskRecord(
        env_id=env_id,
        num_cubes=num_cubes,
        local_task_index=local_task_index,
        goal_hash=digest,
        global_id=global_id,
        canonical_goal=tuple(tuple(map(int, row)) for row in canonical),
        task_data_version=task_data_version,
    )


def build_manifest(
    tasks: Iterable[tuple[str, np.ndarray]],
    *,
    task_data_version: str,
    grid_size: float = 0.04,
) -> list[TaskRecord]:
    records = [
        task_record(
            env_id,
            goal,
            task_data_version=task_data_version,
            grid_size=grid_size,
        )
        for env_id, goal in tasks
    ]
    ids = [record.global_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("continual sequence contains duplicate task identities")
    semantic_ids = [(record.num_cubes, record.goal_hash) for record in records]
    if len(semantic_ids) != len(set(semantic_ids)):
        raise ValueError("continual sequence contains duplicate semantic goals")
    return records


def write_manifest(records: Sequence[TaskRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "tasks": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
