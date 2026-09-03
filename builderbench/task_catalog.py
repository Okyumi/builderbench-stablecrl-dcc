"""BuilderBench targets adapted to the StableCRL direct-cube simulator.

The source coordinates and masks below are copied from
``RajGhugare19/builderbench:builderbench/create_task_data.py``.  A single
rigid X translation moves the original robot workspace into the smaller
StableCRL direct-control workspace.  Rigid translation preserves the target
geometry exactly.

Only tasks used by continual Sequences A and B are included here.  Existing
``creative-N-taskM`` data and identifiers remain untouched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


SOURCE_REVISION = "RajGhugare19/builderbench@de9130b98323"
WORKSPACE_TRANSLATION = np.array([-0.20, 0.0, 0.0], dtype=np.float32)
TASK_ID_PATTERN = re.compile(r"^builderbench-direct-(\d+)-task(\d+)$")


@dataclass(frozen=True)
class DirectBuilderTask:
    """One original BuilderBench target in direct-cube coordinates."""

    num_cubes: int
    source_task_id: int
    name: str
    description: str
    source_starts: np.ndarray
    source_goals: np.ndarray
    goal_mask: np.ndarray
    ordered_reward: bool = False

    @property
    def env_id(self) -> str:
        return (
            f"builderbench-direct-{self.num_cubes}-"
            f"task{self.source_task_id}"
        )

    @property
    def starts(self) -> np.ndarray:
        return self.source_starts + WORKSPACE_TRANSLATION

    @property
    def goals(self) -> np.ndarray:
        return self.source_goals + WORKSPACE_TRANSLATION

    @property
    def start_bounds(self) -> np.ndarray:
        """Return the exact source starts in StableCRL min/max format."""
        return np.stack([self.starts, self.starts], axis=1)


def _points(*rows) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def _line_starts(num_cubes: int) -> np.ndarray:
    if num_cubes == 1:
        return _points((0.4, 0.0, 0.02))
    y = 0.08 * (np.arange(num_cubes, dtype=np.float32)
                - (num_cubes - 1) / 2)
    return np.stack(
        [np.full_like(y, 0.3), y, np.full_like(y, 0.02)], axis=-1
    ).astype(np.float32)


def _task(
    num_cubes: int,
    source_task_id: int,
    name: str,
    description: str,
    goals: np.ndarray,
    goal_mask,
    *,
    starts: np.ndarray | None = None,
    ordered_reward: bool = False,
) -> DirectBuilderTask:
    starts = _line_starts(num_cubes) if starts is None else starts
    goals = np.asarray(goals, dtype=np.float32)
    mask = np.asarray(goal_mask, dtype=bool)
    if starts.shape != (num_cubes, 3):
        raise ValueError(f"invalid starts for cube-{num_cubes}-task-{source_task_id}")
    if goals.shape != (num_cubes, 3):
        raise ValueError(f"invalid goals for cube-{num_cubes}-task-{source_task_id}")
    if mask.shape != (num_cubes,):
        raise ValueError(f"invalid mask for cube-{num_cubes}-task-{source_task_id}")
    return DirectBuilderTask(
        num_cubes=num_cubes,
        source_task_id=source_task_id,
        name=name,
        description=description,
        source_starts=np.asarray(starts, dtype=np.float32),
        source_goals=goals,
        goal_mask=mask,
        ordered_reward=ordered_reward,
    )


_TASKS = (
    _task(
        1,
        1,
        "place-cube",
        "Move one cube to another point on the floor.",
        _points((0.55, 0.0, 0.02)),
        [1],
        starts=_points((0.4, 0.0, 0.02)),
    ),
    _task(
        1,
        2,
        "pick-cube",
        "Lift one cube above the floor.",
        _points((0.55, 0.0, 0.10)),
        [1],
        starts=_points((0.4, 0.0, 0.02)),
    ),
    _task(
        2,
        1,
        "stack",
        "Put one cube on top of the other.",
        _points((0.45, 0.0, 0.02), (0.45, 0.0, 0.06)),
        [1, 1],
    ),
    _task(
        2,
        3,
        "permute",
        "Swap the two cubes' starting positions.",
        _points((0.45, 0.08, 0.02), (0.45, -0.08, 0.02)),
        [1, 1],
        starts=_points((0.45, -0.08, 0.02), (0.45, 0.08, 0.02)),
        ordered_reward=True,
    ),
    _task(
        3,
        1,
        "stack",
        "Build a three-cube vertical tower.",
        _points(
            (0.45, 0.0, 0.02),
            (0.45, 0.0, 0.06),
            (0.45, 0.0, 0.10),
        ),
        [1, 1, 1],
    ),
    _task(
        3,
        2,
        "t-stack",
        "Build a T shape with one base cube and two cubes above it.",
        _points(
            (0.45, 0.0, 0.02),
            (0.45, 0.02, 0.06),
            (0.45, -0.02, 0.06),
        ),
        [1, 1, 1],
    ),
    _task(
        3,
        5,
        "triangular-packing",
        "Pack three cubes into a tight triangle on the floor.",
        _points(
            (0.45, 0.04 / np.sqrt(2), 0.02),
            (0.45, -0.04 / np.sqrt(2), 0.02),
            (0.45 + 0.04 / np.sqrt(2), 0.0, 0.02),
        ),
        [1, 1, 1],
    ),
    _task(
        5,
        2,
        "pyramid",
        "Build a five-cube, three-level pyramid.",
        _points(
            (0.45, -0.035, 0.02),
            (0.45, 0.035, 0.02),
            (0.45, -0.02, 0.06),
            (0.45, 0.02, 0.06),
            (0.45, 0.0, 0.10),
        ),
        [1, 1, 1, 1, 1],
    ),
    _task(
        5,
        3,
        "archway",
        "Build two sides and bridge them with a top cube.",
        _points(
            (0.45, -0.03, 0.02),
            (0.45, 0.03, 0.02),
            (0.45, -0.03, 0.06),
            (0.45, 0.03, 0.06),
            (0.45, 0.0, 0.10),
        ),
        [1, 1, 1, 1, 1],
    ),
    _task(
        7,
        2,
        "2d-tokyo-tower",
        "Build a wide-bottom tower that narrows toward the top.",
        _points(
            (0.45, 0.0, 0.02),
            (0.45, -0.05, 0.02),
            (0.45, 0.05, 0.02),
            (0.45, -0.02, 0.06),
            (0.45, 0.02, 0.06),
            (0.45, 0.0, 0.10),
            (0.45, 0.0, 0.14),
        ),
        [0, 0, 1, 0, 1, 1, 1],
    ),
    _task(
        7,
        6,
        "jenga-tower",
        "Build an asymmetric Jenga-like tower.",
        _points(
            (0.45, 0.02, 0.02),
            (0.45, -0.02, 0.02),
            (0.45, 0.0, 0.06),
            (0.45, 0.045, 0.06),
            (0.45, 0.02, 0.10),
            (0.45, 0.02, 0.14),
            (0.45, 0.02, 0.18),
        ),
        [1, 1, 1, 1, 1, 1, 1],
    ),
    _task(
        7,
        7,
        "maximum-overhang",
        "Build an offset structure that stays balanced.",
        _points(
            (0.45, 0.0, 0.02),
            (0.45, 0.04 * (1 / 32), 0.06),
            (0.45, 0.04 * (1 / 32 + 1 / 16), 0.10),
            (0.45, 0.04 * (1 / 32 + 1 / 16 + 1 / 8), 0.14),
            (0.45, 0.04 * (1 / 4 + 1 / 8 + 1 / 16 + 1 / 32), 0.18),
            (0.45, 0.04 * (1 / 2 + 1 / 4 + 1 / 8 + 1 / 16 + 1 / 32 - 0.1), 0.22),
            (0.45, 0.16, 0.22),
        ),
        [1, 0, 0, 0, 0, 1, 1],
    ),
    _task(
        8,
        2,
        "vertical-portal-easy",
        "Build two vertical sides with an opening between them.",
        _points(
            (0.45, 0.02, 0.02),
            (0.45, -0.02, 0.02),
            (0.45, 0.045, 0.06),
            (0.45, -0.045, 0.06),
            (0.45, 0.02, 0.10),
            (0.45, -0.02, 0.10),
            (0.45, 0.075, 0.02),
            (0.45, -0.075, 0.02),
        ),
        [1, 1, 1, 1, 1, 1, 0, 0],
    ),
)

TASKS = {task.env_id: task for task in _TASKS}

SEQUENCE_A = (
    "builderbench-direct-1-task1",  # place cube
    "builderbench-direct-1-task2",  # lift cube
    "builderbench-direct-2-task1",  # two-cube stack
    "builderbench-direct-2-task3",  # permute
    "builderbench-direct-3-task2",  # T-stack
    "builderbench-direct-3-task5",  # triangular packing
    "builderbench-direct-5-task3",  # archway
    "builderbench-direct-7-task6",  # Jenga tower
    "builderbench-direct-8-task2",  # vertical portal
)

SEQUENCE_B = (
    "builderbench-direct-2-task1",  # two-cube stack
    "builderbench-direct-3-task1",  # three-cube stack
    "builderbench-direct-3-task2",  # T-stack
    "builderbench-direct-5-task2",  # pyramid
    "builderbench-direct-5-task3",  # archway
    "builderbench-direct-7-task2",  # Tokyo tower
    "builderbench-direct-7-task7",  # maximum overhang
    "builderbench-direct-8-task2",  # vertical portal
)


def get_direct_builder_task(env_id: str) -> DirectBuilderTask:
    try:
        return TASKS[env_id]
    except KeyError as error:
        raise ValueError(f"unknown direct BuilderBench task: {env_id!r}") from error


def is_direct_builder_task(env_id: str) -> bool:
    return env_id in TASKS
