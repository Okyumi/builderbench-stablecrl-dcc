#!/usr/bin/env python3
"""Matched continual StableCRL and residual-DCC experiments for Torch.

Every method uses the same semantic wrapper, flat residual backbone, CRTR-12
objective, per-task budget, and seeds. The only intended difference is the
parameter lifecycle or DCC decomposition.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from experiment_configs import (
    SEEDS,
    _cell,
    _shell_value,
    validate_configs,
)


GOAL_ONLY_SEQUENCE = "creative-1-task1,creative-1-task2"
EXPANDING_4STACK_SEQUENCE = (
    "creative-1-task1,creative-2-task1,"
    "creative-3-task1,creative-4-task1"
)
EXPANDING_5STACK_SEQUENCE = (
    f"{EXPANDING_4STACK_SEQUENCE},creative-5-task1"
)

STAGES = {
    "smoke_goal": (0, 2),
    "goal_only": (0, 8),
    "smoke_expanding_4stack": (9, 11),
    "expanding_4stack": (9, 17),
    "core": (0, 17),
    "smoke_expanding_5stack": (18, 20),
    "expanding_5stack": (18, 26),
    "all": (0, 26),
}


def _track_configs(
    track_name: str,
    task_sequence: str,
    max_cubes: int,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for seed in SEEDS:
        configs.extend([
            _cell(
                f"stablecrl_reset_reset_{track_name}",
                seed,
                runner="continual_crl.py",
                task_sequence=task_sequence,
                actor_lifecycle="reset",
                critic_lifecycle="reset",
                observation_layout="semantic",
                vanilla_network_type="flat_upstream",
                max_cubes=max_cubes,
                repetition_factor=12,
            ),
            _cell(
                f"stablecrl_persistent_persistent_{track_name}",
                seed,
                runner="continual_crl.py",
                task_sequence=task_sequence,
                actor_lifecycle="persistent",
                critic_lifecycle="persistent",
                observation_layout="semantic",
                vanilla_network_type="flat_upstream",
                max_cubes=max_cubes,
                repetition_factor=12,
            ),
            _cell(
                f"dcc_residual_{track_name}",
                seed,
                task_sequence=task_sequence,
                carry_actor=True,
                dcc_carry_shared=True,
                max_cubes=max_cubes,
                repetition_factor=12,
            ),
        ])
    return configs


def build_configs() -> list[dict[str, Any]]:
    """Return core and optional five-stack continual comparisons."""
    configs = [
        *_track_configs("goal_only", GOAL_ONLY_SEQUENCE, 1),
        *_track_configs(
            "expanding_4stack",
            EXPANDING_4STACK_SEQUENCE,
            4,
        ),
        *_track_configs(
            "expanding_5stack",
            EXPANDING_5STACK_SEQUENCE,
            5,
        ),
    ]
    validate_configs(configs)
    return configs


def _print_list(configs: list[dict[str, Any]]) -> None:
    print(f"Total: {len(configs)} matched continual configurations")
    print(" idx  name                                             seed  tasks")
    print("----  -----------------------------------------------  ----  -----")
    for index, config in enumerate(configs):
        task_count = len(config["task_sequence"].split(","))
        print(
            f"{index:>4}  {config['name']:<47}  "
            f"{config['seed']:>4}  {task_count:>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate the matched continual Torch experiment batch."
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--setting", type=int)
    actions.add_argument("--json-setting", type=int)
    actions.add_argument("--total", action="store_true")
    actions.add_argument("--list", action="store_true")
    actions.add_argument("--array-max", action="store_true")
    actions.add_argument("--stage-start", choices=tuple(STAGES))
    actions.add_argument("--stage-end", choices=tuple(STAGES))
    actions.add_argument("--stage-total", choices=tuple(STAGES))
    actions.add_argument("--stage-array-max", choices=tuple(STAGES))
    parser.add_argument("--tasks-per-gpu", type=int, default=1)
    args = parser.parse_args()

    configs = build_configs()
    if args.tasks_per_gpu < 1:
        parser.error("--tasks-per-gpu must be positive")
    if args.total:
        print(len(configs))
        return
    if args.stage_start is not None:
        print(STAGES[args.stage_start][0])
        return
    if args.stage_end is not None:
        print(STAGES[args.stage_end][1])
        return
    if args.stage_total is not None:
        start, end = STAGES[args.stage_total]
        print(end - start + 1)
        return
    if args.stage_array_max is not None:
        start, end = STAGES[args.stage_array_max]
        print(math.ceil((end - start + 1) / args.tasks_per_gpu) - 1)
        return
    if args.array_max:
        print(math.ceil(len(configs) / args.tasks_per_gpu) - 1)
        return
    if args.list:
        _print_list(configs)
        return

    index = args.setting if args.setting is not None else args.json_setting
    assert index is not None
    if not 0 <= index < len(configs):
        print(
            f"ERROR: setting {index} out of range [0, {len(configs) - 1}]",
            file=sys.stderr,
        )
        raise SystemExit(1)
    config = configs[index]
    if args.json_setting is not None:
        print(json.dumps(config, sort_keys=True))
        return
    for key, value in config.items():
        print(f"{key.upper()}={_shell_value(value)}")


if __name__ == "__main__":
    main()
