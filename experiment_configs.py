#!/usr/bin/env python3
"""Deterministic StableCRL-DCC experiment cells for NYU Torch HPC.

The active batch mirrors the SGCRL batch used for the continual DCC study:

* matched seeds 5/6/7;
* persistent actor and shared DCC groups, with masked dynamics on/off;
* repetition-12 StableCRL/CRTR probes on two long-horizon tasks.

The shell launcher consumes ``--setting`` output. Values are shell-quoted so
``eval \"$(python experiment_configs.py --setting N)\"`` is safe for every
value produced by this file.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
from typing import Any


SEEDS = (5, 6, 7)
DEFAULT_TASK_SEQUENCE = (
    "creative-1-task1,creative-1-task2,creative-2-task1,"
    "creative-3-task1,creative-4-task1"
)
_TASK_ID = re.compile(r"^creative-(\d+)-task(\d+)$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _cell(name: str, seed: int, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": name,
        "runner": "continual_dcc.py",
        "seed": seed,
        "task_sequence": DEFAULT_TASK_SEQUENCE,
        "carry_actor": True,
        "dcc_carry_shared": True,
        "dcc_dyn_weight": 1.0,
        "dcc_dyn_weight_after_task0": None,
        "dcc_combine_mode": "add",
        "dcc_goal_encoder_mode": "shared",
        "dcc_shared_width": 512,
        "dcc_shared_depth": 3,
        "dcc_task_width": 256,
        "dcc_task_depth": 4,
        "repetition_factor": 1,
        "max_cubes": 8,
        "use_pd": True,
        "pd_duration": 5,
        "wandb_group": name,
    }
    config.update(overrides)
    return config


def build_configs() -> list[dict[str, Any]]:
    """Return the active 12-run batch in stable array-index order."""
    configs = [
        *[
            _cell("dcc_persistent_actor_dynamics", seed)
            for seed in SEEDS
        ],
        *[
            _cell(
                "dcc_persistent_actor_no_dynamics",
                seed,
                dcc_dyn_weight=0.0,
            )
            for seed in SEEDS
        ],
        *[
            _cell(
                "dcc_crtr12_three_stack",
                seed,
                task_sequence="creative-3-task1",
                carry_actor=False,
                repetition_factor=12,
            )
            for seed in SEEDS
        ],
        *[
            _cell(
                "dcc_crtr12_four_stack",
                seed,
                task_sequence="creative-4-task1",
                carry_actor=False,
                repetition_factor=12,
            )
            for seed in SEEDS
        ],
    ]
    validate_configs(configs)
    return configs


def validate_configs(configs: list[dict[str, Any]]) -> None:
    """Reject ambiguous or shape-incompatible cells before submission."""
    required = {
        "name",
        "runner",
        "seed",
        "task_sequence",
        "carry_actor",
        "dcc_carry_shared",
        "dcc_dyn_weight",
        "dcc_combine_mode",
        "dcc_goal_encoder_mode",
        "repetition_factor",
        "max_cubes",
    }
    identities: set[tuple[str, int]] = set()
    for index, config in enumerate(configs):
        missing = sorted(required - config.keys())
        if missing:
            raise ValueError(f"config {index} is missing keys: {missing}")
        if not _NAME.fullmatch(str(config["name"])):
            raise ValueError(f"invalid config name: {config['name']!r}")
        if config["runner"] != "continual_dcc.py":
            raise ValueError(f"unsupported runner: {config['runner']!r}")
        if config["dcc_combine_mode"] not in {"add", "concat"}:
            raise ValueError("dcc_combine_mode must be add or concat")
        if config["dcc_goal_encoder_mode"] not in {"shared", "projected"}:
            raise ValueError(
                "dcc_goal_encoder_mode must be shared or projected"
            )
        if int(config["repetition_factor"]) < 1:
            raise ValueError("repetition_factor must be positive")

        task_ids = str(config["task_sequence"]).split(",")
        matches = [_TASK_ID.fullmatch(task_id) for task_id in task_ids]
        if not task_ids or any(match is None for match in matches):
            raise ValueError(
                f"config {index} has an invalid task sequence: {task_ids}"
            )
        largest_task = max(int(match.group(1)) for match in matches if match)
        if int(config["max_cubes"]) < largest_task:
            raise ValueError(
                f"config {index} max_cubes={config['max_cubes']} cannot fit "
                f"a {largest_task}-cube task"
            )

        identity = (str(config["name"]), int(config["seed"]))
        if identity in identities:
            raise ValueError(f"duplicate run identity: {identity}")
        identities.add(identity)


def _shell_value(value: Any) -> str:
    if value is None:
        value = ""
    elif isinstance(value, bool):
        value = "true" if value else "false"
    return shlex.quote(str(value))


def _print_list(configs: list[dict[str, Any]]) -> None:
    print(f"Total: {len(configs)} configurations")
    print(" idx  name                                  seed  dyn   rep  tasks")
    print("----  ------------------------------------  ----  ----  ---  -----")
    for index, config in enumerate(configs):
        task_count = len(config["task_sequence"].split(","))
        print(
            f"{index:>4}  {config['name']:<36}  {config['seed']:>4}  "
            f"{config['dcc_dyn_weight']:>4}  "
            f"{config['repetition_factor']:>3}  {task_count:>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate the StableCRL-DCC Torch HPC experiment batch."
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--setting", type=int)
    actions.add_argument("--json-setting", type=int)
    actions.add_argument("--total", action="store_true")
    actions.add_argument("--list", action="store_true")
    actions.add_argument("--array-max", action="store_true")
    parser.add_argument("--tasks-per-gpu", type=int, default=1)
    args = parser.parse_args()

    configs = build_configs()
    if args.tasks_per_gpu < 1:
        parser.error("--tasks-per-gpu must be positive")
    if args.total:
        print(len(configs))
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
