#!/usr/bin/env python3
"""Matched continual configurations for diverse BuilderBench Sequence A."""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from builderbench.task_catalog import SEQUENCE_A
from experiment_configs import SEEDS, _cell, _shell_value, validate_configs


SEQUENCE_A_TASKS = ",".join(SEQUENCE_A)
TASK_DATA_VERSION = "builderbench-de9130-direct-v1"
STAGES = {
    "smoke": (0, 2),
    "sequence_a": (0, 8),
    "all": (0, 8),
}


def build_configs() -> list[dict[str, Any]]:
    """Return reset, persistent, and DCC comparisons for Sequence A."""
    configs: list[dict[str, Any]] = []
    for seed in SEEDS:
        configs.extend([
            _cell(
                "stablecrl_reset_reset_diverse_sequence_a",
                seed,
                runner="continual_crl.py",
                task_sequence=SEQUENCE_A_TASKS,
                actor_lifecycle="reset",
                critic_lifecycle="reset",
                observation_layout="semantic",
                vanilla_network_type="flat_upstream",
                max_cubes=8,
                repetition_factor=12,
                task_data_version=TASK_DATA_VERSION,
            ),
            _cell(
                "stablecrl_persistent_persistent_diverse_sequence_a",
                seed,
                runner="continual_crl.py",
                task_sequence=SEQUENCE_A_TASKS,
                actor_lifecycle="persistent",
                critic_lifecycle="persistent",
                observation_layout="semantic",
                vanilla_network_type="flat_upstream",
                max_cubes=8,
                repetition_factor=12,
                task_data_version=TASK_DATA_VERSION,
            ),
            _cell(
                "dcc_residual_diverse_sequence_a",
                seed,
                runner="continual_dcc.py",
                task_sequence=SEQUENCE_A_TASKS,
                carry_actor=True,
                dcc_carry_shared=True,
                observation_layout="semantic",
                max_cubes=8,
                repetition_factor=12,
                task_data_version=TASK_DATA_VERSION,
            ),
        ])
    validate_configs(configs)
    return configs


def _print_list(configs: list[dict[str, Any]]) -> None:
    print(f"Total: {len(configs)} diverse Sequence A configurations")
    print(" idx  name                                                    seed")
    print("----  ------------------------------------------------------  ----")
    for index, config in enumerate(configs):
        print(f"{index:>4}  {config['name']:<54}  {config['seed']:>4}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate diverse BuilderBench Sequence A experiments."
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
    if args.array_max:
        print(math.ceil(len(configs) / args.tasks_per_gpu) - 1)
        return
    if args.list:
        _print_list(configs)
        return
    for action in ("stage_start", "stage_end", "stage_total"):
        stage = getattr(args, action)
        if stage is None:
            continue
        start, end = STAGES[stage]
        values = {
            "stage_start": start,
            "stage_end": end,
            "stage_total": end - start + 1,
        }
        print(values[action])
        return
    if args.stage_array_max is not None:
        start, end = STAGES[args.stage_array_max]
        print(math.ceil((end - start + 1) / args.tasks_per_gpu) - 1)
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
