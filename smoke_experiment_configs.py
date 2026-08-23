#!/usr/bin/env python3
"""Deterministic residual-DCC smoke cells for NYU Torch.

This is deliberately separate from the stable production index registry in
``experiment_configs.py``.  The three cells compile the exact production
network shapes while using short launcher-supplied training budgets.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from experiment_configs import _cell, _shell_value, validate_configs


STAGES = {"smoke": (0, 2)}


def build_configs() -> list[dict]:
    """Return single-task parity and two-task transfer smoke cells."""
    configs = [
        _cell(
            "smoke_dcc_residual_three_stack",
            5,
            task_sequence="creative-3-task1",
            carry_actor=False,
            max_cubes=4,
            repetition_factor=12,
            continual_eval_repeats=2,
        ),
        _cell(
            "smoke_dcc_residual_four_stack",
            5,
            task_sequence="creative-4-task1",
            carry_actor=False,
            max_cubes=4,
            repetition_factor=12,
            continual_eval_repeats=2,
        ),
        _cell(
            "smoke_dcc_residual_two_task_transfer",
            5,
            task_sequence="creative-1-task1,creative-2-task1",
            carry_actor=True,
            dcc_carry_shared=True,
            max_cubes=4,
            repetition_factor=1,
            continual_eval_repeats=2,
        ),
    ]
    validate_configs(configs)
    return configs


def _print_list(configs: list[dict]) -> None:
    print(f"Total: {len(configs)} residual-DCC smoke configurations")
    print(" idx  name                                         seed  tasks  CRTR")
    print("----  -------------------------------------------  ----  -----  ----")
    for index, config in enumerate(configs):
        task_count = len(config["task_sequence"].split(","))
        print(
            f"{index:>4}  {config['name']:<43}  {config['seed']:>4}  "
            f"{task_count:>5}  {config['repetition_factor']:>4}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate the residual-DCC Torch smoke batch."
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
