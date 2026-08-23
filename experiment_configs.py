#!/usr/bin/env python3
"""Deterministic BuilderBench StableCRL/DCC cells for NYU Torch.

Array order is part of the protocol. Indices 0--35 retain their original
meaning. Indices 36--47 isolate padding, semantic transformation, and network
architecture. Indices 48--53 test right-sized set capacity. Indices 54--65
provide goal-only and expanding-capacity
continual protocol cells, gated on the diagnostic results.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
from collections import Counter
from typing import Any


SEEDS = (5, 6, 7)
DEFAULT_TASK_SEQUENCE = (
    "creative-1-task1,creative-1-task2,creative-2-task1,"
    "creative-3-task1,creative-4-task1"
)
INDIVIDUAL_TASKS = (
    ("three_stack", "creative-3-task1"),
    ("four_stack", "creative-4-task1"),
)
BASELINE_FIRST_END = 23
STAGES = {
    "replication": (0, 17),
    "continual_baselines": (18, 35),
    "padding_diagnostics": (36, 53),
    "protocol": (54, 65),
    "legacy_all": (0, 35),
}
_TASK_ID = re.compile(r"^creative-(\d+)-task(\d+)$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _cell(
    name: str,
    seed: int,
    *,
    runner: str = "continual_dcc.py",
    task_sequence: str = DEFAULT_TASK_SEQUENCE,
    **overrides: Any,
) -> dict[str, Any]:
    task_ids = task_sequence.split(",")
    config: dict[str, Any] = {
        "name": name,
        "runner": runner,
        "seed": seed,
        "env_id": task_ids[0] if len(task_ids) == 1 else "",
        "task_sequence": task_sequence,
        "actor_lifecycle": "persistent",
        "critic_lifecycle": "persistent",
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
        "vanilla_width": 512,
        "vanilla_depth": 3,
        "observation_layout": "semantic",
        "vanilla_network_type": "set",
        "architecture": "block",
        "num_blocks": 8,
        "hidden_dim": 1024,
        "repetition_factor": 1,
        "max_cubes": 8,
        "use_pd": True,
        "pd_duration": 5,
        "wandb_group": name,
    }
    config.update(overrides)
    return config


def build_configs() -> list[dict[str, Any]]:
    """Return the 66-run staged batch in stable global index order."""
    configs: list[dict[str, Any]] = []

    # 0--5: upstream algorithm on two paper-style cells.
    for task_name, env_id in INDIVIDUAL_TASKS:
        configs.extend(
            _cell(
                f"upstream_scaled_crtr_{task_name}",
                seed,
                runner="stable_crl.py",
                task_sequence=env_id,
                repetition_factor=12,
            )
            for seed in SEEDS
        )

    # 6--11: same tasks through semantic padding + set networks, no DCC.
    for task_name, env_id in INDIVIDUAL_TASKS:
        configs.extend(
            _cell(
                f"wrapped_vanilla_crl_{task_name}",
                seed,
                runner="continual_crl.py",
                task_sequence=env_id,
                actor_lifecycle="reset",
                critic_lifecycle="reset",
                repetition_factor=12,
            )
            for seed in SEEDS
        )

    # 12--17: one-task DCC controls on the same two tasks.
    for task_name, env_id in INDIVIDUAL_TASKS:
        configs.extend(
            _cell(
                f"dcc_single_{task_name}",
                seed,
                task_sequence=env_id,
                carry_actor=False,
                repetition_factor=12,
            )
            for seed in SEEDS
        )

    # 18--23: requested vanilla continual lifecycle baselines.
    configs.extend(
        _cell(
            "crl_reset_reset",
            seed,
            runner="continual_crl.py",
            actor_lifecycle="reset",
            critic_lifecycle="reset",
        )
        for seed in SEEDS
    )

    configs.extend(
        _cell(
            "crl_persistent_persistent",
            seed,
            runner="continual_crl.py",
            actor_lifecycle="persistent",
            critic_lifecycle="persistent",
        )
        for seed in SEEDS
    )

    # 24--35: existing SGCRL-style continual DCC and CRTR controls.
    configs.extend(
        _cell("dcc_persistent_actor_dynamics", seed)
        for seed in SEEDS
    )
    configs.extend(
        _cell(
            "dcc_persistent_actor_no_dynamics",
            seed,
            dcc_dyn_weight=0.0,
        )
        for seed in SEEDS
    )
    configs.extend(
        _cell(
            "dcc_crtr12_three_stack",
            seed,
            task_sequence="creative-3-task1",
            carry_actor=False,
            repetition_factor=12,
        )
        for seed in SEEDS
    )
    configs.extend(
        _cell(
            "dcc_crtr12_four_stack",
            seed,
            task_sequence="creative-4-task1",
            carry_actor=False,
            repetition_factor=12,
        )
        for seed in SEEDS
    )

    # 36--41: padding-only wrapper + the upstream residual architecture.
    for task_name, env_id in INDIVIDUAL_TASKS:
        configs.extend(
            _cell(
                f"grouped_pad_upstream_{task_name}",
                seed,
                runner="continual_crl.py",
                task_sequence=env_id,
                actor_lifecycle="reset",
                critic_lifecycle="reset",
                observation_layout="grouped",
                vanilla_network_type="flat_upstream",
                max_cubes=4,
                repetition_factor=12,
            )
            for seed in SEEDS
        )

    # 42--47: semantic slot layout + the same upstream residual architecture.
    for task_name, env_id in INDIVIDUAL_TASKS:
        configs.extend(
            _cell(
                f"semantic_pad_upstream_{task_name}",
                seed,
                runner="continual_crl.py",
                task_sequence=env_id,
                actor_lifecycle="reset",
                critic_lifecycle="reset",
                observation_layout="semantic",
                vanilla_network_type="flat_upstream",
                max_cubes=4,
                repetition_factor=12,
            )
            for seed in SEEDS
        )

    # 48--53: isolate oversized capacity in the original set-network control.
    for task_name, env_id in INDIVIDUAL_TASKS:
        configs.extend(
            _cell(
                f"semantic_set_capacity4_{task_name}",
                seed,
                runner="continual_crl.py",
                task_sequence=env_id,
                actor_lifecycle="reset",
                critic_lifecycle="reset",
                observation_layout="semantic",
                vanilla_network_type="set",
                max_cubes=4,
                repetition_factor=12,
            )
            for seed in SEEDS
        )

    # 54--65: protocol separation; run only after indices 36--53 pass.
    configs.extend(
        _cell(
            "flat_crl_goal_only_1cube",
            seed,
            runner="continual_crl.py",
            task_sequence="creative-1-task1,creative-1-task2",
            observation_layout="semantic",
            vanilla_network_type="flat_upstream",
            max_cubes=1,
        )
        for seed in SEEDS
    )
    configs.extend(
        _cell(
            "flat_crl_expanding_stack",
            seed,
            runner="continual_crl.py",
            task_sequence=(
                "creative-1-task1,creative-2-task1,"
                "creative-3-task1,creative-4-task1"
            ),
            observation_layout="semantic",
            vanilla_network_type="flat_upstream",
            max_cubes=4,
        )
        for seed in SEEDS
    )
    configs.extend(
        _cell(
            "dcc_goal_only_1cube",
            seed,
            task_sequence="creative-1-task1,creative-1-task2",
            max_cubes=1,
        )
        for seed in SEEDS
    )
    configs.extend(
        _cell(
            "dcc_expanding_stack",
            seed,
            task_sequence=(
                "creative-1-task1,creative-2-task1,"
                "creative-3-task1,creative-4-task1"
            ),
            max_cubes=4,
        )
        for seed in SEEDS
    )
    validate_configs(configs)
    return configs


def validate_configs(configs: list[dict[str, Any]]) -> None:
    """Reject ambiguous or shape-incompatible cells before submission."""
    required = {
        "name",
        "runner",
        "seed",
        "env_id",
        "task_sequence",
        "actor_lifecycle",
        "critic_lifecycle",
        "carry_actor",
        "dcc_carry_shared",
        "dcc_dyn_weight",
        "dcc_combine_mode",
        "dcc_goal_encoder_mode",
        "observation_layout",
        "vanilla_network_type",
        "repetition_factor",
        "max_cubes",
    }
    allowed_runners = {
        "stable_crl.py",
        "continual_crl.py",
        "continual_dcc.py",
    }
    identities: set[tuple[str, int]] = set()
    for index, config in enumerate(configs):
        missing = sorted(required - config.keys())
        if missing:
            raise ValueError(f"config {index} is missing keys: {missing}")
        if not _NAME.fullmatch(str(config["name"])):
            raise ValueError(f"invalid config name: {config['name']!r}")
        if config["runner"] not in allowed_runners:
            raise ValueError(f"unsupported runner: {config['runner']!r}")
        if config["actor_lifecycle"] not in {"reset", "persistent"}:
            raise ValueError("actor_lifecycle must be reset or persistent")
        if config["critic_lifecycle"] not in {"reset", "persistent"}:
            raise ValueError("critic_lifecycle must be reset or persistent")
        if config["dcc_combine_mode"] not in {"add", "concat"}:
            raise ValueError("dcc_combine_mode must be add or concat")
        if config["dcc_goal_encoder_mode"] not in {"shared", "projected"}:
            raise ValueError(
                "dcc_goal_encoder_mode must be shared or projected"
            )
        if config["observation_layout"] not in {"semantic", "grouped"}:
            raise ValueError("observation_layout must be semantic or grouped")
        if config["vanilla_network_type"] not in {"set", "flat_upstream"}:
            raise ValueError(
                "vanilla_network_type must be set or flat_upstream"
            )
        if (
            config["vanilla_network_type"] == "set"
            and config["observation_layout"] != "semantic"
        ):
            raise ValueError("set networks require the semantic layout")
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
        if config["runner"] == "stable_crl.py":
            if len(task_ids) != 1 or config["env_id"] != task_ids[0]:
                raise ValueError("upstream runs must select exactly one env_id")

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
    counts = Counter(config["runner"] for config in configs)
    print(
        f"Total: {len(configs)} configurations; baseline-first: "
        f"0-{BASELINE_FIRST_END}; runners={dict(counts)}"
    )
    print(" idx  name                                  seed  runner            tasks")
    print("----  ------------------------------------  ----  ----------------  -----")
    for index, config in enumerate(configs):
        task_count = len(config["task_sequence"].split(","))
        print(
            f"{index:>4}  {config['name']:<36}  {config['seed']:>4}  "
            f"{config['runner']:<16}  {task_count:>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate the StableCRL/DCC Torch HPC experiment batch."
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
