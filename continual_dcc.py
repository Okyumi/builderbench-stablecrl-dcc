"""Sequential DCC training on the StableCRL MJX BuilderBench tasks.

Each phase creates a fresh replay buffer and task-specific residual DCC
adapter while carrying the upstream residual actor, shared state-action
encoder, and shared goal encoder. The actor can be reset as an ablation.
There is no dynamics head. Boundary checkpoints contain data only and are
written atomically.
"""
from __future__ import annotations

import os
import pickle
import re
import json
import functools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import jax
import numpy as np
import tyro

from continual.task_manifest import build_manifest, write_manifest
from continual.dcc_networks import make_dcc_networks
from continual.eval_logging import (
    log_continual_eval_to_wandb,
    read_eval_rows,
    write_phase_rows,
)
from continual.evaluation import run_repeated_evaluation
from continual.semantic_layout import SemanticLayout
from continual.semantic_wrapper import SemanticPadWrapper
from builderbench.env_utils import make_env
from stable_crl_dcc import Args as StableCRLArgs
from stable_crl_dcc import make_inference_fn
from stable_crl_dcc import main as train_task
from utils.evaluation import Evaluator
from utils.wrapper import PDWrapper, wrap_env


_TASK_ID = re.compile(r"^creative-(\d+)-task(\d+)$")
_DIRECT_TASK_ID = re.compile(r"^builderbench-direct-(\d+)-task(\d+)$")


@dataclass
class Args(StableCRLArgs):
    task_sequence: str = (
        "creative-1-task1,creative-1-task2,creative-2-task1,"
        "creative-3-task1,creative-4-task1"
    )
    base_steps: int = 200_000_000
    steps_per_task: int = 200_000_000
    carry_actor: bool = True
    boundary_checkpoint_dir: str = "checkpoints/continual_dcc"
    task_data_version: str = "david-6e8d56d"
    resume: bool = True
    critic_family: Literal["dcc"] = "dcc"
    eval_next_task: bool = True
    eval_previous_tasks: bool = True
    report_retention_metrics: bool = True
    log_continual_eval: bool = True
    wandb_eval_tables: bool = True
    continual_eval_repeats: int = 5


def _task_for_env_id(
    env_id: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if _DIRECT_TASK_ID.fullmatch(env_id):
        from builderbench.task_catalog import get_direct_builder_task

        task = get_direct_builder_task(env_id)
        return task.goals, task.goal_mask

    match = _TASK_ID.fullmatch(env_id)
    if match is None:
        raise ValueError(
            "continual_dcc supports creative-N-taskM and "
            "builderbench-direct-N-taskM ids; "
            f"got {env_id!r}"
        )
    num_cubes, one_based_task = map(int, match.groups())
    path = Path(__file__).parent / "builderbench" / "tasks" / (
        f"creative-{num_cubes}.npz"
    )
    with np.load(path) as data:
        goals = data["goals"]
    index = one_based_task - 1
    if not 0 <= index < len(goals):
        raise ValueError(
            f"{env_id} selects task {one_based_task}, but {path.name} has "
            f"{len(goals)} task(s)"
        )
    return goals[index], None


def _goal_for_env_id(env_id: str) -> np.ndarray:
    """Return the unpadded goal for manifest and compatibility callers."""
    return _task_for_env_id(env_id)[0]


def _checkpoint_path(directory: Path, task_index: int) -> Path:
    return directory / f"task_{task_index:02d}.pkl"


def _checkpoint_recipe(args: Args) -> dict:
    """Training semantics that must remain fixed across a resumed run."""
    names = (
        "seed",
        "task_sequence",
        "task_data_version",
        "base_steps",
        "steps_per_task",
        "num_envs",
        "num_eval_envs",
        "continual_eval_repeats",
        "rollout_length",
        "num_eval_steps",
        "num_reset_steps",
        "env_early_termination",
        "env_episode_length",
        "permutation_invariant_reward",
        "actor_learning_rate",
        "critic_learning_rate",
        "discount",
        "entropy_cost",
        "entropy_cost_final",
        "entropy_decay_fraction",
        "logsumexp_cost",
        "rep_size",
        "max_replay_size",
        "min_replay_size",
        "repetition_factor",
        "use_pd",
        "pd_duration",
        "mjx_impl",
        "max_cubes",
        "dcc_task_width",
        "dcc_task_depth",
        "dcc_combine_mode",
        "dcc_goal_encoder_mode",
        "dcc_carry_shared",
        "carry_actor",
        "architecture",
        "num_blocks",
        "hidden_dim",
        "scale_actor_residual_by_depth",
        "scale_critic_residual_by_depth",
        "use_non_residual_critic_encoders",
    )
    recipe = {
        "algorithm": "stablecrl-dcc-flat-residual-no-dynamics-v3",
        **{name: getattr(args, name) for name in names},
    }
    if not (
        args.eval_next_task
        and args.eval_previous_tasks
        and args.report_retention_metrics
    ):
        recipe.update({
            "eval_next_task": args.eval_next_task,
            "eval_previous_tasks": args.eval_previous_tasks,
            "report_retention_metrics": args.report_retention_metrics,
        })
    return recipe


def _save_boundary_checkpoint(
    path: Path,
    *,
    carry: dict,
    global_id: str,
    task_index: int,
    recipe: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "task_index": task_index,
        "global_id": global_id,
        "recipe": recipe,
        "carry": jax.tree_util.tree_map(np.asarray, carry),
    }
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_boundary_checkpoint(
    path: Path, *, global_id: str, task_index: int, recipe: dict
) -> dict:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("schema_version") != 2:
        raise ValueError(f"unsupported checkpoint schema in {path}")
    if payload.get("task_index") != task_index:
        raise ValueError(f"task index mismatch in {path}")
    if payload.get("global_id") != global_id:
        raise ValueError(
            f"semantic task mismatch in {path}: expected {global_id}, "
            f"got {payload.get('global_id')}"
        )
    if payload.get("recipe") != recipe:
        raise ValueError(
            f"training recipe mismatch in {path}; use a fresh checkpoint "
            "directory when changing a DCC/StableCRL ablation"
        )
    return payload["carry"]


def _resume_prefix(
    directory: Path, records, recipe: dict
) -> tuple[int, dict | None]:
    carry = None
    next_task = 0
    for index, record in enumerate(records):
        path = _checkpoint_path(directory, index)
        if not path.is_file() or path.stat().st_size == 0:
            break
        try:
            carry = _load_boundary_checkpoint(
                path,
                global_id=record.global_id,
                task_index=index,
                recipe=recipe,
            )
        except (EOFError, OSError, pickle.UnpicklingError):
            break
        next_task = index + 1
    return next_task, carry


def _evaluate_seen_tasks(args, records, carry, phase_index):
    """Evaluate stored heads plus next-task zero-shot transfer."""
    results = []
    layout = SemanticLayout(max_cubes=args.max_cubes)
    eval_indices = (
        list(range(phase_index + 1))
        if args.eval_previous_tasks
        else [phase_index]
    )
    if args.eval_next_task and phase_index + 1 < len(records):
        eval_indices.append(phase_index + 1)
    for eval_index in eval_indices:
        record = records[eval_index]
        eval_args = replace(args, env_id=record.env_id)
        env_class, config = make_env(eval_args)
        config.impl = args.mjx_impl
        raw_env = SemanticPadWrapper(
            env_class(config=config),
            num_cubes=config.num_cubes,
            max_cubes=args.max_cubes,
        )
        if args.use_pd:
            episode_length = config.episode_length // args.pd_duration
            env = wrap_env(
                PDWrapper(raw_env, duration=args.pd_duration), episode_length
            )
        else:
            episode_length = config.episode_length
            env = wrap_env(raw_env, episode_length)

        dcc = make_dcc_networks(
            layout=layout,
            action_size=env.action_size,
            rep_size=args.rep_size,
            architecture=args.architecture,
            num_blocks=args.num_blocks,
            hidden_dim=args.hidden_dim,
            scale_actor_residual_by_depth=(
                args.scale_actor_residual_by_depth
            ),
            scale_critic_residual_by_depth=(
                args.scale_critic_residual_by_depth
            ),
            use_non_residual_critic_encoders=(
                args.use_non_residual_critic_encoders
            ),
            task_width=args.dcc_task_width,
            task_depth=args.dcc_task_depth,
            combine_mode=args.dcc_combine_mode,
            goal_encoder_mode=args.dcc_goal_encoder_mode,
        )
        critic = dict(carry["critic_shared"])
        head_index = min(eval_index, phase_index)
        critic["phi_task"] = carry["task_bank"][head_index]["phi_task"]
        evaluator = Evaluator(
            env,
            functools.partial(make_inference_fn(dcc), deterministic=True),
            num_eval_envs=args.num_eval_envs,
            episode_length=episode_length,
            key=jax.random.PRNGKey(
                args.seed + 10_000 * phase_index + eval_index
            ),
        )
        metrics = run_repeated_evaluation(
            evaluator,
            policy_params={"actor": carry["actor"], "critic": critic},
            repeats=args.continual_eval_repeats,
            num_eval_envs=args.num_eval_envs,
        )
        row = {
            "phase_index": phase_index,
            "eval_task_index": eval_index,
            "eval_scope": (
                "seen" if eval_index <= phase_index else "next_unseen"
            ),
            "critic_head_task_index": head_index,
            "train_task_global_id": records[phase_index].global_id,
            "eval_task_global_id": record.global_id,
            **{
                key: float(np.asarray(value))
                for key, value in metrics.items()
            },
        }
        if eval_index == phase_index:
            row.update(carry["task_adaptation"][phase_index])
        results.append(row)
    return results


def main(args: Args) -> None:
    if args.continual_eval_repeats < 1:
        raise ValueError("continual_eval_repeats must be positive")
    task_ids = [item.strip() for item in args.task_sequence.split(",")]
    if not task_ids or any(not item for item in task_ids):
        raise ValueError("task_sequence must contain comma-separated task ids")
    tasks = []
    for env_id in task_ids:
        goal, goal_mask = _task_for_env_id(env_id)
        tasks.append(
            (env_id, goal)
            if goal_mask is None
            else (env_id, goal, goal_mask)
        )
    records = build_manifest(
        tasks, task_data_version=args.task_data_version
    )
    checkpoint_dir = Path(args.boundary_checkpoint_dir)
    recipe = _checkpoint_recipe(args)
    recipe_path = checkpoint_dir / "run_recipe.json"
    if recipe_path.is_file():
        existing_recipe = json.loads(recipe_path.read_text())
        if existing_recipe != recipe:
            raise ValueError(
                f"training recipe mismatch in {recipe_path}; use a fresh "
                "checkpoint directory when changing an ablation"
            )
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            json.dumps(recipe, indent=2, sort_keys=True) + "\n"
        )
    write_manifest(records, checkpoint_dir / "task_manifest.json")

    start_task, carry = (0, None)
    if args.resume:
        start_task, carry = _resume_prefix(
            checkpoint_dir, records, recipe
        )
    if start_task:
        print(f"Resuming continual run at task {start_task}/{len(records)}")
    if start_task == len(records) and args.log_continual_eval:
        completed_rows = read_eval_rows(
            checkpoint_dir / "continual_eval.jsonl"
        )
        if completed_rows:
            log_continual_eval_to_wandb(
                args=args,
                recipe=recipe,
                rows=completed_rows,
                phase_index=len(records) - 1,
                log_tables=args.wandb_eval_tables,
            )

    for task_index in range(start_task, len(records)):
        record = records[task_index]
        budget = args.base_steps if task_index == 0 else args.steps_per_task
        task_args = replace(
            args,
            env_id=record.env_id,
            num_timesteps=budget,
            wandb_name_tag=(
                f"continual_t{task_index:02d}_{record.goal_hash}"
            ),
        )
        print(
            f"\n=== Continual task {task_index + 1}/{len(records)}: "
            f"{record.env_id} ({record.global_id}) ==="
        )
        carry = train_task(task_args, carry=carry, task_index=task_index)
        eval_rows = _evaluate_seen_tasks(
            args, records, carry, phase_index=task_index
        )
        all_eval_rows = write_phase_rows(
            checkpoint_dir / "continual_eval.jsonl",
            phase_index=task_index,
            rows=eval_rows,
        )
        if args.log_continual_eval:
            log_continual_eval_to_wandb(
                args=args,
                recipe=recipe,
                rows=all_eval_rows,
                phase_index=task_index,
                log_tables=args.wandb_eval_tables,
            )
        if not args.carry_actor:
            carry["actor"] = None
        _save_boundary_checkpoint(
            _checkpoint_path(checkpoint_dir, task_index),
            carry=carry,
            global_id=record.global_id,
            task_index=task_index,
            recipe=recipe,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
