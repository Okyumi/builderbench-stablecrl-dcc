"""Continual vanilla contrastive-RL baselines for BuilderBench.

The learner uses the same semantic wrapper, masked-set encoders, task
manifest, and evaluation protocol as continual DCC.  Its critic is not
decomposed.  Actor and critic lifecycles are explicit so the standard
reset/reset and persistent/persistent controls cannot be confused.
Replay buffers and optimizer states are always fresh at task boundaries.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import jax
import numpy as np
import tyro

from builderbench.env_utils import make_env
from continual.eval_logging import (
    log_continual_eval_to_wandb,
    read_eval_rows,
    write_phase_rows,
)
from continual.flat_upstream_networks import make_flat_upstream_networks
from continual.grouped_layout import GroupedPadLayout
from continual.grouped_wrapper import GroupedPadWrapper
from continual.semantic_layout import SemanticLayout
from continual.semantic_wrapper import SemanticPadWrapper
from continual.task_manifest import build_manifest, write_manifest
from continual.vanilla_networks import make_vanilla_crl_networks
from continual_dcc import (
    _checkpoint_path,
    _goal_for_env_id,
    _resume_prefix,
    _save_boundary_checkpoint,
)
from stable_crl_dcc import Args as StableCRLArgs
from stable_crl_dcc import make_inference_fn
from stable_crl_dcc import main as train_task
from utils.evaluation import Evaluator
from utils.wrapper import PDWrapper, wrap_env


@dataclass
class Args(StableCRLArgs):
    task_sequence: str = (
        "creative-1-task1,creative-1-task2,creative-2-task1,"
        "creative-3-task1,creative-4-task1"
    )
    base_steps: int = 200_000_000
    steps_per_task: int = 200_000_000
    actor_lifecycle: Literal["reset", "persistent"] = "reset"
    critic_lifecycle: Literal["reset", "persistent"] = "reset"
    boundary_checkpoint_dir: str = "checkpoints/continual_crl"
    task_data_version: str = "david-6e8d56d"
    resume: bool = True
    critic_family: Literal["vanilla"] = "vanilla"
    observation_layout: Literal["semantic", "grouped"] = "semantic"
    vanilla_network_type: Literal["set", "flat_upstream"] = "set"
    eval_next_task: bool = True
    log_continual_eval: bool = True
    wandb_eval_tables: bool = True


def _checkpoint_recipe(args: Args) -> dict:
    """Training semantics that must remain fixed across resume."""
    names = (
        "seed",
        "task_sequence",
        "task_data_version",
        "base_steps",
        "steps_per_task",
        "actor_lifecycle",
        "critic_lifecycle",
        "num_envs",
        "num_eval_envs",
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
        "vanilla_width",
        "vanilla_depth",
    )
    recipe = {
        "algorithm": "stablecrl-vanilla-semantic-set-v1",
        **{name: getattr(args, name) for name in names},
    }
    # Preserve the exact legacy recipe for already-running indices 6--23.
    if (
        args.observation_layout != "semantic"
        or args.vanilla_network_type != "set"
    ):
        recipe.update({
            "algorithm": (
                "stablecrl-vanilla-fixed-layout-flat-upstream-v2"
            ),
            "observation_layout": args.observation_layout,
            "vanilla_network_type": args.vanilla_network_type,
            "architecture": args.architecture,
            "num_blocks": args.num_blocks,
            "hidden_dim": args.hidden_dim,
            "scale_actor_residual_by_depth": (
                args.scale_actor_residual_by_depth
            ),
            "scale_critic_residual_by_depth": (
                args.scale_critic_residual_by_depth
            ),
            "use_non_residual_critic_encoders": (
                args.use_non_residual_critic_encoders
            ),
        })
    return recipe


def _transfer_carry(args: Args, carry: dict | None) -> dict | None:
    """Select parameters visible to the next task from a full snapshot."""
    if carry is None:
        return None
    return {
        "actor": (
            carry["actor"] if args.actor_lifecycle == "persistent" else None
        ),
        "critic": (
            carry["critic"]
            if args.critic_lifecycle == "persistent"
            else None
        ),
    }


def _evaluate_seen_tasks(args, records, carry, phase_index):
    """Evaluate seen tasks and an optional next-task zero-shot control."""
    results = []
    if args.vanilla_network_type == "set" and args.observation_layout != "semantic":
        raise ValueError("set networks require the semantic observation layout")
    layout = (
        SemanticLayout(max_cubes=args.max_cubes)
        if args.observation_layout == "semantic"
        else GroupedPadLayout(max_cubes=args.max_cubes)
    )
    wrapper_type = (
        SemanticPadWrapper
        if args.observation_layout == "semantic"
        else GroupedPadWrapper
    )
    eval_indices = list(range(phase_index + 1))
    if args.eval_next_task and phase_index + 1 < len(records):
        eval_indices.append(phase_index + 1)
    for eval_index in eval_indices:
        record = records[eval_index]
        eval_args = replace(args, env_id=record.env_id)
        env_class, config = make_env(eval_args)
        config.impl = args.mjx_impl
        raw_env = wrapper_type(
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

        if args.vanilla_network_type == "set":
            networks = make_vanilla_crl_networks(
                layout=layout,
                action_size=env.action_size,
                rep_size=args.rep_size,
                width=args.vanilla_width,
                depth=args.vanilla_depth,
            )
        else:
            networks = make_flat_upstream_networks(
                observation_size=layout.observation_size,
                goal_size=layout.goal_size,
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
            )
        evaluator = Evaluator(
            env,
            functools.partial(make_inference_fn(networks), deterministic=True),
            num_eval_envs=args.num_eval_envs,
            episode_length=episode_length,
            key=jax.random.PRNGKey(
                args.seed + 10_000 * phase_index + eval_index
            ),
        )
        metrics = evaluator.run_evaluation(
            policy_params={
                "actor": carry["actor"],
                "critic": carry["critic"],
            },
            training_metrics={},
        )
        results.append({
            "phase_index": phase_index,
            "eval_task_index": eval_index,
            "eval_scope": (
                "seen" if eval_index <= phase_index else "next_unseen"
            ),
            "critic_head_task_index": None,
            "train_task_global_id": records[phase_index].global_id,
            "eval_task_global_id": record.global_id,
            **{
                key: float(np.asarray(value))
                for key, value in metrics.items()
            },
        })
    return results


def main(args: Args) -> None:
    task_ids = [item.strip() for item in args.task_sequence.split(",")]
    if not task_ids or any(not item for item in task_ids):
        raise ValueError("task_sequence must contain comma-separated task ids")
    tasks = [(env_id, _goal_for_env_id(env_id)) for env_id in task_ids]
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
                "checkpoint directory when changing a baseline setting"
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
        print(f"Resuming continual CRL at task {start_task}/{len(records)}")
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
            critic_family="vanilla",
            vanilla_carry_critic=(
                args.critic_lifecycle == "persistent"
            ),
            wandb_name_tag=(
                f"continual_crl_t{task_index:02d}_{record.goal_hash}"
            ),
        )
        print(
            f"\n=== Vanilla CRL task {task_index + 1}/{len(records)}: "
            f"{record.env_id} ({record.global_id}); "
            f"lifecycle={args.actor_lifecycle}/"
            f"{args.critic_lifecycle} ==="
        )
        incoming = _transfer_carry(args, carry)
        carry = train_task(
            task_args, carry=incoming, task_index=task_index
        )
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
        _save_boundary_checkpoint(
            _checkpoint_path(checkpoint_dir, task_index),
            carry=carry,
            global_id=record.global_id,
            task_index=task_index,
            recipe=recipe,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
