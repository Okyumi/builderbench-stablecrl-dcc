"""DCC decomposition on the proven upstream StableCRL residual networks.

The diagnostic experiments showed that fixed semantic padding is compatible
with the upstream residual model while the masked-set network does not learn
the 3- or 4-block controls. DCC therefore keeps the successful flat model and
adds only its continual decomposition::

    z_shared = phi_shared(s, a)
    z_task   = phi_task_t(s, a)
    z_sa     = combine(z_shared, z_task)
    z_goal   = project(psi_shared(g))

``phi_shared``, ``psi_shared``, an optional ``psi_proj``, and the upstream
actor transfer between tasks. ``phi_task_t`` is freshly initialized at every
task boundary and retained in a task bank for evaluation. Its output layer is
zero initialized, so the default additive DCC model starts exactly at the
upstream flat-network function before learning a task residual.

There is deliberately no dynamics head, dynamics target, or dynamics loss.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import variance_scaling

from continual.flat_upstream_networks import (
    FlatActor,
    FlatGoalEncoder,
    FlatStateActionEncoder,
)
from continual.semantic_layout import SemanticLayout


def _lecun_uniform():
    return variance_scaling(1 / 3, "fan_in", "uniform")


class GoalProjection(nn.Module):
    rep_size: int

    @nn.compact
    def __call__(self, goal_repr):
        return nn.Dense(
            self.rep_size,
            kernel_init=_lecun_uniform(),
            bias_init=nn.initializers.zeros,
            name="out",
        )(goal_repr)


@dataclasses.dataclass(frozen=True)
class DCCNetworks:
    phi_shared: nn.Module
    phi_task: nn.Module
    psi_shared: nn.Module
    psi_proj: nn.Module | None
    actor: nn.Module
    combine_mode: str
    goal_encoder_mode: str
    shared_groups: tuple[str, ...]
    init_params: Callable
    init_task_params: Callable
    apply_sa: Callable
    apply_goal: Callable


def make_dcc_networks(
    *,
    layout: SemanticLayout,
    action_size: int,
    rep_size: int = 64,
    architecture: str = "block",
    num_blocks: int = 8,
    hidden_dim: int = 1024,
    scale_actor_residual_by_depth: bool = True,
    scale_critic_residual_by_depth: bool = True,
    use_non_residual_critic_encoders: bool = False,
    task_width: int = 256,
    task_depth: int = 4,
    combine_mode: str = "add",
    goal_encoder_mode: str = "shared",
) -> DCCNetworks:
    """Build a flat residual DCC model with no dynamics auxiliary."""
    if architecture not in {"block", "default"}:
        raise ValueError(
            f"unknown architecture={architecture!r}; expected block/default"
        )
    if combine_mode not in {"add", "concat"}:
        raise ValueError(
            f"combine_mode must be 'add' or 'concat', got {combine_mode!r}"
        )
    if goal_encoder_mode not in {"shared", "projected"}:
        raise ValueError(
            "goal_encoder_mode must be 'shared' or 'projected', got "
            f"{goal_encoder_mode!r}"
        )
    if min(num_blocks, hidden_dim, task_width, task_depth) < 1:
        raise ValueError("DCC residual widths and depths must be positive")

    residual_actor = architecture == "block"
    residual_critic = (
        architecture == "block" and not use_non_residual_critic_encoders
    )
    critic_rep_size = rep_size if combine_mode == "add" else 2 * rep_size
    use_goal_projection = (
        goal_encoder_mode == "projected" or combine_mode == "concat"
    )

    phi_shared = FlatStateActionEncoder(
        rep_size=rep_size,
        residual=residual_critic,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        scale_residual_by_depth=scale_critic_residual_by_depth,
    )
    phi_task = FlatStateActionEncoder(
        rep_size=rep_size,
        residual=residual_critic,
        num_blocks=task_depth,
        hidden_dim=task_width,
        scale_residual_by_depth=True,
        zero_init_output=True,
    )
    psi_shared = FlatGoalEncoder(
        rep_size=rep_size,
        residual=residual_critic,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        scale_residual_by_depth=scale_critic_residual_by_depth,
    )
    psi_proj = (
        GoalProjection(rep_size=critic_rep_size)
        if use_goal_projection
        else None
    )
    actor = FlatActor(
        action_size=action_size,
        residual=residual_actor,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        scale_residual_by_depth=scale_actor_residual_by_depth,
    )

    dummy_obs = np.ones((1, layout.observation_size), dtype=np.float32)
    dummy_action = np.ones((1, action_size), dtype=np.float32)
    dummy_goal = np.ones((1, layout.goal_size), dtype=np.float32)

    def _keys(key):
        if isinstance(key, tuple):
            key_actor, key_shared, key_goal = key
        else:
            key_actor, key_shared, key_goal = jax.random.split(key, 3)
        return (
            key_actor,
            key_shared,
            key_goal,
            jax.random.fold_in(key_shared, 1),
            jax.random.fold_in(key_goal, 1),
        )

    def init_params(key):
        key_actor, key_shared, key_goal, key_task, key_projection = _keys(key)
        parts = {
            "phi_shared": phi_shared.init(
                key_shared, dummy_obs, dummy_action
            ),
            "phi_task": phi_task.init(key_task, dummy_obs, dummy_action),
            "psi_shared": psi_shared.init(key_goal, dummy_goal),
        }
        goal_repr = psi_shared.apply(parts["psi_shared"], dummy_goal)
        if psi_proj is not None:
            parts["psi_proj"] = psi_proj.init(key_projection, goal_repr)
            goal_repr = psi_proj.apply(parts["psi_proj"], goal_repr)
        parts["actor"] = actor.init(key_actor, dummy_obs, goal_repr)
        return parts

    def init_task_params(key):
        return phi_task.init(key, dummy_obs, dummy_action)

    def apply_sa(params, observation, action):
        shared = phi_shared.apply(
            params["phi_shared"], observation, action
        )
        task = phi_task.apply(params["phi_task"], observation, action)
        if combine_mode == "add":
            return shared + task
        return jnp.concatenate([shared, task], axis=-1)

    def apply_goal(params, goal):
        goal_repr = psi_shared.apply(params["psi_shared"], goal)
        if psi_proj is not None:
            goal_repr = psi_proj.apply(params["psi_proj"], goal_repr)
        return goal_repr

    return DCCNetworks(
        phi_shared=phi_shared,
        phi_task=phi_task,
        psi_shared=psi_shared,
        psi_proj=psi_proj,
        actor=actor,
        combine_mode=combine_mode,
        goal_encoder_mode=goal_encoder_mode,
        shared_groups=(
            "phi_shared",
            "psi_shared",
            *(("psi_proj",) if psi_proj is not None else ()),
        ),
        init_params=init_params,
        init_task_params=init_task_params,
        apply_sa=apply_sa,
        apply_goal=apply_goal,
    )
