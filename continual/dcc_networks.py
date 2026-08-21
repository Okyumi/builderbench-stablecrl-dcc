"""Permutation-aware DCC networks for StableCRL BuilderBench.

The decomposition follows DCC's shared/task split::

    z_shared = h_phi(b_shared(s, a))
    z_task   = phi_task(s, a)
    z_sa     = z_shared + z_task
    z_goal   = psi_shared(g)

``b_shared``, ``h_phi``, ``h_dyn`` and ``psi_shared`` transfer between
tasks. ``phi_task`` is reinitialized at every task boundary. The dynamics
head predicts next cube positions per slot with shared weights; its loss is
masked, so padded slots never become artificial training targets.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import variance_scaling

from continual.semantic_layout import CUBE_FEATURE_DIM, POSITION_DIM, SemanticLayout


def _lecun_uniform():
    return variance_scaling(1 / 3, "fan_in", "uniform")


class DenseStack(nn.Module):
    width: int
    depth: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.width, kernel_init=_lecun_uniform())(x)
            x = nn.LayerNorm()(x)
            x = nn.swish(x)
        return x


def _masked_mean(values, mask):
    weights = mask[..., None].astype(values.dtype)
    denominator = jnp.maximum(weights.sum(axis=-2), 1.0)
    return (values * weights).sum(axis=-2) / denominator


def _selector_centers(mask):
    """Continuous selector value used by PDWrapper for every valid slot."""
    num_cubes = jnp.maximum(mask.sum(axis=-1, keepdims=True), 1.0)
    indices = jnp.arange(mask.shape[-1], dtype=mask.dtype)
    return (2.0 * indices + 1.0) / num_cubes - 1.0


def _selection_weights(selector, mask, temperature=0.02):
    centers = _selector_centers(mask)
    logits = -jnp.square(selector - centers) / temperature
    logits = jnp.where(mask > 0, logits, -1e9)
    return nn.softmax(logits, axis=-1)


def _split_observation(observation, max_cubes):
    cube_end = max_cubes * CUBE_FEATURE_DIM
    mask_end = cube_end + max_cubes
    cubes = observation[..., :cube_end].reshape(
        observation.shape[:-1] + (max_cubes, CUBE_FEATURE_DIM)
    )
    mask = observation[..., cube_end:mask_end]
    return cubes, mask


def _split_goal(goal, max_cubes):
    position_end = max_cubes * POSITION_DIM
    positions = goal[..., :position_end].reshape(
        goal.shape[:-1] + (max_cubes, POSITION_DIM)
    )
    return positions, goal[..., position_end:]


class SharedStateActionBackbone(nn.Module):
    max_cubes: int
    width: int = 512
    depth: int = 3

    @nn.compact
    def __call__(self, observation, action):
        cubes, mask = _split_observation(
            observation, self.max_cubes
        )
        selector_weights = _selection_weights(
            action[..., -1:], mask
        )
        tiled_motion = jnp.broadcast_to(
            action[..., None, :-1],
            cubes.shape[:-1] + (action.shape[-1] - 1,),
        )
        slots = DenseStack(self.width, self.depth)(
            jnp.concatenate(
                [cubes, tiled_motion, selector_weights[..., None]], axis=-1
            )
        )
        pooled = _masked_mean(slots, mask)
        selected = (slots * selector_weights[..., None]).sum(axis=-2)
        global_features = jnp.concatenate(
            [pooled, selected, action[..., :-1]], axis=-1
        )
        return DenseStack(self.width, 1)(global_features), slots, mask


class SharedProjection(nn.Module):
    rep_size: int

    @nn.compact
    def __call__(self, hidden):
        return nn.Dense(
            self.rep_size, kernel_init=_lecun_uniform(), name="out"
        )(hidden)


class EquivariantDynamicsHead(nn.Module):
    @nn.compact
    def __call__(self, slot_hidden):
        return nn.Dense(
            POSITION_DIM, kernel_init=_lecun_uniform(), name="next_position"
        )(slot_hidden)


class TaskStateActionEncoder(nn.Module):
    max_cubes: int
    rep_size: int
    width: int = 256
    depth: int = 2

    @nn.compact
    def __call__(self, observation, action):
        cubes, mask = _split_observation(
            observation, self.max_cubes
        )
        selector_weights = _selection_weights(
            action[..., -1:], mask
        )
        tiled_motion = jnp.broadcast_to(
            action[..., None, :-1],
            cubes.shape[:-1] + (action.shape[-1] - 1,),
        )
        slots = DenseStack(self.width, self.depth)(
            jnp.concatenate(
                [cubes, tiled_motion, selector_weights[..., None]], axis=-1
            )
        )
        pooled = _masked_mean(slots, mask)
        selected = (slots * selector_weights[..., None]).sum(axis=-2)
        pooled = jnp.concatenate(
            [pooled, selected, action[..., :-1]], axis=-1
        )
        return nn.Dense(
            self.rep_size, kernel_init=_lecun_uniform(), name="out"
        )(pooled)


class SetGoalEncoder(nn.Module):
    max_cubes: int
    rep_size: int
    width: int = 512
    depth: int = 3

    @nn.compact
    def __call__(self, goal):
        positions, mask = _split_goal(goal, self.max_cubes)
        slots = DenseStack(self.width, self.depth)(positions)
        pooled = _masked_mean(slots, mask)
        # The centroid is permutation invariant and retains absolute target
        # location, which is necessary for control.
        centroid = _masked_mean(positions, mask)
        return nn.Dense(
            self.rep_size, kernel_init=_lecun_uniform(), name="out"
        )(jnp.concatenate([pooled, centroid], axis=-1))


class SetActor(nn.Module):
    max_cubes: int
    action_size: int
    width: int = 512
    depth: int = 3

    LOG_STD_MAX = 2.0
    LOG_STD_MIN = -5.0

    @nn.compact
    def __call__(self, observation, goal_repr):
        cubes, mask = _split_observation(
            observation, self.max_cubes
        )
        slots = DenseStack(self.width, self.depth)(cubes)
        state_repr = _masked_mean(slots, mask)
        goal_per_slot = jnp.broadcast_to(
            goal_repr[..., None, :],
            slots.shape[:-1] + (goal_repr.shape[-1],),
        )
        selector_logits = nn.Dense(
            1, kernel_init=_lecun_uniform(), name="selector_logits"
        )(jnp.concatenate([slots, goal_per_slot], axis=-1))[..., 0]
        selector_logits = jnp.where(mask > 0, selector_logits, -1e9)
        selector_probs = nn.softmax(selector_logits, axis=-1)
        selected_repr = (slots * selector_probs[..., None]).sum(axis=-2)
        hidden = DenseStack(self.width, 2)(jnp.concatenate(
            [state_repr, selected_repr, goal_repr], axis=-1
        ))
        motion_dim = self.action_size - 1
        motion_mean = nn.Dense(
            motion_dim, kernel_init=_lecun_uniform(), name="motion_mean"
        )(hidden)
        motion_log_std = nn.Dense(
            motion_dim, kernel_init=_lecun_uniform(), name="motion_log_std"
        )(hidden)
        selector_center = (
            selector_probs * _selector_centers(mask)
        ).sum(axis=-1, keepdims=True)
        selector_mean = jnp.arctanh(jnp.clip(selector_center, -0.999, 0.999))
        selector_log_std = nn.Dense(
            1, kernel_init=_lecun_uniform(), name="selector_log_std"
        )(hidden)
        mean = jnp.concatenate([motion_mean, selector_mean], axis=-1)
        log_std = jnp.concatenate(
            [motion_log_std, selector_log_std], axis=-1
        )
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (
            self.LOG_STD_MAX - self.LOG_STD_MIN
        ) * (log_std + 1.0)
        return mean, log_std


@dataclasses.dataclass(frozen=True)
class DCCNetworks:
    b_shared: nn.Module
    h_phi: nn.Module
    h_dyn: nn.Module
    phi_task: nn.Module
    psi_shared: nn.Module
    actor: nn.Module
    init_params: Callable
    init_task_params: Callable
    apply_sa: Callable
    apply_goal: Callable
    apply_dynamics: Callable


def make_dcc_networks(
    *,
    layout: SemanticLayout,
    action_size: int,
    rep_size: int = 64,
    shared_width: int = 512,
    task_width: int = 256,
) -> DCCNetworks:
    b_shared = SharedStateActionBackbone(
        max_cubes=layout.max_cubes, width=shared_width
    )
    h_phi = SharedProjection(rep_size=rep_size)
    h_dyn = EquivariantDynamicsHead()
    phi_task = TaskStateActionEncoder(
        max_cubes=layout.max_cubes,
        rep_size=rep_size,
        width=task_width,
    )
    psi_shared = SetGoalEncoder(
        max_cubes=layout.max_cubes,
        rep_size=rep_size,
        width=shared_width,
    )
    actor = SetActor(
        max_cubes=layout.max_cubes,
        action_size=action_size,
        width=shared_width,
    )

    dummy_obs = np.zeros((1, layout.observation_size), dtype=np.float32)
    dummy_action = np.zeros((1, action_size), dtype=np.float32)
    dummy_goal = np.zeros((1, layout.goal_size), dtype=np.float32)
    # A valid first slot avoids a degenerate all-padding init input.
    dummy_obs[:, layout.max_cubes * CUBE_FEATURE_DIM] = 1.0
    dummy_goal[:, layout.max_cubes * POSITION_DIM] = 1.0

    def _init_parts(key):
        keys = jax.random.split(key, 6)
        b_params = b_shared.init(keys[0], dummy_obs, dummy_action)
        hidden, slots, _ = b_shared.apply(b_params, dummy_obs, dummy_action)
        return {
            "b_shared": b_params,
            "h_phi": h_phi.init(keys[1], hidden),
            "h_dyn": h_dyn.init(keys[2], slots),
            "phi_task": phi_task.init(keys[3], dummy_obs, dummy_action),
            "psi_shared": psi_shared.init(keys[4], dummy_goal),
            "actor": actor.init(keys[5], dummy_obs, np.zeros((1, rep_size), np.float32)),
        }

    def init_params(key):
        return _init_parts(key)

    def init_task_params(key):
        return phi_task.init(key, dummy_obs, dummy_action)

    def apply_sa(params, observation, action):
        hidden, _, _ = b_shared.apply(
            params["b_shared"], observation, action
        )
        shared = h_phi.apply(params["h_phi"], hidden)
        task = phi_task.apply(params["phi_task"], observation, action)
        return shared + task

    def apply_goal(params, goal):
        return psi_shared.apply(params["psi_shared"], goal)

    def apply_dynamics(params, observation, action):
        _, slots, mask = b_shared.apply(
            params["b_shared"], observation, action
        )
        prediction = h_dyn.apply(params["h_dyn"], slots)
        return prediction, mask

    return DCCNetworks(
        b_shared=b_shared,
        h_phi=h_phi,
        h_dyn=h_dyn,
        phi_task=phi_task,
        psi_shared=psi_shared,
        actor=actor,
        init_params=init_params,
        init_task_params=init_task_params,
        apply_sa=apply_sa,
        apply_goal=apply_goal,
        apply_dynamics=apply_dynamics,
    )


def masked_dynamics_mse(predicted_positions, next_goal, max_cubes):
    """MSE over real cube slots only; padded slots have exactly zero weight."""
    target_positions, target_mask = _split_goal(next_goal, max_cubes)
    squared_error = jnp.square(predicted_positions - target_positions).sum(-1)
    denominator = jnp.maximum(target_mask.sum(), 1.0)
    return (squared_error * target_mask).sum() / denominator
