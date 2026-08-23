"""Masked-set networks for the legacy semantic-padding CRL control.

These modules are intentionally independent of residual DCC.  They remain in
the repository only for the diagnostic vanilla set-network cells and their
reset/persistent lifecycle controls.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling

from continual.semantic_layout import CUBE_FEATURE_DIM, POSITION_DIM


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
    """Return the continuous PD selector value for every valid cube slot."""
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


class SetStateActionEncoder(nn.Module):
    max_cubes: int
    rep_size: int
    width: int = 256
    depth: int = 2

    @nn.compact
    def __call__(self, observation, action):
        cubes, mask = _split_observation(observation, self.max_cubes)
        selector_weights = _selection_weights(action[..., -1:], mask)
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
        features = jnp.concatenate(
            [pooled, selected, action[..., :-1]], axis=-1
        )
        return nn.Dense(
            self.rep_size, kernel_init=_lecun_uniform(), name="out"
        )(features)


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
        # Absolute target location is needed for control, so retain the
        # permutation-invariant goal centroid alongside learned set features.
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
        cubes, mask = _split_observation(observation, self.max_cubes)
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
        hidden = DenseStack(self.width, 2)(
            jnp.concatenate(
                [state_repr, selected_repr, goal_repr], axis=-1
            )
        )
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
        selector_mean = jnp.arctanh(
            jnp.clip(selector_center, -0.999, 0.999)
        )
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
