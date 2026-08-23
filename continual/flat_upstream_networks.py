"""Upstream StableCRL residual networks on a fixed padded vector.

This is a diagnostic adapter, not a new algorithm.  The residual block,
state-action encoder, goal encoder, actor, initialization, and output
parameterization match ``stable_crl.py``.  Only the input dimensions come
from a fixed-capacity wrapper so the same parameter tree can cross tasks.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import variance_scaling


def _lecun_uniform():
    return variance_scaling(1 / 3, "fan_in", "uniform")


class ResidualBlock(nn.Module):
    hidden_dim: int
    norm_type: str = "layer_norm"
    residual_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        normalize = (
            (lambda value: nn.LayerNorm()(value))
            if self.norm_type == "layer_norm"
            else (lambda value: value)
        )
        residual = x
        for _ in range(4):
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=_lecun_uniform(),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.swish(normalize(x))
        return residual + self.residual_scale * x


class FlatStateActionEncoder(nn.Module):
    rep_size: int
    residual: bool = True
    num_blocks: int = 1
    hidden_dim: int = 1024
    norm_type: str = "layer_norm"
    scale_residual_by_depth: bool = True
    zero_init_output: bool = False

    @nn.compact
    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray):
        normalize = (
            (lambda value: nn.LayerNorm()(value))
            if self.norm_type == "layer_norm"
            else (lambda value: value)
        )
        x = jnp.concatenate([observation, action], axis=-1)
        if self.residual:
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=_lecun_uniform(),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.swish(normalize(x))
            branch_scale = (
                1.0 / np.sqrt(self.num_blocks)
                if self.scale_residual_by_depth and self.num_blocks > 1
                else 1.0
            )
            for _ in range(self.num_blocks):
                x = ResidualBlock(
                    hidden_dim=self.hidden_dim,
                    norm_type=self.norm_type,
                    residual_scale=branch_scale,
                )(x)
        else:
            for _ in range(4):
                x = nn.Dense(
                    1024,
                    kernel_init=_lecun_uniform(),
                    bias_init=nn.initializers.zeros,
                )(x)
                x = nn.swish(normalize(x))
        return nn.Dense(
            self.rep_size,
            kernel_init=(
                nn.initializers.zeros
                if self.zero_init_output
                else _lecun_uniform()
            ),
            bias_init=nn.initializers.zeros,
        )(x)


class FlatGoalEncoder(nn.Module):
    rep_size: int
    residual: bool = True
    num_blocks: int = 1
    hidden_dim: int = 1024
    norm_type: str = "layer_norm"
    scale_residual_by_depth: bool = True

    @nn.compact
    def __call__(self, goal: jnp.ndarray):
        normalize = (
            (lambda value: nn.LayerNorm()(value))
            if self.norm_type == "layer_norm"
            else (lambda value: value)
        )
        x = goal
        if self.residual:
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=_lecun_uniform(),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.swish(normalize(x))
            branch_scale = (
                1.0 / np.sqrt(self.num_blocks)
                if self.scale_residual_by_depth and self.num_blocks > 1
                else 1.0
            )
            for _ in range(self.num_blocks):
                x = ResidualBlock(
                    hidden_dim=self.hidden_dim,
                    norm_type=self.norm_type,
                    residual_scale=branch_scale,
                )(x)
        else:
            for _ in range(4):
                x = nn.Dense(
                    1024,
                    kernel_init=_lecun_uniform(),
                    bias_init=nn.initializers.zeros,
                )(x)
                x = nn.swish(normalize(x))
        return nn.Dense(
            self.rep_size,
            kernel_init=_lecun_uniform(),
            bias_init=nn.initializers.zeros,
        )(x)


class FlatActor(nn.Module):
    action_size: int
    residual: bool = True
    num_blocks: int = 1
    hidden_dim: int = 1024
    norm_type: str = "layer_norm"
    scale_residual_by_depth: bool = True

    LOG_STD_MAX = 2.0
    LOG_STD_MIN = -5.0

    @nn.compact
    def __call__(self, observation, goal_repr):
        normalize = (
            (lambda value: nn.LayerNorm()(value))
            if self.norm_type == "layer_norm"
            else (lambda value: value)
        )
        x = jnp.concatenate([observation, goal_repr], axis=-1)
        if self.residual:
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=_lecun_uniform(),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.swish(normalize(x))
            branch_scale = (
                1.0 / np.sqrt(self.num_blocks)
                if self.scale_residual_by_depth and self.num_blocks > 1
                else 1.0
            )
            for _ in range(self.num_blocks):
                x = ResidualBlock(
                    hidden_dim=self.hidden_dim,
                    norm_type=self.norm_type,
                    residual_scale=branch_scale,
                )(x)
        else:
            for _ in range(4):
                x = nn.Dense(
                    1024,
                    kernel_init=_lecun_uniform(),
                    bias_init=nn.initializers.zeros,
                )(x)
                x = nn.swish(normalize(x))

        mean = nn.Dense(
            self.action_size,
            kernel_init=_lecun_uniform(),
            bias_init=nn.initializers.zeros,
        )(x)
        log_std = nn.Dense(
            self.action_size,
            kernel_init=_lecun_uniform(),
            bias_init=nn.initializers.zeros,
        )(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (
            self.LOG_STD_MAX - self.LOG_STD_MIN
        ) * (log_std + 1.0)
        return mean, log_std


@dataclasses.dataclass(frozen=True)
class FlatUpstreamNetworks:
    sa_encoder: object
    g_encoder: object
    actor: object
    init_params: Callable
    apply_sa: Callable
    apply_goal: Callable


def make_flat_upstream_networks(
    *,
    observation_size: int,
    goal_size: int,
    action_size: int,
    rep_size: int = 64,
    architecture: str = "block",
    num_blocks: int = 8,
    hidden_dim: int = 1024,
    scale_actor_residual_by_depth: bool = True,
    scale_critic_residual_by_depth: bool = True,
    use_non_residual_critic_encoders: bool = False,
) -> FlatUpstreamNetworks:
    """Build an upstream-equivalent actor and monolithic CRL critic."""
    if architecture not in {"block", "default"}:
        raise ValueError(
            f"unknown architecture={architecture!r}; expected block/default"
        )
    if num_blocks < 1 or hidden_dim < 1:
        raise ValueError("num_blocks and hidden_dim must be positive")

    residual_actor = architecture == "block"
    residual_critic = (
        architecture == "block" and not use_non_residual_critic_encoders
    )
    actor = FlatActor(
        action_size=action_size,
        residual=residual_actor,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        scale_residual_by_depth=scale_actor_residual_by_depth,
    )
    sa_encoder = FlatStateActionEncoder(
        rep_size=rep_size,
        residual=residual_critic,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        scale_residual_by_depth=scale_critic_residual_by_depth,
    )
    g_encoder = FlatGoalEncoder(
        rep_size=rep_size,
        residual=residual_critic,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        scale_residual_by_depth=scale_critic_residual_by_depth,
    )
    dummy_obs = np.ones((1, observation_size), dtype=np.float32)
    dummy_action = np.ones((1, action_size), dtype=np.float32)
    dummy_goal = np.ones((1, goal_size), dtype=np.float32)

    def init_params(key):
        # The semantic learner supplies the same three top-level RNG streams
        # as stable_crl.py. A single key remains supported for unit use.
        if isinstance(key, tuple):
            key_actor, key_sa, key_goal = key
        else:
            key_actor, key_sa, key_goal = jax.random.split(key, 3)
        critic = {
            "sa_encoder": sa_encoder.init(key_sa, dummy_obs, dummy_action),
            "g_encoder": g_encoder.init(key_goal, dummy_goal),
        }
        goal_repr = g_encoder.apply(critic["g_encoder"], dummy_goal)
        return {
            **critic,
            "actor": actor.init(key_actor, dummy_obs, goal_repr),
        }

    def apply_sa(params, observation, action):
        return sa_encoder.apply(params["sa_encoder"], observation, action)

    def apply_goal(params, goal):
        return g_encoder.apply(params["g_encoder"], goal)

    return FlatUpstreamNetworks(
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
        actor=actor,
        init_params=init_params,
        apply_sa=apply_sa,
        apply_goal=apply_goal,
    )
