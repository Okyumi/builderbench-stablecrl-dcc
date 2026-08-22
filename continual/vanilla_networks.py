"""Permutation-safe, non-decomposed contrastive networks.

This module is the semantic-wrapper control for DCC.  It deliberately uses
the same masked-set actor and the same padded BuilderBench contract as DCC,
but the critic is a conventional pair of state-action and goal encoders:

    z_sa = phi(s, a)
    z_g  = psi(g)

There is no task-specific encoder, shared/task addition, or dynamics loss.
Consequently reset/reset versus persistent/persistent changes only parameter
lifecycle; it does not silently change the observation representation.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import numpy as np

from continual.dcc_networks import (
    SetActor,
    SetGoalEncoder,
    TaskStateActionEncoder,
)
from continual.semantic_layout import (
    CUBE_FEATURE_DIM,
    POSITION_DIM,
    SemanticLayout,
)


@dataclasses.dataclass(frozen=True)
class VanillaCRLNetworks:
    """Network adapter consumed by the semantic StableCRL learner."""

    sa_encoder: object
    g_encoder: object
    actor: object
    init_params: Callable
    apply_sa: Callable
    apply_goal: Callable


def make_vanilla_crl_networks(
    *,
    layout: SemanticLayout,
    action_size: int,
    rep_size: int = 64,
    width: int = 512,
    depth: int = 3,
) -> VanillaCRLNetworks:
    """Build the monolithic semantic CRL control networks."""
    if width < 1 or depth < 1:
        raise ValueError("vanilla encoder width and depth must be positive")

    sa_encoder = TaskStateActionEncoder(
        max_cubes=layout.max_cubes,
        rep_size=rep_size,
        width=width,
        depth=depth,
    )
    g_encoder = SetGoalEncoder(
        max_cubes=layout.max_cubes,
        rep_size=rep_size,
        width=width,
        depth=depth,
    )
    actor = SetActor(
        max_cubes=layout.max_cubes,
        action_size=action_size,
        width=width,
        depth=depth,
    )

    dummy_obs = np.zeros((1, layout.observation_size), dtype=np.float32)
    dummy_action = np.zeros((1, action_size), dtype=np.float32)
    dummy_goal = np.zeros((1, layout.goal_size), dtype=np.float32)
    # Avoid all-padding inputs during initialization.
    dummy_obs[:, layout.max_cubes * CUBE_FEATURE_DIM] = 1.0
    dummy_goal[:, layout.max_cubes * POSITION_DIM] = 1.0

    def init_params(key):
        key_sa, key_g, key_actor = jax.random.split(key, 3)
        critic = {
            "sa_encoder": sa_encoder.init(
                key_sa, dummy_obs, dummy_action
            ),
            "g_encoder": g_encoder.init(key_g, dummy_goal),
        }
        goal_repr = g_encoder.apply(critic["g_encoder"], dummy_goal)
        return {
            **critic,
            "actor": actor.init(key_actor, dummy_obs, goal_repr),
        }

    def apply_sa(params, observation, action):
        return sa_encoder.apply(
            params["sa_encoder"], observation, action
        )

    def apply_goal(params, goal):
        return g_encoder.apply(params["g_encoder"], goal)

    return VanillaCRLNetworks(
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
        actor=actor,
        init_params=init_params,
        apply_sa=apply_sa,
        apply_goal=apply_goal,
    )
