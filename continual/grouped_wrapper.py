"""JAX wrapper for the padding-only grouped observation layout."""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from continual.grouped_layout import GroupedPadLayout
from utils.wrapper import Wrapper


class GroupedPadWrapper(Wrapper):
    """Pad raw feature groups without changing their ordering or selector."""

    def __init__(self, env: Any, *, num_cubes: int, max_cubes: int):
        super().__init__(env)
        self.num_cubes = num_cubes
        self.layout = GroupedPadLayout(max_cubes=max_cubes)
        self.layout.validate_num_cubes(num_cubes)
        if bool(getattr(env._config, "delta_control", False)):
            raise ValueError(
                "GroupedPadWrapper currently requires delta_control=False"
            )

    @property
    def observation_size(self) -> int:
        return self.layout.observation_size

    @property
    def goal_size(self) -> int:
        return self.layout.goal_size

    def _pack_state(self, state):
        info = dict(state.info)
        info["achieved_goal"] = self.layout.pack_goal(
            info["achieved_goal"], self.num_cubes, xp=jnp
        )
        info["target_goal"] = self.layout.pack_goal(
            info["target_goal"], self.num_cubes, xp=jnp
        )
        return state.replace(
            obs=self.layout.pack_observation(
                state.obs, self.num_cubes, xp=jnp
            ),
            info=info,
        )

    def _unpack_state(self, state):
        info = dict(state.info)
        info["achieved_goal"] = self.layout.unpack_goal(
            info["achieved_goal"], self.num_cubes
        )
        info["target_goal"] = self.layout.unpack_goal(
            info["target_goal"], self.num_cubes
        )
        return state.replace(
            obs=self.layout.unpack_observation(
                state.obs, self.num_cubes, xp=jnp
            ),
            info=info,
        )

    def reset(self, rng):
        return self._pack_state(self.env.reset(rng))

    def step(self, state, action):
        return self._pack_state(self.env.step(self._unpack_state(state), action))
