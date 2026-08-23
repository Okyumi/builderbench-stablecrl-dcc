import unittest

import jax
import jax.numpy as jnp
import numpy as np

from continual.dcc_networks import make_dcc_networks
from continual.flat_upstream_networks import make_flat_upstream_networks
from continual.semantic_layout import SemanticLayout


class DCCNetworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = SemanticLayout(max_cubes=4)
        cls.action_size = 5
        cls.rep_size = 8
        cls.keys = tuple(jax.random.split(jax.random.PRNGKey(7), 3))
        cls.observation = jnp.asarray(
            np.linspace(
                -0.5,
                0.5,
                cls.layout.observation_size,
                dtype=np.float32,
            )[None]
        )
        cls.goal = jnp.asarray(
            np.linspace(
                0.0,
                1.0,
                cls.layout.goal_size,
                dtype=np.float32,
            )[None]
        )
        cls.action = jnp.asarray(
            [[0.1, -0.2, 0.3, 0.0, -0.5]], dtype=jnp.float32
        )

    def _make_dcc(self, **overrides):
        options = {
            "layout": self.layout,
            "action_size": self.action_size,
            "rep_size": self.rep_size,
            "num_blocks": 2,
            "hidden_dim": 16,
            "task_width": 8,
            "task_depth": 1,
        }
        options.update(overrides)
        return make_dcc_networks(**options)

    def test_additive_dcc_starts_exactly_at_flat_upstream_function(self):
        dcc = self._make_dcc()
        flat = make_flat_upstream_networks(
            observation_size=self.layout.observation_size,
            goal_size=self.layout.goal_size,
            action_size=self.action_size,
            rep_size=self.rep_size,
            num_blocks=2,
            hidden_dim=16,
        )
        dcc_params = dcc.init_params(self.keys)
        flat_params = flat.init_params(self.keys)
        dcc_actor = dcc_params.pop("actor")
        flat_actor = flat_params.pop("actor")

        dcc_sa = dcc.apply_sa(
            dcc_params, self.observation, self.action
        )
        flat_sa = flat.apply_sa(
            flat_params, self.observation, self.action
        )
        dcc_goal = dcc.apply_goal(dcc_params, self.goal)
        flat_goal = flat.apply_goal(flat_params, self.goal)
        np.testing.assert_allclose(dcc_sa, flat_sa, atol=1e-6)
        np.testing.assert_allclose(dcc_goal, flat_goal, atol=1e-6)

        dcc_policy = dcc.actor.apply(
            dcc_actor, self.observation, dcc_goal
        )
        flat_policy = flat.actor.apply(
            flat_actor, self.observation, flat_goal
        )
        np.testing.assert_allclose(dcc_policy[0], flat_policy[0], atol=1e-6)
        np.testing.assert_allclose(dcc_policy[1], flat_policy[1], atol=1e-6)

    def test_task_adapter_output_is_zero_at_initialization(self):
        dcc = self._make_dcc()
        params = dcc.init_params(self.keys)
        params.pop("actor")
        task = dcc.phi_task.apply(
            params["phi_task"], self.observation, self.action
        )
        np.testing.assert_array_equal(task, np.zeros_like(task))

    def test_no_dynamics_parameters_or_api_exist(self):
        dcc = self._make_dcc()
        params = dcc.init_params(self.keys)
        self.assertNotIn("h_dyn", params)
        self.assertNotIn("h_dyn", dcc.shared_groups)
        self.assertFalse(hasattr(dcc, "apply_dynamics"))
        self.assertEqual(
            dcc.shared_groups, ("phi_shared", "psi_shared")
        )

    def test_sgcrl_combine_and_goal_projection_modes(self):
        cases = (
            ("add", "shared", 8, False),
            ("add", "projected", 8, True),
            ("concat", "shared", 16, True),
        )
        for combine_mode, goal_mode, expected_size, has_projection in cases:
            with self.subTest(combine=combine_mode, goal=goal_mode):
                dcc = self._make_dcc(
                    combine_mode=combine_mode,
                    goal_encoder_mode=goal_mode,
                )
                params = dcc.init_params(self.keys)
                actor_params = params.pop("actor")
                sa_repr = dcc.apply_sa(
                    params, self.observation, self.action
                )
                goal_repr = dcc.apply_goal(params, self.goal)
                self.assertEqual(sa_repr.shape[-1], expected_size)
                self.assertEqual(goal_repr.shape[-1], expected_size)
                self.assertEqual("psi_proj" in params, has_projection)
                self.assertEqual(
                    "psi_proj" in dcc.shared_groups, has_projection
                )
                mean, log_std = dcc.actor.apply(
                    actor_params, self.observation, goal_repr
                )
                self.assertEqual(mean.shape, (1, self.action_size))
                self.assertEqual(log_std.shape, (1, self.action_size))


if __name__ == "__main__":
    unittest.main()
