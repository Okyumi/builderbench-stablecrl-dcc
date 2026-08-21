import unittest

import jax
import jax.numpy as jnp
import numpy as np

from continual.dcc_networks import make_dcc_networks, masked_dynamics_mse
from continual.semantic_layout import RAW_CUBE_FEATURE_DIM, SemanticLayout


def grouped_observation(cubes, select):
    cubes = np.asarray(cubes, dtype=np.float32)
    return np.concatenate([
        cubes[:, :3].reshape(-1),
        cubes[:, 3:7].reshape(-1),
        cubes[:, 7:10].reshape(-1),
        cubes[:, 10:13].reshape(-1),
        np.array([select], np.float32),
    ])


class DCCNetworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = SemanticLayout(max_cubes=4)
        cls.networks = make_dcc_networks(
            layout=cls.layout,
            action_size=5,
            rep_size=8,
            shared_width=16,
            task_width=8,
        )
        cls.params = cls.networks.init_params(jax.random.PRNGKey(0))
        cls.actor_params = cls.params.pop("actor")
        cls.cubes = np.arange(
            2 * RAW_CUBE_FEATURE_DIM, dtype=np.float32
        ).reshape(2, RAW_CUBE_FEATURE_DIM) / 20.0
        cls.goal = np.array(
            [[0.0, 0.0, 0.0], [0.04, 0.0, 0.04]], np.float32
        )

    def _inputs(self, permuted=False):
        if permuted:
            cubes = self.cubes[::-1]
            goal = self.goal[::-1]
            selector = 0.5  # same physical cube is now raw slot 1
        else:
            cubes = self.cubes
            goal = self.goal
            selector = -0.5
        observation = self.layout.pack_observation(
            grouped_observation(cubes, selector), 2
        )[None]
        packed_goal = self.layout.pack_goal(goal.reshape(-1), 2)[None]
        action = np.array([[0.1, -0.2, 0.3, 0.0, selector]], np.float32)
        return jnp.asarray(observation), jnp.asarray(packed_goal), jnp.asarray(action)

    def test_critic_and_goal_are_permutation_invariant(self):
        observation, goal, action = self._inputs(False)
        perm_obs, perm_goal, perm_action = self._inputs(True)
        np.testing.assert_allclose(
            self.networks.apply_sa(self.params, observation, action),
            self.networks.apply_sa(self.params, perm_obs, perm_action),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            self.networks.apply_goal(self.params, goal),
            self.networks.apply_goal(self.params, perm_goal),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_actor_pointer_is_permutation_equivariant(self):
        observation, goal, _ = self._inputs(False)
        perm_obs, perm_goal, _ = self._inputs(True)
        goal_repr = self.networks.apply_goal(self.params, goal)
        perm_goal_repr = self.networks.apply_goal(self.params, perm_goal)
        mean, log_std = self.networks.actor.apply(
            self.actor_params, observation, goal_repr
        )
        perm_mean, perm_log_std = self.networks.actor.apply(
            self.actor_params, perm_obs, perm_goal_repr
        )
        np.testing.assert_allclose(mean[..., :-1], perm_mean[..., :-1], atol=1e-5)
        np.testing.assert_allclose(log_std, perm_log_std, atol=1e-5)
        np.testing.assert_allclose(
            jnp.tanh(mean[..., -1]),
            -jnp.tanh(perm_mean[..., -1]),
            atol=1e-5,
        )

    def test_dynamics_prediction_is_equivariant_and_masked(self):
        observation, goal, action = self._inputs(False)
        perm_obs, perm_goal, perm_action = self._inputs(True)
        prediction, mask = self.networks.apply_dynamics(
            self.params, observation, action
        )
        perm_prediction, perm_mask = self.networks.apply_dynamics(
            self.params, perm_obs, perm_action
        )
        np.testing.assert_allclose(
            prediction[:, :2][:, ::-1], perm_prediction[:, :2], atol=1e-5
        )
        np.testing.assert_array_equal(mask, perm_mask)
        loss = masked_dynamics_mse(prediction, goal, self.layout.max_cubes)
        changed_padding = goal.at[:, 6:12].set(999.0)
        np.testing.assert_allclose(
            loss,
            masked_dynamics_mse(
                prediction, changed_padding, self.layout.max_cubes
            ),
        )

    def test_sgcrl_combine_and_goal_projection_modes(self):
        observation, goal, action = self._inputs(False)
        cases = (
            ("add", "shared", 8, False),
            ("add", "projected", 8, True),
            ("concat", "shared", 16, True),
        )
        for combine_mode, goal_mode, expected_size, has_projection in cases:
            with self.subTest(combine=combine_mode, goal=goal_mode):
                networks = make_dcc_networks(
                    layout=self.layout,
                    action_size=5,
                    rep_size=8,
                    shared_width=16,
                    task_width=8,
                    shared_depth=1,
                    task_depth=2,
                    combine_mode=combine_mode,
                    goal_encoder_mode=goal_mode,
                )
                params = networks.init_params(jax.random.PRNGKey(4))
                actor_params = params.pop("actor")
                sa_repr = networks.apply_sa(params, observation, action)
                goal_repr = networks.apply_goal(params, goal)
                self.assertEqual(sa_repr.shape[-1], expected_size)
                self.assertEqual(goal_repr.shape[-1], expected_size)
                self.assertEqual("psi_proj" in params, has_projection)
                self.assertEqual(
                    "psi_proj" in networks.shared_groups, has_projection
                )
                mean, log_std = networks.actor.apply(
                    actor_params, observation, goal_repr
                )
                self.assertEqual(mean.shape, (1, 5))
                self.assertEqual(log_std.shape, (1, 5))


if __name__ == "__main__":
    unittest.main()
