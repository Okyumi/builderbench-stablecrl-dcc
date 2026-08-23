import unittest

import jax
import jax.numpy as jnp
import numpy as np

from continual.semantic_layout import RAW_CUBE_FEATURE_DIM, SemanticLayout
from continual.vanilla_networks import make_vanilla_crl_networks


def grouped_observation(cubes, select):
    cubes = np.asarray(cubes, dtype=np.float32)
    return np.concatenate([
        cubes[:, :3].reshape(-1),
        cubes[:, 3:7].reshape(-1),
        cubes[:, 7:10].reshape(-1),
        cubes[:, 10:13].reshape(-1),
        np.array([select], np.float32),
    ])


class VanillaNetworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = SemanticLayout(max_cubes=4)
        cls.networks = make_vanilla_crl_networks(
            layout=cls.layout,
            action_size=5,
            rep_size=8,
            width=16,
            depth=2,
        )
        params = cls.networks.init_params(jax.random.PRNGKey(0))
        cls.actor_params = params.pop("actor")
        cls.params = params
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
            selector = 0.5
        else:
            cubes = self.cubes
            goal = self.goal
            selector = -0.5
        observation = self.layout.pack_observation(
            grouped_observation(cubes, selector), 2
        )[None]
        packed_goal = self.layout.pack_goal(goal.reshape(-1), 2)[None]
        action = np.array(
            [[0.1, -0.2, 0.3, 0.0, selector]], np.float32
        )
        return (
            jnp.asarray(observation),
            jnp.asarray(packed_goal),
            jnp.asarray(action),
        )

    def test_set_control_is_independent_of_residual_dcc(self):
        self.assertEqual(
            type(self.networks.sa_encoder).__module__,
            "continual.set_networks",
        )
        self.assertEqual(
            type(self.networks.g_encoder).__module__,
            "continual.set_networks",
        )
        self.assertEqual(
            type(self.networks.actor).__module__,
            "continual.set_networks",
        )

    def test_monolithic_critic_is_permutation_invariant(self):
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

    def test_actor_remains_pointer_equivariant(self):
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


if __name__ == "__main__":
    unittest.main()
