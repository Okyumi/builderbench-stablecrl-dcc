import unittest

import jax
import jax.numpy as jnp

from continual.flat_upstream_networks import make_flat_upstream_networks


class FlatUpstreamNetworksTest(unittest.TestCase):
    def test_fixed_vector_actor_and_critic_shapes(self):
        networks = make_flat_upstream_networks(
            observation_size=57,
            goal_size=16,
            action_size=5,
            rep_size=8,
            num_blocks=2,
            hidden_dim=16,
        )
        initialized = networks.init_params(jax.random.PRNGKey(0))
        actor_params = initialized.pop("actor")
        observation = jnp.ones((3, 57))
        goal = jnp.ones((3, 16))
        action = jnp.ones((3, 5))
        state_action_repr = networks.apply_sa(
            initialized, observation, action
        )
        goal_repr = networks.apply_goal(initialized, goal)
        mean, log_std = networks.actor.apply(
            actor_params, observation, goal_repr
        )
        self.assertEqual(state_action_repr.shape, (3, 8))
        self.assertEqual(goal_repr.shape, (3, 8))
        self.assertEqual(mean.shape, (3, 5))
        self.assertEqual(log_std.shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
