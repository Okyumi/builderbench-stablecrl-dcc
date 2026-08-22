import unittest

import numpy as np

from continual_crl import Args, _checkpoint_recipe, _transfer_carry


class ContinualCRLTest(unittest.TestCase):
    def setUp(self):
        self.carry = {
            "actor": {"w": np.array([1.0])},
            "critic": {"w": np.array([2.0])},
        }

    def test_reset_reset_hides_both_parameter_sets(self):
        transferred = _transfer_carry(
            Args(actor_lifecycle="reset", critic_lifecycle="reset"),
            self.carry,
        )
        self.assertIsNone(transferred["actor"])
        self.assertIsNone(transferred["critic"])

    def test_persistent_persistent_carries_both_parameter_sets(self):
        transferred = _transfer_carry(
            Args(
                actor_lifecycle="persistent",
                critic_lifecycle="persistent",
            ),
            self.carry,
        )
        self.assertIs(transferred["actor"], self.carry["actor"])
        self.assertIs(transferred["critic"], self.carry["critic"])

    def test_lifecycle_is_checkpointed_in_recipe(self):
        recipe = _checkpoint_recipe(
            Args(
                actor_lifecycle="persistent",
                critic_lifecycle="persistent",
            )
        )
        self.assertEqual(recipe["actor_lifecycle"], "persistent")
        self.assertEqual(recipe["critic_lifecycle"], "persistent")


if __name__ == "__main__":
    unittest.main()
