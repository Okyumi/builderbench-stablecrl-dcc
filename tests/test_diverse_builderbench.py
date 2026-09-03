"""Contracts for the direct-cube adaptation of BuilderBench tasks."""
import unittest
import importlib.util
from types import SimpleNamespace

import numpy as np

from builderbench.task_catalog import (
    SEQUENCE_A,
    SEQUENCE_B,
    SOURCE_REVISION,
    TASKS,
    WORKSPACE_TRANSLATION,
    get_direct_builder_task,
)
from continual.semantic_layout import SemanticLayout
from continual.task_manifest import build_manifest


HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


EXPECTED_SEQUENCE_A = (
    "builderbench-direct-1-task1",
    "builderbench-direct-1-task2",
    "builderbench-direct-2-task1",
    "builderbench-direct-2-task3",
    "builderbench-direct-3-task2",
    "builderbench-direct-3-task5",
    "builderbench-direct-5-task3",
    "builderbench-direct-7-task6",
    "builderbench-direct-8-task2",
)

EXPECTED_SEQUENCE_B = (
    "builderbench-direct-2-task1",
    "builderbench-direct-3-task1",
    "builderbench-direct-3-task2",
    "builderbench-direct-5-task2",
    "builderbench-direct-5-task3",
    "builderbench-direct-7-task2",
    "builderbench-direct-7-task7",
    "builderbench-direct-8-task2",
)


class DirectBuilderBenchCatalogTest(unittest.TestCase):
    def test_both_requested_sequences_are_registered(self):
        self.assertEqual(SEQUENCE_A, EXPECTED_SEQUENCE_A)
        self.assertEqual(SEQUENCE_B, EXPECTED_SEQUENCE_B)
        self.assertEqual(len(SEQUENCE_A), 9)
        self.assertEqual(len(SEQUENCE_B), 8)
        self.assertTrue(set(SEQUENCE_A) | set(SEQUENCE_B) <= set(TASKS))
        self.assertEqual(SOURCE_REVISION, "RajGhugare19/builderbench@de9130b98323")

    def test_source_geometry_and_masks_match_builderbench(self):
        permute = get_direct_builder_task("builderbench-direct-2-task3")
        np.testing.assert_allclose(
            permute.source_goals,
            [[0.45, 0.08, 0.02], [0.45, -0.08, 0.02]],
        )
        np.testing.assert_array_equal(permute.goal_mask, [1, 1])
        self.assertTrue(permute.ordered_reward)

        tokyo = get_direct_builder_task("builderbench-direct-7-task2")
        np.testing.assert_array_equal(
            tokyo.goal_mask, [0, 0, 1, 0, 1, 1, 1]
        )
        portal = get_direct_builder_task("builderbench-direct-8-task2")
        np.testing.assert_array_equal(
            portal.goal_mask, [1, 1, 1, 1, 1, 1, 0, 0]
        )

    def test_workspace_adaptation_is_one_rigid_translation(self):
        for task in TASKS.values():
            np.testing.assert_allclose(
                task.goals - task.source_goals,
                np.broadcast_to(WORKSPACE_TRANSLATION, task.goals.shape),
            )
            source_pairwise = (
                task.source_goals[:, None] - task.source_goals[None, :]
            )
            adapted_pairwise = task.goals[:, None] - task.goals[None, :]
            np.testing.assert_allclose(
                adapted_pairwise, source_pairwise, atol=2e-8
            )
            self.assertTrue(np.all(task.goals >= [-0.05, -0.35, 0.0]))
            self.assertTrue(np.all(task.goals <= [0.45, 0.35, 0.5]))

    @unittest.skipUnless(HAS_MUJOCO, "MuJoCo is not installed")
    def test_direct_env_configuration_keeps_action_and_pd_contract(self):
        from builderbench.env_utils import make_env

        args = SimpleNamespace(
            env_id="builderbench-direct-2-task3",
            env_episode_length=None,
            env_early_termination=False,
            permutation_invariant_reward=True,
            num_envs=1,
        )
        env_class, config = make_env(args)
        self.assertEqual(env_class.__name__, "CreativeCube")
        self.assertEqual(config.num_cubes, 2)
        self.assertEqual(config.episode_length, 200)
        self.assertFalse(config.permutation_invariant_reward)
        self.assertEqual(config.direct_builderbench_task, args.env_id)

    def test_sequence_a_has_fixed_semantic_shapes(self):
        layout = SemanticLayout(max_cubes=8)
        self.assertEqual(layout.observation_size, 120)
        self.assertEqual(layout.goal_size, 32)
        for env_id in SEQUENCE_A:
            task = get_direct_builder_task(env_id)
            layout.validate_num_cubes(task.num_cubes)

    def test_goal_mask_is_in_padded_goal_and_manifest(self):
        task = get_direct_builder_task("builderbench-direct-8-task2")
        layout = SemanticLayout(max_cubes=8)
        packed = layout.pack_goal(
            task.goals.reshape(-1), 8, mask=task.goal_mask
        )
        positions, mask = layout.split_goal(packed)
        np.testing.assert_allclose(positions[:6], task.goals[:6])
        np.testing.assert_array_equal(positions[6:], 0)
        np.testing.assert_array_equal(mask, task.goal_mask)

        manifest = build_manifest(
            [(task.env_id, task.goals, task.goal_mask)],
            task_data_version="builderbench-de9130-direct-v1",
        )
        self.assertEqual(manifest[0].goal_mask, tuple(task.goal_mask))
        self.assertIsNotNone(manifest[0].goal_mask_hash)


if __name__ == "__main__":
    unittest.main()
