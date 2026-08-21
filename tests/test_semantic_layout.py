import unittest

import numpy as np

from continual.semantic_layout import RAW_CUBE_FEATURE_DIM, SemanticLayout
from continual.task_manifest import (
    build_manifest,
    canonical_goal,
    canonical_goal_hash,
)


def grouped_observation(cubes, select=0.5):
    cubes = np.asarray(cubes, dtype=np.float32)
    groups = [
        cubes[:, :3].reshape(-1),
        cubes[:, 3:7].reshape(-1),
        cubes[:, 7:10].reshape(-1),
        cubes[:, 10:13].reshape(-1),
    ]
    return np.concatenate([*groups, np.array([select], np.float32)])


class SemanticLayoutTest(unittest.TestCase):
    def setUp(self):
        self.layout = SemanticLayout(max_cubes=4)
        self.cubes = np.arange(2 * RAW_CUBE_FEATURE_DIM, dtype=np.float32).reshape(
            2, RAW_CUBE_FEATURE_DIM
        )

    def test_observation_round_trip(self):
        raw = grouped_observation(self.cubes)
        packed = self.layout.pack_observation(raw, 2)
        np.testing.assert_array_equal(
            self.layout.unpack_observation(packed, 2), raw
        )
        cubes, mask = self.layout.split_observation(packed)
        np.testing.assert_array_equal(mask, np.array([1, 1, 0, 0]))
        np.testing.assert_array_equal(cubes[:2, -1], np.array([0, 1]))

    def test_goal_round_trip_and_mask(self):
        raw = np.array([0, 0, 0, 0, 0, 0.04], dtype=np.float32)
        packed = self.layout.pack_goal(raw, 2)
        positions, mask = self.layout.split_goal(packed)
        np.testing.assert_array_equal(mask, np.array([1, 1, 0, 0]))
        np.testing.assert_array_equal(positions[2:], 0)
        np.testing.assert_array_equal(self.layout.unpack_goal(packed, 2), raw)

    def test_cube_permutation_only_permutes_valid_slots(self):
        packed = self.layout.pack_observation(
            grouped_observation(self.cubes), 2
        )
        permuted = self.layout.pack_observation(
            grouped_observation(self.cubes[::-1], select=-0.5), 2
        )
        slots, mask = self.layout.split_observation(packed)
        permuted_slots, permuted_mask = self.layout.split_observation(
            permuted
        )
        np.testing.assert_array_equal(slots[:2][::-1], permuted_slots[:2])
        np.testing.assert_array_equal(mask, permuted_mask)
        # A shared encoder followed by sum/mean pooling sees the same set.
        np.testing.assert_array_equal(
            slots[:2].sum(axis=0), permuted_slots[:2].sum(axis=0)
        )

    def test_rejects_unversioned_delta_control_shape(self):
        raw = np.zeros(2 * RAW_CUBE_FEATURE_DIM + 2 * 4 + 1, np.float32)
        with self.assertRaisesRegex(ValueError, "delta_control"):
            self.layout.pack_observation(raw, 2)


class TaskIdentityTest(unittest.TestCase):
    def test_hash_is_permutation_and_horizontal_translation_invariant(self):
        goal = np.array([[0.0, 0.0, 0.0], [0.04, 0.0, 0.04]])
        translated_permuted = goal[::-1] + np.array([0.12, -0.08, 0.0])
        self.assertEqual(
            canonical_goal_hash(goal),
            canonical_goal_hash(translated_permuted),
        )

    def test_height_is_semantic(self):
        place = np.array([[0.0, 0.0, 0.0]])
        pick = np.array([[0.0, 0.0, 0.08]])
        self.assertNotEqual(
            canonical_goal_hash(place), canonical_goal_hash(pick)
        )

    def test_manifest_has_stable_global_ids(self):
        tasks = [
            ("creative-1-task1", np.array([[0.0, 0.0, 0.0]])),
            ("creative-1-task2", np.array([[0.0, 0.0, 0.08]])),
        ]
        first = build_manifest(tasks, task_data_version="test-v1")
        second = build_manifest(tasks, task_data_version="test-v1")
        self.assertEqual(
            [record.global_id for record in first],
            [record.global_id for record in second],
        )


if __name__ == "__main__":
    unittest.main()
