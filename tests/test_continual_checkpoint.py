import tempfile
import unittest
from pathlib import Path

import numpy as np

from continual_dcc import (
    Args,
    _checkpoint_recipe,
    _load_boundary_checkpoint,
    _save_boundary_checkpoint,
)


class ContinualCheckpointTest(unittest.TestCase):
    def test_recipe_round_trip_and_mismatch_rejection(self):
        recipe = _checkpoint_recipe(Args())
        carry = {"value": np.array([1.0, 2.0], dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_00.pkl"
            _save_boundary_checkpoint(
                path,
                carry=carry,
                global_id="version:1:1:goal",
                task_index=0,
                recipe=recipe,
            )
            restored = _load_boundary_checkpoint(
                path,
                global_id="version:1:1:goal",
                task_index=0,
                recipe=recipe,
            )
            np.testing.assert_array_equal(restored["value"], carry["value"])

            changed = dict(recipe)
            changed["dcc_combine_mode"] = "concat"
            with self.assertRaisesRegex(ValueError, "recipe mismatch"):
                _load_boundary_checkpoint(
                    path,
                    global_id="version:1:1:goal",
                    task_index=0,
                    recipe=changed,
                )


if __name__ == "__main__":
    unittest.main()
