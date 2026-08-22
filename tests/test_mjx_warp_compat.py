import importlib.metadata
import unittest

import mujoco.mjx.warp as mjxw
import warp
import warp.types as warp_types

from builderbench.mjx_warp_compat import apply_mjx_warp_compat


class MjxWarpCompatTest(unittest.TestCase):
    def test_pinned_warp_matches_mujoco_37(self):
        self.assertEqual(importlib.metadata.version("warp-lang"), "1.12.0")
        self.assertEqual(importlib.metadata.version("mujoco"), "3.7.0")
        self.assertEqual(importlib.metadata.version("mujoco-mjx"), "3.7.0")
        self.assertEqual(importlib.metadata.version("mujoco-warp"), "3.7.0")

    def test_graph_mode_and_dtype_maps_after_patch(self):
        apply_mjx_warp_compat()
        self.assertTrue(hasattr(mjxw.types.GraphMode, "WARP"))
        self.assertEqual(mjxw.types.GraphMode.WARP.name, "WARP")
        self.assertTrue(hasattr(warp_types, "warp_type_to_np_dtype"))
        self.assertIsInstance(warp_types.warp_type_to_np_dtype, dict)
        self.assertGreater(len(warp_types.warp_type_to_np_dtype), 0)
        self.assertTrue(hasattr(warp, "__version__"))


if __name__ == "__main__":
    unittest.main()
