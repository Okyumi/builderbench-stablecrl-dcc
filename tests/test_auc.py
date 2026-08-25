"""Tests for online learning-curve AUC reporting."""
import unittest

from continual_auc import OnlineNormalizedAUC


class OnlineNormalizedAUCTest(unittest.TestCase):
    def test_trapezoidal_auc_includes_zero_origin(self):
        auc = OnlineNormalizedAUC()
        self.assertAlmostEqual(auc.update(10.0, 0.5), 0.25)
        self.assertAlmostEqual(auc.update(20.0, 1.0), 0.5)

    def test_rejects_invalid_observations(self):
        auc = OnlineNormalizedAUC()
        with self.assertRaises(ValueError):
            auc.update(0.0, 0.5)
        with self.assertRaises(ValueError):
            auc.update(1.0, 1.1)
        with self.assertRaises(ValueError):
            auc.update(float("nan"), 0.5)

    def test_rejects_nonincreasing_steps(self):
        auc = OnlineNormalizedAUC()
        auc.update(10.0, 0.5)
        with self.assertRaises(ValueError):
            auc.update(10.0, 0.6)


if __name__ == "__main__":
    unittest.main()
