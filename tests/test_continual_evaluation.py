import unittest

from continual.evaluation import run_repeated_evaluation


class FakeEvaluator:
    def __init__(self):
        self.index = 0

    def run_evaluation(self, policy_params, training_metrics):
        del policy_params, training_metrics
        values = (0.5, 0.75, 1.0)
        value = values[self.index]
        self.index += 1
        return {
            "eval/episode_success_rate": value,
            "eval/episode_reward": 10.0 * value,
            "eval/walltime": float(self.index),
        }


class ContinualEvaluationTest(unittest.TestCase):
    def test_repeated_evaluation_reports_mean_std_and_episode_count(self):
        result = run_repeated_evaluation(
            FakeEvaluator(),
            policy_params={},
            repeats=3,
            num_eval_envs=128,
        )
        self.assertEqual(result["eval/repeats"], 3)
        self.assertEqual(result["eval/num_episodes"], 384)
        self.assertEqual(result["eval/episode_success_rate"], 0.75)
        self.assertAlmostEqual(
            result["eval/episode_success_rate_std"], 0.2041241452
        )
        self.assertEqual(result["eval/walltime"], 3.0)

    def test_repeats_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            run_repeated_evaluation(
                FakeEvaluator(),
                policy_params={},
                repeats=0,
                num_eval_envs=128,
            )


if __name__ == "__main__":
    unittest.main()
