import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import experiment_configs


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigsTest(unittest.TestCase):
    def test_active_batch_matches_sgcrl_structure(self):
        configs = experiment_configs.build_configs()
        self.assertEqual(len(configs), 12)
        self.assertEqual({config["seed"] for config in configs}, {5, 6, 7})
        self.assertEqual(
            Counter(config["name"] for config in configs),
            {
                "dcc_persistent_actor_dynamics": 3,
                "dcc_persistent_actor_no_dynamics": 3,
                "dcc_crtr12_three_stack": 3,
                "dcc_crtr12_four_stack": 3,
            },
        )

        dynamics = configs[:3]
        no_dynamics = configs[3:6]
        crtr = configs[6:]
        self.assertTrue(all(config["carry_actor"] for config in dynamics))
        self.assertTrue(
            all(config["dcc_dyn_weight"] == 1.0 for config in dynamics)
        )
        self.assertTrue(
            all(config["dcc_dyn_weight"] == 0.0 for config in no_dynamics)
        )
        self.assertTrue(
            all(config["repetition_factor"] == 12 for config in crtr)
        )
        self.assertTrue(
            all(len(config["task_sequence"].split(",")) == 1 for config in crtr)
        )

    def test_cli_counts_and_array_sizes(self):
        def output(*args):
            return subprocess.check_output(
                [sys.executable, "experiment_configs.py", *args],
                cwd=REPO_ROOT,
                text=True,
            ).strip()

        self.assertEqual(output("--total"), "12")
        self.assertEqual(output("--array-max"), "11")
        self.assertEqual(
            output("--array-max", "--tasks-per-gpu", "2"), "5"
        )
        setting = output("--setting", "0")
        self.assertIn("NAME=dcc_persistent_actor_dynamics", setting)
        self.assertIn("SEED=5", setting)
        self.assertIn("DCC_DYN_WEIGHT_AFTER_TASK0=''", setting)

    def test_draft_dry_run_builds_resume_safe_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "true",
                    "CONFIG_INDEX": "0",
                    "TASKS_PER_GPU": "1",
                    "PYTHON_BIN": sys.executable,
                    "SCRATCH": str(root),
                    "LOG_ROOT": str(root / "logs"),
                    "CHECKPOINT_ROOT": str(root / "checkpoints"),
                    "WANDB_DIR": str(root / "wandb"),
                }
            )
            completed = subprocess.run(
                ["bash", "DRAFT.sh"],
                cwd=REPO_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            output = completed.stdout
            self.assertIn("Dry run complete", output)
            self.assertIn("--dcc-dyn-weight 1.0", output)
            self.assertIn("--carry-actor", output)
            self.assertIn(
                str(
                    root
                    / "checkpoints"
                    / "dcc_persistent_actor_dynamics_seed5"
                ),
                output,
            )


if __name__ == "__main__":
    unittest.main()
