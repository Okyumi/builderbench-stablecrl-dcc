"""Configuration and launcher checks for diverse Sequence A."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import diverse_continual_experiment_configs as registry
from builderbench.task_catalog import SEQUENCE_A


REPO_ROOT = Path(__file__).resolve().parents[1]


class DiverseContinualConfigsTest(unittest.TestCase):
    def test_matched_sequence_a_cells(self):
        configs = registry.build_configs()
        self.assertEqual(len(configs), 9)
        self.assertEqual({config["seed"] for config in configs}, {5, 6, 7})
        expected_sequence = ",".join(SEQUENCE_A)
        for offset in range(0, 9, 3):
            reset, persistent, dcc = configs[offset:offset + 3]
            self.assertEqual(
                {cell["task_sequence"] for cell in (reset, persistent, dcc)},
                {expected_sequence},
            )
            self.assertEqual(
                {cell["max_cubes"] for cell in (reset, persistent, dcc)},
                {8},
            )
            self.assertEqual(
                {cell["pd_duration"] for cell in (reset, persistent, dcc)},
                {5},
            )
            self.assertEqual(
                {cell["repetition_factor"] for cell in (
                    reset, persistent, dcc
                )},
                {12},
            )
            self.assertEqual(
                {cell["task_data_version"] for cell in (
                    reset, persistent, dcc
                )},
                {registry.TASK_DATA_VERSION},
            )
            self.assertEqual(reset["actor_lifecycle"], "reset")
            self.assertEqual(reset["critic_lifecycle"], "reset")
            self.assertEqual(persistent["actor_lifecycle"], "persistent")
            self.assertEqual(persistent["critic_lifecycle"], "persistent")
            self.assertEqual(dcc["runner"], "continual_dcc.py")
            for cell in (reset, persistent, dcc):
                self.assertFalse(cell["eval_next_task"])
                self.assertFalse(cell["eval_previous_tasks"])
                self.assertFalse(cell["report_retention_metrics"])

    def test_registry_cli(self):
        def output(*args):
            return subprocess.check_output(
                [sys.executable, "diverse_continual_experiment_configs.py", *args],
                cwd=REPO_ROOT,
                text=True,
            ).strip()

        self.assertEqual(output("--total"), "9")
        self.assertEqual(output("--stage-start", "smoke"), "0")
        self.assertEqual(output("--stage-end", "smoke"), "2")
        self.assertEqual(output("--stage-end", "sequence_a"), "8")
        setting = output("--setting", "2")
        self.assertIn("RUNNER=continual_dcc.py", setting)
        self.assertIn("MAX_CUBES=8", setting)
        self.assertIn("PD_DURATION=5", setting)
        self.assertIn(
            "TASK_DATA_VERSION=builderbench-de9130-direct-v1", setting
        )
        self.assertIn(SEQUENCE_A[0], setting)
        self.assertIn(SEQUENCE_A[-1], setting)

    def test_launcher_dry_run_uses_new_task_data_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = os.environ.copy()
            environment.update({
                "DRY_RUN": "true",
                "CONFIG_INDEX": "2",
                "PYTHON_BIN": sys.executable,
                "SCRATCH": str(root),
                "LOG_ROOT": str(root / "logs"),
                "CHECKPOINT_ROOT": str(root / "checkpoints"),
                "WANDB_DIR": str(root / "wandb"),
                "EXPERIMENT_STAGE": "smoke",
                "RUN_TEST_PREFLIGHT": "false",
            })
            completed = subprocess.run(
                ["bash", "DRAFT_DIVERSE_CONTINUAL.sh"],
                cwd=REPO_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        output = completed.stdout
        self.assertIn("diverse_continual_experiment_configs.py", output)
        self.assertIn("--task-data-version builderbench-de9130-direct-v1", output)
        self.assertIn("--max-cubes 8", output)
        self.assertIn("--pd-duration 5", output)
        self.assertIn("--no-eval-next-task", output)
        self.assertIn("--no-eval-previous-tasks", output)
        self.assertIn("--no-report-retention-metrics", output)
        self.assertIn(SEQUENCE_A[0], output)
        self.assertIn(SEQUENCE_A[-1], output)


if __name__ == "__main__":
    unittest.main()
