"""Tests for the matched continual algorithm benchmark."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import continual_experiment_configs


REPO_ROOT = Path(__file__).resolve().parents[1]


class ContinualExperimentConfigsTest(unittest.TestCase):
    def test_tracks_match_every_non_lifecycle_setting(self):
        configs = continual_experiment_configs.build_configs()
        self.assertEqual(len(configs), 27)
        self.assertEqual({config["seed"] for config in configs}, {5, 6, 7})

        for start, task_count, max_cubes in (
            (0, 2, 1),
            (9, 4, 4),
            (18, 5, 5),
        ):
            track = configs[start:start + 9]
            for seed_offset in range(0, 9, 3):
                reset, persistent, dcc = track[seed_offset:seed_offset + 3]
                self.assertEqual(
                    len(reset["task_sequence"].split(",")),
                    task_count,
                )
                self.assertEqual(
                    {cell["repetition_factor"] for cell in (
                        reset, persistent, dcc
                    )},
                    {12},
                )
                self.assertEqual(
                    {cell["max_cubes"] for cell in (
                        reset, persistent, dcc
                    )},
                    {max_cubes},
                )
                self.assertEqual(reset["actor_lifecycle"], "reset")
                self.assertEqual(reset["critic_lifecycle"], "reset")
                self.assertEqual(
                    persistent["actor_lifecycle"], "persistent"
                )
                self.assertEqual(
                    persistent["critic_lifecycle"], "persistent"
                )
                self.assertEqual(reset["runner"], "continual_crl.py")
                self.assertEqual(persistent["runner"], "continual_crl.py")
                self.assertEqual(dcc["runner"], "continual_dcc.py")
                self.assertEqual(
                    reset["vanilla_network_type"], "flat_upstream"
                )
                self.assertTrue(dcc["carry_actor"])
                self.assertTrue(dcc["dcc_carry_shared"])

    def test_cli_stages_and_smoke_order(self):
        def output(*args):
            return subprocess.check_output(
                [sys.executable, "continual_experiment_configs.py", *args],
                cwd=REPO_ROOT,
                text=True,
            ).strip()

        self.assertEqual(output("--total"), "27")
        self.assertEqual(output("--stage-start", "core"), "0")
        self.assertEqual(output("--stage-end", "core"), "17")
        self.assertEqual(output("--stage-start", "expanding_5stack"), "18")
        self.assertEqual(output("--stage-end", "expanding_5stack"), "26")
        self.assertEqual(output("--stage-end", "smoke_goal"), "2")
        setting = output("--setting", "2")
        self.assertIn("NAME=dcc_residual_goal_only", setting)
        self.assertIn("REPETITION_FACTOR=12", setting)

    def test_draft_builds_matched_smoke_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = os.environ.copy()
            environment.update({
                "DRY_RUN": "true",
                "CONFIG_INDEX": "1",
                "TASKS_PER_GPU": "1",
                "PYTHON_BIN": sys.executable,
                "SCRATCH": str(root),
                "LOG_ROOT": str(root / "logs"),
                "CHECKPOINT_ROOT": str(root / "checkpoints"),
                "WANDB_DIR": str(root / "wandb"),
                "CONFIG_REGISTRY": "continual_experiment_configs.py",
                "EXPERIMENT_STAGE": "smoke_goal",
                "WANDB_GROUP_PREFIX": "torch_dcc_continual_smoke",
                "RUN_TEST_PREFLIGHT": "false",
                "BASE_STEPS": "2097152",
                "STEPS_PER_TASK": "2097152",
                "NUM_ENVS": "256",
                "NUM_EVAL_ENVS": "32",
                "MAX_REPLAY_SIZE": "512",
                "MIN_REPLAY_SIZE": "128",
                "NUM_EVAL_STEPS": "4",
                "NUM_RESET_STEPS": "4",
            })
            completed = subprocess.run(
                ["bash", "DRAFT.sh"],
                cwd=REPO_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        output = completed.stdout
        self.assertIn("continual_crl.py", output)
        self.assertIn("--actor-lifecycle persistent", output)
        self.assertIn("--critic-lifecycle persistent", output)
        self.assertIn("--repetition-factor 12", output)
        self.assertIn(
            r"--task-sequence creative-1-task1\,creative-1-task2",
            output,
        )


if __name__ == "__main__":
    unittest.main()
