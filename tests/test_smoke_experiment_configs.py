"""Tests for the dedicated Torch residual-DCC smoke batch."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import smoke_experiment_configs


REPO_ROOT = Path(__file__).resolve().parents[1]


class SmokeExperimentConfigsTest(unittest.TestCase):
    def test_smoke_cells_cover_parity_and_continual_transfer(self):
        configs = smoke_experiment_configs.build_configs()
        self.assertEqual(len(configs), 3)
        self.assertEqual(
            [config["name"] for config in configs],
            [
                "smoke_dcc_residual_three_stack",
                "smoke_dcc_residual_four_stack",
                "smoke_dcc_residual_two_task_transfer",
            ],
        )
        self.assertTrue(
            all(config["runner"] == "continual_dcc.py" for config in configs)
        )
        self.assertTrue(
            all(
                config["architecture"] == "block"
                and config["num_blocks"] == 8
                and config["hidden_dim"] == 1024
                and config["dcc_task_width"] == 256
                and config["dcc_task_depth"] == 4
                for config in configs
            )
        )
        self.assertTrue(all(config["max_cubes"] == 4 for config in configs))
        self.assertTrue(
            all(config["continual_eval_repeats"] == 2 for config in configs)
        )
        self.assertTrue(
            all("dcc_dyn_weight" not in config for config in configs)
        )

        for config in configs[:2]:
            self.assertEqual(len(config["task_sequence"].split(",")), 1)
            self.assertEqual(config["repetition_factor"], 12)
            self.assertFalse(config["carry_actor"])

        transfer = configs[2]
        self.assertEqual(
            transfer["task_sequence"],
            "creative-1-task1,creative-2-task1",
        )
        self.assertTrue(transfer["carry_actor"])
        self.assertTrue(transfer["dcc_carry_shared"])
        self.assertEqual(transfer["repetition_factor"], 1)

    def test_smoke_registry_cli(self):
        def output(*args):
            return subprocess.check_output(
                [sys.executable, "smoke_experiment_configs.py", *args],
                cwd=REPO_ROOT,
                text=True,
            ).strip()

        self.assertEqual(output("--total"), "3")
        self.assertEqual(output("--array-max"), "2")
        self.assertEqual(output("--stage-start", "smoke"), "0")
        self.assertEqual(output("--stage-end", "smoke"), "2")
        self.assertEqual(
            output(
                "--stage-array-max",
                "smoke",
                "--tasks-per-gpu",
                "2",
            ),
            "1",
        )
        setting = output("--setting", "2")
        self.assertIn("RUNNER=continual_dcc.py", setting)
        self.assertIn(
            "TASK_SEQUENCE=creative-1-task1,creative-2-task1",
            setting,
        )
        self.assertIn("CONTINUAL_EVAL_REPEATS=2", setting)
        self.assertNotIn("DCC_DYN", setting)

    def _launcher_output(
        self, launcher: Path, *, submit_dir: Path | None = None
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = os.environ.copy()
            environment.pop("REPO_DIR", None)
            environment.update(
                {
                    "DRY_RUN": "true",
                    "CONFIG_INDEX": "2",
                    "PYTHON_BIN": sys.executable,
                    "SCRATCH": str(root),
                    "LOG_ROOT": str(root / "logs"),
                    "CHECKPOINT_ROOT": str(root / "checkpoints"),
                    "WANDB_DIR": str(root / "wandb"),
                }
            )
            if submit_dir is not None:
                environment["SLURM_SUBMIT_DIR"] = str(submit_dir)
            completed = subprocess.run(
                ["bash", str(launcher)],
                cwd=launcher.parent,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout

    def test_smoke_launcher_builds_short_production_shape_command(self):
        output = self._launcher_output(REPO_ROOT / "DRAFT_DCC_SMOKE.sh")
        self.assertIn("Dry run complete", output)
        self.assertIn("smoke_experiment_configs.py", output)
        self.assertIn("continual_dcc.py", output)
        self.assertIn("--base-steps 2097152", output)
        self.assertIn("--steps-per-task 2097152", output)
        self.assertIn("--num-envs 256", output)
        self.assertIn("--num-eval-envs 32", output)
        self.assertIn("--max-replay-size 512", output)
        self.assertIn("--min-replay-size 128", output)
        self.assertIn("--architecture block", output)
        self.assertIn("--num-blocks 8", output)
        self.assertIn("--hidden-dim 1024", output)
        self.assertIn("--continual-eval-repeats 2", output)
        self.assertIn(
            "--wandb-group "
            "torch_dcc_smoke__smoke_dcc_residual_two_task_transfer",
            output,
        )
        self.assertNotIn("dcc-dyn", output)

    def test_smoke_launcher_finds_repo_from_slurm_spool(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = Path(directory) / "jobspool"
            spool.mkdir()
            launcher = spool / "DRAFT_DCC_SMOKE.sh"
            shutil.copy(REPO_ROOT / "DRAFT_DCC_SMOKE.sh", launcher)
            output = self._launcher_output(launcher, submit_dir=REPO_ROOT)
        self.assertIn("Dry run complete", output)
        self.assertIn(f"repo={REPO_ROOT}", output)
        self.assertNotIn("jobspool/smoke_experiment_configs.py", output)


if __name__ == "__main__":
    unittest.main()
