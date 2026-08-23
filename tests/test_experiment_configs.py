import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import experiment_configs


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigsTest(unittest.TestCase):
    def test_batch_is_baseline_first_and_retains_sgcrl_cells(self):
        configs = experiment_configs.build_configs()
        self.assertEqual(len(configs), 66)
        self.assertEqual({config["seed"] for config in configs}, {5, 6, 7})
        self.assertEqual(
            Counter(config["name"] for config in configs),
            {
                "upstream_scaled_crtr_three_stack": 3,
                "upstream_scaled_crtr_four_stack": 3,
                "wrapped_vanilla_crl_three_stack": 3,
                "wrapped_vanilla_crl_four_stack": 3,
                "dcc_single_three_stack": 3,
                "dcc_single_four_stack": 3,
                "crl_reset_reset": 3,
                "crl_persistent_persistent": 3,
                "dcc_persistent_actor_dynamics": 3,
                "dcc_persistent_actor_no_dynamics": 3,
                "dcc_crtr12_three_stack": 3,
                "dcc_crtr12_four_stack": 3,
                "grouped_pad_upstream_three_stack": 3,
                "grouped_pad_upstream_four_stack": 3,
                "semantic_pad_upstream_three_stack": 3,
                "semantic_pad_upstream_four_stack": 3,
                "semantic_set_capacity4_three_stack": 3,
                "semantic_set_capacity4_four_stack": 3,
                "flat_crl_goal_only_1cube": 3,
                "flat_crl_expanding_stack": 3,
                "dcc_goal_only_1cube": 3,
                "dcc_expanding_stack": 3,
            },
        )

        self.assertTrue(
            all(config["runner"] == "stable_crl.py" for config in configs[:6])
        )
        self.assertTrue(
            all(
                config["runner"] == "continual_crl.py"
                for config in configs[6:12]
            )
        )
        reset = configs[18:21]
        persistent = configs[21:24]
        self.assertTrue(
            all(
                config["repetition_factor"] == 12
                for config in configs[:18]
            )
        )
        self.assertTrue(
            all(
                config["actor_lifecycle"] == "reset"
                and config["critic_lifecycle"] == "reset"
                for config in reset
            )
        )
        self.assertTrue(
            all(
                config["repetition_factor"] == 1
                for config in reset + persistent
            )
        )
        self.assertTrue(
            all(
                config["actor_lifecycle"] == "persistent"
                and config["critic_lifecycle"] == "persistent"
                for config in persistent
            )
        )

        dynamics = configs[24:27]
        no_dynamics = configs[27:30]
        crtr = configs[30:36]
        self.assertTrue(all(config["carry_actor"] for config in dynamics))
        self.assertTrue(
            all(config["dcc_dyn_weight"] == 1.0 for config in dynamics)
        )
        self.assertTrue(
            all(config["repetition_factor"] == 1 for config in dynamics)
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
        names = {config["name"] for config in configs}
        self.assertEqual(len(names), 22)
        self.assertTrue(
            all(config["wandb_group"] == config["name"] for config in configs)
        )
        self.assertTrue(
            all(
                config["vanilla_network_type"] == "flat_upstream"
                for config in configs[36:48]
            )
        )
        self.assertTrue(
            all(
                config["observation_layout"] == "grouped"
                for config in configs[36:42]
            )
        )

    def test_cli_counts_and_array_sizes(self):
        def output(*args):
            return subprocess.check_output(
                [sys.executable, "experiment_configs.py", *args],
                cwd=REPO_ROOT,
                text=True,
            ).strip()

        self.assertEqual(output("--total"), "66")
        self.assertEqual(output("--array-max"), "65")
        self.assertEqual(
            output("--array-max", "--tasks-per-gpu", "2"), "32"
        )
        self.assertEqual(
            output("--stage-start", "padding_diagnostics"), "36"
        )
        self.assertEqual(
            output(
                "--stage-array-max", "padding_diagnostics",
                "--tasks-per-gpu", "2",
            ),
            "8",
        )
        setting = output("--setting", "0")
        self.assertIn("NAME=upstream_scaled_crtr_three_stack", setting)
        self.assertIn("SEED=5", setting)
        self.assertIn("DCC_DYN_WEIGHT_AFTER_TASK0=''", setting)

    def _draft_output(self, config_index):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "true",
                    "CONFIG_INDEX": str(config_index),
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
            return completed.stdout

    def test_draft_dry_run_builds_all_runner_families(self):
        upstream = self._draft_output(0)
        self.assertIn("Dry run complete", upstream)
        self.assertIn("stable_crl.py", upstream)
        self.assertIn("--env-id creative-3-task1", upstream)
        self.assertIn("--mjx-impl warp", upstream)
        self.assertNotIn("--task-sequence", upstream)
        self.assertIn(
            "--wandb-group torch_dcc__upstream_scaled_crtr_three_stack",
            upstream,
        )
        self.assertIn("--wandb-project-name builderbench-stablecrl-dcc", upstream)
        self.assertNotIn("--wandb-entity", upstream)

        vanilla = self._draft_output(18)
        self.assertIn("continual_crl.py", vanilla)
        self.assertIn("--actor-lifecycle reset", vanilla)
        self.assertIn("--critic-lifecycle reset", vanilla)
        self.assertIn("--resume", vanilla)

        dcc = self._draft_output(24)
        self.assertIn("continual_dcc.py", dcc)
        self.assertIn("--dcc-dyn-weight 1.0", dcc)
        self.assertIn("--carry-actor", dcc)

        padded = self._draft_output(36)
        self.assertIn("--observation-layout grouped", padded)
        self.assertIn("--vanilla-network-type flat_upstream", padded)
        self.assertIn("--eval-next-task", padded)
        self.assertIn("--log-continual-eval", padded)

    def test_draft_finds_repo_when_slurm_copies_the_script(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = Path(directory) / "jobspool"
            spool.mkdir()
            shutil.copy(REPO_ROOT / "DRAFT.sh", spool / "DRAFT.sh")
            environment = os.environ.copy()
            environment.pop("REPO_DIR", None)
            environment.update(
                {
                    "DRY_RUN": "true",
                    "CONFIG_INDEX": "0",
                    "TASKS_PER_GPU": "1",
                    "PYTHON_BIN": sys.executable,
                    "SCRATCH": str(Path(directory) / "scratch"),
                    "SLURM_SUBMIT_DIR": str(REPO_ROOT),
                }
            )
            completed = subprocess.run(
                ["bash", str(spool / "DRAFT.sh")],
                cwd=spool,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Dry run complete", completed.stdout)
            self.assertIn(f"repo={REPO_ROOT}", completed.stdout)
            self.assertIn(str(REPO_ROOT / "stable_crl.py"), completed.stdout)
            self.assertNotIn("jobspool", completed.stdout)


if __name__ == "__main__":
    unittest.main()
