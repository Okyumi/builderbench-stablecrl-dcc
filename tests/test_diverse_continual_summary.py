"""Tests for the Sequence A forward-transfer result summary."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from summarize_diverse_continual import collect_results, upload_results_to_wandb


def _write_row(
    root: Path,
    name: str,
    seed: int,
    task_index: int,
    initial: float,
    auc: float,
) -> None:
    directory = root / f"{name}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    row = {
        "phase_index": task_index,
        "eval_task_index": task_index,
        "env_id": f"task-{task_index}",
        "forward_transfer/initial_success_rate": initial,
        "forward_transfer/initial_easy_success_rate": initial,
        "adaptation/final_success_rate": 0.8,
        "adaptation/final_easy_success_rate": 0.9,
        "adaptation/success_rate_auc": auc,
        "adaptation/easy_success_rate_auc": auc,
    }
    with (directory / "continual_eval.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


class DiverseContinualSummaryTest(unittest.TestCase):
    def test_matched_seed_forward_transfer_and_auc_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reset = "stablecrl_reset_reset_diverse_sequence_a"
            persistent = (
                "stablecrl_persistent_persistent_diverse_sequence_a"
            )
            _write_row(root, reset, 5, 1, initial=0.1, auc=0.4)
            _write_row(root, persistent, 5, 1, initial=0.3, auc=0.7)
            rows = collect_results(root)

        by_method = {row["method"]: row for row in rows}
        self.assertAlmostEqual(
            by_method["persistent_stablecrl"][
                "forward_transfer_gain_vs_reset"
            ],
            0.2,
        )
        self.assertAlmostEqual(
            by_method["persistent_stablecrl"][
                "adaptation_auc_gain_vs_reset"
            ],
            0.3,
        )

    def test_uploads_table_and_aggregate_summaries_to_wandb(self):
        class FakeRun:
            def __init__(self):
                self.logged = []
                self.summary = {}
                self.finished = False

            def log(self, payload):
                self.logged.append(payload)

            def finish(self):
                self.finished = True

        run = FakeRun()
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.init = lambda **kwargs: run
        fake_wandb.Table = lambda **kwargs: kwargs
        rows = [{
            "method": "dcc",
            "seed": 5,
            "task_index": 1,
            "env_id": "task-1",
            "forward_transfer_gain_vs_reset": 0.2,
            "adaptation_auc_gain_vs_reset": 0.3,
        }]
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            upload_results_to_wandb(
                rows,
                project="project",
                entity=None,
                group="group",
                mode="offline",
            )
        self.assertTrue(run.finished)
        self.assertIn("forward_transfer/per_task", run.logged[0])
        self.assertAlmostEqual(
            run.summary["dcc/mean_forward_transfer_gain_vs_reset"], 0.2
        )


if __name__ == "__main__":
    unittest.main()
