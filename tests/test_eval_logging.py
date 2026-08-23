import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from continual.eval_logging import (
    continual_eval_run_id,
    continual_scalars,
    log_continual_eval_to_wandb,
    read_eval_rows,
    write_phase_rows,
)


def row(phase, task, success, scope="seen"):
    return {
        "phase_index": phase,
        "eval_task_index": task,
        "eval_scope": scope,
        "critic_head_task_index": None,
        "train_task_global_id": f"train-{phase}",
        "eval_task_global_id": f"task-{task}",
        "eval/episode_success_rate": success,
        "eval/episode_easy_success_rate": success,
    }


class EvalLoggingTest(unittest.TestCase):
    def test_phase_write_is_atomic_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continual_eval.jsonl"
            write_phase_rows(path, phase_index=0, rows=[row(0, 0, 0.5)])
            write_phase_rows(path, phase_index=0, rows=[row(0, 0, 0.75)])
            rows = read_eval_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["eval/episode_success_rate"], 0.75)

    def test_cl_scalars_include_forgetting_bwt_and_forward_transfer(self):
        rows = [
            row(0, 0, 1.0),
            row(0, 1, 0.25, "next_unseen"),
            row(1, 0, 0.5),
            row(1, 1, 0.75),
        ]
        scalars = continual_scalars(rows, phase_index=1)
        self.assertEqual(scalars["continual/mean_seen_success_rate"], 0.625)
        self.assertEqual(scalars["continual/average_forgetting"], 0.5)
        self.assertEqual(scalars["continual/backward_transfer"], -0.5)

    def test_run_id_is_stable_and_recipe_sensitive(self):
        first = continual_eval_run_id({"algorithm": "a", "seed": 5})
        second = continual_eval_run_id({"seed": 5, "algorithm": "a"})
        changed = continual_eval_run_id({"algorithm": "a", "seed": 6})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_wandb_logging_is_boundary_only_and_best_effort(self):
        class FakeRun:
            def __init__(self):
                self.summary = {}
                self.logged = []
                self.finished = False

            def define_metric(self, *args, **kwargs):
                pass

            def log(self, payload):
                self.logged.append(payload)

            def finish(self):
                self.finished = True

        run = FakeRun()
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.init = lambda **kwargs: run
        fake_wandb.Table = lambda **kwargs: kwargs
        args = SimpleNamespace(
            track=True,
            wandb_name_tag="test",
            wandb_project_name="project",
            wandb_entity="entity",
            wandb_mode="offline",
            wandb_dir=".",
            wandb_group="group",
        )
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            uploaded = log_continual_eval_to_wandb(
                args=args,
                recipe={"algorithm": "test", "seed": 5},
                rows=[row(0, 0, 1.0)],
                phase_index=0,
            )
        self.assertTrue(uploaded)
        self.assertTrue(run.finished)
        self.assertEqual(len(run.logged), 1)
        self.assertIn("continual/success_matrix", run.logged[0])


if __name__ == "__main__":
    unittest.main()
