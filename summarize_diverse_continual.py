#!/usr/bin/env python3
"""Build the Sequence A per-task forward-transfer/AUC comparison CSV."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

from continual.eval_logging import read_eval_rows
from diverse_continual_experiment_configs import build_configs


_METHODS = {
    "stablecrl_reset_reset_diverse_sequence_a": "reset_stablecrl",
    "stablecrl_persistent_persistent_diverse_sequence_a": (
        "persistent_stablecrl"
    ),
    "dcc_residual_diverse_sequence_a": "dcc",
}
_FIELDS = (
    "method",
    "seed",
    "task_index",
    "env_id",
    "initial_success_rate",
    "initial_easy_success_rate",
    "final_success_rate",
    "final_easy_success_rate",
    "success_rate_auc",
    "easy_success_rate_auc",
    "forward_transfer_gain_vs_reset",
    "adaptation_auc_gain_vs_reset",
)


def collect_results(checkpoint_root: Path) -> list[dict[str, Any]]:
    """Read completed task diagonals and add matched-seed reset deltas."""
    results: list[dict[str, Any]] = []
    for config in build_configs():
        method = _METHODS[config["name"]]
        run_dir = checkpoint_root / f"{config['name']}_seed{config['seed']}"
        task_ids = config["task_sequence"].split(",")
        for row in read_eval_rows(run_dir / "continual_eval.jsonl"):
            task_index = int(row["eval_task_index"])
            if int(row["phase_index"]) != task_index:
                continue
            if "adaptation/success_rate_auc" not in row:
                continue
            results.append({
                "method": method,
                "seed": int(config["seed"]),
                "task_index": task_index,
                "env_id": row.get("env_id", task_ids[task_index]),
                "initial_success_rate": row[
                    "forward_transfer/initial_success_rate"
                ],
                "initial_easy_success_rate": row[
                    "forward_transfer/initial_easy_success_rate"
                ],
                "final_success_rate": row[
                    "adaptation/final_success_rate"
                ],
                "final_easy_success_rate": row[
                    "adaptation/final_easy_success_rate"
                ],
                "success_rate_auc": row["adaptation/success_rate_auc"],
                "easy_success_rate_auc": row[
                    "adaptation/easy_success_rate_auc"
                ],
            })

    reset = {
        (row["seed"], row["task_index"]): row
        for row in results
        if row["method"] == "reset_stablecrl"
    }
    for row in results:
        reference = reset.get((row["seed"], row["task_index"]))
        if row["task_index"] == 0 or reference is None:
            row["forward_transfer_gain_vs_reset"] = ""
            row["adaptation_auc_gain_vs_reset"] = ""
            continue
        row["forward_transfer_gain_vs_reset"] = (
            float(row["initial_success_rate"])
            - float(reference["initial_success_rate"])
        )
        row["adaptation_auc_gain_vs_reset"] = (
            float(row["success_rate_auc"])
            - float(reference["success_rate_auc"])
        )
    return sorted(
        results,
        key=lambda row: (row["task_index"], row["seed"], row["method"]),
    )


def write_results(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def upload_results_to_wandb(
    rows: list[dict[str, Any]],
    *,
    project: str,
    entity: str | None,
    group: str,
    mode: str,
) -> None:
    """Upload the completed cross-run comparison after all jobs finish."""
    import wandb

    run = wandb.init(
        project=project,
        entity=entity or None,
        group=group,
        name="sequence_a_forward_transfer_summary",
        job_type="continual-summary",
        mode=mode,
    )
    run.log({
        "forward_transfer/per_task": wandb.Table(
            columns=list(_FIELDS),
            data=[[row.get(field, "") for field in _FIELDS] for row in rows],
        )
    })
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        for field in (
            "forward_transfer_gain_vs_reset",
            "adaptation_auc_gain_vs_reset",
        ):
            values = [
                float(row[field])
                for row in method_rows
                if row.get(field, "") != ""
            ]
            if values:
                run.summary[f"{method}/mean_{field}"] = statistics.fmean(
                    values
                )
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Sequence A task adaptation and forward transfer."
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--upload-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", default="builderbench-stablecrl-dcc"
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-group", default="torch_dcc_diverse_sequence_a"
    )
    parser.add_argument("--wandb-mode", default="online")
    args = parser.parse_args()
    output = args.output or args.checkpoint_root / "forward_transfer_summary.csv"
    rows = collect_results(args.checkpoint_root)
    if not rows:
        parser.error(
            "no completed Sequence A task-adaptation rows were found under "
            f"{args.checkpoint_root}"
        )
    write_results(rows, output)
    print(f"Wrote {len(rows)} per-task rows to {output}")
    if args.upload_wandb:
        upload_results_to_wandb(
            rows,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            mode=args.wandb_mode,
        )
        print("Uploaded the per-task comparison table to W&B")


if __name__ == "__main__":
    main()
