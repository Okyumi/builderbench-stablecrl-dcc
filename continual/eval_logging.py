"""Crash-safe continual evaluation storage and lightweight W&B logging."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


_MATRIX_METRICS = (
    "eval/episode_success_rate",
    "eval/episode_easy_success_rate",
    "eval/episode_reward",
    "eval/episode_obj_goal_dist",
    "eval/avg_episode_length",
)
_MATRIX_STD_METRICS = tuple(f"{metric}_std" for metric in _MATRIX_METRICS)
_EVAL_COUNT_METRICS = ("eval/repeats", "eval/num_episodes")
_ADAPTATION_METRICS = (
    "forward_transfer/initial_success_rate",
    "forward_transfer/initial_easy_success_rate",
    "adaptation/final_success_rate",
    "adaptation/final_easy_success_rate",
    "adaptation/success_rate_auc",
    "adaptation/easy_success_rate_auc",
    "adaptation/budget_env_steps",
)


def read_eval_rows(path: Path) -> list[dict[str, Any]]:
    """Read an evaluation JSONL file written by :func:`write_phase_rows`."""
    if not path.is_file():
        return []
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed continual evaluation row at {path}:"
                    f"{line_number}"
                ) from error
            rows.append(row)
    return rows


def write_phase_rows(
    path: Path,
    *,
    phase_index: int,
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Atomically replace one phase, making boundary resume idempotent."""
    if any(int(row.get("phase_index", -1)) != phase_index for row in rows):
        raise ValueError("every evaluation row must match phase_index")
    row_keys = [
        (int(row["eval_task_index"]), row.get("eval_scope", "seen"))
        for row in rows
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("duplicate evaluation task/scope rows in one phase")

    historical = [
        row for row in read_eval_rows(path)
        if int(row.get("phase_index", -1)) != phase_index
    ]
    combined = historical + list(rows)
    combined.sort(key=lambda row: (
        int(row["phase_index"]),
        int(row["eval_task_index"]),
        str(row.get("eval_scope", "seen")),
    ))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in combined:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return combined


def continual_eval_run_id(recipe: dict[str, Any]) -> str:
    """Return the stable W&B id for a complete continual training recipe."""
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return "ce" + hashlib.sha256(encoded.encode()).hexdigest()[:20]


def _finite_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _seen_rows(
    rows: Iterable[dict[str, Any]], phase_index: int
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if int(row["phase_index"]) == phase_index
        and row.get("eval_scope", "seen") == "seen"
    ]


def continual_scalars(
    rows: Sequence[dict[str, Any]],
    phase_index: int,
    *,
    include_retention: bool = True,
) -> dict[str, float | int]:
    """Compute standard CL scalars from the stored success matrix."""
    current = _seen_rows(rows, phase_index)
    success = [
        value for row in current
        if (value := _finite_metric(
            row, "eval/episode_success_rate"
        )) is not None
    ]
    easy = [
        value for row in current
        if (value := _finite_metric(
            row, "eval/episode_easy_success_rate"
        )) is not None
    ]
    payload: dict[str, float | int] = {
        "continual/phase_index": phase_index,
        "continual/num_seen_tasks": len(current),
    }
    if success:
        payload.update({
            "continual/mean_seen_success_rate": sum(success) / len(success),
            "continual/min_seen_success_rate": min(success),
        })
    if easy:
        payload["continual/mean_seen_easy_success_rate"] = (
            sum(easy) / len(easy)
        )

    current_by_task = {
        int(row["eval_task_index"]): row for row in current
    }
    diagonal = {
        int(row["eval_task_index"]): _finite_metric(
            row, "eval/episode_success_rate"
        )
        for row in rows
        if row.get("eval_scope", "seen") == "seen"
        and int(row["phase_index"]) == int(row["eval_task_index"])
    }
    if include_retention:
        forgetting = []
        backward_transfer = []
        for task_index, row in current_by_task.items():
            value = _finite_metric(row, "eval/episode_success_rate")
            if value is None:
                continue
            if task_index < phase_index:
                history = [
                    _finite_metric(item, "eval/episode_success_rate")
                    for item in rows
                    if item.get("eval_scope", "seen") == "seen"
                    and int(item["eval_task_index"]) == task_index
                    and int(item["phase_index"]) <= phase_index
                ]
                history = [item for item in history if item is not None]
                if history:
                    forgetting.append(max(history) - value)
            reference = diagonal.get(task_index)
            if task_index < phase_index and reference is not None:
                backward_transfer.append(value - reference)
        if forgetting:
            payload["continual/average_forgetting"] = (
                sum(forgetting) / len(forgetting)
            )
        if backward_transfer:
            payload["continual/backward_transfer"] = (
                sum(backward_transfer) / len(backward_transfer)
            )

    next_rows = [
        row for row in rows
        if int(row["phase_index"]) == phase_index
        and row.get("eval_scope") == "next_unseen"
    ]
    if next_rows:
        next_success = _finite_metric(
            next_rows[0], "eval/episode_success_rate"
        )
        if next_success is not None:
            payload["continual/next_task_zero_shot_success_rate"] = (
                next_success
            )

    for row in current:
        task_index = int(row["eval_task_index"])
        for metric in (*_MATRIX_METRICS, *_MATRIX_STD_METRICS):
            value = _finite_metric(row, metric)
            if value is not None:
                short_name = metric.removeprefix("eval/")
                payload[
                    f"continual/task_{task_index:02d}/{short_name}"
                ] = value
        if task_index == phase_index:
            for metric in _ADAPTATION_METRICS:
                value = _finite_metric(row, metric)
                if value is not None:
                    payload[
                        f"continual/task_{task_index:02d}/{metric}"
                    ] = value
    return payload


def _long_table_data(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[str], list[list[Any]]]:
    columns = [
        "phase_index",
        "eval_task_index",
        "eval_scope",
        "critic_head_task_index",
        "train_task_global_id",
        "eval_task_global_id",
        *_EVAL_COUNT_METRICS,
        *_ADAPTATION_METRICS,
        *(
            metric
            for pair in zip(_MATRIX_METRICS, _MATRIX_STD_METRICS)
            for metric in pair
        ),
    ]
    data = [
        [
            row.get(column)
            if column not in (
                *_MATRIX_METRICS,
                *_MATRIX_STD_METRICS,
                *_EVAL_COUNT_METRICS,
                *_ADAPTATION_METRICS,
            )
            else _finite_metric(row, column)
            for column in columns
        ]
        for row in rows
    ]
    return columns, data


def _matrix_table_data(
    rows: Sequence[dict[str, Any]], metric: str
) -> tuple[list[str], list[list[Any]]]:
    seen = [row for row in rows if row.get("eval_scope", "seen") == "seen"]
    if not seen:
        return ["phase_index", "train_task_global_id"], []
    max_task = max(int(row["eval_task_index"]) for row in seen)
    columns = [
        "phase_index",
        "train_task_global_id",
        *(f"task_{index:02d}" for index in range(max_task + 1)),
    ]
    data = []
    for phase_index in sorted({int(row["phase_index"]) for row in seen}):
        phase_rows = _seen_rows(seen, phase_index)
        by_task = {
            int(row["eval_task_index"]): _finite_metric(row, metric)
            for row in phase_rows
        }
        train_id = phase_rows[0].get("train_task_global_id", "")
        data.append([
            phase_index,
            train_id,
            *(by_task.get(index) for index in range(max_task + 1)),
        ])
    return columns, data


def log_continual_eval_to_wandb(
    *,
    args: Any,
    recipe: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    phase_index: int,
    log_tables: bool = True,
) -> bool:
    """Best-effort W&B upload that can never invalidate completed training."""
    if not bool(getattr(args, "track", False)):
        return False

    run = None
    try:
        import wandb

        tag = getattr(args, "wandb_name_tag", "") or recipe["algorithm"]
        run_id = continual_eval_run_id({
            "recipe": recipe,
            "project": args.wandb_project_name,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "name_tag": tag,
        })
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            group=args.wandb_group,
            id=run_id,
            resume="allow",
            reinit=True,
            name=f"continual_eval__{tag}",
            job_type="continual-eval",
            config={
                "continual_eval_schema": 3,
                "training_recipe": recipe,
            },
        )
        run.define_metric("continual/phase_index")
        run.define_metric(
            "continual/*", step_metric="continual/phase_index"
        )
        payload: dict[str, Any] = continual_scalars(
            rows,
            phase_index,
            include_retention=bool(
                getattr(args, "report_retention_metrics", True)
            ),
        )
        if log_tables:
            long_columns, long_data = _long_table_data(rows)
            success_columns, success_data = _matrix_table_data(
                rows, "eval/episode_success_rate"
            )
            easy_columns, easy_data = _matrix_table_data(
                rows, "eval/episode_easy_success_rate"
            )
            success_std_columns, success_std_data = _matrix_table_data(
                rows, "eval/episode_success_rate_std"
            )
            easy_std_columns, easy_std_data = _matrix_table_data(
                rows, "eval/episode_easy_success_rate_std"
            )
            payload.update({
                "continual/eval_rows": wandb.Table(
                    columns=long_columns, data=long_data
                ),
                "continual/success_matrix": wandb.Table(
                    columns=success_columns, data=success_data
                ),
                "continual/easy_success_matrix": wandb.Table(
                    columns=easy_columns, data=easy_data
                ),
                "continual/success_std_matrix": wandb.Table(
                    columns=success_std_columns, data=success_std_data
                ),
                "continual/easy_success_std_matrix": wandb.Table(
                    columns=easy_std_columns, data=easy_std_data
                ),
            })
        run.log(payload)
        run.summary["continual/latest_completed_phase"] = phase_index
        run.summary["continual/eval_row_count"] = len(rows)
        return True
    except Exception as error:  # Logging must not waste a completed HPC phase.
        print(
            "WARNING: continual W&B matrix logging failed; local JSONL is "
            f"complete and training will continue: {error}"
        )
        return False
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as error:
                print(f"WARNING: could not finish continual W&B run: {error}")
