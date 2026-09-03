# Sequence A forward-transfer HPC track

## Comparison

`diverse_continual_experiment_configs.py` defines nine jobs:

- Reset StableCRL, Persistent StableCRL, and DCC;
- seeds 5, 6, and 7;
- the same nine tasks, task order, per-task environment-step budget, CRTR-12,
  PD-5 controller, state/goal shapes, evaluation count, and optimizer settings.

Reset StableCRL initializes a new actor and critic at each task. Persistent
StableCRL carries both networks. DCC carries its actor and shared state-action
and goal encoders, while initializing a fresh zero-output task-specific
state-action adapter.

DCC has **no dynamics prediction head and no dynamics loss**. The
task-specific residual adapter is not a dynamics head; it is the part that
makes this method DCC. Removing it would reduce the method to the persistent
StableCRL comparison.

## Measurements

This track intentionally does not study forgetting. The config disables old
task and speculative next-task evaluations.

For every task, the learner evaluates the policy after the task's actual
reset or transfer initialization and before collecting task data. It reports:

- `forward_transfer/initial_success_rate`;
- `forward_transfer/initial_easy_success_rate`;
- final strict and easy success;
- normalized strict and easy success AUC over environment steps.

The measured initial success is the step-zero point of the AUC. Therefore the
AUC includes both transferred starting ability and subsequent adaptation
speed. The result is stored in the task's W&B run, its
`task_adaptation.json`, and the continual run's `continual_eval.jsonl`.

## Training-progress videos

Every Sequence A config enables video recording. Production uses 50 evaluation
points and a target of 10 videos, so one deterministic evaluation episode is
recorded at evaluation steps 5, 10, ..., 50 for every task, seed, and
algorithm. Each video is saved locally and uploaded in the same W&B history
row as its numerical metrics under `eval/video`.

This produces 810 production videos:

```text
9 tasks x 3 algorithms x 3 seeds x 10 training points = 810 videos
```

“About ten” refers to the per-task cadence. A smoke run has only four
evaluation points, so it records four videos per task.

After all jobs finish, `summarize_diverse_continual.py` creates one row per
method, seed, and task. For tasks 2--9 it reports two matched-seed comparisons
against Reset StableCRL:

- forward-transfer gain = initial success minus reset initial success;
- adaptation-AUC gain = success AUC minus reset success AUC.

Task 1 has no preceding experience, so its transfer gains are left blank.

## Torch commands

First update the Torch checkout to the commit containing this track. Inspect
the nine jobs and the command generated for the seed-5 DCC cell:

```bash
python diverse_continual_experiment_configs.py --list
DRY_RUN=true CONFIG_INDEX=2 RUN_TEST_PREFLIGHT=false \
  bash DRAFT_DIVERSE_CONTINUAL.sh
```

Run a short seed-5, three-method gate:

```bash
EXPERIMENT_STAGE=smoke \
BASE_STEPS=2097152 STEPS_PER_TASK=2097152 \
NUM_ENVS=256 NUM_EVAL_ENVS=32 \
MAX_REPLAY_SIZE=512 MIN_REPLAY_SIZE=128 \
NUM_EVAL_STEPS=4 NUM_RESET_STEPS=4 \
sbatch --array=0-2 DRAFT_DIVERSE_CONTINUAL.sh
```

If all three jobs complete every task, submit the full comparison:

```bash
sbatch DRAFT_DIVERSE_CONTINUAL.sh
```

The script already specifies the Torch account, one GPU per job, and array
indices 0--8. Smoke outputs use the `diverse_sequence_a_smoke` namespace;
production outputs use `diverse_sequence_a`, so production resume cannot pick
up a short-budget checkpoint. To summarize completed production runs:

```bash
python summarize_diverse_continual.py \
  --checkpoint-root /scratch/$USER/builderbench-stablecrl-dcc/checkpoints/diverse_sequence_a \
  --upload-wandb
```

This uploads `forward_transfer/per_task` as a W&B table and stores the mean
matched-seed forward-transfer and adaptation-AUC gains in the summary run.
