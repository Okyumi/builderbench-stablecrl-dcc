# Residual-DCC Torch smoke gate

Date: 2026-08-23

## Scope

This change adds a dedicated, deterministic NYU Torch smoke batch for the
residual, no-dynamics DCC implementation. It is intentionally separate from
the 72-cell production registry, so the existing experiment indices and W&B
identities remain stable.

The implementation adds:

- `smoke_experiment_configs.py`, a three-cell smoke-only registry;
- `DRAFT_DCC_SMOKE.sh`, a short Slurm array wrapper;
- configurable experiment-registry and unit-test preflight support in the
  shared `DRAFT.sh` launcher;
- dependency-light registry and command-generation tests.

## SGCRL launcher conventions retained

The launcher follows the experiment pattern used for SGCRL:

- one deterministic configuration is selected from an integer array index;
- every run has an isolated W&B name, group, log prefix, and checkpoint
  directory;
- the actor and shared DCC groups persist only in the continual transfer
  cell, while each task gets a fresh `phi_task` adapter and optimizer;
- GPU, JAX, MuJoCo, and MuJoCo-Warp availability are checked before launch;
- task-boundary checkpoints are resumable, while interrupted within-task
  optimization is not;
- the same shared launcher builds the runner-specific command, which avoids a
  second copy of the DCC command-line contract drifting from production.

No W&B API key is stored in either script. The existing logged-in W&B state or
`WANDB_API_KEY` supplied to the job environment is used.

## Smoke registry

The smoke array has indices 0 through 2, all at seed 5.

### Index 0: three-cube CRTR initialization

`smoke_dcc_residual_three_stack` trains `creative-3-task1` with repetition
factor 12, semantic padding to four cubes, and actor carry disabled. This
exercises the long-horizon CRTR data path and the three-to-four-cube padding
case.

### Index 1: four-cube CRTR initialization

`smoke_dcc_residual_four_stack` trains `creative-4-task1` with repetition
factor 12 and actor carry disabled. This exercises the largest observation and
goal actually used by the proposed expanding-stack protocol.

### Index 2: two-task continual transfer

`smoke_dcc_residual_two_task_transfer` trains
`creative-1-task1,creative-2-task1` with a persistent actor and persistent
shared DCC groups. It verifies all of the following in one short job:

- the observation/goal tensors retain the same four-cube semantic shape;
- the actor, shared state-action encoder, and shared goal encoder transfer;
- the task-specific residual adapter and optimizer reset at the boundary;
- both task adapters remain available in the task bank;
- next-task zero-shot and seen-task boundary evaluation run;
- mean and standard-deviation continual matrices are written locally and
  uploaded to W&B;
- task-boundary checkpoints can be used by the resume path.

## Architecture and shortened runtime settings

The model shapes are not reduced for the smoke test. Every cell uses the
production residual architecture: eight blocks with hidden width 1024, a
four-layer task adapter with width 256, additive shared/task-specific
composition, and a shared goal encoder. There is no dynamics head or dynamics
loss.

Only the runtime workload is shortened:

- 2,097,152 environment steps for the first task and for each later task;
- 256 parallel training environments and 32 evaluation environments;
- rollout length 64, producing exactly 128 training iterations per task;
- four within-task evaluations and four real-reset intervals;
- replay time-axis capacity 512 with a 128-step prefill;
- two boundary-evaluation repeats, or 64 evaluation episodes per matrix cell;
- videos and sample visualizations disabled.

These settings are intended to detect integration, compilation, checkpoint,
and logging failures and make finite-metric inspection inexpensive. They are
not large enough to judge final BuilderBench performance or compare success
rates with the paper.

## Torch submission

After pulling the branch on Torch, inspect the registry and perform a local
command-generation check:

```bash
cd /scratch/yd2247/builderbench-stablecrl-dcc
python smoke_experiment_configs.py --list
bash -n DRAFT_DCC_SMOKE.sh
DRY_RUN=true CONFIG_INDEX=2 bash DRAFT_DCC_SMOKE.sh
```

Submit all three cells with one run per GPU:

```bash
sbatch --account=torch_pr_XXX_XXXXX DRAFT_DCC_SMOKE.sh
```

The script declares `--array=0-2`. To submit only the continual cell:

```bash
sbatch --account=torch_pr_XXX_XXXXX \
  --array=0-0 \
  --export=ALL,CONFIG_INDEX=2 \
  DRAFT_DCC_SMOKE.sh
```

The default W&B group prefix is `torch_dcc_smoke`. Smoke checkpoints are
stored under `$SCRATCH/builderbench-stablecrl-dcc/checkpoints/smoke`, separate
from production checkpoints.

## Resume check

After index 2 completes, submit exactly the same index and settings again. The
driver should report that both task boundaries are complete and should reuse
the stored continual-evaluation rows rather than retraining:

```bash
sbatch --account=torch_pr_XXX_XXXXX \
  --array=0-0 \
  --export=ALL,CONFIG_INDEX=2 \
  DRAFT_DCC_SMOKE.sh
```

Changing a recipe field such as the number of environments, training budget,
evaluation repeats, or network shape requires a fresh checkpoint root. This
is deliberate: the resume recipe validator rejects incompatible checkpoints.

## Acceptance criteria

The smoke gate passes when:

1. all three Slurm array tasks exit successfully;
2. training losses and evaluation metrics logged to W&B are finite;
3. the two-task run creates `task_00.pkl`, `task_01.pkl`, `run_recipe.json`,
   `task_manifest.json`, and `continual_eval.jsonl`;
4. the continual-evaluation W&B run contains the success and success-standard-
   deviation matrices through phase 1;
5. resubmitting index 2 takes the completed-prefix resume path without a
   recipe mismatch.

Passing this gate authorizes the longer six-cell residual-DCC individual-task
gate at production indices 54 through 59. It does not by itself establish
performance parity.

## Validation performed locally

- both shell launchers passed shell syntax validation;
- the smoke registry generated three valid, shape-compatible DCC cells;
- dry-run command generation covered the two-task smoke cell;
- the Slurm spool-directory resolver was tested;
- 23 dependency-light tests covering registries, launchers, semantic layout,
  continual metrics, and repeated evaluation passed after the change.

The actual JAX/MuJoCo-Warp training smoke remains a Torch GPU task because the
local machine does not have the required CUDA runtime and has insufficient
free disk space for the full environment.

## Known limitations

- The launcher resumes only at completed task boundaries, not in the middle of
  a task.
- The automated preflight checks imports and unit tests, but the user must
  still inspect W&B for finite losses; a numerically invalid run can finish at
  the process level.
- Two evaluation repeats reduce smoke cost and should not be used as the final
  uncertainty estimate. Production evaluation retains five repeats.
