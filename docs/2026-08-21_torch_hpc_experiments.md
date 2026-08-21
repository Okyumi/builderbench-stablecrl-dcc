# NYU Torch HPC experiment launcher

Date: 2026-08-21

## Purpose

`DRAFT.sh` and `experiment_configs.py` reproduce the active SGCRL batch shape
for StableCRL-DCC on BuilderBench. The launcher is intended for NYU Torch,
uses a Slurm array, isolates logs and checkpoints by configuration and seed,
and resumes from completed task boundaries.

The implementation follows current Torch requirements documented by NYU:

- every submission supplies an allocation with `--account`;
- GPU jobs request `--gres=gpu:1`;
- no partition is selected manually;
- optional GPU constraints are `h200`, `l40s`, or a permitted combination.

References:

- <https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/>
- <https://services.rt.nyu.edu/docs/hpc/tools_and_software/utils/>

## SGCRL experiment mapping

The active grid contains 12 matched-seed runs using seeds 5, 6, and 7:

1. Five-task continual DCC, persistent actor, masked dynamics weight 1.0.
2. The same continual run with dynamics weight 0.0.
3. Repetition-12 DCC/StableCRL on `creative-3-task1`.
4. Repetition-12 DCC/StableCRL on `creative-4-task1`.

The first two cells preserve the SGCRL lifecycle: the actor and shared DCC
groups persist while `phi_task` and its optimizer reset. The last two replace
SGCRL's Sawyer task-5/task-8 failure probes with BuilderBench's three- and
four-cube long-horizon stack probes. They are single-task sequences, so actor
carry is disabled and no continual history contaminates the CRTR test.

The SGCRL DCC-SAC and action-contrastive cells are not emitted because those
algorithms do not yet exist in this StableCRL BuilderBench port. The launcher
does not silently map unsupported algorithms to ordinary DCC.

## One-time Torch setup

Clone the repository under scratch and create a Python 3.11 environment:

```bash
cd /scratch/yd2247
git clone https://github.com/Okyumi/builderbench-stablecrl-dcc.git
python3.11 -m venv /scratch/yd2247/.venvs/builderbench-stablecrl-dcc
source /scratch/yd2247/.venvs/builderbench-stablecrl-dcc/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '/scratch/yd2247/builderbench-stablecrl-dcc[all]'
mkdir -p /scratch/yd2247/builderbench-stablecrl-dcc/logs/slurm
```

The project pins the CUDA 12 JAX, MuJoCo, and MuJoCo-Warp packages. The
launcher does not load a CUDA module by default because the installed wheels
provide user-space CUDA libraries. If Torch's environment requires a module,
submit with a valid `CUDA_MODULE` environment value.

## Preflight

Inspect the exact array cells and generate commands without launching:

```bash
cd /scratch/yd2247/builderbench-stablecrl-dcc
python experiment_configs.py --list
python experiment_configs.py --total
python experiment_configs.py --array-max --tasks-per-gpu 1
bash -n DRAFT.sh
DRY_RUN=true CONFIG_INDEX=0 bash DRAFT.sh
```

`--total` returns 12 and the one-run-per-GPU array maximum is 11.

## Submission

Find the available allocation and submit the full array:

```bash
my_slurm_accounts
sbatch --account=torch_pr_XXX_XXXXX DRAFT.sh
```

An optional GPU constraint can be supplied without editing the script:

```bash
sbatch --account=torch_pr_XXX_XXXXX --constraint=h200 DRAFT.sh
```

Submit one cell while debugging:

```bash
sbatch --account=torch_pr_XXX_XXXXX --array=0-0 DRAFT.sh
```

The default is one experiment per GPU. If profiling shows that two runs fit
and satisfy Torch's GPU-utilization policy, use MPS and resize the array:

```bash
TASKS_PER_GPU=2 sbatch \
  --account=torch_pr_XXX_XXXXX \
  --array=0-5 \
  --comment='gpu_mps=yes' \
  DRAFT.sh
```

Do not opt into Torch preemption yet. The continual driver resumes completed
task boundaries, but it does not restore an interrupted task from an
intra-task optimizer/replay checkpoint.

## Runtime and checkpoint contract

Each run uses this identity:

```text
<configuration-name>_seed<seed>
```

Its boundary checkpoint directory is
`$CHECKPOINT_ROOT/<configuration-name>_seed<seed>`. This prevents different
array cells from overwriting `task_00.pkl` and ensures recipe validation is
meaningful. Re-submitting the same cell resumes the completed prefix. Changing
an algorithmic setting requires a new configuration name or checkpoint root.

The launcher checks that JAX sees a GPU and that MuJoCo-Warp imports before
starting experiments. Videos and sample figures are disabled by default on
HPC, while W&B tracking, StableCRL checkpoints, cross-task evaluation, and
task-boundary checkpoints remain enabled.

CRTR requires `NUM_ENVS >= REPETITION_FACTOR`; otherwise the repeated batch
would contain zero unique trajectories. Both the launcher and learner reject
that configuration explicitly. The production defaults use 1024 environments
and repetition factor 12.

## Validation

- All 12 array cells passed command-generation dry runs.
- The complete repository suite passed 16 tests after adding launcher tests.
- `bash -n DRAFT.sh` passed.
- A launcher-level CPU/JAX smoke for the repetition-12 three-stack cell ran
  through environment activation, training, evaluation, logging, boundary
  checkpointing, and process-status collection with finite losses.
