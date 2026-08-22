# Slurm spool path is not the repository

Date: 2026-08-22

## Purpose

Torch copies `#SBATCH` scripts into `/opt/slurm/data/slurmd/job<id>/` before
execution. `DRAFT.sh` previously set `REPO_DIR` from `BASH_SOURCE`, so
`python experiment_configs.py` ran against that empty spool directory:

```text
can't open file '/opt/slurm/data/slurmd/job.../experiment_configs.py'
```

## Fix

The launcher now:

1. Sets `#SBATCH --chdir` to the scratch clone.
2. Resolves `REPO_DIR` by looking for `experiment_configs.py` in the script
   directory, `SLURM_SUBMIT_DIR`, then `$SCRATCH/builderbench-stablecrl-dcc`.
3. Invokes `experiment_configs.py` and the runner with absolute paths.

`REPO_DIR` can still be set explicitly. Local dry runs from the clone keep
working because the script directory already contains the configs module.

## Validation

`tests/test_experiment_configs.py` copies `DRAFT.sh` into a fake spool
directory, sets `SLURM_SUBMIT_DIR` to the real repository, and checks that
the generated command uses the repository paths.
