#!/usr/bin/env bash
#SBATCH --job-name=bb_diverse_a
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=47:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-8
#SBATCH --chdir=/scratch/yd2247/builderbench-stablecrl-dcc
#SBATCH --output=/scratch/yd2247/builderbench-stablecrl-dcc/logs/slurm/%A_%a.out
#SBATCH --error=/scratch/yd2247/builderbench-stablecrl-dcc/logs/slurm/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=FAIL,END

# Diverse Sequence A: reset StableCRL, persistent StableCRL, and DCC for
# seeds 5/6/7.  Set EXPERIMENT_STAGE=smoke and override the step counts for a
# three-cell preflight before launching the full nine-cell sequence_a stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_ROOT="${SCRATCH:-/scratch/${USER}}"
DEFAULT_REPO_DIR="${SCRATCH_ROOT}/builderbench-stablecrl-dcc"

resolve_repo_dir() {
  local candidate
  if [ -n "${REPO_DIR:-}" ]; then
    candidate="$REPO_DIR"
  else
    candidate=""
    for candidate in \
      "$SCRIPT_DIR" \
      "${SLURM_SUBMIT_DIR:-}" \
      "$DEFAULT_REPO_DIR"
    do
      [ -n "$candidate" ] || continue
      if [ -f "$candidate/DRAFT.sh" ] \
        && [ -f "$candidate/diverse_continual_experiment_configs.py" ]; then
        break
      fi
      candidate=""
    done
  fi
  if [ -z "$candidate" ] \
    || [ ! -f "$candidate/DRAFT.sh" ] \
    || [ ! -f "$candidate/diverse_continual_experiment_configs.py" ]; then
    echo "Could not find the diverse continual launcher files." >&2
    echo "Set REPO_DIR to the repository root." >&2
    exit 1
  fi
  cd "$candidate" || exit 1
  REPO_DIR="$(pwd)"
  export REPO_DIR
}

resolve_repo_dir

export CONFIG_REGISTRY="${CONFIG_REGISTRY:-diverse_continual_experiment_configs.py}"
export EXPERIMENT_STAGE="${EXPERIMENT_STAGE:-sequence_a}"
export TASKS_PER_GPU="${TASKS_PER_GPU:-1}"
export RUN_TEST_PREFLIGHT="${RUN_TEST_PREFLIGHT:-true}"
export TASK_DATA_VERSION="${TASK_DATA_VERSION:-builderbench-de9130-direct-v1}"
export EVAL_NEXT_TASK="${EVAL_NEXT_TASK:-false}"
export EVAL_PREVIOUS_TASKS="${EVAL_PREVIOUS_TASKS:-false}"
export REPORT_RETENTION_METRICS="${REPORT_RETENTION_METRICS:-false}"
if [ "$EXPERIMENT_STAGE" = "smoke" ]; then
  DEFAULT_RUN_NAMESPACE="diverse_sequence_a_smoke"
else
  DEFAULT_RUN_NAMESPACE="diverse_sequence_a"
fi
RUN_NAMESPACE="${RUN_NAMESPACE:-$DEFAULT_RUN_NAMESPACE}"
export WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-torch_dcc_${RUN_NAMESPACE}}"
export LOG_ROOT="${LOG_ROOT:-${SCRATCH_ROOT}/builderbench-stablecrl-dcc/logs/${RUN_NAMESPACE}}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRATCH_ROOT}/builderbench-stablecrl-dcc/checkpoints/${RUN_NAMESPACE}}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/builderbench-stablecrl-dcc/wandb/${RUN_NAMESPACE}}"

exec bash "$REPO_DIR/DRAFT.sh"
