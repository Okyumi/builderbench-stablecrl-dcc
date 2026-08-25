#!/usr/bin/env bash
#SBATCH --job-name=bb_cont_smoke
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --chdir=/scratch/yd2247/builderbench-stablecrl-dcc
#SBATCH --output=/scratch/yd2247/builderbench-stablecrl-dcc/logs/slurm/%A_%a.out
#SBATCH --error=/scratch/yd2247/builderbench-stablecrl-dcc/logs/slurm/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=FAIL,END

# Three matched seed-5 smoke cells on the goal-only sequence: reset/reset,
# persistent/persistent, and residual DCC.

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
        && [ -f "$candidate/continual_experiment_configs.py" ]; then
        break
      fi
      candidate=""
    done
  fi
  if [ -z "$candidate" ] \
    || [ ! -f "$candidate/DRAFT.sh" ] \
    || [ ! -f "$candidate/continual_experiment_configs.py" ]; then
    echo "Could not find the matched continual smoke launcher files." >&2
    echo "Set REPO_DIR to the repository root." >&2
    exit 1
  fi
  cd "$candidate" || exit 1
  REPO_DIR="$(pwd)"
  export REPO_DIR
}

resolve_repo_dir

export CONFIG_REGISTRY="${CONFIG_REGISTRY:-continual_experiment_configs.py}"
export EXPERIMENT_STAGE="${EXPERIMENT_STAGE:-smoke_goal}"
export TASKS_PER_GPU="${TASKS_PER_GPU:-1}"
export BASE_STEPS="${BASE_STEPS:-2097152}"
export STEPS_PER_TASK="${STEPS_PER_TASK:-2097152}"
export NUM_ENVS="${NUM_ENVS:-256}"
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-32}"
export ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-64}"
export MAX_REPLAY_SIZE="${MAX_REPLAY_SIZE:-512}"
export MIN_REPLAY_SIZE="${MIN_REPLAY_SIZE:-128}"
export NUM_EVAL_STEPS="${NUM_EVAL_STEPS:-4}"
export NUM_RESET_STEPS="${NUM_RESET_STEPS:-4}"
export TRACK="${TRACK:-true}"
export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
export RESUME="${RESUME:-true}"
export EVAL_NEXT_TASK="${EVAL_NEXT_TASK:-true}"
export LOG_CONTINUAL_EVAL="${LOG_CONTINUAL_EVAL:-true}"
export WANDB_EVAL_TABLES="${WANDB_EVAL_TABLES:-true}"
export RECORD_VIDEOS="${RECORD_VIDEOS:-false}"
export VISUALIZE_SAMPLES="${VISUALIZE_SAMPLES:-false}"
export REQUIRE_GPU="${REQUIRE_GPU:-true}"
export RUN_TEST_PREFLIGHT="${RUN_TEST_PREFLIGHT:-true}"
export WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-torch_dcc_continual_smoke}"
export LOG_ROOT="${LOG_ROOT:-${SCRATCH_ROOT}/builderbench-stablecrl-dcc/logs/continual_smoke}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRATCH_ROOT}/builderbench-stablecrl-dcc/checkpoints/continual_smoke}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/builderbench-stablecrl-dcc/wandb/continual_smoke}"

exec bash "$REPO_DIR/DRAFT.sh"
