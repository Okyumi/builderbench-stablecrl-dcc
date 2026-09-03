#!/usr/bin/env bash
#SBATCH --job-name=bb_stablecrl
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=47:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --chdir=/scratch/yd2247/builderbench-stablecrl-dcc
#SBATCH --output=/scratch/yd2247/builderbench-stablecrl-dcc/logs/slurm/%A_%a.out
#SBATCH --error=/scratch/yd2247/builderbench-stablecrl-dcc/logs/slurm/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=FAIL,END

# NYU Torch requires an account on every submission. Supply it on the command
# line because account names are allocation-specific:
#   sbatch --account=torch_pr_XXX_XXXXX DRAFT.sh
# Torch documentation says not to select a partition manually. The scheduler
# chooses an accessible GPU QoS from --gres and the submitted account.

set -euo pipefail

# Slurm copies the batch script into /opt/slurm/data/slurmd/job*/. Do not treat
# that spool directory as the repository.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS_PER_GPU="${TASKS_PER_GPU:-1}"
DRY_RUN="${DRY_RUN:-false}"
REQUIRE_GPU="${REQUIRE_GPU:-true}"
EXPERIMENT_STAGE="${EXPERIMENT_STAGE:-dcc_residual_gate}"
CONFIG_REGISTRY="${CONFIG_REGISTRY:-experiment_configs.py}"
RUN_TEST_PREFLIGHT="${RUN_TEST_PREFLIGHT:-false}"

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
      if [ -f "$candidate/$CONFIG_REGISTRY" ]; then
        break
      fi
      candidate=""
    done
  fi
  if [ -z "$candidate" ] || [ ! -f "$candidate/$CONFIG_REGISTRY" ]; then
    echo "Could not find experiment registry: $CONFIG_REGISTRY" >&2
    echo "SCRIPT_DIR=${SCRIPT_DIR} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" >&2
    echo "Set REPO_DIR to the repository root." >&2
    exit 1
  fi
  cd "$candidate" || exit 1
  REPO_DIR="$(pwd)"
}

resolve_repo_dir
CONFIG_REGISTRY_PATH="$REPO_DIR/$CONFIG_REGISTRY"
VENV_DIR="${VENV_DIR:-${SCRATCH_ROOT}/.venvs/builderbench-stablecrl-dcc}"
LOG_ROOT="${LOG_ROOT:-${SCRATCH_ROOT}/builderbench-stablecrl-dcc/logs/runs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRATCH_ROOT}/builderbench-stablecrl-dcc/checkpoints}"
WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/builderbench-stablecrl-dcc/wandb}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/.cache/builderbench-stablecrl-dcc}"
JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${XDG_CACHE_HOME}/jax}"
TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp/builderbench-stablecrl-dcc}"

# Shared paper-scale defaults. Per-cell algorithm and lifecycle choices come
# only from experiment_configs.py so every array index is reproducible.
BASE_STEPS="${BASE_STEPS:-200000000}"
STEPS_PER_TASK="${STEPS_PER_TASK:-200000000}"
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-64}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
DISCOUNT="${DISCOUNT:-0.99}"
ENTROPY_COST="${ENTROPY_COST:-0.01}"
LOGSUMEXP_COST="${LOGSUMEXP_COST:-0.1}"
REP_SIZE="${REP_SIZE:-64}"
MAX_REPLAY_SIZE="${MAX_REPLAY_SIZE:-10000}"
MIN_REPLAY_SIZE="${MIN_REPLAY_SIZE:-1000}"
NUM_EVAL_STEPS="${NUM_EVAL_STEPS:-50}"
NUM_RESET_STEPS="${NUM_RESET_STEPS:-50}"
TASK_DATA_VERSION="${TASK_DATA_VERSION:-david-6e8d56d}"
MJX_IMPL="${MJX_IMPL:-warp}"

TRACK="${TRACK:-true}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
RECORD_VIDEOS="${RECORD_VIDEOS:-false}"
VISUALIZE_SAMPLES="${VISUALIZE_SAMPLES:-false}"
RESUME="${RESUME:-true}"
EVAL_NEXT_TASK="${EVAL_NEXT_TASK:-true}"
EVAL_PREVIOUS_TASKS="${EVAL_PREVIOUS_TASKS:-true}"
REPORT_RETENTION_METRICS="${REPORT_RETENTION_METRICS:-true}"
LOG_CONTINUAL_EVAL="${LOG_CONTINUAL_EVAL:-true}"
WANDB_EVAL_TABLES="${WANDB_EVAL_TABLES:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-builderbench-stablecrl-dcc}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-torch_dcc}"

# JAX CUDA wheels normally provide user-space CUDA libraries. Set CUDA_MODULE
# only if the site environment requires an explicit module.
CUDA_MODULE="${CUDA_MODULE:-}"
TOOLCHAIN_MODULE="${TOOLCHAIN_MODULE:-}"

case "$TASKS_PER_GPU" in
  1) DEFAULT_MEM_FRACTION="0.88" ;;
  2) DEFAULT_MEM_FRACTION="0.44" ;;
  3) DEFAULT_MEM_FRACTION="0.29" ;;
  4) DEFAULT_MEM_FRACTION="0.22" ;;
  *)
    echo "TASKS_PER_GPU must be one of 1, 2, 3, or 4" >&2
    exit 2
    ;;
esac
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-$DEFAULT_MEM_FRACTION}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME JAX_COMPILATION_CACHE_DIR TMPDIR REQUIRE_GPU

bool_flag() {
  local flag_name="$1"
  local flag_value="$2"
  case "$flag_value" in
    true) printf '%s' "--${flag_name}" ;;
    false) printf '%s' "--no-${flag_name}" ;;
    *)
      echo "Expected true/false for ${flag_name}, got ${flag_value}" >&2
      return 2
      ;;
  esac
}

setup_environment() {
  if [ "$DRY_RUN" = "true" ]; then
    return
  fi
  if type module >/dev/null 2>&1; then
    module purge
    [ -n "$CUDA_MODULE" ] && module load "$CUDA_MODULE"
    [ -n "$TOOLCHAIN_MODULE" ] && module load "$TOOLCHAIN_MODULE"
  fi
  if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Python environment not found: $VENV_DIR" >&2
    echo "Create it once with Python 3.11 and install -e '.[all]'." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"

  "$PYTHON_BIN" - <<'PY'
import os

import jax
import mujoco
import mujoco_warp  # noqa: F401

devices = jax.devices()
print("JAX devices:", devices)
if (os.environ["REQUIRE_GPU"] == "true"
        and not any(device.platform == "gpu" for device in devices)):
    raise SystemExit("No JAX GPU device is visible in this Torch allocation")
print("MuJoCo:", mujoco.__version__)
PY
}

if ! [[ "$TASKS_PER_GPU" =~ ^[1-4]$ ]]; then
  echo "TASKS_PER_GPU must be a positive integer no larger than four" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$CHECKPOINT_ROOT" "$WANDB_DIR" \
  "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"
setup_environment

if [ "$RUN_TEST_PREFLIGHT" = "true" ] && [ "$DRY_RUN" != "true" ]; then
  echo "Running repository test preflight before launching experiments."
  "$PYTHON_BIN" -m unittest discover -s "$REPO_DIR/tests" -p 'test_*.py'
fi

TOTAL_CONFIGS="$($PYTHON_BIN "$CONFIG_REGISTRY_PATH" --total)"
STAGE_START="$($PYTHON_BIN "$CONFIG_REGISTRY_PATH" \
  --stage-start "$EXPERIMENT_STAGE")"
STAGE_END="$($PYTHON_BIN "$CONFIG_REGISTRY_PATH" \
  --stage-end "$EXPERIMENT_STAGE")"
STAGE_CONFIGS="$($PYTHON_BIN "$CONFIG_REGISTRY_PATH" \
  --stage-total "$EXPERIMENT_STAGE")"
ARRAY_MAX="$($PYTHON_BIN "$CONFIG_REGISTRY_PATH" \
  --stage-array-max \
  "$EXPERIMENT_STAGE" \
  --tasks-per-gpu "$TASKS_PER_GPU")"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ -n "${CONFIG_INDEX:-}" ]; then
  FIRST_CONFIG="$CONFIG_INDEX"
  SLOTS=1
else
  FIRST_CONFIG=$((STAGE_START + TASKS_PER_GPU * ARRAY_TASK_ID))
  SLOTS="$TASKS_PER_GPU"
fi

if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] \
  && [ "$SLURM_ARRAY_TASK_MAX" -ne "$ARRAY_MAX" ]; then
  echo "Warning: this launch needs array 0-${ARRAY_MAX} for " \
    "TASKS_PER_GPU=${TASKS_PER_GPU}; received 0-${SLURM_ARRAY_TASK_MAX}." >&2
fi

echo "============================================================"
echo "BuilderBench StableCRL/DCC on NYU Torch"
echo "job=${SLURM_ARRAY_JOB_ID:-local} array_task=${ARRAY_TASK_ID}"
echo "account=${SLURM_JOB_ACCOUNT:-dry-run-or-unspecified}"
echo "configs=${TOTAL_CONFIGS} stage=${EXPERIMENT_STAGE} " \
  "stage_range=${STAGE_START}-${STAGE_END} stage_configs=${STAGE_CONFIGS}"
echo "array_max=${ARRAY_MAX} tasks_per_gpu=${TASKS_PER_GPU}"
echo "repo=${REPO_DIR}"
echo "config_registry=${CONFIG_REGISTRY_PATH}"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "JAX memory fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "============================================================"

PIDS=()
RUN_IDS=()

terminate_children() {
  if [ "${#PIDS[@]}" -gt 0 ]; then
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap terminate_children INT TERM

for ((slot = 0; slot < SLOTS; slot++)); do
  CONFIG_IDX=$((FIRST_CONFIG + slot))
  if [ -z "${CONFIG_INDEX:-}" ] && [ "$CONFIG_IDX" -gt "$STAGE_END" ]; then
    echo "[slot $slot] config $CONFIG_IDX is beyond stage end; skipping"
    continue
  fi
  if [ "$CONFIG_IDX" -ge "$TOTAL_CONFIGS" ]; then
    echo "[slot $slot] config $CONFIG_IDX is beyond $TOTAL_CONFIGS; skipping"
    continue
  fi

  eval "$("$PYTHON_BIN" "$CONFIG_REGISTRY_PATH" --setting "$CONFIG_IDX")"
  if [ "$NUM_ENVS" -lt "$REPETITION_FACTOR" ]; then
    echo "NUM_ENVS=${NUM_ENVS} must be at least " \
      "REPETITION_FACTOR=${REPETITION_FACTOR} for config ${CONFIG_IDX}" >&2
    exit 2
  fi
  RUN_ID="${NAME}_seed${SEED}"
  RUN_CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
  RUN_LOG_PREFIX="${LOG_ROOT}/${SLURM_ARRAY_JOB_ID:-local}_${ARRAY_TASK_ID}_${CONFIG_IDX}_${RUN_ID}"
  EFFECTIVE_GROUP="${WANDB_GROUP_PREFIX}__${WANDB_GROUP}"
  mkdir -p "$RUN_CHECKPOINT_DIR"

  COMMAND=(
    "$PYTHON_BIN" "$REPO_DIR/$RUNNER"
    --seed "$SEED"
    --num-envs "$NUM_ENVS"
    --num-eval-envs "$NUM_EVAL_ENVS"
    --rollout-length "$ROLLOUT_LENGTH"
    --actor-learning-rate "$ACTOR_LR"
    --critic-learning-rate "$CRITIC_LR"
    --discount "$DISCOUNT"
    --entropy-cost "$ENTROPY_COST"
    --logsumexp-cost "$LOGSUMEXP_COST"
    --rep-size "$REP_SIZE"
    --max-replay-size "$MAX_REPLAY_SIZE"
    --min-replay-size "$MIN_REPLAY_SIZE"
    --num-eval-steps "$NUM_EVAL_STEPS"
    --num-reset-steps "$NUM_RESET_STEPS"
    --repetition-factor "$REPETITION_FACTOR"
    --pd-duration "$PD_DURATION"
    --wandb-project-name "$WANDB_PROJECT_NAME"
    --wandb-mode "$WANDB_MODE"
    --wandb-dir "$WANDB_DIR"
    --wandb-group "$EFFECTIVE_GROUP"
    --wandb-name-tag "$RUN_ID"
  )
  [ -n "$WANDB_ENTITY" ] && COMMAND+=(--wandb-entity "$WANDB_ENTITY")

  if [ "$RUNNER" = "stable_crl.py" ]; then
    COMMAND+=(
      --env-id "$ENV_ID"
      --num-timesteps "$BASE_STEPS"
      --architecture "$ARCHITECTURE"
      --num-blocks "$NUM_BLOCKS"
      --hidden-dim "$HIDDEN_DIM"
      --mjx-impl "$MJX_IMPL"
    )
  else
    COMMAND+=(
      --task-sequence "$TASK_SEQUENCE"
      --base-steps "$BASE_STEPS"
      --steps-per-task "$STEPS_PER_TASK"
      --max-cubes "$MAX_CUBES"
      --boundary-checkpoint-dir "$RUN_CHECKPOINT_DIR"
      --task-data-version "$TASK_DATA_VERSION"
      --mjx-impl "$MJX_IMPL"
      --continual-eval-repeats "$CONTINUAL_EVAL_REPEATS"
    )
    COMMAND+=("$(bool_flag resume "$RESUME")")
    COMMAND+=("$(bool_flag eval-next-task "$EVAL_NEXT_TASK")")
    COMMAND+=("$(bool_flag eval-previous-tasks "$EVAL_PREVIOUS_TASKS")")
    COMMAND+=("$(bool_flag report-retention-metrics "$REPORT_RETENTION_METRICS")")
    COMMAND+=("$(bool_flag log-continual-eval "$LOG_CONTINUAL_EVAL")")
    COMMAND+=("$(bool_flag wandb-eval-tables "$WANDB_EVAL_TABLES")")
  fi

  if [ "$RUNNER" = "continual_crl.py" ]; then
    COMMAND+=(
      --actor-lifecycle "$ACTOR_LIFECYCLE"
      --critic-lifecycle "$CRITIC_LIFECYCLE"
      --vanilla-width "$VANILLA_WIDTH"
      --vanilla-depth "$VANILLA_DEPTH"
      --observation-layout "$OBSERVATION_LAYOUT"
      --vanilla-network-type "$VANILLA_NETWORK_TYPE"
      --architecture "$ARCHITECTURE"
      --num-blocks "$NUM_BLOCKS"
      --hidden-dim "$HIDDEN_DIM"
    )
  elif [ "$RUNNER" = "continual_dcc.py" ]; then
    COMMAND+=(
      --dcc-task-width "$DCC_TASK_WIDTH"
      --dcc-task-depth "$DCC_TASK_DEPTH"
      --dcc-combine-mode "$DCC_COMBINE_MODE"
      --dcc-goal-encoder-mode "$DCC_GOAL_ENCODER_MODE"
      --architecture "$ARCHITECTURE"
      --num-blocks "$NUM_BLOCKS"
      --hidden-dim "$HIDDEN_DIM"
    )
    COMMAND+=("$(bool_flag carry-actor "$CARRY_ACTOR")")
    COMMAND+=("$(bool_flag dcc-carry-shared "$DCC_CARRY_SHARED")")
  fi

  COMMAND+=("$(bool_flag use-pd "$USE_PD")")
  COMMAND+=("$(bool_flag track "$TRACK")")
  COMMAND+=("$(bool_flag save-checkpoint "$SAVE_CHECKPOINT")")
  COMMAND+=("$(bool_flag record-videos "$RECORD_VIDEOS")")
  COMMAND+=("$(bool_flag visualize-samples "$VISUALIZE_SAMPLES")")

  echo "[slot $slot] config=$CONFIG_IDX run=$RUN_ID"
  printf '[slot %s] command: ' "$slot"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'

  if [ "$DRY_RUN" = "true" ]; then
    continue
  fi

  (
    echo "run_id=$RUN_ID config_index=$CONFIG_IDX"
    echo "host=$(hostname) account=${SLURM_JOB_ACCOUNT:-unknown}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    fi
    printf 'command: '
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    "${COMMAND[@]}"
  ) >"${RUN_LOG_PREFIX}.out" 2>"${RUN_LOG_PREFIX}.err" &
  PIDS+=("$!")
  RUN_IDS+=("$RUN_ID")
done

if [ "$DRY_RUN" = "true" ]; then
  echo "Dry run complete; no experiments launched."
  exit 0
fi

FAILURES=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "completed: ${RUN_IDS[$index]}"
  else
    status=$?
    echo "failed (${status}): ${RUN_IDS[$index]}" >&2
    FAILURES=$((FAILURES + 1))
  fi
done

if [ "$FAILURES" -ne 0 ]; then
  echo "$FAILURES experiment(s) failed" >&2
  exit 1
fi
echo "All experiments assigned to this GPU completed."
