# Experiment Commands
All runs are trained for 200,000,000 timesteps.

Baseline CRL (Short horizon/PD controller)
```
python stable_crl.py --env_id creative-4-task1 --use_pd --pd_duration 5 --architecture default
```
StableCRL (Short horizon/PD controller)
```
python stable_crl.py --env_id creative-4-task1 --use_pd --pd_duration 5 --architecture default --repetition_factor 12 --entropy_cost 0.01 
```
StableCRL Scaled (Short horizon/PD controller)
```
python stable_crl.py --env_id creative-4-task1 --use_pd --pd_duration 5 --architecture block --num_blocks 8 --repetition_factor 12 --entropy_cost 0.01 
```

StableCRL Scaled (Long horizon/Raw controller)
```
python stable_crl.py --env_id creative-4-task1 --no-use_pd --architecture block --num_blocks 8 --repetition_factor 12 --entropy_cost 0.01 
```

PPO (Short horizon/PD controller)
```
python ppo_pd.py --env_id creative-4-task1 --use_pd --pd_duration 5
```

PPO (Long horizon/Raw controller)
```
python ppo_pd.py --env_id creative-4-task1 --no-use_pd
```

## DCC continual RL

This fork adds a Decomposed Contrastive Critic on top of StableCRL while
retaining the paper implementation's PD horizon reduction, trajectory
repetition, entropy regularization, and log-sum-exp regularization.

Single-task DCC smoke run:

```bash
python stable_crl_dcc.py \
  --env-id creative-2-task1 \
  --num-timesteps 1000000 \
  --num-envs 256 \
  --repetition-factor 12 \
  --entropy-cost 0.01 \
  --no-track \
  --no-record-videos \
  --no-visualize-samples \
  --mjx-impl jax
```

Use `--mjx-impl warp` (the default) for the paper's GPU path; `jax` is useful
for CPU initialization and correctness smoke tests.

Continual run (the default curriculum is pick/place followed by increasingly
long stacks):

```bash
python continual_dcc.py \
  --task-sequence creative-1-task1,creative-1-task2,creative-2-task1,creative-3-task1,creative-4-task1 \
  --base-steps 200000000 \
  --steps-per-task 200000000 \
  --repetition-factor 12 \
  --entropy-cost 0.01
```

Core SGCRL/DCC ablations are available directly, for example:

```bash
python continual_dcc.py \
  --dcc-combine-mode concat \
  --dcc-goal-encoder-mode shared \
  --dcc-dyn-weight 1.0 \
  --dcc-dyn-weight-after-task0 0.0 \
  --dcc-task-depth 4
```

Changing an algorithmic setting requires a fresh boundary-checkpoint
directory. Resume checkpoints store and validate the full training recipe.

The continual driver writes an immutable task manifest, atomic task-boundary
checkpoints, and a lower-triangular evaluation matrix in
`checkpoints/continual_dcc/continual_eval.jsonl`.

### Why this is more than padding

Variable-cube inputs are represented as masked sets. Shared per-cube encoders
and symmetric pooling make critic and goal representations invariant to cube
permutation. The actor uses an equivariant pointer head: it scores each valid
cube, maps the selected slot to BuilderBench's continuous selector action, and
conditions motion on that cube. Padded slots are masked out of pooling,
selection, and the DCC dynamics loss.

Task identities use a versioned hash of canonical goal geometry. The hash is
invariant to cube permutation and horizontal translation while preserving
height above the ground plane, so `pick` and `place` remain distinct skills.
The default capacity is eight cubes (`--max-cubes 8`). Set it to the largest
known task before training. For open-ended curricula, use 4/8/16 capacity
buckets and reuse the shared set-encoder parameters across separately compiled
shapes rather than assigning permanent semantics to padded indices.

## Implementation notes

- `docs/2026-08-21_stablecrl_upstream_audit.md` records provenance and the
  repository decision.
- `docs/2026-08-21_dcc_continual_implementation.md` records the DCC lifecycle,
  SGCRL parity, semantic layout, validation, and capacity recommendation.
- `docs/2026-08-21_documentation_convention.md` defines the required dated
  implementation-note convention.
- `docs/2026-08-21_torch_hpc_experiments.md` documents the Torch Slurm array,
  SGCRL-matched cells, environment setup, and checkpoint layout.
- `docs/2026-08-22_continual_crl_baselines_and_controls.md` documents the
  encoder design, vanilla reset/reset and persistent/persistent lifecycles,
  individual-task controls, and the baseline-first run order.

## NYU Torch HPC

The active batch contains 36 runs. Indices 0--23 are the baseline-first stage:
upstream individual-task reproductions, wrapper-only vanilla controls,
single-task DCC controls, and vanilla continual reset/reset and
persistent/persistent. Indices 24--35 contain the continual DCC and CRTR
cells. Every group uses matched seeds 5/6/7.

```bash
python experiment_configs.py --list
DRY_RUN=true CONFIG_INDEX=0 bash DRAFT.sh
my_slurm_accounts
sbatch --account=torch_pr_XXX_XXXXX --array=0-23 DRAFT.sh
# After validating the individual-task controls:
sbatch --account=torch_pr_XXX_XXXXX --array=24-35 DRAFT.sh
```

Torch requires the allocation account and does not require a manually chosen
partition. `DRAFT.sh` defaults to one run per GPU and writes a separate
checkpoint directory for every configuration and seed.

## Demos

### Long Horizon (3 Stack)

![Long Horizon (3 Stack)](gifs/3_raw.gif)

### PD-5 (5 stack)

![PD-5 (5 stack)](gifs/5_pd.gif)

<!--1) For the creative cube mode, play around with action scales to make sure which values are best for RL.
2) Directly get mat from data.xmat or something instead of doing mjx.math.quat_to_mat.
3) Try reward normalization for ppo.-->
