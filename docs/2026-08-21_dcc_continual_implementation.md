# StableCRL-DCC continual implementation

Date: 2026-08-21

## Scope

This implementation starts from the paper-author StableCRL BuilderBench code
and adds a sequential DCC experiment path without changing the baseline
`stable_crl.py` entry point.

The new entry points are:

- `stable_crl_dcc.py` for one task;
- `continual_dcc.py` for a sequence with task-boundary transfer, evaluation,
  and resume;
- `continual/` for the DCC networks, semantic layout, environment wrapper,
  and task manifest.

## DCC lifecycle

The shared groups `b_shared`, `h_phi`, `h_dyn`, `psi_shared`, and optional
`psi_proj` are carried across task boundaries. `phi_task` and its optimizer
state are reinitialized for every new task. A copy of each trained
`phi_task` is kept in the task bank for lower-triangular evaluation. The actor
is carried by default and can be reset with `--no-carry-actor`.

The dynamics auxiliary predicts the next cube positions from
`b_shared(s, a)`. `flatten_crl_dcc_fn` keeps the aligned next achieved goal
after HER flattening, and the loss ignores every padded slot.

## SGCRL controls retained

The port deliberately matches the core DCC ablations used in SGCRL:

- `--dcc-combine-mode add|concat`;
- `--dcc-goal-encoder-mode shared|projected`;
- `--dcc-dyn-weight` and `--dcc-dyn-weight-after-task0`;
- `--dcc-task-width`, `--dcc-task-depth`, `--dcc-shared-width`, and
  `--dcc-shared-depth`;
- `--dcc-carry-shared` as the transfer sanity check.

Concatenation doubles the critic representation width and automatically adds
a shared goal projection. `projected` adds the same projection when the
critic uses addition. Checkpoints store the full experimental recipe and
reject resume when any of these semantics change.

The previous BuilderBench DCC port also explored task-specific,
partial-shared, and decomposed goal encoders. They are not silently aliased to
the permutation-safe goal set encoder here; they should be added as explicit
task-banked modules if that richer ablation is needed.

## StableCRL behavior retained

- PD horizon reduction (`--use-pd`, `--pd-duration`);
- trajectory repetition and in-trajectory negatives
  (`--repetition-factor`);
- entropy regularization and optional decay;
- squared log-sum-exp critic regularization;
- 200M-step paper budget defaults;
- Warp GPU execution by default, with JAX MJX for CPU correctness tests.

The original monolithic residual block actor/critic is replaced on the DCC
path by scalable shared/task set encoders because a flattened residual MLP
would restore slot-index semantics. Width and depth flags provide the scaling
axis while preserving permutation invariance.

## Semantic input and action contract

Each cube becomes a 14-value feature record: position, quaternion, linear
velocity, angular velocity, and a flag identifying the previously selected
physical cube. A validity mask is appended. Goals contain padded cube
positions plus the same kind of mask.

For `max_cubes = 8`, observation size is 120 and goal size is 32. Per-cube
encoders share their weights, pooling is masked and symmetric, and the actor
scores valid cubes with a pointer head before mapping that distribution back
to BuilderBench's continuous selector action. Padding never becomes a goal,
negative, selected object, or dynamics target.

## If the padding capacity is insufficient

For a known curriculum, set `--max-cubes` to the largest task **before the
first task**. Do not enlarge it halfway through a saved run: the checkpoint
recipe validator rejects that change because replay shapes and compiled JAX
programs would differ.

For a large or open-ended curriculum, use capacity buckets such as 4, 8, and
16 cubes. Compile one semantic wrapper per bucket, keep the same learned DCC
parameters, and select the smallest bucket that fits each task. This is viable
because every learned per-cube transformation and all pooled representations
have parameter shapes independent of the number of slots; only the compiled
tensor shape changes. The continual driver will need a bucket-aware compile
cache before mixing capacities inside one run.

If tasks eventually contain hundreds of objects, replace dense padded replay
with packed/ragged storage at the data boundary and materialize bounded
mini-batches for JAX. Keep the masked set encoder and semantic task manifest;
do not return to positional slot meanings or per-step sorting.

## Validation completed

- semantic layout and task-hash unit tests;
- permutation invariance of critic and goal features;
- equivariance of the actor selector and dynamics prediction;
- masked dynamics targets;
- SGCRL add/concat and shared/projected shape tests;
- atomic checkpoint round trip and recipe mismatch rejection;
- CPU single-task smoke training with finite DCC loss;
- CPU two-task continual transfer, task-bank evaluation, checkpointing, and
  resume.
