# Residual DCC without dynamics (2026-08-23)

## Scope and decision

The completed padding diagnostics establish the following single-task final
strict-success results across seeds 5, 6, and 7:

| Control | 3 block | 4 block |
| --- | ---: | ---: |
| Raw upstream StableCRL | 98.44 +/- 1.28% | 97.66 +/- 1.91% |
| Semantic padding + upstream residual network | 99.22 +/- 0.64% | 93.75 +/- 1.69% |
| Semantic set network, capacity 4 | 0.00 +/- 0.00% | 0.00 +/- 0.00% |

All 18 diagnostic training configurations reached 200,933,376 environment
steps. Right-sizing the set capacity from eight cubes to four did not produce
nonzero success at any of the 50 evaluation checkpoints. The fixed semantic
wrapper is therefore viable; the masked-set model is not a safe foundation
for the continual protocol.

This change replaces the DCC set model with the proven upstream residual
model and removes the dynamics auxiliary completely. There is no dynamics
module, parameter group, relabelled next-state target, loss, metric, recipe
field, experiment field, or Torch command-line flag.

## Meaning of the residual DCC decomposition

"Port DCC's shared/task-specific decomposition onto the upstream residual
encoders" means that DCC changes parameter lifecycle and factorization without
changing the successful input representation or base neural architecture.

The state-action critic representation is:

```text
z_shared(s, a) = phi_shared(s, a)
z_task_t(s, a) = phi_task_t(s, a)
z_sa(s, a, t)  = z_shared + z_task_t       # default add mode
z_goal(g)      = psi_shared(g)
Q(s, a, g, t)  = -||z_sa - z_goal||_2
```

`phi_shared`, `psi_shared`, and the actor are the same flat residual MLP
families used by the upstream StableCRL adapter: eight residual blocks,
hidden width 1024, LayerNorm, Swish, residual-depth scaling, and the same
tanh-Gaussian action parameterization by default. `phi_task_t` is a smaller
residual state-action adapter, defaulting to four blocks of width 256.

The task adapter's final projection is initialized to exactly zero. In the
default additive mode, the initial task contribution is therefore zero and
the complete DCC state-action representation is exactly equal to the upstream
flat representation. Actor, shared state-action, and goal initialization use
the same top-level RNG streams as the upstream control. This creates a strict
functional parity guarantee at initialization, not merely a similar parameter
count.

At a task boundary:

- the replay buffer and optimizer states are reset;
- the actor is carried or reset according to `--carry-actor`;
- `phi_shared`, `psi_shared`, and optional `psi_proj` are carried when
  `--dcc-carry-shared` is enabled;
- a fresh zero-output `phi_task_t` is initialized;
- the completed task adapter is saved in `task_bank` for seen-task evaluation.

The SGCRL-style `add|concat` combination and `shared|projected` goal modes
remain implemented. Concatenation doubles the critic representation and adds
a goal projection automatically. Add/shared is the parity-safe default.

## Stable task-index semantics

The environment still uses `SemanticPadWrapper` and a fixed `max_cubes` for
the complete run. Each BuilderBench object index occupies the same semantic
cube record across tasks, the previous continuous selector is represented as
a selected-object flag, and an explicit validity mask distinguishes real
cubes from padded slots. The residual network consumes this fixed vector
directly. It does not use the failed masked pooling or pointer actor.

For the expanding-stack protocol, `max_cubes=4` is fixed before task 0. For
the one-cube goal-only protocol, `max_cubes=1`. Capacity cannot grow inside a
saved flat-network run because changing the input width changes parameter and
replay shapes.

## Dynamics removal

The following implementation surfaces were removed:

- `EquivariantDynamicsHead`, `DCCNetworks.h_dyn`, and `apply_dynamics`;
- `masked_dynamics_mse`;
- `next_achieved_goal` insertion and `flatten_crl_dcc_fn`;
- `dcc_dyn_weight` and `dcc_dyn_weight_after_task0`;
- `critic_dynamics_loss` W&B logging;
- `dcc_shared_width` and `dcc_shared_depth`, because the shared model now uses
  the upstream `architecture`, `num_blocks`, and `hidden_dim` settings.

The critic objective is now only the StableCRL contrastive objective:

```text
L_critic = L_InfoNCE + L_logsumexp
```

The checkpoint recipe is versioned as
`stablecrl-dcc-flat-residual-no-dynamics-v3`. Set-network DCC checkpoints are
intentionally incompatible and must not be resumed into this architecture.

## Boundary evaluation and W&B matrices

`continual_crl.py` and `continual_dcc.py` now default to five deterministic
evaluation batches per task-by-phase cell. Each batch contains 128 episodes
under the Torch defaults, for 640 episodes per cell. The compiled evaluator is
reused while its PRNG stream advances deterministically.

The existing metric names store the mean across evaluation batches. Matching
`*_std` fields store population standard deviation across those batches, and
each JSONL row records `eval/repeats` and `eval/num_episodes`. W&B receives:

- the long-form task-by-phase table with means, standard deviations, and
  evaluation counts;
- strict-success and easy-success mean matrices;
- strict-success and easy-success standard-deviation matrices;
- the existing forgetting, backward-transfer, forward-transfer, per-task,
  mean-seen, and minimum-seen scalars computed from the mean matrix.

`--continual-eval-repeats` controls the repeat count. Five repeats add work
only at task boundaries, not inside training updates, so they do not change
the 200M-step training budget. W&B uploads remain boundary-only and
best-effort after the local JSONL is safely written.

## Experiment registry and gate

The registry now contains 72 configurations:

- 36--53: completed padding diagnostics;
- 54--59: residual no-dynamics DCC parity gate on 3-block and 4-block tasks,
  CRTR-12, seeds 5/6/7, fixed capacity 4;
- 60--65: flat-upstream CRL goal-only and expanding-stack protocols;
- 66--71: residual no-dynamics DCC goal-only and expanding-stack protocols.

The earlier dynamics on/off labels at indices 24--29 are replaced by
no-dynamics add/shared and concat/projected controls. Historical W&B runs keep
their original names and remain attributable to the older Git commit; new
results from this commit use the new groups and checkpoint recipe.

The DCC gate is:

1. both 3-block and 4-block mean strict success are at least 90%;
2. each mean is within five percentage points of its semantic-flat control;
3. no seed has zero strict success;
4. boundary re-evaluation is inspected for large repeat standard deviation.

Do not launch indices 66--71 if this gate fails. Indices 60--65 are
representation-safe controls, but using the same post-change code for both
protocol halves keeps their evaluation matrices directly comparable.

## Torch launch commands

`DRAFT.sh` defaults to the six residual-DCC gate configurations and declares
Slurm array `0-5`:

```bash
DRY_RUN=true CONFIG_INDEX=54 bash DRAFT.sh

EXPERIMENT_STAGE=dcc_residual_gate \
  sbatch --account=torch_pr_XXX_XXXXX --array=0-5 DRAFT.sh
```

After the gate passes:

```bash
EXPERIMENT_STAGE=protocol_baselines \
  sbatch --account=torch_pr_XXX_XXXXX --array=0-5 DRAFT.sh

EXPERIMENT_STAGE=protocol_dcc \
  sbatch --account=torch_pr_XXX_XXXXX --array=0-5 DRAFT.sh
```

The combined twelve-cell launch remains available as
`EXPERIMENT_STAGE=protocol` with array `0-11`.

## Validation

The following validation passed in the implementation workspace:

- Python byte-compilation for the continual package, both continual drivers,
  the DCC learner, experiment registry, and tests;
- `bash -n DRAFT.sh`;
- `git diff --check`;
- 19 dependency-light unit tests covering stage/index generation, runner-aware
  dry-run commands, semantic/grouped layouts, task manifests, repeated
  evaluation mean/std/count aggregation, atomic evaluation persistence,
  continual scalars, and W&B mean/std matrix construction;
- a direct dry run of global config 54, which emits residual blocks 8, hidden
  width 1024, no dynamics flags, five evaluation repeats, capacity 4, CRTR-12,
  and the correct no-carry-actor parity setting.

JAX/Flax network tests were added for exact DCC-to-upstream functional parity,
zero task-adapter output, add/concat shapes, goal projection, and the absence
of a dynamics API or parameter group. They could not be executed locally
because the workspace does not contain the pinned JAX/Flax stack and had
insufficient temporary disk space to install it. Run the full test suite and a
short finite-loss initialization smoke test in the existing Torch virtual
environment before submitting the 200M-step gate array.

## Known limitations

- The flat residual model is intentionally slot-sensitive. Cross-task object
  indices must retain BuilderBench's stable semantic meaning; arbitrary
  permutation augmentation would require a separately validated encoder.
- Zero-output initialization guarantees functional parity only for the
  default additive/shared DCC mode. Concat/projected remains an explicit
  architecture ablation.
- Five evaluation batches reduce matrix noise but do not replace reporting
  uncertainty across independent training seeds.
