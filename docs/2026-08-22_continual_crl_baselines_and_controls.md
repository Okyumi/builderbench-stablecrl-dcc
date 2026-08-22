# Continual CRL baselines and individual-task controls

Date: 2026-08-22

This record documents the vanilla continual-CRL implementation, the exact
parameter lifecycles, the single-task controls, and the updated NYU Torch
experiment matrix. It extends, rather than replaces, the DCC implementation
record from 2026-08-21.

## Did the continual setup add another encoder?

Yes, but the precise answer is that the semantic continual path replaces the
upstream flat, position-indexed encoders with permutation-safe set networks;
DCC then decomposes the state-action side into shared and task-specific
parts. It is not one extra encoder appended to the unchanged StableCRL model.

The upstream algorithm path remains `stable_crl.py` and is used for the
individual-task reproduction control. It receives the original observation
and goal shapes and retains the paper's scaled residual architecture. Only
the runtime portability and logging defaults described below changed.

The semantic DCC path in `stable_crl_dcc.py` uses:

```text
b_shared(s, a) -> shared per-cube features and symmetric pooling
h_phi(b_shared(s, a)) -> shared contrastive state-action representation
phi_task(s, a) -> freshly initialized task-specific representation
z_sa = h_phi(...) + phi_task(...)             [add mode]
z_sa = concat(h_phi(...), phi_task(...))       [concat mode]
psi_shared(g) -> masked-set goal representation
h_dyn(shared per-cube features) -> next-position auxiliary prediction
```

`b_shared`, `h_phi`, `h_dyn`, `psi_shared`, and an optional goal projection
are carried between tasks. `phi_task` is reset for every task and stored in a
task bank for retrospective evaluation. The actor is a masked set actor with
a pointer-style selector, rather than the upstream flat actor.

The vanilla semantic control added on 2026-08-22 uses only:

```text
z_sa = phi(s, a)
z_g  = psi(g)
```

Both are masked-set encoders. There is no DCC shared/task split, no
task-specific head, no task bank, and no dynamics auxiliary. The actor is the
same permutation-equivariant pointer actor used by DCC. This is intentional:
the comparison between semantic vanilla CRL and DCC holds the wrapper and
action representation fixed and changes the critic decomposition.

## Continual environment contract

`SemanticPadWrapper` converts every supported BuilderBench task into one
fixed contract determined before training by `max_cubes`.

Each cube contributes 14 observation values: position (3), quaternion (4),
linear velocity (3), angular velocity (3), and a previous-selection indicator
(1). A validity mask of length `max_cubes` follows the cube records. Goals
contain three position values per slot plus a validity mask. At the default
capacity of eight cubes, observations have 120 values and goals have 32.

Padding is never treated as semantic alignment. Learned per-cube functions
share weights across slots; pooling is masked and symmetric; selector logits
for padded cubes are suppressed. Cube permutations therefore leave critic
and goal embeddings unchanged and only permute the actor's selected cube.

The task sequence is validated before training. Every goal receives a global
identifier derived from task-data version, cube count, local task index, and
a canonical geometry hash. This avoids assuming that the same local task
number has the same meaning for different cube counts.

Every task phase creates a fresh environment, replay buffer, critic optimizer,
and actor optimizer. The selected network parameters may be restored into
those fresh optimizers according to the lifecycle below. Task-boundary
checkpoints are atomic and validate the complete training recipe before
resume. After phase `t`, the current model is evaluated on tasks `0..t`,
producing the lower-triangular continual evaluation matrix.

Both continual entry points lock their critic family at the command-line type
level (`dcc` for `continual_dcc.py`, `vanilla` for `continual_crl.py`). Their
resume recipes include environment/evaluation shape controls, MJX backend,
PD settings, replay/sampling settings, network sizes, and lifecycle choices;
a changed recipe requires a fresh checkpoint directory.

## Vanilla CRL lifecycle baselines

The new entry point is `continual_crl.py`. `critic_family` is fixed to
`vanilla`; the lifecycle flags are explicit and recorded in the checkpoint
recipe.

### Reset-reset

```text
--actor-lifecycle reset --critic-lifecycle reset
```

At every task boundary, neither actor nor critic parameters are exposed to
the next task. Both are freshly initialized. Replay and optimizer state are
also fresh. This is independent training in sequence, using the same wrapper,
logging, budgets, and evaluation code as the continual experiment.

### Persistent-persistent

```text
--actor-lifecycle persistent --critic-lifecycle persistent
```

The full actor and the full monolithic contrastive critic are restored for
the next task. Replay remains task-local and optimizer state restarts. This
isolates parameter persistence without mixing old transitions or optimizer
moments into later tasks.

Both requested vanilla continual baselines use repetition factor 1 (plain
CRL). The matched individual-task reproduction controls use repetition factor
12 because they are testing the Scaling-the-Horizon/CRTR paper setting. The
continual DCC dynamics on/off cells also remain at repetition factor 1; only
the explicitly named CRTR probes use 12.

Mixed reset/persistent modes remain representable for diagnostics, but the
committed primary grid contains the two requested endpoints only.

## Individual-task controls before continual conclusions

The first 18 array cells form three matched groups over
`creative-3-task1` and `creative-4-task1`, with seeds 5, 6, and 7 and the
same 200M environment-step budget:

1. **Upstream reproduction (indices 0--5).** Run the upstream
   `stable_crl.py` algorithm
   with the paper-style scaled block network, eight blocks, PD-5,
   repetition factor 12, entropy cost 0.01, and log-sum-exp cost 0.1.
2. **Wrapper control (indices 6--11).** Run one-task `continual_crl.py` with
   semantic padding, monolithic masked-set critic, and pointer actor. This
   measures the combined effect of the wrapper and required
   permutation-safe architecture.
3. **Single-task DCC (indices 12--17).** Run one-task `continual_dcc.py` on
   the same goals. Because there is only one phase, no cross-task transfer can
   explain the result.

The wrapper cannot be validated by comparing DCC against upstream alone,
because both representation and decomposition changed. The primary diagnostic
is upstream reproduction versus wrapped vanilla with per-seed learning
curves. Report final success rate, final return, success-rate AUC against
environment steps, and bootstrap confidence intervals across the three
seeds. Only interpret continual transfer after confirming that the wrapper
control is competitive; if it is worse, tune semantic-network capacity and
selector temperature on the single-task controls without looking at
continual-test results.

## Torch experiment matrix and run order

`experiment_configs.py` now emits 36 deterministic cells:

```text
0--5    upstream StableCRL individual-task reproductions
6--11   semantic-wrapper vanilla CRL individual-task controls
12--17  DCC individual-task controls
18--20  continual vanilla CRL reset-reset
21--23  continual vanilla CRL persistent-persistent
24--26  continual DCC with masked dynamics
27--29  continual DCC without masked dynamics
30--35  existing one-task DCC CRTR probes
```

Run the requested controls and vanilla baselines first:

```bash
python experiment_configs.py --list
DRY_RUN=true CONFIG_INDEX=0 bash DRAFT.sh
sbatch --account=torch_pr_XXX_XXXXX --array=0-23 DRAFT.sh
```

After inspecting the matched individual-task results, launch the DCC stage:

```bash
sbatch --account=torch_pr_XXX_XXXXX --array=24-35 DRAFT.sh
```

The full array is `0-35`. `DRAFT.sh` builds runner-specific commands, so
upstream runs do not accidentally receive continual-wrapper or DCC flags.
Every cell and seed has a unique run name and checkpoint directory.

The upstream learner's algorithm and networks are unchanged. Its two
machine-specific startup constants were made portable: EGL is selected only
on Linux and only when `MUJOCO_GL` is unset, and the JAX compilation cache
uses `JAX_COMPILATION_CACHE_DIR`. `DRAFT.sh` points that cache into scratch;
the previous hard-coded Princeton filesystem path is not valid on NYU Torch.
The inherited author W&B entity default was also removed, so runs use the
caller's account/project unless `WANDB_ENTITY` is set explicitly. The optional
episode-length type annotation was corrected for command-line validation. A
`--mjx-impl warp|jax` runtime selector was added with `warp` as the unchanged
production default; `jax` enables CPU correctness smokes.

## Padding capacity recommendation

For the current benchmark, choose `max_cubes` equal to the largest task before
task zero; eight is the committed default. Do not increase it during a run,
because replay shapes, compiled programs, and the validated checkpoint recipe
would change.

For an open-ended curriculum, use capacity buckets (for example 4, 8, and 16)
and a compile cache keyed by bucket size. The learned set-network parameter
shapes do not depend on slot count, so parameters can transfer while tensor
shapes remain bounded. A bucket-aware driver is still future work; the
current driver intentionally rejects an insufficient fixed capacity rather
than truncating cubes.

## Files implemented on 2026-08-22

- `continual/vanilla_networks.py`: monolithic masked-set CRL critic and shared
  pointer actor adapter.
- `stable_crl_dcc.py`: selects DCC or vanilla semantic critic families and
  disables the DCC auxiliary for vanilla.
- `continual_crl.py`: sequential vanilla driver, reset/persistent lifecycle,
  manifests, resume, and seen-task evaluation.
- `continual_dcc.py`: locks the DCC critic family and expands recipe validation
  to cover environment, evaluation-shape, and MJX settings.
- `experiment_configs.py`: baseline-first 36-cell Torch grid.
- `DRAFT.sh`: runner-aware upstream, vanilla, and DCC command generation.
- `stable_crl.py`: runtime-only EGL/cache portability for NYU Torch; no
  algorithm, network, sampling, or training-hyperparameter change, plus an
  MJX backend selector whose default remains Warp.
- `tests/test_vanilla_networks.py`: critic invariance and actor equivariance.
- `tests/test_continual_crl.py`: lifecycle and recipe tests.
- `tests/test_experiment_configs.py`: array ordering and runner command tests.
- `AGENTS.md`: makes dated implementation notes an explicit repository rule.

## Validation and limitations

Completed locally without launching expensive experiments:

- Python syntax compilation for all new entry points;
- shell syntax validation;
- all 36 configurations enumerate and validate;
- dry-run command generation for upstream, vanilla, and DCC runners;
- all 21 unit tests, including JAX network tests;
- an upstream `stable_crl.py` CPU learner smoke through collection,
  contrastive updates, and evaluation with finite losses;
- a one-task reset/reset CPU learner smoke with finite actor, InfoNCE, and
  log-sum-exp losses, evaluation, and boundary checkpointing;
- a two-task persistent/persistent CPU learner smoke confirming actor/critic
  transfer, fresh task-local training state, seen-task evaluation, boundary
  checkpoints, and completed-prefix resume.

The production 200M-step cells require a Torch GPU allocation and have not
been claimed as completed by this implementation commit. Task-boundary resume
still does not resume an interrupted phase's replay buffer or optimizer, so
preemptible jobs remain unsuitable.
