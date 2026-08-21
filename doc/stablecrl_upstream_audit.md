# StableCRL upstream audit and DCC migration plan

## Repository decision

The relevant implementation is the `david` branch of
`David-Yan1/builderbench` at commit `6e8d56d`. David Yan is an author of
*Scaling the Horizon of Contrastive Reinforcement Learning*, and this branch
contains the paper-facing `stable_crl.py`, GPU/MJX BuilderBench environments,
PD horizon reduction, trajectory repetition (in-trajectory negatives),
entropy regularization, log-sum-exp regularization, and the scaled block
critic architecture.

The user repository `Okyumi/builderbench` is already the available GitHub
fork in the same repository network. Its `main` branch contains the existing
DCC continual implementation and the full modern BuilderBench task suite.
GitHub cannot create a second fork of the same network for the same user, so
the correct setup is:

- `user/main`: preserve the existing DCC and 51-task work.
- `paper/david`: track the paper-author implementation.
- a feature branch: port StableCRL components into DCC in reviewable pieces.

Do not merge `paper/david` wholesale. It replaces the simulator generation,
deletes the modern task suite, and moves most RL files; a merge would silently
discard exactly the continual benchmark that DCC needs.

## StableCRL features to port

1. **Horizon reduction.** Expose the PD controller and `pd_duration`; report
   both raw environment steps and effective policy decisions.
2. **In-trajectory negatives.** Port `repetition_factor` sampling from
   `stable_crl.py`. Repeat a sampled trajectory before HER flattening, then
   draw independent `(anchor, future)` pairs for every repetition.
3. **Critic regularization.** Preserve the squared log-sum-exp penalty and
   make its coefficient explicit in DCC configurations.
4. **Policy regularization.** Match the tuned entropy coefficient and record
   it in checkpoints so resumed runs cannot mix recipes.
5. **Scaled critic.** Port the block architecture behind a flag and keep the
   current DCC decomposed critic as the default until parity tests pass.
6. **Training budget.** Paper commands use 200M environment steps. Continual
   comparisons must fix either total environment steps or total policy
   decisions; publish both because PD changes the conversion.

## Continual task semantics

BuilderBench local task numbers are not semantic labels. For example,
`cube-1-task1` and `cube-2-task1` do not denote the same skill. Continual
experiments must use a manifest whose identity is independent of list order:

```text
global_id = <task-data-version>:<num-cubes>:<local-task-index>:<goal-hash>
```

The goal hash should be computed from a canonical structure representation:

1. translate target X/Y coordinates so the lowest lexicographic cube is at
   the horizontal origin while preserving height above the ground plane;
2. quantize coordinates to the BuilderBench grid;
3. sort cube coordinates lexicographically;
4. hash the sorted integer coordinate array.

This makes the identifier invariant to cube permutation and horizontal
translation without collapsing `pick` into `place`. Store `num_cubes`, source file, local index, human skill label,
canonical coordinates, and goal hash in the manifest. Never use the position
of a task in `task_sequence` as its persistent identity.

Padding alone does **not** establish semantic correspondence. The existing
`PaddedEnvWrapper` gives feature-type consistency (position, quaternion,
velocity, and so on) but preserves arbitrary cube identity. DCC therefore
needs one of these explicitly selected contracts:

- permutation-equivariant cube encoder plus masked pooling (preferred);
- canonical slot assignment at reset, kept fixed for the episode;
- fixed object identity from task metadata, with a validator rejecting
  inconsistent task files.

Dynamic per-step sorting is invalid because cube slots can swap discontinuously
when positions cross. For the current MLP implementation, canonical assignment
once at reset is the minimum safe bridge; a set encoder is the robust target.

## Migration checkpoints

1. Import the paper branch as a read-only comparison remote and add unit tests
   for its replay sampling invariants.
2. Add StableCRL sampling and regularization flags to the existing DCC driver;
   default them off for checkpoint compatibility.
3. Add a versioned task manifest and validate every sequence before JAX model
   initialization.
4. Run CPU shape/sampling tests, then a two-task GPU smoke test with finite
   DCC dynamics loss across the boundary.
5. Compare CRL and StableCRL on one short- and one long-horizon task before
   launching continual runs.

## Provenance

- Original benchmark: `RajGhugare19/builderbench`, `main`.
- Paper-author implementation: `David-Yan1/builderbench`, `david`, `6e8d56d`.
- User fork and DCC implementation: `Okyumi/builderbench`, `main`, audited at
  `f084a94`.
