# StableCRL upstream audit and DCC migration plan

Date: 2026-08-21

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
this work is published as a separate repository based on the paper-author
branch rather than as another GitHub fork.

Do not merge the original BuilderBench `main` branch wholesale into this
codebase. It replaces the simulator generation, deletes the paper branch's
layout, and moves most RL files; a merge could silently discard the exact
StableCRL implementation being adapted.

## StableCRL features retained in the DCC path

1. **Horizon reduction.** The PD controller and `pd_duration` remain
   configurable. Report raw environment steps and effective policy decisions.
2. **In-trajectory negatives.** `repetition_factor` repeats a sampled
   trajectory before HER flattening and draws independent anchor/future pairs.
3. **Critic regularization.** The squared log-sum-exp penalty remains an
   explicit DCC configuration.
4. **Policy regularization.** The tuned entropy coefficient and optional
   annealing remain part of the training recipe.
5. **Scaling.** The DCC path uses configurable shared/task set-encoder width
   and depth. The original monolithic residual block actor/critic cannot be
   reused verbatim because it is not permutation safe.
6. **Training budget.** Paper commands use 200M environment steps. Continual
   comparisons must fix either total environment steps or total policy
   decisions and publish both because PD changes the conversion.

## Continual task semantics

BuilderBench local task numbers are not semantic labels. For example,
`creative-1-task1` and `creative-2-task1` do not denote the same skill.
Continual experiments use a manifest whose identity is independent of list
order:

```text
global_id = <task-data-version>:<num-cubes>:<local-task-index>:<goal-hash>
```

The goal hash is computed from a canonical structure representation:

1. Quantize coordinates to the BuilderBench grid.
2. Sort cube coordinates lexicographically.
3. Translate target X/Y coordinates to a horizontal origin while preserving
   height above the ground plane.
4. Sort again and hash the integer coordinate array.

This makes the identifier invariant to cube permutation and horizontal
translation without collapsing `pick` into `place`. The manifest stores cube
count, local index, canonical coordinates, goal hash, data version, and the
resulting global identifier.

Padding alone does **not** establish semantic correspondence. The semantic
wrapper pairs fixed-size storage with a validity mask. DCC consumes the slots
using shared per-cube encoders and symmetric pooling, while the actor uses an
equivariant pointer head. Dynamic per-step sorting is deliberately avoided
because slots can swap discontinuously when positions cross.

## Migration checkpoints

1. Track the paper branch as a read-only provenance remote.
2. Preserve StableCRL sampling and regularization in a separate DCC driver.
3. Add a versioned task manifest and validate every sequence before model
   initialization.
4. Use masked set observations and a permutation-equivariant selector action.
5. Run semantic, network-equivariance, single-task, two-task, and resume smoke
   tests before full experiments.
6. Compare CRL, StableCRL, and StableCRL-DCC on short- and long-horizon tasks.

## Provenance

- Original benchmark: `RajGhugare19/builderbench`, `main`.
- Paper-author implementation: `David-Yan1/builderbench`, `david`, `6e8d56d`.
- Prior user DCC/SGCRL work: `Okyumi/builderbench`, audited at `f084a94`.
- This separate StableCRL-DCC repository starts from the paper-author commit.
