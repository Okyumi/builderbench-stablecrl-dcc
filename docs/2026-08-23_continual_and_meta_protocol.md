# Continual and meta-learning protocol design (2026-08-23)

## Why the current sequence is not one clean scientific factor

The original five-phase sequence combines two different shifts:

1. `creative-1-task1 -> creative-1-task2` changes the goal while keeping the
   raw state/action morphology fixed.
2. `creative-1-task2 -> creative-2/3/4-task1` changes cube count, horizon,
   raw input dimension, selector discretization, and target complexity.

A single forgetting curve from this mixture cannot distinguish goal
interference from capacity/morphology transfer. The next protocol therefore
reports these axes separately before using the mixed curriculum.

## Wrapper/architecture diagnostic ladder

The zero-success wrapped vanilla result is not evidence that padding alone is
broken: the old control changed the layout, actor, state-action encoder, goal
encoder, and capacity from the upstream run. The staged controls now isolate
those changes on the paper-style three- and four-stack tasks:

| Global indices | Input | Network | Capacity | Question |
| --- | --- | --- | ---: | --- |
| 0--5 | upstream raw | upstream residual StableCRL | task-sized | Can the reference implementation learn? |
| 36--41 | grouped padding | upstream residual StableCRL | 4 | Does fixed-size padding hurt? |
| 42--47 | semantic slots | upstream residual StableCRL | 4 | Does the semantic rewrite hurt? |
| 48--53 | semantic slots | masked-set actor/critic | 4 | Does the set architecture hurt? |
| 6--11 (existing) | semantic slots | masked-set actor/critic | 8 | Does excess capacity contribute? |

Interpret adjacent cells, not only the endpoints. If 0--5 and 36--41 agree,
padding is viable. A drop at 42--47 points to the semantic selector/feature
rewrite. A drop only at 48--53 points to the masked-set networks, especially
the actor's continuous relaxation of discrete cube selection. A difference
between capacity 4 and 8 indicates an oversized fixed contract or optimization
problem.

The default Torch launcher selects `EXPERIMENT_STAGE=padding_diagnostics`,
which maps array slots 0--17 to global configurations 36--53. This avoids
rerunning completed indices 0--35. `CONFIG_INDEX` still launches one explicit
global cell.

## Continual benchmark tracks

After the diagnostics pass, `EXPERIMENT_STAGE=protocol` selects global
indices 54--65:

- Goal-only track: `creative-1-task1 -> creative-1-task2`, with `M=1`. This
  holds state dimension, action dimension, selector meaning, physics, and
  horizon fixed. It tests goal interference but is only a two-task smoke test.
- Expanding-stack track: `creative-1-task1 -> creative-2-task1 ->
  creative-3-task1 -> creative-4-task1`, with `M=4`. This deliberately tests
  expanding object count and structural complexity without inserting the
  one-cube lift task between stack sizes.
- Both tracks have flat-upstream vanilla CRL and DCC cells with seeds 5, 6,
  and 7. These are protocol controls; they do not replace the original mixed
  sequence results.

At every phase the learner receives a fresh replay buffer and optimizer. Actor
and critic/shared-critic lifecycles remain explicit. A global task ID combines
task-data version, cube count, local task index, and a goal hash canonicalized
against horizontal translation and cube permutation.

## Evaluation matrix and W&B semantics

For phase `i` and task `j`, `A[i,j]` is the success rate of the boundary model
on task `j`. Seen-task cells (`j <= i`) use the appropriate stored DCC task
head. One `next_unseen` cell (`j=i+1`) is also evaluated by default:

- monolithic CRL uses its current critic;
- DCC uses the current phase's task head with the transferred shared groups.

The latter is explicitly a zero-shot shared-plus-current-head control, not a
claim that DCC has inferred a new task head. It is recorded with
`critic_head_task_index` so it cannot be confused with seen-task evaluation.

The local `continual_eval.jsonl` is replaced atomically per phase, making
resume idempotent even if a job dies between evaluation and boundary
checkpoint creation. At each boundary a small, resumable W&B evaluation run
logs:

- the long-form evaluation rows;
- success and easy-success matrix tables;
- mean/min seen success, mean seen easy success, average forgetting, backward
  transfer, current per-task metrics, and next-task zero-shot success.

This is outside gradient updates. For the five-task sequence it uploads at
most a few dozen table cells per boundary, so its overhead is negligible next
to a 200M-step phase. Local JSONL is authoritative and W&B exceptions are
warnings rather than training failures.

## What would constitute a real meta-learning benchmark

The checked-in tasks are not numerous enough for a defensible meta-learning
split: one cube has two goals and every larger cube count has only one. Reward
variants reuse the same goals. Calling the two-task goal-only track “meta-RL”
would therefore overstate the evidence.

A real follow-up should generate and version at least dozens of geometries per
cube count, then split by canonical goal hash—not by local task index—into
meta-train, meta-validation, and meta-test. Geometry generators should cover
stacks, bridges, rows, corners, and disconnected structures while checking
reachability, collision stability, and duplicate hashes. Recommended reports
are:

1. zero-shot query performance on held-out goal hashes;
2. adaptation curves after fixed environment-step budgets;
3. shared-frozen/head-only versus full-model adaptation;
4. performance on both adapted task and earlier tasks after adaptation;
5. separate within-morphology and cross-morphology splits.

DCC already has a natural adaptation unit (`phi_task`), but the actor is still
shared. A sound meta-DCC experiment should compare adapting only `phi_task`,
adapting `phi_task` plus a small actor adapter, and full fine-tuning. The
current code implements the continual protocol, diagnostic controls, and
zero-shot next-task evaluation; it does not claim to implement a MAML-style
outer loop.

## Capacity policy when padding is insufficient

Never resize a live JAX parameter tree when a task exceeds `M`: that changes
input layers, optimizer state, checkpoint semantics, and compilation shape.
For a known curriculum, set `M` exactly to the maximum cube count (4 for the
current stack curriculum). The wrappers fail fast when `N > M`.

For an open-ended benchmark, use versioned capacity buckets such as
`M in {4,8,12}`. Compile one model per bucket, include overlap/calibration
tasks in adjacent buckets, and transfer through the shared slot encoder or a
fixed-size latent interface rather than copying shape-dependent input
weights. Report bucket transitions as morphology boundaries. This keeps slot
meaning, mask meaning, checkpoints, and W&B groups explicit while avoiding a
large mostly-empty `M` that weakens optimization and wastes memory.
