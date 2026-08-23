# Continual evaluation and padding controls implementation (2026-08-23)

## Scope

This change prepares the next BuilderBench StableCRL/DCC stage after global
experiment indices 0--35. It adds crash-safe full-matrix tracking, next-task
zero-shot evaluation, a padding-only observation control, an
upstream-equivalent residual network adapter, capacity diagnostics, staged
Torch launch ranges, and non-fatal optional plotting.

## Design and implementation

### Evaluation persistence and W&B

`continual/eval_logging.py` owns the boundary evaluation record. Each phase
replaces rows with the same `phase_index` and atomically renames a temporary
JSONL file. This removes duplicate rows when resuming after a failure between
evaluation and checkpoint creation.

When tracking is enabled, a deterministic W&B evaluation run ID is derived
from the complete training recipe and W&B namespace. The run is resumed after
each task and receives long-form rows, success/easy-success matrix tables, CL
summary scalars, and per-task metrics. Upload happens after local persistence;
all W&B initialization, logging, table, and finish failures are caught and do
not invalidate an expensive completed phase. `--no-log-continual-eval` turns
off the evaluation run, and `--no-wandb-eval-tables` keeps only scalars.

Both continual drivers now evaluate the next task by default in addition to
all seen tasks. DCC uses the current task head for that future task and records
the head index. `--no-eval-next-task` restores lower-triangular-only
evaluation. Re-launching an already-complete checkpoint directory performs no
training but backfills its existing local JSONL into the W&B evaluation run.

### Padding-only and flat-upstream controls

`GroupedPadLayout` and `GroupedPadWrapper` pad the four upstream raw feature
groups independently, retain the continuous previous selector, append an
explicit validity mask, and invert the layout before environment steps. This
isolates fixed input shape without introducing the semantic cube-record
rewrite.

`flat_upstream_networks.py` reproduces StableCRL's residual actor,
state-action encoder, goal encoder, LayerNorm/Swish blocks, residual-depth
scaling, representation size, and tanh-Gaussian parameterization. It accepts
either fixed layout and receives the same top-level actor/state-action/goal RNG
streams as `stable_crl.py`. The learner selects the controls with:

```text
--observation-layout grouped|semantic
--vanilla-network-type flat_upstream|set
```

DCC and set networks reject the grouped layout. The original semantic/set
defaults and their checkpoint recipe remain byte-for-byte compatible so an
existing partial run can still resume.

### Experiment and Torch configuration

Global indices 0--35 are unchanged. New global cells are:

- 36--41: grouped padding, flat upstream residual networks, capacity 4;
- 42--47: semantic padding, flat upstream residual networks, capacity 4;
- 48--53: semantic padding, set networks, capacity 4;
- 54--59: flat CRL goal-only and expanding-stack protocols;
- 60--65: DCC goal-only and expanding-stack protocols.

`experiment_configs.py` exposes named stage bounds. `DRAFT.sh` retains the
Slurm spool-directory resolver and defaults to
`EXPERIMENT_STAGE=padding_diagnostics` and maps Slurm slots 0--17 to global
indices 36--53. This prevents an unqualified next-stage submission from
rerunning the completed legacy batch. Set `EXPERIMENT_STAGE=protocol` only
after the diagnostic gate passes.

The launcher forwards the layout/network choice and all evaluation-logging
booleans. A direct `CONFIG_INDEX` remains a global-index override.

### Optional plotting failure

The final checkpoint metric PNG in both StableCRL runners is optional. Missing
`matplotlib` now prints a warning while preserving the successful process exit,
W&B finish, checkpoint, and `eval_log.jsonl`. This addresses the completed
training jobs previously marked failed only during final plotting.

## Validation

- Python byte-compilation passed for all changed runners, continual modules,
  experiment configuration, and tests.
- Shell syntax validation passed for `DRAFT.sh`.
- Seventeen dependency-light unit tests passed locally, covering stable legacy
  indices, new stage ranges, launcher commands, grouped-layout round trips,
  task identities, atomic/idempotent evaluation writes, CL metrics, and stable
  evaluation run IDs.
- The new JAX/Flax network module was checked statically in this workspace; a
  full GPU smoke run is still required on Torch because the local environment
  does not contain the pinned JAX/Flax/MuJoCo stack.

## Experiment gate

Run the 18 `padding_diagnostics` cells first. Do not spend the full protocol
budget until the grouped-padding/upstream-residual cell is close to the raw
upstream reproduction. If it is not, test a short single-seed, single-task run
with `max_cubes` equal to the task cube count before changing DCC.

## Known limitations

- The padding-only flat network is intentionally order sensitive and is not a
  final continual representation.
- Next-task DCC evaluation reuses the current head; no unseen task head can be
  inferred without adaptation.
- Current BuilderBench goal files are too small for a credible meta-train /
  validation / test split.
- W&B matrix logging is not retroactively present in already-finished remote
  jobs unless their checkpoint directory and JSONL are re-launched with this
  code; the backfill cannot reconstruct files that were deleted from scratch.
- Capacity buckets beyond 8 and a versioned multi-geometry task generator are
  design recommendations, not implemented in this change.
