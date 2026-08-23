# Torch smoke preflight fixes

Date: 2026-08-23

## Scope

The first submission of `DRAFT_DCC_SMOKE.sh` stopped during its repository
test preflight. No training command or W&B run was started. This change fixes
all five reported preflight failures without changing the residual DCC model,
training recipe, smoke indices, or production indices.

## Failure analysis

Three test modules could not import:

- `test_continual_checkpoint`;
- `test_continual_crl`;
- `test_vanilla_networks`.

All three reached `continual/vanilla_networks.py`, which still imported
`SetActor`, `SetGoalEncoder`, and `TaskStateActionEncoder` from
`continual/dcc_networks.py`. Those masked-set classes belonged to the older
set-based DCC implementation and were intentionally removed when DCC moved to
the proven upstream flat residual architecture. The vanilla diagnostic
control was therefore left with a stale import dependency.

Two production-launcher tests also failed. `DRAFT_DCC_SMOKE.sh` correctly
exports `CONFIG_REGISTRY=smoke_experiment_configs.py`,
`EXPERIMENT_STAGE=smoke`, and the smoke W&B group prefix before asking the
shared launcher to run its preflight. The test subprocesses copied that parent
environment, so tests intended to exercise production `DRAFT.sh` accidentally
generated smoke commands. The launcher itself selected the requested smoke
registry correctly; the test isolation was incomplete.

## Implementation

### Independent vanilla set-network module

`continual/set_networks.py` now owns the legacy diagnostic control's:

- masked set pooling;
- continuous-selector slot weighting;
- permutation-invariant goal encoder;
- permutation-safe state-action encoder;
- selector-equivariant actor.

`continual/vanilla_networks.py` imports those modules directly. It no longer
depends on `continual/dcc_networks.py`.

This does not restore the removed DCC set encoder. Residual DCC still uses
only `FlatActor`, `FlatStateActionEncoder`, and `FlatGoalEncoder`, plus its
zero-output task residual adapter. No dynamics head, target, loss, parameter,
metric, or launcher flag was reintroduced.

The set modules remain available solely because historical padding diagnostic
cells and vanilla lifecycle controls explicitly select
`vanilla_network_type=set`.

### Hermetic launcher tests

Production launcher tests now explicitly select:

- `CONFIG_REGISTRY=experiment_configs.py`;
- `EXPERIMENT_STAGE=dcc_residual_gate`;
- `WANDB_GROUP_PREFIX=torch_dcc`.

Smoke launcher tests explicitly select their smoke registry, stage, budgets,
environment counts, replay sizes, evaluation cadence, and W&B prefix. Both
test families therefore verify their own contract even when invoked from the
other launcher's preflight or from an existing Slurm array environment.

## Experiment and configuration impact

There are no changes to:

- the three smoke configurations;
- their seed, task order, CRTR factor, or architecture;
- the 2,097,152-step smoke budgets;
- the 72 production configurations;
- checkpoint recipes or W&B identities.

The failed jobs stopped before per-run checkpoint directories and W&B training
runs were created, so the complete smoke array can be resubmitted directly.
No checkpoint cleanup or new checkpoint root is required.

## Validation

The exact cross-registry contamination was reproduced locally by running the
production launcher tests under exported smoke variables. All four production
registry/launcher tests then passed with the fix.

The inverse check was also run: all four smoke registry/launcher tests passed
under exported production variables. This covers direct execution and the
Slurm copied-script resolver.

The new set-network module was syntax checked and retains the existing
permutation-invariance and selector-equivariance tests. The full JAX/Flax test
must be rerun in the Torch virtual environment because the local environment
does not contain those packages.

## Torch retry

Pull the fix and resubmit the full smoke gate:

```bash
cd /scratch/yd2247/builderbench-stablecrl-dcc
git pull
sbatch --account=torch_pr_301_tandon_advanced DRAFT_DCC_SMOKE.sh
```

The expected sequence is:

1. JAX reports a GPU and MuJoCo-Warp imports;
2. the complete repository preflight passes;
3. each array task prints and launches its assigned DCC command;
4. indices 0 and 1 train one CRTR task each, while index 2 trains two
   continual tasks.

## Known limitations

- This local validation cannot replace the Torch JAX/Flax execution because
  the import failure occurred only once the full HPC dependencies were
  available.
- Preflight still runs once in each array task. This adds a small amount of
  startup work but prevents any smoke training from beginning with a broken
  repository test suite.
