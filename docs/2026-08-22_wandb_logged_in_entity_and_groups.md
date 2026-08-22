# W&B logged-in entity and per-config groups

Date: 2026-08-22

## Purpose

Stop sending training runs to hardcoded third-party W&B entities and keep
Torch jobs grouped by experiment configuration.

## Changes

Every training script that previously set `wandb_entity` to `raj19` or
`david-yan` now defaults to `None`. `stable_crl.py` and `stable_crl_dcc.py`
also default `wandb_project_name` to `builderbench-stablecrl-dcc` and pass
`entity=args.wandb_entity or None` into `wandb.init`.

`DRAFT.sh` still omits `--wandb-entity` unless `WANDB_ENTITY` is set. With a
login on the cluster (`WANDB_API_KEY` or `~/.netrc`), W&B uses that account's
default entity.

## Groups

`experiment_configs.py` sets `wandb_group` to the configuration name. The
launcher prefixes it as `torch_dcc__<name>`. The 36-run batch therefore has
12 W&B groups (one per algorithm cell). Seeds 5, 6, and 7 share a group and
are distinguished by run name (`<name>_seed<seed>`).

## Validation

- `tests/test_experiment_configs.py` checks unique config names, matching
  `wandb_group` values, and that a dry-run command includes the project and
  group flags without `--wandb-entity`.
- Remaining `raj19` / `david-yan` entity strings were searched out of the
  repository.

## Limitations

Compute nodes still need the login credentials available (home `~/.netrc`
and/or an exported `WANDB_API_KEY`). An empty entity does not create a
project; W&B creates `builderbench-stablecrl-dcc` under the logged-in user
on first upload.
