# Matched continual algorithm benchmark (2026-08-25)

## Purpose

The residual-DCC parity runs were intentionally single-task experiments. They
cannot measure forward transfer, backward transfer, or forgetting. The earlier
planned production protocol also used repetition factor 1 even though the
successful individual-task StableCRL and residual-DCC gate used CRTR-12. That
would confound the continual decomposition with a different contrastive
training objective.

This change adds a dedicated matched registry,
`continual_experiment_configs.py`. Every compared method now uses:

- the semantic fixed-capacity wrapper;
- the upstream flat residual actor and critic family;
- eight residual blocks of width 1024;
- PD-5 controls;
- CRTR repetition factor 12;
- entropy cost 0.01 and log-sum-exp cost 0.1 from `DRAFT.sh`;
- 200M environment steps per task in production;
- seeds 5, 6, and 7;
- five repeated boundary-evaluation batches.

Only the parameter lifecycle or DCC decomposition changes.

## Compared algorithms

Each track contains three matched methods:

1. **StableCRL reset/reset.** The actor and monolithic critic are reinitialized
   at every task. Replay and optimizer state are also fresh. This is the
   independent-task lower-triangular reference and should show no retention by
   construction.
2. **StableCRL persistent/persistent.** The monolithic actor and critic persist
   across tasks while replay and optimizer state reset. This measures ordinary
   sequential fine-tuning and catastrophic interference.
3. **Residual DCC.** The actor, shared state-action encoder, and shared goal
   encoder persist. A fresh zero-output task adapter is initialized per task,
   and completed adapters are retained in the task bank for seen-task
   evaluation.

The reset/reset and persistent/persistent baselines are called StableCRL
because they use the paper's CRTR-12 regularization recipe. They are still
vanilla or monolithic contrastive RL with respect to the DCC comparison: they
have no task adapter or decomposition.

## Continual tracks

### Goal-only

`creative-1-task1 -> creative-1-task2`, with `max_cubes=1`.

State width, action width, selector semantics, physics, and horizon stay fixed.
Only the target geometry changes. This is the cleanest test of interference
and transfer, but it has only two tasks.

### Expanding four-stack

`creative-1-task1 -> creative-2-task1 -> creative-3-task1 ->
creative-4-task1`, with `max_cubes=4` fixed before task 0.

This tests morphology, horizon, and object-count transfer while preserving a
single compiled observation and goal contract. The wrapper fails before
training if a task exceeds the fixed capacity.

### Optional expanding five-stack

`creative-1-task1 -> creative-2-task1 -> creative-3-task1 ->
creative-4-task1 -> creative-5-task1`, with `max_cubes=5`.

This is a fresh checkpoint family. A five-stack task cannot be appended to a
`max_cubes=4` run because the flat input layers, replay entries, optimizer
state, and compiled shapes differ. The five-stack stage is kept separate from
the default core launch because it adds substantial compute and should follow
the four-stack diagnostic.

## Evaluation and W&B metrics

At phase `i`, each driver evaluates all seen tasks `j <= i` and one next
unseen task when available. Boundary evaluation uses five deterministic
batches and logs mean and population standard deviation. W&B receives:

- strict- and easy-success mean matrices;
- matching standard-deviation matrices;
- long-form rows including evaluation counts and DCC head identity;
- mean/minimum seen success;
- average forgetting;
- backward transfer and forward-transfer controls;
- per-task boundary reward and object-goal distance.

Training runs now also log cumulative normalized trapezoidal AUC:

- `eval/episode_success_rate_auc`;
- `eval/episode_easy_success_rate_auc`.

The implicit first point is zero success at zero environment steps. AUC is
divided by the current environment-step extent, so it remains in [0, 1] and
its final value is directly comparable across matched task budgets.

## Registry layout

- 0--8: goal-only production cells;
- 9--17: expanding four-stack production cells;
- 18--26: optional expanding five-stack production cells.

Within each nine-cell track, the first three cells are seed-5 reset/reset,
persistent/persistent, and DCC. These form the smoke stage. Seeds 6 and 7
follow in the same method order.

## Torch smoke launch

Run the goal-only lifecycle smoke test first:

```bash
sbatch --account=torch_pr_301_tandon_advanced \
  DRAFT_CONTINUAL_SMOKE.sh
```

Then exercise the same three methods across a morphology boundary:

```bash
EXPERIMENT_STAGE=smoke_expanding_4stack \
  sbatch --account=torch_pr_301_tandon_advanced --array=0-2 \
  DRAFT_CONTINUAL_SMOKE.sh
```

The wrapper supplies 2,097,152 steps per task, 256 training environments, 32
evaluation environments, four training evaluations, and five boundary repeats.
It uses separate smoke checkpoints and W&B groups.

## Production launch

The default production wrapper launches the 18-cell goal-only and expanding
four-stack core:

```bash
sbatch --account=torch_pr_301_tandon_advanced DRAFT_CONTINUAL.sh
```

Launch the five-stack extension separately:

```bash
EXPERIMENT_STAGE=expanding_5stack \
  sbatch --account=torch_pr_301_tandon_advanced --array=0-8 \
  DRAFT_CONTINUAL.sh
```

For a single dry-run command:

```bash
DRY_RUN=true CONFIG_INDEX=2 bash DRAFT_CONTINUAL_SMOKE.sh
```

Configuration 2 is the seed-5 goal-only DCC cell. Configuration 0 is the
matched reset/reset baseline and configuration 1 is persistent/persistent.

## Interpretation order

1. Confirm all three smoke cells compile, train with finite losses, create two
   boundary checkpoints, and upload a 2 x 2 seen-task matrix.
2. Inspect goal-only retention before interpreting morphology transfer.
3. Compare task-0 learning AUC to ensure DCC did not begin from a weaker
   optimization regime.
4. Compare final mean-seen success, minimum-seen success, forgetting, and
   backward transfer across methods.
5. Treat next-task DCC evaluation as shared-plus-current-head zero-shot
   performance, not as inferred selection of an unseen task adapter.
6. Run the five-stack extension only after the four-stack DCC instability is
   understood or explicitly accepted as part of the reported variance.
