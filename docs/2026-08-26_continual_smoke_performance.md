# Continual smoke performance (2026-08-26)

## Scope

This report summarizes the two matched continual-RL smoke batches in the
`builderbench-stablecrl-dcc` W&B project:

- goal-only: `creative-1-task1 -> creative-1-task2`, `max_cubes=1`;
- expanding four-stack: `creative-1-task1 -> creative-2-task1 ->
  creative-3-task1 -> creative-4-task1`, `max_cubes=4`.

Every cell uses seed 5, CRTR-12, PD-5, the flat residual 8 x 1024 encoder,
2,097,152 requested environment steps per task, 256 training environments,
32 training-evaluation environments, and four evaluations per task. The three
methods are:

- **StableCRL R/R:** reset actor and monolithic critic at each task;
- **StableCRL P/P:** persist actor and monolithic critic across tasks;
- **Residual DCC (ours):** persist the actor and shared encoders, initialize a
  fresh residual task adapter, and retain completed adapters. There is no
  dynamics head.

W&B contains all 24 expected records: 18 task-training runs and six continual
evaluation runs. All are finished, have finite summary metrics, reached the
smoke budget, and contain strict/easy AUC.

Data source: [W&B BuilderBench StableCRL/DCC project](https://wandb.ai/nyuad_mmvc/builderbench-stablecrl-dcc).

## Metric definitions

- **Best** is the maximum success rate among the four evaluations within that
  task's training phase.
- **Final** is the last success rate in that task's training phase. It is not
  a retention measurement after later tasks.
- **AUC** is the cumulative trapezoidal success-rate area divided by elapsed
  environment steps. Its implicit first point is `(0 steps, 0 success)`, so
  it is normalized to `[0, 100]` here.
- **Strict** uses BuilderBench's exact success criterion. **Easy** uses its
  relaxed success criterion.
- Terminal retention is evaluated after the final task with five repeated
  batches of 32 environments. Values shown as `mean +/- std` use the
  population standard deviation across those five batch means, not across
  seeds.

The smoke batch contains only seed 5. Therefore, none of the tables below
should be interpreted as an across-seed estimate. The production batch is
needed for seed mean and standard deviation.

## Within-task learning: goal-only track

All entries are percentages. The capacity-one track changes only the goal;
observation and action semantics remain fixed.

| Task | Method | Best strict | Final strict | Strict AUC | Best easy | Final easy | Easy AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| `creative-1-task1` | StableCRL R/R | 100.00 | 71.88 | **75.77** | 100.00 | 81.25 | **82.31** |
| `creative-1-task1` | StableCRL P/P | 100.00 | 81.25 | 64.81 | 100.00 | 87.50 | 67.93 |
| `creative-1-task1` | Residual DCC (ours) | 100.00 | **100.00** | 51.42 | 100.00 | **100.00** | 58.41 |
| `creative-1-task2` | StableCRL R/R | 100.00 | 100.00 | 67.09 | 100.00 | 100.00 | 86.92 |
| `creative-1-task2` | StableCRL P/P | 100.00 | 100.00 | **86.92** | 100.00 | 100.00 | **86.92** |
| `creative-1-task2` | Residual DCC (ours) | 100.00 | 100.00 | **86.92** | 100.00 | 100.00 | **86.92** |

For task 1, reset/reset has the largest strict AUC even though DCC has the
largest final value. For task 2, both persistent methods begin with useful
transferred behavior and attain 86.92 strict AUC, compared with 67.09 for the
reset baseline.

## Within-task learning: expanding four-stack track

All entries are percentages. Capacity is fixed at four cubes before task 0,
so the one-, two-, and three-cube observations are semantically padded to the
same compiled shape as the four-cube task.

| Task | Method | Best strict | Final strict | Strict AUC | Best easy | Final easy | Easy AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| `creative-1-task1` | StableCRL R/R | 100.00 | 71.88 | 56.54 | 100.00 | 84.38 | 59.62 |
| `creative-1-task1` | StableCRL P/P | 100.00 | **100.00** | **63.15** | 100.00 | **100.00** | 65.50 |
| `creative-1-task1` | Residual DCC (ours) | 100.00 | **100.00** | 52.38 | 100.00 | **100.00** | **65.67** |
| `creative-2-task1` | StableCRL R/R | 40.62 | 40.62 | 18.08 | 100.00 | 90.62 | 82.69 |
| `creative-2-task1` | StableCRL P/P | 46.88 | 46.88 | 10.38 | 100.00 | 100.00 | 72.14 |
| `creative-2-task1` | Residual DCC (ours) | **81.25** | **81.25** | **23.08** | 100.00 | 100.00 | **86.92** |
| `creative-3-task1` | StableCRL R/R | 0.00 | 0.00 | 0.00 | 21.88 | 21.88 | 7.31 |
| `creative-3-task1` | StableCRL P/P | **28.12** | **21.88** | **15.77** | **96.88** | **96.88** | **72.40** |
| `creative-3-task1` | Residual DCC (ours) | 18.75 | 15.62 | 7.33 | 93.75 | 75.00 | 51.97 |
| `creative-4-task1` | StableCRL R/R | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `creative-4-task1` | StableCRL P/P | 0.00 | 0.00 | 0.00 | **34.38** | **34.38** | **18.22** |
| `creative-4-task1` | Residual DCC (ours) | **3.12** | 0.00 | **0.77** | 25.00 | 12.50 | 10.02 |

DCC is strongest on the two-cube task: its strict best/final success is 81.25,
versus 46.88 for persistent StableCRL and 40.62 for reset StableCRL. Persistent
StableCRL is strongest on the three-cube task. At the smoke budget, none of the
methods learns reliable strict four-cube success; DCC briefly reaches 3.12,
while persistent StableCRL has the strongest easy-success curve.

## Terminal continual retention: goal-only track

These values are measured after training `creative-1-task2`. Strict task
columns are boundary-evaluation `mean +/- std`, in percent.

| Method | Task 0 strict | Task 1 strict | Mean seen strict | Min seen strict | Mean seen easy | Forgetting | BWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| StableCRL R/R | 0.00 +/- 0.00 | 100.00 +/- 0.00 | 50.00 | 0.00 | 50.00 | 73.13 | -73.13 |
| StableCRL P/P | 0.00 +/- 0.00 | 100.00 +/- 0.00 | 50.00 | 0.00 | **56.88** | 81.88 | -81.88 |
| Residual DCC (ours) | 0.00 +/- 0.00 | 100.00 +/- 0.00 | 50.00 | 0.00 | 50.31 | 100.00 | -100.00 |

Every method learns the current task but loses all strict success on task 0.
DCC's task-0 adapter is retained, but the actor and shared encoders continue
to change; retaining the adapter alone does not guarantee retention.

## Terminal continual retention: expanding four-stack track

These values are measured after training `creative-4-task1`. Strict task
columns are boundary-evaluation `mean +/- std`, in percent.

| Method | Task 0 strict | Task 1 strict | Task 2 strict | Task 3 strict | Mean seen strict | Mean seen easy | Forgetting | BWT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StableCRL R/R | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 | 0.00 | 38.33 | -38.33 |
| StableCRL P/P | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 | **10.62** | 56.46 | -56.46 |
| Residual DCC (ours) | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 | 2.19 | 65.21 | -65.21 |

The final strict matrix is zero for all methods. The different forgetting
values should not be read as reset/reset retaining more knowledge: forgetting
is measured relative to each method's earlier diagonal score, and reset/reset
learned several tasks less well in the first place.

## Conclusions

1. **The continual machinery passes.** Every expected run finished, all task
   boundaries were reached, goal-only evaluation produced four rows, expanding
   evaluation produced thirteen rows, and AUC is present and finite.
2. **DCC shows positive forward learning on task 2.** On the expanding track it
   substantially improves two-cube best success, final success, and AUC.
3. **DCC does not yet show retention at the smoke budget.** Its stored task
   adapters do not prevent drift in the persistent actor and shared encoders.
4. **The four-cube smoke is an execution test, not a performance estimate.**
   Two million steps are insufficient for reliable strict success, so the
   200M-step production cells are required before comparing algorithms.
5. **Production should report seed statistics.** The 18-cell core uses seeds
   5, 6, and 7; its final tables should report mean +/- standard deviation
   across seeds in addition to the within-seed repeated-evaluation deviation.

The smoke results support proceeding to the matched production experiment,
but they do not support a claim that DCC already outperforms persistent
StableCRL as a continual learner.
