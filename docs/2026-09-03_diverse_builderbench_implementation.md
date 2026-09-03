# Diverse direct-cube BuilderBench implementation

## Scope

This change adds the requested diverse continual benchmark while preserving
all existing `creative-*` stacking tasks and experiment indices.

Sequence A is the runnable main track:

```text
place cube
→ lift cube
→ two-cube stack
→ permute
→ T-stack
→ triangular packing
→ five-cube archway
→ seven-cube Jenga tower
→ eight-cube vertical portal
```

Sequence B is registered in `builderbench/task_catalog.py` for later use:

```text
two-cube stack
→ three-cube stack
→ T-stack
→ five-cube pyramid
→ five-cube archway
→ seven-cube Tokyo tower
→ seven-cube maximum overhang
→ eight-cube vertical portal
```

The union contains 13 adapted tasks. They use IDs of the form
`builderbench-direct-N-taskM`, where `N` and `M` retain the original
BuilderBench cube count and task number.

## Geometry provenance

Starts, target coordinates, names, and masks were copied from
`RajGhugare19/builderbench` at revision `de9130b98323`, file
`builderbench/create_task_data.py`.

The direct simulator has a smaller workspace than the robot benchmark. Every
source start and goal receives the same translation:

```text
[-0.20, 0.00, 0.00]
```

This changes only the absolute X location. All distances, heights, offsets,
and target geometry remain identical. A validation script compared all 13
source arrays and masks directly with `upstream/main`.

## Environment behavior

- Physics and control remain the fast StableCRL MJX direct-cube environment.
- Every adapted task uses the unchanged 5-value action and PD duration 5.
- Sequence A fixes `max_cubes=8` before task 1, giving state 120, goal 32,
  and action 5 for every phase.
- The state validity mask marks cubes that exist.
- The goal mask marks cube targets that are required. Masked helper cubes
  still exist and can be controlled, but do not affect reward or success.
  Their coordinates are zeroed in encoder inputs as well as marked with zero
  in the mask, so flat encoders cannot learn from arbitrary helper targets.
- Relabelled achieved goals carry the same goal mask, so contrastive training
  does not learn arbitrary helper-cube destinations.
- `permute` uses ordered matching because permutation-invariant matching would
  erase the meaning of swapping two cubes. Other direct tasks retain
  StableCRL's permutation-invariant matching.
- Masked target markers are hidden in rendering.

Existing `creative-*` task files, targets, IDs, and default stacking configs
were not replaced.

## Experiment configuration

`diverse_continual_experiment_configs.py` provides nine matched cells:

- reset actor + reset critic StableCRL;
- persistent actor + persistent critic StableCRL;
- residual DCC with its actor and shared encoders carried forward;
- seeds 5, 6, and 7 for each method;
- CRTR repetition 12, semantic padding, `max_cubes=8`, and PD-5 throughout.

List or inspect the cells with:

```bash
python diverse_continual_experiment_configs.py --list
python diverse_continual_experiment_configs.py --setting 2
```

Check the generated production command without launching anything:

```bash
DRY_RUN=true CONFIG_INDEX=2 RUN_TEST_PREFLIGHT=false \
  bash DRAFT_DIVERSE_CONTINUAL.sh
```

Launch the seed-5 three-method smoke stage with short budgets first:

```bash
EXPERIMENT_STAGE=smoke \
BASE_STEPS=2097152 STEPS_PER_TASK=2097152 \
MAX_REPLAY_SIZE=512 MIN_REPLAY_SIZE=128 \
NUM_EVAL_STEPS=4 NUM_RESET_STEPS=4 \
sbatch --account=torch_pr_XXX_XXXXX --array=0-2 \
  DRAFT_DIVERSE_CONTINUAL.sh
```

After that passes, launch all nine full cells:

```bash
sbatch --account=torch_pr_XXX_XXXXX DRAFT_DIVERSE_CONTINUAL.sh
```

The launcher uses task-data version `builderbench-de9130-direct-v1`, so its
checkpoints cannot be confused with the earlier simplified task data.

## Validation

- Python syntax compilation passed for every changed Python file.
- The broader dependency-light regression suite passed: 37 tests passed and
  one simulator-only check was skipped.
- All 13 starts, goals, and masks matched the upstream BuilderBench arrays.
- The Sequence A launcher generated the expected nine-task PD-5 command.
- `git diff --check` passed.

The local lightweight Python environment does not contain MuJoCo, Optax, or
the pinned GPU dependencies. A compiled simulator reset/step test therefore
remains for the Torch smoke stage. That stage should be treated as required
before a paper-scale launch.
