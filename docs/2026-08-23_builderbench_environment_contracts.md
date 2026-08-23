# BuilderBench environment contracts (2026-08-23)

## Scope and terminology

This is the source-level contract for every environment family routed by
`builderbench/env_utils.py` in this fork. “State space” below means the vector
visible to the policy. MuJoCo position, velocity, contact, actuator, mocap,
metric, and RNG fields remain in the internal `State` object but are not part
of the agent observation unless listed explicitly.

All continuous policy outputs are expected in `[-1, 1]`. The default physics
step is 0.005 seconds and the raw control step is 0.02 seconds (four MuJoCo
substeps). By default the reward is permutation invariant: a Hungarian
assignment matches physical cubes to target locations before distances,
reward, and success are computed.

## Creative 3-D manipulation family

### Registered task data

The dense IDs are `creative-N-taskM`; the other reward variants add
`sparse-` or `truncated-reward-`. The checked-in task files contain:

| Cubes `N` | Valid goal tasks | Relative target geometry |
| ---: | --- | --- |
| 1 | task1, task2 | place at `z=0`; lift to `z=0.08 m` |
| 2 | task1 | vertical stack at `z=0, 0.04 m` |
| 3 | task1 | vertical stack through `z=0.08 m` |
| 4 | task1 | vertical stack through `z=0.12 m` |
| 5 | task1 | vertical stack through `z=0.16 m` |
| 6 | task1 | vertical stack through `z=0.20 m` |
| 7 | task1 | vertical stack through `z=0.24 m` |
| 8 | task1 | vertical stack through `z=0.28 m` |

At reset a shared horizontal offset is sampled with `x` in
`[0.22, 0.32] m`, `y` in `[-0.10, 0.10] m`, and base `z=0.02 m`; the relative
geometry above is added to that offset. Start boxes are sampled from the task
file and randomly assigned to cubes.

`creative-9.npz` exists, but no creative-9 timing/contact entry exists in
`builderbench/constants.py`, so creative-9 is not currently a runnable ID.
The constants also list `truncated-reward-creative-2-task2`, but
`creative-2.npz` contains only one task; that ID is invalid and must not be
used.

### Raw policy observation

For `N` cubes and `delta_control=False`, the vector is grouped by physical
quantity, not by cube:

```text
[positions_xyz (3N), quaternions_wxyz (4N),
 linear_velocities_xyz (3N), angular_velocities_xyz (3N),
 previous_selector (1)]
```

The raw observation dimension is therefore `13N + 1`; achieved and target
goals contain cube positions only and have dimension `3N`.

| N | Raw observation | Goal | Raw horizon | PD decision horizon |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14 | 3 | 150 | 30 |
| 2 | 27 | 6 | 200 | 40 |
| 3 | 40 | 9 | 250 | 50 |
| 4 | 53 | 12 | 300 | 60 |
| 5 | 66 | 15 | 350 | 70 |
| 6 | 79 | 18 | 400 | 80 |
| 7 | 92 | 21 | 450 | 90 |
| 8 | 105 | 24 | 500 | 100 |

The raw horizon is `100 + 50N`. The PD horizon shown uses the experiment
default `pd_duration=5`; each policy decision executes five raw control steps
and sums their rewards and metrics.

### Raw and PD action spaces

The creative action dimension is always five.

Raw control interprets `[a_x, a_y, a_z, a_yaw, a_select]`. The first four
components are normalized actuator commands for one cube. The cube acted on
at the current raw step is chosen from the selector stored in the previous
state; the current `a_select` becomes the selector for the next raw step.
Thus raw selection has a one-step delay.

The experiments normally place `PDWrapper` outside the fixed-layout wrapper.
Its five components mean `[waypoint_x, waypoint_y, waypoint_z, waypoint_yaw,
cube_selector]`. XYZ is mapped to the workspace, yaw to `[-pi/2, pi/2]`, and
the selector chooses the cube for the current PD decision. A PD controller
converts that waypoint into five raw actions.

### Reward and success

Let `d_i` be the Euclidean distance from matched cube `i` to its assigned
target. Dense creative reward per raw step is

```text
sum_i (1 - tanh(5 d_i)).
```

The truncated-reward variant replaces `d_i` by `max(d_i, 0.02)`. The sparse
variant uses `count(d_i < 0.02) - N`, with range `[-N, 0]`; it still logs the
dense reward separately.

Strict success is true only when every matched cube has `d_i < 0.02 m`.
Easy success uses `d_i < 0.05 m`. `obj_goal_dist` is `sum_i d_i`. The standard
evaluator accumulates the per-step success metric and reports
`eval/episode_success_rate` as the fraction of episodes with success at least
once, not terminal-state success. The same convention holds for easy success.

The raw workspace bounds are `[-0.05,-0.35,0]` to `[0.45,0.35,0.5]` metres.
Out-of-bounds or NaN state is a terminal condition only when
`env_early_termination=True`; StableCRL experiments set it to false and rely
on the horizon.

## Fixed-capacity continual layouts

The final semantic layout converts the raw grouped vector into shared cube
slots:

```text
observation = [cube_0(14), ..., cube_(M-1)(14), valid_mask(M)]
goal        = [positions(3M), valid_mask(M)]
```

Each cube slot contains position 3, quaternion 4, linear velocity 3, angular
velocity 3, and a one-hot previous-selection flag 1. Dimensions are `15M` for
observation and `4M` for goal: `(60,16)` for `M=4` and `(120,32)` for `M=8`.
The wrapper does not change MuJoCo state or dynamics. It changes the policy
input by quantizing the continuous previous selector to a one-hot slot.

The new padding-only diagnostic layout retains upstream feature grouping and
the continuous selector. It pads each group to `M` and appends a mask, giving
observation dimension `14M+1` and goal dimension `4M`: `(57,16)` at `M=4`.
This layout is a control, not the recommended final continual representation,
because a flat network remains slot-order sensitive.

Both wrappers reject `delta_control=True` and fail immediately when
`N > M`. They currently support the creative observation contract only.

## Planar families

The constants provide half-grid size 4 and cube counts 1 through 5. Targets
and achieved goals are XY coordinates; success thresholds and Hungarian
matching are identical to the creative family.

| Environment pattern | Observation | Action | Goal | Horizon |
| --- | --- | --- | --- | --- |
| `planar-4-cube-N` | XY positions `2N`, XY velocities `2N`, previous selector `1` = `4N+1` | continuous `[x,y,selector]`, dim 3 | `2N` | `100N` |
| `planar-position-4-cube-N` | XY positions `2N`, velocities `2N`, controls `2N`, selector `1` = `6N+1` | continuous delta `[dx,dy,selector]`, dim 3 | `2N` | `60(N+1)` |
| `discrete-planar-position-4-cube-N` | XY positions `2N`, velocities `2N`, controls `2N`, active-cube one-hot `N` = `7N` | discrete scalar with `N+4` choices | `2N` | `60(N+1)` |

Sparse prefixes use `count(d_i < 0.02) - N` and have horizons `100N` for
planar, `60N` for planar-position, and `60N` for discrete planar-position.
The four motion choices in the discrete family are right, left, up, and down;
the first `N` actions select the active cube without moving it.

`constants.py` contains stale `discrete-planar-4-cube-N` entries, but
`make_env` has no matching branch. Conversely, the routing regex accepts
other half-grid sizes and cube counts, but they fail unless matching timing
and contact constants exist. The supported set is therefore the explicit
half-grid-4, cube-1-through-5 IDs above.

PDWrapper and the continual semantic/grouped wrappers assume the creative
3-D five-action contract and must not be applied to planar environments
without a separately versioned adapter.

## Known contract risks

- The environment registry is partly regex-based and partly an explicit
  dictionary, so a regex match does not guarantee a runnable environment.
- Goal-task diversity is extremely small: two goals for one cube and one goal
  for each larger cube count. Reward variants do not create new semantic
  tasks.
- Raw cube labels have no persistent physical semantics under random start
  permutation and permutation-invariant reward. Stable meaning must come from
  shared slot processing, masks, and goal hashes—not from assuming slot 0 is a
  particular conceptual cube.
- Eval success is “reached at least once”; terminal success should be logged as
  a separate metric if a later paper uses that definition.
