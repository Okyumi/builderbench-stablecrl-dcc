# Fixed inputs, meta-learning, and the paper setup

## Fixed padding for continual RL

Yes: choose the largest task **before training starts**. Sequence A has at
most 8 cubes, so every task uses 8 cube slots from task 1 onward. The network
shape never changes at a task boundary.

This repository uses a semantic layout, not the old raw `13N + 1` layout:

```text
state = 8 cube slots × 14 values + 8 validity bits = 120
goal  = 8 target slots × 3 values + 8 goal-mask bits = 32
action                                             = 5
```

For a one-cube task, cube slot 0 is filled, slots 1--7 are zero, and the
state validity mask is `[1, 0, 0, 0, 0, 0, 0, 0]`. This is true from the
first task; we do not enlarge the input later.

A **validity mask** says which state slots contain real cubes. It prevents a
network from treating zero padding as real cubes at position zero.

A **goal mask** has a different meaning: it says which real cubes have a
required target. A helper cube can therefore have state-validity `1` and
goal-mask `0`. BuilderBench uses this for structures where some cubes may be
temporary supports or counterweights.

## What each network receives

- The actor receives the fixed 120-value state and an encoded goal.
- The state-action encoder `phi(s, a)` receives the fixed 120-value state and
  the unchanged 5-value PD action.
- The goal encoder `psi(g)` receives the fixed 32-value goal.
- The action is never padded. It is always `[x, y, z, yaw, cube_selector]`.

Thus the shared actor, shared state-action encoder, and shared goal encoder
can be carried through the whole continual sequence.

## Is the proposed meta-learning object encoder correct?

Broadly, yes. A cleaner description is:

1. apply the **same small cube encoder** to every valid cube slot;
2. combine the cube representations with masked pooling or attention;
3. produce one fixed-size state representation;
4. combine it with the action in the state-action encoder;
5. process goal slots in the same way to produce one fixed-size goal
   representation.

This resembles token processing, but each cube is already one structured
token. We do not need a tokenizer that only records the number of cubes. The
mask already provides that information. The state and goal encoders may share
some per-cube weights, but they need not be identical because state cubes
contain velocity and rotation while goal slots contain target positions.

Padding is still useful for batching. The object encoder and mask make the
representation less dependent on slot order and cube count.

## What earlier work does

Most standard meta-RL benchmarks keep observation and action dimensions
fixed. For example, PEARL uses a permutation-invariant encoder for a
variable-size **set of experience transitions**, but the environment state
shape itself is fixed. Variable object counts are usually handled by applying
shared weights to each object and then using sum/mean pooling, a graph
network, or attention. This is the pattern used by
[Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html),
[Set Transformer](https://proceedings.mlr.press/v97/lee19d), and object-centric
RL work that generalizes across object counts
([Wilson and Hermans, 2020](https://proceedings.mlr.press/v100/wilson20a.html);
[OP3](https://proceedings.mlr.press/v100/veerapaneni20a.html)).

So the proposed cube encoder follows established practice. It is an
architectural choice for variable objects, not meta-learning by itself.

## How to train it with online contrastive RL

Do not pretrain only on one easy task and call the result a meta-learner. That
would bias the representation toward one geometry.

A concrete later meta-learning experiment should be:

1. define separate meta-train and held-out meta-test structures;
2. collect replay online from several meta-train tasks, with balanced task
   sampling;
3. train the cube, state-action, goal, and actor networks end to end using the
   same future-goal contrastive loss;
4. use masks in both target goals and relabelled future goals;
5. test unseen structures or cube counts with no update for **zero-shot**
   evaluation, then optionally allow a small number of online updates for a
   separate few-shot result.

A short warm-up with random or exploratory data is fine. A separately
pretrained cube encoder is optional, not required. The decisive test is
whether joint multi-task training learns a representation that transfers.

## Zero-shot does not require an offline-only project

The well-known zero-shot RL study used offline replay to control data coverage
and separate representation quality from exploration
([Touati, Rapin, and Ollivier, 2022](https://arxiv.org/abs/2209.14935)). That
does not mean all zero-shot RL must be offline.

We can collect the pretraining replay **online** on training tasks, freeze the
learned policy/encoders, give an unseen goal, and evaluate without gradient
updates. That is a valid zero-shot test. It will still depend strongly on
whether online exploration covered the skills needed by the unseen goal.

For this paper, zero-shot should be an extension, not the main claim.

## Continual learning versus curriculum learning

- **Curriculum learning:** task order is chosen mainly to make a final task
  easier. Retaining every earlier task is not the main goal.
- **Continual learning:** tasks arrive in sequence, past training data is not
  freely mixed back in, and the agent is evaluated on old tasks after every
  new task. Forward transfer and forgetting are both measured.

Sequence A becomes a continual benchmark because of this training and
evaluation protocol. Its increasing difficulty also gives it a curriculum
shape; task order alone does not decide the label.

## Recommendation for this paper

Use Sequence A as the main continual stream with `max_cubes=8`, PD-5, fresh
per-task replay, no old-task replay, and a full after-each-task evaluation
matrix. Compare reset StableCRL, persistent StableCRL, and DCC under matched
budgets. Report forward transfer, final old-task performance, and forgetting.

Because Sequence A generally becomes harder, also run at least a reversed or
counterbalanced order. Keep Sequence B as a registered construction sequence
and a later robustness test. Present meta-learning and zero-shot transfer as
follow-up experiments unless their held-out results are strong enough to
support a separate claim.
