# Direct Action-Value Ranking

## Goal

The action-value path ranks the actor's complete legal top-K actions with
`Q(s, a)` directly. It does not apply candidate actions to environment copies
at inference time and does not evaluate successor states with `V(s')`.

The actor checkpoint and action-value checkpoint are separate artifacts:

- the actor proposes complete `(high_level, aircraft_id, target_slot)` actions;
- `ActionValueRanker` copies the actor representation weights only at
  initialization;
- Q training may update its independent aircraft, target, and global encoders;
- Q training verifies that every actor parameter remains bitwise unchanged;
- the Q checkpoint records the source actor SHA-256 and cannot be loaded as an
  actor checkpoint.

## Samples And Objective

Collection follows the existing ambiguous-decision stride and reservoir
protocol. Every retained source state stores:

- the source `EncodedObservation`;
- the complete legal action and selected target slot;
- the action-specific target auxiliary representation;
- a CRN group ID;
- the deterministic continuation target:

```text
terminal completed sorties - completed sorties at the source state
```

Every alternative in a group gets an independently cloned environment whose
RNG is initialized with the same derived branch seed. The live environment RNG
state is never copied. Training minimizes Huber regression loss plus pairwise
logistic ranking loss over unequal targets in the same CRN group.

The Q head is explicitly conditioned on the state context, selected aircraft
embedding, high-level action embedding, and selected target representation.

## Seed Protocol

| Purpose | Seeds |
|---|---|
| Q fitting | 42001-42010 |
| Model selection | 41001-41005 |
| Held-out development evaluation | 43001-43005 |

The training and selection partitions must be non-empty, unique, and disjoint.
All action-value CLI paths reject reserved frozen seeds `70001-70050`.

## Reproduction

Run the bounded end-to-end smoke first:

```bash
python -m scripts.train_action_value \
  --actor-checkpoint checkpoints/rl_multiload_bc.pt \
  --output-checkpoint /tmp/rl_action_value_smoke.pt \
  --metrics-output /tmp/rl_action_value_smoke.csv \
  --training-seeds 42001 \
  --selection-seeds 41001 \
  --wave-intervals 60 \
  --waves 1 \
  --stride 50 \
  --max-branch-states 1 \
  --top-k 2 \
  --workers 1 \
  --epochs 1 \
  --minibatch-size 2

python -m scripts.evaluate_rl \
  --checkpoint checkpoints/rl_multiload_bc.pt \
  --action-value-checkpoint /tmp/rl_action_value_smoke.pt \
  --action-value-top-k 2 \
  --seed 43001 \
  --runs 1 \
  --simulation-duration 60 \
  --wave-interval 60
```

Run the complete development experiment:

```bash
PYTHON_BIN=python DEVICE=cuda \
  bash scripts/run_action_value_experiment.sh
```

The default run collects at most 1,600 fitting candidate continuations and 800
selection candidate continuations:

```text
5 intervals * 10 fitting seeds * 8 states * top-4 = 1,600
5 intervals * 5 selection seeds * 8 states * top-4 = 800
```

It then performs 100 Q-training epochs and five online top-3 evaluation
episodes. Collection is CPU simulator work; GPU mainly accelerates Q fitting
and inference. Actual sample counts can be lower if a trajectory has fewer
eligible ambiguous decisions.

Direct inference is also available through the unified solver:

```bash
python -m scripts.solve \
  --solver rl \
  --checkpoint checkpoints/rl_multiload_bc.pt \
  --action-value-checkpoint checkpoints/rl_multiload_action_value.pt \
  --action-value-top-k 3 \
  --seed 43001
```

At each decision, actor candidates are sent through one batched Q forward pass.
Actor rank is the deterministic tie-break. Omitting
`--action-value-checkpoint` preserves the existing actor behavior.
