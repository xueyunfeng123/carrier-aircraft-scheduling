# Offline Value Calibration

## Goal

The retained multi-load policy was trained primarily through behavior cloning.
Its actor is useful, but its critic was previously fitted only on states visited
by that actor's greedy trajectory. This experiment calibrates the critic on
counterfactual successor states from alternative actor-ranked actions and
measures whether that critic can improve decisions through one-step reranking.

The business objective is unchanged: maximize completed launches within the
fixed horizon. With the checkpoint's default PPO reward configuration, the
Monte Carlo value target is the remaining number of completed launches.

## Protocol

The experiment uses five wave intervals and 12 waves per scenario:

```text
47.5, 52.5, 57.5, 62.5, 67.5 minutes
```

Seed groups are disjoint:

| Purpose | Seeds |
|---|---|
| Critic fitting | 42001-42010 |
| Calibration model selection | 41001-41005 |
| Held-out development evaluation | 43001-43005 |

Seeds `70001-70050` are reserved and rejected by the command-line validation.
The held-out development set is not used to fit the value head or select its
hyperparameters.

Collection follows the deterministic base actor and samples decision states at
a configurable stride. At each sampled state it enumerates the actor's top-K
complete legal `(operation, aircraft, target)` actions. Every candidate is
applied to its own deep clone. The live environment RNG is replaced during the
clone operation, so neither its state nor its future draws are exposed.
Candidates from one branch state receive separately instantiated RNGs with the
same seed (common random numbers).

After the candidate action and any forced event advances, the resulting
successor observation is stored. The deterministic base actor then rolls that
candidate to a true terminal state. Its label is:

```text
terminal completed sorties - sorties already completed at the successor
```

This is a direct, undiscounted Monte Carlo remaining-sortie target. The
aircraft encoder, target encoder, global encoder, and all actor heads remain
frozen; only `value_head` is optimized and all non-value tensors are checked
for exact equality after fitting.

The collection controls are:

| Option | Meaning |
|---|---|
| `--stride` | Sample every Nth base-actor decision |
| `--max-branch-states` | Maximum sampled states per `(interval, seed)` trajectory |
| `--top-k` | Complete legal actor actions evaluated per sampled state |
| `--workers` | Concurrent independent candidate rollouts |
| `--branch-seed` | Root seed used to derive per-state CRN streams |

Calibration reports MAE, RMSE, bias, Pearson correlation, and Spearman
correlation overall and by wave interval and horizon third. The calibrated
checkpoint records the target method, collection controls, explicit seed
partitions, load range, and fit statistics. Reranking refuses checkpoints
without compatible metadata or scenarios outside the calibrated interval
range.

## One-Step Reranking

At each decision, the actor enumerates its top `K` complete legal actions.
Each candidate is applied to a deep copy of the current environment. The copy
uses the same newly seeded random stream for every candidate, so ranking does
not inspect or consume the real environment's future random stream. Events are
advanced to the next decision boundary and candidates are scored as:

```text
immediate accumulated reward + gamma * calibrated V(next decision state)
```

The highest score is selected, with actor rank as the deterministic tie-break.
`K=1` preserves the original actor behavior and does not require calibration
metadata.

## Reproduction

Run from the repository root:

```bash
PYTHON_BIN=.venv/bin/python DEVICE=cuda \
  bash scripts/run_value_calibration_experiment.sh
```

For a bounded development smoke run:

```bash
python -m scripts.calibrate_value \
  --checkpoint checkpoints/rl_multiload_bc.pt \
  --output-checkpoint /tmp/rl_counterfactual_smoke.pt \
  --metrics-output /tmp/rl_counterfactual_smoke.csv \
  --calibration-seeds 42001 \
  --selection-seeds 41001 \
  --wave-intervals 60 \
  --waves 1 \
  --stride 50 \
  --max-branch-states 1 \
  --top-k 2 \
  --workers 1 \
  --epochs 1
```

All calibration, selection, and ordinary evaluation lists must be non-empty,
internally unique, pairwise disjoint where applicable, and positive. Seeds
`70001-70050` are rejected and must only be used by the separately controlled
frozen final evaluation, never by these commands.

The script writes:

```text
checkpoints/rl_multiload_value_calibrated.pt
outputs/value_calibration_seed41001_5.csv
outputs/value_calibration_evaluation_seed43001_5.csv
outputs/value_rerank_seed43001_5.csv
```

The first CSV compares critic metrics before and after fitting on the selection
set. The second is the untouched held-out development evaluation. The third
contains paired actor and top-3 rerank sortie counts and runtimes.

## Results

Results are recorded after the reproducible remote run completes. Development
claims use `43001-43005` and report paired wins, ties, losses, mean sortie
delta, exact sign-test p-value, and runtime overhead. Seeds `70001-70050`
remain reserved for a separately authorized frozen final evaluation.
