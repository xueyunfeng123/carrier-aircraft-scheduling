# Offline Value Calibration

## Goal

The retained multi-load policy was trained primarily through behavior cloning.
Its actor is useful, but its critic is not guaranteed to predict complete
episode returns accurately enough for policy-value search. This experiment
fits only the checkpoint's `value_head` on complete deterministic policy
trajectories and measures whether that critic can improve decisions through
one-step reranking.

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
| Final evaluation | 43001-43005 |

Seeds `70001-70050` are reserved and rejected by the command-line validation.
The final evaluation set is not used to fit the value head or select its
hyperparameters.

For every legal decision state, trajectory collection stores the encoded
observation and all shaped rewards through the next decision boundary. Targets
are terminal Monte Carlo returns by default. The aircraft encoder, target
encoder, global encoder, and all actor heads remain frozen; only
`value_head` is optimized.

Calibration reports MAE, RMSE, bias, Pearson correlation, and Spearman
correlation overall and by wave interval and horizon third. The calibrated
checkpoint records the target method, reward configuration, seed sets, load
range, and fit statistics. Reranking refuses checkpoints without compatible
metadata or scenarios outside the calibrated interval range.

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

The script writes:

```text
checkpoints/rl_multiload_value_calibrated.pt
outputs/value_calibration_seed41001_5.csv
outputs/value_calibration_evaluation_seed43001_5.csv
outputs/value_rerank_seed43001_5.csv
```

The first CSV compares critic metrics before and after fitting on the selection
set. The second is the untouched final critic evaluation. The third contains
paired actor and top-3 rerank sortie counts and runtimes.

## Results

Results are recorded after the reproducible remote run completes. Any rerank
claim must use the final `43001-43005` seeds and report paired wins, ties,
losses, mean sortie delta, exact sign-test p-value, and runtime overhead.
