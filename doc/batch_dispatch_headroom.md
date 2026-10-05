# Event-Level Batch Dispatch Headroom

## Decision

Batch-policy training is stopped before using the frozen evaluation set. On
the 47.5-minute, 12-wave scenario, an optimistic trajectory search did not
find the five-sortie margin required to justify training.

This is an empirical gate result, not a mathematical proof that no policy can
ever reach the target.

## Environment Change

`CarrierAircraftSchedulingEnv.step_batch()` starts an ordered list of actions
at one simulation timestamp and then advances to the next event. Each action
is checked against the state left by earlier actions in the batch.

An empty batch explicitly advances time even when legal work remains. This
adds the missing ability to reserve resources or deck space intentionally
without changing the existing `step()` contract.

## Search Protocol

The diagnostic uses development seeds only:

| Purpose | Seeds |
|---|---|
| Candidate search | 66001-66005 |
| Fixed-policy confirmation | 66006-66010 |
| Frozen final evaluation | 70001-70050, not used |

For each search seed, `scripts/diagnose_batch_headroom.py` evaluates 256
complete trajectories. Candidate policies vary:

- launch aircraft ranking;
- recovery aircraft ranking;
- launch/recovery precedence;
- randomized selection among the top 1, 2, 3, or 5 aircraft;
- shortest-target versus randomized target assignment.

Every policy makes online decisions without reading future events or the
environment RNG. The diagnostic then selects the best terminal trajectory for
each seed. That after-the-fact selection is clairvoyant, so its result is an
optimistic headroom estimate and must not be reported as an online policy.

## Search Result

The 1280 trajectory records are in
`outputs/batch_headroom_seed66001_5x256.csv`.

| Seed | CP-SAT | Best of 256 | Optimistic delta |
|---:|---:|---:|---:|
| 66001 | 100 | 104 | +4 |
| 66002 | 101 | 103 | +2 |
| 66003 | 100 | 103 | +3 |
| 66004 | 100 | 103 | +3 |
| 66005 | 100 | 103 | +3 |
| Mean | 100.2 | 103.2 | +3.0 |

No searched trajectory reached its paired CP-SAT score plus five sorties.

Candidate 244 was the best single fixed candidate on the search seeds. Its
configuration is:

```text
launch_mode=duration
recovery_mode=fair
precedence=deadline
launch_top_k=2
recovery_top_k=3
random_targets=false
```

It scored `104/103/102/103/102`, averaging 102.8 and exceeding CP-SAT by
2.6 sorties on the search seeds.

## Independent Confirmation

The fixed candidate was evaluated without retuning on seeds 66006-66010.
Detailed paired results are in
`outputs/batch_headroom_confirmation_seed66006_5.csv`.

| Method | Mean sorties |
|---|---:|
| CP-SAT | 101.0 |
| Fixed batch candidate 244 | 101.4 |

The paired deltas are `0, +1, 0, +1, 0`: two wins, three ties, and a mean
gain of 0.4 sorties. This does not reproduce the search-set gain and is far
below the required +5 threshold.

Additional single-seed probes did not reveal hidden batch headroom:

- launch-before-recovery reduced 98 sorties to 96;
- limiting concurrent service vehicles matched 98 at useful limits and
  degraded sharply at lower limits;
- wave launch quotas peaked at 101 among the tested configurations.

## Conclusion

The batch interface is valid and exposes intentional idling, but the tested
batch policy family does not have enough demonstrated headroom over CP-SAT.
Training a batch BC/PPO model would at best distill a teacher whose independent
gain is 0.4 sorties. Per the pre-registered gate, no batch model is trained and
seeds 70001-70050 remain untouched.
