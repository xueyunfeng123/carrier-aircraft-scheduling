# CLAUDE.md

This file defines the project scope and implementation conventions for agents
working in this repository.

## Sources of truth

There are three distinct layers. Do not silently mix them.

1. `../前置约束.md`
   - Cleaned version of the requirements supplied by the problem owner.
   - Describes the intended full scenario: 75 aircraft, 45 aircraft on deck,
     five spares, failures, sea state, personnel shifts, transport paths, and
     other operational constraints.
   - Treat it as reference material. Ambiguous items listed under its
     "待确认项" are not implementation requirements until confirmed.
2. [Feishu mathematical model](https://mcnubpcdg10x.feishu.cn/docx/COAFdmqXdoj0OnxEIPvciG2jnPc)
   - Defines the current base optimization problem independently of any solver.
   - This is the conceptual source of truth for objective, variables,
     constraints, and model boundaries.
3. Source code under `env/`, `solution/`, and `rl/`
   - Implements a simplified executable instance of the base model.
   - Code behavior is not evidence that an omitted requirement was rejected.

When these layers disagree, state the discrepancy explicitly. Do not expand the
base model to the full supplied scenario without a deliberate modeling change.

## Optimization problem

The problem is a finite-horizon, multi-wave scheduling problem with stochastic
processing times and shared renewable resources.

The optimization problem must remain independent of the solution method.
Heuristics, sampled planning, mathematical programming, and reinforcement
learning are alternative solvers; none of them defines the problem itself.

### Single objective

The only optimization objective in the current base model is:

```text
maximize the expected number of completed launches within the fixed horizon
```

Equivalently:

```text
max E[N_sortie]
```

`total_missed_sorties`, waiting time, and resource utilization are evaluation
metrics, not additional optimization objectives. With a fixed number of launch
opportunities and `y[i,k] + m[i,k] = 1`, minimizing missed sorties is equivalent
to maximizing completed sorties and must not be added as a second objective.

### Base scenario

- 40 aircraft, split into groups A and B with 20 aircraft each.
- 45 abstract parking spots.
- Fixed wave interval `wave_interval`, default 120 minutes.
- Fixed simulation horizon `simulation_duration`, default 720 minutes.
- Wave 0 launches A and has no recovery group.
- Odd waves launch B and recover A.
- Even waves after wave 0 launch A and recover B.
- Aircraft cycle:

```text
launch -> airborne -> recovery -> parking -> fueling || arming -> launch
```

- Fueling and arming may run in parallel after recovery.
- Launch requires both fueling and arming to be complete.
- Decisions are made at event times; time advances to the next queued event
  when no dispatch action is available.

### Base resources

| Resource | Base-model capacity | Supplied requirement |
|---|---:|---:|
| Recovery channel/runway | 1 | 1 |
| Launch channel/position | 1 | 4 launch positions |
| Parking spots | 45 | 45 |
| Fuel servers | 20 | 20 stations, each covering two spots |
| Arm vehicles | 10 | 10 |
| Ammo transport vehicles | 10 | 10 |
| Lower weapon lifts | 6 | 6 |
| Upper weapon lifts | 4 | 4 |
| Active support personnel | 50 | 150 people across 50 three-shift posts |

The base model interprets personnel as one aggregate pool of 50 simultaneously
available workers. It does not model individual workers or shift rotation.

### Stochastic processing assumptions

- Recovery: fixed 1 minute.
- Launch: fixed 1 minute.
- Fueling: truncated normal duration with mean 20 and standard deviation 3.
- Required ammunition quantity:
  - 1 item with probability 0.3;
  - 2 items with probability 0.4;
  - 3 items with probability 0.2;
  - 4 items with probability 0.1.
- Arming stage 1:
  - ammunition extraction sampled uniformly from 5 to 10 minutes;
  - lower-lift time sampled from a normal distribution with mean 3 and
    standard deviation 0.5.
- Arming stage 2:
  - upper-lift time sampled from a normal distribution with mean 3 and
    standard deviation 0.5;
  - parking-to-aircraft transfer time;
  - per-item arming time sampled from a normal distribution with mean 5 and
    variance 2.
- Every sampled processing duration is clamped to at least 0.1 minute.

These distribution parameters are modeling choices made by the executable base
model. The supplied requirements give several means but omit their variance or
standard deviation.

### Core constraints

- An aircraft can launch only during a wave assigned to its group.
- Recovery must complete before fueling or arming starts.
- Arming stage 1 must complete before arming stage 2.
- Fueling and arming may overlap.
- Both fueling and arming must complete before launch.
- A parking spot can contain at most one aircraft at a time.
- Concurrent resource and personnel usage must not exceed configured capacity.
- An aircraft cannot be airborne and perform deck operations simultaneously.
- The same operation instance cannot be started more than once.

## Deliberate simplifications

The executable base model deliberately excludes the following supplied
requirements:

- 75-aircraft full inventory and the five-aircraft spare replacement process;
- aircraft failures, repair distribution, hangar transfer, and return to service;
- sea-state-dependent launch delay, recovery success rate, and wave-off;
- pilots, command staff, individual support staff, rest, fatigue, and shifts;
- four physical launch positions and interference with the recovery area;
- aircraft elevators and tractor allocation;
- explicit taxi routes, path conflicts, jet-blast constraints, and deck topology;
- ammunition types, compatibility, inventory, assembly capacity, and storage;
- fuel state while airborne and recovery queue endurance constraints.

Parking geometry is currently reduced to a scalar transfer-time function:
spot 0 takes 2 minutes, and every two additional spot indices add 1 minute.

Do not describe these excluded items as implemented. Add them only through an
explicit model extension with corresponding state, constraints, tests, and
evaluation scenarios.

## Architecture

| File | Role |
|---|---|
| `env/config.py` | Base capacities, durations, reward constants, and action IDs |
| `env/carrier_aircraft_env.py` | Event queue, aircraft state machine, resource accounting, wave transitions, masks, and metrics |
| `solve.py` | Unified command-line runner and CSV export |
| `solution/random_solver.py` | Uniform random legal-action baseline |
| `solution/heuristic_solver.py` | Wave/deadline-aware slack heuristic |
| `solution/sampled_random_solver.py` | Best-of-N complete random rollout planner |
| `solution/rl_solver.py` | Inference wrapper for a PyTorch checkpoint |
| `rl/obs_encoder.py` | Normalized aircraft/global features and legal-action masks |
| `rl/model.py` | Hierarchical policy/value network |
| `rl/ppo_trainer.py` | Masked action selection and PPO updates |
| `train_rl.py` | Rollout collection, training, checkpointing, and evaluation |
| `evaluate_rl.py` | Checkpoint evaluation |

### Solver contract

Every solver exposes:

```python
choose_action() -> Optional[Dict[str, int]]
```

A dispatch action is:

```python
{"high_level": 0 | 1 | 2 | 3, "aircraft_id": int}
```

The high-level actions are recovery (`R`), fueling (`F`), arming (`M`), and
launch (`L`). Return `None` only when no legal action is available. All solvers
must honor `env.get_action_mask()`.

### Event system

Events are stored in a `heapq` priority queue. Current event types are:

- `wave_start`
- `recover_done`
- `park_done`
- `fuel_done`
- `ammo_to_assembly_done`
- `arm_done`
- `launch_done`

The four principal status dimensions are `recovery_status`, `fuel_status`,
`arm_status`, and `launch_status`. Arming additionally uses `arm_stage`.

## Commands

Run commands from `proj/`.

```bash
# Heuristic baseline
python solve.py
python solve.py --solver heuristic --runs 10 --wave-interval 60 --simulation-duration 720

# Other implemented solvers
python solve.py --solver random --seed 42 --runs 5
python solve.py --solver sampled --sampled-samples 30
python solve.py --solver rl --checkpoint checkpoints/rl_policy.pt

# Export results
python solve.py --solver heuristic --runs 10 \
  --runs-csv runs.csv \
  --timing-csv timing.csv \
  --missed-csv missed.csv

# Train and evaluate PPO
python train_rl.py --checkpoint checkpoints/rl_policy.pt
python evaluate_rl.py --checkpoint checkpoints/rl_policy.pt --runs 10

# Syntax smoke check
python -m compileall -q env solution rl solve.py train_rl.py evaluate_rl.py
```

Core environment and non-RL solvers use the Python standard library. RL
training and inference require PyTorch 2.0 or later. There is currently no
automated test suite or configured linter.

## Development rules

- Preserve the single business objective: maximize completed launches within
  the fixed horizon.
- Keep optimization metrics separate from solver-specific training rewards.
- Do not add a requirement merely because it appears in `../前置约束.md`;
  first resolve its ambiguity and update the mathematical model.
- Keep resource acquisition and release balanced on every event path.
- Never allow action masks and action validation to disagree.
- Preserve deterministic replay for a fixed environment seed and solver seed.
- Compare solvers using identical scenario parameters and random seeds.
- Report at least completed launches, missed sorties, simulation completion,
  and resource feasibility.
- Update this file, the mathematical model, and user-facing documentation when
  the base problem definition changes.
