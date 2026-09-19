# CLAUDE.md

This file defines the project scope and implementation conventions for agents
working in this repository.

## Sources of truth

There are three distinct layers. Do not silently mix them.

1. `doc/前置约束.md`
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

- 45 aircraft share the deck fleet; the final five are tagged as initial
  reserves for observability but obey the same readiness rules.
- 45 capacity-one parking nodes in four Haitian-derived clusters:
  northwest 10, northeast 13, southwest 12, and southeast 10.
- Fixed wave interval `wave_interval`, default 120 minutes.
- Fixed simulation horizon `simulation_duration`, default 720 minutes.
- A/B are alternating wave-demand labels, not immutable aircraft groups.
- Every wave dynamically selects up to 20 ready aircraft from the shared fleet.
- The aircraft actually launched in one wave become recovery candidates in the
  next wave; overdue airborne aircraft remain eligible for recovery.
- Aircraft cycle:

```text
launch -> airborne -> recovery -> parking
       -> fueling || inspection || arming -> launch
```

- Fueling, inspection, and arming may run in parallel after recovery.
- Launch requires fueling, inspection, and arming to be complete.
- Decisions are made at event times; time advances to the next queued event
  when no dispatch action is available.

### Base resources

| Resource | Base-model capacity | Supplied requirement |
|---|---:|---:|
| Recovery channel/runway | 1 | 1 |
| Launch operation channels | 3 | 3 takeoff runways |
| Launch position nodes | 3 | Runways 29, 30, and 31 |
| Parking spots | 45 | 45 |
| Fuel vehicles | 20 | Mobile equipment |
| Inspection vehicles | 10 | Mobile equipment |
| Arm vehicles | 10 | 10 |
| Tow vehicles | 10 | Mobile equipment |
| Ammo transport vehicles | 10 | 10 |
| Lower weapon lifts | 6 | 6 |
| Upper weapon lifts | 4 | 4 |
| Active support personnel | Not scheduled | Deferred |

Personnel state remains observable for compatibility, but personnel capacity
does not mask or consume actions while `personnel_scheduling_enabled=False`.

### Stochastic processing assumptions

- Recovery: fixed 1 minute.
- Launch: fixed 1 minute.
- Fuel is continuous on `[0, 1]`. Aircraft default to `0.2` after a sortie and
  refuel at `0.05` fuel units per minute, so a full refuel takes 16 minutes.
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
  - per-item arming time sampled from a normal distribution with mean 5 and
    variance 2.
- Every sampled processing duration is clamped to at least 0.1 minute.

These distribution parameters are modeling choices made by the executable base
model. The supplied requirements give several means but omit their variance or
standard deviation.

### Core constraints

- Any grounded aircraft can fill a current-wave launch slot after fueling and
  arming are complete.
- At most 20 launch operations may start in one wave.
- Missed sorties are unfilled wave-demand slots, not misses assigned to fixed
  aircraft identities.
- Recovery must complete before fueling, inspection, or arming starts.
- Arming stage 1 must complete before arming stage 2.
- Fueling, inspection, and arming may overlap.
- Fueling, inspection, and arming must complete before launch.
- A parking spot can contain at most one aircraft at a time.
- Launch explicitly selects an available route from the aircraft parking node
  to one of three launch-runway nodes.
- Recovery requires an available route from the recovery runway to an
  explicitly selected unoccupied parking node.
- Fueling, inspection, and arming explicitly select an available service
  vehicle.
- A tow vehicle first travels to the aircraft and then escorts it to the
  selected parking position or launch runway.
- Aircraft and all mobile vehicles share a Cooperative A* reservation table.
  Nodes are mutually exclusive at each time tick; same-edge overlap and
  opposite-direction traversal are prohibited.
- Concurrent equipment usage must not exceed configured capacity.
- An aircraft cannot be airborne and perform deck operations simultaneously.
- The same operation instance cannot be started more than once.

## Deliberate simplifications

The executable base model deliberately excludes the following supplied
requirements:

- 75-aircraft full inventory and failure-driven hangar replacement; five
  initial deck reserves are modeled as part of the shared 45-aircraft pool;
- aircraft failures, repair distribution, hangar transfer, and return to service;
- sea-state-dependent launch delay, recovery success rate, and wave-off;
- pilots, command staff, individual support staff, rest, fatigue, and shifts;
- aircraft elevators;
- measured deck coordinates and the real adjacency matrix;
- continuous coordinates, turning radius, geometric separation distance, and
  jet-blast constraints;
- mutual exclusion between the two launch positions stated to be in the
  recovery area and the recovery runway;
- ammunition types, compatibility, inventory, assembly capacity, and storage;
- fuel burn as a function of airborne time and recovery queue endurance.

Movement uses a schematic graph with 15 pathway nodes and one-minute edges.
Cooperative A* plans on `(node, time tick)` states and inserts waits to resolve
node and edge conflicts. This is a discrete time-space tube, not the continuous
aircraft-convex-hull collision model used by higher-fidelity studies.

Do not describe these excluded items as implemented. Add them only through an
explicit model extension with corresponding state, constraints, tests, and
evaluation scenarios.

## Architecture

| File | Role |
|---|---|
| `env/config.py` | Base capacities, durations, reward constants, and action IDs |
| `env/scenario.py` | Solver-independent scenario, process, mission-plan, duration-distribution, and deck-graph definitions |
| `env/project_deck_graph.py` | Haitian-derived schematic topology and static destination occupancy |
| `env/traffic_planner.py` | Cooperative A* node/edge reservation table shared by aircraft and mobile vehicles |
| `env/cbs_planner.py` | Optimal sum-of-costs CBS for a fixed batch of movement requests |
| `env/carrier_aircraft_env.py` | Event queue, aircraft state machine, resource accounting, wave transitions, masks, and metrics |
| `env/yoon_sortie_env.py` | Executable Yoon 2023 structural replication with mission, movement, runway, lift, hangar, and maintenance events |
| `scripts/solve.py` | Unified command-line runner and CSV export |
| `scripts/inspect_scenario.py` | Read-only exporter for project and paper scenario definitions |
| `scripts/run_yoon_baseline.py` | Multi-case replication runner with confidence intervals and paper-result comparison |
| `solution/random_solver.py` | Uniform random legal-action baseline |
| `solution/priority_rule_solver.py` | FIFO, SPT, and EDD dispatching-rule baselines |
| `solution/heuristic_solver.py` | Wave/deadline-aware slack heuristic |
| `solution/cp_sat_solver.py` | Rolling CP-SAT resource-allocation baseline |
| `solution/rl_solver.py` | Inference wrapper for a PyTorch checkpoint |
| `solution/README.md` | Solver behavior, tradeoffs, and invocation guide |
| `rl/obs_encoder.py` | Normalized aircraft/global features and legal-action masks |
| `rl/model.py` | Hierarchical policy/value network with action-conditioned aircraft logits |
| `rl/behavior_cloning.py` | Heuristic demonstration collection and policy pretraining |
| `rl/ppo_trainer.py` | Masked action selection and PPO updates |
| `scripts/evaluation_defaults.py` | Fixed training and evaluation seeds, horizon, and run count |
| `scripts/train_rl.py` | Rollout collection, training, checkpointing, and evaluation |
| `scripts/evaluate_rl.py` | Checkpoint evaluation |
| `scripts/random_policy_test.py` | Legacy random-policy timing report |
| `scripts/benchmark_non_rl.py` | Reproducible benchmark for non-RL solvers and an optional RL checkpoint |
| `outputs/` | Tracked baseline CSV results and comparison figures |
| `doc/yoon_2023_environment_spec.md` | Traceable Yoon 2023 parameters, missing data, replication scope, and acceptance criteria |
| `doc/` | Requirements, modeling notes, and project guidance |
| `tests/` | Standard-library unit and solver integration tests |

### Solver contract

Every solver exposes:

```python
choose_action() -> Optional[Dict[str, Any]]
```

A dispatch action is:

```python
{
    "high_level": 0 | 1 | 2 | 3 | 4,
    "aircraft_id": int,
    "target_id": int,      # recovery parking or launch runway
    "vehicle_id": str,     # fuel, arm, or inspection vehicle
}
```

The high-level actions are recovery (`R`), fueling (`F`), arming (`M`), launch
(`L`), and inspection (`I`). Only the field relevant to the selected action is
required. Return `None` only when no legal action is available. All solvers
must honor `env.get_action_mask()`.

### Event system

Events are stored in a `heapq` priority queue. Current event types are:

- `wave_start`
- `recover_done`
- `park_done`
- `fuel_done`
- `inspection_done`
- `ammo_to_assembly_done`
- `arm_done`
- `taxi_to_launch_done`
- `launch_done`
- `simulation_end`

The five principal status dimensions are `recovery_status`, `fuel_status`,
`inspection_status`, `arm_status`, and `launch_status`. Arming additionally
uses `arm_stage`.

## Commands

Run commands from `proj/`.

```bash
# Create and activate the Conda environment
conda create --name carrier-aircraft-scheduling --override-channels \
  --channel conda-forge python=3.12 pip -y
conda activate carrier-aircraft-scheduling
python -m pip install -r requirements.txt

# Heuristic baseline
python -m scripts.solve

# Other implemented solvers
python -m scripts.solve --solver random
python -m scripts.solve --solver fifo
python -m scripts.solve --solver spt
python -m scripts.solve --solver edd
python -m scripts.solve --solver cp_sat --cp-sat-max-time 0.05
python -m scripts.solve --solver rl --checkpoint checkpoints/rl_policy.pt

# Export results
python -m scripts.solve --solver heuristic \
  --runs-csv outputs/runs.csv \
  --timing-csv outputs/timing.csv \
  --missed-csv outputs/missed.csv

# Train with heuristic behavior cloning followed by PPO, then evaluate
python -m scripts.train_rl --checkpoint checkpoints/rl_policy.pt
python -m scripts.evaluate_rl --checkpoint checkpoints/rl_policy.pt

# Compare every solver on the fixed evaluation scenario
python -m scripts.benchmark_non_rl \
  --rl-checkpoint checkpoints/rl_policy.pt \
  --rl-label rl_bc_ppo \
  --output outputs/all_solver_benchmark_60min_seed10007.csv

# Syntax smoke check
python -m compileall -q env solution rl scripts
```

The core environment and rule-based solvers use the Python standard library.
The CP-SAT baseline requires OR-Tools, while RL training and inference require
PyTorch 2.0 or later. Tests use the standard-library `unittest` framework;
there is currently no configured linter.

## Environment constraint experiments

Constraint, paper-baseline, and fleet-model experiments are intentionally kept
outside `main`. Do not merge or cherry-pick them without an explicit decision.

### `experiment/recovery-deadlines`

- Commit: `2e37c29`
- Adds a sampled 8-25 minute recovery deadline and a 10-minute retry delay.
- This assumption is not present in `doc/前置约束.md`; it was introduced only
  to test whether heterogeneous recovery urgency separates solver performance.
- It strongly separates recovery-deadline misses but does not consistently
  increase the gap in completed launches.
- Current decision: retain for reference, do not merge as-is.

### `experiment/shared-launch-recovery-zone`

- Commit: `0e88f7d`
- Models four launch positions: two dedicated and two shared with the recovery
  area. Active recovery blocks shared launch positions, and active shared
  launches block recovery.
- The setup is derived from the supplied requirement that two of four launch
  positions are located in the landing area; the mutual-exclusion assumption
  still requires domain confirmation.
- In the 60-minute scenario, the Heuristic-versus-Random gap increases from
  12.3 to 16.0 completed launches over 12 waves. The effect is small or absent
  at longer wave intervals.
- Current decision: retain for review, not merged.

### `experiment/yoon-sgp-baseline`

- Base commit: `4bd07c8`.
- Primary reference: Yoon et al. (2023), *Discrete Event Simulation of
  Aircraft Sortie Generation on an Aircraft Carrier*.
- B0.1 adds a solver-independent scenario layer, all eight published case
  definitions, published SGP/FlyPro parameters, explicit missing-parameter
  records, a weighted deck-graph primitive, event logging, and SGR metrics.
- B0.2 adds a dedicated `YoonSortieGenerationEnv` with fixed-wing and SAR
  mission events, cancellation deadlines, node-level location occupancy,
  stepwise movement, runway priority, hangar/lift flows, and planned
  maintenance after every two sorties.
- The paper profile is an executable structural replication, not an exact
  numerical reproduction. Missing graph weights, movement speeds, full FlyPro,
  and policy details are isolated in `YoonReplicationAssumptions`.
- The existing `CarrierAircraftSchedulingEnv` continues to execute only
  `project_core`; paper and project task semantics are not silently mixed.
- The project-core parking-time rule is represented as an equivalent star
  graph, preserving the historical benchmark exactly.
- Fixed `seed=10007`, 60-minute, 12-wave regression totals remain:
  Random 141, FIFO 143, SPT 140, EDD 143, Heuristic 152, and CP-SAT 140.
- A 100-run eight-case study is stored in
  `outputs/yoon_2023_structural_replication.csv`. The 16-aircraft large-layout
  cases approach the published rates, while 20-aircraft cases remain much
  lower; this unresolved discrepancy must be treated as a replication gap.
- Current decision: continue B0.3 parameter identification and sensitivity
  analysis on this branch; do not merge before the replication gap is reviewed.

### `experiment/dynamic-shared-fleet`

- Parent commit: `7cb0723` on `experiment/yoon-sgp-baseline`.
- Replaces immutable aircraft A/B membership with 45 shared deck aircraft,
  including five aircraft tagged as initial reserves.
- A/B remain wave-demand labels. Each wave accepts any 20 ready aircraft, and
  the actual launched IDs form the next wave's recovery set.
- Missed sorties are recorded as anonymous unfilled demand slots.
- RL observation schema version changes from 1 to 2; old checkpoints are
  rejected and BC+PPO must be retrained.
- Fixed `seed=10007`, 60-minute, 12-wave results are: Random 143, FIFO 152,
  SPT 157, EDD 158, Heuristic 157, and CP-SAT 162.
- Current decision: retain as the preferred interpretation of the supplied
  five-aircraft reserve requirement, pending review before merging.

### `experiment/spatial-deck-graph`

- Parent commit: `689c008` on `experiment/dynamic-shared-fleet`.
- The first structural commit (`e108537`) used a 19-node single pathway,
  four launch positions, and atomic whole-route reservations.
- The active worktree now replaces that provisional graph with a layout
  derived from `doc/海天杯-技术资料0929更新.pdf`: 45 parking spots in four
  clusters, three launch runways, one landing runway, and north/middle/south
  corridors with cross-connections.
- The source document supplies position geometry and interference tables but
  not a taxiway adjacency matrix. Corridor edges and one-minute edge weights
  remain explicit project assumptions.
- Launch reserves a parking-to-launch route before taxi; recovery reserves the
  runway-to-parking route and destination before recovery starts. Conflicting
  routes are removed by the action mask.
- Fuel, inspection, and arming now use separate mobile vehicle pools. Vehicles
  have locations, include travel time in service completion, and may serve the
  same parked aircraft concurrently when their operations do not conflict.
- Tow vehicles travel to aircraft before launch or post-landing movement.
  Aircraft and all mobile vehicles share Cooperative A* node and edge
  reservations; head-on edge traversal is prohibited.
- Recovery selects a parking target, launch selects a runway, and service
  actions select a vehicle. Multi-stop parking transfer remains excluded.
- Fuel is continuous; personnel scheduling and random failures are disabled.
- Sparse parking/runway interference preserves the Haitian physical
  relationships without copying incompatible 28-position indices.
- Inspection is a fifth high-level action. RL uses 23 aircraft features,
  target/vehicle candidate features, 20 global features, and observation
  schema version 10.
- New RL training defaults to the pure-PyTorch sparse heterogeneous relation
  encoder documented in `doc/heterogeneous_gnn_design.md`. Schema-9
  checkpoints remain inference-compatible through the legacy `deepsets`
  encoder; do not partially load them into the heterogeneous encoder.
- Heuristic, priority-rule, CP-SAT, and RL interfaces have been adapted.
- All 71 unit tests pass. In the corrected fixed seed-10007, 60-minute,
  12-wave run,
  Random/FIFO/SPT/EDD/Heuristic/CP-SAT complete
  99/113/118/106/118/119 launches.
- Every non-random baseline exceeds Random for this regression seed; CP-SAT is
  highest at 119. Treat this as a fixed-seed regression, not a statistical
  ranking.
- Detailed assumptions and remaining work are in
  `doc/haitian_deck_environment_spec.md`.
- Fixed `seed=10007`, 60-minute, 12-wave results with the graph enabled are:
  Random 66, FIFO 74, SPT 90, EDD 53, Heuristic 93, and CP-SAT 86. The
  matched graph-disabled totals remain 143/152/157/158/157/162. These values
  belong to the superseded single-pathway graph.
- The large throughput reduction is evidence that the current atomic
  reservation assumption is restrictive, not evidence of real carrier
  capacity. Calibrate topology, edge times, and movement granularity before
  using these values as thesis results.
- Current decision: retain as a structural experiment; do not merge into the
  base environment until the spatial assumptions are reviewed.

### `experiment/rl-target-selection`

- Adds the third masked decision level: parking spot, launch runway, or mobile
  service vehicle, conditioned on operation type and aircraft.
- Adds target/vehicle set encoding, shortest-processing-time and earliest
  reachable-target priors, BC teachers, DAgger, and potential-based readiness
  reward redistribution.
- Fixes selected-runway deadline validation, wave accounting, and clearance
  wake-up events before training.
- Fixed `seed=10007`, 60-minute, 12-wave result: the final BC+PPO checkpoint
  completes 120 launches versus CP-SAT at 119.
- This one-launch advantage uses a constrained elite trajectory from the same
  scenario and is evidence of representational capability, not multi-seed
  generalization or statistical superiority.
- Reproduction: `scripts/run_rl_target_experiment.sh`; details:
  `doc/rl_target_selection_experiment.md`.
- Follow-up work replaces the fixed aircraft-rank coefficient with a
  state- and operation-conditioned learned gate. Training, validation, and
  frozen-test seeds are disjoint.
- On frozen seeds `40001-40020`, learned-gate BC+PPO averages 120.20 launches,
  versus 119.75 for fixed-prior RL and 119.50 for CP-SAT. The paired learned
  policy versus CP-SAT comparison is 14 wins, 5 ties, and 1 loss.
- The formal Heuristic averages 117.60; learned-gate RL wins all 20 paired
  seeds by 2.60 launches on average.
- The `elite_cp` teacher averages 119.80 on the same seeds. Learned-gate RL is
  higher by 0.40 on average but the paired sign test is not significant
  (`p=0.146`); do not claim that the policy has surpassed its teacher.
- Reproduction: `scripts/run_rl_generalization_experiment.sh`; details:
  `doc/rl_generalization_iteration.md`.
- Multi-load training supports disjoint training and validation interval sets.
  The retained checkpoint adapts the fixed-load PPO policy with multi-load BC;
  it does not add PPO updates. On 25 frozen scenarios spanning unseen
  47.5/52.5/57.5/62.5/67.5-minute intervals, `rl_multiload_bc` averages 2.92
  more launches than Heuristic (24 wins, 1 tie) and matches CP-SAT within
  0.24 launches on average.

### `experiment/cbs-path-planning`

- Adds optimal sum-of-costs CBS for fixed batches on top of the existing
  space-time A* low-level search.
- `--cbs-replan` jointly replans service-vehicle and aircraft-towing routes
  dispatched at the same simulation time. Tractor approach routes remain
  fixed.
- New routes replace prioritized reservations only when batch SOC strictly
  decreases; timeout or infeasibility restores the original routes.
- The deterministic deck bottleneck benchmark improves SOC from 11 to 10 and
  makespan from 8 to 6.
- The retained RL checkpoints show no strict batch-SOC improvement in the
  tested 60-minute and 47.5-minute scenarios, so do not claim a sortie gain
  from CBS.

New constraint experiments must be isolated on their own branch, compared
against a matched control, and justified by either the supplied requirements or
a clearly documented physical assumption.

## Remote execution

- GPU host: `xueyf@112.125.89.153`, SSH port `6000`.
- Connect with `ssh -p 6000 xueyf@112.125.89.153`.
- Use SSH public-key authentication. Never store passwords, private keys, or
  access tokens in this repository or its documentation.
- Remote checkout: `/home/xueyf/carrier-aircraft-scheduling`.
- RL optimization work runs from
  `experiment/disruption-domain-randomization`.
- The host provides eight NVIDIA RTX A6000 48 GB GPUs. Check current GPU
  utilization before selecting a device; do not terminate or interfere with
  other users' processes.
- Use an isolated project environment under the remote checkout and preserve
  generated checkpoints, logs, and benchmark outputs with their exact commit
  and seed metadata.

## Development rules

- Preserve the single business objective: maximize completed launches within
  the fixed horizon.
- Keep optimization metrics separate from solver-specific training rewards.
- Default RL training uses heuristic behavior cloning followed by PPO. Formal
  experiments must use disjoint `--train-seeds` and `--validation-seeds` and
  label the result `rl_bc_ppo`, not pure PPO.
- Pure PPO experiments must pass `--bc-episodes 0` and be reported separately.
- Do not add a requirement merely because it appears in `doc/前置约束.md`;
  first resolve its ambiguity and update the mathematical model.
- Keep resource acquisition and release balanced on every event path.
- Never allow action masks and action validation to disagree.
- Preserve deterministic replay for a fixed environment seed and solver seed.
- The fixed baseline uses a 60-minute wave interval, 12 waves, one run, and
  environment seed `10007` for every solver. PPO training uses seed `7`.
- Keep those defaults centralized in `scripts/evaluation_defaults.py`. Treat
  multi-seed robustness studies as separate experiments.
- Report at least completed launches, missed sorties, simulation completion,
  and resource feasibility.
- Update this file, the mathematical model, and user-facing documentation when
  the base problem definition changes.
