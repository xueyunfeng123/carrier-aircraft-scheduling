"""Collect counterfactual oracle labels for repair-control training."""

from __future__ import annotations

import argparse
import copy
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG
from env.disruptions import PROFILE_SETTINGS
from rl.repair_control import (
    CONTROL_BUDGETS_MS,
    CONTROL_FEATURE_NAMES,
    CONTROL_NEIGHBORHOODS,
    RuleBasedRepairController,
    encode_repair_control_state,
)
from rl.checkpoint import load_checkpoint
from scripts.evaluation_protocol import (
    PHASE_SEEDS,
    canonical_json_sha256,
    reject_frozen_final_seeds,
    validate_phase_seeds,
    write_csv,
)
from solution.cp_sat_repair_model import RepairControl
from solution.rl_solver import RLSolver
from solution.rolling_repair_solver import RollingRepairSolver


@dataclass(frozen=True)
class OracleOutcome:
    control: RepairControl
    sorties: int
    missed: int
    deadline_misses: int
    runtime_ms: float

    @property
    def quality_key(self) -> tuple[float, ...]:
        return (
            float(self.sorties),
            -float(self.missed),
            -float(self.deadline_misses),
            -float(self.runtime_ms),
            -float(self.control.trigger),
            -float(self.control.budget_ms),
            -float(self.control.neighborhood_size),
        )


class ForcedThenRuleController:
    def __init__(
        self,
        forced: RepairControl,
        reference_budget_ms: float,
        reference_neighborhood: int,
    ):
        self.forced = forced
        self.used_forced = False
        self.reference = RuleBasedRepairController(
            max_budget_ms=reference_budget_ms,
            max_neighborhood_size=reference_neighborhood,
        )

    def __call__(
        self,
        env: CarrierAircraftSchedulingEnv,
    ) -> RepairControl:
        if not self.used_forced:
            self.used_forced = True
            return self.forced
        return self.reference(env)


class SnapshotCollectorController:
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.reference = RuleBasedRepairController()
        self.snapshots: List[CarrierAircraftSchedulingEnv] = []
        self.seen = set()

    def __call__(
        self,
        env: CarrierAircraftSchedulingEnv,
    ) -> RepairControl:
        features = encode_repair_control_state(env)
        snapshot_key = canonical_json_sha256(
            {
                "time": env.time,
                "wave": env.current_wave_index,
                "event_sequence": env.event_sequence,
                "features": features,
                "mask": env.get_action_mask(),
            }
        )
        if (
            len(self.snapshots) < self.max_samples
            and snapshot_key not in self.seen
        ):
            self.seen.add(snapshot_key)
            self.snapshots.append(copy.deepcopy(env))
        return self.reference(env)


def candidate_controls() -> List[RepairControl]:
    controls = [
        RepairControl(
            trigger=False,
            scope="two_waves",
            budget_ms=10.0,
            neighborhood_size=8,
            horizon_waves=2,
        )
    ]
    controls.extend(
        RepairControl(
            trigger=True,
            scope="two_waves",
            budget_ms=budget_ms,
            neighborhood_size=neighborhood,
            horizon_waves=2,
        )
        for budget_ms in CONTROL_BUDGETS_MS
        for neighborhood in CONTROL_NEIGHBORHOODS
    )
    return controls


def collect_snapshots(
    config: Mapping[str, Any],
    seed: int,
    checkpoint: str,
    device: str,
    max_samples: int,
    max_steps: int,
) -> List[CarrierAircraftSchedulingEnv]:
    env = CarrierAircraftSchedulingEnv(dict(config))
    env.reset(seed=seed)
    actor = RLSolver(
        env,
        checkpoint=checkpoint,
        device=device,
        deterministic=True,
    )
    collector = SnapshotCollectorController(max_samples)
    solver = RollingRepairSolver(
        env,
        budget_ms=50.0,
        neighborhood_size=16,
        horizon_waves=2,
        scope="two_waves",
        control_provider=collector,
        guidance_provider=lambda _: actor.action_pair_scores(),
        fallback_solver=actor,
    )
    steps = 0
    while (
        not env.done
        and steps < max_steps
        and len(collector.snapshots) < max_samples
    ):
        env.step(solver.choose_action())
        steps += 1
    return collector.snapshots


def evaluate_control(
    snapshot: CarrierAircraftSchedulingEnv,
    control: RepairControl,
    checkpoint: str,
    device: str,
    max_steps: int,
) -> OracleOutcome:
    env = copy.deepcopy(snapshot)
    initial_sorties = env.get_evaluation_metrics()[
        "total_sorties_completed"
    ]
    initial_missed = env.get_evaluation_metrics()[
        "total_missed_sorties"
    ]
    target_wave = env.current_wave_index + 2
    actor = RLSolver(
        env,
        checkpoint=checkpoint,
        device=device,
        deterministic=True,
    )
    controller = ForcedThenRuleController(
        control,
        reference_budget_ms=50.0,
        reference_neighborhood=16,
    )
    solver = RollingRepairSolver(
        env,
        budget_ms=control.budget_ms,
        neighborhood_size=control.neighborhood_size,
        horizon_waves=2,
        scope=control.scope,
        control_provider=controller,
        guidance_provider=lambda _: actor.action_pair_scores(),
        fallback_solver=actor,
    )
    started = time.perf_counter()
    steps = 0
    while (
        not env.done
        and env.current_wave_index < target_wave
        and steps < max_steps
    ):
        env.step(solver.choose_action())
        steps += 1
    runtime_ms = (time.perf_counter() - started) * 1000.0
    metrics = env.get_evaluation_metrics()
    telemetry = solver.get_telemetry()
    return OracleOutcome(
        control=control,
        sorties=(
            int(metrics["total_sorties_completed"])
            - int(initial_sorties)
        ),
        missed=(
            int(metrics["total_missed_sorties"])
            - int(initial_missed)
        ),
        deadline_misses=int(telemetry["deadline_misses"]),
        runtime_ms=runtime_ms,
    )


def select_oracle(
    outcomes: Sequence[OracleOutcome],
) -> tuple[OracleOutcome, float, int]:
    if not outcomes:
        raise ValueError("oracle selection requires outcomes")
    ordered = sorted(
        outcomes,
        key=lambda outcome: outcome.quality_key,
        reverse=True,
    )
    best = ordered[0]
    best_primary = (
        best.sorties,
        -best.missed,
        -best.deadline_misses,
    )
    ties = sum(
        (
            outcome.sorties,
            -outcome.missed,
            -outcome.deadline_misses,
        )
        == best_primary
        for outcome in outcomes
    )
    runner_up = ordered[1] if len(ordered) > 1 else best
    margin = float(best.sorties - runner_up.sorties)
    if margin == 0.0:
        margin = float(runner_up.missed - best.missed)
    return best, margin, ties


def build_oracle_row(
    snapshot: CarrierAircraftSchedulingEnv,
    scenario_seed: int,
    sample_index: int,
    outcomes: Sequence[OracleOutcome],
) -> Dict[str, Any]:
    best, margin, ties = select_oracle(outcomes)
    features = dict(
        zip(
            CONTROL_FEATURE_NAMES,
            encode_repair_control_state(snapshot),
        )
    )
    return {
        "schema_version": 1,
        "sample_id": (
            f"{scenario_seed}:{snapshot.time:.9f}:"
            f"{snapshot.event_sequence}:{sample_index}"
        ),
        "scenario_seed": scenario_seed,
        "profile": snapshot.config["disruption_profile"],
        "wave_interval": snapshot.wave_interval,
        "snapshot_time": snapshot.time,
        "wave_index": snapshot.current_wave_index,
        "event_sequence": snapshot.event_sequence,
        "tape_version": snapshot.config[
            "scenario_tape_version"
        ],
        "tape_sha256": (
            snapshot.scenario_tape.sha256
            if snapshot.scenario_tape is not None
            else ""
        ),
        **features,
        "trigger_label": int(best.control.trigger),
        "scope_label": best.control.scope,
        "budget_ms_label": best.control.budget_ms,
        "neighborhood_label": best.control.neighborhood_size,
        "oracle_sorties": best.sorties,
        "oracle_missed": best.missed,
        "oracle_deadline_misses": best.deadline_misses,
        "oracle_runtime_ms": best.runtime_ms,
        "runner_up_margin": margin,
        "tie_count": ties,
        "sample_weight": 1.0 if margin > 0.0 else 0.25,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("train", "selection", "calibration"),
        default="train",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_SETTINGS),
        default=[
            "compound_light",
            "compound_medium",
            "compound_heavy",
        ],
    )
    parser.add_argument(
        "--intervals",
        type=float,
        nargs="+",
        default=[47.5, 57.5, 67.5],
    )
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--samples-per-scenario", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/rl_multiload_bc.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="outputs/repair_control_oracle.csv",
    )
    parser.add_argument(
        "--disable-spatial-graph",
        action="store_true",
    )
    args = parser.parse_args()

    purpose = (
        "training"
        if args.phase == "train"
        else "selection"
    )
    seeds = validate_phase_seeds(
        args.phase,
        args.seeds or PHASE_SEEDS[args.phase],
        purpose=purpose,
    )
    if not Path(args.checkpoint).is_file():
        parser.error(
            f"RL checkpoint does not exist: {args.checkpoint}"
        )
    checkpoint_extra = load_checkpoint(
        args.checkpoint,
        device="cpu",
    ).get("extra", {})
    reject_frozen_final_seeds(
        [
            *checkpoint_extra.get("training_seeds", []),
            *checkpoint_extra.get("validation_seeds", []),
        ],
        "oracle actor development",
    )
    if args.samples_per_scenario < 1:
        parser.error("--samples-per-scenario must be positive")

    rows: List[Dict[str, Any]] = []
    controls = candidate_controls()
    for profile in args.profiles:
        for interval in args.intervals:
            config = dict(DEFAULT_CONFIG)
            config.update(
                {
                    "disruption_profile": profile,
                    "wave_interval": float(interval),
                    "simulation_duration": (
                        float(interval) * args.waves
                    ),
                    "scenario_tape_enabled": True,
                    "spatial_graph_enabled": (
                        not args.disable_spatial_graph
                    ),
                }
            )
            for seed in seeds:
                snapshots = collect_snapshots(
                    config,
                    seed,
                    args.checkpoint,
                    args.device,
                    args.samples_per_scenario,
                    args.max_steps,
                )
                for sample_index, snapshot in enumerate(snapshots):
                    outcomes = [
                        evaluate_control(
                            snapshot,
                            control,
                            args.checkpoint,
                            args.device,
                            args.max_steps,
                        )
                        for control in controls
                    ]
                    rows.append(
                        build_oracle_row(
                            snapshot,
                            seed,
                            sample_index,
                            outcomes,
                        )
                    )
                print(
                    f"oracle,{profile},{interval:g},{seed},"
                    f"samples,{len(snapshots)}",
                    flush=True,
                )
    write_csv(Path(args.output), rows)
    print(f"oracle_csv_written: {args.output}")


if __name__ == "__main__":
    main()
