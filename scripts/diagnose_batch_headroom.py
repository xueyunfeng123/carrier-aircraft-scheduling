"""Estimate batch-dispatch headroom with trajectory-level policy search.

This script is intentionally diagnostic, not a fair online solver. It runs
many randomized, non-anticipatory policies for each development seed and then
selects the best completed trajectory using its terminal score. That
after-the-fact selection is clairvoyant and may only be used as an optimistic
headroom estimate or as a source of training demonstrations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import multiprocessing
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG
from scripts.solve import run_episode
from solution.heuristic_solver import (
    ACTION_LAUNCH,
    ACTION_RECOVERY,
    WaveHeuristicSolver,
)


FROZEN_SEED_START = 70001
FROZEN_SEED_END = 70050


@dataclass(frozen=True)
class SearchPolicy:
    launch_mode: str
    recovery_mode: str
    precedence: str
    launch_top_k: int
    recovery_top_k: int
    random_targets: bool


class RandomizedRouteSolver:
    """Perturb strong heuristic choices without observing future randomness."""

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        policy: SearchPolicy,
        policy_seed: int,
    ):
        self.env = env
        self.policy = policy
        self.rng = random.Random(policy_seed)
        self.base = WaveHeuristicSolver(env)

    def choose_action(self) -> Dict[str, Any] | None:
        mask = self.env.get_action_mask()
        high_mask = mask["high_level"]
        if not any(high_mask):
            return None

        launch_enabled = bool(high_mask[ACTION_LAUNCH])
        recovery_enabled = bool(high_mask[ACTION_RECOVERY])
        if launch_enabled and recovery_enabled:
            high_level = self._movement_precedence()
        elif recovery_enabled:
            high_level = ACTION_RECOVERY
        elif launch_enabled:
            high_level = ACTION_LAUNCH
        else:
            action = self.base.choose_action()
            return self._retarget(action)

        low_mask = mask["low_level_by_high"][high_level]
        aircraft_ids = self.base._candidate_ids(low_mask)
        if high_level == ACTION_LAUNCH:
            ranked = sorted(aircraft_ids, key=self._launch_key)
            top_k = self.policy.launch_top_k
        else:
            ranked = sorted(aircraft_ids, key=self._recovery_key)
            top_k = self.policy.recovery_top_k
        aircraft_id = self.rng.choice(ranked[: min(top_k, len(ranked))])
        return self._complete(high_level, aircraft_id)

    def _movement_precedence(self) -> int:
        if self.policy.precedence == "launch":
            return ACTION_LAUNCH
        if self.policy.precedence == "deadline":
            deadline = min(
                (self.env.current_wave_index + 1) * self.env.wave_interval,
                self.env.simulation_duration,
            )
            remaining = deadline - self.env.time
            if remaining <= 0.25 * self.env.wave_interval:
                return ACTION_LAUNCH
        return ACTION_RECOVERY

    def _launch_key(self, aircraft_id: int) -> Tuple[Any, ...]:
        aircraft = self.env.aircraft[aircraft_id]
        ready_time = (
            aircraft.launch_ready
            if aircraft.launch_ready is not None
            else self.env.time
        )
        duration = self.env._expected_launch_duration(aircraft_id)
        if self.policy.launch_mode == "fair":
            return (
                aircraft.sorties_completed,
                -aircraft.missed_sorties,
                duration,
                ready_time,
                aircraft_id,
            )
        if self.policy.launch_mode == "ready":
            return self.base._launch_priority(aircraft_id)
        if self.policy.launch_mode == "spot":
            return (
                self.env._spot_transfer_time(aircraft.spot_id),
                duration,
                ready_time,
                aircraft_id,
            )
        return (duration, ready_time, aircraft.sorties_completed, aircraft_id)

    def _recovery_key(self, aircraft_id: int) -> Tuple[Any, ...]:
        aircraft = self.env.aircraft[aircraft_id]
        duration = self.env._expected_recovery_duration(aircraft_id)
        if self.policy.recovery_mode == "fair":
            return (
                aircraft.sorties_completed,
                -aircraft.missed_sorties,
                duration,
                aircraft_id,
            )
        if self.policy.recovery_mode == "reverse":
            return (-duration, aircraft.sorties_completed, aircraft_id)
        return self.base._recovery_priority(aircraft_id)

    def _complete(self, high_level: int, aircraft_id: int) -> Dict[str, Any]:
        return self.env.complete_action(
            {
                "high_level": high_level,
                "aircraft_id": aircraft_id,
            },
            rng=self.rng if self.policy.random_targets else None,
        )

    def _retarget(
        self,
        action: Dict[str, Any] | None,
    ) -> Dict[str, Any] | None:
        if action is None or not self.policy.random_targets:
            return action
        return self._complete(
            int(action["high_level"]),
            int(action["aircraft_id"]),
        )


def policy_for_candidate(candidate_id: int) -> SearchPolicy:
    launch_modes = ("duration", "fair", "ready", "spot")
    recovery_modes = ("duration", "fair", "reverse")
    precedences = ("recovery", "launch", "deadline")
    top_k_values = (1, 2, 3, 5)
    return SearchPolicy(
        launch_mode=launch_modes[candidate_id % len(launch_modes)],
        recovery_mode=recovery_modes[
            (candidate_id // len(launch_modes)) % len(recovery_modes)
        ],
        precedence=precedences[
            (candidate_id // (len(launch_modes) * len(recovery_modes)))
            % len(precedences)
        ],
        launch_top_k=top_k_values[
            (candidate_id // 3) % len(top_k_values)
        ],
        recovery_top_k=top_k_values[
            (candidate_id // 7) % len(top_k_values)
        ],
        random_targets=(candidate_id % 16 == 15),
    )


def run_candidate(
    job: Tuple[int, int, Dict[str, Any], int, int],
) -> Dict[str, Any]:
    seed, candidate_id, config, policy_seed, max_steps = job
    env = CarrierAircraftSchedulingEnv(config)
    env.reset(seed=seed)
    policy = policy_for_candidate(candidate_id)
    solver = RandomizedRouteSolver(
        env,
        policy,
        policy_seed + seed * 100_003 + candidate_id,
    )
    steps = 0
    start = time.perf_counter()
    while not env.done and steps < max_steps:
        env.step(solver.choose_action())
        steps += 1
    if not env.done:
        raise RuntimeError(
            f"candidate {candidate_id} on seed {seed} exceeded {max_steps} steps"
        )
    metrics = env.get_evaluation_metrics()
    return {
        "seed": seed,
        "candidate_id": candidate_id,
        "total_sorties_completed": metrics["total_sorties_completed"],
        "total_missed_sorties": metrics["total_missed_sorties"],
        "wave_sorties": ";".join(
            str(record["sorties_completed"])
            for record in env.wave_records
        ),
        "runtime_seconds": time.perf_counter() - start,
        **policy.__dict__,
    }


def validate_development_seeds(seeds: Sequence[int]) -> None:
    frozen = [
        seed
        for seed in seeds
        if FROZEN_SEED_START <= seed <= FROZEN_SEED_END
    ]
    if frozen:
        raise ValueError(
            "headroom search cannot use frozen evaluation seeds: "
            + ", ".join(str(seed) for seed in frozen)
        )


def write_rows(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(66001, 66006)))
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=900_000)
    parser.add_argument("--wave-interval", type=float, default=47.5)
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument("--target-margin", type=float, default=5.0)
    parser.add_argument("--cp-sat-max-time", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument(
        "--output",
        default="outputs/batch_headroom_diagnostic.csv",
    )
    parser.add_argument("--skip-cp-sat", action="store_true")
    args = parser.parse_args()

    validate_development_seeds(args.seeds)
    if args.candidates < 1:
        parser.error("--candidates must be positive")

    config = dict(DEFAULT_CONFIG)
    config["wave_interval"] = float(args.wave_interval)
    config["simulation_duration"] = float(args.wave_interval) * args.waves

    baselines: Dict[int, int] = {}
    if not args.skip_cp_sat:
        for seed in args.seeds:
            result = run_episode(
                "cp_sat",
                config,
                seed=seed,
                max_steps=args.max_steps,
                solver_options={"max_time_seconds": args.cp_sat_max_time},
            )
            baselines[seed] = int(result["total_sorties_completed"])

    jobs = [
        (
            seed,
            candidate_id,
            config,
            args.policy_seed,
            args.max_steps,
        )
        for seed in args.seeds
        for candidate_id in range(args.candidates)
    ]
    workers = args.workers or min(len(jobs), multiprocessing.cpu_count())
    context = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as executor:
        rows = list(executor.map(run_candidate, jobs))
    rows.sort(key=lambda item: (item["seed"], item["candidate_id"]))
    write_rows(args.output, rows)

    oracle_scores = []
    baseline_scores = []
    print(
        "seed,cp_sat,best_candidate,best_total,delta_to_cp_sat,"
        "target_reached,wave_sorties"
    )
    for seed in args.seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        best = max(
            seed_rows,
            key=lambda item: (
                item["total_sorties_completed"],
                -item["candidate_id"],
            ),
        )
        oracle_score = int(best["total_sorties_completed"])
        oracle_scores.append(oracle_score)
        baseline = baselines.get(seed)
        if baseline is not None:
            baseline_scores.append(baseline)
            delta = oracle_score - baseline
            reached = delta >= args.target_margin
        else:
            delta = ""
            reached = ""
        print(
            f"{seed},{baseline if baseline is not None else ''},"
            f"{best['candidate_id']},{oracle_score},{delta},{reached},"
            f"{best['wave_sorties']}"
        )

    print(f"clairvoyant_oracle_mean: {statistics.mean(oracle_scores):.2f}")
    if baseline_scores:
        baseline_mean = statistics.mean(baseline_scores)
        delta_mean = statistics.mean(
            oracle - baseline
            for oracle, baseline in zip(oracle_scores, baseline_scores)
        )
        print(f"cp_sat_mean: {baseline_mean:.2f}")
        print(f"clairvoyant_delta_mean: {delta_mean:.2f}")
        print(
            "mean_target_reached: "
            f"{delta_mean >= args.target_margin}"
        )
    print(f"detail_output: {args.output}")
    print("diagnostic_clairvoyant: true")


if __name__ == "__main__":
    main()
