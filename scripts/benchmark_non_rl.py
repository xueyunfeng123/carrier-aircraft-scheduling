"""Benchmark all non-RL solvers on complete multi-wave scenarios."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from env.config import DEFAULT_CONFIG
from scripts.evaluation_defaults import (
    DEFAULT_EVALUATION_RUNS,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVALUATION_WAVE_INTERVAL,
    DEFAULT_EVALUATION_WAVES,
)
from scripts.solve import run_episode


NON_RL_SOLVERS = ("random", "fifo", "spt", "edd", "heuristic", "sampled", "cp_sat")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intervals",
        type=float,
        nargs="+",
        default=[DEFAULT_EVALUATION_WAVE_INTERVAL],
    )
    parser.add_argument("--waves", type=int, default=DEFAULT_EVALUATION_WAVES)
    parser.add_argument("--runs", type=int, default=DEFAULT_EVALUATION_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--sampled-samples", type=int, default=30)
    parser.add_argument("--cp-sat-max-time", type=float, default=0.05)
    parser.add_argument("--rl-checkpoint", type=str, default="")
    parser.add_argument("--rl-device", type=str, default="cpu")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--output", default="outputs/non_rl_benchmark.csv")
    args = parser.parse_args()

    solver_names = NON_RL_SOLVERS
    if args.rl_checkpoint:
        if not Path(args.rl_checkpoint).is_file():
            parser.error(f"RL checkpoint does not exist: {args.rl_checkpoint}")
        solver_names += ("rl",)

    rows: List[Dict[str, Any]] = []
    print(
        "solver,interval,waves,mean_total,std_total,mean_per_wave,"
        "completion_rate,mean_worst_wave,mean_missed,mean_runtime_s"
    )
    for interval in args.intervals:
        config = dict(DEFAULT_CONFIG)
        config["wave_interval"] = float(interval)
        config["simulation_duration"] = float(interval) * args.waves
        capacity = args.waves * int(config["group_size"])

        for solver_name in solver_names:
            options: Dict[str, Any] = {}
            if solver_name == "sampled":
                options["samples"] = args.sampled_samples
            elif solver_name == "cp_sat":
                options["max_time_seconds"] = args.cp_sat_max_time
            elif solver_name == "rl":
                options = {
                    "checkpoint": args.rl_checkpoint,
                    "device": args.rl_device,
                    "deterministic": True,
                }

            totals: List[int] = []
            missed: List[int] = []
            worst_waves: List[int] = []
            runtimes: List[float] = []
            wave_runs: List[List[int]] = []

            for run_id in range(args.runs):
                start = time.perf_counter()
                result = run_episode(
                    solver_name,
                    config,
                    seed=args.seed + run_id,
                    max_steps=args.max_steps,
                    solver_options=options,
                )
                runtimes.append(time.perf_counter() - start)
                wave_counts = [
                    record["sorties_completed"]
                    for record in result["wave_records"]
                ]
                if len(wave_counts) != args.waves:
                    raise RuntimeError(
                        f"{solver_name}: expected {args.waves} waves, "
                        f"got {len(wave_counts)}"
                    )
                if sum(wave_counts) != result["total_sorties_completed"]:
                    raise RuntimeError(
                        f"{solver_name}: per-wave and total sorties disagree"
                    )
                totals.append(result["total_sorties_completed"])
                missed.append(result["total_missed_sorties"])
                worst_waves.append(min(wave_counts))
                wave_runs.append(wave_counts)

            wave_means = [
                statistics.mean(values)
                for values in zip(*wave_runs)
            ]
            row = {
                "solver": solver_name,
                "wave_interval": float(interval),
                "simulation_duration": config["simulation_duration"],
                "waves": args.waves,
                "runs": args.runs,
                "seed_start": args.seed,
                "mean_total_sorties": statistics.mean(totals),
                "std_total_sorties": statistics.stdev(totals) if args.runs > 1 else 0.0,
                "mean_sorties_per_wave": statistics.mean(totals) / args.waves,
                "completion_rate": statistics.mean(totals) / capacity,
                "mean_worst_wave": statistics.mean(worst_waves),
                "mean_missed_sorties": statistics.mean(missed),
                "mean_runtime_seconds": statistics.mean(runtimes),
                "wave_means": ";".join(f"{value:.2f}" for value in wave_means),
            }
            rows.append(row)
            print(
                f"{solver_name},{interval:g},{args.waves},"
                f"{row['mean_total_sorties']:.2f},"
                f"{row['std_total_sorties']:.2f},"
                f"{row['mean_sorties_per_wave']:.2f},"
                f"{row['completion_rate']:.4f},"
                f"{row['mean_worst_wave']:.2f},"
                f"{row['mean_missed_sorties']:.2f},"
                f"{row['mean_runtime_seconds']:.4f}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"benchmark_csv_written: {output}")


if __name__ == "__main__":
    main()
