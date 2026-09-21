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


NON_RL_SOLVERS = ("random", "fifo", "spt", "edd", "heuristic", "cp_sat")


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
    parser.add_argument("--cp-sat-max-time", type=float, default=0.05)
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=NON_RL_SOLVERS,
        default=list(NON_RL_SOLVERS),
    )
    parser.add_argument(
        "--disable-spatial-graph",
        action="store_true",
        help="run the matched control without deck routes or movement time",
    )
    parser.add_argument("--rl-checkpoint", type=str, default="")
    parser.add_argument("--rl-device", type=str, default="cpu")
    parser.add_argument("--rl-label", type=str, default="rl")
    parser.add_argument("--rl-low-rank-prior", type=float)
    parser.add_argument("--rl-target-rank-prior", type=float)
    parser.add_argument("--rl-low-rank-prior-scale", type=float, default=1.0)
    parser.add_argument("--rl-disable-low-rank-prior", action="store_true")
    parser.add_argument("--rl-value-rerank-top-k", type=int, default=1)
    parser.add_argument("--rl-value-rerank-seed", type=int, default=0)
    parser.add_argument("--cbs-replan", action="store_true")
    parser.add_argument(
        "--cbs-max-expanded-nodes",
        type=int,
        default=DEFAULT_CONFIG["cbs_max_expanded_nodes"],
    )
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--output", default="outputs/non_rl_benchmark.csv")
    parser.add_argument(
        "--detail-output",
        default="",
        help="optional per-solver, per-seed CSV for paired statistical analysis",
    )
    args = parser.parse_args()

    solver_names = tuple(args.solvers)
    if args.rl_checkpoint:
        if not Path(args.rl_checkpoint).is_file():
            parser.error(f"RL checkpoint does not exist: {args.rl_checkpoint}")
        solver_names += ("rl",)

    rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    print(
        "solver,interval,waves,mean_total,std_total,mean_per_wave,"
        "completion_rate,mean_worst_wave,mean_missed,mean_runtime_s"
    )
    for interval in args.intervals:
        config = dict(DEFAULT_CONFIG)
        config["wave_interval"] = float(interval)
        config["simulation_duration"] = float(interval) * args.waves
        config["spatial_graph_enabled"] = not args.disable_spatial_graph
        config["cbs_replan_enabled"] = args.cbs_replan
        config["cbs_max_expanded_nodes"] = (
            args.cbs_max_expanded_nodes
        )
        capacity = args.waves * int(config["group_size"])

        for solver_name in solver_names:
            options: Dict[str, Any] = {}
            if solver_name == "cp_sat":
                options["max_time_seconds"] = args.cp_sat_max_time
            elif solver_name == "rl":
                options = {
                    "checkpoint": args.rl_checkpoint,
                    "device": args.rl_device,
                    "deterministic": True,
                    "low_rank_prior": args.rl_low_rank_prior,
                    "target_rank_prior": args.rl_target_rank_prior,
                    "low_rank_prior_scale": (
                        args.rl_low_rank_prior_scale
                    ),
                    "disable_low_rank_prior": (
                        args.rl_disable_low_rank_prior
                    ),
                    "value_rerank_top_k": (
                        args.rl_value_rerank_top_k
                    ),
                    "value_rerank_seed": (
                        args.rl_value_rerank_seed
                    ),
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
                detail_rows.append(
                    {
                        "solver": (
                            args.rl_label
                            if solver_name == "rl"
                            else solver_name
                        ),
                        "seed": args.seed + run_id,
                        "wave_interval": float(interval),
                        "waves": args.waves,
                        "total_sorties_completed": result[
                            "total_sorties_completed"
                        ],
                        "total_missed_sorties": result[
                            "total_missed_sorties"
                        ],
                        "worst_wave": min(wave_counts),
                        "runtime_seconds": runtimes[-1],
                        "wave_sorties": ";".join(
                            str(value)
                            for value in wave_counts
                        ),
                    }
                )

            wave_means = [
                statistics.mean(values)
                for values in zip(*wave_runs)
            ]
            row = {
                "solver": args.rl_label if solver_name == "rl" else solver_name,
                "scenario_profile": config["scenario_profile"],
                "spatial_graph_enabled": config["spatial_graph_enabled"],
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
                f"{row['solver']},{interval:g},{args.waves},"
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
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"benchmark_csv_written: {output}")
    if args.detail_output:
        detail_output = Path(args.detail_output)
        detail_output.parent.mkdir(parents=True, exist_ok=True)
        with detail_output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(detail_rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"benchmark_detail_csv_written: {detail_output}")


if __name__ == "__main__":
    main()
