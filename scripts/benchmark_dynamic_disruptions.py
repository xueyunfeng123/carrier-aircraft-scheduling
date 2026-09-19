"""Paired headroom benchmark across dynamic disruption profiles."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from env.config import DEFAULT_CONFIG
from scripts.solve import run_episode


PROFILES = ("none", "light", "medium", "heavy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILES,
        default=list(PROFILES),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=65001)
    parser.add_argument("--wave-interval", type=float, default=60.0)
    parser.add_argument(
        "--intervals",
        type=float,
        nargs="+",
        help="optional load matrix; defaults to --wave-interval",
    )
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel scenario processes",
    )
    parser.add_argument("--cp-sat-max-time", type=float, default=0.05)
    parser.add_argument(
        "--fixed-checkpoint",
        default="checkpoints/rl_learned_prior_ppo.pt",
    )
    parser.add_argument(
        "--multiload-checkpoint",
        default="checkpoints/rl_multiload_bc.pt",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        default="",
        help="optional disruption-randomized RL checkpoint",
    )
    parser.add_argument(
        "--candidate-label",
        default="rl_randomized",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--disable-spatial-graph", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/dynamic_disruption_headroom.csv",
    )
    parser.add_argument(
        "--detail-output",
        default="outputs/dynamic_disruption_headroom_detail.csv",
    )
    args = parser.parse_args()

    for checkpoint in (
        args.fixed_checkpoint,
        args.multiload_checkpoint,
        args.candidate_checkpoint,
    ):
        if checkpoint and not Path(checkpoint).is_file():
            parser.error(f"RL checkpoint does not exist: {checkpoint}")

    solver_specs = [
        ("heuristic", "heuristic", {}),
        (
            "cp_sat",
            "cp_sat",
            {"max_time_seconds": args.cp_sat_max_time},
        ),
        (
            "rl_fixed",
            "rl",
            {
                "checkpoint": args.fixed_checkpoint,
                "device": args.device,
                "deterministic": True,
            },
        ),
        (
            "rl_multiload",
            "rl",
            {
                "checkpoint": args.multiload_checkpoint,
                "device": args.device,
                "deterministic": True,
            },
        ),
    ]
    if args.candidate_checkpoint:
        solver_specs.append(
            (
                args.candidate_label,
                "rl",
                {
                    "checkpoint": args.candidate_checkpoint,
                    "device": args.device,
                    "deterministic": True,
                },
            )
        )
    intervals = list(
        dict.fromkeys(
            float(value)
            for value in (
                args.intervals
                if args.intervals
                else [args.wave_interval]
            )
        )
    )
    if any(interval <= 0.0 for interval in intervals):
        parser.error("wave intervals must be positive")

    tasks: List[Dict[str, Any]] = []
    for interval in intervals:
        for profile in args.profiles:
            config = dict(DEFAULT_CONFIG)
            config["wave_interval"] = interval
            config["simulation_duration"] = interval * args.waves
            config["spatial_graph_enabled"] = (
                not args.disable_spatial_graph
            )
            config["disruption_profile"] = profile

            for run_id in range(args.runs):
                seed = args.seed + run_id
                for label, solver_name, options in solver_specs:
                    tasks.append(
                        {
                            "profile": profile,
                            "solver": label,
                            "solver_name": solver_name,
                            "solver_options": options,
                            "seed": seed,
                            "wave_interval": interval,
                            "waves": args.waves,
                            "max_steps": args.max_steps,
                            "config": config,
                        }
                    )

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers == 1:
        detail_rows = [
            run_benchmark_case(task)
            for task in tasks
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            detail_rows = list(
                executor.map(run_benchmark_case, tasks)
            )
    for row in detail_rows:
        print(
            f"{row['profile']},{row['wave_interval']:g},"
            f"{row['solver']},{row['seed']},"
            f"{row['total_sorties_completed']},"
            f"{row['total_missed_sorties']},"
            f"{row['runtime_seconds']:.3f}",
            flush=True,
        )

    summary_rows = build_summary(detail_rows)
    write_csv(Path(args.detail_output), detail_rows)
    write_csv(Path(args.output), summary_rows)
    print(f"benchmark_csv_written: {args.output}")
    print(f"benchmark_detail_csv_written: {args.detail_output}")


def run_benchmark_case(task: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.perf_counter()
    result = run_episode(
        task["solver_name"],
        task["config"],
        seed=task["seed"],
        max_steps=task["max_steps"],
        solver_options=task["solver_options"],
    )
    runtime = time.perf_counter() - started_at
    wave_sorties = [
        int(record["sorties_completed"])
        for record in result["wave_records"]
    ]
    if len(wave_sorties) != task["waves"]:
        raise RuntimeError(
            f"{task['solver']}: expected {task['waves']} waves, "
            f"got {len(wave_sorties)}"
        )
    if sum(wave_sorties) != result["total_sorties_completed"]:
        raise RuntimeError(
            f"{task['solver']}: per-wave and total sorties disagree"
        )
    return {
        "profile": task["profile"],
        "solver": task["solver"],
        "seed": task["seed"],
        "wave_interval": task["wave_interval"],
        "waves": task["waves"],
        "total_sorties_completed": result[
            "total_sorties_completed"
        ],
        "total_missed_sorties": result["total_missed_sorties"],
        "worst_wave": min(wave_sorties),
        "runtime_seconds": runtime,
        "disruptions_scheduled": result["disruptions"][
            "scheduled"
        ],
        "disruptions_started": result["disruptions"]["started"],
        "wave_sorties": ";".join(
            str(value) for value in wave_sorties
        ),
    }


def build_summary(
    detail_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    labels = list(
        dict.fromkeys(str(row["solver"]) for row in detail_rows)
    )
    profiles = list(
        dict.fromkeys(str(row["profile"]) for row in detail_rows)
    )
    intervals = list(
        dict.fromkeys(
            float(row["wave_interval"])
            for row in detail_rows
        )
    )
    summary_rows: List[Dict[str, Any]] = []
    for profile in profiles:
        for interval in intervals:
            summary_rows.extend(
                summarize_group(
                    detail_rows,
                    labels,
                    scope="scenario",
                    profile=profile,
                    wave_interval=interval,
                )
            )
    for profile in profiles:
        summary_rows.extend(
            summarize_group(
                detail_rows,
                labels,
                scope="profile",
                profile=profile,
                wave_interval=None,
            )
        )
    for interval in intervals:
        summary_rows.extend(
            summarize_group(
                detail_rows,
                labels,
                scope="load",
                profile=None,
                wave_interval=interval,
            )
        )
    summary_rows.extend(
        summarize_group(
            detail_rows,
            labels,
            scope="overall",
            profile=None,
            wave_interval=None,
        )
    )
    return summary_rows


def summarize_group(
    detail_rows: List[Dict[str, Any]],
    labels: List[str],
    scope: str,
    profile: str | None,
    wave_interval: float | None,
) -> List[Dict[str, Any]]:
    selected = [
        row
        for row in detail_rows
        if (
            profile is None
            or str(row["profile"]) == profile
        )
        and (
            wave_interval is None
            or float(row["wave_interval"]) == wave_interval
        )
    ]
    means = {
        label: statistics.mean(
            int(row["total_sorties_completed"])
            for row in selected
            if row["solver"] == label
        )
        for label in labels
    }
    rows: List[Dict[str, Any]] = []
    for label in labels:
        solver_rows = [
            row
            for row in selected
            if row["solver"] == label
        ]
        totals = [
            int(row["total_sorties_completed"])
            for row in solver_rows
        ]
        scenario_means = [
            statistics.mean(
                int(row["total_sorties_completed"])
                for row in solver_rows
                if str(row["profile"]) == scenario_profile
                and float(row["wave_interval"]) == interval
            )
            for scenario_profile in dict.fromkeys(
                str(row["profile"]) for row in solver_rows
            )
            for interval in dict.fromkeys(
                float(row["wave_interval"])
                for row in solver_rows
                if str(row["profile"]) == scenario_profile
            )
        ]
        rows.append(
            {
                "scope": scope,
                "profile": profile if profile is not None else "all",
                "wave_interval": (
                    wave_interval
                    if wave_interval is not None
                    else "all"
                ),
                "solver": label,
                "runs": len(solver_rows),
                "mean_total_sorties": statistics.mean(totals),
                "std_total_sorties": (
                    statistics.stdev(totals)
                    if len(totals) > 1
                    else 0.0
                ),
                "worst_scenario_mean": min(scenario_means),
                "mean_missed_sorties": statistics.mean(
                    int(row["total_missed_sorties"])
                    for row in solver_rows
                ),
                "mean_worst_wave": statistics.mean(
                    int(row["worst_wave"])
                    for row in solver_rows
                ),
                "mean_runtime_seconds": statistics.mean(
                    float(row["runtime_seconds"])
                    for row in solver_rows
                ),
                "delta_vs_heuristic": (
                    means[label] - means["heuristic"]
                ),
                "delta_vs_cp_sat": (
                    means[label] - means["cp_sat"]
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
