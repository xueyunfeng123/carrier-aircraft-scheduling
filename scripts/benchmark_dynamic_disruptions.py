"""Paired headroom benchmark across dynamic disruption profiles."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
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
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--cp-sat-max-time", type=float, default=0.05)
    parser.add_argument(
        "--fixed-checkpoint",
        default="checkpoints/rl_learned_prior_ppo.pt",
    )
    parser.add_argument(
        "--multiload-checkpoint",
        default="checkpoints/rl_multiload_bc.pt",
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
    ):
        if not Path(checkpoint).is_file():
            parser.error(f"RL checkpoint does not exist: {checkpoint}")

    solver_specs = (
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
    )
    detail_rows: List[Dict[str, Any]] = []
    for profile in args.profiles:
        config = dict(DEFAULT_CONFIG)
        config["wave_interval"] = float(args.wave_interval)
        config["simulation_duration"] = (
            float(args.wave_interval) * args.waves
        )
        config["spatial_graph_enabled"] = (
            not args.disable_spatial_graph
        )
        config["disruption_profile"] = profile

        for run_id in range(args.runs):
            seed = args.seed + run_id
            for label, solver_name, options in solver_specs:
                started_at = time.perf_counter()
                result = run_episode(
                    solver_name,
                    config,
                    seed=seed,
                    max_steps=args.max_steps,
                    solver_options=options,
                )
                runtime = time.perf_counter() - started_at
                wave_sorties = [
                    int(record["sorties_completed"])
                    for record in result["wave_records"]
                ]
                detail_rows.append(
                    {
                        "profile": profile,
                        "solver": label,
                        "seed": seed,
                        "wave_interval": args.wave_interval,
                        "waves": args.waves,
                        "total_sorties_completed": result[
                            "total_sorties_completed"
                        ],
                        "total_missed_sorties": result[
                            "total_missed_sorties"
                        ],
                        "worst_wave": min(wave_sorties),
                        "runtime_seconds": runtime,
                        "disruptions_scheduled": result[
                            "disruptions"
                        ]["scheduled"],
                        "disruptions_started": result[
                            "disruptions"
                        ]["started"],
                        "wave_sorties": ";".join(
                            str(value) for value in wave_sorties
                        ),
                    }
                )
                print(
                    f"{profile},{label},{seed},"
                    f"{result['total_sorties_completed']},"
                    f"{result['total_missed_sorties']},"
                    f"{runtime:.3f}",
                    flush=True,
                )

    summary_rows = build_summary(detail_rows, args.profiles)
    write_csv(Path(args.detail_output), detail_rows)
    write_csv(Path(args.output), summary_rows)
    print(f"benchmark_csv_written: {args.output}")
    print(f"benchmark_detail_csv_written: {args.detail_output}")


def build_summary(
    detail_rows: List[Dict[str, Any]],
    profiles: List[str],
) -> List[Dict[str, Any]]:
    labels = ("heuristic", "cp_sat", "rl_fixed", "rl_multiload")
    means: Dict[tuple[str, str], float] = {}
    for profile in profiles:
        for label in labels:
            totals = [
                int(row["total_sorties_completed"])
                for row in detail_rows
                if row["profile"] == profile
                and row["solver"] == label
            ]
            means[(profile, label)] = statistics.mean(totals)

    summary_rows: List[Dict[str, Any]] = []
    for profile in profiles:
        for label in labels:
            rows = [
                row
                for row in detail_rows
                if row["profile"] == profile
                and row["solver"] == label
            ]
            totals = [
                int(row["total_sorties_completed"])
                for row in rows
            ]
            missed = [
                int(row["total_missed_sorties"])
                for row in rows
            ]
            baseline = means.get(("none", label), means[(profile, label)])
            summary_rows.append(
                {
                    "profile": profile,
                    "solver": label,
                    "runs": len(rows),
                    "mean_total_sorties": statistics.mean(totals),
                    "std_total_sorties": (
                        statistics.stdev(totals)
                        if len(totals) > 1
                        else 0.0
                    ),
                    "mean_missed_sorties": statistics.mean(missed),
                    "mean_worst_wave": statistics.mean(
                        int(row["worst_wave"]) for row in rows
                    ),
                    "mean_runtime_seconds": statistics.mean(
                        float(row["runtime_seconds"]) for row in rows
                    ),
                    "delta_vs_none": (
                        means[(profile, label)] - baseline
                    ),
                    "delta_vs_heuristic": (
                        means[(profile, label)]
                        - means[(profile, "heuristic")]
                    ),
                    "delta_vs_cp_sat": (
                        means[(profile, label)]
                        - means[(profile, "cp_sat")]
                    ),
                }
            )
    return summary_rows


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
