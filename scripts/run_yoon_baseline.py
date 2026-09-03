"""Run the executable structural replication of Yoon et al. (2023)."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from env.scenario import YOON_2023_CASES
from env.yoon_sortie_env import (
    YoonReplicationAssumptions,
    YoonSortieGenerationEnv,
)
from scripts.solve import write_dict_csv


PUBLISHED_SUCCESS_RATES = {
    "case_1_1": 0.747,
    "case_1_2": 0.500,
    "case_2_1": 0.817,
    "case_2_2": 0.578,
    "case_3_1": 0.854,
    "case_3_2": 0.870,
    "case_4_1": 0.855,
    "case_4_2": 0.868,
}


def run_replication(
    case_id: str,
    assumptions: YoonReplicationAssumptions,
    seed: int,
    validate_each_event: bool,
) -> Dict[str, Any]:
    env = YoonSortieGenerationEnv(
        case_id,
        assumptions,
        record_event_log=False,
    )
    return env.run(
        seed=seed,
        validate_each_event=validate_each_event,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(YOON_2023_CASES),
        default=sorted(YOON_2023_CASES),
    )
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument("--preparation-lead", type=float, default=50.0)
    parser.add_argument("--movement-minutes-per-unit", type=float, default=1.0)
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument("--validate-every-event", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--event-log", default="")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.event_log and (args.runs != 1 or len(args.cases) != 1):
        parser.error("--event-log requires exactly one case and --runs 1")

    assumptions = YoonReplicationAssumptions(
        preparation_lead_minutes=args.preparation_lead,
        movement_minutes_per_graph_unit=args.movement_minutes_per_unit,
        horizon_minutes_override=args.horizon,
    )
    rows: List[Dict[str, Any]] = []
    print(
        "case,runs,published_rate,replication_mean,ci95_low,ci95_high,"
        "absolute_gap,mean_sorties_per_hour,mean_cancelled,mean_maintenance"
    )
    last_env = None
    metrics_by_case: Dict[str, List[Dict[str, Any]]] = {
        case_id: [] for case_id in args.cases
    }
    seeds = [args.seed + run_id for run_id in range(args.runs)]
    if args.event_log:
        case_id = args.cases[0]
        env = YoonSortieGenerationEnv(case_id, assumptions)
        metrics_by_case[case_id] = [
            env.run(
                seed=seeds[0],
                validate_each_event=args.validate_every_event,
            )
        ]
        last_env = env
    elif args.workers == 1:
        for case_id in args.cases:
            metrics_by_case[case_id] = [
                run_replication(
                    case_id,
                    assumptions,
                    seed,
                    args.validate_every_event,
                )
                for seed in seeds
            ]
    else:
        task_cases = [
            case_id
            for case_id in args.cases
            for _ in seeds
        ]
        task_seeds = seeds * len(args.cases)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            task_results = executor.map(
                run_replication,
                task_cases,
                [assumptions] * len(task_cases),
                task_seeds,
                [args.validate_every_event] * len(task_cases),
            )
            for case_id, metrics in zip(task_cases, task_results):
                metrics_by_case[case_id].append(metrics)

    for case_id in args.cases:
        success_rates = []
        sortie_rates = []
        cancelled = []
        maintenance = []
        for metrics in metrics_by_case[case_id]:
            success_rates.append(metrics["fixed_mission_success_rate"])
            sortie_rates.append(metrics["sortie_generation_rate_per_hour"])
            cancelled.append(metrics["cancelled_fixed_missions"])
            maintenance.append(metrics["maintenance_completed"])

        mean_success = statistics.mean(success_rates)
        std_success = statistics.stdev(success_rates) if args.runs > 1 else 0.0
        half_width = 1.96 * std_success / math.sqrt(args.runs)
        published = PUBLISHED_SUCCESS_RATES[case_id]
        row = {
            "case": case_id,
            "runs": args.runs,
            "seed_start": args.seed,
            "fidelity": "structural_replication_with_explicit_assumptions",
            "published_success_rate": published,
            "replication_mean_success_rate": mean_success,
            "replication_std_success_rate": std_success,
            "replication_ci95_low": max(0.0, mean_success - half_width),
            "replication_ci95_high": min(1.0, mean_success + half_width),
            "absolute_gap": abs(mean_success - published),
            "mean_sortie_generation_rate_per_hour": statistics.mean(sortie_rates),
            "mean_cancelled_fixed_missions": statistics.mean(cancelled),
            "mean_maintenance_completed": statistics.mean(maintenance),
            "preparation_lead_minutes": assumptions.preparation_lead_minutes,
            "movement_minutes_per_graph_unit": (
                assumptions.movement_minutes_per_graph_unit
            ),
        }
        rows.append(row)
        print(
            f"{case_id},{args.runs},{published:.4f},{mean_success:.4f},"
            f"{row['replication_ci95_low']:.4f},"
            f"{row['replication_ci95_high']:.4f},"
            f"{row['absolute_gap']:.4f},"
            f"{row['mean_sortie_generation_rate_per_hour']:.4f},"
            f"{row['mean_cancelled_fixed_missions']:.2f},"
            f"{row['mean_maintenance_completed']:.2f}"
        )

    if args.output:
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
        print(f"results_written: {output}")
    if args.event_log and last_env is not None:
        write_dict_csv(args.event_log, last_env.get_event_log())
        print(f"event_log_written: {args.event_log}")


if __name__ == "__main__":
    main()
