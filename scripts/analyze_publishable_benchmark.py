"""Analyze canonical paired benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from scripts.analyze_paired_benchmark import (
    exact_two_sided_sign_test,
)
from scripts.evaluation_protocol import (
    lower_tail_cvar,
    paired_bootstrap_mean_ci,
    upper_tail_cvar,
    write_csv,
)


NUMERIC_FIELDS = {
    "seed": int,
    "wave_interval": float,
    "budget_ms": float,
    "total_sorties_completed": int,
    "total_missed_sorties": int,
    "worst_wave_sorties": int,
    "runtime_seconds": float,
    "decisions": int,
    "solve_calls": int,
    "deadline_misses": int,
    "fallbacks": int,
    "latency_p95_ms": float,
    "latency_p99_ms": float,
    "latency_max_ms": float,
    "solve_latency_p95_ms": float,
    "solve_latency_max_ms": float,
    "recovery_time_mean": float,
    "recovery_time_max": float,
    "recovery_observed": int,
    "recovery_censored": int,
    "sortie_loss_area": float,
    "resilience_index": float,
}


def load_runs(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    required = {
        "manifest_sha256",
        "scenario_id",
        "solver",
        "seed",
        "profile",
        "wave_interval",
        "total_sorties_completed",
    }
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path} is missing columns: {sorted(missing)}"
                )
            for raw_row in reader:
                row: Dict[str, Any] = dict(raw_row)
                for field, converter in NUMERIC_FIELDS.items():
                    if field in row and row[field] != "":
                        row[field] = converter(row[field])
                key = (row["scenario_id"], row["solver"])
                if key in seen:
                    raise ValueError(
                        "duplicate run for scenario/solver: "
                        f"{key[0]}, {key[1]}"
                    )
                seen.add(key)
                rows.append(row)
    manifest_hashes = {
        str(row["manifest_sha256"]) for row in rows
    }
    if len(manifest_hashes) != 1:
        raise ValueError(
            "analysis inputs must share one manifest hash"
        )
    return rows


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    alpha: float,
) -> List[Dict[str, Any]]:
    solvers = list(dict.fromkeys(str(row["solver"]) for row in rows))
    profiles = list(
        dict.fromkeys(str(row["profile"]) for row in rows)
    )
    summary: List[Dict[str, Any]] = []
    for solver in solvers:
        for profile in [*profiles, "all"]:
            selected = [
                row
                for row in rows
                if row["solver"] == solver
                and (
                    profile == "all"
                    or row["profile"] == profile
                )
            ]
            if not selected:
                continue
            completed = [
                float(row["total_sorties_completed"])
                for row in selected
            ]
            solve_calls = sum(
                int(row.get("solve_calls", 0))
                for row in selected
            )
            deadline_misses = sum(
                int(row.get("deadline_misses", 0))
                for row in selected
            )
            observed_recovery_times = [
                float(row["recovery_time_mean"])
                for row in selected
                if math.isfinite(
                    float(row.get("recovery_time_mean", "nan"))
                )
            ]
            losses = [
                float(row.get("sortie_loss_area", 0.0))
                for row in selected
            ]
            summary.append(
                {
                    "scope": (
                        "overall"
                        if profile == "all"
                        else "profile"
                    ),
                    "profile": profile,
                    "solver": solver,
                    "runs": len(selected),
                    "mean_completed": statistics.mean(completed),
                    "std_completed": (
                        statistics.stdev(completed)
                        if len(completed) > 1
                        else 0.0
                    ),
                    f"cvar_{alpha:g}_completed": lower_tail_cvar(
                        completed,
                        alpha,
                    ),
                    f"cvar_{1.0 - alpha:g}_sortie_loss": (
                        upper_tail_cvar(losses, alpha)
                    ),
                    "mean_missed": statistics.mean(
                        float(row["total_missed_sorties"])
                        for row in selected
                    ),
                    "mean_worst_wave": statistics.mean(
                        float(row["worst_wave_sorties"])
                        for row in selected
                    ),
                    "mean_recovery_time": statistics.mean(
                        observed_recovery_times
                    )
                    if observed_recovery_times
                    else float("nan"),
                    "max_recovery_time": max(
                        float(row.get("recovery_time_max", 0.0))
                        for row in selected
                    ),
                    "censored_recoveries": sum(
                        int(row.get("recovery_censored", 0))
                        for row in selected
                    ),
                    "mean_latency_p95_ms": statistics.mean(
                        float(row.get("latency_p95_ms", 0.0))
                        for row in selected
                    ),
                    "max_latency_ms": max(
                        float(row.get("latency_max_ms", 0.0))
                        for row in selected
                    ),
                    "mean_solve_latency_p95_ms": statistics.mean(
                        float(
                            row.get(
                                "solve_latency_p95_ms",
                                0.0,
                            )
                        )
                        for row in selected
                    ),
                    "max_solve_latency_ms": max(
                        float(
                            row.get(
                                "solve_latency_max_ms",
                                0.0,
                            )
                        )
                        for row in selected
                    ),
                    "deadline_miss_rate": (
                        deadline_misses / solve_calls
                        if solve_calls
                        else 0.0
                    ),
                    "mean_runtime_seconds": statistics.mean(
                        float(row.get("runtime_seconds", 0.0))
                        for row in selected
                    ),
                    "mean_resilience_ratio": _mean_resilience_ratio(
                        selected,
                    ),
                }
            )
    return summary


def _mean_resilience_ratio(
    selected: Sequence[Mapping[str, Any]],
) -> float:
    ratios = [
        float(row.get("resilience_index", 1.0))
        for row in selected
    ]
    return statistics.mean(ratios)


def build_comparisons(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    references: Sequence[str],
    bootstrap_samples: int,
) -> List[Dict[str, Any]]:
    table: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        table.setdefault(str(row["solver"]), {})[
            str(row["scenario_id"])
        ] = row
    if candidate not in table:
        raise ValueError(f"missing candidate solver: {candidate}")
    comparisons: List[Dict[str, Any]] = []
    for reference in references:
        if reference not in table:
            raise ValueError(
                f"missing reference solver: {reference}"
            )
        candidate_keys = set(table[candidate])
        reference_keys = set(table[reference])
        if candidate_keys != reference_keys:
            raise ValueError(
                f"unpaired scenarios for {candidate} and {reference}"
            )
        for profile in [
            *dict.fromkeys(
                str(row["profile"])
                for row in table[candidate].values()
            ),
            "all",
        ]:
            selected_keys = [
                key
                for key in sorted(candidate_keys)
                if profile == "all"
                or table[candidate][key]["profile"] == profile
            ]
            keys_by_seed: Dict[int, List[str]] = {}
            for key in selected_keys:
                keys_by_seed.setdefault(
                    int(table[candidate][key]["seed"]),
                    [],
                ).append(key)
            cell_sets = {
                seed: {
                    (
                        str(table[candidate][key]["profile"]),
                        float(
                            table[candidate][key][
                                "wave_interval"
                            ]
                        ),
                    )
                    for key in seed_keys
                }
                for seed, seed_keys in keys_by_seed.items()
            }
            if len(
                {
                    tuple(sorted(cells))
                    for cells in cell_sets.values()
                }
            ) != 1:
                raise ValueError(
                    "scenario matrix is incomplete across seeds"
                )
            candidate_scores = []
            reference_scores = []
            for seed in sorted(keys_by_seed):
                seed_keys = keys_by_seed[seed]
                candidate_scores.append(
                    statistics.mean(
                        float(
                            table[candidate][key][
                                "total_sorties_completed"
                            ]
                        )
                        for key in seed_keys
                    )
                )
                reference_scores.append(
                    statistics.mean(
                        float(
                            table[reference][key][
                                "total_sorties_completed"
                            ]
                        )
                        for key in seed_keys
                    )
                )
            deltas = [
                candidate_value - reference_value
                for candidate_value, reference_value in zip(
                    candidate_scores,
                    reference_scores,
                )
            ]
            sign = exact_two_sided_sign_test(
                candidate_scores,
                reference_scores,
            )
            bootstrap = paired_bootstrap_mean_ci(
                deltas,
                samples=bootstrap_samples,
            )
            comparisons.append(
                {
                    "profile": profile,
                    "candidate": candidate,
                    "reference": reference,
                    "pairs": len(keys_by_seed),
                    "scenario_cells": len(selected_keys),
                    "candidate_mean": statistics.mean(
                        candidate_scores
                    ),
                    "reference_mean": statistics.mean(
                        reference_scores
                    ),
                    **bootstrap,
                    "wins": int(sign["wins"]),
                    "ties": int(sign["ties"]),
                    "losses": int(sign["losses"]),
                    "sign_test_p": sign["p_value"],
                }
            )
    _apply_holm_correction(comparisons)
    return comparisons


def _apply_holm_correction(
    rows: List[Dict[str, Any]],
) -> None:
    ordered = sorted(
        enumerate(rows),
        key=lambda item: float(item[1]["sign_test_p"]),
    )
    running = 0.0
    count = len(ordered)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(
            1.0,
            (count - rank) * float(row["sign_test_p"]),
        )
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--references", nargs="+", required=True)
    parser.add_argument(
        "--summary-output",
        default="outputs/publishable_summary.csv",
    )
    parser.add_argument(
        "--comparison-output",
        default="outputs/publishable_comparisons.csv",
    )
    parser.add_argument("--cvar-alpha", type=float, default=0.10)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )
    args = parser.parse_args()

    rows = load_runs(args.inputs)
    summary = build_summary(rows, args.cvar_alpha)
    comparisons = build_comparisons(
        rows,
        args.candidate,
        args.references,
        args.bootstrap_samples,
    )
    write_csv(Path(args.summary_output), summary)
    write_csv(Path(args.comparison_output), comparisons)
    print(f"summary_csv_written: {args.summary_output}")
    print(f"comparison_csv_written: {args.comparison_output}")


if __name__ == "__main__":
    main()
