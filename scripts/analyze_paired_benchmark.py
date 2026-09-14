"""Summarize paired benchmark runs and compute exact sign tests."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ScenarioKey = Tuple[int, float]
ScoreTable = Dict[str, Dict[ScenarioKey, int]]


def load_scores(paths: Iterable[str]) -> ScoreTable:
    scores: ScoreTable = {}
    required = {
        "solver",
        "seed",
        "wave_interval",
        "total_sorties_completed",
    }
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path} is missing columns: {sorted(missing)}"
                )
            for row in reader:
                solver = row["solver"]
                key = (
                    int(row["seed"]),
                    float(row["wave_interval"]),
                )
                solver_scores = scores.setdefault(solver, {})
                if key in solver_scores:
                    raise ValueError(
                        f"duplicate result for {solver}, seed={key[0]}, "
                        f"interval={key[1]:g}"
                    )
                solver_scores[key] = int(
                    row["total_sorties_completed"]
                )
    return scores


def exact_two_sided_sign_test(
    candidate_scores: Sequence[int],
    reference_scores: Sequence[int],
) -> Dict[str, float]:
    if len(candidate_scores) != len(reference_scores):
        raise ValueError("paired score lists must have equal lengths")
    deltas = [
        candidate - reference
        for candidate, reference in zip(
            candidate_scores,
            reference_scores,
        )
    ]
    wins = sum(delta > 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    non_ties = wins + losses
    if non_ties == 0:
        p_value = 1.0
    else:
        tail = min(wins, losses)
        tail_probability = sum(
            math.comb(non_ties, index)
            for index in range(tail + 1)
        ) / (2 ** non_ties)
        p_value = min(1.0, 2.0 * tail_probability)
    return {
        "mean_delta": statistics.mean(deltas),
        "wins": float(wins),
        "ties": float(ties),
        "losses": float(losses),
        "p_value": p_value,
    }


def validate_paired_scenarios(
    scores: ScoreTable,
    solvers: Sequence[str],
) -> List[ScenarioKey]:
    missing = [solver for solver in solvers if solver not in scores]
    if missing:
        raise ValueError(f"missing solvers: {missing}")
    reference_keys = set(scores[solvers[0]])
    for solver in solvers[1:]:
        solver_keys = set(scores[solver])
        if solver_keys != reference_keys:
            raise ValueError(
                f"scenario mismatch for {solver}: "
                f"missing={sorted(reference_keys - solver_keys)}, "
                f"extra={sorted(solver_keys - reference_keys)}"
            )
    return sorted(reference_keys, key=lambda item: (item[1], item[0]))


def build_summary_rows(
    scores: ScoreTable,
    solvers: Sequence[str],
    candidate: str,
) -> List[Dict[str, object]]:
    keys = validate_paired_scenarios(scores, solvers)
    intervals = sorted({interval for _, interval in keys})
    rows: List[Dict[str, object]] = []
    for interval_value in [*intervals, None]:
        selected = [
            key
            for key in keys
            if interval_value is None or key[1] == interval_value
        ]
        row: Dict[str, object] = {
            "wave_interval": (
                "overall"
                if interval_value is None
                else f"{interval_value:g}"
            ),
            "scenarios": len(selected),
        }
        for solver in solvers:
            row[f"{solver}_mean"] = (
                f"{statistics.mean(scores[solver][key] for key in selected):.2f}"
            )
        for reference in solvers:
            if reference == candidate:
                continue
            row[f"{candidate}_minus_{reference}"] = (
                f"{statistics.mean(
                    scores[candidate][key] - scores[reference][key]
                    for key in selected
                ):.2f}"
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--solvers", nargs="+", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.candidate not in args.solvers:
        parser.error("--candidate must be included in --solvers")

    scores = load_scores(args.inputs)
    keys = validate_paired_scenarios(scores, args.solvers)
    rows = build_summary_rows(
        scores,
        args.solvers,
        args.candidate,
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
    print(f"paired_summary_csv_written: {output}")

    candidate_scores = [scores[args.candidate][key] for key in keys]
    for reference in args.solvers:
        if reference == args.candidate:
            continue
        stats = exact_two_sided_sign_test(
            candidate_scores,
            [scores[reference][key] for key in keys],
        )
        print(
            f"{args.candidate}_vs_{reference},"
            f"mean_delta,{stats['mean_delta']:.2f},"
            f"wins,{int(stats['wins'])},"
            f"ties,{int(stats['ties'])},"
            f"losses,{int(stats['losses'])},"
            f"p_value,{stats['p_value']:.8g}"
        )


if __name__ == "__main__":
    main()
