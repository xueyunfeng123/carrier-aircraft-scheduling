"""Audit oracle-label support before training a repair controller."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def load_rows(paths: Iterable[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    required = {
        "sample_id",
        "scenario_seed",
        "profile",
        "wave_interval",
        "normalized_time",
        "trigger_label",
        "budget_ms_label",
        "neighborhood_label",
        "runner_up_margin",
        "tie_count",
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
            for row in reader:
                sample_id = row["sample_id"]
                sample_key = (
                    sample_id,
                    row["profile"],
                    row["wave_interval"],
                )
                if sample_key in seen:
                    raise ValueError(
                        f"duplicate oracle sample: {sample_id}"
                    )
                seen.add(sample_key)
                parsed = dict(row)
                parsed.setdefault("candidate_count", "10")
                rows.append(parsed)
    if not rows:
        raise ValueError("oracle audit requires at least one row")
    return rows


def audit_rows(
    rows: List[Mapping[str, str]],
    *,
    min_informative_rate: float = 0.20,
    min_trigger_rate: float = 0.10,
    max_trigger_rate: float = 0.70,
) -> Dict[str, Any]:
    trigger = [int(row["trigger_label"]) for row in rows]
    informative = [
        int(row["tie_count"]) < int(row["candidate_count"])
        for row in rows
    ]
    triggered_rows = [
        row
        for row, enabled in zip(rows, trigger)
        if enabled
    ]
    time_bins = collections.Counter(
        min(3, max(0, int(float(row["normalized_time"]) * 4)))
        for row in rows
    )
    trigger_rate = sum(trigger) / len(trigger)
    informative_rate = sum(informative) / len(informative)
    budget_counts = collections.Counter(
        row["budget_ms_label"] for row in triggered_rows
    )
    neighborhood_counts = collections.Counter(
        row["neighborhood_label"] for row in triggered_rows
    )
    failures = []
    if informative_rate < min_informative_rate:
        failures.append("insufficient_informative_labels")
    if not min_trigger_rate <= trigger_rate <= max_trigger_rate:
        failures.append("trigger_rate_out_of_range")
    if triggered_rows and len(budget_counts) < 2:
        failures.append("insufficient_budget_support")
    if triggered_rows and len(neighborhood_counts) < 2:
        failures.append("insufficient_neighborhood_support")
    if len(time_bins) < 3:
        failures.append("insufficient_time_coverage")
    return {
        "samples": len(rows),
        "seeds": sorted(
            {int(row["scenario_seed"]) for row in rows}
        ),
        "profiles": dict(
            collections.Counter(row["profile"] for row in rows)
        ),
        "wave_intervals": dict(
            collections.Counter(
                row["wave_interval"] for row in rows
            )
        ),
        "time_quartiles": {
            str(index): time_bins.get(index, 0)
            for index in range(4)
        },
        "informative_samples": sum(informative),
        "informative_rate": informative_rate,
        "trigger_positive_samples": sum(trigger),
        "trigger_positive_rate": trigger_rate,
        "budget_support_triggered": dict(budget_counts),
        "neighborhood_support_triggered": dict(
            neighborhood_counts
        ),
        "mean_runner_up_margin": (
            sum(float(row["runner_up_margin"]) for row in rows)
            / len(rows)
        ),
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "min_informative_rate": min_informative_rate,
            "min_trigger_rate": min_trigger_rate,
            "max_trigger_rate": max_trigger_rate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--min-informative-rate",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--min-trigger-rate",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--max-trigger-rate",
        type=float,
        default=0.70,
    )
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    report = audit_rows(
        load_rows(args.inputs),
        min_informative_rate=args.min_informative_rate,
        min_trigger_rate=args.min_trigger_rate,
        max_trigger_rate=args.max_trigger_rate,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if args.require_pass and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
