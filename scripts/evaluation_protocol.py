"""Canonical protocol primitives for publishable rescheduling experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROTOCOL_VERSION = 1
PHASE_SEEDS = {
    "train": tuple(range(30001, 30033)),
    "selection": tuple(range(41001, 41021)),
    "calibration": tuple(range(42001, 42021)),
    "dev": tuple(range(43001, 43031)),
    "final": tuple(range(70001, 70051)),
}
FROZEN_FINAL_SEEDS = frozenset(PHASE_SEEDS["final"])
CANONICAL_INTERVALS = (47.5, 52.5, 57.5, 62.5, 67.5)
CANONICAL_PROFILES = (
    "none",
    "compound_light",
    "compound_medium",
    "compound_heavy",
)
CANONICAL_BUDGETS_MS = (10.0, 50.0, 200.0)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_phase_seeds(
    phase: str,
    seeds: Sequence[int],
    purpose: str,
) -> List[int]:
    if phase not in PHASE_SEEDS:
        raise ValueError(f"unknown evaluation phase: {phase}")
    if purpose not in {"training", "selection", "evaluation"}:
        raise ValueError(f"unknown seed purpose: {purpose}")
    selected = list(dict.fromkeys(int(seed) for seed in seeds))
    if not selected:
        raise ValueError("at least one scenario seed is required")
    unexpected = sorted(set(selected) - set(PHASE_SEEDS[phase]))
    if unexpected:
        raise ValueError(
            f"seeds are outside the canonical {phase} partition: "
            f"{unexpected}"
        )
    frozen = sorted(set(selected) & FROZEN_FINAL_SEEDS)
    if frozen and purpose != "evaluation":
        raise ValueError(
            "frozen final seeds 70001-70050 cannot be used for "
            f"{purpose}: {frozen}"
        )
    if phase == "final" and purpose != "evaluation":
        raise ValueError(
            "the final phase is blind evaluation only"
        )
    return selected


def reject_frozen_final_seeds(
    seeds: Iterable[int],
    purpose: str,
) -> None:
    frozen = sorted(set(int(seed) for seed in seeds) & FROZEN_FINAL_SEEDS)
    if frozen:
        raise ValueError(
            "frozen final seeds 70001-70050 cannot be used for "
            f"{purpose}: {frozen}"
        )


def scenario_id(
    profile: str,
    wave_interval: float,
    waves: int,
    seed: int,
    config_sha256: str,
    disruption_schedule_sha256: str = "",
    scenario_tape_version: int = 1,
) -> str:
    return canonical_json_sha256(
        {
            "profile": str(profile),
            "wave_interval": float(wave_interval),
            "waves": int(waves),
            "seed": int(seed),
            "config_sha256": str(config_sha256),
            "disruption_schedule_sha256": str(
                disruption_schedule_sha256
            ),
            "scenario_tape_version": int(
                scenario_tape_version
            ),
        }
    )[:20]


def build_manifest(
    *,
    phase: str,
    seeds: Sequence[int],
    profiles: Sequence[str],
    wave_intervals: Sequence[float],
    budgets_ms: Sequence[float],
    waves: int,
    solver_specs: Sequence[Mapping[str, Any]],
    base_config: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    validated_seeds = validate_phase_seeds(
        phase,
        seeds,
        purpose="evaluation",
    )
    if waves < 1:
        raise ValueError("waves must be positive")
    manifest: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": phase,
        "seeds": validated_seeds,
        "profiles": list(dict.fromkeys(str(item) for item in profiles)),
        "wave_intervals": list(
            dict.fromkeys(float(item) for item in wave_intervals)
        ),
        "budgets_ms": list(
            dict.fromkeys(float(item) for item in budgets_ms)
        ),
        "waves": int(waves),
        "solver_specs": [dict(item) for item in solver_specs],
        "base_config": dict(base_config),
        "base_config_sha256": canonical_json_sha256(dict(base_config)),
        "provenance": dict(provenance or {}),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def verify_manifest(manifest: Mapping[str, Any]) -> str:
    claimed = str(manifest.get("manifest_sha256", ""))
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    actual = canonical_json_sha256(payload)
    if claimed != actual:
        raise ValueError(
            f"manifest hash mismatch: claimed={claimed}, actual={actual}"
        )
    if int(manifest.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError("unsupported evaluation protocol version")
    return actual


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def lower_tail_cvar(
    values: Sequence[float],
    alpha: float = 0.10,
) -> float:
    if not values:
        raise ValueError("CVaR requires at least one value")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("CVaR alpha must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    count = max(1, int(math.ceil(alpha * len(ordered))))
    return statistics.mean(ordered[:count])


def upper_tail_cvar(
    values: Sequence[float],
    alpha: float = 0.10,
) -> float:
    if not values:
        raise ValueError("CVaR requires at least one value")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("CVaR alpha must be in (0, 1]")
    ordered = sorted(
        (float(value) for value in values),
        reverse=True,
    )
    count = max(1, int(math.ceil(alpha * len(ordered))))
    return statistics.mean(ordered[:count])


def compound_disruption_windows(
    event_log: Sequence[Mapping[str, Any]],
    horizon: float,
) -> List[Dict[str, float]]:
    boundary_events = [
        event
        for event in event_log
        if event.get("event_type")
        in {"disruption_start", "disruption_end"}
    ]
    active = 0
    start: float | None = None
    windows: List[Dict[str, float]] = []
    for event in boundary_events:
        event_time = float(event["time"])
        if event["event_type"] == "disruption_start":
            if active == 0:
                start = event_time
            active += 1
        else:
            active = max(0, active - 1)
            if active == 0 and start is not None:
                windows.append(
                    {"start_time": start, "end_time": event_time}
                )
                start = None
    if active > 0 and start is not None:
        windows.append(
            {"start_time": start, "end_time": float(horizon)}
        )
    return windows


def recovery_metrics(
    event_log: Sequence[Mapping[str, Any]],
    wave_records: Sequence[Mapping[str, Any]],
    baseline_wave_records: Sequence[Mapping[str, Any]],
    horizon: float,
    tolerance_sorties: int = 1,
    consecutive_waves: int = 2,
) -> Dict[str, float | int]:
    if tolerance_sorties < 0:
        raise ValueError("recovery tolerance must be non-negative")
    if consecutive_waves < 1:
        raise ValueError("consecutive waves must be positive")
    durations: List[float] = []
    censored = 0
    windows = compound_disruption_windows(event_log, horizon)
    baseline_by_wave = {
        int(record["wave_index"]): int(
            record["sorties_completed"]
        )
        for record in baseline_wave_records
    }
    for window in windows:
        end_time = float(window["end_time"])
        eligible = [
            record
            for record in wave_records
            if float(record["time"]) >= end_time
        ]
        recovered = None
        for index, record in enumerate(eligible):
            window_records = eligible[
                index : index + consecutive_waves
            ]
            if len(window_records) < consecutive_waves:
                break
            if all(
                int(item["sorties_completed"])
                >= baseline_by_wave.get(
                    int(item["wave_index"]),
                    int(item["sorties_completed"]),
                )
                - tolerance_sorties
                for item in window_records
            ):
                recovered = float(record["time"])
                break
        if recovered is None:
            censored += 1
        else:
            durations.append(max(0.0, recovered - end_time))
    return {
        "disruption_windows": len(windows),
        "recovery_time_mean": (
            statistics.mean(durations)
            if durations
            else float("nan")
        ),
        "recovery_time_max": max(durations, default=0.0),
        "recovery_observed": len(durations),
        "recovery_censored": censored,
    }


def paired_bootstrap_mean_ci(
    deltas: Sequence[float],
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 20261007,
) -> Dict[str, float]:
    if not deltas:
        raise ValueError("paired bootstrap requires at least one delta")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    values = [float(delta) for delta in deltas]
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(
            values[rng.randrange(len(values))]
            for _ in values
        )
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = min(
        samples - 1,
        max(0, int(math.floor(tail * samples))),
    )
    high_index = min(
        samples - 1,
        max(0, int(math.ceil((1.0 - tail) * samples)) - 1),
    )
    return {
        "mean_delta": statistics.mean(values),
        "ci_low": means[low_index],
        "ci_high": means[high_index],
    }
