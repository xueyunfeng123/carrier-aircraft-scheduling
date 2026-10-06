"""Reproducible availability disruptions for deck scheduling experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union


DisruptionTarget = Union[int, str]


@dataclass(frozen=True)
class DisruptionSpec:
    disruption_id: int
    kind: str
    target: DisruptionTarget
    start_time: float
    end_time: float
    multiplier: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "disruption_id": self.disruption_id,
            "kind": self.kind,
            "target": self.target,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "multiplier": self.multiplier,
        }


PROFILE_SETTINGS = {
    "none": {
        "vehicle_count": 0,
        "runway_count": 0,
        "aircraft_count": 0,
        "duration": (0.0, 0.0),
        "multiplier": 1.0,
    },
    "light": {
        "vehicle_count": 1,
        "runway_count": 1,
        "aircraft_count": 2,
        "duration": (10.0, 20.0),
        "multiplier": 1.25,
    },
    "medium": {
        "vehicle_count": 3,
        "runway_count": 1,
        "aircraft_count": 5,
        "duration": (25.0, 40.0),
        "multiplier": 1.50,
    },
    "heavy": {
        "vehicle_count": 6,
        "runway_count": 2,
        "aircraft_count": 9,
        "duration": (40.0, 60.0),
        "multiplier": 2.0,
    },
}

DISRUPTION_KINDS = (
    "vehicle_outage",
    "runway_closure",
    "aircraft_hold",
    "service_slowdown",
)


def build_disruption_schedule(
    config: Dict[str, Any],
    rng: random.Random,
    vehicle_ids: Sequence[str],
) -> List[DisruptionSpec]:
    explicit = list(config.get("disruptions", ()))
    if explicit:
        return [
            _parse_explicit_spec(
                disruption_id,
                item,
                float(config["simulation_duration"]),
            )
            for disruption_id, item in enumerate(explicit)
        ]

    profile = str(config.get("disruption_profile", "none"))
    if profile not in PROFILE_SETTINGS:
        raise ValueError(
            f"unknown disruption profile: {profile}"
        )
    settings = PROFILE_SETTINGS[profile]
    vehicle_count = int(settings["vehicle_count"])
    runway_count = int(settings["runway_count"])
    aircraft_count = int(settings["aircraft_count"])
    if vehicle_count == runway_count == aircraft_count == 0:
        return []

    candidates = {
        "vehicle_outage": tuple(vehicle_ids),
        "runway_closure": tuple(
            range(int(config["num_launch_positions"]))
        ),
        "aircraft_hold": tuple(
            range(int(config["num_aircraft"]))
        ),
        "service_slowdown": (
            "fuel",
            "inspection",
            "arm",
        ),
    }
    if any(not candidates[kind] for kind in DISRUPTION_KINDS):
        raise ValueError(
            "disruption profile requires aircraft, runways, and vehicles"
        )

    horizon = float(config["simulation_duration"])
    configured_duration_min, configured_duration_max = settings["duration"]
    duration_max = min(float(configured_duration_max), horizon)
    duration_min = min(float(configured_duration_min), duration_max)
    schedule: List[DisruptionSpec] = []
    disruption_id = 0
    shock_centers = (0.25, 0.50, 0.75)
    for center_fraction in shock_centers:
        center = horizon * center_fraction
        jitter = rng.uniform(-0.03, 0.03) * horizon
        duration = rng.uniform(duration_min, duration_max)
        start_time = min(
            horizon - duration,
            max(0.0, center + jitter),
        )
        end_time = start_time + duration

        vehicle_targets = list(candidates["vehicle_outage"])
        runway_targets = list(candidates["runway_closure"])
        aircraft_targets = list(candidates["aircraft_hold"])
        rng.shuffle(vehicle_targets)
        rng.shuffle(runway_targets)
        rng.shuffle(aircraft_targets)
        selected_targets = (
            (
                "vehicle_outage",
                vehicle_targets[:vehicle_count],
            ),
            (
                "runway_closure",
                runway_targets[:runway_count],
            ),
            (
                "aircraft_hold",
                aircraft_targets[:aircraft_count],
            ),
            ("service_slowdown", ["all"]),
        )
        for kind, targets in selected_targets:
            for target in targets:
                schedule.append(
                    DisruptionSpec(
                        disruption_id=disruption_id,
                        kind=kind,
                        target=target,
                        start_time=start_time,
                        end_time=end_time,
                        multiplier=(
                            float(settings["multiplier"])
                            if kind == "service_slowdown"
                            else 1.0
                        ),
                    )
                )
                disruption_id += 1
    return sorted(
        schedule,
        key=lambda item: (
            item.start_time,
            item.disruption_id,
        ),
    )


def _parse_explicit_spec(
    disruption_id: int,
    item: Dict[str, Any],
    horizon: float,
) -> DisruptionSpec:
    kind = str(item["kind"])
    if kind not in DISRUPTION_KINDS:
        raise ValueError(f"unknown disruption kind: {kind}")
    start_time = float(item["start_time"])
    if "end_time" in item:
        end_time = float(item["end_time"])
    else:
        end_time = start_time + float(item["duration"])
    if not 0.0 <= start_time < end_time <= horizon:
        raise ValueError(
            "disruption times must satisfy "
            "0 <= start_time < end_time <= simulation_duration"
        )
    multiplier = float(item.get("multiplier", 1.0))
    if kind == "service_slowdown" and multiplier < 1.0:
        raise ValueError(
            "service slowdown multiplier must be at least 1"
        )
    return DisruptionSpec(
        disruption_id=disruption_id,
        kind=kind,
        target=item["target"],
        start_time=start_time,
        end_time=end_time,
        multiplier=multiplier,
    )
