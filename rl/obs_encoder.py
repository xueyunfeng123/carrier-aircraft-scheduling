"""Observation and mask encoder for RL policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv


AIRCRAFT_FEATURE_DIM = 20
GLOBAL_FEATURE_DIM = 18


@dataclass
class EncodedObservation:
    aircraft: List[List[float]]
    global_features: List[float]
    high_mask: List[int]
    low_masks: List[List[int]]


def encode_observation(env: CarrierAircraftSchedulingEnv) -> EncodedObservation:
    """Encode the current env state into normalized numeric features."""

    state = env.get_state()
    config = env.config
    simulation_duration = max(1.0, float(config["simulation_duration"]))
    wave_interval = max(1.0, float(config["wave_interval"]))
    num_aircraft = max(1, int(config["num_aircraft"]))
    num_parking_spots = max(1, int(config["num_parking_spots"]))

    aircraft_features = [
        _normalize_aircraft(row, num_aircraft, num_parking_spots, simulation_duration)
        for row in state["aircraft"]
    ]
    global_features = _encode_global_features(state, env, simulation_duration, wave_interval)
    mask = env.get_action_mask()
    low_masks = [
        [int(value) for value in mask["low_level_by_high"][action_id]]
        for action_id in range(4)
    ]
    return EncodedObservation(
        aircraft=aircraft_features,
        global_features=global_features,
        high_mask=[int(value) for value in mask["high_level"]],
        low_masks=low_masks,
    )


def to_torch_batch(encoded: EncodedObservation, device: str = "cpu") -> Dict[str, Any]:
    """Convert one encoded observation to torch tensors with batch dimension."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for RL solver/training. Install it with `pip install torch`."
        ) from exc

    return {
        "aircraft": torch.tensor([encoded.aircraft], dtype=torch.float32, device=device),
        "global": torch.tensor([encoded.global_features], dtype=torch.float32, device=device),
        "high_mask": torch.tensor([encoded.high_mask], dtype=torch.bool, device=device),
        "low_masks": torch.tensor([encoded.low_masks], dtype=torch.bool, device=device),
    }


def batch_to_torch(encoded_items: List[EncodedObservation], device: str = "cpu") -> Dict[str, Any]:
    """Convert multiple encoded observations to torch tensors."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for RL solver/training. Install it with `pip install torch`."
        ) from exc

    return {
        "aircraft": torch.tensor(
            [item.aircraft for item in encoded_items],
            dtype=torch.float32,
            device=device,
        ),
        "global": torch.tensor(
            [item.global_features for item in encoded_items],
            dtype=torch.float32,
            device=device,
        ),
        "high_mask": torch.tensor(
            [item.high_mask for item in encoded_items],
            dtype=torch.bool,
            device=device,
        ),
        "low_masks": torch.tensor(
            [item.low_masks for item in encoded_items],
            dtype=torch.bool,
            device=device,
        ),
    }


def _normalize_aircraft(
    row: List[float],
    num_aircraft: int,
    num_parking_spots: int,
    simulation_duration: float,
) -> List[float]:
    return [
        row[0],  # group
        _scale_nonnegative(row[1], num_parking_spots),
        row[2] / 2.0,
        row[3],
        row[4],
        row[5] / max(1.0, num_aircraft),
        row[6] / max(1.0, num_aircraft),
        row[7] / 2.0,
        row[8] / 2.0,
        row[9] / 2.0,
        row[10] / 4.0,
        row[11] / 3.0,
        row[12] / 4.0,
        row[13] / simulation_duration,
        row[14] / simulation_duration,
        row[15] / simulation_duration,
        row[16] / simulation_duration,
        row[17] / simulation_duration,
        _scale_nonnegative(row[18], simulation_duration),
        row[19] / max(1.0, num_aircraft),
    ]


def _encode_global_features(
    state: Dict[str, Any],
    env: CarrierAircraftSchedulingEnv,
    simulation_duration: float,
    wave_interval: float,
) -> List[float]:
    resources = state["resources"]
    wave = state["wave"]
    config = env.config
    next_wave_time = wave["next_wave_time"]
    if next_wave_time is None:
        time_to_next_wave = 0.0
    else:
        time_to_next_wave = max(0.0, float(next_wave_time) - float(state["time"]))

    active_launch_group = 0.0 if wave["active_launch_group"] == "A" else 1.0
    active_recovery_group = -1.0
    if wave["active_recovery_group"] == "A":
        active_recovery_group = 0.0
    elif wave["active_recovery_group"] == "B":
        active_recovery_group = 1.0

    return [
        float(state["time"]) / simulation_duration,
        float(wave["index"]) / max(1.0, simulation_duration / wave_interval),
        active_launch_group,
        active_recovery_group,
        time_to_next_wave / wave_interval,
        resources["recovery_channels"] / max(1.0, float(config["num_recovery_channels"])),
        resources["fuel_servers"] / max(1.0, float(config["num_fuel_servers"])),
        resources["arm_vehicles"] / max(1.0, float(config["num_arm_vehicles"])),
        resources["ammo_transport_vehicles"] / max(1.0, float(config["num_ammo_transport_vehicles"])),
        resources["lower_weapon_lifts"] / max(1.0, float(config["num_lower_weapon_lifts"])),
        resources["upper_weapon_lifts"] / max(1.0, float(config["num_upper_weapon_lifts"])),
        resources["personnel"] / max(1.0, float(config["num_personnel"])),
        resources["launch_channels"] / max(1.0, float(config["num_launch_channels"])),
        resources["free_parking_spots"] / max(1.0, float(config["num_parking_spots"])),
        len(env.event_queue) / 100.0,
        sum(1 for aircraft in env.aircraft if aircraft.is_airborne) / max(1, env.num_aircraft),
        sum(aircraft.sorties_completed for aircraft in env.aircraft) / max(1, env.num_aircraft * 12),
        sum(aircraft.missed_sorties for aircraft in env.aircraft) / max(1, env.num_aircraft * 12),
    ]


def _scale_nonnegative(value: float, divisor: float) -> float:
    if value < 0:
        return -1.0
    return value / max(1.0, divisor)
