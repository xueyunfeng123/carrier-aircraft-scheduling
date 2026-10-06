"""Observation and mask encoder for RL policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import HIGH_LEVEL_ACTIONS


AIRCRAFT_FEATURE_DIM = 23
GLOBAL_FEATURE_DIM = 20
TARGET_FEATURE_DIM = 15
TARGET_AUX_FEATURE_DIM = 2
OBSERVATION_SCHEMA_VERSION = 9


@dataclass
class EncodedObservation:
    aircraft: List[List[float]]
    low_aux: List[List[List[float]]]
    targets: List[List[float]]
    target_keys: List[Tuple[str, Any]]
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
        _normalize_aircraft(
            row,
            aircraft_id,
            num_aircraft,
            num_parking_spots,
            simulation_duration,
        )
        for aircraft_id, row in enumerate(state["aircraft"])
    ]
    target_features, target_keys = _encode_targets(
        env,
        simulation_duration,
    )
    global_features = _encode_global_features(state, env, simulation_duration, wave_interval)
    mask = env.get_action_mask()
    low_masks = [
        [int(value) for value in mask["low_level_by_high"][action_id]]
        for action_id in HIGH_LEVEL_ACTIONS
    ]
    low_aux = _encode_low_aux(env, low_masks)
    return EncodedObservation(
        aircraft=aircraft_features,
        low_aux=low_aux,
        targets=target_features,
        target_keys=target_keys,
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
        "low_aux": torch.tensor([encoded.low_aux], dtype=torch.float32, device=device),
        "targets": torch.tensor([encoded.targets], dtype=torch.float32, device=device),
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
        "low_aux": torch.tensor(
            [item.low_aux for item in encoded_items],
            dtype=torch.float32,
            device=device,
        ),
        "targets": torch.tensor(
            [item.targets for item in encoded_items],
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
    aircraft_id: int,
    num_aircraft: int,
    num_parking_spots: int,
    simulation_duration: float,
) -> List[float]:
    return [
        row[0],  # initially assigned to the five-aircraft reserve
        _scale_nonnegative(row[1], num_parking_spots),
        row[2] / 2.0,
        row[3],
        row[4],
        row[5] / max(1.0, num_aircraft),
        row[6] / max(1.0, num_aircraft),
        row[7] / 2.0,
        row[8] / 2.0,
        min(1.0, max(0.0, row[9])),
        row[10] / 2.0,
        row[11] / 2.0,
        row[12] / 4.0,
        row[13] / 3.0,
        row[14] / 4.0,
        row[15] / simulation_duration,
        row[16] / simulation_duration,
        row[17] / simulation_duration,
        row[18] / simulation_duration,
        row[19] / simulation_duration,
        row[20] / simulation_duration,
        row[21] / simulation_duration,
        aircraft_id / max(1, num_aircraft - 1),
    ]


def _encode_global_features(
    state: Dict[str, Any],
    env: CarrierAircraftSchedulingEnv,
    simulation_duration: float,
    wave_interval: float,
) -> List[float]:
    resources = state["resources"]
    wave = state["wave"]
    deck = state["deck"]
    config = env.config
    next_wave_time = wave["next_wave_time"]
    if next_wave_time is None:
        time_to_next_wave = 0.0
    else:
        time_to_next_wave = max(0.0, float(next_wave_time) - float(state["time"]))

    launch_fill_ratio = float(wave["launches_started"]) / max(
        1.0,
        float(wave["launch_target"]),
    )
    recovery_pressure = float(wave["pending_recovery_count"]) / max(
        1.0,
        float(wave["launch_target"]),
    )

    return [
        float(state["time"]) / simulation_duration,
        float(wave["index"]) / max(1.0, simulation_duration / wave_interval),
        launch_fill_ratio,
        recovery_pressure,
        time_to_next_wave / wave_interval,
        resources["recovery_channels"] / max(1.0, float(config["num_recovery_channels"])),
        resources["fuel_servers"]
        / max(1.0, float(config["num_fuel_servers"]))
        / env.service_time_multipliers["fuel"],
        resources["inspection_vehicles"]
        / max(1.0, float(config["num_inspection_vehicles"]))
        / env.service_time_multipliers["inspection"],
        resources["arm_vehicles"]
        / max(1.0, float(config["num_arm_vehicles"]))
        / env.service_time_multipliers["arm"],
        resources["ammo_transport_vehicles"] / max(1.0, float(config["num_ammo_transport_vehicles"])),
        resources["lower_weapon_lifts"] / max(1.0, float(config["num_lower_weapon_lifts"])),
        resources["upper_weapon_lifts"] / max(1.0, float(config["num_upper_weapon_lifts"])),
        resources["personnel"] / max(1.0, float(config["num_personnel"])),
        resources["launch_channels"] / max(1.0, float(config["num_launch_channels"])),
        resources["free_parking_spots"] / max(1.0, float(config["num_parking_spots"])),
        float(deck["free_pathway_fraction"]),
        float(state["event_queue_size"]) / 100.0,
        sum(1 for aircraft in env.aircraft if aircraft.is_airborne) / max(1, env.num_aircraft),
        sum(aircraft.sorties_completed for aircraft in env.aircraft) / max(1, env.num_aircraft * 12),
        sum(aircraft.missed_sorties for aircraft in env.aircraft) / max(1, env.num_aircraft * 12),
    ]


def _encode_targets(
    env: CarrierAircraftSchedulingEnv,
    simulation_duration: float,
) -> Tuple[List[List[float]], List[Tuple[str, Any]]]:
    target_features: List[List[float]] = []
    target_keys: List[Tuple[str, Any]] = []
    node_ids = (
        [node.node_id for node in env.deck_layout.graph.nodes]
        if env.deck_layout is not None
        else []
    )
    node_index = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }
    node_divisor = max(1, len(node_ids) - 1)
    section_names = ("northwest", "northeast", "southwest", "southeast")

    for spot_id, occupied_by in enumerate(env.parking_occupancy):
        section = (
            env.deck_layout.parking_sections[spot_id]
            if env.deck_layout is not None
            else ""
        )
        section_one_hot = [
            float(section == section_name)
            for section_name in section_names
        ]
        node_id = (
            env.deck_layout.parking_node(spot_id)
            if env.deck_layout is not None
            else f"parking_{spot_id}"
        )
        release_delay = max(
            0.0,
            env.parking_release_times[spot_id] - env.time,
        )
        target_features.append(
            [
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                float(occupied_by is None),
                spot_id / max(1, env.num_parking_spots - 1),
                node_index.get(node_id, 0) / node_divisor,
                *section_one_hot,
                release_delay / simulation_duration,
            ]
        )
        target_keys.append(("parking", spot_id))

    launch_count = int(env.config["num_launch_positions"])
    for runway_id in range(launch_count):
        node_id = (
            env.deck_layout.launch_nodes[runway_id]
            if env.deck_layout is not None
            else f"launch_{runway_id}"
        )
        release_delay = max(
            0.0,
            env.runway_release_times[runway_id] - env.time,
        )
        target_features.append(
            [
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                float(
                    runway_id not in env.closed_runway_ids
                    and (
                        env.deck_layout is None
                        or not env.deck_occupancy.occupants[node_id]
                    )
                ),
                runway_id / max(1, launch_count - 1),
                node_index.get(node_id, 0) / node_divisor,
                0.0, 0.0, 0.0, 0.0,
                release_delay / simulation_duration,
            ]
        )
        target_keys.append(("runway", runway_id))

    type_offsets = {
        "fuel": 2,
        "inspection": 3,
        "arm": 4,
        "tractor": 5,
    }
    service_counts: Dict[str, int] = {}
    for vehicle in env.service_vehicles:
        service_counts[vehicle.service_type] = (
            service_counts.get(vehicle.service_type, 0) + 1
        )
    service_indices: Dict[str, int] = {}
    for vehicle in env.service_vehicles:
        service_type = vehicle.service_type
        local_index = service_indices.get(service_type, 0)
        service_indices[service_type] = local_index + 1
        type_features = [0.0] * 7
        type_features[type_offsets[service_type]] = 1.0
        target_features.append(
            [
                *type_features,
                float(
                    vehicle.busy_aircraft_id is None
                    and vehicle.vehicle_id
                    not in env.unavailable_vehicle_ids
                ),
                local_index / max(1, service_counts[service_type] - 1),
                node_index.get(vehicle.node_id, 0) / node_divisor,
                0.0, 0.0, 0.0, 0.0,
                0.0,
            ]
        )
        target_keys.append(("vehicle", vehicle.vehicle_id))

    target_features.append(
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0,
        ]
    )
    target_keys.append(("none", None))
    return target_features, target_keys


def _encode_low_aux(
    env: CarrierAircraftSchedulingEnv,
    low_masks: List[List[int]],
) -> List[List[List[float]]]:
    low_aux = [
        [
            [0.0, 0.0]
            for _ in range(env.num_aircraft)
        ]
        for _ in HIGH_LEVEL_ACTIONS
    ]
    for high_level, mask in enumerate(low_masks):
        candidate_ids = [
            aircraft_id
            for aircraft_id, allowed in enumerate(mask)
            if allowed
        ]
        ranked = sorted(
            candidate_ids,
            key=lambda aircraft_id: (
                _expected_action_duration(
                    env,
                    high_level,
                    aircraft_id,
                ),
                aircraft_id,
            ),
        )
        for rank, aircraft_id in enumerate(ranked):
            low_aux[high_level][aircraft_id] = [
                1.0,
                1.0 - rank / max(1, len(ranked) - 1),
            ]
    return low_aux


def _expected_action_duration(
    env: CarrierAircraftSchedulingEnv,
    high_level: int,
    aircraft_id: int,
) -> float:
    action_name = HIGH_LEVEL_ACTIONS[high_level]
    aircraft = env.aircraft[aircraft_id]
    if action_name == "R":
        return env._expected_recovery_duration(aircraft_id)
    if action_name == "L":
        return env._expected_launch_duration(aircraft_id)
    if action_name == "F":
        return (
            env._expected_service_vehicle_travel(
                "fuel",
                aircraft_id,
            )
            + (1.0 - aircraft.fuel_level)
            / float(env.config["fuel_rate_per_minute"])
        )
    if action_name == "I":
        return (
            env._expected_service_vehicle_travel(
                "inspection",
                aircraft_id,
            )
            + float(env.config["inspection_time_mean"])
        )
    extract_mean = (
        float(env.config["ammo_extract_time_min"])
        + float(env.config["ammo_extract_time_max"])
    ) / 2.0
    return (
        env._expected_service_vehicle_travel(
            "arm",
            aircraft_id,
        )
        + extract_mean
        + float(env.config["lower_lift_time_mean"])
        + float(env.config["upper_lift_time_mean"])
        + aircraft.arm_quantity_required
        * float(env.config["arm_unit_time_mean"])
    )


def target_mask_for_action(
    env: CarrierAircraftSchedulingEnv,
    encoded: EncodedObservation,
    high_level: int,
    aircraft_id: int,
) -> List[int]:
    target_mask, _ = target_context_for_action(
        env,
        encoded,
        high_level,
        aircraft_id,
    )
    return target_mask


def target_context_for_action(
    env: CarrierAircraftSchedulingEnv,
    encoded: EncodedObservation,
    high_level: int,
    aircraft_id: int,
) -> Tuple[List[int], List[List[float]]]:
    action_name = HIGH_LEVEL_ACTIONS[high_level]
    candidates = env.get_target_candidates(high_level, aircraft_id)
    if action_name == "R":
        ordered_candidate_keys = [
            ("parking", candidate)
            for candidate in candidates
        ]
    elif action_name == "L":
        ordered_candidate_keys = [
            ("runway", candidate)
            for candidate in candidates
        ]
    else:
        ordered_candidate_keys = [
            ("vehicle", candidate)
            for candidate in candidates
        ]
    if not ordered_candidate_keys:
        ordered_candidate_keys = [("none", None)]
    candidate_keys = set(ordered_candidate_keys)
    target_mask = [
        int(target_key in candidate_keys)
        for target_key in encoded.target_keys
    ]
    target_slot_by_key = {
        target_key: target_slot
        for target_slot, target_key in enumerate(encoded.target_keys)
    }
    candidate_slots = [
        target_slot_by_key[target_key]
        for target_key in ordered_candidate_keys
    ]
    target_aux = [
        [0.0, 0.0]
        for _ in encoded.target_keys
    ]
    for rank, target_slot in enumerate(candidate_slots):
        target_aux[target_slot] = [
            1.0,
            1.0 - rank / max(1, len(candidate_slots) - 1),
        ]
    return target_mask, target_aux


def target_slot_for_action(
    encoded: EncodedObservation,
    action: Dict[str, Any],
) -> int:
    action_name = HIGH_LEVEL_ACTIONS[int(action["high_level"])]
    if action_name == "R":
        target_key = ("parking", action.get("target_id"))
    elif action_name == "L":
        target_key = ("runway", action.get("target_id"))
    else:
        target_key = ("vehicle", action.get("vehicle_id"))
    if target_key[1] is None:
        target_key = ("none", None)
    return encoded.target_keys.index(target_key)


def apply_target_slot(
    encoded: EncodedObservation,
    action: Dict[str, Any],
    target_slot: int,
) -> Dict[str, Any]:
    completed = dict(action)
    target_type, target_value = encoded.target_keys[target_slot]
    completed["target_slot"] = int(target_slot)
    if target_type in ("parking", "runway"):
        completed["target_id"] = int(target_value)
    elif target_type == "vehicle":
        completed["vehicle_id"] = str(target_value)
    return completed


def _scale_nonnegative(value: float, divisor: float) -> float:
    if value < 0:
        return -1.0
    return value / max(1.0, divisor)
