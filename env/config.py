"""Default configuration and action constants for the scheduling environment."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


HIGH_LEVEL_ACTIONS = {
    0: "R",  # Recovery
    1: "F",  # Fueling
    2: "M",  # Arming
    3: "L",  # Launch
    4: "I",  # Inspection
}

ACTION_TO_INDEX = {value: key for key, value in HIGH_LEVEL_ACTIONS.items()}


DEFAULT_CONFIG: Dict[str, Any] = {
    "scenario_profile": "project_core",
    "num_aircraft": 45,
    "num_total_aircraft": 75,
    "group_size": 20,  # Wave demand; retained name for CLI/config compatibility.
    "num_parking_spots": 45,
    "parking_base_transfer_time": 2.0,
    "parking_ring_time_step": 1.0,
    "spatial_graph_enabled": True,
    "num_launch_positions": 3,
    "deck_edge_travel_time": 1.0,
    "traffic_time_step": 1.0,
    "traffic_planning_horizon": 180.0,
    "cbs_replan_enabled": False,
    "cbs_max_expanded_nodes": 2000,
    "scenario_tape_enabled": False,
    "scenario_tape_version": 1,
    "scenario_tape_seed_offset": 0,
    "disruption_profile": "none",
    "disruptions": [],
    "parking_interference_clearance": 2.0,
    "runway_interference_clearance": 0.5,
    "wave_interval": 120.0,
    "simulation_duration": 720.0,
    "num_recovery_channels": 1,
    "num_launch_channels": 3,
    "num_aircraft_lifts": 2,
    "num_fuel_servers": 20,
    "num_inspection_vehicles": 10,
    "num_arm_vehicles": 10,
    "num_tow_vehicles": 10,
    "num_ammo_transport_vehicles": 10,
    "num_lower_weapon_lifts": 6,
    "num_upper_weapon_lifts": 4,
    "num_personnel": 50,
    "hangar_transfer_time": 5.0,
    "personnel_scheduling_enabled": False,
    "routine_inspection_enabled": True,
    "fuel_personnel_required": 4,
    "inspection_personnel_required": 2,
    "arm_personnel_required": 4,
    "recovery_time": 1.0,
    "launch_time": 1.0,
    "fuel_time_mean": 20.0,
    "fuel_time_std": 3.0,
    "fuel_duration_model": "fuel_level_rate",
    "fuel_service_mode": "mobile",
    "fuel_station_coverage": 2,
    "fuel_station_covered_spots": 40,
    "initial_fuel_level": 1.0,
    "post_sortie_fuel_level": 0.2,
    "fuel_rate_per_minute": 0.05,
    "inspection_time_mean": 8.0,
    "inspection_time_std": 1.0,
    "arm_unit_time_mean": 5.0,
    "arm_unit_time_variance": 2.0,
    "ammo_extract_time_min": 5.0,
    "ammo_extract_time_max": 10.0,
    "lower_lift_time_mean": 3.0,
    "lower_lift_time_std": 0.5,
    "upper_lift_time_mean": 3.0,
    "upper_lift_time_std": 0.5,
    "sea_state_effects_enabled": False,
    "sea_state": 0.0,
    "recovery_retry_delay": 1.0,
    "stochastic_flight_failures_enabled": False,
    "aircraft_failure_mtbf_flight_hours": 100.0,
    "repair_time_min": 60.0,
    "repair_time_max": 360.0,
    "repair_time_decay_scale": 120.0,
    "min_process_time": 0.1,
    "arm_quantity_values": [1, 2, 3, 4],
    "arm_quantity_probs": [0.3, 0.4, 0.2, 0.1],
    "alpha": 1.0,
    "beta_recovery": 0.1,
    "beta_fuel": 0.2,
    "beta_inspection": 0.2,
    "beta_arm": 0.2,
    "beta_launch": 1.0,
    "eta_idle": 1.0,
    "invalid_action_penalty": 5.0,
    "terminal_reward": 100.0,
}


HAITIAN_2026_CONFIG: Dict[str, Any] = {
    "scenario_profile": "haitian_2026",
    "num_aircraft": 45,
    "num_total_aircraft": 75,
    "group_size": 20,
    "num_parking_spots": 45,
    "num_launch_positions": 4,
    # Four physical positions feed one serial launch stream (one aircraft/min).
    "num_launch_channels": 1,
    "num_recovery_channels": 1,
    "num_aircraft_lifts": 2,
    "num_fuel_servers": 20,
    "fuel_service_mode": "fixed_station",
    "fuel_station_coverage": 2,
    "fuel_station_covered_spots": 40,
    "fuel_duration_model": "normal",
    "fuel_time_mean": 20.0,
    # The source omits dispersion; 3 minutes is an explicit experiment assumption.
    "fuel_time_std": 3.0,
    "routine_inspection_enabled": False,
    "num_inspection_vehicles": 0,
    "inspection_personnel_required": 0,
    "personnel_scheduling_enabled": True,
    # Fifty simultaneously staffed support posts from 150 personnel in three shifts.
    "num_personnel": 50,
    "fuel_personnel_required": 4,
    # Arming personnel demand is not published and is therefore not invented.
    "arm_personnel_required": 0,
    "sea_state_effects_enabled": True,
    "sea_state": 3.0,
    "stochastic_flight_failures_enabled": True,
    "aircraft_failure_mtbf_flight_hours": 100.0,
    "repair_time_min": 60.0,
    "repair_time_max": 360.0,
    # Truncated exponential decay assumption over the published 1-6 hour range.
    "repair_time_decay_scale": 120.0,
}


SCENARIO_CONFIG_PRESETS: Dict[str, Dict[str, Any]] = {
    "project_core": {},
    "haitian_2026": HAITIAN_2026_CONFIG,
}


def resolve_config(
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply a scenario preset while preserving explicit caller overrides."""

    overrides = dict(config or {})
    profile = str(
        overrides.get(
            "scenario_profile",
            DEFAULT_CONFIG["scenario_profile"],
        )
    )
    try:
        preset = SCENARIO_CONFIG_PRESETS[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(SCENARIO_CONFIG_PRESETS))
        raise ValueError(
            "CarrierAircraftSchedulingEnv currently executes only "
            f"these profiles: {choices}; received {profile!r}"
        ) from exc
    resolved = dict(DEFAULT_CONFIG)
    resolved.update(preset)
    resolved.update(overrides)
    return resolved
