"""Default configuration and action constants for the scheduling environment."""

from __future__ import annotations

from typing import Any, Dict


HIGH_LEVEL_ACTIONS = {
    0: "R",  # Recovery
    1: "F",  # Fueling
    2: "M",  # Arming
    3: "L",  # Launch
}

ACTION_TO_INDEX = {value: key for key, value in HIGH_LEVEL_ACTIONS.items()}


DEFAULT_CONFIG: Dict[str, Any] = {
    "num_aircraft": 40,
    "group_size": 20,
    "num_parking_spots": 45,
    "parking_base_transfer_time": 2.0,
    "parking_ring_time_step": 1.0,
    "wave_interval": 120.0,
    "simulation_duration": 720.0,
    "num_recovery_channels": 1,
    "num_launch_channels": 1,
    "num_fuel_servers": 20,
    "num_arm_vehicles": 10,
    "num_ammo_transport_vehicles": 10,
    "num_lower_weapon_lifts": 6,
    "num_upper_weapon_lifts": 4,
    "num_personnel": 50,
    "fuel_personnel_required": 4,
    "arm_personnel_required": 4,
    "enable_recovery_deadlines": True,
    "recovery_deadline_min": 8.0,
    "recovery_deadline_max": 25.0,
    "recovery_retry_delay": 10.0,
    "recovery_time": 1.0,
    "launch_time": 1.0,
    "fuel_time_mean": 20.0,
    "fuel_time_std": 3.0,
    "arm_unit_time_mean": 5.0,
    "arm_unit_time_variance": 2.0,
    "ammo_extract_time_min": 5.0,
    "ammo_extract_time_max": 10.0,
    "lower_lift_time_mean": 3.0,
    "lower_lift_time_std": 0.5,
    "upper_lift_time_mean": 3.0,
    "upper_lift_time_std": 0.5,
    "min_process_time": 0.1,
    "arm_quantity_values": [1, 2, 3, 4],
    "arm_quantity_probs": [0.3, 0.4, 0.2, 0.1],
    "alpha": 1.0,
    "beta_recovery": 0.1,
    "beta_fuel": 0.2,
    "beta_arm": 0.2,
    "beta_launch": 1.0,
    "eta_idle": 1.0,
    "invalid_action_penalty": 5.0,
    "terminal_reward": 100.0,
}
