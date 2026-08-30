"""Classical dispatching-rule baselines for the scheduling environment."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv


ACTION_RECOVERY = 0
ACTION_FUEL = 1
ACTION_ARM = 2
ACTION_LAUNCH = 3


class PriorityRuleSolver:
    """Select one legal action using a classical priority rule."""

    VALID_RULES = {"fifo", "spt", "edd"}

    def __init__(self, env: CarrierAircraftSchedulingEnv, rule: str):
        if rule not in self.VALID_RULES:
            raise ValueError(f"unsupported priority rule: {rule}")
        self.env = env
        self.rule = rule

    def choose_action(self) -> Optional[Dict[str, int]]:
        candidates = self._legal_actions()
        if not candidates:
            return None

        if self.rule == "fifo":
            high_level, aircraft_id = min(candidates, key=self._fifo_key)
        elif self.rule == "spt":
            high_level, aircraft_id = min(candidates, key=self._spt_key)
        else:
            high_level, aircraft_id = min(candidates, key=self._edd_key)
        return {"high_level": high_level, "aircraft_id": aircraft_id}

    def _legal_actions(self) -> List[Tuple[int, int]]:
        mask = self.env.get_action_mask()
        return [
            (high_level, aircraft_id)
            for high_level, enabled in enumerate(mask["high_level"])
            if enabled
            for aircraft_id, allowed in enumerate(mask["low_level_by_high"][high_level])
            if allowed
        ]

    def _fifo_key(self, candidate: Tuple[int, int]) -> Tuple[float, int, int]:
        high_level, aircraft_id = candidate
        return (
            -self._waiting_time(high_level, aircraft_id),
            aircraft_id,
            high_level,
        )

    def _spt_key(self, candidate: Tuple[int, int]) -> Tuple[float, float, int, int]:
        high_level, aircraft_id = candidate
        return (
            self._expected_action_duration(high_level, aircraft_id),
            self._launch_deadline(aircraft_id),
            aircraft_id,
            high_level,
        )

    def _edd_key(self, candidate: Tuple[int, int]) -> Tuple[float, int, int]:
        high_level, aircraft_id = candidate
        return (
            self._launch_deadline(aircraft_id),
            aircraft_id,
            high_level,
        )

    def _waiting_time(self, high_level: int, aircraft_id: int) -> float:
        aircraft = self.env.aircraft[aircraft_id]
        if high_level == ACTION_RECOVERY:
            wave_start = self.env.current_wave_index * self.env.wave_interval
            return max(0.0, self.env.time - wave_start)
        if high_level == ACTION_FUEL:
            return aircraft.fuel_wait
        if high_level == ACTION_ARM:
            return aircraft.arm_wait
        return aircraft.launch_wait

    def _expected_action_duration(self, high_level: int, aircraft_id: int) -> float:
        config = self.env.config
        if high_level == ACTION_RECOVERY:
            return float(config["recovery_time"])
        if high_level == ACTION_FUEL:
            return float(config["fuel_time_mean"])
        if high_level == ACTION_LAUNCH:
            return float(config["launch_time"])

        aircraft = self.env.aircraft[aircraft_id]
        extract_mean = (
            float(config["ammo_extract_time_min"])
            + float(config["ammo_extract_time_max"])
        ) / 2.0
        return (
            extract_mean
            + float(config["lower_lift_time_mean"])
            + float(config["upper_lift_time_mean"])
            + self.env._spot_transfer_time(aircraft.spot_id)
            + aircraft.arm_quantity_required * float(config["arm_unit_time_mean"])
        )

    def _launch_deadline(self, aircraft_id: int) -> float:
        group = self.env.aircraft[aircraft_id].group
        wave_index = self.env.current_wave_index
        while wave_index * self.env.wave_interval < self.env.simulation_duration:
            launch_group = "A" if wave_index % 2 == 0 else "B"
            if launch_group == group:
                return min(
                    (wave_index + 1) * self.env.wave_interval,
                    self.env.simulation_duration,
                )
            wave_index += 1
        return float("inf")


class FIFOSolver(PriorityRuleSolver):
    """Dispatch the operation that has waited the longest."""

    def __init__(self, env: CarrierAircraftSchedulingEnv):
        super().__init__(env, "fifo")


class SPTSolver(PriorityRuleSolver):
    """Dispatch the legal operation with the shortest expected duration."""

    def __init__(self, env: CarrierAircraftSchedulingEnv):
        super().__init__(env, "spt")


class EDDSolver(PriorityRuleSolver):
    """Dispatch work for the aircraft with the earliest launch deadline."""

    def __init__(self, env: CarrierAircraftSchedulingEnv):
        super().__init__(env, "edd")
