"""Heuristic action selector for the multi-wave scheduling environment."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv


ACTION_RECOVERY = 0
ACTION_FUEL = 1
ACTION_ARM = 2
ACTION_LAUNCH = 3


class WaveHeuristicSolver:
    """Rule-based scheduler for the multi-wave sortie objective.

    The policy keeps launch actions immediate, balances fuel and arming
    personnel usage, and favors aircraft that can be completed within the
    nearest launch window before spending resources on harder misses.
    """

    def __init__(self, env: CarrierAircraftSchedulingEnv):
        self.env = env
        self._fuel_mean = float(env.config["fuel_time_mean"])
        self._arm_unit_mean = float(env.config["arm_unit_time_mean"])

    def choose_action(self) -> Optional[Dict[str, int]]:
        mask = self.env.get_action_mask()
        high_mask = mask["high_level"]
        if not any(high_mask):
            return None

        if high_mask[ACTION_LAUNCH]:
            aircraft_id = self._choose_launch_aircraft(mask["low_level_by_high"][ACTION_LAUNCH])
            return {"high_level": ACTION_LAUNCH, "aircraft_id": aircraft_id}

        fuel_candidate = None
        arm_candidate = None
        if high_mask[ACTION_FUEL]:
            fuel_candidate = self._choose_fuel_aircraft(mask["low_level_by_high"][ACTION_FUEL])
        if high_mask[ACTION_ARM]:
            arm_candidate = self._choose_arm_aircraft(mask["low_level_by_high"][ACTION_ARM])

        if fuel_candidate is not None or arm_candidate is not None:
            if fuel_candidate is None:
                return {"high_level": ACTION_ARM, "aircraft_id": arm_candidate}
            if arm_candidate is None:
                return {"high_level": ACTION_FUEL, "aircraft_id": fuel_candidate}
            high_level, aircraft_id = self._choose_service_action(fuel_candidate, arm_candidate)
            return {"high_level": high_level, "aircraft_id": aircraft_id}

        if high_mask[ACTION_RECOVERY]:
            aircraft_id = self._choose_recovery_aircraft(mask["low_level_by_high"][ACTION_RECOVERY])
            return {"high_level": ACTION_RECOVERY, "aircraft_id": aircraft_id}

        return None

    def _choose_launch_aircraft(self, low_mask: List[int]) -> int:
        return min(self._candidate_ids(low_mask), key=self._launch_priority)

    def _choose_recovery_aircraft(self, low_mask: List[int]) -> int:
        return min(self._candidate_ids(low_mask), key=self._recovery_priority)

    def _choose_fuel_aircraft(self, low_mask: List[int]) -> int:
        return min(self._candidate_ids(low_mask), key=self._fuel_priority)

    def _choose_arm_aircraft(self, low_mask: List[int]) -> int:
        return min(self._candidate_ids(low_mask), key=self._arm_priority)

    def _candidate_ids(self, low_mask: List[int]) -> List[int]:
        return [aircraft_id for aircraft_id, allowed in enumerate(low_mask) if allowed]

    def _launch_priority(self, aircraft_id: int) -> Tuple[float, int, int, float, int]:
        aircraft = self.env.aircraft[aircraft_id]
        ready_time = aircraft.launch_ready if aircraft.launch_ready is not None else self.env.time
        return (
            ready_time,
            aircraft.sorties_completed,
            -aircraft.missed_sorties,
            aircraft.launch_wait,
            aircraft_id,
        )

    def _recovery_priority(self, aircraft_id: int) -> Tuple[int, int, int, int]:
        aircraft = self.env.aircraft[aircraft_id]
        return (
            aircraft.sorties_completed,
            -aircraft.arm_quantity_required,
            -aircraft.missed_sorties,
            aircraft_id,
        )

    def _slack(self, aircraft_id: int) -> float:
        aircraft = self.env.aircraft[aircraft_id]
        remaining = 0.0
        if aircraft.fuel_status == 0:
            remaining = max(remaining, self._fuel_mean)
        if aircraft.arm_status == 1:
            remaining = max(remaining, aircraft.arm_remaining)
        if aircraft.arm_status == 0:
            remaining = max(
                remaining,
                self._expected_ammo_pipeline_time()
                +
                aircraft.arm_quantity_required * self._arm_unit_mean
                + self.env._spot_transfer_time(aircraft.spot_id),
            )
        time_to_launch = self._time_until_group_launch_deadline(aircraft.group, remaining)
        return time_to_launch - remaining

    def _time_until_group_launch_deadline(self, group: str, required_work: float = 0.0) -> float:
        wave_index = self.env.current_wave_index
        if group == self.env.active_launch_group:
            current_deadline = (wave_index + 1) * self.env.wave_interval
            current_time_left = max(0.0, current_deadline - self.env.time)
            launch_queue_time = self._active_launch_queue_time(group)
            if required_work + launch_queue_time <= current_time_left:
                return current_time_left
            next_same_group_deadline = (wave_index + 3) * self.env.wave_interval
            if next_same_group_deadline <= self.env.simulation_duration + self.env.wave_interval:
                return max(0.0, next_same_group_deadline - self.env.time)
            return float("inf")
        else:
            launch_wave = wave_index + 1
            while launch_wave * self.env.wave_interval <= self.env.simulation_duration:
                next_group = "A" if launch_wave % 2 == 0 else "B"
                if next_group == group:
                    break
                launch_wave += 1
            else:
                return float("inf")

        deadline = (launch_wave + 1) * self.env.wave_interval
        return max(0.0, deadline - self.env.time)

    def _active_launch_queue_time(self, group: str) -> float:
        queued = sum(
            1
            for aircraft in self.env.aircraft
            if (
                aircraft.group == group
                and not aircraft.is_airborne
                and aircraft.launch_status == 1
            )
        )
        return (queued + 1) * float(self.env.config["launch_time"])

    def _expected_ammo_pipeline_time(self) -> float:
        extract = (
            float(self.env.config["ammo_extract_time_min"])
            + float(self.env.config["ammo_extract_time_max"])
        ) / 2.0
        lower = float(self.env.config["lower_lift_time_mean"])
        upper = float(self.env.config["upper_lift_time_mean"])
        return extract + lower + upper

    def _fuel_priority(self, aircraft_id: int) -> Tuple[int, float, float, int, float, int, int]:
        aircraft = self.env.aircraft[aircraft_id]
        remaining = max(self._fuel_work(aircraft), self._arm_work(aircraft))
        time_to_deadline = self._time_until_group_launch_deadline(aircraft.group, remaining)
        slack = time_to_deadline - remaining
        return (
            int(slack < 0.0),
            time_to_deadline,
            remaining,
            0 if aircraft.arm_status == 2 else 1,
            -aircraft.fuel_wait,
            aircraft.sorties_completed,
            aircraft_id,
        )

    def _arm_priority(self, aircraft_id: int) -> Tuple[int, float, float, int, float, int, int]:
        aircraft = self.env.aircraft[aircraft_id]
        remaining = max(self._fuel_work(aircraft), self._arm_work(aircraft))
        time_to_deadline = self._time_until_group_launch_deadline(aircraft.group, remaining)
        slack = time_to_deadline - remaining
        return (
            int(slack < 0.0),
            time_to_deadline,
            remaining,
            aircraft.arm_quantity_required,
            self.env._spot_transfer_time(aircraft.spot_id),
            aircraft.sorties_completed,
            aircraft_id,
        )

    def _choose_service_action(self, fuel_candidate: int, arm_candidate: int) -> Tuple[int, int]:
        if self._should_reserve_personnel_for_fuel():
            return ACTION_FUEL, fuel_candidate

        fuel_slack = self._slack(fuel_candidate)
        arm_slack = self._slack(arm_candidate)
        if arm_slack + 3.0 < fuel_slack:
            return ACTION_ARM, arm_candidate
        if fuel_slack + 3.0 < arm_slack:
            return ACTION_FUEL, fuel_candidate

        fuel_work = self._fuel_work(self.env.aircraft[fuel_candidate])
        arm_work = self._arm_work(self.env.aircraft[arm_candidate])
        if arm_work > fuel_work * 1.25:
            return ACTION_ARM, arm_candidate
        return ACTION_FUEL, fuel_candidate

    def _should_reserve_personnel_for_fuel(self) -> bool:
        fuel_waiting = sum(
            1
            for aircraft in self.env.aircraft
            if (
                not aircraft.is_airborne
                and aircraft.parking_status == 2
                and aircraft.recovery_status == 2
                and aircraft.fuel_status == 0
            )
        )
        if fuel_waiting <= 0:
            return False

        active_fuel = int(self.env.config["num_fuel_servers"]) - self.env.free_fuel_servers
        active_arm = sum(1 for aircraft in self.env.aircraft if aircraft.arm_status == 1)
        if active_fuel < min(4, fuel_waiting):
            return True

        target_fuel_for_arm_load = math.ceil(active_arm / self._arm_to_fuel_concurrency_ratio())
        return active_fuel < min(fuel_waiting, target_fuel_for_arm_load)

    def _arm_to_fuel_concurrency_ratio(self) -> float:
        return max(1.0, self._expected_arm_work() / max(0.1, self._fuel_mean))

    def _expected_arm_work(self) -> float:
        quantity_values = list(self.env.config["arm_quantity_values"])
        quantity_probs = list(self.env.config["arm_quantity_probs"])
        total_prob = sum(float(prob) for prob in quantity_probs)
        expected_quantity = sum(
            int(value) * float(prob)
            for value, prob in zip(quantity_values, quantity_probs)
        ) / total_prob
        expected_spot = sum(self.env.parking_transfer_times) / len(self.env.parking_transfer_times)
        return self._expected_ammo_pipeline_time() + expected_quantity * self._arm_unit_mean + expected_spot

    def _fuel_work(self, aircraft) -> float:
        if aircraft.fuel_status == 0:
            return self._fuel_mean
        if aircraft.fuel_status == 1:
            return aircraft.fuel_remaining
        return 0.0

    def _arm_work(self, aircraft) -> float:
        if aircraft.arm_status == 0:
            return (
                self._expected_ammo_pipeline_time()
                + aircraft.arm_quantity_required * self._arm_unit_mean
                + self.env._spot_transfer_time(aircraft.spot_id)
            )
        if aircraft.arm_status == 1:
            return aircraft.arm_remaining
        return 0.0
