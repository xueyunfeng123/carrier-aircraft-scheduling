"""Event-driven carrier aircraft scheduling environment.

The environment follows a Gym-like interface but intentionally has no third
party dependency. Actions can be dictionaries such as
{"high_level": 1, "aircraft_id": 3} or tuples such as (1, 3).
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from env.config import ACTION_TO_INDEX, DEFAULT_CONFIG, HIGH_LEVEL_ACTIONS


@dataclass(order=True)
class Event:
    """A future completion event stored in a priority queue."""

    time: float
    sequence: int
    event_type: str = field(compare=False)
    aircraft_id: int = field(compare=False)


@dataclass
class AircraftRecord:
    """State and timing information for one aircraft."""

    group: str = "A"
    spot_id: int = -1
    parking_status: int = 0
    is_airborne: bool = False
    pending_recovery: bool = False
    sorties_completed: int = 0
    missed_sorties: int = 0
    last_missed_wave: Optional[int] = None
    recovery_status: int = 0
    fuel_status: int = 0
    arm_status: int = 0
    arm_stage: int = 0
    launch_status: int = 0
    arm_quantity_required: int = 0
    fuel_remaining: float = 0.0
    arm_remaining: float = 0.0
    fuel_wait: float = 0.0
    arm_wait: float = 0.0
    launch_wait: float = 0.0
    recovery_start: Optional[float] = None
    recovery_end: Optional[float] = None
    park_start: Optional[float] = None
    park_end: Optional[float] = None
    fuel_start: Optional[float] = None
    fuel_end: Optional[float] = None
    arm_start: Optional[float] = None
    ammo_extract_start: Optional[float] = None
    ammo_to_assembly_end: Optional[float] = None
    deck_arm_start: Optional[float] = None
    arm_end: Optional[float] = None
    launch_ready: Optional[float] = None
    launch_start: Optional[float] = None
    launch_end: Optional[float] = None

    def as_vector(self) -> List[float]:
        return [
            0 if self.group == "A" else 1,
            self.spot_id,
            self.parking_status,
            int(self.is_airborne),
            int(self.pending_recovery),
            self.sorties_completed,
            self.missed_sorties,
            self.recovery_status,
            self.fuel_status,
            self.arm_status,
            self.arm_stage,
            self.launch_status,
            self.arm_quantity_required,
            self.fuel_remaining,
            self.arm_remaining,
            self.fuel_wait,
            self.arm_wait,
            self.launch_wait,
        ]


class CarrierAircraftSchedulingEnv:
    """Event-triggered scheduling environment for carrier aircraft operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.num_aircraft = int(self.config["num_aircraft"])
        self.group_size = int(self.config["group_size"])
        if self.num_aircraft != self.group_size * 2:
            raise ValueError("num_aircraft must equal group_size * 2 for A/B wave simulation")
        self.num_parking_spots = int(self.config["num_parking_spots"])
        if self.num_parking_spots < self.num_aircraft:
            raise ValueError("num_parking_spots must be at least num_aircraft")
        self.rng = random.Random()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)

        self.time = 0.0
        self.event_sequence = 0
        self.event_queue: List[Event] = []
        self.wave_interval = float(self.config["wave_interval"])
        self.simulation_duration = float(self.config["simulation_duration"])
        self.current_wave_index = 0
        self.active_launch_group = "A"
        self.active_recovery_group: Optional[str] = None
        self.wave_records: List[Dict[str, Any]] = []
        self.missed_sortie_records: List[Dict[str, Any]] = []
        self.parking_transfer_times = self._build_parking_transfer_times()
        self.parking_occupancy: List[Optional[int]] = [None] * self.num_parking_spots
        self.aircraft: List[AircraftRecord] = []
        for aircraft_id in range(self.num_aircraft):
            group = "A" if aircraft_id < self.group_size else "B"
            spot_id = aircraft_id
            self.parking_occupancy[spot_id] = aircraft_id
            self.aircraft.append(
                AircraftRecord(
                    group=group,
                    spot_id=spot_id,
                    parking_status=2,
                    is_airborne=False,
                    pending_recovery=False,
                    recovery_status=2,
                    fuel_status=2,
                    arm_status=2,
                    launch_status=1,
                )
            )
        self._start_wave(0)
        self._schedule_wave_events()

        self.free_recovery_channels = int(self.config["num_recovery_channels"])
        self.free_launch_channels = int(self.config["num_launch_channels"])
        self.free_fuel_servers = int(self.config["num_fuel_servers"])
        self.free_arm_vehicles = int(self.config["num_arm_vehicles"])
        self.free_ammo_transport_vehicles = int(self.config["num_ammo_transport_vehicles"])
        self.free_lower_weapon_lifts = int(self.config["num_lower_weapon_lifts"])
        self.free_upper_weapon_lifts = int(self.config["num_upper_weapon_lifts"])
        self.free_personnel = int(self.config["num_personnel"])

        self.total_reward = 0.0
        self.done = False
        self.last_completed_counts = {
            "recovery": 0,
            "fuel": 0,
            "arm": 0,
            "launch": 0,
        }
        return self.get_state()

    def step(self, action: Optional[Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one dispatch action or advance time if no action is available.

        If actions are available and action is None, the environment applies the
        idle penalty. If no actions are available, action is ignored and time is
        advanced to the next event.
        """

        if self.done:
            return self.get_state(), 0.0, True, self._make_info()

        high_mask = self.get_high_level_action_mask()
        has_action = any(high_mask)

        if not has_action:
            reward, completed = self._advance_time_to_next_event()
            self.total_reward += reward
            info = self._make_info(completed_counts=completed)
            return self.get_state(), reward, self.done, info

        if action is None:
            reward = -float(self.config["eta_idle"])
            self.total_reward += reward
            info = self._make_info(invalid_action=True, message="idle_when_action_available")
            return self.get_state(), reward, self.done, info

        parsed = self._parse_action(action)
        if parsed is None:
            reward = -float(self.config["invalid_action_penalty"])
            self.total_reward += reward
            info = self._make_info(invalid_action=True, message="invalid_action_format")
            return self.get_state(), reward, self.done, info

        high_level, aircraft_id = parsed
        if not self._is_action_valid(high_level, aircraft_id):
            reward = -float(self.config["invalid_action_penalty"])
            self.total_reward += reward
            info = self._make_info(invalid_action=True, message="illegal_action")
            return self.get_state(), reward, self.done, info

        action_name = HIGH_LEVEL_ACTIONS[high_level]
        if action_name == "R":
            self._start_recovery(aircraft_id)
        elif action_name == "F":
            self._start_fueling(aircraft_id)
        elif action_name == "M":
            self._start_arming(aircraft_id)
        elif action_name == "L":
            self._start_launch(aircraft_id)

        info = self._make_info(action_started=action_name)
        return self.get_state(), 0.0, self.done, info

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "aircraft": [record.as_vector() for record in self.aircraft],
            "resources": {
                "recovery_channels": self.free_recovery_channels,
                "fuel_servers": self.free_fuel_servers,
                "arm_vehicles": self.free_arm_vehicles,
                "ammo_transport_vehicles": self.free_ammo_transport_vehicles,
                "lower_weapon_lifts": self.free_lower_weapon_lifts,
                "upper_weapon_lifts": self.free_upper_weapon_lifts,
                "personnel": self.free_personnel,
                "launch_channels": self.free_launch_channels,
                "free_parking_spots": sum(1 for item in self.parking_occupancy if item is None),
            },
            "event_queue_size": len(self.event_queue),
            "parking_transfer_times": list(self.parking_transfer_times),
            "wave": {
                "index": self.current_wave_index,
                "active_launch_group": self.active_launch_group,
                "active_recovery_group": self.active_recovery_group,
                "next_wave_time": self._next_wave_time(),
            },
        }

    def get_action_mask(self, high_level_action: Optional[int] = None) -> Dict[str, Any]:
        high_level_mask = self.get_high_level_action_mask()
        low_level_masks = {
            action_id: self.get_low_level_action_mask(action_id)
            for action_id in HIGH_LEVEL_ACTIONS
        }
        result: Dict[str, Any] = {
            "high_level": high_level_mask,
            "low_level_by_high": low_level_masks,
        }
        if high_level_action is not None:
            result["low_level"] = self.get_low_level_action_mask(high_level_action)
        return result

    def get_high_level_action_mask(self) -> List[int]:
        candidates = self._candidate_sets()
        return [
            int(len(candidates["R"]) > 0 and self.free_recovery_channels > 0),
            int(
                len(candidates["F"]) > 0
                and self.free_fuel_servers > 0
                and self.free_personnel >= int(self.config["fuel_personnel_required"])
            ),
            int(
                len(candidates["M"]) > 0
                and self.free_ammo_transport_vehicles > 0
                and self.free_lower_weapon_lifts > 0
                and self.free_personnel >= int(self.config["arm_personnel_required"])
            ),
            int(len(candidates["L"]) > 0 and self.free_launch_channels > 0),
        ]

    def get_low_level_action_mask(self, high_level_action: int) -> List[int]:
        action_name = HIGH_LEVEL_ACTIONS.get(high_level_action)
        mask = [0] * self.num_aircraft
        if action_name is None:
            return mask
        for aircraft_id in self._candidate_sets()[action_name]:
            mask[aircraft_id] = 1
        return mask

    def get_aircraft_timing_records(self) -> List[Dict[str, Optional[float]]]:
        records = []
        for aircraft_id, item in enumerate(self.aircraft):
            records.append(
                {
                    "aircraft_id": aircraft_id,
                    "group": item.group,
                    "spot_id": item.spot_id,
                    "spot_transfer_time": self._spot_transfer_time(item.spot_id),
                    "parking_status": item.parking_status,
                    "sorties_completed": item.sorties_completed,
                    "missed_sorties": item.missed_sorties,
                    "is_airborne": item.is_airborne,
                    "pending_recovery": item.pending_recovery,
                    "arm_quantity_required": item.arm_quantity_required,
                    "arm_stage": item.arm_stage,
                    "recovery_start": item.recovery_start,
                    "recovery_end": item.recovery_end,
                    "park_start": item.park_start,
                    "park_end": item.park_end,
                    "fuel_start": item.fuel_start,
                    "fuel_end": item.fuel_end,
                    "arm_start": item.arm_start,
                    "ammo_extract_start": item.ammo_extract_start,
                    "ammo_to_assembly_end": item.ammo_to_assembly_end,
                    "deck_arm_start": item.deck_arm_start,
                    "arm_end": item.arm_end,
                    "launch_ready": item.launch_ready,
                    "launch_start": item.launch_start,
                    "launch_end": item.launch_end,
                }
            )
        return records

    def get_wave_records(self) -> List[Dict[str, Any]]:
        return list(self.wave_records)

    def get_missed_sortie_records(self) -> List[Dict[str, Any]]:
        return list(self.missed_sortie_records)

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        total_sorties = sum(aircraft.sorties_completed for aircraft in self.aircraft)
        total_missed = sum(aircraft.missed_sorties for aircraft in self.aircraft)
        group_metrics = {}
        for group in ("A", "B"):
            group_aircraft = [aircraft for aircraft in self.aircraft if aircraft.group == group]
            group_metrics[group] = {
                "sorties_completed": sum(aircraft.sorties_completed for aircraft in group_aircraft),
                "missed_sorties": sum(aircraft.missed_sorties for aircraft in group_aircraft),
            }
        return {
            "simulation_duration": self.simulation_duration,
            "total_sorties_completed": total_sorties,
            "total_missed_sorties": total_missed,
            "group_metrics": group_metrics,
        }

    def _advance_time_to_next_event(self) -> Tuple[float, Dict[str, int]]:
        if not self.event_queue:
            self.done = self.time >= self.simulation_duration
            completed = {"recovery": 0, "fuel": 0, "arm": 0, "launch": 0}
            return 0.0, completed

        next_time = self.event_queue[0].time
        if next_time > self.simulation_duration:
            delta = self.simulation_duration - self.time
            self._update_waiting_and_remaining(delta)
            self.time = self.simulation_duration
            self.done = True
            completed = {"recovery": 0, "fuel": 0, "arm": 0, "launch": 0}
            reward = self._calculate_time_reward(delta, completed)
            return reward, completed

        delta = next_time - self.time
        self._update_waiting_and_remaining(delta)
        self.time = next_time
        completed = self._process_events_at_current_time()

        reward = self._calculate_time_reward(delta, completed)
        if self.time >= self.simulation_duration:
            self.done = True
            reward += float(self.config["terminal_reward"])
        self.last_completed_counts = completed
        return reward, completed

    def _process_events_at_current_time(self) -> Dict[str, int]:
        completed = {"recovery": 0, "fuel": 0, "arm": 0, "launch": 0}
        events: List[Event] = []
        while self.event_queue and self.event_queue[0].time == self.time:
            events.append(heapq.heappop(self.event_queue))
        events.sort(key=lambda item: item.event_type == "wave_start")

        for event in events:
            if event.event_type == "wave_start":
                self._start_wave(int(event.time / self.wave_interval))
                continue

            aircraft = self.aircraft[event.aircraft_id]
            if event.event_type == "recover_done":
                aircraft.recovery_status = 2
                aircraft.recovery_end = self.time
                aircraft.pending_recovery = False
                aircraft.is_airborne = False
                aircraft.launch_status = 0
                aircraft.parking_status = 1
                aircraft.spot_id = self._assign_parking_spot(event.aircraft_id)
                aircraft.park_start = self.time
                aircraft.arm_quantity_required = self._sample_arm_quantity()
                self._push_event(
                    self.time,
                    "park_done",
                    event.aircraft_id,
                )
                self.free_recovery_channels += 1
                completed["recovery"] += 1
            elif event.event_type == "park_done":
                aircraft.parking_status = 2
                aircraft.park_end = self.time
            elif event.event_type == "fuel_done":
                aircraft.fuel_status = 2
                aircraft.fuel_remaining = 0.0
                aircraft.fuel_end = self.time
                self.free_fuel_servers += 1
                self.free_personnel += int(self.config["fuel_personnel_required"])
                completed["fuel"] += 1
            elif event.event_type == "arm_done":
                aircraft.arm_status = 2
                aircraft.arm_stage = 4
                aircraft.arm_remaining = 0.0
                aircraft.arm_end = self.time
                self.free_arm_vehicles += 1
                self.free_upper_weapon_lifts += 1
                self.free_personnel += int(self.config["arm_personnel_required"])
                completed["arm"] += 1
            elif event.event_type == "ammo_to_assembly_done":
                aircraft.arm_stage = 2
                aircraft.ammo_to_assembly_end = self.time
                self.free_ammo_transport_vehicles += 1
                self.free_lower_weapon_lifts += 1
            elif event.event_type == "launch_done":
                aircraft.launch_status = 3
                aircraft.launch_end = self.time
                aircraft.sorties_completed += 1
                aircraft.is_airborne = True
                aircraft.pending_recovery = False
                self._release_parking_spot(event.aircraft_id)
                aircraft.parking_status = 0
                aircraft.spot_id = -1
                aircraft.recovery_status = 0
                aircraft.fuel_status = 0
                aircraft.arm_status = 0
                aircraft.arm_stage = 0
                aircraft.arm_quantity_required = 0
                self.free_launch_channels += 1
                completed["launch"] += 1

        for aircraft in self.aircraft:
            if (
                not aircraft.is_airborne
                and aircraft.parking_status == 2
                and aircraft.recovery_status == 2
                and aircraft.fuel_status == 2
                and aircraft.arm_status == 2
                and aircraft.launch_status == 0
            ):
                aircraft.launch_status = 1
                aircraft.launch_ready = self.time
        self._try_start_waiting_deck_arming()
        return completed

    def _start_recovery(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        aircraft.recovery_status = 1
        aircraft.recovery_start = self.time
        self.free_recovery_channels -= 1
        self._push_event(self.time + float(self.config["recovery_time"]), "recover_done", aircraft_id)

    def _start_fueling(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        duration = self._sample_duration("fuel_time_mean", "fuel_time_std")
        aircraft.fuel_status = 1
        aircraft.fuel_start = self.time
        aircraft.fuel_remaining = duration
        self.free_fuel_servers -= 1
        self.free_personnel -= int(self.config["fuel_personnel_required"])
        self._push_event(self.time + duration, "fuel_done", aircraft_id)

    def _start_arming(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        first_stage_duration = self._sample_ammo_extract_time() + self._sample_duration(
            "lower_lift_time_mean",
            "lower_lift_time_std",
        )
        deck_stage_duration = (
            self._sample_duration("upper_lift_time_mean", "upper_lift_time_std")
            + self._spot_transfer_time(aircraft.spot_id)
            + self._sample_arming_duration(aircraft.arm_quantity_required)
        )
        aircraft.arm_status = 1
        aircraft.arm_stage = 1
        aircraft.arm_start = self.time
        aircraft.ammo_extract_start = self.time
        aircraft.arm_remaining = first_stage_duration + deck_stage_duration
        self.free_ammo_transport_vehicles -= 1
        self.free_lower_weapon_lifts -= 1
        self.free_personnel -= int(self.config["arm_personnel_required"])
        self._push_event(self.time + first_stage_duration, "ammo_to_assembly_done", aircraft_id)

    def _start_launch(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        aircraft.launch_status = 2
        aircraft.launch_start = self.time
        self.free_launch_channels -= 1
        self._push_event(self.time + float(self.config["launch_time"]), "launch_done", aircraft_id)

    def _try_start_waiting_deck_arming(self) -> None:
        for aircraft_id, aircraft in enumerate(self.aircraft):
            if (
                aircraft.arm_status == 1
                and aircraft.arm_stage == 2
                and self.free_arm_vehicles > 0
                and self.free_upper_weapon_lifts > 0
            ):
                self._start_deck_arming(aircraft_id)

    def _start_deck_arming(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        duration = (
            self._sample_duration("upper_lift_time_mean", "upper_lift_time_std")
            + self._spot_transfer_time(aircraft.spot_id)
            + self._sample_arming_duration(aircraft.arm_quantity_required)
        )
        aircraft.arm_stage = 3
        aircraft.deck_arm_start = self.time
        self.free_arm_vehicles -= 1
        self.free_upper_weapon_lifts -= 1
        self._push_event(self.time + duration, "arm_done", aircraft_id)

    def _candidate_sets(self) -> Dict[str, List[int]]:
        candidates = {"R": [], "F": [], "M": [], "L": []}
        for aircraft_id, aircraft in enumerate(self.aircraft):
            if (
                aircraft.group == self.active_recovery_group
                and aircraft.pending_recovery
                and aircraft.recovery_status == 0
            ):
                candidates["R"].append(aircraft_id)
            if (
                not aircraft.is_airborne
                and aircraft.parking_status == 2
                and aircraft.recovery_status == 2
                and aircraft.fuel_status == 0
            ):
                candidates["F"].append(aircraft_id)
            if (
                not aircraft.is_airborne
                and aircraft.parking_status == 2
                and aircraft.recovery_status == 2
                and aircraft.arm_status == 0
            ):
                candidates["M"].append(aircraft_id)
            if (
                aircraft.group == self.active_launch_group
                and not aircraft.is_airborne
                and aircraft.parking_status == 2
                and aircraft.fuel_status == 2
                and aircraft.arm_status == 2
                and aircraft.launch_status == 1
            ):
                candidates["L"].append(aircraft_id)
        return candidates

    def _schedule_wave_events(self) -> None:
        wave_index = 1
        while wave_index * self.wave_interval <= self.simulation_duration:
            self._push_event(wave_index * self.wave_interval, "wave_start", -1)
            wave_index += 1

    def _start_wave(self, wave_index: int) -> None:
        if wave_index > self.current_wave_index:
            self._record_missed_sorties(self.current_wave_index, self.active_launch_group)

        self.current_wave_index = wave_index
        self.active_launch_group = "A" if wave_index % 2 == 0 else "B"
        self.active_recovery_group = None if wave_index == 0 else ("B" if wave_index % 2 == 0 else "A")

        if self.active_recovery_group is not None:
            for aircraft in self.aircraft:
                if aircraft.group == self.active_recovery_group and aircraft.is_airborne:
                    aircraft.pending_recovery = True
                    aircraft.recovery_status = 0

        self.wave_records.append(
            {
                "wave_index": wave_index,
                "time": self.time,
                "launch_group": self.active_launch_group,
                "recovery_group": self.active_recovery_group,
            }
        )

    def _record_missed_sorties(self, wave_index: int, group: str) -> None:
        for aircraft_id, aircraft in enumerate(self.aircraft):
            if (
                aircraft.group == group
                and not aircraft.is_airborne
                and aircraft.launch_status != 2
                and aircraft.last_missed_wave != wave_index
            ):
                aircraft.missed_sorties += 1
                aircraft.last_missed_wave = wave_index
                self.missed_sortie_records.append(
                    {
                        "wave_index": wave_index,
                        "time": self.time,
                        "group": group,
                        "aircraft_id": aircraft_id,
                    }
                )

    def _next_wave_time(self) -> Optional[float]:
        next_time = (self.current_wave_index + 1) * self.wave_interval
        if next_time > self.simulation_duration:
            return None
        return next_time

    def _build_parking_transfer_times(self) -> List[float]:
        times = []
        base = float(self.config["parking_base_transfer_time"])
        step = float(self.config["parking_ring_time_step"])
        for spot_id in range(self.num_parking_spots):
            if spot_id == 0:
                layer = 0
            else:
                layer = (spot_id + 1) // 2
            times.append(base + layer * step)
        return times

    def _spot_transfer_time(self, spot_id: int) -> float:
        if spot_id < 0:
            return 0.0
        return float(self.parking_transfer_times[spot_id])

    def _assign_parking_spot(self, aircraft_id: int) -> int:
        free_spots = [
            spot_id
            for spot_id, occupied_by in enumerate(self.parking_occupancy)
            if occupied_by is None
        ]
        if not free_spots:
            raise RuntimeError("no free parking spot available")
        spot_id = self._select_parking_spot(aircraft_id, free_spots)
        self.parking_occupancy[spot_id] = aircraft_id
        return spot_id

    def _select_parking_spot(self, aircraft_id: int, free_spots: List[int]) -> int:
        aircraft = self.aircraft[aircraft_id]
        deadline = self._time_until_group_launch(aircraft.group)
        if deadline <= self.wave_interval:
            return min(free_spots, key=lambda spot_id: (self._spot_transfer_time(spot_id), spot_id))
        return min(free_spots, key=lambda spot_id: (spot_id, self._spot_transfer_time(spot_id)))

    def _release_parking_spot(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        spot_id = aircraft.spot_id
        if 0 <= spot_id < len(self.parking_occupancy):
            if self.parking_occupancy[spot_id] == aircraft_id:
                self.parking_occupancy[spot_id] = None

    def _time_until_group_launch(self, group: str) -> float:
        if group == self.active_launch_group:
            return 0.0
        next_wave = self.current_wave_index + 1
        while next_wave * self.wave_interval <= self.simulation_duration:
            next_group = "A" if next_wave % 2 == 0 else "B"
            if next_group == group:
                return next_wave * self.wave_interval - self.time
            next_wave += 1
        return float("inf")

    def _push_event(self, time: float, event_type: str, aircraft_id: int) -> None:
        self.event_sequence += 1
        heapq.heappush(
            self.event_queue,
            Event(time=time, sequence=self.event_sequence, event_type=event_type, aircraft_id=aircraft_id),
        )

    def _sample_duration(self, mean_key: str, std_key: str) -> float:
        mean = float(self.config[mean_key])
        std = float(self.config[std_key])
        min_process_time = float(self.config["min_process_time"])
        if std <= 0:
            return max(min_process_time, mean)
        return max(min_process_time, self.rng.gauss(mean, std))

    def _sample_ammo_extract_time(self) -> float:
        low = float(self.config["ammo_extract_time_min"])
        high = float(self.config["ammo_extract_time_max"])
        if high < low:
            raise ValueError("ammo_extract_time_max must be >= ammo_extract_time_min")
        return self.rng.uniform(low, high)

    def _sample_arming_duration(self, quantity: int) -> float:
        mean = float(self.config["arm_unit_time_mean"])
        variance = float(self.config["arm_unit_time_variance"])
        if variance < 0:
            raise ValueError("arm_unit_time_variance must be non-negative")
        std = math.sqrt(variance)
        min_process_time = float(self.config["min_process_time"])
        return sum(
            max(min_process_time, self.rng.gauss(mean, std))
            for _ in range(max(0, int(quantity)))
        )

    def _sample_arm_quantity(self) -> int:
        values = list(self.config["arm_quantity_values"])
        probs = list(self.config["arm_quantity_probs"])
        if len(values) != len(probs):
            raise ValueError("arm_quantity_values and arm_quantity_probs must have the same length")
        total_prob = sum(float(prob) for prob in probs)
        if total_prob <= 0:
            raise ValueError("arm_quantity_probs must sum to a positive value")

        draw = self.rng.random() * total_prob
        cumulative = 0.0
        for value, prob in zip(values, probs):
            cumulative += float(prob)
            if draw <= cumulative:
                return int(value)
        return int(values[-1])

    def _update_waiting_and_remaining(self, delta: float) -> None:
        if delta <= 0:
            return
        for aircraft in self.aircraft:
            if aircraft.recovery_status == 2 and aircraft.fuel_status == 0:
                aircraft.fuel_wait += delta
            if aircraft.recovery_status == 2 and aircraft.arm_status == 0:
                aircraft.arm_wait += delta
            if aircraft.launch_status == 1:
                aircraft.launch_wait += delta
            if aircraft.fuel_status == 1:
                aircraft.fuel_remaining = max(0.0, aircraft.fuel_remaining - delta)
            if aircraft.arm_status == 1 and aircraft.arm_stage in (1, 3):
                aircraft.arm_remaining = max(0.0, aircraft.arm_remaining - delta)

    def _calculate_time_reward(self, delta: float, completed: Dict[str, int]) -> float:
        return (
            -float(self.config["alpha"]) * delta
            + float(self.config["beta_recovery"]) * completed["recovery"]
            + float(self.config["beta_fuel"]) * completed["fuel"]
            + float(self.config["beta_arm"]) * completed["arm"]
            + float(self.config["beta_launch"]) * completed["launch"]
        )

    def _parse_action(self, action: Any) -> Optional[Tuple[int, int]]:
        if isinstance(action, dict):
            high_level = action.get("high_level")
            aircraft_id = action.get("aircraft_id")
        elif isinstance(action, (tuple, list)) and len(action) == 2:
            high_level, aircraft_id = action
        else:
            return None

        if isinstance(high_level, str):
            high_level = ACTION_TO_INDEX.get(high_level.upper())
        try:
            high_level_int = int(high_level)
            aircraft_id_int = int(aircraft_id)
        except (TypeError, ValueError):
            return None
        return high_level_int, aircraft_id_int

    def _is_action_valid(self, high_level: int, aircraft_id: int) -> bool:
        if high_level not in HIGH_LEVEL_ACTIONS:
            return False
        if not 0 <= aircraft_id < self.num_aircraft:
            return False
        high_mask = self.get_high_level_action_mask()
        if high_mask[high_level] != 1:
            return False
        low_mask = self.get_low_level_action_mask(high_level)
        return low_mask[aircraft_id] == 1

    def _all_launched(self) -> bool:
        return all(aircraft.launch_status == 3 for aircraft in self.aircraft)

    def _make_info(
        self,
        completed_counts: Optional[Dict[str, int]] = None,
        invalid_action: bool = False,
        message: str = "",
        action_started: Optional[str] = None,
    ) -> Dict[str, Any]:
        launched = sum(1 for aircraft in self.aircraft if aircraft.launch_status == 3)
        metrics = self.get_evaluation_metrics()
        return {
            "time": self.time,
            "makespan": None,
            "simulation_duration": self.simulation_duration,
            "completed_counts": completed_counts or dict(self.last_completed_counts),
            "launched": launched,
            "total_sorties_completed": metrics["total_sorties_completed"],
            "total_missed_sorties": metrics["total_missed_sorties"],
            "resources": self.get_state()["resources"],
            "invalid_action": invalid_action,
            "message": message,
            "action_started": action_started,
        }
