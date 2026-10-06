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

from env.cbs_planner import CBSExpansionLimitError, RouteRequest
from env.config import ACTION_TO_INDEX, DEFAULT_CONFIG, HIGH_LEVEL_ACTIONS
from env.disruptions import (
    DisruptionSpec,
    build_disruption_schedule,
)
from env.project_deck_graph import (
    DeckOccupancy,
    DeckRouteReservation,
    ProjectDeckGraphAssumptions,
    ProjectDeckLayout,
    build_project_deck_layout,
)
from env.scenario import PROJECT_CORE_PROFILE, build_project_core_profile
from env.traffic_planner import SpaceTimeTrafficPlanner, TimedRoute


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

    group: str = "shared"
    initial_role: str = "mission"
    spot_id: int = -1
    parking_status: int = 0
    is_airborne: bool = False
    pending_recovery: bool = False
    sorties_completed: int = 0
    missed_sorties: int = 0
    last_missed_wave: Optional[int] = None
    recovery_status: int = 0
    fuel_status: int = 0
    fuel_level: float = 1.0
    inspection_status: int = 0
    arm_status: int = 0
    arm_stage: int = 0
    launch_status: int = 0
    arm_quantity_required: int = 0
    fuel_remaining: float = 0.0
    inspection_remaining: float = 0.0
    arm_remaining: float = 0.0
    fuel_wait: float = 0.0
    inspection_wait: float = 0.0
    arm_wait: float = 0.0
    launch_wait: float = 0.0
    recovery_start: Optional[float] = None
    recovery_end: Optional[float] = None
    park_start: Optional[float] = None
    park_end: Optional[float] = None
    fuel_start: Optional[float] = None
    fuel_end: Optional[float] = None
    inspection_start: Optional[float] = None
    inspection_end: Optional[float] = None
    arm_start: Optional[float] = None
    ammo_extract_start: Optional[float] = None
    ammo_to_assembly_end: Optional[float] = None
    deck_arm_start: Optional[float] = None
    arm_vehicle_ready_time: Optional[float] = None
    arm_end: Optional[float] = None
    launch_ready: Optional[float] = None
    launch_start: Optional[float] = None
    launch_end: Optional[float] = None
    taxi_start: Optional[float] = None
    taxi_end: Optional[float] = None
    assigned_launch_wave: Optional[int] = None
    recovery_due_wave: Optional[int] = None

    def as_vector(self) -> List[float]:
        return [
            1 if self.initial_role == "reserve" else 0,
            self.spot_id,
            self.parking_status,
            int(self.is_airborne),
            int(self.pending_recovery),
            self.sorties_completed,
            self.missed_sorties,
            self.recovery_status,
            self.fuel_status,
            self.fuel_level,
            self.inspection_status,
            self.arm_status,
            self.arm_stage,
            self.launch_status,
            self.arm_quantity_required,
            self.fuel_remaining,
            self.inspection_remaining,
            self.arm_remaining,
            self.fuel_wait,
            self.inspection_wait,
            self.arm_wait,
            self.launch_wait,
        ]


@dataclass
class ServiceVehicleRecord:
    vehicle_id: str
    service_type: str
    node_id: str
    busy_aircraft_id: Optional[int] = None


@dataclass(frozen=True)
class TractorApproach:
    tractor_id: str
    start_time: float
    route: TimedRoute


@dataclass(frozen=True)
class TowPlan:
    tractor_id: str
    approach_start: float
    approach: TimedRoute
    towing: TimedRoute


class CarrierAircraftSchedulingEnv:
    """Event-triggered scheduling environment for carrier aircraft operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.num_aircraft = int(self.config["num_aircraft"])
        self.group_size = int(self.config["group_size"])
        if self.num_aircraft < self.group_size * 2:
            raise ValueError(
                "num_aircraft must provide two wave loads; additional aircraft "
                "form the shared reserve"
            )
        self.num_reserve_aircraft = self.num_aircraft - self.group_size * 2
        self.num_parking_spots = int(self.config["num_parking_spots"])
        if self.num_parking_spots < self.num_aircraft:
            raise ValueError("num_parking_spots must be at least num_aircraft")
        scenario_profile = str(self.config.get("scenario_profile", PROJECT_CORE_PROFILE))
        if scenario_profile != PROJECT_CORE_PROFILE:
            raise ValueError(
                "CarrierAircraftSchedulingEnv currently executes only the "
                f"{PROJECT_CORE_PROFILE!r} profile; use YoonSortieGenerationEnv "
                "for the paper baseline"
            )
        self.scenario_profile = build_project_core_profile(self.config)
        self.spatial_graph_enabled = bool(self.config["spatial_graph_enabled"])
        self.deck_layout: Optional[ProjectDeckLayout] = None
        if self.spatial_graph_enabled:
            self.deck_layout = build_project_deck_layout(
                self.num_parking_spots,
                ProjectDeckGraphAssumptions(
                    launch_positions=int(self.config["num_launch_positions"]),
                    edge_travel_minutes=float(
                        self.config["deck_edge_travel_time"]
                    ),
                ),
            )
        self.rng = random.Random()
        self.disruption_rng = random.Random()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)
            self.disruption_rng.seed(int(seed) + 1_000_003)

        self.time = 0.0
        self.event_sequence = 0
        self.event_queue: List[Event] = []
        self.wave_interval = float(self.config["wave_interval"])
        self.simulation_duration = float(self.config["simulation_duration"])
        self.current_wave_index = 0
        self.active_launch_group = "A"
        self.active_recovery_group: Optional[str] = None
        self.event_log: List[Dict[str, Any]] = []
        self.wave_records: List[Dict[str, Any]] = []
        self.missed_sortie_records: List[Dict[str, Any]] = []
        self.parking_transfer_times = self._build_parking_transfer_times()
        self.parking_occupancy: List[Optional[int]] = [None] * self.num_parking_spots
        self.parking_release_times = [float("-inf")] * self.num_parking_spots
        self.runway_release_times = [float("-inf")] * int(
            self.config["num_launch_positions"]
        )
        self.deck_occupancy: Optional[DeckOccupancy] = None
        self.traffic_planner: Optional[SpaceTimeTrafficPlanner] = None
        self.active_deck_routes: Dict[int, DeckRouteReservation] = {}
        if self.deck_layout is not None:
            self.deck_occupancy = DeckOccupancy(self.deck_layout.graph)
            self.traffic_planner = SpaceTimeTrafficPlanner(
                self.deck_layout.graph,
                time_step=float(self.config["traffic_time_step"]),
                max_planning_horizon=float(
                    self.config["traffic_planning_horizon"]
                ),
            )
        self._candidate_cache: Optional[Dict[str, List[int]]] = None
        self._launch_route_cache: Dict[
            Tuple[int, Optional[int]],
            Optional[Tuple[str, Tuple[str, ...], float]],
        ] = {}
        self._recovery_route_cache: Dict[
            Tuple[int, Optional[int]],
            Optional[Tuple[str, Tuple[str, ...], float]],
        ] = {}
        self._tow_preview_cache: Dict[
            Tuple[int, str, str, float],
            Optional[TowPlan],
        ] = {}
        self._tractor_approach_cache: Dict[
            Tuple[str, float],
            Optional[TractorApproach],
        ] = {}
        self._service_route_cache: Dict[
            Tuple[str, int],
            Optional[TimedRoute],
        ] = {}
        self.aircraft: List[AircraftRecord] = []
        for aircraft_id in range(self.num_aircraft):
            spot_id = aircraft_id
            self.parking_occupancy[spot_id] = aircraft_id
            if self.deck_occupancy is not None and self.deck_layout is not None:
                self.deck_occupancy.occupy(
                    self.deck_layout.parking_node(spot_id),
                    aircraft_id,
                )
            self.aircraft.append(
                AircraftRecord(
                    initial_role=(
                        "reserve"
                        if aircraft_id >= self.group_size * 2
                        else "mission"
                    ),
                    spot_id=spot_id,
                    parking_status=2,
                    is_airborne=False,
                    pending_recovery=False,
                    recovery_status=2,
                    fuel_status=2,
                    fuel_level=float(self.config["initial_fuel_level"]),
                    inspection_status=2,
                    arm_status=2,
                    launch_status=1,
                )
            )
        self._start_wave(0)
        self._schedule_wave_events()

        self.free_recovery_channels = int(self.config["num_recovery_channels"])
        self.free_launch_channels = int(self.config["num_launch_channels"])
        self.free_fuel_servers = int(self.config["num_fuel_servers"])
        self.free_inspection_vehicles = int(
            self.config["num_inspection_vehicles"]
        )
        self.free_arm_vehicles = int(self.config["num_arm_vehicles"])
        self.free_ammo_transport_vehicles = int(self.config["num_ammo_transport_vehicles"])
        self.free_lower_weapon_lifts = int(self.config["num_lower_weapon_lifts"])
        self.free_upper_weapon_lifts = int(self.config["num_upper_weapon_lifts"])
        self.free_personnel = int(self.config["num_personnel"])
        self.service_vehicles = self._build_service_vehicle_fleet()
        self.active_service_vehicles: Dict[Tuple[str, int], str] = {}
        self.active_tow_vehicles: Dict[int, str] = {}
        self._cbs_route_dispatch_times: Dict[str, float] = {}
        self.cbs_replan_stats: Dict[str, float] = {
            "calls": 0,
            "improved_batches": 0,
            "fallbacks": 0,
            "sum_of_costs_before": 0.0,
            "sum_of_costs_after": 0.0,
        }
        self.active_disruptions: Dict[int, DisruptionSpec] = {}
        self.unavailable_vehicle_ids: set[str] = set()
        self.closed_runway_ids: set[int] = set()
        self.unavailable_aircraft_ids: set[int] = set()
        self.service_time_multipliers: Dict[str, float] = {
            "fuel": 1.0,
            "inspection": 1.0,
            "arm": 1.0,
        }
        self.disruption_schedule = build_disruption_schedule(
            self.config,
            self.disruption_rng,
            [vehicle.vehicle_id for vehicle in self.service_vehicles],
        )
        self.disruption_metrics: Dict[str, Any] = {
            "profile": str(self.config["disruption_profile"]),
            "scheduled": len(self.disruption_schedule),
            "started": 0,
            "ended": 0,
            "started_by_kind": {},
        }
        self._schedule_disruption_events()

        self.total_reward = 0.0
        self.done = False
        self.last_completed_counts = {
            "recovery": 0,
            "fuel": 0,
            "inspection": 0,
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

        high_level, aircraft_id, target_id, vehicle_id = parsed
        if not self._is_action_valid(
            high_level,
            aircraft_id,
            target_id,
            vehicle_id,
        ):
            reward = -float(self.config["invalid_action_penalty"])
            self.total_reward += reward
            info = self._make_info(invalid_action=True, message="illegal_action")
            return self.get_state(), reward, self.done, info

        action_name = HIGH_LEVEL_ACTIONS[high_level]
        if action_name == "R":
            self._start_recovery(aircraft_id, target_id)
        elif action_name == "F":
            self._start_fueling(aircraft_id, vehicle_id)
        elif action_name == "M":
            self._start_arming(aircraft_id, vehicle_id)
        elif action_name == "L":
            self._start_launch(aircraft_id, target_id)
        elif action_name == "I":
            self._start_inspection(aircraft_id, vehicle_id)

        self._invalidate_planning_cache()
        info = self._make_info(action_started=action_name)
        return self.get_state(), 0.0, self.done, info

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "aircraft": [record.as_vector() for record in self.aircraft],
            "resources": {
                "recovery_channels": self.free_recovery_channels,
                "fuel_servers": min(
                    self.free_fuel_servers,
                    len(self._available_service_vehicles("fuel")),
                ),
                "inspection_vehicles": min(
                    self.free_inspection_vehicles,
                    len(
                        self._available_service_vehicles(
                            "inspection"
                        )
                    ),
                ),
                "arm_vehicles": min(
                    self.free_arm_vehicles,
                    len(self._available_service_vehicles("arm")),
                ),
                "ammo_transport_vehicles": self.free_ammo_transport_vehicles,
                "lower_weapon_lifts": self.free_lower_weapon_lifts,
                "upper_weapon_lifts": self.free_upper_weapon_lifts,
                "personnel": self.free_personnel,
                "launch_channels": min(
                    self.free_launch_channels,
                    int(self.config["num_launch_positions"])
                    - len(self.closed_runway_ids),
                ),
                "free_parking_spots": sum(1 for item in self.parking_occupancy if item is None),
            },
            "event_queue_size": sum(
                event.event_type
                not in ("disruption_start", "disruption_end")
                for event in self.event_queue
            ),
            "parking_transfer_times": list(self.parking_transfer_times),
            "wave": {
                "index": self.current_wave_index,
                "active_launch_group": self.active_launch_group,
                "active_recovery_group": self.active_recovery_group,
                "launch_target": self.group_size,
                "launches_started": self.wave_records[-1]["launches_started"],
                "sorties_completed": self.wave_records[-1]["sorties_completed"],
                "pending_recovery_count": sum(
                    aircraft.pending_recovery for aircraft in self.aircraft
                ),
                "next_wave_time": self._next_wave_time(),
            },
            "scenario": {
                "name": self.scenario_profile.name,
                "fidelity": self.scenario_profile.fidelity,
                "fleet_model": "shared_dynamic",
                "reserve_aircraft": self.num_reserve_aircraft,
            },
            "deck": self._deck_state(),
            "service_vehicles": [
                {
                    "vehicle_id": vehicle.vehicle_id,
                    "service_type": vehicle.service_type,
                    "node_id": vehicle.node_id,
                    "busy_aircraft_id": vehicle.busy_aircraft_id,
                    "available": (
                        vehicle.busy_aircraft_id is None
                        and vehicle.vehicle_id
                        not in self.unavailable_vehicle_ids
                    ),
                }
                for vehicle in self.service_vehicles
            ],
            "disruptions": {
                "profile": self.disruption_metrics["profile"],
                "active": [
                    spec.as_dict()
                    for spec in self.active_disruptions.values()
                ],
                "unavailable_vehicle_ids": sorted(
                    self.unavailable_vehicle_ids
                ),
                "closed_runway_ids": sorted(
                    self.closed_runway_ids
                ),
                "unavailable_aircraft_ids": sorted(
                    self.unavailable_aircraft_ids
                ),
                "service_time_multipliers": dict(
                    self.service_time_multipliers
                ),
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

    def get_target_candidates(
        self,
        high_level: int,
        aircraft_id: int,
    ) -> List[Any]:
        action_name = HIGH_LEVEL_ACTIONS.get(high_level)
        if action_name == "R":
            if self.deck_layout is None:
                return [
                    spot_id
                    for spot_id, occupied_by in enumerate(
                        self.parking_occupancy
                    )
                    if occupied_by is None
                ][:1]
            options = [
                (route[2], spot_id)
                for spot_id, occupied_by in enumerate(self.parking_occupancy)
                if occupied_by is None
                and self._landing_target_available(spot_id)
                and (
                    route := self._find_recovery_route(
                        aircraft_id,
                        spot_id,
                    )
                )
                is not None
            ]
            return [spot_id for _, spot_id in sorted(options)]
        if action_name == "L":
            if self.deck_layout is None:
                for runway_id in range(
                    int(self.config["num_launch_positions"])
                ):
                    if self._launch_runway_available(
                        runway_id,
                        aircraft_id,
                    ):
                        return [runway_id]
                return []
            deadline = min(
                (self.current_wave_index + 1) * self.wave_interval,
                self.simulation_duration,
            )
            options = [
                (route[2], runway_id)
                for runway_id in range(len(self.deck_layout.launch_nodes))
                if self._launch_runway_available(runway_id, aircraft_id)
                and (
                    route := self._find_launch_route(
                        aircraft_id,
                        runway_id,
                    )
                )
                is not None
                and self.time
                + route[2]
                + float(self.config["launch_time"])
                <= deadline
            ]
            return [runway_id for _, runway_id in sorted(options)]
        service_type = {
            "F": "fuel",
            "M": "arm",
            "I": "inspection",
        }.get(action_name)
        if service_type is None:
            return []
        options = [
            (route.end_time, vehicle.vehicle_id)
            for vehicle in self.service_vehicles
            if vehicle.service_type == service_type
            and vehicle.busy_aircraft_id is None
            and vehicle.vehicle_id
            not in self.unavailable_vehicle_ids
            and (
                route := self._preview_service_vehicle_route(
                    vehicle,
                    aircraft_id,
                )
            )
            is not None
        ]
        return [vehicle_id for _, vehicle_id in sorted(options)]

    def complete_action(
        self,
        action: Dict[str, Any],
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        completed = dict(action)
        high_level = int(completed["high_level"])
        aircraft_id = int(completed["aircraft_id"])
        candidates = self.get_target_candidates(high_level, aircraft_id)
        if not candidates:
            return completed
        selected = rng.choice(candidates) if rng is not None else candidates[0]
        action_name = HIGH_LEVEL_ACTIONS[high_level]
        if action_name in ("R", "L"):
            completed["target_id"] = int(selected)
        elif action_name in ("F", "M", "I"):
            completed["vehicle_id"] = str(selected)
        return completed

    def get_high_level_action_mask(self) -> List[int]:
        candidates = self._candidate_sets()
        return [
            int(len(candidates["R"]) > 0 and self.free_recovery_channels > 0),
            int(
                len(candidates["F"]) > 0
                and self.free_fuel_servers > 0
                and bool(self._available_service_vehicles("fuel"))
                and self._personnel_available(
                    int(self.config["fuel_personnel_required"])
                )
            ),
            int(
                len(candidates["M"]) > 0
                and self.free_arm_vehicles > 0
                and bool(self._available_service_vehicles("arm"))
                and self.free_ammo_transport_vehicles > 0
                and self.free_lower_weapon_lifts > 0
                and self._personnel_available(
                    int(self.config["arm_personnel_required"])
                )
            ),
            int(len(candidates["L"]) > 0 and self.free_launch_channels > 0),
            int(
                len(candidates["I"]) > 0
                and self.free_inspection_vehicles > 0
                and bool(
                    self._available_service_vehicles("inspection")
                )
                and self._personnel_available(
                    int(self.config["inspection_personnel_required"])
                )
            ),
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
                    "initial_role": item.initial_role,
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
                    "inspection_start": item.inspection_start,
                    "inspection_end": item.inspection_end,
                    "arm_start": item.arm_start,
                    "ammo_extract_start": item.ammo_extract_start,
                    "ammo_to_assembly_end": item.ammo_to_assembly_end,
                    "deck_arm_start": item.deck_arm_start,
                    "arm_end": item.arm_end,
                    "launch_ready": item.launch_ready,
                    "launch_start": item.launch_start,
                    "launch_end": item.launch_end,
                    "taxi_start": item.taxi_start,
                    "taxi_end": item.taxi_end,
                    "assigned_launch_wave": item.assigned_launch_wave,
                    "recovery_due_wave": item.recovery_due_wave,
                }
            )
        return records

    def get_wave_records(self) -> List[Dict[str, Any]]:
        return list(self.wave_records)

    def get_missed_sortie_records(self) -> List[Dict[str, Any]]:
        return list(self.missed_sortie_records)

    def get_event_log(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.event_log]

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        total_sorties = sum(aircraft.sorties_completed for aircraft in self.aircraft)
        total_missed = len(self.missed_sortie_records)
        horizon_hours = self.simulation_duration / 60.0
        sortie_opportunities = len(self.wave_records) * self.group_size
        group_metrics = {
            label: {
                "sorties_completed": sum(
                    record["sorties_completed"]
                    for record in self.wave_records
                    if record["launch_group"] == label
                ),
                "missed_sorties": sum(
                    record["missed_sorties"]
                    for record in self.wave_records
                    if record["launch_group"] == label
                ),
            }
            for label in ("A", "B")
        }
        group_metrics["shared"] = {
            "sorties_completed": total_sorties,
            "missed_sorties": total_missed,
        }
        return {
            "simulation_duration": self.simulation_duration,
            "total_sorties_completed": total_sorties,
            "total_missed_sorties": total_missed,
            "sortie_generation_rate_per_hour": (
                total_sorties / horizon_hours if horizon_hours > 0 else 0.0
            ),
            "sortie_completion_rate": (
                total_sorties / sortie_opportunities
                if sortie_opportunities > 0
                else 0.0
            ),
            "group_metrics": group_metrics,
            "cbs_replan": dict(self.cbs_replan_stats),
            "disruptions": {
                **self.disruption_metrics,
                "started_by_kind": dict(
                    self.disruption_metrics["started_by_kind"]
                ),
                "active_at_end": len(self.active_disruptions),
            },
        }

    def _advance_time_to_next_event(self) -> Tuple[float, Dict[str, int]]:
        if not self.event_queue:
            self.done = self.time >= self.simulation_duration
            completed = self._empty_completed_counts()
            return 0.0, completed

        next_time = self.event_queue[0].time
        if next_time > self.simulation_duration:
            delta = self.simulation_duration - self.time
            self._update_waiting_and_remaining(delta)
            self.time = self.simulation_duration
            self.done = True
            completed = self._empty_completed_counts()
            reward = self._calculate_time_reward(delta, completed)
            return reward, completed

        delta = next_time - self.time
        self._update_waiting_and_remaining(delta)
        self.time = next_time
        if self.traffic_planner is not None:
            self.traffic_planner.prune(self.time)
        completed = self._process_events_at_current_time()
        self._invalidate_planning_cache()

        reward = self._calculate_time_reward(delta, completed)
        if self.time >= self.simulation_duration:
            self.done = True
            reward += float(self.config["terminal_reward"])
        self.last_completed_counts = completed
        return reward, completed

    def _process_events_at_current_time(self) -> Dict[str, int]:
        completed = self._empty_completed_counts()
        events: List[Event] = []
        while self.event_queue and self.event_queue[0].time == self.time:
            events.append(heapq.heappop(self.event_queue))
        boundary_events = {"wave_start", "simulation_end"}
        events.sort(key=lambda item: item.event_type in boundary_events)

        for event in events:
            if event.event_type == "wave_start":
                self._start_wave(int(event.time / self.wave_interval))
                continue
            if event.event_type == "simulation_end":
                self._record_missed_sorties(self.current_wave_index)
                self._log_event("simulation_end")
                continue
            if event.event_type in (
                "disruption_start",
                "disruption_end",
            ):
                self._process_disruption_event(event)
                continue
            if event.event_type == "clearance_done":
                self._log_event("clearance_done")
                continue

            aircraft = self.aircraft[event.aircraft_id]
            self._log_event(event.event_type, event.aircraft_id)
            if event.event_type == "recover_done":
                if self.spatial_graph_enabled:
                    reservation = self._complete_deck_route(
                        event.aircraft_id,
                        clear_operation=True,
                    )
                    self._release_tow_vehicle(
                        event.aircraft_id,
                        reservation.target,
                    )
                    aircraft.spot_id = self._route_target_spot(reservation)
                    aircraft.taxi_end = self.time
                    aircraft.parking_status = 2
                    aircraft.park_start = aircraft.taxi_start or self.time
                    aircraft.park_end = self.time
                else:
                    aircraft.spot_id = self._assign_parking_spot(
                        event.aircraft_id
                    )
                    aircraft.parking_status = 1
                    aircraft.park_start = self.time
                    self._push_event(
                        self.time,
                        "park_done",
                        event.aircraft_id,
                    )
                aircraft.recovery_status = 2
                aircraft.recovery_end = self.time
                aircraft.pending_recovery = False
                aircraft.recovery_due_wave = None
                aircraft.is_airborne = False
                aircraft.launch_status = 0
                aircraft.inspection_status = 0
                aircraft.arm_quantity_required = self._sample_arm_quantity()
                self.free_recovery_channels += 1
                completed["recovery"] += 1
            elif event.event_type == "park_done":
                aircraft.parking_status = 2
                aircraft.park_end = self.time
            elif event.event_type == "fuel_done":
                aircraft.fuel_status = 2
                aircraft.fuel_level = 1.0
                aircraft.fuel_remaining = 0.0
                aircraft.fuel_end = self.time
                self.free_fuel_servers += 1
                self._release_service_vehicle("fuel", event.aircraft_id)
                self._release_personnel(
                    int(self.config["fuel_personnel_required"])
                )
                completed["fuel"] += 1
            elif event.event_type == "inspection_done":
                aircraft.inspection_status = 2
                aircraft.inspection_remaining = 0.0
                aircraft.inspection_end = self.time
                self.free_inspection_vehicles += 1
                self._release_service_vehicle("inspection", event.aircraft_id)
                self._release_personnel(
                    int(self.config["inspection_personnel_required"])
                )
                completed["inspection"] += 1
            elif event.event_type == "arm_done":
                aircraft.arm_status = 2
                aircraft.arm_stage = 4
                aircraft.arm_remaining = 0.0
                aircraft.arm_end = self.time
                self.free_arm_vehicles += 1
                self._release_service_vehicle("arm", event.aircraft_id)
                self.free_upper_weapon_lifts += 1
                self._release_personnel(
                    int(self.config["arm_personnel_required"])
                )
                completed["arm"] += 1
            elif event.event_type == "ammo_to_assembly_done":
                aircraft.arm_stage = 2
                aircraft.ammo_to_assembly_end = self.time
                self.free_ammo_transport_vehicles += 1
                self.free_lower_weapon_lifts += 1
            elif event.event_type == "taxi_to_launch_done":
                reservation = self._complete_deck_route(event.aircraft_id)
                self._release_tow_vehicle(
                    event.aircraft_id,
                    reservation.target,
                )
                aircraft.taxi_end = self.time
                aircraft.launch_start = self.time
                self._log_event(
                    "launch_position_ready",
                    event.aircraft_id,
                    launch_position=reservation.target,
                )
                self._push_event(
                    self.time + float(self.config["launch_time"]),
                    "launch_done",
                    event.aircraft_id,
                )
            elif event.event_type == "launch_done":
                launch_wave = aircraft.assigned_launch_wave
                if launch_wave is None:
                    raise RuntimeError("launch completed without an assigned wave")
                wave_record = self._wave_record(launch_wave)
                if wave_record["missed_recorded"]:
                    raise RuntimeError(
                        "launch completed after its assigned wave was closed"
                    )
                aircraft.launch_status = 3
                aircraft.launch_end = self.time
                aircraft.sorties_completed += 1
                wave_record["sorties_completed"] += 1
                wave_record["launched_aircraft_ids"].append(event.aircraft_id)
                aircraft.is_airborne = True
                aircraft.pending_recovery = False
                aircraft.recovery_due_wave = launch_wave + 1
                self._release_launch_position(event.aircraft_id)
                if not self.spatial_graph_enabled:
                    self._release_parking_spot(event.aircraft_id)
                aircraft.parking_status = 0
                aircraft.spot_id = -1
                aircraft.recovery_status = 0
                aircraft.fuel_status = 0
                aircraft.fuel_level = float(
                    self.config["post_sortie_fuel_level"]
                )
                aircraft.inspection_status = 0
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
                and aircraft.inspection_status == 2
                and aircraft.arm_status == 2
                and aircraft.launch_status == 0
            ):
                aircraft.launch_status = 1
                aircraft.launch_ready = self.time
        self._try_start_waiting_deck_arming()
        if self.deck_occupancy is not None:
            self.deck_occupancy.validate()
        return completed

    def _start_recovery(
        self,
        aircraft_id: int,
        target_spot_id: Optional[int] = None,
    ) -> None:
        aircraft = self.aircraft[aircraft_id]
        movement_duration = 0.0
        if self.spatial_graph_enabled:
            reservation = self._reserve_recovery_route(
                aircraft_id,
                target_spot_id,
            )
            self.active_deck_routes[aircraft_id] = reservation
            aircraft.spot_id = self._route_target_spot(reservation)
            aircraft.parking_status = 1
            aircraft.taxi_start = self.time + float(self.config["recovery_time"])
            movement_duration = self._route_duration(reservation)
        aircraft.recovery_status = 1
        aircraft.recovery_start = self.time
        self.free_recovery_channels -= 1
        self._log_event(
            "recover_start",
            aircraft_id,
            route=(
                list(self.active_deck_routes[aircraft_id].path)
                if self.spatial_graph_enabled
                else []
            ),
            movement_duration=movement_duration,
        )
        self._push_event(
            self.time + float(self.config["recovery_time"]) + movement_duration,
            "recover_done",
            aircraft_id,
        )
        if self.spatial_graph_enabled:
            self._register_cbs_route(f"tow:{aircraft_id}")

    def _start_fueling(
        self,
        aircraft_id: int,
        vehicle_id: Optional[str] = None,
    ) -> None:
        aircraft = self.aircraft[aircraft_id]
        rate = float(self.config["fuel_rate_per_minute"])
        if rate <= 0:
            raise ValueError("fuel_rate_per_minute must be positive")
        duration = max(
            float(self.config["min_process_time"]),
            (1.0 - aircraft.fuel_level) / rate,
        ) * self.service_time_multipliers["fuel"]
        vehicle, travel_duration = self._dispatch_service_vehicle(
            "fuel",
            aircraft_id,
            vehicle_id,
        )
        aircraft.fuel_status = 1
        aircraft.fuel_start = self.time
        aircraft.fuel_remaining = travel_duration + duration
        self.free_fuel_servers -= 1
        self._acquire_personnel(int(self.config["fuel_personnel_required"]))
        self._log_event(
            "fuel_start",
            aircraft_id,
            duration=duration,
            vehicle_id=vehicle.vehicle_id,
            travel_duration=travel_duration,
        )
        self._push_event(
            self.time + travel_duration + duration,
            "fuel_done",
            aircraft_id,
        )
        self._register_cbs_route(vehicle.vehicle_id)

    def _start_inspection(
        self,
        aircraft_id: int,
        vehicle_id: Optional[str] = None,
    ) -> None:
        aircraft = self.aircraft[aircraft_id]
        duration = self._sample_duration(
            "inspection_time_mean",
            "inspection_time_std",
        ) * self.service_time_multipliers["inspection"]
        vehicle, travel_duration = self._dispatch_service_vehicle(
            "inspection",
            aircraft_id,
            vehicle_id,
        )
        aircraft.inspection_status = 1
        aircraft.inspection_start = self.time
        aircraft.inspection_remaining = travel_duration + duration
        self.free_inspection_vehicles -= 1
        self._acquire_personnel(
            int(self.config["inspection_personnel_required"])
        )
        self._log_event(
            "inspection_start",
            aircraft_id,
            duration=duration,
            vehicle_id=vehicle.vehicle_id,
            travel_duration=travel_duration,
        )
        self._push_event(
            self.time + travel_duration + duration,
            "inspection_done",
            aircraft_id,
        )
        self._register_cbs_route(vehicle.vehicle_id)

    def _start_arming(
        self,
        aircraft_id: int,
        vehicle_id: Optional[str] = None,
    ) -> None:
        aircraft = self.aircraft[aircraft_id]
        arm_multiplier = self.service_time_multipliers["arm"]
        first_stage_duration = arm_multiplier * (
            self._sample_ammo_extract_time()
            + self._sample_duration(
                "lower_lift_time_mean",
                "lower_lift_time_std",
            )
        )
        deck_stage_duration = arm_multiplier * (
            self._sample_duration("upper_lift_time_mean", "upper_lift_time_std")
            + self._sample_arming_duration(aircraft.arm_quantity_required)
        )
        aircraft.arm_status = 1
        aircraft.arm_stage = 1
        aircraft.arm_start = self.time
        aircraft.ammo_extract_start = self.time
        vehicle, vehicle_travel_duration = self._dispatch_service_vehicle(
            "arm",
            aircraft_id,
            vehicle_id,
        )
        aircraft.arm_vehicle_ready_time = self.time + vehicle_travel_duration
        aircraft.arm_remaining = (
            max(first_stage_duration, vehicle_travel_duration)
            + deck_stage_duration
        )
        self.free_arm_vehicles -= 1
        self.free_ammo_transport_vehicles -= 1
        self.free_lower_weapon_lifts -= 1
        self._acquire_personnel(int(self.config["arm_personnel_required"]))
        self._log_event(
            "arm_start",
            aircraft_id,
            first_stage_duration=first_stage_duration,
            deck_stage_duration=deck_stage_duration,
            vehicle_id=vehicle.vehicle_id,
            vehicle_travel_duration=vehicle_travel_duration,
        )
        self._push_event(self.time + first_stage_duration, "ammo_to_assembly_done", aircraft_id)
        self._register_cbs_route(vehicle.vehicle_id)

    def _start_launch(
        self,
        aircraft_id: int,
        runway_id: Optional[int] = None,
    ) -> None:
        aircraft = self.aircraft[aircraft_id]
        movement_duration = 0.0
        if self.spatial_graph_enabled:
            reservation = self._reserve_launch_route(
                aircraft_id,
                runway_id,
            )
            self.active_deck_routes[aircraft_id] = reservation
            movement_duration = self._route_duration(reservation)
            aircraft.taxi_start = self.time
            aircraft.parking_status = 0
            self._release_parking_spot(aircraft_id)
            aircraft.spot_id = -1
        aircraft.launch_status = 2
        aircraft.assigned_launch_wave = self.current_wave_index
        self.wave_records[-1]["launches_started"] += 1
        self.wave_records[-1]["started_aircraft_ids"].append(aircraft_id)
        self.free_launch_channels -= 1
        self._log_event(
            "launch_start",
            aircraft_id,
            wave_index=self.current_wave_index,
            route=(
                list(self.active_deck_routes[aircraft_id].path)
                if self.spatial_graph_enabled
                else []
            ),
            movement_duration=movement_duration,
        )
        if self.spatial_graph_enabled:
            self._push_event(
                self.time + movement_duration,
                "taxi_to_launch_done",
                aircraft_id,
            )
            self._register_cbs_route(f"tow:{aircraft_id}")
        else:
            aircraft.launch_start = self.time
            self._push_event(
                self.time + float(self.config["launch_time"]),
                "launch_done",
                aircraft_id,
            )

    def _try_start_waiting_deck_arming(self) -> None:
        for aircraft_id, aircraft in enumerate(self.aircraft):
            if (
                aircraft.arm_status == 1
                and aircraft.arm_stage == 2
                and self.free_upper_weapon_lifts > 0
            ):
                self._start_deck_arming(aircraft_id)

    def _start_deck_arming(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        vehicle_id = self.active_service_vehicles[("arm", aircraft_id)]
        vehicle = next(
            item for item in self.service_vehicles
            if item.vehicle_id == vehicle_id
        )
        vehicle_wait = max(
            0.0,
            (aircraft.arm_vehicle_ready_time or self.time) - self.time,
        )
        operation_duration = self.service_time_multipliers[
            "arm"
        ] * (
            self._sample_duration("upper_lift_time_mean", "upper_lift_time_std")
            + self._sample_arming_duration(aircraft.arm_quantity_required)
        )
        duration = operation_duration + vehicle_wait
        aircraft.arm_stage = 3
        aircraft.deck_arm_start = self.time
        aircraft.arm_remaining = duration
        self.free_upper_weapon_lifts -= 1
        self._log_event(
            "deck_arm_start",
            aircraft_id,
            duration=duration,
            vehicle_id=vehicle.vehicle_id,
            travel_duration=vehicle_wait,
        )
        self._push_event(self.time + duration, "arm_done", aircraft_id)

    def _candidate_sets(self) -> Dict[str, List[int]]:
        if self._candidate_cache is not None:
            return self._candidate_cache
        candidates = {"R": [], "F": [], "M": [], "L": [], "I": []}
        launch_capacity_available = (
            self.wave_records[-1]["launches_started"] < self.group_size
        )
        launch_runway_available = any(
            self._launch_runway_available(runway_id)
            for runway_id in range(
                int(self.config["num_launch_positions"])
            )
        )
        for aircraft_id, aircraft in enumerate(self.aircraft):
            if (
                aircraft.pending_recovery
                and aircraft.recovery_status == 0
                and (
                    not self.spatial_graph_enabled
                    or self._find_recovery_route(aircraft_id) is not None
                )
            ):
                candidates["R"].append(aircraft_id)
            if aircraft_id in self.unavailable_aircraft_ids:
                continue
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
                and aircraft.inspection_status == 0
            ):
                candidates["I"].append(aircraft_id)
            if (
                not aircraft.is_airborne
                and aircraft.parking_status == 2
                and aircraft.recovery_status == 2
                and aircraft.arm_status == 0
            ):
                candidates["M"].append(aircraft_id)
            launch_route = (
                self._find_launch_route(aircraft_id)
                if self.spatial_graph_enabled
                and not aircraft.is_airborne
                and aircraft.parking_status == 2
                else None
            )
            launch_movement_duration = (
                self._route_duration_values(launch_route[2])
                if launch_route is not None
                else 0.0
            )
            if (
                launch_capacity_available
                and launch_runway_available
                and not aircraft.is_airborne
                and aircraft.parking_status == 2
                and aircraft.fuel_status == 2
                and aircraft.inspection_status == 2
                and aircraft.arm_status == 2
                and aircraft.launch_status == 1
                and (
                    not self.spatial_graph_enabled
                    or launch_route is not None
                )
                and self.time
                + launch_movement_duration
                + float(self.config["launch_time"])
                <= min(
                    (self.current_wave_index + 1) * self.wave_interval,
                    self.simulation_duration,
                )
            ):
                candidates["L"].append(aircraft_id)
        self._candidate_cache = candidates
        return candidates

    def _schedule_wave_events(self) -> None:
        wave_index = 1
        while wave_index * self.wave_interval < self.simulation_duration:
            self._push_event(wave_index * self.wave_interval, "wave_start", -1)
            wave_index += 1
        self._push_event(self.simulation_duration, "simulation_end", -1)

    def _schedule_disruption_events(self) -> None:
        self.disruptions_by_id = {
            spec.disruption_id: spec
            for spec in self.disruption_schedule
        }
        for spec in self.disruption_schedule:
            self._validate_disruption_spec(spec)
            if spec.start_time <= self.time:
                self._process_disruption_event(
                    Event(
                        time=self.time,
                        sequence=0,
                        event_type="disruption_start",
                        aircraft_id=spec.disruption_id,
                    )
                )
            else:
                self._push_event(
                    spec.start_time,
                    "disruption_start",
                    spec.disruption_id,
                )
            self._push_event(
                spec.end_time,
                "disruption_end",
                spec.disruption_id,
            )

    def _validate_disruption_spec(
        self,
        spec: DisruptionSpec,
    ) -> None:
        if spec.kind == "vehicle_outage":
            vehicle_ids = {
                vehicle.vehicle_id
                for vehicle in self.service_vehicles
            }
            if str(spec.target) not in vehicle_ids:
                raise ValueError(
                    f"unknown disruption vehicle: {spec.target}"
                )
        elif spec.kind == "runway_closure":
            runway_id = int(spec.target)
            if not 0 <= runway_id < int(
                self.config["num_launch_positions"]
            ):
                raise ValueError(
                    f"unknown disruption runway: {spec.target}"
                )
        elif spec.kind == "aircraft_hold":
            aircraft_id = int(spec.target)
            if not 0 <= aircraft_id < self.num_aircraft:
                raise ValueError(
                    f"unknown disruption aircraft: {spec.target}"
                )
        elif (
            spec.kind == "service_slowdown"
            and str(spec.target)
            not in ("fuel", "inspection", "arm", "all")
        ):
            raise ValueError(
                f"unknown slowdown service: {spec.target}"
            )

    def _process_disruption_event(self, event: Event) -> None:
        spec = self.disruptions_by_id[event.aircraft_id]
        if event.event_type == "disruption_start":
            self.active_disruptions[spec.disruption_id] = spec
            self.disruption_metrics["started"] += 1
            by_kind = self.disruption_metrics["started_by_kind"]
            by_kind[spec.kind] = by_kind.get(spec.kind, 0) + 1
        else:
            self.active_disruptions.pop(
                spec.disruption_id,
                None,
            )
            self.disruption_metrics["ended"] += 1
        self._refresh_disruption_state()
        self._invalidate_planning_cache()
        self._log_event(
            event.event_type,
            disruption_id=spec.disruption_id,
            disruption_kind=spec.kind,
            disruption_target=spec.target,
            multiplier=spec.multiplier,
        )

    def _refresh_disruption_state(self) -> None:
        active = tuple(self.active_disruptions.values())
        self.unavailable_vehicle_ids = {
            str(spec.target)
            for spec in active
            if spec.kind == "vehicle_outage"
        }
        self.closed_runway_ids = {
            int(spec.target)
            for spec in active
            if spec.kind == "runway_closure"
        }
        self.unavailable_aircraft_ids = {
            int(spec.target)
            for spec in active
            if spec.kind == "aircraft_hold"
        }
        multipliers = {
            "fuel": 1.0,
            "inspection": 1.0,
            "arm": 1.0,
        }
        for spec in active:
            if spec.kind != "service_slowdown":
                continue
            target = str(spec.target)
            services = (
                tuple(multipliers)
                if target == "all"
                else (target,)
            )
            for service_type in services:
                multipliers[service_type] = max(
                    multipliers[service_type],
                    spec.multiplier,
                )
        self.service_time_multipliers = multipliers

    def _start_wave(self, wave_index: int) -> None:
        self._invalidate_planning_cache()
        if wave_index > self.current_wave_index:
            self._record_missed_sorties(self.current_wave_index)

        self.current_wave_index = wave_index
        self.active_launch_group = "A" if wave_index % 2 == 0 else "B"
        self.active_recovery_group = None if wave_index == 0 else ("B" if wave_index % 2 == 0 else "A")

        recovery_aircraft_ids = []
        if wave_index > 0:
            for aircraft_id, aircraft in enumerate(self.aircraft):
                if (
                    aircraft.is_airborne
                    and aircraft.recovery_due_wave is not None
                    and aircraft.recovery_due_wave <= wave_index
                ):
                    aircraft.pending_recovery = True
                    aircraft.recovery_status = 0
                    recovery_aircraft_ids.append(aircraft_id)

        self.wave_records.append(
            {
                "wave_index": wave_index,
                "time": self.time,
                "launch_group": self.active_launch_group,
                "recovery_group": self.active_recovery_group,
                "launch_target": self.group_size,
                "launches_started": 0,
                "sorties_completed": 0,
                "missed_sorties": 0,
                "missed_recorded": False,
                "started_aircraft_ids": [],
                "launched_aircraft_ids": [],
                "recovery_aircraft_ids": recovery_aircraft_ids,
            }
        )
        self._log_event(
            "wave_start",
            launch_group=self.active_launch_group,
            recovery_group=self.active_recovery_group,
            recovery_aircraft_ids=recovery_aircraft_ids,
            wave_index=wave_index,
        )

    def _record_missed_sorties(self, wave_index: int) -> None:
        record = self._wave_record(wave_index)
        if record["missed_recorded"]:
            return
        missed = max(0, self.group_size - record["sorties_completed"])
        record["missed_sorties"] = missed
        record["missed_recorded"] = True
        if record["sorties_completed"] + missed != self.group_size:
            raise RuntimeError("wave sortie accounting is inconsistent")
        for slot_index in range(missed):
            self.missed_sortie_records.append(
                {
                    "wave_index": wave_index,
                    "time": self.time,
                    "group": record["launch_group"],
                    "slot_index": record["sorties_completed"] + slot_index,
                    "aircraft_id": None,
                }
            )
            self._log_event(
                "missed_sortie",
                group=record["launch_group"],
                wave_index=wave_index,
                slot_index=record["sorties_completed"] + slot_index,
            )

    def _wave_record(self, wave_index: int) -> Dict[str, Any]:
        for record in self.wave_records:
            if record["wave_index"] == wave_index:
                return record
        raise RuntimeError(f"unknown wave index: {wave_index}")

    def _next_wave_time(self) -> Optional[float]:
        next_time = (self.current_wave_index + 1) * self.wave_interval
        if next_time > self.simulation_duration:
            return None
        return next_time

    def _find_launch_route(
        self,
        aircraft_id: int,
        runway_id: Optional[int] = None,
    ) -> Optional[Tuple[str, Tuple[str, ...], float]]:
        cache_key = (aircraft_id, runway_id)
        if cache_key in self._launch_route_cache:
            return self._launch_route_cache[cache_key]
        if (
            self.deck_layout is None
            or self.deck_occupancy is None
            or self.traffic_planner is None
            or not self._available_tow_vehicles()
        ):
            self._launch_route_cache[cache_key] = None
            return None
        aircraft = self.aircraft[aircraft_id]
        if aircraft.spot_id < 0:
            self._launch_route_cache[cache_key] = None
            return None
        source = self.deck_layout.parking_node(aircraft.spot_id)
        runway_ids = (
            (runway_id,)
            if runway_id is not None
            else tuple(range(len(self.deck_layout.launch_nodes)))
        )
        routes = []
        for candidate_id in runway_ids:
            if not self._launch_runway_available(candidate_id, aircraft_id):
                continue
            target = self.deck_layout.launch_nodes[candidate_id]
            if self.deck_occupancy.occupants[target]:
                continue
            tow_plan = self._preview_tow_vehicle(
                aircraft_id,
                source,
                target,
                self.time,
            )
            if tow_plan is not None:
                routes.append(
                    (
                        tow_plan.towing.end_time,
                        target,
                        tow_plan.towing.nodes,
                        tow_plan.towing.end_time - self.time,
                    )
                )
        if not routes:
            self._launch_route_cache[cache_key] = None
            return None
        _, target, path, duration = min(routes)
        result = (target, path, duration)
        self._launch_route_cache[cache_key] = result
        return result

    def _find_recovery_route(
        self,
        aircraft_id: int,
        target_spot_id: Optional[int] = None,
    ) -> Optional[Tuple[str, Tuple[str, ...], float]]:
        cache_key = (aircraft_id, target_spot_id)
        if cache_key in self._recovery_route_cache:
            return self._recovery_route_cache[cache_key]
        if (
            self.deck_layout is None
            or self.deck_occupancy is None
            or self.traffic_planner is None
            or not self._available_tow_vehicles()
        ):
            self._recovery_route_cache[cache_key] = None
            return None
        free_spot_ids = [
            spot_id
            for spot_id, occupied_by in enumerate(self.parking_occupancy)
            if occupied_by is None
            and self._landing_target_available(spot_id)
        ]
        if target_spot_id is not None:
            free_spot_ids = [
                spot_id for spot_id in free_spot_ids
                if spot_id == target_spot_id
            ]
        else:
            recovery_node = self.deck_layout.recovery_node
            free_spot_ids.sort(
                key=lambda spot_id: (
                    self.deck_layout.graph.shortest_distance(
                        recovery_node,
                        self.deck_layout.parking_node(spot_id),
                    ),
                    spot_id,
                )
            )
        routes = []
        start_time = self.time + float(self.config["recovery_time"])
        for spot_id in free_spot_ids:
            target = self.deck_layout.parking_node(spot_id)
            tow_plan = self._preview_tow_vehicle(
                aircraft_id,
                self.deck_layout.recovery_node,
                target,
                start_time,
            )
            if tow_plan is not None:
                routes.append(
                    (
                        tow_plan.towing.end_time,
                        target,
                        tow_plan.towing.nodes,
                        tow_plan.towing.end_time - start_time,
                    )
                )
                if target_spot_id is None:
                    break
        if not routes:
            self._recovery_route_cache[cache_key] = None
            return None
        _, target, path, duration = min(routes)
        result = (target, path, duration)
        self._recovery_route_cache[cache_key] = result
        return result

    def _reserve_launch_route(
        self,
        aircraft_id: int,
        runway_id: Optional[int] = None,
    ) -> DeckRouteReservation:
        if (
            self.deck_layout is None
            or self.deck_occupancy is None
            or self.traffic_planner is None
        ):
            raise RuntimeError("spatial graph is not initialized")
        route = self._find_launch_route(aircraft_id, runway_id)
        if route is None:
            raise RuntimeError("no available launch route")
        target, _, _ = route
        source = self.deck_layout.parking_node(
            self.aircraft[aircraft_id].spot_id
        )
        _, timed_route = self._dispatch_tow_vehicle(
            aircraft_id,
            source,
            target,
            self.time,
        )
        return self.deck_occupancy.reserve_route(
            aircraft_id,
            "launch_taxi",
            source,
            target,
            timed_route.nodes,
            timed_route.end_time - self.time,
            release_source=True,
        )

    def _reserve_recovery_route(
        self,
        aircraft_id: int,
        target_spot_id: Optional[int] = None,
    ) -> DeckRouteReservation:
        if (
            self.deck_layout is None
            or self.deck_occupancy is None
            or self.traffic_planner is None
        ):
            raise RuntimeError("spatial graph is not initialized")
        route = self._find_recovery_route(aircraft_id, target_spot_id)
        if route is None:
            raise RuntimeError("no available recovery route")
        target, _, _ = route
        spot_id = self.deck_layout.parking_spot(target)
        if self.parking_occupancy[spot_id] is not None:
            raise RuntimeError("recovery parking destination became unavailable")
        tow_start = self.time + float(self.config["recovery_time"])
        _, timed_route = self._dispatch_tow_vehicle(
            aircraft_id,
            self.deck_layout.recovery_node,
            target,
            tow_start,
        )
        reservation = self.deck_occupancy.reserve_route(
            aircraft_id,
            "recovery_taxi",
            self.deck_layout.recovery_node,
            target,
            timed_route.nodes,
            timed_route.end_time - tow_start,
            release_source=False,
        )
        self.parking_occupancy[spot_id] = aircraft_id
        return reservation

    def _complete_deck_route(
        self,
        aircraft_id: int,
        clear_operation: bool = False,
    ) -> DeckRouteReservation:
        if self.deck_occupancy is None:
            raise RuntimeError("spatial graph is not initialized")
        reservation = self.deck_occupancy.complete_route(aircraft_id)
        if clear_operation:
            self.active_deck_routes.pop(aircraft_id, None)
        self.deck_occupancy.validate()
        return reservation

    def _release_launch_position(self, aircraft_id: int) -> None:
        if not self.spatial_graph_enabled:
            return
        if self.deck_occupancy is None:
            raise RuntimeError("spatial graph is not initialized")
        reservation = self.active_deck_routes.pop(aircraft_id)
        runway_id = self.deck_layout.launch_nodes.index(reservation.target)
        self.runway_release_times[runway_id] = self.time
        self._schedule_clearance_event(
            self.time + float(self.config["runway_interference_clearance"])
        )
        self.deck_occupancy.release_target(aircraft_id, reservation.target)
        self.deck_occupancy.validate()

    def _route_target_spot(self, reservation: DeckRouteReservation) -> int:
        if self.deck_layout is None:
            raise RuntimeError("spatial graph is not initialized")
        return self.deck_layout.parking_spot(reservation.target)

    def _route_duration(self, reservation: DeckRouteReservation) -> float:
        return self._route_duration_values(reservation.distance)

    def _route_duration_values(self, distance: float) -> float:
        return max(
            float(self.config["min_process_time"]),
            distance,
        )

    def _expected_launch_duration(self, aircraft_id: int) -> float:
        movement_duration = 0.0
        if self.spatial_graph_enabled:
            route = self._find_launch_route(aircraft_id)
            if route is None:
                return float("inf")
            movement_duration = self._route_duration_values(route[2])
        return movement_duration + float(self.config["launch_time"])

    def _expected_recovery_duration(self, aircraft_id: int) -> float:
        movement_duration = 0.0
        if self.spatial_graph_enabled:
            route = self._find_recovery_route(aircraft_id)
            if route is None:
                return float("inf")
            movement_duration = self._route_duration_values(route[2])
        return movement_duration + float(self.config["recovery_time"])

    def _deck_state(self) -> Dict[str, Any]:
        if self.deck_layout is None or self.deck_occupancy is None:
            return {
                "enabled": False,
                "active_movements": 0,
                "free_pathway_fraction": 1.0,
                "free_launch_positions": int(
                    self.config.get("num_launch_positions", 1)
                ),
            }
        reserved_pathways = set()
        if self.traffic_planner is not None:
            pathway_nodes = set(self.deck_layout.pathway_nodes)
            reserved_pathways = {
                node_id
                for (node_id, tick), _ in (
                    self.traffic_planner.node_reservations.items()
                )
                if node_id in pathway_nodes
                and tick * self.traffic_planner.time_step >= self.time
            }
        return {
            "enabled": True,
            "active_movements": len(self.deck_occupancy.active_routes),
            "free_pathway_fraction": (
                1.0
                - len(reserved_pathways)
                / max(1, len(self.deck_layout.pathway_nodes))
            ),
            "free_launch_positions": sum(
                not self.deck_occupancy.occupants[node_id]
                for node_id in self.deck_layout.launch_nodes
            ),
            "topology_assumption": "haitian_45_parking_dual_corridor",
        }

    def _build_parking_transfer_times(self) -> List[float]:
        graph = self.scenario_profile.deck_graph
        if graph is None:
            raise RuntimeError("project_core scenario must define a deck graph")
        return [
            graph.shortest_distance(f"parking_{spot_id}", "arm_service")
            for spot_id in range(self.num_parking_spots)
        ]

    def _spot_transfer_time(self, spot_id: int) -> float:
        if spot_id < 0:
            return 0.0
        return float(self.parking_transfer_times[spot_id])

    def _build_service_vehicle_fleet(self) -> List[ServiceVehicleRecord]:
        vehicles: List[ServiceVehicleRecord] = []
        specifications = (
            ("fuel", int(self.config["num_fuel_servers"])),
            ("inspection", int(self.config["num_inspection_vehicles"])),
            ("arm", int(self.config["num_arm_vehicles"])),
            ("tractor", int(self.config["num_tow_vehicles"])),
        )
        depot_spots = (0, 10, 23, 35)
        for service_type, count in specifications:
            for index in range(count):
                spot_id = depot_spots[index % len(depot_spots)]
                spot_id = min(spot_id, self.num_parking_spots - 1)
                node_id = (
                    self.deck_layout.parking_node(spot_id)
                    if self.deck_layout is not None
                    else f"parking_{spot_id}"
                )
                vehicles.append(
                    ServiceVehicleRecord(
                        vehicle_id=f"{service_type}_{index}",
                        service_type=service_type,
                        node_id=node_id,
                    )
                )
        return vehicles

    def _available_service_vehicles(
        self,
        service_type: str,
    ) -> List[ServiceVehicleRecord]:
        return [
            vehicle
            for vehicle in self.service_vehicles
            if vehicle.service_type == service_type
            and vehicle.busy_aircraft_id is None
            and vehicle.vehicle_id
            not in self.unavailable_vehicle_ids
        ]

    def _available_tow_vehicles(self) -> List[ServiceVehicleRecord]:
        return self._available_service_vehicles("tractor")

    def _preview_tow_vehicle(
        self,
        aircraft_id: int,
        source: str,
        target: str,
        tow_not_before: float,
    ) -> Optional[TowPlan]:
        if self.traffic_planner is None or self.deck_layout is None:
            return None
        cache_key = (aircraft_id, source, target, tow_not_before)
        if cache_key in self._tow_preview_cache:
            return self._tow_preview_cache[cache_key]
        approach = self._preview_tractor_approach(
            source,
            tow_not_before,
        )
        if approach is None:
            self._tow_preview_cache[cache_key] = None
            return None
        towing = self.traffic_planner.plan(
            f"preview:tow:{aircraft_id}:{approach.tractor_id}",
            source,
            target,
            max(tow_not_before, approach.route.end_time),
            shared_nodes=(source, target),
            reserve=False,
        )
        if towing is None:
            self._tow_preview_cache[cache_key] = None
            return None
        result = TowPlan(
            tractor_id=approach.tractor_id,
            approach_start=approach.start_time,
            approach=approach.route,
            towing=towing,
        )
        self._tow_preview_cache[cache_key] = result
        return result

    def _preview_tractor_approach(
        self,
        source: str,
        tow_not_before: float,
    ) -> Optional[TractorApproach]:
        if self.traffic_planner is None or self.deck_layout is None:
            return None
        cache_key = (source, tow_not_before)
        if cache_key in self._tractor_approach_cache:
            return self._tractor_approach_cache[cache_key]
        options: List[TractorApproach] = []
        for tractor in self._available_tow_vehicles():
            static_approach = self.deck_layout.graph.shortest_distance(
                tractor.node_id,
                source,
            )
            approach_start = max(
                self.time,
                tow_not_before - static_approach,
            )
            approach = self.traffic_planner.plan(
                f"preview:{tractor.vehicle_id}",
                tractor.node_id,
                source,
                approach_start,
                shared_nodes=(tractor.node_id, source),
                reserve=False,
            )
            if approach is None:
                continue
            options.append(
                TractorApproach(
                    tractor_id=tractor.vehicle_id,
                    start_time=approach_start,
                    route=approach,
                )
            )
        if not options:
            self._tractor_approach_cache[cache_key] = None
            return None
        result = min(
            options,
            key=lambda item: (item.route.end_time, item.tractor_id),
        )
        self._tractor_approach_cache[cache_key] = result
        return result

    def _dispatch_tow_vehicle(
        self,
        aircraft_id: int,
        source: str,
        target: str,
        tow_not_before: float,
    ) -> Tuple[ServiceVehicleRecord, TimedRoute]:
        if self.traffic_planner is None:
            raise RuntimeError("traffic planner is not initialized")
        preview = self._preview_tow_vehicle(
            aircraft_id,
            source,
            target,
            tow_not_before,
        )
        if preview is None:
            raise RuntimeError("no conflict-free tractor route")
        tractor = next(
            vehicle
            for vehicle in self.service_vehicles
            if vehicle.vehicle_id == preview.tractor_id
        )
        approach = self.traffic_planner.plan(
            tractor.vehicle_id,
            tractor.node_id,
            source,
            preview.approach_start,
            shared_nodes=(tractor.node_id, source),
            reserve=True,
        )
        if approach is None:
            raise RuntimeError("tractor approach route became unavailable")
        towing = self.traffic_planner.plan(
            f"tow:{aircraft_id}",
            source,
            target,
            max(tow_not_before, approach.end_time),
            shared_nodes=(source, target),
            reserve=True,
        )
        if towing is None:
            self.traffic_planner.release(tractor.vehicle_id)
            raise RuntimeError("aircraft towing route became unavailable")
        tractor.busy_aircraft_id = aircraft_id
        self.active_tow_vehicles[aircraft_id] = tractor.vehicle_id
        return tractor, towing

    def _release_tow_vehicle(
        self,
        aircraft_id: int,
        node_id: str,
    ) -> None:
        tractor_id = self.active_tow_vehicles.pop(aircraft_id)
        tractor = next(
            vehicle
            for vehicle in self.service_vehicles
            if vehicle.vehicle_id == tractor_id
        )
        tractor.node_id = node_id
        tractor.busy_aircraft_id = None

    def _register_cbs_route(self, entity_id: str) -> None:
        if (
            not bool(self.config["cbs_replan_enabled"])
            or self.traffic_planner is None
            or entity_id not in self.traffic_planner.routes
        ):
            return
        self._cbs_route_dispatch_times[entity_id] = self.time
        self._replan_current_cbs_batch()

    def _replan_current_cbs_batch(self) -> None:
        if self.traffic_planner is None:
            return
        entity_ids = sorted(
            entity_id
            for entity_id, dispatch_time in (
                self._cbs_route_dispatch_times.items()
            )
            if dispatch_time == self.time
            and entity_id in self.traffic_planner.routes
        )
        if len(entity_ids) < 2:
            return

        old_routes = {
            entity_id: self.traffic_planner.routes[entity_id]
            for entity_id in entity_ids
        }
        requests = tuple(
            RouteRequest(
                entity_id=route.entity_id,
                source=route.source,
                target=route.target,
                start_time=route.start_time,
                shared_nodes=(route.source, route.target),
            )
            for route in old_routes.values()
        )
        for entity_id in entity_ids:
            self.traffic_planner.release(entity_id)

        self.cbs_replan_stats["calls"] += 1
        expansion_limit = int(
            self.config["cbs_max_expanded_nodes"]
        )
        try:
            plan = self.traffic_planner.plan_batch(
                requests,
                reserve=True,
                max_expanded_nodes=(
                    expansion_limit
                    if expansion_limit > 0
                    else None
                ),
            )
        except CBSExpansionLimitError:
            plan = None
        except Exception:
            self._restore_traffic_routes(old_routes, requests)
            raise

        if plan is None:
            self._restore_traffic_routes(old_routes, requests)
            self.cbs_replan_stats["fallbacks"] += 1
            self._log_event(
                "cbs_replan_fallback",
                batch_size=len(requests),
            )
            return

        old_soc = sum(
            route.duration
            for route in old_routes.values()
        )
        self.cbs_replan_stats["sum_of_costs_before"] += old_soc
        if plan.sum_of_costs >= old_soc - 1.0e-9:
            for entity_id in entity_ids:
                self.traffic_planner.release(entity_id)
            self._restore_traffic_routes(old_routes, requests)
            self.cbs_replan_stats["sum_of_costs_after"] += old_soc
            self._log_event(
                "cbs_replan_no_improvement",
                batch_size=len(requests),
                sum_of_costs=old_soc,
                expanded_nodes=plan.expanded_nodes,
            )
            return

        self._apply_cbs_route_updates(
            old_routes,
            plan.routes,
        )
        self.cbs_replan_stats["sum_of_costs_after"] += (
            plan.sum_of_costs
        )
        self.cbs_replan_stats["improved_batches"] += 1
        self._log_event(
            "cbs_replan",
            batch_size=len(requests),
            sum_of_costs_before=old_soc,
            sum_of_costs_after=plan.sum_of_costs,
            expanded_nodes=plan.expanded_nodes,
        )

    def _restore_traffic_routes(
        self,
        routes: Dict[str, TimedRoute],
        requests: Tuple[RouteRequest, ...],
    ) -> None:
        if self.traffic_planner is None:
            return
        request_by_entity = {
            request.entity_id: request
            for request in requests
        }
        for entity_id, route in routes.items():
            self.traffic_planner.reserve(
                route,
                request_by_entity[entity_id].shared_nodes,
            )

    def _apply_cbs_route_updates(
        self,
        old_routes: Dict[str, TimedRoute],
        new_routes: Dict[str, TimedRoute],
    ) -> None:
        event_times_changed = False
        for entity_id, old_route in old_routes.items():
            new_route = new_routes[entity_id]
            end_time_delta = (
                new_route.end_time - old_route.end_time
            )
            if entity_id.startswith("tow:"):
                aircraft_id = int(entity_id.split(":", 1)[1])
                reservation = self.active_deck_routes[aircraft_id]
                updated = DeckRouteReservation(
                    aircraft_id=reservation.aircraft_id,
                    operation=reservation.operation,
                    source=reservation.source,
                    target=reservation.target,
                    path=new_route.nodes,
                    distance=reservation.distance + end_time_delta,
                )
                self.active_deck_routes[aircraft_id] = updated
                if self.deck_occupancy is None:
                    raise RuntimeError(
                        "CBS tow route requires deck occupancy"
                    )
                self.deck_occupancy.active_routes[
                    aircraft_id
                ] = updated
                event_type = (
                    "taxi_to_launch_done"
                    if reservation.operation == "launch_taxi"
                    else "recover_done"
                )
                self._shift_scheduled_event(
                    event_type,
                    aircraft_id,
                    end_time_delta,
                )
                event_times_changed = True
                continue

            assignment = next(
                (
                    key
                    for key, vehicle_id in (
                        self.active_service_vehicles.items()
                    )
                    if vehicle_id == entity_id
                ),
                None,
            )
            if assignment is None:
                raise RuntimeError(
                    "CBS service route has no active assignment"
                )
            service_type, aircraft_id = assignment
            aircraft = self.aircraft[aircraft_id]
            if service_type == "fuel":
                aircraft.fuel_remaining = max(
                    0.0,
                    aircraft.fuel_remaining + end_time_delta,
                )
                self._shift_scheduled_event(
                    "fuel_done",
                    aircraft_id,
                    end_time_delta,
                )
                event_times_changed = True
            elif service_type == "inspection":
                aircraft.inspection_remaining = max(
                    0.0,
                    aircraft.inspection_remaining + end_time_delta,
                )
                self._shift_scheduled_event(
                    "inspection_done",
                    aircraft_id,
                    end_time_delta,
                )
                event_times_changed = True
            elif service_type == "arm":
                ammo_events = [
                    event
                    for event in self.event_queue
                    if event.event_type
                    == "ammo_to_assembly_done"
                    and event.aircraft_id == aircraft_id
                ]
                if len(ammo_events) != 1:
                    raise RuntimeError(
                        "CBS arm route update expected one "
                        "ammunition event"
                    )
                first_stage_remaining = max(
                    0.0,
                    ammo_events[0].time - self.time,
                )
                old_vehicle_remaining = max(
                    0.0,
                    old_route.end_time - self.time,
                )
                deck_stage_remaining = max(
                    0.0,
                    aircraft.arm_remaining
                    - max(
                        first_stage_remaining,
                        old_vehicle_remaining,
                    ),
                )
                new_vehicle_remaining = max(
                    0.0,
                    new_route.end_time - self.time,
                )
                aircraft.arm_vehicle_ready_time = (
                    new_route.end_time
                )
                aircraft.arm_remaining = (
                    max(
                        first_stage_remaining,
                        new_vehicle_remaining,
                    )
                    + deck_stage_remaining
                )
        if event_times_changed:
            heapq.heapify(self.event_queue)

    def _shift_scheduled_event(
        self,
        event_type: str,
        aircraft_id: int,
        delta: float,
    ) -> None:
        matches = [
            event
            for event in self.event_queue
            if event.event_type == event_type
            and event.aircraft_id == aircraft_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "CBS route update expected one completion event: "
                f"{event_type}, aircraft {aircraft_id}"
            )
        matches[0].time += delta

    def _dispatch_service_vehicle(
        self,
        service_type: str,
        aircraft_id: int,
        vehicle_id: Optional[str] = None,
    ) -> Tuple[ServiceVehicleRecord, float]:
        available = [
            vehicle
            for vehicle in self.service_vehicles
            if vehicle.service_type == service_type
            and vehicle.busy_aircraft_id is None
            and vehicle.vehicle_id
            not in self.unavailable_vehicle_ids
            and (vehicle_id is None or vehicle.vehicle_id == vehicle_id)
        ]
        if not available:
            raise RuntimeError(f"no free {service_type} service vehicle")
        aircraft = self.aircraft[aircraft_id]
        if aircraft.spot_id < 0:
            raise RuntimeError("service vehicle target has no parking spot")

        def travel_time(vehicle: ServiceVehicleRecord) -> float:
            route = self._preview_service_vehicle_route(
                vehicle,
                aircraft_id,
            )
            if route is None:
                if self.deck_layout is not None:
                    return float("inf")
                return self._spot_transfer_time(aircraft.spot_id)
            return route.duration

        vehicle = min(
            available,
            key=lambda item: (travel_time(item), item.vehicle_id),
        )
        duration = travel_time(vehicle)
        if not math.isfinite(duration):
            raise RuntimeError(f"no conflict-free route for {vehicle.vehicle_id}")
        if self.traffic_planner is not None and self.deck_layout is not None:
            target = self.deck_layout.parking_node(aircraft.spot_id)
            route = self.traffic_planner.plan(
                vehicle.vehicle_id,
                vehicle.node_id,
                target,
                self.time,
                shared_nodes=(vehicle.node_id, target),
                reserve=True,
            )
            if route is None:
                raise RuntimeError(
                    f"service route became unavailable: {vehicle.vehicle_id}"
                )
            duration = route.duration
        vehicle.busy_aircraft_id = aircraft_id
        self.active_service_vehicles[(service_type, aircraft_id)] = (
            vehicle.vehicle_id
        )
        return vehicle, duration

    def _preview_service_vehicle_route(
        self,
        vehicle: ServiceVehicleRecord,
        aircraft_id: int,
    ) -> Optional[TimedRoute]:
        cache_key = (vehicle.vehicle_id, aircraft_id)
        if cache_key in self._service_route_cache:
            return self._service_route_cache[cache_key]
        aircraft = self.aircraft[aircraft_id]
        if aircraft.spot_id < 0:
            self._service_route_cache[cache_key] = None
            return None
        if self.deck_layout is None or self.traffic_planner is None:
            result = TimedRoute(
                entity_id=vehicle.vehicle_id,
                source=vehicle.node_id,
                target=f"parking_{aircraft.spot_id}",
                nodes=(vehicle.node_id,),
                ticks=(0,),
                start_time=self.time,
                end_time=self.time + self._spot_transfer_time(
                    aircraft.spot_id
                ),
            )
            self._service_route_cache[cache_key] = result
            return result
        target = self.deck_layout.parking_node(aircraft.spot_id)
        result = self.traffic_planner.plan(
            vehicle.vehicle_id,
            vehicle.node_id,
            target,
            self.time,
            shared_nodes=(vehicle.node_id, target),
            reserve=False,
        )
        self._service_route_cache[cache_key] = result
        return result

    def _expected_service_vehicle_travel(
        self,
        service_type: str,
        aircraft_id: int,
    ) -> float:
        available = [
            vehicle
            for vehicle in self.service_vehicles
            if vehicle.service_type == service_type
            and vehicle.busy_aircraft_id is None
            and vehicle.vehicle_id
            not in self.unavailable_vehicle_ids
        ]
        if not available:
            available = [
                vehicle
                for vehicle in self.service_vehicles
                if vehicle.service_type == service_type
                and vehicle.vehicle_id
                not in self.unavailable_vehicle_ids
            ]
        aircraft = self.aircraft[aircraft_id]
        if aircraft.spot_id < 0:
            return float("inf")
        if self.deck_layout is None:
            return self._spot_transfer_time(aircraft.spot_id)
        durations = [
            route.duration
            for vehicle in available
            if (route := self._preview_service_vehicle_route(
                vehicle,
                aircraft_id,
            )) is not None
        ]
        return min(durations, default=float("inf"))

    def _release_service_vehicle(
        self,
        service_type: str,
        aircraft_id: int,
    ) -> None:
        vehicle_id = self.active_service_vehicles.pop(
            (service_type, aircraft_id)
        )
        vehicle = next(
            item for item in self.service_vehicles
            if item.vehicle_id == vehicle_id
        )
        aircraft = self.aircraft[aircraft_id]
        vehicle.node_id = (
            self.deck_layout.parking_node(aircraft.spot_id)
            if self.deck_layout is not None
            else f"parking_{aircraft.spot_id}"
        )
        vehicle.busy_aircraft_id = None

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
        deadline = self._time_until_launch_window()
        if deadline <= self.wave_interval:
            return min(free_spots, key=lambda spot_id: (self._spot_transfer_time(spot_id), spot_id))
        return min(free_spots, key=lambda spot_id: (spot_id, self._spot_transfer_time(spot_id)))

    def _release_parking_spot(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        spot_id = aircraft.spot_id
        if 0 <= spot_id < len(self.parking_occupancy):
            if self.parking_occupancy[spot_id] == aircraft_id:
                self.parking_occupancy[spot_id] = None
                self.parking_release_times[spot_id] = self.time
                self._schedule_clearance_event(
                    self.time
                    + float(self.config["parking_interference_clearance"])
                )

    def _landing_target_available(self, spot_id: int) -> bool:
        if self.deck_layout is None:
            return True
        clearance = float(self.config["parking_interference_clearance"])
        for target, interfering in self.deck_layout.landing_interference_pairs:
            if target != spot_id:
                continue
            if self.parking_occupancy[interfering] is not None:
                return False
            if self.time < self.parking_release_times[interfering] + clearance:
                return False
        return True

    def _launch_runway_available(
        self,
        runway_id: int,
        aircraft_id: Optional[int] = None,
    ) -> bool:
        if runway_id in self.closed_runway_ids:
            return False
        if self.deck_layout is None or self.deck_occupancy is None:
            return True
        if not 0 <= runway_id < len(self.deck_layout.launch_nodes):
            return False
        clearance = float(self.config["parking_interference_clearance"])
        for spot_id in self.deck_layout.launch_blocking_spots[runway_id]:
            if (
                aircraft_id is not None
                and self.aircraft[aircraft_id].spot_id == spot_id
            ):
                continue
            if self.parking_occupancy[spot_id] is not None:
                return False
            if self.time < self.parking_release_times[spot_id] + clearance:
                return False
        runway_clearance = float(
            self.config["runway_interference_clearance"]
        )
        for conflict_id in self.deck_layout.runway_conflicts[runway_id]:
            conflict_node = self.deck_layout.launch_nodes[conflict_id]
            if self.deck_occupancy.occupants[conflict_node]:
                return False
            if self.time < self.runway_release_times[conflict_id] + runway_clearance:
                return False
        return True

    def _time_until_group_launch(self, group: str) -> float:
        del group
        return self._time_until_launch_window()

    def _schedule_clearance_event(self, event_time: float) -> None:
        if self.time < event_time < self.simulation_duration:
            self._push_event(event_time, "clearance_done", -1)

    def _time_until_launch_window(self) -> float:
        current_record = self.wave_records[-1]
        if current_record["launches_started"] < self.group_size:
            return max(
                0.0,
                min(
                    (self.current_wave_index + 1) * self.wave_interval,
                    self.simulation_duration,
                )
                - self.time,
            )
        next_deadline = min(
            (self.current_wave_index + 2) * self.wave_interval,
            self.simulation_duration,
        )
        return max(0.0, next_deadline - self.time)

    def _push_event(self, time: float, event_type: str, aircraft_id: int) -> None:
        self.event_sequence += 1
        heapq.heappush(
            self.event_queue,
            Event(time=time, sequence=self.event_sequence, event_type=event_type, aircraft_id=aircraft_id),
        )

    def _log_event(
        self,
        event_type: str,
        aircraft_id: int = -1,
        **details: Any,
    ) -> None:
        record: Dict[str, Any] = {
            "time": self.time,
            "event_type": event_type,
            "aircraft_id": aircraft_id,
        }
        record.update(details)
        self.event_log.append(record)

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
            if (
                aircraft.recovery_status == 2
                and aircraft.inspection_status == 0
            ):
                aircraft.inspection_wait += delta
            if aircraft.recovery_status == 2 and aircraft.arm_status == 0:
                aircraft.arm_wait += delta
            if aircraft.launch_status == 1:
                aircraft.launch_wait += delta
            if aircraft.fuel_status == 1:
                aircraft.fuel_remaining = max(0.0, aircraft.fuel_remaining - delta)
            if aircraft.inspection_status == 1:
                aircraft.inspection_remaining = max(
                    0.0,
                    aircraft.inspection_remaining - delta,
                )
            if aircraft.arm_status == 1 and aircraft.arm_stage in (1, 3):
                aircraft.arm_remaining = max(0.0, aircraft.arm_remaining - delta)

    def _calculate_time_reward(self, delta: float, completed: Dict[str, int]) -> float:
        return (
            -float(self.config["alpha"]) * delta
            + float(self.config["beta_recovery"]) * completed["recovery"]
            + float(self.config["beta_fuel"]) * completed["fuel"]
            + float(self.config["beta_inspection"]) * completed["inspection"]
            + float(self.config["beta_arm"]) * completed["arm"]
            + float(self.config["beta_launch"]) * completed["launch"]
        )

    def _personnel_available(self, required: int) -> bool:
        return (
            not bool(self.config["personnel_scheduling_enabled"])
            or self.free_personnel >= required
        )

    def _acquire_personnel(self, required: int) -> None:
        if bool(self.config["personnel_scheduling_enabled"]):
            self.free_personnel -= required

    def _release_personnel(self, required: int) -> None:
        if bool(self.config["personnel_scheduling_enabled"]):
            self.free_personnel += required

    def _invalidate_planning_cache(self) -> None:
        self._candidate_cache = None
        self._launch_route_cache.clear()
        self._recovery_route_cache.clear()
        self._tow_preview_cache.clear()
        self._tractor_approach_cache.clear()
        self._service_route_cache.clear()

    @staticmethod
    def _empty_completed_counts() -> Dict[str, int]:
        return {
            "recovery": 0,
            "fuel": 0,
            "inspection": 0,
            "arm": 0,
            "launch": 0,
        }

    def _parse_action(
        self,
        action: Any,
    ) -> Optional[Tuple[int, int, Optional[int], Optional[str]]]:
        if isinstance(action, dict):
            high_level = action.get("high_level")
            aircraft_id = action.get("aircraft_id")
            target_id = action.get("target_id")
            vehicle_id = action.get("vehicle_id")
        elif isinstance(action, (tuple, list)) and len(action) == 2:
            high_level, aircraft_id = action
            target_id = None
            vehicle_id = None
        else:
            return None

        if isinstance(high_level, str):
            high_level = ACTION_TO_INDEX.get(high_level.upper())
        try:
            high_level_int = int(high_level)
            aircraft_id_int = int(aircraft_id)
        except (TypeError, ValueError):
            return None
        target_id_int: Optional[int] = None
        if target_id is not None:
            try:
                target_id_int = int(target_id)
            except (TypeError, ValueError):
                return None
        return (
            high_level_int,
            aircraft_id_int,
            target_id_int,
            str(vehicle_id) if vehicle_id is not None else None,
        )

    def _is_action_valid(
        self,
        high_level: int,
        aircraft_id: int,
        target_id: Optional[int] = None,
        vehicle_id: Optional[str] = None,
    ) -> bool:
        if high_level not in HIGH_LEVEL_ACTIONS:
            return False
        if not 0 <= aircraft_id < self.num_aircraft:
            return False
        high_mask = self.get_high_level_action_mask()
        if high_mask[high_level] != 1:
            return False
        low_mask = self.get_low_level_action_mask(high_level)
        if low_mask[aircraft_id] != 1:
            return False
        action_name = HIGH_LEVEL_ACTIONS[high_level]
        candidates = self.get_target_candidates(high_level, aircraft_id)
        if action_name in ("R", "L"):
            return bool(candidates) and (
                target_id is None or target_id in candidates
            )
        if action_name in ("F", "M", "I"):
            return bool(candidates) and (
                vehicle_id is None or vehicle_id in candidates
            )
        return True

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
            "scenario_profile": self.scenario_profile.name,
            "simulation_duration": self.simulation_duration,
            "completed_counts": completed_counts or dict(self.last_completed_counts),
            "launched": launched,
            "total_sorties_completed": metrics["total_sorties_completed"],
            "total_missed_sorties": metrics["total_missed_sorties"],
            "sortie_generation_rate_per_hour": metrics["sortie_generation_rate_per_hour"],
            "sortie_completion_rate": metrics["sortie_completion_rate"],
            "resources": self.get_state()["resources"],
            "invalid_action": invalid_action,
            "message": message,
            "action_started": action_started,
        }
