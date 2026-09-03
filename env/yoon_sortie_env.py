"""Executable structural replication of the Yoon et al. (2023) DES scenario.

The paper publishes process durations, mission cadence, parking capacities and
policy constraints, but not the complete deck matrices, movement speeds or
FlyPro rows. Missing values are isolated in ``YoonReplicationAssumptions`` so
the simulator cannot silently present assumptions as published facts.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from env.scenario import (
    DeckGraph,
    LocationEdge,
    LocationNode,
    ScenarioProfile,
    build_yoon_2023_profile,
)


@dataclass(frozen=True)
class YoonReplicationAssumptions:
    """Parameters required for execution but omitted from the paper."""

    preparation_lead_minutes: float = 50.0
    first_fixed_wing_launch_minute: float = 100.0
    sar_first_launch_minute: float = 0.0
    movement_minutes_per_graph_unit: float = 1.0
    pathway_capacity: int = 19
    lift_capacity: int = 2
    runway_staging_capacity: int = 8
    sar_blocks_runway_during_refueling: bool = True
    horizon_minutes_override: Optional[float] = None

    def __post_init__(self) -> None:
        if self.preparation_lead_minutes < 0:
            raise ValueError("preparation lead time must be non-negative")
        if self.first_fixed_wing_launch_minute < 0:
            raise ValueError("first fixed-wing launch time must be non-negative")
        if self.sar_first_launch_minute < 0:
            raise ValueError("first SAR launch time must be non-negative")
        if self.movement_minutes_per_graph_unit <= 0:
            raise ValueError("movement time per graph unit must be positive")
        if self.pathway_capacity <= 0:
            raise ValueError("pathway capacity must be positive")
        if self.lift_capacity <= 0:
            raise ValueError("lift capacity must be positive")
        if self.runway_staging_capacity <= 0:
            raise ValueError("runway staging capacity must be positive")
        if (
            self.horizon_minutes_override is not None
            and self.horizon_minutes_override <= 0
        ):
            raise ValueError("horizon override must be positive")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "preparation_lead_minutes": self.preparation_lead_minutes,
            "first_fixed_wing_launch_minute": self.first_fixed_wing_launch_minute,
            "sar_first_launch_minute": self.sar_first_launch_minute,
            "movement_minutes_per_graph_unit": self.movement_minutes_per_graph_unit,
            "pathway_capacity": self.pathway_capacity,
            "lift_capacity": self.lift_capacity,
            "runway_staging_capacity": self.runway_staging_capacity,
            "sar_blocks_runway_during_refueling": (
                self.sar_blocks_runway_during_refueling
            ),
            "horizon_minutes_override": self.horizon_minutes_override,
        }


@dataclass(order=True)
class YoonEvent:
    time: float
    priority: int
    sequence: int
    event_type: str = field(compare=False)
    entity_id: str = field(compare=False)
    details: Dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass(order=True)
class MovementRequest:
    priority: int
    sequence: int
    aircraft_id: int = field(compare=False)
    target_type: str = field(compare=False)
    completion_event: str = field(compare=False)
    details: Dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class YoonAircraftRecord:
    aircraft_id: int
    state: str
    location: str
    sorties_completed: int = 0
    mission_id: Optional[str] = None
    maintenance_due: bool = False


@dataclass
class YoonMissionRecord:
    mission_id: str
    kind: str
    scheduled_launch: float
    duration: float
    formation_size: int
    cancellation_deadline: float
    status: str = "scheduled"
    eligible: bool = False
    aircraft_ids: Tuple[int, ...] = ()
    actual_launch: Optional[float] = None
    completed_at: Optional[float] = None


def build_yoon_assumed_graph(profile: ScenarioProfile) -> DeckGraph:
    """Build a schematic graph from Figure 1 with unit edge distances.

    Node counts follow the paper figure (19 pathway and 8 runway nodes).
    Exact coordinates and weights are unavailable, so all edge weights are
    explicitly replication assumptions.
    """

    capacities = dict(profile.parking_capacities)
    bow_count = capacities["parking_bow"]
    stern_count = capacities["parking_stern"]
    nodes: List[LocationNode] = []
    nodes.extend(
        LocationNode(f"bow_{index}", "parking_bow")
        for index in range(1, bow_count + 1)
    )
    nodes.extend(
        LocationNode(f"stern_{index}", "parking_stern")
        for index in range(1, stern_count + 1)
    )
    nodes.extend(
        LocationNode(f"pathway_{index}", "pathway")
        for index in range(1, 20)
    )
    nodes.extend(
        LocationNode(f"runway_{index}", "runway")
        for index in range(1, 9)
    )
    nodes.extend(
        (
            LocationNode("lift_port", "lift"),
            LocationNode("lift_starboard", "lift"),
            LocationNode(
                "hangar",
                "hangar",
                capacity=profile.num_fighters + profile.mission_plan.sar_aircraft,
            ),
            LocationNode("sar_base", "runway_front"),
        )
    )

    edges: List[LocationEdge] = []
    edges.extend(
        LocationEdge(f"pathway_{index}", f"pathway_{index + 1}", 1.0)
        for index in range(1, 19)
    )
    edges.extend(
        LocationEdge(f"runway_{index}", f"runway_{index + 1}", 1.0)
        for index in range(1, 8)
    )
    for index in range(1, bow_count + 1):
        pathway_index = min(index, 19)
        edges.append(LocationEdge(f"bow_{index}", f"pathway_{pathway_index}", 1.0))
    for index in range(1, stern_count + 1):
        pathway_index = max(1, 20 - stern_count + index - 1)
        edges.append(
            LocationEdge(f"stern_{index}", f"pathway_{pathway_index}", 1.0)
        )

    for runway_index in range(1, 9):
        pathway_index = round(1 + (runway_index - 1) * 18 / 7)
        edges.append(
            LocationEdge(
                f"runway_{runway_index}",
                f"pathway_{pathway_index}",
                1.0,
            )
        )
    edges.extend(
        (
            LocationEdge("lift_port", "pathway_17", 1.0),
            LocationEdge("lift_starboard", "runway_6", 1.0),
            LocationEdge("lift_port", "hangar", 1.0),
            LocationEdge("lift_starboard", "hangar", 1.0),
            LocationEdge("sar_base", "runway_1", 1.0),
        )
    )
    return DeckGraph(tuple(nodes), tuple(edges))


class YoonSortieGenerationEnv:
    """Autonomous DES baseline using the paper's published operation policy."""

    def __init__(
        self,
        case_id: str = "case_4_1",
        assumptions: Optional[YoonReplicationAssumptions] = None,
        record_event_log: bool = True,
    ):
        self.profile = build_yoon_2023_profile(case_id)
        self.assumptions = assumptions or YoonReplicationAssumptions()
        self.record_event_log = record_event_log
        self.deck_graph = build_yoon_assumed_graph(self.profile)
        self.processes = {
            process.name: process for process in self.profile.processes
        }
        self.rng = random.Random()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)
        self.time = 0.0
        self.horizon = (
            self.assumptions.horizon_minutes_override
            or self.profile.mission_plan.horizon_minutes
        )
        self.event_sequence = 0
        self.movement_sequence = 0
        self.event_queue: List[YoonEvent] = []
        self.event_log: List[Dict[str, Any]] = []
        self.pending_movements: List[MovementRequest] = []
        self.pending_movement_aircraft: set[int] = set()
        self.movement_paths: Dict[int, Tuple[str, ...]] = {}
        self.movement_progress: Dict[int, int] = {}
        self.movement_completions: Dict[int, Tuple[str, Dict[str, Any]]] = {}
        self.movement_targets: Dict[int, str] = {}
        self.movement_directions: Dict[int, str] = {}
        self.movement_edges_in_progress: set[int] = set()
        self.movement_uses_lift: set[int] = set()
        self.reserved_destinations: Dict[str, int] = {}
        self.pathway_in_use = 0
        self.lifts_in_use = 0
        self.runway_operation: Optional[Tuple[str, str]] = None
        self.fixed_landing_queue: List[str] = []
        self.sar_landing_queue: List[str] = []
        self.sar_launch_queue: List[str] = []
        self.sar_available = True
        self.movement_count = 0
        self.maintenance_completed = 0
        self.done = False

        self.occupancy: Dict[str, set[int]] = {
            node.node_id: set() for node in self.deck_graph.nodes
        }
        parking_nodes = self._parking_nodes()
        self.aircraft: List[YoonAircraftRecord] = []
        for aircraft_id in range(self.profile.num_fighters):
            if aircraft_id < len(parking_nodes):
                location = parking_nodes[aircraft_id]
                state = "waiting"
            else:
                location = "hangar"
                state = "waiting_hangar"
            self.occupancy[location].add(aircraft_id)
            self.aircraft.append(
                YoonAircraftRecord(
                    aircraft_id=aircraft_id,
                    state=state,
                    location=location,
                )
            )

        self.missions: Dict[str, YoonMissionRecord] = {}
        self._schedule_fixed_wing_missions()
        self._schedule_sar_missions()
        self._log_event(
            "simulation_start",
            profile=self.profile.name,
            assumptions=self.assumptions.as_dict(),
        )
        self._assert_invariants()
        return self.get_state()

    def run(
        self,
        seed: Optional[int] = None,
        max_events: int = 1_000_000,
        validate_each_event: bool = True,
    ) -> Dict[str, Any]:
        if seed is not None:
            self.reset(seed)
        processed = 0
        stopped_at_horizon = False
        while self.event_queue and processed < max_events:
            event = heapq.heappop(self.event_queue)
            if event.time > self.horizon:
                stopped_at_horizon = True
                break
            self.time = event.time
            self._handle_event(event)
            self._dispatch_policy()
            if validate_each_event:
                self._assert_invariants()
            processed += 1
        if processed >= max_events:
            raise RuntimeError("maximum event count reached before simulation end")
        if (
            not stopped_at_horizon
            and self.time < self.horizon
            and (
                self.pathway_in_use
                or self.pending_movements
                or self.runway_operation is not None
                or self.fixed_landing_queue
                or self.sar_landing_queue
            )
        ):
            raise RuntimeError("simulation deadlocked before the configured horizon")
        self.time = self.horizon
        self.done = True
        self._log_event("simulation_end")
        self._assert_invariants()
        return self.get_evaluation_metrics()

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "horizon": self.horizon,
            "profile": self.profile.name,
            "fidelity": "structural_replication_with_explicit_assumptions",
            "aircraft": [
                {
                    "aircraft_id": item.aircraft_id,
                    "state": item.state,
                    "location": item.location,
                    "sorties_completed": item.sorties_completed,
                    "mission_id": item.mission_id,
                    "maintenance_due": item.maintenance_due,
                }
                for item in self.aircraft
            ],
            "missions": {
                mission_id: {
                    "kind": mission.kind,
                    "scheduled_launch": mission.scheduled_launch,
                    "status": mission.status,
                    "aircraft_ids": list(mission.aircraft_ids),
                    "actual_launch": mission.actual_launch,
                }
                for mission_id, mission in self.missions.items()
            },
            "resources": {
                "pathway_in_use": self.pathway_in_use,
                "pathway_capacity": self.assumptions.pathway_capacity,
                "lifts_in_use": self.lifts_in_use,
                "lift_capacity": self.assumptions.lift_capacity,
                "runway_busy": self.runway_operation is not None,
                "free_parking": len(self._free_nodes(self._parking_nodes())),
                "runway_staging_occupancy": sum(
                    len(self.occupancy[node_id])
                    for node_id in self._runway_nodes()
                ),
            },
        }

    def get_event_log(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.event_log]

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        fixed_missions = [
            mission for mission in self.missions.values() if mission.kind == "fighter"
        ]
        sar_missions = [
            mission for mission in self.missions.values() if mission.kind == "sar"
        ]
        successful_fixed = [
            mission for mission in fixed_missions if mission.actual_launch is not None
        ]
        successful_sar = [
            mission for mission in sar_missions if mission.actual_launch is not None
        ]
        cancelled_fixed = [
            mission for mission in fixed_missions if mission.status == "cancelled"
        ]
        launch_delays = [
            mission.actual_launch - mission.scheduled_launch
            for mission in successful_fixed
            if mission.actual_launch is not None
        ]
        total_aircraft_sorties = sum(
            aircraft.sorties_completed for aircraft in self.aircraft
        )
        horizon_hours = self.horizon / 60.0
        return {
            "profile": self.profile.name,
            "case_id": dict(self.profile.metadata)["case_id"],
            "fidelity": "structural_replication_with_explicit_assumptions",
            "scheduled_fixed_missions": len(fixed_missions),
            "successful_fixed_missions": len(successful_fixed),
            "cancelled_fixed_missions": len(cancelled_fixed),
            "fixed_mission_success_rate": (
                len(successful_fixed) / len(fixed_missions)
                if fixed_missions
                else 0.0
            ),
            "scheduled_sar_missions": len(sar_missions),
            "successful_sar_missions": len(successful_sar),
            "total_aircraft_sorties": total_aircraft_sorties,
            "sortie_generation_rate_per_hour": (
                total_aircraft_sorties / horizon_hours if horizon_hours > 0 else 0.0
            ),
            "average_fixed_launch_delay": (
                sum(launch_delays) / len(launch_delays)
                if launch_delays
                else 0.0
            ),
            "movement_count": self.movement_count,
            "maintenance_completed": self.maintenance_completed,
            "assumptions": self.assumptions.as_dict(),
        }

    def validate_invariants(self) -> None:
        self._assert_invariants()

    def _schedule_fixed_wing_missions(self) -> None:
        plan = self.profile.mission_plan
        launch_time = self.assumptions.first_fixed_wing_launch_minute
        mission_index = 0
        while launch_time < self.horizon:
            mission_id = f"fighter_{mission_index}"
            mission = YoonMissionRecord(
                mission_id=mission_id,
                kind="fighter",
                scheduled_launch=launch_time,
                duration=plan.mission_duration_minutes,
                formation_size=plan.formation_size,
                cancellation_deadline=(
                    launch_time + plan.cancellation_threshold_minutes
                ),
            )
            self.missions[mission_id] = mission
            self._push_event(
                max(0.0, launch_time - self.assumptions.preparation_lead_minutes),
                "mission_prepare",
                mission_id,
                priority=5,
            )
            self._push_event(
                mission.cancellation_deadline,
                "mission_deadline",
                mission_id,
                priority=20,
            )
            launch_time += plan.mission_interval_minutes
            mission_index += 1

    def _schedule_sar_missions(self) -> None:
        plan = self.profile.mission_plan
        if not plan.sar_aircraft or not plan.sar_missions_per_day:
            return
        days = math.ceil(self.horizon / (24.0 * 60.0))
        for day in range(days):
            day_start = day * 24.0 * 60.0
            for mission_in_day in range(plan.sar_missions_per_day):
                launch_time = (
                    day_start
                    + self.assumptions.sar_first_launch_minute
                    + mission_in_day * plan.mission_interval_minutes
                )
                if launch_time >= self.horizon:
                    continue
                mission_id = f"sar_{day}_{mission_in_day}"
                mission = YoonMissionRecord(
                    mission_id=mission_id,
                    kind="sar",
                    scheduled_launch=launch_time,
                    duration=float(plan.sar_mission_duration_minutes or 0.0),
                    formation_size=1,
                    cancellation_deadline=(
                        launch_time + plan.cancellation_threshold_minutes
                    ),
                )
                self.missions[mission_id] = mission
                self._push_event(
                    launch_time,
                    "sar_launch_request",
                    mission_id,
                    priority=5,
                )
                self._push_event(
                    mission.cancellation_deadline,
                    "mission_deadline",
                    mission_id,
                    priority=20,
                )

    def _handle_event(self, event: YoonEvent) -> None:
        self._log_event(event.event_type, entity_id=event.entity_id, **event.details)
        if event.event_type == "mission_prepare":
            mission = self.missions[event.entity_id]
            mission.eligible = True
            if mission.status == "scheduled":
                mission.status = "waiting_aircraft"
        elif event.event_type == "mission_deadline":
            self._handle_mission_deadline(event.entity_id)
        elif event.event_type == "start_process_done":
            self._handle_start_process_done(
                int(event.entity_id),
                str(event.details["mission_id"]),
            )
        elif event.event_type == "movement_step_done":
            self._handle_movement_step_done(event)
        elif event.event_type == "fixed_launch_done":
            self._handle_fixed_launch_done(event.entity_id)
        elif event.event_type == "fixed_mission_return":
            self.fixed_landing_queue.append(event.entity_id)
        elif event.event_type == "fixed_landing_done":
            self._handle_fixed_landing_done(event.entity_id)
        elif event.event_type == "refuel_done":
            self._handle_refuel_done(int(event.entity_id))
        elif event.event_type == "weapon_unload_done":
            self._request_movement(
                int(event.entity_id),
                "hangar",
                "arrived_hangar",
                priority=0,
            )
        elif event.event_type == "maintenance_done":
            aircraft = self.aircraft[int(event.entity_id)]
            aircraft.maintenance_due = False
            aircraft.state = "waiting_hangar"
            self.maintenance_completed += 1
        elif event.event_type == "sar_launch_request":
            mission = self.missions[event.entity_id]
            mission.eligible = True
            mission.status = "waiting_runway"
            self.sar_launch_queue.append(event.entity_id)
        elif event.event_type == "sar_launch_done":
            self._handle_sar_launch_done(event.entity_id)
        elif event.event_type == "sar_mission_return":
            self.sar_landing_queue.append(event.entity_id)
        elif event.event_type == "sar_landing_done":
            self._handle_sar_landing_done(event.entity_id)
        elif event.event_type == "sar_refuel_done":
            self.sar_available = True
            mission = self.missions[event.entity_id]
            mission.status = "completed"
            mission.completed_at = self.time
            if self.assumptions.sar_blocks_runway_during_refueling:
                self.runway_operation = None
        else:
            raise RuntimeError(f"unsupported Yoon event: {event.event_type}")

    def _dispatch_policy(self) -> None:
        self._try_assign_waiting_missions()
        self._try_start_runway_operation()
        if (
            self.runway_operation is None
            and self._runway_is_empty()
            and (self.fixed_landing_queue or self.sar_landing_queue)
        ):
            self._advance_active_movements()
            if self.pathway_in_use == 0:
                self._try_start_runway_operation()
            return
        self._stage_hangar_aircraft()
        self._dispatch_movement()
        self._try_start_runway_operation()

    def _try_assign_waiting_missions(self) -> None:
        pending = sorted(
            (
                mission
                for mission in self.missions.values()
                if mission.kind == "fighter"
                and mission.eligible
                and mission.status == "waiting_aircraft"
                and self.time <= mission.cancellation_deadline
            ),
            key=lambda mission: (mission.scheduled_launch, mission.mission_id),
        )
        for mission in pending:
            available = sorted(
                (
                    aircraft
                    for aircraft in self.aircraft
                    if aircraft.state == "waiting"
                    and aircraft.location.startswith(("bow_", "stern_"))
                ),
                key=lambda aircraft: (
                    aircraft.sorties_completed,
                    aircraft.aircraft_id,
                ),
            )
            if len(available) < mission.formation_size:
                continue
            selected = available[: mission.formation_size]
            mission.aircraft_ids = tuple(
                aircraft.aircraft_id for aircraft in selected
            )
            mission.status = "preparing"
            for aircraft in selected:
                aircraft.state = "preparing"
                aircraft.mission_id = mission.mission_id
                duration = self.processes["start_process"].duration.sample(self.rng)
                self._push_event(
                    self.time + duration,
                    "start_process_done",
                    str(aircraft.aircraft_id),
                    mission_id=mission.mission_id,
                    duration=duration,
                )

    def _handle_start_process_done(self, aircraft_id: int, mission_id: str) -> None:
        aircraft = self.aircraft[aircraft_id]
        mission = self.missions[mission_id]
        if aircraft.mission_id != mission_id or aircraft.state != "preparing":
            return
        if mission.status == "cancelled":
            aircraft.state = "waiting"
            aircraft.mission_id = None
            return
        aircraft.state = "waiting_taxi_to_runway"
        self._request_movement(
            aircraft_id,
            "runway",
            "ready_for_launch",
            priority=1,
            mission_id=mission_id,
        )

    def _handle_mission_deadline(self, mission_id: str) -> None:
        mission = self.missions[mission_id]
        if mission.actual_launch is not None or mission.status in {
            "airborne",
            "completed",
        }:
            return
        mission.status = "cancelled"
        self._log_event("mission_cancelled", entity_id=mission_id)
        if mission.kind == "sar":
            self.sar_launch_queue = [
                item for item in self.sar_launch_queue if item != mission_id
            ]
            return
        for aircraft_id in mission.aircraft_ids:
            aircraft = self.aircraft[aircraft_id]
            if aircraft.state == "ready_launch":
                aircraft.state = "cancelled_waiting_runway"
            elif aircraft.state == "waiting_taxi_to_runway":
                self._cancel_pending_movement(aircraft_id)
                aircraft.state = "waiting"
                aircraft.mission_id = None
        self._release_cancelled_mission_aircraft(mission)

    def _release_cancelled_mission_aircraft(
        self,
        mission: YoonMissionRecord,
    ) -> None:
        if any(
            self.aircraft[aircraft_id].state == "moving"
            for aircraft_id in mission.aircraft_ids
        ):
            return
        free_parking = len(self._free_nodes(self._parking_nodes()))
        pending_to_parking = sum(
            request.target_type == "parking"
            for request in self.pending_movements
        )
        available_slots = max(0, free_parking - pending_to_parking)
        for aircraft_id in mission.aircraft_ids:
            aircraft = self.aircraft[aircraft_id]
            if aircraft.state == "cancelled_waiting_runway":
                if available_slots > 0:
                    self._request_movement(
                        aircraft_id,
                        "parking",
                        "cancelled_returned_to_parking",
                        priority=0,
                    )
                    available_slots -= 1
                else:
                    aircraft.mission_id = None
                    self._start_weapon_unloading(aircraft_id)

    def _try_start_runway_operation(self) -> None:
        if self.runway_operation is not None or self.pathway_in_use > 0:
            return

        fixed_ready = [
            mission
            for mission in self.missions.values()
            if mission.kind == "fighter"
            and mission.status == "preparing"
            and mission.aircraft_ids
            and all(
                self.aircraft[aircraft_id].state == "ready_launch"
                for aircraft_id in mission.aircraft_ids
            )
            and self.time <= mission.cancellation_deadline
        ]
        sar_ready = [
            self.missions[mission_id]
            for mission_id in self.sar_launch_queue
            if self.missions[mission_id].status != "cancelled"
            and self.time <= self.missions[mission_id].cancellation_deadline
            and self.sar_available
        ]
        launch_candidates = sorted(
            [(mission.scheduled_launch, "fighter", mission) for mission in fixed_ready]
            + [(mission.scheduled_launch, "sar", mission) for mission in sar_ready],
            key=lambda item: (item[0], item[1], item[2].mission_id),
        )
        for _, kind, mission in launch_candidates:
            if kind == "fighter" and self._runway_contains_only(
                set(mission.aircraft_ids)
            ):
                self._start_fixed_launch(mission)
                return
            if kind == "sar" and self._runway_is_empty():
                self._start_sar_launch(mission)
                return

        if not self._runway_is_empty():
            return
        if self.fixed_landing_queue:
            mission_id = self.fixed_landing_queue.pop(0)
            self._start_fixed_landing(self.missions[mission_id])
            return
        if self.sar_landing_queue:
            mission_id = self.sar_landing_queue.pop(0)
            self._start_sar_landing(self.missions[mission_id])

    def _start_fixed_launch(self, mission: YoonMissionRecord) -> None:
        mission.status = "launching"
        mission.actual_launch = self.time
        self.runway_operation = ("fighter_launch", mission.mission_id)
        duration = sum(
            self.processes["launch"].duration.sample(self.rng)
            for _ in mission.aircraft_ids
        )
        self._log_event(
            "fixed_launch_start",
            entity_id=mission.mission_id,
            duration=duration,
            launch_delay=self.time - mission.scheduled_launch,
        )
        self._push_event(
            self.time + duration,
            "fixed_launch_done",
            mission.mission_id,
            duration=duration,
        )

    def _handle_fixed_launch_done(self, mission_id: str) -> None:
        mission = self.missions[mission_id]
        for aircraft_id in mission.aircraft_ids:
            aircraft = self.aircraft[aircraft_id]
            self._release_node(aircraft.location, aircraft_id)
            aircraft.location = "airborne"
            aircraft.state = "airborne"
            aircraft.sorties_completed += 1
        mission.status = "airborne"
        self.runway_operation = None
        self._push_event(
            self.time + mission.duration,
            "fixed_mission_return",
            mission_id,
        )

    def _start_fixed_landing(self, mission: YoonMissionRecord) -> None:
        runway_nodes = self._free_nodes(self._runway_nodes())
        if len(runway_nodes) < mission.formation_size:
            self.fixed_landing_queue.insert(0, mission.mission_id)
            return
        self.runway_operation = ("fighter_landing", mission.mission_id)
        mission.status = "landing"
        for aircraft_id, runway_node in zip(mission.aircraft_ids, runway_nodes):
            self._reserve_node(runway_node, aircraft_id)
            aircraft = self.aircraft[aircraft_id]
            aircraft.location = runway_node
            aircraft.state = "landing"
        duration = sum(
            self.processes["landing"].duration.sample(self.rng)
            for _ in mission.aircraft_ids
        )
        self._log_event(
            "fixed_landing_start",
            entity_id=mission.mission_id,
            duration=duration,
        )
        self._push_event(
            self.time + duration,
            "fixed_landing_done",
            mission.mission_id,
            duration=duration,
        )

    def _handle_fixed_landing_done(self, mission_id: str) -> None:
        mission = self.missions[mission_id]
        mission.status = "completed"
        mission.completed_at = self.time
        self.runway_operation = None
        free_parking = len(self._free_nodes(self._parking_nodes()))
        for index, aircraft_id in enumerate(mission.aircraft_ids):
            aircraft = self.aircraft[aircraft_id]
            aircraft.mission_id = None
            aircraft.maintenance_due = aircraft.sorties_completed % 2 == 0
            aircraft.state = "landed"
            if index < free_parking:
                self._request_movement(
                    aircraft_id,
                    "parking",
                    "arrived_parking",
                    priority=0,
                )
            else:
                self._start_weapon_unloading(aircraft_id)

    def _handle_refuel_done(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        if aircraft.maintenance_due:
            self._start_weapon_unloading(aircraft_id)
        else:
            aircraft.state = "waiting"
            aircraft.mission_id = None

    def _start_weapon_unloading(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        aircraft.state = "weapon_unloading"
        duration = self.processes["weapon_unloading"].duration.sample(self.rng)
        self._push_event(
            self.time + duration,
            "weapon_unload_done",
            str(aircraft_id),
            duration=duration,
        )

    def _start_sar_launch(self, mission: YoonMissionRecord) -> None:
        self.sar_launch_queue.remove(mission.mission_id)
        self.sar_available = False
        mission.status = "launching"
        mission.actual_launch = self.time
        self.runway_operation = ("sar_launch", mission.mission_id)
        duration = self.processes["sar_launch"].duration.sample(self.rng)
        self._log_event(
            "sar_launch_start",
            entity_id=mission.mission_id,
            duration=duration,
        )
        self._push_event(
            self.time + duration,
            "sar_launch_done",
            mission.mission_id,
            duration=duration,
        )

    def _handle_sar_launch_done(self, mission_id: str) -> None:
        mission = self.missions[mission_id]
        mission.status = "airborne"
        self.runway_operation = None
        self._push_event(
            self.time + mission.duration,
            "sar_mission_return",
            mission_id,
        )

    def _start_sar_landing(self, mission: YoonMissionRecord) -> None:
        mission.status = "landing"
        self.runway_operation = ("sar_landing", mission.mission_id)
        duration = self.processes["sar_landing"].duration.sample(self.rng)
        self._log_event(
            "sar_landing_start",
            entity_id=mission.mission_id,
            duration=duration,
        )
        self._push_event(
            self.time + duration,
            "sar_landing_done",
            mission.mission_id,
            duration=duration,
        )

    def _handle_sar_landing_done(self, mission_id: str) -> None:
        duration = self.processes["sar_refueling"].duration.sample(self.rng)
        if not self.assumptions.sar_blocks_runway_during_refueling:
            self.runway_operation = None
        else:
            self.runway_operation = ("sar_refueling", mission_id)
        self._push_event(
            self.time + duration,
            "sar_refuel_done",
            mission_id,
            duration=duration,
        )

    def _request_movement(
        self,
        aircraft_id: int,
        target_type: str,
        completion_event: str,
        priority: int,
        **details: Any,
    ) -> None:
        if aircraft_id in self.pending_movement_aircraft:
            return
        self.movement_sequence += 1
        heapq.heappush(
            self.pending_movements,
            MovementRequest(
                priority=priority,
                sequence=self.movement_sequence,
                aircraft_id=aircraft_id,
                target_type=target_type,
                completion_event=completion_event,
                details=details,
            ),
        )
        self.pending_movement_aircraft.add(aircraft_id)

    def _cancel_pending_movement(self, aircraft_id: int) -> None:
        self.pending_movements = [
            request
            for request in self.pending_movements
            if request.aircraft_id != aircraft_id
        ]
        heapq.heapify(self.pending_movements)
        self.pending_movement_aircraft.discard(aircraft_id)

    def _dispatch_movement(self) -> None:
        while (
            self.pending_movements
            and self.pathway_in_use < self.assumptions.pathway_capacity
        ):
            selected = self._select_feasible_movement()
            if selected is None:
                break
            request, target, path = selected
            self._start_movement(request, target, path)
        self._advance_active_movements()

    def _select_feasible_movement(
        self,
    ) -> Optional[Tuple[MovementRequest, str, Tuple[str, ...]]]:
        deferred: List[MovementRequest] = []
        selected = None
        while self.pending_movements:
            request = heapq.heappop(self.pending_movements)
            aircraft = self.aircraft[request.aircraft_id]
            route = self._choose_movement_route(
                aircraft.location,
                request.target_type,
                aircraft.aircraft_id,
            )
            if route is None:
                deferred.append(request)
                continue
            target, path = route
            direction = self._movement_direction(aircraft.location, target)
            active_directions = set(self.movement_directions.values())
            if active_directions and direction not in active_directions:
                deferred.append(request)
                continue
            uses_lift = any(
                self.deck_graph.node(node_id).location_type == "lift"
                for node_id in path
            )
            if uses_lift and self.lifts_in_use >= self.assumptions.lift_capacity:
                deferred.append(request)
                continue
            selected = (request, target, path)
            break
        for request in deferred:
            heapq.heappush(self.pending_movements, request)
        return selected

    def _start_movement(
        self,
        request: MovementRequest,
        target: str,
        path: Tuple[str, ...],
    ) -> None:
        aircraft = self.aircraft[request.aircraft_id]
        source = aircraft.location
        uses_lift = any(
            self.deck_graph.node(node_id).location_type == "lift"
            for node_id in path
        )
        if uses_lift:
            self.lifts_in_use += 1
        self.pathway_in_use += 1
        self.pending_movement_aircraft.remove(aircraft.aircraft_id)
        self.movement_paths[aircraft.aircraft_id] = path
        self.movement_progress[aircraft.aircraft_id] = 0
        self.movement_completions[aircraft.aircraft_id] = (
            request.completion_event,
            dict(request.details),
        )
        self.movement_targets[aircraft.aircraft_id] = target
        self.movement_directions[aircraft.aircraft_id] = self._movement_direction(
            source,
            target,
        )
        if self.deck_graph.node(target).capacity == 1:
            self.reserved_destinations[target] = aircraft.aircraft_id
        if uses_lift:
            self.movement_uses_lift.add(aircraft.aircraft_id)
        aircraft.state = "moving"
        self.movement_count += 1
        self._log_event(
            "movement_start",
            entity_id=str(aircraft.aircraft_id),
            source=source,
            target=target,
            path=list(path),
            uses_lift=uses_lift,
        )

    def _advance_active_movements(self) -> None:
        for aircraft_id in sorted(self.movement_paths):
            if aircraft_id in self.movement_edges_in_progress:
                continue
            aircraft = self.aircraft[aircraft_id]
            current_node = aircraft.location
            target = self.movement_targets[aircraft_id]
            if current_node == target:
                self._complete_movement(aircraft_id)
                continue

            blocked_nodes = {
                node_id
                for node_id, occupants in self.occupancy.items()
                if len(occupants) >= self.deck_graph.node(node_id).capacity
                and aircraft_id not in occupants
            }
            blocked_nodes.update(
                node_id
                for node_id, owner in self.reserved_destinations.items()
                if owner != aircraft_id
            )
            _, route = self.deck_graph.shortest_route_avoiding(
                current_node,
                target,
                blocked_nodes,
            )
            if len(route) < 2:
                continue
            self.movement_paths[aircraft_id] = route
            self.movement_progress[aircraft_id] = 0
            next_node = route[1]

            self._reserve_node(next_node, aircraft_id)
            distance = self._edge_distance(current_node, next_node)
            duration = (
                distance * self.assumptions.movement_minutes_per_graph_unit
            )
            if (
                self.deck_graph.node(next_node).location_type == "lift"
            ):
                duration += (
                    self.processes["lashing"].duration.expected_value
                    + self.processes["lift_down_or_up"].duration.expected_value
                )
            if (
                self.deck_graph.node(current_node).location_type == "lift"
            ):
                duration += self.processes["unlashing"].duration.expected_value
            duration = max(0.1, duration)
            self.movement_edges_in_progress.add(aircraft_id)
            self._push_event(
                self.time + duration,
                "movement_step_done",
                str(aircraft_id),
                aircraft_id=aircraft_id,
                current_node=current_node,
                next_node=next_node,
                duration=duration,
            )

    def _handle_movement_step_done(self, event: YoonEvent) -> None:
        aircraft_id = int(event.details["aircraft_id"])
        current_node = str(event.details["current_node"])
        next_node = str(event.details["next_node"])
        self._release_node(current_node, aircraft_id)
        aircraft = self.aircraft[aircraft_id]
        aircraft.location = next_node
        self.movement_edges_in_progress.discard(aircraft_id)
        self._log_event(
            "movement_node_reached",
            entity_id=str(aircraft_id),
            location=next_node,
        )
        if next_node == self.movement_targets[aircraft_id]:
            self._complete_movement(aircraft_id)

    def _complete_movement(self, aircraft_id: int) -> None:
        aircraft = self.aircraft[aircraft_id]
        target = self.movement_targets.pop(aircraft_id)
        completion_event, completion_details = self.movement_completions.pop(
            aircraft_id
        )
        self.movement_paths.pop(aircraft_id)
        self.movement_progress.pop(aircraft_id)
        self.movement_directions.pop(aircraft_id)
        self.reserved_destinations.pop(target, None)
        self.pathway_in_use -= 1
        if aircraft_id in self.movement_uses_lift:
            self.movement_uses_lift.remove(aircraft_id)
            self.lifts_in_use -= 1
        self._log_event(
            "movement_done",
            entity_id=str(aircraft_id),
            target=target,
            completion_event=completion_event,
        )

        if completion_event == "ready_for_launch":
            mission_id = str(completion_details["mission_id"])
            mission = self.missions[mission_id]
            if mission.status == "cancelled":
                aircraft.state = "cancelled_waiting_runway"
                self._release_cancelled_mission_aircraft(mission)
            else:
                aircraft.state = "ready_launch"
        elif completion_event == "arrived_parking":
            aircraft.state = "refueling"
            duration = self.processes["refueling"].duration.sample(self.rng)
            self._push_event(
                self.time + duration,
                "refuel_done",
                str(aircraft_id),
                duration=duration,
            )
        elif completion_event == "cancelled_returned_to_parking":
            aircraft.state = "waiting"
            aircraft.mission_id = None
        elif completion_event == "arrived_hangar":
            if aircraft.maintenance_due:
                aircraft.state = "maintenance"
                duration = self.processes["maintenance"].duration.sample(self.rng)
                self._log_event(
                    "maintenance_start",
                    entity_id=str(aircraft_id),
                    duration=duration,
                )
                self._push_event(
                    self.time + duration,
                    "maintenance_done",
                    str(aircraft_id),
                    duration=duration,
                )
            else:
                aircraft.state = "waiting_hangar"
        elif completion_event == "arrived_deck":
            aircraft.state = "waiting"
            aircraft.mission_id = None
        else:
            raise RuntimeError(f"unsupported movement completion: {completion_event}")

    def _stage_hangar_aircraft(self) -> None:
        free_parking = len(self._free_nodes(self._parking_nodes()))
        pending_to_parking = sum(
            request.target_type == "parking"
            for request in self.pending_movements
        )
        slots = max(0, free_parking - pending_to_parking)
        if slots <= 0:
            return
        candidates = [
            aircraft
            for aircraft in self.aircraft
            if aircraft.state == "waiting_hangar"
            and aircraft.aircraft_id not in self.pending_movement_aircraft
        ]
        for aircraft in candidates[:slots]:
            self._request_movement(
                aircraft.aircraft_id,
                "parking",
                "arrived_deck",
                priority=2,
            )

    def _choose_movement_route(
        self,
        source: str,
        target_type: str,
        aircraft_id: int,
    ) -> Optional[Tuple[str, Tuple[str, ...]]]:
        del aircraft_id
        if target_type == "parking":
            free = self._free_nodes(self._parking_nodes())
        elif target_type == "runway":
            free = self._free_nodes(self._runway_nodes())
        elif target_type == "hangar":
            free = self._free_nodes(("hangar",))
        else:
            raise ValueError(f"unsupported movement target type: {target_type}")
        routes = []
        for target in free:
            path = self.deck_graph.shortest_path(source, target)
            if not path:
                continue
            routes.append(
                (
                    self.deck_graph.shortest_distance(source, target),
                    target,
                    path,
                )
            )
        if not routes:
            return None
        _, target, path = min(routes, key=lambda item: (item[0], item[1]))
        return target, path

    def _movement_direction(self, source: str, target: str) -> str:
        if target == "hangar":
            return "to_hangar"
        if source == "hangar":
            return "to_deck"
        source_type = self.deck_graph.node(source).location_type
        target_type = self.deck_graph.node(target).location_type
        if source_type.startswith("parking") and target_type == "runway":
            return "to_runway"
        if source_type == "runway" and target_type.startswith("parking"):
            return "from_runway"
        return "within_deck"

    def _parking_nodes(self) -> Tuple[str, ...]:
        return tuple(
            node.node_id
            for node in self.deck_graph.nodes
            if node.location_type in {"parking_bow", "parking_stern"}
        )

    def _runway_nodes(self) -> Tuple[str, ...]:
        nodes = tuple(
            node.node_id
            for node in self.deck_graph.nodes
            if node.location_type == "runway"
        )
        return nodes[: self.assumptions.runway_staging_capacity]

    def _free_nodes(self, node_ids: Tuple[str, ...]) -> List[str]:
        return [
            node_id
            for node_id in node_ids
            if len(self.occupancy[node_id]) < self.deck_graph.node(node_id).capacity
            and node_id not in self.reserved_destinations
        ]

    def _edge_distance(self, source: str, target: str) -> float:
        try:
            return self.deck_graph.edge_distance(source, target)
        except ValueError as exc:
            raise RuntimeError(
                f"movement path contains non-edge: {source} -> {target}"
            ) from exc

    def _reserve_node(self, node_id: str, aircraft_id: int) -> None:
        node = self.deck_graph.node(node_id)
        occupants = self.occupancy[node_id]
        if len(occupants) >= node.capacity:
            raise RuntimeError(f"location capacity exceeded: {node_id}")
        occupants.add(aircraft_id)

    def _release_node(self, node_id: str, aircraft_id: int) -> None:
        if node_id in self.occupancy:
            self.occupancy[node_id].discard(aircraft_id)

    def _runway_is_empty(self) -> bool:
        return all(not self.occupancy[node_id] for node_id in self._runway_nodes())

    def _runway_contains_only(self, aircraft_ids: set[int]) -> bool:
        occupants = set().union(
            *(self.occupancy[node_id] for node_id in self._runway_nodes())
        )
        return occupants == aircraft_ids

    def _push_event(
        self,
        time: float,
        event_type: str,
        entity_id: str,
        priority: int = 0,
        **details: Any,
    ) -> None:
        self.event_sequence += 1
        heapq.heappush(
            self.event_queue,
            YoonEvent(
                time=float(time),
                priority=priority,
                sequence=self.event_sequence,
                event_type=event_type,
                entity_id=entity_id,
                details=details,
            ),
        )

    def _log_event(
        self,
        event_type: str,
        entity_id: str = "",
        **details: Any,
    ) -> None:
        if not self.record_event_log:
            return
        record: Dict[str, Any] = {
            "time": self.time,
            "event_type": event_type,
            "entity_id": entity_id,
        }
        record.update(details)
        self.event_log.append(record)

    def _assert_invariants(self) -> None:
        if self.pathway_in_use < 0:
            raise RuntimeError("pathway usage became negative")
        if self.pathway_in_use > self.assumptions.pathway_capacity:
            raise RuntimeError("pathway capacity exceeded")
        if self.lifts_in_use < 0:
            raise RuntimeError("lift usage became negative")
        if self.lifts_in_use > self.assumptions.lift_capacity:
            raise RuntimeError("lift capacity exceeded")

        seen: Dict[int, List[str]] = {}
        for node in self.deck_graph.nodes:
            occupants = self.occupancy[node.node_id]
            if len(occupants) > node.capacity:
                raise RuntimeError(f"location capacity exceeded: {node.node_id}")
            for aircraft_id in occupants:
                seen.setdefault(aircraft_id, []).append(node.node_id)

        for aircraft in self.aircraft:
            if aircraft.state == "airborne":
                if aircraft.aircraft_id in seen:
                    raise RuntimeError(
                        f"airborne aircraft {aircraft.aircraft_id} occupies a location"
                    )
            elif aircraft.state == "moving":
                occupied_nodes = tuple(seen.get(aircraft.aircraft_id, ()))
                if not occupied_nodes:
                    raise RuntimeError(
                        f"moving aircraft {aircraft.aircraft_id} has no reserved target"
                    )
                path = self.movement_paths.get(aircraft.aircraft_id, ())
                if not set(occupied_nodes).issubset(set(path)):
                    raise RuntimeError(
                        f"moving aircraft {aircraft.aircraft_id} path reservation "
                        "disagrees with occupancy"
                    )
            elif seen.get(aircraft.aircraft_id) != [aircraft.location]:
                raise RuntimeError(
                    f"aircraft {aircraft.aircraft_id} location disagrees with occupancy"
                )
