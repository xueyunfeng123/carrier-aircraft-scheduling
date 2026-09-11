"""Scenario definitions shared by simulation environments and experiment tools."""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


PROJECT_CORE_PROFILE = "project_core"
YOON_2023_PROFILE = "paper_yoon_2023"


@dataclass(frozen=True)
class DurationDistribution:
    """A small, serializable processing-time distribution."""

    kind: str
    parameters: Tuple[float, ...]

    def __post_init__(self) -> None:
        expected_counts = {
            "deterministic": 1,
            "uniform": 2,
            "triangular": 3,
            "normal": 2,
        }
        if self.kind not in expected_counts:
            raise ValueError(f"unsupported duration distribution: {self.kind}")
        if len(self.parameters) != expected_counts[self.kind]:
            raise ValueError(
                f"{self.kind} requires {expected_counts[self.kind]} parameters"
            )
        if self.kind == "uniform" and self.parameters[1] < self.parameters[0]:
            raise ValueError("uniform maximum must be >= minimum")
        if self.kind == "triangular":
            low, mode, high = self.parameters
            if not low <= mode <= high:
                raise ValueError("triangular parameters must satisfy low <= mode <= high")
        if self.kind == "normal" and self.parameters[1] < 0:
            raise ValueError("normal standard deviation must be non-negative")

    @property
    def expected_value(self) -> float:
        if self.kind == "deterministic":
            return self.parameters[0]
        if self.kind == "uniform":
            low, high = self.parameters
            return (low + high) / 2.0
        if self.kind == "triangular":
            low, mode, high = self.parameters
            return (low + mode + high) / 3.0
        return self.parameters[0]

    def sample(self, rng: random.Random, minimum: float = 0.0) -> float:
        if self.kind == "deterministic":
            value = self.parameters[0]
        elif self.kind == "uniform":
            value = rng.uniform(*self.parameters)
        elif self.kind == "triangular":
            low, mode, high = self.parameters
            value = rng.triangular(low, high, mode)
        else:
            value = rng.gauss(*self.parameters)
        return max(float(minimum), float(value))

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "parameters": list(self.parameters)}


@dataclass(frozen=True)
class ProcessSpec:
    """One operation in a sortie-generation process."""

    name: str
    phase: str
    duration: DurationDistribution
    allowed_location_types: Tuple[str, ...]
    required_resources: Tuple[Tuple[str, int], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "duration": self.duration.as_dict(),
            "allowed_location_types": list(self.allowed_location_types),
            "required_resources": dict(self.required_resources),
        }


@dataclass(frozen=True)
class LocationNode:
    """A discrete location whose capacity can constrain aircraft movement."""

    node_id: str
    location_type: str
    capacity: int = 1

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("location node_id must not be empty")
        if self.capacity <= 0:
            raise ValueError("location capacity must be positive")


@dataclass(frozen=True)
class LocationEdge:
    """A weighted connection between two deck locations."""

    source: str
    target: str
    distance: float
    bidirectional: bool = True

    def __post_init__(self) -> None:
        if self.distance < 0:
            raise ValueError("location edge distance must be non-negative")


@dataclass
class DeckGraph:
    """Validated weighted deck graph with deterministic shortest paths."""

    nodes: Tuple[LocationNode, ...]
    edges: Tuple[LocationEdge, ...]
    _nodes_by_id: Dict[str, LocationNode] = field(init=False, repr=False)
    _adjacency: Dict[str, Tuple[Tuple[str, float], ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nodes_by_id = {node.node_id: node for node in self.nodes}
        if len(nodes_by_id) != len(self.nodes):
            raise ValueError("deck graph node IDs must be unique")

        adjacency: Dict[str, list[Tuple[str, float]]] = {
            node_id: [] for node_id in nodes_by_id
        }
        for edge in self.edges:
            if edge.source not in nodes_by_id or edge.target not in nodes_by_id:
                raise ValueError(
                    f"edge references unknown node: {edge.source} -> {edge.target}"
                )
            adjacency[edge.source].append((edge.target, float(edge.distance)))
            if edge.bidirectional:
                adjacency[edge.target].append((edge.source, float(edge.distance)))

        self._nodes_by_id = nodes_by_id
        self._adjacency = {
            node_id: tuple(sorted(neighbors))
            for node_id, neighbors in adjacency.items()
        }

    def node(self, node_id: str) -> LocationNode:
        try:
            return self._nodes_by_id[node_id]
        except KeyError as exc:
            raise ValueError(f"unknown deck node: {node_id}") from exc

    def edge_distance(self, source: str, target: str) -> float:
        self.node(source)
        self.node(target)
        for neighbor, distance in self._adjacency[source]:
            if neighbor == target:
                return distance
        raise ValueError(f"nodes are not directly connected: {source} -> {target}")

    def neighbors(self, node_id: str) -> Tuple[Tuple[str, float], ...]:
        self.node(node_id)
        return self._adjacency[node_id]

    def shortest_distance(self, source: str, target: str) -> float:
        distance, _ = self._shortest_route(source, target)
        return distance

    def shortest_path(self, source: str, target: str) -> Tuple[str, ...]:
        _, path = self._shortest_route(source, target)
        return path

    def shortest_route_avoiding(
        self,
        source: str,
        target: str,
        blocked_nodes: Iterable[str],
    ) -> Tuple[float, Tuple[str, ...]]:
        return self._shortest_route(source, target, frozenset(blocked_nodes))

    def shortest_route_to_any_avoiding(
        self,
        source: str,
        targets: Iterable[str],
        blocked_nodes: Iterable[str],
    ) -> Tuple[Optional[str], float, Tuple[str, ...]]:
        """Find the nearest reachable target with one deterministic search."""

        self.node(source)
        target_ids = frozenset(targets)
        for target in target_ids:
            self.node(target)
        blocked = frozenset(blocked_nodes)
        available_targets = target_ids - blocked
        if source in blocked or not available_targets:
            return None, math.inf, ()

        distances = {source: 0.0}
        predecessors: Dict[str, str] = {}
        queue = [(0.0, source)]
        best_target: Optional[str] = None
        best_distance = math.inf
        while queue:
            distance, node_id = heapq.heappop(queue)
            if distance > distances[node_id]:
                continue
            if distance > best_distance:
                break
            if node_id in available_targets:
                if (distance, node_id) < (best_distance, best_target or ""):
                    best_distance = distance
                    best_target = node_id
                continue
            for neighbor, weight in self._adjacency[node_id]:
                if neighbor in blocked:
                    continue
                next_distance = distance + weight
                if next_distance < distances.get(neighbor, math.inf):
                    distances[neighbor] = next_distance
                    predecessors[neighbor] = node_id
                    heapq.heappush(queue, (next_distance, neighbor))

        if best_target is None:
            return None, math.inf, ()
        path = [best_target]
        while path[-1] != source:
            path.append(predecessors[path[-1]])
        path.reverse()
        return best_target, best_distance, tuple(path)

    def _shortest_route(
        self,
        source: str,
        target: str,
        blocked_nodes: frozenset[str] = frozenset(),
    ) -> Tuple[float, Tuple[str, ...]]:
        self.node(source)
        self.node(target)
        if source in blocked_nodes or target in blocked_nodes:
            return math.inf, ()
        if source == target:
            return 0.0, (source,)

        distances = {source: 0.0}
        predecessors: Dict[str, str] = {}
        queue = [(0.0, source)]
        while queue:
            distance, node_id = heapq.heappop(queue)
            if node_id == target:
                path = [target]
                while path[-1] != source:
                    path.append(predecessors[path[-1]])
                path.reverse()
                return distance, tuple(path)
            if distance > distances[node_id]:
                continue
            for neighbor, weight in self._adjacency[node_id]:
                if neighbor in blocked_nodes:
                    continue
                next_distance = distance + weight
                if next_distance < distances.get(neighbor, math.inf):
                    distances[neighbor] = next_distance
                    predecessors[neighbor] = node_id
                    heapq.heappush(queue, (next_distance, neighbor))
        return math.inf, ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "location_type": node.location_type,
                    "capacity": node.capacity,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "distance": edge.distance,
                    "bidirectional": edge.bidirectional,
                }
                for edge in self.edges
            ],
        }


@dataclass(frozen=True)
class MissionPlanSpec:
    """Published mission-plan parameters without inventing missing entries."""

    horizon_minutes: float
    formation_size: int
    mission_interval_minutes: float
    mission_duration_minutes: float
    cancellation_threshold_minutes: float
    sar_aircraft: int = 0
    sar_mission_duration_minutes: Optional[float] = None
    sar_missions_per_day: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "horizon_minutes": self.horizon_minutes,
            "formation_size": self.formation_size,
            "mission_interval_minutes": self.mission_interval_minutes,
            "mission_duration_minutes": self.mission_duration_minutes,
            "cancellation_threshold_minutes": self.cancellation_threshold_minutes,
            "sar_aircraft": self.sar_aircraft,
            "sar_mission_duration_minutes": self.sar_mission_duration_minutes,
            "sar_missions_per_day": self.sar_missions_per_day,
        }


@dataclass(frozen=True)
class ScenarioProfile:
    """Traceable scenario data independent of any scheduling algorithm."""

    name: str
    source: str
    fidelity: str
    num_fighters: int
    parking_capacities: Tuple[Tuple[str, int], ...]
    processes: Tuple[ProcessSpec, ...]
    mission_plan: MissionPlanSpec
    deck_graph: Optional[DeckGraph] = None
    policy_rules: Tuple[str, ...] = ()
    unresolved_parameters: Tuple[str, ...] = ()
    metadata: Tuple[Tuple[str, Any], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "fidelity": self.fidelity,
            "num_fighters": self.num_fighters,
            "parking_capacities": dict(self.parking_capacities),
            "processes": [process.as_dict() for process in self.processes],
            "mission_plan": self.mission_plan.as_dict(),
            "deck_graph": self.deck_graph.as_dict() if self.deck_graph else None,
            "policy_rules": list(self.policy_rules),
            "unresolved_parameters": list(self.unresolved_parameters),
            "metadata": dict(self.metadata),
        }


YOON_2023_CASES: Mapping[str, Tuple[int, int, int]] = {
    "case_1_1": (7, 5, 16),
    "case_1_2": (7, 5, 20),
    "case_2_1": (8, 6, 16),
    "case_2_2": (8, 6, 20),
    "case_3_1": (9, 7, 16),
    "case_3_2": (9, 7, 20),
    "case_4_1": (10, 8, 16),
    "case_4_2": (10, 8, 20),
}


def build_yoon_2023_profile(case_id: str = "case_4_1") -> ScenarioProfile:
    """Return the parameters explicitly reported by Yoon et al. (2023).

    The paper does not publish the graph adjacency/weight matrices or the full
    FlyPro table, so this profile deliberately remains a specification rather
    than claiming to be an exact executable replication.
    """

    try:
        bow_capacity, stern_capacity, num_fighters = YOON_2023_CASES[case_id]
    except KeyError as exc:
        choices = ", ".join(sorted(YOON_2023_CASES))
        raise ValueError(
            f"unknown Yoon 2023 case {case_id!r}; choose from {choices}"
        ) from exc

    fighter_processes = (
        ProcessSpec(
            "start_process",
            "pre_launch",
            DurationDistribution("triangular", (30.0, 40.0, 50.0)),
            ("parking_bow", "parking_stern"),
        ),
        ProcessSpec(
            "launch",
            "launch",
            DurationDistribution("uniform", (1.0, 2.0)),
            ("runway",),
        ),
        ProcessSpec(
            "landing",
            "post_launch",
            DurationDistribution("uniform", (0.5, 1.0)),
            ("runway",),
        ),
        ProcessSpec(
            "refueling",
            "post_launch",
            DurationDistribution("triangular", (30.0, 40.0, 50.0)),
            ("parking_bow", "parking_stern"),
        ),
        ProcessSpec(
            "weapon_unloading",
            "post_launch",
            DurationDistribution("uniform", (20.0, 30.0)),
            ("parking_bow", "parking_stern", "runway"),
        ),
        ProcessSpec(
            "maintenance",
            "post_launch",
            DurationDistribution("deterministic", (120.0,)),
            ("hangar",),
        ),
        ProcessSpec(
            "lift_down_or_up",
            "movement",
            DurationDistribution("deterministic", (1.0,)),
            ("lift",),
        ),
        ProcessSpec(
            "lashing",
            "movement",
            DurationDistribution("deterministic", (3.0,)),
            ("parking_bow", "parking_stern", "lift"),
        ),
        ProcessSpec(
            "unlashing",
            "movement",
            DurationDistribution("deterministic", (3.0,)),
            ("parking_bow", "parking_stern", "lift"),
        ),
        ProcessSpec(
            "sar_launch",
            "launch",
            DurationDistribution("uniform", (4.5, 5.5)),
            ("runway_front",),
        ),
        ProcessSpec(
            "sar_landing",
            "post_launch",
            DurationDistribution("uniform", (4.5, 5.5)),
            ("runway_front",),
        ),
        ProcessSpec(
            "sar_refueling",
            "post_launch",
            DurationDistribution("triangular", (15.0, 20.0, 25.0)),
            ("runway_front",),
        ),
    )
    return ScenarioProfile(
        name=f"{YOON_2023_PROFILE}:{case_id}",
        source="https://doi.org/10.1109/WSC60868.2023.10407756",
        fidelity="published_parameters_without_layout_graph_or_full_flypro",
        num_fighters=num_fighters,
        parking_capacities=(
            ("parking_bow", bow_capacity),
            ("parking_stern", stern_capacity),
        ),
        processes=fighter_processes,
        mission_plan=MissionPlanSpec(
            horizon_minutes=2.0 * 24.0 * 60.0,
            formation_size=4,
            mission_interval_minutes=100.0,
            mission_duration_minutes=60.0,
            cancellation_threshold_minutes=20.0,
            sar_aircraft=1,
            sar_mission_duration_minutes=70.0,
            sar_missions_per_day=8,
        ),
        policy_rules=(
            "all assigned fighters and the runway must be ready before launch",
            "pathway and runway must be clear for launch",
            "launch has priority over recovery for runway occupation",
            "planned maintenance occurs after every two sorties",
            "aircraft without deck parking after landing move through temporary runway handling to hangar",
        ),
        unresolved_parameters=(
            "deck graph node coordinates, adjacency matrix, and edge distances",
            "cold-move and hot-move speeds",
            "complete fixed-wing FlyPro mission table and aircraft types",
            "time window during which runway use blocks the port-side lift",
            "probability model for stochastic failures mentioned by the framework",
            "complete deterministic deck-operation policy and tie breakers",
        ),
        metadata=(
            ("case_id", case_id),
            ("simulation_replications", 1000),
            ("maintenance_interval_sorties", 2),
            ("reported_kpi", "mission_success_rate"),
        ),
    )


def build_project_core_profile(config: Mapping[str, Any]) -> ScenarioProfile:
    """Describe the existing executable environment without changing behavior."""

    num_parking_spots = int(config["num_parking_spots"])
    base = float(config["parking_base_transfer_time"])
    step = float(config["parking_ring_time_step"])
    nodes = [
        LocationNode(f"parking_{spot_id}", "parking")
        for spot_id in range(num_parking_spots)
    ]
    nodes.append(
        LocationNode(
            "arm_service",
            "service",
            capacity=max(1, int(config["num_arm_vehicles"])),
        )
    )
    edges = []
    for spot_id in range(num_parking_spots):
        layer = 0 if spot_id == 0 else (spot_id + 1) // 2
        edges.append(
            LocationEdge(
                f"parking_{spot_id}",
                "arm_service",
                base + layer * step,
            )
        )

    personnel_enabled = bool(config["personnel_scheduling_enabled"])
    fuel_resources = (("fuel_vehicle", 1),)
    inspection_resources = (("inspection_vehicle", 1),)
    arm_resources = (("arm_vehicle", 1),)
    if personnel_enabled:
        fuel_resources += (
            ("personnel", int(config["fuel_personnel_required"])),
        )
        inspection_resources += (
            ("personnel", int(config["inspection_personnel_required"])),
        )
        arm_resources += (
            ("personnel", int(config["arm_personnel_required"])),
        )
    refuel_duration = (
        1.0 - float(config["post_sortie_fuel_level"])
    ) / float(config["fuel_rate_per_minute"])

    processes = (
        ProcessSpec(
            "recovery",
            "post_launch",
            DurationDistribution("deterministic", (float(config["recovery_time"]),)),
            ("recovery_channel",),
        ),
        ProcessSpec(
            "fueling",
            "support",
            DurationDistribution(
                "deterministic",
                (refuel_duration,),
            ),
            ("parking",),
            fuel_resources,
        ),
        ProcessSpec(
            "inspection",
            "support",
            DurationDistribution(
                "normal",
                (
                    float(config["inspection_time_mean"]),
                    float(config["inspection_time_std"]),
                ),
            ),
            ("parking",),
            inspection_resources,
        ),
        ProcessSpec(
            "ammo_extract",
            "support",
            DurationDistribution(
                "uniform",
                (
                    float(config["ammo_extract_time_min"]),
                    float(config["ammo_extract_time_max"]),
                ),
            ),
            ("ammunition_storage",),
            (("ammo_transport_vehicle", 1),),
        ),
        ProcessSpec(
            "lower_weapon_lift",
            "support",
            DurationDistribution(
                "normal",
                (
                    float(config["lower_lift_time_mean"]),
                    float(config["lower_lift_time_std"]),
                ),
            ),
            ("lower_weapon_lift",),
            (("lower_weapon_lift", 1),),
        ),
        ProcessSpec(
            "upper_weapon_lift",
            "support",
            DurationDistribution(
                "normal",
                (
                    float(config["upper_lift_time_mean"]),
                    float(config["upper_lift_time_std"]),
                ),
            ),
            ("upper_weapon_lift",),
            (("upper_weapon_lift", 1),),
        ),
        ProcessSpec(
            "arming_unit",
            "support",
            DurationDistribution(
                "normal",
                (
                    float(config["arm_unit_time_mean"]),
                    math.sqrt(float(config["arm_unit_time_variance"])),
                ),
            ),
            ("parking",),
            arm_resources,
        ),
        ProcessSpec(
            "launch",
            "launch",
            DurationDistribution("deterministic", (float(config["launch_time"]),)),
            ("launch_channel",),
        ),
    )
    return ScenarioProfile(
        name=PROJECT_CORE_PROFILE,
        source="proj/doc/前置约束.md and executable environment",
        fidelity="executable_project_model",
        num_fighters=int(config["num_aircraft"]),
        parking_capacities=(("parking", num_parking_spots),),
        processes=processes,
        mission_plan=MissionPlanSpec(
            horizon_minutes=float(config["simulation_duration"]),
            formation_size=int(config["group_size"]),
            mission_interval_minutes=float(config["wave_interval"]),
            mission_duration_minutes=float(config["wave_interval"]),
            cancellation_threshold_minutes=float(config["wave_interval"]),
        ),
        deck_graph=DeckGraph(tuple(nodes), tuple(edges)),
        policy_rules=(
            "A and B are alternating wave-demand labels",
            "fueling, inspection, and arming may overlap",
            "launch requires fueling, inspection, and arming completion",
        ),
        unresolved_parameters=(),
        metadata=(
            ("reported_kpi", "completed_sorties_within_horizon"),
            ("fleet_model", "shared_dynamic"),
            (
                "reserve_aircraft",
                int(config["num_aircraft"]) - 2 * int(config["group_size"]),
            ),
        ),
    )


def list_scenario_profiles() -> Tuple[str, ...]:
    return (PROJECT_CORE_PROFILE, YOON_2023_PROFILE)


def summarize_profiles(profiles: Iterable[ScenarioProfile]) -> Dict[str, Dict[str, Any]]:
    return {profile.name: profile.as_dict() for profile in profiles}
