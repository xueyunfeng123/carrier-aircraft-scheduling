"""Structural flight-deck graph for the project scheduling environment.

The supplied requirements define area capacities but not an adjacency matrix
or physical distances. This module therefore keeps the topology assumptions
explicit and separate from the scheduling policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from env.scenario import DeckGraph, LocationEdge, LocationNode


@dataclass(frozen=True)
class ProjectDeckGraphAssumptions:
    pathway_nodes: int = 19
    launch_area_spots: int = 10
    recovery_area_spots: int = 10
    launch_positions: int = 4
    edge_travel_minutes: float = 1.0

    def __post_init__(self) -> None:
        if self.pathway_nodes != 19:
            raise ValueError("project deck graph currently requires 19 pathway nodes")
        if self.launch_area_spots < 0 or self.recovery_area_spots < 0:
            raise ValueError("parking-area sizes must be non-negative")
        if self.launch_positions != 4:
            raise ValueError("project deck graph currently requires four launch positions")
        if self.edge_travel_minutes <= 0:
            raise ValueError("edge travel time must be positive")


@dataclass(frozen=True)
class ProjectDeckLayout:
    graph: DeckGraph
    parking_nodes: Tuple[str, ...]
    pathway_nodes: Tuple[str, ...]
    launch_nodes: Tuple[str, ...]
    recovery_node: str
    parking_sections: Tuple[str, ...]
    assumptions: ProjectDeckGraphAssumptions

    def parking_node(self, spot_id: int) -> str:
        if not 0 <= spot_id < len(self.parking_nodes):
            raise ValueError(f"unknown parking spot: {spot_id}")
        return self.parking_nodes[spot_id]

    def parking_spot(self, node_id: str) -> int:
        try:
            return self.parking_nodes.index(node_id)
        except ValueError as exc:
            raise ValueError(f"node is not a parking spot: {node_id}") from exc


@dataclass(frozen=True)
class DeckRouteReservation:
    aircraft_id: int
    operation: str
    source: str
    target: str
    path: Tuple[str, ...]
    distance: float


def build_project_deck_layout(
    num_parking_spots: int,
    assumptions: Optional[ProjectDeckGraphAssumptions] = None,
) -> ProjectDeckLayout:
    """Build a zoned schematic graph from documented capacity information."""

    assumptions = assumptions or ProjectDeckGraphAssumptions()
    if num_parking_spots <= 0:
        raise ValueError("project deck graph requires at least one parking spot")
    launch_area_spots = min(assumptions.launch_area_spots, num_parking_spots)
    recovery_area_spots = min(
        assumptions.recovery_area_spots,
        num_parking_spots - launch_area_spots,
    )
    minimum_spots = launch_area_spots + recovery_area_spots

    parking_nodes = tuple(
        f"deck_parking_{spot_id}" for spot_id in range(num_parking_spots)
    )
    pathway_nodes = tuple(
        f"deck_pathway_{index}" for index in range(assumptions.pathway_nodes)
    )
    launch_nodes = tuple(
        f"deck_launch_{index}" for index in range(assumptions.launch_positions)
    )
    recovery_node = "deck_recovery_runway"
    edge_travel_minutes = assumptions.edge_travel_minutes

    support_spots = (
        num_parking_spots
        - launch_area_spots
        - recovery_area_spots
    )
    sections = []
    for spot_id in range(num_parking_spots):
        if spot_id < launch_area_spots:
            sections.append("launch_parking")
        elif spot_id < launch_area_spots + recovery_area_spots:
            sections.append("recovery_parking")
        else:
            sections.append("support_parking")

    nodes: List[LocationNode] = [
        LocationNode(node_id, sections[index])
        for index, node_id in enumerate(parking_nodes)
    ]
    nodes.extend(LocationNode(node_id, "pathway") for node_id in pathway_nodes)
    nodes.extend(
        LocationNode(
            node_id,
            "launch_dedicated" if index < 2 else "launch_recovery_area",
        )
        for index, node_id in enumerate(launch_nodes)
    )
    nodes.append(LocationNode(recovery_node, "recovery_runway"))

    edges: List[LocationEdge] = [
        LocationEdge(
            pathway_nodes[index],
            pathway_nodes[index + 1],
            edge_travel_minutes,
        )
        for index in range(len(pathway_nodes) - 1)
    ]

    for spot_id, parking_node in enumerate(parking_nodes):
        if spot_id < launch_area_spots:
            local_index = spot_id
            pathway_index = _spread_index(
                local_index,
                launch_area_spots,
                0,
                4,
            )
        elif spot_id < minimum_spots:
            local_index = spot_id - launch_area_spots
            pathway_index = _spread_index(
                local_index,
                recovery_area_spots,
                14,
                18,
            )
        else:
            local_index = spot_id - minimum_spots
            pathway_index = _spread_index(
                local_index,
                support_spots,
                5,
                13,
            )
        edges.append(
            LocationEdge(
                parking_node,
                pathway_nodes[pathway_index],
                edge_travel_minutes,
            )
        )

    launch_connections = (0, 2, 16, 18)
    for index, launch_node in enumerate(launch_nodes):
        connection = launch_connections[min(index, len(launch_connections) - 1)]
        edges.append(
            LocationEdge(
                launch_node,
                pathway_nodes[connection],
                edge_travel_minutes,
            )
        )
    edges.append(
        LocationEdge(
            recovery_node,
            pathway_nodes[18],
            edge_travel_minutes,
        )
    )

    return ProjectDeckLayout(
        graph=DeckGraph(tuple(nodes), tuple(edges)),
        parking_nodes=parking_nodes,
        pathway_nodes=pathway_nodes,
        launch_nodes=launch_nodes,
        recovery_node=recovery_node,
        parking_sections=tuple(sections),
        assumptions=assumptions,
    )


class DeckOccupancy:
    """Capacity accounting and atomic path reservation for one deck graph."""

    def __init__(self, graph: DeckGraph):
        self.graph = graph
        self.occupants: Dict[str, set[int]] = {
            node.node_id: set() for node in graph.nodes
        }
        self.active_routes: Dict[int, DeckRouteReservation] = {}

    def occupy(self, node_id: str, aircraft_id: int) -> None:
        node = self.graph.node(node_id)
        occupants = self.occupants[node_id]
        if aircraft_id in occupants:
            return
        if len(occupants) >= node.capacity:
            raise RuntimeError(f"deck node capacity exceeded: {node_id}")
        occupants.add(aircraft_id)

    def release(self, node_id: str, aircraft_id: int) -> None:
        self.occupants[node_id].discard(aircraft_id)

    def find_route(
        self,
        aircraft_id: int,
        source: str,
        targets: Iterable[str],
    ) -> Optional[Tuple[str, Tuple[str, ...], float]]:
        blocked = {
            node_id
            for node_id, occupants in self.occupants.items()
            if len(occupants) >= self.graph.node(node_id).capacity
            and aircraft_id not in occupants
        }
        target, distance, path = self.graph.shortest_route_to_any_avoiding(
            source,
            targets,
            blocked,
        )
        if target is None:
            return None
        return target, path, distance

    def reserve_route(
        self,
        aircraft_id: int,
        operation: str,
        source: str,
        target: str,
        path: Tuple[str, ...],
        distance: float,
        release_source: bool,
    ) -> DeckRouteReservation:
        if aircraft_id in self.active_routes:
            raise RuntimeError(f"aircraft already has an active route: {aircraft_id}")
        nodes_to_reserve = path[1:] if release_source else path
        for node_id in nodes_to_reserve:
            node = self.graph.node(node_id)
            occupants = self.occupants[node_id]
            if (
                aircraft_id not in occupants
                and len(occupants) >= node.capacity
            ):
                raise RuntimeError(f"deck route became unavailable: {node_id}")
        for node_id in nodes_to_reserve:
            self.occupy(node_id, aircraft_id)
        if release_source:
            self.release(source, aircraft_id)
        reservation = DeckRouteReservation(
            aircraft_id=aircraft_id,
            operation=operation,
            source=source,
            target=target,
            path=path,
            distance=distance,
        )
        self.active_routes[aircraft_id] = reservation
        return reservation

    def complete_route(self, aircraft_id: int) -> DeckRouteReservation:
        reservation = self.active_routes.pop(aircraft_id)
        for node_id in reservation.path:
            if node_id != reservation.target:
                self.release(node_id, aircraft_id)
        return reservation

    def release_target(self, aircraft_id: int, target: str) -> None:
        self.release(target, aircraft_id)

    def free_fraction(self, node_type: str) -> float:
        nodes = [
            node for node in self.graph.nodes if node.location_type == node_type
        ]
        if not nodes:
            return 1.0
        free = sum(
            len(self.occupants[node.node_id]) < node.capacity for node in nodes
        )
        return free / len(nodes)

    def validate(self) -> None:
        for node in self.graph.nodes:
            if len(self.occupants[node.node_id]) > node.capacity:
                raise RuntimeError(f"deck node capacity exceeded: {node.node_id}")


def _spread_index(
    local_index: int,
    count: int,
    start: int,
    end: int,
) -> int:
    if count <= 1:
        return start
    return round(start + local_index * (end - start) / (count - 1))
