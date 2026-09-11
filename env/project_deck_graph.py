"""Structural flight-deck graph for the project scheduling environment.

The supplied requirements define area capacities but not an adjacency matrix
or physical distances. This module therefore keeps the topology assumptions
explicit and separate from the scheduling policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from env.scenario import DeckGraph, LocationEdge, LocationNode


@dataclass(frozen=True)
class ProjectDeckGraphAssumptions:
    parking_cluster_sizes: Tuple[int, int, int, int] = (10, 13, 12, 10)
    launch_positions: int = 3
    edge_travel_minutes: float = 1.0

    def __post_init__(self) -> None:
        if any(size < 0 for size in self.parking_cluster_sizes):
            raise ValueError("parking-cluster sizes must be non-negative")
        if self.launch_positions != 3:
            raise ValueError("Haitian layout requires three launch runways")
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
    launch_blocking_spots: Tuple[Tuple[int, ...], ...]
    landing_interference_pairs: Tuple[Tuple[int, int], ...]
    runway_conflicts: Tuple[Tuple[int, ...], ...]
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
    """Build a four-cluster graph derived from the Haitian competition layout.

    The source document supplies position geometry and interference rules, but
    no taxiway adjacency matrix. The graph therefore preserves its four parking
    clusters, western landing runway, and three eastern launch runways while
    making the derived dual-corridor topology explicit.
    """

    assumptions = assumptions or ProjectDeckGraphAssumptions()
    if num_parking_spots <= 0:
        raise ValueError("project deck graph requires at least one parking spot")
    cluster_sizes = _scaled_cluster_sizes(
        num_parking_spots,
        assumptions.parking_cluster_sizes,
    )

    parking_nodes = tuple(
        f"deck_parking_{spot_id}" for spot_id in range(num_parking_spots)
    )
    pathway_nodes = tuple(
        f"deck_pathway_{row}_{index}"
        for row in ("north", "middle", "south")
        for index in range(5)
    )
    launch_nodes = tuple(
        f"deck_launch_{index}" for index in range(assumptions.launch_positions)
    )
    recovery_node = "deck_recovery_runway"
    edge_travel_minutes = assumptions.edge_travel_minutes

    cluster_names = ("northwest", "northeast", "southwest", "southeast")
    sections = [
        cluster_name
        for cluster_name, size in zip(cluster_names, cluster_sizes)
        for _ in range(size)
    ]

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

    pathway_index = {
        (row, index): pathway_nodes[row_index * 5 + index]
        for row_index, row in enumerate(("north", "middle", "south"))
        for index in range(5)
    }
    edges: List[LocationEdge] = []
    for row in ("north", "middle", "south"):
        for index in range(4):
            edges.append(
                LocationEdge(
                    pathway_index[(row, index)],
                    pathway_index[(row, index + 1)],
                    edge_travel_minutes,
                )
            )
    for index in (1, 2, 3):
        edges.extend(
            (
                LocationEdge(
                    pathway_index[("north", index)],
                    pathway_index[("middle", index)],
                    edge_travel_minutes,
                ),
                LocationEdge(
                    pathway_index[("middle", index)],
                    pathway_index[("south", index)],
                    edge_travel_minutes,
                ),
            )
        )

    cluster_anchors = (
        tuple(pathway_index[("north", index)] for index in (0, 1)),
        tuple(pathway_index[("north", index)] for index in (3, 4)),
        tuple(pathway_index[("south", index)] for index in (0, 1)),
        tuple(pathway_index[("south", index)] for index in (3, 4)),
    )
    cluster_ranges: List[range] = []
    spot_id = 0
    for cluster_size, anchors in zip(cluster_sizes, cluster_anchors):
        cluster_ranges.append(range(spot_id, spot_id + cluster_size))
        for local_index in range(cluster_size):
            parking_node = parking_nodes[spot_id]
            anchor = anchors[
                _spread_index(local_index, cluster_size, 0, len(anchors) - 1)
            ]
            edges.append(
                LocationEdge(
                    parking_node,
                    anchor,
                    edge_travel_minutes,
                )
            )
            spot_id += 1

    northeast_east = tuple(
        spot
        for local_index, spot in enumerate(cluster_ranges[1])
        if _spread_index(local_index, cluster_sizes[1], 0, 1) == 1
    )
    southeast_east = tuple(
        spot
        for local_index, spot in enumerate(cluster_ranges[3])
        if _spread_index(local_index, cluster_sizes[3], 0, 1) == 1
    )
    # The PDF supplies a 28-position interference table that cannot be copied
    # position-for-position into the scaled 45-position graph. Keep only the
    # closest east-side spot for runways 29 and 30; runway 31 remains clear.
    # This preserves the physical dependency without deadlocking a full deck.
    launch_blocking_spots = (
        northeast_east[-1:],
        southeast_east[-1:],
        (),
    )
    landing_interference_pairs = tuple(
        pair
        for cluster_range in cluster_ranges
        for pair in _cluster_interference_pairs(cluster_range)
    )
    runway_conflicts = ((1,), (0,), ())

    launch_connections = (
        pathway_index[("north", 4)],
        pathway_index[("middle", 4)],
        pathway_index[("south", 4)],
    )
    for index, launch_node in enumerate(launch_nodes):
        edges.append(
            LocationEdge(
                launch_node,
                launch_connections[index],
                edge_travel_minutes,
            )
        )
    edges.append(
        LocationEdge(
            recovery_node,
            pathway_index[("middle", 0)],
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
        launch_blocking_spots=launch_blocking_spots,
        landing_interference_pairs=landing_interference_pairs,
        runway_conflicts=runway_conflicts,
        assumptions=assumptions,
    )


class DeckOccupancy:
    """Static location capacity accounting for the project deck graph."""

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
        nodes_to_reserve = (target,)
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


def _scaled_cluster_sizes(
    total: int,
    requested: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    if sum(requested) == total:
        return requested
    weight = sum(requested)
    if weight <= 0:
        raise ValueError("parking-cluster weights must contain capacity")
    raw = [total * size / weight for size in requested]
    sizes = [math.floor(value) for value in raw]
    remainder = total - sum(sizes)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - sizes[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        sizes[index] += 1
    return tuple(sizes)  # type: ignore[return-value]


def _cluster_interference_pairs(
    cluster_range: range,
) -> Tuple[Tuple[int, int], ...]:
    spots = tuple(cluster_range)
    if len(spots) < 6:
        return ()
    return ((spots[-1], spots[-2]),)
