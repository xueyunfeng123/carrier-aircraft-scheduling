"""Optimal conflict-based search for a fixed batch of deck movements."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from env.traffic_planner import SpaceTimeTrafficPlanner, TimedRoute


VertexConstraint = Tuple[str, int]
EdgeConstraint = Tuple[str, str, int]


@dataclass(frozen=True)
class RouteRequest:
    """One fixed start, goal, and release time for CBS."""

    entity_id: str
    source: str
    target: str
    start_time: float
    shared_nodes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CBSPlan:
    """A conflict-free route set minimizing sum of route durations."""

    routes: Mapping[str, TimedRoute]
    sum_of_costs: float
    makespan: float
    expanded_nodes: int


@dataclass(frozen=True)
class _Conflict:
    first_entity: str
    second_entity: str
    tick: int
    node: Optional[str] = None
    first_edge: Optional[Tuple[str, str]] = None
    second_edge: Optional[Tuple[str, str]] = None


@dataclass
class _SearchNode:
    routes: Dict[str, TimedRoute]
    vertex_constraints: Dict[str, frozenset[VertexConstraint]]
    edge_constraints: Dict[str, frozenset[EdgeConstraint]]


class CBSExpansionLimitError(RuntimeError):
    """Raised when an explicitly bounded CBS search exhausts its budget."""


class ConflictBasedSearchPlanner:
    """Jointly plan a fixed movement batch with optimal sum of costs.

    Existing reservations in ``traffic_planner`` are immutable obstacles.
    Requests in the new batch are planned jointly with standard CBS. The
    optimality guarantee applies only to this fixed batch and the configured
    finite low-level planning horizon.
    """

    def __init__(self, traffic_planner: SpaceTimeTrafficPlanner):
        self.traffic_planner = traffic_planner

    def plan(
        self,
        requests: Sequence[RouteRequest],
        reserve: bool = True,
        max_expanded_nodes: Optional[int] = None,
    ) -> Optional[CBSPlan]:
        normalized = tuple(requests)
        if not normalized:
            return CBSPlan({}, 0.0, 0.0, 0)
        self._validate_requests(normalized)

        request_by_entity = {
            request.entity_id: request
            for request in normalized
        }
        vertex_constraints = {
            request.entity_id: frozenset()
            for request in normalized
        }
        edge_constraints = {
            request.entity_id: frozenset()
            for request in normalized
        }
        routes: Dict[str, TimedRoute] = {}
        for request in normalized:
            route = self._plan_request(
                request,
                vertex_constraints[request.entity_id],
                edge_constraints[request.entity_id],
            )
            if route is None:
                return None
            routes[request.entity_id] = route

        root = _SearchNode(
            routes=routes,
            vertex_constraints=vertex_constraints,
            edge_constraints=edge_constraints,
        )
        sequence = itertools.count()
        queue = []
        self._push(queue, root, next(sequence))
        visited = {self._constraint_signature(root)}
        expanded_nodes = 0

        while queue:
            if (
                max_expanded_nodes is not None
                and expanded_nodes >= max_expanded_nodes
            ):
                raise CBSExpansionLimitError(
                    "CBS high-level expansion limit reached"
                )
            _, _, _, _, node = heapq.heappop(queue)
            expanded_nodes += 1
            conflict = self._first_conflict(
                node.routes,
                request_by_entity,
            )
            if conflict is None:
                if reserve:
                    for request in normalized:
                        self.traffic_planner.reserve(
                            node.routes[request.entity_id],
                            request.shared_nodes,
                        )
                return CBSPlan(
                    routes=dict(node.routes),
                    sum_of_costs=self._sum_of_costs(node.routes),
                    makespan=max(
                        route.end_time
                        for route in node.routes.values()
                    ),
                    expanded_nodes=expanded_nodes,
                )

            for entity_id in (
                conflict.first_entity,
                conflict.second_entity,
            ):
                child = self._branch(node, conflict, entity_id)
                signature = self._constraint_signature(child)
                if signature in visited:
                    continue
                visited.add(signature)
                request = request_by_entity[entity_id]
                route = self._plan_request(
                    request,
                    child.vertex_constraints[entity_id],
                    child.edge_constraints[entity_id],
                )
                if route is None:
                    continue
                child.routes[entity_id] = route
                self._push(queue, child, next(sequence))
        return None

    def _plan_request(
        self,
        request: RouteRequest,
        vertex_constraints: Iterable[VertexConstraint],
        edge_constraints: Iterable[EdgeConstraint],
    ) -> Optional[TimedRoute]:
        return self.traffic_planner.plan(
            request.entity_id,
            request.source,
            request.target,
            request.start_time,
            shared_nodes=request.shared_nodes,
            reserve=False,
            vertex_constraints=vertex_constraints,
            edge_constraints=edge_constraints,
        )

    def _validate_requests(
        self,
        requests: Sequence[RouteRequest],
    ) -> None:
        entity_ids = [request.entity_id for request in requests]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("CBS route request entity IDs must be unique")
        active_ids = set(self.traffic_planner.routes)
        duplicates = active_ids.intersection(entity_ids)
        if duplicates:
            raise ValueError(
                "CBS requests conflict with active route IDs: "
                f"{sorted(duplicates)}"
            )

    def _branch(
        self,
        node: _SearchNode,
        conflict: _Conflict,
        entity_id: str,
    ) -> _SearchNode:
        vertex_constraints = dict(node.vertex_constraints)
        edge_constraints = dict(node.edge_constraints)
        if conflict.node is not None:
            updated = set(vertex_constraints[entity_id])
            updated.add((conflict.node, conflict.tick))
            vertex_constraints[entity_id] = frozenset(updated)
        else:
            edge = (
                conflict.first_edge
                if entity_id == conflict.first_entity
                else conflict.second_edge
            )
            if edge is None:
                raise RuntimeError("edge conflict is missing a transition")
            updated = set(edge_constraints[entity_id])
            updated.add((edge[0], edge[1], conflict.tick))
            edge_constraints[entity_id] = frozenset(updated)
        return _SearchNode(
            routes=dict(node.routes),
            vertex_constraints=vertex_constraints,
            edge_constraints=edge_constraints,
        )

    def _first_conflict(
        self,
        routes: Mapping[str, TimedRoute],
        requests: Mapping[str, RouteRequest],
    ) -> Optional[_Conflict]:
        entity_ids = sorted(routes)
        best: Optional[Tuple[int, int, _Conflict]] = None
        for first_index, first_entity in enumerate(entity_ids):
            for second_entity in entity_ids[first_index + 1 :]:
                conflict = self._pair_conflict(
                    routes[first_entity],
                    requests[first_entity],
                    routes[second_entity],
                    requests[second_entity],
                )
                if conflict is None:
                    continue
                priority = (
                    conflict.tick,
                    0 if conflict.node is not None else 1,
                    conflict,
                )
                if best is None or priority[:2] < best[:2]:
                    best = priority
        return None if best is None else best[2]

    def _pair_conflict(
        self,
        first_route: TimedRoute,
        first_request: RouteRequest,
        second_route: TimedRoute,
        second_request: RouteRequest,
    ) -> Optional[_Conflict]:
        first_nodes = {
            (node, tick)
            for node, tick in zip(
                first_route.nodes,
                first_route.ticks,
            )
            if node not in first_request.shared_nodes
        }
        second_nodes = {
            (node, tick)
            for node, tick in zip(
                second_route.nodes,
                second_route.ticks,
            )
            if node not in second_request.shared_nodes
        }
        shared_occupancies = first_nodes.intersection(second_nodes)
        node_conflict = None
        if shared_occupancies:
            node, tick = min(
                shared_occupancies,
                key=lambda item: (item[1], item[0]),
            )
            node_conflict = _Conflict(
                first_entity=first_route.entity_id,
                second_entity=second_route.entity_id,
                tick=tick,
                node=node,
            )

        first_edges = self._edge_occupancies(first_route)
        second_edges = self._edge_occupancies(second_route)
        edge_conflict = None
        for tick in sorted(set(first_edges).intersection(second_edges)):
            first_edge = first_edges[tick]
            second_edge = second_edges[tick]
            if frozenset(first_edge) != frozenset(second_edge):
                continue
            edge_conflict = _Conflict(
                first_entity=first_route.entity_id,
                second_entity=second_route.entity_id,
                tick=tick,
                first_edge=first_edge,
                second_edge=second_edge,
            )
            break

        conflicts = [
            conflict
            for conflict in (node_conflict, edge_conflict)
            if conflict is not None
        ]
        if not conflicts:
            return None
        return min(
            conflicts,
            key=lambda item: (
                item.tick,
                0 if item.node is not None else 1,
            ),
        )

    @staticmethod
    def _edge_occupancies(
        route: TimedRoute,
    ) -> Dict[int, Tuple[str, str]]:
        occupancies = {}
        for index in range(1, len(route.nodes)):
            source = route.nodes[index - 1]
            target = route.nodes[index]
            if source == target:
                continue
            for tick in range(
                route.ticks[index - 1],
                route.ticks[index],
            ):
                occupancies[tick] = (source, target)
        return occupancies

    def _push(
        self,
        queue,
        node: _SearchNode,
        sequence: int,
    ) -> None:
        heapq.heappush(
            queue,
            (
                self._sum_of_costs(node.routes),
                max(route.end_time for route in node.routes.values()),
                self._constraint_count(node),
                sequence,
                node,
            ),
        )

    @staticmethod
    def _sum_of_costs(routes: Mapping[str, TimedRoute]) -> float:
        return sum(
            route.end_time - route.start_time
            for route in routes.values()
        )

    @staticmethod
    def _constraint_count(node: _SearchNode) -> int:
        return sum(
            len(items)
            for items in node.vertex_constraints.values()
        ) + sum(
            len(items)
            for items in node.edge_constraints.values()
        )

    @staticmethod
    def _constraint_signature(node: _SearchNode) -> Tuple:
        return (
            tuple(
                (entity_id, tuple(sorted(items)))
                for entity_id, items in sorted(
                    node.vertex_constraints.items()
                )
            ),
            tuple(
                (entity_id, tuple(sorted(items)))
                for entity_id, items in sorted(
                    node.edge_constraints.items()
                )
            ),
        )
