"""Time-expanded conflict-aware routing for deck aircraft and vehicles."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from env.scenario import DeckGraph


@dataclass(frozen=True)
class TimedRoute:
    entity_id: str
    source: str
    target: str
    nodes: Tuple[str, ...]
    ticks: Tuple[int, ...]
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class SpaceTimeTrafficPlanner:
    """Cooperative A* reservation table on a discrete weighted graph.

    Node reservations prevent two moving entities from occupying the same
    pathway at the same tick. Edge reservations additionally prevent both
    same-direction overlap and head-on traversal.
    """

    def __init__(
        self,
        graph: DeckGraph,
        time_step: float = 1.0,
        max_planning_horizon: float = 120.0,
    ):
        if time_step <= 0:
            raise ValueError("traffic time step must be positive")
        if max_planning_horizon <= 0:
            raise ValueError("traffic planning horizon must be positive")
        self.graph = graph
        self.time_step = float(time_step)
        self.max_ticks = int(math.ceil(max_planning_horizon / time_step))
        self.node_reservations: Dict[Tuple[str, int], str] = {}
        self.edge_reservations: Dict[Tuple[str, str, int], str] = {}
        self.routes: Dict[str, TimedRoute] = {}
        self._distance_cache: Dict[Tuple[str, str], float] = {}

    def plan(
        self,
        entity_id: str,
        source: str,
        target: str,
        start_time: float,
        shared_nodes: Iterable[str] = (),
        reserve: bool = True,
    ) -> Optional[TimedRoute]:
        self.graph.node(source)
        self.graph.node(target)
        shared = frozenset(shared_nodes)
        start_tick = self._ceil_tick(start_time)
        queue = [(0.0, start_tick, source)]
        costs = {(source, start_tick): 0.0}
        predecessor: Dict[
            Tuple[str, int],
            Tuple[Tuple[str, int], bool],
        ] = {}
        goal: Optional[Tuple[str, int]] = None

        while queue:
            _, tick, node = heapq.heappop(queue)
            state = (node, tick)
            cost = costs[state]
            if node == target:
                goal = state
                break
            if tick - start_tick >= self.max_ticks:
                continue

            transitions = [(node, 1, True)]
            transitions.extend(
                (neighbor, self._edge_ticks(distance), False)
                for neighbor, distance in self.graph.neighbors(node)
            )
            for next_node, duration, is_wait in transitions:
                next_tick = tick + duration
                if next_tick - start_tick > self.max_ticks:
                    continue
                if not self._transition_available(
                    entity_id,
                    node,
                    next_node,
                    tick,
                    next_tick,
                    shared,
                    is_wait,
                ):
                    continue
                next_state = (next_node, next_tick)
                next_cost = cost + duration
                if next_cost >= costs.get(next_state, math.inf):
                    continue
                costs[next_state] = next_cost
                predecessor[next_state] = (state, is_wait)
                distance_key = (next_node, target)
                heuristic = self._distance_cache.get(distance_key)
                if heuristic is None:
                    heuristic = self.graph.shortest_distance(
                        next_node,
                        target,
                    )
                    self._distance_cache[distance_key] = heuristic
                heapq.heappush(
                    queue,
                    (next_cost + heuristic / self.time_step, next_tick, next_node),
                )

        if goal is None:
            return None
        states = [goal]
        while states[-1] != (source, start_tick):
            states.append(predecessor[states[-1]][0])
        states.reverse()
        route = TimedRoute(
            entity_id=entity_id,
            source=source,
            target=target,
            nodes=tuple(node for node, _ in states),
            ticks=tuple(tick for _, tick in states),
            start_time=start_tick * self.time_step,
            end_time=goal[1] * self.time_step,
        )
        if reserve:
            self.reserve(route, shared)
        return route

    def reserve(
        self,
        route: TimedRoute,
        shared_nodes: Iterable[str] = (),
    ) -> None:
        shared = frozenset(shared_nodes)
        for index, (node, tick) in enumerate(zip(route.nodes, route.ticks)):
            if node not in shared:
                self.node_reservations[(node, tick)] = route.entity_id
            if index == 0:
                continue
            source = route.nodes[index - 1]
            start_tick = route.ticks[index - 1]
            if source == node:
                continue
            for edge_tick in range(start_tick, tick):
                self.edge_reservations[
                    (source, node, edge_tick)
                ] = route.entity_id
        self.routes[route.entity_id] = route

    def prune(self, current_time: float) -> None:
        current_tick = int(math.floor(current_time / self.time_step))
        self.node_reservations = {
            key: owner
            for key, owner in self.node_reservations.items()
            if key[1] >= current_tick
        }
        self.edge_reservations = {
            key: owner
            for key, owner in self.edge_reservations.items()
            if key[2] >= current_tick
        }
        self.routes = {
            entity_id: route
            for entity_id, route in self.routes.items()
            if route.end_time >= current_time
        }

    def release(self, entity_id: str) -> None:
        """Remove every future reservation owned by one moving entity."""

        self.node_reservations = {
            key: owner
            for key, owner in self.node_reservations.items()
            if owner != entity_id
        }
        self.edge_reservations = {
            key: owner
            for key, owner in self.edge_reservations.items()
            if owner != entity_id
        }
        self.routes.pop(entity_id, None)

    def _transition_available(
        self,
        entity_id: str,
        source: str,
        target: str,
        start_tick: int,
        end_tick: int,
        shared_nodes: frozenset[str],
        is_wait: bool,
    ) -> bool:
        if target not in shared_nodes:
            owner = self.node_reservations.get((target, end_tick))
            if owner is not None and owner != entity_id:
                return False
        if is_wait:
            return True
        for tick in range(start_tick, end_tick):
            owners = (
                self.edge_reservations.get((source, target, tick)),
                self.edge_reservations.get((target, source, tick)),
            )
            if any(owner is not None and owner != entity_id for owner in owners):
                return False
        return True

    def _ceil_tick(self, time_value: float) -> int:
        return int(math.ceil(time_value / self.time_step - 1.0e-9))

    def _edge_ticks(self, distance: float) -> int:
        return max(1, int(math.ceil(distance / self.time_step)))
