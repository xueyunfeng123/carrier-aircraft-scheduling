"""Tests for optimal fixed-batch conflict-based search."""

from __future__ import annotations

import unittest

from env.cbs_planner import (
    CBSExpansionLimitError,
    RouteRequest,
)
from env.project_deck_graph import build_project_deck_layout
from env.scenario import DeckGraph, LocationEdge, LocationNode
from env.traffic_planner import SpaceTimeTrafficPlanner


def build_asymmetric_conflict_graph() -> DeckGraph:
    return DeckGraph(
        tuple(
            LocationNode(node_id, "pathway")
            for node_id in ("s", "c", "d", "p")
        ),
        (
            LocationEdge("s", "c", 1.0),
            LocationEdge("c", "d", 5.0),
            LocationEdge("s", "p", 3.0),
            LocationEdge("p", "d", 4.0),
        ),
    )


class ConflictBasedSearchPlannerTest(unittest.TestCase):
    def test_cbs_improves_fixed_priority_sum_of_costs(self) -> None:
        graph = build_asymmetric_conflict_graph()
        sequential = SpaceTimeTrafficPlanner(
            graph,
            max_planning_horizon=20.0,
        )
        first = sequential.plan(
            "a",
            "s",
            "d",
            0.0,
            shared_nodes=("s", "d"),
        )
        second = sequential.plan(
            "b",
            "c",
            "d",
            0.0,
            shared_nodes=("c", "d"),
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        sequential_cost = first.duration + second.duration

        planner = SpaceTimeTrafficPlanner(
            graph,
            max_planning_horizon=20.0,
        )
        plan = planner.plan_batch(
            (
                RouteRequest(
                    "a",
                    "s",
                    "d",
                    0.0,
                    ("s", "d"),
                ),
                RouteRequest(
                    "b",
                    "c",
                    "d",
                    0.0,
                    ("c", "d"),
                ),
            ),
            reserve=False,
        )

        self.assertIsNotNone(plan)
        self.assertEqual(sequential_cost, 15.0)
        self.assertEqual(plan.sum_of_costs, 12.0)
        self.assertEqual(plan.routes["a"].nodes, ("s", "p", "d"))
        self.assertEqual(plan.routes["b"].nodes, ("c", "d"))
        self.assertEqual(planner.routes, {})

    def test_cbs_commits_every_route_after_finding_solution(self) -> None:
        planner = SpaceTimeTrafficPlanner(
            build_asymmetric_conflict_graph(),
            max_planning_horizon=20.0,
        )
        requests = (
            RouteRequest("a", "s", "d", 0.0, ("s", "d")),
            RouteRequest("b", "c", "d", 0.0, ("c", "d")),
        )

        plan = planner.plan_batch(requests)

        self.assertIsNotNone(plan)
        self.assertEqual(set(planner.routes), {"a", "b"})
        self.assertEqual(
            planner.routes["a"],
            plan.routes["a"],
        )
        self.assertEqual(
            planner.routes["b"],
            plan.routes["b"],
        )

    def test_cbs_reduces_soc_on_project_deck_graph(self) -> None:
        layout = build_project_deck_layout(45)
        requests = (
            RouteRequest(
                "first",
                "deck_pathway_middle_3",
                "deck_parking_42",
                0.0,
                (
                    "deck_pathway_middle_3",
                    "deck_parking_42",
                ),
            ),
            RouteRequest(
                "second",
                "deck_pathway_south_4",
                "deck_parking_7",
                0.0,
                (
                    "deck_pathway_south_4",
                    "deck_parking_7",
                ),
            ),
        )
        sequential = SpaceTimeTrafficPlanner(layout.graph)
        sequential_routes = [
            sequential.plan(
                request.entity_id,
                request.source,
                request.target,
                request.start_time,
                request.shared_nodes,
            )
            for request in requests
        ]
        self.assertTrue(all(sequential_routes))

        joint = SpaceTimeTrafficPlanner(layout.graph)
        plan = joint.plan_batch(requests, reserve=False)

        self.assertIsNotNone(plan)
        self.assertEqual(
            sum(route.duration for route in sequential_routes),
            11.0,
        )
        self.assertEqual(plan.sum_of_costs, 10.0)

    def test_cbs_respects_preexisting_reservations(self) -> None:
        graph = DeckGraph(
            tuple(
                LocationNode(node_id, "pathway")
                for node_id in ("a", "b", "c")
            ),
            (
                LocationEdge("a", "b", 1.0),
                LocationEdge("b", "c", 1.0),
            ),
        )
        planner = SpaceTimeTrafficPlanner(graph)
        blocker = planner.plan("blocker", "a", "b", 0.0)
        self.assertIsNotNone(blocker)

        plan = planner.plan_batch(
            (
                RouteRequest(
                    "new",
                    "c",
                    "b",
                    0.0,
                    ("c",),
                ),
            ),
            reserve=False,
        )

        self.assertIsNotNone(plan)
        self.assertEqual(
            plan.routes["new"].nodes,
            ("c", "c", "b"),
        )

    def test_expansion_limit_is_reported(self) -> None:
        planner = SpaceTimeTrafficPlanner(
            build_asymmetric_conflict_graph(),
            max_planning_horizon=20.0,
        )

        with self.assertRaises(CBSExpansionLimitError):
            planner.plan_batch(
                (
                    RouteRequest("a", "s", "d", 0.0),
                    RouteRequest("b", "c", "d", 0.0),
                ),
                reserve=False,
                max_expanded_nodes=1,
            )


if __name__ == "__main__":
    unittest.main()
