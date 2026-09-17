"""Tests for optimal fixed-batch conflict-based search."""

from __future__ import annotations

import unittest

from env.cbs_planner import (
    CBSExpansionLimitError,
    RouteRequest,
)
from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.project_deck_graph import build_project_deck_layout
from env.scenario import DeckGraph, LocationEdge, LocationNode
from env.traffic_planner import SpaceTimeTrafficPlanner, TimedRoute


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

    def test_environment_replans_current_service_batch(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {"cbs_replan_enabled": True}
        )
        env.reset(seed=7)
        first = env.traffic_planner.plan(
            "fuel_0",
            "deck_pathway_middle_3",
            "deck_parking_42",
            0.0,
            shared_nodes=(
                "deck_pathway_middle_3",
                "deck_parking_42",
            ),
        )
        second = env.traffic_planner.plan(
            "inspection_0",
            "deck_pathway_south_4",
            "deck_parking_7",
            0.0,
            shared_nodes=(
                "deck_pathway_south_4",
                "deck_parking_7",
            ),
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        env.active_service_vehicles = {
            ("fuel", 0): "fuel_0",
            ("inspection", 1): "inspection_0",
        }
        env.aircraft[0].fuel_remaining = 13.0
        env.aircraft[1].inspection_remaining = 16.0
        env._push_event(13.0, "fuel_done", 0)
        env._push_event(16.0, "inspection_done", 1)
        env._cbs_route_dispatch_times = {
            "fuel_0": 0.0,
            "inspection_0": 0.0,
        }

        env._replan_current_cbs_batch()

        self.assertEqual(
            env.cbs_replan_stats["improved_batches"],
            1,
        )
        self.assertEqual(
            env.cbs_replan_stats["sum_of_costs_before"],
            11.0,
        )
        self.assertEqual(
            env.cbs_replan_stats["sum_of_costs_after"],
            10.0,
        )
        event_times = {
            (event.event_type, event.aircraft_id): event.time
            for event in env.event_queue
        }
        self.assertEqual(event_times[("fuel_done", 0)], 14.0)
        self.assertEqual(
            event_times[("inspection_done", 1)],
            14.0,
        )

    def test_environment_updates_tow_completion_after_cbs(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(
            {"cbs_replan_enabled": True}
        )
        env.reset(seed=7)
        source = env.deck_layout.parking_node(0)
        target = env.deck_layout.launch_nodes[0]
        old_route = TimedRoute(
            entity_id="tow:0",
            source=source,
            target=target,
            nodes=(source, target),
            ticks=(0, 2),
            start_time=0.0,
            end_time=2.0,
        )
        new_route = TimedRoute(
            entity_id="tow:0",
            source=source,
            target=target,
            nodes=(source, source, target),
            ticks=(0, 1, 3),
            start_time=0.0,
            end_time=3.0,
        )
        reservation = env.deck_occupancy.reserve_route(
            aircraft_id=0,
            operation="launch_taxi",
            source=source,
            target=target,
            path=old_route.nodes,
            distance=old_route.duration,
            release_source=True,
        )
        env.active_deck_routes[0] = reservation
        env._push_event(
            old_route.end_time,
            "taxi_to_launch_done",
            0,
        )

        env._apply_cbs_route_updates(
            {"tow:0": old_route},
            {"tow:0": new_route},
        )

        updated = env.active_deck_routes[0]
        self.assertEqual(updated.path, new_route.nodes)
        self.assertEqual(updated.distance, 3.0)
        self.assertEqual(env.event_queue[0].time, 3.0)

    def test_environment_recomputes_parallel_arm_remaining(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(
            {"cbs_replan_enabled": True}
        )
        env.reset(seed=7)
        source = "deck_pathway_middle_3"
        target = "deck_parking_42"
        old_route = TimedRoute(
            entity_id="arm_0",
            source=source,
            target=target,
            nodes=(source, target),
            ticks=(0, 12),
            start_time=0.0,
            end_time=12.0,
        )
        new_route = TimedRoute(
            entity_id="arm_0",
            source=source,
            target=target,
            nodes=(source, target),
            ticks=(0, 7),
            start_time=0.0,
            end_time=7.0,
        )
        env.active_service_vehicles = {
            ("arm", 0): "arm_0",
        }
        env.aircraft[0].arm_remaining = 16.0
        env._push_event(
            10.0,
            "ammo_to_assembly_done",
            0,
        )

        env._apply_cbs_route_updates(
            {"arm_0": old_route},
            {"arm_0": new_route},
        )

        self.assertEqual(
            env.aircraft[0].arm_vehicle_ready_time,
            7.0,
        )
        self.assertEqual(env.aircraft[0].arm_remaining, 14.0)


if __name__ == "__main__":
    unittest.main()
