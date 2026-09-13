"""Tests for shared aircraft and vehicle traffic reservations."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.scenario import DeckGraph, LocationEdge, LocationNode
from env.traffic_planner import SpaceTimeTrafficPlanner


class SpaceTimeTrafficPlannerTest(unittest.TestCase):
    def test_node_conflict_is_resolved_by_waiting(self) -> None:
        graph = DeckGraph(
            (
                LocationNode("a", "pathway"),
                LocationNode("b", "pathway"),
                LocationNode("c", "pathway"),
            ),
            (
                LocationEdge("a", "b", 1.0),
                LocationEdge("c", "b", 1.0),
            ),
        )
        planner = SpaceTimeTrafficPlanner(graph)

        first = planner.plan("first", "a", "b", 0.0)
        second = planner.plan("second", "c", "b", 0.0)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(second.nodes, ("c", "c", "b"))
        self.assertEqual(second.ticks, (0, 1, 2))

    def test_head_on_edge_swap_is_rejected(self) -> None:
        graph = DeckGraph(
            (
                LocationNode("a", "pathway"),
                LocationNode("b", "pathway"),
            ),
            (LocationEdge("a", "b", 1.0),),
        )
        planner = SpaceTimeTrafficPlanner(graph, max_planning_horizon=5.0)

        self.assertIsNotNone(planner.plan("first", "a", "b", 0.0))
        self.assertIsNone(planner.plan("second", "b", "a", 0.0))

    def test_same_direction_edge_use_is_delayed(self) -> None:
        graph = DeckGraph(
            (
                LocationNode("a", "pathway"),
                LocationNode("b", "pathway"),
                LocationNode("c", "pathway"),
            ),
            (
                LocationEdge("a", "b", 1.0),
                LocationEdge("b", "c", 1.0),
            ),
        )
        planner = SpaceTimeTrafficPlanner(graph)

        first = planner.plan("first", "a", "c", 0.0)
        second = planner.plan(
            "second",
            "a",
            "c",
            0.0,
            shared_nodes=("a",),
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertGreater(second.end_time, first.end_time)


class ExplicitAssignmentTest(unittest.TestCase):
    def test_launch_selects_runway_and_dispatches_tractor(self) -> None:
        env = CarrierAircraftSchedulingEnv()
        env.reset(seed=7)

        action = env.complete_action(
            {"high_level": 3, "aircraft_id": 0}
        )
        _, _, _, info = env.step(action)

        self.assertFalse(info["invalid_action"])
        self.assertIn("target_id", action)
        self.assertEqual(
            env.active_deck_routes[0].target,
            env.deck_layout.launch_nodes[action["target_id"]],
        )
        tractor_id = env.active_tow_vehicles[0]
        tractor = next(
            vehicle
            for vehicle in env.service_vehicles
            if vehicle.vehicle_id == tractor_id
        )
        self.assertEqual(tractor.busy_aircraft_id, 0)
        self.assertIn(tractor_id, env.traffic_planner.routes)
        self.assertIn("tow:0", env.traffic_planner.routes)

        while 0 in env.active_tow_vehicles:
            env._advance_time_to_next_event()
        self.assertNotIn(0, env.active_tow_vehicles)
        self.assertIsNone(tractor.busy_aircraft_id)

    def test_service_action_selects_a_reachable_vehicle(self) -> None:
        env = CarrierAircraftSchedulingEnv()
        env.reset(seed=7)
        aircraft = env.aircraft[0]
        aircraft.fuel_status = 0
        aircraft.fuel_level = 0.2
        aircraft.launch_status = 0

        action = env.complete_action(
            {"high_level": 1, "aircraft_id": 0}
        )
        _, _, _, info = env.step(action)

        self.assertFalse(info["invalid_action"])
        self.assertIn("vehicle_id", action)
        self.assertEqual(
            env.active_service_vehicles[("fuel", 0)],
            action["vehicle_id"],
        )

    def test_continuous_fuel_duration_and_disabled_personnel(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "fuel_rate_per_minute": 0.1,
                "num_personnel": 0,
                "personnel_scheduling_enabled": False,
            }
        )
        env.reset(seed=7)
        aircraft = env.aircraft[0]
        aircraft.fuel_status = 0
        aircraft.fuel_level = 0.25
        aircraft.launch_status = 0

        action = env.complete_action(
            {"high_level": 1, "aircraft_id": 0}
        )
        env.step(action)
        fuel_event = next(
            event
            for event in env.event_queue
            if event.event_type == "fuel_done"
        )
        vehicle = next(
            item
            for item in env.service_vehicles
            if item.vehicle_id == action["vehicle_id"]
        )
        route = env.traffic_planner.routes[vehicle.vehicle_id]

        self.assertAlmostEqual(
            fuel_event.time,
            route.duration + 7.5,
        )
        self.assertEqual(env.free_personnel, 0)


if __name__ == "__main__":
    unittest.main()
