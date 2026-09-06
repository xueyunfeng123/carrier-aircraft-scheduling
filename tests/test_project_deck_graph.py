"""Tests for the project-core structural deck graph and route constraints."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.project_deck_graph import (
    DeckOccupancy,
    ProjectDeckGraphAssumptions,
    build_project_deck_layout,
)


class ProjectDeckLayoutTest(unittest.TestCase):
    def test_default_layout_matches_documented_area_capacities(self) -> None:
        layout = build_project_deck_layout(45)

        self.assertEqual(len(layout.parking_nodes), 45)
        self.assertEqual(len(layout.pathway_nodes), 19)
        self.assertEqual(len(layout.launch_nodes), 4)
        self.assertEqual(layout.parking_sections.count("launch_parking"), 10)
        self.assertEqual(layout.parking_sections.count("recovery_parking"), 10)
        self.assertEqual(layout.parking_sections.count("support_parking"), 25)

    def test_reserved_route_blocks_same_destination_until_release(self) -> None:
        layout = build_project_deck_layout(45)
        occupancy = DeckOccupancy(layout.graph)
        source = layout.parking_node(0)
        occupancy.occupy(source, 0)
        target, path, distance = occupancy.find_route(
            0,
            source,
            (layout.launch_nodes[0],),
        )

        occupancy.reserve_route(
            0,
            "launch_taxi",
            source,
            target,
            path,
            distance,
            release_source=True,
        )

        self.assertIsNone(
            occupancy.find_route(
                1,
                layout.parking_node(1),
                (layout.launch_nodes[0],),
            )
        )
        completed = occupancy.complete_route(0)
        self.assertEqual(completed.target, layout.launch_nodes[0])
        self.assertTrue(occupancy.occupants[layout.launch_nodes[0]])
        occupancy.release_target(0, layout.launch_nodes[0])
        self.assertFalse(occupancy.occupants[layout.launch_nodes[0]])

    def test_edge_travel_time_sets_route_duration(self) -> None:
        layout = build_project_deck_layout(
            45,
            ProjectDeckGraphAssumptions(edge_travel_minutes=2.0),
        )

        self.assertEqual(
            layout.graph.shortest_distance(
                layout.parking_node(0),
                layout.launch_nodes[0],
            ),
            4.0,
        )


class SpatialCarrierEnvironmentTest(unittest.TestCase):
    def test_launch_includes_taxi_and_releases_route(self) -> None:
        env = CarrierAircraftSchedulingEnv()
        env.reset(seed=7)

        env.step({"high_level": 3, "aircraft_id": 0})
        self.assertEqual(env.event_queue[0].event_type, "taxi_to_launch_done")
        self.assertEqual(env.event_queue[0].time, 2.0)
        self.assertEqual(env.get_state()["deck"]["active_movements"], 1)
        self.assertIsNone(env.parking_occupancy[0])

        env.step(None)
        self.assertEqual(env.time, 2.0)
        self.assertEqual(env.event_queue[0].event_type, "launch_done")
        self.assertEqual(env.get_state()["deck"]["active_movements"], 0)
        self.assertEqual(env.get_state()["deck"]["free_launch_positions"], 3)

        env.step(None)
        self.assertEqual(env.time, 3.0)
        self.assertTrue(env.aircraft[0].is_airborne)
        self.assertEqual(env.get_state()["deck"]["free_launch_positions"], 4)

    def test_recovery_route_blocks_conflicting_launch_paths(self) -> None:
        env = CarrierAircraftSchedulingEnv()
        env.reset(seed=7)
        env.step({"high_level": 3, "aircraft_id": 0})
        env.step(None)
        env.step(None)
        env._start_wave(1)

        env.step({"high_level": 0, "aircraft_id": 0})

        self.assertEqual(env.get_state()["deck"]["active_movements"], 1)
        self.assertEqual(env.get_high_level_action_mask()[3], 0)
        reserved_spot = env.aircraft[0].spot_id
        self.assertEqual(env.parking_occupancy[reserved_spot], 0)

        env.step(None)
        self.assertFalse(env.aircraft[0].is_airborne)
        self.assertEqual(env.aircraft[0].parking_status, 2)
        self.assertEqual(env.get_state()["deck"]["active_movements"], 0)
        env.deck_occupancy.validate()


if __name__ == "__main__":
    unittest.main()
