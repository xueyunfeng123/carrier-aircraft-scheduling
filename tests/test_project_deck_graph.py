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
        self.assertEqual(len(layout.pathway_nodes), 15)
        self.assertEqual(len(layout.launch_nodes), 3)
        self.assertEqual(layout.parking_sections.count("northwest"), 10)
        self.assertEqual(layout.parking_sections.count("northeast"), 13)
        self.assertEqual(layout.parking_sections.count("southwest"), 12)
        self.assertEqual(layout.parking_sections.count("southeast"), 10)

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
            12.0,
        )


class SpatialCarrierEnvironmentTest(unittest.TestCase):
    def test_launch_includes_taxi_and_releases_route(self) -> None:
        env = CarrierAircraftSchedulingEnv()
        env.reset(seed=7)

        env.step({"high_level": 3, "aircraft_id": 0})
        taxi_event = next(
            event
            for event in env.event_queue
            if event.event_type == "taxi_to_launch_done"
        )
        taxi_end = taxi_event.time
        self.assertGreater(taxi_end, 0.0)
        self.assertEqual(env.get_state()["deck"]["active_movements"], 1)
        self.assertIsNone(env.parking_occupancy[0])

        while env.time < taxi_end:
            env._advance_time_to_next_event()
        self.assertEqual(env.time, taxi_end)
        self.assertEqual(env.event_queue[0].event_type, "launch_done")
        self.assertEqual(env.get_state()["deck"]["active_movements"], 0)
        self.assertEqual(env.get_state()["deck"]["free_launch_positions"], 2)

        env._advance_time_to_next_event()
        self.assertEqual(env.time, taxi_end + 1.0)
        self.assertTrue(env.aircraft[0].is_airborne)
        self.assertEqual(env.get_state()["deck"]["free_launch_positions"], 3)

    def test_recovery_route_keeps_non_conflicting_launch_paths_available(self) -> None:
        env = CarrierAircraftSchedulingEnv()
        env.reset(seed=7)
        env.step({"high_level": 3, "aircraft_id": 0})
        while not env.aircraft[0].is_airborne:
            env._advance_time_to_next_event()
        env._start_wave(1)

        env.step({"high_level": 0, "aircraft_id": 0})

        self.assertEqual(env.get_state()["deck"]["active_movements"], 1)
        self.assertEqual(env.get_high_level_action_mask()[3], 1)
        reserved_spot = env.aircraft[0].spot_id
        self.assertEqual(env.parking_occupancy[reserved_spot], 0)

        while env.aircraft[0].is_airborne:
            env._advance_time_to_next_event()
        self.assertFalse(env.aircraft[0].is_airborne)
        self.assertEqual(env.aircraft[0].parking_status, 2)
        self.assertEqual(env.get_state()["deck"]["active_movements"], 0)
        env.deck_occupancy.validate()

    def test_compatible_service_vehicles_share_aircraft_parking_spot(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "fuel_time_mean": 30.0,
                "fuel_time_std": 0.0,
                "inspection_time_mean": 30.0,
                "inspection_time_std": 0.0,
                "ammo_extract_time_min": 0.1,
                "ammo_extract_time_max": 0.1,
                "lower_lift_time_mean": 0.1,
                "lower_lift_time_std": 0.0,
            }
        )
        env.reset(seed=7)
        aircraft = env.aircraft[0]
        aircraft.fuel_status = 0
        aircraft.fuel_level = 0.2
        aircraft.inspection_status = 0
        aircraft.arm_status = 0
        aircraft.arm_quantity_required = 1
        aircraft.launch_status = 0

        env.step({"high_level": 1, "aircraft_id": 0})
        env.step({"high_level": 4, "aircraft_id": 0})
        env.step({"high_level": 2, "aircraft_id": 0})
        env._advance_time_to_next_event()

        active_types = {
            vehicle.service_type
            for vehicle in env.service_vehicles
            if vehicle.busy_aircraft_id == 0
        }
        self.assertEqual(active_types, {"fuel", "inspection", "arm"})
        self.assertEqual(env.parking_occupancy[0], 0)


if __name__ == "__main__":
    unittest.main()
