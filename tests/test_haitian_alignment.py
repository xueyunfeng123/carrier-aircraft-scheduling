"""Contract tests for the executable Haitian competition alignment."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import resolve_config


class HaitianAlignmentConfigTest(unittest.TestCase):
    def test_profile_applies_published_resources_and_declared_assumptions(
        self,
    ) -> None:
        config = resolve_config({"scenario_profile": "haitian_2026"})

        self.assertEqual(config["num_aircraft"], 45)
        self.assertEqual(config["num_total_aircraft"], 75)
        self.assertEqual(config["group_size"], 20)
        self.assertEqual(config["num_launch_positions"], 4)
        self.assertEqual(config["num_launch_channels"], 1)
        self.assertEqual(config["num_aircraft_lifts"], 2)
        self.assertEqual(config["fuel_service_mode"], "fixed_station")
        self.assertEqual(config["fuel_duration_model"], "normal")
        self.assertEqual(config["fuel_time_mean"], 20.0)
        self.assertEqual(config["fuel_time_std"], 3.0)
        self.assertFalse(config["routine_inspection_enabled"])
        self.assertTrue(config["personnel_scheduling_enabled"])
        self.assertEqual(config["num_personnel"], 50)
        self.assertTrue(config["sea_state_effects_enabled"])
        self.assertTrue(config["stochastic_flight_failures_enabled"])

    def test_profile_exposes_four_launch_positions_and_assumptions(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(
            {"scenario_profile": "haitian_2026"}
        )

        self.assertEqual(env.scenario_profile.name, "haitian_2026")
        self.assertEqual(len(env.deck_layout.launch_nodes), 4)
        self.assertTrue(env.scenario_profile.unresolved_parameters)
        metadata = dict(env.scenario_profile.metadata)
        self.assertEqual(metadata["total_inventory"], 75)
        self.assertEqual(
            metadata["fuel_time_std_assumption_minutes"],
            3.0,
        )


class HaitianAlignmentBehaviorTest(unittest.TestCase):
    def test_fixed_fuel_station_has_two_spot_coverage_and_no_travel(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "scenario_profile": "haitian_2026",
                "scenario_tape_enabled": True,
                "stochastic_flight_failures_enabled": False,
            }
        )
        env.reset(seed=7)
        aircraft = env.aircraft[0]
        aircraft.fuel_status = 0
        aircraft.fuel_level = 0.2
        aircraft.launch_status = 0

        action = env.complete_action(
            {"high_level": 1, "aircraft_id": 0}
        )
        _, _, _, info = env.step(action)
        fuel_event = next(
            event
            for event in env.event_queue
            if event.event_type == "fuel_done"
            and event.aircraft_id == 0
        )
        fuel_start = next(
            event
            for event in env.event_log
            if event["event_type"] == "fuel_start"
            and event["aircraft_id"] == 0
        )

        self.assertFalse(info["invalid_action"])
        self.assertEqual(action["vehicle_id"], "fuel_0")
        station = next(
            vehicle
            for vehicle in env.service_vehicles
            if vehicle.vehicle_id == "fuel_0"
        )
        self.assertEqual(station.covered_spot_ids, (0, 1))
        self.assertEqual(fuel_start["travel_duration"], 0.0)
        self.assertEqual(fuel_event.time, fuel_start["duration"])
        self.assertEqual(env.free_personnel, 46)
        self.assertNotIn(40, {
            spot
            for vehicle in env.service_vehicles
            if vehicle.service_type == "fuel"
            for spot in vehicle.covered_spot_ids
        })

    def test_recovery_does_not_create_routine_inspection_work(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "scenario_profile": "haitian_2026",
                "spatial_graph_enabled": False,
                "sea_state_effects_enabled": False,
                "stochastic_flight_failures_enabled": False,
            }
        )
        env.reset(seed=7)
        env.step({"high_level": 3, "aircraft_id": 0})
        env._advance_time_to_next_event()
        env._start_wave(1)
        env.step({"high_level": 0, "aircraft_id": 0})
        env._advance_time_to_next_event()

        self.assertEqual(env.aircraft[0].inspection_status, 2)
        self.assertNotIn(0, env._candidate_sets()["I"])

    def test_sea_state_above_six_blocks_launch_and_recovery(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "scenario_profile": "haitian_2026",
                "sea_state": 6.1,
                "stochastic_flight_failures_enabled": False,
            }
        )
        env.reset(seed=7)

        self.assertEqual(env.get_high_level_action_mask()[3], 0)
        env.aircraft[0].is_airborne = True
        env.aircraft[0].pending_recovery = True
        env.aircraft[0].recovery_status = 0
        env._invalidate_planning_cache()
        self.assertEqual(env.get_high_level_action_mask()[0], 0)

    def test_runtime_failure_uses_mtbf_and_one_to_six_hour_repair(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "scenario_profile": "haitian_2026",
                "spatial_graph_enabled": False,
                "sea_state_effects_enabled": False,
                "scenario_tape_enabled": True,
                "aircraft_failure_mtbf_flight_hours": 0.0001,
            }
        )
        env.reset(seed=7)
        env.step({"high_level": 3, "aircraft_id": 0})
        env._advance_time_to_next_event()

        failures = [
            spec
            for spec in env.disruption_schedule
            if spec.kind == "aircraft_failure"
            and int(spec.target) == 0
        ]
        self.assertEqual(len(failures), 1)
        duration = failures[0].end_time - failures[0].start_time
        self.assertGreaterEqual(duration, 60.0)
        self.assertLessEqual(duration, 360.0)

    def test_two_aircraft_lifts_bound_failure_transfers(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "scenario_profile": "haitian_2026",
                "stochastic_flight_failures_enabled": False,
                "disruptions": [
                    {
                        "kind": "aircraft_failure",
                        "target": 0,
                        "start_time": 0.0,
                        "end_time": 120.0,
                    },
                    {
                        "kind": "aircraft_failure",
                        "target": 1,
                        "start_time": 0.0,
                        "end_time": 120.0,
                    },
                ],
            }
        )
        env.reset(seed=7)

        transfer_events = [
            event
            for event in env.event_queue
            if event.event_type
            in {
                "hangar_transfer_done",
                "replacement_transfer_done",
            }
        ]
        self.assertEqual(env.free_aircraft_lifts, 0)
        self.assertEqual(len(transfer_events), 2)
        self.assertEqual(len(env.failure_transfer_started), 1)

    def test_long_repairs_do_not_overlap_later_aircraft_holds(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "scenario_profile": "haitian_2026",
                "disruption_profile": "compound_heavy",
                "stochastic_flight_failures_enabled": False,
            }
        )
        env.reset(seed=43001)
        aircraft_specs = [
            spec
            for spec in env.disruption_schedule
            if spec.kind in {"aircraft_failure", "aircraft_hold"}
        ]

        for index, first in enumerate(aircraft_specs):
            for second in aircraft_specs[index + 1 :]:
                if int(first.target) != int(second.target):
                    continue
                overlap = max(
                    first.start_time,
                    second.start_time,
                ) < min(first.end_time, second.end_time)
                self.assertFalse(
                    overlap
                    and "aircraft_failure"
                    in {first.kind, second.kind}
                )


if __name__ == "__main__":
    unittest.main()
