"""Tests for failure, hangar repair, and replacement inventory flow."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv


class AircraftFailureLifecycleTest(unittest.TestCase):
    def test_ground_failure_is_replaced_and_repaired(self) -> None:
        env = self._env(num_total_aircraft=6)

        self._advance_to(env, 1.0)
        self.assertEqual(
            env.aircraft[0].lifecycle_status,
            "to_hangar",
        )
        self.assertIn(0, env.failure_blocked_slots)

        self._advance_to(env, 2.0)
        self.assertEqual(
            env.aircraft[0].physical_aircraft_id,
            4,
        )
        self.assertEqual(
            env.aircraft[0].lifecycle_status,
            "deck_serviceable",
        )
        self.assertNotIn(0, env.failure_blocked_slots)
        self.assertEqual(env.under_repair_aircraft_ids, {0})
        self.assertEqual(env.hangar_ready_aircraft_ids, {5})

        self._advance_to(env, 5.0)
        self.assertEqual(env.under_repair_aircraft_ids, set())
        self.assertEqual(env.hangar_ready_aircraft_ids, {0, 5})
        self.assertEqual(
            env.failure_metrics,
            {
                "detected": 1,
                "replacements": 1,
                "repairs_completed": 1,
            },
        )
        event_types = [
            item["event_type"]
            for item in env.get_event_log()
        ]
        for event_type in (
            "aircraft_failure_detected",
            "hangar_transfer_start",
            "hangar_transfer_done",
            "replacement_transfer_start",
            "replacement_transfer_done",
            "repair_start",
            "repair_done",
        ):
            self.assertIn(event_type, event_types)
        env._validate_aircraft_inventory()

    def test_airborne_failure_still_allows_safe_recovery(self) -> None:
        env = self._env(num_total_aircraft=6)
        aircraft = env.aircraft[0]
        env.parking_occupancy[aircraft.spot_id] = None
        aircraft.spot_id = -1
        aircraft.parking_status = 0
        aircraft.is_airborne = True
        aircraft.pending_recovery = True
        aircraft.recovery_status = 0
        aircraft.launch_status = 3
        env._invalidate_planning_cache()

        self._advance_to(env, 1.0)
        self.assertEqual(
            aircraft.lifecycle_status,
            "failure_pending_recovery",
        )
        self.assertIn(0, env._candidate_sets()["R"])
        for action_name in ("F", "M", "L", "I"):
            self.assertNotIn(0, env._candidate_sets()[action_name])

        action = env.complete_action(
            {"high_level": 0, "aircraft_id": 0}
        )
        _, _, _, info = env.step(action)
        self.assertFalse(info["invalid_action"])
        self._advance_to(env, 2.0)
        env._advance_time_to_next_event()
        self.assertEqual(aircraft.lifecycle_status, "to_hangar")

    def test_no_spare_waits_for_repaired_aircraft(self) -> None:
        env = self._env(num_total_aircraft=4)

        self._advance_to(env, 2.0)
        self.assertEqual(env.hangar_ready_aircraft_ids, set())
        self.assertIn(0, env.failure_blocked_slots)
        self.assertEqual(
            env.aircraft[0].lifecycle_status,
            "under_repair",
        )

        self._advance_to(env, 5.0)
        self.assertIn(0, env.replacement_in_transit)
        self.assertIn(0, env.failure_blocked_slots)

        self._advance_to(env, 6.0)
        self.assertEqual(
            env.aircraft[0].physical_aircraft_id,
            0,
        )
        self.assertNotIn(0, env.failure_blocked_slots)
        env._validate_aircraft_inventory()

    def test_total_inventory_cannot_be_smaller_than_deck_fleet(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "num_total_aircraft",
        ):
            CarrierAircraftSchedulingEnv(
                {
                    "num_aircraft": 4,
                    "num_total_aircraft": 3,
                    "group_size": 2,
                    "num_parking_spots": 4,
                }
            )

    def _env(
        self,
        num_total_aircraft: int,
    ) -> CarrierAircraftSchedulingEnv:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 4,
                "num_total_aircraft": num_total_aircraft,
                "group_size": 2,
                "num_parking_spots": 4,
                "spatial_graph_enabled": False,
                "wave_interval": 10.0,
                "simulation_duration": 10.0,
                "hangar_transfer_time": 1.0,
                "disruptions": [
                    {
                        "kind": "aircraft_failure",
                        "target": 0,
                        "start_time": 1.0,
                        "end_time": 5.0,
                    }
                ],
            }
        )
        env.reset(seed=7)
        return env

    def _advance_to(
        self,
        env: CarrierAircraftSchedulingEnv,
        target_time: float,
    ) -> None:
        while env.time < target_time:
            env._advance_time_to_next_event()
        self.assertEqual(env.time, target_time)


if __name__ == "__main__":
    unittest.main()
