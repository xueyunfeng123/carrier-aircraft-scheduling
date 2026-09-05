"""Regression tests for dynamic wave assignment from a shared deck fleet."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv


class DynamicSharedFleetTest(unittest.TestCase):
    def test_default_fleet_contains_five_ready_reserve_aircraft(self) -> None:
        env = CarrierAircraftSchedulingEnv()
        env.reset(seed=7)

        self.assertEqual(env.num_aircraft, 45)
        self.assertEqual(env.group_size, 20)
        self.assertEqual(env.num_reserve_aircraft, 5)
        self.assertEqual(
            sum(aircraft.initial_role == "reserve" for aircraft in env.aircraft),
            5,
        )
        self.assertEqual(len(env._candidate_sets()["L"]), 45)

    def test_any_ready_aircraft_can_fill_current_wave(self) -> None:
        env = self._small_env()

        self.assertIn(4, env._candidate_sets()["L"])
        self._launch_and_complete(env, 4)
        self._launch_and_complete(env, 2)

        self.assertEqual(env.wave_records[0]["started_aircraft_ids"], [4, 2])
        self.assertEqual(env.wave_records[0]["launched_aircraft_ids"], [4, 2])
        self.assertEqual(env._candidate_sets()["L"], [])

    def test_next_wave_recovers_actual_previous_wave_aircraft(self) -> None:
        env = self._small_env()
        self._launch_and_complete(env, 4)
        self._launch_and_complete(env, 2)

        while env.time < 10.0:
            env.step(None)

        self.assertEqual(env.current_wave_index, 1)
        self.assertEqual(env.wave_records[1]["recovery_aircraft_ids"], [2, 4])
        self.assertEqual(
            {index for index, item in enumerate(env.aircraft) if item.pending_recovery},
            {2, 4},
        )
        self.assertEqual(env._candidate_sets()["L"], [0, 1, 3])

    def test_unfilled_wave_records_anonymous_missed_slots(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 4,
                "group_size": 2,
                "num_parking_spots": 4,
                "launch_time": 2.0,
                "wave_interval": 1.0,
                "simulation_duration": 2.0,
            }
        )
        env.reset(seed=7)

        while env.time < 1.0:
            env.step(None)

        self.assertEqual(env.wave_records[0]["missed_sorties"], 2)
        self.assertEqual(len(env.get_missed_sortie_records()), 2)
        self.assertTrue(
            all(
                record["aircraft_id"] is None
                for record in env.get_missed_sortie_records()
            )
        )

    def _small_env(self) -> CarrierAircraftSchedulingEnv:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 5,
                "group_size": 2,
                "num_parking_spots": 5,
                "launch_time": 1.0,
                "wave_interval": 10.0,
                "simulation_duration": 30.0,
            }
        )
        env.reset(seed=7)
        return env

    def _launch_and_complete(
        self,
        env: CarrierAircraftSchedulingEnv,
        aircraft_id: int,
    ) -> None:
        _, _, _, info = env.step(
            {"high_level": 3, "aircraft_id": aircraft_id}
        )
        self.assertFalse(info["invalid_action"])
        env.step(None)


if __name__ == "__main__":
    unittest.main()
