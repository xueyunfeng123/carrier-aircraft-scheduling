"""Tests for per-wave launch accounting."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv


class WaveRecordsTest(unittest.TestCase):
    def test_launch_completion_increments_active_wave(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 2,
                "group_size": 1,
                "num_parking_spots": 2,
                "wave_interval": 10.0,
                "simulation_duration": 20.0,
            }
        )
        env.reset(seed=7)

        env.step({"high_level": 3, "aircraft_id": 0})
        env.step(None)

        self.assertEqual(env.get_wave_records()[0]["sorties_completed"], 1)

    def test_launch_at_wave_boundary_counts_toward_previous_wave(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 2,
                "group_size": 1,
                "num_parking_spots": 2,
                "launch_time": 1.0,
                "wave_interval": 1.0,
                "simulation_duration": 2.0,
            }
        )
        env.reset(seed=7)

        env.step({"high_level": 3, "aircraft_id": 0})
        env.step(None)

        records = env.get_wave_records()
        self.assertEqual(records[0]["sorties_completed"], 1)
        self.assertEqual(records[1]["sorties_completed"], 0)

    def test_simulation_end_does_not_create_empty_wave(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 2,
                "group_size": 1,
                "num_parking_spots": 2,
                "num_recovery_channels": 0,
                "launch_time": 1.0,
                "wave_interval": 1.0,
                "simulation_duration": 2.0,
            }
        )
        env.reset(seed=7)

        env.step({"high_level": 3, "aircraft_id": 0})
        env.step(None)
        env.step({"high_level": 3, "aircraft_id": 1})
        env.step(None)

        records = env.get_wave_records()
        self.assertTrue(env.done)
        self.assertEqual([record["wave_index"] for record in records], [0, 1])
        self.assertEqual([record["sorties_completed"] for record in records], [1, 1])
        self.assertEqual(
            sum(record["sorties_completed"] for record in records),
            env.get_evaluation_metrics()["total_sorties_completed"],
        )


if __name__ == "__main__":
    unittest.main()
