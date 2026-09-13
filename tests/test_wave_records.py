"""Tests for per-wave launch accounting."""

from __future__ import annotations

import unittest
from unittest.mock import patch

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
                "spatial_graph_enabled": False,
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
                "spatial_graph_enabled": False,
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
                "spatial_graph_enabled": False,
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

    def test_launch_targets_respect_current_wave_deadline(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "wave_interval": 5.0,
                "simulation_duration": 10.0,
            }
        )
        env.reset(seed=7)
        durations = {0: 2.0, 1: 5.0, 2: 6.0}

        with (
            patch.object(
                env,
                "_launch_runway_available",
                return_value=True,
            ),
            patch.object(
                env,
                "_find_launch_route",
                side_effect=lambda aircraft_id, runway_id: (
                    f"runway_{runway_id}",
                    ("source", f"runway_{runway_id}"),
                    durations[runway_id],
                ),
            ),
        ):
            candidates = env.get_target_candidates(3, 0)

        self.assertEqual(candidates, [0])

    def test_closed_wave_accounting_matches_demand(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 2,
                "group_size": 1,
                "num_parking_spots": 2,
                "wave_interval": 1.0,
                "simulation_duration": 2.0,
                "spatial_graph_enabled": False,
            }
        )
        env.reset(seed=7)

        while not env.done:
            mask = env.get_action_mask()
            action = None
            for high_level, high_allowed in enumerate(
                mask["high_level"]
            ):
                if not high_allowed:
                    continue
                aircraft_id = next(
                    index
                    for index, allowed in enumerate(
                        mask["low_level_by_high"][high_level]
                    )
                    if allowed
                )
                action = {
                    "high_level": high_level,
                    "aircraft_id": aircraft_id,
                }
                break
            env.step(action)

        self.assertTrue(
            all(
                record["sorties_completed"] + record["missed_sorties"]
                == env.group_size
                for record in env.get_wave_records()
            )
        )


if __name__ == "__main__":
    unittest.main()
