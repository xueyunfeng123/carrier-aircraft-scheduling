"""Tests for event-level batch dispatch semantics."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from scripts.diagnose_batch_headroom import validate_development_seeds


class BatchDispatchTest(unittest.TestCase):
    def _env(self, **overrides) -> CarrierAircraftSchedulingEnv:
        config = {
            "num_aircraft": 4,
            "group_size": 2,
            "num_parking_spots": 4,
            "num_launch_positions": 2,
            "num_launch_channels": 2,
            "num_tow_vehicles": 2,
            "wave_interval": 20.0,
            "simulation_duration": 40.0,
            "spatial_graph_enabled": False,
        }
        config.update(overrides)
        env = CarrierAircraftSchedulingEnv(config)
        env.reset(seed=7)
        return env

    def test_batch_starts_multiple_actions_before_advancing(self) -> None:
        env = self._env()

        _, reward, done, info = env.step_batch(
            [
                {"high_level": 3, "aircraft_id": 0},
                {"high_level": 3, "aircraft_id": 1},
            ]
        )

        self.assertFalse(done)
        self.assertEqual(env.time, 1.0)
        self.assertEqual(env.aircraft[0].launch_start, 0.0)
        self.assertEqual(env.aircraft[1].launch_start, 0.0)
        self.assertEqual(info["batch_started_actions"], ["L", "L"])
        self.assertEqual(info["batch_size"], 2)
        self.assertTrue(info["batch_advanced"])
        self.assertEqual(reward, -1.0 + 2.0)

    def test_empty_batch_is_explicit_idle_until_next_event(self) -> None:
        env = self._env()

        _, reward, _, info = env.step_batch([])

        self.assertEqual(env.time, 20.0)
        self.assertEqual(reward, -20.0)
        self.assertFalse(info["invalid_action"])
        self.assertEqual(info["batch_size"], 0)
        self.assertTrue(info["batch_advanced"])

    def test_batch_revalidates_capacity_after_each_action(self) -> None:
        env = self._env(num_launch_channels=1)

        _, reward, _, info = env.step_batch(
            [
                {"high_level": 3, "aircraft_id": 0},
                {"high_level": 3, "aircraft_id": 1},
            ]
        )

        self.assertEqual(env.time, 0.0)
        self.assertEqual(reward, -5.0)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["batch_invalid_index"], 1)
        self.assertEqual(info["batch_started_actions"], ["L"])
        self.assertFalse(info["batch_advanced"])

    def test_headroom_diagnostic_rejects_frozen_seeds(self) -> None:
        validate_development_seeds([66001, 66002])
        with self.assertRaisesRegex(ValueError, "70001"):
            validate_development_seeds([66001, 70001])


if __name__ == "__main__":
    unittest.main()
