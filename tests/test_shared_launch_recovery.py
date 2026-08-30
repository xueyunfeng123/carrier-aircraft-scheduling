"""Tests for launch positions shared with the recovery area."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv


class SharedLaunchRecoveryTest(unittest.TestCase):
    def _make_conflict_env(self) -> CarrierAircraftSchedulingEnv:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 8,
                "group_size": 4,
                "num_parking_spots": 9,
                "num_launch_channels": 4,
                "num_shared_launch_channels": 2,
                "wave_interval": 20.0,
                "simulation_duration": 40.0,
            }
        )
        env.reset(seed=7)
        env.active_launch_group = "B"
        env.active_recovery_group = "A"

        aircraft = env.aircraft[0]
        env._release_parking_spot(0)
        aircraft.spot_id = -1
        aircraft.parking_status = 0
        aircraft.is_airborne = True
        aircraft.pending_recovery = True
        aircraft.recovery_status = 0
        return env

    def test_active_recovery_limits_launches_to_dedicated_positions(self) -> None:
        env = self._make_conflict_env()
        env.step({"high_level": 0, "aircraft_id": 0})
        env.step({"high_level": 3, "aircraft_id": 4})
        env.step({"high_level": 3, "aircraft_id": 5})

        mask = env.get_high_level_action_mask()
        self.assertEqual(env.free_dedicated_launch_channels, 0)
        self.assertEqual(env.free_shared_launch_channels, 2)
        self.assertEqual(mask[3], 0)

    def test_shared_launch_blocks_recovery(self) -> None:
        env = self._make_conflict_env()
        env.step({"high_level": 3, "aircraft_id": 4})
        env.step({"high_level": 3, "aircraft_id": 5})
        env.step({"high_level": 3, "aircraft_id": 6})

        mask = env.get_high_level_action_mask()
        self.assertEqual(env.free_shared_launch_channels, 1)
        self.assertEqual(mask[0], 0)

    def test_invalid_shared_capacity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CarrierAircraftSchedulingEnv(
                {
                    "num_launch_channels": 2,
                    "num_shared_launch_channels": 3,
                }
            )


if __name__ == "__main__":
    unittest.main()
