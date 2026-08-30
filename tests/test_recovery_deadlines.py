"""Tests for heterogeneous recovery deadlines."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from solution import EDDSolver


class RecoveryDeadlineTest(unittest.TestCase):
    def test_missed_deadline_requeues_aircraft_after_delay(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 2,
                "group_size": 1,
                "num_parking_spots": 2,
                "wave_interval": 10.0,
                "simulation_duration": 20.0,
                "recovery_deadline_min": 1.0,
                "recovery_deadline_max": 1.0,
                "recovery_retry_delay": 2.0,
            }
        )
        env.reset(seed=7)
        aircraft = env.aircraft[0]
        aircraft.is_airborne = True
        aircraft.pending_recovery = True
        aircraft.recovery_status = 0
        env._schedule_recovery_deadline(0)

        env._advance_time_to_next_event()
        self.assertEqual(env.time, 1.0)
        self.assertFalse(aircraft.pending_recovery)
        self.assertEqual(aircraft.recovery_deadline_misses, 1)

        env._advance_time_to_next_event()
        self.assertEqual(env.time, 3.0)
        self.assertTrue(aircraft.pending_recovery)
        self.assertEqual(aircraft.recovery_deadline, 4.0)

    def test_edd_selects_earliest_recovery_deadline(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 4,
                "group_size": 2,
                "num_parking_spots": 4,
            }
        )
        env.reset(seed=7)
        env.active_launch_group = "B"
        env.free_launch_channels = 0

        for aircraft_id, deadline in ((0, 5.0), (1, 3.0)):
            aircraft = env.aircraft[aircraft_id]
            aircraft.is_airborne = True
            aircraft.pending_recovery = True
            aircraft.recovery_status = 0
            aircraft.recovery_deadline = deadline

        action = EDDSolver(env).choose_action()
        self.assertEqual(action, {"high_level": 0, "aircraft_id": 1})


if __name__ == "__main__":
    unittest.main()
