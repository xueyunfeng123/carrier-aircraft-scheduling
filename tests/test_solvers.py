"""Integration tests for registered non-RL solvers."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from solution import CPSATSolver, EDDSolver, FIFOSolver, SPTSolver


class SolverIntegrationTest(unittest.TestCase):
    def test_priority_and_cp_sat_solvers_only_emit_legal_actions(self) -> None:
        solver_classes = [FIFOSolver, SPTSolver, EDDSolver, CPSATSolver]
        config = {
            "num_aircraft": 4,
            "group_size": 2,
            "num_parking_spots": 4,
            "wave_interval": 20.0,
            "simulation_duration": 40.0,
            "num_fuel_servers": 2,
            "num_arm_vehicles": 2,
            "num_ammo_transport_vehicles": 2,
            "num_lower_weapon_lifts": 2,
            "num_upper_weapon_lifts": 2,
            "num_personnel": 8,
        }

        for solver_class in solver_classes:
            with self.subTest(solver=solver_class.__name__):
                env = CarrierAircraftSchedulingEnv(config)
                env.reset(seed=7)
                solver = solver_class(env)
                steps = 0

                while not env.done and steps < 10_000:
                    action = solver.choose_action()
                    mask = env.get_action_mask()
                    if action is None:
                        self.assertFalse(any(mask["high_level"]))
                    else:
                        high_level = action["high_level"]
                        aircraft_id = action["aircraft_id"]
                        self.assertEqual(mask["high_level"][high_level], 1)
                        self.assertEqual(
                            mask["low_level_by_high"][high_level][aircraft_id],
                            1,
                        )
                    env.step(action)
                    steps += 1

                self.assertTrue(env.done)
                self.assertLess(steps, 10_000)


if __name__ == "__main__":
    unittest.main()
