"""Tests for rolling CP-SAT repair and deadline-safe fallback behavior."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from solution.cp_sat_repair_model import (
    CPSATRepairModel,
    RepairControl,
)
from solution.rolling_repair_solver import RollingRepairSolver


class RollingRepairSolverTest(unittest.TestCase):
    def test_model_respects_single_fuel_vehicle_capacity(self) -> None:
        env = self._service_env(num_fuel_servers=1)
        for aircraft_id in (0, 1):
            aircraft = env.aircraft[aircraft_id]
            aircraft.fuel_status = 0
            aircraft.fuel_level = 0.2
        env._invalidate_planning_cache()

        plan = CPSATRepairModel(env).solve(
            RepairControl(
                budget_ms=500.0,
                neighborhood_size=4,
            )
        )

        self.assertIn(plan.status, {"optimal", "feasible"})
        first = plan.start_ticks[(1, 0)]
        second = plan.start_ticks[(1, 1)]
        self.assertNotEqual(first, second)
        self.assertEqual(min(first, second), 0)

    def test_solver_returns_legal_repair_action(self) -> None:
        env = self._service_env()
        aircraft = env.aircraft[0]
        aircraft.inspection_status = 0
        aircraft.launch_status = 0
        env._invalidate_planning_cache()
        solver = RollingRepairSolver(
            env,
            budget_ms=500.0,
            neighborhood_size=4,
        )

        action = solver.choose_action()

        self.assertIsNotNone(action)
        parsed = env._parse_action(action)
        self.assertIsNotNone(parsed)
        self.assertTrue(env._is_action_valid(*parsed))
        telemetry = solver.get_telemetry()
        self.assertEqual(telemetry["solve_calls"], 1)
        self.assertEqual(telemetry["fallbacks"], 0)

    def test_zero_budget_uses_incumbent_fallback(self) -> None:
        env = self._service_env()
        env.aircraft[0].inspection_status = 0
        env.aircraft[0].launch_status = 0
        env._invalidate_planning_cache()
        solver = RollingRepairSolver(env, budget_ms=0.0)

        action = solver.choose_action()

        self.assertIsNotNone(action)
        parsed = env._parse_action(action)
        self.assertTrue(env._is_action_valid(*parsed))
        telemetry = solver.get_telemetry()
        self.assertEqual(telemetry["solve_calls"], 0)
        self.assertEqual(telemetry["fallbacks"], 1)

    def test_cached_batch_actions_are_revalidated(self) -> None:
        env = self._service_env(
            num_inspection_vehicles=2,
        )
        for aircraft_id in (0, 1):
            aircraft = env.aircraft[aircraft_id]
            aircraft.inspection_status = 0
            aircraft.launch_status = 0
        env._invalidate_planning_cache()
        solver = RollingRepairSolver(
            env,
            budget_ms=500.0,
            neighborhood_size=4,
        )

        first = solver.choose_action()
        env.step(first)
        second = solver.choose_action()

        self.assertIsNotNone(second)
        self.assertNotEqual(
            first["aircraft_id"],
            second["aircraft_id"],
        )
        parsed = env._parse_action(second)
        self.assertTrue(env._is_action_valid(*parsed))
        self.assertEqual(
            solver.get_telemetry()["cached_actions"],
            1,
        )

    def _service_env(
        self,
        **overrides: object,
    ) -> CarrierAircraftSchedulingEnv:
        config = {
            "num_aircraft": 4,
            "num_total_aircraft": 6,
            "group_size": 2,
            "num_parking_spots": 4,
            "spatial_graph_enabled": False,
            "wave_interval": 20.0,
            "simulation_duration": 40.0,
        }
        config.update(overrides)
        env = CarrierAircraftSchedulingEnv(config)
        env.reset(seed=7)
        for aircraft in env.aircraft:
            aircraft.launch_status = 0
        env._invalidate_planning_cache()
        return env


if __name__ == "__main__":
    unittest.main()
