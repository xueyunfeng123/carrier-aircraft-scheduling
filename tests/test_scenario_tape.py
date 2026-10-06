"""Tests for order-invariant stochastic scenario inputs."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.scenario_tape import ScenarioTape


class ScenarioTapeTest(unittest.TestCase):
    def test_semantic_draws_are_order_invariant(self) -> None:
        tape = ScenarioTape(seed=70001)
        keys = [
            ("inspection", 0, 1),
            ("inspection", 1, 1),
            ("arming", 0, 1, 0),
        ]

        forward = {
            key: tape.normal(key, 8.0, 1.0)
            for key in keys
        }
        reverse = {
            key: tape.normal(key, 8.0, 1.0)
            for key in reversed(keys)
        }

        self.assertEqual(forward, reverse)

    def test_seed_and_key_change_draws(self) -> None:
        first = ScenarioTape(seed=7)
        second = ScenarioTape(seed=8)

        self.assertNotEqual(
            first.normal(("inspection", 0, 1), 8.0, 1.0),
            second.normal(("inspection", 0, 1), 8.0, 1.0),
        )
        self.assertNotEqual(
            first.normal(("inspection", 0, 1), 8.0, 1.0),
            first.normal(("inspection", 1, 1), 8.0, 1.0),
        )

    def test_environment_operation_draws_do_not_depend_on_call_order(
        self,
    ) -> None:
        config = {
            "num_aircraft": 4,
            "group_size": 2,
            "num_parking_spots": 4,
            "spatial_graph_enabled": False,
            "scenario_tape_enabled": True,
        }
        first = CarrierAircraftSchedulingEnv(config)
        second = CarrierAircraftSchedulingEnv(config)
        first.reset(seed=19)
        second.reset(seed=19)
        first.aircraft[0].sorties_completed = 1
        first.aircraft[1].sorties_completed = 2
        second.aircraft[0].sorties_completed = 1
        second.aircraft[1].sorties_completed = 2

        first_zero = first._sample_duration(
            "inspection_time_mean",
            "inspection_time_std",
            first._scenario_key(0, "inspection"),
        )
        first_one = first._sample_duration(
            "inspection_time_mean",
            "inspection_time_std",
            first._scenario_key(1, "inspection"),
        )
        second_one = second._sample_duration(
            "inspection_time_mean",
            "inspection_time_std",
            second._scenario_key(1, "inspection"),
        )
        second_zero = second._sample_duration(
            "inspection_time_mean",
            "inspection_time_std",
            second._scenario_key(0, "inspection"),
        )

        self.assertEqual(first_zero, second_zero)
        self.assertEqual(first_one, second_one)
        self.assertEqual(
            first.get_evaluation_metrics()["scenario_tape"],
            second.get_evaluation_metrics()["scenario_tape"],
        )

    def test_tape_is_opt_in_for_historical_compatibility(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 4,
                "group_size": 2,
                "num_parking_spots": 4,
                "spatial_graph_enabled": False,
            }
        )
        env.reset(seed=7)

        self.assertIsNone(env.scenario_tape)
        self.assertEqual(
            env.get_evaluation_metrics()["scenario_tape"],
            {"enabled": False},
        )


if __name__ == "__main__":
    unittest.main()
