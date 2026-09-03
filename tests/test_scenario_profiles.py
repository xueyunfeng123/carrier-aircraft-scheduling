"""Tests for paper-traceable scenario definitions and monitoring."""

from __future__ import annotations

import math
import random
import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG
from env.scenario import (
    DeckGraph,
    DurationDistribution,
    LocationEdge,
    LocationNode,
    YOON_2023_CASES,
    build_project_core_profile,
    build_yoon_2023_profile,
)


class DurationDistributionTest(unittest.TestCase):
    def test_expected_values_and_samples(self) -> None:
        rng = random.Random(7)
        deterministic = DurationDistribution("deterministic", (3.0,))
        uniform = DurationDistribution("uniform", (1.0, 2.0))
        triangular = DurationDistribution("triangular", (30.0, 40.0, 50.0))

        self.assertEqual(deterministic.expected_value, 3.0)
        self.assertEqual(deterministic.sample(rng), 3.0)
        self.assertEqual(uniform.expected_value, 1.5)
        self.assertGreaterEqual(uniform.sample(rng), 1.0)
        self.assertLessEqual(uniform.sample(rng), 2.0)
        self.assertEqual(triangular.expected_value, 40.0)

    def test_invalid_distribution_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DurationDistribution("uniform", (2.0, 1.0))
        with self.assertRaises(ValueError):
            DurationDistribution("normal", (1.0, -1.0))


class DeckGraphTest(unittest.TestCase):
    def test_shortest_distance_uses_weighted_edges(self) -> None:
        graph = DeckGraph(
            nodes=(
                LocationNode("a", "parking"),
                LocationNode("b", "pathway"),
                LocationNode("c", "runway"),
            ),
            edges=(
                LocationEdge("a", "c", 10.0),
                LocationEdge("a", "b", 2.0),
                LocationEdge("b", "c", 3.0),
            ),
        )

        self.assertEqual(graph.shortest_distance("a", "c"), 5.0)
        self.assertEqual(graph.shortest_distance("c", "a"), 5.0)
        self.assertTrue(math.isfinite(graph.shortest_distance("a", "b")))


class ScenarioProfileTest(unittest.TestCase):
    def test_yoon_cases_match_published_experiment_matrix(self) -> None:
        self.assertEqual(len(YOON_2023_CASES), 8)
        self.assertEqual(YOON_2023_CASES["case_1_1"], (7, 5, 16))
        self.assertEqual(YOON_2023_CASES["case_4_2"], (10, 8, 20))

    def test_yoon_profile_records_published_parameters_and_gaps(self) -> None:
        profile = build_yoon_2023_profile("case_4_1")
        process_by_name = {process.name: process for process in profile.processes}

        self.assertEqual(profile.num_fighters, 16)
        self.assertEqual(dict(profile.parking_capacities), {"parking_bow": 10, "parking_stern": 8})
        self.assertEqual(profile.mission_plan.formation_size, 4)
        self.assertEqual(profile.mission_plan.mission_interval_minutes, 100.0)
        self.assertEqual(profile.mission_plan.mission_duration_minutes, 60.0)
        self.assertEqual(profile.mission_plan.cancellation_threshold_minutes, 20.0)
        self.assertEqual(process_by_name["refueling"].duration.expected_value, 40.0)
        self.assertEqual(process_by_name["maintenance"].duration.expected_value, 120.0)
        self.assertIsNone(profile.deck_graph)
        self.assertTrue(
            any("adjacency" in item for item in profile.unresolved_parameters)
        )

    def test_project_core_graph_preserves_legacy_transfer_times(self) -> None:
        config = dict(DEFAULT_CONFIG)
        config.update({"num_aircraft": 4, "group_size": 2, "num_parking_spots": 5})
        profile = build_project_core_profile(config)

        distances = [
            profile.deck_graph.shortest_distance(f"parking_{spot_id}", "arm_service")
            for spot_id in range(5)
        ]
        self.assertEqual(distances, [2.0, 3.0, 3.0, 4.0, 4.0])


class EnvironmentMonitoringTest(unittest.TestCase):
    def test_environment_exposes_profile_sgr_and_event_log(self) -> None:
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

        metrics = env.get_evaluation_metrics()
        event_types = [item["event_type"] for item in env.get_event_log()]

        self.assertEqual(env.get_state()["scenario"]["name"], "project_core")
        self.assertEqual(metrics["total_sorties_completed"], 1)
        self.assertEqual(metrics["sortie_generation_rate_per_hour"], 3.0)
        self.assertEqual(metrics["sortie_completion_rate"], 1.0)
        self.assertIn("wave_start", event_types)
        self.assertIn("launch_start", event_types)
        self.assertIn("launch_done", event_types)

    def test_non_executable_profile_is_rejected_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "currently executes only"):
            CarrierAircraftSchedulingEnv({"scenario_profile": "paper_yoon_2023"})


if __name__ == "__main__":
    unittest.main()
