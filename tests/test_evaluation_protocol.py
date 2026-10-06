"""Tests for the canonical publishable evaluation protocol."""

from __future__ import annotations

import math
import unittest

from scripts.evaluation_protocol import (
    build_manifest,
    lower_tail_cvar,
    paired_bootstrap_mean_ci,
    recovery_metrics,
    reject_frozen_final_seeds,
    scenario_id,
    upper_tail_cvar,
    validate_phase_seeds,
    verify_manifest,
)


class EvaluationProtocolTest(unittest.TestCase):
    def test_frozen_final_seeds_are_evaluation_only(self) -> None:
        self.assertEqual(
            validate_phase_seeds(
                "final",
                [70001, 70050],
                "evaluation",
            ),
            [70001, 70050],
        )
        with self.assertRaisesRegex(ValueError, "frozen final"):
            reject_frozen_final_seeds(
                [30001, 70001],
                "training",
            )
        with self.assertRaisesRegex(ValueError, "frozen final"):
            validate_phase_seeds(
                "final",
                [70001],
                "selection",
            )

    def test_manifest_hash_detects_mutation(self) -> None:
        manifest = build_manifest(
            phase="dev",
            seeds=[43001],
            profiles=["none"],
            wave_intervals=[52.5],
            budgets_ms=[50.0],
            waves=2,
            solver_specs=[
                {
                    "label": "heuristic",
                    "solver": "heuristic",
                }
            ],
            base_config={"num_aircraft": 4},
            provenance={"git_commit": "abc"},
        )

        self.assertEqual(
            verify_manifest(manifest),
            manifest["manifest_sha256"],
        )
        manifest["waves"] = 3
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_manifest(manifest)

    def test_scenario_id_binds_stochastic_scenario_not_solver(self) -> None:
        first = scenario_id(
            "compound_light",
            52.5,
            12,
            43001,
            "config",
            "schedule-a",
            1,
        )
        repeated = scenario_id(
            "compound_light",
            52.5,
            12,
            43001,
            "config",
            "schedule-a",
            1,
        )
        changed = scenario_id(
            "compound_light",
            52.5,
            12,
            43001,
            "config",
            "schedule-b",
            1,
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)

    def test_cvar_uses_expected_tail(self) -> None:
        values = [1.0, 2.0, 3.0, 100.0]

        self.assertEqual(lower_tail_cvar(values, 0.25), 1.0)
        self.assertEqual(upper_tail_cvar(values, 0.25), 100.0)

    def test_recovery_uses_paired_baseline_and_censoring(self) -> None:
        event_log = [
            {"time": 5.0, "event_type": "disruption_start"},
            {"time": 15.0, "event_type": "disruption_end"},
        ]
        baseline = [
            {
                "wave_index": index,
                "time": index * 10.0,
                "sorties_completed": value,
            }
            for index, value in enumerate([8, 9, 10, 10, 9])
        ]
        recovered = [
            {
                "wave_index": index,
                "time": index * 10.0,
                "sorties_completed": value,
            }
            for index, value in enumerate([8, 4, 9, 10, 9])
        ]

        metrics = recovery_metrics(
            event_log,
            recovered,
            baseline,
            horizon=50.0,
        )

        self.assertEqual(metrics["recovery_observed"], 1)
        self.assertEqual(metrics["recovery_censored"], 0)
        self.assertEqual(metrics["recovery_time_mean"], 5.0)

        censored = recovery_metrics(
            event_log,
            recovered[:3],
            baseline,
            horizon=30.0,
        )
        self.assertEqual(censored["recovery_censored"], 1)
        self.assertTrue(
            math.isnan(float(censored["recovery_time_mean"]))
        )

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap_mean_ci(
            [1.0, 2.0, 3.0],
            samples=100,
            seed=7,
        )
        second = paired_bootstrap_mean_ci(
            [1.0, 2.0, 3.0],
            samples=100,
            seed=7,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["mean_delta"], 2.0)


if __name__ == "__main__":
    unittest.main()
