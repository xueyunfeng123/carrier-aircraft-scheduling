"""Tests for publishable benchmark aggregation and pairing."""

from __future__ import annotations

import unittest

from scripts.analyze_publishable_benchmark import (
    build_comparisons,
    build_summary,
)


class PublishableAnalysisTest(unittest.TestCase):
    def test_summary_reports_tail_loss_resilience_and_latency(self) -> None:
        rows = [
            self._row("a", "candidate", 10, 0.0, 1.0),
            self._row("b", "candidate", 8, 2.0, 0.8),
        ]

        summary = build_summary(rows, alpha=0.5)
        overall = next(
            row for row in summary if row["scope"] == "overall"
        )

        self.assertEqual(overall["mean_completed"], 9.0)
        self.assertEqual(overall["cvar_0.5_completed"], 8.0)
        self.assertEqual(
            overall["cvar_0.5_sortie_loss"],
            2.0,
        )
        self.assertEqual(overall["mean_resilience_ratio"], 0.9)
        self.assertEqual(overall["mean_latency_p95_ms"], 5.0)

    def test_comparison_requires_exact_scenario_pairing(self) -> None:
        rows = [
            self._row("a", "candidate", 11, 0.0, 1.0),
            self._row("b", "candidate", 12, 0.0, 1.0),
            self._row("a", "baseline", 10, 0.0, 1.0),
            self._row("b", "baseline", 10, 0.0, 1.0),
        ]

        comparisons = build_comparisons(
            rows,
            "candidate",
            ["baseline"],
            bootstrap_samples=100,
        )
        overall = next(
            row
            for row in comparisons
            if row["profile"] == "all"
        )

        self.assertEqual(overall["mean_delta"], 1.5)
        self.assertEqual(overall["pairs"], 1)
        self.assertEqual(overall["scenario_cells"], 2)
        self.assertEqual(overall["wins"], 1)
        self.assertIn("holm_adjusted_p", overall)

        with self.assertRaisesRegex(ValueError, "unpaired"):
            build_comparisons(
                rows[:-1],
                "candidate",
                ["baseline"],
                bootstrap_samples=10,
            )

    def _row(
        self,
        scenario_id: str,
        solver: str,
        completed: int,
        loss: float,
        resilience: float,
    ) -> dict:
        return {
            "scenario_id": scenario_id,
            "solver": solver,
            "profile": "compound_light",
            "seed": 43001,
            "wave_interval": 52.5,
            "total_sorties_completed": completed,
            "total_missed_sorties": 20 - completed,
            "worst_wave_sorties": completed,
            "runtime_seconds": 1.0,
            "decisions": 1,
            "solve_calls": 1,
            "deadline_misses": 0,
            "fallbacks": 0,
            "latency_p95_ms": 5.0,
            "latency_p99_ms": 5.0,
            "latency_max_ms": 5.0,
            "recovery_time_mean": 1.0,
            "recovery_time_max": 1.0,
            "recovery_observed": 1,
            "recovery_censored": 0,
            "sortie_loss_area": loss,
            "resilience_index": resilience,
        }


if __name__ == "__main__":
    unittest.main()
