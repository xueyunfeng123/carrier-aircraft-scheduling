import unittest

from scripts.analyze_paired_benchmark import (
    build_summary_rows,
    exact_two_sided_sign_test,
    validate_paired_scenarios,
)


class PairedBenchmarkAnalysisTest(unittest.TestCase):
    def test_exact_sign_test_ignores_ties(self) -> None:
        stats = exact_two_sided_sign_test(
            [3, 4, 5, 7],
            [2, 4, 4, 6],
        )

        self.assertEqual(stats["wins"], 3)
        self.assertEqual(stats["ties"], 1)
        self.assertEqual(stats["losses"], 0)
        self.assertEqual(stats["p_value"], 0.25)

    def test_summary_preserves_paired_scenarios(self) -> None:
        scores = {
            "heuristic": {
                (1, 45.0): 90,
                (2, 45.0): 92,
                (1, 55.0): 110,
                (2, 55.0): 112,
            },
            "rl": {
                (1, 45.0): 94,
                (2, 45.0): 96,
                (1, 55.0): 111,
                (2, 55.0): 115,
            },
        }

        rows = build_summary_rows(
            scores,
            ["heuristic", "rl"],
            "rl",
        )

        self.assertEqual(rows[-1]["wave_interval"], "overall")
        self.assertEqual(rows[-1]["scenarios"], 4)
        self.assertEqual(rows[-1]["rl_mean"], "104.00")
        self.assertEqual(
            rows[-1]["rl_minus_heuristic"],
            "3.00",
        )

    def test_validation_rejects_unpaired_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "scenario mismatch"):
            validate_paired_scenarios(
                {
                    "heuristic": {(1, 60.0): 118},
                    "rl": {(2, 60.0): 120},
                },
                ["heuristic", "rl"],
            )


if __name__ == "__main__":
    unittest.main()
