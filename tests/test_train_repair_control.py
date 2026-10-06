"""Tests for repair-control dataset and checkpoint contracts."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.checkpoint import save_checkpoint
from rl.repair_control import (
    CONTROL_BUDGETS_MS,
    CONTROL_FEATURE_NAMES,
    CONTROL_NEIGHBORHOODS,
    CONTROL_SCOPES,
    REPAIR_CONTROL_CHECKPOINT_TYPE,
    LearnedRepairController,
    RepairControlNet,
)
from scripts.train_repair_control import (
    load_control_dataset,
    validate_dataset_partition,
)
from scripts.run_publishable_benchmark import (
    checkpoint_provenance,
)
from scripts.collect_repair_control_oracle import (
    OracleOutcome,
    select_oracle,
)
from solution.cp_sat_repair_model import RepairControl


class RepairControlTrainingContractTest(unittest.TestCase):
    def test_oracle_prioritizes_sorties_before_runtime(self) -> None:
        fast = OracleOutcome(
            control=RepairControl(
                trigger=False,
                budget_ms=10.0,
                neighborhood_size=8,
            ),
            sorties=3,
            missed=1,
            deadline_misses=0,
            runtime_ms=1.0,
        )
        productive = OracleOutcome(
            control=RepairControl(
                trigger=True,
                budget_ms=200.0,
                neighborhood_size=32,
            ),
            sorties=4,
            missed=0,
            deadline_misses=1,
            runtime_ms=100.0,
        )

        best, margin, ties = select_oracle(
            [fast, productive]
        )

        self.assertEqual(best, productive)
        self.assertEqual(margin, 1.0)
        self.assertEqual(ties, 1)

    def test_dataset_loader_and_seed_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.csv"
            validation_path = Path(directory) / "validation.csv"
            self._write_dataset(train_path, 30001)
            self._write_dataset(validation_path, 41001)

            training = load_control_dataset(train_path)
            validation = load_control_dataset(validation_path)
            validate_dataset_partition(training, validation)

            self.assertEqual(len(training), 2)
            self.assertEqual(training.seeds, [30001, 30001])

    def test_frozen_seed_is_rejected_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.csv"
            validation_path = Path(directory) / "validation.csv"
            self._write_dataset(train_path, 70001)
            self._write_dataset(validation_path, 41001)

            with self.assertRaisesRegex(ValueError, "frozen final"):
                validate_dataset_partition(
                    load_control_dataset(train_path),
                    load_control_dataset(validation_path),
                )

    def test_public_controller_reloads_schema_checked_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "control.pt"
            model = RepairControlNet(hidden_dim=8)
            save_checkpoint(
                str(checkpoint),
                model,
                extra={
                    "checkpoint_type": (
                        REPAIR_CONTROL_CHECKPOINT_TYPE
                    ),
                    "control_dataset_schema_version": 1,
                    "feature_names": list(CONTROL_FEATURE_NAMES),
                    "scopes": list(CONTROL_SCOPES),
                    "budgets_ms": list(CONTROL_BUDGETS_MS),
                    "neighborhoods": list(
                        CONTROL_NEIGHBORHOODS
                    ),
                    "training_seeds": [30001],
                    "validation_seeds": [41001],
                },
            )
            controller = LearnedRepairController(
                str(checkpoint),
                max_budget_ms=10.0,
                max_neighborhood_size=8,
            )
            provenance = checkpoint_provenance(
                str(checkpoint),
                require_lineage=True,
            )
            env = CarrierAircraftSchedulingEnv(
                {
                    "num_aircraft": 4,
                    "num_total_aircraft": 6,
                    "group_size": 2,
                    "num_parking_spots": 4,
                    "spatial_graph_enabled": False,
                    "simulation_duration": 20.0,
                    "wave_interval": 10.0,
                }
            )
            env.reset(seed=7)

            control = controller(env)

            self.assertLessEqual(control.budget_ms, 10.0)
            self.assertLessEqual(control.neighborhood_size, 8)
            self.assertEqual(
                provenance["training_seeds"],
                [30001],
            )

    def _write_dataset(self, path: Path, seed: int) -> None:
        fieldnames = [
            "schema_version",
            "sample_id",
            "scenario_seed",
            *CONTROL_FEATURE_NAMES,
            "trigger_label",
            "scope_label",
            "budget_ms_label",
            "neighborhood_label",
            "sample_weight",
        ]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for index, trigger in enumerate((0, 1)):
                writer.writerow(
                    {
                        "schema_version": 1,
                        "sample_id": f"{seed}:{index}",
                        "scenario_seed": seed,
                        **{
                            name: float(index)
                            for name in CONTROL_FEATURE_NAMES
                        },
                        "trigger_label": trigger,
                        "scope_label": "two_waves",
                        "budget_ms_label": 50.0,
                        "neighborhood_label": 16,
                        "sample_weight": 1.0,
                    }
                )


if __name__ == "__main__":
    unittest.main()
