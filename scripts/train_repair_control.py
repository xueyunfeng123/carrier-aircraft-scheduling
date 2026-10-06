"""Train the multi-head repair controller from canonical oracle labels."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
from scripts.evaluation_protocol import reject_frozen_final_seeds


CONTROL_DATASET_SCHEMA_VERSION = 1
LABEL_COLUMNS = (
    "trigger_label",
    "scope_label",
    "budget_ms_label",
    "neighborhood_label",
)


@dataclass
class ControlDataset:
    features: List[List[float]]
    trigger_labels: List[int]
    scope_labels: List[int]
    budget_labels: List[int]
    neighborhood_labels: List[int]
    weights: List[float]
    seeds: List[int]
    sample_ids: List[str]

    def __len__(self) -> int:
        return len(self.features)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_control_dataset(path: Path) -> ControlDataset:
    required = {
        *CONTROL_FEATURE_NAMES,
        *LABEL_COLUMNS,
        "schema_version",
        "scenario_seed",
    }
    features: List[List[float]] = []
    trigger_labels: List[int] = []
    scope_labels: List[int] = []
    budget_labels: List[int] = []
    neighborhood_labels: List[int] = []
    weights: List[float] = []
    seeds: List[int] = []
    sample_ids: List[str] = []
    seen_ids = set()
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )
        for row_index, row in enumerate(reader, start=2):
            if (
                int(row["schema_version"])
                != CONTROL_DATASET_SCHEMA_VERSION
            ):
                raise ValueError(
                    "unsupported repair-control dataset schema"
                )
            sample_id = row.get(
                "sample_id",
                f"{path.name}:{row_index}",
            )
            if sample_id in seen_ids:
                raise ValueError(
                    f"duplicate control sample_id: {sample_id}"
                )
            seen_ids.add(sample_id)
            feature_row = [
                _finite_float(
                    row[name],
                    f"{path}:{row_index}:{name}",
                )
                for name in CONTROL_FEATURE_NAMES
            ]
            trigger = int(row["trigger_label"])
            if trigger not in (0, 1):
                raise ValueError(
                    "trigger_label must be either 0 or 1"
                )
            scope = _category_index(
                row["scope_label"],
                CONTROL_SCOPES,
                "scope_label",
            )
            budget = _category_index(
                float(row["budget_ms_label"]),
                CONTROL_BUDGETS_MS,
                "budget_ms_label",
            )
            neighborhood = _category_index(
                int(row["neighborhood_label"]),
                CONTROL_NEIGHBORHOODS,
                "neighborhood_label",
            )
            weight = _finite_float(
                row.get("sample_weight", "1"),
                f"{path}:{row_index}:sample_weight",
            )
            if weight <= 0.0:
                raise ValueError("sample_weight must be positive")
            features.append(feature_row)
            trigger_labels.append(trigger)
            scope_labels.append(scope)
            budget_labels.append(budget)
            neighborhood_labels.append(neighborhood)
            weights.append(weight)
            seeds.append(int(row["scenario_seed"]))
            sample_ids.append(sample_id)
    if not features:
        raise ValueError(f"control dataset is empty: {path}")
    return ControlDataset(
        features=features,
        trigger_labels=trigger_labels,
        scope_labels=scope_labels,
        budget_labels=budget_labels,
        neighborhood_labels=neighborhood_labels,
        weights=weights,
        seeds=seeds,
        sample_ids=sample_ids,
    )


def _finite_float(value: str, context: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite value at {context}")
    return parsed


def _category_index(
    value: Any,
    categories: Sequence[Any],
    label: str,
) -> int:
    try:
        return list(categories).index(value)
    except ValueError as exc:
        raise ValueError(
            f"unknown {label}: {value}; expected {list(categories)}"
        ) from exc


def validate_dataset_partition(
    training: ControlDataset,
    validation: ControlDataset,
) -> None:
    reject_frozen_final_seeds(
        training.seeds,
        "repair-control training",
    )
    reject_frozen_final_seeds(
        validation.seeds,
        "repair-control checkpoint selection",
    )
    overlap = sorted(set(training.seeds) & set(validation.seeds))
    if overlap:
        raise ValueError(
            f"repair-control train/validation seeds overlap: {overlap}"
        )


def train_controller(
    training: ControlDataset,
    validation: ControlDataset,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    device: str,
) -> tuple[Any, Dict[str, float], int]:
    try:
        import torch
        from torch.nn import functional as functional
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for repair-control training"
        ) from exc
    random.seed(seed)
    torch.manual_seed(seed)
    model = RepairControlNet(hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_tensors = _to_tensors(training, torch, device)
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = evaluate_controller(
        model,
        validation,
        device,
    )
    best_score = _selection_score(best_metrics)
    best_epoch = 0
    stale_epochs = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(
            len(training),
            generator=generator,
        )
        for start in range(0, len(training), batch_size):
            indexes = permutation[start : start + batch_size].to(
                device
            )
            batch = tuple(
                tensor.index_select(0, indexes)
                for tensor in train_tensors
            )
            features, trigger, scope, budget, neighborhood, weights = (
                batch
            )
            outputs = model(features)
            trigger_loss = functional.cross_entropy(
                outputs[0],
                trigger,
                reduction="none",
            )
            downstream_loss = (
                functional.cross_entropy(
                    outputs[1],
                    scope,
                    reduction="none",
                )
                + functional.cross_entropy(
                    outputs[2],
                    budget,
                    reduction="none",
                )
                + functional.cross_entropy(
                    outputs[3],
                    neighborhood,
                    reduction="none",
                )
            )
            loss = (
                weights
                * (
                    trigger_loss
                    + trigger.float() * downstream_loss
                )
            ).sum() / weights.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate_controller(model, validation, device)
        score = _selection_score(metrics)
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if patience > 0 and stale_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_metrics, best_epoch


def _to_tensors(dataset: ControlDataset, torch, device: str):
    return (
        torch.tensor(
            dataset.features,
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            dataset.trigger_labels,
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            dataset.scope_labels,
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            dataset.budget_labels,
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            dataset.neighborhood_labels,
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            dataset.weights,
            dtype=torch.float32,
            device=device,
        ),
    )


def evaluate_controller(
    model,
    dataset: ControlDataset,
    device: str,
) -> Dict[str, float]:
    import torch
    from torch.nn import functional as functional

    features, trigger, scope, budget, neighborhood, weights = (
        _to_tensors(dataset, torch, device)
    )
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        trigger_loss = functional.cross_entropy(
            outputs[0],
            trigger,
            reduction="none",
        )
        downstream_loss = (
            functional.cross_entropy(
                outputs[1],
                scope,
                reduction="none",
            )
            + functional.cross_entropy(
                outputs[2],
                budget,
                reduction="none",
            )
            + functional.cross_entropy(
                outputs[3],
                neighborhood,
                reduction="none",
            )
        )
        loss = (
            weights
            * (
                trigger_loss
                + trigger.float() * downstream_loss
            )
        ).sum() / weights.sum()
        predictions = [
            output.argmax(dim=-1) for output in outputs
        ]
    trigger_mask = trigger.bool()
    triggered = int(trigger_mask.sum().item())
    true_positive = int(
        ((predictions[0] == 1) & (trigger == 1)).sum().item()
    )
    false_positive = int(
        ((predictions[0] == 1) & (trigger == 0)).sum().item()
    )
    false_negative = int(
        ((predictions[0] == 0) & (trigger == 1)).sum().item()
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    joint = (
        (predictions[0] == trigger)
        & (
            ~trigger_mask
            | (
                (predictions[1] == scope)
                & (predictions[2] == budget)
                & (predictions[3] == neighborhood)
            )
        )
    )
    return {
        "loss": float(loss.item()),
        "trigger_accuracy": float(
            (predictions[0] == trigger).float().mean().item()
        ),
        "trigger_f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "joint_accuracy": float(
            joint.float().mean().item()
        ),
        "scope_accuracy_triggered": _masked_accuracy(
            predictions[1],
            scope,
            trigger_mask,
            triggered,
        ),
        "budget_accuracy_triggered": _masked_accuracy(
            predictions[2],
            budget,
            trigger_mask,
            triggered,
        ),
        "neighborhood_accuracy_triggered": _masked_accuracy(
            predictions[3],
            neighborhood,
            trigger_mask,
            triggered,
        ),
    }


def _masked_accuracy(
    prediction,
    target,
    mask,
    count: int,
) -> float:
    if count == 0:
        return 0.0
    return float(
        (prediction[mask] == target[mask])
        .float()
        .mean()
        .item()
    )


def _selection_score(
    metrics: Dict[str, float],
) -> tuple[float, float, float]:
    return (
        metrics["joint_accuracy"],
        metrics["trigger_f1"],
        -metrics["loss"],
    )


def verify_saved_checkpoint(
    checkpoint: Path,
    validation: ControlDataset,
    device: str,
) -> Dict[str, float]:
    controller = LearnedRepairController(
        str(checkpoint),
        device=device,
    )
    return evaluate_controller(
        controller.model,
        validation,
        device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/repair_control.pt",
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/repair_control_metrics.json",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    validation_path = Path(args.validation_csv)
    training = load_control_dataset(train_path)
    validation = load_control_dataset(validation_path)
    validate_dataset_partition(training, validation)
    model, metrics, best_epoch = train_controller(
        training,
        validation,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    checkpoint_path = Path(args.checkpoint)
    save_checkpoint(
        str(checkpoint_path),
        model,
        extra={
            "checkpoint_type": REPAIR_CONTROL_CHECKPOINT_TYPE,
            "control_dataset_schema_version": (
                CONTROL_DATASET_SCHEMA_VERSION
            ),
            "feature_names": list(CONTROL_FEATURE_NAMES),
            "scopes": list(CONTROL_SCOPES),
            "budgets_ms": list(CONTROL_BUDGETS_MS),
            "neighborhoods": list(CONTROL_NEIGHBORHOODS),
            "training_seeds": sorted(set(training.seeds)),
            "validation_seeds": sorted(set(validation.seeds)),
            "training_dataset_sha256": file_sha256(train_path),
            "validation_dataset_sha256": file_sha256(
                validation_path
            ),
            "best_epoch": best_epoch,
            "validation_metrics": metrics,
            "selection_objective": (
                "joint accuracy, trigger F1, negative loss"
            ),
        },
    )
    reloaded_metrics = verify_saved_checkpoint(
        checkpoint_path,
        validation,
        args.device,
    )
    if reloaded_metrics != metrics:
        raise RuntimeError(
            "reloaded repair-control checkpoint changed predictions"
        )
    output = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "metrics": reloaded_metrics,
    }
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
