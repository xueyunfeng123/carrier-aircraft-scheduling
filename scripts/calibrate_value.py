"""Calibrate an existing policy checkpoint's critic on complete trajectories."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG
from rl.checkpoint import load_checkpoint, save_checkpoint
from rl.train_config import PPOConfig
from rl.value_calibration import (
    calibrate_value_head,
    calibration_metadata,
    calibration_metric_rows,
    collect_value_samples,
    predict_values,
)
from solution.rl_solver import RLSolver


DEFAULT_CALIBRATION_SEEDS = tuple(range(42001, 42011))
DEFAULT_SELECTION_SEEDS = tuple(range(41001, 41006))
DEFAULT_INTERVALS = (47.5, 52.5, 57.5, 62.5, 67.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/rl_multiload_bc.pt",
    )
    parser.add_argument(
        "--output-checkpoint",
        default="checkpoints/rl_multiload_value_calibrated.pt",
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/value_calibration_seed41001_5.csv",
    )
    parser.add_argument(
        "--calibration-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_CALIBRATION_SEEDS),
    )
    parser.add_argument(
        "--selection-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SELECTION_SEEDS),
    )
    parser.add_argument(
        "--wave-intervals",
        type=float,
        nargs="+",
        default=list(DEFAULT_INTERVALS),
    )
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument(
        "--target-method",
        choices=("mc", "gae"),
        default="mc",
    )
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--loss",
        choices=("huber", "mse"),
        default="huber",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=100_000)
    args = parser.parse_args()

    validate_seed_protocol(
        args.calibration_seeds,
        args.selection_seeds,
    )
    configs = build_calibration_configs(
        args.wave_intervals,
        args.waves,
    )
    payload = load_checkpoint(args.checkpoint, device=args.device)
    solver = RLSolver(
        CarrierAircraftSchedulingEnv(configs[0]),
        checkpoint=args.checkpoint,
        device=args.device,
    )
    model = solver.model
    reward_config = ppo_config_from_checkpoint(payload)
    if args.gamma is not None:
        reward_config.gamma = float(args.gamma)

    calibration_samples = collect_value_samples(
        model,
        configs,
        args.calibration_seeds,
        args.device,
        reward_config,
        target_method=args.target_method,
        gae_lambda=args.gae_lambda,
        max_steps=args.max_steps,
    )
    selection_samples = collect_value_samples(
        model,
        configs,
        args.selection_seeds,
        args.device,
        reward_config,
        target_method=args.target_method,
        gae_lambda=args.gae_lambda,
        max_steps=args.max_steps,
    )
    before_predictions = predict_values(
        model,
        selection_samples,
        args.device,
        args.minibatch_size,
    )
    fit_stats = calibrate_value_head(
        model,
        calibration_samples,
        args.device,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        loss_name=args.loss,
        seed=args.calibration_seeds[0],
    )
    after_predictions = predict_values(
        model,
        selection_samples,
        args.device,
        args.minibatch_size,
    )
    rows = [
        {"stage": stage, **row}
        for stage, predictions in (
            ("before", before_predictions),
            ("after", after_predictions),
        )
        for row in calibration_metric_rows(
            selection_samples,
            predictions,
        )
    ]
    write_rows(args.metrics_output, rows)

    extra = dict(payload.get("extra", {}))
    extra["ppo_config"] = asdict(reward_config)
    metadata = calibration_metadata(
        calibration_samples,
        args.target_method,
        reward_config,
        args.gae_lambda,
    )
    metadata.update(
        {
            "source_checkpoint": args.checkpoint,
            "calibration_seeds": list(args.calibration_seeds),
            "selection_seeds": list(args.selection_seeds),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "minibatch_size": args.minibatch_size,
            "loss": args.loss,
            "fit_stats": fit_stats,
            "selection_metrics_before": _overall_row(
                rows,
                "before",
            ),
            "selection_metrics_after": _overall_row(
                rows,
                "after",
            ),
            "actor_frozen": True,
        }
    )
    extra["value_calibration"] = metadata
    save_checkpoint(
        args.output_checkpoint,
        model,
        optimizer=None,
        extra=extra,
    )

    before = metadata["selection_metrics_before"]
    after = metadata["selection_metrics_after"]
    print(
        "value_calibration,"
        f"samples,{len(calibration_samples)},"
        f"selection_samples,{len(selection_samples)},"
        f"mae_before,{before['mae']:.6f},"
        f"mae_after,{after['mae']:.6f},"
        f"pearson_before,{before['pearson_r']:.6f},"
        f"pearson_after,{after['pearson_r']:.6f}"
    )
    print(f"calibrated_checkpoint_written: {args.output_checkpoint}")
    print(f"calibration_metrics_written: {args.metrics_output}")


def build_calibration_configs(
    intervals: Sequence[float],
    waves: int,
) -> list[Dict[str, Any]]:
    if waves < 1:
        raise ValueError("waves must be positive")
    configs = []
    for interval in dict.fromkeys(float(value) for value in intervals):
        if interval <= 0.0:
            raise ValueError("wave intervals must be positive")
        config = dict(DEFAULT_CONFIG)
        config["wave_interval"] = interval
        config["simulation_duration"] = interval * waves
        configs.append(config)
    if not configs:
        raise ValueError("at least one wave interval is required")
    return configs


def validate_seed_protocol(
    calibration_seeds: Sequence[int],
    selection_seeds: Sequence[int],
) -> None:
    calibration = {int(seed) for seed in calibration_seeds}
    selection = {int(seed) for seed in selection_seeds}
    if not calibration or not selection:
        raise ValueError("calibration and selection seeds are required")
    overlap = sorted(calibration & selection)
    if overlap:
        raise ValueError(
            f"calibration and selection seeds overlap: {overlap}"
        )
    forbidden = sorted(
        seed
        for seed in calibration | selection
        if 70001 <= seed <= 70050
    )
    if forbidden:
        raise ValueError(
            f"seeds 70001-70050 are forbidden: {forbidden}"
        )


def ppo_config_from_checkpoint(
    payload: Dict[str, Any],
) -> PPOConfig:
    stored = payload.get("extra", {}).get("ppo_config", {})
    compatible = {
        key: value
        for key, value in stored.items()
        if key in PPOConfig.__dataclass_fields__
    }
    return PPOConfig(**compatible)


def write_rows(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metrics report")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _overall_row(
    rows: Sequence[Dict[str, Any]],
    stage: str,
) -> Dict[str, Any]:
    return next(
        dict(row)
        for row in rows
        if row["stage"] == stage
        and row["group_type"] == "overall"
    )


if __name__ == "__main__":
    main()
