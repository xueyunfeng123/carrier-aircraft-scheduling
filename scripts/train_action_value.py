"""Train a direct action-conditioned Q ranker from CRN continuations."""

from __future__ import annotations

import argparse
from pathlib import Path

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.action_value import (
    ACTION_VALUE_CHECKPOINT_TYPE,
    ActionValueRanker,
    action_value_metadata,
    checkpoint_sha256,
    collect_counterfactual_action_value_samples,
    fit_action_value_ranker,
    predict_action_values,
)
from rl.checkpoint import load_checkpoint, save_checkpoint
from rl.obs_encoder import OBSERVATION_SCHEMA_VERSION
from rl.value_calibration import calibration_metric_rows
from scripts.calibrate_value import (
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_INTERVALS,
    DEFAULT_SELECTION_SEEDS,
    build_calibration_configs,
    validate_seed_protocol,
    write_rows,
)
from solution.rl_solver import RLSolver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor-checkpoint",
        default="checkpoints/rl_multiload_bc.pt",
    )
    parser.add_argument(
        "--output-checkpoint",
        default="checkpoints/rl_multiload_action_value.pt",
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/action_value_selection_seed41001_5.csv",
    )
    parser.add_argument(
        "--training-seeds",
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
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--max-branch-states", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--branch-seed", type=int, default=52_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--ranking-loss-coef", type=float, default=1.0)
    parser.add_argument("--q-hidden-dim", type=int)
    parser.add_argument(
        "--freeze-q-encoder",
        action="store_true",
        help="fit only the Q head instead of the independent encoder copy",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=100_000)
    args = parser.parse_args()

    actor_path = Path(args.actor_checkpoint)
    output_path = Path(args.output_checkpoint)
    if not actor_path.is_file():
        parser.error(f"actor checkpoint does not exist: {actor_path}")
    if actor_path.resolve() == output_path.resolve():
        parser.error(
            "action-value output checkpoint must differ from actor checkpoint"
        )
    validate_seed_protocol(
        args.training_seeds,
        args.selection_seeds,
    )
    configs = build_calibration_configs(
        args.wave_intervals,
        args.waves,
    )

    actor_payload = load_checkpoint(
        str(actor_path),
        device=args.device,
    )
    actor = RLSolver(
        CarrierAircraftSchedulingEnv(configs[0]),
        checkpoint=str(actor_path),
        device=args.device,
    ).model
    ranker = ActionValueRanker.from_actor(
        actor,
        q_hidden_dim=args.q_hidden_dim,
    ).to(args.device)
    collection_options = {
        "stride": args.stride,
        "max_branch_states": args.max_branch_states,
        "top_k": args.top_k,
        "workers": args.workers,
        "branch_seed": args.branch_seed,
        "max_steps": args.max_steps,
    }
    training_samples = collect_counterfactual_action_value_samples(
        actor,
        configs,
        args.training_seeds,
        args.device,
        **collection_options,
    )
    selection_samples = collect_counterfactual_action_value_samples(
        actor,
        configs,
        args.selection_seeds,
        args.device,
        **collection_options,
    )
    before_predictions = predict_action_values(
        ranker,
        selection_samples,
        args.device,
        args.minibatch_size,
    )
    fit_stats = fit_action_value_ranker(
        ranker,
        training_samples,
        args.device,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        ranking_loss_coef=args.ranking_loss_coef,
        update_encoder=not args.freeze_q_encoder,
        seed=args.training_seeds[0],
        protected_actor=actor,
    )
    after_predictions = predict_action_values(
        ranker,
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

    actor_digest = checkpoint_sha256(str(actor_path))
    metadata = action_value_metadata(
        training_samples,
        str(actor_path),
        actor_digest,
    )
    metadata.update(
        {
            "training_seeds": list(args.training_seeds),
            "selection_seeds": list(args.selection_seeds),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "minibatch_size": args.minibatch_size,
            "ranking_loss_coef": args.ranking_loss_coef,
            "q_encoder_updated": not args.freeze_q_encoder,
            "fit_stats": fit_stats,
            "collection": {
                "method": "counterfactual_source_action",
                "state_filter": "distinct_top_k_at_least_2",
                "sampling": "ambiguous_stride_reservoir",
                **collection_options,
            },
            "selection_metrics_before": _overall_row(rows, "before"),
            "selection_metrics_after": _overall_row(rows, "after"),
            "actor_frozen": True,
        }
    )
    save_checkpoint(
        str(output_path),
        ranker,
        optimizer=None,
        extra={
            "checkpoint_type": ACTION_VALUE_CHECKPOINT_TYPE,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "source_actor_model_config": actor_payload.get(
                "model_config",
                {},
            ),
            "action_value": metadata,
        },
    )

    before = metadata["selection_metrics_before"]
    after = metadata["selection_metrics_after"]
    print(
        "action_value_training,"
        f"samples,{len(training_samples)},"
        f"selection_samples,{len(selection_samples)},"
        f"mae_before,{before['mae']:.6f},"
        f"mae_after,{after['mae']:.6f},"
        f"spearman_before,{before['spearman_r']:.6f},"
        f"spearman_after,{after['spearman_r']:.6f}"
    )
    print(f"action_value_checkpoint_written: {output_path}")
    print(f"action_value_metrics_written: {args.metrics_output}")


def _overall_row(rows, stage: str):
    return next(
        {
            key: value
            for key, value in row.items()
            if key != "stage"
        }
        for row in rows
        if row["stage"] == stage
        and row["group_type"] == "overall"
    )


if __name__ == "__main__":
    main()
