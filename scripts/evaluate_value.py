"""Evaluate critic calibration and optional one-step value reranking."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Dict, Sequence

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.checkpoint import load_checkpoint
from rl.value_calibration import (
    calibration_metric_rows,
    collect_counterfactual_value_samples,
    collect_value_samples,
    predict_values,
    validate_development_seeds,
    validate_rerank_calibration,
)
from scripts.analyze_paired_benchmark import exact_two_sided_sign_test
from scripts.calibrate_value import (
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_INTERVALS,
    DEFAULT_SELECTION_SEEDS,
    build_calibration_configs,
    ppo_config_from_checkpoint,
    validate_seed_protocol,
    write_rows,
)
from scripts.solve import run_episode
from solution.rl_solver import RLSolver


DEFAULT_EVALUATION_SEEDS = tuple(range(43001, 43006))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/rl_multiload_value_calibrated.pt",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVALUATION_SEEDS),
    )
    parser.add_argument(
        "--wave-intervals",
        type=float,
        nargs="+",
        default=list(DEFAULT_INTERVALS),
    )
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-branch-states", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--branch-seed", type=int)
    parser.add_argument(
        "--metrics-output",
        default="outputs/value_calibration_evaluation_seed43001_5.csv",
    )
    parser.add_argument("--rerank-top-k", type=int, default=1)
    parser.add_argument("--rerank-seed", type=int, default=91_000)
    parser.add_argument(
        "--benefit-output",
        default="outputs/value_rerank_seed43001_5.csv",
    )
    args = parser.parse_args()

    configs = build_calibration_configs(
        args.wave_intervals,
        args.waves,
    )
    payload = load_checkpoint(args.checkpoint, device=args.device)
    validate_evaluation_seeds(args.seeds, payload)
    metadata = validate_rerank_calibration(
        payload,
        min(args.wave_intervals),
    )
    validate_rerank_calibration(
        payload,
        max(args.wave_intervals),
    )
    solver = RLSolver(
        CarrierAircraftSchedulingEnv(configs[0]),
        checkpoint=args.checkpoint,
        device=args.device,
    )
    reward_config = ppo_config_from_checkpoint(payload)
    collection = metadata.get("collection", {})
    if collection.get("method") == "counterfactual_successor":
        samples = collect_counterfactual_value_samples(
            solver.model,
            configs,
            args.seeds,
            args.device,
            reward_config,
            stride=_option_or_metadata(
                args.stride,
                collection,
                "stride",
            ),
            max_branch_states=_option_or_metadata(
                args.max_branch_states,
                collection,
                "max_branch_states",
            ),
            top_k=_option_or_metadata(
                args.top_k,
                collection,
                "top_k",
            ),
            workers=_option_or_metadata(
                args.workers,
                collection,
                "workers",
            ),
            branch_seed=_option_or_metadata(
                args.branch_seed,
                collection,
                "branch_seed",
            ),
            max_steps=args.max_steps,
        )
    else:
        samples = collect_value_samples(
            solver.model,
            configs,
            args.seeds,
            args.device,
            reward_config,
            target_method=metadata["target_method"],
            gae_lambda=float(metadata["gae_lambda"]),
            max_steps=args.max_steps,
        )
    predictions = predict_values(
        solver.model,
        samples,
        args.device,
        args.batch_size,
    )
    metric_rows = calibration_metric_rows(samples, predictions)
    write_rows(args.metrics_output, metric_rows)
    overall = next(
        row
        for row in metric_rows
        if row["group_type"] == "overall"
    )
    print(
        "value_evaluation,"
        f"samples,{len(samples)},"
        f"mae,{overall['mae']:.6f},"
        f"rmse,{overall['rmse']:.6f},"
        f"pearson,{overall['pearson_r']:.6f},"
        f"spearman,{overall['spearman_r']:.6f}"
    )
    print(f"value_metrics_written: {args.metrics_output}")

    if args.rerank_top_k > 1:
        benefit_rows = evaluate_rerank_benefit(
            args.checkpoint,
            configs,
            args.seeds,
            args.device,
            args.rerank_top_k,
            args.rerank_seed,
            args.max_steps,
        )
        write_rows(args.benefit_output, benefit_rows)
        actor_scores = [
            int(row["actor_sorties"])
            for row in benefit_rows
        ]
        rerank_scores = [
            int(row["rerank_sorties"])
            for row in benefit_rows
        ]
        stats = exact_two_sided_sign_test(
            rerank_scores,
            actor_scores,
        )
        print(
            "value_rerank,"
            f"top_k,{args.rerank_top_k},"
            f"actor_mean,{statistics.mean(actor_scores):.6f},"
            f"rerank_mean,{statistics.mean(rerank_scores):.6f},"
            f"mean_delta,{stats['mean_delta']:.6f},"
            f"wins,{int(stats['wins'])},"
            f"ties,{int(stats['ties'])},"
            f"losses,{int(stats['losses'])},"
            f"p_value,{stats['p_value']:.8g}"
        )
        print(f"value_rerank_benefit_written: {args.benefit_output}")


def validate_evaluation_seeds(
    seeds: Sequence[int],
    checkpoint_payload: Dict[str, Any] | None = None,
) -> None:
    evaluation = set(
        validate_development_seeds(seeds, "evaluation")
    )

    metadata = (
        checkpoint_payload.get("extra", {}).get(
            "value_calibration",
            {},
        )
        if checkpoint_payload is not None
        else {}
    )
    if (
        int(metadata.get("version", 1)) >= 2
        and (
            "calibration_seeds" not in metadata
            or "selection_seeds" not in metadata
        )
    ):
        raise ValueError(
            "version 2 value calibration metadata must record explicit "
            "calibration and selection seed partitions"
        )
    calibration_values = metadata.get(
        "calibration_seeds",
        DEFAULT_CALIBRATION_SEEDS,
    )
    selection_values = metadata.get(
        "selection_seeds",
        DEFAULT_SELECTION_SEEDS,
    )
    validate_seed_protocol(calibration_values, selection_values)
    calibration = {int(seed) for seed in calibration_values}
    selection = {int(seed) for seed in selection_values}
    overlap = sorted(evaluation & (calibration | selection))
    if overlap:
        raise ValueError(
            "evaluation seeds overlap calibration or selection seeds: "
            f"{overlap}"
        )


def _option_or_metadata(
    option: int | None,
    metadata: Dict[str, Any],
    name: str,
) -> int:
    value = metadata.get(name) if option is None else option
    if value is None:
        raise ValueError(
            f"counterfactual calibration metadata is missing {name}"
        )
    return int(value)


def evaluate_rerank_benefit(
    checkpoint: str,
    configs: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    device: str,
    top_k: int,
    rerank_seed: int,
    max_steps: int,
) -> list[Dict[str, Any]]:
    rows = []
    for config in configs:
        for seed in seeds:
            common_options = {
                "checkpoint": checkpoint,
                "device": device,
                "deterministic": True,
            }
            started = time.perf_counter()
            actor = run_episode(
                "rl",
                config,
                int(seed),
                max_steps,
                solver_options=common_options,
            )
            actor_runtime = time.perf_counter() - started
            started = time.perf_counter()
            reranked = run_episode(
                "rl",
                config,
                int(seed),
                max_steps,
                solver_options={
                    **common_options,
                    "value_rerank_top_k": top_k,
                    "value_rerank_seed": (
                        int(rerank_seed)
                        + int(seed) * 10_000
                        + int(round(config["wave_interval"] * 10))
                    ),
                },
            )
            rerank_runtime = time.perf_counter() - started
            rows.append(
                {
                    "seed": int(seed),
                    "wave_interval": float(
                        config["wave_interval"]
                    ),
                    "actor_sorties": actor[
                        "total_sorties_completed"
                    ],
                    "rerank_sorties": reranked[
                        "total_sorties_completed"
                    ],
                    "delta_sorties": (
                        reranked["total_sorties_completed"]
                        - actor["total_sorties_completed"]
                    ),
                    "actor_missed": actor[
                        "total_missed_sorties"
                    ],
                    "rerank_missed": reranked[
                        "total_missed_sorties"
                    ],
                    "actor_runtime_seconds": actor_runtime,
                    "rerank_runtime_seconds": rerank_runtime,
                }
            )
    return rows


if __name__ == "__main__":
    main()
