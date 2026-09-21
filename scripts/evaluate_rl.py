"""Evaluate a trained RL checkpoint."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from env.config import DEFAULT_CONFIG
from scripts.evaluation_defaults import (
    DEFAULT_EVALUATION_DURATION,
    DEFAULT_EVALUATION_RUNS,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVALUATION_WAVE_INTERVAL,
)
from scripts.solve import build_config, run_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rl_policy.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--low-rank-prior", type=float)
    parser.add_argument("--target-rank-prior", type=float)
    parser.add_argument("--low-rank-prior-scale", type=float, default=1.0)
    parser.add_argument("--disable-low-rank-prior", action="store_true")
    parser.add_argument(
        "--value-rerank-top-k",
        type=int,
        default=1,
        help="rerank the actor's top K complete actions with one-step value estimates",
    )
    parser.add_argument("--value-rerank-seed", type=int, default=0)
    parser.add_argument(
        "--value-rerank-include-heuristic",
        action="store_true",
    )
    parser.add_argument(
        "--value-rerank-include-cp-sat",
        action="store_true",
    )
    parser.add_argument(
        "--value-rerank-cp-sat-time",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--value-rerank-min-advantage",
        type=float,
        default=0.0,
    )
    parser.add_argument("--value-rerank-samples", type=int, default=1)
    parser.add_argument("--cbs-replan", action="store_true")
    parser.add_argument(
        "--cbs-max-expanded-nodes",
        type=int,
        default=DEFAULT_CONFIG["cbs_max_expanded_nodes"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--runs", type=int, default=DEFAULT_EVALUATION_RUNS)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--num-aircraft", type=int, default=DEFAULT_CONFIG["num_aircraft"])
    parser.add_argument(
        "--wave-size",
        "--group-size",
        dest="group_size",
        type=int,
        default=DEFAULT_CONFIG["group_size"],
    )
    parser.add_argument("--num-parking-spots", type=int, default=DEFAULT_CONFIG["num_parking_spots"])
    parser.add_argument("--parking-base-transfer-time", type=float, default=DEFAULT_CONFIG["parking_base_transfer_time"])
    parser.add_argument("--parking-ring-time-step", type=float, default=DEFAULT_CONFIG["parking_ring_time_step"])
    parser.add_argument("--simulation-duration", type=float, default=DEFAULT_EVALUATION_DURATION)
    parser.add_argument("--wave-interval", type=float, default=DEFAULT_EVALUATION_WAVE_INTERVAL)
    parser.add_argument("--num-ammo-transport-vehicles", type=int, default=DEFAULT_CONFIG["num_ammo_transport_vehicles"])
    parser.add_argument("--num-lower-weapon-lifts", type=int, default=DEFAULT_CONFIG["num_lower_weapon_lifts"])
    parser.add_argument("--num-upper-weapon-lifts", type=int, default=DEFAULT_CONFIG["num_upper_weapon_lifts"])
    args = parser.parse_args()

    config = build_config(args)
    results = [
        run_episode(
            "rl",
            config,
            seed=args.seed + run_id,
            max_steps=args.max_steps,
            solver_options={
                "checkpoint": args.checkpoint,
                "device": args.device,
                "deterministic": not args.stochastic,
                "low_rank_prior": args.low_rank_prior,
                "target_rank_prior": args.target_rank_prior,
                "low_rank_prior_scale": args.low_rank_prior_scale,
                "disable_low_rank_prior": args.disable_low_rank_prior,
                "value_rerank_top_k": args.value_rerank_top_k,
                "value_rerank_seed": args.value_rerank_seed,
                "value_rerank_include_heuristic": (
                    args.value_rerank_include_heuristic
                ),
                "value_rerank_include_cp_sat": (
                    args.value_rerank_include_cp_sat
                ),
                "value_rerank_cp_sat_time": (
                    args.value_rerank_cp_sat_time
                ),
                "value_rerank_min_advantage": (
                    args.value_rerank_min_advantage
                ),
                "value_rerank_samples": args.value_rerank_samples,
            },
        )
        for run_id in range(args.runs)
    ]

    print("run,seed,total_sorties_completed,total_missed_sorties,total_reward")
    for run_id, result in enumerate(results, start=1):
        print(
            f"{run_id},"
            f"{result['seed']},"
            f"{result['total_sorties_completed']},"
            f"{result['total_missed_sorties']},"
            f"{result['total_reward']:.2f}"
        )
    print(f"mean_total_sorties_completed: {statistics.mean(r['total_sorties_completed'] for r in results):.2f}")
    print(f"mean_total_missed_sorties: {statistics.mean(r['total_missed_sorties'] for r in results):.2f}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "run": run_id,
                "seed": result["seed"],
                "total_sorties_completed": result[
                    "total_sorties_completed"
                ],
                "total_missed_sorties": result[
                    "total_missed_sorties"
                ],
                "total_reward": result["total_reward"],
                "cbs_calls": int(
                    result["cbs_replan"]["calls"]
                ),
                "cbs_improved_batches": int(
                    result["cbs_replan"]["improved_batches"]
                ),
                "cbs_fallbacks": int(
                    result["cbs_replan"]["fallbacks"]
                ),
                "cbs_sum_of_costs_before": result[
                    "cbs_replan"
                ]["sum_of_costs_before"],
                "cbs_sum_of_costs_after": result[
                    "cbs_replan"
                ]["sum_of_costs_after"],
            }
            for run_id, result in enumerate(results, start=1)
        ]
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"evaluation_csv_written: {output}")


if __name__ == "__main__":
    main()
