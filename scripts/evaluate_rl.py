"""Evaluate a trained RL checkpoint."""

from __future__ import annotations

import argparse
import statistics

from env.config import DEFAULT_CONFIG
from scripts.solve import build_config, run_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rl_policy.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--num-aircraft", type=int, default=DEFAULT_CONFIG["num_aircraft"])
    parser.add_argument("--group-size", type=int, default=DEFAULT_CONFIG["group_size"])
    parser.add_argument("--num-parking-spots", type=int, default=DEFAULT_CONFIG["num_parking_spots"])
    parser.add_argument("--parking-base-transfer-time", type=float, default=DEFAULT_CONFIG["parking_base_transfer_time"])
    parser.add_argument("--parking-ring-time-step", type=float, default=DEFAULT_CONFIG["parking_ring_time_step"])
    parser.add_argument("--simulation-duration", type=float, default=DEFAULT_CONFIG["simulation_duration"])
    parser.add_argument("--wave-interval", type=float, default=DEFAULT_CONFIG["wave_interval"])
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


if __name__ == "__main__":
    main()
