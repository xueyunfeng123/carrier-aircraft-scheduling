"""External runner for solution classes."""

from __future__ import annotations

import argparse
import csv
import statistics
from typing import Any, Dict, List, Type

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG
from solution import (
    CPSATSolver,
    EDDSolver,
    FIFOSolver,
    RandomSolver,
    RLSolver,
    SPTSolver,
    SampledRandomSolver,
    WaveHeuristicSolver,
)


SOLVERS: Dict[str, Type] = {
    "cp_sat": CPSATSolver,
    "edd": EDDSolver,
    "fifo": FIFOSolver,
    "heuristic": WaveHeuristicSolver,
    "random": RandomSolver,
    "rl": RLSolver,
    "spt": SPTSolver,
    "sampled": SampledRandomSolver,
}


def run_episode(
    solver_name: str,
    config: Dict[str, Any],
    seed: int,
    max_steps: int,
    solver_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    env = CarrierAircraftSchedulingEnv(config)
    env.reset(seed=seed)
    solver_cls = SOLVERS[solver_name]
    solver_options = solver_options or {}
    if solver_name == "random":
        solver = solver_cls(env, seed=seed)
    elif solver_name == "sampled":
        solver = solver_cls(env, seed=seed, max_steps=max_steps, **solver_options)
    elif solver_name == "cp_sat":
        solver = solver_cls(env, **solver_options)
    elif solver_name == "rl":
        solver = solver_cls(env, **solver_options)
    else:
        solver = solver_cls(env)

    total_reward = 0.0
    steps = 0
    started_actions = {"R": 0, "F": 0, "M": 0, "L": 0}

    while not env.done and steps < max_steps:
        action = solver.choose_action()
        _, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if info["action_started"]:
            started_actions[info["action_started"]] += 1
        if done:
            break

    metrics = env.get_evaluation_metrics()
    return {
        "solver": solver_name,
        "seed": seed,
        "steps": steps,
        "done": env.done,
        "simulation_time": env.time,
        "total_reward": total_reward,
        "started_actions": started_actions,
        "total_sorties_completed": metrics["total_sorties_completed"],
        "total_missed_sorties": metrics["total_missed_sorties"],
        "total_recovery_deadline_misses": metrics["total_recovery_deadline_misses"],
        "group_metrics": metrics["group_metrics"],
        "timing_records": env.get_aircraft_timing_records(),
        "wave_records": env.get_wave_records(),
        "missed_sortie_records": env.get_missed_sortie_records(),
    }


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config["num_aircraft"] = args.num_aircraft
    config["group_size"] = args.group_size
    config["num_parking_spots"] = args.num_parking_spots
    config["parking_base_transfer_time"] = args.parking_base_transfer_time
    config["parking_ring_time_step"] = args.parking_ring_time_step
    config["num_ammo_transport_vehicles"] = args.num_ammo_transport_vehicles
    config["num_lower_weapon_lifts"] = args.num_lower_weapon_lifts
    config["num_upper_weapon_lifts"] = args.num_upper_weapon_lifts
    config["simulation_duration"] = args.simulation_duration
    config["wave_interval"] = args.wave_interval
    return config


def write_dict_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_runs_csv(path: str, results: List[Dict[str, Any]]) -> None:
    rows = []
    for run_id, result in enumerate(results, start=1):
        rows.append(
            {
                "run": run_id,
                "solver": result["solver"],
                "seed": result["seed"],
                "done": result["done"],
                "steps": result["steps"],
                "simulation_time": f"{result['simulation_time']:.6f}",
                "total_sorties_completed": result["total_sorties_completed"],
                "total_missed_sorties": result["total_missed_sorties"],
                "total_recovery_deadline_misses": result["total_recovery_deadline_misses"],
                "total_reward": f"{result['total_reward']:.6f}",
                "A_sorties": result["group_metrics"]["A"]["sorties_completed"],
                "A_missed": result["group_metrics"]["A"]["missed_sorties"],
                "B_sorties": result["group_metrics"]["B"]["sorties_completed"],
                "B_missed": result["group_metrics"]["B"]["missed_sorties"],
            }
        )
    write_dict_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=sorted(SOLVERS), default="heuristic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--num-aircraft", type=int, default=DEFAULT_CONFIG["num_aircraft"])
    parser.add_argument("--group-size", type=int, default=DEFAULT_CONFIG["group_size"])
    parser.add_argument("--num-parking-spots", type=int, default=DEFAULT_CONFIG["num_parking_spots"])
    parser.add_argument(
        "--parking-base-transfer-time",
        type=float,
        default=DEFAULT_CONFIG["parking_base_transfer_time"],
    )
    parser.add_argument(
        "--parking-ring-time-step",
        type=float,
        default=DEFAULT_CONFIG["parking_ring_time_step"],
    )
    parser.add_argument("--simulation-duration", type=float, default=DEFAULT_CONFIG["simulation_duration"])
    parser.add_argument("--wave-interval", type=float, default=DEFAULT_CONFIG["wave_interval"])
    parser.add_argument(
        "--num-ammo-transport-vehicles",
        type=int,
        default=DEFAULT_CONFIG["num_ammo_transport_vehicles"],
    )
    parser.add_argument(
        "--num-lower-weapon-lifts",
        type=int,
        default=DEFAULT_CONFIG["num_lower_weapon_lifts"],
    )
    parser.add_argument(
        "--num-upper-weapon-lifts",
        type=int,
        default=DEFAULT_CONFIG["num_upper_weapon_lifts"],
    )
    parser.add_argument("--runs-csv", type=str, default="")
    parser.add_argument("--timing-csv", type=str, default="")
    parser.add_argument("--missed-csv", type=str, default="")
    parser.add_argument("--sampled-samples", type=int, default=30)
    parser.add_argument("--cp-sat-max-time", type=float, default=0.05)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--rl-device", type=str, default="cpu")
    parser.add_argument("--rl-stochastic", action="store_true")
    parser.add_argument("--rl-hidden-dim", type=int, default=128)
    parser.add_argument("--rl-aircraft-embed-dim", type=int, default=64)
    args = parser.parse_args()

    config = build_config(args)
    solver_options = {}
    if args.solver == "sampled":
        solver_options = {
            "samples": args.sampled_samples,
        }
    elif args.solver == "cp_sat":
        solver_options = {
            "max_time_seconds": args.cp_sat_max_time,
        }
    elif args.solver == "rl":
        solver_options = {
            "checkpoint": args.checkpoint,
            "device": args.rl_device,
            "deterministic": not args.rl_stochastic,
            "hidden_dim": args.rl_hidden_dim,
            "aircraft_embed_dim": args.rl_aircraft_embed_dim,
        }
    results = [
        run_episode(
            args.solver,
            config,
            seed=args.seed + run_id,
            max_steps=args.max_steps,
            solver_options=solver_options,
        )
        for run_id in range(args.runs)
    ]

    print(
        "run,solver,seed,done,steps,simulation_time,total_sorties_completed,"
        "total_missed_sorties,total_recovery_deadline_misses,total_reward"
    )
    for run_id, result in enumerate(results, start=1):
        print(
            f"{run_id},"
            f"{result['solver']},"
            f"{result['seed']},"
            f"{result['done']},"
            f"{result['steps']},"
            f"{result['simulation_time']:.2f},"
            f"{result['total_sorties_completed']},"
            f"{result['total_missed_sorties']},"
            f"{result['total_recovery_deadline_misses']},"
            f"{result['total_reward']:.2f}"
        )

    if args.runs > 1:
        mean_completed = statistics.mean(result["total_sorties_completed"] for result in results)
        mean_missed = statistics.mean(result["total_missed_sorties"] for result in results)
        mean_recovery_missed = statistics.mean(
            result["total_recovery_deadline_misses"]
            for result in results
        )
        print(f"mean_total_sorties_completed: {mean_completed:.2f}")
        print(f"mean_total_missed_sorties: {mean_missed:.2f}")
        print(f"mean_total_recovery_deadline_misses: {mean_recovery_missed:.2f}")
    best = max(
        results,
        key=lambda item: (
            item["total_sorties_completed"],
            -item["total_missed_sorties"],
        ),
    )

    if args.runs_csv:
        write_runs_csv(args.runs_csv, results)
        print(f"runs_csv_written: {args.runs_csv}")
    if args.timing_csv:
        write_dict_csv(args.timing_csv, best["timing_records"])
        print(f"timing_csv_written: {args.timing_csv}")
    if args.missed_csv:
        write_dict_csv(args.missed_csv, best["missed_sortie_records"])
        print(f"missed_csv_written: {args.missed_csv}")


if __name__ == "__main__":
    main()
