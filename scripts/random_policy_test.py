"""Run a simple random policy on the carrier aircraft scheduling environment."""

from __future__ import annotations

import argparse
import csv
import statistics
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG, HIGH_LEVEL_ACTIONS
from scripts.evaluation_defaults import (
    DEFAULT_EVALUATION_DURATION,
    DEFAULT_EVALUATION_RUNS,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVALUATION_WAVE_INTERVAL,
)
from solution import RandomSolver


def print_timing_table(records: List[Dict[str, Optional[float]]]) -> None:
    headers = [
        "id",
        "group",
        "spot",
        "spot_t",
        "park_st",
        "sorties",
        "missed",
        "airborne",
        "pending_rec",
        "q_arm",
        "arm_stage",
        "rec_s",
        "rec_e",
        "fuel_s",
        "fuel_e",
        "ammo_s",
        "ammo_e",
        "deck_arm_s",
        "arm_s",
        "arm_e",
        "ready",
        "launch_s",
        "launch_e",
    ]
    print(",".join(headers))
    for record in records:
        row = [
            record["aircraft_id"],
            record["group"],
            record["spot_id"],
            record["spot_transfer_time"],
            record["parking_status"],
            record["sorties_completed"],
            record["missed_sorties"],
            record["is_airborne"],
            record["pending_recovery"],
            record["arm_quantity_required"],
            record["arm_stage"],
            record["recovery_start"],
            record["recovery_end"],
            record["fuel_start"],
            record["fuel_end"],
            record["ammo_extract_start"],
            record["ammo_to_assembly_end"],
            record["deck_arm_start"],
            record["arm_start"],
            record["arm_end"],
            record["launch_ready"],
            record["launch_start"],
            record["launch_end"],
        ]
        print(",".join(format_value(value) for value in row))


def format_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_timing_csv(path: str, records: List[Dict[str, Optional[float]]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def run_episode(config: Dict[str, Any], seed: int, max_steps: int) -> Dict[str, Any]:
    env = CarrierAircraftSchedulingEnv(config)
    env.reset(seed=seed)
    solver = RandomSolver(env, seed=seed)

    total_reward = 0.0
    steps = 0
    started_actions = {name: 0 for name in HIGH_LEVEL_ACTIONS.values()}

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
        "seed": seed,
        "steps": steps,
        "done": env.done,
        "simulation_time": env.time,
        "total_reward": total_reward,
        "total_sorties_completed": metrics["total_sorties_completed"],
        "total_missed_sorties": metrics["total_missed_sorties"],
        "group_metrics": metrics["group_metrics"],
        "started_actions": started_actions,
        "timing_records": env.get_aircraft_timing_records(),
    }


def write_runs_csv(path: str, results: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "run",
        "seed",
        "done",
        "steps",
        "simulation_time",
        "total_sorties_completed",
        "total_missed_sorties",
        "total_reward",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run_id, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "run": run_id,
                    "seed": result["seed"],
                    "done": result["done"],
                    "steps": result["steps"],
                    "simulation_time": f"{result['simulation_time']:.6f}",
                    "total_sorties_completed": result["total_sorties_completed"],
                    "total_missed_sorties": result["total_missed_sorties"],
                    "total_reward": f"{result['total_reward']:.6f}",
                }
            )


def print_run_summary(results: List[Dict[str, Any]]) -> None:
    missed_sorties = [result["total_missed_sorties"] for result in results if result["done"]]
    completed_sorties = [result["total_sorties_completed"] for result in results if result["done"]]
    failed = len(results) - len(missed_sorties)

    print("run,seed,done,steps,simulation_time,total_sorties_completed,total_missed_sorties,total_reward")
    for run_id, result in enumerate(results, start=1):
        print(
            f"{run_id},"
            f"{result['seed']},"
            f"{result['done']},"
            f"{result['steps']},"
            f"{result['simulation_time']:.2f},"
            f"{result['total_sorties_completed']},"
            f"{result['total_missed_sorties']},"
            f"{result['total_reward']:.2f}"
        )

    if not missed_sorties:
        print("summary: no completed runs")
        return

    print(f"completed_runs: {len(missed_sorties)}")
    print(f"failed_runs: {failed}")
    print(f"mean_sorties_completed: {statistics.mean(completed_sorties):.2f}")
    print(f"mean_missed_sorties: {statistics.mean(missed_sorties):.2f}")
    print(f"min_missed_sorties: {min(missed_sorties)}")
    print(f"max_missed_sorties: {max(missed_sorties)}")
    if len(missed_sorties) >= 2:
        print(f"std_missed_sorties: {statistics.stdev(missed_sorties):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--num-aircraft", type=int, default=DEFAULT_CONFIG["num_aircraft"])
    parser.add_argument("--group-size", type=int, default=DEFAULT_CONFIG["group_size"])
    parser.add_argument("--simulation-duration", type=float, default=DEFAULT_EVALUATION_DURATION)
    parser.add_argument("--wave-interval", type=float, default=DEFAULT_EVALUATION_WAVE_INTERVAL)
    parser.add_argument("--runs", type=int, default=DEFAULT_EVALUATION_RUNS)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--runs-csv", type=str, default="")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["num_aircraft"] = args.num_aircraft
    config["group_size"] = args.group_size
    config["simulation_duration"] = args.simulation_duration
    config["wave_interval"] = args.wave_interval

    results = []
    iterator = range(args.runs)
    if tqdm and args.runs > 1:
        iterator = tqdm(iterator, desc="runs", ncols=80)
    for run_id in iterator:
        results.append(run_episode(config, seed=args.seed + run_id, max_steps=args.max_steps))

    if args.runs == 1:
        result = results[0]
        records = result["timing_records"]
        print(f"steps: {result['steps']}")
        print(f"done: {result['done']}")
        print(f"simulation_time: {result['simulation_time']:.2f}")
        print(f"total_sorties_completed: {result['total_sorties_completed']}")
        print(f"total_missed_sorties: {result['total_missed_sorties']}")
        print(f"group_metrics: {result['group_metrics']}")
        print(f"total_reward: {result['total_reward']:.2f}")
        print(f"started_actions: {result['started_actions']}")
        print_timing_table(records)
    else:
        print_run_summary(results)

    if args.csv:
        write_timing_csv(args.csv, results[-1]["timing_records"])
        print(f"csv_written: {args.csv}")

    if args.runs_csv:
        write_runs_csv(args.runs_csv, results)
        print(f"runs_csv_written: {args.runs_csv}")


if __name__ == "__main__":
    main()
