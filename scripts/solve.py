"""External runner for solution classes."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from typing import Any, Dict, List, Type

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG, HIGH_LEVEL_ACTIONS
from env.disruptions import PROFILE_SETTINGS
from scripts.evaluation_defaults import (
    DEFAULT_EVALUATION_DURATION,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVALUATION_WAVE_INTERVAL,
)
from solution import (
    CPSATSolver,
    EDDSolver,
    FIFOSolver,
    RandomSolver,
    RLSolver,
    RollingRepairSolver,
    SPTSolver,
    WaveHeuristicSolver,
)
from solution.event_triggered_repair_solver import (
    EventTriggeredRepairSolver,
)
from solution.adaptive_rl_cp_solver import AdaptiveRLCPSolver
from solution.hybrid_rescheduling_solver import (
    ActorFallbackRepairSolver,
    HybridReschedulingSolver,
    RLGuidedRepairSolver,
    StrongFallbackGuidedRepairSolver,
)


SOLVERS: Dict[str, Type] = {
    "adaptive_rl_cp": AdaptiveRLCPSolver,
    "cp_sat": CPSATSolver,
    "cp_sat_repair": RollingRepairSolver,
    "edd": EDDSolver,
    "event_cp_sat_repair": EventTriggeredRepairSolver,
    "fifo": FIFOSolver,
    "heuristic": WaveHeuristicSolver,
    "random": RandomSolver,
    "rl": RLSolver,
    "rl_cp_sat": HybridReschedulingSolver,
    "rl_fallback_cp_sat": ActorFallbackRepairSolver,
    "rl_guided_cp_sat": RLGuidedRepairSolver,
    "rl_guided_strong_cp_sat": (
        StrongFallbackGuidedRepairSolver
    ),
    "spt": SPTSolver,
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
    elif solver_name in (
        "cp_sat",
        "cp_sat_repair",
        "event_cp_sat_repair",
        "adaptive_rl_cp",
        "rl_cp_sat",
        "rl_fallback_cp_sat",
        "rl_guided_cp_sat",
        "rl_guided_strong_cp_sat",
    ):
        solver = solver_cls(env, **solver_options)
    elif solver_name == "rl":
        solver = solver_cls(env, **solver_options)
    else:
        solver = solver_cls(env)

    total_reward = 0.0
    steps = 0
    started_actions = {name: 0 for name in HIGH_LEVEL_ACTIONS.values()}
    generic_decisions: List[Dict[str, Any]] = []

    while not env.done and steps < max_steps:
        decision_started = time.perf_counter()
        action = solver.choose_action()
        decision_latency_ms = (
            time.perf_counter() - decision_started
        ) * 1000.0
        generic_decisions.append(
            {
                "decision_index": steps,
                "simulation_time": float(env.time),
                "source": "solver",
                "high_level": (
                    int(action["high_level"])
                    if action is not None
                    else ""
                ),
                "aircraft_id": (
                    int(action["aircraft_id"])
                    if action is not None
                    else ""
                ),
                "latency_ms": decision_latency_ms,
            }
        )
        _, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if info["action_started"]:
            started_actions[info["action_started"]] += 1
        if done:
            break

    metrics = env.get_evaluation_metrics()
    if hasattr(solver, "get_telemetry"):
        solver_telemetry = solver.get_telemetry()
    else:
        ordered_latencies = sorted(
            float(record["latency_ms"])
            for record in generic_decisions
        )
        solver_telemetry = {
            "decisions": len(generic_decisions),
            "solve_calls": 0,
            "deadline_misses": 0,
            "fallbacks": 0,
            "cached_actions": 0,
            "latency_p50_ms": _latency_quantile(
                ordered_latencies,
                0.50,
            ),
            "latency_p95_ms": _latency_quantile(
                ordered_latencies,
                0.95,
            ),
            "latency_p99_ms": _latency_quantile(
                ordered_latencies,
                0.99,
            ),
            "latency_max_ms": max(
                ordered_latencies,
                default=0.0,
            ),
            "solve_latency_p50_ms": 0.0,
            "solve_latency_p95_ms": 0.0,
            "solve_latency_p99_ms": 0.0,
            "solve_latency_max_ms": 0.0,
            "mean_solve_ms": 0.0,
        }
    decision_records = (
        solver.get_decision_records()
        if hasattr(solver, "get_decision_records")
        else generic_decisions
    )
    return {
        "solver": solver_name,
        "scenario_profile": env.scenario_profile.name,
        "seed": seed,
        "steps": steps,
        "done": env.done,
        "simulation_time": env.time,
        "total_reward": total_reward,
        "started_actions": started_actions,
        "total_sorties_completed": metrics["total_sorties_completed"],
        "total_missed_sorties": metrics["total_missed_sorties"],
        "sortie_generation_rate_per_hour": metrics["sortie_generation_rate_per_hour"],
        "sortie_completion_rate": metrics["sortie_completion_rate"],
        "group_metrics": metrics["group_metrics"],
        "cbs_replan": metrics["cbs_replan"],
        "disruptions": metrics["disruptions"],
        "scenario_tape": metrics["scenario_tape"],
        "solver_telemetry": solver_telemetry,
        "decision_records": decision_records,
        "timing_records": env.get_aircraft_timing_records(),
        "wave_records": env.get_wave_records(),
        "missed_sortie_records": env.get_missed_sortie_records(),
        "event_log": env.get_event_log(),
    }


def _latency_quantile(
    values: List[float],
    fraction: float,
) -> float:
    if not values:
        return 0.0
    index = min(
        len(values) - 1,
        max(0, int(round((len(values) - 1) * fraction))),
    )
    return float(values[index])


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config["num_aircraft"] = args.num_aircraft
    config["group_size"] = args.group_size
    config["num_parking_spots"] = args.num_parking_spots
    config["parking_base_transfer_time"] = args.parking_base_transfer_time
    config["parking_ring_time_step"] = args.parking_ring_time_step
    config["spatial_graph_enabled"] = not getattr(
        args,
        "disable_spatial_graph",
        False,
    )
    config["deck_edge_travel_time"] = getattr(
        args,
        "deck_edge_travel_time",
        DEFAULT_CONFIG["deck_edge_travel_time"],
    )
    config["cbs_replan_enabled"] = getattr(
        args,
        "cbs_replan",
        DEFAULT_CONFIG["cbs_replan_enabled"],
    )
    config["cbs_max_expanded_nodes"] = getattr(
        args,
        "cbs_max_expanded_nodes",
        DEFAULT_CONFIG["cbs_max_expanded_nodes"],
    )
    config["disruption_profile"] = getattr(
        args,
        "disruption_profile",
        DEFAULT_CONFIG["disruption_profile"],
    )
    config["num_ammo_transport_vehicles"] = args.num_ammo_transport_vehicles
    config["num_lower_weapon_lifts"] = args.num_lower_weapon_lifts
    config["num_upper_weapon_lifts"] = args.num_upper_weapon_lifts
    config["simulation_duration"] = args.simulation_duration
    config["wave_interval"] = args.wave_interval
    return config


def write_dict_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_runs_csv(path: str, results: List[Dict[str, Any]]) -> None:
    rows = []
    for run_id, result in enumerate(results, start=1):
        rows.append(
            {
                "run": run_id,
                "solver": result["solver"],
                "scenario_profile": result["scenario_profile"],
                "seed": result["seed"],
                "done": result["done"],
                "steps": result["steps"],
                "simulation_time": f"{result['simulation_time']:.6f}",
                "total_sorties_completed": result["total_sorties_completed"],
                "total_missed_sorties": result["total_missed_sorties"],
                "sortie_generation_rate_per_hour": (
                    f"{result['sortie_generation_rate_per_hour']:.6f}"
                ),
                "sortie_completion_rate": f"{result['sortie_completion_rate']:.6f}",
                "total_reward": f"{result['total_reward']:.6f}",
                "disruption_profile": result["disruptions"]["profile"],
                "disruptions_scheduled": result["disruptions"]["scheduled"],
                "disruptions_started": result["disruptions"]["started"],
                "disruptions_ended": result["disruptions"]["ended"],
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
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--runs", type=int, default=1)
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
    parser.add_argument(
        "--disable-spatial-graph",
        action="store_true",
        help="disable deck routes and movement time for matched controls",
    )
    parser.add_argument(
        "--deck-edge-travel-time",
        type=float,
        default=DEFAULT_CONFIG["deck_edge_travel_time"],
    )
    parser.add_argument("--cbs-replan", action="store_true")
    parser.add_argument(
        "--cbs-max-expanded-nodes",
        type=int,
        default=DEFAULT_CONFIG["cbs_max_expanded_nodes"],
    )
    parser.add_argument(
        "--disruption-profile",
        choices=tuple(PROFILE_SETTINGS),
        default=DEFAULT_CONFIG["disruption_profile"],
    )
    parser.add_argument("--simulation-duration", type=float, default=DEFAULT_EVALUATION_DURATION)
    parser.add_argument("--wave-interval", type=float, default=DEFAULT_EVALUATION_WAVE_INTERVAL)
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
    parser.add_argument("--event-log-csv", type=str, default="")
    parser.add_argument("--cp-sat-max-time", type=float, default=0.05)
    parser.add_argument("--repair-budget-ms", type=float, default=50.0)
    parser.add_argument(
        "--repair-neighborhood-size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--repair-horizon-waves",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--repair-guidance-strength",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--repair-max-trigger-severity",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--repair-scope",
        choices=("current_wave", "two_waves", "affected"),
        default="two_waves",
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--repair-control-checkpoint",
        type=str,
        default="",
    )
    parser.add_argument("--rl-device", type=str, default="cpu")
    parser.add_argument("--rl-stochastic", action="store_true")
    parser.add_argument("--rl-hidden-dim", type=int, default=128)
    parser.add_argument("--rl-aircraft-embed-dim", type=int, default=64)
    args = parser.parse_args()

    config = build_config(args)
    solver_options = {}
    if args.solver == "cp_sat":
        solver_options = {
            "max_time_seconds": args.cp_sat_max_time,
        }
    elif args.solver in (
        "cp_sat_repair",
        "event_cp_sat_repair",
    ):
        solver_options = {
            "budget_ms": args.repair_budget_ms,
            "neighborhood_size": args.repair_neighborhood_size,
            "horizon_waves": args.repair_horizon_waves,
            "scope": args.repair_scope,
            "guidance_strength": (
                args.repair_guidance_strength
            ),
        }
    elif args.solver == "rl":
        solver_options = {
            "checkpoint": args.checkpoint,
            "device": args.rl_device,
            "deterministic": not args.rl_stochastic,
            "hidden_dim": args.rl_hidden_dim,
            "aircraft_embed_dim": args.rl_aircraft_embed_dim,
        }
    elif args.solver == "adaptive_rl_cp":
        solver_options = {
            "checkpoint": args.checkpoint,
            "device": args.rl_device,
            "deterministic": not args.rl_stochastic,
            "max_time_seconds": args.cp_sat_max_time,
        }
    elif args.solver in (
        "rl_cp_sat",
        "rl_fallback_cp_sat",
        "rl_guided_cp_sat",
        "rl_guided_strong_cp_sat",
    ):
        solver_options = {
            "checkpoint": args.checkpoint,
            "device": args.rl_device,
            "deterministic": not args.rl_stochastic,
            "control_checkpoint": (
                args.repair_control_checkpoint
            ),
            "budget_ms": args.repair_budget_ms,
            "neighborhood_size": args.repair_neighborhood_size,
            "horizon_waves": args.repair_horizon_waves,
            "scope": args.repair_scope,
            "guidance_strength": (
                args.repair_guidance_strength
            ),
            "max_trigger_severity": (
                args.repair_max_trigger_severity
            ),
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

    print("run,solver,seed,done,steps,simulation_time,total_sorties_completed,total_missed_sorties,total_reward")
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
            f"{result['total_reward']:.2f}"
        )

    if args.runs > 1:
        mean_completed = statistics.mean(result["total_sorties_completed"] for result in results)
        mean_missed = statistics.mean(result["total_missed_sorties"] for result in results)
        print(f"mean_total_sorties_completed: {mean_completed:.2f}")
        print(f"mean_total_missed_sorties: {mean_missed:.2f}")
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
    if args.event_log_csv:
        write_dict_csv(args.event_log_csv, best["event_log"])
        print(f"event_log_csv_written: {args.event_log_csv}")


if __name__ == "__main__":
    main()
