"""Run the canonical paired benchmark for hybrid rescheduling research."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG
from env.disruptions import PROFILE_SETTINGS
from scripts.evaluation_protocol import (
    CANONICAL_BUDGETS_MS,
    CANONICAL_INTERVALS,
    CANONICAL_PROFILES,
    PHASE_SEEDS,
    build_manifest,
    canonical_json_sha256,
    recovery_metrics,
    reject_frozen_final_seeds,
    scenario_id,
    validate_phase_seeds,
    write_csv,
    write_json,
)
from scripts.solve import SOLVERS, run_episode


DEFAULT_SOLVERS = (
    "heuristic",
    "cp_sat_repair",
    "event_cp_sat_repair",
    "rl_fallback_cp_sat",
    "rl_guided_cp_sat",
    "rl_cp_sat",
)


def build_solver_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for solver in args.solvers:
        if solver in {
            "cp_sat_repair",
            "event_cp_sat_repair",
            "rl_cp_sat",
            "rl_fallback_cp_sat",
            "rl_guided_cp_sat",
        }:
            for budget_ms in args.budgets_ms:
                options: Dict[str, Any] = {
                    "budget_ms": float(budget_ms),
                    "neighborhood_size": args.neighborhood_size,
                    "horizon_waves": args.horizon_waves,
                    "scope": args.scope,
                }
                if solver in {
                    "rl_cp_sat",
                    "rl_fallback_cp_sat",
                    "rl_guided_cp_sat",
                }:
                    options.update(
                        {
                            "checkpoint": args.rl_checkpoint,
                            "device": args.device,
                            "deterministic": True,
                            "control_checkpoint": (
                                args.control_checkpoint
                            ),
                            "control_device": args.device,
                        }
                    )
                specs.append(
                    {
                        "label": f"{solver}_b{budget_ms:g}",
                        "solver": solver,
                        "budget_ms": float(budget_ms),
                        "options": options,
                    }
                )
        elif solver == "cp_sat":
            for budget_ms in args.budgets_ms:
                specs.append(
                    {
                        "label": f"cp_sat_b{budget_ms:g}",
                        "solver": solver,
                        "budget_ms": float(budget_ms),
                        "options": {
                            "max_time_seconds": (
                                float(budget_ms) / 1000.0
                            )
                        },
                    }
                )
        elif solver == "rl":
            specs.append(
                {
                    "label": "rl",
                    "solver": solver,
                    "budget_ms": 0.0,
                    "options": {
                        "checkpoint": args.rl_checkpoint,
                        "device": args.device,
                        "deterministic": True,
                    },
                }
            )
        else:
            specs.append(
                {
                    "label": solver,
                    "solver": solver,
                    "budget_ms": 0.0,
                    "options": {},
                }
            )
    return specs


def build_tasks(
    manifest: Mapping[str, Any],
    solver_specs: Sequence[Mapping[str, Any]],
    max_steps: int,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    base_config = dict(manifest["base_config"])
    for profile in manifest["profiles"]:
        for interval in manifest["wave_intervals"]:
            config = dict(base_config)
            config.update(
                {
                    "disruption_profile": str(profile),
                    "wave_interval": float(interval),
                    "simulation_duration": (
                        float(interval) * int(manifest["waves"])
                    ),
                    "scenario_tape_enabled": True,
                }
            )
            config_sha256 = canonical_json_sha256(config)
            for seed in manifest["seeds"]:
                preview_env = CarrierAircraftSchedulingEnv(config)
                preview_env.reset(seed=int(seed))
                disruption_schedule_sha256 = (
                    canonical_json_sha256(
                        [
                            spec.as_dict()
                            for spec in preview_env.disruption_schedule
                        ]
                    )
                )
                case_id = scenario_id(
                    str(profile),
                    float(interval),
                    int(manifest["waves"]),
                    int(seed),
                    config_sha256,
                    disruption_schedule_sha256,
                    int(config["scenario_tape_version"]),
                )
                for spec in solver_specs:
                    tasks.append(
                        {
                            "manifest_sha256": manifest[
                                "manifest_sha256"
                            ],
                            "protocol_version": manifest[
                                "protocol_version"
                            ],
                            "phase": manifest["phase"],
                            "scenario_id": case_id,
                            "config_sha256": config_sha256,
                            "disruption_schedule_sha256": (
                                disruption_schedule_sha256
                            ),
                            "profile": profile,
                            "wave_interval": float(interval),
                            "waves": int(manifest["waves"]),
                            "seed": int(seed),
                            "solver_label": spec["label"],
                            "solver": spec["solver"],
                            "budget_ms": spec["budget_ms"],
                            "solver_options": dict(spec["options"]),
                            "config": config,
                            "max_steps": int(max_steps),
                        }
                    )
    return tasks


def run_case(task: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    result = run_episode(
        str(task["solver"]),
        dict(task["config"]),
        seed=int(task["seed"]),
        max_steps=int(task["max_steps"]),
        solver_options=dict(task["solver_options"]),
    )
    runtime_seconds = time.perf_counter() - started
    if not result["done"]:
        raise RuntimeError(
            f"{task['solver_label']} did not finish scenario "
            f"{task['scenario_id']}"
        )
    return {
        "task": dict(task),
        "result": result,
        "runtime_seconds": runtime_seconds,
    }


def result_rows(
    completed: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    wave_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    baselines = {
        (
            item["task"]["solver_label"],
            int(item["task"]["seed"]),
            float(item["task"]["wave_interval"]),
        ): item["result"]
        for item in completed
        if item["task"]["profile"] == "none"
    }
    for item in completed:
        task = item["task"]
        result = item["result"]
        telemetry = result["solver_telemetry"]
        baseline_key = (
            task["solver_label"],
            int(task["seed"]),
            float(task["wave_interval"]),
        )
        baseline = baselines.get(baseline_key)
        if baseline is None and task["profile"] != "none":
            raise ValueError(
                "disrupted scenarios require a paired none baseline"
            )
        baseline = baseline or result
        recovery = recovery_metrics(
            result["event_log"],
            result["wave_records"],
            baseline["wave_records"],
            float(task["config"]["simulation_duration"]),
        )
        baseline_wave_sorties = {
            int(record["wave_index"]): int(
                record["sorties_completed"]
            )
            for record in baseline["wave_records"]
        }
        sortie_loss_area = sum(
            max(
                0,
                baseline_wave_sorties[
                    int(record["wave_index"])
                ]
                - int(record["sorties_completed"]),
            )
            for record in result["wave_records"]
        )
        baseline_area = sum(baseline_wave_sorties.values())
        run_key = (
            f"{task['scenario_id']}:{task['solver_label']}"
        )
        run_rows.append(
            {
                "protocol_version": task["protocol_version"],
                "manifest_sha256": task["manifest_sha256"],
                "run_key": run_key,
                "scenario_id": task["scenario_id"],
                "phase": task["phase"],
                "solver": task["solver_label"],
                "solver_family": task["solver"],
                "seed": task["seed"],
                "profile": task["profile"],
                "wave_interval": task["wave_interval"],
                "waves": task["waves"],
                "budget_ms": task["budget_ms"],
                "config_sha256": task["config_sha256"],
                "disruption_schedule_sha256": task[
                    "disruption_schedule_sha256"
                ],
                "scenario_tape_sha256": result[
                    "scenario_tape"
                ].get("sha256", ""),
                "total_sorties_completed": result[
                    "total_sorties_completed"
                ],
                "total_missed_sorties": result[
                    "total_missed_sorties"
                ],
                "sortie_generation_rate_per_hour": result[
                    "sortie_generation_rate_per_hour"
                ],
                "sortie_completion_rate": result[
                    "sortie_completion_rate"
                ],
                "worst_wave_sorties": min(
                    int(record["sorties_completed"])
                    for record in result["wave_records"]
                ),
                "runtime_seconds": item["runtime_seconds"],
                "decisions": telemetry.get("decisions", 0),
                "solve_calls": telemetry.get("solve_calls", 0),
                "deadline_misses": telemetry.get(
                    "deadline_misses",
                    0,
                ),
                "fallbacks": telemetry.get("fallbacks", 0),
                "latency_p50_ms": telemetry.get(
                    "latency_p50_ms",
                    0.0,
                ),
                "latency_p95_ms": telemetry.get(
                    "latency_p95_ms",
                    0.0,
                ),
                "latency_p99_ms": telemetry.get(
                    "latency_p99_ms",
                    0.0,
                ),
                "latency_max_ms": telemetry.get(
                    "latency_max_ms",
                    0.0,
                ),
                "solve_latency_p50_ms": telemetry.get(
                    "solve_latency_p50_ms",
                    0.0,
                ),
                "solve_latency_p95_ms": telemetry.get(
                    "solve_latency_p95_ms",
                    0.0,
                ),
                "solve_latency_p99_ms": telemetry.get(
                    "solve_latency_p99_ms",
                    0.0,
                ),
                "solve_latency_max_ms": telemetry.get(
                    "solve_latency_max_ms",
                    0.0,
                ),
                "sortie_loss_area": sortie_loss_area,
                "resilience_index": (
                    1.0 - sortie_loss_area / baseline_area
                    if baseline_area
                    else 1.0
                ),
                **recovery,
            }
        )
        for record in result["wave_records"]:
            wave_rows.append(
                {
                    "manifest_sha256": task["manifest_sha256"],
                    "run_key": run_key,
                    "scenario_id": task["scenario_id"],
                    "solver": task["solver_label"],
                    "seed": task["seed"],
                    "profile": task["profile"],
                    "wave_interval": task["wave_interval"],
                    "budget_ms": task["budget_ms"],
                    "wave_index": record["wave_index"],
                    "wave_time": record["time"],
                    "launch_group": record["launch_group"],
                    "launch_target": record["launch_target"],
                    "launches_started": record["launches_started"],
                    "sorties_completed": record[
                        "sorties_completed"
                    ],
                    "missed_sorties": record["missed_sorties"],
                }
            )
        for record in result["decision_records"]:
            decision_rows.append(
                {
                    "manifest_sha256": task["manifest_sha256"],
                    "run_key": run_key,
                    "scenario_id": task["scenario_id"],
                    "solver": task["solver_label"],
                    "seed": task["seed"],
                    "profile": task["profile"],
                    "wave_interval": task["wave_interval"],
                    "budget_cap_ms": task["budget_ms"],
                    **record,
                }
            )
    run_rows.sort(key=_row_sort_key)
    wave_rows.sort(
        key=lambda row: (
            row["profile"],
            float(row["wave_interval"]),
            int(row["seed"]),
            row["solver"],
            int(row["wave_index"]),
        )
    )
    decision_rows.sort(
        key=lambda row: (
            row["profile"],
            float(row["wave_interval"]),
            int(row["seed"]),
            row["solver"],
            int(row["decision_index"]),
        )
    )
    return run_rows, wave_rows, decision_rows


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["profile"],
        float(row["wave_interval"]),
        int(row["seed"]),
        row["solver"],
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_provenance(
    path: str,
    require_lineage: bool,
) -> Dict[str, Any]:
    from rl.checkpoint import load_checkpoint

    checkpoint_path = Path(path)
    payload = load_checkpoint(str(checkpoint_path), device="cpu")
    extra = payload.get("extra", {})
    training_seeds = [
        int(seed) for seed in extra.get("training_seeds", [])
    ]
    validation_seeds = [
        int(seed) for seed in extra.get("validation_seeds", [])
    ]
    if require_lineage and not training_seeds:
        raise ValueError(
            "final evaluation requires checkpoint training seed lineage: "
            f"{path}"
        )
    if require_lineage:
        invalid_training = sorted(
            set(training_seeds) - set(PHASE_SEEDS["train"])
        )
        invalid_validation = sorted(
            set(validation_seeds)
            - set(PHASE_SEEDS["selection"])
        )
        if invalid_training or invalid_validation:
            raise ValueError(
                "final evaluation requires canonical checkpoint "
                "lineage: "
                f"invalid_training={invalid_training}, "
                f"invalid_validation={invalid_validation}"
            )
    reject_frozen_final_seeds(
        [*training_seeds, *validation_seeds],
        f"checkpoint development ({path})",
    )
    return {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "training_seeds": training_seeds,
        "validation_seeds": validation_seeds,
        "checkpoint_type": extra.get(
            "checkpoint_type",
            "actor_policy",
        ),
    }


def repository_provenance() -> Dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        text=True,
    )
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"],
    )
    source_prefixes = (
        "env/",
        "rl/",
        "scripts/",
        "solution/",
        "tests/",
        "doc/",
        "README.md",
        "requirements.txt",
        "environment.yml",
    )
    untracked_source = {}
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        raw_path = line[3:]
        if not raw_path.startswith(source_prefixes):
            continue
        path = Path(raw_path)
        if path.is_file():
            untracked_source[raw_path] = file_sha256(path)
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_source_sha256": canonical_json_sha256(
            untracked_source
        ),
        "untracked_source_files": untracked_source,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "ortools", "torch")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=tuple(PHASE_SEEDS),
        default="dev",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_SETTINGS),
        default=list(CANONICAL_PROFILES),
    )
    parser.add_argument(
        "--intervals",
        type=float,
        nargs="+",
        default=list(CANONICAL_INTERVALS),
    )
    parser.add_argument(
        "--budgets-ms",
        type=float,
        nargs="+",
        default=list(CANONICAL_BUDGETS_MS),
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=tuple(sorted(SOLVERS)),
        default=list(DEFAULT_SOLVERS),
    )
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--rl-checkpoint",
        default="checkpoints/rl_multiload_bc.pt",
    )
    parser.add_argument("--control-checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--neighborhood-size", type=int, default=32)
    parser.add_argument("--horizon-waves", type=int, default=2)
    parser.add_argument(
        "--scope",
        choices=("current_wave", "two_waves", "affected"),
        default="two_waves",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/publishable_dev",
    )
    parser.add_argument(
        "--disable-spatial-graph",
        action="store_true",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.waves < 1:
        parser.error("--waves must be positive")
    if any(value <= 0.0 for value in args.intervals):
        parser.error("--intervals must be positive")
    if any(value <= 0.0 for value in args.budgets_ms):
        parser.error("--budgets-ms must be positive")
    if any(
        solver
        in {
            "rl",
            "rl_cp_sat",
            "rl_fallback_cp_sat",
            "rl_guided_cp_sat",
        }
        for solver in args.solvers
    ) and not Path(args.rl_checkpoint).is_file():
        parser.error(
            f"RL checkpoint does not exist: {args.rl_checkpoint}"
        )
    if (
        args.control_checkpoint
        and not Path(args.control_checkpoint).is_file()
    ):
        parser.error(
            "repair control checkpoint does not exist: "
            f"{args.control_checkpoint}"
        )

    seeds = validate_phase_seeds(
        args.phase,
        args.seeds or PHASE_SEEDS[args.phase],
        purpose="evaluation",
    )
    base_config = dict(DEFAULT_CONFIG)
    base_config["spatial_graph_enabled"] = (
        not args.disable_spatial_graph
    )
    solver_specs = build_solver_specs(args)
    provenance = repository_provenance()
    checkpoint_paths = []
    if any(
        solver
        in {
            "rl",
            "rl_cp_sat",
            "rl_fallback_cp_sat",
            "rl_guided_cp_sat",
        }
        for solver in args.solvers
    ):
        checkpoint_paths.append(args.rl_checkpoint)
    if args.control_checkpoint:
        checkpoint_paths.append(args.control_checkpoint)
    provenance["checkpoints"] = [
        checkpoint_provenance(
            path,
            require_lineage=args.phase == "final",
        )
        for path in dict.fromkeys(checkpoint_paths)
    ]
    manifest = build_manifest(
        phase=args.phase,
        seeds=seeds,
        profiles=args.profiles,
        wave_intervals=args.intervals,
        budgets_ms=args.budgets_ms,
        waves=args.waves,
        solver_specs=solver_specs,
        base_config=base_config,
        provenance=provenance,
    )
    tasks = build_tasks(manifest, solver_specs, args.max_steps)
    if args.workers == 1:
        completed = []
        for index, task in enumerate(tasks, start=1):
            completed.append(run_case(task))
            print(
                f"completed,{index},{len(tasks)},"
                f"{task['solver_label']},{task['profile']},"
                f"{task['wave_interval']:g},{task['seed']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            completed = list(executor.map(run_case, tasks))

    run_rows, wave_rows, decision_rows = result_rows(completed)
    output_dir = Path(args.output_dir)
    write_json(output_dir / "manifest.json", manifest)
    write_csv(output_dir / "runs.csv", run_rows)
    write_csv(output_dir / "waves.csv", wave_rows)
    write_csv(output_dir / "decisions.csv", decision_rows)
    print(f"manifest_sha256: {manifest['manifest_sha256']}")
    print(f"runs_csv_written: {output_dir / 'runs.csv'}")
    print(f"waves_csv_written: {output_dir / 'waves.csv'}")
    print(f"decisions_csv_written: {output_dir / 'decisions.csv'}")


if __name__ == "__main__":
    main()
