"""State-gated hybrid between the RL actor and route-aware CP-SAT."""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.disruptions import PROFILE_SETTINGS
from solution.cp_sat_solver import CPSATSolver
from solution.rl_solver import RLSolver


class AdaptiveRLCPSolver:
    """Use RL in nominal flow and CP-SAT while disruptions are active."""

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        checkpoint: str = "",
        device: str = "cpu",
        deterministic: bool = True,
        max_time_seconds: float = 0.05,
    ):
        self.env = env
        self.actor = RLSolver(
            env,
            checkpoint=checkpoint,
            device=device,
            deterministic=deterministic,
        )
        self.cp_solver = CPSATSolver(
            env,
            max_time_seconds=max_time_seconds,
        )
        self.decision_records: List[Dict[str, Any]] = []

    def choose_action(self) -> Optional[Dict[str, Any]]:
        started = time.perf_counter()
        use_cp = self._repair_mode()
        solver = self.cp_solver if use_cp else self.actor
        action = solver.choose_action()
        self.decision_records.append(
            {
                "decision_index": len(self.decision_records),
                "simulation_time": float(self.env.time),
                "source": "cp_sat" if use_cp else "actor",
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
                "latency_ms": (
                    time.perf_counter() - started
                )
                * 1000.0,
            }
        )
        return action

    def get_decision_records(self) -> List[Dict[str, Any]]:
        return [dict(record) for record in self.decision_records]

    def get_telemetry(self) -> Dict[str, Any]:
        latencies = sorted(
            float(record["latency_ms"])
            for record in self.decision_records
        )
        return {
            "decisions": len(self.decision_records),
            "solve_calls": sum(
                record["source"] == "cp_sat"
                for record in self.decision_records
            ),
            "deadline_misses": 0,
            "fallbacks": 0,
            "cached_actions": 0,
            "latency_p50_ms": self._quantile(
                latencies,
                0.50,
            ),
            "latency_p95_ms": self._quantile(
                latencies,
                0.95,
            ),
            "latency_p99_ms": self._quantile(
                latencies,
                0.99,
            ),
            "latency_max_ms": max(latencies, default=0.0),
            "solve_latency_p50_ms": 0.0,
            "solve_latency_p95_ms": 0.0,
            "solve_latency_p99_ms": 0.0,
            "solve_latency_max_ms": 0.0,
            "mean_solve_ms": 0.0,
            "actor_decisions": sum(
                record["source"] == "actor"
                for record in self.decision_records
            ),
            "cp_decisions": sum(
                record["source"] == "cp_sat"
                for record in self.decision_records
            ),
            "mean_latency_ms": (
                statistics.mean(latencies)
                if latencies
                else 0.0
            ),
        }

    def _repair_mode(self) -> bool:
        disruptions = self.env.get_state()["disruptions"]
        lifecycle = disruptions["failure_lifecycle"]
        return bool(disruptions["active"]) or any(
            int(lifecycle[key]) > 0
            for key in (
                "blocked_slots",
                "under_repair",
                "replacements_in_transit",
            )
        )

    @staticmethod
    def _quantile(
        values: List[float],
        fraction: float,
    ) -> float:
        if not values:
            return 0.0
        index = min(
            len(values) - 1,
            max(
                0,
                int(round((len(values) - 1) * fraction)),
            ),
        )
        return float(values[index])


class RiskAwareAdaptiveRLCPSolver(AdaptiveRLCPSolver):
    """Select a conservative CP regime under high pressure or severe risk."""

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        checkpoint: str = "",
        device: str = "cpu",
        deterministic: bool = True,
        max_time_seconds: float = 0.05,
        pressure_interval_threshold: float = 47.5,
        elevated_risk_multiplier: float = 1.5,
        elevated_relief_interval: float = 57.5,
        severe_risk_multiplier: float = 2.0,
        severe_relief_interval: float = 67.5,
    ):
        super().__init__(
            env,
            checkpoint=checkpoint,
            device=device,
            deterministic=deterministic,
            max_time_seconds=max_time_seconds,
        )
        self.pressure_interval_threshold = float(
            pressure_interval_threshold
        )
        self.elevated_risk_multiplier = float(
            elevated_risk_multiplier
        )
        self.elevated_relief_interval = float(
            elevated_relief_interval
        )
        self.severe_risk_multiplier = float(
            severe_risk_multiplier
        )
        self.severe_relief_interval = float(
            severe_relief_interval
        )
        self.force_cp_regime = self._force_cp_regime()

    def _repair_mode(self) -> bool:
        return self.force_cp_regime or super()._repair_mode()

    def _force_cp_regime(self) -> bool:
        interval = float(self.env.config["wave_interval"])
        if interval <= self.pressure_interval_threshold:
            return True
        profile = str(
            self.env.config.get("disruption_profile", "none")
        )
        risk_multiplier = float(
            PROFILE_SETTINGS.get(profile, {}).get(
                "multiplier",
                1.0,
            )
        )
        if (
            risk_multiplier >= self.severe_risk_multiplier
            and interval <= self.severe_relief_interval
        ):
            return True
        return (
            risk_multiplier >= self.elevated_risk_multiplier
            and interval <= self.elevated_relief_interval
        )

    def get_telemetry(self) -> Dict[str, Any]:
        telemetry = super().get_telemetry()
        telemetry.update(
            {
                "meta_controller": "risk_aware",
                "force_cp_regime": self.force_cp_regime,
                "pressure_interval_threshold": (
                    self.pressure_interval_threshold
                ),
                "elevated_risk_multiplier": (
                    self.elevated_risk_multiplier
                ),
                "elevated_relief_interval": (
                    self.elevated_relief_interval
                ),
                "severe_risk_multiplier": (
                    self.severe_risk_multiplier
                ),
                "severe_relief_interval": (
                    self.severe_relief_interval
                ),
            }
        )
        return telemetry
