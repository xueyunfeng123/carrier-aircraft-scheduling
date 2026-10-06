"""Online rolling CP-SAT repair solver with incumbent-safe fallbacks."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from solution.cp_sat_repair_model import (
    ActionKey,
    CPSATRepairModel,
    RepairControl,
    RepairPlan,
)
from solution.heuristic_solver import WaveHeuristicSolver
from solution.priority_rule_solver import (
    ACTION_LAUNCH,
    ACTION_RECOVERY,
)


ControlProvider = Callable[
    [CarrierAircraftSchedulingEnv],
    RepairControl,
]
GuidanceProvider = Callable[
    [CarrierAircraftSchedulingEnv],
    Dict[ActionKey, float],
]


@dataclass
class RepairTelemetry:
    decisions: int = 0
    solve_calls: int = 0
    deadline_misses: int = 0
    fallbacks: int = 0
    cached_actions: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    solve_latencies_ms: List[float] = field(default_factory=list)
    decision_records: List[Dict[str, Any]] = field(
        default_factory=list
    )

    def as_dict(self) -> Dict[str, Any]:
        ordered = sorted(self.latencies_ms)
        ordered_solve = sorted(self.solve_latencies_ms)
        return {
            "decisions": self.decisions,
            "solve_calls": self.solve_calls,
            "deadline_misses": self.deadline_misses,
            "fallbacks": self.fallbacks,
            "cached_actions": self.cached_actions,
            "latency_p50_ms": self._quantile(ordered, 0.50),
            "latency_p95_ms": self._quantile(ordered, 0.95),
            "latency_p99_ms": self._quantile(ordered, 0.99),
            "latency_max_ms": max(ordered, default=0.0),
            "solve_latency_p50_ms": self._quantile(
                ordered_solve,
                0.50,
            ),
            "solve_latency_p95_ms": self._quantile(
                ordered_solve,
                0.95,
            ),
            "solve_latency_p99_ms": self._quantile(
                ordered_solve,
                0.99,
            ),
            "solve_latency_max_ms": max(
                ordered_solve,
                default=0.0,
            ),
            "mean_solve_ms": (
                statistics.mean(self.solve_latencies_ms)
                if self.solve_latencies_ms
                else 0.0
            ),
        }

    @staticmethod
    def _quantile(values: List[float], fraction: float) -> float:
        if not values:
            return 0.0
        index = min(
            len(values) - 1,
            max(0, int(round((len(values) - 1) * fraction))),
        )
        return float(values[index])


class RollingRepairSolver:
    """Replans service dispatches and commits one revalidated action at a time."""

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        budget_ms: float = 50.0,
        neighborhood_size: int = 20,
        horizon_waves: int = 2,
        scope: str = "two_waves",
        control_provider: Optional[ControlProvider] = None,
        guidance_provider: Optional[GuidanceProvider] = None,
        fallback_solver: Optional[Any] = None,
        guidance_strength: float = 100.0,
    ):
        self.env = env
        self.default_control = RepairControl(
            budget_ms=float(budget_ms),
            neighborhood_size=int(neighborhood_size),
            horizon_waves=int(horizon_waves),
            scope=scope,
        )
        self.control_provider = control_provider
        self.guidance_provider = guidance_provider
        self.fallback_solver = (
            fallback_solver
            if fallback_solver is not None
            else WaveHeuristicSolver(env)
        )
        self.repair_model = CPSATRepairModel(
            env,
            guidance_strength=guidance_strength,
        )
        self.telemetry = RepairTelemetry()
        self.last_plan: Optional[RepairPlan] = None
        self.incumbent_starts: Dict[ActionKey, int] = {}
        self.cached_actions: List[ActionKey] = []
        self.cached_time: Optional[float] = None
        self.control_signature: Optional[tuple[Any, ...]] = None
        self.pending_control: Optional[RepairControl] = None

    def choose_action(self) -> Optional[Dict[str, Any]]:
        started = time.perf_counter()
        self.telemetry.decisions += 1
        decision = self._decision_context()

        cached = self._choose_cached_action()
        if cached is not None:
            self.telemetry.cached_actions += 1
            self._record_latency(
                started,
                decision,
                source="cache",
                action=cached,
            )
            return cached

        self._refresh_pending_control()
        fallback = self.fallback_solver.choose_action()
        if fallback is None:
            self._record_latency(
                started,
                decision,
                source="none",
            )
            return None
        if int(fallback["high_level"]) in (
            ACTION_RECOVERY,
            ACTION_LAUNCH,
        ):
            self._record_latency(
                started,
                decision,
                source="priority",
                action=fallback,
            )
            return fallback

        if self.control_provider is not None:
            control = self.pending_control or RepairControl(
                trigger=False,
                scope=self.default_control.scope,
                budget_ms=self.default_control.budget_ms,
                neighborhood_size=(
                    self.default_control.neighborhood_size
                ),
                horizon_waves=self.default_control.horizon_waves,
            )
            self.pending_control = None
        else:
            control = self.default_control
        if not control.trigger or control.budget_ms <= 0.0:
            self.telemetry.fallbacks += 1
            self._record_latency(
                started,
                decision,
                source="fallback",
                action=fallback,
                control=control,
                plan_status="skipped",
            )
            return fallback

        guidance = (
            self.guidance_provider(self.env)
            if self.guidance_provider is not None
            else {}
        )
        self.telemetry.solve_calls += 1
        plan = self.repair_model.solve(
            control,
            incumbent_starts=self.incumbent_starts,
            guidance_scores=guidance,
        )
        plan.deadline_missed = (
            plan.deadline_missed
            or (time.perf_counter() - started) * 1000.0
            > control.budget_ms
        )
        self.last_plan = plan
        self.telemetry.solve_latencies_ms.append(plan.solve_ms)
        if plan.deadline_missed:
            self.telemetry.deadline_misses += 1
        usable = (
            not plan.deadline_missed
            and plan.status in {"optimal", "feasible"}
            and bool(plan.actions)
        )
        if not usable:
            self.telemetry.fallbacks += 1
            self._record_latency(
                started,
                decision,
                source="fallback",
                action=fallback,
                control=control,
                plan=plan,
            )
            return fallback

        self.incumbent_starts = dict(plan.start_ticks)
        ordered = (
            plan.immediate_actions
            if plan.immediate_actions
            else plan.actions[:1]
        )
        self.cached_actions = list(ordered)
        self.cached_time = self.env.time
        selected = self._choose_cached_action()
        if selected is None:
            self.telemetry.fallbacks += 1
            selected = fallback
            source = "fallback"
        else:
            source = "repair"
        self._record_latency(
            started,
            decision,
            source=source,
            action=selected,
            control=control,
            plan=plan,
        )
        return selected

    def get_telemetry(self) -> Dict[str, Any]:
        return self.telemetry.as_dict()

    def get_decision_records(self) -> List[Dict[str, Any]]:
        return [
            dict(record)
            for record in self.telemetry.decision_records
        ]

    def _choose_cached_action(self) -> Optional[Dict[str, Any]]:
        if self.cached_time != self.env.time:
            self.cached_actions.clear()
            self.cached_time = None
            return None
        while self.cached_actions:
            high_level, aircraft_id = self.cached_actions.pop(0)
            action = self.env.complete_action(
                {
                    "high_level": high_level,
                    "aircraft_id": aircraft_id,
                }
            )
            parsed = self.env._parse_action(action)
            if parsed is not None and self.env._is_action_valid(*parsed):
                return action
        return None

    def _decision_context(self) -> Dict[str, Any]:
        from rl.repair_control import (
            CONTROL_FEATURE_NAMES,
            encode_repair_control_state,
        )

        return {
            "decision_index": self.telemetry.decisions - 1,
            "simulation_time": float(self.env.time),
            **dict(
                zip(
                    CONTROL_FEATURE_NAMES,
                    encode_repair_control_state(self.env),
                )
            ),
        }

    def _refresh_pending_control(self) -> None:
        if self.control_provider is None:
            return
        from rl.repair_control import repair_event_signature

        signature = repair_event_signature(self.env)
        if signature == self.control_signature:
            return
        self.control_signature = signature
        self.pending_control = self.control_provider(self.env)

    def _record_latency(
        self,
        started: float,
        decision: Dict[str, Any],
        source: str,
        action: Optional[Dict[str, Any]] = None,
        control: Optional[RepairControl] = None,
        plan: Optional[RepairPlan] = None,
        plan_status: str = "",
    ) -> None:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.telemetry.latencies_ms.append(latency_ms)
        record = {
            **decision,
            "source": source,
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
            "trigger": (
                int(control.trigger)
                if control is not None
                else ""
            ),
            "scope": control.scope if control is not None else "",
            "budget_ms": (
                float(control.budget_ms)
                if control is not None
                else 0.0
            ),
            "neighborhood_size": (
                int(control.neighborhood_size)
                if control is not None
                else 0
            ),
            "plan_status": (
                plan.status
                if plan is not None
                else plan_status
            ),
            "solve_ms": (
                float(plan.solve_ms)
                if plan is not None
                else 0.0
            ),
            "deadline_missed": (
                int(plan.deadline_missed)
                if plan is not None
                else 0
            ),
            "latency_ms": latency_ms,
        }
        self.telemetry.decision_records.append(record)
