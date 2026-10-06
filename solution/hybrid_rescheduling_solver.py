"""RL-guided event-triggered rolling CP-SAT rescheduling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.repair_control import (
    LearnedRepairController,
    RuleBasedRepairController,
)
from solution.rl_solver import RLSolver
from solution.heuristic_solver import WaveHeuristicSolver
from solution.rolling_repair_solver import RollingRepairSolver


class HybridReschedulingSolver:
    """Use an actor for proposals and a controller for bounded CP repair."""

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        checkpoint: str = "",
        device: str = "cpu",
        control_checkpoint: str = "",
        control_device: str = "cpu",
        budget_ms: float = 50.0,
        neighborhood_size: int = 20,
        horizon_waves: int = 2,
        scope: str = "two_waves",
        deterministic: bool = True,
        fallback_policy: str = "actor",
        use_actor_guidance: bool = True,
        guidance_strength: float = 100.0,
    ):
        if fallback_policy not in {"actor", "heuristic"}:
            raise ValueError(
                f"unsupported hybrid fallback: {fallback_policy}"
            )
        self.env = env
        self.fallback_policy = fallback_policy
        self.use_actor_guidance = bool(use_actor_guidance)
        self.actor = RLSolver(
            env,
            checkpoint=checkpoint,
            device=device,
            deterministic=deterministic,
        )
        controller = (
            LearnedRepairController(
                control_checkpoint,
                device=control_device,
                max_budget_ms=budget_ms,
                max_neighborhood_size=neighborhood_size,
            )
            if control_checkpoint
            else RuleBasedRepairController(
                max_budget_ms=budget_ms,
                max_neighborhood_size=neighborhood_size,
            )
        )
        self.repair_solver = RollingRepairSolver(
            env,
            budget_ms=budget_ms,
            neighborhood_size=neighborhood_size,
            horizon_waves=horizon_waves,
            scope=scope,
            control_provider=controller,
            guidance_provider=(
                (lambda _: self.actor.action_pair_scores())
                if self.use_actor_guidance
                and guidance_strength > 0.0
                else None
            ),
            fallback_solver=(
                self.actor
                if fallback_policy == "actor"
                else WaveHeuristicSolver(env)
            ),
            guidance_strength=guidance_strength,
        )

    def choose_action(self) -> Optional[Dict[str, Any]]:
        return self.repair_solver.choose_action()

    def get_telemetry(self) -> Dict[str, Any]:
        telemetry = self.repair_solver.get_telemetry()
        telemetry["controller"] = (
            "learned"
            if isinstance(
                self.repair_solver.control_provider,
                LearnedRepairController,
            )
            else "rule"
        )
        telemetry["fallback_policy"] = self.fallback_policy
        telemetry["actor_guidance"] = self.use_actor_guidance
        return telemetry

    def get_decision_records(self) -> List[Dict[str, Any]]:
        return self.repair_solver.get_decision_records()


class RLGuidedRepairSolver(HybridReschedulingSolver):
    """Actor guidance with a matched heuristic fallback."""

    def __init__(self, env: CarrierAircraftSchedulingEnv, **kwargs):
        super().__init__(
            env,
            fallback_policy="heuristic",
            use_actor_guidance=True,
            **kwargs,
        )


class ActorFallbackRepairSolver(HybridReschedulingSolver):
    """Actor fallback without actor guidance in the repair objective."""

    def __init__(self, env: CarrierAircraftSchedulingEnv, **kwargs):
        super().__init__(
            env,
            fallback_policy="actor",
            use_actor_guidance=False,
            **kwargs,
        )
