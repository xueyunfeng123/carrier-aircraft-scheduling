"""RL-guided event-triggered rolling CP-SAT rescheduling."""

from __future__ import annotations

from typing import Any, Dict, Optional

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.repair_control import (
    LearnedRepairController,
    RuleBasedRepairController,
)
from solution.rl_solver import RLSolver
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
    ):
        self.env = env
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
            )
            if control_checkpoint
            else RuleBasedRepairController()
        )
        self.repair_solver = RollingRepairSolver(
            env,
            budget_ms=budget_ms,
            neighborhood_size=neighborhood_size,
            horizon_waves=horizon_waves,
            scope=scope,
            control_provider=controller,
            guidance_provider=lambda _: (
                self.actor.action_pair_scores()
            ),
            fallback_solver=self.actor,
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
        return telemetry
