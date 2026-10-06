"""Non-learning event-triggered CP-SAT repair baseline."""

from __future__ import annotations

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.repair_control import RuleBasedRepairController
from solution.heuristic_solver import WaveHeuristicSolver
from solution.rolling_repair_solver import RollingRepairSolver


class EventTriggeredRepairSolver(RollingRepairSolver):
    """Use the same event trigger as the hybrid method without RL guidance."""

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        budget_ms: float = 50.0,
        neighborhood_size: int = 20,
        horizon_waves: int = 2,
        scope: str = "two_waves",
    ):
        controller = RuleBasedRepairController(
            max_budget_ms=budget_ms,
            max_neighborhood_size=neighborhood_size,
        )
        super().__init__(
            env,
            budget_ms=budget_ms,
            neighborhood_size=neighborhood_size,
            horizon_waves=horizon_waves,
            scope=scope,
            control_provider=controller,
            fallback_solver=WaveHeuristicSolver(env),
        )
