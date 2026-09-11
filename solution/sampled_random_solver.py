"""Sample-based planner using best-of-N random complete rollouts."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from solution.random_solver import RandomSolver


class SampledRandomSolver:
    """Plan once by selecting the best complete random rollout.

    This solver is intentionally stronger than ``RandomSolver``: it samples
    multiple legal random action sequences inside copied environments, scores
    complete episodes, then replays the best action sequence in the real env.
    """

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        seed: int = 7,
        samples: int = 30,
        max_steps: int = 100000,
    ):
        self.env = env
        self.seed = seed
        self.samples = samples
        self.max_steps = max_steps
        self._planned_actions: Optional[List[Optional[Dict[str, Any]]]] = None
        self._cursor = 0

    def choose_action(self) -> Optional[Dict[str, Any]]:
        if self._planned_actions is None:
            self._planned_actions = self._build_plan()
            self._cursor = 0

        if self._cursor >= len(self._planned_actions):
            return None

        action = self._planned_actions[self._cursor]
        self._cursor += 1
        if action is not None and not self._is_action_valid(action):
            return None
        return action

    def _build_plan(self) -> List[Optional[Dict[str, Any]]]:
        best_score = None
        best_actions: List[Optional[Dict[str, Any]]] = []

        for sample_id in range(self.samples):
            env = copy.deepcopy(self.env)
            solver = RandomSolver(env, seed=self.seed + sample_id)
            actions: List[Optional[Dict[str, Any]]] = []
            steps = 0

            while not env.done and steps < self.max_steps:
                action = solver.choose_action()
                actions.append(copy.deepcopy(action))
                env.step(action)
                steps += 1

            score = self._score(env)
            if best_score is None or score > best_score:
                best_score = score
                best_actions = actions

        return best_actions

    def _score(self, env: CarrierAircraftSchedulingEnv) -> Tuple[int, int, float]:
        metrics = env.get_evaluation_metrics()
        return (
            metrics["total_sorties_completed"],
            -metrics["total_missed_sorties"],
            -env.time,
        )

    def _is_action_valid(self, action: Dict[str, Any]) -> bool:
        parsed = self.env._parse_action(action)
        return parsed is not None and self.env._is_action_valid(*parsed)
