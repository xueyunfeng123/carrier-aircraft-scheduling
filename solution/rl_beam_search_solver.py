"""Policy-guided short-horizon beam search for inference."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.scoring import readiness_potential
from solution.rl_solver import RLSolver


class RankedPolicy(Protocol):
    def rank_actions(
        self,
        top_k: int,
        env: Optional[CarrierAircraftSchedulingEnv] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        ...


@dataclass
class _BeamNode:
    env: CarrierAircraftSchedulingEnv
    first_action: Dict[str, Any]
    policy_log_probability: float
    score: Tuple[float, ...]


class RLBeamSearchSolver:
    """Improve a checkpoint policy with simulation-guided beam search.

    Search clones the current observable simulator state, replaces its random
    generator before any candidate action is executed, and uses only normal
    ``env.step`` transitions. The live environment and its RNG are never
    advanced by planning. The checkpoint value head is deliberately unused.
    """

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        checkpoint: str = "",
        device: str = "cpu",
        deterministic: bool = True,
        beam_width: int = 2,
        expansion_k: int = 2,
        search_depth: int = 2,
        rollout_events: int = 2,
        rollout_samples: int = 1,
        search_seed: int = 17,
        max_rollout_actions: int = 1000,
        value_leaf_weight: float = 0.0,
        risk_weight: float = 0.0,
        risk_alpha: float = 0.25,
        policy: Optional[RankedPolicy] = None,
        **rl_options: Any,
    ):
        for name, value in (
            ("beam_width", beam_width),
            ("expansion_k", expansion_k),
            ("search_depth", search_depth),
            ("rollout_events", rollout_events),
            ("rollout_samples", rollout_samples),
            ("max_rollout_actions", max_rollout_actions),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if value_leaf_weight < 0.0:
            raise ValueError("value_leaf_weight must be non-negative")
        if not 0.0 <= risk_weight <= 1.0:
            raise ValueError("risk_weight must be in [0, 1]")
        if not 0.0 < risk_alpha <= 1.0:
            raise ValueError("risk_alpha must be in (0, 1]")

        self.env = env
        self.beam_width = int(beam_width)
        self.expansion_k = int(expansion_k)
        self.search_depth = int(search_depth)
        self.rollout_events = int(rollout_events)
        self.rollout_samples = int(rollout_samples)
        self.search_seed = int(search_seed)
        self.max_rollout_actions = int(max_rollout_actions)
        self.value_leaf_weight = float(value_leaf_weight)
        self.risk_weight = float(risk_weight)
        self.risk_alpha = float(risk_alpha)
        self.policy = policy or RLSolver(
            env,
            checkpoint=checkpoint,
            device=device,
            deterministic=deterministic,
            require_calibrated_value=self.value_leaf_weight > 0.0,
            **rl_options,
        )
        if self.value_leaf_weight > 0.0 and not callable(
            getattr(self.policy, "estimate_terminal_sorties", None)
        ):
            raise ValueError(
                "value-guided search requires a calibrated value policy"
            )
        self.decision_index = 0

    def choose_action(self) -> Optional[Dict[str, Any]]:
        if not any(self.env.get_high_level_action_mask()):
            return None

        planning_env = self._clone_without_live_rng(
            self.env,
            self._forecast_seed(0),
        )
        beam: List[_BeamNode] = []

        for depth in range(self.search_depth):
            expanded: List[_BeamNode] = []
            parents = beam or [None]
            for parent in parents:
                parent_env = (
                    planning_env
                    if parent is None
                    else parent.env
                )
                if parent_env.done:
                    if parent is not None:
                        expanded.append(parent)
                    continue
                ranked_actions = self.policy.rank_actions(
                    self.expansion_k,
                    parent_env,
                )
                if not ranked_actions:
                    if parent is not None:
                        expanded.append(parent)
                    continue
                for action, log_probability in ranked_actions:
                    child_env = copy.deepcopy(parent_env)
                    _, _, _, info = child_env.step(action)
                    if info["invalid_action"]:
                        raise RuntimeError(
                            "RL beam policy emitted an illegal action"
                        )
                    first_action = (
                        dict(action)
                        if parent is None
                        else parent.first_action
                    )
                    cumulative_log_probability = (
                        log_probability
                        if parent is None
                        else (
                            parent.policy_log_probability
                            + log_probability
                        )
                    )
                    expanded.append(
                        _BeamNode(
                            env=child_env,
                            first_action=first_action,
                            policy_log_probability=(
                                cumulative_log_probability
                            ),
                            score=self._score_node(
                                child_env,
                                cumulative_log_probability,
                                depth,
                            ),
                        )
                    )
            if not expanded:
                break
            expanded.sort(
                key=lambda node: (
                    node.score,
                    self._action_tiebreak(node.first_action),
                ),
                reverse=True,
            )
            beam = expanded[: self.beam_width]

        self.decision_index += 1
        if not beam:
            return None
        return dict(beam[0].first_action)

    def _score_node(
        self,
        env: CarrierAircraftSchedulingEnv,
        policy_log_probability: float,
        depth: int,
    ) -> Tuple[float, ...]:
        scores = []
        terminal_estimates = []
        for sample_index in range(self.rollout_samples):
            rollout_env = copy.deepcopy(env)
            rollout_env.rng = random.Random(
                self._forecast_seed(
                    1
                    + depth * self.rollout_samples
                    + sample_index
                )
            )
            self._greedy_rollout(rollout_env)
            metrics = rollout_env.get_evaluation_metrics()
            completed = float(metrics["total_sorties_completed"])
            scores.append(
                (
                    completed,
                    readiness_potential(rollout_env),
                    -float(metrics["total_missed_sorties"]),
                )
            )
            terminal_estimate = completed
            if self.value_leaf_weight > 0.0:
                terminal_estimate = float(
                    self.policy.estimate_terminal_sorties(rollout_env)
                )
            terminal_estimates.append(
                completed
                + self.value_leaf_weight
                * (terminal_estimate - completed)
            )
        sample_count = float(len(scores))
        mean_terminal = sum(terminal_estimates) / sample_count
        lower_count = max(
            1,
            int(math.ceil(self.risk_alpha * len(terminal_estimates))),
        )
        lower_cvar = (
            sum(sorted(terminal_estimates)[:lower_count])
            / float(lower_count)
        )
        robust_terminal = (
            (1.0 - self.risk_weight) * mean_terminal
            + self.risk_weight * lower_cvar
        )
        return (
            robust_terminal,
            sum(score[0] for score in scores) / sample_count,
            sum(score[1] for score in scores) / sample_count,
            sum(score[2] for score in scores) / sample_count,
            policy_log_probability,
        )

    def _greedy_rollout(
        self,
        env: CarrierAircraftSchedulingEnv,
    ) -> None:
        events_advanced = 0
        actions = 0
        while (
            not env.done
            and events_advanced < self.rollout_events
            and actions < self.max_rollout_actions
        ):
            ranked = self.policy.rank_actions(1, env)
            action = ranked[0][0] if ranked else None
            previous_time = env.time
            _, _, _, info = env.step(action)
            if info["invalid_action"]:
                raise RuntimeError(
                    "RL rollout policy emitted an illegal action"
                )
            actions += 1
            if env.time > previous_time:
                events_advanced += 1
        if (
            not env.done
            and events_advanced < self.rollout_events
            and actions >= self.max_rollout_actions
        ):
            raise RuntimeError(
                "RL beam rollout exceeded max_rollout_actions"
            )

    def _forecast_seed(self, offset: int) -> int:
        return (
            self.search_seed
            + self.decision_index * 1_000_003
            + offset * 10_007
        )

    @staticmethod
    def _clone_without_live_rng(
        env: CarrierAircraftSchedulingEnv,
        forecast_seed: int,
    ) -> CarrierAircraftSchedulingEnv:
        forecast_rng = random.Random(forecast_seed)
        return copy.deepcopy(
            env,
            {id(env.rng): forecast_rng},
        )

    @staticmethod
    def _action_tiebreak(action: Dict[str, Any]) -> Tuple[str, ...]:
        return tuple(
            f"{key}={action[key]}"
            for key in sorted(action)
        )
