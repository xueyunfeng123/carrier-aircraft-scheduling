"""Focused tests for policy-guided short-horizon search."""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from solution.rl_beam_search_solver import RLBeamSearchSolver


class _ToyEnvironment:
    def __init__(self) -> None:
        self.rng = random.Random(123)
        self.time = 0.0
        self.done = False
        self.pending_action = None
        self.aircraft = [
            SimpleNamespace(
                fuel_status=2,
                inspection_status=2,
                arm_status=2,
                is_airborne=False,
                recovery_status=2,
                sorties_completed=0,
            )
            for _ in range(2)
        ]

    def get_high_level_action_mask(self):
        return [1] if self.pending_action is None and not self.done else [0]

    def step(self, action):
        if action is not None:
            self.pending_action = action["aircraft_id"]
        else:
            self.time += 1.0
            if self.pending_action == 1:
                self.aircraft[1].sorties_completed += 1
            self.done = True
        return {}, 0.0, self.done, {"invalid_action": False}

    def get_evaluation_metrics(self):
        return {
            "total_sorties_completed": sum(
                item.sorties_completed
                for item in self.aircraft
            ),
            "total_missed_sorties": 0,
        }


class _ToyPolicy:
    def rank_actions(self, top_k, env=None):
        if env is None or not any(env.get_high_level_action_mask()):
            return []
        return [
            ({"high_level": 0, "aircraft_id": 0}, -0.1),
            ({"high_level": 0, "aircraft_id": 1}, -1.0),
        ][:top_k]


class _NonProgressEnvironment(_ToyEnvironment):
    def get_high_level_action_mask(self):
        return [1]

    def step(self, action):
        self.pending_action = action["aircraft_id"]
        return {}, 0.0, False, {"invalid_action": False}


class _ValueEnvironment(_ToyEnvironment):
    def step(self, action):
        if action is not None:
            self.pending_action = action["aircraft_id"]
        else:
            self.time += 1.0
        return {}, 0.0, False, {"invalid_action": False}


class _ValuePolicy(_ToyPolicy):
    def estimate_terminal_sorties(self, env):
        return 5.0 if env.pending_action == 1 else 0.0


class _ForbiddenLiveRandom:
    def __deepcopy__(self, memo):
        raise AssertionError("live RNG state must not be copied")


class RLBeamSearchTest(unittest.TestCase):
    def test_rollout_score_can_improve_on_greedy_policy(self) -> None:
        env = _ToyEnvironment()
        solver = RLBeamSearchSolver(
            env,
            beam_width=2,
            expansion_k=2,
            search_depth=1,
            rollout_events=1,
            policy=_ToyPolicy(),
        )

        action = solver.choose_action()

        self.assertEqual(action["aircraft_id"], 1)

    def test_planning_does_not_mutate_live_state_or_rng(self) -> None:
        env = _ToyEnvironment()
        rng_state = env.rng.getstate()
        solver = RLBeamSearchSolver(
            env,
            beam_width=2,
            expansion_k=2,
            search_depth=1,
            rollout_events=1,
            policy=_ToyPolicy(),
        )

        solver.choose_action()

        self.assertEqual(env.time, 0.0)
        self.assertIsNone(env.pending_action)
        self.assertEqual(env.rng.getstate(), rng_state)
        self.assertEqual(
            [item.sorties_completed for item in env.aircraft],
            [0, 0],
        )

    def test_planning_does_not_copy_live_rng_state(self) -> None:
        env = _ToyEnvironment()
        env.rng = _ForbiddenLiveRandom()
        solver = RLBeamSearchSolver(
            env,
            beam_width=2,
            expansion_k=2,
            search_depth=1,
            rollout_events=1,
            policy=_ToyPolicy(),
        )

        action = solver.choose_action()

        self.assertEqual(action["aircraft_id"], 1)

    def test_positive_search_limits_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "beam_width"):
            RLBeamSearchSolver(
                _ToyEnvironment(),
                beam_width=0,
                policy=_ToyPolicy(),
            )

    def test_calibrated_leaf_value_can_override_policy_rank(self) -> None:
        solver = RLBeamSearchSolver(
            _ValueEnvironment(),
            beam_width=2,
            expansion_k=2,
            search_depth=1,
            rollout_events=1,
            value_leaf_weight=1.0,
            policy=_ValuePolicy(),
        )

        action = solver.choose_action()

        self.assertEqual(action["aircraft_id"], 1)

    def test_value_guidance_requires_value_capable_policy(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "calibrated value policy",
        ):
            RLBeamSearchSolver(
                _ToyEnvironment(),
                value_leaf_weight=1.0,
                policy=_ToyPolicy(),
            )

    def test_risk_weight_must_be_a_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "risk_weight"):
            RLBeamSearchSolver(
                _ToyEnvironment(),
                risk_weight=1.1,
                policy=_ToyPolicy(),
            )

    def test_rollout_may_finish_exactly_at_action_limit(self) -> None:
        solver = RLBeamSearchSolver(
            _ToyEnvironment(),
            beam_width=2,
            expansion_k=2,
            search_depth=1,
            rollout_events=1,
            max_rollout_actions=1,
            policy=_ToyPolicy(),
        )

        action = solver.choose_action()

        self.assertEqual(action["aircraft_id"], 1)

    def test_rollout_raises_when_action_limit_prevents_progress(self) -> None:
        solver = RLBeamSearchSolver(
            _NonProgressEnvironment(),
            beam_width=2,
            expansion_k=2,
            search_depth=1,
            rollout_events=1,
            max_rollout_actions=1,
            policy=_ToyPolicy(),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "exceeded max_rollout_actions",
        ):
            solver.choose_action()


if __name__ == "__main__":
    unittest.main()
