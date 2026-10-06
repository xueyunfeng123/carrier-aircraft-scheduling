"""Tests for event-triggered repair control and the hybrid solver."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.repair_control import (
    CONTROL_FEATURE_DIM,
    RepairControlNet,
    RuleBasedRepairController,
    encode_repair_control_state,
)
from solution.hybrid_rescheduling_solver import (
    ActorFallbackRepairSolver,
    HybridReschedulingSolver,
    RLGuidedRepairSolver,
    StrongFallbackGuidedRepairSolver,
)
from solution.event_triggered_repair_solver import (
    EventTriggeredRepairSolver,
)
from solution.adaptive_rl_cp_solver import (
    AdaptiveRLCPSolver,
    RiskAwareAdaptiveRLCPSolver,
)


class RepairControlTest(unittest.TestCase):
    def test_control_features_hide_future_disruptions(self) -> None:
        base = self._env(
            disruption_profile="none",
            wave_interval=60.0,
            simulation_duration=720.0,
        )
        disrupted = self._env(
            disruption_profile="heavy",
            wave_interval=60.0,
            simulation_duration=720.0,
        )
        base.reset(seed=19)
        disrupted.reset(seed=19)

        self.assertEqual(
            encode_repair_control_state(base),
            encode_repair_control_state(disrupted),
        )
        self.assertEqual(
            len(encode_repair_control_state(base)),
            CONTROL_FEATURE_DIM,
        )

    def test_rule_controller_triggers_on_event_change(self) -> None:
        env = self._env(
            disruptions=[
                {
                    "kind": "runway_closure",
                    "target": 0,
                    "start_time": 1.0,
                    "end_time": 3.0,
                }
            ]
        )
        controller = RuleBasedRepairController()

        initial = controller(env)
        repeated = controller(env)
        env._advance_time_to_next_event()
        disrupted = controller(env)

        self.assertTrue(initial.trigger)
        self.assertFalse(repeated.trigger)
        self.assertTrue(disrupted.trigger)
        self.assertGreater(
            disrupted.budget_ms,
            initial.budget_ms,
        )

    def test_rule_controller_respects_experiment_caps(self) -> None:
        env = self._env(
            disruptions=[
                {
                    "kind": "runway_closure",
                    "target": 0,
                    "start_time": 1.0,
                    "end_time": 3.0,
                }
            ]
        )
        controller = RuleBasedRepairController(
            max_budget_ms=10.0,
            max_neighborhood_size=8,
        )
        controller(env)
        env._advance_time_to_next_event()

        disrupted = controller(env)

        self.assertEqual(disrupted.budget_ms, 10.0)
        self.assertEqual(disrupted.neighborhood_size, 8)

    def test_rule_controller_can_skip_extreme_disruptions(
        self,
    ) -> None:
        env = self._env(
            disruptions=[
                {
                    "kind": "service_slowdown",
                    "target": "all",
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "multiplier": 2.0,
                }
            ]
        )
        controller = RuleBasedRepairController(
            max_trigger_severity=0.375,
        )
        controller(env)
        env._advance_time_to_next_event()

        disrupted = controller(env)

        self.assertFalse(disrupted.trigger)
        self.assertEqual(disrupted.scope, "affected")

    def test_control_network_heads_have_valid_shapes(self) -> None:
        import torch

        model = RepairControlNet(hidden_dim=16)
        outputs = model(
            torch.zeros((2, CONTROL_FEATURE_DIM))
        )

        self.assertEqual(
            [tuple(output.shape) for output in outputs],
            [(2, 2), (2, 3), (2, 3), (2, 3)],
        )

    def test_hybrid_solver_returns_legal_action(self) -> None:
        env = self._env()
        for aircraft in env.aircraft:
            aircraft.launch_status = 0
        env.aircraft[0].inspection_status = 0
        env._invalidate_planning_cache()
        solver = HybridReschedulingSolver(
            env,
            checkpoint="",
            device="cpu",
            budget_ms=200.0,
            neighborhood_size=4,
        )

        action = solver.choose_action()

        self.assertIsNotNone(action)
        parsed = env._parse_action(action)
        self.assertIsNotNone(parsed)
        self.assertTrue(env._is_action_valid(*parsed))
        self.assertEqual(
            solver.get_telemetry()["controller"],
            "rule",
        )

    def test_non_learning_event_repair_returns_legal_action(
        self,
    ) -> None:
        env = self._env()
        for aircraft in env.aircraft:
            aircraft.launch_status = 0
        env.aircraft[0].inspection_status = 0
        env._invalidate_planning_cache()
        solver = EventTriggeredRepairSolver(
            env,
            budget_ms=200.0,
            neighborhood_size=4,
        )

        action = solver.choose_action()

        self.assertIsNotNone(action)
        parsed = env._parse_action(action)
        self.assertIsNotNone(parsed)
        self.assertTrue(env._is_action_valid(*parsed))
        self.assertEqual(
            solver.get_telemetry()["solve_calls"],
            1,
        )

    def test_hybrid_ablation_variants_separate_guidance_and_fallback(
        self,
    ) -> None:
        variants = (
            (
                RLGuidedRepairSolver,
                "heuristic",
                True,
            ),
            (
                ActorFallbackRepairSolver,
                "actor",
                False,
            ),
            (
                StrongFallbackGuidedRepairSolver,
                "cp_sat",
                True,
            ),
        )
        for solver_class, fallback, guidance in variants:
            with self.subTest(solver=solver_class.__name__):
                env = self._env()
                solver = solver_class(
                    env,
                    checkpoint="",
                    budget_ms=10.0,
                    neighborhood_size=4,
                )

                action = solver.choose_action()
                telemetry = solver.get_telemetry()

                self.assertIsNotNone(action)
                self.assertTrue(
                    env._is_action_valid(*env._parse_action(action))
                )
                self.assertEqual(
                    telemetry["fallback_policy"],
                    fallback,
                )
                self.assertEqual(
                    telemetry["actor_guidance"],
                    guidance,
                )

    def test_adaptive_hybrid_switches_to_cp_during_disruption(
        self,
    ) -> None:
        env = self._env(
            disruptions=[
                {
                    "kind": "service_slowdown",
                    "target": "all",
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "multiplier": 1.5,
                }
            ]
        )
        solver = AdaptiveRLCPSolver(
            env,
            checkpoint="",
            max_time_seconds=0.01,
        )

        first = solver.choose_action()
        env.step(first)
        if env.time < 1.0:
            env._advance_time_to_next_event()
        second = solver.choose_action()

        self.assertIsNotNone(second)
        records = solver.get_decision_records()
        self.assertEqual(records[0]["source"], "actor")
        self.assertEqual(records[-1]["source"], "cp_sat")

    def test_risk_aware_hybrid_forces_cp_under_pressure(
        self,
    ) -> None:
        env = self._env(
            disruption_profile="none",
            wave_interval=47.5,
            simulation_duration=95.0,
        )
        solver = RiskAwareAdaptiveRLCPSolver(
            env,
            checkpoint="",
            max_time_seconds=0.01,
        )

        action = solver.choose_action()

        self.assertIsNotNone(action)
        self.assertTrue(solver.force_cp_regime)
        self.assertEqual(
            solver.get_decision_records()[0]["source"],
            "cp_sat",
        )

    def test_risk_aware_hybrid_preserves_actor_with_relief(
        self,
    ) -> None:
        env = self._env(
            disruption_profile="none",
            wave_interval=67.5,
            simulation_duration=135.0,
        )
        env.config["disruption_profile"] = "compound_heavy"
        solver = RiskAwareAdaptiveRLCPSolver(
            env,
            checkpoint="",
            max_time_seconds=0.01,
        )

        action = solver.choose_action()

        self.assertIsNotNone(action)
        self.assertFalse(solver.force_cp_regime)
        self.assertEqual(
            solver.get_decision_records()[0]["source"],
            "actor",
        )

    def _env(self, **overrides) -> CarrierAircraftSchedulingEnv:
        config = {
            "num_aircraft": 4,
            "num_total_aircraft": 6,
            "group_size": 2,
            "num_parking_spots": 4,
            "num_launch_positions": 2,
            "num_launch_channels": 2,
            "spatial_graph_enabled": False,
            "wave_interval": 20.0,
            "simulation_duration": 40.0,
        }
        config.update(overrides)
        env = CarrierAircraftSchedulingEnv(config)
        env.reset(seed=7)
        return env


if __name__ == "__main__":
    unittest.main()
