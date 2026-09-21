"""Tests for offline critic calibration and one-step value reranking."""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.checkpoint import save_checkpoint
from rl.model import CarrierPolicyValueNet
from rl.obs_encoder import (
    AIRCRAFT_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    OBSERVATION_SCHEMA_VERSION,
    encode_observation,
)
from rl.ppo_trainer import PPOTrainer
from rl.train_config import PPOConfig
from rl.value_calibration import (
    ValueSample,
    calibrate_value_head,
    calibration_metric_rows,
    clone_environment_for_counterfactual,
    collect_complete_trajectory,
    collect_counterfactual_value_samples,
    compute_value_targets,
    trajectories_to_samples,
    validate_development_seeds,
    validate_rerank_calibration,
)
from scripts.calibrate_value import (
    DEFAULT_SELECTION_SEEDS,
    validate_seed_protocol,
)
from scripts.evaluate_value import (
    DEFAULT_EVALUATION_SEEDS,
    validate_evaluation_seeds,
)
from solution.rl_solver import RLSolver


def small_env_config():
    return {
        "num_aircraft": 2,
        "group_size": 1,
        "num_parking_spots": 2,
        "launch_time": 0.5,
        "wave_interval": 60.0,
        "simulation_duration": 0.5,
        "spatial_graph_enabled": False,
    }


def counterfactual_env_config():
    return {
        "num_aircraft": 2,
        "group_size": 1,
        "num_parking_spots": 2,
        "launch_time": 0.5,
        "wave_interval": 10.0,
        "simulation_duration": 20.0,
        "spatial_graph_enabled": False,
    }


class ValueTargetTest(unittest.TestCase):
    def test_mc_and_terminal_gae_lambda_one_match(self) -> None:
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, -0.25, 4.0]

        monte_carlo = compute_value_targets(
            rewards,
            values,
            gamma=0.9,
            method="mc",
        )
        gae = compute_value_targets(
            rewards,
            values,
            gamma=0.9,
            method="gae",
            gae_lambda=1.0,
        )

        self.assertEqual(monte_carlo, [5.23, 4.7, 3.0])
        for actual, expected in zip(gae, monte_carlo):
            self.assertAlmostEqual(actual, expected)

    def test_target_validation_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            compute_value_targets([1.0], [], gamma=1.0)
        with self.assertRaisesRegex(ValueError, "target method"):
            compute_value_targets([], [], gamma=1.0, method="td")

    def test_complete_trajectory_reaches_terminal_return(self) -> None:
        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
        )

        trajectory = collect_complete_trajectory(
            model,
            small_env_config(),
            seed=7,
            device="cpu",
            reward_config=PPOConfig(),
        )
        samples = trajectories_to_samples(
            [trajectory],
            gamma=1.0,
            method="mc",
            gae_lambda=0.98,
        )

        self.assertEqual(len(trajectory.observations), 1)
        self.assertEqual(trajectory.rewards, [1.0])
        self.assertEqual([sample.target for sample in samples], [1.0])


class CounterfactualCollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
            hidden_dim=16,
            aircraft_embed_dim=8,
            target_embed_dim=8,
        )

    def test_collects_top_k_successors_with_common_random_numbers(
        self,
    ) -> None:
        samples = collect_counterfactual_value_samples(
            self.model,
            [counterfactual_env_config()],
            [42001],
            device="cpu",
            reward_config=PPOConfig(),
            stride=1,
            max_branch_states=1,
            top_k=2,
            workers=1,
            branch_seed=99,
            max_steps=1000,
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(
            [sample.candidate_rank for sample in samples],
            [0, 1],
        )
        self.assertEqual(
            len({sample.action_key for sample in samples}),
            2,
        )
        self.assertEqual(
            len({sample.branch_seed for sample in samples}),
            1,
        )
        self.assertEqual(
            [sample.target for sample in samples],
            [1.0, 1.0],
        )
        self.assertTrue(
            all(
                sample.source_decision_index == 0
                for sample in samples
            )
        )

    def test_stride_limit_and_workers_are_deterministic(self) -> None:
        options = {
            "model": self.model,
            "configs": [counterfactual_env_config()],
            "seeds": [42001],
            "device": "cpu",
            "reward_config": PPOConfig(),
            "stride": 2,
            "max_branch_states": 2,
            "top_k": 2,
            "branch_seed": 99,
            "max_steps": 1000,
        }

        serial = collect_counterfactual_value_samples(
            **options,
            workers=1,
        )
        parallel = collect_counterfactual_value_samples(
            **options,
            workers=2,
        )

        serial_rows = [
            (
                sample.source_decision_index,
                sample.candidate_rank,
                sample.action_key,
                sample.branch_seed,
                sample.target,
            )
            for sample in serial
        ]
        parallel_rows = [
            (
                sample.source_decision_index,
                sample.candidate_rank,
                sample.action_key,
                sample.branch_seed,
                sample.target,
            )
            for sample in parallel
        ]
        self.assertEqual(serial_rows, parallel_rows)
        self.assertEqual(
            sorted({sample.source_decision_index for sample in serial}),
            [0, 2],
        )

    def test_clone_replaces_live_rng_without_copying_its_state(self) -> None:
        class ForbiddenLiveRandom:
            def __deepcopy__(self, memo):
                raise AssertionError("live RNG state must not be copied")

        env = SimpleNamespace(rng=ForbiddenLiveRandom(), marker=[1])

        first = clone_environment_for_counterfactual(env, 123)
        second = clone_environment_for_counterfactual(env, 123)

        self.assertIs(first.rng.__class__, random.Random)
        self.assertIs(second.rng.__class__, random.Random)
        self.assertEqual(first.rng.random(), second.rng.random())
        self.assertIsNot(first.rng, second.rng)
        self.assertIs(env.rng.__class__, ForbiddenLiveRandom)

    def test_collection_controls_require_positive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "stride must be positive"):
            collect_counterfactual_value_samples(
                self.model,
                [counterfactual_env_config()],
                [42001],
                device="cpu",
                reward_config=PPOConfig(),
                stride=0,
            )


class CriticFitTest(unittest.TestCase):
    def test_calibration_updates_only_value_head(self) -> None:
        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
            hidden_dim=16,
            aircraft_embed_dim=8,
            target_embed_dim=8,
        )
        env = CarrierAircraftSchedulingEnv(small_env_config())
        env.reset(seed=7)
        base_observation = encode_observation(env)
        samples = []
        for index in range(6):
            observation = copy.deepcopy(base_observation)
            observation.global_features[0] = index / 5.0
            samples.append(
                ValueSample(
                    observation=observation,
                    target=float(index),
                    seed=7,
                    wave_interval=60.0,
                    time_fraction=index / 5.0,
                )
            )
        actor_before = {
            name: value.clone()
            for name, value in model.state_dict().items()
            if not name.startswith("value_head.")
        }
        value_before = {
            name: value.clone()
            for name, value in model.value_head.state_dict().items()
        }

        stats = calibrate_value_head(
            model,
            samples,
            device="cpu",
            learning_rate=0.05,
            epochs=100,
            minibatch_size=6,
            loss_name="mse",
            seed=11,
        )

        self.assertLess(stats["final_mse"], stats["initial_mse"])
        self.assertTrue(
            any(
                not torch.equal(value, model.value_head.state_dict()[name])
                for name, value in value_before.items()
            )
        )
        for name, value in actor_before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))

    def test_metrics_include_load_and_horizon_groups(self) -> None:
        env = CarrierAircraftSchedulingEnv(small_env_config())
        env.reset(seed=7)
        observation = encode_observation(env)
        samples = [
            ValueSample(
                observation=observation,
                target=float(index),
                seed=index,
                wave_interval=interval,
                time_fraction=time_fraction,
            )
            for index, (interval, time_fraction) in enumerate(
                [
                    (50.0, 0.1),
                    (50.0, 0.5),
                    (60.0, 0.7),
                    (60.0, 0.9),
                ]
            )
        ]

        rows = calibration_metric_rows(
            samples,
            [sample.target for sample in samples],
        )
        groups = {
            (row["group_type"], row["group"])
            for row in rows
        }
        overall = next(
            row
            for row in rows
            if row["group_type"] == "overall"
        )

        self.assertEqual(overall["mae"], 0.0)
        self.assertAlmostEqual(overall["pearson_r"], 1.0)
        self.assertIn(("load", "50"), groups)
        self.assertIn(("horizon", "early"), groups)
        self.assertIn(("load_horizon", "60:late"), groups)


class ValueRerankTest(unittest.TestCase):
    def test_rank_actions_returns_legal_actions_and_greedy_first(self) -> None:
        env = CarrierAircraftSchedulingEnv(small_env_config())
        env.reset(seed=7)
        encoded = encode_observation(env)
        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
        )
        trainer = PPOTrainer(
            model,
            optimizer=None,
            config=PPOConfig(),
        )
        greedy, _, _ = trainer.select_action(
            encoded,
            env,
            deterministic=True,
        )

        ranked = trainer.rank_actions(encoded, env, top_k=4)

        self.assertGreaterEqual(len(ranked), 2)
        self.assertEqual(
            (
                ranked[0][0]["high_level"],
                ranked[0][0]["aircraft_id"],
                ranked[0][0]["target_slot"],
            ),
            (
                greedy["high_level"],
                greedy["aircraft_id"],
                greedy["target_slot"],
            ),
        )
        keys = set()
        for action, _ in ranked:
            parsed = env._parse_action(action)
            self.assertIsNotNone(parsed)
            self.assertTrue(env._is_action_valid(*parsed))
            keys.add(
                (
                    action["high_level"],
                    action["aircraft_id"],
                    action["target_slot"],
                )
            )
        self.assertEqual(len(keys), len(ranked))

    def test_old_checkpoint_loads_but_cannot_enable_rerank(self) -> None:
        env = CarrierAircraftSchedulingEnv(small_env_config())
        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            save_checkpoint(
                str(checkpoint),
                model,
                extra={
                    "observation_schema_version": (
                        OBSERVATION_SCHEMA_VERSION
                    ),
                },
            )

            solver = RLSolver(env, checkpoint=str(checkpoint))
            self.assertIsNotNone(solver.choose_action())
            with self.assertRaisesRegex(
                ValueError,
                "value_calibration metadata",
            ):
                RLSolver(
                    env,
                    checkpoint=str(checkpoint),
                    value_rerank_top_k=2,
                )

    def test_calibration_metadata_checks_load_range(self) -> None:
        reward_config = PPOConfig()
        payload = {
            "extra": {
                "ppo_config": {
                    "gamma": reward_config.gamma,
                    "env_reward_scale": reward_config.env_reward_scale,
                    "sortie_bonus": reward_config.sortie_bonus,
                    "miss_penalty": reward_config.miss_penalty,
                    "progress_shaping": reward_config.progress_shaping,
                },
                "value_calibration": {
                    "version": 1,
                    "wave_interval_range": [47.5, 67.5],
                    "gamma": reward_config.gamma,
                    "reward_config": {
                        "env_reward_scale": (
                            reward_config.env_reward_scale
                        ),
                        "sortie_bonus": reward_config.sortie_bonus,
                        "miss_penalty": reward_config.miss_penalty,
                        "progress_shaping": (
                            reward_config.progress_shaping
                        ),
                    },
                }
            }
        }

        validate_rerank_calibration(payload, 60.0)
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            validate_rerank_calibration(payload, 70.0)
        payload["extra"]["ppo_config"]["gamma"] = 0.9
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_rerank_calibration(payload, 60.0)

    def test_version_two_metadata_requires_counterfactual_target(self) -> None:
        reward_config = PPOConfig()
        payload = {
            "extra": {
                "ppo_config": {
                    "gamma": reward_config.gamma,
                    "env_reward_scale": reward_config.env_reward_scale,
                    "sortie_bonus": reward_config.sortie_bonus,
                    "miss_penalty": reward_config.miss_penalty,
                    "progress_shaping": reward_config.progress_shaping,
                },
                "value_calibration": {
                    "version": 2,
                    "target_method": "mc",
                    "wave_interval_range": [47.5, 67.5],
                    "gamma": reward_config.gamma,
                    "reward_config": {
                        "env_reward_scale": 0.0,
                        "sortie_bonus": 1.0,
                        "miss_penalty": 0.0,
                        "progress_shaping": 0.0,
                    },
                },
            }
        }

        with self.assertRaisesRegex(ValueError, "counterfactual"):
            validate_rerank_calibration(payload, 60.0)

    def test_rerank_selects_highest_one_step_value(self) -> None:
        env = CarrierAircraftSchedulingEnv(small_env_config())
        env.reset(seed=7)
        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
        )
        trainer = PPOTrainer(
            model,
            optimizer=None,
            config=PPOConfig(),
        )
        candidates = trainer.rank_actions(
            encode_observation(env),
            env,
            top_k=2,
        )
        solver = object.__new__(RLSolver)
        solver.trainer = trainer
        solver.env = env
        solver.value_rerank_top_k = 2
        solver.value_rerank_seed = 13
        solver.value_rerank_decisions = 0
        preferred = candidates[1][0]
        preferred_key = (
            preferred["high_level"],
            preferred["aircraft_id"],
            preferred["target_slot"],
        )
        solver._one_step_value = lambda action, seed: (
            1.0
            if (
                action["high_level"],
                action["aircraft_id"],
                action["target_slot"],
            )
            == preferred_key
            else 0.0
        )

        selected = solver._choose_value_reranked_action(
            encode_observation(env)
        )

        self.assertEqual(
            (
                selected["high_level"],
                selected["aircraft_id"],
                selected["target_slot"],
            ),
            preferred_key,
        )


class CalibrationSeedProtocolTest(unittest.TestCase):
    def test_selection_seed_default_is_frozen_range(self) -> None:
        self.assertEqual(
            DEFAULT_SELECTION_SEEDS,
            (41001, 41002, 41003, 41004, 41005),
        )
        self.assertEqual(
            DEFAULT_EVALUATION_SEEDS,
            (43001, 43002, 43003, 43004, 43005),
        )

    def test_seed_protocol_rejects_overlap_and_forbidden_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_seed_protocol([41001], [41001])
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_seed_protocol([42001], [70001])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_seed_protocol([42001, 42001], [41001])
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_development_seeds([0], "calibration")

    def test_evaluation_seeds_are_disjoint(self) -> None:
        payload = {
            "extra": {
                "value_calibration": {
                    "calibration_seeds": [42001, 42002],
                    "selection_seeds": [41001, 41002],
                }
            }
        }

        validate_evaluation_seeds([43001], payload)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_evaluation_seeds([41001], payload)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_evaluation_seeds([70050], payload)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_evaluation_seeds([43001, 43001], payload)

    def test_version_two_metadata_requires_explicit_partitions(self) -> None:
        payload = {
            "extra": {
                "value_calibration": {
                    "version": 2,
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "explicit"):
            validate_evaluation_seeds([43001], payload)

        payload["extra"]["value_calibration"].update(
            {
                "calibration_seeds": [42001],
                "selection_seeds": [42001],
            }
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_evaluation_seeds([43001], payload)


if __name__ == "__main__":
    unittest.main()
