"""Tests for direct action-conditioned Q collection, fitting, and inference."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.action_value import (
    ACTION_VALUE_CHECKPOINT_TYPE,
    ACTION_VALUE_TARGET_METHOD,
    ACTION_VALUE_VERSION,
    ActionValueRanker,
    ActionValueSample,
    checkpoint_sha256,
    collect_counterfactual_action_value_samples,
    fit_action_value_ranker,
    predict_action_values_for_actions,
    validate_action_value_evaluation_seeds,
)
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


def small_actor():
    torch.manual_seed(1)
    return CarrierPolicyValueNet(
        AIRCRAFT_FEATURE_DIM,
        GLOBAL_FEATURE_DIM,
        hidden_dim=16,
        aircraft_embed_dim=8,
        target_embed_dim=8,
    )


def actor_candidates(actor, env, top_k=3):
    observation = encode_observation(env)
    trainer = PPOTrainer(
        actor,
        optimizer=None,
        config=PPOConfig(),
    )
    return observation, trainer.rank_actions(
        observation,
        env,
        top_k=top_k,
    )


def write_actor_and_action_value_checkpoints(
    directory: str,
    actor,
) -> tuple[Path, Path]:
    actor_path = Path(directory) / "actor.pt"
    action_value_path = Path(directory) / "action_value.pt"
    save_checkpoint(
        str(actor_path),
        actor,
        extra={
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        },
    )
    ranker = ActionValueRanker.from_actor(actor)
    save_checkpoint(
        str(action_value_path),
        ranker,
        extra={
            "checkpoint_type": ACTION_VALUE_CHECKPOINT_TYPE,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "action_value": {
                "version": ACTION_VALUE_VERSION,
                "checkpoint_type": ACTION_VALUE_CHECKPOINT_TYPE,
                "target_method": ACTION_VALUE_TARGET_METHOD,
                "source_actor_sha256": checkpoint_sha256(
                    str(actor_path)
                ),
                "wave_interval_range": [60.0, 60.0],
                "training_seeds": [42001],
                "selection_seeds": [41001],
            },
        },
    )
    return actor_path, action_value_path


class ActionValueCollectionTest(unittest.TestCase):
    def test_samples_keep_source_state_complete_legal_action_and_crn_group(
        self,
    ) -> None:
        actor = small_actor()
        config = small_env_config()
        env = CarrierAircraftSchedulingEnv(config)
        env.reset(seed=42001)
        expected_source = encode_observation(env)

        samples = collect_counterfactual_action_value_samples(
            actor,
            [config],
            [42001],
            device="cpu",
            stride=1,
            max_branch_states=1,
            top_k=2,
            workers=1,
            branch_seed=99,
            max_steps=100,
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(len({sample.group_id for sample in samples}), 1)
        self.assertEqual(len({sample.branch_seed for sample in samples}), 1)
        self.assertEqual(
            [sample.candidate_rank for sample in samples],
            [0, 1],
        )
        self.assertEqual(
            [sample.observation.aircraft for sample in samples],
            [expected_source.aircraft, expected_source.aircraft],
        )
        for sample in samples:
            high, aircraft_id, target_slot = sample.action_key
            self.assertEqual(sample.observation.high_mask[high], 1)
            self.assertEqual(
                sample.observation.low_masks[high][aircraft_id],
                1,
            )
            self.assertEqual(sample.target_aux[target_slot][0], 1.0)
            self.assertIn(
                "target_id",
                sample.action,
            )
            self.assertEqual(sample.target, 1.0)

    def test_collection_rejects_frozen_evaluation_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "70001-70050"):
            collect_counterfactual_action_value_samples(
                small_actor(),
                [small_env_config()],
                [70001],
                device="cpu",
            )

    def test_evaluation_seeds_are_disjoint_and_not_frozen(self) -> None:
        payload = {
            "extra": {
                "action_value": {
                    "training_seeds": [42001],
                    "selection_seeds": [41001],
                }
            }
        }

        self.assertEqual(
            validate_action_value_evaluation_seeds(
                [43001],
                payload,
            ),
            [43001],
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_action_value_evaluation_seeds(
                [42001],
                payload,
            )
        with self.assertRaisesRegex(ValueError, "70001-70050"):
            validate_action_value_evaluation_seeds(
                [70001],
                payload,
            )


class ActionValueRankerTest(unittest.TestCase):
    def test_from_actor_copies_representations_without_sharing_storage(
        self,
    ) -> None:
        actor = small_actor()
        ranker = ActionValueRanker.from_actor(actor)

        for actor_module, ranker_module in (
            (actor.aircraft_encoder, ranker.aircraft_encoder),
            (actor.target_encoder, ranker.target_encoder),
            (actor.global_encoder, ranker.global_encoder),
            (
                actor.high_action_embedding,
                ranker.high_action_embedding,
            ),
        ):
            for actor_parameter, ranker_parameter in zip(
                actor_module.parameters(),
                ranker_module.parameters(),
            ):
                self.assertTrue(
                    torch.equal(actor_parameter, ranker_parameter)
                )
                self.assertNotEqual(
                    actor_parameter.data_ptr(),
                    ranker_parameter.data_ptr(),
                )

    def test_actor_candidates_are_scored_in_one_q_batch(self) -> None:
        env = CarrierAircraftSchedulingEnv(small_env_config())
        env.reset(seed=7)
        actor = small_actor()
        observation, ranked = actor_candidates(actor, env)
        actions = [action for action, _ in ranked]
        ranker = ActionValueRanker.from_actor(actor)

        with patch.object(
            ranker.q_head,
            "forward",
            wraps=ranker.q_head.forward,
        ) as q_forward:
            values = predict_action_values_for_actions(
                ranker,
                observation,
                actions,
                "cpu",
            )

        self.assertEqual(len(values), len(actions))
        self.assertEqual(q_forward.call_count, 1)
        self.assertEqual(
            q_forward.call_args.args[0].shape[0],
            len(actions),
        )

    def test_training_updates_independent_q_without_changing_actor(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(small_env_config())
        env.reset(seed=7)
        actor = small_actor()
        observation, ranked = actor_candidates(actor, env, top_k=2)
        ranker = ActionValueRanker.from_actor(actor)
        actor_before = {
            name: tensor.clone()
            for name, tensor in actor.state_dict().items()
        }
        q_encoder_before = {
            name: tensor.clone()
            for name, tensor in ranker.aircraft_encoder.state_dict().items()
        }
        samples = [
            ActionValueSample(
                observation=copy.deepcopy(observation),
                action={
                    key: value
                    for key, value in action.items()
                    if not key.startswith("_")
                },
                target_aux=copy.deepcopy(action["_target_aux"]),
                target=float(index * 2),
                group_id=(7, 0, 0, 99),
                seed=7,
                wave_interval=60.0,
                time_fraction=0.0,
                source_decision_index=0,
                candidate_rank=index,
                branch_seed=99,
            )
            for index, (action, _) in enumerate(ranked)
        ]

        stats = fit_action_value_ranker(
            ranker,
            samples,
            device="cpu",
            learning_rate=0.01,
            epochs=5,
            minibatch_size=2,
            ranking_loss_coef=1.0,
            update_encoder=True,
            seed=11,
            protected_actor=actor,
        )

        self.assertGreater(stats["last_ranking_loss"], 0.0)
        self.assertTrue(
            any(
                not torch.equal(
                    before,
                    ranker.aircraft_encoder.state_dict()[name],
                )
                for name, before in q_encoder_before.items()
            )
        )
        for name, before in actor_before.items():
            self.assertTrue(torch.equal(before, actor.state_dict()[name]))


class ActionValueSolverTest(unittest.TestCase):
    def test_q_rerank_uses_batch_prediction_without_future_access(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(small_env_config())
        env.reset(seed=7)
        actor = small_actor()
        with tempfile.TemporaryDirectory() as directory:
            actor_path, action_value_path = (
                write_actor_and_action_value_checkpoints(
                    directory,
                    actor,
                )
            )
            solver = RLSolver(
                env,
                checkpoint=str(actor_path),
                action_value_checkpoint=str(action_value_path),
                action_value_top_k=2,
            )

            candidates = solver.trainer.rank_actions(
                encode_observation(env),
                env,
                top_k=2,
            )
            preferred = candidates[1][0]
            with (
                patch(
                    "solution.rl_solver.copy.deepcopy",
                    side_effect=AssertionError("future clone accessed"),
                ),
                patch.object(
                    env,
                    "step",
                    side_effect=AssertionError("future step accessed"),
                ),
                patch.object(
                    solver,
                    "_predict_action_values_for_actions",
                    return_value=[0.0, 1.0],
                ) as predict,
            ):
                selected = solver.choose_action()

        self.assertEqual(
            (
                selected["high_level"],
                selected["aircraft_id"],
                selected["target_slot"],
            ),
            (
                preferred["high_level"],
                preferred["aircraft_id"],
                preferred["target_slot"],
            ),
        )
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(len(predict.call_args.args[2]), 2)

    def test_no_action_value_checkpoint_preserves_actor_choice(self) -> None:
        env = CarrierAircraftSchedulingEnv(small_env_config())
        env.reset(seed=7)
        solver = RLSolver(env)
        expected, _, _ = solver.trainer.select_action(
            encode_observation(env),
            env,
            deterministic=True,
        )

        actual = solver.choose_action()

        self.assertEqual(
            (
                actual["high_level"],
                actual["aircraft_id"],
                actual["target_slot"],
            ),
            (
                expected["high_level"],
                expected["aircraft_id"],
                expected["target_slot"],
            ),
        )

    def test_action_value_checkpoint_cannot_be_used_as_actor(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(small_env_config())
        actor = small_actor()
        with tempfile.TemporaryDirectory() as directory:
            _, action_value_path = write_actor_and_action_value_checkpoints(
                directory,
                actor,
            )
            with self.assertRaisesRegex(
                ValueError,
                "action-value checkpoint",
            ):
                RLSolver(
                    env,
                    checkpoint=str(action_value_path),
                )

    def test_action_value_checkpoint_rejects_different_actor(
        self,
    ) -> None:
        env = CarrierAircraftSchedulingEnv(small_env_config())
        actor = small_actor()
        with tempfile.TemporaryDirectory() as directory:
            _, action_value_path = write_actor_and_action_value_checkpoints(
                directory,
                actor,
            )
            other_actor_path = Path(directory) / "other_actor.pt"
            with torch.no_grad():
                next(actor.parameters()).add_(1.0)
            save_checkpoint(
                str(other_actor_path),
                actor,
                extra={
                    "observation_schema_version": (
                        OBSERVATION_SCHEMA_VERSION
                    ),
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "different actor",
            ):
                RLSolver(
                    env,
                    checkpoint=str(other_actor_path),
                    action_value_checkpoint=str(action_value_path),
                )


if __name__ == "__main__":
    unittest.main()
