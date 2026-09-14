"""Regression tests for PPO rollout collection."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.behavior_cloning import (
    collect_heuristic_demonstrations,
    pretrain_behavior_cloning,
)
from rl.model import CarrierPolicyValueNet
from rl.obs_encoder import (
    AIRCRAFT_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    OBSERVATION_SCHEMA_VERSION,
    TARGET_AUX_FEATURE_DIM,
    TARGET_FEATURE_DIM,
    encode_observation,
    target_context_for_action,
    target_mask_for_action,
)
from rl.ppo_trainer import PPOTrainer
from rl.train_config import PPOConfig
from scripts.train_rl import (
    EpisodeSeedScheduler,
    build_ppo_optimizer,
    collect_rollout,
    resolve_training_seeds,
    resolve_validation_seeds,
    select_dagger_seeds,
    validate_seed_partition,
)
from solution.rl_solver import RLSolver


class FixedLaunchTrainer:
    def select_action(self, encoded, env, deterministic: bool = False):
        del env, deterministic
        target_slot = encoded.target_keys.index(("runway", 0))
        target_mask = [
            int(index == target_slot)
            for index in range(len(encoded.targets))
        ]
        target_aux = [
            [float(enabled), float(enabled)]
            for enabled in target_mask
        ]
        return {
            "high_level": 3,
            "aircraft_id": 0,
            "target_id": 0,
            "target_slot": target_slot,
            "_target_mask": target_mask,
            "_target_aux": target_aux,
        }, 0.0, 0.0


class RolloutCollectionTest(unittest.TestCase):
    def test_terminal_event_reward_stays_with_preceding_action(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 2,
                "group_size": 1,
                "num_parking_spots": 2,
                "launch_time": 0.5,
                "wave_interval": 60.0,
                "simulation_duration": 0.5,
                "spatial_graph_enabled": False,
            }
        )
        env.reset(seed=7)
        config = PPOConfig(
            rollout_steps=1,
            env_reward_scale=1.0,
            sortie_bonus=0.0,
            miss_penalty=0.0,
        )

        buffer = collect_rollout(env, FixedLaunchTrainer(), config, seed=8)

        self.assertEqual(len(buffer), 1)
        self.assertTrue(buffer.dones[0])
        self.assertAlmostEqual(buffer.rewards[0], 100.5)

    def test_default_reward_ignores_environment_terminal_reward(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 2,
                "group_size": 1,
                "num_parking_spots": 2,
                "launch_time": 0.5,
                "wave_interval": 60.0,
                "simulation_duration": 0.5,
                "spatial_graph_enabled": False,
            }
        )
        env.reset(seed=7)

        buffer = collect_rollout(
            env,
            FixedLaunchTrainer(),
            PPOConfig(rollout_steps=1),
            seed=8,
        )

        self.assertEqual(buffer.rewards, [1.0])


class TrainingSeedProtocolTest(unittest.TestCase):
    def test_explicit_seed_sets_are_deduplicated_in_order(self) -> None:
        self.assertEqual(
            resolve_training_seeds(7, 5, [11, 12, 11]),
            [11, 12],
        )
        self.assertEqual(
            resolve_validation_seeds(10007, 5, [21, 22, 21]),
            [21, 22],
        )

    def test_training_and_validation_seeds_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_seed_partition([7, 8], [8, 9])
        validate_seed_partition([7, 8], [8, 9], allow_overlap=True)

    def test_dagger_samples_only_from_training_partition(self) -> None:
        self.assertEqual(
            select_dagger_seeds([10, 20, 30], iteration=1, episodes=4),
            [20, 30, 10, 20],
        )

    def test_episode_seed_scheduler_cycles_training_partition(self) -> None:
        scheduler = EpisodeSeedScheduler([10, 20, 30])

        self.assertEqual(
            [scheduler.next_seed() for _ in range(5)],
            [10, 20, 30, 10, 20],
        )

    def test_adaptive_rank_gate_can_use_a_separate_learning_rate(self) -> None:
        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
            adaptive_low_rank_prior=True,
        )

        optimizer = build_ppo_optimizer(
            model,
            learning_rate=1.0e-5,
            rank_gate_learning_rate=1.0e-3,
        )

        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [1.0e-5, 1.0e-3],
        )


class PolicyNetworkTest(unittest.TestCase):
    def test_spatial_observation_reports_pathway_availability(self) -> None:
        env = CarrierAircraftSchedulingEnv()
        env.reset(seed=7)

        initial = encode_observation(env)
        env.step({"high_level": 3, "aircraft_id": 0})
        moving = encode_observation(env)

        self.assertEqual(OBSERVATION_SCHEMA_VERSION, 9)
        self.assertEqual(
            len(initial.aircraft[0]),
            AIRCRAFT_FEATURE_DIM,
        )
        self.assertEqual(len(initial.global_features), GLOBAL_FEATURE_DIM)
        self.assertEqual(len(initial.targets[0]), TARGET_FEATURE_DIM)
        self.assertEqual(initial.global_features[15], 1.0)
        self.assertLess(moving.global_features[15], 1.0)

    def test_low_level_logits_are_conditioned_on_high_level_action(self) -> None:
        import torch

        model = CarrierPolicyValueNet(AIRCRAFT_FEATURE_DIM, GLOBAL_FEATURE_DIM)
        high_logits, low_logits, values = model(
            torch.zeros(2, 40, AIRCRAFT_FEATURE_DIM),
            torch.zeros(2, GLOBAL_FEATURE_DIM),
        )

        self.assertEqual(tuple(high_logits.shape), (2, 5))
        self.assertEqual(tuple(low_logits.shape), (2, 5, 40))
        self.assertEqual(tuple(values.shape), (2,))

    def test_target_logits_are_conditioned_on_selected_aircraft(self) -> None:
        import torch

        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
            target_feature_dim=(
                TARGET_FEATURE_DIM + TARGET_AUX_FEATURE_DIM
            ),
        )
        _, _, target_logits, values = model.forward_with_targets(
            torch.zeros(2, 45, AIRCRAFT_FEATURE_DIM),
            torch.zeros(2, GLOBAL_FEATURE_DIM),
            torch.zeros(2, 99, TARGET_FEATURE_DIM),
            torch.zeros(2, 99, TARGET_AUX_FEATURE_DIM),
            torch.tensor([0, 3]),
            torch.tensor([1, 2]),
        )

        self.assertEqual(tuple(target_logits.shape), (2, 99))
        self.assertEqual(tuple(values.shape), (2,))

    def test_policy_selects_a_complete_masked_action(self) -> None:
        import torch

        env = CarrierAircraftSchedulingEnv()
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

        action, _, _ = trainer.select_action(
            encoded,
            env,
            deterministic=True,
        )

        self.assertIn("target_slot", action)
        self.assertIn("target_id", action)
        self.assertTrue(
            action["_target_mask"][action["target_slot"]]
        )
        self.assertTrue(env._is_action_valid(*env._parse_action(action)))

    def test_target_mask_matches_launch_candidates(self) -> None:
        env = CarrierAircraftSchedulingEnv()
        env.reset(seed=7)
        encoded = encode_observation(env)

        mask = target_mask_for_action(env, encoded, 3, 0)
        masked_runways = {
            value
            for enabled, (target_type, value) in zip(
                mask,
                encoded.target_keys,
            )
            if enabled and target_type == "runway"
        }

        self.assertEqual(
            masked_runways,
            set(env.get_target_candidates(3, 0)),
        )
        _, target_aux = target_context_for_action(
            env,
            encoded,
            3,
            0,
        )
        first_runway = env.get_target_candidates(3, 0)[0]
        first_slot = encoded.target_keys.index(
            ("runway", first_runway)
        )
        self.assertEqual(target_aux[first_slot][1], 1.0)

    def test_legacy_low_level_head_shape_remains_available(self) -> None:
        import torch

        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
            action_conditioned_low_head=False,
        )
        _, low_logits, _ = model(
            torch.zeros(2, 40, AIRCRAFT_FEATURE_DIM),
            torch.zeros(2, GLOBAL_FEATURE_DIM),
        )

        self.assertEqual(tuple(low_logits.shape), (2, 40))

    def test_rank_prior_strengths_are_checkpointed(self) -> None:
        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
            low_rank_prior=1.5,
            target_rank_prior=0.5,
            adaptive_low_rank_prior=True,
        )

        config = model.checkpoint_config()

        self.assertEqual(config["low_rank_prior"], 1.5)
        self.assertEqual(config["target_rank_prior"], 0.5)
        self.assertTrue(config["adaptive_low_rank_prior"])

    def test_adaptive_rank_gate_starts_from_configured_prior(self) -> None:
        import torch

        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
            low_rank_prior=2.5,
            adaptive_low_rank_prior=True,
        )

        self.assertTrue(
            torch.equal(
                model.low_rank_gate.weight,
                torch.zeros_like(model.low_rank_gate.weight),
            )
        )
        self.assertTrue(
            torch.equal(
                model.low_rank_gate.bias,
                torch.full_like(model.low_rank_gate.bias, 2.5),
            )
        )

    def test_solver_can_override_rank_priors_for_ablation(self) -> None:
        env = CarrierAircraftSchedulingEnv()
        solver = RLSolver(
            env,
            low_rank_prior=0.0,
            target_rank_prior=0.0,
            disable_low_rank_prior=True,
        )

        self.assertEqual(solver.model.low_rank_prior, 0.0)
        self.assertEqual(solver.model.target_rank_prior, 0.0)
        self.assertEqual(solver.model.low_rank_prior_scale, 0.0)


class BehaviorCloningTest(unittest.TestCase):
    def test_heuristic_demonstrations_can_pretrain_policy(self) -> None:
        import torch

        config = {
            "num_aircraft": 4,
            "group_size": 2,
            "num_parking_spots": 4,
            "wave_interval": 20.0,
            "simulation_duration": 40.0,
            "spatial_graph_enabled": False,
            "num_fuel_servers": 2,
            "num_arm_vehicles": 2,
            "num_ammo_transport_vehicles": 2,
            "num_lower_weapon_lifts": 2,
            "num_upper_weapon_lifts": 2,
            "num_personnel": 8,
        }
        demonstrations = collect_heuristic_demonstrations(config, [7])
        model = CarrierPolicyValueNet(AIRCRAFT_FEATURE_DIM, GLOBAL_FEATURE_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

        stats = pretrain_behavior_cloning(
            model,
            optimizer,
            demonstrations,
            epochs=1,
            minibatch_size=32,
            device="cpu",
        )

        self.assertGreater(len(demonstrations), 0)
        self.assertGreaterEqual(stats["loss"], 0.0)
        self.assertGreaterEqual(stats["accuracy"], 0.0)
        self.assertLessEqual(stats["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
