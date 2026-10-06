"""Regression tests for PPO rollout collection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.behavior_cloning import (
    collect_heuristic_demonstrations,
    collect_self_imitation_demonstrations,
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
    average_rollout_stats,
    build_interval_configs,
    build_scenario_configs,
    build_ppo_optimizer,
    collect_rollout,
    format_interval_scores,
    format_profile_scores,
    format_scenario_scores,
    evaluate_policy,
    resolve_training_seeds,
    resolve_validation_seeds,
    robust_checkpoint_score,
    self_imitation_parameters,
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
    def test_interval_configs_preserve_wave_count(self) -> None:
        configs = build_interval_configs(
            {
                "wave_interval": 60.0,
                "simulation_duration": 720.0,
            },
            [45.0, 55.0],
            waves_per_scenario=12,
        )

        self.assertEqual(
            [
                (
                    config["wave_interval"],
                    config["simulation_duration"],
                )
                for config in configs
            ],
            [(45.0, 540.0), (55.0, 660.0)],
        )

    def test_scenario_configs_are_profile_load_cartesian_product(self) -> None:
        configs = build_scenario_configs(
            {
                "wave_interval": 60.0,
                "simulation_duration": 720.0,
                "disruption_profile": "none",
            },
            [50.0, 70.0],
            ["light", "heavy"],
            waves_per_scenario=12,
        )

        self.assertEqual(
            [
                (
                    config["disruption_profile"],
                    config["wave_interval"],
                    config["simulation_duration"],
                )
                for config in configs
            ],
            [
                ("light", 50.0, 600.0),
                ("heavy", 50.0, 600.0),
                ("light", 70.0, 840.0),
                ("heavy", 70.0, 840.0),
            ],
        )

    def test_scenario_configs_default_to_base_profile_and_duration(self) -> None:
        base = {
            "wave_interval": 60.0,
            "simulation_duration": 725.0,
            "disruption_profile": "medium",
        }

        self.assertEqual(
            build_scenario_configs(base, None, None, 12),
            [base],
        )

    def test_rollout_statistics_are_averaged_across_loads(self) -> None:
        stats = average_rollout_stats(
            [
                {"elite_mean": 100.0, "improved_seeds": 1.0},
                {"elite_mean": 120.0, "improved_seeds": 3.0},
            ]
        )

        self.assertEqual(stats["elite_mean"], 110.0)
        self.assertEqual(stats["improved_seeds"], 4.0)

    def test_interval_scores_have_stable_order(self) -> None:
        formatted = format_interval_scores(
            {
                "completed_by_interval": {
                    70.0: 133.5,
                    50.0: 106.0,
                    60.0: 120.0,
                }
            }
        )

        self.assertEqual(
            formatted,
            "50:106.00;60:120.00;70:133.50",
        )

    def test_profile_and_scenario_scores_are_formatted(self) -> None:
        stats = {
            "completed_by_profile": {
                "none": 120.0,
                "heavy": 112.5,
            },
            "completed_by_scenario": {
                ("none", 60.0): 120.0,
                ("heavy", 70.0): 118.0,
                ("heavy", 50.0): 107.0,
            },
        }

        self.assertEqual(
            format_profile_scores(stats),
            "none:120.00;heavy:112.50",
        )
        self.assertEqual(
            format_scenario_scores(stats),
            "heavy@50:107.00;heavy@70:118.00;none@60:120.00",
        )

    def test_robust_checkpoint_score_prioritizes_worst_scenario(self) -> None:
        robust = {
            "worst_scenario_completed": 110.0,
            "completed": 118.0,
            "worst_completed": 108.0,
            "missed": 122.0,
        }
        fragile = {
            "worst_scenario_completed": 109.0,
            "completed": 125.0,
            "worst_completed": 109.0,
            "missed": 115.0,
        }

        self.assertGreater(
            robust_checkpoint_score(robust),
            robust_checkpoint_score(fragile),
        )

    @patch("scripts.train_rl.run_episode")
    def test_policy_evaluation_aggregates_profiles_and_loads(
        self,
        run_episode,
    ) -> None:
        totals = iter([100, 102, 90, 94, 120, 124, 110, 112])

        def fake_episode(*args, **kwargs):
            del args, kwargs
            completed = next(totals)
            return {
                "total_sorties_completed": completed,
                "total_missed_sorties": 240 - completed,
            }

        run_episode.side_effect = fake_episode
        configs = build_scenario_configs(
            {
                "wave_interval": 60.0,
                "simulation_duration": 720.0,
                "disruption_profile": "none",
            },
            [50.0, 70.0],
            ["none", "heavy"],
            waves_per_scenario=12,
        )

        stats = evaluate_policy(
            configs,
            checkpoint="unused.pt",
            seeds=[1, 2],
            device="cpu",
        )

        self.assertEqual(
            stats["completed_by_scenario"],
            {
                ("none", 50.0): 101,
                ("heavy", 50.0): 92,
                ("none", 70.0): 122,
                ("heavy", 70.0): 111,
            },
        )
        self.assertEqual(
            stats["completed_by_profile"],
            {"none": 111.5, "heavy": 101.5},
        )
        self.assertEqual(
            stats["completed_by_interval"],
            {50.0: 96.5, 70.0: 116.5},
        )
        self.assertEqual(stats["worst_scenario_completed"], 92)

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

    def test_self_imitation_can_update_only_rank_gate(self) -> None:
        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
            adaptive_low_rank_prior=True,
        )

        selected = self_imitation_parameters(model, "gate")

        self.assertEqual(
            {id(parameter) for parameter in selected},
            {
                id(parameter)
                for parameter in model.low_rank_gate.parameters()
            },
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

    def test_empty_stochastic_levels_match_deterministic_action(self) -> None:
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

        deterministic, _, _ = trainer.select_action(
            encoded,
            env,
            deterministic=True,
        )
        no_sampling, _, _ = trainer.select_action(
            encoded,
            env,
            stochastic_levels=(),
        )

        self.assertEqual(
            (
                deterministic["high_level"],
                deterministic["aircraft_id"],
                deterministic["target_slot"],
            ),
            (
                no_sampling["high_level"],
                no_sampling["aircraft_id"],
                no_sampling["target_slot"],
            ),
        )

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
            low_rank_prior_scale=0.5,
            disable_low_rank_prior=True,
        )

        self.assertEqual(solver.model.low_rank_prior, 0.0)
        self.assertEqual(solver.model.target_rank_prior, 0.0)
        self.assertEqual(solver.model.low_rank_prior_scale, 0.0)

        scaled_solver = RLSolver(
            env,
            low_rank_prior_scale=0.5,
        )
        self.assertEqual(
            scaled_solver.model.low_rank_prior_scale,
            0.5,
        )


class BehaviorCloningTest(unittest.TestCase):
    def test_self_imitation_keeps_best_complete_trajectory(self) -> None:
        config = {
            "num_aircraft": 4,
            "group_size": 2,
            "num_parking_spots": 4,
            "wave_interval": 20.0,
            "simulation_duration": 40.0,
            "spatial_graph_enabled": False,
            "num_fuel_servers": 2,
            "num_inspection_vehicles": 2,
            "num_arm_vehicles": 2,
            "num_ammo_transport_vehicles": 2,
            "num_lower_weapon_lifts": 2,
            "num_upper_weapon_lifts": 2,
            "num_tow_vehicles": 2,
            "num_personnel": 8,
        }
        model = CarrierPolicyValueNet(
            AIRCRAFT_FEATURE_DIM,
            GLOBAL_FEATURE_DIM,
        )

        demonstrations, stats = collect_self_imitation_demonstrations(
            model,
            config,
            seeds=[7],
            candidates_per_seed=2,
            device="cpu",
            ppo_config=PPOConfig(),
            policy_seed=19,
        )

        self.assertGreater(len(demonstrations), 0)
        self.assertGreaterEqual(
            stats["elite_mean"],
            stats["deterministic_mean"],
        )

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
