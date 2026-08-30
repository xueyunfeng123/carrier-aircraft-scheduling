"""Regression tests for PPO rollout collection."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.behavior_cloning import (
    collect_heuristic_demonstrations,
    pretrain_behavior_cloning,
)
from rl.model import CarrierPolicyValueNet
from rl.obs_encoder import AIRCRAFT_FEATURE_DIM, GLOBAL_FEATURE_DIM
from rl.train_config import PPOConfig
from scripts.train_rl import collect_rollout


class FixedLaunchTrainer:
    def select_action(self, encoded, deterministic: bool = False):
        del encoded, deterministic
        return {"high_level": 3, "aircraft_id": 0}, 0.0, 0.0


class RolloutCollectionTest(unittest.TestCase):
    def test_terminal_event_reward_stays_with_preceding_action(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 2,
                "group_size": 1,
                "num_parking_spots": 2,
                "launch_time": 1.0,
                "wave_interval": 60.0,
                "simulation_duration": 0.5,
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
        self.assertAlmostEqual(buffer.rewards[0], 99.5)

    def test_default_reward_ignores_environment_terminal_reward(self) -> None:
        env = CarrierAircraftSchedulingEnv(
            {
                "num_aircraft": 2,
                "group_size": 1,
                "num_parking_spots": 2,
                "launch_time": 1.0,
                "wave_interval": 60.0,
                "simulation_duration": 0.5,
            }
        )
        env.reset(seed=7)

        buffer = collect_rollout(
            env,
            FixedLaunchTrainer(),
            PPOConfig(rollout_steps=1),
            seed=8,
        )

        self.assertEqual(buffer.rewards, [0.0])


class PolicyNetworkTest(unittest.TestCase):
    def test_low_level_logits_are_conditioned_on_high_level_action(self) -> None:
        import torch

        model = CarrierPolicyValueNet(AIRCRAFT_FEATURE_DIM, GLOBAL_FEATURE_DIM)
        high_logits, low_logits, values = model(
            torch.zeros(2, 40, AIRCRAFT_FEATURE_DIM),
            torch.zeros(2, GLOBAL_FEATURE_DIM),
        )

        self.assertEqual(tuple(high_logits.shape), (2, 4))
        self.assertEqual(tuple(low_logits.shape), (2, 4, 40))
        self.assertEqual(tuple(values.shape), (2,))

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


class BehaviorCloningTest(unittest.TestCase):
    def test_heuristic_demonstrations_can_pretrain_policy(self) -> None:
        import torch

        config = {
            "num_aircraft": 4,
            "group_size": 2,
            "num_parking_spots": 4,
            "wave_interval": 20.0,
            "simulation_duration": 40.0,
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
