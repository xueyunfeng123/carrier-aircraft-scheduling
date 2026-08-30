"""Regression tests for PPO rollout collection."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
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
        config = PPOConfig(rollout_steps=1, miss_penalty=0.0)

        buffer = collect_rollout(env, FixedLaunchTrainer(), config, seed=8)

        self.assertEqual(len(buffer), 1)
        self.assertTrue(buffer.dones[0])
        self.assertAlmostEqual(buffer.rewards[0], 99.5)


if __name__ == "__main__":
    unittest.main()
