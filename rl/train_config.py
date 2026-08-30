"""Default hyperparameters for PPO training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    rollout_steps: int = 512
    total_updates: int = 200
    learning_rate: float = 3.0e-4
    gamma: float = 1.0
    gae_lambda: float = 0.98
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 128
    hidden_dim: int = 128
    aircraft_embed_dim: int = 64
    env_reward_scale: float = 0.0
    sortie_bonus: float = 1.0
    miss_penalty: float = 0.0
