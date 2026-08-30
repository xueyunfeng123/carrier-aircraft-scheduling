"""Rollout storage and GAE computation for PPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from rl.obs_encoder import EncodedObservation


@dataclass
class RolloutBuffer:
    observations: List[EncodedObservation] = field(default_factory=list)
    high_actions: List[int] = field(default_factory=list)
    low_actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)

    def add(
        self,
        observation: EncodedObservation,
        high_action: int,
        low_action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        self.observations.append(observation)
        self.high_actions.append(high_action)
        self.low_actions.append(low_action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.observations.clear()
        self.high_actions.clear()
        self.low_actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages.clear()
        self.returns.clear()

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        self.advantages = [0.0] * len(self.rewards)
        self.returns = [0.0] * len(self.rewards)
        gae = 0.0
        next_value = float(last_value)
        for index in reversed(range(len(self.rewards))):
            nonterminal = 0.0 if self.dones[index] else 1.0
            delta = self.rewards[index] + gamma * next_value * nonterminal - self.values[index]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            self.advantages[index] = gae
            self.returns[index] = gae + self.values[index]
            next_value = self.values[index]

