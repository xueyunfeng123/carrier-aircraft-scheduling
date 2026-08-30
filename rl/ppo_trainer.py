"""PPO trainer and masked action selection utilities."""

from __future__ import annotations

import random
from typing import Dict, Tuple

from rl.masked_categorical import masked_categorical
from rl.model import select_low_action_logits
from rl.obs_encoder import batch_to_torch, to_torch_batch


class PPOTrainer:
    def __init__(self, model, optimizer, config, device: str = "cpu"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch is required for RL solver/training. Install it with `pip install torch`."
            ) from exc

        self.torch = torch
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device

    def select_action(self, encoded, deterministic: bool = False) -> Tuple[Dict[str, int], float, float]:
        batch = to_torch_batch(encoded, self.device)
        with self.torch.no_grad():
            high_logits, low_logits, value = self.model(batch["aircraft"], batch["global"])
            high_dist = masked_categorical(high_logits, batch["high_mask"])
            if deterministic:
                high_action = high_logits.masked_fill(~batch["high_mask"], -1.0e9).argmax(dim=-1)
            else:
                high_action = high_dist.sample()

            low_mask = batch["low_masks"][self.torch.arange(1, device=self.device), high_action]
            selected_low_logits = select_low_action_logits(low_logits, high_action)
            low_dist = masked_categorical(selected_low_logits, low_mask)
            if deterministic:
                low_action = selected_low_logits.masked_fill(~low_mask, -1.0e9).argmax(dim=-1)
            else:
                low_action = low_dist.sample()

            log_prob = high_dist.log_prob(high_action) + low_dist.log_prob(low_action)

        action = {
            "high_level": int(high_action.item()),
            "aircraft_id": int(low_action.item()),
        }
        return action, float(log_prob.item()), float(value.item())

    def update(self, buffer) -> Dict[str, float]:
        torch = self.torch
        if len(buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        self.model.train()
        batch = batch_to_torch(buffer.observations, self.device)
        high_actions = torch.tensor(buffer.high_actions, dtype=torch.long, device=self.device)
        low_actions = torch.tensor(buffer.low_actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(buffer.returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(buffer.advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

        indices = list(range(len(buffer)))
        last_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.config.update_epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                mb = indices[start : start + self.config.minibatch_size]
                mb_tensor = torch.tensor(mb, dtype=torch.long, device=self.device)

                high_logits, low_logits, values = self.model(
                    batch["aircraft"][mb_tensor],
                    batch["global"][mb_tensor],
                )
                high_mask = batch["high_mask"][mb_tensor]
                low_masks = batch["low_masks"][mb_tensor]
                high_dist = masked_categorical(high_logits, high_mask)

                mb_high_actions = high_actions[mb_tensor]
                selected_low_mask = low_masks[
                    torch.arange(len(mb), device=self.device),
                    mb_high_actions,
                ]
                selected_low_logits = select_low_action_logits(low_logits, mb_high_actions)
                low_dist = masked_categorical(selected_low_logits, selected_low_mask)

                mb_low_actions = low_actions[mb_tensor]
                log_probs = high_dist.log_prob(mb_high_actions) + low_dist.log_prob(mb_low_actions)
                entropy = high_dist.entropy().mean() + low_dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs[mb_tensor])
                mb_advantages = advantages[mb_tensor]
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                )
                policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns[mb_tensor])
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                last_stats = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }
        return last_stats
