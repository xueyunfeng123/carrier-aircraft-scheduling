"""PPO trainer and masked action selection utilities."""

from __future__ import annotations

import random
from typing import Dict, Tuple

from rl.masked_categorical import masked_categorical
from rl.model import select_low_action_logits
from rl.obs_encoder import (
    apply_target_slot,
    batch_to_torch,
    target_context_for_action,
    to_torch_batch,
)


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

    def select_action(
        self,
        encoded,
        env,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, int], float, float]:
        batch = to_torch_batch(encoded, self.device)
        with self.torch.no_grad():
            high_logits, low_logits, value = self.model(
                batch["aircraft"],
                batch["global"],
                batch["targets"],
                batch["low_aux"],
            )
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

            target_mask_values, target_aux_values = (
                target_context_for_action(
                env,
                encoded,
                int(high_action.item()),
                int(low_action.item()),
                )
            )
            target_mask = self.torch.tensor(
                [target_mask_values],
                dtype=self.torch.bool,
                device=self.device,
            )
            target_aux = self.torch.tensor(
                [target_aux_values],
                dtype=self.torch.float32,
                device=self.device,
            )
            _, _, target_logits, _ = self.model.forward_with_targets(
                batch["aircraft"],
                batch["global"],
                batch["targets"],
                target_aux,
                high_action,
                low_action,
                batch["low_aux"],
            )
            target_dist = masked_categorical(target_logits, target_mask)
            if deterministic:
                target_action = target_logits.masked_fill(
                    ~target_mask,
                    -1.0e9,
                ).argmax(dim=-1)
            else:
                target_action = target_dist.sample()

            log_prob = (
                high_dist.log_prob(high_action)
                + low_dist.log_prob(low_action)
                + target_dist.log_prob(target_action)
            )

        action = apply_target_slot(
            encoded,
            {
            "high_level": int(high_action.item()),
            "aircraft_id": int(low_action.item()),
            },
            int(target_action.item()),
        )
        action["_target_mask"] = target_mask_values
        action["_target_aux"] = target_aux_values
        return action, float(log_prob.item()), float(value.item())

    def update(self, buffer) -> Dict[str, float]:
        torch = self.torch
        if len(buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        self.model.train()
        batch = batch_to_torch(buffer.observations, self.device)
        high_actions = torch.tensor(buffer.high_actions, dtype=torch.long, device=self.device)
        low_actions = torch.tensor(buffer.low_actions, dtype=torch.long, device=self.device)
        target_actions = torch.tensor(
            buffer.target_actions,
            dtype=torch.long,
            device=self.device,
        )
        target_masks = torch.tensor(
            buffer.target_masks,
            dtype=torch.bool,
            device=self.device,
        )
        target_aux = torch.tensor(
            buffer.target_aux,
            dtype=torch.float32,
            device=self.device,
        )
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

                mb_high_actions = high_actions[mb_tensor]
                mb_low_actions = low_actions[mb_tensor]
                high_logits, low_logits, target_logits, values = (
                    self.model.forward_with_targets(
                    batch["aircraft"][mb_tensor],
                    batch["global"][mb_tensor],
                    batch["targets"][mb_tensor],
                    target_aux[mb_tensor],
                    mb_high_actions,
                    mb_low_actions,
                    batch["low_aux"][mb_tensor],
                    )
                )
                high_mask = batch["high_mask"][mb_tensor]
                low_masks = batch["low_masks"][mb_tensor]
                high_dist = masked_categorical(high_logits, high_mask)

                selected_low_mask = low_masks[
                    torch.arange(len(mb), device=self.device),
                    mb_high_actions,
                ]
                selected_low_logits = select_low_action_logits(low_logits, mb_high_actions)
                low_dist = masked_categorical(selected_low_logits, selected_low_mask)

                target_dist = masked_categorical(
                    target_logits,
                    target_masks[mb_tensor],
                )
                mb_target_actions = target_actions[mb_tensor]
                log_probs = (
                    high_dist.log_prob(mb_high_actions)
                    + low_dist.log_prob(mb_low_actions)
                    + target_dist.log_prob(mb_target_actions)
                )
                entropy = (
                    high_dist.entropy().mean()
                    + low_dist.entropy().mean()
                    + target_dist.entropy().mean()
                )

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
