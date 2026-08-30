"""Heuristic demonstration collection and behavior-cloning pretraining."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch.nn import functional as F

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.masked_categorical import masked_logits
from rl.model import select_low_action_logits
from rl.obs_encoder import EncodedObservation, batch_to_torch, encode_observation
from solution.heuristic_solver import WaveHeuristicSolver


@dataclass
class Demonstrations:
    observations: List[EncodedObservation] = field(default_factory=list)
    high_actions: List[int] = field(default_factory=list)
    low_actions: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.high_actions)


def collect_heuristic_demonstrations(
    env_config: Dict,
    seeds: List[int],
    max_steps: int = 100_000,
) -> Demonstrations:
    demonstrations = Demonstrations()
    for seed in seeds:
        env = CarrierAircraftSchedulingEnv(env_config)
        env.reset(seed=seed)
        solver = WaveHeuristicSolver(env)
        steps = 0

        while not env.done and steps < max_steps:
            action = solver.choose_action()
            if action is not None:
                demonstrations.observations.append(encode_observation(env))
                demonstrations.high_actions.append(action["high_level"])
                demonstrations.low_actions.append(action["aircraft_id"])
            env.step(action)
            steps += 1

        if not env.done:
            raise RuntimeError(f"Heuristic demonstration exceeded {max_steps} steps")
    return demonstrations


def pretrain_behavior_cloning(
    model,
    optimizer,
    demonstrations: Demonstrations,
    epochs: int,
    minibatch_size: int,
    device: str,
) -> Dict[str, float]:
    if not demonstrations:
        return {"loss": 0.0, "accuracy": 0.0}

    batch = batch_to_torch(demonstrations.observations, device)
    high_actions = torch.tensor(
        demonstrations.high_actions,
        dtype=torch.long,
        device=device,
    )
    low_actions = torch.tensor(
        demonstrations.low_actions,
        dtype=torch.long,
        device=device,
    )
    indices = list(range(len(demonstrations)))
    last_loss = 0.0

    model.train()
    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            minibatch = indices[start : start + minibatch_size]
            mb = torch.tensor(minibatch, dtype=torch.long, device=device)
            mb_high_actions = high_actions[mb]
            mb_low_actions = low_actions[mb]

            high_logits, low_logits, _ = model(
                batch["aircraft"][mb],
                batch["global"][mb],
            )
            legal_high_logits = masked_logits(high_logits, batch["high_mask"][mb])
            selected_low_logits = select_low_action_logits(low_logits, mb_high_actions)
            selected_low_masks = batch["low_masks"][mb][
                torch.arange(len(minibatch), device=device),
                mb_high_actions,
            ]
            legal_low_logits = masked_logits(selected_low_logits, selected_low_masks)
            loss = F.cross_entropy(legal_high_logits, mb_high_actions)
            loss += F.cross_entropy(legal_low_logits, mb_low_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        high_logits, low_logits, _ = model(batch["aircraft"], batch["global"])
        predicted_high = masked_logits(high_logits, batch["high_mask"]).argmax(dim=-1)
        selected_low_logits = select_low_action_logits(low_logits, high_actions)
        selected_low_masks = batch["low_masks"][
            torch.arange(len(demonstrations), device=device),
            high_actions,
        ]
        predicted_low = masked_logits(selected_low_logits, selected_low_masks).argmax(dim=-1)
        joint_accuracy = (
            (predicted_high == high_actions) & (predicted_low == low_actions)
        ).float().mean()

    return {
        "loss": last_loss,
        "accuracy": float(joint_accuracy.item()),
    }
