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
from rl.obs_encoder import (
    EncodedObservation,
    batch_to_torch,
    encode_observation,
    target_context_for_action,
    target_slot_for_action,
)
from solution.cp_sat_solver import CPSATSolver
from solution.heuristic_solver import WaveHeuristicSolver
from solution.priority_rule_solver import (
    ACTION_ARM,
    ACTION_FUEL,
    ACTION_INSPECTION,
    ACTION_LAUNCH,
    ACTION_RECOVERY,
)


@dataclass
class Demonstrations:
    observations: List[EncodedObservation] = field(default_factory=list)
    high_actions: List[int] = field(default_factory=list)
    low_actions: List[int] = field(default_factory=list)
    target_actions: List[int] = field(default_factory=list)
    target_masks: List[List[int]] = field(default_factory=list)
    target_aux: List[List[List[float]]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.high_actions)

    def extend(self, other: "Demonstrations") -> None:
        self.observations.extend(other.observations)
        self.high_actions.extend(other.high_actions)
        self.low_actions.extend(other.low_actions)
        self.target_actions.extend(other.target_actions)
        self.target_masks.extend(other.target_masks)
        self.target_aux.extend(other.target_aux)


def build_demonstration_teacher(env, teacher: str):
    if teacher == "heuristic":
        return WaveHeuristicSolver(env)
    if teacher == "cp_sat":
        return CPSATSolver(env)
    if teacher == "elite_cp":
        return EliteCPSATTeacher(env)
    if teacher == "mixed":
        return None
    raise ValueError(f"unknown demonstration teacher: {teacher}")


class EliteCPSATTeacher(CPSATSolver):
    """Training-only expert selected by constrained trajectory search."""

    def __init__(self, env, policy_seed: int = 7):
        super().__init__(env, max_time_seconds=0.02)
        self.policy_rng = random.Random(policy_seed)

    def _pick_top(self, items, key, limit: int):
        ranked = sorted(items, key=key)
        return self.policy_rng.choice(ranked[: min(limit, len(ranked))])

    def choose_action(self):
        mask = self.env.get_action_mask()
        recovery_ids = (
            self._candidate_ids(mask, ACTION_RECOVERY)
            if mask["high_level"][ACTION_RECOVERY]
            else []
        )
        if recovery_ids:
            aircraft_id = self._pick_top(
                recovery_ids,
                lambda item: (
                    self.env._expected_recovery_duration(item),
                    item,
                ),
                2,
            )
            return self.env.complete_action(
                {
                    "high_level": ACTION_RECOVERY,
                    "aircraft_id": aircraft_id,
                }
            )

        launch_ids = (
            self._candidate_ids(mask, ACTION_LAUNCH)
            if mask["high_level"][ACTION_LAUNCH]
            else []
        )
        if launch_ids:
            aircraft_id = self._pick_top(
                launch_ids,
                lambda item: (
                    self.env._expected_launch_duration(item),
                    item,
                ),
                2,
            )
            return self.env.complete_action(
                {
                    "high_level": ACTION_LAUNCH,
                    "aircraft_id": aircraft_id,
                }
            )

        service_actions = [
            (high_level, aircraft_id)
            for high_level in (
                ACTION_FUEL,
                ACTION_ARM,
                ACTION_INSPECTION,
            )
            if mask["high_level"][high_level]
            for aircraft_id in self._candidate_ids(mask, high_level)
        ]
        if not service_actions:
            return None
        selected = (
            self._solve_service_batch(service_actions)
            or service_actions
        )
        high_level, aircraft_id = self._pick_top(
            selected,
            self._priority._spt_key,
            4,
        )
        return self.env.complete_action(
            {
                "high_level": high_level,
                "aircraft_id": aircraft_id,
            }
        )


def append_demonstration(
    demonstrations: Demonstrations,
    env: CarrierAircraftSchedulingEnv,
    encoded: EncodedObservation,
    action: Dict,
) -> None:
    high_action = int(action["high_level"])
    low_action = int(action["aircraft_id"])
    demonstrations.observations.append(encoded)
    demonstrations.high_actions.append(high_action)
    demonstrations.low_actions.append(low_action)
    demonstrations.target_actions.append(
        target_slot_for_action(encoded, action)
    )
    target_mask, target_aux = target_context_for_action(
        env,
        encoded,
        high_action,
        low_action,
    )
    demonstrations.target_masks.append(target_mask)
    demonstrations.target_aux.append(target_aux)


def collect_heuristic_demonstrations(
    env_config: Dict,
    seeds: List[int],
    max_steps: int = 100_000,
    teacher: str = "heuristic",
) -> Demonstrations:
    demonstrations = Demonstrations()
    for seed in seeds:
        env = CarrierAircraftSchedulingEnv(env_config)
        env.reset(seed=seed)
        if teacher == "mixed":
            solver = (
                CPSATSolver(env)
                if seed % 2 == 0
                else WaveHeuristicSolver(env)
            )
        else:
            solver = build_demonstration_teacher(env, teacher)
        steps = 0

        while not env.done and steps < max_steps:
            action = solver.choose_action()
            if action is not None:
                encoded = encode_observation(env)
                append_demonstration(
                    demonstrations,
                    env,
                    encoded,
                    action,
                )
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
    low_loss_weight: float = 1.0,
    target_loss_weight: float = 1.0,
) -> Dict[str, float]:
    if not demonstrations:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "high_accuracy": 0.0,
            "low_accuracy": 0.0,
            "target_accuracy": 0.0,
        }

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
    target_actions = torch.tensor(
        demonstrations.target_actions,
        dtype=torch.long,
        device=device,
    )
    target_masks = torch.tensor(
        demonstrations.target_masks,
        dtype=torch.bool,
        device=device,
    )
    target_aux = torch.tensor(
        demonstrations.target_aux,
        dtype=torch.float32,
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

            high_logits, low_logits, target_logits, _ = (
                model.forward_with_targets(
                batch["aircraft"][mb],
                batch["global"][mb],
                batch["targets"][mb],
                target_aux[mb],
                mb_high_actions,
                mb_low_actions,
                batch["low_aux"][mb],
                )
            )
            legal_high_logits = masked_logits(high_logits, batch["high_mask"][mb])
            selected_low_logits = select_low_action_logits(low_logits, mb_high_actions)
            selected_low_masks = batch["low_masks"][mb][
                torch.arange(len(minibatch), device=device),
                mb_high_actions,
            ]
            legal_low_logits = masked_logits(selected_low_logits, selected_low_masks)
            loss = F.cross_entropy(legal_high_logits, mb_high_actions)
            loss += low_loss_weight * F.cross_entropy(
                legal_low_logits,
                mb_low_actions,
            )
            legal_target_logits = masked_logits(
                target_logits,
                target_masks[mb],
            )
            loss += target_loss_weight * F.cross_entropy(
                legal_target_logits,
                target_actions[mb],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        high_logits, low_logits, target_logits, _ = (
            model.forward_with_targets(
                batch["aircraft"],
                batch["global"],
                batch["targets"],
                target_aux,
                high_actions,
                low_actions,
                batch["low_aux"],
            )
        )
        predicted_high = masked_logits(high_logits, batch["high_mask"]).argmax(dim=-1)
        selected_low_logits = select_low_action_logits(low_logits, high_actions)
        selected_low_masks = batch["low_masks"][
            torch.arange(len(demonstrations), device=device),
            high_actions,
        ]
        predicted_low = masked_logits(selected_low_logits, selected_low_masks).argmax(dim=-1)
        predicted_target = masked_logits(
            target_logits,
            target_masks,
        ).argmax(dim=-1)
        joint_accuracy = (
            (predicted_high == high_actions)
            & (predicted_low == low_actions)
            & (predicted_target == target_actions)
        ).float().mean()

    return {
        "loss": last_loss,
        "accuracy": float(joint_accuracy.item()),
        "high_accuracy": float(
            (predicted_high == high_actions).float().mean().item()
        ),
        "low_accuracy": float(
            (predicted_low == low_actions).float().mean().item()
        ),
        "target_accuracy": float(
            (predicted_target == target_actions).float().mean().item()
        ),
    }
