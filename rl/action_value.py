"""Direct action-conditioned Q ranking for actor-proposed scheduling actions."""

from __future__ import annotations

import copy
import hashlib
import random
import statistics
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.obs_encoder import EncodedObservation, batch_to_torch, encode_observation
from rl.ppo_trainer import PPOTrainer
from rl.train_config import PPOConfig
from rl.value_calibration import (
    _counterfactual_branch_seed,
    _distinct_ranked_actions,
    clone_environment_for_counterfactual,
    pairwise_ranking_loss,
    validate_development_seeds,
)


ACTION_VALUE_VERSION = 1
ACTION_VALUE_CHECKPOINT_TYPE = "action_value_ranker"
ACTION_VALUE_TARGET_METHOD = "counterfactual_source_action_remaining_sorties"
ActionValueGroupId = Tuple[int, int, int, int]


@dataclass
class ActionValueSample:
    """One source-state action and its CRN terminal continuation target."""

    observation: EncodedObservation
    action: Dict[str, Any]
    target_aux: List[List[float]]
    target: float
    group_id: ActionValueGroupId
    seed: int
    wave_interval: float
    time_fraction: float
    source_decision_index: int
    candidate_rank: int
    branch_seed: int

    @property
    def action_key(self) -> Tuple[int, int, int]:
        return (
            int(self.action["high_level"]),
            int(self.action["aircraft_id"]),
            int(self.action["target_slot"]),
        )


class ActionValueRanker(nn.Module):
    """Q(s, a) model with an encoder copy independent from the actor."""

    def __init__(
        self,
        aircraft_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 128,
        aircraft_embed_dim: int = 64,
        target_feature_dim: int = 17,
        target_embed_dim: int = 64,
        num_high_actions: int = 5,
        q_hidden_dim: int = 128,
    ):
        super().__init__()
        self.aircraft_feature_dim = int(aircraft_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.aircraft_embed_dim = int(aircraft_embed_dim)
        self.target_feature_dim = int(target_feature_dim)
        self.target_embed_dim = int(target_embed_dim)
        self.num_high_actions = int(num_high_actions)
        self.q_hidden_dim = int(q_hidden_dim)

        self.aircraft_encoder = nn.Sequential(
            nn.Linear(self.aircraft_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.aircraft_embed_dim),
            nn.ReLU(),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(self.target_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.target_embed_dim),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(
                self.global_feature_dim
                + self.aircraft_embed_dim * 2
                + self.target_embed_dim * 2,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.high_action_embedding = nn.Embedding(
            self.num_high_actions,
            self.aircraft_embed_dim,
        )
        self.q_head = nn.Sequential(
            nn.Linear(
                self.hidden_dim
                + self.aircraft_embed_dim * 2
                + self.target_embed_dim,
                self.q_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.q_hidden_dim, 1),
        )

    @classmethod
    def from_actor(
        cls,
        actor,
        q_hidden_dim: Optional[int] = None,
    ) -> "ActionValueRanker":
        """Initialize an independent Q network from actor representation weights."""

        ranker = cls(
            aircraft_feature_dim=actor.aircraft_feature_dim,
            global_feature_dim=actor.global_feature_dim,
            hidden_dim=actor.hidden_dim,
            aircraft_embed_dim=actor.aircraft_embed_dim,
            target_feature_dim=actor.target_feature_dim,
            target_embed_dim=actor.target_embed_dim,
            num_high_actions=actor.num_high_actions,
            q_hidden_dim=(
                actor.hidden_dim
                if q_hidden_dim is None
                else int(q_hidden_dim)
            ),
        )
        ranker.aircraft_encoder.load_state_dict(
            copy.deepcopy(actor.aircraft_encoder.state_dict())
        )
        ranker.target_encoder.load_state_dict(
            copy.deepcopy(actor.target_encoder.state_dict())
        )
        ranker.global_encoder.load_state_dict(
            copy.deepcopy(actor.global_encoder.state_dict())
        )
        ranker.high_action_embedding.load_state_dict(
            copy.deepcopy(actor.high_action_embedding.state_dict())
        )
        return ranker

    def _encode_state(self, aircraft, global_features, targets):
        aircraft_embed = self.aircraft_encoder(aircraft)
        mean_pool = aircraft_embed.mean(dim=1)
        max_pool = aircraft_embed.max(dim=1).values
        padding_width = self.target_feature_dim - targets.shape[-1]
        if padding_width < 0:
            raise ValueError("target feature tensor is too wide")
        state_targets = targets
        if padding_width:
            state_targets = F.pad(state_targets, (0, padding_width))
        state_target_embed = self.target_encoder(state_targets)
        context = self.global_encoder(
            torch.cat(
                [
                    global_features,
                    mean_pool,
                    max_pool,
                    state_target_embed.mean(dim=1),
                    state_target_embed.max(dim=1).values,
                ],
                dim=-1,
            )
        )
        return aircraft_embed, context

    def forward(
        self,
        aircraft,
        global_features,
        targets,
        target_aux,
        high_actions,
        aircraft_actions,
        target_slots,
    ):
        """Predict Q for a batch of explicit complete actions."""

        if any(
            actions.ndim != 1
            for actions in (
                high_actions,
                aircraft_actions,
                target_slots,
            )
        ):
            raise ValueError("action tensors must be one-dimensional")
        batch_size = aircraft.shape[0]
        if not all(
            len(actions) == batch_size
            for actions in (
                high_actions,
                aircraft_actions,
                target_slots,
            )
        ):
            raise ValueError("action tensors must match observation batch size")
        if target_aux.shape[:2] != targets.shape[:2]:
            raise ValueError("target auxiliary features must align with targets")

        aircraft_embed, context = self._encode_state(
            aircraft,
            global_features,
            targets,
        )
        target_embed = self.target_encoder(
            torch.cat([targets, target_aux], dim=-1)
        )
        batch_indices = torch.arange(
            batch_size,
            device=aircraft.device,
        )
        selected_aircraft = aircraft_embed[
            batch_indices,
            aircraft_actions,
        ]
        selected_high = self.high_action_embedding(high_actions)
        selected_target = target_embed[
            batch_indices,
            target_slots,
        ]
        return self.q_head(
            torch.cat(
                [
                    context,
                    selected_aircraft,
                    selected_high,
                    selected_target,
                ],
                dim=-1,
            )
        ).squeeze(-1)

    def checkpoint_config(self) -> Dict[str, Any]:
        return {
            "aircraft_feature_dim": self.aircraft_feature_dim,
            "global_feature_dim": self.global_feature_dim,
            "hidden_dim": self.hidden_dim,
            "aircraft_embed_dim": self.aircraft_embed_dim,
            "target_feature_dim": self.target_feature_dim,
            "target_embed_dim": self.target_embed_dim,
            "num_high_actions": self.num_high_actions,
            "q_hidden_dim": self.q_hidden_dim,
        }


def collect_counterfactual_action_value_samples(
    actor_model,
    configs: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    device: str,
    stride: int = 20,
    max_branch_states: int = 8,
    top_k: int = 4,
    workers: int = 1,
    branch_seed: int = 52_000,
    max_steps: int = 100_000,
) -> List[ActionValueSample]:
    """Collect source-state action targets from ambiguous actor decisions."""

    for name, value in (
        ("stride", stride),
        ("max_branch_states", max_branch_states),
        ("top_k", top_k),
        ("workers", workers),
        ("max_steps", max_steps),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if int(top_k) < 2:
        raise ValueError("top_k must be at least 2 for ambiguous decisions")
    normalized_seeds = validate_development_seeds(
        seeds,
        "action-value",
    )
    if not configs:
        raise ValueError("at least one environment config is required")

    actor_model.eval()
    executor: Optional[Executor] = None
    if workers > 1:
        executor = ThreadPoolExecutor(max_workers=int(workers))
    try:
        samples: List[ActionValueSample] = []
        for config_index, config in enumerate(configs):
            for seed in normalized_seeds:
                samples.extend(
                    _collect_action_value_trajectory(
                        actor_model=actor_model,
                        env_config=config,
                        seed=seed,
                        config_index=config_index,
                        device=device,
                        stride=int(stride),
                        max_branch_states=int(max_branch_states),
                        top_k=int(top_k),
                        branch_seed=int(branch_seed),
                        max_steps=int(max_steps),
                        executor=executor,
                    )
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if not samples:
        raise RuntimeError("action-value collection produced no samples")
    return samples


def _collect_action_value_trajectory(
    actor_model,
    env_config: Dict[str, Any],
    seed: int,
    config_index: int,
    device: str,
    stride: int,
    max_branch_states: int,
    top_k: int,
    branch_seed: int,
    max_steps: int,
    executor: Optional[Executor],
) -> List[ActionValueSample]:
    env = CarrierAircraftSchedulingEnv(env_config)
    env.reset(seed=seed)
    trainer = PPOTrainer(
        actor_model,
        optimizer=None,
        config=PPOConfig(),
        device=device,
    )
    reservoir: List[List[tuple]] = []
    reservoir_rng = random.Random(
        _counterfactual_branch_seed(
            branch_seed,
            seed,
            config_index,
            -1,
        )
    )
    decision_index = 0
    ambiguous_index = 0
    eligible_states = 0
    steps = 0

    while not env.done and steps < max_steps:
        encoded = encode_observation(env)
        if not any(encoded.high_mask):
            _, _, done, info = env.step(None)
            if info["invalid_action"]:
                raise RuntimeError(
                    "base actor trajectory failed to advance legally"
                )
            steps += 1
            if done:
                break
            continue

        candidates = _distinct_ranked_actions(
            trainer.rank_actions(
                encoded,
                env,
                top_k=top_k,
            )
        )
        if len(candidates) >= 2:
            if ambiguous_index % stride == 0:
                shared_seed = _counterfactual_branch_seed(
                    branch_seed,
                    seed,
                    config_index,
                    decision_index,
                )
                source_observation = copy.deepcopy(encoded)
                source_sorties = int(
                    env.get_evaluation_metrics()[
                        "total_sorties_completed"
                    ]
                )
                group_id: ActionValueGroupId = (
                    int(seed),
                    int(config_index),
                    int(decision_index),
                    int(shared_seed),
                )
                prepared = []
                for rank, (action, _) in enumerate(candidates):
                    parsed = env._parse_action(action)
                    if (
                        parsed is None
                        or not env._is_action_valid(*parsed)
                    ):
                        raise RuntimeError(
                            "actor ranking emitted an illegal candidate action"
                        )
                    prepared.append(
                        (
                            trainer,
                            clone_environment_for_counterfactual(
                                env,
                                shared_seed,
                            ),
                            source_observation,
                            action,
                            rank,
                            group_id,
                            source_sorties,
                            seed,
                            decision_index,
                            shared_seed,
                            max_steps,
                        )
                    )
                eligible_states += 1
                if len(reservoir) < max_branch_states:
                    reservoir.append(prepared)
                else:
                    replacement = reservoir_rng.randrange(eligible_states)
                    if replacement < max_branch_states:
                        reservoir[replacement] = prepared
            ambiguous_index += 1

        action, _, _ = trainer.select_action(
            encoded,
            env,
            deterministic=True,
        )
        _, _, _, info = env.step(action)
        if info["invalid_action"]:
            raise RuntimeError("base actor emitted an illegal action")
        decision_index += 1
        steps += 1

    if not env.done:
        raise RuntimeError(
            f"base actor trajectory exceeded {max_steps} steps for seed {seed}"
        )
    reservoir.sort(key=lambda prepared: prepared[0][8])
    samples: List[ActionValueSample] = []
    for prepared in reservoir:
        if executor is None:
            branch_samples = [
                _rollout_action_value_candidate(*arguments)
                for arguments in prepared
            ]
        else:
            branch_samples = list(
                executor.map(
                    _rollout_action_value_candidate_from_tuple,
                    prepared,
                )
            )
        samples.extend(branch_samples)
    return samples


def _rollout_action_value_candidate_from_tuple(
    arguments,
) -> ActionValueSample:
    return _rollout_action_value_candidate(*arguments)


def _rollout_action_value_candidate(
    trainer: PPOTrainer,
    env: CarrierAircraftSchedulingEnv,
    source_observation: EncodedObservation,
    action: Dict[str, Any],
    candidate_rank: int,
    group_id: ActionValueGroupId,
    source_sorties: int,
    source_seed: int,
    source_decision_index: int,
    branch_seed: int,
    max_steps: int,
) -> ActionValueSample:
    _, _, _, info = env.step(action)
    if info["invalid_action"]:
        raise RuntimeError("actor ranking emitted an illegal candidate action")
    steps = 1
    while not env.done and steps < max_steps:
        encoded = encode_observation(env)
        if any(encoded.high_mask):
            rollout_action, _, _ = trainer.select_action(
                encoded,
                env,
                deterministic=True,
            )
        else:
            rollout_action = None
        _, _, _, info = env.step(rollout_action)
        if info["invalid_action"]:
            raise RuntimeError(
                "deterministic actor rollout emitted an illegal action"
            )
        steps += 1
    if not env.done:
        raise RuntimeError(
            "counterfactual actor rollout exceeded max_steps"
        )

    final_sorties = int(
        env.get_evaluation_metrics()["total_sorties_completed"]
    )
    clean_action = {
        key: value
        for key, value in action.items()
        if not key.startswith("_")
    }
    return ActionValueSample(
        observation=source_observation,
        action=clean_action,
        target_aux=copy.deepcopy(action["_target_aux"]),
        target=float(final_sorties - source_sorties),
        group_id=group_id,
        seed=int(source_seed),
        wave_interval=float(env.config["wave_interval"]),
        time_fraction=(
            float(source_observation.global_features[0])
        ),
        source_decision_index=int(source_decision_index),
        candidate_rank=int(candidate_rank),
        branch_seed=int(branch_seed),
    )


def _action_batch(
    observations: Sequence[EncodedObservation],
    actions: Sequence[Dict[str, Any]],
    target_aux_values: Sequence[Sequence[Sequence[float]]],
    device: str,
) -> Dict[str, Any]:
    if not observations:
        raise ValueError("at least one action is required")
    if not (
        len(observations)
        == len(actions)
        == len(target_aux_values)
    ):
        raise ValueError(
            "observations, actions, and target auxiliary values "
            "must have equal lengths"
        )
    batch = batch_to_torch(list(observations), device)
    batch.update(
        {
            "target_aux": torch.tensor(
                target_aux_values,
                dtype=torch.float32,
                device=device,
            ),
            "high_actions": torch.tensor(
                [int(action["high_level"]) for action in actions],
                dtype=torch.long,
                device=device,
            ),
            "aircraft_actions": torch.tensor(
                [int(action["aircraft_id"]) for action in actions],
                dtype=torch.long,
                device=device,
            ),
            "target_slots": torch.tensor(
                [int(action["target_slot"]) for action in actions],
                dtype=torch.long,
                device=device,
            ),
        }
    )
    return batch


def predict_action_values(
    model: ActionValueRanker,
    samples: Sequence[ActionValueSample],
    device: str,
    batch_size: int = 1024,
) -> List[float]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    predictions: List[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            minibatch = samples[start : start + batch_size]
            batch = _action_batch(
                [sample.observation for sample in minibatch],
                [sample.action for sample in minibatch],
                [sample.target_aux for sample in minibatch],
                device,
            )
            values = model(
                batch["aircraft"],
                batch["global"],
                batch["targets"],
                batch["target_aux"],
                batch["high_actions"],
                batch["aircraft_actions"],
                batch["target_slots"],
            )
            predictions.extend(
                float(value)
                for value in values.detach().cpu().tolist()
            )
    return predictions


def predict_action_values_for_actions(
    model: ActionValueRanker,
    observation: EncodedObservation,
    actions: Sequence[Dict[str, Any]],
    device: str,
) -> List[float]:
    """Score actor candidates in one batched Q forward pass."""

    if not actions:
        return []
    missing_aux = [
        index
        for index, action in enumerate(actions)
        if "_target_aux" not in action
    ]
    if missing_aux:
        raise ValueError(
            "candidate actions are missing target context at indices "
            f"{missing_aux}"
        )
    batch = _action_batch(
        [observation] * len(actions),
        actions,
        [action["_target_aux"] for action in actions],
        device,
    )
    model.eval()
    with torch.no_grad():
        values = model(
            batch["aircraft"],
            batch["global"],
            batch["targets"],
            batch["target_aux"],
            batch["high_actions"],
            batch["aircraft_actions"],
            batch["target_slots"],
        )
    return [
        float(value)
        for value in values.detach().cpu().tolist()
    ]


def fit_action_value_ranker(
    model: ActionValueRanker,
    samples: Sequence[ActionValueSample],
    device: str,
    learning_rate: float = 1.0e-3,
    epochs: int = 100,
    minibatch_size: int = 512,
    ranking_loss_coef: float = 1.0,
    update_encoder: bool = True,
    seed: int = 0,
    protected_actor=None,
) -> Dict[str, float]:
    """Fit Huber plus within-CRN-group pairwise ranking loss."""

    if not samples:
        raise ValueError("at least one action-value sample is required")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if minibatch_size < 1:
        raise ValueError("minibatch_size must be positive")
    if ranking_loss_coef < 0.0:
        raise ValueError("ranking_loss_coef must be non-negative")

    protected_before = (
        {
            name: tensor.detach().cpu().clone()
            for name, tensor in protected_actor.state_dict().items()
        }
        if protected_actor is not None
        else {}
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            bool(update_encoder) or name.startswith("q_head.")
        )
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
    )
    targets = torch.tensor(
        [sample.target for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    group_ids: List[Hashable] = [
        sample.group_id
        for sample in samples
    ]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    initial_predictions = predict_action_values(
        model,
        samples,
        device,
        batch_size=minibatch_size,
    )
    initial_mse = statistics.mean(
        (prediction - sample.target) ** 2
        for prediction, sample in zip(initial_predictions, samples)
    )

    model.train()
    last_loss = 0.0
    last_regression_loss = 0.0
    last_ranking_loss = 0.0
    for _ in range(epochs):
        for indices in _grouped_minibatches(
            group_ids,
            minibatch_size,
            generator,
        ):
            minibatch = [samples[index] for index in indices]
            batch = _action_batch(
                [sample.observation for sample in minibatch],
                [sample.action for sample in minibatch],
                [sample.target_aux for sample in minibatch],
                device,
            )
            values = model(
                batch["aircraft"],
                batch["global"],
                batch["targets"],
                batch["target_aux"],
                batch["high_actions"],
                batch["aircraft_actions"],
                batch["target_slots"],
            )
            mb_targets = targets[
                torch.tensor(indices, dtype=torch.long, device=device)
            ]
            regression_loss = F.smooth_l1_loss(
                values,
                mb_targets,
            )
            ranking_loss = pairwise_ranking_loss(
                values,
                mb_targets,
                [group_ids[index] for index in indices],
            )
            loss = regression_loss + ranking_loss_coef * ranking_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            last_regression_loss = float(regression_loss.item())
            last_ranking_loss = float(ranking_loss.item())

    changed_actor_parameters = [
        name
        for name, before in protected_before.items()
        if not torch.equal(
            before,
            protected_actor.state_dict()[name].detach().cpu(),
        )
    ]
    if changed_actor_parameters:
        raise RuntimeError(
            "action-value training changed actor parameters: "
            + ", ".join(changed_actor_parameters)
        )
    final_predictions = predict_action_values(
        model,
        samples,
        device,
        batch_size=minibatch_size,
    )
    final_mse = statistics.mean(
        (prediction - sample.target) ** 2
        for prediction, sample in zip(final_predictions, samples)
    )
    return {
        "samples": float(len(samples)),
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "last_batch_loss": last_loss,
        "last_regression_loss": last_regression_loss,
        "last_ranking_loss": last_ranking_loss,
    }


def _grouped_minibatches(
    group_ids: Sequence[Hashable],
    minibatch_size: int,
    generator: torch.Generator,
) -> List[List[int]]:
    grouped: Dict[Hashable, List[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(index)
    groups = list(grouped.values())
    order = torch.randperm(
        len(groups),
        generator=generator,
    ).tolist()
    minibatches: List[List[int]] = []
    current: List[int] = []
    for group_index in order:
        group = groups[group_index]
        if current and len(current) + len(group) > minibatch_size:
            minibatches.append(current)
            current = []
        current.extend(group)
    if current:
        minibatches.append(current)
    return minibatches


def action_value_metadata(
    samples: Sequence[ActionValueSample],
    source_actor_checkpoint: str,
    source_actor_sha256: str,
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("action-value metadata requires samples")
    intervals = sorted({sample.wave_interval for sample in samples})
    return {
        "version": ACTION_VALUE_VERSION,
        "checkpoint_type": ACTION_VALUE_CHECKPOINT_TYPE,
        "target_method": ACTION_VALUE_TARGET_METHOD,
        "source_actor_checkpoint": source_actor_checkpoint,
        "source_actor_sha256": source_actor_sha256,
        "training_wave_intervals": intervals,
        "wave_interval_range": [min(intervals), max(intervals)],
        "training_samples": len(samples),
        "training_groups": len(
            {sample.group_id for sample in samples}
        ),
    }


def validate_action_value_checkpoint(
    checkpoint_payload: Dict[str, Any],
    wave_interval: float,
    actor_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    extra = checkpoint_payload.get("extra", {})
    metadata = extra.get("action_value")
    if (
        extra.get("checkpoint_type") != ACTION_VALUE_CHECKPOINT_TYPE
        or not metadata
        or metadata.get("version") != ACTION_VALUE_VERSION
        or metadata.get("checkpoint_type")
        != ACTION_VALUE_CHECKPOINT_TYPE
        or metadata.get("target_method")
        != ACTION_VALUE_TARGET_METHOD
    ):
        raise ValueError(
            "action-value inference requires a dedicated "
            "ActionValueRanker checkpoint"
        )
    interval_range = metadata.get("wave_interval_range")
    if (
        not isinstance(interval_range, list)
        or len(interval_range) != 2
        or not float(interval_range[0])
        <= float(wave_interval)
        <= float(interval_range[1])
    ):
        raise ValueError(
            "action-value checkpoint is not calibrated for wave interval "
            f"{wave_interval:g}"
        )
    expected_actor_sha256 = metadata.get("source_actor_sha256")
    if (
        actor_sha256 is not None
        and expected_actor_sha256
        and actor_sha256 != expected_actor_sha256
    ):
        raise ValueError(
            "action-value checkpoint was initialized from a different "
            "actor checkpoint"
        )
    return metadata


def validate_action_value_evaluation_seeds(
    seeds: Sequence[int],
    checkpoint_payload: Dict[str, Any],
) -> List[int]:
    evaluation = validate_development_seeds(
        seeds,
        "action-value evaluation",
    )
    metadata = checkpoint_payload.get("extra", {}).get(
        "action_value",
        {},
    )
    if (
        "training_seeds" not in metadata
        or "selection_seeds" not in metadata
    ):
        raise ValueError(
            "action-value metadata must record training and selection seeds"
        )
    training = set(
        validate_development_seeds(
            metadata["training_seeds"],
            "action-value training",
        )
    )
    selection = set(
        validate_development_seeds(
            metadata["selection_seeds"],
            "action-value selection",
        )
    )
    partition_overlap = sorted(training & selection)
    if partition_overlap:
        raise ValueError(
            "action-value training and selection seeds overlap: "
            f"{partition_overlap}"
        )
    evaluation_overlap = sorted(
        set(evaluation) & (training | selection)
    )
    if evaluation_overlap:
        raise ValueError(
            "action-value evaluation seeds overlap training or selection "
            f"seeds: {evaluation_overlap}"
        )
    return evaluation


def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
