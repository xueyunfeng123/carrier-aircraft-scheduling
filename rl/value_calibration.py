"""Offline critic calibration from complete policy trajectories."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.nn import functional as F

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.obs_encoder import EncodedObservation, batch_to_torch, encode_observation
from rl.ppo_trainer import PPOTrainer
from rl.train_config import PPOConfig


VALUE_CALIBRATION_VERSION = 1


@dataclass
class ValueTrajectory:
    observations: List[EncodedObservation]
    rewards: List[float]
    values: List[float]
    time_fractions: List[float]
    seed: int
    wave_interval: float


@dataclass
class ValueSample:
    observation: EncodedObservation
    target: float
    seed: int
    wave_interval: float
    time_fraction: float


def compute_value_targets(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    method: str = "mc",
    gae_lambda: float = 0.98,
) -> List[float]:
    """Compute terminal Monte Carlo or GAE lambda-return targets."""

    if len(rewards) != len(values):
        raise ValueError("rewards and values must have equal lengths")
    if method not in ("mc", "gae"):
        raise ValueError("value target method must be 'mc' or 'gae'")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")

    if method == "mc":
        targets = [0.0] * len(rewards)
        return_so_far = 0.0
        for index in reversed(range(len(rewards))):
            return_so_far = float(rewards[index]) + gamma * return_so_far
            targets[index] = return_so_far
        return targets

    targets = [0.0] * len(rewards)
    advantage = 0.0
    next_value = 0.0
    for index in reversed(range(len(rewards))):
        value = float(values[index])
        delta = float(rewards[index]) + gamma * next_value - value
        advantage = delta + gamma * gae_lambda * advantage
        targets[index] = value + advantage
        next_value = value
    return targets


def transition_reward(
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    environment_reward: float,
    before_progress: float,
    after_progress: float,
    done: bool,
    config: PPOConfig,
) -> float:
    """Apply the same reward definition used by PPO training."""

    delta_sorties = (
        after_metrics["total_sorties_completed"]
        - before_metrics["total_sorties_completed"]
    )
    delta_missed = (
        after_metrics["total_missed_sorties"]
        - before_metrics["total_missed_sorties"]
    )
    terminal_progress = 0.0 if done else after_progress
    return float(
        config.env_reward_scale * environment_reward
        + config.sortie_bonus * delta_sorties
        - config.miss_penalty * delta_missed
        + config.progress_shaping
        * (config.gamma * terminal_progress - before_progress)
    )


def readiness_potential(env: CarrierAircraftSchedulingEnv) -> float:
    service_progress = sum(
        int(aircraft.fuel_status == 2)
        + int(aircraft.inspection_status == 2)
        + int(aircraft.arm_status == 2)
        for aircraft in env.aircraft
        if not aircraft.is_airborne
        and aircraft.recovery_status == 2
    )
    completed_sorties = sum(
        aircraft.sorties_completed
        for aircraft in env.aircraft
    )
    return float(3 * completed_sorties + service_progress)


def step_with_calibration_reward(
    env: CarrierAircraftSchedulingEnv,
    action,
    config: PPOConfig,
) -> tuple[float, bool]:
    before_metrics = env.get_evaluation_metrics()
    before_progress = readiness_potential(env)
    _, environment_reward, done, _ = env.step(action)
    after_metrics = env.get_evaluation_metrics()
    after_progress = readiness_potential(env)
    return (
        transition_reward(
            before_metrics,
            after_metrics,
            environment_reward,
            before_progress,
            after_progress,
            done,
            config,
        ),
        done,
    )


def collect_complete_trajectory(
    model,
    env_config: Dict[str, Any],
    seed: int,
    device: str,
    reward_config: PPOConfig,
    deterministic: bool = True,
    max_steps: int = 100_000,
) -> ValueTrajectory:
    """Collect every decision state until a true terminal state."""

    env = CarrierAircraftSchedulingEnv(env_config)
    env.reset(seed=int(seed))
    trainer = PPOTrainer(
        model,
        optimizer=None,
        config=reward_config,
        device=device,
    )
    trajectory = ValueTrajectory(
        observations=[],
        rewards=[],
        values=[],
        time_fractions=[],
        seed=int(seed),
        wave_interval=float(env.config["wave_interval"]),
    )
    model.eval()
    steps = 0

    while not env.done and steps < max_steps:
        encoded = encode_observation(env)
        if not any(encoded.high_mask):
            reward, _ = step_with_calibration_reward(
                env,
                None,
                reward_config,
            )
            if trajectory.rewards:
                trajectory.rewards[-1] += reward
            steps += 1
            continue

        action, _, value = trainer.select_action(
            encoded,
            env,
            deterministic=deterministic,
        )
        trajectory.observations.append(encoded)
        trajectory.values.append(value)
        trajectory.time_fractions.append(
            float(env.time)
            / max(1.0, float(env.config["simulation_duration"]))
        )
        reward, done = step_with_calibration_reward(
            env,
            action,
            reward_config,
        )
        steps += 1
        while not done and steps < max_steps:
            next_encoded = encode_observation(env)
            if any(next_encoded.high_mask):
                break
            event_reward, done = step_with_calibration_reward(
                env,
                None,
                reward_config,
            )
            reward += event_reward
            steps += 1
        trajectory.rewards.append(reward)

    if not env.done:
        raise RuntimeError(
            f"value trajectory exceeded {max_steps} steps for seed {seed}"
        )
    if not trajectory.observations:
        raise RuntimeError(
            f"value trajectory has no decision states for seed {seed}"
        )
    return trajectory


def trajectories_to_samples(
    trajectories: Iterable[ValueTrajectory],
    gamma: float,
    method: str,
    gae_lambda: float,
) -> List[ValueSample]:
    samples: List[ValueSample] = []
    for trajectory in trajectories:
        targets = compute_value_targets(
            trajectory.rewards,
            trajectory.values,
            gamma,
            method=method,
            gae_lambda=gae_lambda,
        )
        samples.extend(
            ValueSample(
                observation=observation,
                target=target,
                seed=trajectory.seed,
                wave_interval=trajectory.wave_interval,
                time_fraction=time_fraction,
            )
            for observation, target, time_fraction in zip(
                trajectory.observations,
                targets,
                trajectory.time_fractions,
            )
        )
    return samples


def collect_value_samples(
    model,
    configs: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    device: str,
    reward_config: PPOConfig,
    target_method: str = "mc",
    gae_lambda: float = 0.98,
    deterministic: bool = True,
    max_steps: int = 100_000,
) -> List[ValueSample]:
    trajectories = [
        collect_complete_trajectory(
            model,
            config,
            int(seed),
            device,
            reward_config,
            deterministic=deterministic,
            max_steps=max_steps,
        )
        for config in configs
        for seed in seeds
    ]
    return trajectories_to_samples(
        trajectories,
        reward_config.gamma,
        target_method,
        gae_lambda,
    )


def predict_values(
    model,
    samples: Sequence[ValueSample],
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
            batch = batch_to_torch(
                [sample.observation for sample in minibatch],
                device,
            )
            _, _, values = model(
                batch["aircraft"],
                batch["global"],
                batch["targets"],
                batch["low_aux"],
            )
            predictions.extend(
                float(value)
                for value in values.detach().cpu().tolist()
            )
    return predictions


def calibrate_value_head(
    model,
    samples: Sequence[ValueSample],
    device: str,
    learning_rate: float = 1.0e-3,
    epochs: int = 100,
    minibatch_size: int = 512,
    loss_name: str = "huber",
    seed: int = 0,
    parameter_scope: str = "head",
) -> Dict[str, float]:
    """Fit the value head, optionally with its shared state encoders."""

    if not samples:
        raise ValueError("at least one value sample is required")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if minibatch_size < 1:
        raise ValueError("minibatch_size must be positive")
    if loss_name not in ("huber", "mse"):
        raise ValueError("loss_name must be 'huber' or 'mse'")
    if parameter_scope not in ("head", "encoder"):
        raise ValueError("parameter_scope must be 'head' or 'encoder'")

    trainable_prefixes = ("value_head.",)
    if parameter_scope == "encoder":
        trainable_prefixes = (
            "aircraft_encoder.",
            "target_encoder.",
            "global_encoder.",
            "value_head.",
        )
    frozen_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith(trainable_prefixes)
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith(trainable_prefixes):
            parameter.requires_grad_(True)
            trainable_parameters.append(parameter)

    batch = batch_to_torch(
        [sample.observation for sample in samples],
        device,
    )
    targets = torch.tensor(
        [sample.target for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    initial_predictions = predict_values(
        model,
        samples,
        device,
        batch_size=minibatch_size,
    )
    initial_loss = statistics.mean(
        (prediction - sample.target) ** 2
        for prediction, sample in zip(initial_predictions, samples)
    )

    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        order = torch.randperm(
            len(samples),
            generator=generator,
        ).tolist()
        for start in range(0, len(order), minibatch_size):
            indices = order[start : start + minibatch_size]
            mb = torch.tensor(
                indices,
                dtype=torch.long,
                device=device,
            )
            _, _, values = model(
                batch["aircraft"][mb],
                batch["global"][mb],
                batch["targets"][mb],
                batch["low_aux"][mb],
            )
            if loss_name == "huber":
                loss = F.smooth_l1_loss(values, targets[mb])
            else:
                loss = F.mse_loss(values, targets[mb])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    state_after = model.state_dict()
    changed_frozen_parameters = [
        name
        for name, before in frozen_before.items()
        if not torch.equal(before, state_after[name].detach().cpu())
    ]
    if changed_frozen_parameters:
        raise RuntimeError(
            "critic calibration changed frozen parameters: "
            + ", ".join(changed_frozen_parameters)
        )
    model.eval()
    final_predictions = predict_values(
        model,
        samples,
        device,
        batch_size=minibatch_size,
    )
    final_loss = statistics.mean(
        (prediction - sample.target) ** 2
        for prediction, sample in zip(final_predictions, samples)
    )
    return {
        "samples": float(len(samples)),
        "initial_mse": initial_loss,
        "final_mse": final_loss,
        "last_batch_loss": last_loss,
    }


def calibration_metric_rows(
    samples: Sequence[ValueSample],
    predictions: Sequence[float],
) -> List[Dict[str, Any]]:
    """Return overall and load/time-stratified critic metrics."""

    if len(samples) != len(predictions):
        raise ValueError("samples and predictions must have equal lengths")
    if not samples:
        return []

    grouped: Dict[tuple[str, str], List[int]] = {
        ("overall", "all"): list(range(len(samples))),
    }
    for index, sample in enumerate(samples):
        load = f"{sample.wave_interval:g}"
        horizon = _time_bucket(sample.time_fraction)
        grouped.setdefault(("load", load), []).append(index)
        grouped.setdefault(("horizon", horizon), []).append(index)
        grouped.setdefault(
            ("load_horizon", f"{load}:{horizon}"),
            [],
        ).append(index)

    rows = []
    for (group_type, group), indices in grouped.items():
        targets = [samples[index].target for index in indices]
        estimates = [float(predictions[index]) for index in indices]
        errors = [
            estimate - target
            for estimate, target in zip(estimates, targets)
        ]
        rows.append(
            {
                "group_type": group_type,
                "group": group,
                "count": len(indices),
                "mae": statistics.mean(abs(error) for error in errors),
                "rmse": math.sqrt(
                    statistics.mean(error * error for error in errors)
                ),
                "bias": statistics.mean(errors),
                "pearson_r": _pearson_correlation(targets, estimates),
                "spearman_r": _pearson_correlation(
                    _ranks(targets),
                    _ranks(estimates),
                ),
            }
        )
    return rows


def calibration_metadata(
    samples: Sequence[ValueSample],
    target_method: str,
    reward_config: PPOConfig,
    gae_lambda: float,
) -> Dict[str, Any]:
    intervals = sorted({sample.wave_interval for sample in samples})
    return {
        "version": VALUE_CALIBRATION_VERSION,
        "target_method": target_method,
        "gamma": reward_config.gamma,
        "gae_lambda": gae_lambda,
        "reward_config": {
            "env_reward_scale": reward_config.env_reward_scale,
            "sortie_bonus": reward_config.sortie_bonus,
            "miss_penalty": reward_config.miss_penalty,
            "progress_shaping": reward_config.progress_shaping,
        },
        "training_wave_intervals": intervals,
        "wave_interval_range": [min(intervals), max(intervals)],
        "training_samples": len(samples),
    }


def validate_rerank_calibration(
    checkpoint_payload: Dict[str, Any],
    wave_interval: float,
) -> Dict[str, Any]:
    extra = checkpoint_payload.get("extra", {})
    metadata = extra.get(
        "value_calibration"
    )
    if not metadata or metadata.get("version") != VALUE_CALIBRATION_VERSION:
        raise ValueError(
            "value rerank requires a checkpoint with value_calibration metadata"
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
            "value rerank checkpoint is not calibrated for wave interval "
            f"{wave_interval:g}"
        )
    stored_config = extra.get("ppo_config", {})
    calibrated_config = {
        "gamma": metadata.get("gamma"),
        **metadata.get("reward_config", {}),
    }
    mismatches = [
        key
        for key, calibrated_value in calibrated_config.items()
        if calibrated_value is None
        or key not in stored_config
        or not math.isclose(
            float(calibrated_value),
            float(stored_config[key]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    if mismatches:
        raise ValueError(
            "value calibration reward config does not match checkpoint "
            f"ppo_config: {', '.join(mismatches)}"
        )
    return metadata


def _time_bucket(time_fraction: float) -> str:
    if time_fraction < 1.0 / 3.0:
        return "early"
    if time_fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def _pearson_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) < 2:
        return 0.0
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
    )
    right_scale = math.sqrt(
        sum((value - right_mean) ** 2 for value in right)
    )
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _ranks(values: Sequence[float]) -> List[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while (
            end < len(ordered)
            and values[ordered[end]] == values[ordered[start]]
        ):
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average_rank
        start = end
    return ranks
