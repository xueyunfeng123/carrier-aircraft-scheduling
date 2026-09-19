"""Measure policy parameter count and masked action-selection latency."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.model import CarrierPolicyValueNet
from rl.obs_encoder import (
    AIRCRAFT_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    NUM_EDGE_TYPES,
    NUM_NODE_TYPES,
    TARGET_AUX_FEATURE_DIM,
    TARGET_FEATURE_DIM,
    encode_observation,
)
from rl.ppo_trainer import PPOTrainer
from rl.train_config import PPOConfig


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def benchmark(
    encoder_type: str,
    env: CarrierAircraftSchedulingEnv,
    device: str,
    warmup: int,
    iterations: int,
) -> None:
    torch.manual_seed(41001)
    model = CarrierPolicyValueNet(
        AIRCRAFT_FEATURE_DIM,
        GLOBAL_FEATURE_DIM,
        target_feature_dim=(
            TARGET_FEATURE_DIM + TARGET_AUX_FEATURE_DIM
        ),
        encoder_type=encoder_type,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
    ).to(device)
    model.eval()
    trainer = PPOTrainer(
        model,
        optimizer=None,
        config=PPOConfig(),
        device=device,
    )
    encoded = encode_observation(env)

    for _ in range(warmup):
        trainer.select_action(encoded, env, deterministic=True)
    synchronize(device)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        trainer.select_action(encoded, env, deterministic=True)
        synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    print(
        f"{encoder_type},"
        f"parameters,{parameters},"
        f"mean_ms,{statistics.mean(timings):.4f},"
        f"median_ms,{statistics.median(timings):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    env = CarrierAircraftSchedulingEnv()
    env.reset(seed=args.seed)
    for encoder_type in ("deepsets", "hetero"):
        benchmark(
            encoder_type,
            env,
            args.device,
            args.warmup,
            args.iterations,
        )


if __name__ == "__main__":
    main()
