"""Train a masked PPO policy for the carrier aircraft scheduling environment."""

from __future__ import annotations

import argparse
import statistics
from typing import Dict

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment.
    raise SystemExit("PyTorch is required for training. Install it with `pip install torch`.") from exc

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG
from rl.checkpoint import save_checkpoint
from rl.model import CarrierPolicyValueNet
from rl.obs_encoder import (
    AIRCRAFT_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    encode_observation,
    to_torch_batch,
)
from rl.ppo_trainer import PPOTrainer
from rl.rollout_buffer import RolloutBuffer
from rl.train_config import PPOConfig
from scripts.solve import build_config, run_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-aircraft", type=int, default=DEFAULT_CONFIG["num_aircraft"])
    parser.add_argument("--group-size", type=int, default=DEFAULT_CONFIG["group_size"])
    parser.add_argument("--num-parking-spots", type=int, default=DEFAULT_CONFIG["num_parking_spots"])
    parser.add_argument("--parking-base-transfer-time", type=float, default=DEFAULT_CONFIG["parking_base_transfer_time"])
    parser.add_argument("--parking-ring-time-step", type=float, default=DEFAULT_CONFIG["parking_ring_time_step"])
    parser.add_argument("--simulation-duration", type=float, default=DEFAULT_CONFIG["simulation_duration"])
    parser.add_argument("--wave-interval", type=float, default=DEFAULT_CONFIG["wave_interval"])
    parser.add_argument("--num-ammo-transport-vehicles", type=int, default=DEFAULT_CONFIG["num_ammo_transport_vehicles"])
    parser.add_argument("--num-lower-weapon-lifts", type=int, default=DEFAULT_CONFIG["num_lower_weapon_lifts"])
    parser.add_argument("--num-upper-weapon-lifts", type=int, default=DEFAULT_CONFIG["num_upper_weapon_lifts"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rl_policy.pt")
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--aircraft-embed-dim", type=int, default=64)
    parser.add_argument("--sortie-bonus", type=float, default=0.0)
    parser.add_argument("--miss-penalty", type=float, default=1.0)
    parser.add_argument("--eval-runs", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = build_config(args)
    ppo_config = PPOConfig(
        rollout_steps=args.rollout_steps,
        total_updates=args.total_updates,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_dim=args.hidden_dim,
        aircraft_embed_dim=args.aircraft_embed_dim,
        sortie_bonus=args.sortie_bonus,
        miss_penalty=args.miss_penalty,
    )

    env = CarrierAircraftSchedulingEnv(config)
    env.reset(seed=args.seed)
    model = CarrierPolicyValueNet(
        AIRCRAFT_FEATURE_DIM,
        GLOBAL_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        aircraft_embed_dim=args.aircraft_embed_dim,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = PPOTrainer(model, optimizer, ppo_config, device=args.device)

    for update in range(1, args.total_updates + 1):
        buffer = collect_rollout(env, trainer, ppo_config, args.seed + update)
        last_value = estimate_value(model, env, args.device)
        buffer.compute_gae(last_value, ppo_config.gamma, ppo_config.gae_lambda)
        stats = trainer.update(buffer)

        if update % args.save_every == 0 or update == 1 or update == args.total_updates:
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                extra={
                    "update": update,
                    "env_config": config,
                    "ppo_config": ppo_config.__dict__,
                },
            )

        if update % args.save_every == 0 or update == 1:
            eval_stats = evaluate_policy(config, args.checkpoint, args.seed, args.eval_runs, args.device)
            print(
                "update,"
                f"{update},"
                f"policy_loss,{stats['policy_loss']:.6f},"
                f"value_loss,{stats['value_loss']:.6f},"
                f"entropy,{stats['entropy']:.6f},"
                f"eval_completed,{eval_stats['completed']:.2f},"
                f"eval_missed,{eval_stats['missed']:.2f}"
            )


def collect_rollout(
    env: CarrierAircraftSchedulingEnv,
    trainer: PPOTrainer,
    config: PPOConfig,
    seed: int,
) -> RolloutBuffer:
    buffer = RolloutBuffer()
    pending_reward = 0.0
    if env.done:
        env.reset(seed=seed)

    while len(buffer) < config.rollout_steps:
        encoded = encode_observation(env)
        if not any(encoded.high_mask):
            reward, done = step_with_shaping(env, None, config)
            pending_reward += reward
            if done:
                env.reset(seed=seed + len(buffer))
                pending_reward = 0.0
            continue

        action, log_prob, value = trainer.select_action(encoded, deterministic=False)
        reward, done = step_with_shaping(env, action, config)
        buffer.add(
            encoded,
            action["high_level"],
            action["aircraft_id"],
            pending_reward + reward,
            done,
            value,
            log_prob,
        )
        pending_reward = 0.0
        if done:
            env.reset(seed=seed + len(buffer))
    return buffer


def step_with_shaping(env: CarrierAircraftSchedulingEnv, action, config: PPOConfig):
    before = env.get_evaluation_metrics()
    _, reward, done, _ = env.step(action)
    after = env.get_evaluation_metrics()
    delta_sorties = after["total_sorties_completed"] - before["total_sorties_completed"]
    delta_missed = after["total_missed_sorties"] - before["total_missed_sorties"]
    shaped_reward = reward + config.sortie_bonus * delta_sorties - config.miss_penalty * delta_missed
    return shaped_reward, done


def estimate_value(model, env: CarrierAircraftSchedulingEnv, device: str) -> float:
    if env.done:
        return 0.0
    encoded = encode_observation(env)
    batch = to_torch_batch(encoded, device)
    with torch.no_grad():
        _, _, value = model(batch["aircraft"], batch["global"])
    return float(value.item())


def evaluate_policy(config: Dict, checkpoint: str, seed: int, runs: int, device: str) -> Dict[str, float]:
    results = [
        run_episode(
            "rl",
            config,
            seed=seed + run_id,
            max_steps=100000,
            solver_options={
                "checkpoint": checkpoint,
                "device": device,
                "deterministic": True,
            },
        )
        for run_id in range(runs)
    ]
    return {
        "completed": statistics.mean(item["total_sorties_completed"] for item in results),
        "missed": statistics.mean(item["total_missed_sorties"] for item in results),
    }


if __name__ == "__main__":
    main()
