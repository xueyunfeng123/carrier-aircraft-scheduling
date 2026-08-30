"""Train a masked PPO policy for the carrier aircraft scheduling environment."""

from __future__ import annotations

import argparse
import copy
import random
import statistics
from typing import Dict

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment.
    raise SystemExit("PyTorch is required for training. Install it with `pip install torch`.") from exc

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG
from rl.behavior_cloning import (
    collect_heuristic_demonstrations,
    pretrain_behavior_cloning,
)
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
from scripts.evaluation_defaults import (
    DEFAULT_EVALUATION_DURATION,
    DEFAULT_EVALUATION_RUNS,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVALUATION_WAVE_INTERVAL,
    DEFAULT_TRAINING_SEED,
)
from scripts.solve import build_config, run_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--num-aircraft", type=int, default=DEFAULT_CONFIG["num_aircraft"])
    parser.add_argument("--group-size", type=int, default=DEFAULT_CONFIG["group_size"])
    parser.add_argument("--num-parking-spots", type=int, default=DEFAULT_CONFIG["num_parking_spots"])
    parser.add_argument("--parking-base-transfer-time", type=float, default=DEFAULT_CONFIG["parking_base_transfer_time"])
    parser.add_argument("--parking-ring-time-step", type=float, default=DEFAULT_CONFIG["parking_ring_time_step"])
    parser.add_argument("--simulation-duration", type=float, default=DEFAULT_EVALUATION_DURATION)
    parser.add_argument("--wave-interval", type=float, default=DEFAULT_EVALUATION_WAVE_INTERVAL)
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
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--aircraft-embed-dim", type=int, default=64)
    parser.add_argument("--bc-episodes", type=int, default=5)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-minibatch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--env-reward-scale", type=float, default=0.0)
    parser.add_argument("--sortie-bonus", type=float, default=1.0)
    parser.add_argument("--miss-penalty", type=float, default=0.0)
    parser.add_argument("--eval-runs", type=int, default=DEFAULT_EVALUATION_RUNS)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = build_config(args)
    ppo_config = PPOConfig(
        rollout_steps=args.rollout_steps,
        total_updates=args.total_updates,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_dim=args.hidden_dim,
        aircraft_embed_dim=args.aircraft_embed_dim,
        env_reward_scale=args.env_reward_scale,
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

    bc_stats = None
    if args.bc_episodes > 0 and args.bc_epochs > 0:
        demonstration_seeds = [
            args.seed + run_id
            for run_id in range(args.bc_episodes)
        ]
        demonstrations = collect_heuristic_demonstrations(
            config,
            demonstration_seeds,
        )
        bc_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.bc_learning_rate,
        )
        bc_stats = pretrain_behavior_cloning(
            model,
            bc_optimizer,
            demonstrations,
            epochs=args.bc_epochs,
            minibatch_size=args.bc_minibatch_size,
            device=args.device,
        )
        print(
            "bc,"
            f"episodes,{args.bc_episodes},"
            f"samples,{len(demonstrations)},"
            f"loss,{bc_stats['loss']:.6f},"
            f"accuracy,{bc_stats['accuracy']:.6f}"
        )
        del demonstrations, bc_optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = PPOTrainer(model, optimizer, ppo_config, device=args.device)
    best_score = (float("-inf"), float("-inf"))
    best_model_state = None
    best_optimizer_state = None
    best_update = -1
    updates_completed = 0

    def checkpoint_extra(update: int) -> Dict:
        return {
            "update": update,
            "best_update": best_update,
            "updates_completed": updates_completed,
            "eval_seed": args.eval_seed,
            "env_config": config,
            "ppo_config": ppo_config.__dict__,
            "behavior_cloning": {
                "episodes": args.bc_episodes,
                "epochs": args.bc_epochs,
                "minibatch_size": args.bc_minibatch_size,
                "learning_rate": args.bc_learning_rate,
                "stats": bc_stats,
            },
        }

    def evaluate_and_track(update: int) -> Dict[str, float]:
        nonlocal best_model_state, best_optimizer_state, best_score, best_update
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            extra=checkpoint_extra(update),
        )
        eval_stats = evaluate_policy(
            config,
            args.checkpoint,
            args.eval_seed,
            args.eval_runs,
            args.device,
        )
        score = (eval_stats["completed"], -eval_stats["missed"])
        if score > best_score:
            best_score = score
            best_update = update
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        return eval_stats

    if bc_stats is not None:
        eval_stats = evaluate_and_track(0)
        print(
            "bc_eval,"
            f"completed,{eval_stats['completed']:.2f},"
            f"missed,{eval_stats['missed']:.2f}"
        )

    for update in range(1, args.total_updates + 1):
        buffer = collect_rollout(env, trainer, ppo_config, args.seed + update)
        last_value = estimate_value(model, env, args.device)
        buffer.compute_gae(last_value, ppo_config.gamma, ppo_config.gae_lambda)
        stats = trainer.update(buffer)
        updates_completed = update

        should_evaluate = (
            update % args.save_every == 0
            or update == 1
            or update == args.total_updates
        )
        if should_evaluate:
            eval_stats = evaluate_and_track(update)
            print(
                "update,"
                f"{update},"
                f"policy_loss,{stats['policy_loss']:.6f},"
                f"value_loss,{stats['value_loss']:.6f},"
                f"entropy,{stats['entropy']:.6f},"
                f"eval_completed,{eval_stats['completed']:.2f},"
                f"eval_missed,{eval_stats['missed']:.2f}"
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            extra=checkpoint_extra(best_update),
        )
        print(
            "best,"
            f"update,{best_update},"
            f"eval_completed,{best_score[0]:.2f},"
            f"eval_missed,{-best_score[1]:.2f}"
        )


def collect_rollout(
    env: CarrierAircraftSchedulingEnv,
    trainer: PPOTrainer,
    config: PPOConfig,
    seed: int,
) -> RolloutBuffer:
    buffer = RolloutBuffer()
    if env.done:
        env.reset(seed=seed)

    while len(buffer) < config.rollout_steps:
        encoded = encode_observation(env)
        if not any(encoded.high_mask):
            _, done = step_with_shaping(env, None, config)
            if done:
                env.reset(seed=seed + len(buffer))
            continue

        action, log_prob, value = trainer.select_action(encoded, deterministic=False)
        reward, done = step_with_shaping(env, action, config)
        while not done:
            next_encoded = encode_observation(env)
            if any(next_encoded.high_mask):
                break
            event_reward, done = step_with_shaping(env, None, config)
            reward += event_reward

        buffer.add(
            encoded,
            action["high_level"],
            action["aircraft_id"],
            reward,
            done,
            value,
            log_prob,
        )
        if done:
            env.reset(seed=seed + len(buffer))
    return buffer


def step_with_shaping(env: CarrierAircraftSchedulingEnv, action, config: PPOConfig):
    before = env.get_evaluation_metrics()
    _, reward, done, _ = env.step(action)
    after = env.get_evaluation_metrics()
    delta_sorties = after["total_sorties_completed"] - before["total_sorties_completed"]
    delta_missed = after["total_missed_sorties"] - before["total_missed_sorties"]
    shaped_reward = (
        config.env_reward_scale * reward
        + config.sortie_bonus * delta_sorties
        - config.miss_penalty * delta_missed
    )
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
