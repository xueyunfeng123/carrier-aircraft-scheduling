"""Train a masked PPO policy for the carrier aircraft scheduling environment."""

from __future__ import annotations

import argparse
import copy
import random
import statistics
from typing import Dict, List, Optional, Sequence, Union

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment.
    raise SystemExit("PyTorch is required for training. Install it with `pip install torch`.") from exc

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from env.config import DEFAULT_CONFIG
from rl.behavior_cloning import (
    Demonstrations,
    append_demonstration,
    build_demonstration_teacher,
    collect_heuristic_demonstrations,
    collect_self_imitation_demonstrations,
    pretrain_behavior_cloning,
)
from rl.checkpoint import load_checkpoint, save_checkpoint
from rl.model import CarrierPolicyValueNet
from rl.obs_encoder import (
    AIRCRAFT_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    OBSERVATION_SCHEMA_VERSION,
    TARGET_AUX_FEATURE_DIM,
    TARGET_FEATURE_DIM,
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
    parser.add_argument("--train-seeds", type=int, nargs="+")
    parser.add_argument("--validation-seeds", type=int, nargs="+")
    parser.add_argument(
        "--allow-seed-overlap",
        action="store_true",
        help="allow training and validation seeds to overlap for legacy overfit experiments",
    )
    parser.add_argument("--num-aircraft", type=int, default=DEFAULT_CONFIG["num_aircraft"])
    parser.add_argument(
        "--wave-size",
        "--group-size",
        dest="group_size",
        type=int,
        default=DEFAULT_CONFIG["group_size"],
    )
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
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--rank-gate-learning-rate",
        type=float,
        default=None,
        help="optional separate PPO learning rate for the adaptive rank gate",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--aircraft-embed-dim", type=int, default=64)
    parser.add_argument("--target-embed-dim", type=int, default=64)
    parser.add_argument("--low-rank-prior", type=float, default=4.0)
    parser.add_argument("--target-rank-prior", type=float, default=4.0)
    parser.add_argument(
        "--adaptive-low-rank-prior",
        action="store_true",
        help="learn state- and action-dependent weights for the aircraft rank feature",
    )
    parser.add_argument("--bc-episodes", type=int, default=5)
    parser.add_argument(
        "--bc-teacher",
        choices=("heuristic", "cp_sat", "elite_cp", "mixed"),
        default="heuristic",
    )
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-minibatch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--bc-low-loss-weight", type=float, default=1.0)
    parser.add_argument("--bc-target-loss-weight", type=float, default=1.0)
    parser.add_argument("--dagger-iterations", type=int, default=0)
    parser.add_argument("--dagger-episodes", type=int, default=1)
    parser.add_argument("--dagger-epochs", type=int, default=10)
    parser.add_argument("--self-imitation-iterations", type=int, default=0)
    parser.add_argument("--self-imitation-candidates", type=int, default=4)
    parser.add_argument(
        "--self-imitation-seeds-per-iteration",
        type=int,
        default=0,
    )
    parser.add_argument("--self-imitation-epochs", type=int, default=10)
    parser.add_argument(
        "--self-imitation-learning-rate",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--self-imitation-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--self-imitation-sample-levels",
        nargs="+",
        choices=("high", "low", "target"),
        default=["low"],
    )
    parser.add_argument(
        "--self-imitation-only-improvements",
        action="store_true",
    )
    parser.add_argument(
        "--self-imitation-parameter-scope",
        choices=("all", "low", "gate"),
        default="all",
    )
    parser.add_argument("--env-reward-scale", type=float, default=0.0)
    parser.add_argument("--sortie-bonus", type=float, default=1.0)
    parser.add_argument("--miss-penalty", type=float, default=0.0)
    parser.add_argument("--progress-shaping", type=float, default=0.0)
    parser.add_argument("--eval-runs", type=int, default=DEFAULT_EVALUATION_RUNS)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    training_seeds = resolve_training_seeds(
        args.seed,
        args.bc_episodes,
        args.train_seeds,
    )
    validation_seeds = resolve_validation_seeds(
        args.eval_seed,
        args.eval_runs,
        args.validation_seeds,
    )
    validate_seed_partition(
        training_seeds,
        validation_seeds,
        allow_overlap=args.allow_seed_overlap,
    )
    config = build_config(args)
    ppo_config = PPOConfig(
        rollout_steps=args.rollout_steps,
        total_updates=args.total_updates,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_dim=args.hidden_dim,
        aircraft_embed_dim=args.aircraft_embed_dim,
        env_reward_scale=args.env_reward_scale,
        sortie_bonus=args.sortie_bonus,
        miss_penalty=args.miss_penalty,
        progress_shaping=args.progress_shaping,
    )

    env = CarrierAircraftSchedulingEnv(config)
    rollout_seeds = EpisodeSeedScheduler(training_seeds)
    env.reset(seed=rollout_seeds.next_seed())
    model = CarrierPolicyValueNet(
        AIRCRAFT_FEATURE_DIM,
        GLOBAL_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        aircraft_embed_dim=args.aircraft_embed_dim,
        target_feature_dim=(
            TARGET_FEATURE_DIM + TARGET_AUX_FEATURE_DIM
        ),
        target_embed_dim=args.target_embed_dim,
        low_rank_prior=args.low_rank_prior,
        target_rank_prior=args.target_rank_prior,
        adaptive_low_rank_prior=args.adaptive_low_rank_prior,
    ).to(args.device)
    if args.init_checkpoint:
        initial_payload = load_checkpoint(
            args.init_checkpoint,
            device=args.device,
        )
        initial_schema = initial_payload.get("extra", {}).get(
            "observation_schema_version",
            1,
        )
        if initial_schema != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                "initial RL checkpoint observation schema is incompatible"
            )
        incompatible = model.load_state_dict(
            initial_payload["model_state"],
            strict=False,
        )
        allowed_missing = (
            {"low_rank_gate.weight", "low_rank_gate.bias"}
            if args.adaptive_low_rank_prior
            and not initial_payload.get("model_config", {}).get(
                "adaptive_low_rank_prior",
                False,
            )
            else set()
        )
        if (
            set(incompatible.missing_keys) != allowed_missing
            or incompatible.unexpected_keys
        ):
            raise ValueError(
                "initial checkpoint model architecture is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    bc_stats = None
    if args.bc_episodes > 0 and args.bc_epochs > 0:
        demonstrations = collect_heuristic_demonstrations(
            config,
            training_seeds,
            teacher=args.bc_teacher,
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
            low_loss_weight=args.bc_low_loss_weight,
            target_loss_weight=args.bc_target_loss_weight,
        )
        for dagger_iteration in range(args.dagger_iterations):
            dagger_demonstrations = collect_dagger_demonstrations(
                model,
                config,
                select_dagger_seeds(
                    training_seeds,
                    dagger_iteration,
                    args.dagger_episodes,
                ),
                args.device,
                args.bc_teacher,
                ppo_config,
            )
            demonstrations.extend(dagger_demonstrations)
            bc_stats = pretrain_behavior_cloning(
                model,
                bc_optimizer,
                demonstrations,
                epochs=args.dagger_epochs,
                minibatch_size=args.bc_minibatch_size,
                device=args.device,
                low_loss_weight=args.bc_low_loss_weight,
                target_loss_weight=args.bc_target_loss_weight,
            )
            print(
                "dagger,"
                f"iteration,{dagger_iteration + 1},"
                f"new_samples,{len(dagger_demonstrations)},"
                f"total_samples,{len(demonstrations)},"
                f"accuracy,{bc_stats['accuracy']:.6f},"
                f"high_accuracy,{bc_stats['high_accuracy']:.6f},"
                f"low_accuracy,{bc_stats['low_accuracy']:.6f},"
                f"target_accuracy,{bc_stats['target_accuracy']:.6f}"
            )
        print(
            "bc,"
            f"episodes,{args.bc_episodes},"
            f"samples,{len(demonstrations)},"
            f"loss,{bc_stats['loss']:.6f},"
            f"accuracy,{bc_stats['accuracy']:.6f},"
            f"high_accuracy,{bc_stats['high_accuracy']:.6f},"
            f"low_accuracy,{bc_stats['low_accuracy']:.6f},"
            f"target_accuracy,{bc_stats['target_accuracy']:.6f}"
        )
        del demonstrations, bc_optimizer

    self_imitation_stats = []
    if args.self_imitation_iterations > 0:
        self_imitation_optimizer = torch.optim.Adam(
            self_imitation_parameters(
                model,
                args.self_imitation_parameter_scope,
            ),
            lr=args.self_imitation_learning_rate,
        )
        seeds_per_iteration = (
            args.self_imitation_seeds_per_iteration
            or len(training_seeds)
        )
        for iteration in range(args.self_imitation_iterations):
            iteration_seeds = select_dagger_seeds(
                training_seeds,
                iteration,
                seeds_per_iteration,
            )
            demonstrations, rollout_stats = (
                collect_self_imitation_demonstrations(
                    model,
                    config,
                    iteration_seeds,
                    args.self_imitation_candidates,
                    args.device,
                    ppo_config,
                    args.seed
                    + iteration
                    * seeds_per_iteration
                    * args.self_imitation_candidates,
                    temperature=args.self_imitation_temperature,
                    stochastic_levels=(
                        args.self_imitation_sample_levels
                    ),
                    only_improvements=(
                        args.self_imitation_only_improvements
                    ),
                )
            )
            imitation_stats = pretrain_behavior_cloning(
                model,
                self_imitation_optimizer,
                demonstrations,
                epochs=args.self_imitation_epochs,
                minibatch_size=args.bc_minibatch_size,
                device=args.device,
                low_loss_weight=args.bc_low_loss_weight,
                target_loss_weight=args.bc_target_loss_weight,
            )
            iteration_stats = {
                **rollout_stats,
                **{
                    f"imitation_{key}": value
                    for key, value in imitation_stats.items()
                },
            }
            self_imitation_stats.append(iteration_stats)
            print(
                "self_imitation,"
                f"iteration,{iteration + 1},"
                f"seeds,{len(iteration_seeds)},"
                f"candidates,{args.self_imitation_candidates},"
                f"deterministic_mean,{rollout_stats['deterministic_mean']:.2f},"
                f"elite_mean,{rollout_stats['elite_mean']:.2f},"
                f"improved_seeds,{int(rollout_stats['improved_seeds'])},"
                f"selected_seeds,{int(rollout_stats['selected_seeds'])},"
                f"accuracy,{imitation_stats['accuracy']:.6f}"
            )
            del demonstrations
        del self_imitation_optimizer

    optimizer = build_ppo_optimizer(
        model,
        args.learning_rate,
        args.rank_gate_learning_rate,
    )
    trainer = PPOTrainer(model, optimizer, ppo_config, device=args.device)
    best_score = (
        float("-inf"),
        float("-inf"),
        float("-inf"),
    )
    best_model_state = None
    best_optimizer_state = None
    best_update = -1
    updates_completed = 0

    def checkpoint_extra(update: int) -> Dict:
        return {
            "update": update,
            "best_update": best_update,
            "updates_completed": updates_completed,
            "eval_seed": validation_seeds[0],
            "training_seeds": training_seeds,
            "validation_seeds": validation_seeds,
            "init_checkpoint": args.init_checkpoint,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "env_config": config,
            "ppo_config": ppo_config.__dict__,
            "rank_gate_learning_rate": args.rank_gate_learning_rate,
            "self_imitation": {
                "iterations": args.self_imitation_iterations,
                "candidates": args.self_imitation_candidates,
                "seeds_per_iteration": (
                    args.self_imitation_seeds_per_iteration
                ),
                "epochs": args.self_imitation_epochs,
                "learning_rate": args.self_imitation_learning_rate,
                "temperature": args.self_imitation_temperature,
                "sample_levels": args.self_imitation_sample_levels,
                "only_improvements": (
                    args.self_imitation_only_improvements
                ),
                "parameter_scope": (
                    args.self_imitation_parameter_scope
                ),
                "stats": self_imitation_stats,
            },
            "behavior_cloning": {
                "episodes": args.bc_episodes,
                "teacher": args.bc_teacher,
                "epochs": args.bc_epochs,
                "minibatch_size": args.bc_minibatch_size,
                "learning_rate": args.bc_learning_rate,
                "low_loss_weight": args.bc_low_loss_weight,
                "target_loss_weight": args.bc_target_loss_weight,
                "dagger_iterations": args.dagger_iterations,
                "dagger_episodes": args.dagger_episodes,
                "dagger_epochs": args.dagger_epochs,
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
            validation_seeds,
            args.device,
        )
        score = (
            eval_stats["completed"],
            eval_stats["worst_completed"],
            -eval_stats["missed"],
        )
        if score > best_score:
            best_score = score
            best_update = update
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        return eval_stats

    if bc_stats is not None or args.init_checkpoint:
        eval_stats = evaluate_and_track(0)
        print(
            "initial_eval,"
            f"completed,{eval_stats['completed']:.2f},"
            f"std,{eval_stats['completed_std']:.2f},"
            f"worst_completed,{eval_stats['worst_completed']:.2f},"
            f"missed,{eval_stats['missed']:.2f}"
        )

    for update in range(1, args.total_updates + 1):
        buffer = collect_rollout(
            env,
            trainer,
            ppo_config,
            rollout_seeds,
        )
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
                f"eval_worst_completed,{eval_stats['worst_completed']:.2f},"
                f"eval_missed,{eval_stats['missed']:.2f}"
            )

    if best_model_state is not None:
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
            f"eval_worst_completed,{best_score[1]:.2f},"
            f"eval_missed,{-best_score[2]:.2f}"
        )


def resolve_training_seeds(
    seed: int,
    bc_episodes: int,
    explicit_seeds: Optional[Sequence[int]],
) -> List[int]:
    if explicit_seeds:
        return list(dict.fromkeys(int(value) for value in explicit_seeds))
    return [
        int(seed) + run_id
        for run_id in range(max(1, int(bc_episodes)))
    ]


def resolve_validation_seeds(
    eval_seed: int,
    eval_runs: int,
    explicit_seeds: Optional[Sequence[int]],
) -> List[int]:
    if explicit_seeds:
        return list(dict.fromkeys(int(value) for value in explicit_seeds))
    return [
        int(eval_seed) + run_id
        for run_id in range(max(1, int(eval_runs)))
    ]


def validate_seed_partition(
    training_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    allow_overlap: bool = False,
) -> None:
    overlap = sorted(set(training_seeds) & set(validation_seeds))
    if overlap and not allow_overlap:
        raise ValueError(
            "training and validation seeds overlap: "
            f"{overlap}; use --allow-seed-overlap only for legacy overfit runs"
        )


def select_dagger_seeds(
    training_seeds: Sequence[int],
    iteration: int,
    episodes: int,
) -> List[int]:
    return [
        int(training_seeds[
            (iteration * episodes + run_id) % len(training_seeds)
        ])
        for run_id in range(episodes)
    ]


class EpisodeSeedScheduler:
    def __init__(self, seeds: Sequence[int]):
        if not seeds:
            raise ValueError("at least one training seed is required")
        self.seeds = [int(seed) for seed in seeds]
        self.index = 0

    def next_seed(self) -> int:
        seed = self.seeds[self.index % len(self.seeds)]
        self.index += 1
        return seed


def build_ppo_optimizer(
    model,
    learning_rate: float,
    rank_gate_learning_rate: Optional[float] = None,
):
    if (
        rank_gate_learning_rate is None
        or not getattr(model, "adaptive_low_rank_prior", False)
    ):
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    gate_parameters = list(model.low_rank_gate.parameters())
    gate_parameter_ids = {
        id(parameter)
        for parameter in gate_parameters
    }
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in gate_parameter_ids
    ]
    return torch.optim.Adam(
        [
            {
                "params": base_parameters,
                "lr": learning_rate,
            },
            {
                "params": gate_parameters,
                "lr": rank_gate_learning_rate,
            },
        ]
    )


def self_imitation_parameters(model, scope: str):
    if scope == "all":
        return list(model.parameters())
    if scope == "gate":
        if not getattr(model, "adaptive_low_rank_prior", False):
            raise ValueError(
                "gate-only self-imitation requires "
                "--adaptive-low-rank-prior"
            )
        return list(model.low_rank_gate.parameters())
    if scope == "low":
        modules = [
            model.low_context,
            model.low_head,
        ]
        if getattr(model, "adaptive_low_rank_prior", False):
            modules.append(model.low_rank_gate)
        return [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]
    raise ValueError(f"unknown self-imitation parameter scope: {scope}")


def collect_rollout(
    env: CarrierAircraftSchedulingEnv,
    trainer: PPOTrainer,
    config: PPOConfig,
    seed: Union[int, EpisodeSeedScheduler],
) -> RolloutBuffer:
    buffer = RolloutBuffer()
    episode_index = 0

    def next_seed() -> int:
        nonlocal episode_index
        if isinstance(seed, EpisodeSeedScheduler):
            return seed.next_seed()
        selected = int(seed) + episode_index
        episode_index += 1
        return selected

    if env.done:
        env.reset(seed=next_seed())

    actions_in_episode = 0
    while len(buffer) < config.rollout_steps:
        encoded = encode_observation(env)
        if not any(encoded.high_mask):
            _, done = step_with_shaping(env, None, config)
            if done:
                if actions_in_episode == 0:
                    raise RuntimeError(
                        "episode ended without any legal scheduling action"
                    )
                env.reset(seed=next_seed())
                actions_in_episode = 0
            continue

        action, log_prob, value = trainer.select_action(
            encoded,
            env,
            deterministic=False,
        )
        actions_in_episode += 1
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
            action["target_slot"],
            action["_target_mask"],
            action["_target_aux"],
            reward,
            done,
            value,
            log_prob,
        )
        if done:
            env.reset(seed=next_seed())
            actions_in_episode = 0
    return buffer


def collect_dagger_demonstrations(
    model,
    env_config: Dict,
    seeds,
    device: str,
    teacher_name: str,
    ppo_config: PPOConfig,
) -> Demonstrations:
    demonstrations = Demonstrations()
    learner = PPOTrainer(
        model,
        optimizer=None,
        config=ppo_config,
        device=device,
    )
    model.eval()
    for seed in seeds:
        env = CarrierAircraftSchedulingEnv(env_config)
        env.reset(seed=seed)
        teacher = (
            build_demonstration_teacher(
                env,
                "cp_sat" if seed % 2 == 0 else "heuristic",
            )
            if teacher_name == "mixed"
            else build_demonstration_teacher(env, teacher_name)
        )
        steps = 0
        while not env.done and steps < 100_000:
            encoded = encode_observation(env)
            if not any(encoded.high_mask):
                env.step(None)
                steps += 1
                continue
            teacher_action = teacher.choose_action()
            if teacher_action is None:
                raise RuntimeError(
                    "demonstration teacher returned no legal action"
                )
            append_demonstration(
                demonstrations,
                env,
                encoded,
                teacher_action,
            )
            learner_action, _, _ = learner.select_action(
                encoded,
                env,
                deterministic=True,
            )
            env.step(learner_action)
            steps += 1
        if not env.done:
            raise RuntimeError(
                "DAgger demonstration exceeded 100000 steps"
            )
    return demonstrations


def step_with_shaping(env: CarrierAircraftSchedulingEnv, action, config: PPOConfig):
    before = env.get_evaluation_metrics()
    before_progress = readiness_potential(env)
    _, reward, done, _ = env.step(action)
    after = env.get_evaluation_metrics()
    after_progress = 0.0 if done else readiness_potential(env)
    delta_sorties = after["total_sorties_completed"] - before["total_sorties_completed"]
    delta_missed = after["total_missed_sorties"] - before["total_missed_sorties"]
    shaped_reward = (
        config.env_reward_scale * reward
        + config.sortie_bonus * delta_sorties
        - config.miss_penalty * delta_missed
        + config.progress_shaping
        * (config.gamma * after_progress - before_progress)
    )
    return shaped_reward, done


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


def estimate_value(model, env: CarrierAircraftSchedulingEnv, device: str) -> float:
    if env.done:
        return 0.0
    encoded = encode_observation(env)
    batch = to_torch_batch(encoded, device)
    with torch.no_grad():
        _, _, value = model(
            batch["aircraft"],
            batch["global"],
            batch["targets"],
            batch["low_aux"],
        )
    return float(value.item())


def evaluate_policy(
    config: Dict,
    checkpoint: str,
    seeds: Sequence[int],
    device: str,
) -> Dict[str, float]:
    results = [
        run_episode(
            "rl",
            config,
            seed=seed,
            max_steps=100000,
            solver_options={
                "checkpoint": checkpoint,
                "device": device,
                "deterministic": True,
            },
        )
        for seed in seeds
    ]
    completed = [
        item["total_sorties_completed"]
        for item in results
    ]
    return {
        "completed": statistics.mean(completed),
        "completed_std": (
            statistics.stdev(completed)
            if len(completed) > 1
            else 0.0
        ),
        "worst_completed": min(completed),
        "missed": statistics.mean(item["total_missed_sorties"] for item in results),
    }


if __name__ == "__main__":
    main()
