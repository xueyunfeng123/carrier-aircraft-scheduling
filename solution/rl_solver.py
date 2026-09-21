"""RL policy solver for inference through scripts.solve."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, Optional

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.obs_encoder import (
    AIRCRAFT_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    OBSERVATION_SCHEMA_VERSION,
    TARGET_AUX_FEATURE_DIM,
    TARGET_FEATURE_DIM,
    encode_observation,
    to_torch_batch,
)


class RLSolver:
    """Loads a PyTorch policy checkpoint and selects masked hierarchical actions."""

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        checkpoint: str = "",
        device: str = "cpu",
        deterministic: bool = True,
        hidden_dim: int = 128,
        aircraft_embed_dim: int = 64,
        target_embed_dim: int = 64,
        low_rank_prior: Optional[float] = None,
        target_rank_prior: Optional[float] = None,
        low_rank_prior_scale: float = 1.0,
        disable_low_rank_prior: bool = False,
        value_rerank_top_k: int = 1,
        value_rerank_seed: int = 0,
    ):
        try:
            import torch
            from rl.checkpoint import load_checkpoint
            from rl.model import CarrierPolicyValueNet
            from rl.ppo_trainer import PPOTrainer
            from rl.train_config import PPOConfig
            from rl.value_calibration import (
                step_with_calibration_reward,
                validate_rerank_calibration,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch is required for `--solver rl`. Install it with `pip install torch`."
            ) from exc

        self.env = env
        self.device = device
        self.deterministic = deterministic
        self.torch = torch
        self.value_rerank_top_k = int(value_rerank_top_k)
        self.value_rerank_seed = int(value_rerank_seed)
        self.value_rerank_decisions = 0
        if self.value_rerank_top_k < 1:
            raise ValueError("value_rerank_top_k must be positive")

        model_config = {
            "aircraft_feature_dim": AIRCRAFT_FEATURE_DIM,
            "global_feature_dim": GLOBAL_FEATURE_DIM,
            "hidden_dim": hidden_dim,
            "aircraft_embed_dim": aircraft_embed_dim,
            "target_feature_dim": (
                TARGET_FEATURE_DIM + TARGET_AUX_FEATURE_DIM
            ),
            "target_embed_dim": target_embed_dim,
        }
        checkpoint_path = Path(checkpoint) if checkpoint else None
        checkpoint_payload = None
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_payload = load_checkpoint(str(checkpoint_path), device=device)
            checkpoint_schema = checkpoint_payload.get("extra", {}).get(
                "observation_schema_version",
                1,
            )
            if checkpoint_schema != OBSERVATION_SCHEMA_VERSION:
                raise ValueError(
                    "RL checkpoint observation schema is incompatible with the "
                    "current environment; retrain BC+PPO"
                )
            checkpoint_model_config = checkpoint_payload.get("model_config", {})
            model_config.update(checkpoint_model_config)
            if "action_conditioned_low_head" not in checkpoint_model_config:
                model_config["action_conditioned_low_head"] = False
        if low_rank_prior is not None:
            model_config["low_rank_prior"] = float(low_rank_prior)
        if target_rank_prior is not None:
            model_config["target_rank_prior"] = float(target_rank_prior)

        self.model = CarrierPolicyValueNet(**model_config).to(device)
        if checkpoint_payload is not None:
            self.model.load_state_dict(checkpoint_payload["model_state"])
        self.model.low_rank_prior_scale = (
            0.0
            if disable_low_rank_prior
            else float(low_rank_prior_scale)
        )
        self.model.eval()

        # Reuse action selection code without an optimizer during inference.
        ppo_config_values = (
            checkpoint_payload.get("extra", {}).get("ppo_config", {})
            if checkpoint_payload is not None
            else {}
        )
        compatible_ppo_config = {
            key: value
            for key, value in ppo_config_values.items()
            if key in PPOConfig.__dataclass_fields__
        }
        self.ppo_config = PPOConfig(**compatible_ppo_config)
        self.trainer = PPOTrainer(
            self.model,
            optimizer=None,
            config=self.ppo_config,
            device=device,
        )
        self._step_with_calibration_reward = (
            step_with_calibration_reward
        )
        if self.value_rerank_top_k > 1:
            if checkpoint_payload is None:
                raise ValueError(
                    "value rerank requires an existing checkpoint"
                )
            validate_rerank_calibration(
                checkpoint_payload,
                float(self.env.config["wave_interval"]),
            )

    def choose_action(self) -> Optional[Dict[str, Any]]:
        encoded = encode_observation(self.env)
        if not any(encoded.high_mask):
            return None
        if self.value_rerank_top_k == 1:
            action, _, _ = self.trainer.select_action(
                encoded,
                self.env,
                deterministic=self.deterministic,
            )
        else:
            action = self._choose_value_reranked_action(encoded)
        action.pop("_target_mask", None)
        action.pop("_target_aux", None)
        return action

    def _choose_value_reranked_action(self, encoded) -> Dict[str, Any]:
        candidates = self.trainer.rank_actions(
            encoded,
            self.env,
            self.value_rerank_top_k,
        )
        scenario_seed = (
            self.value_rerank_seed
            + self.value_rerank_decisions
        )
        scored = [
            (
                self._one_step_value(
                    action,
                    scenario_seed,
                ),
                -rank,
                action,
            )
            for rank, (action, _) in enumerate(candidates)
        ]
        self.value_rerank_decisions += 1
        return max(scored, key=lambda item: (item[0], item[1]))[2]

    def _one_step_value(
        self,
        action: Dict[str, Any],
        scenario_seed: int,
    ) -> float:
        candidate_env = copy.deepcopy(self.env)
        # Do not expose the real environment's future random stream to reranking.
        candidate_env.rng = random.Random(scenario_seed)
        reward, done = self._step_with_calibration_reward(
            candidate_env,
            action,
            self.ppo_config,
        )
        steps = 1
        while not done and steps < 100_000:
            next_encoded = encode_observation(candidate_env)
            if any(next_encoded.high_mask):
                break
            event_reward, done = self._step_with_calibration_reward(
                candidate_env,
                None,
                self.ppo_config,
            )
            reward += event_reward
            steps += 1
        if done:
            return reward
        if steps >= 100_000:
            raise RuntimeError("value rerank one-step simulation did not advance")

        next_encoded = encode_observation(candidate_env)
        encoded_batch = to_torch_batch(next_encoded, self.device)
        with self.torch.no_grad():
            _, _, next_value = self.model(
                encoded_batch["aircraft"],
                encoded_batch["global"],
                encoded_batch["targets"],
                encoded_batch["low_aux"],
            )
        return reward + self.ppo_config.gamma * float(
            next_value.item()
        )
