"""RL policy solver for inference through scripts.solve."""

from __future__ import annotations

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
    ):
        try:
            import torch
            from rl.checkpoint import load_checkpoint
            from rl.model import CarrierPolicyValueNet
            from rl.ppo_trainer import PPOTrainer
            from rl.train_config import PPOConfig
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch is required for `--solver rl`. Install it with `pip install torch`."
            ) from exc

        self.env = env
        self.device = device
        self.deterministic = deterministic
        self.torch = torch

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
        self.trainer = PPOTrainer(
            self.model,
            optimizer=None,
            config=PPOConfig(),
            device=device,
        )

    def choose_action(self) -> Optional[Dict[str, Any]]:
        encoded = encode_observation(self.env)
        if not any(encoded.high_mask):
            return None
        action, _, _ = self.trainer.select_action(
            encoded,
            self.env,
            deterministic=self.deterministic,
        )
        action.pop("_target_mask", None)
        action.pop("_target_aux", None)
        return action
