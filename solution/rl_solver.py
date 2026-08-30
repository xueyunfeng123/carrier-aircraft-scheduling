"""RL policy solver for inference through scripts.solve."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.obs_encoder import AIRCRAFT_FEATURE_DIM, GLOBAL_FEATURE_DIM, encode_observation


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
        }
        checkpoint_path = Path(checkpoint) if checkpoint else None
        checkpoint_payload = None
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_payload = load_checkpoint(str(checkpoint_path), device=device)
            model_config.update(checkpoint_payload.get("model_config", {}))

        self.model = CarrierPolicyValueNet(**model_config).to(device)
        if checkpoint_payload is not None:
            self.model.load_state_dict(checkpoint_payload["model_state"])
        self.model.eval()

        # Reuse action selection code without an optimizer during inference.
        self.trainer = PPOTrainer(
            self.model,
            optimizer=None,
            config=PPOConfig(),
            device=device,
        )

    def choose_action(self) -> Optional[Dict[str, int]]:
        encoded = encode_observation(self.env)
        if not any(encoded.high_mask):
            return None
        action, _, _ = self.trainer.select_action(encoded, deterministic=self.deterministic)
        return action
