"""Checkpoint helpers for RL policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def save_checkpoint(path: str, model, optimizer=None, extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for RL solver/training. Install it with `pip install torch`."
        ) from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.checkpoint_config(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, target)


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for RL solver/training. Install it with `pip install torch`."
        ) from exc

    return torch.load(path, map_location=device)

