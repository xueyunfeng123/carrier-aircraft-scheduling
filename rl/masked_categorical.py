"""Categorical distribution helpers with invalid-action masking."""

from __future__ import annotations


def masked_logits(logits, mask):
    """Set invalid logits to a large negative value."""

    return logits.masked_fill(~mask.bool(), -1.0e9)


def masked_categorical(logits, mask):
    try:
        import torch
        from torch.distributions import Categorical
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for RL solver/training. Install it with `pip install torch`."
        ) from exc

    safe_mask = mask.bool()
    if not torch.all(safe_mask.any(dim=-1)):
        raise ValueError("masked_categorical received a row with no legal actions")
    return Categorical(logits=masked_logits(logits, safe_mask))

