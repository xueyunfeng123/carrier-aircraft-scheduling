"""Policy-value network for hierarchical masked scheduling actions."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without torch installed.
    raise ModuleNotFoundError(
        "PyTorch is required for RL solver/training. Install it with `pip install torch`."
    ) from exc


def select_low_action_logits(low_logits, high_actions):
    """Select the aircraft logits associated with each high-level action."""

    if low_logits.ndim == 2:
        return low_logits
    return low_logits[
        torch.arange(len(high_actions), device=high_actions.device),
        high_actions,
    ]


class CarrierPolicyValueNet(nn.Module):
    """Shared aircraft encoder plus global high/low action heads."""

    def __init__(
        self,
        aircraft_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 128,
        aircraft_embed_dim: int = 64,
        action_conditioned_low_head: bool = True,
    ):
        super().__init__()
        self.aircraft_feature_dim = aircraft_feature_dim
        self.global_feature_dim = global_feature_dim
        self.hidden_dim = hidden_dim
        self.aircraft_embed_dim = aircraft_embed_dim
        self.action_conditioned_low_head = action_conditioned_low_head

        self.aircraft_encoder = nn.Sequential(
            nn.Linear(aircraft_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, aircraft_embed_dim),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim + aircraft_embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.high_head = nn.Linear(hidden_dim, 4)
        self.low_context = nn.Linear(hidden_dim, aircraft_embed_dim)
        low_output_dim = 4 if action_conditioned_low_head else 1
        self.low_head = nn.Linear(aircraft_embed_dim * 2, low_output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, aircraft, global_features):
        aircraft_embed = self.aircraft_encoder(aircraft)
        mean_pool = aircraft_embed.mean(dim=1)
        max_pool = aircraft_embed.max(dim=1).values
        global_input = torch.cat([global_features, mean_pool, max_pool], dim=-1)
        context = self.global_encoder(global_input)

        high_logits = self.high_head(context)
        context_for_aircraft = self.low_context(context).unsqueeze(1).expand_as(aircraft_embed)
        low_input = torch.cat([aircraft_embed, context_for_aircraft], dim=-1)
        low_logits = self.low_head(low_input)
        if self.action_conditioned_low_head:
            low_logits = low_logits.transpose(1, 2)
        else:
            low_logits = low_logits.squeeze(-1)
        value = self.value_head(context).squeeze(-1)
        return high_logits, low_logits, value

    def checkpoint_config(self):
        return {
            "aircraft_feature_dim": self.aircraft_feature_dim,
            "global_feature_dim": self.global_feature_dim,
            "hidden_dim": self.hidden_dim,
            "aircraft_embed_dim": self.aircraft_embed_dim,
            "action_conditioned_low_head": self.action_conditioned_low_head,
        }
