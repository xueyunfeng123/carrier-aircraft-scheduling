"""Policy-value network for hierarchical masked scheduling actions."""

from __future__ import annotations

import math

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
        target_feature_dim: int = 17,
        target_embed_dim: int = 64,
        target_rank_prior: float = 4.0,
        low_rank_prior: float = 4.0,
        adaptive_low_rank_prior: bool = False,
        action_conditioned_low_head: bool = True,
        num_high_actions: int = 5,
    ):
        super().__init__()
        self.aircraft_feature_dim = aircraft_feature_dim
        self.global_feature_dim = global_feature_dim
        self.hidden_dim = hidden_dim
        self.aircraft_embed_dim = aircraft_embed_dim
        self.target_feature_dim = target_feature_dim
        self.target_embed_dim = target_embed_dim
        self.target_rank_prior = float(target_rank_prior)
        self.low_rank_prior = float(low_rank_prior)
        self.low_rank_prior_scale = 1.0
        self.adaptive_low_rank_prior = adaptive_low_rank_prior
        self.action_conditioned_low_head = action_conditioned_low_head
        self.num_high_actions = num_high_actions

        self.aircraft_encoder = nn.Sequential(
            nn.Linear(aircraft_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, aircraft_embed_dim),
            nn.ReLU(),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(target_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_embed_dim),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(
                global_feature_dim
                + aircraft_embed_dim * 2
                + target_embed_dim * 2,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.high_head = nn.Linear(hidden_dim, num_high_actions)
        self.low_context = nn.Linear(hidden_dim, aircraft_embed_dim)
        low_output_dim = num_high_actions if action_conditioned_low_head else 1
        self.low_head = nn.Linear(aircraft_embed_dim * 2, low_output_dim)
        if adaptive_low_rank_prior:
            self.low_rank_gate = nn.Linear(hidden_dim, num_high_actions)
            nn.init.zeros_(self.low_rank_gate.weight)
            nn.init.constant_(
                self.low_rank_gate.bias,
                self.low_rank_prior,
            )
        self.high_action_embedding = nn.Embedding(
            num_high_actions,
            aircraft_embed_dim,
        )
        self.target_query = nn.Sequential(
            nn.Linear(
                hidden_dim + aircraft_embed_dim * 2,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_embed_dim),
        )
        self.target_bias = nn.Linear(target_embed_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _encode_state(self, aircraft, global_features, targets=None):
        aircraft_embed = self.aircraft_encoder(aircraft)
        mean_pool = aircraft_embed.mean(dim=1)
        max_pool = aircraft_embed.max(dim=1).values
        if targets is None:
            target_mean = torch.zeros(
                len(aircraft),
                self.target_embed_dim,
                device=aircraft.device,
                dtype=aircraft.dtype,
            )
            target_max = target_mean
        else:
            padding_width = self.target_feature_dim - targets.shape[-1]
            if padding_width < 0:
                raise ValueError("target feature tensor is too wide")
            if padding_width:
                targets = torch.nn.functional.pad(
                    targets,
                    (0, padding_width),
                )
            target_embed = self.target_encoder(targets)
            target_mean = target_embed.mean(dim=1)
            target_max = target_embed.max(dim=1).values
        global_input = torch.cat(
            [
                global_features,
                mean_pool,
                max_pool,
                target_mean,
                target_max,
            ],
            dim=-1,
        )
        context = self.global_encoder(global_input)
        return aircraft_embed, context

    def _state_heads(self, aircraft_embed, context, low_aux=None):
        high_logits = self.high_head(context)
        context_for_aircraft = self.low_context(context).unsqueeze(1).expand_as(aircraft_embed)
        low_input = torch.cat([aircraft_embed, context_for_aircraft], dim=-1)
        low_logits = self.low_head(low_input)
        if self.action_conditioned_low_head:
            low_logits = low_logits.transpose(1, 2)
            if low_aux is not None:
                rank_weight = self.low_rank_prior
                if self.adaptive_low_rank_prior:
                    rank_weight = self.low_rank_gate(
                        context
                    ).unsqueeze(-1)
                low_logits = (
                    low_logits
                    + self.low_rank_prior_scale
                    * rank_weight
                    * low_aux[..., 1]
                )
        else:
            low_logits = low_logits.squeeze(-1)
        value = self.value_head(context).squeeze(-1)
        return high_logits, low_logits, value

    def forward(
        self,
        aircraft,
        global_features,
        targets=None,
        low_aux=None,
    ):
        aircraft_embed, context = self._encode_state(
            aircraft,
            global_features,
            targets,
        )
        return self._state_heads(aircraft_embed, context, low_aux)

    def forward_with_targets(
        self,
        aircraft,
        global_features,
        targets,
        target_aux,
        high_actions,
        low_actions,
        low_aux=None,
    ):
        aircraft_embed, context = self._encode_state(
            aircraft,
            global_features,
            targets,
        )
        high_logits, low_logits, value = self._state_heads(
            aircraft_embed,
            context,
            low_aux,
        )
        batch_indices = torch.arange(
            len(high_actions),
            device=high_actions.device,
        )
        selected_aircraft = aircraft_embed[batch_indices, low_actions]
        selected_high = self.high_action_embedding(high_actions)
        query = self.target_query(
            torch.cat(
                [context, selected_aircraft, selected_high],
                dim=-1,
            )
        )
        target_embed = self.target_encoder(
            torch.cat([targets, target_aux], dim=-1)
        )
        target_logits = (
            target_embed * query.unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(self.target_embed_dim)
        target_logits = target_logits + self.target_bias(
            target_embed
        ).squeeze(-1)
        target_logits = (
            target_logits
            + self.target_rank_prior * target_aux[..., 1]
        )
        return high_logits, low_logits, target_logits, value

    def checkpoint_config(self):
        return {
            "aircraft_feature_dim": self.aircraft_feature_dim,
            "global_feature_dim": self.global_feature_dim,
            "hidden_dim": self.hidden_dim,
            "aircraft_embed_dim": self.aircraft_embed_dim,
            "target_feature_dim": self.target_feature_dim,
            "target_embed_dim": self.target_embed_dim,
            "target_rank_prior": self.target_rank_prior,
            "low_rank_prior": self.low_rank_prior,
            "adaptive_low_rank_prior": self.adaptive_low_rank_prior,
            "action_conditioned_low_head": self.action_conditioned_low_head,
            "num_high_actions": self.num_high_actions,
        }
