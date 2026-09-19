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


class SparseRelationAttentionLayer(nn.Module):
    """Sparse multi-head attention with relation-specific key/value gates."""

    def __init__(
        self,
        embed_dim: int,
        num_relations: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(
                "heterogeneous embedding dimension must be divisible "
                "by the number of attention heads"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.relation_key = nn.Parameter(
            torch.ones(num_relations, num_heads, self.head_dim)
        )
        self.relation_value = nn.Parameter(
            torch.ones(num_relations, num_heads, self.head_dim)
        )
        self.relation_bias = nn.Parameter(
            torch.zeros(num_relations, num_heads)
        )
        self.output = nn.Linear(embed_dim, embed_dim)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _gather_nodes(values, indices):
        trailing_shape = values.shape[2:]
        gather_index = indices.reshape(
            *indices.shape,
            *([1] * len(trailing_shape)),
        ).expand(*indices.shape, *trailing_shape)
        return torch.gather(values, 1, gather_index)

    def forward(
        self,
        nodes,
        edge_sources,
        edge_targets,
        edge_types,
        edge_mask,
    ):
        batch_size, node_count, _ = nodes.shape
        query = self.query(nodes).reshape(
            batch_size,
            node_count,
            self.num_heads,
            self.head_dim,
        )
        key = self.key(nodes).reshape_as(query)
        value = self.value(nodes).reshape_as(query)
        source_key = self._gather_nodes(key, edge_sources)
        source_value = self._gather_nodes(value, edge_sources)
        target_query = self._gather_nodes(query, edge_targets)
        relation_key = self.relation_key[edge_types]
        relation_value = self.relation_value[edge_types]
        scores = (
            target_query * source_key * relation_key
        ).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores + self.relation_bias[edge_types]
        scores = scores.masked_fill(
            ~edge_mask.unsqueeze(-1),
            float("-inf"),
        )

        target_index = edge_targets.unsqueeze(-1).expand(
            -1,
            -1,
            self.num_heads,
        )
        maxima = scores.new_full(
            (batch_size, node_count, self.num_heads),
            float("-inf"),
        )
        maxima.scatter_reduce_(
            1,
            target_index,
            scores,
            reduce="amax",
            include_self=True,
        )
        edge_maxima = torch.gather(maxima, 1, target_index)
        stabilized = torch.where(
            edge_mask.unsqueeze(-1),
            scores - edge_maxima,
            torch.zeros_like(scores),
        )
        weights = stabilized.exp() * edge_mask.unsqueeze(-1)
        denominators = scores.new_zeros(
            (batch_size, node_count, self.num_heads)
        )
        denominators.scatter_add_(1, target_index, weights)
        competing_sources = torch.gather(
            denominators,
            1,
            target_index,
        ).clamp_min(1.0e-12)
        weights = weights / competing_sources

        messages = (
            source_value
            * relation_value
            * weights.unsqueeze(-1)
        )
        aggregated = nodes.new_zeros(
            (
                batch_size,
                node_count,
                self.num_heads,
                self.head_dim,
            )
        )
        aggregated.scatter_add_(
            1,
            edge_targets[..., None, None].expand_as(messages),
            messages,
        )
        attended = self.output(
            aggregated.reshape(batch_size, node_count, self.embed_dim)
        )
        hidden = self.attention_norm(
            nodes + self.dropout(attended)
        )
        return self.output_norm(
            hidden + self.dropout(self.feed_forward(hidden))
        )


class CarrierPolicyValueNet(nn.Module):
    """Hierarchical policy with Deep Sets or sparse heterogeneous encoding."""

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
        encoder_type: str = "hetero",
        hetero_embed_dim: int = 64,
        hetero_layers: int = 1,
        hetero_heads: int = 4,
        hetero_dropout: float = 0.0,
        num_node_types: int = 6,
        num_edge_types: int = 18,
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
        self.encoder_type = encoder_type
        self.hetero_embed_dim = hetero_embed_dim
        self.hetero_layers = hetero_layers
        self.hetero_heads = hetero_heads
        self.hetero_dropout = hetero_dropout
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if encoder_type == "deepsets":
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
            global_input_dim = (
                global_feature_dim
                + aircraft_embed_dim * 2
                + target_embed_dim * 2
            )
        elif encoder_type == "hetero":
            if hetero_layers < 1:
                raise ValueError(
                    "heterogeneous encoder requires at least one layer"
                )
            self.aircraft_encoder = nn.Sequential(
                nn.Linear(aircraft_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hetero_embed_dim),
            )
            self.target_encoder = nn.Sequential(
                nn.Linear(target_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hetero_embed_dim),
            )
            self.global_node_encoder = nn.Sequential(
                nn.Linear(global_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hetero_embed_dim),
            )
            self.node_type_embedding = nn.Embedding(
                num_node_types,
                hetero_embed_dim,
            )
            self.relation_encoder = nn.ModuleList(
                SparseRelationAttentionLayer(
                    hetero_embed_dim,
                    num_edge_types,
                    hetero_heads,
                    hetero_dropout,
                )
                for _ in range(hetero_layers)
            )
            self.aircraft_projection = nn.Sequential(
                nn.Linear(hetero_embed_dim, aircraft_embed_dim),
                nn.ReLU(),
            )
            self.target_projection = nn.Sequential(
                nn.Linear(hetero_embed_dim, target_embed_dim),
                nn.ReLU(),
            )
            self.target_aux_encoder = nn.Linear(
                2,
                target_embed_dim,
                bias=False,
            )
            global_input_dim = (
                hetero_embed_dim
                + aircraft_embed_dim * 2
                + target_embed_dim * 2
            )
        else:
            raise ValueError(
                f"unknown state encoder type: {encoder_type}"
            )
        self.global_encoder = nn.Sequential(
            nn.Linear(
                global_input_dim,
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

    def _pad_target_features(self, targets):
        padding_width = self.target_feature_dim - targets.shape[-1]
        if padding_width < 0:
            raise ValueError("target feature tensor is too wide")
        if padding_width:
            targets = torch.nn.functional.pad(
                targets,
                (0, padding_width),
            )
        return targets

    def _encode_state_deepsets(
        self,
        aircraft,
        global_features,
        targets,
    ):
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
            target_embed = None
        else:
            target_embed = self.target_encoder(
                self._pad_target_features(targets)
            )
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
        return aircraft_embed, target_embed, context

    def _encode_state_hetero(
        self,
        aircraft,
        global_features,
        targets,
        node_types,
        edge_sources,
        edge_targets,
        edge_types,
        edge_mask,
    ):
        if any(
            item is None
            for item in (
                node_types,
                edge_sources,
                edge_targets,
                edge_types,
                edge_mask,
            )
        ):
            raise ValueError(
                "heterogeneous encoder requires typed graph tensors"
            )
        aircraft_hidden = self.aircraft_encoder(aircraft)
        if targets is None:
            target_hidden = aircraft_hidden.new_zeros(
                (len(aircraft), 0, self.hetero_embed_dim)
            )
        else:
            target_hidden = self.target_encoder(
                self._pad_target_features(targets)
            )
        global_hidden = self.global_node_encoder(
            global_features
        ).unsqueeze(1)
        nodes = torch.cat(
            [aircraft_hidden, target_hidden, global_hidden],
            dim=1,
        )
        if nodes.shape[:2] != node_types.shape:
            raise ValueError(
                "node type tensor does not match encoded graph nodes"
            )
        nodes = nodes + self.node_type_embedding(node_types)
        for layer in self.relation_encoder:
            nodes = layer(
                nodes,
                edge_sources,
                edge_targets,
                edge_types,
                edge_mask,
            )

        aircraft_count = aircraft.shape[1]
        target_count = 0 if targets is None else targets.shape[1]
        aircraft_embed = self.aircraft_projection(
            nodes[:, :aircraft_count]
        )
        target_embed = self.target_projection(
            nodes[
                :,
                aircraft_count : aircraft_count + target_count,
            ]
        )
        target_mean = (
            target_embed.mean(dim=1)
            if target_count
            else aircraft_embed.new_zeros(
                (len(aircraft), self.target_embed_dim)
            )
        )
        target_max = (
            target_embed.max(dim=1).values
            if target_count
            else target_mean
        )
        global_input = torch.cat(
            [
                nodes[:, -1],
                aircraft_embed.mean(dim=1),
                aircraft_embed.max(dim=1).values,
                target_mean,
                target_max,
            ],
            dim=-1,
        )
        context = self.global_encoder(global_input)
        return aircraft_embed, target_embed, context

    def _encode_state(
        self,
        aircraft,
        global_features,
        targets=None,
        node_types=None,
        edge_sources=None,
        edge_targets=None,
        edge_types=None,
        edge_mask=None,
    ):
        if self.encoder_type == "deepsets":
            return self._encode_state_deepsets(
                aircraft,
                global_features,
                targets,
            )
        return self._encode_state_hetero(
            aircraft,
            global_features,
            targets,
            node_types,
            edge_sources,
            edge_targets,
            edge_types,
            edge_mask,
        )

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
        node_types=None,
        edge_sources=None,
        edge_targets=None,
        edge_types=None,
        edge_mask=None,
    ):
        aircraft_embed, _, context = self._encode_state(
            aircraft,
            global_features,
            targets,
            node_types,
            edge_sources,
            edge_targets,
            edge_types,
            edge_mask,
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
        node_types=None,
        edge_sources=None,
        edge_targets=None,
        edge_types=None,
        edge_mask=None,
    ):
        aircraft_embed, target_embed, context = self._encode_state(
            aircraft,
            global_features,
            targets,
            node_types,
            edge_sources,
            edge_targets,
            edge_types,
            edge_mask,
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
        if self.encoder_type == "deepsets":
            target_embed = self.target_encoder(
                torch.cat([targets, target_aux], dim=-1)
            )
        else:
            target_embed = (
                target_embed + self.target_aux_encoder(target_aux)
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
            "encoder_type": self.encoder_type,
            "hetero_embed_dim": self.hetero_embed_dim,
            "hetero_layers": self.hetero_layers,
            "hetero_heads": self.hetero_heads,
            "hetero_dropout": self.hetero_dropout,
            "num_node_types": self.num_node_types,
            "num_edge_types": self.num_edge_types,
        }
