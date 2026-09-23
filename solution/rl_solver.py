"""RL policy solver for inference through scripts.solve."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        require_calibrated_value: bool = False,
        value_rerank_include_heuristic: bool = False,
        value_rerank_include_cp_sat: bool = False,
        value_rerank_cp_sat_time: float = 0.05,
        value_checkpoint: str = "",
        action_value_checkpoint: str = "",
        action_value_top_k: int = 3,
    ):
        try:
            import torch
            from rl.action_value import (
                ActionValueRanker,
                checkpoint_sha256,
                predict_action_values_for_actions,
                validate_action_value_checkpoint,
            )
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
        self.value_rerank_include_heuristic = bool(
            value_rerank_include_heuristic
        )
        self.value_rerank_include_cp_sat = bool(
            value_rerank_include_cp_sat
        )
        self.value_rerank_enabled = (
            self.value_rerank_top_k > 1
            or self.value_rerank_include_heuristic
            or self.value_rerank_include_cp_sat
        )
        self.action_value_top_k = int(action_value_top_k)
        self.action_value_enabled = bool(action_value_checkpoint)
        if self.value_rerank_top_k < 1:
            raise ValueError("value_rerank_top_k must be positive")
        if self.action_value_top_k < 1:
            raise ValueError("action_value_top_k must be positive")
        if self.action_value_enabled and self.value_rerank_enabled:
            raise ValueError(
                "action-value ranking and successor-value reranking "
                "cannot be enabled together"
            )
        if self.action_value_enabled and value_checkpoint:
            raise ValueError(
                "action-value and successor-value checkpoints cannot "
                "be loaded together"
            )
        if value_rerank_cp_sat_time <= 0.0:
            raise ValueError("value_rerank_cp_sat_time must be positive")

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
        if self.action_value_enabled and (
            checkpoint_path is None
            or not checkpoint_path.is_file()
        ):
            raise ValueError(
                "action-value inference requires an existing actor checkpoint"
            )
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_payload = load_checkpoint(str(checkpoint_path), device=device)
            checkpoint_extra = checkpoint_payload.get("extra", {})
            checkpoint_calibration = checkpoint_extra.get(
                "value_calibration",
                {},
            )
            if (
                checkpoint_extra.get("value_only_checkpoint", False)
                or checkpoint_calibration.get(
                    "value_only_checkpoint",
                    False,
                )
            ):
                raise ValueError(
                    "a value-only checkpoint cannot be used as the policy "
                    "checkpoint"
                )
            if checkpoint_extra.get("checkpoint_type") == (
                "action_value_ranker"
            ):
                raise ValueError(
                    "an action-value checkpoint cannot be used as the "
                    "policy checkpoint"
                )
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
        self.value_model = self.model
        value_checkpoint_payload = None
        if value_checkpoint:
            value_checkpoint_path = Path(value_checkpoint)
            if not value_checkpoint_path.exists():
                raise ValueError(
                    "value checkpoint does not exist: "
                    f"{value_checkpoint}"
                )
            value_checkpoint_payload = load_checkpoint(
                str(value_checkpoint_path),
                device=device,
            )
            value_schema = value_checkpoint_payload.get("extra", {}).get(
                "observation_schema_version",
                1,
            )
            if value_schema != OBSERVATION_SCHEMA_VERSION:
                raise ValueError(
                    "value checkpoint observation schema is incompatible "
                    "with the current environment"
                )
            value_model_config = dict(model_config)
            value_checkpoint_config = value_checkpoint_payload.get(
                "model_config",
                {},
            )
            value_model_config.update(value_checkpoint_config)
            if (
                "action_conditioned_low_head"
                not in value_checkpoint_config
            ):
                value_model_config["action_conditioned_low_head"] = False
            self.value_model = CarrierPolicyValueNet(
                **value_model_config
            ).to(device)
            self.value_model.load_state_dict(
                value_checkpoint_payload["model_state"]
            )
            self.value_model.eval()
        self.action_value_model = None
        self._predict_action_values_for_actions = (
            predict_action_values_for_actions
        )
        if self.action_value_enabled:
            action_value_path = Path(action_value_checkpoint)
            if not action_value_path.is_file():
                raise ValueError(
                    "action-value checkpoint does not exist: "
                    f"{action_value_checkpoint}"
                )
            action_value_payload = load_checkpoint(
                str(action_value_path),
                device=device,
            )
            action_value_schema = action_value_payload.get(
                "extra",
                {},
            ).get("observation_schema_version", 1)
            if action_value_schema != OBSERVATION_SCHEMA_VERSION:
                raise ValueError(
                    "action-value checkpoint observation schema is "
                    "incompatible with the current environment"
                )
            validate_action_value_checkpoint(
                action_value_payload,
                float(self.env.config["wave_interval"]),
                actor_sha256=checkpoint_sha256(str(checkpoint_path)),
            )
            self.action_value_model = ActionValueRanker(
                **action_value_payload["model_config"]
            ).to(device)
            self.action_value_model.load_state_dict(
                action_value_payload["model_state"]
            )
            self.action_value_model.eval()

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
        if self.value_rerank_enabled or require_calibrated_value:
            calibration_payload = (
                value_checkpoint_payload
                if value_checkpoint_payload is not None
                else checkpoint_payload
            )
            if calibration_payload is None:
                raise ValueError(
                    "calibrated value inference requires an existing checkpoint"
                )
            validate_rerank_calibration(
                calibration_payload,
                float(self.env.config["wave_interval"]),
            )
        self._proposal_solvers = []
        if self.value_rerank_include_heuristic:
            from solution.heuristic_solver import WaveHeuristicSolver

            self._proposal_solvers.append(
                WaveHeuristicSolver(self.env)
            )
        if self.value_rerank_include_cp_sat:
            from solution.cp_sat_solver import CPSATSolver

            self._proposal_solvers.append(
                CPSATSolver(
                    self.env,
                    max_time_seconds=float(value_rerank_cp_sat_time),
                )
            )

    def choose_action(self) -> Optional[Dict[str, Any]]:
        encoded = encode_observation(self.env)
        if not any(encoded.high_mask):
            return None
        if self.action_value_enabled:
            action = self._choose_action_value_ranked_action(encoded)
        elif not self.value_rerank_enabled:
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

    def rank_actions(
        self,
        top_k: int,
        env: Optional[CarrierAircraftSchedulingEnv] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return joint actions from a legal hierarchical Top-K beam.

        Each conditional level is restricted to ``top_k`` candidates before
        the resulting joint actions are sorted by policy log probability.
        """

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        target_env = env or self.env
        encoded = encode_observation(target_env)
        if not any(encoded.high_mask):
            return []

        from rl.model import select_low_action_logits
        from rl.obs_encoder import (
            apply_target_slot,
            target_context_for_action,
            to_torch_batch,
        )

        torch = self.torch
        batch = to_torch_batch(encoded, self.device)
        candidates: List[Tuple[Dict[str, Any], float]] = []
        with torch.no_grad():
            high_logits, low_logits, _ = self.model(
                batch["aircraft"],
                batch["global"],
                batch["targets"],
                batch["low_aux"],
            )
            high_log_probs = torch.log_softmax(
                high_logits.masked_fill(~batch["high_mask"], -torch.inf),
                dim=-1,
            )[0]
            high_ids = self._top_masked_indices(
                high_log_probs,
                batch["high_mask"][0],
                top_k,
            )
            for high_id in high_ids:
                high_action = torch.tensor(
                    [high_id],
                    dtype=torch.long,
                    device=self.device,
                )
                low_mask = batch["low_masks"][0, high_id]
                selected_low_logits = select_low_action_logits(
                    low_logits,
                    high_action,
                )[0]
                low_log_probs = torch.log_softmax(
                    selected_low_logits.masked_fill(
                        ~low_mask,
                        -torch.inf,
                    ),
                    dim=-1,
                )
                low_ids = self._top_masked_indices(
                    low_log_probs,
                    low_mask,
                    top_k,
                )
                for low_id in low_ids:
                    target_mask_values, target_aux_values = (
                        target_context_for_action(
                            target_env,
                            encoded,
                            high_id,
                            low_id,
                        )
                    )
                    target_mask = torch.tensor(
                        [target_mask_values],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    target_aux = torch.tensor(
                        [target_aux_values],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    low_action = torch.tensor(
                        [low_id],
                        dtype=torch.long,
                        device=self.device,
                    )
                    _, _, target_logits, _ = (
                        self.model.forward_with_targets(
                            batch["aircraft"],
                            batch["global"],
                            batch["targets"],
                            target_aux,
                            high_action,
                            low_action,
                            batch["low_aux"],
                        )
                    )
                    target_log_probs = torch.log_softmax(
                        target_logits.masked_fill(
                            ~target_mask,
                            -torch.inf,
                        ),
                        dim=-1,
                    )[0]
                    target_ids = self._top_masked_indices(
                        target_log_probs,
                        target_mask[0],
                        top_k,
                    )
                    for target_id in target_ids:
                        action = apply_target_slot(
                            encoded,
                            {
                                "high_level": high_id,
                                "aircraft_id": low_id,
                            },
                            target_id,
                        )
                        log_probability = (
                            high_log_probs[high_id]
                            + low_log_probs[low_id]
                            + target_log_probs[target_id]
                        )
                        candidates.append(
                            (action, float(log_probability.item()))
                        )

        candidates.sort(
            key=lambda item: (
                -item[1],
                item[0]["high_level"],
                item[0]["aircraft_id"],
                item[0]["target_slot"],
            )
        )
        return candidates[:top_k]

    def _top_masked_indices(self, values, mask, top_k: int) -> List[int]:
        legal_count = int(mask.sum().item())
        if legal_count == 0:
            return []
        count = min(top_k, legal_count)
        return [
            int(index)
            for index in self.torch.topk(
                values,
                k=count,
            ).indices.tolist()
        ]

    def estimate_terminal_sorties(
        self,
        env: Optional[CarrierAircraftSchedulingEnv] = None,
    ) -> float:
        """Estimate final completed sorties from an observable search state."""

        target_env = env or self.env
        completed = float(
            target_env.get_evaluation_metrics()[
                "total_sorties_completed"
            ]
        )
        if target_env.done:
            return completed

        encoded = encode_observation(target_env)
        batch = to_torch_batch(encoded, self.device)
        with self.torch.no_grad():
            _, _, remaining_value = self.value_model(
                batch["aircraft"],
                batch["global"],
                batch["targets"],
                batch["low_aux"],
            )
        return completed + float(remaining_value.item())

    def _choose_action_value_ranked_action(
        self,
        encoded,
    ) -> Dict[str, Any]:
        candidates = self.trainer.rank_actions(
            encoded,
            self.env,
            self.action_value_top_k,
        )
        if not candidates:
            raise RuntimeError("actor produced no legal action-value candidates")
        actions = [action for action, _ in candidates]
        values = self._predict_action_values_for_actions(
            self.action_value_model,
            encoded,
            actions,
            self.device,
        )
        best_index = max(
            range(len(actions)),
            key=lambda index: (values[index], -index),
        )
        return actions[best_index]

    def _choose_value_reranked_action(self, encoded) -> Dict[str, Any]:
        candidates = self.trainer.rank_actions(
            encoded,
            self.env,
            self.value_rerank_top_k,
        )
        candidate_keys = {
            self._action_key(action)
            for action, _ in candidates
        }
        for proposal_solver in getattr(self, "_proposal_solvers", ()):
            proposal = proposal_solver.choose_action()
            if proposal is None:
                continue
            key = self._action_key(proposal)
            if key in candidate_keys:
                continue
            candidates.append((proposal, float("-inf")))
            candidate_keys.add(key)
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

    @staticmethod
    def _action_key(action: Dict[str, Any]) -> Tuple[int, int, int]:
        return (
            int(action["high_level"]),
            int(action["aircraft_id"]),
            int(action["target_slot"]),
        )

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
            _, _, next_value = self.value_model(
                encoded_batch["aircraft"],
                encoded_batch["global"],
                encoded_batch["targets"],
                encoded_batch["low_aux"],
            )
        return reward + self.ppo_config.gamma * float(
            next_value.item()
        )
