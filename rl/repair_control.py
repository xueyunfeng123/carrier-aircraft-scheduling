"""Event-trigger, neighborhood, and budget control for hybrid repair."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from solution.cp_sat_repair_model import RepairControl


CONTROL_FEATURE_DIM = 13
CONTROL_FEATURE_NAMES = (
    "normalized_time",
    "wave_launch_progress",
    "normalized_time_to_deadline",
    "pending_recovery_ratio",
    "candidate_action_ratio",
    "active_disruption_ratio",
    "unavailable_service_ratio",
    "closed_runway_ratio",
    "unavailable_aircraft_ratio",
    "service_slowdown_excess",
    "minimum_service_capacity_ratio",
    "launch_capacity_ratio",
    "blocked_failure_slot_ratio",
)
CONTROL_SCOPES = ("current_wave", "two_waves", "affected")
CONTROL_BUDGETS_MS = (10.0, 50.0, 200.0)
CONTROL_NEIGHBORHOODS = (8, 16, 32)
REPAIR_CONTROL_CHECKPOINT_TYPE = "repair_control_policy"
REPAIR_CONTROL_SCHEMA_VERSION = 1


def repair_event_signature(
    env: CarrierAircraftSchedulingEnv,
) -> Tuple[Any, ...]:
    disruptions = env.get_state()["disruptions"]
    lifecycle = disruptions["failure_lifecycle"]
    active = tuple(
        sorted(
            (
                item["kind"],
                str(item["target"]),
                int(item["disruption_id"]),
            )
            for item in disruptions["active"]
        )
    )
    return (
        int(env.current_wave_index),
        active,
        int(lifecycle["blocked_slots"]),
        int(lifecycle["hangar_ready"]),
        int(lifecycle["under_repair"]),
        int(lifecycle["replacements_in_transit"]),
    )


def encode_repair_control_state(
    env: CarrierAircraftSchedulingEnv,
) -> List[float]:
    state = env.get_state()
    resources = state["resources"]
    config = env.config
    wave = state["wave"]
    disruptions = state["disruptions"]
    next_wave_time = wave["next_wave_time"]
    time_to_deadline = (
        0.0
        if next_wave_time is None
        else max(0.0, float(next_wave_time) - env.time)
    )
    candidate_count = sum(
        sum(mask)
        for mask in env.get_action_mask()[
            "low_level_by_high"
        ].values()
    )
    unavailable_service = len(
        disruptions["unavailable_vehicle_ids"]
    )
    total_service = max(1, len(env.service_vehicles))
    slowdown = max(
        disruptions["service_time_multipliers"].values(),
        default=1.0,
    )
    service_ratios = (
        resources["fuel_servers"]
        / max(1.0, float(config["num_fuel_servers"])),
        resources["inspection_vehicles"]
        / max(
            1.0,
            float(config["num_inspection_vehicles"]),
        ),
        resources["arm_vehicles"]
        / max(1.0, float(config["num_arm_vehicles"])),
    )
    failure_state = disruptions["failure_lifecycle"]
    return [
        env.time / max(1.0, env.simulation_duration),
        float(wave["launches_started"])
        / max(1.0, float(wave["launch_target"])),
        time_to_deadline / max(1.0, env.wave_interval),
        float(wave["pending_recovery_count"])
        / max(1.0, float(wave["launch_target"])),
        candidate_count
        / max(1.0, env.num_aircraft * 5.0),
        len(disruptions["active"]) / 20.0,
        unavailable_service / total_service,
        len(disruptions["closed_runway_ids"])
        / max(1.0, float(config["num_launch_positions"])),
        len(disruptions["unavailable_aircraft_ids"])
        / max(1.0, float(env.num_aircraft)),
        max(0.0, slowdown - 1.0),
        min(service_ratios),
        resources["launch_channels"]
        / max(1.0, float(config["num_launch_channels"])),
        float(failure_state["blocked_slots"])
        / max(1.0, float(env.num_aircraft)),
    ]


class RuleBasedRepairController:
    """Deterministic event trigger used when no learned control is loaded."""

    def __init__(
        self,
        max_budget_ms: Optional[float] = None,
        max_neighborhood_size: Optional[int] = None,
        max_trigger_severity: Optional[float] = None,
    ):
        if (
            max_trigger_severity is not None
            and max_trigger_severity < 0.0
        ):
            raise ValueError(
                "maximum trigger severity must be non-negative"
            )
        self.last_signature: Optional[Tuple[Any, ...]] = None
        self.max_budget_ms = max_budget_ms
        self.max_neighborhood_size = max_neighborhood_size
        self.max_trigger_severity = max_trigger_severity

    def __call__(
        self,
        env: CarrierAircraftSchedulingEnv,
    ) -> RepairControl:
        features = encode_repair_control_state(env)
        signature = repair_event_signature(env)
        trigger = self.last_signature != signature
        self.last_signature = signature

        severity = max(
            features[6],
            features[7],
            features[8],
            features[9] / 2.0,
            features[12],
        )
        if (
            self.max_trigger_severity is not None
            and severity > self.max_trigger_severity
        ):
            trigger = False
        if severity >= 0.25:
            budget_ms = CONTROL_BUDGETS_MS[2]
            neighborhood = CONTROL_NEIGHBORHOODS[2]
            scope = "affected"
        elif severity > 0.0:
            budget_ms = CONTROL_BUDGETS_MS[1]
            neighborhood = CONTROL_NEIGHBORHOODS[1]
            scope = "two_waves"
        else:
            budget_ms = CONTROL_BUDGETS_MS[0]
            neighborhood = CONTROL_NEIGHBORHOODS[0]
            scope = "current_wave"
        return RepairControl(
            trigger=trigger,
            scope=scope,
            budget_ms=(
                min(budget_ms, self.max_budget_ms)
                if self.max_budget_ms is not None
                else budget_ms
            ),
            neighborhood_size=(
                min(neighborhood, self.max_neighborhood_size)
                if self.max_neighborhood_size is not None
                else neighborhood
            ),
            horizon_waves=(
                1 if scope == "current_wave" else 2
            ),
        )


class RepairControlNet:
    """Small multi-head policy over trigger, scope, budget, and neighborhood."""

    def __new__(
        cls,
        hidden_dim: int = 64,
    ):
        try:
            import torch
            from torch import nn
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch is required for learned repair control"
            ) from exc

        class _Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(CONTROL_FEATURE_DIM, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.trigger_head = nn.Linear(hidden_dim, 2)
                self.scope_head = nn.Linear(
                    hidden_dim,
                    len(CONTROL_SCOPES),
                )
                self.budget_head = nn.Linear(
                    hidden_dim,
                    len(CONTROL_BUDGETS_MS),
                )
                self.neighborhood_head = nn.Linear(
                    hidden_dim,
                    len(CONTROL_NEIGHBORHOODS),
                )

            def forward(self, features):
                encoded = self.encoder(features)
                return (
                    self.trigger_head(encoded),
                    self.scope_head(encoded),
                    self.budget_head(encoded),
                    self.neighborhood_head(encoded),
                )

            def checkpoint_config(self) -> Dict[str, Any]:
                return {
                    "feature_dim": CONTROL_FEATURE_DIM,
                    "hidden_dim": hidden_dim,
                    "scopes": list(CONTROL_SCOPES),
                    "budgets_ms": list(CONTROL_BUDGETS_MS),
                    "neighborhoods": list(
                        CONTROL_NEIGHBORHOODS
                    ),
                }

        return _Network()


class LearnedRepairController:
    """Loads a trained control policy and emits deterministic repair controls."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        max_budget_ms: Optional[float] = None,
        max_neighborhood_size: Optional[int] = None,
    ):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch is required for learned repair control"
            ) from exc
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(
                f"repair control checkpoint does not exist: {checkpoint}"
            )
        payload = torch.load(path, map_location=device)
        extra = payload.get("extra", {})
        if (
            extra.get("checkpoint_type")
            != REPAIR_CONTROL_CHECKPOINT_TYPE
        ):
            raise ValueError(
                "checkpoint is not a repair control policy"
            )
        if (
            int(extra.get("control_dataset_schema_version", -1))
            != REPAIR_CONTROL_SCHEMA_VERSION
        ):
            raise ValueError(
                "repair control checkpoint schema is incompatible"
            )
        if tuple(extra.get("feature_names", ())) != tuple(
            CONTROL_FEATURE_NAMES
        ):
            raise ValueError(
                "repair control checkpoint feature order is incompatible"
            )
        expected_categories = {
            "scopes": list(CONTROL_SCOPES),
            "budgets_ms": list(CONTROL_BUDGETS_MS),
            "neighborhoods": list(CONTROL_NEIGHBORHOODS),
        }
        for key, expected in expected_categories.items():
            if list(extra.get(key, ())) != expected:
                raise ValueError(
                    "repair control checkpoint categories are "
                    f"incompatible: {key}"
                )
        hidden_dim = int(
            payload.get("model_config", {}).get(
                "hidden_dim",
                64,
            )
        )
        self.model = RepairControlNet(hidden_dim).to(device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.device = device
        self.torch = torch
        self.max_budget_ms = max_budget_ms
        self.max_neighborhood_size = max_neighborhood_size
        self.last_signature: Optional[Tuple[Any, ...]] = None

    def __call__(
        self,
        env: CarrierAircraftSchedulingEnv,
    ) -> RepairControl:
        signature = repair_event_signature(env)
        event_changed = self.last_signature != signature
        self.last_signature = signature
        if not event_changed:
            return RepairControl(
                trigger=False,
                scope="two_waves",
                budget_ms=(
                    min(50.0, self.max_budget_ms)
                    if self.max_budget_ms is not None
                    else 50.0
                ),
                neighborhood_size=(
                    min(16, self.max_neighborhood_size)
                    if self.max_neighborhood_size is not None
                    else 16
                ),
                horizon_waves=2,
            )
        features = self.torch.tensor(
            [encode_repair_control_state(env)],
            dtype=self.torch.float32,
            device=self.device,
        )
        with self.torch.no_grad():
            outputs = self.model(features)
        trigger_index, scope_index, budget_index, neighborhood_index = (
            int(output.argmax(dim=-1).item())
            for output in outputs
        )
        scope = CONTROL_SCOPES[scope_index]
        return RepairControl(
            trigger=bool(trigger_index),
            scope=scope,
            budget_ms=(
                min(
                    CONTROL_BUDGETS_MS[budget_index],
                    self.max_budget_ms,
                )
                if self.max_budget_ms is not None
                else CONTROL_BUDGETS_MS[budget_index]
            ),
            neighborhood_size=(
                min(
                    CONTROL_NEIGHBORHOODS[neighborhood_index],
                    self.max_neighborhood_size,
                )
                if self.max_neighborhood_size is not None
                else CONTROL_NEIGHBORHOODS[neighborhood_index]
            ),
            horizon_waves=(
                1 if scope == "current_wave" else 2
            ),
        )
