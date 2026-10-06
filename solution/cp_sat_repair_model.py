"""Rolling-horizon CP-SAT repair model for deck support dispatching."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from solution.priority_rule_solver import (
    ACTION_ARM,
    ACTION_FUEL,
    ACTION_INSPECTION,
    PriorityRuleSolver,
)


ActionKey = Tuple[int, int]


@dataclass(frozen=True)
class RepairControl:
    """One online repair decision supplied by a baseline or controller."""

    trigger: bool = True
    scope: str = "two_waves"
    budget_ms: float = 50.0
    neighborhood_size: int = 20
    horizon_waves: int = 2

    def __post_init__(self) -> None:
        if self.scope not in {
            "current_wave",
            "two_waves",
            "affected",
        }:
            raise ValueError(f"unsupported repair scope: {self.scope}")
        if self.budget_ms < 0.0:
            raise ValueError("repair budget must be non-negative")
        if self.neighborhood_size < 1:
            raise ValueError("repair neighborhood must be positive")
        if self.horizon_waves < 1:
            raise ValueError("repair horizon must be positive")


@dataclass(frozen=True)
class PlanningTask:
    action_key: ActionKey
    duration_ticks: int
    priority: int


@dataclass
class RepairPlan:
    snapshot_id: Tuple[Any, ...]
    status: str
    actions: List[ActionKey] = field(default_factory=list)
    immediate_actions: List[ActionKey] = field(default_factory=list)
    start_ticks: Dict[ActionKey, int] = field(default_factory=dict)
    objective: Optional[float] = None
    best_bound: Optional[float] = None
    runtime_ms: float = 0.0
    solve_ms: float = 0.0
    deadline_missed: bool = False


def planning_snapshot_id(
    env: CarrierAircraftSchedulingEnv,
) -> Tuple[Any, ...]:
    active_disruptions = tuple(
        sorted(
            (
                spec.kind,
                str(spec.target),
                int(spec.disruption_id),
            )
            for spec in env.active_disruptions.values()
        )
    )
    return (
        round(float(env.time), 9),
        int(env.current_wave_index),
        int(env.event_sequence),
        active_disruptions,
    )


class CPSATRepairModel:
    """Build and solve an expected-duration rolling service schedule."""

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        time_scale: int = 10,
        guidance_strength: float = 100.0,
    ):
        try:
            from ortools.sat.python import cp_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OR-Tools is required for rolling CP-SAT repair"
            ) from exc
        if time_scale < 1:
            raise ValueError("time_scale must be positive")
        if guidance_strength < 0.0:
            raise ValueError(
                "guidance strength must be non-negative"
            )
        self.env = env
        self.cp_model = cp_model
        self.time_scale = int(time_scale)
        self.guidance_strength = float(guidance_strength)
        self.priority = PriorityRuleSolver(env, "edd")

    def solve(
        self,
        control: RepairControl,
        incumbent_starts: Optional[Mapping[ActionKey, int]] = None,
        guidance_scores: Optional[Mapping[ActionKey, float]] = None,
    ) -> RepairPlan:
        started = time.perf_counter()
        snapshot_id = planning_snapshot_id(self.env)
        if not control.trigger or control.budget_ms <= 0.0:
            return RepairPlan(
                snapshot_id=snapshot_id,
                status="skipped",
                runtime_ms=(time.perf_counter() - started) * 1000.0,
            )

        tasks = self._planning_tasks(
            control,
            guidance_scores or {},
        )
        if not tasks:
            return RepairPlan(
                snapshot_id=snapshot_id,
                status="empty",
                runtime_ms=(time.perf_counter() - started) * 1000.0,
            )

        model = self.cp_model.CpModel()
        horizon_ticks = self._ticks(
            control.horizon_waves * self.env.wave_interval
        )
        max_end = horizon_ticks + sum(
            task.duration_ticks for task in tasks
        )
        deadline_ticks = self._ticks(
            max(
                0.0,
                min(
                    (
                        self.env.current_wave_index + 1
                    )
                    * self.env.wave_interval,
                    self.env.simulation_duration,
                )
                - self.env.time,
            )
        )
        starts: Dict[ActionKey, Any] = {}
        ends: Dict[ActionKey, Any] = {}
        intervals_by_resource: Dict[str, List[Any]] = {
            "fuel": [],
            "inspection": [],
            "arm": [],
            "ammo_transport": [],
            "lower_lift": [],
            "upper_lift": [],
            "personnel": [],
        }
        personnel_demands: List[int] = []
        objective_terms = []
        secondary_objective_bound = 0
        incumbent_starts = dict(incumbent_starts or {})

        for task in tasks:
            action, aircraft_id = task.action_key
            name = f"{action}_{aircraft_id}"
            start = model.NewIntVar(0, max_end, f"start_{name}")
            end = model.NewIntVar(0, max_end, f"end_{name}")
            starts[task.action_key] = start
            ends[task.action_key] = end

            if action == ACTION_ARM:
                stage_one = self._arm_stage_one_ticks()
                stage_two = max(1, task.duration_ticks - stage_one)
                stage_one_end = model.NewIntVar(
                    0,
                    max_end,
                    f"stage_one_end_{name}",
                )
                model.Add(stage_one_end == start + stage_one)
                model.Add(end == stage_one_end + stage_two)
                full_interval = model.NewIntervalVar(
                    start,
                    task.duration_ticks,
                    end,
                    f"arm_full_{name}",
                )
                lower_interval = model.NewIntervalVar(
                    start,
                    stage_one,
                    stage_one_end,
                    f"arm_lower_{name}",
                )
                upper_interval = model.NewIntervalVar(
                    stage_one_end,
                    stage_two,
                    end,
                    f"arm_upper_{name}",
                )
                intervals_by_resource["arm"].append(full_interval)
                intervals_by_resource["ammo_transport"].append(
                    lower_interval
                )
                intervals_by_resource["lower_lift"].append(
                    lower_interval
                )
                intervals_by_resource["upper_lift"].append(
                    upper_interval
                )
                personnel_interval = full_interval
                personnel_demand = int(
                    self.env.config["arm_personnel_required"]
                )
            else:
                interval = model.NewIntervalVar(
                    start,
                    task.duration_ticks,
                    end,
                    f"service_{name}",
                )
                resource = (
                    "fuel"
                    if action == ACTION_FUEL
                    else "inspection"
                )
                intervals_by_resource[resource].append(interval)
                personnel_interval = interval
                personnel_demand = int(
                    self.env.config[
                        (
                            "fuel_personnel_required"
                            if action == ACTION_FUEL
                            else "inspection_personnel_required"
                        )
                    ]
                )

            if bool(
                self.env.config["personnel_scheduling_enabled"]
            ):
                intervals_by_resource["personnel"].append(
                    personnel_interval
                )
                personnel_demands.append(personnel_demand)

            old_start = incumbent_starts.get(task.action_key)
            if old_start is not None:
                clipped = min(max_end, max(0, int(old_start)))
                model.AddHint(start, clipped)
                deviation = model.NewIntVar(
                    0,
                    max_end,
                    f"deviation_{name}",
                )
                model.AddAbsEquality(deviation, start - clipped)
                objective_terms.append(-10 * deviation)
                secondary_objective_bound += 10 * max_end
            objective_terms.append(-task.priority * end)
            secondary_objective_bound += task.priority * max_end

        self._add_active_resource_occupancy(
            model,
            intervals_by_resource,
            personnel_demands,
            max_end,
        )
        capacities = self._resource_capacities()
        self._add_cumulative(
            model,
            intervals_by_resource["fuel"],
            int(capacities["fuel_servers"]),
        )
        self._add_cumulative(
            model,
            intervals_by_resource["inspection"],
            int(capacities["inspection_vehicles"]),
        )
        self._add_cumulative(
            model,
            intervals_by_resource["arm"],
            int(capacities["arm_vehicles"]),
        )
        self._add_cumulative(
            model,
            intervals_by_resource["ammo_transport"],
            int(capacities["ammo_transport_vehicles"]),
        )
        self._add_cumulative(
            model,
            intervals_by_resource["lower_lift"],
            int(capacities["lower_weapon_lifts"]),
        )
        self._add_cumulative(
            model,
            intervals_by_resource["upper_lift"],
            int(capacities["upper_weapon_lifts"]),
        )
        if intervals_by_resource["personnel"]:
            capacity = int(capacities["personnel"])
            model.AddCumulative(
                intervals_by_resource["personnel"],
                personnel_demands,
                max(0, capacity),
            )

        task_keys_by_aircraft: Dict[int, List[ActionKey]] = {}
        for task in tasks:
            task_keys_by_aircraft.setdefault(
                task.action_key[1],
                [],
            ).append(task.action_key)
        sortie_weight = secondary_objective_bound + 1
        active_completion_ticks = (
            self._active_completion_ticks_by_aircraft(max_end)
        )
        for aircraft_id, action_keys in task_keys_by_aircraft.items():
            completion = model.NewIntVar(
                0,
                max_end,
                f"completion_{aircraft_id}",
            )
            model.AddMaxEquality(
                completion,
                [
                    ends[action_key]
                    for action_key in action_keys
                ]
                + (
                    [
                        model.NewConstant(
                            active_completion_ticks[aircraft_id]
                        )
                    ]
                    if aircraft_id in active_completion_ticks
                    else []
                ),
            )
            ready = model.NewBoolVar(f"ready_{aircraft_id}")
            model.Add(completion <= deadline_ticks).OnlyEnforceIf(
                ready
            )
            model.Add(completion > deadline_ticks).OnlyEnforceIf(
                ready.Not()
            )
            objective_terms.append(sortie_weight * ready)

        model.Maximize(sum(objective_terms))
        build_ms = (time.perf_counter() - started) * 1000.0
        remaining_seconds = max(
            0.001,
            (control.budget_ms - build_ms) / 1000.0,
        )
        solver = self.cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining_seconds
        solver.parameters.num_search_workers = 1
        solve_started = time.perf_counter()
        status = solver.Solve(model)
        solve_ms = (time.perf_counter() - solve_started) * 1000.0
        runtime_ms = (time.perf_counter() - started) * 1000.0

        valid_statuses = {
            self.cp_model.OPTIMAL,
            self.cp_model.FEASIBLE,
        }
        if status not in valid_statuses:
            return RepairPlan(
                snapshot_id=snapshot_id,
                status=solver.StatusName(status).lower(),
                runtime_ms=runtime_ms,
                solve_ms=solve_ms,
                deadline_missed=runtime_ms > control.budget_ms,
            )

        start_ticks = {
            action_key: int(solver.Value(variable))
            for action_key, variable in starts.items()
        }
        actions = sorted(
            start_ticks,
            key=lambda action_key: (
                start_ticks[action_key],
                action_key[0],
                action_key[1],
            ),
        )
        immediate = [
            action_key
            for action_key in actions
            if start_ticks[action_key] == 0
        ]
        return RepairPlan(
            snapshot_id=snapshot_id,
            status=solver.StatusName(status).lower(),
            actions=actions,
            immediate_actions=immediate,
            start_ticks=start_ticks,
            objective=float(solver.ObjectiveValue()),
            best_bound=float(solver.BestObjectiveBound()),
            runtime_ms=runtime_ms,
            solve_ms=solve_ms,
            deadline_missed=runtime_ms > control.budget_ms,
        )

    def _planning_tasks(
        self,
        control: RepairControl,
        guidance_scores: Mapping[ActionKey, float],
    ) -> List[PlanningTask]:
        mask = self.env.get_action_mask()
        candidates = [
            (action, aircraft_id)
            for action in (
                ACTION_FUEL,
                ACTION_INSPECTION,
                ACTION_ARM,
            )
            for aircraft_id, allowed in enumerate(
                mask["low_level_by_high"][action]
            )
            if allowed
        ]
        if control.scope == "affected":
            affected_actions: set[ActionKey] = set()
            action_by_service = {
                "fuel": ACTION_FUEL,
                "inspection": ACTION_INSPECTION,
                "arm": ACTION_ARM,
            }
            for spec in self.env.active_disruptions.values():
                if spec.kind in {
                    "runway_closure",
                    "aircraft_failure",
                }:
                    affected_actions.update(candidates)
                elif spec.kind == "aircraft_hold":
                    affected_actions.update(
                        action_key
                        for action_key in candidates
                        if action_key[1] == int(spec.target)
                    )
                elif spec.kind == "service_slowdown":
                    target = str(spec.target)
                    affected_actions.update(
                        action_key
                        for action_key in candidates
                        if target == "all"
                        or action_key[0]
                        == action_by_service.get(target)
                    )
                elif spec.kind == "vehicle_outage":
                    service = str(spec.target).split("_", 1)[0]
                    affected_actions.update(
                        action_key
                        for action_key in candidates
                        if action_key[0]
                        == action_by_service.get(service)
                    )
            if affected_actions:
                candidates = [
                    action_key
                    for action_key in candidates
                    if action_key in affected_actions
                ]
        normalized_guidance = self._normalized_guidance(
            candidates,
            guidance_scores,
        )
        affected = set(self.env.unavailable_aircraft_ids)
        aircraft_guidance = {
            aircraft_id: max(
                normalized_guidance.get(
                    (action, aircraft_id),
                    0.0,
                )
                for action, candidate_id in candidates
                if candidate_id == aircraft_id
            )
            for aircraft_id in {
                candidate_id for _, candidate_id in candidates
            }
        }
        ranked_aircraft = sorted(
            {aircraft_id for _, aircraft_id in candidates},
            key=lambda aircraft_id: (
                0 if aircraft_id in affected else 1,
                self.priority._launch_deadline(aircraft_id),
                (
                    -aircraft_guidance[aircraft_id]
                    if self.guidance_strength > 0.0
                    else 0.0
                ),
                min(
                    self.priority._expected_action_duration(
                        action,
                        aircraft_id,
                    )
                    for action, candidate_id in candidates
                    if candidate_id == aircraft_id
                ),
                aircraft_id,
            ),
        )
        scope_limit = control.neighborhood_size
        launches_remaining = max(
            1,
            self.env.group_size
            - int(
                self.env.wave_records[-1]["launches_started"]
            ),
        )
        if control.scope == "current_wave":
            scope_limit = min(scope_limit, launches_remaining)
        elif control.scope == "two_waves":
            scope_limit = min(
                scope_limit,
                launches_remaining + self.env.group_size,
            )
        selected_aircraft = set(ranked_aircraft[:scope_limit])
        tasks = []
        for action_key in candidates:
            action, aircraft_id = action_key
            if aircraft_id not in selected_aircraft:
                continue
            duration = self.priority._expected_action_duration(
                action,
                aircraft_id,
            )
            if not math.isfinite(duration):
                continue
            guidance = normalized_guidance.get(action_key, 0.0)
            urgency = max(
                1,
                int(
                    round(
                        1000.0
                        + self.guidance_strength * guidance
                    )
                ),
            )
            tasks.append(
                PlanningTask(
                    action_key=action_key,
                    duration_ticks=self._ticks(duration),
                    priority=urgency,
                )
            )
        return tasks

    def _active_completion_ticks_by_aircraft(
        self,
        max_end: int,
    ) -> Dict[int, int]:
        event_by_key = {
            (event.event_type, event.aircraft_id): event.time
            for event in self.env.event_queue
        }
        result = {}
        for aircraft_id, aircraft in enumerate(self.env.aircraft):
            remaining = [
                self._active_service_remaining(
                    service_type,
                    aircraft_id,
                    event_by_key,
                )
                for service_type, active in (
                    ("fuel", aircraft.fuel_status == 1),
                    (
                        "inspection",
                        aircraft.inspection_status == 1,
                    ),
                    ("arm", aircraft.arm_status == 1),
                )
                if active
            ]
            if remaining:
                result[aircraft_id] = min(
                    max_end,
                    self._ticks(max(remaining)),
                )
        return result

    @staticmethod
    def _normalized_guidance(
        candidates: Sequence[ActionKey],
        guidance_scores: Mapping[ActionKey, float],
    ) -> Dict[ActionKey, float]:
        scored = {
            action_key: float(guidance_scores[action_key])
            for action_key in candidates
            if action_key in guidance_scores
            and math.isfinite(float(guidance_scores[action_key]))
        }
        unique = sorted(set(scored.values()))
        if not unique:
            return {}
        if len(unique) == 1:
            return {
                action_key: 0.0
                for action_key in scored
            }
        rank_by_value = {
            value: 2.0 * rank / (len(unique) - 1) - 1.0
            for rank, value in enumerate(unique)
        }
        return {
            action_key: rank_by_value[value]
            for action_key, value in scored.items()
        }

    def _resource_capacities(self) -> Dict[str, int]:
        available_service = {
            service_type: sum(
                vehicle.service_type == service_type
                and vehicle.vehicle_id
                not in self.env.unavailable_vehicle_ids
                for vehicle in self.env.service_vehicles
            )
            for service_type in ("fuel", "inspection", "arm")
        }
        return {
            "fuel_servers": available_service["fuel"],
            "inspection_vehicles": available_service[
                "inspection"
            ],
            "arm_vehicles": available_service["arm"],
            "ammo_transport_vehicles": int(
                self.env.config[
                    "num_ammo_transport_vehicles"
                ]
            ),
            "lower_weapon_lifts": int(
                self.env.config["num_lower_weapon_lifts"]
            ),
            "upper_weapon_lifts": int(
                self.env.config["num_upper_weapon_lifts"]
            ),
            "personnel": int(self.env.config["num_personnel"]),
        }

    def _add_active_resource_occupancy(
        self,
        model,
        intervals_by_resource: Dict[str, List[Any]],
        personnel_demands: List[int],
        max_end: int,
    ) -> None:
        event_by_key = {
            (event.event_type, event.aircraft_id): event.time
            for event in self.env.event_queue
        }
        for vehicle in self.env.service_vehicles:
            aircraft_id = vehicle.busy_aircraft_id
            if (
                aircraft_id is None
                or vehicle.service_type
                not in {"fuel", "inspection", "arm"}
                or vehicle.vehicle_id
                in self.env.unavailable_vehicle_ids
            ):
                continue
            remaining = self._active_service_remaining(
                vehicle.service_type,
                aircraft_id,
                event_by_key,
            )
            self._append_fixed_interval(
                model,
                intervals_by_resource[vehicle.service_type],
                remaining,
                max_end,
                f"active_{vehicle.service_type}_{aircraft_id}",
            )

        for aircraft_id, aircraft in enumerate(self.env.aircraft):
            if aircraft.arm_status != 1:
                continue
            if aircraft.arm_stage == 1:
                remaining = max(
                    0.0,
                    event_by_key.get(
                        (
                            "ammo_to_assembly_done",
                            aircraft_id,
                        ),
                        self.env.time,
                    )
                    - self.env.time,
                )
                for resource in (
                    "ammo_transport",
                    "lower_lift",
                ):
                    self._append_fixed_interval(
                        model,
                        intervals_by_resource[resource],
                        remaining,
                        max_end,
                        f"active_{resource}_{aircraft_id}",
                    )
            elif aircraft.arm_stage == 3:
                remaining = max(
                    0.0,
                    event_by_key.get(
                        ("arm_done", aircraft_id),
                        self.env.time,
                    )
                    - self.env.time,
                )
                self._append_fixed_interval(
                    model,
                    intervals_by_resource["upper_lift"],
                    remaining,
                    max_end,
                    f"active_upper_lift_{aircraft_id}",
                )

        if not bool(
            self.env.config["personnel_scheduling_enabled"]
        ):
            return
        personnel_by_service = {
            "fuel": int(
                self.env.config["fuel_personnel_required"]
            ),
            "inspection": int(
                self.env.config[
                    "inspection_personnel_required"
                ]
            ),
            "arm": int(
                self.env.config["arm_personnel_required"]
            ),
        }
        for aircraft_id, aircraft in enumerate(self.env.aircraft):
            for service_type, active in (
                ("fuel", aircraft.fuel_status == 1),
                (
                    "inspection",
                    aircraft.inspection_status == 1,
                ),
                ("arm", aircraft.arm_status == 1),
            ):
                if not active:
                    continue
                remaining = self._active_service_remaining(
                    service_type,
                    aircraft_id,
                    event_by_key,
                )
                before = len(
                    intervals_by_resource["personnel"]
                )
                self._append_fixed_interval(
                    model,
                    intervals_by_resource["personnel"],
                    remaining,
                    max_end,
                    (
                        "active_personnel_"
                        f"{service_type}_{aircraft_id}"
                    ),
                )
                if (
                    len(intervals_by_resource["personnel"])
                    > before
                ):
                    personnel_demands.append(
                        personnel_by_service[service_type]
                    )

    def _active_service_remaining(
        self,
        service_type: str,
        aircraft_id: int,
        event_by_key: Mapping[Tuple[str, int], float],
    ) -> float:
        event_type = {
            "fuel": "fuel_done",
            "inspection": "inspection_done",
            "arm": "arm_done",
        }[service_type]
        event_time = event_by_key.get((event_type, aircraft_id))
        if event_time is not None:
            return max(0.0, event_time - self.env.time)
        if service_type != "arm":
            return 0.0
        aircraft = self.env.aircraft[aircraft_id]
        first_stage_remaining = max(
            0.0,
            event_by_key.get(
                ("ammo_to_assembly_done", aircraft_id),
                self.env.time,
            )
            - self.env.time,
        )
        upper_stage = (
            float(self.env.config["upper_lift_time_mean"])
            + aircraft.arm_quantity_required
            * float(self.env.config["arm_unit_time_mean"])
        ) * self.env.service_time_multipliers["arm"]
        return first_stage_remaining + upper_stage

    def _append_fixed_interval(
        self,
        model,
        intervals: List[Any],
        remaining: float,
        max_end: int,
        name: str,
    ) -> None:
        if remaining <= 0.0:
            return
        duration = min(max_end, self._ticks(remaining))
        intervals.append(
            model.NewFixedSizeIntervalVar(0, duration, name)
        )

    def _arm_stage_one_ticks(self) -> int:
        duration = (
            (
                float(self.env.config["ammo_extract_time_min"])
                + float(self.env.config["ammo_extract_time_max"])
            )
            / 2.0
            + float(self.env.config["lower_lift_time_mean"])
        )
        return self._ticks(duration)

    def _ticks(self, duration: float) -> int:
        return max(1, int(math.ceil(duration * self.time_scale)))

    @staticmethod
    def _add_cumulative(model, intervals: List[Any], capacity: int) -> None:
        if not intervals:
            return
        if capacity <= 0:
            raise ValueError(
                "repair model received work for an unavailable resource"
            )
        model.AddCumulative(
            intervals,
            [1] * len(intervals),
            capacity,
        )
