"""Rolling resource-allocation baseline implemented with OR-Tools CP-SAT."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from solution.priority_rule_solver import (
    ACTION_ARM,
    ACTION_FUEL,
    ACTION_LAUNCH,
    ACTION_RECOVERY,
    PriorityRuleSolver,
)


class CPSATSolver:
    """Optimize the currently dispatchable service batch at each decision time.

    This is a rolling-horizon baseline, not a globally optimal solution of the
    full stochastic episode. Launch and recovery use independent resources and
    are dispatched immediately. CP-SAT allocates the shared personnel and
    fueling/arming resources among currently eligible service operations.
    """

    def __init__(
        self,
        env: CarrierAircraftSchedulingEnv,
        max_time_seconds: float = 0.05,
    ):
        try:
            from ortools.sat.python import cp_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OR-Tools is required for `--solver cp_sat`. "
                "Install it with `pip install ortools`."
            ) from exc

        self.env = env
        self.cp_model = cp_model
        self.max_time_seconds = max_time_seconds
        self._priority = PriorityRuleSolver(env, "edd")

    def choose_action(self) -> Optional[Dict[str, int]]:
        mask = self.env.get_action_mask()
        if not any(mask["high_level"]):
            return None

        recovery_ids = (
            self._candidate_ids(mask, ACTION_RECOVERY)
            if mask["high_level"][ACTION_RECOVERY]
            else []
        )
        launch_ids = (
            self._candidate_ids(mask, ACTION_LAUNCH)
            if mask["high_level"][ACTION_LAUNCH]
            else []
        )
        if recovery_ids and launch_ids and self.env.num_shared_launch_channels > 0:
            aircraft_id = min(recovery_ids)
            return {"high_level": ACTION_RECOVERY, "aircraft_id": aircraft_id}
        if launch_ids:
            aircraft_id = min(
                launch_ids,
                key=lambda item: (
                    self.env.aircraft[item].launch_ready or self.env.time,
                    item,
                ),
            )
            return {"high_level": ACTION_LAUNCH, "aircraft_id": aircraft_id}
        if recovery_ids:
            aircraft_id = min(
                recovery_ids,
                key=lambda item: (self._priority._launch_deadline(item), item),
            )
            return {"high_level": ACTION_RECOVERY, "aircraft_id": aircraft_id}

        service_actions = [
            (high_level, aircraft_id)
            for high_level in (ACTION_FUEL, ACTION_ARM)
            if mask["high_level"][high_level]
            for aircraft_id in self._candidate_ids(mask, high_level)
        ]
        if not service_actions:
            return None
        if len(service_actions) == 1:
            high_level, aircraft_id = service_actions[0]
            return {"high_level": high_level, "aircraft_id": aircraft_id}

        selected = self._solve_service_batch(service_actions)
        if not selected:
            selected = service_actions
        high_level, aircraft_id = min(selected, key=self._priority._edd_key)
        return {"high_level": high_level, "aircraft_id": aircraft_id}

    def _candidate_ids(self, mask: Dict[str, object], high_level: int) -> List[int]:
        low_mask = mask["low_level_by_high"][high_level]
        return [
            aircraft_id
            for aircraft_id, allowed in enumerate(low_mask)
            if allowed
        ]

    def _solve_service_batch(
        self,
        service_actions: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        model = self.cp_model.CpModel()
        variables = {
            candidate: model.NewBoolVar(f"a_{candidate[0]}_{candidate[1]}")
            for candidate in service_actions
        }

        fuel_vars = [
            variable
            for (action, _), variable in variables.items()
            if action == ACTION_FUEL
        ]
        arm_vars = [
            variable
            for (action, _), variable in variables.items()
            if action == ACTION_ARM
        ]
        model.Add(sum(fuel_vars) <= self.env.free_fuel_servers)
        model.Add(
            sum(arm_vars)
            <= min(
                self.env.free_ammo_transport_vehicles,
                self.env.free_lower_weapon_lifts,
            )
        )

        fuel_personnel = int(self.env.config["fuel_personnel_required"])
        arm_personnel = int(self.env.config["arm_personnel_required"])
        model.Add(
            fuel_personnel * sum(fuel_vars)
            + arm_personnel * sum(arm_vars)
            <= self.env.free_personnel
        )

        objective_terms = []
        candidate_aircraft = sorted({aircraft_id for _, aircraft_id in service_actions})
        for aircraft_id in candidate_aircraft:
            required_vars = [
                variables[candidate]
                for candidate in service_actions
                if candidate[1] == aircraft_id
            ]
            ready = model.NewBoolVar(f"ready_{aircraft_id}")
            for variable in required_vars:
                model.Add(ready <= variable)
            model.Add(ready >= sum(required_vars) - len(required_vars) + 1)
            objective_terms.append(1_000_000 * ready)

        horizon = max(1.0, self.env.simulation_duration - self.env.time)
        for candidate, variable in variables.items():
            high_level, aircraft_id = candidate
            deadline_remaining = max(
                0.0,
                self._priority._launch_deadline(aircraft_id) - self.env.time,
            )
            duration = self._priority._expected_action_duration(
                high_level,
                aircraft_id,
            )
            urgency = max(0, int(round((horizon - min(horizon, deadline_remaining)) * 10)))
            shortness = max(0, 1000 - int(round(duration * 10)))
            objective_terms.append((1000 + urgency + shortness) * variable)

        model.Maximize(sum(objective_terms))
        solver = self.cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.max_time_seconds
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        if status not in (self.cp_model.OPTIMAL, self.cp_model.FEASIBLE):
            return []
        return [
            candidate
            for candidate, variable in variables.items()
            if solver.Value(variable)
        ]
