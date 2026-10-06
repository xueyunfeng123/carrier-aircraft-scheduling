"""Tests for reproducible dynamic resource disruptions."""

from __future__ import annotations

import unittest

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv
from rl.obs_encoder import encode_observation
from scripts.benchmark_dynamic_disruptions import build_summary


class DynamicDisruptionTest(unittest.TestCase):
    def test_benchmark_summary_groups_scenarios_profiles_and_loads(
        self,
    ) -> None:
        rows = []
        for profile, interval, heuristic, cp_sat, candidate in (
            ("none", 50.0, 100, 101, 102),
            ("none", 70.0, 120, 121, 122),
            ("heavy", 50.0, 90, 94, 96),
            ("heavy", 70.0, 110, 112, 113),
        ):
            for solver, completed in (
                ("heuristic", heuristic),
                ("cp_sat", cp_sat),
                ("rl_randomized", candidate),
            ):
                rows.append(
                    {
                        "profile": profile,
                        "wave_interval": interval,
                        "solver": solver,
                        "total_sorties_completed": completed,
                        "total_missed_sorties": 240 - completed,
                        "worst_wave": 5,
                        "runtime_seconds": 1.0,
                    }
                )

        summary = build_summary(rows)
        candidate_overall = next(
            row
            for row in summary
            if row["scope"] == "overall"
            and row["solver"] == "rl_randomized"
        )
        heavy_candidate = next(
            row
            for row in summary
            if row["scope"] == "profile"
            and row["profile"] == "heavy"
            and row["solver"] == "rl_randomized"
        )

        self.assertEqual(candidate_overall["mean_total_sorties"], 108.25)
        self.assertEqual(candidate_overall["worst_scenario_mean"], 96)
        self.assertEqual(candidate_overall["delta_vs_cp_sat"], 1.25)
        self.assertEqual(heavy_candidate["mean_total_sorties"], 104.5)

    def test_profile_rng_is_reproducible_and_independent(self) -> None:
        base = self._config(disruption_profile="none")
        heavy = self._config(disruption_profile="heavy")

        no_disruption = CarrierAircraftSchedulingEnv(base)
        no_disruption.reset(seed=19)
        first = CarrierAircraftSchedulingEnv(heavy)
        first.reset(seed=19)
        second = CarrierAircraftSchedulingEnv(heavy)
        second.reset(seed=19)
        different = CarrierAircraftSchedulingEnv(heavy)
        different.reset(seed=20)

        self.assertEqual(
            [item.as_dict() for item in first.disruption_schedule],
            [item.as_dict() for item in second.disruption_schedule],
        )
        self.assertNotEqual(
            [item.as_dict() for item in first.disruption_schedule],
            [item.as_dict() for item in different.disruption_schedule],
        )
        self.assertEqual(
            [no_disruption.rng.random() for _ in range(5)],
            [first.rng.random() for _ in range(5)],
        )

    def test_future_disruptions_do_not_leak_into_observation(self) -> None:
        baseline = CarrierAircraftSchedulingEnv(
            self._config(disruption_profile="none")
        )
        disrupted = CarrierAircraftSchedulingEnv(
            self._config(disruption_profile="light")
        )
        baseline.reset(seed=19)
        disrupted.reset(seed=19)

        baseline_observation = encode_observation(baseline)
        disrupted_observation = encode_observation(disrupted)

        self.assertEqual(
            baseline_observation,
            disrupted_observation,
        )
        self.assertNotEqual(
            baseline.disruption_schedule,
            disrupted.disruption_schedule,
        )

    def test_profiles_are_nested_for_a_paired_seed(self) -> None:
        schedules = {}
        for profile in ("light", "medium", "heavy"):
            env = CarrierAircraftSchedulingEnv(
                {"disruption_profile": profile}
            )
            env.reset(seed=19)
            schedules[profile] = {
                (item.start_time, item.kind, item.target): item.end_time
                for item in env.disruption_schedule
            }

        self.assertEqual(
            [len(schedules[name]) for name in schedules],
            [15, 30, 54],
        )
        self.assertTrue(
            set(schedules["light"]) <= set(schedules["medium"])
        )
        self.assertTrue(
            set(schedules["medium"]) <= set(schedules["heavy"])
        )
        for key, light_end in schedules["light"].items():
            self.assertLessEqual(
                light_end,
                schedules["medium"][key],
            )
        for key, medium_end in schedules["medium"].items():
            self.assertLessEqual(
                medium_end,
                schedules["heavy"][key],
            )

    def test_vehicle_outage_masks_target_and_restores_it(self) -> None:
        env = self._explicit_env(
            {
                "kind": "vehicle_outage",
                "target": "fuel_0",
                "start_time": 1.0,
                "end_time": 3.0,
            },
            num_fuel_servers=2,
        )
        aircraft = env.aircraft[0]
        aircraft.fuel_status = 0
        aircraft.fuel_level = 0.2
        env._invalidate_planning_cache()

        self.assertEqual(
            env.get_target_candidates(1, 0),
            ["fuel_0", "fuel_1"],
        )
        self._advance_to(env, 1.0)
        self.assertEqual(env.get_target_candidates(1, 0), ["fuel_1"])
        self.assertEqual(env.get_state()["resources"]["fuel_servers"], 1)

        self._advance_to(env, 3.0)
        self.assertEqual(
            env.get_target_candidates(1, 0),
            ["fuel_0", "fuel_1"],
        )
        self.assertEqual(env.get_state()["resources"]["fuel_servers"], 2)

    def test_runway_closure_blocks_non_spatial_launch(self) -> None:
        env = self._explicit_env(
            {
                "kind": "runway_closure",
                "target": 0,
                "start_time": 1.0,
                "end_time": 3.0,
            },
            num_launch_positions=1,
            num_launch_channels=1,
        )

        self.assertEqual(env.get_target_candidates(3, 0), [0])
        self._advance_to(env, 1.0)
        self.assertEqual(env.get_target_candidates(3, 0), [])
        self.assertEqual(env.get_high_level_action_mask()[3], 0)
        encoded = encode_observation(env)
        runway_slot = encoded.target_keys.index(("runway", 0))
        self.assertEqual(encoded.targets[runway_slot][7], 0.0)

        self._advance_to(env, 3.0)
        self.assertEqual(env.get_target_candidates(3, 0), [0])
        self.assertEqual(env.get_high_level_action_mask()[3], 1)

    def test_aircraft_hold_allows_recovery_but_blocks_new_work(self) -> None:
        env = self._explicit_env(
            {
                "kind": "aircraft_hold",
                "target": 0,
                "start_time": 1.0,
                "end_time": 3.0,
            }
        )
        aircraft = env.aircraft[0]
        env.parking_occupancy[aircraft.spot_id] = None
        aircraft.spot_id = -1
        aircraft.parking_status = 0
        aircraft.is_airborne = True
        aircraft.pending_recovery = True
        aircraft.recovery_status = 0
        aircraft.fuel_status = 0
        aircraft.inspection_status = 0
        aircraft.arm_status = 0
        aircraft.launch_status = 0
        env._invalidate_planning_cache()

        self._advance_to(env, 1.0)
        candidates = env._candidate_sets()
        self.assertIn(0, candidates["R"])
        for action_name in ("F", "M", "L", "I"):
            self.assertNotIn(0, candidates[action_name])

        self._advance_to(env, 3.0)
        candidates = env._candidate_sets()
        self.assertIn(0, candidates["R"])

    def test_service_slowdown_only_changes_new_work_duration(self) -> None:
        env = self._explicit_env(
            {
                "kind": "service_slowdown",
                "target": "fuel",
                "start_time": 1.0,
                "end_time": 5.0,
                "multiplier": 2.0,
            },
            simulation_duration=40.0,
            wave_interval=40.0,
        )
        self._advance_to(env, 1.0)
        self.assertAlmostEqual(
            encode_observation(env).global_features[6],
            0.5,
        )
        aircraft = env.aircraft[0]
        aircraft.fuel_status = 0
        aircraft.fuel_level = 0.5
        env._invalidate_planning_cache()

        action = env.complete_action(
            {"high_level": 1, "aircraft_id": 0}
        )
        _, _, _, info = env.step(action)
        self.assertFalse(info["invalid_action"])
        start_record = next(
            item
            for item in reversed(env.event_log)
            if item["event_type"] == "fuel_start"
        )
        self.assertAlmostEqual(start_record["duration"], 20.0)
        completion_time = next(
            event.time
            for event in env.event_queue
            if event.event_type == "fuel_done"
            and event.aircraft_id == 0
        )

        self._advance_to(env, 5.0)
        self.assertEqual(env.service_time_multipliers["fuel"], 1.0)
        self.assertEqual(
            next(
                event.time
                for event in env.event_queue
                if event.event_type == "fuel_done"
                and event.aircraft_id == 0
            ),
            completion_time,
        )

        self._advance_to(env, completion_time)
        self.assertEqual(aircraft.fuel_status, 2)

    def _explicit_env(
        self,
        disruption: dict,
        **overrides: object,
    ) -> CarrierAircraftSchedulingEnv:
        config = self._config(
            disruptions=[disruption],
            **overrides,
        )
        env = CarrierAircraftSchedulingEnv(config)
        env.reset(seed=7)
        return env

    def _config(self, **overrides: object) -> dict:
        config = {
            "num_aircraft": 4,
            "group_size": 2,
            "num_parking_spots": 4,
            "simulation_duration": 20.0,
            "wave_interval": 10.0,
            "spatial_graph_enabled": False,
        }
        config.update(overrides)
        return config

    def _advance_to(
        self,
        env: CarrierAircraftSchedulingEnv,
        target_time: float,
    ) -> None:
        while env.time < target_time:
            env._advance_time_to_next_event()
        self.assertEqual(env.time, target_time)


if __name__ == "__main__":
    unittest.main()
