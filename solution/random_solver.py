"""Random action selector for the scheduling environment."""

from __future__ import annotations

import random
from typing import Dict, Optional

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv


class RandomSolver:
    """Uniformly samples one legal high-level action and one legal aircraft."""

    def __init__(self, env: CarrierAircraftSchedulingEnv, seed: Optional[int] = None):
        self.env = env
        self.rng = random.Random(seed)

    def choose_action(self) -> Optional[Dict[str, int]]:
        mask = self.env.get_action_mask()
        available_high_levels = [
            action_id
            for action_id, is_available in enumerate(mask["high_level"])
            if is_available
        ]
        if not available_high_levels:
            return None

        high_level = self.rng.choice(available_high_levels)
        low_mask = mask["low_level_by_high"][high_level]
        available_aircraft = [
            aircraft_id
            for aircraft_id, is_available in enumerate(low_mask)
            if is_available
        ]
        aircraft_id = self.rng.choice(available_aircraft)
        return {"high_level": high_level, "aircraft_id": aircraft_id}

