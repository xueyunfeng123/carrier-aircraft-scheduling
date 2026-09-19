"""Shared state-quality scores for RL training and inference search."""

from __future__ import annotations

from env.carrier_aircraft_env import CarrierAircraftSchedulingEnv


def readiness_potential(env: CarrierAircraftSchedulingEnv) -> float:
    """Count completed sorties and completed turnaround services."""

    service_progress = sum(
        int(aircraft.fuel_status == 2)
        + int(aircraft.inspection_status == 2)
        + int(aircraft.arm_status == 2)
        for aircraft in env.aircraft
        if not aircraft.is_airborne
        and aircraft.recovery_status == 2
    )
    completed_sorties = sum(
        aircraft.sorties_completed
        for aircraft in env.aircraft
    )
    return float(3 * completed_sorties + service_progress)
