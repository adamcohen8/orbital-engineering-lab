"""Bounded, deterministic trajectory-targeting workflows."""

from sim.trajectory_targeting.apogee import ApogeeTargetResult, ApogeeTargetSpec, target_apogee
from sim.trajectory_targeting.perigee import PerigeeTargetResult, PerigeeTargetSpec, target_perigee
from sim.trajectory_targeting.scalar import (
    EventObservation,
    ScalarTargetProblem,
    ScalarTargetResult,
    solve_scalar_target,
)

__all__ = [
    "ApogeeTargetSpec",
    "ApogeeTargetResult",
    "target_apogee",
    "PerigeeTargetSpec",
    "PerigeeTargetResult",
    "target_perigee",
    "EventObservation",
    "ScalarTargetProblem",
    "ScalarTargetResult",
    "solve_scalar_target",
]
