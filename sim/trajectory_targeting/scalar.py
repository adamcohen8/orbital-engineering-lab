"""Bounded scalar vary/propagate-to-event/achieve correction.

The caller supplies an OEL trajectory evaluator. This module only adjusts one
scalar variable; it does not implement or substitute orbital dynamics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class EventObservation:
    value: float
    time_s: float | None
    state_eci: tuple[float, ...] | None


@dataclass(frozen=True)
class ScalarTargetProblem:
    variable_name: str
    event_name: str
    metric_name: str
    target_value: float
    lower_bound: float
    upper_bound: float
    tolerance: float
    max_iterations: int = 24


@dataclass(frozen=True)
class ScalarTargetResult:
    status: str
    reason: str
    variable_value: float | None
    observation: EventObservation | None
    residual: float | None
    iterations: tuple[dict[str, float], ...]

    def to_dict(self) -> dict:
        return asdict(self)


def solve_scalar_target(
    problem: ScalarTargetProblem,
    evaluate: Callable[[float], EventObservation],
) -> ScalarTargetResult:
    """Solve one bracketed target with safeguarded differential correction.

    The bracket endpoints must straddle the target. The evaluator must return
    finite measurements and is responsible for propagating to the named event.
    """
    numbers = (problem.target_value, problem.lower_bound, problem.upper_bound, problem.tolerance)
    if not all(math.isfinite(float(value)) for value in numbers):
        raise ValueError("Target, variable bounds, and tolerance must be finite")
    if problem.lower_bound >= problem.upper_bound or problem.tolerance <= 0.0:
        raise ValueError("Variable bounds must increase and tolerance must be positive")
    if not 1 <= problem.max_iterations <= 100:
        raise ValueError("max_iterations must be in [1, 100]")
    if not problem.variable_name or not problem.event_name or not problem.metric_name:
        raise ValueError("Variable, event, and metric names are required")

    cache: dict[float, EventObservation] = {}

    def observed(variable: float) -> EventObservation:
        if variable not in cache:
            value = evaluate(variable)
            if not math.isfinite(float(value.value)):
                raise ValueError("Trajectory evaluator returned a non-finite metric")
            cache[variable] = value
        return cache[variable]

    low_value = observed(problem.lower_bound).value
    high_value = observed(problem.upper_bound).value
    direction = 1.0 if high_value > low_value else -1.0
    if low_value == high_value:
        return ScalarTargetResult("unreachable", "Bound endpoints have the same event metric", None, None, None, ())
    if abs(low_value - problem.target_value) <= problem.tolerance:
        obs = observed(problem.lower_bound)
        return ScalarTargetResult(
            "converged", "Lower bound satisfies target", problem.lower_bound, obs, obs.value - problem.target_value, ()
        )
    if abs(high_value - problem.target_value) <= problem.tolerance:
        obs = observed(problem.upper_bound)
        return ScalarTargetResult(
            "converged", "Upper bound satisfies target", problem.upper_bound, obs, obs.value - problem.target_value, ()
        )
    if direction * (low_value - problem.target_value) > 0.0 or direction * (high_value - problem.target_value) < 0.0:
        return ScalarTargetResult(
            "unreachable", "Target is outside the evaluated variable bounds", None, None, None, ()
        )

    lo, hi = problem.lower_bound, problem.upper_bound
    variable = 0.5 * (lo + hi)
    rows: list[dict[str, float]] = []
    for index in range(problem.max_iterations):
        obs = observed(variable)
        residual = obs.value - problem.target_value
        rows.append(
            {
                "iteration": float(index),
                "variable_value": float(variable),
                "event_value": float(obs.value),
                "residual": float(residual),
            }
        )
        if abs(residual) <= problem.tolerance:
            return ScalarTargetResult(
                "converged", "Event metric satisfies tolerance", variable, obs, residual, tuple(rows)
            )
        if direction * residual < 0.0:
            lo = variable
        else:
            hi = variable
        perturbation = min(max(0.001, 1.0e-3 * max(abs(variable), 1.0)), 0.25 * (hi - lo))
        probe = variable + perturbation if variable + perturbation < hi else variable - perturbation
        derivative = (observed(probe).value - obs.value) / (probe - variable)
        proposal = variable - residual / derivative if direction * derivative > 0.0 else math.nan
        variable = proposal if math.isfinite(proposal) and lo < proposal < hi else 0.5 * (lo + hi)
    return ScalarTargetResult(
        "nonconverged",
        "Differential correction exhausted max_iterations",
        None,
        None,
        rows[-1]["residual"],
        tuple(rows),
    )
