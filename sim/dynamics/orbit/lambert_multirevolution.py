"""Enumerate elliptic two-body Lambert solutions at a fixed positive revolution count."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.lambert import LambertSolution, _validated_endpoint, _validated_tolerance


@dataclass(frozen=True)
class MultiRevolutionLambertBranch(LambertSolution):
    branch: str
    phase_rad: float


@dataclass(frozen=True)
class MultiRevolutionLambertResult:
    minimum_time_s: float
    branches: tuple[MultiRevolutionLambertBranch, ...]


def solve_multi_revolution_lambert(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    time_of_flight_s: float,
    *,
    revolutions: int,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    short_way: bool = True,
    tolerance_s: float = 1.0e-7,
) -> MultiRevolutionLambertResult:
    """Return all branches at the given geometry, orientation and revolution count.

    Phase is the elliptic eccentric-anomaly increment modulo 2π. Branches are
    ordered by phase. Times within tolerance_s of the minimum yield one
    tangent branch. Collinear endpoints require a separate plane convention.
    """
    if isinstance(revolutions, bool) or not isinstance(revolutions, Integral) or revolutions < 1:
        raise ValueError("Multi-revolution Lambert requires a positive integer revolution count.")
    if not isinstance(short_way, (bool, np.bool_)):
        raise ValueError("short_way must be a boolean.")
    tof = float(time_of_flight_s)
    mu = float(mu_km3_s2)
    tolerance = _validated_tolerance(tolerance_s)
    if not math.isfinite(tof) or not math.isfinite(mu) or min(tof, mu) <= 0:
        raise ValueError("Lambert flight time and gravitational parameter must be finite and positive.")
    r1 = _validated_endpoint(r1_km, "r1_km")
    r2 = _validated_endpoint(r2_km, "r2_km")
    n1, n2 = float(np.linalg.norm(r1)), float(np.linalg.norm(r2))
    u1, u2 = r1 / n1, r2 / n2
    sine = float(np.linalg.norm(np.cross(u1, u2)))
    if sine <= 1.0e-14:
        raise ValueError("Multi-revolution Lambert requires noncollinear endpoints.")
    angle = math.atan2(sine, float(np.dot(u1, u2)))
    geometric_mean = math.sqrt(n1) * math.sqrt(n2)
    k = 2.0 * geometric_mean * math.cos(angle / 2)
    # Evaluate r1+r2-|k| without subtracting nearly equal orbital radii.
    y_floor = (math.sqrt(n1) - math.sqrt(n2)) ** 2 + 4 * geometric_mean * math.sin(angle / 4) ** 2
    a = (k if short_way else -k) / math.sqrt(2.0)
    sqrt_mu = math.sqrt(mu)
    turns = math.tau * int(revolutions)

    def evaluate(phase):
        # Classical universal variables, sqrt(z) = 2πM + phase. Factoring
        # the Stumpff functions removes their poles from the open bracket.
        half_sin = math.sin(phase / 2)
        geometry_phase = phase if short_way else math.tau - phase
        y = y_floor + 2 * k * math.sin(geometry_phase / 4) ** 2
        root_y = math.sqrt(y)
        scale = root_y / (math.sqrt(2.0) * half_sin)
        d = turns + phase - math.sin(phase)
        time = (scale**3 * d + a * root_y) / sqrt_mu
        dy = (k if short_way else -k) * half_sin / 2
        # Sign of dT/dphase after removal of its positive scale factor.
        slope = 2 * half_sin**2 + 1.5 * d * (dy / y - math.cos(phase / 2) / half_sin) + a * dy / (2 * root_y * scale**3)
        return time, slope, y

    # On each positive-revolution interval time diverges at both ends and
    # has a single minimum. Bracket it before searching either monotone side.
    lo, hi = 0.0, math.tau
    for _ in range(64):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if evaluate(mid)[1] < 0:
            lo = mid
        else:
            hi = mid
    minimum_phase = (lo + hi) / 2
    minimum_time = evaluate(minimum_phase)[0]

    def branch(phase, label, iterations):
        actual, _, y = evaluate(phase)
        g = a * math.sqrt(y / mu)
        f, gdot = 1 - y / n1, 1 - y / n2
        return MultiRevolutionLambertBranch(
            v1_km_s=(r2 - f * r1) / g,
            v2_km_s=(gdot * r2 - r1) / g,
            time_of_flight_s=tof,
            short_way=short_way,
            revolutions=int(revolutions),
            converged=True,
            iterations=iterations,
            residual_s=actual - tof,
            branch=label,
            phase_rad=phase,
        )

    if abs(minimum_time - tof) <= tolerance:
        branches = (branch(minimum_phase, "minimum", 0),)
    elif tof < minimum_time:
        branches = ()
    else:

        def root(left, right, increasing, label):
            for iteration in range(1, 129):
                phase = (left + right) / 2
                if phase in (left, right):
                    break
                error = evaluate(phase)[0] - tof
                if abs(error) <= tolerance:
                    return branch(phase, label, iteration)
                if (error < 0) == increasing:
                    left = phase
                else:
                    right = phase
            raise ValueError("Lambert time tolerance is below the attainable precision on this branch.")

        branches = (root(0, minimum_phase, False, "lower_phase"), root(minimum_phase, math.tau, True, "higher_phase"))
    return MultiRevolutionLambertResult(minimum_time_s=minimum_time, branches=branches)
