from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim import rust_control_backend as _rust_control_backend
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.numeric_backend import normalize_numeric_backend


@dataclass(frozen=True)
class LambertSolution:
    v1_km_s: np.ndarray
    v2_km_s: np.ndarray
    time_of_flight_s: float
    short_way: bool
    revolutions: int
    converged: bool
    iterations: int
    residual_s: float


def solve_lambert_universal_variable(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    time_of_flight_s: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    short_way: bool = True,
    revolutions: int = 0,
    max_iterations: int = 100,
    tolerance_s: float = 1.0e-7,
    backend: str = "rust",
) -> LambertSolution:
    """Solve the single-revolution two-body Lambert boundary-value problem.

    The implementation uses the classical universal-variable formulation with
    a bracketed bisection search on the Stumpff parameter.  Multi-revolution
    Lambert transfers are intentionally rejected for this first deterministic
    planner surface.
    """

    if int(revolutions) != 0:
        raise ValueError("Only zero-revolution Lambert transfers are currently supported.")
    tof = float(time_of_flight_s)
    if not np.isfinite(tof) or tof <= 0.0:
        raise ValueError("Lambert time_of_flight_s must be finite and positive.")
    mu = float(mu_km3_s2)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("Lambert mu_km3_s2 must be finite and positive.")
    max_iter = _validated_max_iterations(max_iterations)
    tolerance = _validated_tolerance(tolerance_s)
    r1 = _validated_endpoint(r1_km, "r1_km")
    r2 = _validated_endpoint(r2_km, "r2_km")
    selected_backend = normalize_numeric_backend(backend, error_message="Lambert backend must be python or rust.")
    if selected_backend == "rust":
        v1, v2, residual, iterations, converged = _rust_control_backend.lambert(
            r1,
            r2,
            tof,
            mu_km3_s2=mu,
            short_way=bool(short_way),
            max_iterations=max_iter,
            tolerance_s=tolerance,
        )
        return LambertSolution(
            v1_km_s=v1,
            v2_km_s=v2,
            time_of_flight_s=tof,
            short_way=bool(short_way),
            revolutions=0,
            converged=bool(converged),
            iterations=int(iterations),
            residual_s=float(residual),
        )

    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    if r1_norm <= 0.0 or r2_norm <= 0.0:
        raise ValueError("Lambert endpoint position norms must be positive.")

    cos_dtheta = float(np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0))
    sin_dtheta = float(np.linalg.norm(np.cross(r1, r2)) / (r1_norm * r2_norm))
    if not bool(short_way):
        sin_dtheta = -sin_dtheta
    one_minus_cos = 1.0 - cos_dtheta
    if one_minus_cos <= 1.0e-14 or abs(sin_dtheta) <= 1.0e-14:
        raise ValueError("Lambert transfer angle is singular for the current solver.")
    a_lambert = sin_dtheta * float(np.sqrt(r1_norm * r2_norm / one_minus_cos))
    if abs(a_lambert) <= 1.0e-14:
        raise ValueError("Lambert geometry produced a singular transfer parameter.")

    def time_for_z(z: float) -> tuple[float, float, float, float]:
        c = _stumpff_c(z)
        s = _stumpff_s(z)
        if c <= 0.0:
            return float("nan"), float("nan"), c, s
        y = r1_norm + r2_norm + a_lambert * ((z * s - 1.0) / float(np.sqrt(c)))
        if y < 0.0:
            return float("nan"), y, c, s
        x = float(np.sqrt(y / c))
        dt = (x * x * x * s + a_lambert * float(np.sqrt(y))) / float(np.sqrt(mu))
        return float(dt), float(y), c, s

    lower = -4.0 * np.pi * np.pi
    # Zero-revolution transfers live below the first positive Stumpff pole.
    # Expand toward that pole by shrinking the angular gap; doubling z can
    # cross the pole and make bisection converge to a later-revolution root.
    upper = 0.0
    dt_upper, _, _, _ = time_for_z(float(upper))
    pole_gap_rad = float(np.pi)
    expand_count = 0
    while (not np.isfinite(dt_upper) or dt_upper < tof) and expand_count < 64:
        candidate_upper = (2.0 * float(np.pi) - pole_gap_rad) ** 2
        if candidate_upper <= upper:
            break
        candidate_dt, _, _, _ = time_for_z(float(candidate_upper))
        if not np.isfinite(candidate_dt):
            break
        upper = float(candidate_upper)
        dt_upper = float(candidate_dt)
        pole_gap_rad *= 0.5
        expand_count += 1
    if not np.isfinite(dt_upper) or dt_upper < tof:
        raise ValueError("Lambert solver could not bracket the requested time of flight.")

    z_mid = 0.0
    dt_mid = float("nan")
    y_mid = float("nan")
    c_mid = float("nan")
    s_mid = float("nan")
    residual = float("inf")
    iterations = 0
    for iteration_index in range(1, max_iter + 1):
        iterations = int(iteration_index)
        z_mid = 0.5 * (float(lower) + float(upper))
        dt_mid, y_mid, c_mid, s_mid = time_for_z(z_mid)
        if not np.isfinite(dt_mid) or y_mid < 0.0:
            lower = z_mid
            continue
        residual = dt_mid - tof
        if abs(residual) <= tolerance:
            break
        if residual <= 0.0:
            lower = z_mid
        else:
            upper = z_mid

    if not np.isfinite(dt_mid) or not np.isfinite(y_mid) or y_mid < 0.0:
        raise ValueError("Lambert solver did not converge to a finite transfer.")
    f = 1.0 - y_mid / r1_norm
    g = a_lambert * float(np.sqrt(y_mid / mu))
    gdot = 1.0 - y_mid / r2_norm
    if abs(g) <= 1.0e-14:
        raise ValueError("Lambert solver produced a singular Lagrange g coefficient.")
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    converged = bool(abs(residual) <= tolerance)
    return LambertSolution(
        v1_km_s=np.asarray(v1, dtype=float),
        v2_km_s=np.asarray(v2, dtype=float),
        time_of_flight_s=tof,
        short_way=bool(short_way),
        revolutions=0,
        converged=converged,
        iterations=int(iterations),
        residual_s=float(residual),
    )


def solve_lambert_universal_variable_batch(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    time_of_flight_s: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    short_way: bool = True,
    revolutions: int = 0,
    max_iterations: int = 100,
    tolerance_s: float = 1.0e-7,
    backend: str = "rust",
) -> tuple[tuple[LambertSolution | None, ...], tuple[tuple[int, str], ...]]:
    """Evaluate independent Lambert candidates with one native crossing.

    The Python backend intentionally reuses the scalar reference solver.  The
    Rust backend returns one slot per candidate and a sparse error list so a
    singular candidate cannot corrupt the remaining sweep.
    """

    if int(revolutions) != 0:
        raise ValueError("Only zero-revolution Lambert transfers are currently supported.")
    first = np.asarray(r1_km, dtype=float).reshape(-1, 3)
    second = np.asarray(r2_km, dtype=float).reshape(-1, 3)
    times = np.asarray(time_of_flight_s, dtype=float).reshape(-1)
    if first.shape != second.shape or first.shape[0] != times.size:
        raise ValueError("Lambert batch arrays have inconsistent shapes")
    mu = float(mu_km3_s2)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("Lambert mu_km3_s2 must be finite and positive.")
    max_iter = _validated_max_iterations(max_iterations)
    tolerance = _validated_tolerance(tolerance_s)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Lambert batch endpoint positions must be finite.")
    if np.any(np.linalg.norm(first, axis=1) <= 0.0) or np.any(np.linalg.norm(second, axis=1) <= 0.0):
        raise ValueError("Lambert batch endpoint position norms must be positive.")
    if not np.isfinite(times).all():
        raise ValueError("Lambert batch time_of_flight_s values must be finite.")
    selected_backend = normalize_numeric_backend("python" if backend is None else backend, error_message="Lambert backend must be python or rust.")
    if selected_backend == "python":
        values: list[LambertSolution | None] = []
        errors: list[tuple[int, str]] = []
        for index, (left, right, tof) in enumerate(zip(first, second, times, strict=True)):
            try:
                values.append(
                    solve_lambert_universal_variable(
                        left,
                        right,
                        float(tof),
                        mu_km3_s2=mu,
                        short_way=short_way,
                        revolutions=0,
                        max_iterations=max_iter,
                        tolerance_s=tolerance,
                        backend="python",
                    )
                )
            except (ValueError, FloatingPointError) as exc:
                values.append(None)
                errors.append((index, str(exc)))
        return tuple(values), tuple(errors)
    rows, raw_errors = _rust_control_backend.lambert_batch(
        first,
        second,
        times,
        mu_km3_s2=mu,
        short_way=bool(short_way),
        max_iterations=max_iter,
        tolerance_s=tolerance,
    )
    error_by_index = {int(index): str(message) for index, message in raw_errors}
    values = []
    for index, tof in enumerate(times):
        if index in error_by_index:
            values.append(None)
            continue
        row = rows[index]
        encoded_iterations = float(row[7])
        # Successful native rows encode a positive integer plus zero or 0.5.
        iterations = int(encoded_iterations)
        converged = abs(encoded_iterations - iterations) < 0.25
        values.append(
            LambertSolution(
                v1_km_s=row[0:3].copy(),
                v2_km_s=row[3:6].copy(),
                time_of_flight_s=float(tof),
                short_way=bool(short_way),
                revolutions=0,
                converged=converged,
                iterations=iterations,
                residual_s=float(row[6]),
            )
        )
    return tuple(values), tuple(raw_errors)


def _validated_endpoint(value: np.ndarray, name: str) -> np.ndarray:
    try:
        endpoint = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Lambert {name} must contain exactly three numeric values.") from exc
    if endpoint.size != 3:
        raise ValueError(f"Lambert {name} must contain exactly three numeric values.")
    endpoint = endpoint.reshape(3)
    if not np.isfinite(endpoint).all():
        raise ValueError(f"Lambert {name} must be finite.")
    if float(np.linalg.norm(endpoint)) <= 0.0:
        raise ValueError(f"Lambert {name} position norm must be positive.")
    return endpoint


def _validated_max_iterations(value: int) -> int:
    try:
        iterations = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Lambert max_iterations must be a positive integer.") from exc
    if iterations <= 0:
        raise ValueError("Lambert max_iterations must be a positive integer.")
    return iterations


def _validated_tolerance(value: float) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Lambert tolerance_s must be finite and positive.") from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Lambert tolerance_s must be finite and positive.")
    return tolerance


def _stumpff_c(z: float) -> float:
    zf = float(z)
    if zf > 1.0e-8:
        root = float(np.sqrt(zf))
        return float((1.0 - np.cos(root)) / zf)
    if zf < -1.0e-8:
        root = float(np.sqrt(-zf))
        return float((np.cosh(root) - 1.0) / (-zf))
    return 0.5 - zf / 24.0 + zf * zf / 720.0 - zf * zf * zf / 40320.0


def _stumpff_s(z: float) -> float:
    zf = float(z)
    if zf > 1.0e-8:
        root = float(np.sqrt(zf))
        return float((root - np.sin(root)) / (root * root * root))
    if zf < -1.0e-8:
        root = float(np.sqrt(-zf))
        return float((np.sinh(root) - root) / (root * root * root))
    return 1.0 / 6.0 - zf / 120.0 + zf * zf / 5040.0 - zf * zf * zf / 362880.0
