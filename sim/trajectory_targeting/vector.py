"""Bounded vector differential correction for deterministic OEL trajectory observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sim.numeric_backend import normalize_numeric_backend


@dataclass(frozen=True)
class VectorTargetProblem:
    variable_names: tuple[str, ...]
    goal_names: tuple[str, ...]
    initial: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    targets: tuple[float, ...]
    tolerances: tuple[float, ...]
    variable_scales: tuple[float, ...]
    max_iterations: int = 24
    derivative_step_fraction: float = 1.0e-4
    numeric_backend: str = "rust"


@dataclass(frozen=True)
class VectorTargetResult:
    status: str
    reason: str
    variables: tuple[float, ...]
    observed: tuple[float, ...] | None
    residuals: tuple[float, ...] | None
    iterations: tuple[dict, ...]


def solve_vector_target(
    problem: VectorTargetProblem,
    observe: Callable[[np.ndarray], tuple[float, ...] | np.ndarray],
) -> VectorTargetResult:
    """Correct scaled variables with central differences and bounded line search.

    A missing event or invalid trajectory is returned as a named failure, not
    silently substituted with a penalty observation.
    """
    x = np.asarray(problem.initial, dtype=float)
    lo = np.asarray(problem.lower_bounds, dtype=float)
    hi = np.asarray(problem.upper_bounds, dtype=float)
    target = np.asarray(problem.targets, dtype=float)
    tol = np.asarray(problem.tolerances, dtype=float)
    scale = np.asarray(problem.variable_scales, dtype=float)
    n, m = x.size, target.size
    backend = normalize_numeric_backend(problem.numeric_backend, error_message="numeric_backend must be python or rust")
    if (
        n == 0 or m == 0 or len(problem.variable_names) != n or len(problem.goal_names) != m
        or any(a.size != n for a in (lo, hi, scale)) or tol.size != m
        or not all(np.isfinite(a).all() for a in (x, lo, hi, target, tol, scale))
        or np.any(lo >= hi) or np.any(x < lo) or np.any(x > hi)
        or np.any(tol <= 0) or np.any(scale <= 0)
        or not 1 <= problem.max_iterations <= 100
        or not 1.0e-8 <= problem.derivative_step_fraction <= 0.1
    ):
        raise ValueError("Invalid vector target dimensions, bounds, scales, or tolerances")

    history: list[dict] = []

    def evaluate(at: np.ndarray) -> np.ndarray:
        value = np.asarray(observe(at.copy()), dtype=float)
        if value.shape != (m,) or not np.isfinite(value).all():
            raise ValueError("trajectory observation must return finite goal values")
        return value

    def result(status: str, reason: str, observed: np.ndarray | None) -> VectorTargetResult:
        residual = None if observed is None else observed - target
        return VectorTargetResult(
            status, reason, tuple(float(v) for v in x),
            None if observed is None else tuple(float(v) for v in observed),
            None if residual is None else tuple(float(v) for v in residual),
            tuple(history),
        )

    try:
        y = evaluate(x)
    except ValueError as exc:
        return result("failed", f"initial_observation: {exc}", None)
    for iteration in range(problem.max_iterations + 1):
        residual = (y - target) / tol
        if backend == "rust":
            from sim.rust_control_backend import scaled_residual_norms

            native_norm, native_max = scaled_residual_norms(y.reshape(1, m), target, tol)
            norm = float(native_norm[0])
            max_abs_residual = float(native_max[0])
        else:
            norm = float(np.linalg.norm(residual))
            max_abs_residual = float(np.max(np.abs(residual)))
        row: dict = {
            "iteration": iteration,
            "variables": x.tolist(),
            "observed": y.tolist(),
            "residuals": (y - target).tolist(),
            "scaled_residual_norm": norm,
        }
        if backend != "python":
            row["numeric_backend"] = backend
        history.append(row)
        if max_abs_residual <= 1.0:
            return result("converged", "all_goals_within_tolerance", y)
        if iteration == problem.max_iterations:
            return result("failed", "iteration_limit", y)
        left_observations = np.empty((n, m), dtype=float)
        right_observations = np.empty((n, m), dtype=float)
        jacobian_steps = np.empty(n, dtype=float)
        for j in range(n):
            h = max(scale[j] * problem.derivative_step_fraction, 1.0e-7)
            left, right = x.copy(), x.copy()
            left[j] = max(lo[j], x[j] - h)
            right[j] = min(hi[j], x[j] + h)
            if right[j] <= left[j]:
                return result("failed", f"bound_locked:{problem.variable_names[j]}", y)
            try:
                yl, yr = evaluate(left), evaluate(right)
            except ValueError as exc:
                return result("failed", f"jacobian_observation:{exc}", y)
            left_observations[j] = yl
            right_observations[j] = yr
            jacobian_steps[j] = (right[j] - left[j]) / scale[j]
        if backend == "rust":
            from sim.rust_control_backend import finite_difference_jacobian

            jac = finite_difference_jacobian(
                left_observations,
                right_observations,
                jacobian_steps,
                tol,
            )
        else:
            jac = np.empty((m, n), dtype=float)
            for j in range(n):
                jac[:, j] = (right_observations[j] - left_observations[j]) / (
                    jacobian_steps[j] * tol
                )
        singular = np.linalg.svd(jac, compute_uv=False)
        rank = int(np.linalg.matrix_rank(jac, tol=1.0e-9 * max(float(singular[0]), 1.0)))
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
        row["jacobian_scaled"] = jac.tolist()
        row["jacobian_rank"] = rank
        row["jacobian_condition"] = condition
        if rank < min(m, n):
            return result("failed", "rank_deficient_jacobian", y)
        direction = np.linalg.lstsq(jac, -residual, rcond=1.0e-12)[0] * scale
        accepted = False
        for factor in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            trial = np.clip(x + factor * direction, lo, hi)
            if np.array_equal(trial, x):
                continue
            try:
                trial_y = evaluate(trial)
            except ValueError:
                continue
            if backend == "rust":
                from sim.rust_control_backend import scaled_residual_norms

                trial_norm = float(
                    scaled_residual_norms(trial_y.reshape(1, m), target, tol)[0][0]
                )
            else:
                trial_norm = float(np.linalg.norm((trial_y - target) / tol))
            if trial_norm < norm - 1.0e-10:
                row["step_factor"] = factor
                row["accepted_scaled_residual_norm"] = trial_norm
                x, y = trial, trial_y
                accepted = True
                break
        if not accepted:
            return result("failed", "no_improving_step_within_bounds", y)
    raise AssertionError("unreachable")
