from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AdaptiveStepInfo:
    method: str
    accepted_steps: int = 0
    rejected_steps: int = 0
    attempted_steps: int = 0
    min_step_s: float | None = None
    max_step_s: float | None = None
    final_step_s: float | None = None
    suggested_next_step_s: float | None = None
    max_error_ratio: float = 0.0

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "method": self.method,
            "accepted_steps": int(self.accepted_steps),
            "rejected_steps": int(self.rejected_steps),
            "attempted_steps": int(self.attempted_steps),
            "min_step_s": self.min_step_s,
            "max_step_s": self.max_step_s,
            "final_step_s": self.final_step_s,
            "suggested_next_step_s": self.suggested_next_step_s,
            "max_error_ratio": float(self.max_error_ratio),
        }


def combine_adaptive_step_info(method: str, items: list[AdaptiveStepInfo]) -> AdaptiveStepInfo:
    nonempty = [item for item in items if item.attempted_steps > 0]
    if not nonempty:
        return AdaptiveStepInfo(method=method)
    min_steps = [float(item.min_step_s) for item in nonempty if item.min_step_s is not None]
    max_steps = [float(item.max_step_s) for item in nonempty if item.max_step_s is not None]
    return AdaptiveStepInfo(
        method=method,
        accepted_steps=sum(int(item.accepted_steps) for item in nonempty),
        rejected_steps=sum(int(item.rejected_steps) for item in nonempty),
        attempted_steps=sum(int(item.attempted_steps) for item in nonempty),
        min_step_s=min(min_steps) if min_steps else None,
        max_step_s=max(max_steps) if max_steps else None,
        final_step_s=nonempty[-1].final_step_s,
        suggested_next_step_s=nonempty[-1].suggested_next_step_s,
        max_error_ratio=max(float(item.max_error_ratio) for item in nonempty),
    )


def rk4_step_state(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> np.ndarray:
    k1 = deriv_fn(t_s, x)
    k2 = deriv_fn(t_s + 0.5 * dt_s, x + 0.5 * dt_s * k1)
    k3 = deriv_fn(t_s + 0.5 * dt_s, x + 0.5 * dt_s * k2)
    k4 = deriv_fn(t_s + dt_s, x + dt_s * k3)
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rkf78_stage_trace(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> list[dict[str, np.ndarray | float | str]]:
    x = np.array(x, dtype=float, copy=True)
    stages: list[dict[str, np.ndarray | float | str]] = []

    def record(name: str, stage_t: float, stage_x: np.ndarray) -> np.ndarray:
        k = deriv_fn(stage_t, stage_x)
        stages.append(
            {
                "name": name,
                "t": float(stage_t),
                "x": np.array(stage_x, dtype=float, copy=True),
                "k": np.array(k, dtype=float, copy=True),
            }
        )
        return k

    k1 = record("k1", t_s, x)
    k2 = record("k2", t_s + dt_s * (2.0 / 27.0), x + dt_s * ((2.0 / 27.0) * k1))
    k3 = record("k3", t_s + dt_s * (1.0 / 9.0), x + dt_s * ((1.0 / 36.0) * k1 + (1.0 / 12.0) * k2))
    k4 = record("k4", t_s + dt_s * (1.0 / 6.0), x + dt_s * ((1.0 / 24.0) * k1 + (1.0 / 8.0) * k3))
    k5 = record(
        "k5",
        t_s + dt_s * (5.0 / 12.0),
        x + dt_s * ((5.0 / 12.0) * k1 - (25.0 / 16.0) * k3 + (25.0 / 16.0) * k4),
    )
    k6 = record(
        "k6",
        t_s + dt_s * 0.5,
        x + dt_s * ((1.0 / 20.0) * k1 + 0.25 * k4 + 0.2 * k5),
    )
    k7 = record(
        "k7",
        t_s + dt_s * (5.0 / 6.0),
        x + dt_s * (-(25.0 / 108.0) * k1 + (125.0 / 108.0) * k4 - (65.0 / 27.0) * k5 + (125.0 / 54.0) * k6),
    )
    k8 = record(
        "k8",
        t_s + dt_s * (1.0 / 6.0),
        x + dt_s * ((31.0 / 300.0) * k1 + (61.0 / 225.0) * k5 - (2.0 / 9.0) * k6 + (13.0 / 900.0) * k7),
    )
    k9 = record(
        "k9",
        t_s + dt_s * (2.0 / 3.0),
        x
        + dt_s
        * (2.0 * k1 - (53.0 / 6.0) * k4 + (704.0 / 45.0) * k5 - (107.0 / 9.0) * k6 + (67.0 / 90.0) * k7 + 3.0 * k8),
    )
    k10 = record(
        "k10",
        t_s + dt_s * (1.0 / 3.0),
        x
        + dt_s
        * (
            -(91.0 / 108.0) * k1
            + (23.0 / 108.0) * k4
            - (976.0 / 135.0) * k5
            + (311.0 / 54.0) * k6
            - (19.0 / 60.0) * k7
            + (17.0 / 6.0) * k8
            - (1.0 / 12.0) * k9
        ),
    )
    record(
        "k11",
        t_s + dt_s,
        x
        + dt_s
        * (
            (2383.0 / 4100.0) * k1
            - (341.0 / 164.0) * k4
            + (4496.0 / 1025.0) * k5
            - (301.0 / 82.0) * k6
            + (2133.0 / 4100.0) * k7
            + (45.0 / 82.0) * k8
            + (45.0 / 164.0) * k9
            + (18.0 / 41.0) * k10
        ),
    )
    record(
        "k12",
        t_s,
        x
        + dt_s
        * (
            (3.0 / 205.0) * k1
            - (6.0 / 41.0) * k6
            - (3.0 / 205.0) * k7
            - (3.0 / 41.0) * k8
            + (3.0 / 41.0) * k9
            + (6.0 / 41.0) * k10
        ),
    )
    record(
        "k13",
        t_s + dt_s,
        x
        + dt_s
        * (
            -(1777.0 / 4100.0) * k1
            - (341.0 / 164.0) * k4
            + (4496.0 / 1025.0) * k5
            - (289.0 / 82.0) * k6
            + (2193.0 / 4100.0) * k7
            + (51.0 / 82.0) * k8
            + (33.0 / 164.0) * k9
            + (12.0 / 41.0) * k10
            + stages[-1]["k"]
        ),
    )
    return stages


def rkf78_step(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Fehlberg embedded Runge-Kutta 7(8) step.

    Returns the propagated state and the embedded error estimate used for
    adaptive step-size control.
    """
    # The diagnostic trace retains a copy of every stage state and derivative.
    # Ordinary propagation only needs the derivatives, so evaluate the same
    # tableau directly and avoid retaining thirteen unused stage-state copies
    # and dictionaries on every attempted adaptive step.
    x = np.array(x, dtype=float, copy=True)
    k1 = np.asarray(deriv_fn(t_s, x), dtype=float)
    k2 = np.asarray(
        deriv_fn(t_s + dt_s * (2.0 / 27.0), x + dt_s * ((2.0 / 27.0) * k1)),
        dtype=float,
    )
    k3 = np.asarray(
        deriv_fn(t_s + dt_s * (1.0 / 9.0), x + dt_s * ((1.0 / 36.0) * k1 + (1.0 / 12.0) * k2)),
        dtype=float,
    )
    k4 = np.asarray(
        deriv_fn(t_s + dt_s * (1.0 / 6.0), x + dt_s * ((1.0 / 24.0) * k1 + (1.0 / 8.0) * k3)),
        dtype=float,
    )
    k5 = np.asarray(
        deriv_fn(
            t_s + dt_s * (5.0 / 12.0),
            x + dt_s * ((5.0 / 12.0) * k1 - (25.0 / 16.0) * k3 + (25.0 / 16.0) * k4),
        ),
        dtype=float,
    )
    k6 = np.asarray(
        deriv_fn(t_s + dt_s * 0.5, x + dt_s * ((1.0 / 20.0) * k1 + 0.25 * k4 + 0.2 * k5)),
        dtype=float,
    )
    k7 = np.asarray(
        deriv_fn(
            t_s + dt_s * (5.0 / 6.0),
            x
            + dt_s
            * (-(25.0 / 108.0) * k1 + (125.0 / 108.0) * k4 - (65.0 / 27.0) * k5 + (125.0 / 54.0) * k6),
        ),
        dtype=float,
    )
    k8 = np.asarray(
        deriv_fn(
            t_s + dt_s * (1.0 / 6.0),
            x + dt_s * ((31.0 / 300.0) * k1 + (61.0 / 225.0) * k5 - (2.0 / 9.0) * k6 + (13.0 / 900.0) * k7),
        ),
        dtype=float,
    )
    k9 = np.asarray(
        deriv_fn(
            t_s + dt_s * (2.0 / 3.0),
            x
            + dt_s
            * (2.0 * k1 - (53.0 / 6.0) * k4 + (704.0 / 45.0) * k5 - (107.0 / 9.0) * k6 + (67.0 / 90.0) * k7 + 3.0 * k8),
        ),
        dtype=float,
    )
    k10 = np.asarray(
        deriv_fn(
            t_s + dt_s * (1.0 / 3.0),
            x
            + dt_s
            * (
                -(91.0 / 108.0) * k1
                + (23.0 / 108.0) * k4
                - (976.0 / 135.0) * k5
                + (311.0 / 54.0) * k6
                - (19.0 / 60.0) * k7
                + (17.0 / 6.0) * k8
                - (1.0 / 12.0) * k9
            ),
        ),
        dtype=float,
    )
    k11 = np.asarray(
        deriv_fn(
            t_s + dt_s,
            x
            + dt_s
            * (
                (2383.0 / 4100.0) * k1
                - (341.0 / 164.0) * k4
                + (4496.0 / 1025.0) * k5
                - (301.0 / 82.0) * k6
                + (2133.0 / 4100.0) * k7
                + (45.0 / 82.0) * k8
                + (45.0 / 164.0) * k9
                + (18.0 / 41.0) * k10
            ),
        ),
        dtype=float,
    )
    k12 = np.asarray(
        deriv_fn(
            t_s,
            x
            + dt_s
            * (
                (3.0 / 205.0) * k1
                - (6.0 / 41.0) * k6
                - (3.0 / 205.0) * k7
                - (3.0 / 41.0) * k8
                + (3.0 / 41.0) * k9
                + (6.0 / 41.0) * k10
            ),
        ),
        dtype=float,
    )
    k13 = np.asarray(
        deriv_fn(
            t_s + dt_s,
            x
            + dt_s
            * (
                -(1777.0 / 4100.0) * k1
                - (341.0 / 164.0) * k4
                + (4496.0 / 1025.0) * k5
                - (289.0 / 82.0) * k6
                + (2193.0 / 4100.0) * k7
                + (51.0 / 82.0) * k8
                + (33.0 / 164.0) * k9
                + (12.0 / 41.0) * k10
                + k12
            ),
        ),
        dtype=float,
    )

    x_next = x + dt_s * (
        (41.0 / 840.0) * k1
        + (34.0 / 105.0) * k6
        + (9.0 / 35.0) * k7
        + (9.0 / 35.0) * k8
        + (9.0 / 280.0) * k9
        + (9.0 / 280.0) * k10
        + (41.0 / 840.0) * k11
    )
    err = dt_s * (41.0 / 840.0) * (k1 + k11 - k12 - k13)
    return x_next, err


def integrate_rkf78_hpop(
    deriv_fn,
    t_s: float,
    x: np.ndarray,
    dt_s: float,
    *,
    tolerance: float = 1e-10,
    h_init: float | None = None,
    max_attempts: int = 12,
    return_info: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, AdaptiveStepInfo]:
    accepted_steps = 0
    rejected_steps = 0
    attempted_steps = 0
    min_step_s: float | None = None
    max_step_s: float | None = None
    max_error_ratio = 0.0

    def _finish(y_out: np.ndarray, h_out: float, final_step_s: float | None) -> Any:
        info = AdaptiveStepInfo(
            method="rkf78_hpop",
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            attempted_steps=attempted_steps,
            min_step_s=min_step_s,
            max_step_s=max_step_s,
            final_step_s=final_step_s,
            suggested_next_step_s=float(h_out),
            max_error_ratio=max_error_ratio,
        )
        if return_info:
            return y_out, float(h_out), info
        return y_out, float(h_out)

    dt_s = float(dt_s)
    if not np.isfinite(dt_s):
        raise ValueError("dt_s must be finite.")
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("HPOP RKF78 tolerance must be positive and finite.")
    if dt_s == 0.0:
        return _finish(np.array(x, dtype=float, copy=True), float(h_init if h_init is not None else 0.01), 0.0)

    min_scale = 0.125
    max_scale = 4.0
    err_exponent = 1.0 / 7.0

    x_now = float(t_s)
    x_end = float(t_s + dt_s)
    h = float(h_init if h_init is not None else 0.01)
    if h <= 0.0:
        h = 0.01
    y = np.array(x, dtype=float, copy=True)
    if h > (x_end - x_now):
        h = x_end - x_now

    tol_per_unit = tolerance / (x_end - x_now)

    while x_now < x_end:
        scale = 1.0
        for _attempt in range(max_attempts):
            attempted_steps += 1
            h_attempt = float(h)
            y_trial, err_vec = rkf78_step(deriv_fn, x_now, y, h)
            err = float(np.linalg.norm(err_vec))
            if err == 0.0:
                scale = max_scale
                accepted_steps += 1
                min_step_s = h_attempt if min_step_s is None else min(min_step_s, h_attempt)
                max_step_s = h_attempt if max_step_s is None else max(max_step_s, h_attempt)
                break
            y_norm = float(np.linalg.norm(y))
            yy = tol_per_unit if y_norm == 0.0 else y_norm
            err_ratio = float(err / max(tol_per_unit * yy, 1e-300))
            max_error_ratio = max(max_error_ratio, err_ratio)
            scale = 0.8 * (tol_per_unit * yy / err) ** err_exponent
            scale = min(max(scale, min_scale), max_scale)
            if err < (tol_per_unit * yy):
                accepted_steps += 1
                min_step_s = h_attempt if min_step_s is None else min(min_step_s, h_attempt)
                max_step_s = h_attempt if max_step_s is None else max(max_step_s, h_attempt)
                break
            rejected_steps += 1
            h *= scale
            if x_now + h > x_end:
                h = x_end - x_now
            elif x_now + h + 0.5 * h > x_end:
                h = 0.5 * h
        else:
            raise RuntimeError(f"HPOP-style RKF78 failed to converge within {max_attempts} attempts at t={x_now:.9f}s.")

        y = y_trial
        x_now += h
        h *= scale
        h_next = h
        if x_now >= x_end:
            return _finish(y, h_next, float(h / scale) if scale != 0.0 else float(h))
        if x_now + h > x_end:
            h = x_end - x_now
        elif x_now + h + 0.5 * h > x_end:
            h = 0.5 * h

    return _finish(y, float(h), None)


def dopri45_step(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    k1 = deriv_fn(t_s, x)
    k2 = deriv_fn(t_s + dt_s * 1 / 5, x + dt_s * (1 / 5) * k1)
    k3 = deriv_fn(t_s + dt_s * 3 / 10, x + dt_s * (3 / 40 * k1 + 9 / 40 * k2))
    k4 = deriv_fn(t_s + dt_s * 4 / 5, x + dt_s * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3))
    k5 = deriv_fn(
        t_s + dt_s * 8 / 9,
        x + dt_s * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4),
    )
    k6 = deriv_fn(
        t_s + dt_s,
        x + dt_s * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3 + 49 / 176 * k4 - 5103 / 18656 * k5),
    )
    k7 = deriv_fn(
        t_s + dt_s,
        x + dt_s * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6),
    )

    x5 = x + dt_s * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)
    x4 = x + dt_s * (
        5179 / 57600 * k1 + 7571 / 16695 * k3 + 393 / 640 * k4 - 92097 / 339200 * k5 + 187 / 2100 * k6 + 1 / 40 * k7
    )
    err = x5 - x4
    return x5, err


def integrate_adaptive(
    deriv_fn,
    t_s: float,
    x: np.ndarray,
    dt_s: float,
    atol: float = 1e-9,
    rtol: float = 1e-7,
    max_substeps: int = 4096,
    method: str = "rkf78",
    h_init: float | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, AdaptiveStepInfo]:
    def _finish(state: np.ndarray, info: AdaptiveStepInfo) -> np.ndarray | tuple[np.ndarray, AdaptiveStepInfo]:
        if return_info:
            return state, info
        return state

    dt_s = float(dt_s)
    if not np.isfinite(dt_s):
        raise ValueError("dt_s must be finite.")
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative.")

    method_name = str(method).strip().lower()
    if method_name not in {"rkf78", "dopri5", "dopri45"}:
        raise ValueError(f"Unknown adaptive integrator method '{method}'.")
    atol = float(atol)
    rtol = float(rtol)
    if (
        not np.isfinite(atol)
        or atol < 0.0
        or not np.isfinite(rtol)
        or rtol < 0.0
    ):
        raise ValueError("Adaptive integrator tolerances must be finite and non-negative.")
    if dt_s == 0.0:
        return _finish(np.array(x, dtype=float, copy=True), AdaptiveStepInfo(method=method_name))

    if method_name == "rkf78":
        step_fn = rkf78_step
        growth_exponent = -1.0 / 8.0
    elif method_name in ("dopri5", "dopri45"):
        step_fn = dopri45_step
        growth_exponent = -1.0 / 5.0

    t = t_s
    xk = x
    remain = dt_s
    h = min(dt_s, float(h_init if h_init is not None and h_init > 0.0 else 1.0))
    min_h = max(1e-12, 1e-12 * max(1.0, abs(dt_s)))
    steps = 0
    accepted_steps = 0
    rejected_steps = 0
    attempted_steps = 0
    min_step_s: float | None = None
    max_step_s: float | None = None
    final_step_s: float | None = None
    max_error_ratio = 0.0

    while remain > 0.0 and steps < max_substeps:
        h = min(h, remain)
        x_next, err = step_fn(deriv_fn, t, xk, h)
        attempted_steps += 1
        scale = atol + rtol * np.maximum(np.abs(xk), np.abs(x_next))
        err_ratio = float(np.max(np.abs(err) / np.maximum(scale, 1e-14)))
        max_error_ratio = max(max_error_ratio, err_ratio)

        if err_ratio <= 1.0:
            t += h
            xk = x_next
            remain -= h
            accepted_steps += 1
            final_step_s = float(h)
            min_step_s = float(h) if min_step_s is None else min(min_step_s, float(h))
            max_step_s = float(h) if max_step_s is None else max(max_step_s, float(h))
            if err_ratio < 1e-10:
                h *= 2.0
            else:
                h *= min(2.0, max(0.5, 0.9 * err_ratio**growth_exponent))
        else:
            rejected_steps += 1
            h *= max(0.1, 0.9 * err_ratio**growth_exponent)
            if h < min_h:
                raise RuntimeError(
                    f"Adaptive integrator step size underflow at t={t:.9f}s while trying to cover dt={dt_s:.9f}s."
                )
        steps += 1

    if remain > max(min_h, 1e-9 * max(1.0, abs(dt_s))):
        raise RuntimeError(
            f"Adaptive integrator exhausted {max_substeps} internal substeps with {remain:.9e}s remaining."
        )
    info = AdaptiveStepInfo(
        method=method_name,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        attempted_steps=attempted_steps,
        min_step_s=min_step_s,
        max_step_s=max_step_s,
        final_step_s=final_step_s,
        suggested_next_step_s=float(h),
        max_error_ratio=max_error_ratio,
    )
    return _finish(xk, info)
