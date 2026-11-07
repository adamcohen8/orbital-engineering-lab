"""Native kernels for repeated guidance and actuator arithmetic.

The adapter is intentionally small.  It does not construct controllers or
change command policy. Rust is the primary numerical backend; callers can
select the Python reference path explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _extension() -> Any:
    try:
        import oel_rust_orbit

        return oel_rust_orbit
    except ImportError as exc:  # pragma: no cover - depends on optional wheel
        raise RuntimeError(
            "Rust control backend requested but oel_rust_orbit is unavailable; "
            "install the optional oel_rust_orbit wheel first"
        ) from exc


def _rust_function(name: str) -> Any:
    function = getattr(_extension(), name, None)
    if function is None:
        raise RuntimeError(
            f"Installed oel_rust_orbit wheel does not provide {name}; "
            "install a control/targeting-enabled wheel"
        )
    return function


def _packed(*values: np.ndarray) -> bytes:
    """Portable little-endian f64 buffers avoid per-scalar Python objects."""
    return b"".join(np.asarray(value, dtype="<f8").tobytes() for value in values)


def _packed_f64(*values: np.ndarray) -> bytes:
    # Callers have already normalized these arrays to native f64.
    if np.little_endian:
        return b"".join(value.tobytes() for value in values)
    return _packed(*values)


def _vector(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array if array.ndim == 1 else array.reshape(-1)


def _unpacked(value: bytes) -> np.ndarray:
    return np.frombuffer(value, dtype="<f8").copy()


def reaction_wheel_step(
    prepared: bytes, motor_torque: np.ndarray, omega: np.ndarray,
    command: np.ndarray, dt_s: float, torque_tau_s: float,
) -> np.ndarray | None:
    """Advance maintained motor/friction/saturation dynamics in one call."""
    function = getattr(_extension(), "control_reaction_wheel_step_packed", None)
    if function is None:
        return None
    return _unpacked(function(prepared + _packed(motor_torque, omega, command),
                              int(command.size), float(dt_s), float(torque_tau_s)))


def project_sequence(controls: np.ndarray, max_norm: float) -> np.ndarray:
    """Project each control row onto an independent Euclidean norm ball."""

    values = np.asarray(controls, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("controls must be a two-dimensional sequence")
    result = _rust_function("control_project_sequence")(
        values.reshape(-1).tolist(), int(values.shape[0]), int(values.shape[1]), float(max_norm)
    )
    return np.asarray(result, dtype=np.float64).reshape(values.shape)


def mat_vec_clip(
    matrix: np.ndarray,
    vector: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Apply a dense matrix-vector map and clip its output in native code."""

    values = np.asarray(matrix, dtype=np.float64)
    x = _vector(vector)
    lo = _vector(lower)
    hi = _vector(upper)
    if values.ndim != 2 or x.shape != (values.shape[1],) or lo.shape != (values.shape[0],) or hi.shape != (values.shape[0],):
        raise ValueError("matrix-vector dimensions are inconsistent")
    packed = getattr(_extension(), "control_mat_vec_packed", None)
    if packed is not None:
        return _unpacked(packed(_packed_f64(values, x, lo, hi), int(values.shape[0]), int(values.shape[1]), True))
    result = _rust_function("control_mat_vec_clip")(
        values.reshape(-1).tolist(),
        int(values.shape[0]),
        int(values.shape[1]),
        x.tolist(),
        lo.tolist(),
        hi.tolist(),
    )
    return np.asarray(result, dtype=np.float64)


def mat_vec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Apply a dense matrix-vector map in native code without clipping."""

    values = np.asarray(matrix, dtype=np.float64)
    x = _vector(vector)
    if values.ndim != 2 or x.shape != (values.shape[1],):
        raise ValueError("matrix-vector dimensions are inconsistent")
    packed = getattr(_extension(), "control_mat_vec_packed", None)
    if packed is not None:
        return _unpacked(packed(_packed_f64(values, x), int(values.shape[0]), int(values.shape[1]), False))
    result = _rust_function("control_mat_vec")(
        values.reshape(-1).tolist(), int(values.shape[0]), int(values.shape[1]), x.tolist()
    )
    return np.asarray(result, dtype=np.float64)


def mpc_rollout_cost_gradient(
    x0: np.ndarray,
    target: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    terminal: np.ndarray,
    r: np.ndarray,
    rd: np.ndarray,
    previous_control: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Evaluate a linear discrete MPC rollout and analytic control gradient."""

    x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    terminal_arr = np.asarray(terminal, dtype=np.float64).reshape(-1)
    r_arr = np.asarray(r, dtype=np.float64).reshape(-1)
    rd_arr = np.asarray(rd, dtype=np.float64).reshape(-1)
    previous_arr = np.asarray(previous_control, dtype=np.float64).reshape(-1)
    controls_arr = np.asarray(controls, dtype=np.float64)
    if controls_arr.ndim != 2 or a_arr.shape != (x0_arr.size, x0_arr.size) or b_arr.shape != (x0_arr.size, controls_arr.shape[1]):
        raise ValueError("MPC matrix dimensions are inconsistent")
    if target_arr.size != x0_arr.size or q_arr.size != x0_arr.size or terminal_arr.size != x0_arr.size or any(value.size != controls_arr.shape[1] for value in (r_arr, rd_arr, previous_arr)):
        raise ValueError("MPC dimensions are inconsistent")
    packed = getattr(_extension(), "control_mpc_packed", None)
    if packed is not None:
        states, cost, gradient = packed(
            _packed_f64(x0_arr, target_arr, a_arr, b_arr, q_arr, terminal_arr, r_arr,
                    rd_arr, previous_arr, controls_arr),
            int(x0_arr.size), int(controls_arr.shape[1]), True,
        )
        return (_unpacked(states).reshape(controls_arr.shape[0] + 1, x0_arr.size),
                float(cost), _unpacked(gradient).reshape(controls_arr.shape))
    states, cost, gradient = _rust_function("control_mpc_rollout_cost_gradient")(
        x0_arr.tolist(),
        target_arr.tolist(),
        a_arr.reshape(-1).tolist(),
        b_arr.reshape(-1).tolist(),
        q_arr.tolist(),
        terminal_arr.tolist(),
        r_arr.tolist(),
        rd_arr.tolist(),
        previous_arr.tolist(),
        controls_arr.reshape(-1).tolist(),
        int(x0_arr.size),
        int(controls_arr.shape[1]),
    )
    return (
        np.asarray(states, dtype=np.float64).reshape(controls_arr.shape[0] + 1, x0_arr.size),
        float(cost),
        np.asarray(gradient, dtype=np.float64).reshape(controls_arr.shape),
    )


def mpc_rollout_cost(
    x0: np.ndarray, target: np.ndarray, a: np.ndarray, b: np.ndarray,
    q: np.ndarray, terminal: np.ndarray, r: np.ndarray, rd: np.ndarray,
    previous_control: np.ndarray, controls: np.ndarray,
) -> float:
    """Evaluate the identical forward objective without an unused gradient."""
    x = np.asarray(x0, dtype=np.float64).reshape(-1)
    u = np.asarray(controls, dtype=np.float64)
    a_values = np.asarray(a, dtype=np.float64)
    b_values = np.asarray(b, dtype=np.float64)
    if u.ndim != 2 or a_values.shape != (x.size, x.size) or b_values.shape != (x.size, u.shape[1]):
        raise ValueError("MPC matrix dimensions are inconsistent")
    # Match the full adapter's shape checking rather than allowing packed
    # fields to shift when a malformed caller supplies a short weight vector.
    arrays = [x, np.asarray(target, dtype=np.float64).reshape(-1), a_values, b_values,
              np.asarray(q, dtype=np.float64).reshape(-1), np.asarray(terminal, dtype=np.float64).reshape(-1),
              np.asarray(r, dtype=np.float64).reshape(-1), np.asarray(rd, dtype=np.float64).reshape(-1),
              np.asarray(previous_control, dtype=np.float64).reshape(-1), u]
    if any(arrays[i].size != x.size for i in (1, 4, 5)) or any(arrays[i].size != u.shape[1] for i in (6, 7, 8)):
        raise ValueError("MPC dimensions are inconsistent")
    packed = getattr(_extension(), "control_mpc_packed", None)
    if packed is None:
        return float(mpc_rollout_cost_gradient(x0, target, a, b, q, terminal, r, rd, previous_control, controls)[1])
    return float(packed(_packed_f64(*arrays), int(x.size), int(u.shape[1]), False)[1])


def has_relative_mpc_cost_batch() -> bool:
    """Older native wheels retain the maintained per-step Rust MPC route."""
    return getattr(_extension(), "control_relative_mpc_cost_batch_packed", None) is not None


def relative_mpc_cost_batch(
    x_chaser0: np.ndarray, x_target0: np.ndarray, previous_control: np.ndarray,
    target_relative: np.ndarray, state_signs: np.ndarray, q: np.ndarray,
    terminal: np.ndarray, r: np.ndarray, rd: np.ndarray, controls: np.ndarray,
    *, dt_s: float, mu_km3_s2: float, max_accel_km_s2: float,
) -> np.ndarray:
    """Evaluate complete projected nonlinear relative-MPC trial costs."""
    sequences = np.asarray(controls, dtype=np.float64)
    if sequences.ndim != 3 or sequences.shape[0] == 0 or sequences.shape[1] == 0 or sequences.shape[2] != 3:
        raise ValueError("relative MPC controls must have shape (candidates, horizon, 3)")
    fields = (
        (x_chaser0, 6), (x_target0, 6), (previous_control, 3),
        (target_relative, 6), (state_signs, 6), (q, 6),
        (terminal, 6), (r, 3), (rd, 3),
    )
    prepared = []
    for value, length in fields:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != length:
            raise ValueError("relative MPC prepared input dimensions are inconsistent")
        prepared.append(array)
    output = _rust_function("control_relative_mpc_cost_batch_packed")(
        _packed_f64(*prepared, sequences),
        int(sequences.shape[0]), int(sequences.shape[1]),
        float(dt_s), float(mu_km3_s2), float(max_accel_km_s2),
    )
    return _unpacked(output)


def lambert(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    time_of_flight_s: float,
    *,
    mu_km3_s2: float,
    short_way: bool,
    max_iterations: int,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    """Solve one native universal-variable Lambert transfer."""

    r1 = np.asarray(r1_km, dtype=np.float64)
    r2 = np.asarray(r2_km, dtype=np.float64)
    if r1.shape != (3,):
        r1 = r1.reshape(3)
    if r2.shape != (3,):
        r2 = r2.reshape(3)
    packed = getattr(_extension(), "targeting_lambert_packed", None)
    if packed is not None:
        result, residual, iterations, converged = packed(
            r1.tobytes() + r2.tobytes() if np.little_endian else _packed(r1, r2),
            float(time_of_flight_s), float(mu_km3_s2), bool(short_way), int(max_iterations), float(tolerance_s),
        )
        values = _unpacked(result)
        return values[:3], values[3:], float(residual), int(iterations), bool(converged)
    v1, v2, residual, iterations, converged = _rust_function("targeting_lambert")(
        r1.tolist(), r2.tolist(), float(time_of_flight_s), float(mu_km3_s2), bool(short_way), int(max_iterations), float(tolerance_s)
    )
    return np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64), float(residual), int(iterations), bool(converged)


def lambert_batch(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    time_of_flight_s: np.ndarray,
    *,
    mu_km3_s2: float,
    short_way: bool,
    max_iterations: int,
    tolerance_s: float,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Solve a batch of Lambert candidates in one Python/Rust crossing."""

    first = np.asarray(r1_km, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(r2_km, dtype=np.float64).reshape(-1, 3)
    times = np.asarray(time_of_flight_s, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or first.shape[0] != times.size:
        raise ValueError("Lambert batch arrays have inconsistent shapes")
    packed = getattr(_extension(), "targeting_lambert_batch_packed", None)
    if packed is not None:
        output, errors = packed(_packed_f64(first, second, times), float(mu_km3_s2), bool(short_way), int(max_iterations), float(tolerance_s))
        return _unpacked(output).reshape(-1, 8), [(int(index), str(message)) for index, message in errors]
    output, errors = _rust_function("targeting_lambert_batch")(
        first.reshape(-1).tolist(), second.reshape(-1).tolist(), times.tolist(), float(mu_km3_s2), bool(short_way), int(max_iterations), float(tolerance_s)
    )
    return np.asarray(output, dtype=np.float64).reshape(-1, 8), [(int(index), str(message)) for index, message in errors]


def scaled_residual_norms(
    observed: np.ndarray,
    targets: np.ndarray,
    tolerances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-candidate scaled residual L2 norms and maximum errors."""

    values = np.asarray(observed, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("observed candidates must be two-dimensional")
    goal = np.asarray(targets, dtype=np.float64).reshape(-1)
    tol = np.asarray(tolerances, dtype=np.float64).reshape(-1)
    if values.shape[1] != goal.size or tol.size != goal.size:
        raise ValueError("target residual dimensions are inconsistent")
    packed = getattr(_extension(), "targeting_scaled_residual_norms_packed", None)
    if packed is not None:
        norms, maxima = packed(_packed_f64(values, goal, tol), int(values.shape[0]), int(values.shape[1]))
        return _unpacked(norms), _unpacked(maxima)
    norms, maxima = _rust_function("targeting_scaled_residual_norms")(
        values.reshape(-1).tolist(), goal.tolist(), tol.tolist(), int(values.shape[0]), int(values.shape[1])
    )
    return np.asarray(norms, dtype=np.float64), np.asarray(maxima, dtype=np.float64)


def finite_difference_jacobian(
    left: np.ndarray,
    right: np.ndarray,
    variable_steps: np.ndarray,
    goal_tolerances: np.ndarray,
) -> np.ndarray:
    """Build a scaled target Jacobian from already evaluated perturbations."""

    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.ndim != 2 or right_arr.shape != left_arr.shape:
        raise ValueError("finite-difference observations must have matching two-dimensional shapes")
    steps = np.asarray(variable_steps, dtype=np.float64).reshape(-1)
    tolerances = np.asarray(goal_tolerances, dtype=np.float64).reshape(-1)
    if left_arr.shape[0] != steps.size or left_arr.shape[1] != tolerances.size:
        raise ValueError("finite-difference dimensions are inconsistent")
    result = _rust_function("targeting_finite_difference_jacobian")(
        left_arr.reshape(-1).tolist(), right_arr.reshape(-1).tolist(), steps.tolist(), tolerances.tolist(), int(steps.size), int(tolerances.size)
    )
    return np.asarray(result, dtype=np.float64).reshape(tolerances.size, steps.size)


def prepare_rcs_geometry(force_matrix: np.ndarray, torque_matrix: np.ndarray) -> Any:
    """Copy content-bound matrices once; older wheels retain scalar matvecs."""
    force = np.asarray(force_matrix, dtype=np.float64)
    torque = np.asarray(torque_matrix, dtype=np.float64)
    if (
        force.ndim != 2
        or force.shape[0] != 3
        or force.shape[1] == 0
        or torque.shape != force.shape
        or not np.all(np.isfinite(force))
        or not np.all(np.isfinite(torque))
    ):
        raise ValueError("RCS force and torque matrices must be finite with matching shape (3, 1..4096)")
    cls = getattr(_extension(), "RCSGeometryContext", None)
    if cls is None or force.shape[1] > 4096:
        return None
    return cls(_packed(force), _packed(torque), int(force.shape[1]))


def rcs_achieved(context: Any, forces: np.ndarray, body_to_eci: np.ndarray) -> np.ndarray:
    return _unpacked(context.achieved(_packed(forces), _packed(body_to_eci))).reshape(3, 3)
