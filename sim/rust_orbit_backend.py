"""Optional, explicitly selected binding to the parallel Rust ONP core."""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES, ExtensionFileLoader
from pathlib import Path
from threading import RLock
from types import ModuleType

import numpy as np

_NATIVE_MODULE = None
_NATIVE_IMPORTER = None
_NATIVE_MODULE_LOCK = RLock()


def _extension():
    global _NATIVE_MODULE, _NATIVE_IMPORTER
    module = _NATIVE_MODULE
    if module is None or _NATIVE_IMPORTER is not import_module:
        with _NATIVE_MODULE_LOCK:
            module = _NATIVE_MODULE
            if module is None or _NATIVE_IMPORTER is not import_module:
                try:
                    module = import_module("oel_rust_orbit")
                except ImportError as exc:
                    raise RuntimeError(
                        "Rust orbit backend requested but oel_rust_orbit is unavailable; "
                        "install the optional oel_rust_orbit wheel first"
                    ) from exc
                _NATIVE_MODULE, _NATIVE_IMPORTER = module, import_module
    return module


def native_extension_path() -> Path:
    """Return the verified file for the loaded ``oel_rust_orbit`` binary.

    Wheels may expose the extension as a flat module or through the package's
    ``oel_rust_orbit`` child module. Resolve only those already-loaded module
    aliases; never search the filesystem for a similarly named binary.
    """

    def fail(reason: str) -> RuntimeError:
        return RuntimeError(f"Unable to identify the loaded oel_rust_orbit extension: {reason}.")

    package = _extension()
    if not isinstance(package, ModuleType) or package.__name__ != "oel_rust_orbit":
        raise fail("the imported package alias is not the expected module")
    if sys.modules.get("oel_rust_orbit") is not package:
        raise fail("the imported package alias is not registered in sys.modules")

    package_file = getattr(package, "__file__", None)
    if isinstance(package_file, str) and any(package_file.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        native = package
        expected_name = "oel_rust_orbit"
    else:
        expected_name = "oel_rust_orbit.oel_rust_orbit"
        native = getattr(package, "oel_rust_orbit", None)
        if not isinstance(native, ModuleType) or native.__name__ != expected_name:
            raise fail("the package does not expose its compiled child module")
        if getattr(package, "oel_rust_orbit", None) is not native:
            raise fail("the package child alias does not identify the loaded module")

    if sys.modules.get(expected_name) is not native:
        raise fail("the native module alias is not registered in sys.modules")
    spec = getattr(native, "__spec__", None)
    if (
        spec is None
        or spec.name != expected_name
        or not isinstance(spec.loader, ExtensionFileLoader)
        or not isinstance(spec.origin, str)
    ):
        raise fail("the loaded module does not have a native extension spec")

    module_file = getattr(native, "__file__", None)
    if not isinstance(module_file, str) or not Path(module_file).is_absolute():
        raise fail("the loaded module has no absolute file path")
    module_path = Path(module_file)
    origin_path = Path(spec.origin)
    if not origin_path.is_absolute():
        raise fail("the native extension spec has no absolute origin")
    if not any(module_path.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise fail("the loaded module path has no recognized extension suffix")
    if not any(origin_path.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise fail("the native extension origin has no recognized extension suffix")

    try:
        resolved_module_path = module_path.resolve(strict=True)
        resolved_origin_path = origin_path.resolve(strict=True)
        if resolved_module_path != resolved_origin_path or not resolved_module_path.is_file():
            raise fail("the module file and native extension origin do not identify one regular file")
        if resolved_module_path.stat().st_size <= 0:
            raise fail("the loaded native extension file is empty")
    except OSError as exc:
        raise fail("the loaded native extension file is unavailable") from exc

    identity_export = getattr(native, "rk4_step_eci", None)
    if not callable(identity_export) or getattr(package, "rk4_step_eci", None) is not identity_export:
        raise fail("the package and loaded native module do not share the identity export")
    return resolved_module_path


def _state(value: np.ndarray) -> list[float]:
    state = np.asarray(value, dtype=np.float64)
    if state.shape != (6,):
        raise ValueError("ECI state must have shape (6,)")
    return state.tolist()


def _command(value: np.ndarray) -> list[float]:
    command = np.asarray(value, dtype=np.float64)
    if command.shape != (3,):
        raise ValueError("command acceleration must have shape (3,)")
    return command.tolist()


def native_harmonic_degree_limit() -> int:
    """Older compatible wheels retain their degree-64 native envelope."""
    return int(getattr(_extension(), "MAX_HARMONIC_DEGREE", 64))


def rk4_step_eci(
    state: np.ndarray,
    dt_s: float,
    mu_km3_s2: float,
    *,
    include_j2: bool,
    command_accel_eci_km_s2: np.ndarray,
) -> np.ndarray:
    """Propagate one ECI RK4 step with two-body gravity and optional Earth J2."""

    result = _extension().rk4_step_eci(
        _state(state),
        float(dt_s),
        float(mu_km3_s2),
        bool(include_j2),
        _command(command_accel_eci_km_s2),
    )
    return np.asarray(result, dtype=np.float64)


def propagate_history_eci(
    initial_state: np.ndarray,
    dt_s: float,
    steps: int,
    mu_km3_s2: float,
    *,
    include_j2: bool,
    command_accel_eci_km_s2: np.ndarray,
) -> np.ndarray:
    """Return initial state plus every step in one Rust call, shape (steps + 1, 6)."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    native = _extension()
    packed = getattr(native, "propagate_history_eci_bytes", None)
    if packed is not None:
        values = packed(
            _state(initial_state), float(dt_s), steps, float(mu_km3_s2),
            bool(include_j2), _command(command_accel_eci_km_s2),
        )
        return np.frombuffer(values, dtype="<f8").copy().reshape(steps + 1, 6)
    flat = native.propagate_history_eci(
        _state(initial_state),
        float(dt_s),
        steps,
        float(mu_km3_s2),
        bool(include_j2),
        _command(command_accel_eci_km_s2),
    )
    return np.asarray(flat, dtype=np.float64).reshape(steps + 1, 6)


def propagate_passive_segment_eci(
    initial_state: np.ndarray,
    dt_s: float,
    steps: int,
    mu_km3_s2: float,
    *,
    include_j2: bool,
    command_accel_eci_km_s2: np.ndarray | None = None,
    boundary_steps: tuple[int, ...] = (),
    segment_commands_eci_km_s2: tuple[np.ndarray, ...] | None = None,
) -> np.ndarray:
    """Propagate a fixed-step passive segment without crossing caller boundaries.

    The native history routine is called once per interval between the supplied
    ``boundary_steps``.  This keeps hard event and control boundaries visible
    to the caller while retaining one native history call for each uninterrupted
    coast.  ``segment_commands_eci_km_s2`` can supply the command active after
    each boundary; a single ``command_accel_eci_km_s2`` is reused when it is not
    supplied.  The returned rows always include the initial state and every
    fixed-step boundary, with no samples skipped.

    This helper is intentionally limited to fixed-step ECI two-body/J2
    propagation.  Event detection, controller decisions, frame conversion, and
    output writing remain at the Python caller boundary.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if not np.isfinite(float(dt_s)) or float(dt_s) == 0.0:
        raise ValueError("dt_s must be finite and nonzero")
    initial = np.asarray(initial_state, dtype=np.float64)
    if initial.shape != (6,):
        raise ValueError("ECI state must have shape (6,)")
    if command_accel_eci_km_s2 is None:
        default_command = np.zeros(3, dtype=np.float64)
    else:
        default_command = np.asarray(command_accel_eci_km_s2, dtype=np.float64)
        if default_command.shape != (3,):
            raise ValueError("command acceleration must have shape (3,)")

    raw_boundaries = tuple(boundary_steps)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_boundaries):
        raise ValueError("boundary_steps must contain integers")
    if any(value < 0 or value > steps for value in raw_boundaries):
        raise ValueError("boundary_steps must lie within the propagation interval")
    boundaries = [0, *sorted({value for value in raw_boundaries if 0 < value < steps}), steps]
    interval_count = len(boundaries) - 1

    if segment_commands_eci_km_s2 is None:
        commands = [default_command] * interval_count
    else:
        if len(segment_commands_eci_km_s2) != interval_count:
            raise ValueError("segment_commands must contain one command per uninterrupted interval")
        commands = []
        for command in segment_commands_eci_km_s2:
            values = np.asarray(command, dtype=np.float64)
            if values.shape != (3,):
                raise ValueError("each segment command must have shape (3,)")
            commands.append(values)

    rows = [initial.copy()]
    current = initial.copy()
    for interval, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if stop == start:
            continue
        segment = propagate_history_eci(
            current,
            dt_s,
            stop - start,
            mu_km3_s2,
            include_j2=include_j2,
            command_accel_eci_km_s2=commands[interval],
        )
        rows.extend(segment[1:])
        current = segment[-1].copy()
    return np.asarray(rows, dtype=np.float64)


def rk4_zonal_step_eci(
    state: np.ndarray,
    dt_s: float,
    mu_km3_s2: float,
    *,
    force_codes: list[int],
    command_accel_eci_km_s2: np.ndarray,
) -> np.ndarray:
    """Advance one RK4 step with ordered J2/J3/J4 forces."""

    result = _extension().rk4_zonal_step_eci(
        _state(state),
        float(dt_s),
        float(mu_km3_s2),
        force_codes,
        _command(command_accel_eci_km_s2),
    )
    return np.asarray(result, dtype=np.float64)


def rkf78_zonal_step_eci(
    state: np.ndarray,
    t_s: float,
    dt_s: float,
    mu_km3_s2: float,
    *,
    force_codes: list[int],
    command_accel_eci_km_s2: np.ndarray,
    atol: float,
    rtol: float,
    h_init: float | None,
):
    """Advance one adaptive interval and return OEL-compatible step evidence."""

    from sim.dynamics.orbit.integrators import AdaptiveStepInfo

    state_out, raw = _extension().rkf78_zonal_step_eci(
        _state(state),
        float(t_s),
        float(dt_s),
        float(mu_km3_s2),
        force_codes,
        _command(command_accel_eci_km_s2),
        float(atol),
        float(rtol),
        None if h_init is None else float(h_init),
    )
    info = AdaptiveStepInfo(
        method="rkf78",
        accepted_steps=raw[0],
        rejected_steps=raw[1],
        attempted_steps=raw[2],
        min_step_s=raw[3],
        max_step_s=raw[4],
        final_step_s=raw[5],
        suggested_next_step_s=raw[6],
        max_error_ratio=raw[7],
    )
    return np.asarray(state_out, dtype=np.float64), info


def rkf78_force_plan_eci(
    state: np.ndarray, t_s: float, dt_s: float, *, codes: list[int],
    scalars: list[float], shadow_model: int, harmonic_dims: tuple[int, int],
    tables: list[list[float]], command_accel_eci_km_s2: np.ndarray,
    atol: float, rtol: float, h_init: float | None, stage_callback,
    force_context=None,
):
    """Advance an ordered, staged built-in force plan with Rust RKF78."""

    return _adaptive_force_plan_eci(
        "rkf78", state, t_s, dt_s, codes=codes, scalars=scalars,
        shadow_model=shadow_model, harmonic_dims=harmonic_dims, tables=tables,
        command_accel_eci_km_s2=command_accel_eci_km_s2, atol=atol,
        rtol=rtol, h_init=h_init, stage_callback=stage_callback,
        force_context=force_context,
    )


def dopri5_force_plan_eci(
    state: np.ndarray, t_s: float, dt_s: float, *, codes: list[int],
    scalars: list[float], shadow_model: int, harmonic_dims: tuple[int, int],
    tables: list[list[float]], command_accel_eci_km_s2: np.ndarray,
    atol: float, rtol: float, h_init: float | None, stage_callback,
    force_context=None,
):
    """Advance an ordered, staged built-in force plan with Rust DOPRI5."""

    return _adaptive_force_plan_eci(
        "dopri5", state, t_s, dt_s, codes=codes, scalars=scalars,
        shadow_model=shadow_model, harmonic_dims=harmonic_dims, tables=tables,
        command_accel_eci_km_s2=command_accel_eci_km_s2, atol=atol,
        rtol=rtol, h_init=h_init, stage_callback=stage_callback,
        force_context=force_context,
    )


def _adaptive_force_plan_eci(
    method: str, state: np.ndarray, t_s: float, dt_s: float, *, codes: list[int],
    scalars: list[float], shadow_model: int, harmonic_dims: tuple[int, int],
    tables: list[list[float]], command_accel_eci_km_s2: np.ndarray,
    atol: float, rtol: float, h_init: float | None, stage_callback,
    force_context=None,
):
    """Shared Python binding for the Rust adaptive built-in force plans."""

    from sim.dynamics.orbit.integrators import AdaptiveStepInfo

    if force_context is None:
        legacy_tables = [np.asarray(table, dtype=np.float64).reshape(-1).tolist() for table in tables]
        state_out, raw = getattr(_extension(), f"{method}_force_plan_eci")(
            _state(state), float(t_s), float(dt_s), codes, scalars, shadow_model,
            harmonic_dims, legacy_tables, _command(command_accel_eci_km_s2),
            float(atol), float(rtol), None if h_init is None else float(h_init),
            stage_callback,
        )
    else:
        state_out, raw = force_context.adaptive(
            _state(state), float(t_s), float(dt_s), _command(command_accel_eci_km_s2),
            float(atol), float(rtol), None if h_init is None else float(h_init),
            method, stage_callback,
        )
    info = AdaptiveStepInfo(
        method=method, accepted_steps=raw[0], rejected_steps=raw[1],
        attempted_steps=raw[2], min_step_s=raw[3], max_step_s=raw[4],
        final_step_s=raw[5], suggested_next_step_s=raw[6], max_error_ratio=raw[7],
    )
    return np.asarray(state_out, dtype=np.float64), info


def rk4_force_plan_eci(
    state: np.ndarray, t_s: float, dt_s: float, *, codes: list[int],
    scalars: list[float], shadow_model: int, harmonic_dims: tuple[int, int],
    tables: list[list[float]], command_accel_eci_km_s2: np.ndarray, stage_callback,
    force_context=None,
) -> np.ndarray:
    """Advance an ordered, staged built-in force plan with Rust RK4."""

    if force_context is None:
        legacy_tables = [np.asarray(table, dtype=np.float64).reshape(-1).tolist() for table in tables]
        result = _extension().rk4_force_plan_eci(
            _state(state), float(t_s), float(dt_s), codes, scalars, shadow_model,
            harmonic_dims, legacy_tables, _command(command_accel_eci_km_s2), stage_callback,
        )
    else:
        result = force_context.rk4(
            _state(state), float(t_s), float(dt_s), _command(command_accel_eci_km_s2), stage_callback,
        )
    return np.asarray(result, dtype=np.float64)


def rk4_callback_eci(
    state: np.ndarray,
    t_s: float,
    dt_s: float,
    acceleration_callback,
) -> np.ndarray:
    """Advance RK4 in Rust with an authoritative Python acceleration callback."""

    result = _extension().rk4_callback_eci(
        _state(state), float(t_s), float(dt_s), acceleration_callback,
    )
    return np.asarray(result, dtype=np.float64)


def adaptive_callback_eci(
    state: np.ndarray,
    t_s: float,
    dt_s: float,
    *,
    atol: float,
    rtol: float,
    h_init: float | None,
    method: str,
    acceleration_callback,
):
    """Advance RKF78 or DOPRI5 in Rust with Python stage accelerations."""

    from sim.dynamics.orbit.integrators import AdaptiveStepInfo

    state_out, raw = _extension().adaptive_callback_eci(
        _state(state), float(t_s), float(dt_s), float(atol), float(rtol),
        None if h_init is None else float(h_init), method, acceleration_callback,
    )
    info = AdaptiveStepInfo(
        method=method,
        accepted_steps=raw[0], rejected_steps=raw[1], attempted_steps=raw[2],
        min_step_s=raw[3], max_step_s=raw[4], final_step_s=raw[5],
        suggested_next_step_s=raw[6], max_error_ratio=raw[7],
    )
    return np.asarray(state_out, dtype=np.float64), info


def rk4_pair_callback_eci(states: np.ndarray, t_s: float, dt_s: float, derivative_callback) -> np.ndarray:
    """Advance two synchronized six-state objects through Rust RK4 stages."""

    values = np.asarray(states, dtype=np.float64)
    if values.shape != (12,):
        raise ValueError("pair states must have shape (12,)")
    result = _extension().rk4_pair_callback_eci(
        values.tolist(), float(t_s), float(dt_s), derivative_callback,
    )
    return np.asarray(result, dtype=np.float64)


def rk4_coupled_callback(state: np.ndarray, quaternion: np.ndarray, t_s: float, dt_s: float, stage_callback):
    """Advance coupled orbit and attitude stage arithmetic in Rust."""

    values = np.asarray(state, dtype=np.float64).reshape(-1)
    attitude = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if values.size < 10 or attitude.size != 4:
        raise ValueError("coupled state or attitude has invalid dimensions")
    next_state, next_quaternion = _extension().rk4_coupled_callback(
        values.tolist(), attitude.tolist(), float(t_s), float(dt_s), stage_callback,
    )
    return np.asarray(next_state, dtype=np.float64), np.asarray(next_quaternion, dtype=np.float64)
