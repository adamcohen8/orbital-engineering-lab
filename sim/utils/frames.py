from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock

import numpy as np

from sim.acceleration.settings import acceleration_cache_key, acceleration_settings_from_mode
from sim.numeric_backend import normalize_numeric_backend

# These names remain as narrowly scoped compatibility seams for existing tests
# and downstream monkeypatches.  Normal dispatch never writes them: selecting
# acceleration is per execution context below, rather than process-global.
eci_relative_to_ric_rect_kernel = None
ric_angular_rate_eci_from_rv_kernel = None
ric_angular_rate_eci_from_rva_kernel = None
ric_curv_to_rect_kernel = None
ric_dcm_ir_from_rv_kernel = None
ric_rect_state_to_eci_kernel = None
ric_rect_state_to_eci_rva_kernel = None
ric_rect_to_curv_kernel = None
eci_relative_to_ric_rect_rva_kernel = None


@dataclass(frozen=True)
class _FrameKernelBundle:
    """Immutable frame-kernel selection for one acceleration context."""

    cache_key: tuple[str, bool]
    enabled: bool
    eci_relative_to_ric_rect_kernel: Callable[..., np.ndarray] | None = None
    ric_angular_rate_eci_from_rv_kernel: Callable[..., np.ndarray] | None = None
    ric_angular_rate_eci_from_rva_kernel: Callable[..., np.ndarray] | None = None
    ric_curv_to_rect_kernel: Callable[..., np.ndarray] | None = None
    ric_dcm_ir_from_rv_kernel: Callable[..., np.ndarray] | None = None
    ric_rect_state_to_eci_kernel: Callable[..., np.ndarray] | None = None
    ric_rect_state_to_eci_rva_kernel: Callable[..., np.ndarray] | None = None
    ric_rect_to_curv_kernel: Callable[..., np.ndarray] | None = None
    eci_relative_to_ric_rect_rva_kernel: Callable[..., np.ndarray] | None = None


_FRAME_KERNEL_BUNDLES: dict[tuple[str, bool], _FrameKernelBundle] = {}
_FRAME_KERNEL_CONTEXT: ContextVar[_FrameKernelBundle | None] = ContextVar(
    "oel_frame_kernel_bundle", default=None
)
_FRAME_KERNEL_LOCK = RLock()


def _load_frame_kernel_bundle(cache_key: tuple[str, bool], enabled: bool) -> _FrameKernelBundle:
    """Return a process-cached immutable bundle, loading kernels once."""

    key = (str(cache_key[0]), bool(cache_key[1]))
    with _FRAME_KERNEL_LOCK:
        cached = _FRAME_KERNEL_BUNDLES.get(key)
        if cached is not None:
            return cached
        if not enabled:
            bundle = _FrameKernelBundle(cache_key=key, enabled=False)
        else:
            from sim.acceleration.kernels.frames import (
                eci_relative_to_ric_rect_kernel,
                eci_relative_to_ric_rect_rva_kernel,
                ric_angular_rate_eci_from_rv_kernel,
                ric_angular_rate_eci_from_rva_kernel,
                ric_curv_to_rect_kernel,
                ric_dcm_ir_from_rv_kernel,
                ric_rect_state_to_eci_kernel,
                ric_rect_state_to_eci_rva_kernel,
                ric_rect_to_curv_kernel,
            )

            bundle = _FrameKernelBundle(
                cache_key=key,
                enabled=True,
                eci_relative_to_ric_rect_kernel=eci_relative_to_ric_rect_kernel,
                ric_angular_rate_eci_from_rv_kernel=ric_angular_rate_eci_from_rv_kernel,
                ric_angular_rate_eci_from_rva_kernel=ric_angular_rate_eci_from_rva_kernel,
                ric_curv_to_rect_kernel=ric_curv_to_rect_kernel,
                ric_dcm_ir_from_rv_kernel=ric_dcm_ir_from_rv_kernel,
                ric_rect_state_to_eci_kernel=ric_rect_state_to_eci_kernel,
                ric_rect_state_to_eci_rva_kernel=ric_rect_state_to_eci_rva_kernel,
                ric_rect_to_curv_kernel=ric_rect_to_curv_kernel,
                eci_relative_to_ric_rect_rva_kernel=eci_relative_to_ric_rect_rva_kernel,
            )
        _FRAME_KERNEL_BUNDLES[key] = bundle
        return bundle


def _frame_kernel_bundle() -> _FrameKernelBundle:
    cache_key = acceleration_cache_key()
    current = _FRAME_KERNEL_CONTEXT.get()
    if current is not None and current.cache_key == cache_key:
        return current
    bundle = _load_frame_kernel_bundle(cache_key, bool(acceleration_settings_from_mode().enabled))
    _FRAME_KERNEL_CONTEXT.set(bundle)
    return bundle


def _frame_kernel(bundle: _FrameKernelBundle, name: str) -> Callable[..., np.ndarray]:
    """Resolve a bundle member, honoring legacy test monkeypatch seams."""

    override = globals().get(name)
    if override is not None:
        return override
    kernel = getattr(bundle, name)
    if kernel is None:
        raise RuntimeError(f"Frame kernel {name} is unavailable in an enabled context.")
    return kernel


def _frame_acceleration_enabled() -> bool:
    return _frame_kernel_bundle().enabled


def ric_dcm_ir_from_rv(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
    if _frame_acceleration_enabled():
        bundle = _frame_kernel_bundle()
        return _frame_kernel(bundle, "ric_dcm_ir_from_rv_kernel")(
            np.asarray(r_eci_km, dtype=float).reshape(3),
            np.asarray(v_eci_km_s, dtype=float).reshape(3),
        )
    r = np.asarray(r_eci_km, dtype=float).reshape(3)
    v = np.asarray(v_eci_km_s, dtype=float).reshape(3)
    r_norm = float(np.sqrt(np.dot(r, r)))
    if not np.isfinite(r_norm) or r_norm <= 1e-12:
        raise ValueError("RIC frame is undefined for a zero or non-finite position vector.")
    r_hat = r / r_norm
    h = np.array(
        [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ],
        dtype=float,
    )
    h_norm = float(np.sqrt(np.dot(h, h)))
    if not np.isfinite(h_norm) or h_norm <= 1e-12:
        raise ValueError("RIC frame is undefined for zero angular momentum.")
    c_hat = h / h_norm
    i_hat = np.array(
        [
            c_hat[1] * r_hat[2] - c_hat[2] * r_hat[1],
            c_hat[2] * r_hat[0] - c_hat[0] * r_hat[2],
            c_hat[0] * r_hat[1] - c_hat[1] * r_hat[0],
        ],
        dtype=float,
    )
    i_norm = float(np.sqrt(np.dot(i_hat, i_hat)))
    if not np.isfinite(i_norm) or i_norm <= 1e-12:
        raise ValueError("RIC frame is undefined for a degenerate basis.")
    i_hat = i_hat / i_norm
    return np.column_stack((r_hat, i_hat, c_hat))


def _ric_angular_rate_eci_from_rva_python(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    a_eci_km_s2: np.ndarray,
) -> np.ndarray:
    r = np.asarray(r_eci_km, dtype=float).reshape(3)
    v = np.asarray(v_eci_km_s, dtype=float).reshape(3)
    acceleration = np.asarray(a_eci_km_s2, dtype=float).reshape(3)
    r_norm = float(np.linalg.norm(r))
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if not np.isfinite(r_norm) or r_norm <= 1e-12:
        raise ValueError("RIC frame is undefined for a zero or non-finite position vector.")
    if not np.isfinite(h_norm) or h_norm <= 1e-12:
        raise ValueError("RIC frame is undefined for zero angular momentum.")
    if not np.all(np.isfinite(acceleration)):
        raise ValueError("RIC angular rate requires a finite acceleration vector.")
    transverse_speed = h_norm / r_norm
    if not np.isfinite(transverse_speed) or transverse_speed <= 1e-12:
        raise ValueError("RIC angular rate is undefined for negligible transverse speed.")
    r_hat = r / r_norm
    c_hat = h / h_norm
    return h / (r_norm * r_norm) + (float(acceleration @ c_hat) / transverse_speed) * r_hat


def ric_angular_rate_eci_from_rv(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    *,
    chief_accel_eci_km_s2: np.ndarray | None = None,
) -> np.ndarray:
    """Return RIC angular rate, optionally including cross-track acceleration.

    The historical r/v-only call retains its fixed-plane behavior.  Supplying
    the chief acceleration adds the instantaneous radial basis rate required
    when the orbit-plane normal moves.
    """

    if chief_accel_eci_km_s2 is not None:
        r = np.asarray(r_eci_km, dtype=float).reshape(3)
        v = np.asarray(v_eci_km_s, dtype=float).reshape(3)
        acceleration = np.asarray(chief_accel_eci_km_s2, dtype=float).reshape(3)
        if _frame_acceleration_enabled():
            bundle = _frame_kernel_bundle()
            return _frame_kernel(bundle, "ric_angular_rate_eci_from_rva_kernel")(r, v, acceleration)
        return _ric_angular_rate_eci_from_rva_python(r, v, acceleration)
    if _frame_acceleration_enabled():
        bundle = _frame_kernel_bundle()
        return _frame_kernel(bundle, "ric_angular_rate_eci_from_rv_kernel")(
            np.asarray(r_eci_km, dtype=float).reshape(3),
            np.asarray(v_eci_km_s, dtype=float).reshape(3),
        )
    r = np.asarray(r_eci_km, dtype=float).reshape(3)
    v = np.asarray(v_eci_km_s, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    if r2 <= 1e-12:
        return np.zeros(3, dtype=float)
    return (
        np.array(
            [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ],
            dtype=float,
        )
        / r2
    )


def ric_rect_state_to_eci(
    x_rel_ric_rect: np.ndarray,
    r_chief_eci_km: np.ndarray,
    v_chief_eci_km_s: np.ndarray,
    *,
    chief_accel_eci_km_s2: np.ndarray | None = None,
    numeric_backend: str = "rust",
) -> np.ndarray:
    """Convert a rectangular RIC relative state to ECI.

    ``chief_accel_eci_km_s2`` is optional for compatibility.  Without it the
    existing fixed-plane r/v-only rate is used; with it the full instantaneous
    RIC basis rate is applied.
    """

    backend = normalize_numeric_backend(numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")
    if backend == "rust" and chief_accel_eci_km_s2 is None:
        from sim.rust_relative_backend import ric_relative_to_eci_batch

        return ric_relative_to_eci_batch(
            np.asarray(x_rel_ric_rect, dtype=float).reshape(1, 6),
            np.hstack((np.asarray(r_chief_eci_km, dtype=float).reshape(3),
                       np.asarray(v_chief_eci_km_s, dtype=float).reshape(3))).reshape(1, 6),
        )[0]
    if chief_accel_eci_km_s2 is not None:
        x_rel = np.asarray(x_rel_ric_rect, dtype=float).reshape(6)
        r = np.asarray(r_chief_eci_km, dtype=float).reshape(3)
        v = np.asarray(v_chief_eci_km_s, dtype=float).reshape(3)
        acceleration = np.asarray(chief_accel_eci_km_s2, dtype=float).reshape(3)
        if _frame_acceleration_enabled():
            bundle = _frame_kernel_bundle()
            return _frame_kernel(bundle, "ric_rect_state_to_eci_rva_kernel")(x_rel, r, v, acceleration)
        c_ir = ric_dcm_ir_from_rv(r, v)
        omega_ric_eci = _ric_angular_rate_eci_from_rva_python(r, v, acceleration)
        dr_eci = c_ir @ x_rel[:3]
        omega_cross_dr = np.cross(omega_ric_eci, dr_eci)
        dv_eci = c_ir @ x_rel[3:] + omega_cross_dr
        return np.hstack((r + dr_eci, v + dv_eci))
    if _frame_acceleration_enabled():
        bundle = _frame_kernel_bundle()
        return _frame_kernel(bundle, "ric_rect_state_to_eci_kernel")(
            np.asarray(x_rel_ric_rect, dtype=float).reshape(6),
            np.asarray(r_chief_eci_km, dtype=float).reshape(3),
            np.asarray(v_chief_eci_km_s, dtype=float).reshape(3),
        )
    x_rel = np.array(x_rel_ric_rect, dtype=float).reshape(6)
    c_ir = ric_dcm_ir_from_rv(r_chief_eci_km, v_chief_eci_km_s)
    omega_ric_eci = ric_angular_rate_eci_from_rv(r_chief_eci_km, v_chief_eci_km_s)
    dr_eci = c_ir @ x_rel[:3]
    omega_cross_dr = np.array(
        [
            omega_ric_eci[1] * dr_eci[2] - omega_ric_eci[2] * dr_eci[1],
            omega_ric_eci[2] * dr_eci[0] - omega_ric_eci[0] * dr_eci[2],
            omega_ric_eci[0] * dr_eci[1] - omega_ric_eci[1] * dr_eci[0],
        ],
        dtype=float,
    )
    dv_eci = c_ir @ x_rel[3:] + omega_cross_dr
    return np.hstack(
        (
            np.array(r_chief_eci_km, dtype=float).reshape(3) + dr_eci,
            np.array(v_chief_eci_km_s, dtype=float).reshape(3) + dv_eci,
        )
    )


def eci_relative_to_ric_rect(
    x_dep_eci: np.ndarray,
    x_chief_eci: np.ndarray,
    *,
    chief_accel_eci_km_s2: np.ndarray | None = None,
    numeric_backend: str = "rust",
) -> np.ndarray:
    """Convert an ECI deputy/chief pair to rectangular RIC coordinates.

    ``chief_accel_eci_km_s2`` is optional for compatibility.  Without it the
    existing fixed-plane r/v-only rate is used; with it the full instantaneous
    RIC basis rate is applied.
    """

    backend = normalize_numeric_backend(numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")
    if backend == "rust" and chief_accel_eci_km_s2 is None:
        from sim.rust_relative_backend import eci_relative_to_ric_batch

        return eci_relative_to_ric_batch(
            np.asarray(x_dep_eci, dtype=float).reshape(1, 6),
            np.asarray(x_chief_eci, dtype=float).reshape(1, 6),
        )[0]
    if chief_accel_eci_km_s2 is not None:
        deputy = np.asarray(x_dep_eci, dtype=float).reshape(6)
        chief = np.asarray(x_chief_eci, dtype=float).reshape(6)
        acceleration = np.asarray(chief_accel_eci_km_s2, dtype=float).reshape(3)
        if _frame_acceleration_enabled():
            bundle = _frame_kernel_bundle()
            return _frame_kernel(bundle, "eci_relative_to_ric_rect_rva_kernel")(deputy, chief, acceleration)
        r_chief = chief[:3]
        v_chief = chief[3:]
        c_ir = ric_dcm_ir_from_rv(r_chief, v_chief)
        omega_ric_eci = _ric_angular_rate_eci_from_rva_python(r_chief, v_chief, acceleration)
        dr_eci = deputy[:3] - r_chief
        dv_eci = deputy[3:] - v_chief
        dr_ric = c_ir.T @ dr_eci
        dv_ric = c_ir.T @ (dv_eci - np.cross(omega_ric_eci, dr_eci))
        return np.hstack((dr_ric, dv_ric))
    if _frame_acceleration_enabled():
        bundle = _frame_kernel_bundle()
        return _frame_kernel(bundle, "eci_relative_to_ric_rect_kernel")(
            np.asarray(x_dep_eci, dtype=float).reshape(6),
            np.asarray(x_chief_eci, dtype=float).reshape(6),
        )
    x_dep = np.array(x_dep_eci, dtype=float).reshape(6)
    x_chief = np.array(x_chief_eci, dtype=float).reshape(6)
    r_chief = x_chief[:3]
    v_chief = x_chief[3:]
    c_ir = ric_dcm_ir_from_rv(r_chief, v_chief)
    omega_ric_eci = ric_angular_rate_eci_from_rv(r_chief, v_chief)
    dr_eci = x_dep[:3] - r_chief
    dv_eci = x_dep[3:] - v_chief
    dr_ric = c_ir.T @ dr_eci
    omega_cross_dr = np.array(
        [
            omega_ric_eci[1] * dr_eci[2] - omega_ric_eci[2] * dr_eci[1],
            omega_ric_eci[2] * dr_eci[0] - omega_ric_eci[0] * dr_eci[2],
            omega_ric_eci[0] * dr_eci[1] - omega_ric_eci[1] * dr_eci[0],
        ],
        dtype=float,
    )
    dv_ric = c_ir.T @ (dv_eci - omega_cross_dr)
    return np.hstack((dr_ric, dv_ric))


def dcm_to_euler_321(dcm: np.ndarray) -> np.ndarray:
    psi = np.arctan2(dcm[1, 0], dcm[0, 0])
    theta = -np.arcsin(np.clip(dcm[2, 0], -1.0, 1.0))
    phi = np.arctan2(dcm[2, 1], dcm[2, 2])
    return np.array([phi, theta, psi])


def ric_curv_to_rect(x_ric_curv: np.ndarray, r0_km: float, eps: float = 1e-12) -> np.ndarray:
    if _frame_acceleration_enabled():
        bundle = _frame_kernel_bundle()
        return _frame_kernel(bundle, "ric_curv_to_rect_kernel")(
            np.asarray(x_ric_curv, dtype=float).reshape(6), float(r0_km), float(eps)
        )
    x_r_curv, x_i_curv, x_c_curv, x_r_curv_dot, x_i_curv_dot, x_c_curv_dot = np.array(x_ric_curv, dtype=float).reshape(
        6
    )
    r0 = max(float(r0_km), eps)

    r = max(r0 + x_r_curv, eps)
    theta_i = x_i_curv / r0
    theta_c = x_c_curv / r0

    c_i = np.cos(theta_i)
    s_i = np.sin(theta_i)
    c_c = np.cos(theta_c)
    s_c = np.sin(theta_c)

    x = r * c_c * c_i
    y = r * c_c * s_i
    z = r * s_c

    x_r = x - r0
    x_i = y
    x_c = z

    r_dot = x_r_curv_dot
    theta_i_dot = x_i_curv_dot / r0
    theta_c_dot = x_c_curv_dot / r0

    xdot = r_dot * c_c * c_i - r * s_c * theta_c_dot * c_i - r * c_c * s_i * theta_i_dot
    ydot = r_dot * c_c * s_i - r * s_c * theta_c_dot * s_i + r * c_c * c_i * theta_i_dot
    zdot = r_dot * s_c + r * c_c * theta_c_dot

    return np.array([x_r, x_i, x_c, xdot, ydot, zdot], dtype=float)


def ric_rect_to_curv(x_ric_rect: np.ndarray, r0_km: float, eps: float = 1e-12) -> np.ndarray:
    if _frame_acceleration_enabled():
        bundle = _frame_kernel_bundle()
        return _frame_kernel(bundle, "ric_rect_to_curv_kernel")(
            np.asarray(x_ric_rect, dtype=float).reshape(6), float(r0_km), float(eps)
        )
    x_r, x_i, x_c, x_rdot, x_idot, x_cdot = np.array(x_ric_rect, dtype=float).reshape(6)
    r0 = max(float(r0_km), eps)

    x = r0 + x_r
    y = x_i
    z = x_c
    r = np.sqrt(x * x + y * y + z * z)
    r = max(r, eps)
    p2 = x * x + y * y
    p = np.sqrt(max(p2, eps))

    theta_i = np.arctan2(y, x)
    theta_c = np.arctan2(z, p)

    x_r_curv = r - r0
    x_i_curv = r0 * theta_i
    x_c_curv = r0 * theta_c

    r_dot = (x * x_rdot + y * x_idot + z * x_cdot) / r
    theta_i_dot = (x * x_idot - y * x_rdot) / max(p2, eps)
    p_dot = (x * x_rdot + y * x_idot) / p
    theta_c_dot = (p * x_cdot - z * p_dot) / (r * r)

    return np.array(
        [
            x_r_curv,
            x_i_curv,
            x_c_curv,
            r_dot,
            r0 * theta_i_dot,
            r0 * theta_c_dot,
        ],
        dtype=float,
    )
