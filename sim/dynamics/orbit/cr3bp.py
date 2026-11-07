from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, MOON_MU_KM3_S2
from sim.dynamics.orbit.integrators import AdaptiveStepInfo, integrate_adaptive, rk4_step_state

EARTH_MOON_DISTANCE_KM = 384400.0
EARTH_MOON_MU = MOON_MU_KM3_S2 / (EARTH_MU_KM3_S2 + MOON_MU_KM3_S2)
EARTH_MOON_MEAN_MOTION_RAD_S = float(np.sqrt((EARTH_MU_KM3_S2 + MOON_MU_KM3_S2) / (EARTH_MOON_DISTANCE_KM**3)))


@dataclass(frozen=True)
class CR3BPSystem:
    name: str = "earth_moon"
    distance_km: float = EARTH_MOON_DISTANCE_KM
    mu: float = EARTH_MOON_MU
    mean_motion_rad_s: float = EARTH_MOON_MEAN_MOTION_RAD_S

    def __post_init__(self) -> None:
        distance = float(self.distance_km)
        mass_ratio = float(self.mu)
        mean_motion = float(self.mean_motion_rad_s)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("CR3BP distance_km must be finite and positive.")
        if not np.isfinite(mass_ratio) or not 0.0 < mass_ratio < 1.0:
            raise ValueError("CR3BP mu must be finite and satisfy 0 < mu < 1.")
        if not np.isfinite(mean_motion) or mean_motion <= 0.0:
            raise ValueError("CR3BP mean_motion_rad_s must be finite and positive.")
        object.__setattr__(self, "distance_km", distance)
        object.__setattr__(self, "mu", mass_ratio)
        object.__setattr__(self, "mean_motion_rad_s", mean_motion)


EARTH_MOON_CR3BP = CR3BPSystem()


def cr3bp_system(name: str = "earth_moon") -> CR3BPSystem:
    key = str(name or "earth_moon").strip().lower().replace("-", "_")
    if key in {"earth_moon", "earthmoon", "em"}:
        return EARTH_MOON_CR3BP
    raise ValueError(f"Unsupported CR3BP system '{name}'.")


def cr3bp_l1_position_km(system: CR3BPSystem | None = None) -> float:
    sys = EARTH_MOON_CR3BP if system is None else system
    return float(_collinear_libration_x(sys.mu, point="l1") * sys.distance_km)


def cr3bp_l1_state_km_s(system: CR3BPSystem | None = None) -> np.ndarray:
    state = np.zeros(6, dtype=float)
    state[0] = cr3bp_l1_position_km(system)
    return state


def cr3bp_moon_position_km(system: CR3BPSystem | None = None) -> float:
    sys = EARTH_MOON_CR3BP if system is None else system
    return float((1.0 - float(sys.mu)) * float(sys.distance_km))


def cr3bp_moon_state_km_s(system: CR3BPSystem | None = None) -> np.ndarray:
    state = np.zeros(6, dtype=float)
    state[0] = cr3bp_moon_position_km(system)
    return state


def cr3bp_halo_seed_state_km_s(
    *,
    system: CR3BPSystem | None = None,
    family: str = "l1_northern",
) -> np.ndarray:
    """Return a deterministic Earth-Moon CR3BP orbit seed in rotating coordinates.

    The larger L1 family is a corrected periodic northern halo orbit. The NRHO
    family is a corrected southern L2 near-rectilinear halo orbit. The default
    family remains the original small training seed for backward compatibility
    with existing scenarios.
    """

    sys = EARTH_MOON_CR3BP if system is None else system
    key = str(family or "l1_northern").strip().lower().replace("-", "_")
    if key in {"l1_northern_large", "l1_large", "large", "large_northern", "l1_northern_corrected_large"}:
        state_nd = np.array(
            [
                0.825758855860872,
                0.0,
                0.08,
                0.0,
                0.193697258505661,
                0.0,
            ],
            dtype=float,
        )
        return cr3bp_dimensional_state(state_nd, system=sys)
    if key in {"l2_nrho_southern", "nrho", "southern_nrho", "l2_southern_nrho", "gateway_nrho"}:
        state_nd = np.array(
            [
                1.0213444229227893,
                0.0,
                -0.181626,
                0.0,
                -0.10177667502757823,
                0.0,
            ],
            dtype=float,
        )
        return cr3bp_dimensional_state(state_nd, system=sys)
    if key not in {"l1_northern", "northern", "north"}:
        raise ValueError(f"Unsupported CR3BP orbit seed family '{family}'.")
    state_nd = np.array(
        [
            0.823385182067467,
            0.0,
            0.022277556,
            0.0,
            0.134184,
            0.0,
        ],
        dtype=float,
    )
    return cr3bp_dimensional_state(state_nd, system=sys)


def cr3bp_dimensional_state(state_nd: np.ndarray, *, system: CR3BPSystem | None = None) -> np.ndarray:
    sys = EARTH_MOON_CR3BP if system is None else system
    state = np.array(state_nd, dtype=float).reshape(6).copy()
    state[:3] *= float(sys.distance_km)
    state[3:] *= float(sys.distance_km) * float(sys.mean_motion_rad_s)
    return state


def cr3bp_nondimensional_state(state_km_s: np.ndarray, *, system: CR3BPSystem | None = None) -> np.ndarray:
    sys = EARTH_MOON_CR3BP if system is None else system
    state = np.array(state_km_s, dtype=float).reshape(6).copy()
    state[:3] /= float(sys.distance_km)
    state[3:] /= float(sys.distance_km) * float(sys.mean_motion_rad_s)
    return state


def cr3bp_relative_state(deputy_state: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
    return np.array(deputy_state, dtype=float).reshape(6) - np.array(reference_state, dtype=float).reshape(6)


def _integrate_cr3bp(deriv, state, dt_s, t_s, *, integrator, adaptive_atol, adaptive_rtol, h_init):
    method = str(integrator).strip().lower()
    dt = float(dt_s)
    if not np.isfinite(dt) or (method != "rk4" and dt < 0.0):
        raise ValueError("CR3BP dt_s must be finite and nonnegative.")
    epoch = float(t_s)
    if not np.isfinite(epoch):
        raise ValueError("CR3BP t_s must be finite.")
    if method == "rk4":
        return rk4_step_state(deriv_fn=deriv, t_s=epoch, x=state, dt_s=dt), None
    if method not in {"rkf78", "adaptive", "dopri5"}:
        raise ValueError(f"Unsupported CR3BP integrator '{integrator}'.")
    if not np.isfinite([adaptive_atol, adaptive_rtol]).all() or adaptive_atol <= 0 or adaptive_rtol <= 0:
        raise ValueError("CR3BP adaptive tolerances must be positive and finite.")
    return integrate_adaptive(
        deriv_fn=deriv,
        t_s=epoch,
        x=state,
        dt_s=dt,
        atol=adaptive_atol,
        rtol=adaptive_rtol,
        method="rkf78" if method == "adaptive" else method,
        h_init=h_init,
        return_info=True,
    )


def propagate_cr3bp_state(
    state_km_s: np.ndarray,
    dt_s: float,
    t_s: float,
    command_accel_km_s2: np.ndarray | None = None,
    *,
    system: CR3BPSystem | None = None,
    integrator: str = "rk4",
    adaptive_atol: float = 1e-9,
    adaptive_rtol: float = 1e-7,
    h_init: float | None = None,
    return_info: bool = False,
    numeric_backend: str = "rust",
) -> np.ndarray | tuple[np.ndarray, AdaptiveStepInfo | None]:
    """Propagate physical rotating state; optional diagnostics are None for RK4."""
    sys = EARTH_MOON_CR3BP if system is None else system
    state = _validated_cr3bp_state(state_km_s, "state_km_s")
    signed_rk4 = str(integrator).strip().lower() == "rk4"
    dt = _validated_cr3bp_time(dt_s, "dt_s", nonnegative=not signed_rk4)
    epoch = _validated_cr3bp_time(t_s, "t_s")
    _validate_cr3bp_integrator_options(integrator, adaptive_atol, adaptive_rtol)
    command = (
        None
        if command_accel_km_s2 is None
        else np.asarray(command_accel_km_s2, dtype=float).reshape(3)
    )
    if command is not None and not np.isfinite(command).all():
        raise ValueError("CR3BP command_accel_km_s2 must be finite.")

    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown CR3BP numeric backend {numeric_backend!r}")
    if backend == "rust":
        from sim.dynamics.orbit.rust_cr3bp import integrate

        out, info = integrate(
            state, dt, epoch, system=sys, command=command,
            integrator=integrator, adaptive_atol=adaptive_atol,
            adaptive_rtol=adaptive_rtol, h_init=h_init, collect_info=return_info,
        )
        return (out, info) if return_info else out

    if command is None:
        command = np.zeros(3, dtype=float)

    def deriv(t_local: float, x_local: np.ndarray) -> np.ndarray:
        return cr3bp_derivative_physical(x_local, command_accel_km_s2=command, system=sys)

    out, info = _integrate_cr3bp(
        deriv,
        state,
        dt,
        epoch,
        integrator=integrator,
        adaptive_atol=adaptive_atol,
        adaptive_rtol=adaptive_rtol,
        h_init=h_init,
    )
    return (out, info) if return_info else out


def propagate_cr3bp_reference_stm(
    reference_state_km_s: np.ndarray,
    stm: np.ndarray,
    dt_s: float,
    t_s: float,
    *,
    system: CR3BPSystem | None = None,
    integrator: str = "rk4",
    adaptive_atol: float = 1e-9,
    adaptive_rtol: float = 1e-7,
    h_init: float | None = None,
    return_info: bool = False,
    numeric_backend: str = "rust",
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, AdaptiveStepInfo | None]:
    """Integrate the 42-component reference/STM system with shared adaptive steps.

    Scalar tolerances apply componentwise to the physical state and STM entries.
    Default return shape and RK4 stepping preserve Trainer compatibility.
    """
    sys = EARTH_MOON_CR3BP if system is None else system
    reference = _validated_cr3bp_state(reference_state_km_s, "reference_state_km_s")
    phi = np.asarray(stm, dtype=float).reshape(6, 6).copy()
    if not np.isfinite(phi).all():
        raise ValueError("CR3BP stm must be finite.")
    signed_rk4 = str(integrator).strip().lower() == "rk4"
    dt = _validated_cr3bp_time(dt_s, "dt_s", nonnegative=not signed_rk4)
    epoch = _validated_cr3bp_time(t_s, "t_s")
    _validate_cr3bp_integrator_options(integrator, adaptive_atol, adaptive_rtol)
    augmented = np.empty(42, dtype=float)
    augmented[:6] = reference
    augmented[6:] = phi.reshape(36)

    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown CR3BP numeric backend {numeric_backend!r}")
    if backend == "rust":
        from sim.dynamics.orbit.rust_cr3bp import integrate

        out, info = integrate(
            augmented, dt, epoch, system=sys, stm=True,
            integrator=integrator, adaptive_atol=adaptive_atol,
            adaptive_rtol=adaptive_rtol, h_init=h_init, collect_info=return_info,
        )
        result = (out[:6].copy(), out[6:].reshape(6, 6).copy())
        return (*result, info) if return_info else result

    def deriv(t_local: float, x_local: np.ndarray) -> np.ndarray:
        reference = np.array(x_local[:6], dtype=float)
        phi = np.array(x_local[6:], dtype=float).reshape(6, 6)
        a = cr3bp_jacobian_physical(reference, system=sys)
        return np.hstack((cr3bp_derivative_physical(reference, system=sys), (a @ phi).reshape(36)))

    out, info = _integrate_cr3bp(
        deriv,
        augmented,
        dt,
        epoch,
        integrator=integrator,
        adaptive_atol=adaptive_atol,
        adaptive_rtol=adaptive_rtol,
        h_init=h_init,
    )
    result = (out[:6].copy(), out[6:].reshape(6, 6).copy())
    return (*result, info) if return_info else result


def cr3bp_derivative_physical(
    state_km_s: np.ndarray,
    *,
    command_accel_km_s2: np.ndarray | None = None,
    system: CR3BPSystem | None = None,
) -> np.ndarray:
    sys = EARTH_MOON_CR3BP if system is None else system
    state_nd = cr3bp_nondimensional_state(state_km_s, system=sys)
    deriv_nd = cr3bp_derivative_nondimensional(state_nd, mu=sys.mu)
    deriv = np.empty(6, dtype=float)
    deriv[:3] = np.array(state_km_s, dtype=float).reshape(6)[3:]
    deriv[3:] = deriv_nd[3:] * float(sys.distance_km) * float(sys.mean_motion_rad_s) ** 2
    if command_accel_km_s2 is not None:
        deriv[3:] += np.array(command_accel_km_s2, dtype=float).reshape(3)
    return deriv


def _validated_cr3bp_state(value: np.ndarray, name: str) -> np.ndarray:
    try:
        state = np.asarray(value, dtype=float).reshape(6)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CR3BP {name} must contain six finite values.") from exc
    if not np.isfinite(state).all():
        raise ValueError(f"CR3BP {name} must contain six finite values.")
    return state.copy()


def _validated_cr3bp_time(value: float, name: str, *, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"CR3BP {name} must be finite.") from exc
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"CR3BP {name} must be {qualifier}.")
    return result


def _validate_cr3bp_integrator_options(integrator: str, adaptive_atol: float, adaptive_rtol: float) -> None:
    method = str(integrator).strip().lower()
    if method not in {"rk4", "rkf78", "adaptive", "dopri5"}:
        raise ValueError(f"Unsupported CR3BP integrator '{integrator}'.")
    try:
        atol = float(adaptive_atol)
        rtol = float(adaptive_rtol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CR3BP adaptive tolerances must be positive and finite.") from exc
    if not np.isfinite([atol, rtol]).all() or atol <= 0.0 or rtol <= 0.0:
        raise ValueError("CR3BP adaptive tolerances must be positive and finite.")


def cr3bp_jacobian_physical(state_km_s: np.ndarray, *, system: CR3BPSystem | None = None) -> np.ndarray:
    sys = EARTH_MOON_CR3BP if system is None else system
    state_nd = cr3bp_nondimensional_state(state_km_s, system=sys)
    u_rr_nd = cr3bp_potential_hessian_nondimensional(state_nd, mu=sys.mu)
    n = float(sys.mean_motion_rad_s)
    a = np.zeros((6, 6), dtype=float)
    a[:3, 3:] = np.eye(3, dtype=float)
    a[3:, :3] = (n**2) * u_rr_nd
    a[3:, 3:] = n * np.array(
        [
            [0.0, 2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return a


def cr3bp_derivative_nondimensional(state_nd: np.ndarray, *, mu: float = EARTH_MOON_MU) -> np.ndarray:
    x, y, z, vx, vy, vz = np.array(state_nd, dtype=float).reshape(6)
    mu2 = float(mu)
    mu1 = 1.0 - mu2
    r1 = float(np.sqrt((x + mu2) ** 2 + y * y + z * z))
    r2 = float(np.sqrt((x - mu1) ** 2 + y * y + z * z))
    r1 = max(r1, 1.0e-12)
    r2 = max(r2, 1.0e-12)
    d_omega_dx = x - mu1 * (x + mu2) / (r1**3) - mu2 * (x - mu1) / (r2**3)
    d_omega_dy = y - mu1 * y / (r1**3) - mu2 * y / (r2**3)
    d_omega_dz = -mu1 * z / (r1**3) - mu2 * z / (r2**3)
    return np.array(
        [
            vx,
            vy,
            vz,
            2.0 * vy + d_omega_dx,
            -2.0 * vx + d_omega_dy,
            d_omega_dz,
        ],
        dtype=float,
    )


def cr3bp_potential_hessian_nondimensional(state_nd: np.ndarray, *, mu: float = EARTH_MOON_MU) -> np.ndarray:
    x, y, z = np.array(state_nd, dtype=float).reshape(6)[:3]
    mu2 = float(mu)
    mu1 = 1.0 - mu2
    x1 = x + mu2
    x2 = x - mu1
    r1_sq = x1 * x1 + y * y + z * z
    r2_sq = x2 * x2 + y * y + z * z
    r1 = max(float(np.sqrt(r1_sq)), 1.0e-12)
    r2 = max(float(np.sqrt(r2_sq)), 1.0e-12)
    r1_3 = r1**3
    r2_3 = r2**3
    r1_5 = r1**5
    r2_5 = r2**5

    u_xx = 1.0 - mu1 * (1.0 / r1_3 - 3.0 * x1 * x1 / r1_5) - mu2 * (1.0 / r2_3 - 3.0 * x2 * x2 / r2_5)
    u_yy = 1.0 - mu1 * (1.0 / r1_3 - 3.0 * y * y / r1_5) - mu2 * (1.0 / r2_3 - 3.0 * y * y / r2_5)
    u_zz = -mu1 * (1.0 / r1_3 - 3.0 * z * z / r1_5) - mu2 * (1.0 / r2_3 - 3.0 * z * z / r2_5)
    u_xy = 3.0 * mu1 * x1 * y / r1_5 + 3.0 * mu2 * x2 * y / r2_5
    u_xz = 3.0 * mu1 * x1 * z / r1_5 + 3.0 * mu2 * x2 * z / r2_5
    u_yz = 3.0 * mu1 * y * z / r1_5 + 3.0 * mu2 * y * z / r2_5
    return np.array(
        [
            [u_xx, u_xy, u_xz],
            [u_xy, u_yy, u_yz],
            [u_xz, u_yz, u_zz],
        ],
        dtype=float,
    )


def _collinear_libration_x(mu: float, *, point: str) -> float:
    key = str(point or "l1").strip().lower()
    if key != "l1":
        raise ValueError(f"Unsupported collinear libration point '{point}'.")
    x = 1.0 - float(mu) - (float(mu) / 3.0) ** (1.0 / 3.0)
    for _ in range(50):
        f = _collinear_equilibrium_f(x, float(mu))
        h = 1.0e-7
        df = (_collinear_equilibrium_f(x + h, float(mu)) - _collinear_equilibrium_f(x - h, float(mu))) / (2.0 * h)
        if abs(df) <= 1.0e-14:
            break
        step = f / df
        x -= step
        if abs(step) <= 1.0e-14:
            break
    return float(x)


def _collinear_equilibrium_f(x: float, mu: float) -> float:
    mu2 = float(mu)
    mu1 = 1.0 - mu2
    r1 = abs(float(x) + mu2)
    r2 = abs(float(x) - mu1)
    return float(x) - mu1 * (float(x) + mu2) / (r1**3) - mu2 * (float(x) - mu1) / (r2**3)
