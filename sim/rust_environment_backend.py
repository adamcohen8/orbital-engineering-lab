"""Opt-in bindings for the Rust ONP environment and perturbation kernels.

The adapter deliberately owns no resource loading.  EOP tables, DE440 files,
FES coefficients, and Python callbacks stay on the reference side; callers
pass validated numeric arrays into Rust only when the orbit numeric backend is
explicitly ``rust``.  Missing symbols return ``None`` so an older installed
wheel can retain the established Python implementation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


def _extension() -> Any:
    try:
        return import_module("oel_rust_orbit")
    except ImportError as exc:  # pragma: no cover - exercised by installation tests
        raise RuntimeError(
            "Rust environment backend requested but oel_rust_orbit is unavailable; "
            "install an environment-enabled oel_rust_orbit wheel first"
        ) from exc


def supports(name: str) -> bool:
    """Return whether the installed wheel exposes one of these kernels."""

    try:
        return callable(getattr(_extension(), str(name), None))
    except RuntimeError:
        return False


def try_create_environment_context(nutation_coefficients: Any, nutation_terms: Any) -> Any | None:
    """Create a persistent native frame/ephemeris context when available.

    The arrays are copied once at this boundary.  Python remains responsible
    for selecting and validating the source tables and for interpolating EOP.
    """
    extension = _extension()
    # Older contexts embed the legacy apparent sidereal-time calculation.
    # Retain the authoritative Python preparation rather than silently mixing
    # a corrected frame owner with an incompatible installed native wheel.
    if getattr(extension, "ONP_IAU76_80_SIDEREAL_VERSION", 0) != 2:
        return None
    native = getattr(extension, "ONPEnvironmentContext", None)
    if not callable(native):
        return None
    coefficients = np.asarray(nutation_coefficients, dtype=np.float64).reshape(-1)
    terms = np.asarray(nutation_terms, dtype=np.float64).reshape(-1)
    if coefficients.size == 0 or terms.size == 0 or not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(terms)):
        raise ValueError("nutation tables must be finite and nonempty")
    return native(coefficients.tolist(), terms.tolist())


def try_configure_environment_de440_body(
    context: Any,
    body_index: int,
    *,
    row_starts_jd_tdb: Any,
    row_ends_jd_tdb: Any,
    coeff_count: int,
    segments: int,
    span_days: float,
    x_rows: Any,
    y_rows: Any,
    z_rows: Any,
) -> bool:
    """Copy one validated lightbank body into a persistent native context."""
    setter = getattr(context, "set_de440_body", None)
    if not callable(setter):
        return False
    starts = np.asarray(row_starts_jd_tdb, dtype=np.float64).reshape(-1)
    ends = np.asarray(row_ends_jd_tdb, dtype=np.float64).reshape(-1)
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in (x_rows, y_rows, z_rows)]
    if starts.shape != ends.shape or any(not np.all(np.isfinite(value)) for value in (starts, ends, *arrays)):
        raise ValueError("DE440 context body inputs must be finite and have matching row metadata")
    setter(
        int(body_index), starts.tolist(), ends.tolist(), int(coeff_count), int(segments), float(span_days),
        *(value.tolist() for value in arrays),
    )
    return True


def try_context_rotation_iau76_80(
    context: Any,
    t_s: float,
    *,
    jd_utc_start: float,
    xp_arcsec: float,
    yp_arcsec: float,
    dut1_s: float,
    dat_s: float,
    ddpsi_rad: float,
    ddeps_rad: float,
) -> np.ndarray | None:
    native = getattr(context, "rotation_iau76_80", None)
    if not callable(native):
        return None
    values = native(
        float(t_s), float(jd_utc_start), float(xp_arcsec), float(yp_arcsec),
        float(dut1_s), float(dat_s), float(ddpsi_rad), float(ddeps_rad),
    )
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def try_context_de440_sun_moon_km(
    context: Any,
    jd_tdb: float,
    *,
    earth_moon_mass_ratio: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    native = getattr(context, "de440_sun_moon_km", None)
    if not callable(native):
        return None
    result = native(float(jd_tdb), float(earth_moon_mass_ratio))
    return (
        _vector(result[0], (3,), "native Sun position"),
        _vector(result[1], (3,), "native Moon position"),
    )


def try_create_force_context(
    codes: Any,
    scalars: Any,
    shadow_model: int,
    harmonic_dims: tuple[int, int],
    tables: Any,
    *, precision_specs=None,
) -> Any | None:
    """Create a persistent native force context when the installed wheel supports it."""
    native = getattr(_extension(), "ONPForceContext", None)
    if not callable(native):
        return None
    kwargs = {} if precision_specs is None else {"precision_specs": precision_specs}
    if precision_specs is not None and not supports_precision_force_plan():
        return None
    return native(
        [int(value) for value in codes],
        np.asarray(scalars, dtype=np.float64).reshape(-1).tolist(),
        int(shadow_model),
        (int(harmonic_dims[0]), int(harmonic_dims[1])),
        [[float(value) for value in table] for table in tables],
        **kwargs,
    )


def supports_precision_force_plan() -> bool:
    """An explicit capability guards compatibility with older native wheels."""
    return getattr(_extension(), "ONP_PRECISION_FORCE_PLAN_VERSION", 0) == 1


def _vector(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must have shape {shape} and contain finite values")
    return result


def try_rotation(t_s: float, *, jd_utc_start: float | None, earth_rotation_rad_s: float) -> np.ndarray | None:
    native = getattr(_extension(), "environment_rotation_batch", None)
    if not callable(native):
        return None
    values = native([float(t_s)], None if jd_utc_start is None else float(jd_utc_start), float(earth_rotation_rad_s))
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def try_simple_relative_velocity(
    position_km: Any, velocity_km_s: Any, time_s: float,
    *, jd_utc_start: float | None, atmosphere_rotation_rad_s: float,
) -> np.ndarray | None:
    """Return simple-frame atmosphere-relative velocity in ECI coordinates."""
    native = getattr(_extension(), "environment_simple_relative_velocity", None)
    if not callable(native):
        return None
    result = native(np.asarray(position_km, dtype=np.float64).reshape(3),
                    np.asarray(velocity_km_s, dtype=np.float64).reshape(3), float(time_s),
                    None if jd_utc_start is None else float(jd_utc_start), float(atmosphere_rotation_rad_s))
    return np.asarray(result, dtype=np.float64).reshape(3)


def try_state_batch(
    times_s: Any,
    positions_eci_km: Any,
    velocities_eci_km_s: Any,
    *,
    jd_utc_start: float | None,
    earth_rotation_rad_s: float,
    subtract_atmosphere_rotation: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    native = getattr(_extension(), "environment_state_batch", None)
    if not callable(native):
        return None
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions_eci_km, dtype=np.float64).reshape(-1, 3)
    velocities = np.asarray(velocities_eci_km_s, dtype=np.float64).reshape(-1, 3)
    if positions.shape != velocities.shape or positions.shape[0] != times.size:
        raise ValueError("environment state batch dimensions must match")
    if not np.all(np.isfinite(np.concatenate((times, positions.ravel(), velocities.ravel())))):
        raise ValueError("environment state batch values must be finite")
    result = native(
        times.tolist(),
        positions.ravel().tolist(),
        velocities.ravel().tolist(),
        None if jd_utc_start is None else float(jd_utc_start),
        float(earth_rotation_rad_s),
        bool(subtract_atmosphere_rotation),
    )
    return (
        np.asarray(result[0], dtype=np.float64).reshape(-1, 3),
        np.asarray(result[1], dtype=np.float64).reshape(-1, 3),
    )


def try_geodetic_batch(positions_ecef_km: Any) -> np.ndarray | None:
    """Convert finite ECEF rows through the existing reference-order kernel."""
    native = getattr(_extension(), "environment_geodetic_batch", None)
    if not callable(native):
        return None
    positions = np.asarray(positions_ecef_km, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        raise ValueError("ECEF positions must be finite (rows, 3) values")
    if positions.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(native(positions.reshape(-1).tolist()), dtype=np.float64).reshape(-1, 3)


def try_geodetic(position_ecef_km: Any) -> np.ndarray | None:
    native = getattr(_extension(), "environment_geodetic_batch", None)
    if not callable(native):
        return None
    position = _vector(position_ecef_km, (3,), "ECEF position")
    return np.asarray(native(position.tolist()), dtype=np.float64).reshape(3)


def try_exponential_density(
    altitude_km: float,
    *,
    reference_density_kg_m3: float,
    reference_altitude_km: float,
    scale_height_km: float,
    ceiling_km: float,
) -> float | None:
    native = getattr(_extension(), "environment_exponential_density", None)
    if not callable(native):
        return None
    return float(
        native(
            float(altitude_km),
            float(reference_density_kg_m3),
            float(reference_altitude_km),
            float(scale_height_km),
            float(ceiling_km),
        )
    )


def try_radial_exponential_density(
    position_km: Any,
    *,
    radius_km: float,
    reference_density_kg_m3: float,
    reference_altitude_km: float,
    scale_height_km: float,
    ceiling_km: float,
) -> float | None:
    """Evaluate radial altitude and density together for one current state."""
    native = getattr(_extension(), "environment_radial_exponential_density", None)
    if not callable(native):
        return None
    position = np.asarray(position_km, dtype=np.float64).reshape(3)
    return float(native(position, float(radius_km), float(reference_density_kg_m3),
                        float(reference_altitude_km), float(scale_height_km), float(ceiling_km)))


def _batch(value: Any, width: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype="<f8")
    if result.ndim != 2 or result.shape[1] != width:
        raise ValueError(f"{label} must have shape (N, {width})")
    return result


def try_radial_exponential_density_batch(
    positions_km: Any,
    *,
    radius_km: float,
    reference_density_kg_m3: float,
    reference_altitude_km: float,
    scale_height_km: float,
    ceiling_km: float,
) -> np.ndarray | None:
    """Evaluate each radial altitude and density in a packed native batch.

    The model parameters apply to all rows; position and altitude are evaluated
    independently for every row.  An older wheel returns ``None``.
    """
    native = getattr(_extension(), "environment_radial_exponential_density_batch", None)
    if not callable(native):
        return None
    positions = _batch(positions_km, 3, "radial density positions")
    result = native(positions.tobytes(), float(radius_km), float(reference_density_kg_m3),
                    float(reference_altitude_km), float(scale_height_km), float(ceiling_km))
    return np.frombuffer(result, dtype="<f8").copy()


def try_simple_environment_batch(
    times_s: Any,
    states_eci: Any,
    *,
    jd_utc_start: float | None,
    earth_rotation_rad_s: float,
    subtract_atmosphere_rotation: bool,
    radius_km: float,
    reference_density_kg_m3: float,
    reference_altitude_km: float,
    scale_height_km: float,
    ceiling_km: float,
) -> dict[str, np.ndarray] | None:
    """Prepare simple-frame rotation/state, WGS-84 coordinates, and density.

    Each row evaluates its own time and ECI state.  Velocities are in ECEF,
    relative to a stationary atmosphere when requested.  This data-free API
    leaves IAU/EOP and ephemeris resource policy with their existing owners.
    """
    native = getattr(_extension(), "environment_simple_batch", None)
    if not callable(native):
        return None
    states = _batch(states_eci, 6, "simple environment states")
    times = np.asarray(times_s, dtype="<f8")
    if times.shape != (states.shape[0],):
        raise ValueError("simple environment batch dimensions must match")
    result = native(times.tobytes(), states.tobytes(),
                    None if jd_utc_start is None else float(jd_utc_start),
                    float(earth_rotation_rad_s), bool(subtract_atmosphere_rotation),
                    float(radius_km), float(reference_density_kg_m3),
                    float(reference_altitude_km), float(scale_height_km), float(ceiling_km))
    rows = np.frombuffer(result, dtype="<f8").reshape(-1, 19)
    return {"rotations": rows[:, :9].reshape(-1, 3, 3).copy(),
            "positions": rows[:, 9:12].copy(), "velocities": rows[:, 12:15].copy(),
            "geodetic": rows[:, 15:18].copy(), "density": rows[:, 18].copy()}


def try_jb_lower_atmosphere(altitude_km: float, temperature_coefficients: Any) -> tuple | None:
    """Evaluate the established JB2006/JB2008 lower quadrature in native code."""
    native = getattr(_extension(), "environment_jb_lower_atmosphere", None)
    if not callable(native):
        return None
    return tuple(native(float(altitude_km), tuple(float(value) for value in temperature_coefficients)))


def try_jb_upper_atmosphere(
    zend: float, z_current: float, z_target: float, r_step: float,
    ain: float, temperature_coefficients: Any,
) -> tuple | None:
    """Evaluate the shared JB2006/JB2008 upper quadrature in native code."""
    native = getattr(_extension(), "environment_jb_upper_atmosphere", None)
    if not callable(native):
        return None
    return tuple(native(float(zend), float(z_current), float(z_target), float(r_step),
                        float(ain), tuple(float(value) for value in temperature_coefficients)))


def try_de440_chebyshev_batch(
    times_jd_tdb: Any,
    *,
    row_starts_jd_tdb: Any,
    row_ends_jd_tdb: Any,
    coeff_count: int,
    segments: int,
    span_days: float,
    x_rows: Any,
    y_rows: Any,
    z_rows: Any,
) -> np.ndarray | None:
    """Evaluate DE440 Chebyshev rows and return positions in kilometres.

    The Rust extension follows the DE440 resource convention used by the
    existing Python evaluator and returns the polynomial result in metres
    (the extension applies the resource's ``1e3`` scale factor).  This
    adapter is the Python boundary used by the orbit ephemeris API, whose
    public contract is kilometres, so convert once here rather than making
    every caller remember the native unit.
    """
    native = getattr(_extension(), "environment_de440_chebyshev_batch", None)
    if not callable(native):
        return None
    times = np.asarray(times_jd_tdb, dtype=np.float64).reshape(-1)
    starts = np.asarray(row_starts_jd_tdb, dtype=np.float64).reshape(-1)
    ends = np.asarray(row_ends_jd_tdb, dtype=np.float64).reshape(-1)
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in (x_rows, y_rows, z_rows)]
    if starts.shape != ends.shape or any(not np.all(np.isfinite(value)) for value in (times, starts, ends, *arrays)):
        raise ValueError("DE440 batch inputs must be finite and have matching row metadata")
    result = native(
        times.tolist(),
        starts.tolist(),
        ends.tolist(),
        int(coeff_count),
        int(segments),
        float(span_days),
        *(array.tolist() for array in arrays),
    )
    return np.asarray(result, dtype=np.float64).reshape(-1, 3) / 1.0e3


def try_zonal_acceleration(
    position_km: Any,
    *,
    mu_km3_s2: float,
    j2: float,
    j3: float,
    j4: float,
    radius_km: float,
    codes: list[int],
) -> np.ndarray | None:
    native = getattr(_extension(), "perturbation_zonal_acceleration", None)
    if not callable(native):
        return None
    position = _vector(position_km, (3,), "position")
    return np.asarray(native(position.tolist(), float(mu_km3_s2), float(j2), float(j3), float(j4), float(radius_km), list(codes)), dtype=np.float64).reshape(3)


def try_third_body_acceleration(position_km: Any, body_position_km: Any, *, mu_km3_s2: float) -> np.ndarray | None:
    native = getattr(_extension(), "perturbation_third_body", None)
    if not callable(native):
        return None
    position = _vector(position_km, (3,), "position")
    body = _vector(body_position_km, (3,), "body position")
    return np.asarray(native(position.tolist(), body.tolist(), float(mu_km3_s2)), dtype=np.float64).reshape(3)


def try_srp_acceleration(
    position_km: Any,
    sun_position_km: Any,
    *,
    mass_kg: float,
    area_m2: float,
    reflectivity: float,
    pressure_pa: float,
    au_km: float,
    earth_radius_km: float,
    sun_radius_km: float,
    shadow_model: int,
) -> np.ndarray | None:
    native = getattr(_extension(), "perturbation_srp", None)
    if not callable(native):
        return None
    position = _vector(position_km, (3,), "position")
    sun = _vector(sun_position_km, (3,), "Sun position")
    return np.asarray(native(position.tolist(), sun.tolist(), float(mass_kg), float(area_m2), float(reflectivity), float(pressure_pa), float(au_km), float(earth_radius_km), float(sun_radius_km), int(shadow_model)), dtype=np.float64).reshape(3)


def _schwarzschild_prepared_acceleration(native: Any, state: Any, *, mu_km3_s2: float) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError("state must have shape (6,) and contain finite values")
    try:
        mu = float(mu_km3_s2)
    except Exception:
        # Preserve the existing state-before-mu validation order even when
        # conversion of mu fails. The successful path validates in Rust.
        _vector(values, (6,), "state")
        raise
    return np.asarray(native(values, mu), dtype=np.float64).reshape(3)


def prepare_schwarzschild_acceleration() -> Any | None:
    """Bind the prepared native symbol once for a persistent force plugin."""
    native = getattr(_extension(), "perturbation_schwarzschild_prepared", None)
    if not callable(native):
        return None
    from functools import partial

    return partial(_schwarzschild_prepared_acceleration, native)


def try_schwarzschild_acceleration(state: Any, *, mu_km3_s2: float) -> np.ndarray | None:
    extension = _extension()
    native = getattr(extension, "perturbation_schwarzschild_prepared", None)
    if callable(native):
        return _schwarzschild_prepared_acceleration(native, state, mu_km3_s2=mu_km3_s2)
    native = getattr(extension, "perturbation_schwarzschild", None)
    if not callable(native):
        return None
    values = _vector(state, (6,), "state")
    return np.asarray(native(values.tolist(), float(mu_km3_s2)), dtype=np.float64).reshape(3)


def try_tidal_acceleration(position_km: Any, c_nm: Any, s_nm: Any, *, mu_km3_s2: float, radius_km: float) -> np.ndarray | None:
    native = getattr(_extension(), "perturbation_tidal_acceleration", None)
    if not callable(native):
        return None
    position = _vector(position_km, (3,), "position")
    c = np.asarray(c_nm, dtype=np.float64)
    s = np.asarray(s_nm, dtype=np.float64)
    if c.ndim != 2 or c.shape != s.shape or c.shape[0] != c.shape[1] or not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
        raise ValueError("tidal coefficient matrices must be matching finite square arrays")
    return np.asarray(native(position.tolist(), c.ravel().tolist(), s.ravel().tolist(), int(c.shape[0] - 1), float(mu_km3_s2), float(radius_km)), dtype=np.float64).reshape(3)


def try_earth_radiation_components(
    position_fixed_km: Any,
    sun_fixed_km: Any,
    elapsed_s: float,
    *,
    order: int,
    include_albedo: bool,
    include_infrared: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    native = getattr(_extension(), "perturbation_earth_radiation", None)
    if not callable(native):
        return None
    from sim.pro_perturbations.earth_radiation import _quadrature

    position = _vector(position_fixed_km, (3,), "fixed spacecraft position")
    sun = _vector(sun_fixed_km, (3,), "fixed Sun position")
    nodes, weights, cos_azimuth, sin_azimuth = _quadrature(int(order))
    result = native(
        position.tolist(),
        sun.tolist(),
        float(elapsed_s),
        6378.137,
        np.asarray(nodes, dtype=np.float64).tolist(),
        np.asarray(weights, dtype=np.float64).tolist(),
        np.asarray(cos_azimuth, dtype=np.float64).tolist(),
        np.asarray(sin_azimuth, dtype=np.float64).tolist(),
        bool(include_albedo),
        bool(include_infrared),
        None,
        None,
    )
    return np.asarray(result[0], dtype=np.float64).reshape(3), np.asarray(result[1], dtype=np.float64).reshape(3)


def try_earth_radiation_components_batch(
    positions_fixed_km: Any,
    sun_positions_fixed_km: Any,
    elapsed_times_s: Any,
    *,
    order: int,
    include_albedo: bool,
    include_infrared: bool,
    radius_km: float = 6378.137,
    uniform_coefficients: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return separate albedo/infrared pressures for every supplied geometry.

    Callers retain frame, epoch, resource, and quadrature policy.  This adapter
    uses the same established quadrature as the scalar production path.
    """
    native = getattr(_extension(), "perturbation_earth_radiation_batch", None)
    if not callable(native):
        return None
    from sim.pro_perturbations.earth_radiation import _quadrature

    positions = _batch(positions_fixed_km, 3, "fixed spacecraft positions")
    suns = _batch(sun_positions_fixed_km, 3, "fixed Sun positions")
    times = np.asarray(elapsed_times_s, dtype="<f8")
    if positions.shape != suns.shape or times.shape != (positions.shape[0],):
        raise ValueError("Earth radiation batch dimensions must match")
    if uniform_coefficients is not None and len(uniform_coefficients) != 2:
        raise ValueError("uniform coefficients must contain albedo and emissivity")
    a, e = (None, None) if uniform_coefficients is None else uniform_coefficients
    nodes, weights, cp, sp = _quadrature(order)
    result = native(positions.tobytes(), suns.tobytes(), times.tobytes(), float(radius_km),
                    nodes.tolist(), weights.tolist(), cp.tolist(), sp.tolist(),
                    bool(include_albedo), bool(include_infrared), a, e)
    components = np.frombuffer(result, dtype="<f8").reshape(-1, 6)
    return components[:, :3].copy(), components[:, 3:].copy()


def try_force_bundle_batch(
    states_eci: Any,
    body_positions_eci_km: Any,
    sun_positions_eci_km: Any,
    c_nm: Any,
    s_nm: Any,
    *,
    mu_km3_s2: float,
    j2: float,
    j3: float,
    j4: float,
    radius_km: float,
    body_mu_km3_s2: float,
    mass_kg: float,
    area_m2: float,
    reflectivity: float,
    pressure_pa: float,
    au_km: float,
    sun_radius_km: float,
    shadow_model: int,
    codes: tuple[int, ...] = (2, 3, 4),
) -> np.ndarray | None:
    """Batch additive zonal, third-body, SRP, Schwarzschild, and tidal forces.

    This excludes central gravity.  Each row follows the existing scalar force
    addition order and receives its own spacecraft, body, and Sun state.
    Prepared coefficient matrices and physical constants apply to all rows.
    """
    native = getattr(_extension(), "perturbation_force_bundle_batch", None)
    if not callable(native):
        return None
    states = _batch(states_eci, 6, "force bundle states")
    bodies = _batch(body_positions_eci_km, 3, "force bundle body positions")
    suns = _batch(sun_positions_eci_km, 3, "force bundle Sun positions")
    if bodies.shape != suns.shape or bodies.shape[0] != states.shape[0]:
        raise ValueError("force bundle batch dimensions must match")
    c = np.asarray(c_nm, dtype=np.float64)
    s = np.asarray(s_nm, dtype=np.float64)
    if c.ndim != 2 or c.shape != s.shape or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        raise ValueError("tidal coefficient matrices must be matching nonempty square arrays")
    scalars = (mu_km3_s2, j2, j3, j4, radius_km, body_mu_km3_s2,
               mass_kg, area_m2, reflectivity, pressure_pa, au_km, sun_radius_km)
    result = native(states.tobytes(), bodies.tobytes(), suns.tobytes(),
                    tuple(float(value) for value in scalars), int(shadow_model),
                    tuple(int(code) for code in codes), c.ravel().tolist(), s.ravel().tolist(), c.shape[0] - 1)
    return np.frombuffer(result, dtype="<f8").reshape(-1, 3).copy()


def try_solid_tides_acceleration(position_km, sun_fixed_km, moon_fixed_km, *, mu_km3_s2, radius_km,
                                  jd_tt, jd_ut1, tide_system, pole_xy_arcsec, sun_mu, moon_mu):
    """Evaluate IERS solid-tide coefficients and their gradient in one Rust call."""
    native = getattr(_extension(), "perturbation_solid_tides", None)
    if not callable(native):
        return None
    position = _vector(position_km, (3,), "position")
    sun = _vector(sun_fixed_km, (3,), "Sun position")
    moon = _vector(moon_fixed_km, (3,), "Moon position")
    result = native(position.tolist(), sun.tolist(), moon.tolist(), float(mu_km3_s2), float(radius_km),
                    float(jd_tt), float(jd_ut1), str(tide_system), pole_xy_arcsec, float(sun_mu), float(moon_mu))
    return np.asarray(result, dtype=np.float64).reshape(3)


def try_create_ocean_tides_context(factors, coefficients):
    """Bind validated, immutable FES wave arrays to an optional native context."""
    factory = getattr(_extension(), "OceanTidesContext", None)
    if not callable(factory):
        return None
    factors = np.asarray(factors, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if (factors.ndim != 2 or factors.shape[1] != 6 or coefficients.ndim != 4
            or coefficients.shape[0] != factors.shape[0] or coefficients.shape[-1] != 4
            or coefficients.shape[1] != coefficients.shape[2]
            or not np.all(np.isfinite(factors)) or not np.all(np.isfinite(coefficients))):
        raise ValueError("Ocean tide context requires finite factors (N,6) and coefficients (N,D,D,4).")
    return factory(factors.ravel().tolist(), coefficients.ravel().tolist(), coefficients.shape[1] - 1)


def try_context_ocean_tides_acceleration(context, position_km, *, jd_tt, jd_ut1, pole_xy_arcsec,
                                        mu_km3_s2, radius_km):
    """Evaluate one content-bound ocean tide force without Python wave matrices."""
    native = getattr(context, "acceleration", None)
    if not callable(native):
        return None
    position = _vector(position_km, (3,), "position")
    result = native(position.tolist(), float(jd_tt), float(jd_ut1), pole_xy_arcsec,
                    float(mu_km3_s2), float(radius_km))
    return np.asarray(result, dtype=np.float64).reshape(3)
