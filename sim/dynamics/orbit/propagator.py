from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from threading import RLock
from typing import Callable

import numpy as np

from sim.acceleration.settings import acceleration_enabled_from_mode, acceleration_settings_from_mode
from sim.dynamics.orbit.accelerations import (
    OrbitContext,
    accel_drag_resolved,
    accel_j2,
    accel_j3,
    accel_j4,
    accel_lift,
    accel_srp_resolved,
    accel_third_body,
    accel_two_body,
)
from sim.dynamics.orbit.atmosphere import (
    _datetime_from_env_t_s,
    _nrlmsise00_sun_longitude_rad,
    density_from_model,
)
from sim.dynamics.orbit.cr3bp import cr3bp_system, propagate_cr3bp_state
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.environment import (
    EARTH_RADIUS_KM,
    EARTH_ROT_RATE_RAD_S,
    JUPITER_MU_KM3_S2,
    MARS_MU_KM3_S2,
    MERCURY_MU_KM3_S2,
    MOON_MU_KM3_S2,
    NEPTUNE_MU_KM3_S2,
    PLUTO_MU_KM3_S2,
    SATURN_MU_KM3_S2,
    SUN_MU_KM3_S2,
    SUN_RADIUS_KM,
    URANUS_MU_KM3_S2,
    VENUS_MU_KM3_S2,
    srp_pressure_n_m2,
)
from sim.dynamics.orbit.epoch import (
    AU_KM,
    resolve_body_position_eci_km,
    resolve_sun_moon_positions,
    resolve_time_dependent_env,
)
from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    _interp_eop,
    _interp_eop_nutation,
    _load_nut80_table,
    _validate_eop_elapsed_interval,
    eci_to_ecef_rotation,
    eci_to_ecef_rotation_hpop_like,
    normalize_frame_model,
)
from sim.dynamics.orbit.integrators import (
    AdaptiveStepInfo,
    combine_adaptive_step_info,
    integrate_adaptive,
    rk4_step_state,
)
from sim.dynamics.orbit.spherical_harmonics import (
    accel_spherical_harmonics_terms,
    compile_spherical_harmonic_terms,
    load_real_earth_gravity_terms,
    parse_spherical_harmonic_terms,
)

AccelerationPlugin = Callable[[float, np.ndarray, dict, OrbitContext], np.ndarray]
PLANETARY_MU_KM3_S2 = {
    "mercury": MERCURY_MU_KM3_S2,
    "venus": VENUS_MU_KM3_S2,
    "mars": MARS_MU_KM3_S2,
    "jupiter": JUPITER_MU_KM3_S2,
    "saturn": SATURN_MU_KM3_S2,
    "uranus": URANUS_MU_KM3_S2,
    "neptune": NEPTUNE_MU_KM3_S2,
    "pluto": PLUTO_MU_KM3_S2,
}
_ZERO3 = np.zeros(3, dtype=float)
_IDENTITY3 = np.eye(3, dtype=float)
_DUMMY_MATRIX = np.zeros((1, 1), dtype=float)
_DUMMY_VECTOR = np.zeros(1, dtype=float)
rk4_zonal_step_state = None
_RK4_ZONAL_KERNEL_LOCK = RLock()


def _load_rk4_zonal_step_state():
    global rk4_zonal_step_state
    kernel = rk4_zonal_step_state
    if kernel is None:
        with _RK4_ZONAL_KERNEL_LOCK:
            kernel = rk4_zonal_step_state
            if kernel is None:
                from sim.acceleration.kernels.orbit import rk4_zonal_step_state as accelerated_step

                kernel = accelerated_step
                rk4_zonal_step_state = kernel
    return kernel


@lru_cache(maxsize=1)
def _compiled_builtin_force_plan_step():
    from sim.acceleration.kernels.orbit_force_plan import rk4_builtin_force_plan_step_kernel

    return rk4_builtin_force_plan_step_kernel


@lru_cache(maxsize=1)
def _compiled_builtin_force_components():
    from sim.acceleration.kernels.orbit_force_plan import builtin_force_components_kernel

    return builtin_force_components_kernel


@lru_cache(maxsize=1)
def _compiled_drag_force_component():
    from sim.acceleration.kernels.orbit_force_plan import _drag_acceleration

    return _drag_acceleration


@lru_cache(maxsize=1)
def _compiled_lift_force_component():
    from sim.acceleration.kernels.orbit_force_plan import _lift_acceleration

    return _lift_acceleration


@lru_cache(maxsize=1)
def _compiled_iau76_80_rotation():
    from sim.acceleration.kernels.frames import eci_to_ecef_iau76_80_kernel

    return eci_to_ecef_iau76_80_kernel



def j2_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    if env.get("_rust_numeric_backend") == "rust":
        from sim.dynamics.orbit.environment import EARTH_J2, EARTH_J3, EARTH_J4
        from sim.rust_environment_backend import try_zonal_acceleration

        native = try_zonal_acceleration(
            x_eci[:3], mu_km3_s2=ctx.mu_km3_s2, j2=EARTH_J2, j3=EARTH_J3, j4=EARTH_J4,
            radius_km=EARTH_RADIUS_KM, codes=[2],
        )
        if native is not None:
            return native
    return accel_j2(x_eci[:3], ctx.mu_km3_s2)


def j3_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    if env.get("_rust_numeric_backend") == "rust":
        from sim.dynamics.orbit.environment import EARTH_J2, EARTH_J3, EARTH_J4
        from sim.rust_environment_backend import try_zonal_acceleration

        native = try_zonal_acceleration(
            x_eci[:3], mu_km3_s2=ctx.mu_km3_s2, j2=EARTH_J2, j3=EARTH_J3, j4=EARTH_J4,
            radius_km=EARTH_RADIUS_KM, codes=[3],
        )
        if native is not None:
            return native
    return accel_j3(x_eci[:3], ctx.mu_km3_s2)


def j4_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    if env.get("_rust_numeric_backend") == "rust":
        from sim.dynamics.orbit.environment import EARTH_J2, EARTH_J3, EARTH_J4
        from sim.rust_environment_backend import try_zonal_acceleration

        native = try_zonal_acceleration(
            x_eci[:3], mu_km3_s2=ctx.mu_km3_s2, j2=EARTH_J2, j3=EARTH_J3, j4=EARTH_J4,
            radius_km=EARTH_RADIUS_KM, codes=[4],
        )
        if native is not None:
            return native
    return accel_j4(x_eci[:3], ctx.mu_km3_s2)


def _evaluate_spherical_harmonics_plugin(
    t_s: float,
    x_eci: np.ndarray,
    env: dict,
    ctx: OrbitContext,
    *,
    use_acceleration: bool,
) -> np.ndarray:
    """
    Generic spherical-harmonics perturbation plugin.

    Expects one of:
    1) `env["spherical_harmonics_terms"]` as list[dict], each with:
    - n: degree
    - m: order
    - c_nm (or c): cosine coefficient
    - s_nm (or s): sine coefficient (optional)
    - normalized: whether coefficients are fully normalized (optional; default False)

    2) Real-coefficient mode:
    - spherical_harmonics_use_real_coefficients: bool (True)
    - spherical_harmonics_model: e.g., "EGM96" (optional; default EGM96)
    - spherical_harmonics_coeff_path: local .gfc path (optional)
    - spherical_harmonics_max_degree: int (optional; default 8)
    - spherical_harmonics_max_order: int (optional; default max_degree)
    - spherical_harmonics_allow_download: bool (optional; default True)

    Optional env fields:
    - spherical_harmonics_fd_step_km
    """
    terms = env.get("_parsed_spherical_harmonics_terms")
    if terms is None:
        terms = parse_spherical_harmonic_terms(env.get("spherical_harmonics_terms"))
        if terms:
            env["_parsed_spherical_harmonics_terms"] = terms
    if not terms and bool(env.get("spherical_harmonics_use_real_coefficients", False)):
        n_max = int(env.get("spherical_harmonics_max_degree", 8))
        m_max = int(env.get("spherical_harmonics_max_order", n_max))
        model = str(env.get("spherical_harmonics_model", "EGM96"))
        coeff_path = env.get("spherical_harmonics_coeff_path")
        allow_download = bool(env.get("spherical_harmonics_allow_download", True))
        cache_key = (
            n_max,
            m_max,
            model,
            None if coeff_path is None else str(coeff_path),
            allow_download,
        )
        cached_terms = env.get("_real_spherical_harmonics_cache")
        if cached_terms is None or cached_terms[0] != cache_key:
            terms = load_real_earth_gravity_terms(
                max_degree=n_max,
                max_order=m_max,
                model=model,
                coeff_path=None if coeff_path is None else str(coeff_path),
                allow_download=allow_download,
            )
            env["_real_spherical_harmonics_cache"] = (cache_key, terms)
        else:
            terms = cached_terms[1]
    if not terms:
        return np.zeros(3)
    compiled = env.get("_compiled_spherical_harmonics_terms")
    if compiled is None:
        compiled = compile_spherical_harmonic_terms(terms)
        if compiled is not None:
            env["_compiled_spherical_harmonics_terms"] = compiled
    fd_step_km = float(env.get("spherical_harmonics_fd_step_km", 1e-3))
    jd_utc_start = env.get("jd_utc_start")
    re_km = float(env.get("spherical_harmonics_reference_radius_km", EARTH_RADIUS_KM))
    frame_model = str(env.get("spherical_harmonics_frame_model", "simple"))
    eop_path = env.get("spherical_harmonics_eop_path")
    dut1_s = env.get("dut1_s")
    xp_arcsec = env.get("xp_arcsec")
    yp_arcsec = env.get("yp_arcsec")
    dat_s = env.get("dat_s")
    tt_minus_utc_s = env.get("tt_minus_utc_s")
    ddpsi_rad = float(env.get("ddpsi_rad", 0.0) or 0.0)
    ddeps_rad = float(env.get("ddeps_rad", 0.0) or 0.0)
    if jd_utc_start is None and "jd_utc" in env:
        jd_utc_start = float(env["jd_utc"]) - float(t_s) / 86400.0
    return accel_spherical_harmonics_terms(
        r_eci_km=x_eci[:3],
        t_s=t_s,
        terms=terms,
        mu_km3_s2=float(env.get("spherical_harmonics_mu_km3_s2", ctx.mu_km3_s2)),
        re_km=re_km,
        fd_step_km=fd_step_km,
        jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
        frame_model=frame_model,
        eop_path=None if eop_path is None else str(eop_path),
        dut1_s=None if dut1_s is None else float(dut1_s),
        xp_arcsec=None if xp_arcsec is None else float(xp_arcsec),
        yp_arcsec=None if yp_arcsec is None else float(yp_arcsec),
        dat_s=None if dat_s is None else float(dat_s),
        tt_minus_utc_s=None if tt_minus_utc_s is None else float(tt_minus_utc_s),
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
        eop_extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
        compiled=compiled,
        use_acceleration=use_acceleration,
    )


def spherical_harmonics_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    """Evaluate spherical-harmonic perturbations with the Python reference path."""

    return _evaluate_spherical_harmonics_plugin(
        t_s,
        x_eci,
        env,
        ctx,
        use_acceleration=False,
    )


def _accelerated_spherical_harmonics_plugin(
    t_s: float,
    x_eci: np.ndarray,
    env: dict,
    ctx: OrbitContext,
) -> np.ndarray:
    return _evaluate_spherical_harmonics_plugin(
        t_s,
        x_eci,
        env,
        ctx,
        use_acceleration=True,
    )


def drag_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    density = env.get("density_kg_m3")
    if density is None:
        if env.get("atmosphere_model") in (None, ""):
            raise ValueError(
                "Drag requires an explicit environment.atmosphere_model or density_kg_m3; "
                "no orbital-decay atmosphere is selected implicitly."
            )
        atmo_model = str(env.get("atmosphere_model")).lower()
        density = density_from_model(
            atmo_model,
            x_eci[:3],
            t_s,
            env=env,
        )
    omega_raw = env.get("drag_earth_rotation_rad_s")
    return accel_drag_resolved(
        r_eci_km=x_eci[:3],
        v_eci_km_s=x_eci[3:],
        t_s=t_s,
        mass_kg=ctx.mass_kg,
        cd=float(env.get("drag_coefficient", ctx.cd)),
        density_kg_m3=float(density),
        area_eff_m2=float(env.get("drag_area_m2", ctx.area_m2)),
        drag_frame_model=str(env.get("drag_frame_model", "simple")).strip().lower(),
        jd_utc_start=(None if env.get("jd_utc_start") is None else float(env.get("jd_utc_start"))),
        drag_eop_path=(None if env.get("drag_eop_path") is None else str(env.get("drag_eop_path"))),
        omega_earth_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
        dut1_s=None if env.get("dut1_s") is None else float(env["dut1_s"]),
        xp_arcsec=None if env.get("xp_arcsec") is None else float(env["xp_arcsec"]),
        yp_arcsec=None if env.get("yp_arcsec") is None else float(env["yp_arcsec"]),
        dat_s=None if env.get("dat_s") is None else float(env["dat_s"]),
        tt_minus_utc_s=None if env.get("tt_minus_utc_s") is None else float(env["tt_minus_utc_s"]),
        ddpsi_rad=float(env.get("ddpsi_rad", 0.0) or 0.0),
        ddeps_rad=float(env.get("ddeps_rad", 0.0) or 0.0),
        eop_extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
    )


def lift_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    cl = float(env.get("lift_coefficient", env.get("cl", 0.0)) or 0.0)
    lift_direction = env.get("lift_direction_eci")
    if lift_direction is None or cl == 0.0:
        return np.zeros(3)
    density = env.get("density_kg_m3")
    if density is None:
        if env.get("atmosphere_model") in (None, ""):
            raise ValueError("Lift requires an explicit environment.atmosphere_model or density_kg_m3.")
        atmo_model = str(env.get("atmosphere_model")).lower()
        density = density_from_model(
            atmo_model,
            x_eci[:3],
            t_s,
            env=env,
        )
    return accel_lift(
        x_eci[:3],
        x_eci[3:],
        t_s,
        ctx.mass_kg,
        ctx.area_m2,
        cl,
        np.array(lift_direction, dtype=float).reshape(3),
        {
            "density_kg_m3": density,
            "lift_area_m2": env.get("lift_area_m2", env.get("drag_area_m2", ctx.area_m2)),
            "jd_utc_start": env.get("jd_utc_start"),
            "drag_frame_model": env.get("drag_frame_model", "inertial_z"),
            "drag_eop_path": env.get("drag_eop_path"),
            "drag_earth_rotation_rad_s": env.get("drag_earth_rotation_rad_s"),
            "dut1_s": env.get("dut1_s"),
            "xp_arcsec": env.get("xp_arcsec"),
            "yp_arcsec": env.get("yp_arcsec"),
            "dat_s": env.get("dat_s"),
            "tt_minus_utc_s": env.get("tt_minus_utc_s"),
            "ddpsi_rad": env.get("ddpsi_rad"),
            "ddeps_rad": env.get("ddeps_rad"),
            "eop_extrapolation": env.get("eop_extrapolation", "error"),
        },
    )


def srp_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    sun_position = env.get("sun_pos_eci_km")
    ellipsoid_shadow = str(env.get("srp_shadow_model", "")).lower() == "conical_ellipsoid"
    if env.get("_rust_numeric_backend") == "rust" and sun_position is not None and not ellipsoid_shadow:
        shadow_name = str(env.get("srp_shadow_model", "conical")).lower()
        shadow_model = 0 if shadow_name in ("none", "off", "disabled") else 1 if shadow_name in ("cylindrical", "cylinder") else 2
        from sim.rust_environment_backend import try_srp_acceleration

        native = try_srp_acceleration(
            x_eci[:3], np.asarray(sun_position, dtype=float).reshape(3),
            mass_kg=ctx.mass_kg, area_m2=float(env.get("srp_area_m2", ctx.area_m2)),
            reflectivity=ctx.cr, pressure_pa=srp_pressure_n_m2(env), au_km=AU_KM,
            earth_radius_km=EARTH_RADIUS_KM, sun_radius_km=SUN_RADIUS_KM, shadow_model=shadow_model,
        )
        if native is not None:
            return native
    if acceleration_enabled_from_mode() and sun_position is not None and not ellipsoid_shadow:
        shadow_name = str(env.get("srp_shadow_model", "conical")).lower()
        shadow_model = (
            0 if shadow_name in ("none", "off", "disabled") else 1 if shadow_name in ("cylindrical", "cylinder") else 2
        )
        return _compiled_srp_acceleration()(
            x_eci[:3],
            np.asarray(sun_position, dtype=float).reshape(3),
            float(ctx.mass_kg),
            float(env.get("srp_area_m2", ctx.area_m2)),
            float(ctx.cr),
            srp_pressure_n_m2(env),
            float(AU_KM),
            float(EARTH_RADIUS_KM),
            float(SUN_RADIUS_KM),
            shadow_model,
        )
    srp_geometry = resolve_srp_geometry(x_eci[:3], t_s, env)
    shadow = srp_shadow_factor(
        r_sc_eci_km=x_eci[:3],
        t_s=t_s,
        env=env,
        srp_geometry=srp_geometry,
    )
    return accel_srp_resolved(
        sun_dir_eci=srp_geometry["sun_dir_sc_eci"],
        mass_kg=ctx.mass_kg,
        area_eff_m2=float(env.get("srp_area_m2", ctx.area_m2)),
        cr=ctx.cr,
        distance_scale=float(srp_geometry["distance_scale"]),
        shadow_factor=shadow,
        pressure_n_m2=srp_pressure_n_m2(env),
    )


@lru_cache(maxsize=1)
def _compiled_srp_acceleration():
    from sim.acceleration.kernels.srp import srp_acceleration_kernel

    return srp_acceleration_kernel


def third_body_moon_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    moon = env.get("moon_pos_eci_km")
    if moon is None:
        _, moon = resolve_sun_moon_positions(env, t_s)
    if env.get("_rust_numeric_backend") == "rust":
        from sim.rust_environment_backend import try_third_body_acceleration

        native = try_third_body_acceleration(x_eci[:3], moon, mu_km3_s2=MOON_MU_KM3_S2)
        if native is not None:
            return native
    return accel_third_body(x_eci[:3], moon, MOON_MU_KM3_S2)


def third_body_sun_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    sun = env.get("sun_pos_eci_km")
    if sun is None:
        sun, _ = resolve_sun_moon_positions(env, t_s)
    if env.get("_rust_numeric_backend") == "rust":
        from sim.rust_environment_backend import try_third_body_acceleration

        native = try_third_body_acceleration(x_eci[:3], sun, mu_km3_s2=SUN_MU_KM3_S2)
        if native is not None:
            return native
    return accel_third_body(x_eci[:3], sun, SUN_MU_KM3_S2)


def third_body_planets_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    selected = env.get("third_body_planets", [])
    if isinstance(selected, str):
        selected_names = [selected.strip().lower()]
    else:
        selected_names = [str(v).strip().lower() for v in selected]
    if any(v in ("all", "*") for v in selected_names):
        selected_names = list(PLANETARY_MU_KM3_S2.keys())

    acc = np.zeros(3)
    for name in selected_names:
        if name not in PLANETARY_MU_KM3_S2:
            continue
        pos = resolve_body_position_eci_km(name, env=env, t_s=t_s)
        mu = float(env.get(f"{name}_mu_km3_s2", PLANETARY_MU_KM3_S2[name]))
        acc += accel_third_body(x_eci[:3], pos, mu)
    return acc


@dataclass
class OrbitPropagator:
    model: str = "two_body"
    cr3bp_system_name: str = "earth_moon"
    integrator: str = "rk4"
    plugins: list[AccelerationPlugin] = field(default_factory=list)
    adaptive_atol: float = 1e-9
    adaptive_rtol: float = 1e-7
    acceleration_mode: str = "off"
    numeric_backend: str = "rust"
    last_numeric_path: str | None = field(default=None, init=False)
    _rkf78_h_next: float | None = field(default=None, init=False, repr=False)
    _rkf78_last_t_s: float | None = field(default=None, init=False, repr=False)
    _acceleration_enabled_cache: bool | None = field(default=None, init=False, repr=False)
    _zonal_rk4_fast_path_checked: bool = field(default=False, init=False, repr=False)
    _zonal_rk4_fast_path_flags_cache: tuple[bool, bool, bool] | None = field(default=None, init=False, repr=False)
    _builtin_rk4_fast_path_checked: bool = field(default=False, init=False, repr=False)
    _builtin_rk4_fast_path_enabled_cache: bool = field(default=False, init=False, repr=False)
    _builtin_needs_time_env_cache: bool = field(default=False, init=False, repr=False)
    _builtin_rk4_k1: np.ndarray = field(default_factory=lambda: np.empty(6, dtype=float), init=False, repr=False)
    _builtin_rk4_k2: np.ndarray = field(default_factory=lambda: np.empty(6, dtype=float), init=False, repr=False)
    _builtin_rk4_k3: np.ndarray = field(default_factory=lambda: np.empty(6, dtype=float), init=False, repr=False)
    _builtin_rk4_k4: np.ndarray = field(default_factory=lambda: np.empty(6, dtype=float), init=False, repr=False)
    _builtin_rk4_stage: np.ndarray = field(default_factory=lambda: np.empty(6, dtype=float), init=False, repr=False)
    _builtin_rk4_accel: np.ndarray = field(default_factory=lambda: np.empty(3, dtype=float), init=False, repr=False)
    _builtin_time_env_cache: dict[tuple, dict[str, np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    _compiled_force_codes_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _compiled_force_plugins_cache: tuple[AccelerationPlugin, ...] | None = field(default=None, init=False, repr=False)
    _staged_force_plan_cache_key: tuple | None = field(default=None, init=False, repr=False)
    # Native extension objects are process-local and intentionally excluded
    # from OrbitPropagator serialization; they are recreated on first use in
    # the worker that receives the pickled propagator.
    _rust_force_context_cache: tuple | None = field(default=None, init=False, repr=False)
    _rust_environment_context_cache: tuple | None = field(default=None, init=False, repr=False)
    _staged_planet_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((len(PLANETARY_MU_KM3_S2), 3), dtype=float),
        init=False,
        repr=False,
    )
    _staged_planet_mu: np.ndarray = field(
        default_factory=lambda: np.zeros(len(PLANETARY_MU_KM3_S2), dtype=float),
        init=False,
        repr=False,
    )
    _compiled_harmonic_rotations: np.ndarray = field(
        default_factory=lambda: np.empty((3, 3, 3), dtype=float), init=False, repr=False
    )
    _compiled_density_rotations: np.ndarray = field(
        default_factory=lambda: np.empty((3, 3, 3), dtype=float), init=False, repr=False
    )
    _compiled_drag_rotations: np.ndarray = field(
        default_factory=lambda: np.empty((3, 3, 3), dtype=float), init=False, repr=False
    )
    _compiled_sun_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=float), init=False, repr=False
    )
    _compiled_moon_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=float), init=False, repr=False
    )
    _compiled_atmosphere_inputs: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 6), dtype=float), init=False, repr=False
    )
    _compiled_endpoint_cache_key: tuple | None = field(default=None, init=False, repr=False)
    _compiled_endpoint_harmonic_rotation: np.ndarray = field(
        default_factory=lambda: np.empty((3, 3), dtype=float), init=False, repr=False
    )
    _compiled_endpoint_density_rotation: np.ndarray = field(
        default_factory=lambda: np.empty((3, 3), dtype=float), init=False, repr=False
    )
    _compiled_endpoint_drag_rotation: np.ndarray = field(
        default_factory=lambda: np.empty((3, 3), dtype=float), init=False, repr=False
    )
    _compiled_endpoint_sun_position: np.ndarray = field(
        default_factory=lambda: np.empty(3, dtype=float), init=False, repr=False
    )
    _compiled_endpoint_moon_position: np.ndarray = field(
        default_factory=lambda: np.empty(3, dtype=float), init=False, repr=False
    )
    _compiled_endpoint_atmosphere_inputs: np.ndarray = field(
        default_factory=lambda: np.empty(6, dtype=float), init=False, repr=False
    )
    _compiled_scalar_parameters: np.ndarray = field(
        default_factory=lambda: np.empty(15, dtype=float), init=False, repr=False
    )
    last_adaptive_step_info: AdaptiveStepInfo | None = field(default=None, init=False, repr=False)
    adaptive_step_info: AdaptiveStepInfo | None = field(default=None, init=False, repr=False)

    def __getstate__(self) -> dict:
        """Serialize the Python reference state while dropping native caches.

        PyO3 context instances retain Rust allocations and are not a stable
        cross-process serialization contract.  Resource selection and table
        validation remain in the Python state, so each process can recreate
        an equivalent context lazily after unpickling.
        """

        state = dict(self.__dict__)
        state.pop("_rust_force_context_cache", None)
        state.pop("_rust_environment_context_cache", None)
        state.pop("_rust_mechanical_context_cache", None)
        state.pop("_rust_prepared_frame_cache", None)
        state.pop("_rust_forecast_zonal_context", None)
        state.pop("_rust_stage_context_cache", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._rust_force_context_cache = None
        self._rust_environment_context_cache = None
        self._rust_mechanical_context_cache = None
        self._rust_prepared_frame_cache = None
        self._rust_forecast_zonal_context = None
        self._rust_stage_context_cache = None

    @property
    def state_frame(self) -> str:
        """Frame carried by the six-component numerical state."""

        return "cr3bp_rotating" if str(self.model or "two_body").strip().lower() == "cr3bp" else "eci"

    def propagation_metadata(self) -> dict[str, str]:
        """Return frame-aware metadata for numerical propagation evidence."""

        frame = self.state_frame
        return {
            "propagation_method": "special",
            "propagator_family": "CR3BP" if frame == "cr3bp_rotating" else "ONP",
            "propagator_name": (
                f"{self.cr3bp_system_name} CR3BP" if frame == "cr3bp_rotating" else "OEL Numerical Propagator"
            ),
            "general_model": "",
            "native_frame": frame,
            "output_frame": frame,
            "state_history_frame": frame,
            "frame_transform": "native",
            "command_acceleration_frame": frame,
        }

    def acceleration_at(
        self,
        *,
        t_s: float,
        x_eci: np.ndarray,
        command_accel_eci_km_s2: np.ndarray,
        env: dict,
        ctx: OrbitContext,
    ) -> np.ndarray:
        """Evaluate the ECI acceleration for one coupled-integrator stage."""

        if self.state_frame != "eci":
            raise ValueError("stage acceleration is available only for ECI ONP propagation")
        state = np.asarray(x_eci, dtype=float).reshape(6)
        acceleration = accel_two_body(state[:3], ctx.mu_km3_s2) + np.asarray(
            command_accel_eci_km_s2, dtype=float
        ).reshape(3)
        accelerate_spherical_harmonics = self._acceleration_enabled() and spherical_harmonics_plugin in self.plugins
        plugin_env = env
        if str(self.numeric_backend).strip().lower() == "rust":
            # The plugin itself decides whether its prepared numeric kernel is
            # available.  Keeping the marker local preserves custom Python
            # plugins and the default Python reference path.
            plugin_env = dict(env)
            plugin_env["_rust_numeric_backend"] = "rust"
        for plugin in self.plugins:
            if accelerate_spherical_harmonics and plugin is spherical_harmonics_plugin:
                acceleration += _accelerated_spherical_harmonics_plugin(t_s, state, plugin_env, ctx)
            else:
                acceleration += plugin(t_s, state, plugin_env, ctx)
        return acceleration

    def try_propagate_fixed_steps(
        self, x_eci: np.ndarray, dt_s: float, steps: int, t_s: float,
        command_accel_eci_km_s2: np.ndarray, env: dict, ctx: OrbitContext,
        *, sample_stride: int = 0,
    ) -> np.ndarray | None:
        """Advance a constant-command native segment, or leave ordinary stepping in charge.

        This numerical API has no controller, event or scenario lifecycle.
        Callers must own the interval and visit their usual lifecycle boundaries.
        Zero stride retains only the final state; positive stride includes the
        initial state, requested samples and the final state.
        """
        if self.numeric_backend != "rust" or self.integrator != "rk4" or self.model != "two_body" or self.state_frame != "eci":
            return None
        from sim.dynamics.orbit.rust_force_plan import force_codes, make_plan
        from sim.rust_orbit_backend import _command, _state, native_harmonic_degree_limit

        if force_codes(self, env) is None:
            return None
        if spherical_harmonics_plugin in self.plugins:
            if env.get("_compiled_spherical_harmonics_terms") is None:
                spherical_harmonics_plugin(t_s, np.asarray(x_eci, dtype=float), env, ctx)
            harmonic = env.get("_compiled_spherical_harmonics_terms")
            if harmonic is None or not harmonic.all_normalized or harmonic.n_max > native_harmonic_degree_limit():
                return None
        *_, stage_callback, force_context = make_plan(self, x_eci, t_s, env, ctx)
        function = getattr(force_context, "fixed_history", None)
        if function is None:
            return None
        raw = function(_state(x_eci), float(t_s), float(dt_s), steps, sample_stride,
                       _command(command_accel_eci_km_s2), stage_callback)
        self.last_numeric_path = "rust_native_force_plan_history"
        return np.frombuffer(raw, dtype="<f8").copy().reshape(-1, 6)

    def try_propagate_sampled_steps(
        self, x_eci: np.ndarray, step_widths, sample_ends, t_s: float,
        command_accel_eci_km_s2: np.ndarray, env: dict, ctx: OrbitContext,
        *, step_times=None,
    ) -> np.ndarray | None:
        """Visit an exact caller-owned RK4 schedule with ordinary stage forces.

        Callers own every controller/event/output boundary. Explicit substep
        start times preserve nested decimal simulation-clock semantics; without
        them this numerical API advances one cumulative local clock.
        """
        if self.numeric_backend != "rust" or self.integrator != "rk4" or self.model != "two_body" or self.state_frame != "eci":
            return None
        from sim.dynamics.orbit.rust_force_plan import force_codes, make_plan
        from sim.rust_orbit_backend import _command, _state, native_harmonic_degree_limit

        if force_codes(self, env) is None:
            return None
        if spherical_harmonics_plugin in self.plugins:
            if env.get("_compiled_spherical_harmonics_terms") is None:
                spherical_harmonics_plugin(t_s, np.asarray(x_eci, dtype=float), env, ctx)
            harmonic = env.get("_compiled_spherical_harmonics_terms")
            if harmonic is None or not harmonic.all_normalized or harmonic.n_max > native_harmonic_degree_limit():
                return None
        *_, stage_callback, force_context = make_plan(self, x_eci, t_s, env, ctx)
        function = getattr(force_context, "sampled_history", None)
        if not callable(function):
            return None
        widths = np.asarray(step_widths, dtype="<f8").reshape(-1)
        if step_times is None:
            times = np.empty(widths.size, dtype="<f8")
            clock = float(t_s)
            for index, width in enumerate(widths):
                times[index] = clock
                clock += float(width)
        else:
            times = np.asarray(step_times, dtype="<f8").reshape(-1)
        ends = [int(value) for value in sample_ends]
        from sim.dynamics.orbit.rust_precision_force_plan import immutable_sampled_stages
        stage_callback = immutable_sampled_stages(self, stage_callback, np.asarray(x_eci,dtype=float), times, widths, ends)
        density_callback = getattr(stage_callback, "_simple_drag_density_callback", None)
        if (callable(density_callback) and not getattr(stage_callback, "native_preparation", False)
                and callable(getattr(force_context, "supports_simple_drag_stages", None))):
            raw = function(_state(x_eci), times.tobytes(), widths.tobytes(), ends,
                           _command(command_accel_eci_km_s2), density_callback, True,
                           getattr(stage_callback, "_simple_drag_jd_utc_start", None))
        else:
            raw = function(_state(x_eci), times.tobytes(), widths.tobytes(), ends,
                           _command(command_accel_eci_km_s2), stage_callback)
        self.last_numeric_path = "rust_native_force_plan_sampled_history"
        return np.frombuffer(raw, dtype="<f8").copy().reshape(len(ends) + 1, 6)

    def try_propagate_adaptive_sampled_steps(
        self, x_eci, step_widths, sample_ends, t_s, command_accel_eci_km_s2,
        env, ctx, *, step_times,
    ):
        """Prepare caller-owned full-force adaptive samples without advancing policy.

        Return states and per-interval diagnostics. The caller commits adaptive
        state only as its ordinary lifecycle consumes each interval.
        """
        if (self.numeric_backend != "rust" or self.integrator not in {"rkf78", "adaptive", "dopri5"}
                or self.model != "two_body" or self.state_frame != "eci"):
            return None
        from sim.dynamics.orbit.rust_force_plan import immutable_builtin_plan_signature, make_plan
        from sim.rust_orbit_backend import _command, _state

        if immutable_builtin_plan_signature(self, env, ctx) is None:
            return None
        *_, stage_callback, force_context = make_plan(self, x_eci, t_s, env, ctx)
        function = getattr(force_context, "sampled_adaptive_history", None)
        if not callable(function):
            return None
        h_init = self._rkf78_h_next
        if self._rkf78_last_t_s is None or float(t_s) < self._rkf78_last_t_s - 1.0e-12:
            h_init = None
        widths = np.asarray(step_widths, dtype="<f8").reshape(-1)
        times = np.asarray(step_times, dtype="<f8").reshape(-1)
        ends = [int(value) for value in sample_ends]
        raw, infos = function(
            _state(x_eci), times.tobytes(), widths.tobytes(), ends,
            _command(command_accel_eci_km_s2), float(self.adaptive_atol), float(self.adaptive_rtol),
            h_init, "dopri5" if self.integrator == "dopri5" else "rkf78", stage_callback,
        )
        return (np.frombuffer(raw, dtype="<f8").copy().reshape(len(ends) + 1, 6),
                np.frombuffer(infos, dtype="<f8").copy().reshape(len(widths), 8))

    def propagate(
        self,
        x_eci: np.ndarray,
        dt_s: float,
        t_s: float,
        command_accel_eci_km_s2: np.ndarray,
        env: dict,
        ctx: OrbitContext,
    ) -> np.ndarray:
        backend = str(self.numeric_backend).strip().lower()
        if backend not in {"python", "rust"}:
            raise ValueError(f"Unknown orbit numeric backend {self.numeric_backend!r}")
        if str(self.model or "two_body").strip().lower() == "cr3bp":
            self.last_numeric_path = "rust_native_cr3bp" if backend == "rust" else "python"
            if self._rkf78_last_t_s is None or float(t_s) < float(self._rkf78_last_t_s) - 1e-12:
                self._rkf78_h_next = None
            x_next, step_info = propagate_cr3bp_state(
                x_eci,
                dt_s,
                t_s,
                command_accel_eci_km_s2,
                system=cr3bp_system(self.cr3bp_system_name),
                integrator=self.integrator,
                adaptive_atol=self.adaptive_atol,
                adaptive_rtol=self.adaptive_rtol,
                h_init=self._rkf78_h_next,
                return_info=True,
                numeric_backend=backend,
            )
            self.last_adaptive_step_info = step_info
            if step_info is not None:
                self._rkf78_h_next = step_info.suggested_next_step_s
                self._rkf78_last_t_s = float(t_s + dt_s)
                previous = [] if self.adaptive_step_info is None else [self.adaptive_step_info]
                self.adaptive_step_info = combine_adaptive_step_info(step_info.method, [*previous, step_info])
            return x_next

        if backend == "rust":
            if str(self.model or "two_body").strip().lower() != "two_body":
                raise ValueError("Rust orbit backend supports only two-body ONP")
            codes = ((j2_plugin, 2), (j3_plugin, 3), (j4_plugin, 4))
            zonal_only = all(any(plugin is known for known, _ in codes) for plugin in self.plugins)
            force_codes = (
                [next(code for known, code in codes if plugin is known) for plugin in self.plugins]
                if zonal_only else []
            )
            from sim.dynamics.orbit.rust_force_plan import force_codes as native_force_codes
            native_plan = native_force_codes(self, env) is not None
            if native_plan and any(plugin is spherical_harmonics_plugin for plugin in self.plugins):
                from sim.rust_orbit_backend import native_harmonic_degree_limit

                if env.get("_compiled_spherical_harmonics_terms") is None:
                    spherical_harmonics_plugin(t_s, np.asarray(x_eci, dtype=float), env, ctx)
                harmonic_terms = env.get("_compiled_spherical_harmonics_terms")
                native_plan = bool(
                    harmonic_terms is not None
                    and harmonic_terms.all_normalized
                    and harmonic_terms.n_max <= native_harmonic_degree_limit()
                    and harmonic_terms.m_max <= native_harmonic_degree_limit()
                )
            integrator = str(self.integrator).strip().lower()

            def acceleration_callback(stage_t, stage_x):
                return self.acceleration_at(
                    t_s=float(stage_t), x_eci=np.asarray(stage_x, dtype=float),
                    command_accel_eci_km_s2=command_accel_eci_km_s2,
                    env=env, ctx=ctx,
                ).tolist()

            if integrator == "rk4":
                if not zonal_only:
                    if native_plan:
                        from sim.dynamics.orbit.rust_force_plan import make_plan
                        from sim.rust_orbit_backend import rk4_force_plan_eci

                        plan_codes, scalars, shadow, dims, tables, stage_callback, force_context = make_plan(
                            self, x_eci, t_s, env, ctx,
                        )
                        self.last_numeric_path = "rust_native_force_plan"
                        return rk4_force_plan_eci(
                            x_eci, t_s, dt_s, codes=plan_codes, scalars=scalars,
                            shadow_model=shadow, harmonic_dims=dims, tables=tables,
                            command_accel_eci_km_s2=command_accel_eci_km_s2,
                            stage_callback=stage_callback, force_context=force_context,
                        )
                    from sim.rust_orbit_backend import rk4_callback_eci

                    self.last_numeric_path = "rust_python_force_callback"
                    return rk4_callback_eci(x_eci, t_s, dt_s, acceleration_callback)
                if force_codes in ([], [2]):
                    from sim.rust_orbit_backend import rk4_step_eci

                    self.last_numeric_path = "rust_native_zonal"
                    return rk4_step_eci(
                        x_eci,
                        dt_s,
                        ctx.mu_km3_s2,
                        include_j2=bool(force_codes),
                        command_accel_eci_km_s2=command_accel_eci_km_s2,
                    )
                from sim.rust_orbit_backend import rk4_zonal_step_eci

                self.last_numeric_path = "rust_native_zonal"
                return rk4_zonal_step_eci(
                    x_eci, dt_s, ctx.mu_km3_s2,
                    force_codes=force_codes,
                    command_accel_eci_km_s2=command_accel_eci_km_s2,
                )
            if integrator in ("rkf78", "adaptive", "dopri5"):
                from sim.rust_orbit_backend import (
                    adaptive_callback_eci,
                    dopri5_force_plan_eci,
                    rkf78_force_plan_eci,
                    rkf78_zonal_step_eci,
                )

                if self._rkf78_last_t_s is None or float(t_s) < float(self._rkf78_last_t_s) - 1e-12:
                    self._rkf78_h_next = None
                method = "dopri5" if integrator == "dopri5" else "rkf78"
                if zonal_only and method == "rkf78":
                    self.last_numeric_path = "rust_native_zonal"
                    x_next, info = rkf78_zonal_step_eci(
                        x_eci, t_s, dt_s, ctx.mu_km3_s2,
                        force_codes=force_codes,
                        command_accel_eci_km_s2=command_accel_eci_km_s2,
                        atol=self.adaptive_atol,
                        rtol=self.adaptive_rtol,
                        h_init=self._rkf78_h_next,
                    )
                elif native_plan:
                    from sim.dynamics.orbit.rust_force_plan import make_plan

                    self.last_numeric_path = "rust_native_force_plan"
                    plan_codes, scalars, shadow, dims, tables, stage_callback, force_context = make_plan(
                        self, x_eci, t_s, env, ctx,
                    )
                    force_step = dopri5_force_plan_eci if method == "dopri5" else rkf78_force_plan_eci
                    x_next, info = force_step(
                        x_eci, t_s, dt_s, codes=plan_codes, scalars=scalars,
                        shadow_model=shadow, harmonic_dims=dims, tables=tables,
                        command_accel_eci_km_s2=command_accel_eci_km_s2,
                        atol=self.adaptive_atol, rtol=self.adaptive_rtol,
                        h_init=self._rkf78_h_next, stage_callback=stage_callback,
                        force_context=force_context,
                    )
                else:
                    self.last_numeric_path = "rust_python_force_callback"
                    x_next, info = adaptive_callback_eci(
                        x_eci, t_s, dt_s, atol=self.adaptive_atol,
                        rtol=self.adaptive_rtol, h_init=self._rkf78_h_next,
                        method=method, acceleration_callback=acceleration_callback,
                    )
                self._rkf78_h_next = info.suggested_next_step_s
                self._rkf78_last_t_s = float(t_s + dt_s)
                self.last_adaptive_step_info = info
                previous = [] if self.adaptive_step_info is None else [self.adaptive_step_info]
                self.adaptive_step_info = combine_adaptive_step_info(method, [*previous, info])
                return x_next
            raise ValueError("Rust orbit backend supports only RK4, RKF78, and DOPRI5")
        self.last_numeric_path = "python"

        fast_flags = self._zonal_rk4_fast_path_flags()
        acceleration_enabled = self._acceleration_enabled()
        if acceleration_enabled and fast_flags is not None:
            include_j2, include_j3, include_j4 = fast_flags
            return _load_rk4_zonal_step_state()(
                np.asarray(x_eci, dtype=float).reshape(6),
                float(dt_s),
                np.asarray(command_accel_eci_km_s2, dtype=float).reshape(3),
                float(ctx.mu_km3_s2),
                include_j2,
                include_j3,
                include_j4,
            )
        compiled_result = self._try_propagate_compiled_builtin_rk4(
            x_eci=x_eci,
            dt_s=dt_s,
            t_s=t_s,
            command_accel_eci_km_s2=command_accel_eci_km_s2,
            env=env,
            ctx=ctx,
        )
        if compiled_result is not None:
            return compiled_result
        staged_result = self._try_propagate_staged_compiled(
            x_eci=x_eci,
            dt_s=dt_s,
            t_s=t_s,
            command_accel_eci_km_s2=command_accel_eci_km_s2,
            env=env,
            ctx=ctx,
        )
        if staged_result is not None:
            return staged_result
        if self._builtin_rk4_fast_path_enabled():
            return self._propagate_builtin_rk4(
                x_eci=x_eci,
                dt_s=dt_s,
                t_s=t_s,
                command_accel_eci_km_s2=command_accel_eci_km_s2,
                env=env,
                ctx=ctx,
            )

        accelerate_spherical_harmonics = acceleration_enabled and spherical_harmonics_plugin in self.plugins

        def deriv(t_local: float, x_local: np.ndarray) -> np.ndarray:
            dx = np.empty(6, dtype=float)
            dx[:3] = x_local[3:]
            a = accel_two_body(x_local[:3], ctx.mu_km3_s2) + command_accel_eci_km_s2
            for plugin in self.plugins:
                if accelerate_spherical_harmonics and plugin is spherical_harmonics_plugin:
                    a += _accelerated_spherical_harmonics_plugin(t_local, x_local, env, ctx)
                else:
                    a += plugin(t_local, x_local, env, ctx)
            dx[3:] = a
            return dx

        if self.integrator in ("rkf78", "dopri5", "adaptive"):
            adaptive_method = "rkf78" if self.integrator in ("rkf78", "adaptive") else "dopri5"
            if self._rkf78_last_t_s is None or float(t_s) < float(self._rkf78_last_t_s) - 1e-12:
                self._rkf78_h_next = None
            x_next, step_info = integrate_adaptive(
                deriv_fn=deriv,
                t_s=t_s,
                x=x_eci,
                dt_s=dt_s,
                atol=self.adaptive_atol,
                rtol=self.adaptive_rtol,
                method=adaptive_method,
                h_init=self._rkf78_h_next,
                return_info=True,
            )
            self._rkf78_h_next = step_info.suggested_next_step_s
            self._rkf78_last_t_s = float(t_s + dt_s)
            self.last_adaptive_step_info = step_info
            previous = [] if self.adaptive_step_info is None else [self.adaptive_step_info]
            self.adaptive_step_info = combine_adaptive_step_info(adaptive_method, [*previous, step_info])
            return x_next
        return rk4_step_state(deriv_fn=deriv, t_s=t_s, x=x_eci, dt_s=dt_s)

    def _acceleration_enabled(self) -> bool:
        if self._acceleration_enabled_cache is None:
            self._acceleration_enabled_cache = bool(acceleration_settings_from_mode(self.acceleration_mode).enabled)
        if self._acceleration_enabled_cache:
            _load_rk4_zonal_step_state()
        return bool(self._acceleration_enabled_cache)

    def _zonal_rk4_fast_path_flags(self) -> tuple[bool, bool, bool] | None:
        if self._zonal_rk4_fast_path_checked:
            return self._zonal_rk4_fast_path_flags_cache
        self._zonal_rk4_fast_path_checked = True
        if str(self.integrator).strip().lower() != "rk4":
            return None
        supported = (j2_plugin, j3_plugin, j4_plugin)
        if any(not any(plugin is known for known in supported) for plugin in self.plugins):
            return None
        self._zonal_rk4_fast_path_flags_cache = (
            j2_plugin in self.plugins,
            j3_plugin in self.plugins,
            j4_plugin in self.plugins,
        )
        return self._zonal_rk4_fast_path_flags_cache

    def _builtin_rk4_fast_path_enabled(self) -> bool:
        if self._builtin_rk4_fast_path_checked:
            return bool(self._builtin_rk4_fast_path_enabled_cache)
        self._builtin_rk4_fast_path_checked = True
        if str(self.integrator).strip().lower() != "rk4":
            self._builtin_rk4_fast_path_enabled_cache = False
            return False
        supported = {
            j2_plugin,
            j3_plugin,
            j4_plugin,
            spherical_harmonics_plugin,
            drag_plugin,
            lift_plugin,
            srp_plugin,
            third_body_moon_plugin,
            third_body_sun_plugin,
            third_body_planets_plugin,
        }
        if any(not any(plugin is known for known in supported) for plugin in self.plugins):
            self._builtin_rk4_fast_path_enabled_cache = False
            return False
        time_env_plugins = {
            srp_plugin,
            third_body_moon_plugin,
            third_body_sun_plugin,
            third_body_planets_plugin,
        }
        self._builtin_needs_time_env_cache = any(
            any(plugin is known for known in time_env_plugins) for plugin in self.plugins
        )
        self._builtin_rk4_fast_path_enabled_cache = True
        return True

    @staticmethod
    def _compiled_rotation(env: dict, t_s: float, *, model_key: str, path_key: str) -> np.ndarray:
        model = normalize_frame_model(env.get(model_key, "simple"))
        jd_utc_start = env.get("jd_utc_start")
        if model == FRAME_MODEL_IAU76_80_EOP:
            eop_path = env.get(path_key)
            return eci_to_ecef_rotation_hpop_like(
                float(t_s),
                jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
                eop_path=None if eop_path is None else str(eop_path),
                dut1_s=None if env.get("dut1_s") is None else float(env["dut1_s"]),
                xp_arcsec=None if env.get("xp_arcsec") is None else float(env["xp_arcsec"]),
                yp_arcsec=None if env.get("yp_arcsec") is None else float(env["yp_arcsec"]),
                dat_s=None if env.get("dat_s") is None else float(env["dat_s"]),
                tt_minus_utc_s=(None if env.get("tt_minus_utc_s") is None else float(env["tt_minus_utc_s"])),
                ddpsi_rad=float(env.get("ddpsi_rad", 0.0) or 0.0),
                ddeps_rad=float(env.get("ddeps_rad", 0.0) or 0.0),
                eop_extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
            )
        return eci_to_ecef_rotation(
            float(t_s),
            jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
        )

    @staticmethod
    def _accelerated_compiled_rotation(env: dict, t_s: float, *, model_key: str, path_key: str) -> np.ndarray:
        """Evaluate a numerically equivalent IAU frame on the accelerated path."""

        model = normalize_frame_model(env.get(model_key, "simple"))
        jd_utc_start = env.get("jd_utc_start")
        if model == FRAME_MODEL_IAU76_80_EOP:
            eop_path = env.get(path_key)
            has_eop_path = eop_path not in (None, "")
            if has_eop_path and jd_utc_start is not None:
                _validate_eop_elapsed_interval(float(jd_utc_start), t_s, str(eop_path))
            dut1_s = env.get("dut1_s")
            xp_arcsec = env.get("xp_arcsec")
            yp_arcsec = env.get("yp_arcsec")
            dat_s = env.get("dat_s")
            has_manual_eop = any(
                value is not None for value in (dut1_s, xp_arcsec, yp_arcsec, dat_s)
            ) or (
                float(env.get("ddpsi_rad", 0.0) or 0.0) != 0.0
                or float(env.get("ddeps_rad", 0.0) or 0.0) != 0.0
            )
            if jd_utc_start is not None and (has_eop_path or has_manual_eop):
                dp, de = 0.0, 0.0
                if has_eop_path:
                    jd_utc = float(jd_utc_start) + float(t_s) / 86400.0
                    xp_arcsec, yp_arcsec, dut1_s, dat_s = _interp_eop(
                        jd_utc - 2400000.5,
                        str(eop_path),
                        extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
                    )
                    dp, de = _interp_eop_nutation(
                        jd_utc - 2400000.5, str(eop_path),
                        extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
                    )
                else:
                    xp_arcsec = 0.0 if xp_arcsec is None else float(xp_arcsec)
                    yp_arcsec = 0.0 if yp_arcsec is None else float(yp_arcsec)
                    dut1_s = 0.0 if dut1_s is None else float(dut1_s)
                    if dat_s is None:
                        tt_minus_utc_s = env.get("tt_minus_utc_s")
                        dat_s = (
                            69.184 if tt_minus_utc_s is None else float(tt_minus_utc_s)
                        ) - 32.184
                    else:
                        dat_s = float(dat_s)
                nutation_coefficients, nutation_terms = _load_nut80_table()
                return _compiled_iau76_80_rotation()(
                    float(t_s),
                    float(jd_utc_start),
                    float(xp_arcsec),
                    float(yp_arcsec),
                    float(dut1_s),
                    float(dat_s),
                    float(env.get("ddpsi_rad", 0.0) or 0.0) + dp,
                    float(env.get("ddeps_rad", 0.0) or 0.0) + de,
                    nutation_coefficients,
                    nutation_terms,
                )
            return eci_to_ecef_rotation_hpop_like(
                float(t_s),
                jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
                eop_path=None if eop_path is None else str(eop_path),
                dut1_s=None if dut1_s is None else float(dut1_s),
                xp_arcsec=None if xp_arcsec is None else float(xp_arcsec),
                yp_arcsec=None if yp_arcsec is None else float(yp_arcsec),
                dat_s=None if dat_s is None else float(dat_s),
                tt_minus_utc_s=(None if env.get("tt_minus_utc_s") is None else float(env["tt_minus_utc_s"])),
                ddpsi_rad=float(env.get("ddpsi_rad", 0.0) or 0.0),
                ddeps_rad=float(env.get("ddeps_rad", 0.0) or 0.0),
                eop_extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
            )
        return eci_to_ecef_rotation(
            float(t_s),
            jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
        )

    @staticmethod
    def _compiled_rotation_key(env: dict, t_s: float, *, model_key: str, path_key: str) -> tuple:
        """Return the exact numeric request represented by a frame lookup."""

        model = normalize_frame_model(env.get(model_key, "simple"))
        jd_utc_start = env.get("jd_utc_start")
        if model != FRAME_MODEL_IAU76_80_EOP:
            return (
                model,
                float(t_s),
                None if jd_utc_start is None else float(jd_utc_start),
            )
        return (
            model,
            float(t_s),
            None if jd_utc_start is None else float(jd_utc_start),
            None if env.get(path_key) is None else str(env[path_key]),
            None if env.get("dut1_s") is None else float(env["dut1_s"]),
            None if env.get("xp_arcsec") is None else float(env["xp_arcsec"]),
            None if env.get("yp_arcsec") is None else float(env["yp_arcsec"]),
            None if env.get("dat_s") is None else float(env["dat_s"]),
            None if env.get("tt_minus_utc_s") is None else float(env["tt_minus_utc_s"]),
            float(env.get("ddpsi_rad", 0.0) or 0.0),
            float(env.get("ddeps_rad", 0.0) or 0.0),
            str(env.get("eop_extrapolation", "error") or "error"),
        )

    @classmethod
    def _compiled_rotation_definition_key(cls, env: dict, *, model_key: str, path_key: str) -> tuple:
        request = cls._compiled_rotation_key(env, 0.0, model_key=model_key, path_key=path_key)
        return request[:1] + request[2:]

    @staticmethod
    def _compiled_endpoint_environment_signature(env: dict, plugins: tuple) -> tuple | None:
        """Describe immutable time-only inputs eligible for cross-step reuse.

        Explicit ephemeris arrays and callables can be mutated between calls, so
        those configurations deliberately retain the established preparation
        path. Scalar configuration and quiet-Ap sequences are safe to compare.
        """

        unsafe_keys = (
            "ephemeris_callable",
            "ephemeris_body_callable",
            "spice_ephemeris_callable",
            "spice_body_ephemeris_callable",
            "nrlmsise00_density_callable",
            "sun_ephemeris_time_s",
            "sun_ephemeris_eci_km",
            "moon_ephemeris_time_s",
            "moon_ephemeris_eci_km",
            "sun_pos_eci_km",
            "moon_pos_eci_km",
            "sun_dir_eci",
        )
        if any(key in env for key in unsafe_keys):
            return None
        if str(env.get("ephemeris_mode", "analytic_enhanced")).strip().lower() in {
            "spice",
            "spiceypy",
        }:
            # SPICE positions depend on mutable kernel lists and several
            # target/frame settings. Recompute the shared endpoint rather than
            # risk reusing positions prepared for a different SPICE context.
            return None
        scalar_keys = (
            "jd_utc",
            "jd_utc_start",
            "atmo_epoch_utc",
            "spherical_harmonics_frame_model",
            "spherical_harmonics_eop_path",
            "drag_frame_model",
            "drag_eop_path",
            "density_frame_model",
            "density_eop_path",
            "dut1_s",
            "xp_arcsec",
            "yp_arcsec",
            "dat_s",
            "tt_minus_utc_s",
            "ddpsi_rad",
            "ddeps_rad",
            "eop_extrapolation",
            "atmosphere_model",
            "geodetic_model",
            "f107",
            "f107a",
            "ap",
            "nrlmsise00_f107",
            "nrlmsise00_f107a",
            "nrlmsise00_ap",
            "nrlmsise00_sw_path",
            "nrlmsise00_lst_hr",
            "msis_sw_path",
            "ephemeris_mode",
            "de440_coeff_path",
            "de440_eop_path",
            "de440_tai_utc_s",
        )
        values: list[object] = [plugins]
        for key in scalar_keys:
            value = env.get(key)
            if value is None or isinstance(value, (str, bool, int, float)):
                values.append(value)
            elif hasattr(value, "isoformat"):
                values.append(value.isoformat())
            else:
                return None
        ap_history = env.get("nrlmsise00_ap_a")
        if ap_history is None:
            values.append(None)
        elif isinstance(ap_history, (list, tuple)):
            values.append(tuple(float(value) for value in ap_history))
        else:
            return None
        # Solar geometry now follows the selected ephemeris and density
        # rotation. File-backed endpoint inputs must be refreshed after a
        # rewrite or deletion, just like their authoritative resource owners.
        from sim.dynamics.orbit.de440_hpop import _resource_signature, default_de440_coeff_path

        for key in (
            "density_eop_path", "drag_eop_path", "spherical_harmonics_eop_path",
            "de440_eop_path", "de440_coeff_path",
        ):
            path = env.get(key)
            if path not in (None, ""):
                values.append((key, _resource_signature(path)))
        if str(env.get("ephemeris_mode", "analytic_enhanced")).lower() in {
            "de440", "de440_hpop", "hpop_de440",
        } and env.get("de440_coeff_path") in (None, ""):
            values.append(("de440_coeff_path", _resource_signature(default_de440_coeff_path())))
        return tuple(values)

    def _try_propagate_compiled_builtin_rk4(
        self,
        *,
        x_eci: np.ndarray,
        dt_s: float,
        t_s: float,
        command_accel_eci_km_s2: np.ndarray,
        env: dict,
        ctx: OrbitContext,
    ) -> np.ndarray | None:
        if str(env.get("srp_shadow_model", "")).lower() == "conical_ellipsoid":
            return None  # This older fused kernel has spherical eclipse geometry.
        if not self._acceleration_enabled() or str(self.integrator).strip().lower() != "rk4":
            return None
        supported = {
            spherical_harmonics_plugin,
            drag_plugin,
            srp_plugin,
            third_body_sun_plugin,
            third_body_moon_plugin,
        }
        if not self.plugins or any(
            not any(plugin is known for known in supported) for plugin in self.plugins
        ):
            return None
        if drag_plugin in self.plugins:
            try:
                drag_frame = normalize_frame_model(env.get("drag_frame_model", "simple"))
            except ValueError:
                return None
            if drag_frame == FRAME_MODEL_IAU76_80_EOP:
                # The accelerated kernel models only a fixed z-axis Earth
                # rotation.  The authoritative EOP path now includes the full
                # frame-rotation derivative, so retain the Python evaluator
                # until that derivative is part of the compiled contract.
                return None

        from sim.acceleration.kernels.orbit_force_plan import (
            DENSITY_CONSTANT,
            DENSITY_NRLMSISE00_QUIET_THERMOSPHERE,
            FORCE_DRAG,
            FORCE_SPHERICAL_HARMONICS,
            FORCE_SRP,
            FORCE_THIRD_BODY_MOON,
            FORCE_THIRD_BODY_SUN,
        )

        compiled_harmonics = env.get("_compiled_spherical_harmonics_terms")
        if spherical_harmonics_plugin in self.plugins and (
            compiled_harmonics is None or not compiled_harmonics.all_normalized
        ):
            return None

        density_mode = DENSITY_CONSTANT
        constant_density = float(env.get("density_kg_m3", 0.0) or 0.0)
        if drag_plugin in self.plugins and env.get("density_kg_m3") is None:
            if str(env.get("atmosphere_model", "")).strip().lower() != "nrlmsise00":
                return None
            if str(env.get("geodetic_model", "")).strip().lower() != "wgs84":
                return None
            if callable(env.get("nrlmsise00_density_callable")):
                return None
            if env.get("nrlmsise00_lst_hr") is not None and not math.isfinite(float(env["nrlmsise00_lst_hr"])):
                # NaN is the packed "derive solar time" sentinel. Preserve
                # the reference owner's handling of non-finite overrides.
                return None
            try:
                density_frame = normalize_frame_model(
                    env.get("density_frame_model", env.get("drag_frame_model", "simple"))
                )
            except ValueError:
                return None
            if density_frame != FRAME_MODEL_IAU76_80_EOP:
                return None
            density_mode = DENSITY_NRLMSISE00_QUIET_THERMOSPHERE
        # The existing per-force kernels remain faster for plans without the
        # state-dependent NRLMSISE workload. Keep those configurations on the
        # established path instead of introducing a general-case regression.
        if density_mode != DENSITY_NRLMSISE00_QUIET_THERMOSPHERE:
            return None

        plugin_code = {
            spherical_harmonics_plugin: FORCE_SPHERICAL_HARMONICS,
            drag_plugin: FORCE_DRAG,
            srp_plugin: FORCE_SRP,
            third_body_sun_plugin: FORCE_THIRD_BODY_SUN,
            third_body_moon_plugin: FORCE_THIRD_BODY_MOON,
        }
        plugin_tuple = tuple(self.plugins)
        if self._compiled_force_plugins_cache != plugin_tuple:
            self._compiled_force_codes_cache = np.asarray(
                [plugin_code[plugin] for plugin in self.plugins], dtype=np.int64
            )
            self._compiled_force_plugins_cache = plugin_tuple

        if compiled_harmonics is None:
            c_nm = s_nm = _DUMMY_MATRIX
            diag = subdiag = _DUMMY_VECTOR
            recur_a = recur_b = recur_c = _DUMMY_MATRIX
            n_max = m_max = 0
        else:
            c_nm = compiled_harmonics.c_nm
            s_nm = compiled_harmonics.s_nm
            diag = compiled_harmonics.legendre_diag_scale
            subdiag = compiled_harmonics.legendre_subdiag_scale
            recur_a = compiled_harmonics.legendre_recur_a
            recur_b = compiled_harmonics.legendre_recur_b
            recur_c = compiled_harmonics.legendre_recur_c
            n_max = int(compiled_harmonics.n_max)
            m_max = int(compiled_harmonics.m_max)

        stage_times = (float(t_s), float(t_s) + 0.5 * float(dt_s), float(t_s) + float(dt_s))
        from sim.dynamics.orbit.nrlmsise00_backend import (
            _ALPHA,
            _ZN1,
            PD1,
            PDL1,
            PDM1,
            PMA1,
            PS1,
            PT1,
            PTL1,
            PTM1,
            _solar_geomagnetic_inputs,
        )

        needs_harmonics = spherical_harmonics_plugin in self.plugins
        needs_drag = drag_plugin in self.plugins
        needs_sun = srp_plugin in self.plugins or third_body_sun_plugin in self.plugins
        needs_moon = third_body_moon_plugin in self.plugins
        endpoint_environment_signature = self._compiled_endpoint_environment_signature(env, plugin_tuple)
        harmonic_rotation_definition = (
            self._compiled_rotation_definition_key(
                env,
                model_key="spherical_harmonics_frame_model",
                path_key="spherical_harmonics_eop_path",
            )
            if needs_harmonics
            else None
        )
        drag_rotation_definition = (
            self._compiled_rotation_definition_key(
                env,
                model_key="drag_frame_model",
                path_key="drag_eop_path",
            )
            if needs_drag
            else None
        )
        density_rotation_definition = (
            self._compiled_rotation_definition_key(
                env,
                model_key="density_frame_model",
                path_key="density_eop_path",
            )
            if density_mode == DENSITY_NRLMSISE00_QUIET_THERMOSPHERE
            else None
        )
        for stage_index, stage_time in enumerate(stage_times):
            endpoint_key = (
                None
                if endpoint_environment_signature is None
                else (float(stage_time), endpoint_environment_signature)
            )
            if (
                stage_index == 0
                and endpoint_key is not None
                and endpoint_key == self._compiled_endpoint_cache_key
            ):
                if needs_harmonics:
                    self._compiled_harmonic_rotations[stage_index] = self._compiled_endpoint_harmonic_rotation
                if needs_drag:
                    self._compiled_drag_rotations[stage_index] = self._compiled_endpoint_drag_rotation
                    self._compiled_density_rotations[stage_index] = self._compiled_endpoint_density_rotation
                    self._compiled_atmosphere_inputs[stage_index] = self._compiled_endpoint_atmosphere_inputs
                if needs_sun:
                    self._compiled_sun_positions[stage_index] = self._compiled_endpoint_sun_position
                if needs_moon:
                    self._compiled_moon_positions[stage_index] = self._compiled_endpoint_moon_position
                continue

            # The fused plan consumes only time-derived numeric artifacts. It
            # does not need the generic per-stage environment dictionary (or
            # its derived Sun direction), so prepare those artifacts directly
            # from the authoritative base environment.
            stage_env = env
            harmonic_rotation = None
            if needs_harmonics:
                harmonic_rotation = self._accelerated_compiled_rotation(
                    stage_env,
                    stage_time,
                    model_key="spherical_harmonics_frame_model",
                    path_key="spherical_harmonics_eop_path",
                )
                self._compiled_harmonic_rotations[stage_index] = harmonic_rotation
            drag_rotation = None
            if needs_drag:
                drag_rotation = (
                    harmonic_rotation
                    if harmonic_rotation is not None
                    and drag_rotation_definition == harmonic_rotation_definition
                    else self._accelerated_compiled_rotation(
                        stage_env,
                        stage_time,
                        model_key="drag_frame_model",
                        path_key="drag_eop_path",
                    )
                )
                self._compiled_drag_rotations[stage_index] = drag_rotation
            if density_mode == DENSITY_NRLMSISE00_QUIET_THERMOSPHERE:
                density_rotation = (
                    harmonic_rotation
                    if harmonic_rotation is not None
                    and density_rotation_definition == harmonic_rotation_definition
                    else drag_rotation
                    if drag_rotation is not None and density_rotation_definition == drag_rotation_definition
                    else self._accelerated_compiled_rotation(
                        stage_env,
                        stage_time,
                        model_key="density_frame_model",
                        path_key="density_eop_path",
                    )
                )
                self._compiled_density_rotations[stage_index] = density_rotation
                dt_utc = _datetime_from_env_t_s(stage_env, stage_time)
                f107a, f107, _ap, ap_a = _solar_geomagnetic_inputs(dt_utc, stage_env)
                if any(float(ap_a[index]) != 4.0 for index in range(2, 8)):
                    return None
                lst_override = stage_env.get("nrlmsise00_lst_hr")
                sun_lon = (
                    _nrlmsise00_sun_longitude_rad(dt_utc, stage_time, stage_env, rotation=density_rotation)
                    if lst_override is None else 0.0
                )
                self._compiled_atmosphere_inputs[stage_index] = (
                    float(dt_utc.timetuple().tm_yday),
                    float(dt_utc.hour * 3600.0 + dt_utc.minute * 60.0 + dt_utc.second + dt_utc.microsecond * 1.0e-6),
                    float(f107a),
                    float(f107),
                    float(sun_lon),
                    math.nan if lst_override is None else float(lst_override),
                )
            if needs_sun or needs_moon:
                sun = stage_env.get("sun_pos_eci_km")
                moon = stage_env.get("moon_pos_eci_km")
                if (needs_sun and sun is None) or (needs_moon and moon is None):
                    resolved_sun, resolved_moon = resolve_sun_moon_positions(stage_env, stage_time)
                    sun = resolved_sun if sun is None else sun
                    moon = resolved_moon if moon is None else moon
                if needs_sun:
                    self._compiled_sun_positions[stage_index] = np.asarray(sun, dtype=float).reshape(3)
                if needs_moon:
                    self._compiled_moon_positions[stage_index] = np.asarray(moon, dtype=float).reshape(3)

            if stage_index == 2 and endpoint_key is not None:
                if needs_harmonics:
                    self._compiled_endpoint_harmonic_rotation[:] = self._compiled_harmonic_rotations[stage_index]
                if needs_drag:
                    self._compiled_endpoint_drag_rotation[:] = self._compiled_drag_rotations[stage_index]
                    self._compiled_endpoint_density_rotation[:] = self._compiled_density_rotations[stage_index]
                    self._compiled_endpoint_atmosphere_inputs[:] = self._compiled_atmosphere_inputs[stage_index]
                if needs_sun:
                    self._compiled_endpoint_sun_position[:] = self._compiled_sun_positions[stage_index]
                if needs_moon:
                    self._compiled_endpoint_moon_position[:] = self._compiled_moon_positions[stage_index]
                self._compiled_endpoint_cache_key = endpoint_key

        shadow_name = str(env.get("srp_shadow_model", "conical")).lower()
        shadow_model = (
            0 if shadow_name in ("none", "off", "disabled") else 1 if shadow_name in ("cylindrical", "cylinder") else 2
        )
        parameters = self._compiled_scalar_parameters
        parameters[:] = (
            float(ctx.mu_km3_s2),
            float(env.get("spherical_harmonics_reference_radius_km", EARTH_RADIUS_KM)),
            float(ctx.mass_kg),
            float(ctx.cd),
            float(env.get("drag_area_m2", ctx.area_m2)),
            float(
                EARTH_ROT_RATE_RAD_S
                if env.get("drag_earth_rotation_rad_s") is None
                else env.get("drag_earth_rotation_rad_s")
            ),
            float(env.get("srp_area_m2", ctx.area_m2)),
            float(ctx.cr),
            srp_pressure_n_m2(env),
            float(AU_KM),
            float(EARTH_RADIUS_KM),
            float(SUN_RADIUS_KM),
            float(SUN_MU_KM3_S2),
            float(MOON_MU_KM3_S2),
            float(env.get("spherical_harmonics_mu_km3_s2", ctx.mu_km3_s2)),
        )
        result, valid = _compiled_builtin_force_plan_step()(
            np.asarray(x_eci, dtype=float).reshape(6),
            float(dt_s),
            np.asarray(command_accel_eci_km_s2, dtype=float).reshape(3),
            self._compiled_force_codes_cache,
            self._compiled_harmonic_rotations,
            self._compiled_density_rotations,
            self._compiled_drag_rotations,
            self._compiled_sun_positions,
            self._compiled_moon_positions,
            self._compiled_atmosphere_inputs,
            parameters,
            density_mode,
            constant_density,
            shadow_model,
            c_nm,
            s_nm,
            diag,
            subdiag,
            recur_a,
            recur_b,
            recur_c,
            n_max,
            m_max,
            PT1,
            PS1,
            PD1,
            PDL1,
            PTM1,
            PDM1,
            PTL1,
            PMA1,
            _ZN1,
            _ALPHA,
        )
        return result if bool(valid) else None

    def _try_propagate_staged_compiled(
        self,
        *,
        x_eci: np.ndarray,
        dt_s: float,
        t_s: float,
        command_accel_eci_km_s2: np.ndarray,
        env: dict,
        ctx: OrbitContext,
    ) -> np.ndarray | None:
        if str(env.get("srp_shadow_model", "")).lower() == "conical_ellipsoid":
            return None
        integrator_name = str(self.integrator).strip().lower()
        if not self._acceleration_enabled() or integrator_name not in {
            "rk4",
            "rkf78",
            "adaptive",
            "dopri5",
        }:
            return None

        from sim.acceleration.kernels.orbit_force_plan import (
            FORCE_DRAG,
            FORCE_J2,
            FORCE_J3,
            FORCE_J4,
            FORCE_LIFT,
            FORCE_SPHERICAL_HARMONICS,
            FORCE_SRP,
            FORCE_THIRD_BODY_MOON,
            FORCE_THIRD_BODY_PLANETS,
            FORCE_THIRD_BODY_SUN,
        )

        compiled_harmonics = env.get("_compiled_spherical_harmonics_terms")
        plugin_code = {
            j2_plugin: FORCE_J2,
            j3_plugin: FORCE_J3,
            j4_plugin: FORCE_J4,
            drag_plugin: FORCE_DRAG,
            lift_plugin: FORCE_LIFT,
            srp_plugin: FORCE_SRP,
            third_body_sun_plugin: FORCE_THIRD_BODY_SUN,
            third_body_moon_plugin: FORCE_THIRD_BODY_MOON,
            third_body_planets_plugin: FORCE_THIRD_BODY_PLANETS,
        }
        try:
            eop_aerodynamics = normalize_frame_model(
                env.get("drag_frame_model", "simple")
            ) == FRAME_MODEL_IAU76_80_EOP
        except ValueError:
            return None
        staged_cache_key = (
            tuple(self.plugins),
            bool(compiled_harmonics is not None and compiled_harmonics.all_normalized),
            eop_aerodynamics,
        )
        if self._staged_force_plan_cache_key != staged_cache_key:
            self._compiled_force_codes_cache = np.asarray(
                [
                    (
                        0
                        if eop_aerodynamics and (plugin is drag_plugin or plugin is lift_plugin)
                        else (
                            FORCE_SPHERICAL_HARMONICS
                            if plugin is spherical_harmonics_plugin
                            and compiled_harmonics is not None
                            and compiled_harmonics.all_normalized
                            else next((code for known, code in plugin_code.items() if plugin is known), 0)
                        )
                    )
                    for plugin in self.plugins
                ],
                dtype=np.int64,
            )
            self._staged_force_plan_cache_key = staged_cache_key
        force_codes = self._compiled_force_codes_cache
        assert force_codes is not None
        compiled_force_count = int(np.count_nonzero(force_codes))
        has_lift_planet_pair = FORCE_LIFT in force_codes and FORCE_THIRD_BODY_PLANETS in force_codes
        # Crossing the Python/compiled boundary is not free. Small plans are
        # faster on their existing specialized evaluators, while richer plans
        # win by batching component math here. This is a general profitability
        # rule; every force code remains supported when it participates in a
        # staged plan, and small plans retain their already-compiled kernels.
        if compiled_force_count < 4 and not has_lift_planet_pair:
            return None
        stage_env_cache: dict[float, dict] = {}
        command = np.asarray(command_accel_eci_km_s2, dtype=float).reshape(3)

        def deriv(stage_time: float, stage_state: np.ndarray) -> np.ndarray:
            return self._staged_compiled_derivative(
                t_s=float(stage_time),
                x_eci=np.asarray(stage_state, dtype=float).reshape(6),
                command_accel=command,
                env=env,
                stage_env_cache=stage_env_cache,
                ctx=ctx,
                force_codes=force_codes,
                compiled_harmonics=compiled_harmonics,
            )

        if integrator_name == "rk4":
            return rk4_step_state(
                deriv_fn=deriv,
                t_s=float(t_s),
                x=np.asarray(x_eci, dtype=float).reshape(6),
                dt_s=float(dt_s),
            )

        adaptive_method = "rkf78" if integrator_name in ("rkf78", "adaptive") else "dopri5"
        if self._rkf78_last_t_s is None or float(t_s) < float(self._rkf78_last_t_s) - 1e-12:
            self._rkf78_h_next = None
        x_next, step_info = integrate_adaptive(
            deriv_fn=deriv,
            t_s=float(t_s),
            x=np.asarray(x_eci, dtype=float).reshape(6),
            dt_s=float(dt_s),
            atol=self.adaptive_atol,
            rtol=self.adaptive_rtol,
            method=adaptive_method,
            h_init=self._rkf78_h_next,
            return_info=True,
        )
        self._rkf78_h_next = step_info.suggested_next_step_s
        self._rkf78_last_t_s = float(t_s + dt_s)
        self.last_adaptive_step_info = step_info
        previous = [] if self.adaptive_step_info is None else [self.adaptive_step_info]
        self.adaptive_step_info = combine_adaptive_step_info(adaptive_method, [*previous, step_info])
        return x_next

    def _staged_compiled_derivative(
        self,
        *,
        t_s: float,
        x_eci: np.ndarray,
        command_accel: np.ndarray,
        env: dict,
        stage_env_cache: dict[float, dict],
        ctx: OrbitContext,
        force_codes: np.ndarray,
        compiled_harmonics,
    ) -> np.ndarray:
        from sim.acceleration.kernels.orbit_force_plan import (
            FORCE_DRAG,
            FORCE_LIFT,
            FORCE_SPHERICAL_HARMONICS,
            FORCE_SRP,
            FORCE_THIRD_BODY_MOON,
            FORCE_THIRD_BODY_PLANETS,
            FORCE_THIRD_BODY_SUN,
        )

        needs_time_env = any(
            any(
                plugin is known for known in (
                    srp_plugin, third_body_sun_plugin,
                    third_body_moon_plugin, third_body_planets_plugin,
                )
            )
            for plugin in self.plugins
        )
        if needs_time_env:
            stage_env = stage_env_cache.get(float(t_s))
            if stage_env is None:
                stage_env = resolve_time_dependent_env(
                    env,
                    float(t_s),
                    cache_override=self._builtin_time_env_cache,
                )
                stage_env_cache[float(t_s)] = stage_env
                while len(self._builtin_time_env_cache) > 8:
                    self._builtin_time_env_cache.pop(next(iter(self._builtin_time_env_cache)))
        else:
            stage_env = env

        if compiled_harmonics is None:
            c_nm = s_nm = _DUMMY_MATRIX
            diag = subdiag = _DUMMY_VECTOR
            recur_a = recur_b = recur_c = _DUMMY_MATRIX
            n_max = m_max = 0
            harmonic_rotation = _IDENTITY3
        else:
            c_nm = compiled_harmonics.c_nm
            s_nm = compiled_harmonics.s_nm
            diag = compiled_harmonics.legendre_diag_scale
            subdiag = compiled_harmonics.legendre_subdiag_scale
            recur_a = compiled_harmonics.legendre_recur_a
            recur_b = compiled_harmonics.legendre_recur_b
            recur_c = compiled_harmonics.legendre_recur_c
            n_max = int(compiled_harmonics.n_max)
            m_max = int(compiled_harmonics.m_max)
            harmonic_rotation = (
                self._compiled_rotation(
                    stage_env,
                    t_s,
                    model_key="spherical_harmonics_frame_model",
                    path_key="spherical_harmonics_eop_path",
                )
                if FORCE_SPHERICAL_HARMONICS in force_codes
                else _IDENTITY3
            )

        has_aerodynamics = FORCE_DRAG in force_codes or FORCE_LIFT in force_codes
        drag_rotation = (
            self._compiled_rotation(
                stage_env,
                t_s,
                model_key="drag_frame_model",
                path_key="drag_eop_path",
            )
            if has_aerodynamics
            else _IDENTITY3
        )
        plugin_densities = np.zeros(len(self.plugins), dtype=float)
        lift_coefficient = float(stage_env.get("lift_coefficient", stage_env.get("cl", 0.0)) or 0.0)
        lift_direction_raw = stage_env.get("lift_direction_eci")
        lift_direction = (
            np.zeros(3, dtype=float)
            if lift_direction_raw is None
            else np.asarray(lift_direction_raw, dtype=float).reshape(3)
        )
        for index, force_code in enumerate(force_codes):
            if force_code == FORCE_DRAG or (
                force_code == FORCE_LIFT and lift_direction_raw is not None and lift_coefficient != 0.0
            ):
                density = stage_env.get("density_kg_m3")
                if density is None:
                    atmosphere_model = stage_env.get("atmosphere_model")
                    if atmosphere_model in (None, ""):
                        requirement = "Drag" if force_code == FORCE_DRAG else "Lift"
                        if requirement == "Drag":
                            raise ValueError(
                                "Drag requires an explicit environment.atmosphere_model or density_kg_m3; "
                                "no orbital-decay atmosphere is selected implicitly."
                            )
                        raise ValueError("Lift requires an explicit environment.atmosphere_model or density_kg_m3.")
                    density = density_from_model(
                        str(atmosphere_model).lower(),
                        x_eci[:3],
                        t_s,
                        env=env,
                    )
                plugin_densities[index] = float(density)

        needs_sun = any(code in (FORCE_SRP, FORCE_THIRD_BODY_SUN) for code in force_codes)
        needs_moon = FORCE_THIRD_BODY_MOON in force_codes
        sun = stage_env.get("sun_pos_eci_km")
        moon = stage_env.get("moon_pos_eci_km")
        if (needs_sun and sun is None) or (needs_moon and moon is None):
            resolved_sun, resolved_moon = resolve_sun_moon_positions(stage_env, t_s)
            sun = resolved_sun if sun is None else sun
            moon = resolved_moon if moon is None else moon
        sun_position = np.zeros(3, dtype=float) if sun is None else np.asarray(sun, dtype=float).reshape(3)
        moon_position = np.zeros(3, dtype=float) if moon is None else np.asarray(moon, dtype=float).reshape(3)

        planet_count = 0
        if FORCE_THIRD_BODY_PLANETS in force_codes:
            selected = stage_env.get("third_body_planets", [])
            selected_names = (
                [selected.strip().lower()]
                if isinstance(selected, str)
                else [str(value).strip().lower() for value in selected]
            )
            if any(value in ("all", "*") for value in selected_names):
                selected_names = list(PLANETARY_MU_KM3_S2)
            valid_planet_names = [name for name in selected_names if name in PLANETARY_MU_KM3_S2]
            if len(valid_planet_names) > self._staged_planet_positions.shape[0]:
                self._staged_planet_positions = np.zeros((len(valid_planet_names), 3), dtype=float)
                self._staged_planet_mu = np.zeros(len(valid_planet_names), dtype=float)
            for name in valid_planet_names:
                self._staged_planet_positions[planet_count] = resolve_body_position_eci_km(
                    name,
                    env=stage_env,
                    t_s=t_s,
                )
                self._staged_planet_mu[planet_count] = float(
                    stage_env.get(f"{name}_mu_km3_s2", PLANETARY_MU_KM3_S2[name])
                )
                planet_count += 1

        shadow_name = str(stage_env.get("srp_shadow_model", "conical")).lower()
        shadow_model = (
            0 if shadow_name in ("none", "off", "disabled") else 1 if shadow_name in ("cylindrical", "cylinder") else 2
        )
        omega_raw = stage_env.get("drag_earth_rotation_rad_s")
        parameters = self._compiled_scalar_parameters
        parameters[:] = (
            float(ctx.mu_km3_s2),
            float(stage_env.get("spherical_harmonics_reference_radius_km", EARTH_RADIUS_KM)),
            float(ctx.mass_kg),
            float(ctx.cd),
            float(stage_env.get("drag_area_m2", ctx.area_m2)),
            float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
            float(stage_env.get("srp_area_m2", ctx.area_m2)),
            float(ctx.cr),
            srp_pressure_n_m2(stage_env),
            float(AU_KM),
            float(EARTH_RADIUS_KM),
            float(SUN_RADIUS_KM),
            float(SUN_MU_KM3_S2),
            float(MOON_MU_KM3_S2),
            float(stage_env.get("spherical_harmonics_mu_km3_s2", ctx.mu_km3_s2)),
        )
        lift_area_m2 = float(stage_env.get("lift_area_m2", stage_env.get("drag_area_m2", ctx.area_m2)))
        if force_codes.size == 1 and force_codes[0] == FORCE_DRAG:
            components = np.empty((1, 3), dtype=float)
            components[0] = _compiled_drag_force_component()(
                x_eci[:3],
                x_eci[3:],
                drag_rotation,
                plugin_densities[0],
                parameters[2],
                parameters[3],
                parameters[4],
                parameters[5],
            )
        elif force_codes.size == 1 and force_codes[0] == FORCE_LIFT:
            components = np.empty((1, 3), dtype=float)
            components[0] = _compiled_lift_force_component()(
                x_eci[:3],
                x_eci[3:],
                drag_rotation,
                plugin_densities[0],
                parameters[2],
                lift_coefficient,
                lift_area_m2,
                parameters[5],
                lift_direction,
            )
        else:
            components = _compiled_builtin_force_components()(
                x_eci,
                force_codes,
                plugin_densities,
                harmonic_rotation,
                drag_rotation,
                sun_position,
                moon_position,
                self._staged_planet_positions,
                self._staged_planet_mu,
                int(planet_count),
                parameters,
                shadow_model,
                lift_direction,
                lift_coefficient,
                lift_area_m2,
                c_nm,
                s_nm,
                diag,
                subdiag,
                recur_a,
                recur_b,
                recur_c,
                n_max,
                m_max,
            )
        out = np.empty(6, dtype=float)
        out[:3] = x_eci[3:]
        acceleration = np.add(accel_two_body(x_eci[:3], ctx.mu_km3_s2), command_accel)
        for index, plugin in enumerate(self.plugins):
            if force_codes[index] == 0:
                plugin_env = stage_env if plugin is spherical_harmonics_plugin else env
                acceleration += plugin(t_s, x_eci, plugin_env, ctx)
            else:
                acceleration += components[index]
        out[3:] = acceleration
        return out

    def _propagate_builtin_rk4(
        self,
        *,
        x_eci: np.ndarray,
        dt_s: float,
        t_s: float,
        command_accel_eci_km_s2: np.ndarray,
        env: dict,
        ctx: OrbitContext,
    ) -> np.ndarray:
        x0 = np.asarray(x_eci, dtype=float).reshape(6)
        h = float(dt_s)
        t0 = float(t_s)
        command_accel = np.asarray(command_accel_eci_km_s2, dtype=float).reshape(3)
        stage_env_cache: dict[float, dict] = {}
        k1 = self._builtin_rk4_k1
        k2 = self._builtin_rk4_k2
        k3 = self._builtin_rk4_k3
        k4 = self._builtin_rk4_k4
        x_stage = self._builtin_rk4_stage

        self._builtin_deriv_into(t0, x0, env, stage_env_cache, ctx, command_accel, k1)
        np.multiply(k1, 0.5 * h, out=x_stage)
        x_stage += x0
        self._builtin_deriv_into(t0 + 0.5 * h, x_stage, env, stage_env_cache, ctx, command_accel, k2)
        np.multiply(k2, 0.5 * h, out=x_stage)
        x_stage += x0
        self._builtin_deriv_into(t0 + 0.5 * h, x_stage, env, stage_env_cache, ctx, command_accel, k3)
        np.multiply(k3, h, out=x_stage)
        x_stage += x0
        self._builtin_deriv_into(t0 + h, x_stage, env, stage_env_cache, ctx, command_accel, k4)
        return x0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _builtin_deriv_into(
        self,
        t_local: float,
        x_local: np.ndarray,
        env: dict,
        stage_env_cache: dict[float, dict],
        ctx: OrbitContext,
        command_accel: np.ndarray,
        out: np.ndarray,
    ) -> None:
        stage_env = self._builtin_stage_env(env, t_local, stage_env_cache)
        out[:3] = x_local[3:]
        a = self._builtin_rk4_accel
        np.add(accel_two_body(x_local[:3], ctx.mu_km3_s2), command_accel, out=a)
        for plugin in self.plugins:
            if self._acceleration_enabled() and plugin is spherical_harmonics_plugin:
                a += _accelerated_spherical_harmonics_plugin(t_local, x_local, stage_env, ctx)
            else:
                a += plugin(t_local, x_local, stage_env, ctx)
        out[3:] = a

    def _builtin_stage_env(self, env: dict, t_s: float, cache: dict[float, dict]) -> dict:
        if not self._builtin_needs_time_env_cache:
            return env
        key = float(t_s)
        stage_env = cache.get(key)
        if stage_env is None:
            stage_env = resolve_time_dependent_env(
                env,
                key,
                cache_override=self._builtin_time_env_cache,
            )
            cache[key] = stage_env
            while len(self._builtin_time_env_cache) > 8:
                self._builtin_time_env_cache.pop(next(iter(self._builtin_time_env_cache)))
        return stage_env
