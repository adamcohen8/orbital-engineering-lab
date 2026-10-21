"""Stage adapter for the opt-in Rust ONP force plan.

Python retains configuration and resource policy. Persistent Rust contexts
assemble built-in stages; atmosphere and custom models retain their callbacks.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import fields
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

from sim.aero.core import atmosphere_relative_velocity_eci_km_s
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.dynamics.orbit.de440_hpop import (
    _load_de440_light,
    _resolve_path,
    default_de440_coeff_path,
)
from sim.dynamics.orbit.environment import (
    EARTH_RADIUS_KM,
    EARTH_ROT_RATE_RAD_S,
    MOON_MU_KM3_S2,
    SUN_MU_KM3_S2,
    SUN_RADIUS_KM,
    srp_pressure_n_m2,
)
from sim.dynamics.orbit.epoch import AU_KM, resolve_sun_moon_positions, resolve_time_dependent_env
from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    FrameContext,
    PreparedFrameEvaluator,
    _load_nut80_table,
    frame_context_from_environment,
    normalize_frame_model,
)
from sim.dynamics.orbit.propagator import (
    drag_plugin,
    j2_plugin,
    j3_plugin,
    j4_plugin,
    spherical_harmonics_plugin,
    srp_plugin,
    third_body_moon_plugin,
    third_body_sun_plugin,
)
from sim.dynamics.orbit.spherical_harmonics import SphericalHarmonicTerm
from sim.rust_environment_backend import (
    try_configure_environment_de440_body,
    try_create_environment_context,
    try_create_force_context,
)

_CODES = {
    spherical_harmonics_plugin: 1, drag_plugin: 2, srp_plugin: 3,
    third_body_sun_plugin: 4, third_body_moon_plugin: 5,
    j2_plugin: 6, j3_plugin: 7, j4_plugin: 8,
}


def force_codes(propagator, env):
    from sim import rust_environment_backend
    from sim.dynamics.orbit.rust_precision_force_plan import precision_code

    supports_precision_force_plan = getattr(rust_environment_backend, "supports_precision_force_plan", lambda: False)
    codes = []
    for plugin in propagator.plugins:
        builtin = next((code for known, code in _CODES.items() if plugin is known), None)
        code = builtin if builtin is not None else precision_code(plugin)
        if code is None or (code > 8 and (env.get("_rust_precision_force_plan_disabled") or not supports_precision_force_plan())):
            return None
        codes.append(code)
    if 1 in codes and str(env.get("srp_shadow_model", "")).lower() == "conical_ellipsoid":
        context = frame_context_from_environment(env)
        if (context.model != normalize_frame_model(env.get("spherical_harmonics_frame_model", "simple"))
                or context.eop_path != env.get("spherical_harmonics_eop_path")):
            # The compact native stage has one shared Earth rotation. Distinct
            # gravity/shadow frames remain on the authoritative plugin route.
            return None
    return codes


def immutable_builtin_plan_signature(propagator, env: dict, ctx) -> tuple | None:
    """Bind speculative passive rows to all built-in numeric inputs.

    Arbitrary callables/objects are excluded from speculation. Mutable world
    truth and internal scratch caches are not read by these built-in forces;
    compiled harmonic coefficient content is bound separately.
    """
    if force_codes(propagator, env) is None:
        return None

    def frozen(value):
        if type(value) is SphericalHarmonicTerm:
            return ("SphericalHarmonicTerm", tuple(
                (field.name, frozen(getattr(value, field.name)))
                for field in fields(SphericalHarmonicTerm)
            ))
        if type(value) is FrameContext:
            return ("FrameContext", tuple((field.name, frozen(getattr(value, field.name)))
                    for field in fields(FrameContext)),
                    None if value.eop_path is None else _resource_signature(os.path.expanduser(value.eop_path)))
        if value is None or isinstance(value, (str, bool, int, float, datetime)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return _table_signature(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (tuple, list)):
            return tuple(frozen(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((str(key), frozen(item)) for key, item in value.items()))
        raise ValueError("unsupported speculative environment value")

    try:
        values = tuple(sorted(
            (str(key), frozen(value)) for key, value in env.items()
            if not str(key).startswith("_") and key not in {"world_truth", "sim_t_s"}
        ))
        resources = tuple(sorted(
            (str(key), _resource_signature(os.path.expanduser(str(value))))
            for key, value in env.items() if str(key).endswith("_path") and value not in (None, "")
        ))
        harmonic = env.get("_compiled_spherical_harmonics_terms")
        harmonic_signature = None if harmonic is None else (
            harmonic.n_max, harmonic.m_max, harmonic.all_normalized,
            tuple(_table_signature(getattr(harmonic, name)) for name in (
                "c_nm", "s_nm", "legendre_diag_scale", "legendre_subdiag_scale",
                "legendre_recur_a", "legendre_recur_b", "legendre_recur_c",
            )),
        )
        return (
            id(propagator), str(propagator.numeric_backend), str(propagator.integrator), str(propagator.model),
            tuple(force_codes(propagator, env)),
            tuple(frozen(getattr(plugin, f.name)) for plugin in propagator.plugins if hasattr(plugin, "__dataclass_fields__") for f in fields(plugin) if f.init),
            tuple((_table_signature(plugin._factors), _table_signature(plugin._coefficients)) for plugin in propagator.plugins if hasattr(plugin, "_coefficients")),
            float(ctx.mu_km3_s2), float(ctx.mass_kg), float(ctx.area_m2), float(ctx.cd), float(ctx.cr),
            values, resources, harmonic_signature,
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _resource_signature(path) -> tuple:
    """Return a small invalidation key for a native copied resource."""
    try:
        stat = os.stat(path)
    except OSError:
        return (str(path), None)
    return (str(path), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))


def _table_signature(values) -> tuple:
    array = np.asarray(values)
    return (array.shape, str(array.dtype), hashlib.sha256(array.tobytes()).hexdigest())


def _configure_native_environment_context(propagator, env: dict, *, need_de440: bool):
    """Create/reuse a native context while preserving Python resource policy."""
    coefficients, terms = _load_nut80_table()
    coeff_signature = (_table_signature(coefficients), _table_signature(terms))
    coeff_path = None
    if need_de440:
        coeff_path_raw = env.get("de440_coeff_path")
        coeff_path = default_de440_coeff_path() if coeff_path_raw is None else _resolve_path(str(coeff_path_raw))
    de440_signature = _resource_signature(coeff_path) if coeff_path is not None and coeff_path.suffix.lower() == ".npz" else None
    key = (coeff_signature, de440_signature)
    cached = getattr(propagator, "_rust_environment_context_cache", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == key:
        return cached[1]
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] != key:
        # Prepared Sun/Moon rows are keyed by epoch and configuration.  A
        # rewritten DE440/EOP resource must invalidate those rows before the
        # newly configured native context is used.
        prepared = getattr(propagator, "_builtin_time_env_cache", None)
        if isinstance(prepared, dict):
            prepared.clear()
    context = try_create_environment_context(coefficients, terms)
    if context is None:
        return None
    if de440_signature is not None:
        light = _load_de440_light(str(coeff_path))
        bodies = light.get("body_set", light.get("bodies", ()))
        if not {"earthmoon", "moon", "sun"}.issubset(bodies):
            raise RuntimeError("OEL DE440 light file must include earthmoon, moon, and sun")
        for body_index, body in enumerate(("earthmoon", "moon", "sun")):
            starts = np.asarray(light.get(f"{body}_start_jd_tdb", light["row_start_jd_tdb"]), dtype=float)
            ends = np.asarray(light.get(f"{body}_end_jd_tdb", light["row_end_jd_tdb"]), dtype=float)
            coeff_count, segments, span_days = light["body_specs"][body]
            records = light["body_records"][body]
            try_configure_environment_de440_body(
                context,
                body_index,
                row_starts_jd_tdb=starts,
                row_ends_jd_tdb=ends,
                coeff_count=int(coeff_count),
                segments=int(segments),
                span_days=float(span_days),
                x_rows=np.asarray(records[2], dtype=float),
                y_rows=np.asarray(records[3], dtype=float),
                z_rows=np.asarray(records[4], dtype=float),
            )
    propagator._rust_environment_context_cache = (key, context)
    return context


def _native_force_context(propagator, codes, scalars, shadow, dims, tables, ctx=None):
    table_signature = tuple(_table_signature(table) for table in tables)
    from sim.dynamics.orbit.rust_precision_force_plan import precision_code, precision_signature, specifications
    precision_key = tuple(precision_signature(plugin, ctx) for plugin in propagator.plugins if precision_code(plugin) is not None) if ctx is not None else ()
    key = (precision_key, tuple(codes), tuple(float(value) for value in scalars), int(shadow), tuple(dims), table_signature)
    cached = getattr(propagator, "_rust_force_context_cache", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == key:
        return cached[1]
    kwargs = {"precision_specs": specifications(propagator.plugins, ctx)} if precision_key else {}
    context = try_create_force_context(codes, scalars, shadow, dims, tables, **kwargs)
    if context is not None:
        propagator._rust_force_context_cache = (key, context)
    return context


def _prepared_frame(propagator, env, model_key, path_key, native_context):
    context = FrameContext(
        model=env.get(model_key, "simple"), jd_utc_start=env.get("jd_utc_start"),
        eop_path=env.get(path_key), eop_extrapolation=env.get("eop_extrapolation", "error") or "error",
        tt_minus_utc_s=69.184 if env.get("tt_minus_utc_s") is None else float(env["tt_minus_utc_s"]),
        dut1_s=env.get("dut1_s"), xp_arcsec=env.get("xp_arcsec"), yp_arcsec=env.get("yp_arcsec"),
        dat_s=env.get("dat_s"), ddpsi_rad=float(env.get("ddpsi_rad", 0.0) or 0.0),
        ddeps_rad=float(env.get("ddeps_rad", 0.0) or 0.0), source="rust_onp",
    )
    cache = getattr(propagator, "_rust_prepared_frame_cache", None)
    if cache is None:
        cache = {}
        propagator._rust_prepared_frame_cache = cache
    key = (context, id(native_context))
    prepared = cache.get(key)
    if prepared is None:
        prepared = PreparedFrameEvaluator(context, native_context)
        cache[key] = prepared
        while len(cache) > 8:
            cache.pop(next(iter(cache)))
    return prepared


def make_plan(propagator, state, t_s, env, ctx):
    """Return immutable Rust plan inputs and an authoritative stage callback."""

    if force_codes(propagator, env) is None:
        raise ValueError("Rust force plan does not support a selected plugin")
    codes = force_codes(propagator, env)
    if 1 in codes and env.get("_compiled_spherical_harmonics_terms") is None:
        spherical_harmonics_plugin(t_s, np.asarray(state, dtype=float), env, ctx)
    harmonic = env.get("_compiled_spherical_harmonics_terms")
    if 1 in codes and (harmonic is None or not harmonic.all_normalized):
        raise ValueError("Rust harmonics require compiled, fully normalized terms")
    dims = (0, 0) if harmonic is None else (int(harmonic.n_max), int(harmonic.m_max))
    tables = [] if harmonic is None else [
        np.asarray(getattr(harmonic, key), dtype=float).ravel()
        for key in (
            "c_nm", "s_nm", "legendre_diag_scale", "legendre_subdiag_scale",
            "legendre_recur_a", "legendre_recur_b", "legendre_recur_c",
        )
    ]
    shadow_name = str(env.get("srp_shadow_model", "conical")).lower()
    shadow = 0 if shadow_name in ("none", "off", "disabled") else 1 if shadow_name in ("cylindrical", "cylinder") else 3 if shadow_name == "conical_ellipsoid" else 2
    omega_raw = env.get("drag_earth_rotation_rad_s")
    scalars = [
        float(ctx.mu_km3_s2),
        float(env.get("spherical_harmonics_reference_radius_km", EARTH_RADIUS_KM)),
        float(ctx.mass_kg),
        float(env.get("drag_coefficient", ctx.cd)),
        float(env.get("drag_area_m2", ctx.area_m2)),
        float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
        float(env.get("srp_area_m2", ctx.area_m2)),
        float(ctx.cr),
        srp_pressure_n_m2(env),
        float(AU_KM), float(EARTH_RADIUS_KM), float(SUN_RADIUS_KM),
        float(SUN_MU_KM3_S2), float(MOON_MU_KM3_S2),
        float(env.get("spherical_harmonics_mu_km3_s2", ctx.mu_km3_s2)),
    ]
    if all(code in (6, 7, 8) for code in codes):
        # These ordered zonal forces have no stage environment dependencies.
        # Rust still evaluates all four state derivatives at every RK4 step.
        force_context = _native_force_context(propagator, codes, scalars, shadow, dims, tables, ctx)
        empty_stage = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 10
        return codes, scalars, shadow, dims, tables, lambda _time, _state: empty_stage, force_context
    stage_env_cache = {}
    # Rust owns the repeated arithmetic only. Resource loading, EOP coverage,
    # ephemeris resolution, and custom callbacks remain authoritative in this
    # Python preparation closure.
    native_env = dict(env)
    native_env["_rust_numeric_backend"] = "rust"
    ephemeris_mode = str(env.get("ephemeris_mode", "analytic_enhanced") or "analytic_enhanced").strip().lower()
    need_de440 = bool(
        any(code in codes for code in (3, 4, 5))
        and ephemeris_mode in {"de440", "hpop_de440", "de440_hpop"}
    )
    need_native_frame = bool(
        (1 in codes and normalize_frame_model(env.get("spherical_harmonics_frame_model", "simple")) == FRAME_MODEL_IAU76_80_EOP)
        or (2 in codes and any(normalize_frame_model(model) == FRAME_MODEL_IAU76_80_EOP for model in (
            env.get("drag_frame_model", "simple"),
            env.get("density_frame_model", env.get("drag_frame_model", "simple")),
        )))
    )
    native_environment_context = _configure_native_environment_context(
        propagator, env, need_de440=need_de440,
    ) if (need_de440 or need_native_frame or any(code in codes for code in (3, 4, 5))) else None
    if native_environment_context is not None:
        native_env["_rust_environment_context"] = native_environment_context
    harmonic_frame = _prepared_frame(
        propagator, env, "spherical_harmonics_frame_model", "spherical_harmonics_eop_path",
        native_environment_context,
    ) if 1 in codes else (PreparedFrameEvaluator(frame_context_from_environment(env), native_environment_context)
                         if shadow == 3 and 3 in codes else None)
    drag_frame = density_frame = None
    if 2 in codes:
        drag_frame = _prepared_frame(
            propagator, env, "drag_frame_model", "drag_eop_path", native_environment_context,
        )
        density_env = dict(env)
        density_env.setdefault("density_frame_model", env.get("drag_frame_model", "simple"))
        density_env.setdefault("density_eop_path", env.get("drag_eop_path"))
        density_frame = _prepared_frame(
            propagator, density_env, "density_frame_model", "density_eop_path", native_environment_context,
        )
        native_env["_prepared_density_frame"] = density_frame
    prepared_frames = set(frame for frame in (harmonic_frame, drag_frame, density_frame) if frame is not None)
    de440_path = None
    if need_de440:
        raw_path = env.get("de440_coeff_path")
        de440_path = default_de440_coeff_path() if raw_path is None else _resolve_path(str(raw_path))
    de440_signature = _resource_signature(de440_path) if de440_path is not None else None
    eop_paths = tuple(sorted({str(env[key]) for key in (
        "spherical_harmonics_eop_path", "drag_eop_path", "density_eop_path", "de440_eop_path", "eop_path",
    ) if env.get(key)}))
    eop_signature = tuple(_resource_signature(path) for path in eop_paths)
    force_context = _native_force_context(propagator, codes, scalars, shadow, dims, tables, ctx)
    # Caller-owned, bounded reuse of exact frame preparation. The aero owner
    # keys every frame input and EOP metadata; state values stay uncached.
    frame_cache = getattr(propagator, "_rust_atmosphere_frame_cache", None)
    if frame_cache is None:
        frame_cache = {}
        propagator._rust_atmosphere_frame_cache = frame_cache
    empty_stage = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 10
    has_bodies = any(code in codes for code in (3, 4, 5))

    def stage_inputs(stage_time, raw_state, *, _density_only=False):
        nonlocal native_environment_context, de440_signature, eop_signature
        for frame in prepared_frames:
            frame.refresh()
        current_eop_signature = tuple(_resource_signature(path) for path in eop_paths)
        if current_eop_signature != eop_signature:
            stage_env_cache.clear()
            propagator._builtin_time_env_cache.clear()
            eop_signature = current_eop_signature
        # A batch keeps native coefficients alive across steps, but a rewritten
        # or deleted ephemeris file must still invalidate prepared stage rows.
        if de440_path is not None:
            signature = _resource_signature(de440_path)
            if signature != de440_signature:
                native_environment_context = _configure_native_environment_context(
                    propagator, env, need_de440=True,
                )
                native_env["_rust_environment_context"] = native_environment_context
                stage_env_cache.clear()
                de440_signature = signature
        x = np.asarray(raw_state, dtype=float).reshape(6)
        if has_bodies:
            stage_env = stage_env_cache.get(stage_time)
            if stage_env is None:
                stage_env = resolve_time_dependent_env(
                    native_env, float(stage_time), cache_override=propagator._builtin_time_env_cache,
                )
                stage_env_cache[stage_time] = stage_env
                while len(stage_env_cache) > 8:
                    stage_env_cache.pop(next(iter(stage_env_cache)))
                while len(propagator._builtin_time_env_cache) > 8:
                    propagator._builtin_time_env_cache.pop(next(iter(propagator._builtin_time_env_cache)))
        else:
            stage_env = env
        result = empty_stage.copy()
        if harmonic_frame is not None:
            result[:9] = harmonic_frame.rotation(float(stage_time)).ravel().tolist()
        if 2 in codes:
            density = stage_env.get("density_kg_m3")
            if density is None:
                model = stage_env.get("atmosphere_model")
                if model in (None, ""):
                    raise ValueError("Drag requires an explicit atmosphere_model or density_kg_m3")
                density = density_from_model(str(model).lower(), x[:3], stage_time, env=native_env)
            if _density_only:
                return float(density)
            omega = stage_env.get("drag_earth_rotation_rad_s")
            v_rel = atmosphere_relative_velocity_eci_km_s(
                x[:3], x[3:], t_s=float(stage_time),
                earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega is None else omega),
                frame_model=str(stage_env.get("drag_frame_model", "simple")),
                jd_utc_start=stage_env.get("jd_utc_start"),
                eop_path=stage_env.get("drag_eop_path"),
                dut1_s=stage_env.get("dut1_s"),
                xp_arcsec=stage_env.get("xp_arcsec"),
                yp_arcsec=stage_env.get("yp_arcsec"),
                dat_s=stage_env.get("dat_s"),
                tt_minus_utc_s=stage_env.get("tt_minus_utc_s"),
                ddpsi_rad=float(stage_env.get("ddpsi_rad", 0.0) or 0.0),
                ddeps_rad=float(stage_env.get("ddeps_rad", 0.0) or 0.0),
                eop_extrapolation=str(stage_env.get("eop_extrapolation", "error") or "error"),
                _frame_cache=frame_cache,
                _numeric_backend="rust",
                _prepared_frame=drag_frame,
            )
            result[9] = float(density)
            result[10:13] = v_rel.tolist()
        sun = stage_env.get("sun_pos_eci_km")
        moon = stage_env.get("moon_pos_eci_km")
        if ((3 in codes or 4 in codes) and sun is None) or (5 in codes and moon is None):
            resolved_sun, resolved_moon = resolve_sun_moon_positions(stage_env, stage_time)
            sun = resolved_sun if sun is None else sun
            moon = resolved_moon if moon is None else moon
        if sun is not None:
            result[13:16] = np.asarray(sun, dtype=float).reshape(3).tolist()
        if moon is not None:
            result[16:19] = np.asarray(moon, dtype=float).reshape(3).tolist()
        return result

    if 2 in codes and all(code in (2, 6, 7, 8) for code in codes) and normalize_frame_model(
        env.get("drag_frame_model", "simple")
    ) != FRAME_MODEL_IAU76_80_EOP:
        stage_inputs._simple_drag_density_callback = partial(stage_inputs, _density_only=True)
        stage_inputs._simple_drag_jd_utc_start = env.get("jd_utc_start")
    from sim.dynamics.orbit.rust_stage_context import try_native_stage_context
    try:
        native_stage = try_native_stage_context(
            propagator, codes, scalars, env, native_env, native_environment_context,
            harmonic_frame, drag_frame, density_frame, de440_path, stage_inputs,
        )
    except (TypeError, ValueError):
        # Unsupported snapshot metadata retains the reference adapter and its
        # force-specific validation, including unused atmosphere fields.
        native_stage = None
    if any(code > 8 for code in codes):
        from sim.dynamics.orbit.rust_precision_force_plan import wrap_stages
        return codes, scalars, shadow, dims, tables, wrap_stages(
            native_stage if native_stage is not None else stage_inputs, propagator.plugins, env, ctx,
            immutable_resources=(tuple(eop_paths)+((str(de440_path),) if de440_path else ())
                                 if native_stage is not None and 2 not in codes
                                 and env.get("_precision_immutable_sampled",False) else None)), force_context
    return codes, scalars, shadow, dims, tables, native_stage if native_stage is not None else stage_inputs, force_context
