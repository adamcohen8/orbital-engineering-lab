from __future__ import annotations

from typing import Any

from sim.config.scenario.models import (
    SimulatorSection,
)
from sim.config.scenario.objects import _parse_algorithm_pointer
from sim.config.scenario.primitives import (
    _REENTRY_TERMINATION_LIMIT_FIELDS,
    _SIMULATOR_UNSUPPORTED_ALIASES,
    _as_dict,
    _parse_bool,
    _parse_float,
    _parse_optional_float,
    _reject_unknown_fields,
    _reject_unsupported_aliases,
    _validate_sim_timing,
)
from sim.numeric_backend import normalize_numeric_backend

__all__ = [
    '_normalize_reentry_termination_block',
    '_normalize_reentry_section',
    '_normalize_simulator_termination_block',
    '_parse_simulator_section',
    '_normalize_dynamics_section',
    '_parse_simulator_frames_section',
    '_parse_resource_profile',
    '_parse_acceleration_section',
    '_parse_simulator_execution_section',
    '_normalize_spherical_harmonics_section',
]


# Closed, user-facing environment contract. Runtime-derived cache/state keys are
# deliberately absent; callers that need non-OEL metadata can use `extensions`.
_ENVIRONMENT_ALLOWED_FIELDS = {
    "air_gamma", "air_gas_constant_j_kg_k", "atmosphere_env", "atmosphere_model",
    "atmosphere_sun_model", "dat_s", "ddeps_rad", "ddpsi_rad", "de440_coeff_path",
    "de440_eop_path", "de440_tai_utc_s", "density_eop_path", "density_frame_model",
    "density_kg_m3", "drag_earth_rotation_rad_s", "drag_eop_path", "drag_frame_model",
    "dut1_s", "earth_radius_km", "eop_extrapolation", "ephemeris_body_callable",
    "ephemeris_callable", "ephemeris_mode", "extensions", "geodetic_model", "jd_utc",
    "jd_utc_start", "moon_pos_eci_km", "nrlmsise00_density_callable", "physical_aerodynamics",
    "solar_irradiance_w_m2", "spice_abcorr", "spice_body_ephemeris_callable",
    "spice_body_target_map", "spice_ephemeris_callable", "spice_frame", "spice_kernels",
    "spice_moon_target", "spice_observer", "spice_sun_target", "srp_distance_scale",
    "srp_geometry", "srp_pressure_n_m2", "srp_shadow_factor", "srp_shadow_model",
    "srp_sun_dir_eci", "sun_dir_eci", "sun_pos_eci_km", "third_body_planets",
    "tt_minus_utc_s", "xp_arcsec", "yp_arcsec",
}

_ATMOSPHERE_ENV_ALLOWED_FIELDS = {
    "ap", "atmo_epoch_utc", "dat_s", "ddeps_rad", "ddpsi_rad", "density_eop_path",
    "density_frame_model", "drag_earth_rotation_rad_s", "drag_eop_path", "drag_frame_model",
    "dut1_s", "eop_extrapolation", "exponential_ceiling_altitude_km",
    "exponential_reference_altitude_km", "exponential_reference_density_kg_m3",
    "exponential_scale_height_km", "extensions", "f107", "f107a", "harris_priester_coeff_path",
    "harris_priester_f107", "harris_priester_lower_limit_km", "harris_priester_n",
    "harris_priester_ra_lag_rad", "harris_priester_upper_limit_km", "hp_coeff_path", "hp_f107",
    "jacchia70_ap", "jacchia70_density_callable", "jacchia70_f10", "jacchia70_f10b",
    "jacchia70_gast_rad", "jacchia70_geomagnetic_index_type", "jacchia70_sw_path",
    "jb2006_ap_path", "jb2006_density_callable", "jb2006_sol_path", "jb2008_density_callable",
    "jb2008_dtc_path", "jb2008_sol_path", "jd_utc", "jd_utc_start", "msis86_ap",
    "msis86_ap_a", "msis86_density_callable", "msis86_f107", "msis86_f107a",
    "msis86_hpop_angle_compat", "msis86_lst_hr", "msis86_sw_path", "msis_sw_path",
    "nrlmsise00_ap", "nrlmsise00_ap_a", "nrlmsise00_density_callable", "nrlmsise00_f107",
    "nrlmsise00_f107a", "nrlmsise00_lst_hr", "nrlmsise00_sw_path", "solar_flux_f107",
    "spherical_harmonics_eop_path", "tt_minus_utc_s", "xp_arcsec", "yp_arcsec",
}

# Atmospheric inputs may be written either directly in `environment` or in the
# nested `atmosphere_env` block; the single-run boundary normalizes both forms.
_ENVIRONMENT_ALLOWED_FIELDS.update(_ATMOSPHERE_ENV_ALLOWED_FIELDS)


def _normalize_environment_section(value: Any) -> dict[str, Any]:
    environment = _as_dict(value, "simulator.environment")
    _reject_unknown_fields(environment, "simulator.environment", _ENVIRONMENT_ALLOWED_FIELDS)
    normalized = dict(environment)
    atmosphere_env = _as_dict(normalized.get("atmosphere_env"), "simulator.environment.atmosphere_env")
    _reject_unknown_fields(
        atmosphere_env,
        "simulator.environment.atmosphere_env",
        _ATMOSPHERE_ENV_ALLOWED_FIELDS,
    )
    extensions = normalized.get("extensions", {}) or {}
    if not isinstance(extensions, dict):
        raise ValueError("simulator.environment.extensions must be a mapping.")
    atmosphere_extensions = atmosphere_env.get("extensions", {}) or {}
    if not isinstance(atmosphere_extensions, dict):
        raise ValueError("simulator.environment.atmosphere_env.extensions must be a mapping.")
    if "atmosphere_env" in normalized:
        normalized["atmosphere_env"] = atmosphere_env
    if "extensions" in normalized:
        normalized["extensions"] = dict(extensions)
    atmosphere_models = {
        "exponential", "ussa1976", "nrlmsise00", "msis86", "msis-86",
        "hpop_msis86", "jacchia70", "jacchia-70", "hpop_jacchia70",
        "jb2006", "jb2008", "harris_priester", "harris-priester", "hp",
        "hpop_harris_priester",
    }
    for container, container_path in (
        (normalized, "simulator.environment"),
        (atmosphere_env, "simulator.environment.atmosphere_env"),
    ):
        if container.get("atmosphere_model") not in (None, ""):
            model = str(container["atmosphere_model"]).strip().lower()
            if model not in atmosphere_models:
                raise ValueError(f"{container_path}.atmosphere_model is unsupported: {model!r}.")
            container["atmosphere_model"] = model
        for key in ("drag_frame_model", "density_frame_model"):
            if container.get(key) not in (None, ""):
                from sim.dynamics.orbit.frames import normalize_frame_model

                configured_model = str(container[key]).strip().lower()
                normalize_frame_model(configured_model)
                container[key] = configured_model
        for key in (
            "density_kg_m3",
            "exponential_reference_density_kg_m3",
            "drag_earth_rotation_rad_s",
        ):
            if container.get(key) is not None:
                container[key] = _parse_float(container[key], f"{container_path}.{key}")
                if key != "drag_earth_rotation_rad_s" and container[key] < 0.0:
                    raise ValueError(f"{container_path}.{key} must be >= 0.")
    if "atmosphere_env" in normalized:
        normalized["atmosphere_env"] = atmosphere_env
    return normalized


def _parse_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a nonnegative integer.")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a nonnegative integer.") from exc
    if not isinstance(value, int) and str(value).strip() not in {str(out), f"+{out}"}:
        raise ValueError(f"{field_name} must be a nonnegative integer.")
    if out < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer.")
    return out


def _normalize_spherical_harmonics_section(value: Any) -> dict[str, Any]:
    path = "simulator.dynamics.orbit.spherical_harmonics"
    raw = _as_dict(value, path)
    _reject_unknown_fields(
        raw,
        path,
        {
            "enabled",
            "degree",
            "order",
            "source",
            "model",
            "coeff_path",
            "source_path",
            "terms",
            "normalized",
            "reference_radius_km",
            "frame_model",
            "eop_path",
            "fd_step_km",
            "allow_download",
        },
    )
    out = dict(raw)
    out["enabled"] = _parse_bool(out.get("enabled", False), f"{path}.enabled")

    if out.get("source") not in (None, "") and out.get("model") not in (None, ""):
        source = str(out["source"]).strip().lower()
        model = str(out["model"]).strip().lower()
        if source != model:
            raise ValueError(f"{path}.source and {path}.model cannot select different gravity models.")
    source = str(out.get("source", out.get("model", "")) or "").strip().lower()
    if not source and (out.get("coeff_path") not in (None, "") or out.get("source_path") not in (None, "")):
        source = "icgem"
    if source:
        out["source"] = source
    out.pop("model", None)

    terms = out.get("terms")
    max_term_degree = -1
    max_term_order = -1
    if terms not in (None, ""):
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"{path}.terms must be a non-empty list of coefficient mappings.")
        normalized_terms: list[dict[str, Any]] = []
        for index, item in enumerate(terms):
            term_path = f"{path}.terms[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{term_path} must be a coefficient mapping.")
            term = dict(item)
            _reject_unknown_fields(term, term_path, {"n", "m", "c_nm", "s_nm", "c", "s", "normalized"})
            if "n" not in term or "m" not in term:
                raise ValueError(f"{term_path} requires integer n and m fields.")
            coefficient_keys = {"c_nm", "s_nm", "c", "s"} & set(term)
            if not coefficient_keys:
                raise ValueError(f"{term_path} requires at least one C or S coefficient field.")
            n = _parse_nonnegative_int(term["n"], f"{term_path}.n")
            m = _parse_nonnegative_int(term["m"], f"{term_path}.m")
            if n < 2:
                raise ValueError(f"{term_path}.n must be >= 2 for a perturbing geopotential term.")
            if m > n:
                raise ValueError(f"{term_path}.m must satisfy 0 <= m <= n.")
            term["n"] = n
            term["m"] = m
            for key in coefficient_keys:
                term[key] = _parse_float(term[key], f"{term_path}.{key}")
            if "normalized" in term:
                term["normalized"] = _parse_bool(term["normalized"], f"{term_path}.normalized")
            normalized_terms.append(term)
            max_term_degree = max(max_term_degree, n)
            max_term_order = max(max_term_order, m)
        out["terms"] = normalized_terms

    if out.get("degree") is not None:
        out["degree"] = _parse_nonnegative_int(out["degree"], f"{path}.degree")
    elif max_term_degree >= 2:
        out["degree"] = max_term_degree
    if out.get("order") is not None:
        out["order"] = _parse_nonnegative_int(out["order"], f"{path}.order")
    elif max_term_order >= 0:
        out["order"] = max_term_order
    elif out.get("degree") is not None:
        out["order"] = int(out["degree"])

    if out.get("degree") is not None and out.get("order") is not None and int(out["order"]) > int(out["degree"]):
        raise ValueError(f"{path}.order must satisfy 0 <= order <= degree.")
    for key in ("normalized", "allow_download"):
        if key in out:
            out[key] = _parse_bool(out[key], f"{path}.{key}")
    for key in ("reference_radius_km", "fd_step_km"):
        if out.get(key) is None:
            continue
        out[key] = _parse_float(out[key], f"{path}.{key}")
        if float(out[key]) <= 0.0:
            raise ValueError(f"{path}.{key} must be positive.")
    return out

def _normalize_reentry_termination_block(
    value: Any,
    path: str,
    *,
    fill_bool_defaults: bool,
) -> dict[str, Any]:
    termination = _as_dict(value, path)
    _reject_unknown_fields(
        termination,
        path,
        {"enabled", "terminate_on_entry", *_REENTRY_TERMINATION_LIMIT_FIELDS},
    )
    if fill_bool_defaults or "enabled" in termination:
        termination["enabled"] = _parse_bool(
            termination.get("enabled", False),
            f"{path}.enabled",
        )
    if fill_bool_defaults or "terminate_on_entry" in termination:
        termination["terminate_on_entry"] = _parse_bool(
            termination.get("terminate_on_entry", False),
            f"{path}.terminate_on_entry",
        )
    for key in _REENTRY_TERMINATION_LIMIT_FIELDS:
        if termination.get(key) is None:
            continue
        termination[key] = _parse_float(termination.get(key), f"{path}.{key}")
        if float(termination[key]) < 0.0:
            raise ValueError(f"{path}.{key} must be >= 0.")
    return termination


def _normalize_reentry_section(dynamics: dict[str, Any]) -> dict[str, Any]:
    out = dict(dynamics or {})
    if "reentry" not in out:
        return out
    raw = _as_dict(out.get("reentry"), "simulator.dynamics.reentry")
    _reject_unknown_fields(
        raw,
        "simulator.dynamics.reentry",
        {
            "enabled",
            "begin_altitude_km",
            "object_ids",
            "nose_radius_m",
            "heat_rate_coefficient",
            "atmosphere_model",
            "termination",
            "numeric_backend",
        },
    )
    normalized = dict(raw)
    if "numeric_backend" in normalized:
        normalized["numeric_backend"] = _normalize_numeric_backend(
            normalized["numeric_backend"], "simulator.dynamics.reentry.numeric_backend",
        )
    normalized["enabled"] = _parse_bool(
        normalized.get("enabled", False),
        "simulator.dynamics.reentry.enabled",
    )
    normalized["begin_altitude_km"] = _parse_float(
        normalized.get("begin_altitude_km", 300.0),
        "simulator.dynamics.reentry.begin_altitude_km",
    )
    if normalized["begin_altitude_km"] < 0.0:
        raise ValueError("simulator.dynamics.reentry.begin_altitude_km must be >= 0.")
    if normalized.get("nose_radius_m") is not None:
        normalized["nose_radius_m"] = _parse_float(
            normalized.get("nose_radius_m"),
            "simulator.dynamics.reentry.nose_radius_m",
        )
        if normalized["nose_radius_m"] <= 0.0:
            raise ValueError("simulator.dynamics.reentry.nose_radius_m must be positive.")
    normalized["heat_rate_coefficient"] = _parse_float(
        normalized.get("heat_rate_coefficient", 1.83e-4),
        "simulator.dynamics.reentry.heat_rate_coefficient",
    )
    if normalized["heat_rate_coefficient"] < 0.0:
        raise ValueError("simulator.dynamics.reentry.heat_rate_coefficient must be >= 0.")
    object_ids = normalized.get("object_ids", [])
    if isinstance(object_ids, str):
        normalized["object_ids"] = [object_ids]
    elif isinstance(object_ids, list):
        normalized["object_ids"] = [str(item) for item in object_ids]
    else:
        raise ValueError("simulator.dynamics.reentry.object_ids must be a string or list of strings.")
    atmosphere_model = normalized.get("atmosphere_model")
    if atmosphere_model is not None:
        atmosphere_model = str(atmosphere_model).strip().lower()
        if atmosphere_model not in {
            "",
            "exponential",
            "ussa1976",
            "msis86",
            "msis-86",
            "hpop_msis86",
            "nrlmsise00",
            "jacchia70",
            "jacchia-70",
            "hpop_jacchia70",
            "jb2006",
            "jb2008",
            "harris_priester",
            "harris-priester",
            "hp",
            "hpop_harris_priester",
        }:
            raise ValueError(
                "simulator.dynamics.reentry.atmosphere_model must be one of: "
                "exponential, ussa1976, msis86, nrlmsise00, jacchia70, jb2006, jb2008, harris_priester."
            )
        normalized["atmosphere_model"] = atmosphere_model
    termination = dict(normalized.get("termination", {}) or {})
    if termination:
        by_object_raw = termination.pop("by_object", {})
        termination = _normalize_reentry_termination_block(
            termination,
            "simulator.dynamics.reentry.termination",
            fill_bool_defaults=True,
        )
        if by_object_raw is not None:
            if not isinstance(by_object_raw, dict):
                raise ValueError("simulator.dynamics.reentry.termination.by_object must be a mapping.")
            by_object: dict[str, Any] = {}
            for object_id, object_termination in by_object_raw.items():
                object_path = f"simulator.dynamics.reentry.termination.by_object.{object_id}"
                by_object[str(object_id)] = _normalize_reentry_termination_block(
                    object_termination,
                    object_path,
                    fill_bool_defaults=False,
                )
            termination["by_object"] = by_object
    normalized["termination"] = termination
    out["reentry"] = normalized
    return out
def _normalize_simulator_termination_block(
    value: Any,
    path: str,
    *,
    fill_defaults: bool,
) -> dict[str, Any]:
    termination = _as_dict(value, path)
    _reject_unknown_fields(termination, path, {"enabled", "earth_impact_enabled", "earth_radius_km"})
    if "enabled" in termination and "earth_impact_enabled" not in termination:
        termination["earth_impact_enabled"] = termination.pop("enabled")
    elif "enabled" in termination:
        termination.pop("enabled")
    if fill_defaults or "earth_impact_enabled" in termination:
        termination["earth_impact_enabled"] = _parse_bool(
            termination.get("earth_impact_enabled", True),
            f"{path}.earth_impact_enabled",
        )
    if fill_defaults or "earth_radius_km" in termination:
        termination["earth_radius_km"] = _parse_float(
            termination.get("earth_radius_km", 6378.137),
            f"{path}.earth_radius_km",
        )
        if float(termination["earth_radius_km"]) <= 0.0:
            raise ValueError(f"{path}.earth_radius_km must be positive.")
    return termination


def _parse_simulator_section(value: Any) -> SimulatorSection:
    d = _as_dict(value, "simulator")
    _reject_unsupported_aliases(d, "simulator", _SIMULATOR_UNSUPPORTED_ALIASES)
    _reject_unknown_fields(
        d,
        "simulator",
        {
            "duration_s",
            "dt_s",
            "initial_jd_utc",
            "resource_profile",
            "acceleration",
            "execution",
            "frames",
            "dynamics",
            "environment",
            "plugin_validation",
            "termination",
            "system_force_models",
            "collisions",
            "central_body",
            "planetary",
            "additional_gravity",
        },
    )
    plugin_validation = {"strict": True}
    plugin_validation.update(dict(d.get("plugin_validation", {}) or {}))
    _reject_unknown_fields(plugin_validation, "simulator.plugin_validation", {"strict", "strict_runtime"})
    termination_raw = dict(d.get("termination", {}) or {})
    termination_by_object_raw = termination_raw.pop("by_object", {})
    termination = {"earth_impact_enabled": True, "earth_radius_km": 6378.137}
    termination.update(termination_raw)
    plugin_validation["strict"] = _parse_bool(
        plugin_validation.get("strict", True), "simulator.plugin_validation.strict"
    )
    plugin_validation["strict_runtime"] = _parse_bool(
        plugin_validation.get("strict_runtime", False), "simulator.plugin_validation.strict_runtime"
    )
    termination = _normalize_simulator_termination_block(
        termination,
        "simulator.termination",
        fill_defaults=True,
    )
    if termination_by_object_raw is not None:
        if not isinstance(termination_by_object_raw, dict):
            raise ValueError("simulator.termination.by_object must be a mapping.")
        by_object: dict[str, Any] = {}
        for object_id, object_termination in termination_by_object_raw.items():
            object_path = f"simulator.termination.by_object.{object_id}"
            by_object[str(object_id)] = _normalize_simulator_termination_block(
                object_termination,
                object_path,
                fill_defaults=False,
            )
        termination["by_object"] = by_object
    body_id = str(d.get("central_body", "earth")).strip().lower()
    planetary = None
    dynamics_raw = dict(d.get("dynamics", {}) or {})
    if body_id != "earth":
        from .planetary import normalize_planetary
        planetary = normalize_planetary(d.get("planetary"), body_id)
        if d.get("frames") or d.get("environment") or d.get("termination"):
            raise ValueError("Non-Earth frames, environment and impact policy belong in simulator.planetary.")
        dynamics_raw.setdefault("attitude", {"enabled": False})
        if isinstance(dynamics_raw["attitude"], dict):
            dynamics_raw["attitude"].setdefault("enabled", False)
        dynamics_raw.setdefault("orbit", {})
        dynamics_raw["orbit"].setdefault("numeric_backend", "rust")
        termination = {"earth_impact_enabled": False, "earth_radius_km": 6378.137, "by_object": {}}
    elif d.get("planetary") is not None:
        raise ValueError("simulator.planetary is only valid with a non-Earth central_body.")
    dynamics = _normalize_dynamics_section(dynamics_raw)
    additional_gravity = None
    if d.get("additional_gravity") is not None:
        from .additional_gravity import normalize_additional_gravity
        additional_gravity = normalize_additional_gravity(d["additional_gravity"], body_id)
    system_force_models = d.get("system_force_models", [])
    if not isinstance(system_force_models, list) or any(item is None for item in system_force_models):
        raise ValueError("simulator.system_force_models must be a list without null entries.")
    collisions = _as_dict(d.get("collisions"), "simulator.collisions")
    _reject_unknown_fields(collisions, "simulator.collisions", {"enabled", "radii_m", "numeric_backend"})
    collision_backend = _normalize_numeric_backend(
        collisions.get("numeric_backend", "rust"), "simulator.collisions.numeric_backend",
    )
    collision_enabled = _parse_bool(collisions.get("enabled", False), "simulator.collisions.enabled")
    radii_raw = collisions.get("radii_m", {}) or {}
    if not isinstance(radii_raw, dict):
        raise ValueError("simulator.collisions.radii_m must be a mapping of object IDs to radii.")
    radii_m = {
        str(object_id): _parse_float(radius, f"simulator.collisions.radii_m.{object_id}")
        for object_id, radius in radii_raw.items()
    }
    if any(not object_id or radius <= 0.0 for object_id, radius in radii_m.items()):
        raise ValueError("simulator.collisions.radii_m requires non-empty object IDs and positive radii.")
    if collision_enabled and len(radii_m) != 2:
        raise ValueError("simulator.collisions requires radii_m for exactly two objects.")
    out = SimulatorSection(
        central_body=body_id,
        planetary=planetary,
        additional_gravity=additional_gravity,
        duration_s=_parse_float(d.get("duration_s", 3600.0), "simulator.duration_s"),
        dt_s=_parse_float(d.get("dt_s", 1.0), "simulator.dt_s"),
        initial_jd_utc=_parse_optional_float(d.get("initial_jd_utc"), "simulator.initial_jd_utc"),
        resource_profile=_parse_resource_profile(d.get("resource_profile"), "simulator.resource_profile"),
        acceleration=_parse_acceleration_section(d.get("acceleration")),
        execution=_parse_simulator_execution_section(d.get("execution")),
        frames=_parse_simulator_frames_section(d.get("frames")),
        dynamics=dynamics,
        environment=_normalize_environment_section(d.get("environment")),
        plugin_validation=plugin_validation,
        termination=termination,
        system_force_models=[_parse_algorithm_pointer(item) for item in system_force_models],
        collisions=(
            {"enabled": True, "radii_m": radii_m, **(
                {"numeric_backend": collision_backend} if "numeric_backend" in collisions else {}
            )}
            if collision_enabled else {}
        ),
    )
    if str(out.frames.model).strip().lower() in {"iau76_80_eop", "iau76_fk5_iau80_eop", "hpop_like", "hpop"}:
        has_manual_eop = any(out.frames.get(key) is not None for key in ("dut1_s", "xp_arcsec", "yp_arcsec", "dat_s"))
        has_manual_eop = has_manual_eop or any(
            float(out.frames.get(key, 0.0) or 0.0) != 0.0 for key in ("ddpsi_rad", "ddeps_rad")
        )
        if (out.frames.eop_path is not None or has_manual_eop) and out.initial_jd_utc is None:
            raise ValueError("simulator.frames EOP settings require simulator.initial_jd_utc for frame rotation.")
    _validate_sim_timing(out)
    return out


def _normalize_numeric_backend(value: Any, field_name: str) -> str:
    return normalize_numeric_backend(value, field_name=field_name)


def _normalize_dynamics_section(value: dict[str, Any]) -> dict[str, Any]:
    dynamics = dict(value or {})
    _reject_unknown_fields(dynamics, "simulator.dynamics", {"orbit", "attitude", "rocket", "reentry"})
    orbit = _as_dict(dynamics.get("orbit"), "simulator.dynamics.orbit")
    _reject_unknown_fields(
        orbit,
        "simulator.dynamics.orbit",
        {
            "model",
            "cr3bp_system",
            "propagation_method",
            "integrator",
            "numeric_backend",
            "adaptive_atol",
            "adaptive_rtol",
            "orbit_substep_s",
            "j2",
            "j3",
            "j4",
            "drag",
            "lift",
            "srp",
            "third_body_sun",
            "third_body_moon",
            "atmosphere_model",
            "drag_frame_model",
            "drag_eop_path",
            "drag_earth_rotation_rad_s",
            "de440_coeff_path",
            "de440_eop_path",
            "spherical_harmonics",
            "ocean_tides",
            "solid_earth_tides",
            "schwarzschild",
            "earth_radiation",
        },
    )
    if "schwarzschild" in orbit:
        path = "simulator.dynamics.orbit.schwarzschild"
        relativity = _as_dict(orbit["schwarzschild"], path)
        _reject_unknown_fields(relativity, path, {"enabled"})
        relativity["enabled"] = _parse_bool(relativity.get("enabled", False), f"{path}.enabled")
        orbit["schwarzschild"] = relativity
    if "earth_radiation" in orbit:
        path = "simulator.dynamics.orbit.earth_radiation"
        radiation = _as_dict(orbit["earth_radiation"], path)
        _reject_unknown_fields(radiation, path, {"enabled", "albedo", "infrared", "quadrature_order"})
        for key, default in (("enabled", False), ("albedo", True), ("infrared", True)):
            radiation[key] = _parse_bool(radiation.get(key, default), f"{path}.{key}")
        order = radiation.get("quadrature_order", 32)
        if isinstance(order, bool) or not isinstance(order, int) or not 8 <= order <= 128:
            raise ValueError(f"{path}.quadrature_order must be an integer from 8 to 128.")
        radiation["quadrature_order"] = order
        if radiation["enabled"] and not radiation["albedo"] and not radiation["infrared"]:
            raise ValueError(f"{path} requires albedo and/or infrared.")
        orbit["earth_radiation"] = radiation
    if "ocean_tides" in orbit:
        path = "simulator.dynamics.orbit.ocean_tides"
        ocean = _as_dict(orbit["ocean_tides"], path)
        _reject_unknown_fields(ocean, path, {"enabled", "pole_tide", "degree", "order", "coeff_path"})
        ocean["enabled"] = _parse_bool(ocean.get("enabled", False), f"{path}.enabled")
        ocean["pole_tide"] = _parse_bool(ocean.get("pole_tide", True), f"{path}.pole_tide")
        for key in ("degree", "order"):
            value = ocean.get(key, 6)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 20:
                raise ValueError(f"{path}.{key} must be an integer from 0 to 20.")
            ocean[key] = value
        if ocean["degree"] < 2 or ocean["order"] > ocean["degree"]:
            raise ValueError(f"{path} requires 2 <= degree <= 20 and order <= degree.")
        if ocean["pole_tide"] and ocean["order"] < 1:
            raise ValueError(f"{path}.pole_tide requires order >= 1.")
        if ocean["enabled"] and (not isinstance(ocean.get("coeff_path"), str) or not ocean["coeff_path"].strip()):
            raise ValueError(f"{path}.coeff_path must name a normalized FES Cnm-Snm file.")
        orbit["ocean_tides"] = ocean
    if "solid_earth_tides" in orbit:
        path = "simulator.dynamics.orbit.solid_earth_tides"
        tides = _as_dict(orbit["solid_earth_tides"], path)
        _reject_unknown_fields(tides, path, {"enabled", "tide_system", "pole_tide"})
        tides["enabled"] = _parse_bool(tides.get("enabled", False), f"{path}.enabled")
        tides["pole_tide"] = _parse_bool(tides.get("pole_tide", True), f"{path}.pole_tide")
        if "tide_system" in tides:
            tides["tide_system"] = str(tides["tide_system"]).strip().lower()
            if tides["tide_system"] not in {"tide_free", "zero_tide"}:
                raise ValueError(f"{path}.tide_system must be tide_free or zero_tide.")
        if tides["enabled"] and "tide_system" not in tides:
            raise ValueError(f"{path}.tide_system must explicitly describe the static gravity field.")
        orbit["solid_earth_tides"] = tides
    if "spherical_harmonics" in orbit:
        orbit["spherical_harmonics"] = _normalize_spherical_harmonics_section(orbit.get("spherical_harmonics"))
    backend = _normalize_numeric_backend(orbit.get("numeric_backend", "rust"),
                                         "simulator.dynamics.orbit.numeric_backend")
    if backend not in {"python", "rust"}:
        raise ValueError("simulator.dynamics.orbit.numeric_backend must be python or rust.")
    if "numeric_backend" in orbit:
        orbit["numeric_backend"] = backend
    for key in ("adaptive_atol", "adaptive_rtol"):
        if orbit.get(key) is not None:
            orbit[key] = _parse_float(orbit[key], f"simulator.dynamics.orbit.{key}")
            if orbit[key] <= 0.0:
                raise ValueError(f"simulator.dynamics.orbit.{key} must be positive.")
    if orbit.get("atmosphere_model") not in (None, ""):
        orbit["atmosphere_model"] = str(orbit["atmosphere_model"]).strip().lower()
        if orbit["atmosphere_model"] not in {
            "exponential", "ussa1976", "nrlmsise00", "msis86", "msis-86",
            "hpop_msis86", "jacchia70", "jacchia-70", "hpop_jacchia70",
            "jb2006", "jb2008", "harris_priester", "harris-priester", "hp",
            "hpop_harris_priester",
        }:
            raise ValueError("simulator.dynamics.orbit.atmosphere_model is unsupported.")
    if orbit.get("drag_frame_model") not in (None, ""):
        from sim.dynamics.orbit.frames import normalize_frame_model

        configured_model = str(orbit["drag_frame_model"]).strip().lower()
        normalize_frame_model(configured_model)
        orbit["drag_frame_model"] = configured_model
    attitude = _as_dict(dynamics.get("attitude"), "simulator.dynamics.attitude")
    _reject_unknown_fields(
        attitude,
        "simulator.dynamics.attitude",
        {"enabled", "attitude_substep_s", "disturbance_torques", "guardrail_policy", "numeric_backend"},
    )
    if "numeric_backend" in attitude:
        attitude["numeric_backend"] = _normalize_numeric_backend(
            attitude["numeric_backend"], "simulator.dynamics.attitude.numeric_backend",
        )
    if "rocket" in dynamics:
        rocket = _as_dict(dynamics["rocket"], "simulator.dynamics.rocket")
        if "numeric_backend" in rocket:
            rocket["numeric_backend"] = _normalize_numeric_backend(
                rocket["numeric_backend"], "simulator.dynamics.rocket.numeric_backend",
            )
        dynamics["rocket"] = rocket
    guardrail_policy = str(attitude.get("guardrail_policy", "error") or "error").strip().lower()
    if guardrail_policy not in {"sanitize", "error"}:
        raise ValueError("simulator.dynamics.attitude.guardrail_policy must be one of: sanitize, error.")
    if "guardrail_policy" in attitude:
        attitude["guardrail_policy"] = guardrail_policy
    dynamics["orbit"] = orbit
    dynamics["attitude"] = attitude
    return _normalize_reentry_section(dynamics)


def _parse_simulator_frames_section(value: Any) -> dict[str, Any]:
    raw = _as_dict(value, "simulator.frames")
    _reject_unknown_fields(
        raw,
        "simulator.frames",
        {
            "model",
            "frame_model",
            "eop_path",
            "eop_extrapolation",
            "time_scale_model",
            "tt_minus_utc_s",
            "dut1_s",
            "xp_arcsec",
            "yp_arcsec",
            "dat_s",
            "ddpsi_rad",
            "ddeps_rad",
        },
    )
    out = dict(raw)
    model = str(out.get("model", out.get("frame_model", "simple_gmst")) or "simple_gmst").strip().lower()
    allowed = {
        "simple",
        "simple_gmst",
        "simple_earth_rotation",
        "gmst",
        "hpop_like",
        "hpop",
        "iau76_80_eop",
        "iau76_fk5_iau80_eop",
    }
    if model not in allowed:
        raise ValueError(
            "simulator.frames.model must be one of: simple_gmst, simple_earth_rotation, "
            "hpop_like, iau76_80_eop."
        )
    out["model"] = model
    for key in ("tt_minus_utc_s", "dut1_s", "xp_arcsec", "yp_arcsec", "dat_s", "ddpsi_rad", "ddeps_rad"):
        if out.get(key) is not None:
            out[key] = _parse_float(out.get(key), f"simulator.frames.{key}")
    if out.get("eop_path") in ("",):
        out["eop_path"] = None
    eop_extrapolation = str(out.get("eop_extrapolation", "error") or "error").strip().lower()
    if eop_extrapolation not in {"error", "hold"}:
        raise ValueError("simulator.frames.eop_extrapolation must be 'error' or 'hold'.")
    out["eop_extrapolation"] = eop_extrapolation
    if out.get("time_scale_model") is not None:
        out["time_scale_model"] = str(out.get("time_scale_model"))
    return out


def _parse_resource_profile(value: Any, field_name: str) -> str | None:
    if value in (None, ""):
        return None
    profile = str(value).strip().lower()
    if profile not in {"config", "laptop-safe", "standard", "aggressive", "off"}:
        raise ValueError(f"{field_name} must be one of: config, laptop-safe, standard, aggressive, off.")
    return profile


def _parse_acceleration_section(value: Any) -> dict[str, Any]:
    d = _as_dict(value, "simulator.acceleration")
    _reject_unknown_fields(d, "simulator.acceleration", {"mode", "warmup", "env_override"})
    mode = str(d.get("mode", "off") or "off").strip().lower()
    if mode not in {"off", "auto", "numba"}:
        raise ValueError("simulator.acceleration.mode must be one of: off, auto, numba.")
    return {
        **d,
        "mode": mode,
        "warmup": _parse_bool(d.get("warmup", False), "simulator.acceleration.warmup"),
        "env_override": _parse_bool(d.get("env_override", True), "simulator.acceleration.env_override"),
    }


def _parse_simulator_execution_section(value: Any) -> dict[str, Any]:
    d = _as_dict(value, "simulator.execution")
    _reject_unknown_fields(
        d,
        "simulator.execution",
        {"policy", "object_parallelism", "runtime_profiler", "controller"},
    )
    object_parallelism = _as_dict(d.get("object_parallelism"), "simulator.execution.object_parallelism")
    runtime_profiler = _as_dict(d.get("runtime_profiler"), "simulator.execution.runtime_profiler")
    controller = _as_dict(d.get("controller"), "simulator.execution.controller")
    backend = str(object_parallelism.get("backend", "serial") or "serial").strip().lower()
    if backend not in {"serial", "process_pool", "thread_pool"}:
        raise ValueError("simulator.execution.object_parallelism.backend must be one of: serial, process_pool, thread_pool.")
    thread_scope = str(object_parallelism.get("thread_scope", "auto") or "auto").strip().lower()
    if thread_scope not in {"auto", "object_step", "native_history"}:
        raise ValueError(
            "simulator.execution.object_parallelism.thread_scope must be one of: auto, object_step, native_history."
        )
    raw_workers = object_parallelism.get("workers", 0)
    raw_max_workers = object_parallelism.get("max_workers", 0)
    raw_reserve_workers = object_parallelism.get("reserve_workers", 1)
    raw_min_objects = object_parallelism.get("min_objects", 3)
    workers = _parse_nonnegative_int(
        0 if raw_workers is None else raw_workers,
        "simulator.execution.object_parallelism.workers",
    )
    max_workers = _parse_nonnegative_int(
        0 if raw_max_workers is None else raw_max_workers,
        "simulator.execution.object_parallelism.max_workers",
    )
    reserve_workers = _parse_nonnegative_int(
        0 if raw_reserve_workers is None else raw_reserve_workers,
        "simulator.execution.object_parallelism.reserve_workers",
    )
    min_objects = _parse_nonnegative_int(
        0 if raw_min_objects is None else raw_min_objects,
        "simulator.execution.object_parallelism.min_objects",
    )
    if workers < 0:
        raise ValueError("simulator.execution.object_parallelism.workers must be >= 0.")
    if max_workers < 0:
        raise ValueError("simulator.execution.object_parallelism.max_workers must be >= 0.")
    if reserve_workers < 0:
        raise ValueError("simulator.execution.object_parallelism.reserve_workers must be >= 0.")
    if min_objects < 1:
        raise ValueError("simulator.execution.object_parallelism.min_objects must be >= 1.")
    policy = str(d.get("policy", "configured") or "configured").strip().lower()
    raw_enabled = object_parallelism.get("enabled", False)
    if isinstance(raw_enabled, str) and raw_enabled.strip().lower() == "auto":
        if policy not in {"configured", "auto"}:
            raise ValueError(
                "simulator.execution.object_parallelism.enabled=auto conflicts with "
                f"simulator.execution.policy={policy!r}."
            )
        policy = "auto"
        enabled = True
    else:
        enabled = _parse_bool(
            raw_enabled,
            "simulator.execution.object_parallelism.enabled",
        )
    if policy not in {"configured", "serial", "parallel", "auto"}:
        raise ValueError(
            "simulator.execution.policy must be one of: auto, configured, parallel, serial."
        )
    if backend == "thread_pool" and policy == "auto":
        raise ValueError("simulator.execution.object_parallelism.backend=thread_pool requires policy=configured or parallel.")
    if policy == "serial":
        enabled = False
        backend = "serial"
    elif policy in {"parallel", "auto"}:
        enabled = True
        if backend != "thread_pool":
            backend = "process_pool"
    orbit_budget_ms = _parse_float(
        controller.get("orbit_budget_ms", 2.0),
        "simulator.execution.controller.orbit_budget_ms",
    )
    attitude_budget_ms = _parse_float(
        controller.get("attitude_budget_ms", 2.0),
        "simulator.execution.controller.attitude_budget_ms",
    )
    if orbit_budget_ms <= 0.0 or attitude_budget_ms <= 0.0:
        raise ValueError("simulator.execution.controller budgets must be positive.")
    deadline_policy = str(controller.get("deadline_policy", "record") or "record").strip().lower()
    if deadline_policy not in {"record", "zero_command", "error"}:
        raise ValueError(
            "simulator.execution.controller.deadline_policy must be one of: error, record, zero_command."
        )
    return {
        **d,
        "policy": policy,
        "object_parallelism": {
            **object_parallelism,
            "enabled": enabled,
            "backend": backend,
            "workers": workers,
            "max_workers": max_workers,
            "reserve_workers": reserve_workers,
            "min_objects": min_objects,
            "thread_scope": thread_scope,
        },
        "runtime_profiler": {
            **runtime_profiler,
            "enabled": _parse_bool(
                runtime_profiler.get("enabled", True),
                "simulator.execution.runtime_profiler.enabled",
            ),
        },
        "controller": {
            **controller,
            "orbit_budget_ms": orbit_budget_ms,
            "attitude_budget_ms": attitude_budget_ms,
            "deadline_policy": deadline_policy,
        },
    }
