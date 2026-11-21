"""Public orbit-dynamics API with lazy implementation imports.

Importing any orbit submodule used to initialize every propagation family,
atmosphere backend, and optional utility.  Spawned ONP workers therefore paid
the memory and startup cost of OGP, SDP4, Lambert, SPICE, and all atmosphere
models even when their scenario did not use them.  PEP 562 lazy attributes keep
the same package API while loading only requested implementations.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {}


def _export(module: str, *names: str) -> None:
    for name in names:
        _EXPORTS[name] = (module, name)


_export(
    "sim.dynamics.orbit.accelerations",
    "OrbitContext",
    "accel_drag",
    "accel_j2",
    "accel_j3",
    "accel_j4",
    "accel_srp",
    "accel_third_body",
    "accel_two_body",
)
_export(
    "sim.dynamics.orbit.atmosphere",
    "AtmosphereModelName",
    "density_exponential",
    "density_from_model",
    "density_harris_priester",
    "density_jacchia70",
    "density_jb2006",
    "density_jb2008",
    "density_msis86",
    "density_nrlmsise00",
    "density_ussa1976",
)
_export(
    "sim.dynamics.orbit.cr3bp",
    "CR3BPSystem",
    "EARTH_MOON_CR3BP",
    "EARTH_MOON_DISTANCE_KM",
    "EARTH_MOON_MEAN_MOTION_RAD_S",
    "EARTH_MOON_MU",
    "cr3bp_halo_seed_state_km_s",
    "cr3bp_l1_position_km",
    "cr3bp_l1_state_km_s",
    "cr3bp_system",
    "propagate_cr3bp_state",
)
_export(
    "sim.dynamics.orbit.cr3bp_research",
    "transform_cr3bp_state",
    "cr3bp_effective_potential",
    "cr3bp_jacobi_constant",
    "cr3bp_jacobi_diagnostics",
    "cr3bp_libration_points",
    "cr3bp_zero_velocity_grid",
)
_export("sim.dynamics.orbit.eclipse", "srp_shadow_factor")
_export(
    "sim.dynamics.orbit.elements",
    "ClassicalOrbitalElements",
    "OrbitalElementFeedbackResult",
    "coe_to_rv_eci",
    "coes_mapping_to_element_targets",
    "coes_mapping_to_rv_eci",
    "coes_target_state_at_current_true_anomaly",
    "orbital_element_feedback_accel",
    "rv_to_coe_eci",
)
_export(
    "sim.dynamics.orbit.environment",
    "EARTH_J2",
    "EARTH_J3",
    "EARTH_J4",
    "EARTH_MU_KM3_S2",
    "EARTH_RADIUS_KM",
)
_export(
    "sim.dynamics.orbit.epoch",
    "datetime_to_julian_date",
    "gmst_angle_rad_from_jd",
    "julian_date_to_datetime",
    "moon_position_eci_km_enhanced",
    "moon_position_eci_km_simple",
    "resolve_body_position_eci_km",
    "resolve_sun_moon_positions",
    "resolve_time_dependent_env",
    "resolved_jd_utc",
    "sun_position_eci_km_enhanced",
    "sun_position_eci_km_simple",
)
_export(
    "sim.dynamics.orbit.frames",
    "FRAME_MODEL_IAU76_80_EOP",
    "FRAME_MODEL_SIMPLE_GMST",
    "FrameContext",
    "frame_context_from_environment",
    "frame_context_from_mapping",
    "normalize_frame_model",
    "rotation_between",
    "teme_to_eci_matrix_vallado_iau80",
    "teme_to_eci_vallado_iau80",
    "transform_position",
    "transform_state",
)
_export("sim.dynamics.orbit.lambert", "LambertSolution", "solve_lambert_universal_variable")
_export("sim.dynamics.orbit.lambert_multirevolution", "MultiRevolutionLambertBranch",
        "MultiRevolutionLambertResult", "solve_multi_revolution_lambert")
_export(
    "sim.dynamics.orbit.ogp",
    "OGP_DEEP_SPACE_PERIOD_THRESHOLD_MIN",
    "ogp_propagate_teme",
    "ogp_propagate_teme_batch_accelerated",
    "ogp_propagate_teme_batch_reference",
    "ogp_propagator_name_for_elements",
    "ogp_regime_for_elements",
)
_export(
    "sim.dynamics.orbit.propagator",
    "OrbitPropagator",
    "drag_plugin",
    "j2_plugin",
    "j3_plugin",
    "j4_plugin",
    "spherical_harmonics_plugin",
    "srp_plugin",
    "third_body_moon_plugin",
    "third_body_planets_plugin",
    "third_body_sun_plugin",
)
_export(
    "sim.dynamics.orbit.relative_linear",
    "RELATIVE_LINEAR_MODELS",
    "RelativeLinearDynamics",
    "normalize_relative_linear_model",
)
_export(
    "sim.dynamics.orbit.sdp4",
    "SDP4Context",
    "sdp4_initialize",
    "sdp4_propagate_teme_from_context",
)
_export(
    "sim.dynamics.orbit.sgp4",
    "SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN",
    "SGP4BatchResult",
    "SGP4EphemerisProvider",
    "SGP4EphemerisState",
    "SGP4State",
    "sgp4_propagate_teme",
    "sgp4_propagate_teme_batch_numba",
    "sgp4_propagate_teme_batch_reference",
)
_export(
    "sim.dynamics.orbit.spherical_harmonics",
    "SphericalHarmonicTerm",
    "accel_spherical_harmonics_terms",
    "load_hpop_ggm03_terms",
    "load_icgem_gfc_terms",
    "load_real_earth_gravity_terms",
    "parse_spherical_harmonic_terms",
)
_export(
    "sim.dynamics.orbit.spice",
    "spice_sun_moon_positions_eci_km",
    "spice_supported_body_names",
)

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
