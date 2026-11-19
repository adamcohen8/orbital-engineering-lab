"""Rocket stack, guidance, and runtime construction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from sim.aero import aero_spec_get, resolve_vehicle_aero_properties
from sim.config import SimulationScenarioConfig
from sim.core.models import StateBelief
from sim.digital_twin.mass_properties import resolve_inertia_kg_m2
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import frame_context_from_environment, frame_context_from_mapping, transform_position
from sim.numeric_backend import normalize_numeric_backend
from sim.presets.rockets import BASIC_1ST_STAGE, BASIC_SSTO_ROCKET, BASIC_TWO_STAGE_STACK, RocketStackPreset
from sim.rocket import (
    LandingGear,
    MaxQThrottleLimiterGuidance,
    OpenLoopPitchProgramGuidance,
    OrbitInsertionCutoffGuidance,
    RocketAscentSimulator,
    RocketGuidanceLaw,
    RocketSimConfig,
    RocketTouchdownLimits,
    RocketVehicleConfig,
    TVCSteeringGuidance,
)
from sim.rocket.aero import RocketAeroConfig
from sim.rocket.navigation import _orbital_elements_basic as _navigation_orbital_elements_basic
from sim.runtime.commands import _rocket_state_to_truth
from sim.runtime.compat import _module_obj
from sim.runtime.models import AgentRuntime
from sim.utils.geodesy import ecef_to_geodetic_deg_km


def _earth_impact_policy_for_object(termination: Any, object_id: str) -> dict[str, Any]:
    root = dict(termination or {})
    policy = {
        "earth_impact_enabled": bool(root.get("earth_impact_enabled", True)),
        "earth_radius_km": float(root.get("earth_radius_km", EARTH_RADIUS_KM)),
    }
    by_object = root.get("by_object", {}) or {}
    if not isinstance(by_object, dict):
        return policy
    for key in ("*", "all", str(object_id)):
        override = by_object.get(key)
        if not isinstance(override, dict):
            continue
        if "earth_impact_enabled" in override:
            policy["earth_impact_enabled"] = bool(override.get("earth_impact_enabled"))
        elif "enabled" in override:
            policy["earth_impact_enabled"] = bool(override.get("enabled"))
        if override.get("earth_radius_km") is not None:
            policy["earth_radius_km"] = float(override.get("earth_radius_km"))
    object_policy = by_object.get(str(object_id), {})
    if isinstance(object_policy, dict) and "touchdown" in object_policy:
        policy["touchdown"] = object_policy["touchdown"]
    return policy


def _resolve_rocket_stack(specs: dict[str, Any]) -> RocketStackPreset:
    preset = str(specs.get("preset_stack", "BASIC_TWO_STAGE_STACK")).strip().upper()
    ssto_stack = RocketStackPreset(name="Basic SSTO Stack", stages=(BASIC_SSTO_ROCKET,))
    by_name: dict[str, RocketStackPreset] = {
        "BASIC_TWO_STAGE_STACK": BASIC_TWO_STAGE_STACK,
        "BASIC_SSTO_STACK": ssto_stack,
        "BASIC_SSTO_ROCKET": ssto_stack,
        "BASIC_1ST_STAGE_STACK": RocketStackPreset(name="Basic 1st Stage Stack", stages=(BASIC_1ST_STAGE,)),
    }
    if preset not in by_name:
        valid = ", ".join(sorted(by_name.keys()))
        raise ValueError(f"Unknown rocket.specs.preset_stack '{preset}'. Valid options: {valid}")
    stack = by_name[preset]
    scales = {
        "dry_mass_scale": float(specs.get("dry_mass_scale", 1.0)),
        "propellant_mass_scale": float(specs.get("propellant_mass_scale", 1.0)),
        "thrust_scale": float(specs.get("thrust_scale", 1.0)),
        "isp_scale": float(specs.get("isp_scale", 1.0)),
    }
    if any((not np.isfinite(value)) or value <= 0.0 for value in scales.values()):
        raise ValueError("Rocket stage performance scales must be finite and positive.")
    if all(abs(value - 1.0) <= 1e-15 for value in scales.values()):
        return stack
    stages = tuple(
        replace(
            stage,
            dry_mass_kg=float(stage.dry_mass_kg) * scales["dry_mass_scale"],
            propellant_mass_kg=float(stage.propellant_mass_kg) * scales["propellant_mass_scale"],
            max_thrust_n=float(stage.max_thrust_n) * scales["thrust_scale"],
            isp_s=float(stage.isp_s) * scales["isp_scale"],
            sea_level_thrust_n=(
                None
                if stage.sea_level_thrust_n is None
                else float(stage.sea_level_thrust_n) * scales["thrust_scale"]
            ),
            vacuum_thrust_n=(
                None
                if stage.vacuum_thrust_n is None
                else float(stage.vacuum_thrust_n) * scales["thrust_scale"]
            ),
            sea_level_isp_s=(
                None if stage.sea_level_isp_s is None else float(stage.sea_level_isp_s) * scales["isp_scale"]
            ),
            vacuum_isp_s=(
                None if stage.vacuum_isp_s is None else float(stage.vacuum_isp_s) * scales["isp_scale"]
            ),
        )
        for stage in stack.stages
    )
    return RocketStackPreset(name=f"{stack.name} (scaled)", stages=stages)


def _build_rocket_guidance(agent_cfg: Any) -> RocketGuidanceLaw:
    base_pointer = getattr(agent_cfg, "base_guidance", None) or getattr(agent_cfg, "guidance", None)
    guidance = _module_obj(base_pointer) or OpenLoopPitchProgramGuidance()
    for modifier_pointer in list(getattr(agent_cfg, "guidance_modifiers", []) or []):
        modifier_obj = _module_obj(modifier_pointer, extra_kwargs={"base_guidance": guidance})
        if modifier_obj is not None:
            guidance = modifier_obj
    return guidance


def _create_rocket_runtime(
    cfg: SimulationScenarioConfig,
    object_id: str = "rocket",
    agent_cfg: Any | None = None,
) -> AgentRuntime:
    rc = cfg.rocket if agent_cfg is None else agent_cfg
    r_init = dict(rc.initial_state or {})
    r_specs = dict(rc.specs or {})
    orbit_dyn = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    att_dyn = dict(cfg.simulator.dynamics.get("attitude", {}) or {})
    rocket_dyn = dict(cfg.simulator.dynamics.get("rocket", {}) or {})
    object_aero = resolve_vehicle_aero_properties(
        r_specs,
        default_reference_area_m2=10.0,
        default_cd=0.20,
        default_cl=0.0,
        default_nose_radius_m=0.5,
        default_reference_length_m=30.0,
    )
    aero_dyn = dict(rocket_dyn.get("aero", {}) or {})
    rocket_area_ref_m2 = (
        float(rocket_dyn.get("area_ref_m2"))
        if rocket_dyn.get("area_ref_m2") is not None
        else (
            object_aero.reference_area_m2
            if aero_spec_get(r_specs, ("reference_area_m2", "area_ref_m2", "area_m2")) is not None
            else None
        )
    )
    atmosphere_env = dict(cfg.simulator.environment.get("atmosphere_env", {}) or {})
    earth_impact_policy = _earth_impact_policy_for_object(cfg.simulator.termination, object_id)
    touchdown = earth_impact_policy.get("touchdown")
    if touchdown is not None:
        atmosphere_env["frame_context"] = frame_context_from_mapping(
            dict(cfg.simulator.frames or {}), jd_utc_start=cfg.simulator.initial_jd_utc,
        )
        if cfg.simulator.initial_jd_utc is not None:
            atmosphere_env["jd_utc_start"] = cfg.simulator.initial_jd_utc
    aero_cfg = RocketAeroConfig(
        enabled=bool(rocket_dyn.get("aero_model_enabled", True)),
        reference_area_m2=float(aero_dyn.get("reference_area_m2", object_aero.reference_area_m2)),
        reference_length_m=float(aero_dyn.get("reference_length_m", object_aero.reference_length_m)),
        cp_offset_body_m=np.array(aero_dyn.get("cp_offset_body_m", object_aero.cp_offset_body_m), dtype=float),
        cd_base=float(aero_dyn.get("cd_base", aero_dyn.get("cd", object_aero.cd))),
        cd_alpha2=float(aero_dyn.get("cd_alpha2", 0.10)),
        cd_supersonic=float(aero_dyn.get("cd_supersonic", 0.28)),
        transonic_peak_cd=float(aero_dyn.get("transonic_peak_cd", 0.22)),
        transonic_mach=float(aero_dyn.get("transonic_mach", 1.0)),
        transonic_width=float(aero_dyn.get("transonic_width", 0.22)),
        cl_alpha_per_rad=float(aero_dyn.get("cl_alpha_per_rad", 0.15)),
        cy_beta_per_rad=float(aero_dyn.get("cy_beta_per_rad", 0.15)),
        cm_alpha_per_rad=float(aero_dyn.get("cm_alpha_per_rad", -0.02)),
        cn_beta_per_rad=float(aero_dyn.get("cn_beta_per_rad", -0.02)),
        cl_roll_per_rad=float(aero_dyn.get("cl_roll_per_rad", -0.01)),
        alpha_limit_deg=float(aero_dyn.get("alpha_limit_deg", 20.0)),
        beta_limit_deg=float(aero_dyn.get("beta_limit_deg", 20.0)),
    )
    rocket_inertia_kg_m2 = resolve_inertia_kg_m2(
        r_specs,
        default=np.array([[8.0e5, 0.0, 0.0], [0.0, 8.0e5, 0.0], [0.0, 0.0, 2.0e4]], dtype=float),
    )
    sim_cfg = RocketSimConfig(
        numeric_backend=normalize_numeric_backend(
            rocket_dyn.get("numeric_backend", "rust"),
            field_name="simulator.dynamics.rocket.numeric_backend",
        ),
        dt_s=float(cfg.simulator.dt_s),
        max_time_s=float(cfg.simulator.duration_s),
        target_altitude_km=float(rocket_dyn.get("target_altitude_km", 400.0)),
        target_altitude_tolerance_km=float(rocket_dyn.get("target_altitude_tolerance_km", 25.0)),
        target_eccentricity_max=float(rocket_dyn.get("target_eccentricity_max", 0.02)),
        insertion_hold_time_s=float(rocket_dyn.get("insertion_hold_time_s", 30.0)),
        launch_lat_deg=float(r_init.get("launch_lat_deg", 0.0)),
        launch_lon_deg=float(r_init.get("launch_lon_deg", 0.0)),
        launch_alt_km=float(r_init.get("launch_alt_km", 0.0)),
        launch_azimuth_deg=float(r_init.get("launch_azimuth_deg", 90.0)),
        atmosphere_model=str(rocket_dyn.get("atmosphere_model", "ussa1976")),
        enable_drag=bool(orbit_dyn.get("drag", True)),
        enable_srp=bool(orbit_dyn.get("srp", False)),
        enable_j2=bool(orbit_dyn.get("j2", True)),
        enable_j3=bool(orbit_dyn.get("j3", False)),
        enable_j4=bool(orbit_dyn.get("j4", False)),
        terminate_on_earth_impact=bool(earth_impact_policy.get("earth_impact_enabled", True)),
        earth_impact_radius_km=float(earth_impact_policy.get("earth_radius_km", 6378.137)),
        touchdown=RocketTouchdownLimits(**touchdown) if touchdown is not None else None,
        area_ref_m2=rocket_area_ref_m2,
        use_stagewise_aero_geometry=bool(rocket_dyn.get("use_stagewise_aero_geometry", True)),
        cd=float(rocket_dyn.get("cd", 0.35)),
        cr=float(rocket_dyn.get("cr", 1.2)),
        aero=aero_cfg,
        atmosphere_env=atmosphere_env,
        use_wgs84_geodesy=bool(rocket_dyn.get("use_wgs84_geodesy", True)),
        wind_enu_m_s=np.array(rocket_dyn.get("wind_enu_m_s", [0.0, 0.0, 0.0]), dtype=float),
        inertia_kg_m2=rocket_inertia_kg_m2,
        attitude_substep_s=float(rocket_dyn.get("attitude_substep_s", att_dyn.get("attitude_substep_s", 0.02)) or 0.02),
        attitude_mode=str(rocket_dyn.get("attitude_mode", "dynamic")),
        tvc_time_constant_s=float(rocket_dyn.get("tvc_time_constant_s", 0.1)),
        tvc_max_gimbal_deg=float(rocket_dyn.get("tvc_max_gimbal_deg", 6.0)),
        tvc_rate_limit_deg_s=float(rocket_dyn.get("tvc_rate_limit_deg_s", 20.0)),
        tvc_pivot_offset_body_m=np.array(rocket_dyn.get("tvc_pivot_offset_body_m", [0.0, 0.0, 0.0]), dtype=float),
    )
    vehicle_cfg = RocketVehicleConfig(
        stack=_resolve_rocket_stack(dict(rc.specs or {})),
        payload_mass_kg=float(r_specs.get("payload_mass_kg", 150.0)),
        thrust_axis_body=np.array(r_specs.get("thrust_axis_body", [1.0, 0.0, 0.0]), dtype=float),
        landing_gear=LandingGear.from_mapping(r_specs["landing_gear"]) if "landing_gear" in r_specs else None,
    )
    guidance = _build_rocket_guidance(rc)
    if bool(rocket_dyn.get("tvc_steering_enabled", False)):
        guidance = TVCSteeringGuidance(
            base_guidance=guidance, pass_through_attitude=bool(rocket_dyn.get("tvc_pass_through_attitude", True))
        )
    if bool(rocket_dyn.get("orbit_insertion_cutoff_enabled", False)):
        guidance = OrbitInsertionCutoffGuidance(
            base_guidance=guidance,
            min_cutoff_alt_km=float(rocket_dyn.get("cutoff_min_alt_km", 80.0)),
            min_periapsis_alt_km=float(rocket_dyn.get("cutoff_min_periapsis_alt_km", 120.0)),
            apoapsis_margin_km=float(rocket_dyn.get("cutoff_apoapsis_margin_km", 5.0)),
            energy_margin_km2_s2=float(rocket_dyn.get("cutoff_energy_margin_km2_s2", 0.0)),
            ecc_relax_factor=float(rocket_dyn.get("cutoff_ecc_relax_factor", 2.0)),
            hard_escape_cutoff=bool(rocket_dyn.get("cutoff_hard_escape_enabled", True)),
            near_escape_speed_margin_frac=float(rocket_dyn.get("cutoff_near_escape_speed_margin_frac", 0.03)),
        )
    if bool(rocket_dyn.get("max_q_limiter_enabled", False)):
        guidance = MaxQThrottleLimiterGuidance(
            base_guidance=guidance,
            max_q_pa=float(rocket_dyn.get("max_q_pa", 45000.0)),
            min_throttle=float(rocket_dyn.get("min_throttle", 0.0)),
        )
    rocket_sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=guidance)
    rocket_state = rocket_sim.initial_state()
    truth = _rocket_state_to_truth(rocket_state)
    belief = StateBelief(
        state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
        covariance=np.eye(6) * 1e-4,
        last_update_t_s=0.0,
    )
    bridge = _module_obj(rc.bridge) if (rc.bridge is not None and rc.bridge.enabled) else None
    mission_strategy = _module_obj(getattr(rc, "mission_strategy", None))
    mission_execution = _module_obj(getattr(rc, "mission_execution", None))
    mission_modules = [_module_obj(pointer) for pointer in list(rc.mission_objectives or [])]
    mission_modules = [module for module in mission_modules if module is not None]
    return AgentRuntime(
        object_id=object_id,
        kind="rocket",
        enabled=bool(rc.enabled),
        active=bool(rc.enabled),
        truth=truth,
        belief=belief,
        sensor=None,
        estimator=None,
        orbit_controller=None,
        attitude_controller=None,
        dynamics=None,
        knowledge_base=None,
        bridge=bridge,
        mission_strategy=mission_strategy,
        mission_execution=mission_execution,
        rocket_sim=rocket_sim,
        rocket_state=rocket_state,
        rocket_guidance=guidance,
        deploy_source=None,
        deploy_time_s=None,
        deploy_dv_body_m_s=None,
        initialization_delay_s=0.0,
        control_available_time_s=0.0,
        mission_modules=mission_modules,
        waiting_for_launch=False,
        orbital_isp_s=None,
        dry_mass_kg=None,
        fuel_capacity_kg=None,
        orbital_max_thrust_n=None,
        thruster_direction_body=None,
        thruster_position_body_m=None,
        mass_properties=dict(r_specs.get("mass_properties", {}) or {}),
    )


def _rocket_altitude_km(r_eci_km: np.ndarray, t_s: float, sim_cfg: RocketSimConfig) -> float:
    if not bool(getattr(sim_cfg, "use_wgs84_geodesy", False)):
        return float(np.linalg.norm(r_eci_km) - EARTH_RADIUS_KM)
    frame_context = frame_context_from_environment(dict(getattr(sim_cfg, "atmosphere_env", {}) or {}))
    r_ecef = transform_position(np.array(r_eci_km, dtype=float), "eci", "ecef", t_s=float(t_s), context=frame_context)
    _, _, alt_km = ecef_to_geodetic_deg_km(r_ecef)
    return float(alt_km)


def _orbital_elements_basic(
    r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = EARTH_MU_KM3_S2
) -> tuple[float, float]:
    if (
        type(r_km) is np.ndarray
        and type(v_km_s) is np.ndarray
        and r_km.shape == (3,)
        and v_km_s.shape == (3,)
        and r_km.dtype == np.dtype("float64")
        and v_km_s.dtype == np.dtype("float64")
    ):
        return _navigation_orbital_elements_basic(r_km, v_km_s, mu_km3_s2)
    r = float(np.linalg.norm(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu_km3_s2 / r
    a = np.inf if abs(eps) < 1e-14 else float(-mu_km3_s2 / (2.0 * eps))
    h = np.cross(r_km, v_km_s)
    e_vec = np.cross(v_km_s, h) / mu_km3_s2 - r_km / r
    return a, float(np.linalg.norm(e_vec))
