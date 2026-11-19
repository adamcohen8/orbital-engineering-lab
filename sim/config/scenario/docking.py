"""Configuration and bounded runtime contract for cooperative docking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import SimulationScenarioConfig
from .primitives import _as_dict, _parse_bool, _parse_float, _reject_unknown_fields

_PATH = "simulator.docking"
_LATCH_FIELDS = (
    "capture_lateral_m", "capture_angle_rad", "capture_speed_m_s", "capture_rate_rad_s",
    "stiffness_n_m", "damping_n_s_m", "angular_stiffness_n_m_rad", "angular_damping_n_m_s_rad",
    "max_travel_m", "max_force_n", "max_torque_n_m",
)
_LOCK_FIELDS = ("position_m", "angle_rad", "speed_m_s", "rate_rad_s", "dwell_s")


def _positive_fields(block, path, names):
    values = {name: _parse_float(block.get(name), f"{path}.{name}") for name in names}
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{path}.{name} must be positive.")
    return values


def _vector(value, path, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{path} must contain {size} numbers.")
    return [_parse_float(item, f"{path}[{index}]") for index, item in enumerate(value)]


def normalize_docking(value: Any) -> dict[str, Any]:
    raw = _as_dict(value, _PATH)
    _reject_unknown_fields(raw, _PATH, {
        "enabled", "probe", "receiver", "max_step_s", "contact_stiffness_n_m",
        "contact_damping_n_s_m", "max_compression_m", "latch", "hard_mate", "commanded",
    })
    if not _parse_bool(raw.get("enabled", False), f"{_PATH}.enabled"):
        if set(raw) - {"enabled"}:
            raise ValueError(f"{_PATH} parameters require enabled: true.")
        return {}

    normalized = {"enabled": True, **_positive_fields(
        raw, _PATH, ("max_step_s", "contact_stiffness_n_m", "max_compression_m"),
    )}
    if _parse_bool(raw.get("commanded", False), f"{_PATH}.commanded"):
        normalized["commanded"] = True
    damping = _parse_float(raw.get("contact_damping_n_s_m", 0.0), f"{_PATH}.contact_damping_n_s_m")
    if damping < 0:
        raise ValueError(f"{_PATH}.contact_damping_n_s_m must be nonnegative.")
    normalized["contact_damping_n_s_m"] = damping
    for role in ("probe", "receiver"):
        path = f"{_PATH}.{role}"
        port = _as_dict(raw.get(role), path)
        _reject_unknown_fields(port, path, {"object_id", "offset_body_m", "quat_port_from_body", "radius_m"})
        object_id = port.get("object_id")
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError(f"{path}.object_id must name a spacecraft.")
        quat = _vector(port.get("quat_port_from_body", [1, 0, 0, 0]), f"{path}.quat_port_from_body", 4)
        if not math.isclose(math.hypot(*quat), 1.0, abs_tol=1e-10, rel_tol=0):
            raise ValueError(f"{path}.quat_port_from_body must be a unit quaternion.")
        normalized[role] = {
            "object_id": object_id,
            "offset_body_m": _vector(port.get("offset_body_m"), f"{path}.offset_body_m", 3),
            "quat_port_from_body": quat,
            **_positive_fields(port, path, ("radius_m",)),
        }
    if normalized["probe"]["object_id"] == normalized["receiver"]["object_id"]:
        raise ValueError(f"{_PATH} requires distinct probe and receiver objects.")
    if normalized["probe"]["radius_m"] > normalized["receiver"]["radius_m"]:
        raise ValueError(f"{_PATH}.receiver.radius_m must cover the centered probe ring.")

    for name, fields in (("latch", _LATCH_FIELDS), ("hard_mate", _LOCK_FIELDS)):
        block = _as_dict(raw.get(name), f"{_PATH}.{name}")
        _reject_unknown_fields(block, f"{_PATH}.{name}", set(fields))
        if block:
            normalized[name] = _positive_fields(block, f"{_PATH}.{name}", fields)
    latch = normalized.get("latch")
    if normalized.get("commanded") and not latch:
        raise ValueError(f"{_PATH}.commanded requires a latch.")
    lock = normalized.get("hard_mate")
    if latch and latch["capture_angle_rad"] >= math.pi / 2:
        raise ValueError(f"{_PATH}.latch.capture_angle_rad must be less than pi/2.")
    if lock and (not latch or any(lock[name] > latch[f"capture_{name}"] for name in
                                    ("angle_rad", "speed_m_s", "rate_rad_s"))
                 or lock["position_m"] > min(latch["capture_lateral_m"], latch["max_travel_m"])):
        raise ValueError(f"{_PATH}.hard_mate requires a latch and lock tolerances within its capture envelope.")
    return normalized


def validate_docking_scenario(cfg: SimulationScenarioConfig) -> None:
    docking = cfg.simulator.docking
    if not docking:
        return
    simulator = cfg.simulator
    dynamics = simulator.dynamics
    orbit = dynamics.orbit
    attitude = dynamics.attitude
    active = {oid: obj for oid, obj in cfg.objects.items() if obj.enabled}
    port_ids = {docking[role]["object_id"] for role in ("probe", "receiver")}
    commanded = docking.get("commanded", False)
    profile = "flight_software" if commanded else "trajectory_only"
    if set(active) != port_ids or any(
        obj.kind != "satellite" or obj.runtime_profile != profile for obj in active.values()
    ):
        raise ValueError(f"{_PATH} requires exactly the two named, enabled {profile} satellites.")
    if commanded:
        from sim.runtime.actuator_factory import _resolve_satellite_isp_s

        if (cfg.ground_segment or {}).get("enabled"):
            raise ValueError(f"{_PATH}.commanded does not support ground-segment routing.")
        for obj in active.values():
            fsw = obj.flight_software
            hardware = obj.specs.get("flight_software_hardware") or {}
            if (fsw is None or not fsw.module or fsw.hardware_profile != "hardware.ideal_wrench.v1"
                    or fsw.checkpoint is not None or obj.knowledge.get("targets")
                    or str(hardware.get("actuator_id", fsw.params.get("actuator_id", "wrench"))) == "docking"):
                raise ValueError(
                    f"{_PATH}.commanded requires custom stacks with ideal wrench hardware, a separate "
                    "wrench actuator ID, and no checkpoint or separate knowledge targets."
                )
            if _resolve_satellite_isp_s(obj.specs) != 0. or obj.specs.get("fuel_mass_kg", 0.):
                raise ValueError(f"{_PATH}.commanded currently requires fixed-mass ideal wrench hardware.")
    if (simulator.central_body != "earth" or orbit.get("model", "two_body") != "two_body"
            or orbit.get("propagation_method", "special") != "special"
            or orbit.get("integrator", "rk4") != "rk4" or not attitude.get("enabled", True)
            or any(obj.propagation_method not in (None, "", "special") for obj in active.values())):
        raise ValueError(f"{_PATH} requires attitude-enabled Earth-centered two-body ONP with RK4.")
    perturbations = ("j2", "j3", "j4", "drag", "lift", "srp", "third_body_sun", "third_body_moon")
    precision_forces = ("spherical_harmonics", "schwarzschild", "earth_radiation", "ocean_tides", "solid_earth_tides")
    if (any(orbit.get(name, False) for name in perturbations)
            or any((orbit.get(name) or {}).get("enabled", False) for name in precision_forces)
            or any((attitude.get("disturbance_torques") or {}).values())
            or simulator.additional_gravity is not None or simulator.system_force_models
            or simulator.collisions or any(obj.force_models for obj in active.values())):
        raise ValueError(f"{_PATH} requires central gravity and excludes other forces and disturbance torques.")
    execution = simulator.execution
    if (execution.get("policy", "configured") in {"auto", "parallel"}
            or (execution.get("object_parallelism") or {}).get("enabled", False)):
        raise ValueError(f"{_PATH} requires serial object execution.")
    if any((obj.bridge is not None and obj.bridge.enabled)
           or (obj.reference_orbit or {}).get("enabled", False)
           or any((obj.specs.get(name) or {}).get("enabled", False) for name in ("thermal", "power"))
           for obj in active.values()):
        raise ValueError(f"{_PATH} does not support bridges, reference orbits, or spacecraft resources.")
    if any("position_eci_km" not in obj.initial_state or "velocity_eci_km_s" not in obj.initial_state
           for obj in active.values()):
        raise ValueError(f"{_PATH} requires explicit Cartesian ECI initial states.")
    from sim.dynamics.docking import DockingInterface

    pair = np.array([
        [*state["position_eci_km"], *state["velocity_eci_km_s"],
         *state.get("attitude_quat_bn", (1, 0, 0, 0)), *state.get("angular_rate_body_rad_s", (0, 0, 0))]
        for state in (active[docking[role]["object_id"]].initial_state for role in ("probe", "receiver"))
    ])
    DockingInterface(docking).validate_initial_geometry(pair)
