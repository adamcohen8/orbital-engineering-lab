"""Bind compiled engineering properties to the active v2 flight-software hardware."""

from __future__ import annotations

import copy
import math


def hardware_parameters(placements, com):
    thrusters, wheels, sensors = [], [], []
    for p in placements:
        values = p["parameters"]
        if p["kind"] == "thruster":
            thrusters.append(
                {
                    "thruster_id": p["id"],
                    "direction_body": p["axis_body"],
                    "position_body_m": [x - c for x, c in zip(p["position_body_m"], com, strict=True)],
                    "max_thrust_n": values["max_thrust_n"],
                    "isp_s": values["isp_s"],
                }
            )
        elif p["kind"] == "reaction_wheel":
            wheels.append(p)
        elif p["kind"] == "tracking_sensor":
            sensors.append(p)
    params = {}
    if thrusters:
        params["rcs_thrusters"] = thrusters
    if wheels:
        params.update(
            {
                "wheel_axes_body": [p["axis_body"] for p in wheels],
                "wheel_max_torque_n_m": [p["parameters"]["max_torque_nm"] for p in wheels],
                "wheel_max_momentum_n_m_s": [
                    min(
                        p["parameters"]["max_momentum_nms"],
                        p["parameters"]["inertia_kg_m2"] * p["parameters"]["max_speed_rad_s"],
                    )
                    for p in wheels
                ],
                "wheel_inertia_kg_m2": [p["parameters"]["inertia_kg_m2"] for p in wheels],
                "wheel_max_speed_rad_s": [p["parameters"]["max_speed_rad_s"] for p in wheels],
            }
        )
    if len(sensors) > 1:
        raise ValueError("The current knowledge runtime supports one mounted tracking sensor per observer.")
    if sensors:
        sensor = sensors[0]
        values = sensor["parameters"]
        params["tracking_sensor"] = {
            "refresh_rate_s": values["refresh_rate_s"],
            "conditions": {
                "sensor_position_body_m": [x - c for x, c in zip(sensor["position_body_m"], com, strict=True)],
                "sensor_boresight_body": sensor["axis_body"],
                "max_range_km": values["max_range_km"],
                "fov_half_angle_rad": values["fov_half_angle_rad"],
            },
            "sensor_error": {
                key: values[key] for key in ("range_sigma_km", "angle_sigma_rad", "range_rate_sigma_km_s")
            },
        }
    return params


def bind(template_object, compiled):
    """Preserve mission/controller choices; refuse hardware overrides and unsupported stacks."""
    section = copy.deepcopy(template_object.get("flight_software", {}))
    stack = section.get("stack")
    if section.get("profile") or not stack:
        raise ValueError(
            "Twin materialization requires an explicit flight_software.stack; expand profiles before binding."
        )
    params = dict(section.get("params", {}))
    hardware = compiled["hardware_parameters"]
    inactive = []
    if stack == "fsw.passive":
        if section.get("hardware_profile") not in {None, "hardware.passive.v1"}:
            raise ValueError("Passive stack requires passive hardware.")
        section["hardware_profile"] = "hardware.passive.v1"
        inactive = [p["id"] for p in compiled["placements"]]
    elif stack == "fsw.attitude_reference":
        if "wheel_axes_body" not in hardware:
            raise ValueError("Attitude twin binding currently requires reaction wheels.")
        _profile(section, "hardware.reaction_wheels.v1")
        _merge(params, {k: v for k, v in hardware.items() if k.startswith("wheel_")})
        inactive = [p["id"] for p in compiled["placements"] if p["kind"] == "thruster"]
    elif stack == "fsw.low_thrust_reference":
        raise ValueError(
            "Low-thrust twin binding requires a continuous-engine adapter; mounted RCS thrusters cannot represent it."
        )
    elif stack in {"fsw.orbit_reference", "fsw.rpo_reference"}:
        if "rcs_thrusters" not in hardware:
            raise ValueError("Translation twin binding currently requires mounted RCS thrusters.")
        _profile(section, "hardware.rcs.v1")
        _merge(params, {"rcs_thrusters": hardware["rcs_thrusters"]})
        compiled_max_force_n = float(sum(p["max_thrust_n"] for p in hardware["rcs_thrusters"]))
        requested_max_force_n = params.get("max_force_n")
        if requested_max_force_n is None:
            params["max_force_n"] = compiled_max_force_n
        else:
            if isinstance(requested_max_force_n, bool):
                raise ValueError("Template max_force_n must be a finite positive number.")
            try:
                requested_max_force_n = float(requested_max_force_n)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Template max_force_n must be a finite positive number.") from exc
            if not math.isfinite(requested_max_force_n) or requested_max_force_n <= 0.0:
                raise ValueError("Template max_force_n must be a finite positive number.")
            if requested_max_force_n > compiled_max_force_n:
                raise ValueError(
                    "Template max_force_n cannot exceed compiled mounted RCS capacity "
                    f"({compiled_max_force_n:g} N)."
                )
            # A lower study-specific cap is a legitimate user constraint; keep
            # it while preventing the scenario from claiming more physical
            # force than the reviewed twin provides.
            params["max_force_n"] = requested_max_force_n
        if params.get("attitude_reference_mode") not in (None, "", "none"):
            if "wheel_axes_body" not in hardware:
                raise ValueError(
                    "Coupled attitude binding requires twin reaction wheels; ideal torque is not substituted."
                )
            _merge(
                params,
                {
                    "attitude_hardware_profile": "hardware.reaction_wheels.v1",
                    **{k: v for k, v in hardware.items() if k.startswith("wheel_")},
                },
            )
        else:
            inactive = [p["id"] for p in compiled["placements"] if p["kind"] == "reaction_wheel"]
    else:
        raise ValueError(f"No qualified twin hardware binding for flight-software stack {stack}.")
    section["params"] = params
    return section, {
        "stack": stack,
        "hardware_profile": section["hardware_profile"],
        "inactive_components": inactive,
        "note": "Inactive devices remain in the engineering twin but are not actuated by this selected study stack.",
    }


def _merge(params, owned):
    for key, value in owned.items():
        if key in params and params[key] != value:
            raise ValueError(f"Template overrides twin-owned hardware parameter: {key}")
        params[key] = value


def _profile(section, expected):
    if section.get("hardware_profile") not in {None, expected}:
        raise ValueError(f"Twin binding requires {expected}, not {section['hardware_profile']}.")
    section["hardware_profile"] = expected


def bind_sensor(template_object, compiled):
    knowledge = copy.deepcopy(template_object.get("knowledge", {}))
    sensor = compiled["hardware_parameters"].get("tracking_sensor")
    if not sensor or not knowledge.get("targets"):
        return knowledge, False
    _merge(knowledge, {"refresh_rate_s": sensor["refresh_rate_s"]})
    for key in ("conditions", "sensor_error"):
        block = knowledge.setdefault(key, {})
        _merge(block, sensor[key])
    return knowledge, True
