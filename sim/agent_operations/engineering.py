"""Spacecraft package and OEL forecast; initialized only from delivered observations."""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from sim.api import SimulationConfig, SimulationSession
from sim.flight_software import (
    ClockScale,
    ClockTag,
    GroundCommandKind,
    GroundCommandPayload,
    InputEvent,
    InputKind,
    PacketId,
    Quality,
    TelemetryField,
)

from .contracts import NAV_FIELDS, PARAMETERS, PointingAction, Policy, digest


class SpacecraftPackage:
    def __init__(self, config: dict, spacecraft_id: str = "oel-sat-a"):
        self._config = deepcopy(config)
        self.spacecraft_id = spacecraft_id
        self.object_id = next(iter(config["objects"]))
        if len(config["objects"]) != 1:
            raise ValueError("operations v1 requires one spacecraft")
        obj = config["objects"][self.object_id]
        if (
            obj["flight_software"].get("stack") != "fsw.attitude_reference"
            or obj["flight_software"]["params"].get("reference_mode") != "target"
        ):
            raise ValueError("operations v1 requires the built-in target reference stack")
        if any(k in obj["flight_software"] for k in ("module", "class_name", "checkpoint")):
            raise ValueError("external stack or checkpoint is outside this exercise")
        if obj["flight_software"].get("hardware_profile") != "hardware.reaction_wheels_magnetorquer.v1":
            raise ValueError("operations v1 requires the three-wheel hardware package")
        if obj.get("knowledge") or obj["flight_software"]["params"].get("navigation_initialization", "ideal") != "ideal":
            raise ValueError("operations v1 represents ideal onboard navigation only")
        SimulationConfig.from_dict(self._config)  # validate reviewed built-in package
        sim_root = Path(__file__).resolve().parents[1]
        owners = (
            "dynamics",
            "actuators",
            "spacecraft_resources",
            "gnc",
            "runtime",
            "execution",
            "core",
            "flight_software",
            "agent_operations",
        )
        files = sorted(
            {path for owner in owners for path in (sim_root / owner).rglob("*.py")} | {sim_root / "single_run.py"}
        )
        self.model_digest = digest(
            {str(path.relative_to(sim_root)): sha256(path.read_bytes()).hexdigest() for path in files}
        )
        self.identity = digest(
            {
                "schema": "oel.spacecraft_package.v1",
                "spacecraft_id": spacecraft_id,
                "scenario": self._config,
                "model_source_digest": self.model_digest,
            }
        )

    def config(self):
        return deepcopy(self._config)

    @property
    def recovery_target(self):
        return tuple(self._config["objects"][self.object_id]["flight_software"]["params"]["target_position_eci_m"])

    def describe(self):
        obj = self._config["objects"][self.object_id]
        return {
            "spacecraft_id": self.spacecraft_id,
            "package_digest": self.identity,
            "model_source_digest": self.model_digest,
            "command": "PointTarget",
            "target_frame": "OEL/ECI/J2000",
            "target_units": "m",
            "quaternion": "scalar-first wxyz; passive inertial-to-body rotation",
            "quaternion_rotation": {
                "mapping": "v_body = C_bn @ v_eci; boresight_eci = C_bn.T @ boresight_body",
                "matrix": "C_bn = (w*w - dot(v,v))*I + 2*outer(v,v) - 2*w*skew(v), v=[x,y,z]",
                "skew": "skew(v) @ u = cross(v,u)",
                "example": "q=[cos(theta/2),0,0,sin(theta/2)] maps body +X to ECI [cos(theta),sin(theta),0]",
                "owner": "sim.utils.quaternion.quaternion_to_dcm_bn",
            },
            "boresight_body": obj["flight_software"]["params"]["boresight_body"],
            "battery_capacity_wh": obj["specs"]["power"]["battery_capacity_wh"],
            "recovery_target_eci_m": self.recovery_target,
            "navigation": "ideal onboard navigation delivered through Yamcs; not sensor qualification",
            "model_limits": [
                "deterministic nominal hardware",
                "no uncertainty qualification",
                "no thermal constraint",
                "no external spacecraft connection",
            ],
        }


def command_event(
    action: PointingAction, time_s: float, *, sequence: int = 0, command_id: str | None = None, object_id: str = "sat"
):
    clock = ClockTag(f"{object_id}/onboard", round(time_s * 1e9), 1, ClockScale.ONBOARD)
    fields = (TelemetryField("operation", "set_target_eci"),) + tuple(
        TelemetryField(f"target_{a}_eci_m", float(v), "m") for a, v in zip("xyz", action.target_eci_m)
    )
    return InputEvent(
        PacketId("guarded-ground", "operations-v1", sequence),
        InputKind.GROUND_COMMAND,
        clock,
        clock,
        Quality(),
        GroundCommandPayload(command_id or f"point-{action.request_id}", GroundCommandKind.STACK_COMMAND, fields),
    )


def onboard_fields(session: SimulationSession, object_id: str) -> dict:
    records = session.flight_software_observations(object_id)
    fields = {field["name"]: field["value"] for record in records["telemetry"] for field in record["fields"]}
    for record in records["actuator_telemetry"]:
        for field in record["payload"]["fields"]:
            fields[field["name"]] = field["value"]
    return fields


def utc_epoch(jd: float):
    return datetime.fromtimestamp((jd - 2440587.5) * 86400, timezone.utc).isoformat().replace("+00:00", "Z")


class OelForecast:
    def __init__(self, package: SpacecraftPackage, policy: Policy):
        self.package, self.policy = package, policy

    def evaluate(self, action: PointingAction, observation: dict) -> dict:
        values = observation["values"]
        if set(PARAMETERS) - set(values):
            return {"disposition": "insufficient_evidence", "reason": "missing required telemetry"}
        if values["package_digest"] != self.package.identity or values["spacecraft_id"] != self.package.spacecraft_id:
            return {"disposition": "rejected", "reason": "spacecraft package mismatch"}
        if not all(
            type(values[k]) in (int, float) and math.isfinite(values[k])
            for k in (
                *NAV_FIELDS,
                "battery_soc",
                "sim_time_s",
                "wheel_x_momentum",
                "wheel_y_momentum",
                "wheel_z_momentum",
            )
        ):
            return {"disposition": "insufficient_evidence", "reason": "invalid numeric telemetry"}
        if values["mode"] != "nominal" or values["command_status"] == "active":
            return {"disposition": "rejected", "reason": "spacecraft unavailable or command active"}
        if action.hold_s > self.policy.max_hold_s:
            return {"disposition": "rejected", "reason": "hold exceeds delegated authority"}
        if not 0 <= values["battery_soc"] <= 1 or abs(sum(values[f"q_{a}"] ** 2 for a in "wxyz") - 1) > 1e-5:
            return {"disposition": "insufficient_evidence", "reason": "invalid state telemetry"}
        distance = math.sqrt(
            sum((action.target_eci_m[i] - values[f"position_{a}_m"]) ** 2 for i, a in enumerate("xyz"))
        )
        if distance < 1000 or max(abs(v) for v in action.target_eci_m) > 1e12:
            return {"disposition": "rejected", "reason": "target outside represented geometry"}
        config = self.package.config()
        obj = config["objects"][self.package.object_id]
        obj["initial_state"] = {
            "position_eci_km": [values[f"position_{a}_m"] / 1000 for a in "xyz"],
            "velocity_eci_km_s": [values[f"velocity_{a}_m_s"] / 1000 for a in "xyz"],
            "attitude_quat_bn": [values[f"q_{a}"] for a in "wxyz"],
            "angular_rate_body_rad_s": [values[f"omega_{a}_rad_s"] for a in "xyz"],
        }
        obj["specs"]["power"]["initial_soc"] = values["battery_soc"]
        params = obj["flight_software"]["params"]
        params["target_position_eci_m"] = list(action.target_eci_m)
        params["wheel_initial_momentum_n_m_s"] = [values[f"wheel_{a}_momentum"] for a in "xyz"]
        config["simulator"]["initial_jd_utc"] += values["sim_time_s"] / 86400
        duration = action.hold_s + self.policy.recovery_s
        config["simulator"]["duration_s"] = duration
        config["scenario_name"] = "agent_operations_forecast"
        config["outputs"]["output_dir"] = "build/agent-operations/forecast-unused"
        session = SimulationSession(config, history_mode="dynamic", max_history_samples=8)
        session.reset()
        min_soc, max_wheel, max_rate, unmet = values["battery_soc"], 0.0, 0.0, 0.0
        hold_error = None
        recovered = False
        time_s = 0.0
        while not session.done:
            boundary = duration if recovered else action.hold_s
            snap = session.step(dt_s=min(1.0, boundary - time_s))
            time_s = snap.time_s
            fields = onboard_fields(session, self.package.object_id)
            resource = snap.spacecraft_resources[self.package.object_id]
            min_soc = min(min_soc, resource["battery_soc"])
            max_wheel = max(max_wheel, float(fields["wheel_momentum_fraction"]))
            max_rate = max(max_rate, float(fields["angular_rate_norm_rad_s"]))
            unmet = max(unmet, resource["unmet_load_w"])
            if snap.time_s >= action.hold_s and not recovered:
                hold_error = float(fields["attitude_error_rad"])
                recovery = PointingAction(action.request_id, self.package.recovery_target, 1.0)
                session.publish_fsw_input(
                    self.package.object_id,
                    command_event(
                        recovery, snap.time_s, command_id="forecast-recovery", object_id=self.package.object_id
                    ),
                )
                recovered = True
        metrics = {
            "minimum_battery_soc": min_soc,
            "maximum_wheel_fraction": max_wheel,
            "maximum_rate_rad_s": max_rate,
            "maximum_unmet_load_w": unmet,
            "hold_end_error_rad": hold_error,
            "forecast_duration_s": duration,
        }
        violations = []
        if min_soc < self.policy.battery_reserve + self.policy.battery_margin:
            violations.append("battery reserve plus margin")
        if max_wheel > self.policy.wheel_fraction_limit:
            violations.append("wheel momentum")
        if max_rate > self.policy.max_rate_rad_s:
            violations.append("angular rate")
        if unmet > 1e-6:
            violations.append("unserved electrical load")
        return {
            "disposition": "rejected" if violations else "permitted",
            "reason": ", ".join(violations) or "represented constraints satisfied",
            "metrics": metrics,
            "observation_digest": digest(observation),
            "model_digest": self.package.identity,
            "policy_digest": self.policy.identity,
            "forecast_config_digest": digest(config),
        }
