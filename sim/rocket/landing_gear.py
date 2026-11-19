"""Deployable rocket hardware and rigid-body footpad contact."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from math import isfinite, sqrt
from typing import TYPE_CHECKING, Literal

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import (
    eci_to_ecef_rotation_context,
    eci_to_ecef_rotation_derivative_context,
    frame_context_from_environment,
)
from sim.utils.geodesy import ecef_to_enu_rotation, ecef_to_geodetic_deg_km
from sim.utils.quaternion import quaternion_to_dcm_bn

if TYPE_CHECKING:
    from sim.rocket.models import RocketSimConfig, RocketState


def _number(value, name, minimum=0.0, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value < minimum or (positive and value == minimum):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}.")


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite numbers.")
    for component in value:
        _number(component, name, minimum=-np.inf)
    return tuple(value)


@dataclass(frozen=True)
class LandingLeg:
    id: str
    mount_body_m: tuple[float, float, float]
    stowed_offset_m: tuple[float, float, float]
    deployed_offset_m: tuple[float, float, float]
    travel_time_s: float
    stiffness_n_m: float
    damping_n_s_m: float
    max_stroke_m: float
    friction_coefficient: float
    initially_deployed: bool = False
    retractable: bool = False

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip() or self.id == "body":
            raise ValueError("Leg id must be nonempty and distinct from 'body'.")
        for name in ("mount_body_m", "stowed_offset_m", "deployed_offset_m"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        for name in ("travel_time_s", "stiffness_n_m", "max_stroke_m"):
            _number(getattr(self, name), name, positive=True)
        for name in ("damping_n_s_m", "friction_coefficient"):
            _number(getattr(self, name), name)
        if not isinstance(self.initially_deployed, bool) or not isinstance(self.retractable, bool):
            raise ValueError("initially_deployed and retractable must be booleans.")


@dataclass(frozen=True)
class LandingGearCommand:
    leg_id: str
    action: Literal["deploy", "stow"]

    def __post_init__(self):
        if not isinstance(self.leg_id, str) or not self.leg_id.strip() or self.action not in ("deploy", "stow"):
            raise ValueError("Landing gear commands require a leg ID and a deploy or stow action.")


@dataclass(frozen=True)
class ScheduledGearCommand:
    time_s: float
    leg_id: str
    action: Literal["deploy", "stow"]

    def __post_init__(self):
        _number(self.time_s, "time_s")
        LandingGearCommand(self.leg_id, self.action)


@dataclass(frozen=True)
class LandingGear:
    legs: tuple[LandingLeg, ...]
    body_contact_points_m: tuple[tuple[float, float, float], ...]
    commands: tuple[ScheduledGearCommand, ...] = ()
    max_step_s: float = 0.01
    settle_time_s: float = 0.5
    settle_speed_m_s: float = 0.05
    settle_rate_deg_s: float = 0.5
    friction_transition_speed_m_s: float = 0.01

    def __post_init__(self):
        if not self.legs or len({leg.id for leg in self.legs}) != len(self.legs):
            raise ValueError("Landing gear requires legs with unique IDs.")
        if not self.body_contact_points_m:
            raise ValueError("Landing gear requires at least one body contact point.")
        object.__setattr__(self, "body_contact_points_m", tuple(
            _vector(point, "body_contact_points_m") for point in self.body_contact_points_m
        ))
        for name in ("max_step_s", "settle_time_s", "settle_speed_m_s", "settle_rate_deg_s",
                     "friction_transition_speed_m_s"):
            _number(getattr(self, name), name, positive=True)
        legs_by_id = {leg.id: leg for leg in self.legs}
        keys = [(command.time_s, command.leg_id) for command in self.commands]
        if len(set(keys)) != len(keys) or any(leg not in legs_by_id for _, leg in keys):
            raise ValueError("Scheduled gear commands must reference known legs and be unique per leg and time.")
        if any(command.action == "stow" and not legs_by_id[command.leg_id].retractable for command in self.commands):
            raise ValueError("Scheduled stow commands require retractable legs.")
        object.__setattr__(self, "commands", tuple(sorted(self.commands, key=lambda item: item.time_s)))

    @classmethod
    def from_mapping(cls, data):
        def parse(model, mapping):
            if not isinstance(mapping, dict):
                raise ValueError(f"{model.__name__} must be a mapping.")
            unknown = mapping.keys() - {field.name for field in fields(model)}
            if unknown:
                raise ValueError(f"Unsupported {model.__name__} fields: {', '.join(sorted(map(str, unknown)))}.")
            try:
                return model(**mapping)
            except TypeError as error:
                raise ValueError(str(error)) from error

        if not isinstance(data, dict):
            raise ValueError("landing_gear must be a mapping.")
        values = dict(data)
        for name, model in (("legs", LandingLeg), ("commands", ScheduledGearCommand)):
            if name in values:
                if not isinstance(values[name], (list, tuple)):
                    raise ValueError(f"{name} must be a sequence.")
                values[name] = tuple(parse(model, row) for row in values[name])
        if not isinstance(values.get("body_contact_points_m"), (list, tuple)):
            raise ValueError("body_contact_points_m must be a sequence of body points.")
        return parse(cls, values)


@dataclass(frozen=True)
class LegState:
    extension: float
    target_deployed: bool
    peak_compression_m: float = 0.0
    peak_normal_load_n: float = 0.0
    last_command_time_s: float | None = None
    last_lock_time_s: float | None = None
    compression_m: float = 0.0
    normal_load_n: float = 0.0
    last_rejected_action: str | None = None
    last_rejection_time_s: float | None = None

    @property
    def status(self):
        if self.target_deployed:
            return "down_locked" if self.extension == 1.0 else "deploying"
        return "stowed" if self.extension == 0.0 else "stowing"


@dataclass(frozen=True)
class ContactPoint:
    id: str
    leg: LandingLeg | None
    status: str
    gap_m: float
    position_ecef_km: np.ndarray
    velocity_ecef_m_s: np.ndarray
    normal_ecef: np.ndarray
    arm_body_m: np.ndarray
    normal_load_n: float
    force_ecef_n: np.ndarray


class GearContactModel:
    def __init__(self, config: LandingGear, sim: RocketSimConfig):
        self.config = config
        self.surface_altitude_m = sim.touchdown.surface_altitude_m
        self.context = frame_context_from_environment(sim.atmosphere_env)
        self.radius_km = sim.earth_impact_radius_km
        self.legs_by_id = {leg.id: leg for leg in config.legs}
        self.ellipsoid = sim.use_wgs84_geodesy and self.radius_km == EARTH_RADIUS_KM
        inertia = np.asarray(sim.inertia_kg_m2, dtype=float)
        if (inertia.shape != (3, 3) or not np.all(np.isfinite(inertia))
                or not np.allclose(inertia, inertia.T) or np.min(np.linalg.eigvalsh(inertia)) <= 0.0):
            raise ValueError("Landing gear requires a positive definite body inertia tensor.")
        inverse_inertia = 1.0 / float(np.min(np.linalg.eigvalsh(inertia)))
        self.angular_mobility = {
            leg.id: inverse_inertia * max(
                float(np.linalg.norm(np.add(leg.mount_body_m, offset)))**2
                for offset in (leg.stowed_offset_m, leg.deployed_offset_m)
            )
            for leg in config.legs
        }

    def initial_legs(self):
        return {
            leg.id: LegState(float(leg.initially_deployed), leg.initially_deployed,
                             last_lock_time_s=0.0 if leg.initially_deployed else None)
            for leg in self.config.legs
        }

    def command(self, state: RocketState, commands):
        if any(not isinstance(command, LandingGearCommand) for command in commands):
            raise ValueError("Landing gear commands must be LandingGearCommand instances.")
        if any(command.leg_id not in self.legs_by_id for command in commands):
            raise ValueError("Landing gear command references an unknown leg.")
        if len({command.leg_id for command in commands}) != len(commands):
            raise ValueError("Multiple landing gear commands for the same leg at one boundary.")
        for command in commands:
            leg = self.legs_by_id[command.leg_id]
            current = state.landing_legs[leg.id]
            deploy = command.action == "deploy"
            if current.target_deployed == deploy:
                continue
            if not deploy and (not leg.retractable or state.gear_touchdown is not None):
                state.landing_legs[leg.id] = replace(
                    current, last_rejected_action="stow", last_rejection_time_s=state.t_s,
                )
                continue
            state.landing_legs[leg.id] = replace(current, target_deployed=deploy, last_command_time_s=state.t_s)

    def advance(self, state: RocketState, dt_s: float):
        for leg in self.config.legs:
            current = state.landing_legs[leg.id]
            rate = dt_s / leg.travel_time_s
            extension = current.extension + (rate if current.target_deployed else -rate)
            extension = 1.0 if extension >= 1.0 - 1e-12 else max(0.0, extension)
            if extension < 1e-12 and not current.target_deployed:
                extension = 0.0
            lock_time = state.t_s if extension == 1.0 and current.extension < 1.0 else current.last_lock_time_s
            state.landing_legs[leg.id] = replace(current, extension=extension, last_lock_time_s=lock_time)

    def motion_boundary(self, state: RocketState):
        return min((
            abs(float(state.landing_legs[leg.id].target_deployed) - state.landing_legs[leg.id].extension)
            * leg.travel_time_s
            for leg in self.config.legs
            if state.landing_legs[leg.id].status in ("deploying", "stowing")
        ), default=float("inf"))

    def contact_step(self, mass_kg: float, points: list[ContactPoint]):
        # Use an upper bound on the combined contact-mode frequency and damping rate.
        omega_squared = damping_rate = friction_rate = 0.0
        for point in points:
            if point.leg is None or point.status != "down_locked" or point.gap_m > 1e-6:
                continue
            leg = point.leg
            mobility = 1.0 / mass_kg + self.angular_mobility[leg.id]
            omega_squared += leg.stiffness_n_m * mobility
            damping_rate += leg.damping_n_s_m * mobility
            friction_rate += leg.friction_coefficient * point.normal_load_n * mobility / self.config.friction_transition_speed_m_s
        return min(self.config.max_step_s, 0.12 / max(sqrt(omega_squared), damping_rate, friction_rate, 1e-12))

    def sample(self, state: RocketState):
        body_to_eci = quaternion_to_dcm_bn(state.attitude_quat_bn).T
        eci_to_ecef = eci_to_ecef_rotation_context(state.t_s, self.context)
        frame_derivative = eci_to_ecef_rotation_derivative_context(state.t_s, self.context)
        position = eci_to_ecef @ state.position_eci_km
        velocity = eci_to_ecef @ state.velocity_eci_km_s + frame_derivative @ state.position_eci_km
        # The derivative of the Earth-fixed rotation gives the local velocity
        # of each body axis, including frame-rate effects on short lever arms.
        body_to_ecef = eci_to_ecef @ body_to_eci
        axes_velocity = frame_derivative @ body_to_eci

        def point(point_id, leg, status, arm, deployment_velocity):
            point_position = position + body_to_ecef @ arm / 1000.0
            point_velocity = 1000.0 * velocity + axes_velocity @ arm + body_to_ecef @ (
                np.cross(state.angular_rate_body_rad_s, arm) + deployment_velocity
            )
            if self.ellipsoid:
                lat, lon, altitude = ecef_to_geodetic_deg_km(point_position)
                normal = ecef_to_enu_rotation(lat, lon)[2]
            else:
                radius = np.linalg.norm(point_position)
                altitude = radius - self.radius_km
                normal = point_position / radius
            gap = float(1000.0 * altitude - self.surface_altitude_m)
            normal_speed = float(point_velocity @ normal)
            load = 0.0
            force = np.zeros(3)
            if leg is not None and status == "down_locked" and gap <= 0.0:
                load = max(0.0, -leg.stiffness_n_m * gap - leg.damping_n_s_m * normal_speed)
                tangential = point_velocity - normal_speed * normal
                speed = float(np.linalg.norm(tangential))
                force = load * (normal - leg.friction_coefficient * tangential /
                                max(speed, self.config.friction_transition_speed_m_s))
            return ContactPoint(point_id, leg, status, gap, point_position, point_velocity, normal, arm, load, force)

        points = [
            point(f"body:{index}", None, "body", np.asarray(arm), np.zeros(3))
            for index, arm in enumerate(self.config.body_contact_points_m)
        ]
        for leg in self.config.legs:
            current = state.landing_legs[leg.id]
            offset_delta = np.subtract(leg.deployed_offset_m, leg.stowed_offset_m)
            arm = np.add(leg.mount_body_m, leg.stowed_offset_m) + current.extension * offset_delta
            movement = {"deploying": 1.0, "stowing": -1.0}.get(current.status, 0.0)
            points.append(point(leg.id, leg, current.status, arm, movement * offset_delta / leg.travel_time_s))
        return points, body_to_ecef, eci_to_ecef

    def wrench(self, points, body_to_ecef, eci_to_ecef):
        force_eci = eci_to_ecef.T @ sum((point.force_ecef_n for point in points), np.zeros(3))
        torque_body = sum((np.cross(point.arm_body_m, body_to_ecef.T @ point.force_ecef_n) for point in points), np.zeros(3))
        return force_eci, torque_body

    def update_loads(self, state: RocketState, points):
        for point in points:
            if point.leg is not None:
                leg = state.landing_legs[point.id]
                state.landing_legs[point.id] = replace(
                    leg, peak_compression_m=max(leg.peak_compression_m, -point.gap_m),
                    peak_normal_load_n=max(leg.peak_normal_load_n, point.normal_load_n),
                    compression_m=max(0.0, -point.gap_m), normal_load_n=point.normal_load_n,
                )

    def supported(self, state: RocketState, points):
        loaded = [point for point in points if point.normal_load_n > 0.0]
        if len(loaded) < 3 or any(leg.status != "down_locked" for leg in state.landing_legs.values()):
            return False
        if max(np.linalg.norm(point.velocity_ecef_m_s) for point in loaded) > self.config.settle_speed_m_s:
            return False
        rotation = eci_to_ecef_rotation_context(state.t_s, self.context)
        frame_rate = eci_to_ecef_rotation_derivative_context(state.t_s, self.context) @ rotation.T
        omega_earth_ecef = np.array([frame_rate[1, 2], frame_rate[2, 0], frame_rate[0, 1]])
        omega_vehicle_ecef = rotation @ quaternion_to_dcm_bn(state.attitude_quat_bn).T @ state.angular_rate_body_rad_s
        if np.rad2deg(np.linalg.norm(omega_vehicle_ecef - omega_earth_ecef)) > self.config.settle_rate_deg_s:
            return False
        center = rotation @ state.position_eci_km
        lat, lon, _ = ecef_to_geodetic_deg_km(center)
        east_north = ecef_to_enu_rotation(lat, lon)[:2]
        offsets = np.array([east_north @ (point.position_ecef_km - center) for point in loaded])
        # A central foot does not define a bearing or change the outer footprint.
        offsets = offsets[np.linalg.norm(offsets, axis=1) >= 1e-9]
        if len(offsets) < 3:
            return False
        # Strictly inside the convex support polygon iff the foot bearings
        # leave no gap of 180 degrees or more around the projected center.
        angles = np.sort(np.arctan2(offsets[:, 1], offsets[:, 0]))
        return bool(np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])) < np.pi - 1e-8)


def gear_evidence(state: RocketState) -> dict:
    return {
        "touchdown": state.gear_touchdown,
        "terminal_event": state.gear_terminal,
        "first_rebound_time_s": state.gear_rebound_time_s,
        "legs": {
            leg_id: {
                "status": leg.status, "extension": leg.extension,
                "peak_compression_m": leg.peak_compression_m, "peak_normal_load_n": leg.peak_normal_load_n,
                "last_command_time_s": leg.last_command_time_s, "last_lock_time_s": leg.last_lock_time_s,
                "compression_m": leg.compression_m, "normal_load_n": leg.normal_load_n,
                "last_rejected_action": leg.last_rejected_action,
                "last_rejection_time_s": leg.last_rejection_time_s,
            }
            for leg_id, leg in state.landing_legs.items()
        },
    }
