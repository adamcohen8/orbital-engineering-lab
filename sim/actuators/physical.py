"""Reference physical actuator state evolution for the v2 command bus."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, isfinite, isnan, sin, sqrt

import numpy as np

from sim.actuators.command_bus import ActuatorDemand, DemandMode
from sim.flight_software.contracts import (
    AerodynamicEffectorPositionCommand,
    CmgGimbalRateCommand,
    ContinuousEngineCommand,
    IdealWrenchCommand,
    MagnetorquerDipoleCommand,
    PacketId,
    ReactionWheelTorqueCommand,
    TelemetryField,
    ThrusterOnOffCommand,
    ThrusterPulseCommand,
    Vector3,
)


@dataclass(frozen=True, slots=True)
class ActuatorRealization:
    actuator_id: str
    interval_start_ns: int
    interval_end_ns: int
    source_command_id: PacketId | None
    demand_mode: DemandMode
    requested_force_n: Vector3
    requested_torque_n_m: Vector3
    realized_force_n: Vector3
    realized_torque_n_m: Vector3
    mass_flow_kg_s: float = 0.0
    device_state: tuple[TelemetryField, ...] = ()
    saturated: bool = False

    def __post_init__(self) -> None:
        if not isfinite(float(self.mass_flow_kg_s)) or float(self.mass_flow_kg_s) < 0.0:
            raise ValueError("mass_flow_kg_s must be finite and nonnegative")


class _NativeIntervalCacheState:
    def __getstate__(self):
        state = self.__dict__.copy()
        # The native context caches immutable hardware parameters; device state
        # remains in the ordinary Python fields and recreates the cache lazily.
        state.pop("_native_interval_context", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class IdealWrenchHardware(_NativeIntervalCacheState):
    def __init__(
        self,
        actuator_id: str,
        *,
        max_force_n: float = float("inf"),
        max_torque_n_m: float = float("inf"),
        response_time_constant_s: float = 0.0,
        specific_impulse_s: float | None = None,
    ) -> None:
        if not actuator_id.strip():
            raise ValueError("actuator_id must be non-empty")
        max_force = float(max_force_n)
        max_torque = float(max_torque_n_m)
        response_time_constant = float(response_time_constant_s)
        # Positive infinity is the declared unbounded default for force and
        # torque. NaN and negative infinity are never valid limits, and
        # response time must always be finite.
        if (
            isnan(max_force)
            or max_force < 0.0
            or isnan(max_torque)
            or max_torque < 0.0
            or not isfinite(response_time_constant)
            or response_time_constant < 0.0
        ):
            raise ValueError("hardware limits must be nonnegative and response time must be finite and nonnegative")
        if specific_impulse_s is not None and (not isfinite(specific_impulse_s) or specific_impulse_s <= 0.0):
            raise ValueError("specific_impulse_s must be positive and finite when provided")
        self.actuator_id = actuator_id
        self.max_force_n = max_force
        self.max_torque_n_m = max_torque
        self.response_time_constant_s = response_time_constant
        self.specific_impulse_s = None if specific_impulse_s is None else float(specific_impulse_s)
        self.realized_force_n: Vector3 = (0.0, 0.0, 0.0)
        self.realized_torque_n_m: Vector3 = (0.0, 0.0, 0.0)

    def advance(self, demand: ActuatorDemand, *, start_time_ns: int, end_time_ns: int) -> ActuatorRealization:
        if end_time_ns < start_time_ns:
            raise ValueError("actuator interval must be nonnegative")
        payload = demand.payload
        if payload is not None and not isinstance(payload, IdealWrenchCommand):
            raise TypeError("ideal wrench hardware requires IdealWrenchCommand demand")
        if type(self) is IdealWrenchHardware and getattr(self, "_native_actuator_intervals", False):
            from sim.actuators.rust_physical import advance_hardware
            return advance_hardware(self, demand, start_time_ns, end_time_ns)
        if isinstance(payload, IdealWrenchCommand):
            requested_force = payload.force_n
            requested_torque = payload.torque_n_m
        else:
            requested_force = (0.0, 0.0, 0.0)
            requested_torque = (0.0, 0.0, 0.0)
        target_force, force_saturated = _limit_norm(requested_force, self.max_force_n)
        target_torque, torque_saturated = _limit_norm(requested_torque, self.max_torque_n_m)
        dt_s = (end_time_ns - start_time_ns) / 1.0e9
        alpha = 1.0 if self.response_time_constant_s <= 0.0 else 1.0 - exp(-dt_s / self.response_time_constant_s)
        self.realized_force_n = _blend(self.realized_force_n, target_force, alpha)
        self.realized_torque_n_m = _blend(self.realized_torque_n_m, target_torque, alpha)
        return ActuatorRealization(
            actuator_id=self.actuator_id,
            interval_start_ns=start_time_ns,
            interval_end_ns=end_time_ns,
            source_command_id=None if demand.source_command is None else demand.source_command.command_id,
            demand_mode=demand.mode,
            requested_force_n=requested_force,
            requested_torque_n_m=requested_torque,
            realized_force_n=self.realized_force_n,
            realized_torque_n_m=self.realized_torque_n_m,
            mass_flow_kg_s=_mass_flow_kg_s(self.realized_force_n, self.specific_impulse_s),
            saturated=force_saturated or torque_saturated,
        )

    def snapshot_state(self) -> dict[str, object]:
        return {
            "realized_force_n": list(self.realized_force_n),
            "realized_torque_n_m": list(self.realized_torque_n_m),
        }

    def restore_state(self, state: object) -> None:
        if not isinstance(state, dict):
            raise ValueError("ideal-wrench checkpoint must be an object")
        force = tuple(float(value) for value in state.get("realized_force_n", ()))
        torque = tuple(float(value) for value in state.get("realized_torque_n_m", ()))
        if (
            len(force) != 3
            or len(torque) != 3
            or not all(isfinite(value) for value in force)
            or not all(isfinite(value) for value in torque)
        ):
            raise ValueError("ideal-wrench checkpoint vectors must have three finite components")
        self.realized_force_n = force  # type: ignore[assignment]
        self.realized_torque_n_m = torque  # type: ignore[assignment]


class ReactionWheelHardware(_NativeIntervalCacheState):
    """Body-torque realization with explicit wheel momentum storage."""

    def __init__(
        self,
        actuator_id: str,
        *,
        axes_body: tuple[tuple[float, float, float], ...],
        max_torque_n_m: tuple[float, ...],
        max_momentum_n_m_s: tuple[float, ...],
        initial_momentum_n_m_s: tuple[float, ...] | None = None,
        inertia_kg_m2: tuple[float, ...] | None = None,
        max_speed_rad_s: tuple[float, ...] | None = None,
    ) -> None:
        axes = np.asarray(axes_body, dtype=float)
        if not actuator_id.strip() or axes.ndim != 2 or axes.shape[1] != 3 or not np.all(np.isfinite(axes)):
            raise ValueError("reaction-wheel identity and finite Nx3 axes are required")
        axis_norms = np.linalg.norm(axes, axis=1)
        if np.any(~np.isclose(axis_norms, 1.0, rtol=1.0e-9, atol=1.0e-9)):
            raise ValueError("reaction-wheel axes must be unit vectors")
        self.actuator_id = actuator_id
        self.axes_body = axes
        self.max_torque_n_m = _coordinate_limits(max_torque_n_m, axes.shape[0], "wheel torque")
        self.max_momentum_n_m_s = _coordinate_limits(max_momentum_n_m_s, axes.shape[0], "wheel momentum")
        if (inertia_kg_m2 is None) != (max_speed_rad_s is None):
            raise ValueError("wheel inertia and speed limits must be supplied together")
        self.inertia_kg_m2 = None
        if inertia_kg_m2 is not None:
            self.inertia_kg_m2 = _coordinate_limits(inertia_kg_m2, axes.shape[0], "wheel inertia")
            speed_limits = _coordinate_limits(max_speed_rad_s, axes.shape[0], "wheel speed")
            if np.any(self.inertia_kg_m2 <= 0) or np.any(speed_limits <= 0):
                raise ValueError("wheel inertia and speed limits must be positive")
            self.max_momentum_n_m_s = np.minimum(self.max_momentum_n_m_s, self.inertia_kg_m2 * speed_limits)
        initial = (
            np.zeros(axes.shape[0])
            if initial_momentum_n_m_s is None
            else np.asarray(initial_momentum_n_m_s, dtype=float)
        )
        if (
            initial.shape != (axes.shape[0],)
            or not np.all(np.isfinite(initial))
            or np.any(np.abs(initial) > self.max_momentum_n_m_s)
        ):
            raise ValueError("initial wheel momentum must be finite, match wheel count, and remain within limits")
        self.momentum_n_m_s = initial.copy()

    def advance(self, demand: ActuatorDemand, *, start_time_ns: int, end_time_ns: int) -> ActuatorRealization:
        if end_time_ns < start_time_ns:
            raise ValueError("actuator interval must be nonnegative")
        payload = demand.payload
        if payload is not None and not isinstance(payload, ReactionWheelTorqueCommand):
            raise TypeError("reaction-wheel hardware requires ReactionWheelTorqueCommand demand")
        if type(self) is ReactionWheelHardware and getattr(self, "_native_actuator_intervals", False):
            from sim.actuators.rust_physical import advance_hardware
            return advance_hardware(self, demand, start_time_ns, end_time_ns)
        requested = np.zeros(self.axes_body.shape[0]) if payload is None else np.asarray(payload.torque_n_m, dtype=float)
        if requested.size != self.axes_body.shape[0]:
            raise ValueError("reaction-wheel command count must match configured wheels")
        clipped = np.clip(requested, -self.max_torque_n_m, self.max_torque_n_m)
        dt_s = (end_time_ns - start_time_ns) / 1.0e9
        if dt_s > 0.0:
            next_momentum = np.clip(
                self.momentum_n_m_s + clipped * dt_s,
                -self.max_momentum_n_m_s,
                self.max_momentum_n_m_s,
            )
            realized_wheel_torque = (next_momentum - self.momentum_n_m_s) / dt_s
            self.momentum_n_m_s = next_momentum
        else:
            realized_wheel_torque = clipped
        requested_body = -(self.axes_body.T @ requested)
        realized_body = -(self.axes_body.T @ realized_wheel_torque)
        return ActuatorRealization(
            self.actuator_id,
            start_time_ns,
            end_time_ns,
            None if demand.source_command is None else demand.source_command.command_id,
            demand.mode,
            (0.0, 0.0, 0.0),
            tuple(float(value) for value in requested_body),
            (0.0, 0.0, 0.0),
            tuple(float(value) for value in realized_body),
            device_state=tuple(
                TelemetryField(f"wheel_{index}_momentum_n_m_s", float(value), "N*m*s")
                for index, value in enumerate(self.momentum_n_m_s)
            ) + (() if self.inertia_kg_m2 is None else tuple(
                TelemetryField(f"wheel_{index}_speed_rad_s", float(value), "rad/s")
                for index, value in enumerate(self.momentum_n_m_s / self.inertia_kg_m2)
            )),
            saturated=bool(np.any(np.abs(requested - realized_wheel_torque) > 1.0e-15)),
        )

    def snapshot_state(self) -> dict[str, object]:
        return {"momentum_n_m_s": self.momentum_n_m_s.tolist()}

    def restore_state(self, state: object) -> None:
        if not isinstance(state, dict):
            raise ValueError("reaction-wheel checkpoint must be an object")
        momentum = np.asarray(state.get("momentum_n_m_s"), dtype=float)
        if (
            momentum.shape != self.momentum_n_m_s.shape
            or not np.all(np.isfinite(momentum))
            or np.any(np.abs(momentum) > self.max_momentum_n_m_s)
        ):
            raise ValueError("reaction-wheel checkpoint momentum is invalid")
        self.momentum_n_m_s = momentum


class MagnetorquerHardware(_NativeIntervalCacheState):
    def __init__(
        self,
        actuator_id: str,
        *,
        max_dipole_a_m2: tuple[float, ...],
        magnetic_field_body_t: Vector3,
    ) -> None:
        if not actuator_id.strip():
            raise ValueError("magnetorquer actuator_id must be non-empty")
        self.actuator_id = actuator_id
        self.max_dipole_a_m2 = _coordinate_limits(max_dipole_a_m2, 3, "magnetorquer dipole")
        self.magnetic_field_body_t = np.asarray(magnetic_field_body_t, dtype=float).reshape(3)
        if not np.all(np.isfinite(self.magnetic_field_body_t)):
            raise ValueError("magnetic field must contain three finite values")

    def advance(self, demand: ActuatorDemand, *, start_time_ns: int, end_time_ns: int) -> ActuatorRealization:
        if end_time_ns < start_time_ns:
            raise ValueError("actuator interval end must not precede start")
        payload = demand.payload
        if payload is not None and not isinstance(payload, MagnetorquerDipoleCommand):
            raise TypeError("magnetorquer hardware requires MagnetorquerDipoleCommand demand")
        if type(self) is MagnetorquerHardware and getattr(self, "_native_actuator_intervals", False):
            from sim.actuators.rust_physical import advance_hardware
            return advance_hardware(self, demand, start_time_ns, end_time_ns)
        requested = np.zeros(3) if payload is None else np.asarray(payload.dipole_a_m2, dtype=float)
        realized = np.clip(requested, -self.max_dipole_a_m2, self.max_dipole_a_m2)
        requested_torque = np.cross(requested, self.magnetic_field_body_t)
        realized_torque = np.cross(realized, self.magnetic_field_body_t)
        return ActuatorRealization(
            self.actuator_id,
            start_time_ns,
            end_time_ns,
            None if demand.source_command is None else demand.source_command.command_id,
            demand.mode,
            (0.0, 0.0, 0.0),
            tuple(float(value) for value in requested_torque),
            (0.0, 0.0, 0.0),
            tuple(float(value) for value in realized_torque),
            device_state=tuple(
                TelemetryField(f"dipole_{axis}_a_m2", float(value), "A*m^2")
                for axis, value in zip("xyz", realized)
            ),
            saturated=bool(np.any(requested != realized)),
        )

    def snapshot_state(self) -> dict[str, object]:
        return {}

    def restore_state(self, state: object) -> None:
        if state != {}:
            raise ValueError("magnetorquer checkpoint must be empty")


class CmgHardware(_NativeIntervalCacheState):
    def __init__(
        self,
        actuator_id: str,
        *,
        momentum_n_m_s: tuple[float, ...],
        max_gimbal_rate_rad_s: tuple[float, ...],
    ) -> None:
        if not actuator_id.strip():
            raise ValueError("CMG actuator_id must be non-empty")
        self.actuator_id = actuator_id
        self.momentum_n_m_s = _coordinate_limits(momentum_n_m_s, 3, "CMG momentum")
        self.max_gimbal_rate_rad_s = _coordinate_limits(max_gimbal_rate_rad_s, 3, "CMG gimbal rate")
        self.gimbal_angle_rad = np.zeros(3)

    def advance(self, demand: ActuatorDemand, *, start_time_ns: int, end_time_ns: int) -> ActuatorRealization:
        if end_time_ns < start_time_ns:
            raise ValueError("actuator interval end must not precede start")
        payload = demand.payload
        if payload is not None and not isinstance(payload, CmgGimbalRateCommand):
            raise TypeError("CMG hardware requires CmgGimbalRateCommand demand")
        if type(self) is CmgHardware and getattr(self, "_native_actuator_intervals", False):
            from sim.actuators.rust_physical import advance_hardware
            return advance_hardware(self, demand, start_time_ns, end_time_ns)
        requested = np.zeros(3) if payload is None else np.asarray(payload.gimbal_rate_rad_s, dtype=float)
        realized = np.clip(requested, -self.max_gimbal_rate_rad_s, self.max_gimbal_rate_rad_s)
        self.gimbal_angle_rad += realized * ((end_time_ns - start_time_ns) / 1.0e9)
        requested_torque = requested * self.momentum_n_m_s
        realized_torque = realized * self.momentum_n_m_s
        return ActuatorRealization(
            self.actuator_id,
            start_time_ns,
            end_time_ns,
            None if demand.source_command is None else demand.source_command.command_id,
            demand.mode,
            (0.0, 0.0, 0.0),
            tuple(float(value) for value in requested_torque),
            (0.0, 0.0, 0.0),
            tuple(float(value) for value in realized_torque),
            device_state=tuple(
                TelemetryField(f"gimbal_{axis}_angle_rad", float(value), "rad")
                for axis, value in zip("xyz", self.gimbal_angle_rad)
            ),
            saturated=bool(np.any(requested != realized)),
        )

    def snapshot_state(self) -> dict[str, object]:
        return {"gimbal_angle_rad": self.gimbal_angle_rad.tolist()}

    def restore_state(self, state: object) -> None:
        if not isinstance(state, dict):
            raise ValueError("CMG checkpoint must be an object")
        angles = np.asarray(state.get("gimbal_angle_rad"), dtype=float)
        if angles.shape != (3,) or not np.all(np.isfinite(angles)):
            raise ValueError("CMG checkpoint angles are invalid")
        self.gimbal_angle_rad = angles


class ContinuousEngineHardware(_NativeIntervalCacheState):
    """Physical continuous engine with body-frame yaw/pitch gimbals."""

    def __init__(self, actuator_id: str, *, max_thrust_n: float, specific_impulse_s: float | None = None) -> None:
        if not actuator_id.strip() or not isfinite(max_thrust_n) or max_thrust_n <= 0.0:
            raise ValueError("continuous engine identity and positive finite max thrust are required")
        self.actuator_id = actuator_id
        self.max_thrust_n = float(max_thrust_n)
        if specific_impulse_s is not None and (not isfinite(specific_impulse_s) or specific_impulse_s <= 0.0):
            raise ValueError("specific_impulse_s must be positive and finite when provided")
        self.specific_impulse_s = None if specific_impulse_s is None else float(specific_impulse_s)

    def advance(
        self,
        demand: ActuatorDemand,
        *,
        start_time_ns: int,
        end_time_ns: int,
        attitude_quat_bn: Vector3 | tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> ActuatorRealization:
        if end_time_ns < start_time_ns:
            raise ValueError("actuator interval must be nonnegative")
        payload = demand.payload
        if payload is not None and not isinstance(payload, ContinuousEngineCommand):
            raise TypeError("continuous engine hardware requires ContinuousEngineCommand demand")
        if type(self) is ContinuousEngineHardware and getattr(self, "_native_actuator_intervals", False):
            from sim.actuators.rust_physical import advance_hardware
            return advance_hardware(self, demand, start_time_ns, end_time_ns)
        throttle = float(payload.throttle_0_1) if isinstance(payload, ContinuousEngineCommand) else 0.0
        angles = payload.gimbal_angles_rad if isinstance(payload, ContinuousEngineCommand) else ()
        yaw = float(angles[0]) if len(angles) >= 1 else 0.0
        pitch = float(angles[1]) if len(angles) >= 2 else 0.0
        body_direction = np.array(
            [
                cos(pitch) * cos(yaw),
                cos(pitch) * sin(yaw),
                -sin(pitch),
            ],
            dtype=float,
        )
        # The device command and realization are in the actuator/body frame;
        # the coupled dynamics owner performs the stage-consistent transform.
        realized = tuple(float(value) for value in body_direction * throttle * self.max_thrust_n)
        return ActuatorRealization(
            actuator_id=self.actuator_id,
            interval_start_ns=start_time_ns,
            interval_end_ns=end_time_ns,
            source_command_id=None if demand.source_command is None else demand.source_command.command_id,
            demand_mode=demand.mode,
            requested_force_n=realized,
            requested_torque_n_m=(0.0, 0.0, 0.0),
            realized_force_n=realized,
            realized_torque_n_m=(0.0, 0.0, 0.0),
            mass_flow_kg_s=_mass_flow_kg_s(realized, self.specific_impulse_s),
            device_state=(
                TelemetryField("throttle", throttle, "fraction"),
                TelemetryField("gimbal_yaw", yaw, "rad"),
                TelemetryField("gimbal_pitch", pitch, "rad"),
            ),
        )

    def snapshot_state(self) -> dict[str, object]:
        return {}

    def restore_state(self, state: object) -> None:
        if state != {}:
            raise ValueError("continuous-engine checkpoint must be empty")


class RcsThrusterHardware:
    """One body-fixed RCS jet with an optional COM-relative mounting arm."""

    def __init__(
        self,
        actuator_id: str,
        *,
        direction_body: Vector3,
        max_thrust_n: float,
        position_body_m: Vector3 = (0.0, 0.0, 0.0),
        specific_impulse_s: float | None = None,
    ) -> None:
        direction = np.asarray(direction_body, dtype=float)
        norm = float(np.linalg.norm(direction))
        if not actuator_id.strip() or not isfinite(max_thrust_n) or max_thrust_n <= 0.0:
            raise ValueError("RCS identity and positive finite thrust are required")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)) or abs(norm - 1.0) > 1.0e-10:
            raise ValueError("RCS body direction must be a normalized three-vector")
        position = np.asarray(position_body_m, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("RCS position must be a finite COM-relative body-frame three-vector")
        self.position_body_m = position
        self.actuator_id = actuator_id
        self.direction_body = direction
        self.max_thrust_n = float(max_thrust_n)
        if specific_impulse_s is not None and (not isfinite(specific_impulse_s) or specific_impulse_s <= 0.0):
            raise ValueError("specific_impulse_s must be positive and finite when provided")
        self.specific_impulse_s = None if specific_impulse_s is None else float(specific_impulse_s)

    def advance(self, demand: ActuatorDemand, *, start_time_ns: int, end_time_ns: int) -> ActuatorRealization:
        if end_time_ns < start_time_ns:
            raise ValueError("actuator interval must be nonnegative")
        payload = demand.payload
        if payload is not None and not isinstance(payload, (ThrusterPulseCommand, ThrusterOnOffCommand)):
            raise TypeError("RCS hardware requires a thruster pulse or on/off demand")
        enabled = isinstance(payload, ThrusterPulseCommand) or (
            isinstance(payload, ThrusterOnOffCommand) and payload.enabled
        )
        if payload is not None and payload.thruster_id != self.actuator_id:
            raise ValueError("RCS command thruster_id does not match physical device")
        force = tuple(float(value) for value in self.direction_body * (self.max_thrust_n if enabled else 0.0))
        torque = tuple(float(value) if value != 0.0 else 0.0 for value in np.cross(self.position_body_m, force))
        return ActuatorRealization(
            self.actuator_id,
            start_time_ns,
            end_time_ns,
            None if demand.source_command is None else demand.source_command.command_id,
            demand.mode,
            force,
            torque,
            force,
            torque,
            _mass_flow_kg_s(force, self.specific_impulse_s),
            (TelemetryField("valve_open", enabled),),
        )

    def snapshot_state(self) -> dict[str, object]:
        return {}

    def restore_state(self, state: object) -> None:
        if state != {}:
            raise ValueError("RCS checkpoint must be empty")


class AerodynamicEffectorHardware:
    def __init__(
        self,
        actuator_id: str,
        coordinate_id: str,
        *,
        unit: str,
        minimum: float,
        maximum: float,
        neutral: float,
        rate_limit_per_s: float,
    ) -> None:
        values = (minimum, maximum, neutral, rate_limit_per_s)
        if not actuator_id.strip() or not coordinate_id.strip() or unit not in ("rad", "m", "fraction"):
            raise ValueError("effector identity and supported unit are required")
        if not all(isfinite(float(value)) for value in values) or minimum > neutral or neutral > maximum:
            raise ValueError("effector limits must be finite and contain neutral")
        if rate_limit_per_s < 0.0:
            raise ValueError("rate_limit_per_s must be nonnegative")
        self.actuator_id = actuator_id
        self.coordinate_id = coordinate_id
        self.unit = unit
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.neutral = float(neutral)
        self.rate_limit_per_s = float(rate_limit_per_s)
        self.position = float(neutral)

    def validate(self, payload: object) -> tuple[bool, str | None]:
        if not isinstance(payload, AerodynamicEffectorPositionCommand):
            return False, "wrong_effector_payload"
        if payload.coordinate_id != self.coordinate_id or payload.unit != self.unit:
            return False, "effector_coordinate_mismatch"
        if not self.minimum <= payload.position <= self.maximum:
            return False, "effector_position_out_of_range"
        return True, None

    def advance(self, demand: ActuatorDemand, *, start_time_ns: int, end_time_ns: int) -> ActuatorRealization:
        if end_time_ns < start_time_ns:
            raise ValueError("actuator interval must be nonnegative")
        if demand.mode is DemandMode.ZERO:
            target = self.neutral
        elif isinstance(demand.payload, AerodynamicEffectorPositionCommand):
            target = float(demand.payload.position)
        else:
            target = self.position
        dt_s = (end_time_ns - start_time_ns) / 1.0e9
        max_delta = self.rate_limit_per_s * dt_s
        delta = max(-max_delta, min(max_delta, target - self.position))
        self.position += delta
        return ActuatorRealization(
            actuator_id=self.actuator_id,
            interval_start_ns=start_time_ns,
            interval_end_ns=end_time_ns,
            source_command_id=None if demand.source_command is None else demand.source_command.command_id,
            demand_mode=demand.mode,
            requested_force_n=(0.0, 0.0, 0.0),
            requested_torque_n_m=(0.0, 0.0, 0.0),
            realized_force_n=(0.0, 0.0, 0.0),
            realized_torque_n_m=(0.0, 0.0, 0.0),
            device_state=(TelemetryField("position", self.position, self.unit),),
            saturated=abs(self.position - target) > 1.0e-12,
        )

    def snapshot_state(self) -> dict[str, object]:
        return {"position": self.position}

    def restore_state(self, state: object) -> None:
        if not isinstance(state, dict) or set(state) != {"position"}:
            raise ValueError("aerodynamic-effector checkpoint is invalid")
        position = float(state["position"])
        if not self.minimum <= position <= self.maximum:
            raise ValueError("aerodynamic-effector checkpoint position is outside device limits")
        self.position = position


def _limit_norm(vector: Vector3, maximum: float) -> tuple[Vector3, bool]:
    values = tuple(float(value) for value in vector)
    if len(values) != 3 or not all(isfinite(value) for value in values):
        raise ValueError("wrench command vectors must be finite three-vectors")
    if isnan(float(maximum)) or maximum < 0.0:
        raise ValueError("wrench limit must be nonnegative and not NaN")
    norm = sqrt(sum(value**2 for value in values))
    if norm <= maximum or norm == 0.0:
        return values, False  # type: ignore[return-value]
    scale = maximum / norm
    return tuple(value * scale for value in values), True  # type: ignore[return-value]


def _coordinate_limits(values: tuple[float, ...], count: int, name: str) -> np.ndarray:
    limits = np.asarray(values, dtype=float).reshape(-1)
    if limits.size == 1:
        limits = np.full(count, float(limits[0]))
    if limits.size != count or not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError(f"{name} limits must be positive finite scalar or length {count}")
    return limits


def _blend(previous: Vector3, target: Vector3, alpha: float) -> Vector3:
    return tuple(float(old) + alpha * (float(new) - float(old)) for old, new in zip(previous, target))  # type: ignore[return-value]


def _mass_flow_kg_s(force_n: Vector3, specific_impulse_s: float | None) -> float:
    """Return positive propellant flow for a realized thrust vector."""

    if specific_impulse_s is None:
        return 0.0
    standard_gravity_m_s2 = 9.80665
    return sqrt(sum(float(value) ** 2 for value in force_n)) / (specific_impulse_s * standard_gravity_m_s2)
