"""Versioned, truth-free boundary contracts for satellite flight software.

This module intentionally depends only on the Python standard library.  It is
safe for public custom stacks and must not import simulator truth, dynamics,
sensor, actuator-realization, or runtime implementation types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite, sqrt
from typing import Literal, Protocol, runtime_checkable

CONTRACT_VERSION = "1.0"
INPUT_BATCH_SCHEMA = "oel.flight_software.input_batch.v1"
OUTPUT_SCHEMA = "oel.flight_software.output.v1"
ACTUATOR_COMMAND_SCHEMA = "oel.flight_software.actuator_command.v1"
SNAPSHOT_SCHEMA = "oel.flight_software.snapshot.v1"
NATIVE_SNAPSHOT_SCHEMA = "oel.flight_software.native_snapshot.v1"
PILOT_INPUT_SCHEMA = "oel.flight_software.pilot_input.v1"
GROUND_COMMAND_SCHEMA = "oel.flight_software.ground_command.v1"


class ClockScale(str, Enum):
    ONBOARD = "onboard"
    GPS = "gps"
    TAI = "tai"
    UTC = "utc"


class TimeValidity(str, Enum):
    VALID = "valid"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"


class DataValidity(str, Enum):
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"


class InputKind(str, Enum):
    MEASUREMENT = "measurement"
    ACTUATOR_RECEIPT = "actuator_receipt"
    ACTUATOR_TELEMETRY = "actuator_telemetry"
    MISSION_LOAD = "mission_load"
    STACK_LOAD = "stack_load"
    GROUND_COMMAND = "ground_command"
    PILOT_INPUT = "pilot_input"
    CROSSLINK = "crosslink"
    CLOCK_EVENT = "clock_event"
    MODELED_FAULT_INDICATION = "modeled_fault_indication"


class GroundCommandKind(str, Enum):
    ACTION_REQUEST = "action_request"
    GOAL_UPDATE = "goal_update"
    STACK_COMMAND = "stack_command"


class CommandDisposition(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED_SCHEMA = "rejected_schema"
    REJECTED_VERSION = "rejected_version"
    REJECTED_TARGET = "rejected_target"
    REJECTED_SEQUENCE = "rejected_sequence"
    REJECTED_TIME = "rejected_time"
    REJECTED_FRAME = "rejected_frame"
    REJECTED_VALUE = "rejected_value"
    REJECTED_INTERLOCK = "rejected_interlock"
    REJECTED_DEVICE_STATE = "rejected_device_state"


ScalarTelemetryValue = str | int | float | bool | None
Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
Matrix = tuple[tuple[float, ...], ...]


def _require_nonempty(name: str, value: str) -> str:
    text = str(value)
    if not text.strip():
        raise ValueError(f"{name} must be non-empty")
    return text


def _require_int(name: str, value: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_vector(name: str, value: tuple[float, ...], *, length: int) -> tuple[float, ...]:
    result = tuple(_require_finite(f"{name}[{index}]", item) for index, item in enumerate(value))
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return result


def _require_unit_vector(name: str, value: tuple[float, ...], *, tolerance: float = 1.0e-10) -> tuple[float, ...]:
    result = _require_vector(name, value, length=3)
    norm = sqrt(sum(item * item for item in result))
    if abs(norm - 1.0) > tolerance:
        raise ValueError(f"{name} must be normalized within {tolerance:g}")
    return result


def _require_quaternion(name: str, value: tuple[float, ...]) -> tuple[float, ...]:
    result = _require_vector(name, value, length=4)
    norm = sqrt(sum(item * item for item in result))
    if abs(norm - 1.0) > 1.0e-10:
        raise ValueError(f"{name} must be scalar-first and normalized within 1e-10")
    return result


def _require_covariance(name: str, covariance: Matrix, *, size: int | None = None) -> Matrix:
    matrix = tuple(
        tuple(_require_finite(f"{name}[{row}][{col}]", item) for col, item in enumerate(values))
        for row, values in enumerate(covariance)
    )
    if size is None:
        size = len(matrix)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError(f"{name} must be a {size}x{size} matrix")
    for row in range(size):
        for col in range(row + 1, size):
            scale = max(abs(matrix[row][col]), abs(matrix[col][row]), 1.0)
            if abs(matrix[row][col] - matrix[col][row]) > 1.0e-12 * scale:
                raise ValueError(f"{name} must be symmetric within relative tolerance 1e-12")
    _require_positive_semidefinite(name, matrix)
    return matrix


def _require_positive_semidefinite(name: str, matrix: Matrix) -> None:
    """Validate PSD using a tolerance-aware LDL decomposition.

    A zero pivot may occur in a valid semidefinite matrix. In that case every
    remaining residual coupled to the pivot must also be zero within the
    declared covariance tolerance.
    """

    size = len(matrix)
    scale = max((abs(value) for row in matrix for value in row), default=1.0)
    tolerance = 1.0e-12 * max(scale, 1.0)
    lower = [[0.0] * size for _ in range(size)]
    diagonal = [0.0] * size
    for row in range(size):
        lower[row][row] = 1.0
        for col in range(row):
            residual = matrix[row][col] - sum(
                lower[row][index] * diagonal[index] * lower[col][index] for index in range(col)
            )
            if abs(diagonal[col]) <= tolerance:
                if abs(residual) > tolerance:
                    raise ValueError(f"{name} must be positive semidefinite within relative tolerance 1e-12")
                lower[row][col] = 0.0
            else:
                lower[row][col] = residual / diagonal[col]
        pivot = matrix[row][row] - sum(lower[row][index] * lower[row][index] * diagonal[index] for index in range(row))
        if pivot < -tolerance:
            raise ValueError(f"{name} must be positive semidefinite within relative tolerance 1e-12")
        diagonal[row] = 0.0 if abs(pivot) <= tolerance else pivot


@dataclass(frozen=True, slots=True)
class ClockTag:
    clock_id: str
    ticks: int
    tick_period_ns: int
    scale: ClockScale
    validity: TimeValidity = TimeValidity.VALID
    reset_counter: int = 0

    def __post_init__(self) -> None:
        _require_nonempty("clock_id", self.clock_id)
        _require_int("ticks", self.ticks)
        _require_int("tick_period_ns", self.tick_period_ns, minimum=1)
        _require_int("reset_counter", self.reset_counter)
        if not isinstance(self.scale, ClockScale):
            raise TypeError("scale must be ClockScale")
        if not isinstance(self.validity, TimeValidity):
            raise TypeError("validity must be TimeValidity")


@dataclass(frozen=True, slots=True)
class PacketId:
    source_id: str
    boot_id: str
    sequence: int

    def __post_init__(self) -> None:
        _require_nonempty("source_id", self.source_id)
        _require_nonempty("boot_id", self.boot_id)
        _require_int("sequence", self.sequence)
        if self.sequence >= 2**64:
            raise ValueError("sequence must fit in an unsigned 64-bit integer")


@dataclass(frozen=True, slots=True)
class FrameId:
    name: str
    registry_version: str

    def __post_init__(self) -> None:
        _require_nonempty("name", self.name)
        _require_nonempty("registry_version", self.registry_version)


@dataclass(frozen=True, slots=True)
class ValidityInterval:
    not_before: ClockTag
    expires_at: ClockTag | None = None

    def __post_init__(self) -> None:
        if self.expires_at is None:
            return
        if (
            self.not_before.clock_id != self.expires_at.clock_id
            or self.not_before.reset_counter != self.expires_at.reset_counter
            or self.not_before.tick_period_ns != self.expires_at.tick_period_ns
            or self.not_before.scale != self.expires_at.scale
        ):
            raise ValueError("validity interval endpoints must use the same clock domain and reset")
        if self.expires_at.ticks < self.not_before.ticks:
            raise ValueError("expires_at must not precede not_before")


@dataclass(frozen=True, slots=True)
class Quality:
    validity: DataValidity = DataValidity.VALID
    status_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.validity, DataValidity):
            raise TypeError("validity must be DataValidity")
        for index, code in enumerate(self.status_codes):
            _require_nonempty(f"status_codes[{index}]", code)


@dataclass(frozen=True, slots=True)
class TelemetryField:
    name: str
    value: ScalarTelemetryValue
    unit: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty("name", self.name)
        if isinstance(self.value, float):
            _require_finite("value", self.value)
        if self.unit is not None:
            _require_nonempty("unit", self.unit)


@dataclass(frozen=True, slots=True)
class GyroMeasurement:
    angular_rate_rad_s: Vector3
    covariance_rad2_s2: Matrix | None = None
    schema: Literal["gyro.v1"] = "gyro.v1"

    def __post_init__(self) -> None:
        _require_vector("angular_rate_rad_s", self.angular_rate_rad_s, length=3)
        if self.covariance_rad2_s2 is not None:
            _require_covariance("covariance_rad2_s2", self.covariance_rad2_s2, size=3)


@dataclass(frozen=True, slots=True)
class StarTrackerMeasurement:
    quat_sensor_from_inertial: Quaternion
    covariance_small_angle_rad2: Matrix | None = None
    schema: Literal["star_tracker.v1"] = "star_tracker.v1"

    def __post_init__(self) -> None:
        _require_quaternion("quat_sensor_from_inertial", self.quat_sensor_from_inertial)
        if self.covariance_small_angle_rad2 is not None:
            _require_covariance("covariance_small_angle_rad2", self.covariance_small_angle_rad2, size=3)


@dataclass(frozen=True, slots=True)
class SunVectorMeasurement:
    unit_vector: Vector3
    irradiance_w_m2: float | None = None
    covariance: Matrix | None = None
    schema: Literal["sun_vector.v1"] = "sun_vector.v1"

    def __post_init__(self) -> None:
        _require_unit_vector("unit_vector", self.unit_vector)
        if self.irradiance_w_m2 is not None and _require_finite("irradiance_w_m2", self.irradiance_w_m2) < 0.0:
            raise ValueError("irradiance_w_m2 must be nonnegative")
        if self.covariance is not None:
            _require_covariance("covariance", self.covariance, size=3)


@dataclass(frozen=True, slots=True)
class MagnetometerMeasurement:
    magnetic_flux_density_t: Vector3
    covariance_t2: Matrix | None = None
    schema: Literal["magnetometer.v1"] = "magnetometer.v1"

    def __post_init__(self) -> None:
        _require_vector("magnetic_flux_density_t", self.magnetic_flux_density_t, length=3)
        if self.covariance_t2 is not None:
            _require_covariance("covariance_t2", self.covariance_t2, size=3)


@dataclass(frozen=True, slots=True)
class GnssOwnStateMeasurement:
    position_m: Vector3
    velocity_m_s: Vector3
    receiver_clock_bias_s: float = 0.0
    receiver_clock_drift_s_s: float = 0.0
    covariance: Matrix | None = None
    schema: Literal["gnss_own_state.v1"] = "gnss_own_state.v1"

    def __post_init__(self) -> None:
        _require_vector("position_m", self.position_m, length=3)
        _require_vector("velocity_m_s", self.velocity_m_s, length=3)
        _require_finite("receiver_clock_bias_s", self.receiver_clock_bias_s)
        _require_finite("receiver_clock_drift_s_s", self.receiver_clock_drift_s_s)
        if self.covariance is not None:
            _require_covariance("covariance", self.covariance, size=8)


@dataclass(frozen=True, slots=True)
class RelativeObservationMeasurement:
    range_m: float | None = None
    range_rate_m_s: float | None = None
    los_unit: Vector3 | None = None
    angular_rate_rad_s: Vector3 | None = None
    covariance_order: tuple[str, ...] = ()
    covariance: Matrix | None = None
    target_track_id: str | None = None
    schema: Literal["relative_observation.v1"] = "relative_observation.v1"

    def __post_init__(self) -> None:
        if all(value is None for value in (self.range_m, self.range_rate_m_s, self.los_unit, self.angular_rate_rad_s)):
            raise ValueError("relative observation must contain at least one observable")
        if self.range_m is not None and _require_finite("range_m", self.range_m) < 0.0:
            raise ValueError("range_m must be nonnegative")
        if self.range_rate_m_s is not None:
            _require_finite("range_rate_m_s", self.range_rate_m_s)
        if self.los_unit is not None:
            _require_unit_vector("los_unit", self.los_unit)
        if self.angular_rate_rad_s is not None:
            _require_vector("angular_rate_rad_s", self.angular_rate_rad_s, length=3)
        if self.target_track_id is not None:
            _require_nonempty("target_track_id", self.target_track_id)
        if self.covariance is not None:
            if not self.covariance_order:
                raise ValueError("covariance_order is required when covariance is supplied")
            _require_covariance("covariance", self.covariance, size=len(self.covariance_order))


@dataclass(frozen=True, slots=True)
class IdealOwnStateMeasurement:
    position_m: Vector3 | None = None
    velocity_m_s: Vector3 | None = None
    attitude_quat_body_from_inertial: Quaternion | None = None
    angular_rate_body_rad_s: Vector3 | None = None
    mass_kg: float | None = None
    covariance_order: tuple[str, ...] = ()
    covariance: Matrix | None = None
    schema: Literal["ideal_own_state.v1"] = "ideal_own_state.v1"

    def __post_init__(self) -> None:
        values = (
            self.position_m,
            self.velocity_m_s,
            self.attitude_quat_body_from_inertial,
            self.angular_rate_body_rad_s,
            self.mass_kg,
        )
        if all(value is None for value in values):
            raise ValueError("ideal own-state measurement must enable at least one observable")
        if self.position_m is not None:
            _require_vector("position_m", self.position_m, length=3)
        if self.velocity_m_s is not None:
            _require_vector("velocity_m_s", self.velocity_m_s, length=3)
        if self.attitude_quat_body_from_inertial is not None:
            _require_quaternion("attitude_quat_body_from_inertial", self.attitude_quat_body_from_inertial)
        if self.angular_rate_body_rad_s is not None:
            _require_vector("angular_rate_body_rad_s", self.angular_rate_body_rad_s, length=3)
        if self.mass_kg is not None and _require_finite("mass_kg", self.mass_kg) < 0.0:
            raise ValueError("mass_kg must be nonnegative")
        if self.covariance is not None:
            if not self.covariance_order:
                raise ValueError("covariance_order is required when covariance is supplied")
            _require_covariance("covariance", self.covariance, size=len(self.covariance_order))


@dataclass(frozen=True, slots=True)
class IdealTrackedObjectStateMeasurement:
    target_id: str
    position_m: Vector3
    velocity_m_s: Vector3
    schema: Literal["ideal_tracked_object_state.v1"] = "ideal_tracked_object_state.v1"

    def __post_init__(self) -> None:
        _require_nonempty("target_id", self.target_id)
        _require_vector("position_m", self.position_m, length=3)
        _require_vector("velocity_m_s", self.velocity_m_s, length=3)


@dataclass(frozen=True, slots=True)
class VehicleResourceMeasurement:
    """Onboard platform-resource measurement, independent of simulator truth."""

    battery_soc: float | None = None
    available_power_w: float | None = None
    maximum_temperature_k: float | None = None
    storage_used_bytes: float | None = None
    storage_capacity_bytes: float | None = None
    propellant_kg: float | None = None
    schema: Literal["vehicle_resources.v1"] = "vehicle_resources.v1"

    def __post_init__(self) -> None:
        values = (
            self.battery_soc,
            self.available_power_w,
            self.maximum_temperature_k,
            self.storage_used_bytes,
            self.storage_capacity_bytes,
            self.propellant_kg,
        )
        if all(value is None for value in values):
            raise ValueError("vehicle resource measurement must contain at least one observable")
        if self.battery_soc is not None and not 0.0 <= _require_finite("battery_soc", self.battery_soc) <= 1.0:
            raise ValueError("battery_soc must be in [0, 1]")
        for name, value in (
            ("available_power_w", self.available_power_w),
            ("maximum_temperature_k", self.maximum_temperature_k),
            ("storage_used_bytes", self.storage_used_bytes),
            ("storage_capacity_bytes", self.storage_capacity_bytes),
            ("propellant_kg", self.propellant_kg),
        ):
            if value is not None and _require_finite(name, value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.maximum_temperature_k is not None and self.maximum_temperature_k <= 0.0:
            raise ValueError("maximum_temperature_k must be positive")
        if (
            self.storage_used_bytes is not None
            and self.storage_capacity_bytes is not None
            and self.storage_used_bytes > self.storage_capacity_bytes
        ):
            raise ValueError("storage_used_bytes cannot exceed storage_capacity_bytes")


@dataclass(frozen=True, slots=True)
class ActuatorTelemetryPayload:
    actuator_id: str
    fields: tuple[TelemetryField, ...]
    schema: Literal["actuator_telemetry.v1"] = "actuator_telemetry.v1"

    def __post_init__(self) -> None:
        _require_nonempty("actuator_id", self.actuator_id)


MeasurementPayload = (
    GyroMeasurement
    | StarTrackerMeasurement
    | SunVectorMeasurement
    | MagnetometerMeasurement
    | GnssOwnStateMeasurement
    | RelativeObservationMeasurement
    | IdealOwnStateMeasurement
    | IdealTrackedObjectStateMeasurement
    | VehicleResourceMeasurement
)


@dataclass(frozen=True, slots=True)
class MeasurementEvent:
    sensor_id: str
    measurement_type: str
    sample_time: ClockTag
    frame: FrameId
    payload: MeasurementPayload
    schema: Literal["oel.flight_software.measurement_event.v1"] = "oel.flight_software.measurement_event.v1"

    def __post_init__(self) -> None:
        _require_nonempty("sensor_id", self.sensor_id)
        _require_nonempty("measurement_type", self.measurement_type)
        if self.measurement_type != self.payload.schema:
            raise ValueError("measurement_type must match payload.schema")


@dataclass(frozen=True, slots=True)
class ControlAxisSample:
    control_id: str
    value: float

    def __post_init__(self) -> None:
        _require_nonempty("control_id", self.control_id)
        value = _require_finite("value", self.value)
        if not -1.0 <= value <= 1.0:
            raise ValueError("control axis value must be in [-1, 1]")


@dataclass(frozen=True, slots=True)
class PilotInputPayload:
    input_profile_id: str
    axes: tuple[ControlAxisSample, ...] = ()
    pressed_actions: tuple[str, ...] = ()
    released_actions: tuple[str, ...] = ()
    schema: Literal["oel.flight_software.pilot_input.v1"] = PILOT_INPUT_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty("input_profile_id", self.input_profile_id)
        axis_names = [item.control_id for item in self.axes]
        if len(axis_names) != len(set(axis_names)):
            raise ValueError("pilot input axes must have unique control_id values")
        for name, values in (("pressed_actions", self.pressed_actions), ("released_actions", self.released_actions)):
            for index, value in enumerate(values):
                _require_nonempty(f"{name}[{index}]", value)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class GroundCommandPayload:
    command_id: str
    kind: GroundCommandKind
    parameters: tuple[TelemetryField, ...] = ()
    execute_at: ClockTag | None = None
    schema: Literal["oel.flight_software.ground_command.v1"] = GROUND_COMMAND_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty("command_id", self.command_id)
        if not isinstance(self.kind, GroundCommandKind):
            raise TypeError("kind must be GroundCommandKind")
        names = [field.name for field in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("ground-command parameter names must be unique")


@dataclass(frozen=True, slots=True)
class ModeledFaultIndicationPayload:
    component_id: str
    fault_code: str
    active: bool
    detected_by: str
    schema: Literal["oel.flight_software.modeled_fault_indication.v1"] = (
        "oel.flight_software.modeled_fault_indication.v1"
    )

    def __post_init__(self) -> None:
        _require_nonempty("component_id", self.component_id)
        _require_nonempty("fault_code", self.fault_code)
        _require_nonempty("detected_by", self.detected_by)
        if not isinstance(self.active, bool):
            raise TypeError("active must be boolean")


@dataclass(frozen=True, slots=True)
class ReactionWheelTorqueCommand:
    torque_n_m: tuple[float, ...]
    schema: Literal["reaction_wheel_torque.v1"] = "reaction_wheel_torque.v1"

    def __post_init__(self) -> None:
        if not self.torque_n_m:
            raise ValueError("torque_n_m must address at least one wheel axis")
        _require_vector("torque_n_m", self.torque_n_m, length=len(self.torque_n_m))


@dataclass(frozen=True, slots=True)
class MagnetorquerDipoleCommand:
    dipole_a_m2: tuple[float, ...]
    schema: Literal["magnetorquer_dipole.v1"] = "magnetorquer_dipole.v1"

    def __post_init__(self) -> None:
        if not self.dipole_a_m2:
            raise ValueError("dipole_a_m2 must address at least one torquer axis")
        _require_vector("dipole_a_m2", self.dipole_a_m2, length=len(self.dipole_a_m2))


@dataclass(frozen=True, slots=True)
class CmgGimbalRateCommand:
    gimbal_rate_rad_s: tuple[float, ...]
    schema: Literal["cmg_gimbal_rate.v1"] = "cmg_gimbal_rate.v1"

    def __post_init__(self) -> None:
        if not self.gimbal_rate_rad_s:
            raise ValueError("gimbal_rate_rad_s must address at least one CMG")
        _require_vector("gimbal_rate_rad_s", self.gimbal_rate_rad_s, length=len(self.gimbal_rate_rad_s))


@dataclass(frozen=True, slots=True)
class ThrusterPulseCommand:
    thruster_id: str
    start_at: ClockTag
    duration_s: float
    schema: Literal["thruster_pulse.v1"] = "thruster_pulse.v1"

    def __post_init__(self) -> None:
        _require_nonempty("thruster_id", self.thruster_id)
        if _require_finite("duration_s", self.duration_s) <= 0.0:
            raise ValueError("duration_s must be positive")


@dataclass(frozen=True, slots=True)
class ThrusterOnOffCommand:
    thruster_id: str
    enabled: bool
    schema: Literal["thruster_on_off.v1"] = "thruster_on_off.v1"

    def __post_init__(self) -> None:
        _require_nonempty("thruster_id", self.thruster_id)
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")


@dataclass(frozen=True, slots=True)
class ContinuousEngineCommand:
    throttle_0_1: float
    gimbal_angles_rad: tuple[float, ...] = ()
    schema: Literal["continuous_engine.v1"] = "continuous_engine.v1"

    def __post_init__(self) -> None:
        throttle = _require_finite("throttle_0_1", self.throttle_0_1)
        if not 0.0 <= throttle <= 1.0:
            raise ValueError("throttle_0_1 must be in [0, 1]")
        _require_vector("gimbal_angles_rad", self.gimbal_angles_rad, length=len(self.gimbal_angles_rad))


@dataclass(frozen=True, slots=True)
class AerodynamicEffectorPositionCommand:
    coordinate_id: str
    position: float
    unit: Literal["rad", "m", "fraction"]
    schema: Literal["aerodynamic_effector_position.v1"] = "aerodynamic_effector_position.v1"

    def __post_init__(self) -> None:
        _require_nonempty("coordinate_id", self.coordinate_id)
        position = _require_finite("position", self.position)
        if self.unit == "fraction" and not 0.0 <= position <= 1.0:
            raise ValueError("dimensionless effector deployment fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class IdealWrenchCommand:
    force_n: Vector3
    torque_n_m: Vector3
    schema: Literal["ideal_wrench.v1"] = "ideal_wrench.v1"

    def __post_init__(self) -> None:
        _require_vector("force_n", self.force_n, length=3)
        _require_vector("torque_n_m", self.torque_n_m, length=3)


ActuatorCommandPayload = (
    ReactionWheelTorqueCommand
    | MagnetorquerDipoleCommand
    | CmgGimbalRateCommand
    | ThrusterPulseCommand
    | ThrusterOnOffCommand
    | ContinuousEngineCommand
    | AerodynamicEffectorPositionCommand
    | IdealWrenchCommand
)


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    command_id: PacketId
    satellite_id: str
    actuator_id: str
    issued_at: ClockTag
    validity: ValidityInterval
    frame: FrameId
    payload: ActuatorCommandPayload
    schema: Literal["oel.flight_software.actuator_command.v1"] = ACTUATOR_COMMAND_SCHEMA
    contract_version: Literal["1.0"] = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_nonempty("satellite_id", self.satellite_id)
        _require_nonempty("actuator_id", self.actuator_id)
        issued_identity = (
            self.issued_at.clock_id,
            self.issued_at.reset_counter,
            self.issued_at.tick_period_ns,
            self.issued_at.scale,
        )
        validity_identity = (
            self.validity.not_before.clock_id,
            self.validity.not_before.reset_counter,
            self.validity.not_before.tick_period_ns,
            self.validity.not_before.scale,
        )
        if validity_identity != issued_identity:
            raise ValueError("issued_at and validity must use the same clock domain")


@dataclass(frozen=True, slots=True)
class ActuatorCommandReceipt:
    command_id: PacketId
    received_at: ClockTag
    disposition: CommandDisposition
    status_codes: tuple[str, ...] = ()
    schema: Literal["oel.flight_software.actuator_command_receipt.v1"] = (
        "oel.flight_software.actuator_command_receipt.v1"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, CommandDisposition):
            raise TypeError("disposition must be CommandDisposition")
        for index, code in enumerate(self.status_codes):
            _require_nonempty(f"status_codes[{index}]", code)


InputPayload = (
    MeasurementEvent
    | PilotInputPayload
    | GroundCommandPayload
    | ActuatorCommandReceipt
    | ActuatorTelemetryPayload
    | object
)


@dataclass(frozen=True, slots=True)
class InputEvent:
    packet_id: PacketId
    kind: InputKind
    source_time: ClockTag
    delivery_time: ClockTag
    quality: Quality
    payload: InputPayload
    schema: Literal["oel.flight_software.input_event.v1"] = "oel.flight_software.input_event.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, InputKind):
            raise TypeError("kind must be InputKind")
        expected = {
            InputKind.MEASUREMENT: MeasurementEvent,
            InputKind.PILOT_INPUT: PilotInputPayload,
            InputKind.GROUND_COMMAND: GroundCommandPayload,
            InputKind.ACTUATOR_RECEIPT: ActuatorCommandReceipt,
            InputKind.ACTUATOR_TELEMETRY: ActuatorTelemetryPayload,
            InputKind.MODELED_FAULT_INDICATION: ModeledFaultIndicationPayload,
        }.get(self.kind)
        if expected is not None and not isinstance(self.payload, expected):
            raise TypeError(f"{self.kind.value} event payload must be {expected.__name__}")


@dataclass(frozen=True, slots=True)
class FlightSoftwareInputBatch:
    satellite_id: str
    invocation_id: int
    invocation_time: ClockTag
    events: tuple[InputEvent, ...] = ()
    schema: Literal["oel.flight_software.input_batch.v1"] = INPUT_BATCH_SCHEMA
    contract_version: Literal["1.0"] = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_nonempty("satellite_id", self.satellite_id)
        _require_int("invocation_id", self.invocation_id)
        identities = [event.packet_id for event in self.events]
        if len(identities) != len(set(identities)):
            raise ValueError("input batch cannot contain duplicate packet identities")


@dataclass(frozen=True, slots=True)
class DiagnosticTelemetry:
    topic: str
    generated_at: ClockTag
    fields: tuple[TelemetryField, ...] = ()
    schema: Literal["oel.flight_software.diagnostic_telemetry.v1"] = "oel.flight_software.diagnostic_telemetry.v1"

    def __post_init__(self) -> None:
        _require_nonempty("topic", self.topic)


@dataclass(frozen=True, slots=True)
class TaskReleaseRequest:
    task_id: str
    release_at: ClockTag
    schema: Literal["oel.flight_software.task_release_request.v1"] = "oel.flight_software.task_release_request.v1"

    def __post_init__(self) -> None:
        _require_nonempty("task_id", self.task_id)


@dataclass(frozen=True, slots=True)
class TaskRelease:
    task_id: str
    release_time: ClockTag
    modeled_execution_duration_ns: int = 0
    execution_budget_ns: int | None = None
    schema: Literal["oel.flight_software.task_release.v1"] = "oel.flight_software.task_release.v1"

    def __post_init__(self) -> None:
        _require_nonempty("task_id", self.task_id)
        _require_int("modeled_execution_duration_ns", self.modeled_execution_duration_ns)
        if self.execution_budget_ns is not None:
            _require_int("execution_budget_ns", self.execution_budget_ns, minimum=1)


@dataclass(frozen=True, slots=True)
class StackIdentity:
    stack_id: str
    stack_version: str
    contract_major: int
    implementation_hash: str
    checkpointable: bool

    def __post_init__(self) -> None:
        _require_nonempty("stack_id", self.stack_id)
        _require_nonempty("stack_version", self.stack_version)
        _require_int("contract_major", self.contract_major, minimum=1)
        if len(self.implementation_hash) != 64 or any(ch not in "0123456789abcdef" for ch in self.implementation_hash):
            raise ValueError("implementation_hash must be a 64-character lowercase SHA-256 digest")
        if not isinstance(self.checkpointable, bool):
            raise TypeError("checkpointable must be boolean")


@dataclass(frozen=True, slots=True)
class FlightSoftwareOutput:
    satellite_id: str
    invocation_id: int
    commands: tuple[ActuatorCommand, ...] = ()
    telemetry: tuple[DiagnosticTelemetry, ...] = ()
    requested_next_invocations: tuple[TaskReleaseRequest, ...] = ()
    schema: Literal["oel.flight_software.output.v1"] = OUTPUT_SCHEMA
    contract_version: Literal["1.0"] = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_nonempty("satellite_id", self.satellite_id)
        _require_int("invocation_id", self.invocation_id)
        if any(command.satellite_id != self.satellite_id for command in self.commands):
            raise ValueError("all commands must target the output satellite_id")


@dataclass(frozen=True, slots=True)
class FlightSoftwareSnapshot:
    stack_id: str
    stack_version: str
    boot_id: str
    invocation_id: int
    active_load_id: str | None
    active_load_revision: int | None
    state_bytes: bytes
    state_hash_sha256: str
    schema: Literal["oel.flight_software.snapshot.v1"] = SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty("stack_id", self.stack_id)
        _require_nonempty("stack_version", self.stack_version)
        _require_nonempty("boot_id", self.boot_id)
        _require_int("invocation_id", self.invocation_id)
        if (self.active_load_id is None) != (self.active_load_revision is None):
            raise ValueError("active load ID and revision must either both be present or both be absent")
        if self.active_load_id is not None:
            _require_nonempty("active_load_id", self.active_load_id)
            _require_int("active_load_revision", self.active_load_revision)
        if len(self.state_hash_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.state_hash_sha256):
            raise ValueError("state_hash_sha256 must be a 64-character lowercase SHA-256 digest")
        if sha256(self.state_bytes).hexdigest() != self.state_hash_sha256:
            raise ValueError("state_hash_sha256 does not match state_bytes")


@dataclass(frozen=True, slots=True)
class NativeFlightSoftwareSnapshot:
    """Host-bound envelope for a checkpoint returned by a native endpoint.

    The endpoint still speaks the public ``FlightSoftwareSnapshot`` v1 record.
    The host wraps that record after decoding it and binds the resulting
    checkpoint to a verified native implementation.  Keeping this as a
    separate record preserves the Python stack snapshot wire contract.
    """

    stack_id: str
    stack_version: str
    boot_id: str
    invocation_id: int
    active_load_id: str | None
    active_load_revision: int | None
    state_bytes: bytes
    state_hash_sha256: str
    host_implementation_hash: str
    schema: Literal["oel.flight_software.native_snapshot.v1"] = NATIVE_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NATIVE_SNAPSHOT_SCHEMA:
            raise ValueError(f"schema must equal {NATIVE_SNAPSHOT_SCHEMA!r}")
        _require_nonempty("stack_id", self.stack_id)
        _require_nonempty("stack_version", self.stack_version)
        _require_nonempty("boot_id", self.boot_id)
        _require_int("invocation_id", self.invocation_id)
        if (self.active_load_id is None) != (self.active_load_revision is None):
            raise ValueError("active load ID and revision must either both be present or both be absent")
        if self.active_load_id is not None:
            _require_nonempty("active_load_id", self.active_load_id)
            _require_int("active_load_revision", self.active_load_revision)
        if len(self.state_hash_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.state_hash_sha256):
            raise ValueError("state_hash_sha256 must be a 64-character lowercase SHA-256 digest")
        if sha256(self.state_bytes).hexdigest() != self.state_hash_sha256:
            raise ValueError("state_hash_sha256 does not match state_bytes")
        if len(self.host_implementation_hash) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.host_implementation_hash
        ):
            raise ValueError("host_implementation_hash must be a 64-character lowercase SHA-256 digest")

    def as_flight_software_snapshot(self) -> FlightSoftwareSnapshot:
        """Return the endpoint-compatible v1 payload without host metadata."""

        return FlightSoftwareSnapshot(
            stack_id=self.stack_id,
            stack_version=self.stack_version,
            boot_id=self.boot_id,
            invocation_id=self.invocation_id,
            active_load_id=self.active_load_id,
            active_load_revision=self.active_load_revision,
            state_bytes=self.state_bytes,
            state_hash_sha256=self.state_hash_sha256,
        )


FlightSoftwareCheckpoint = FlightSoftwareSnapshot | NativeFlightSoftwareSnapshot


@dataclass(frozen=True, slots=True)
class BootEvent:
    satellite_id: str
    boot_id: str
    boot_time: ClockTag

    def __post_init__(self) -> None:
        _require_nonempty("satellite_id", self.satellite_id)
        _require_nonempty("boot_id", self.boot_id)


@dataclass(frozen=True, slots=True)
class ShutdownEvent:
    satellite_id: str
    shutdown_time: ClockTag
    reason: str

    def __post_init__(self) -> None:
        _require_nonempty("satellite_id", self.satellite_id)
        _require_nonempty("reason", self.reason)


@runtime_checkable
class SatelliteFlightSoftware(Protocol):
    @property
    def identity(self) -> StackIdentity: ...

    def boot(self, event: BootEvent) -> None: ...

    def step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput: ...

    def snapshot(self) -> FlightSoftwareCheckpoint: ...

    def restore(self, snapshot: FlightSoftwareCheckpoint) -> None: ...

    def shutdown(self, event: ShutdownEvent) -> None: ...


BOUNDARY_RECORD_TYPES = (
    ClockTag,
    PacketId,
    FrameId,
    ValidityInterval,
    Quality,
    TelemetryField,
    GyroMeasurement,
    StarTrackerMeasurement,
    SunVectorMeasurement,
    MagnetometerMeasurement,
    GnssOwnStateMeasurement,
    RelativeObservationMeasurement,
    IdealOwnStateMeasurement,
    IdealTrackedObjectStateMeasurement,
    VehicleResourceMeasurement,
    ActuatorTelemetryPayload,
    MeasurementEvent,
    ControlAxisSample,
    PilotInputPayload,
    GroundCommandPayload,
    ModeledFaultIndicationPayload,
    ReactionWheelTorqueCommand,
    MagnetorquerDipoleCommand,
    CmgGimbalRateCommand,
    ThrusterPulseCommand,
    ThrusterOnOffCommand,
    ContinuousEngineCommand,
    AerodynamicEffectorPositionCommand,
    IdealWrenchCommand,
    ActuatorCommand,
    ActuatorCommandReceipt,
    InputEvent,
    FlightSoftwareInputBatch,
    DiagnosticTelemetry,
    TaskReleaseRequest,
    TaskRelease,
    StackIdentity,
    FlightSoftwareOutput,
    FlightSoftwareSnapshot,
    NativeFlightSoftwareSnapshot,
    BootEvent,
    ShutdownEvent,
)
