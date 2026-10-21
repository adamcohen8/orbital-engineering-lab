"""OEL-owned boundary adapter between physical truth and satellite FSW.

This module is deliberately the only v2 runtime owner allowed to observe both
simulator truth and the truth-free ``SatelliteFlightSoftware`` protocol.  It
turns physical sensor samples into typed packets, validates device commands,
advances physical hardware, and exposes only realized forces/torques to the
dynamics owner.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field, replace
from hashlib import sha256
from time import perf_counter_ns
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from sim.actuators.command_bus import ActuatorCommandBus, ActuatorDeviceDefinition, ExpiryBehavior
from sim.actuators.physical import (
    ActuatorRealization,
    AerodynamicEffectorHardware,
    CmgHardware,
    ContinuousEngineHardware,
    IdealWrenchHardware,
    MagnetorquerHardware,
    RcsThrusterHardware,
    ReactionWheelHardware,
)
from sim.core.models import StateTruth
from sim.dynamics.orbit.epoch import sun_position_eci_km_enhanced
from sim.flight_software.contracts import (
    CONTRACT_VERSION,
    OUTPUT_SCHEMA,
    ActuatorCommandReceipt,
    ActuatorTelemetryPayload,
    AerodynamicEffectorPositionCommand,
    BootEvent,
    ClockScale,
    ClockTag,
    CmgGimbalRateCommand,
    ContinuousEngineCommand,
    DataValidity,
    FlightSoftwareInputBatch,
    FlightSoftwareOutput,
    FlightSoftwareSnapshot,
    FrameId,
    GnssOwnStateMeasurement,
    GyroMeasurement,
    IdealOwnStateMeasurement,
    IdealTrackedObjectStateMeasurement,
    IdealWrenchCommand,
    InputEvent,
    InputKind,
    MagnetometerMeasurement,
    MagnetorquerDipoleCommand,
    MeasurementEvent,
    PacketId,
    Quality,
    ReactionWheelTorqueCommand,
    RelativeObservationMeasurement,
    SatelliteFlightSoftware,
    ShutdownEvent,
    StarTrackerMeasurement,
    SunVectorMeasurement,
    TelemetryField,
    ThrusterOnOffCommand,
    ThrusterPulseCommand,
)
from sim.flight_software.delivery import InputDeliveryQueue
from sim.flight_software.loads import OnboardMissionConfigurationLoad
from sim.flight_software.schemas import (
    _to_primitive_trusted,
    assert_truth_free,
    from_primitive,
    to_primitive,
)
from sim.knowledge.object_tracking import KnowledgeConditionConfig, ObjectDetectionGate
from sim.utils.frames import eci_relative_to_ric_rect
from sim.utils.quaternion import (
    normalize_quaternion,
    quaternion_delta_from_body_rate,
    quaternion_multiply,
    quaternion_to_dcm_bn,
)


@dataclass(frozen=True, slots=True)
class SatellitePhysicalCommand:
    force_eci_n: tuple[float, float, float]
    force_body_n: tuple[float, float, float]
    torque_body_n_m: tuple[float, float, float]
    mass_flow_kg_s: float
    realizations: tuple[ActuatorRealization, ...]
    device_positions: tuple[tuple[str, float], ...] = ()


def _aggregate_realizations(
    records: list[tuple[object, ActuatorRealization]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    force_eci = np.zeros(3)
    force_body = np.zeros(3)
    torque_body = np.zeros(3)
    mass_flow_kg_s = 0.0
    for hardware, realization in records:
        if isinstance(hardware, (IdealWrenchHardware, ContinuousEngineHardware, RcsThrusterHardware)):
            force_body += np.asarray(realization.realized_force_n, dtype=float)
        else:
            force_eci += np.asarray(realization.realized_force_n, dtype=float)
        torque_body += np.asarray(realization.realized_torque_n_m, dtype=float)
        mass_flow_kg_s += float(realization.mass_flow_kg_s)
    return force_eci, force_body, torque_body, mass_flow_kg_s


def _scale_translational_realizations(
    records: list[tuple[object, ActuatorRealization]],
    scale: float,
    *,
    propellant_only: bool = False,
) -> list[tuple[object, ActuatorRealization]]:
    """Limit translational realization without corrupting attitude devices."""

    result: list[tuple[object, ActuatorRealization]] = []
    for hardware, realization in records:
        applies = (not propellant_only) or float(realization.mass_flow_kg_s) > 0.0
        has_force = any(abs(float(value)) > 0.0 for value in realization.realized_force_n)
        if applies and has_force:
            realization = replace(
                realization,
                realized_force_n=tuple(float(value) * scale for value in realization.realized_force_n),
                realized_torque_n_m=(
                    tuple(float(value) * scale for value in realization.realized_torque_n_m)
                    if isinstance(hardware, RcsThrusterHardware) else realization.realized_torque_n_m
                ),
                mass_flow_kg_s=float(realization.mass_flow_kg_s) * scale,
                saturated=True,
            )
        result.append((hardware, realization))
    return result


def _synchronize_stateful_realizations(records: list[tuple[object, ActuatorRealization]]) -> None:
    """Keep ideal-device response state aligned with the force sent to dynamics."""

    for hardware, realization in records:
        if isinstance(hardware, IdealWrenchHardware):
            hardware.realized_force_n = realization.realized_force_n


@dataclass(slots=True)
class FlightSoftwareRuntimeEvidence:
    invocations: list[dict[str, Any]] = field(default_factory=list)
    input_events: list[InputEvent] = field(default_factory=list)
    outputs: list[FlightSoftwareOutput] = field(default_factory=list)
    receipts: list[ActuatorCommandReceipt] = field(default_factory=list)
    realizations: list[ActuatorRealization] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    sensor_access: list[dict[str, Any]] = field(default_factory=list)


InputPublisher = Callable[[ClockTag], Iterable[InputEvent]]

_TASK_RELEASING_INPUT_KINDS = frozenset(
    {
        InputKind.MEASUREMENT,
        InputKind.MISSION_LOAD,
        InputKind.STACK_LOAD,
        InputKind.GROUND_COMMAND,
        InputKind.PILOT_INPUT,
        InputKind.CROSSLINK,
        InputKind.CLOCK_EVENT,
        InputKind.MODELED_FAULT_INDICATION,
    }
)


class SatelliteFlightSoftwareRuntime:
    """Deterministic physical wrapper for one complete satellite stack."""

    def __init__(
        self,
        *,
        satellite_id: str,
        stack: SatelliteFlightSoftware,
        devices: tuple[ActuatorDeviceDefinition, ...],
        hardware: Mapping[
            str,
            IdealWrenchHardware
            | ContinuousEngineHardware
            | RcsThrusterHardware
            | AerodynamicEffectorHardware
            | ReactionWheelHardware
            | MagnetorquerHardware
            | CmgHardware,
        ],
        inertial_frame: FrameId,
        body_frame: FrameId,
        task_period_ns: int,
        sensor_period_ns: int | None = None,
        tick_period_ns: int = 1_000_000,
        boot_id: str = "boot-0",
        profile_id: str | None = None,
        profile_params: Mapping[str, object] | None = None,
        reference_object_id: str | None = None,
        initial_mission_load: OnboardMissionConfigurationLoad | None = None,
        ideal_sun_sensor: bool = False,
        ideal_magnetic_field_body_t: tuple[float, float, float] | None = None,
        initial_jd_utc: float | None = None,
        dry_mass_kg: float = 0.0,
        ideal_navigation: bool = True,
        sensor_error: Mapping[str, object] | None = None,
        sensor_seed: int = 0,
        sensor_conditions: KnowledgeConditionConfig | None = None,
        initial_checkpoint: Mapping[str, object] | None = None,
    ) -> None:
        if task_period_ns <= 0 or tick_period_ns <= 0 or task_period_ns % tick_period_ns:
            raise ValueError("task period must be a positive whole number of clock ticks")
        if sensor_period_ns is not None and (sensor_period_ns <= 0 or sensor_period_ns % tick_period_ns):
            raise ValueError("sensor period must be a positive whole number of clock ticks")
        if set(hardware) != {device.actuator_id for device in devices}:
            raise ValueError("every declared actuator device must have exactly one physical hardware model")
        self.satellite_id = satellite_id
        self.stack = stack
        # OEL-owned reference stacks already expose a private execution hook
        # for a boundary that this runtime has validated.  Calling the base
        # implementation directly prevents a subclass from weakening the
        # firewall; third-party stacks continue through their public step().
        from sim.flight_software.reference_stacks import ReferenceStackBase

        self._builtin_reference_stack_base = ReferenceStackBase if isinstance(stack, ReferenceStackBase) else None
        self._native_evidence_encoder = None
        self._validate_boundary = assert_truth_free
        self._quaternion_to_dcm_bn = quaternion_to_dcm_bn
        self._relative_ric_state = eci_relative_to_ric_rect
        if getattr(stack, "numeric_backend", "rust") == "rust":
            from sim.flight_software.rust_game_frames import quaternion_to_dcm_bn as native_dcm
            from sim.flight_software.rust_game_frames import relative_ric_state
            from sim.flight_software.rust_game_packets import boundary_validator, trusted_evidence_encoder

            self._native_evidence_encoder = getattr(stack, "_native_packet_encoder", None) or trusted_evidence_encoder()
            self._validate_boundary = (getattr(stack, "_native_packet_validator", None) or boundary_validator()).check
            if not getattr(stack, "_native_actuator_pipeline", False):
                self._quaternion_to_dcm_bn = native_dcm
                self._relative_ric_state = relative_ric_state
        self.inertial_frame = inertial_frame
        self.body_frame = body_frame
        self.task_period_ns = int(task_period_ns)
        self.sensor_period_ns = int(sensor_period_ns or task_period_ns)
        self.tick_period_ns = int(tick_period_ns)
        self.boot_id = boot_id
        self.profile_id = None if profile_id in (None, "") else str(profile_id)
        self.profile_params = dict(profile_params or {})
        self.reference_object_id = reference_object_id
        self.ideal_sun_sensor = bool(ideal_sun_sensor)
        self.ideal_magnetic_field_body_t = ideal_magnetic_field_body_t
        self.initial_jd_utc = None if initial_jd_utc is None else float(initial_jd_utc)
        if not np.isfinite(float(dry_mass_kg)) or float(dry_mass_kg) < 0.0:
            raise ValueError("dry_mass_kg must be finite and nonnegative")
        self.dry_mass_kg = float(dry_mass_kg)
        self.ideal_navigation = bool(ideal_navigation)
        self.sensor_error = dict(sensor_error or {})
        self._sensor_rng = np.random.default_rng(int(sensor_seed))
        # Independent noise/access streams keep existing unbounded measurements
        # stable and make packet loss replayable without consuming noise draws.
        self._target_access = (
            None
            if sensor_conditions is None
            else ObjectDetectionGate(sensor_conditions, np.random.default_rng(int(sensor_seed)))
        )
        self.command_bus = ActuatorCommandBus(devices)
        self.hardware = dict(hardware)
        if getattr(stack, "_native_actuator_pipeline", False):
            from sim.actuators.rust_physical import enable_hardware
            self.command_bus._native_evidence_encoder = self._native_evidence_encoder
            self.command_bus._native_boundary_validator = self._validate_boundary.__self__
            for model in self.hardware.values():
                enable_hardware(model)
        self.inputs = InputDeliveryQueue()
        self.evidence = FlightSoftwareRuntimeEvidence()
        checkpoint_snapshot, checkpoint_state = self._decode_initial_checkpoint(initial_checkpoint)
        self._clock_offset_ns = int(checkpoint_state.get("checkpoint_time_ns", 0))
        self._invocation_id = 0
        self._sensor_sequence = 0
        self._receipt_sequence = 0
        self._telemetry_sequence = 0
        self._last_invocation_ns: int | None = None
        self._next_task_ns = 0
        self._next_sensor_ns = 0
        self._missed_task_releases = 0
        self._missed_sensor_releases = 0
        self._requested_release_ns: set[int] = set()
        self._publisher_poll_requested_ns: int | None = None
        self._input_publishers: list[InputPublisher] = []
        self._runtime_owned_input_ids: set[int] = set()
        self.aerodynamic_config: dict[str, float] | None = None
        self.max_delta_v_m_s: float | None = None
        self.used_delta_v_m_s = 0.0
        self._shutdown = False
        if checkpoint_snapshot is not None:
            boot_id = checkpoint_snapshot.boot_id
            self.boot_id = boot_id
        self.stack.boot(BootEvent(satellite_id, boot_id, self.clock_tag(0)))
        if checkpoint_snapshot is not None:
            self.stack.restore(checkpoint_snapshot)
            self._restore_runtime_state(checkpoint_state)
        elif initial_mission_load is not None:
            at = self.clock_tag(0)
            self.enqueue(
                InputEvent(
                    PacketId(f"{satellite_id}/scenario_mission_load", boot_id, 0),
                    InputKind.MISSION_LOAD,
                    at,
                    at,
                    Quality(),
                    initial_mission_load,
                )
            )
        self._record_snapshot(invocation_id=self._invocation_id, run_time_ns=0)

    def clock_tag(self, time_ns: int) -> ClockTag:
        if time_ns < 0 or time_ns % self.tick_period_ns:
            raise ValueError("runtime time must align with the onboard clock quantum")
        return ClockTag(
            f"{self.satellite_id}/onboard",
            (self._clock_offset_ns + time_ns) // self.tick_period_ns,
            self.tick_period_ns,
            ClockScale.ONBOARD,
        )

    def add_input_publisher(self, publisher: InputPublisher) -> None:
        if not callable(publisher):
            raise TypeError("input publisher must be callable")
        self._input_publishers.append(publisher)
        # Interactive publishers are commonly attached after the engine has
        # emitted its time-zero invocation. Request one event-driven poll at
        # that same boundary so the first physical interval sees the input.
        if self._last_invocation_ns is not None:
            self._publisher_poll_requested_ns = self._last_invocation_ns

    def request_input_publisher_poll(self, *, time_ns: int) -> None:
        """Request one publisher poll at the current physical boundary."""

        run_time_ns = int(time_ns)
        if run_time_ns < 0 or run_time_ns % self.tick_period_ns:
            raise ValueError("publisher poll time must align with the onboard clock quantum")
        if not self._input_publishers:
            raise RuntimeError("publisher poll requires at least one attached input publisher")
        onboard_time_ns = self._clock_offset_ns + run_time_ns
        if self._last_invocation_ns is not None and onboard_time_ns < self._last_invocation_ns:
            raise ValueError("publisher poll cannot be requested before the latest invocation")
        pending = self._publisher_poll_requested_ns
        self._publisher_poll_requested_ns = onboard_time_ns if pending is None else min(pending, onboard_time_ns)

    def enqueue(self, event: InputEvent) -> None:
        self.inputs.enqueue(event)

    def _enqueue_runtime_owned(self, event: InputEvent) -> None:
        """Queue one immutable packet constructed by this boundary adapter."""

        self._runtime_owned_input_ids.add(id(event))
        self.inputs.enqueue(event)

    def enqueue_all(self, events: Iterable[InputEvent]) -> None:
        for event in events:
            self.enqueue(event)

    def prepare_interval(
        self,
        truth: StateTruth,
        *,
        start_time_ns: int,
        world_truth: Mapping[str, StateTruth] | None = None,
    ) -> None:
        """Release any task or delivered input event due at an interval boundary."""

        onboard_start_ns = self._clock_offset_ns + start_time_ns
        publisher_due = (
            self._publisher_poll_requested_ns is not None and self._publisher_poll_requested_ns <= onboard_start_ns
        )
        input_due = self.inputs.next_delivery_time_ns_for(_TASK_RELEASING_INPUT_KINDS)
        if (
            self._last_invocation_ns == onboard_start_ns
            and not publisher_due
            and not (input_due is not None and input_due <= onboard_start_ns)
        ):
            return
        task_due = self._next_task_ns <= onboard_start_ns
        sensor_due = self._next_sensor_ns <= onboard_start_ns
        missed_task_releases = (onboard_start_ns - self._next_task_ns) // self.task_period_ns if task_due else 0
        missed_sensor_releases = (onboard_start_ns - self._next_sensor_ns) // self.sensor_period_ns if sensor_due else 0
        requested_due = any(value <= onboard_start_ns for value in self._requested_release_ns)
        if (
            task_due
            or sensor_due
            or requested_due
            or publisher_due
            or (input_due is not None and input_due <= onboard_start_ns)
        ):
            self._missed_task_releases += int(missed_task_releases)
            self._missed_sensor_releases += int(missed_sensor_releases)
            release_reasons = tuple(
                reason
                for reason, active in (
                    ("scheduled_task", task_due),
                    ("sensor_sample", sensor_due),
                    ("requested_release", requested_due),
                    ("publisher_poll", publisher_due),
                    ("delivered_input", input_due is not None and input_due <= onboard_start_ns),
                )
                if active
            )
            self._invoke(
                truth,
                start_time_ns,
                world_truth=world_truth,
                sample_sensors=sensor_due,
                release_reasons=release_reasons,
            )
            self._requested_release_ns = {value for value in self._requested_release_ns if value > onboard_start_ns}
            if publisher_due:
                self._publisher_poll_requested_ns = None
            if task_due:
                self._next_task_ns += (missed_task_releases + 1) * self.task_period_ns
            if sensor_due:
                self._next_sensor_ns += (missed_sensor_releases + 1) * self.sensor_period_ns

    def passive_prediction_boundaries_ns(self, *, start_time_ns: int, end_time_ns: int) -> list[int] | None:
        """Read-only calendar for orbit prediction with the built-in non-actuating stack.

        This does not execute or export FSW telemetry. Only the exact built-in
        passive stack with no hardware, queued inputs, publishers, requested
        releases or checkpoint clock offset has a state-independent calendar.
        Every other runtime must execute the ordinary scheduler. The passive
        stack's optional navigation reporting never commands physical devices.
        """
        from sim.flight_software.reference_stacks import PassiveFlightSoftwareStack

        if (type(self) is not SatelliteFlightSoftwareRuntime or type(self.stack) is not PassiveFlightSoftwareStack
                or self.hardware or self._clock_offset_ns or self._shutdown
                or self.inputs.next_delivery_time_ns is not None or self._input_publishers
                or self._requested_release_ns or self._publisher_poll_requested_ns is not None
                or "_step" in vars(self.stack) or "next_hard_boundary_ns" in vars(self)):
            return None
        if start_time_ns < 0 or end_time_ns < start_time_ns:
            raise ValueError("prediction calendar requires ordered nonnegative times")
        boundaries = {int(end_time_ns)}
        for next_release, period in ((self._next_task_ns, self.task_period_ns),
                                     (self._next_sensor_ns, self.sensor_period_ns)):
            first = next_release + max(0, (start_time_ns - next_release) // period + 1) * period
            boundaries.update(range(first, end_time_ns + 1, period))
        return sorted(boundaries)

    def next_hard_boundary_ns(self, *, after_time_ns: int, before_time_ns: int) -> int:
        """Return the next task, delivery, command start, or expiry boundary."""

        onboard_after_ns = self._clock_offset_ns + after_time_ns
        onboard_before_ns = self._clock_offset_ns + before_time_ns
        candidates = [onboard_before_ns]
        candidates.append(self._next_task_ns)
        candidates.append(self._next_sensor_ns)
        candidates.extend(self._requested_release_ns)
        if self._publisher_poll_requested_ns is not None:
            candidates.append(self._publisher_poll_requested_ns)
        delivery = self.inputs.next_delivery_time_ns_for(_TASK_RELEASING_INPUT_KINDS)
        if delivery is not None:
            candidates.append(delivery)
        candidates.extend(self.command_bus.hard_event_times_ns(after_time_ns=onboard_after_ns))
        future = [value for value in candidates if onboard_after_ns < value <= onboard_before_ns]
        return min(future, default=onboard_before_ns) - self._clock_offset_ns

    def command_interval(
        self,
        truth: StateTruth,
        *,
        start_time_ns: int,
        end_time_ns: int,
        world_truth: Mapping[str, StateTruth] | None = None,
    ) -> SatellitePhysicalCommand:
        if end_time_ns < start_time_ns:
            raise ValueError("physical interval must be nonnegative")
        self.prepare_interval(truth, start_time_ns=start_time_ns, world_truth=world_truth)
        at = self.clock_tag(start_time_ns)
        realization_records: list[tuple[object, ActuatorRealization]] = []
        positions: list[tuple[str, float]] = []
        for actuator_id, hardware in self.hardware.items():
            demand = self.command_bus.demand(satellite_id=self.satellite_id, actuator_id=actuator_id, at=at)
            if isinstance(hardware, ContinuousEngineHardware):
                realization = hardware.advance(
                    demand,
                    start_time_ns=start_time_ns,
                    end_time_ns=end_time_ns,
                    attitude_quat_bn=tuple(float(value) for value in truth.attitude_quat_bn),
                )
            else:
                realization = hardware.advance(demand, start_time_ns=start_time_ns, end_time_ns=end_time_ns)
            realization_records.append((hardware, realization))
            if isinstance(hardware, AerodynamicEffectorHardware):
                positions.append((actuator_id, float(hardware.position)))

        force, force_body, torque, mass_flow_kg_s = _aggregate_realizations(realization_records)
        if self.max_delta_v_m_s is not None and end_time_ns > start_time_ns:
            dt_s = (end_time_ns - start_time_ns) / 1.0e9
            mass_kg = max(float(truth.mass_kg), 1.0e-12)
            dcm_bn = self._quaternion_to_dcm_bn(np.asarray(truth.attitude_quat_bn, dtype=float))
            requested_delta_v = float(np.linalg.norm(force + dcm_bn.T @ force_body)) / mass_kg * dt_s
            remaining = max(float(self.max_delta_v_m_s) - self.used_delta_v_m_s, 0.0)
            scale = 1.0 if requested_delta_v <= remaining or requested_delta_v <= 0.0 else remaining / requested_delta_v
            if scale < 1.0:
                realization_records = _scale_translational_realizations(realization_records, scale)
                force, force_body, torque, mass_flow_kg_s = _aggregate_realizations(realization_records)
        if end_time_ns > start_time_ns and mass_flow_kg_s > 0.0:
            dt_s = (end_time_ns - start_time_ns) / 1.0e9
            available_mass_kg = max(float(truth.mass_kg) - self.dry_mass_kg, 0.0)
            requested_mass_kg = mass_flow_kg_s * dt_s
            fuel_scale = min(1.0, available_mass_kg / requested_mass_kg) if requested_mass_kg > 0.0 else 1.0
            if fuel_scale < 1.0:
                realization_records = _scale_translational_realizations(
                    realization_records,
                    fuel_scale,
                    propellant_only=True,
                )
                force, force_body, torque, mass_flow_kg_s = _aggregate_realizations(realization_records)
        if self.max_delta_v_m_s is not None and end_time_ns > start_time_ns:
            dt_s = (end_time_ns - start_time_ns) / 1.0e9
            mass_kg = max(float(truth.mass_kg), 1.0e-12)
            dcm_bn = self._quaternion_to_dcm_bn(np.asarray(truth.attitude_quat_bn, dtype=float))
            self.used_delta_v_m_s += float(np.linalg.norm(force + dcm_bn.T @ force_body)) / mass_kg * dt_s
        _synchronize_stateful_realizations(realization_records)
        realizations = [item for _, item in realization_records]
        telemetry_time = self.clock_tag(end_time_ns)
        for realization in realizations:
            self._queue_actuator_telemetry(realization, telemetry_time)
        self.evidence.realizations.extend(realizations)
        return SatellitePhysicalCommand(
            tuple(float(value) for value in force),
            tuple(float(value) for value in force_body),
            tuple(float(value) for value in torque),
            float(mass_flow_kg_s),
            tuple(realizations),
            tuple(sorted(positions)),
        )

    def physics_environment(
        self,
        truth: StateTruth,
        command: SatellitePhysicalCommand,
    ) -> dict[str, object]:
        """Resolve physical aero inputs from realized devices, never FSW flags."""

        config = self.aerodynamic_config
        if config is None:
            return {}
        positions = dict(command.device_positions)
        deployment = float(np.clip(positions.get("deployment", 0.5), 0.0, 1.0))
        bc_min = config["ballistic_coefficient_min_kg_m2"]
        bc_max = config["ballistic_coefficient_max_kg_m2"]
        ballistic_coefficient = bc_min + deployment * (bc_max - bc_min)
        drag_coefficient = config["drag_coefficient"]
        drag_area = float(truth.mass_kg) / max(drag_coefficient * ballistic_coefficient, 1.0e-12)
        velocity = np.asarray(truth.velocity_eci_km_s, dtype=float)
        velocity_hat = velocity / max(float(np.linalg.norm(velocity)), 1.0e-15)
        radial = np.asarray(truth.position_eci_km, dtype=float)
        radial = radial / max(float(np.linalg.norm(radial)), 1.0e-15)
        lift_reference = radial - float(radial @ velocity_hat) * velocity_hat
        lift_reference /= max(float(np.linalg.norm(lift_reference)), 1.0e-15)
        bank = float(positions.get("bank", 0.0))
        lift_direction = (
            lift_reference * np.cos(bank)
            + np.cross(velocity_hat, lift_reference) * np.sin(bank)
            + velocity_hat * float(velocity_hat @ lift_reference) * (1.0 - np.cos(bank))
        )
        return {
            "physical_aerodynamics": True,
            "drag_area_m2": drag_area,
            "drag_coefficient": drag_coefficient,
            "lift_area_m2": config["lift_area_m2"],
            "lift_coefficient": config["lift_coefficient"],
            "lift_direction_eci": lift_direction,
            "aerodynamic_ballistic_coefficient_kg_m2": ballistic_coefficient,
            "aerodynamic_device_positions": positions,
        }

    def _invoke(
        self,
        truth: StateTruth,
        time_ns: int,
        *,
        world_truth: Mapping[str, StateTruth] | None,
        sample_sensors: bool,
        release_reasons: tuple[str, ...],
    ) -> None:
        now = self.clock_tag(time_ns)
        if sample_sensors:
            self._enqueue_runtime_owned(
                self._ideal_own_state_event(truth, now)
                if self.ideal_navigation
                else self._measured_own_state_event(truth, now)
            )
            if self.ideal_sun_sensor:
                self._enqueue_runtime_owned(self._ideal_sun_event(truth, now))
            if self.ideal_magnetic_field_body_t is not None:
                self._enqueue_runtime_owned(self._ideal_magnetometer_event(now))
        reference_id = self.reference_object_id
        if (
            sample_sensors
            and reference_id is not None
            and world_truth is not None
            and reference_id in world_truth
            and self._target_detected(truth, world_truth[reference_id], now)
        ):
            if self.ideal_navigation:
                self._enqueue_runtime_owned(
                    self._ideal_tracked_state_event(reference_id, world_truth[reference_id], now)
                )
            else:
                self._enqueue_runtime_owned(
                    self._relative_observation_event(reference_id, truth, world_truth[reference_id], now)
                )
        for publisher in self._input_publishers:
            self.enqueue_all(publisher(now))
        onboard_time_ns = self._clock_offset_ns + time_ns
        events = self.inputs.deliver_due(onboard_time_ns)
        self.evidence.input_events.extend(events)
        self._invocation_id += 1
        batch = FlightSoftwareInputBatch(self.satellite_id, self._invocation_id, now, events)
        # Runtime-owned sensor/receipt records are immutable boundary types
        # assembled here from explicit observable values.  Anything entering
        # through a publisher, public enqueue, or restored queue is checked
        # recursively immediately before the stack sees it.
        untrusted = tuple(event for event in events if id(event) not in self._runtime_owned_input_ids)
        validator = getattr(self._validate_boundary, "__self__", None)
        check_many = getattr(validator, "check_many", None)
        if check_many is not None:
            check_many(untrusted)
        else:
            for event in untrusted:
                self._validate_boundary(event)
        self._runtime_owned_input_ids.difference_update(id(event) for event in events)
        execution_started_ns = perf_counter_ns()
        try:
            if self._builtin_reference_stack_base is None:
                output = self.stack.step(batch)
            else:
                output = self._builtin_reference_stack_base._step_after_boundary_validation(
                    self.stack,
                    batch,
                )
        except Exception as exc:
            # Preserve the adapter's packet-digest trail across the failed
            # run boundary.  The CLI writes this partial evidence separately;
            # a failed exchange cannot reach completed payload assembly.
            if type(exc).__name__ == "CfsBridgeFailure":
                transport_evidence = getattr(self.stack, "transport_evidence", None)
                if callable(transport_evidence):
                    exc.bridge_failure_evidence = {
                        "satellite_id": self.satellite_id,
                        "bridge_transport": to_primitive(transport_evidence()),
                    }
            raise
        host_execution_duration_ns = max(0, perf_counter_ns() - execution_started_ns)
        # Keep one adapter-owned egress check for every stack.  Built-ins avoid
        # only the duplicate traversal formerly repeated inside the stack base.
        self._validate_boundary(output)
        if output.schema != OUTPUT_SCHEMA:
            raise ValueError(
                f"flight-software output schema {output.schema!r} is not supported"
            )
        if output.contract_version != CONTRACT_VERSION:
            raise ValueError(
                f"flight-software output contract_version {output.contract_version!r} is not supported"
            )
        if output.satellite_id != self.satellite_id:
            raise ValueError(
                f"flight-software output satellite_id {output.satellite_id!r} does not match "
                f"runtime satellite {self.satellite_id!r}"
            )
        if output.invocation_id != self._invocation_id:
            raise ValueError(
                f"flight-software output invocation_id {output.invocation_id} does not match "
                f"runtime invocation {self._invocation_id}"
            )
        for request in output.requested_next_invocations:
            release_ns = self._validate_requested_release(request.release_at, now)
            if release_ns <= onboard_time_ns:
                raise ValueError("requested next invocation must be later than the requesting invocation")
            self._requested_release_ns.add(release_ns)
        receipts = self.command_bus._publish_all_boundary_validated(
            output.commands,
            received_at=now,
        )
        identity = self.stack.identity
        if callable(identity):
            identity = identity()
        if self._native_evidence_encoder is None:
            input_packet_ids = [_to_primitive_trusted(event.packet_id) for event in events]
            command_ids = [_to_primitive_trusted(command.command_id) for command in output.commands]
            requested_invocations = [_to_primitive_trusted(request) for request in output.requested_next_invocations]
        else:
            # One crossing for the three identifier groups, after the same
            # ingress/egress checks and command contract validation as Python.
            input_packet_ids, command_ids, requested_invocations = self._native_evidence_encoder.convert_many((
                tuple(event.packet_id for event in events),
                tuple(command.command_id for command in output.commands),
                output.requested_next_invocations,
            ))
        self.evidence.invocations.append(
            {
                "satellite_id": self.satellite_id,
                "invocation_id": self._invocation_id,
                "invocation_time_ns": time_ns,
                "stack_id": identity.stack_id,
                "stack_version": identity.stack_version,
                "profile_id": self.profile_id,
                "profile_params": dict(self.profile_params),
                "input_packet_ids": input_packet_ids,
                "command_ids": command_ids,
                "telemetry_count": len(output.telemetry),
                "missed_task_releases": self._missed_task_releases,
                "missed_sensor_releases": self._missed_sensor_releases,
                "requested_next_invocations": requested_invocations,
                "task_releases": [
                    {
                        "task_id": "stack.step",
                        "release_time_ns": time_ns,
                        # Host profiling is observational evidence, not a
                        # deterministic simulated execution-time model.
                        "modeled_execution_duration_ns": 0,
                        "host_execution_duration_ns": host_execution_duration_ns,
                        "execution_budget_ns": self.task_period_ns,
                        "deadline_missed": False,
                        "release_reasons": list(release_reasons),
                    }
                ],
            }
        )
        self.evidence.outputs.append(output)
        self.evidence.receipts.extend(receipts)
        for receipt in receipts:
            self._enqueue_runtime_owned(self._receipt_event(receipt, now))
        self._last_invocation_ns = onboard_time_ns

    def _target_detected(self, observer: StateTruth, target: StateTruth, now: ClockTag) -> bool:
        if self._target_access is None:
            return True
        detected = self._target_access.detect(observer, target, now.ticks * now.tick_period_ns / 1e9)
        self.evidence.sensor_access.append(
            {
                "target_id": self.reference_object_id,
                "sample_time": _to_primitive_trusted(now),
                "detected": detected,
                "status": self._target_access.last_detection_status,
            }
        )
        return detected

    def _validate_requested_release(self, release_at: ClockTag, now: ClockTag) -> int:
        if (
            release_at.clock_id,
            release_at.tick_period_ns,
            release_at.scale,
            release_at.reset_counter,
        ) != (now.clock_id, now.tick_period_ns, now.scale, now.reset_counter):
            raise ValueError("requested next invocation must use the runtime onboard clock domain")
        return int(release_at.ticks * release_at.tick_period_ns)

    def realizations_since(self, index: int) -> tuple[tuple[ActuatorRealization, ...], int]:
        """Return each physical realization once without rescanning prior history."""

        start = max(int(index), 0)
        return tuple(self.evidence.realizations[start:]), len(self.evidence.realizations)

    def shutdown(self, *, time_ns: int, reason: str = "simulation_complete") -> None:
        if self._shutdown:
            return
        self._record_snapshot(invocation_id=self._invocation_id, run_time_ns=time_ns)
        self.stack.shutdown(ShutdownEvent(self.satellite_id, self.clock_tag(time_ns), reason))
        self._shutdown = True

    def checkpoint(self) -> object | None:
        identity = self.stack.identity
        if callable(identity):
            identity = identity()
        return self.stack.snapshot() if identity.checkpointable else None

    def _record_snapshot(self, *, invocation_id: int, run_time_ns: int) -> None:
        identity = self.stack.identity
        if callable(identity):
            identity = identity()
        if not identity.checkpointable:
            return
        snapshot = self.stack.snapshot()
        if self._native_evidence_encoder is not None:
            self._validate_boundary(snapshot)
        runtime_state_bytes = json.dumps(
            self._runtime_state(run_time_ns=run_time_ns),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.evidence.snapshots.append(
            {
                "invocation_id": invocation_id,
                "stack_id": snapshot.stack_id,
                "stack_version": snapshot.stack_version,
                "profile_id": self.profile_id,
                "active_load_id": snapshot.active_load_id,
                "active_load_revision": snapshot.active_load_revision,
                "state_hash_sha256": snapshot.state_hash_sha256,
                "boot_id": snapshot.boot_id,
                "run_time_ns": int(run_time_ns),
                "checkpoint_time_ns": self._clock_offset_ns + int(run_time_ns),
                "implementation_hash": identity.implementation_hash,
                "fsw_snapshot": (to_primitive(snapshot) if self._native_evidence_encoder is None
                                 else self._native_evidence_encoder.convert(snapshot)),
                "runtime_state_bytes_base64": base64.b64encode(runtime_state_bytes).decode("ascii"),
                "runtime_state_hash_sha256": sha256(runtime_state_bytes).hexdigest(),
                "checkpoint_schema": "oel.satellite_runtime_checkpoint.v1",
            }
        )

    def _runtime_state(self, *, run_time_ns: int) -> dict[str, object]:
        identity = self.stack.identity
        if callable(identity):
            identity = identity()
        return {
            "schema": "oel.satellite_runtime_state.v1",
            "checkpoint_time_ns": self._clock_offset_ns + int(run_time_ns),
            "profile_id": self.profile_id,
            "profile_params": to_primitive(self.profile_params),
            "stack_identity": to_primitive(identity),
            "invocation_id": self._invocation_id,
            "sensor_sequence": self._sensor_sequence,
            "receipt_sequence": self._receipt_sequence,
            "telemetry_sequence": self._telemetry_sequence,
            "last_invocation_ns": self._last_invocation_ns,
            "next_task_ns": self._next_task_ns,
            "next_sensor_ns": self._next_sensor_ns,
            "missed_task_releases": self._missed_task_releases,
            "missed_sensor_releases": self._missed_sensor_releases,
            "requested_release_ns": sorted(self._requested_release_ns),
            "publisher_poll_requested_ns": self._publisher_poll_requested_ns,
            "max_delta_v_m_s": self.max_delta_v_m_s,
            "used_delta_v_m_s": self.used_delta_v_m_s,
            "sensor_rng_state": self._sensor_rng.bit_generator.state,
            "target_access_state": (
                None
                if self._target_access is None
                else {
                    "rng_state": self._target_access.rng.bit_generator.state,
                    "last_update_t_s": (
                        float(self._target_access.access._last_update_t_s)
                        if np.isfinite(self._target_access.access._last_update_t_s)
                        else None
                    ),
                }
            ),
            "external_publisher_count": len(self._input_publishers),
            "command_bus": self.command_bus.snapshot_state(),
            "input_delivery": self.inputs.snapshot_state(),
            "hardware": {
                actuator_id: hardware.snapshot_state() for actuator_id, hardware in sorted(self.hardware.items())
            },
        }

    @staticmethod
    def _decode_initial_checkpoint(
        checkpoint: Mapping[str, object] | None,
    ) -> tuple[FlightSoftwareSnapshot | None, dict[str, object]]:
        if checkpoint is None:
            return None, {}
        raw = dict(checkpoint)
        if raw.get("checkpoint_schema") != "oel.satellite_runtime_checkpoint.v1":
            raise ValueError("unsupported flight-software checkpoint schema")
        snapshot_raw = raw.get("fsw_snapshot")
        if isinstance(snapshot_raw, dict):
            snapshot_raw = dict(snapshot_raw)
            snapshot_raw.setdefault("active_load_id", None)
            snapshot_raw.setdefault("active_load_revision", None)
        snapshot = from_primitive(FlightSoftwareSnapshot, snapshot_raw)
        encoded = raw.get("runtime_state_bytes_base64")
        expected_hash = str(raw.get("runtime_state_hash_sha256", "") or "")
        if not isinstance(encoded, str):
            raise ValueError("flight-software checkpoint runtime state must be base64 text")
        try:
            state_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("flight-software checkpoint runtime state contains invalid base64") from exc
        if sha256(state_bytes).hexdigest() != expected_hash:
            raise ValueError("flight-software checkpoint runtime-state hash mismatch")
        try:
            state = json.loads(state_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("flight-software checkpoint runtime state is invalid JSON") from exc
        if not isinstance(state, dict) or state.get("schema") != "oel.satellite_runtime_state.v1":
            raise ValueError("flight-software checkpoint runtime state has an unsupported schema")
        return snapshot, state

    def _restore_runtime_state(self, state: Mapping[str, object]) -> None:
        if int(state.get("checkpoint_time_ns", -1)) != self._clock_offset_ns:
            raise ValueError("runtime checkpoint clock offset is inconsistent")
        if int(state.get("invocation_id", -1)) != self.stack.snapshot().invocation_id:
            raise ValueError("runtime and flight-software checkpoint invocation IDs differ")
        if state.get("profile_id") != self.profile_id:
            raise ValueError("runtime checkpoint flight-software profile_id is incompatible")
        if state.get("profile_params") != to_primitive(self.profile_params):
            raise ValueError("runtime checkpoint flight-software profile parameters are incompatible")
        identity = self.stack.identity
        if callable(identity):
            identity = identity()
        if state.get("stack_identity") != to_primitive(identity):
            raise ValueError("runtime checkpoint flight-software implementation identity is incompatible")
        self._invocation_id = int(state["invocation_id"])
        self._sensor_sequence = int(state["sensor_sequence"])
        self._receipt_sequence = int(state["receipt_sequence"])
        self._telemetry_sequence = int(state["telemetry_sequence"])
        self._last_invocation_ns = None if state.get("last_invocation_ns") is None else int(state["last_invocation_ns"])
        self._next_task_ns = int(state["next_task_ns"])
        self._next_sensor_ns = int(state["next_sensor_ns"])
        self._missed_task_releases = int(state.get("missed_task_releases", 0))
        self._missed_sensor_releases = int(state.get("missed_sensor_releases", 0))
        self._requested_release_ns = {int(value) for value in state.get("requested_release_ns", [])}  # type: ignore[arg-type]
        self._publisher_poll_requested_ns = (
            None if state.get("publisher_poll_requested_ns") is None else int(state["publisher_poll_requested_ns"])
        )
        self.max_delta_v_m_s = None if state.get("max_delta_v_m_s") is None else float(state["max_delta_v_m_s"])
        self.used_delta_v_m_s = float(state.get("used_delta_v_m_s", 0.0))
        if int(state.get("external_publisher_count", 0)) != 0:
            raise ValueError(
                "runtime checkpoint depends on external input publishers and cannot be restored by a scenario alone"
            )
        sensor_rng_state = state.get("sensor_rng_state")
        if not isinstance(sensor_rng_state, dict):
            raise ValueError("runtime checkpoint sensor RNG state is invalid")
        self._sensor_rng.bit_generator.state = sensor_rng_state
        access_state = state.get("target_access_state")
        if access_state is not None:
            if self._target_access is None or not isinstance(access_state, dict):
                raise ValueError("runtime checkpoint target access configuration is incompatible")
            self._target_access.rng.bit_generator.state = access_state["rng_state"]
            last_update = access_state.get("last_update_t_s")
            self._target_access.access._last_update_t_s = -np.inf if last_update is None else float(last_update)
        self.command_bus.restore_state(state.get("command_bus"))
        self.inputs.restore_state(state.get("input_delivery"))
        hardware_state = state.get("hardware")
        if not isinstance(hardware_state, dict) or set(hardware_state) != set(self.hardware):
            raise ValueError("runtime checkpoint hardware identities are incompatible")
        for actuator_id, hardware in self.hardware.items():
            hardware.restore_state(hardware_state[actuator_id])

    def review_evidence(self) -> dict[str, object]:
        """Return JSON-safe typed boundary evidence for reporting owners."""

        records = [
            ("invocations", self.evidence.invocations),
            ("input_events", self.evidence.input_events),
            ("outputs", self.evidence.outputs),
            ("receipts", self.evidence.receipts),
            ("realizations", self.evidence.realizations),
            ("snapshots", self.evidence.snapshots),
        ]
        if self.evidence.sensor_access:
            records.append(("sensor_access", self.evidence.sensor_access))
        if self._native_evidence_encoder is None:
            evidence = {name: _to_primitive_trusted(value) for name, value in records}
        else:
            values = self._native_evidence_encoder.convert_many(tuple(value for _name, value in records))
            evidence = dict(zip((name for name, _value in records), values, strict=True))
        transport_evidence = getattr(self.stack, "transport_evidence", None)
        if callable(transport_evidence):
            # Bridge evidence is supplied by an external transport owner and
            # therefore keeps the complete public boundary traversal.
            evidence["bridge_transport"] = to_primitive(transport_evidence())
        return evidence

    def onboard_state_vector(self) -> np.ndarray | None:
        """Report published navigation; preserve legacy widths with NaN for gaps."""
        measured = self.onboard_measurement_vector()
        unavailable = None if measured is None else np.full(measured.shape, np.nan)
        if not self.evidence.outputs:
            return unavailable
        fields = None
        for output in reversed(self.evidence.outputs):
            for record in output.telemetry:
                if record.topic == "oel.navigation_state.v1":
                    # An invocation may publish diagnostics without a new
                    # navigation packet. Hold the latest published packet.
                    fields = {item.name: item.value for item in record.fields}
                    break
            if fields is not None:
                break
        if (
            not fields
            or fields.get("frame_id") != self.inertial_frame.name
            or fields.get("frame_registry_version") != self.inertial_frame.registry_version
        ):
            return unavailable
        position = [fields.get(f"position_{axis}_m") for axis in "xyz"]
        velocity = [fields.get(f"velocity_{axis}_m_s") for axis in "xyz"]
        values = [np.nan if value is None else float(value) / 1000 for value in (*position, *velocity)]
        quaternion = [fields.get(f"q_{axis}") for axis in "wxyz"]
        rate = [fields.get(f"omega_{axis}_rad_s") for axis in "xyz"]
        if any(value is not None for value in quaternion) or (measured is not None and measured.size >= 10):
            values.extend(np.nan if value is None else float(value) for value in quaternion)
            if any(value is not None for value in rate) or (measured is not None and measured.size >= 13):
                values.extend(np.nan if value is None else float(value) for value in rate)
        return np.asarray(values, dtype=float)

    def onboard_measurement_vector(self) -> np.ndarray | None:
        """Return the latest onboard own-state measurement for reporting only."""

        for event in reversed(self.evidence.input_events):
            measurement = event.payload
            if not isinstance(measurement, MeasurementEvent) or not isinstance(
                measurement.payload, (IdealOwnStateMeasurement, GnssOwnStateMeasurement)
            ):
                continue
            payload = measurement.payload
            if isinstance(payload, IdealOwnStateMeasurement) and (
                payload.position_m is None or payload.velocity_m_s is None
            ):
                return None
            values = [
                *(float(value) / 1.0e3 for value in payload.position_m),
                *(float(value) / 1.0e3 for value in payload.velocity_m_s),
            ]
            if isinstance(payload, IdealOwnStateMeasurement) and payload.attitude_quat_body_from_inertial is not None:
                values.extend(float(value) for value in payload.attitude_quat_body_from_inertial)
            if isinstance(payload, IdealOwnStateMeasurement) and payload.angular_rate_body_rad_s is not None:
                values.extend(float(value) for value in payload.angular_rate_body_rad_s)
            return np.asarray(values, dtype=float)
        return None

    def _ideal_own_state_event(self, truth: StateTruth, now: ClockTag) -> InputEvent:
        packet_id = PacketId(f"{self.satellite_id}/ideal_own_state", self.boot_id, self._sensor_sequence)
        self._sensor_sequence += 1
        attitude = np.asarray(truth.attitude_quat_bn, dtype=float).copy()
        attitude /= max(float(np.linalg.norm(attitude)), 1.0e-15)
        if attitude[0] < 0.0:
            attitude *= -1.0
        payload = IdealOwnStateMeasurement(
            position_m=tuple(float(value) * 1.0e3 for value in truth.position_eci_km),
            velocity_m_s=tuple(float(value) * 1.0e3 for value in truth.velocity_eci_km_s),
            attitude_quat_body_from_inertial=tuple(float(value) for value in attitude),
            angular_rate_body_rad_s=tuple(float(value) for value in truth.angular_rate_body_rad_s),
            mass_kg=float(truth.mass_kg),
        )
        measurement = MeasurementEvent("ideal_own_state", payload.schema, now, self.inertial_frame, payload)
        return InputEvent(packet_id, InputKind.MEASUREMENT, now, now, Quality(DataValidity.VALID), measurement)

    def _measured_own_state_event(self, truth: StateTruth, now: ClockTag) -> InputEvent:
        events: list[InputEvent] = []
        position_sigma = _expand3(self.sensor_error.get("pos_sigma_km", 0.0)) * 1.0e3
        velocity_sigma = _expand3(self.sensor_error.get("vel_sigma_km_s", 0.0)) * 1.0e3
        position_bias = _expand3(self.sensor_error.get("pos_bias_km", 0.0)) * 1.0e3
        velocity_bias = _expand3(self.sensor_error.get("vel_bias_km_s", 0.0)) * 1.0e3
        position_m = np.asarray(truth.position_eci_km, dtype=float) * 1.0e3
        velocity_m_s = np.asarray(truth.velocity_eci_km_s, dtype=float) * 1.0e3
        gnss = GnssOwnStateMeasurement(
            tuple(float(value) for value in position_m + position_bias + self._sensor_rng.normal(0.0, position_sigma)),
            tuple(
                float(value) for value in velocity_m_s + velocity_bias + self._sensor_rng.normal(0.0, velocity_sigma)
            ),
        )
        events.append(self._measurement_input("gnss", gnss, now, self.inertial_frame))
        omega_sigma = _expand3(self.sensor_error.get("omega_sigma_rad_s", 0.0))
        gyro = GyroMeasurement(
            tuple(
                float(value)
                for value in np.asarray(truth.angular_rate_body_rad_s, dtype=float)
                + self._sensor_rng.normal(0.0, omega_sigma)
            )
        )
        events.append(self._measurement_input("gyro", gyro, now, self.body_frame))
        quaternion = np.asarray(truth.attitude_quat_bn, dtype=float)
        quaternion_sigma = _expand3(self.sensor_error.get("quat_sigma", 0.0))
        if np.any(quaternion_sigma > 0.0):
            delta = self._sensor_rng.normal(0.0, quaternion_sigma)
            quaternion = normalize_quaternion(
                quaternion_multiply(quaternion, quaternion_delta_from_body_rate(delta, 1.0))
            )
        star = StarTrackerMeasurement(tuple(float(value) for value in quaternion))
        events.append(self._measurement_input("star_tracker", star, now, self.body_frame))
        # Return the first event and queue the rest at the same delivery time.
        for event in events[1:]:
            self._enqueue_runtime_owned(event)
        return events[0]

    def _measurement_input(
        self,
        sensor_id: str,
        payload: GnssOwnStateMeasurement | GyroMeasurement | StarTrackerMeasurement,
        now: ClockTag,
        frame: FrameId,
    ) -> InputEvent:
        packet = PacketId(f"{self.satellite_id}/{sensor_id}", self.boot_id, self._sensor_sequence)
        self._sensor_sequence += 1
        measurement = MeasurementEvent(sensor_id, payload.schema, now, frame, payload)
        return InputEvent(packet, InputKind.MEASUREMENT, now, now, Quality(), measurement)

    def _relative_observation_event(
        self,
        target_id: str,
        observer: StateTruth,
        target: StateTruth,
        now: ClockTag,
    ) -> InputEvent:
        target_state = np.hstack((target.position_eci_km, target.velocity_eci_km_s))
        if self._target_access is None:
            observer_position = observer.position_eci_km
            observer_velocity = observer.velocity_eci_km_s
        else:
            # The access gate and the measurement must use the same sensor
            # phase center, including its rotational velocity about the COM.
            observer_position, observer_velocity, _ = self._target_access._sensor_state_eci(observer)
        observer_state = np.hstack((observer_position, observer_velocity))
        # Canonical RPO convention: the controlled satellite is the deputy and
        # the tracked/reference object is the chief.  State is therefore
        # deputy relative to chief, expressed in the chief's RIC frame.
        relative = self._relative_ric_state(observer_state, target_state)
        position_m = np.asarray(relative[:3], dtype=float) * 1.0e3
        velocity_m_s = np.asarray(relative[3:6], dtype=float) * 1.0e3
        position_m += self._sensor_rng.normal(
            0.0,
            _expand3(self.sensor_error.get("relative_pos_sigma_km", self.sensor_error.get("pos_sigma_km", 0.0)))
            * 1.0e3,
        )
        velocity_m_s += self._sensor_rng.normal(
            0.0,
            _expand3(self.sensor_error.get("relative_vel_sigma_km_s", self.sensor_error.get("vel_sigma_km_s", 0.0)))
            * 1.0e3,
        )
        range_m = float(np.linalg.norm(position_m))
        los = position_m / max(range_m, 1.0e-15)
        range_rate = float(los @ velocity_m_s)
        angular_rate = np.cross(los, velocity_m_s) / max(range_m, 1.0e-15)
        payload = RelativeObservationMeasurement(
            range_m=range_m,
            range_rate_m_s=range_rate,
            los_unit=tuple(float(value) for value in los),
            angular_rate_rad_s=tuple(float(value) for value in angular_rate),
            target_track_id=target_id,
        )
        packet = PacketId(f"{self.satellite_id}/relative/{target_id}", self.boot_id, self._sensor_sequence)
        self._sensor_sequence += 1
        relative_frame = FrameId(f"OEL/RIC/{target_id}", "frames-v1")
        measurement = MeasurementEvent("relative", payload.schema, now, relative_frame, payload)
        return InputEvent(packet, InputKind.MEASUREMENT, now, now, Quality(), measurement)

    def _receipt_event(self, receipt: ActuatorCommandReceipt, now: ClockTag) -> InputEvent:
        packet_id = PacketId(f"{self.satellite_id}/command_bus", self.boot_id, self._receipt_sequence)
        self._receipt_sequence += 1
        return InputEvent(packet_id, InputKind.ACTUATOR_RECEIPT, now, now, Quality(), receipt)

    def _ideal_tracked_state_event(self, target_id: str, truth: StateTruth, now: ClockTag) -> InputEvent:
        packet_id = PacketId(f"{self.satellite_id}/ideal_track/{target_id}", self.boot_id, self._sensor_sequence)
        self._sensor_sequence += 1
        payload = IdealTrackedObjectStateMeasurement(
            target_id,
            tuple(float(value) * 1.0e3 for value in truth.position_eci_km),
            tuple(float(value) * 1.0e3 for value in truth.velocity_eci_km_s),
        )
        measurement = MeasurementEvent(f"ideal_track/{target_id}", payload.schema, now, self.inertial_frame, payload)
        return InputEvent(packet_id, InputKind.MEASUREMENT, now, now, Quality(), measurement)

    def _ideal_sun_event(self, truth: StateTruth, now: ClockTag) -> InputEvent:
        packet_id = PacketId(f"{self.satellite_id}/ideal_sun", self.boot_id, self._sensor_sequence)
        self._sensor_sequence += 1
        if self.initial_jd_utc is None:
            sun_eci = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            elapsed_days = now.ticks * now.tick_period_ns / 86_400.0e9
            sun_eci = sun_position_eci_km_enhanced(self.initial_jd_utc + elapsed_days) - np.asarray(
                truth.position_eci_km, dtype=float
            )
        sun_eci /= max(float(np.linalg.norm(sun_eci)), 1.0e-15)
        sun_body = self._quaternion_to_dcm_bn(np.asarray(truth.attitude_quat_bn, dtype=float)) @ sun_eci
        payload = SunVectorMeasurement(tuple(float(value) for value in sun_body))
        measurement = MeasurementEvent("ideal_sun", payload.schema, now, self.body_frame, payload)
        return InputEvent(packet_id, InputKind.MEASUREMENT, now, now, Quality(), measurement)

    def _ideal_magnetometer_event(self, now: ClockTag) -> InputEvent:
        packet_id = PacketId(f"{self.satellite_id}/ideal_magnetometer", self.boot_id, self._sensor_sequence)
        self._sensor_sequence += 1
        assert self.ideal_magnetic_field_body_t is not None
        payload = MagnetometerMeasurement(self.ideal_magnetic_field_body_t)
        measurement = MeasurementEvent("ideal_magnetometer", payload.schema, now, self.body_frame, payload)
        return InputEvent(packet_id, InputKind.MEASUREMENT, now, now, Quality(), measurement)

    def _queue_actuator_telemetry(self, realization: ActuatorRealization, at: ClockTag) -> None:
        fields = (
            TelemetryField("demand_mode", realization.demand_mode.value),
            TelemetryField("saturated", realization.saturated),
            *realization.device_state,
        )
        payload = ActuatorTelemetryPayload(realization.actuator_id, fields)
        packet_id = PacketId(
            f"{self.satellite_id}/{realization.actuator_id}/telemetry", self.boot_id, self._telemetry_sequence
        )
        self._telemetry_sequence += 1
        self._enqueue_runtime_owned(InputEvent(packet_id, InputKind.ACTUATOR_TELEMETRY, at, at, Quality(), payload))


def ideal_wrench_device(
    satellite_id: str,
    actuator_id: str,
    frame: FrameId,
    *,
    max_force_n: float,
    max_torque_n_m: float,
    specific_impulse_s: float | None = None,
) -> tuple[ActuatorDeviceDefinition, IdealWrenchHardware]:
    return (
        ActuatorDeviceDefinition(
            satellite_id,
            actuator_id,
            frame,
            (IdealWrenchCommand,),
            ExpiryBehavior.ZERO,
        ),
        IdealWrenchHardware(
            actuator_id,
            max_force_n=max_force_n,
            max_torque_n_m=max_torque_n_m,
            specific_impulse_s=specific_impulse_s,
        ),
    )


def continuous_engine_device(
    satellite_id: str,
    actuator_id: str,
    frame: FrameId,
    *,
    max_thrust_n: float,
    specific_impulse_s: float | None = None,
) -> tuple[ActuatorDeviceDefinition, ContinuousEngineHardware]:
    return (
        ActuatorDeviceDefinition(
            satellite_id,
            actuator_id,
            frame,
            (ContinuousEngineCommand,),
            ExpiryBehavior.ZERO,
        ),
        ContinuousEngineHardware(
            actuator_id,
            max_thrust_n=max_thrust_n,
            specific_impulse_s=specific_impulse_s,
        ),
    )


def reaction_wheel_device(
    satellite_id: str,
    actuator_id: str,
    frame: FrameId,
    *,
    axes_body: tuple[tuple[float, float, float], ...],
    max_torque_n_m: tuple[float, ...],
    max_momentum_n_m_s: tuple[float, ...],
    initial_momentum_n_m_s: tuple[float, ...] | None = None,
    inertia_kg_m2: tuple[float, ...] | None = None,
    max_speed_rad_s: tuple[float, ...] | None = None,
) -> tuple[ActuatorDeviceDefinition, ReactionWheelHardware]:
    return (
        ActuatorDeviceDefinition(
            satellite_id,
            actuator_id,
            frame,
            (ReactionWheelTorqueCommand,),
            ExpiryBehavior.ZERO,
            validator=lambda payload: (
                len(payload.torque_n_m) == len(axes_body),
                "reaction_wheel_command_cardinality",
            ),
        ),
        ReactionWheelHardware(
            actuator_id,
            axes_body=axes_body,
            max_torque_n_m=max_torque_n_m,
            max_momentum_n_m_s=max_momentum_n_m_s,
            initial_momentum_n_m_s=initial_momentum_n_m_s,
            inertia_kg_m2=inertia_kg_m2,
            max_speed_rad_s=max_speed_rad_s,
        ),
    )


def magnetorquer_device(
    satellite_id: str,
    actuator_id: str,
    frame: FrameId,
    *,
    max_dipole_a_m2: tuple[float, ...],
    magnetic_field_body_t: tuple[float, float, float],
) -> tuple[ActuatorDeviceDefinition, MagnetorquerHardware]:
    return (
        ActuatorDeviceDefinition(
            satellite_id,
            actuator_id,
            frame,
            (MagnetorquerDipoleCommand,),
            ExpiryBehavior.ZERO,
            validator=lambda payload: (
                len(payload.dipole_a_m2) == 3,
                "magnetorquer_command_cardinality",
            ),
        ),
        MagnetorquerHardware(
            actuator_id,
            max_dipole_a_m2=max_dipole_a_m2,
            magnetic_field_body_t=magnetic_field_body_t,
        ),
    )


def cmg_device(
    satellite_id: str,
    actuator_id: str,
    frame: FrameId,
    *,
    momentum_n_m_s: tuple[float, ...],
    max_gimbal_rate_rad_s: tuple[float, ...],
) -> tuple[ActuatorDeviceDefinition, CmgHardware]:
    return (
        ActuatorDeviceDefinition(
            satellite_id,
            actuator_id,
            frame,
            (CmgGimbalRateCommand,),
            ExpiryBehavior.ZERO,
            validator=lambda payload: (
                len(payload.gimbal_rate_rad_s) == 3,
                "cmg_command_cardinality",
            ),
        ),
        CmgHardware(
            actuator_id,
            momentum_n_m_s=momentum_n_m_s,
            max_gimbal_rate_rad_s=max_gimbal_rate_rad_s,
        ),
    )


def rcs_thruster_device(
    satellite_id: str,
    actuator_id: str,
    frame: FrameId,
    *,
    direction_body: tuple[float, float, float],
    max_thrust_n: float,
    position_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    specific_impulse_s: float | None = None,
) -> tuple[ActuatorDeviceDefinition, RcsThrusterHardware]:
    return (
        ActuatorDeviceDefinition(
            satellite_id,
            actuator_id,
            frame,
            (ThrusterPulseCommand, ThrusterOnOffCommand),
            ExpiryBehavior.ZERO,
        ),
        RcsThrusterHardware(
            actuator_id,
            direction_body=direction_body,
            max_thrust_n=max_thrust_n,
            position_body_m=position_body_m,
            specific_impulse_s=specific_impulse_s,
        ),
    )


def aerodynamic_effector_device(
    satellite_id: str,
    actuator_id: str,
    coordinate_id: str,
    frame: FrameId,
    *,
    unit: str,
    minimum: float,
    maximum: float,
    neutral: float,
    rate_limit_per_s: float,
) -> tuple[ActuatorDeviceDefinition, AerodynamicEffectorHardware]:
    hardware = AerodynamicEffectorHardware(
        actuator_id,
        coordinate_id,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        neutral=neutral,
        rate_limit_per_s=rate_limit_per_s,
    )
    return (
        ActuatorDeviceDefinition(
            satellite_id,
            actuator_id,
            frame,
            (AerodynamicEffectorPositionCommand,),
            ExpiryBehavior.LATCH,
            validator=hardware.validate,
        ),
        hardware,
    )


def _expand3(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(3, float(array[0]))
    if array.size == 3 and np.all(np.isfinite(array)):
        return array
    raise ValueError("sensor-error values must be scalar or contain three finite values")
