"""Cooperative port alignment and docking for the two-vehicle example."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sim.flight_software.contracts import (
    ActuatorCommand,
    BootEvent,
    DataValidity,
    DiagnosticTelemetry,
    DockingMechanismCommand,
    DockingPortMeasurement,
    FlightSoftwareInputBatch,
    FlightSoftwareOutput,
    FrameId,
    IdealOwnStateMeasurement,
    IdealWrenchCommand,
    MeasurementEvent,
    PacketId,
    ShutdownEvent,
    StackIdentity,
    TelemetryField,
    ValidityInterval,
)
from sim.utils.quaternion import quaternion_to_dcm_bn


class CooperativeDockingStack:
    """Example complete-stack plugin using ideal sensors and bounded wrench commands."""

    numeric_backend = "python"

    def __init__(
        self, role: str, arm_capture: bool = True, request_hard_mate: bool = True,
        navigation_initialization: str = "ideal",
    ):
        if navigation_initialization != "ideal":
            raise ValueError("The docking example requires ideal own-state measurements.")
        if role not in {"probe", "receiver"}:
            raise ValueError("Docking example role must be probe or receiver.")
        if not isinstance(arm_capture, bool) or not isinstance(request_hard_mate, bool):
            raise ValueError("Docking example authority settings must be booleans.")
        self.role = role
        self.arm_capture = arm_capture
        self.request_hard_mate = request_hard_mate
        self._identity = StackIdentity(
            f"example.docking.{role}", "1.0", 1, sha256(Path(__file__).read_bytes()).hexdigest(), False,
        )

    @property
    def identity(self):
        return self._identity

    def boot(self, event: BootEvent) -> None:
        self.boot_id = event.boot_id
        self.own = None
        self.port = None
        self.own_time = None
        self.port_time = None
        self.aborted = False

    def snapshot(self):
        raise ValueError("The docking example does not support checkpoints.")

    def restore(self, snapshot):
        raise ValueError("The docking example does not support checkpoints.")

    def shutdown(self, event: ShutdownEvent) -> None:
        pass

    def step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput:
        for event in batch.events:
            if event.quality.validity is not DataValidity.VALID or not isinstance(event.payload, MeasurementEvent):
                continue
            measurement = event.payload
            if isinstance(measurement.payload, IdealOwnStateMeasurement):
                self.own, self.own_time = measurement.payload, measurement.sample_time
            elif isinstance(measurement.payload, DockingPortMeasurement):
                self.port, self.port_time = measurement.payload, measurement.sample_time
        now = batch.invocation_time
        fresh = all(
            stamp is not None and stamp.clock_id == now.clock_id
            and 0 <= (now.ticks - stamp.ticks) * now.tick_period_ns <= 100_000_000
            for stamp in (self.own_time, self.port_time)
        )
        force, torque = np.zeros(3), np.zeros(3)
        request = DockingMechanismCommand()
        mode = "hold_for_measurements"
        if fresh and all(value is not None for value in (
            self.own.attitude_quat_body_from_inertial,
            self.own.angular_rate_body_rad_s,
            self.own.mass_kg,
        )):
            port, own = self.port, self.own
            if port.phase == "mated":
                mode = "docked"
            elif port.phase == "captured":
                mode = "verify_capture"
                request = DockingMechanismCommand(request_hard_mate=self.request_hard_mate)
            elif port.phase == "contact":
                mode = "contact_hold"
            else:
                self.aborted |= port.phase == "separated"
                body_from_eci = quaternion_to_dcm_bn(own.attitude_quat_body_from_inertial)
                gap = port.offset_m[2]
                lateral = np.linalg.norm(port.offset_m[:2])
                angle = np.linalg.norm(port.rotation_error_eci_rad)
                ready = 0. <= gap <= 1. and lateral < .012 and angle < .035
                request = DockingMechanismCommand(arm_capture=bool(self.arm_capture and ready and not self.aborted))
                if self.role == "receiver":
                    # Hold the inertial attitude supplied for the cooperative
                    # receiver in the example. Translation is passive.
                    error_body = body_from_eci @ Rotation.from_matrix(body_from_eci.T).as_rotvec()
                    torque = -9. * error_body - 23. * np.asarray(own.angular_rate_body_rad_s)
                    mode = "receiver_ready" if request.arm_capture else "receiver_hold"
                else:
                    error_body = body_from_eci @ port.rotation_error_eci_rad
                    relative_rate = body_from_eci @ port.relative_rate_eci_rad_s
                    torque = -9. * error_body - 23. * relative_rate
                    if self.aborted:
                        desired_normal_speed, mode = .06, "retreat"
                    elif request.arm_capture and port.peer_capture_armed:
                        desired_normal_speed, mode = -.045, "closing"
                    else:
                        desired_normal_speed, mode = 0., "aligning"
                    velocity_error = np.asarray(port.velocity_m_s).copy()
                    velocity_error[2] -= desired_normal_speed
                    accel_receiver = -.85 * velocity_error
                    accel_receiver[:2] -= .23 * np.asarray(port.offset_m[:2])
                    force = own.mass_kg * (body_from_eci @ np.asarray(port.receiver_axes_eci).T @ accel_receiver)
        expires = replace(now, ticks=now.ticks + 100_000_000 // now.tick_period_ns)
        commands = tuple(
            ActuatorCommand(
                PacketId(f"{batch.satellite_id}/docking_fsw", self.boot_id, 2 * batch.invocation_id + index),
                batch.satellite_id, device, now, ValidityInterval(now, expires),
                FrameId(f"OEL/ACTUATOR/{batch.satellite_id}/{device}", "frames-v1"), payload,
            )
            for index, (device, payload) in enumerate((
                ("wrench", IdealWrenchCommand(tuple(force), tuple(torque))), ("docking", request),
            ))
        )
        return FlightSoftwareOutput(
            batch.satellite_id, batch.invocation_id, commands,
            (DiagnosticTelemetry("oel.example.docking.v1", now, (
                TelemetryField("role", self.role), TelemetryField("mode", mode),
                TelemetryField("capture_armed", request.arm_capture),
                TelemetryField("hard_mate_requested", request.request_hard_mate),
            )),),
        )
