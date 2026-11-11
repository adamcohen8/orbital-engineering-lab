"""Deterministic loopback cFS/SIL v2 endpoint used by the maintained mission."""

from __future__ import annotations

import argparse
import json
import socket
from dataclasses import replace

import numpy as np

from integrations.cfs_sil.v2_codec import V2PacketKind, decode_v2_packet, encode_v2_packet
from sim.flight_software import (
    ActuatorCommand,
    BootEvent,
    FlightSoftwareInputBatch,
    FlightSoftwareOutput,
    FrameId,
    IdealOwnStateMeasurement,
    IdealWrenchCommand,
    InputKind,
    PacketId,
    ValidityInterval,
)
from sim.flight_software.schemas import from_primitive


class MockCfsV2Endpoint:
    def __init__(self, *, kp: float, kd: float, max_torque_n_m: float) -> None:
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_torque_n_m = float(max_torque_n_m)
        self.boot_id = "cfs-boot"
        self.command_sequence = 0

    def handle(self, packet: bytes) -> bytes:
        request = decode_v2_packet(packet)
        primitive = json.loads(request.payload.decode("utf-8"))
        if request.kind is V2PacketKind.BOOT:
            event = from_primitive(BootEvent, primitive)
            self.boot_id = event.boot_id
            return encode_v2_packet(V2PacketKind.BOOT, request.sequence, {})
        if request.kind is V2PacketKind.INPUT_BATCH:
            batch = from_primitive(FlightSoftwareInputBatch, primitive)
            return encode_v2_packet(V2PacketKind.OUTPUT, request.sequence, self._step(batch))
        if request.kind is V2PacketKind.SHUTDOWN:
            return encode_v2_packet(V2PacketKind.SHUTDOWN, request.sequence, {})
        raise ValueError(f"mock cFS v2 endpoint does not support {request.kind.name}")

    def _step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput:
        measurement = None
        for event in batch.events:
            if event.kind is InputKind.MEASUREMENT and hasattr(event.payload, "payload"):
                candidate = event.payload.payload
                if isinstance(candidate, IdealOwnStateMeasurement):
                    measurement = candidate
        commands = ()
        if measurement is not None:
            quaternion = np.asarray(measurement.attitude_quat_body_from_inertial, dtype=float)
            if quaternion[0] < 0.0:
                quaternion *= -1.0
            torque = -self.kp * quaternion[1:] - self.kd * np.asarray(
                measurement.angular_rate_body_rad_s,
                dtype=float,
            )
            norm = float(np.linalg.norm(torque))
            if norm > self.max_torque_n_m > 0.0:
                torque *= self.max_torque_n_m / norm
            command_id = PacketId("cfs/wrench", self.boot_id, self.command_sequence)
            self.command_sequence += 1
            expires = replace(batch.invocation_time, ticks=batch.invocation_time.ticks + 1_000_000_000)
            command = ActuatorCommand(
                command_id,
                batch.satellite_id,
                "wrench",
                batch.invocation_time,
                ValidityInterval(batch.invocation_time, expires),
                FrameId(f"OEL/ACTUATOR/{batch.satellite_id}/wrench", "frames-v1"),
                IdealWrenchCommand((0.0, 0.0, 0.0), tuple(float(value) for value in torque)),
            )
            commands = (command,)
        return FlightSoftwareOutput(batch.satellite_id, batch.invocation_id, commands)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--bind-port", type=int, default=50101)
    parser.add_argument("--att-kp", type=float, default=0.10)
    parser.add_argument("--att-kd", type=float, default=0.05)
    parser.add_argument("--max-torque-nm", type=float, default=0.05)
    args = parser.parse_args()
    if args.bind_host not in {"127.0.0.1", "localhost"}:
        raise ValueError("the maintained mock cFS v2 endpoint is loopback-only")
    endpoint = MockCfsV2Endpoint(kp=args.att_kp, kd=args.att_kd, max_torque_n_m=args.max_torque_nm)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.bind((args.bind_host, int(args.bind_port)))
        while True:
            packet, source = udp.recvfrom(1 << 20)
            udp.sendto(endpoint.handle(packet), source)


if __name__ == "__main__":
    raise SystemExit(main())
