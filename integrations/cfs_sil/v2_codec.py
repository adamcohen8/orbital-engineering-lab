"""Canonical cFS/SIL v2 transport for the common SatelliteFlightSoftware ICD."""

from __future__ import annotations

import socket
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
from typing import Callable

from sim.flight_software import (
    BootEvent,
    FlightSoftwareInputBatch,
    FlightSoftwareOutput,
    FlightSoftwareSnapshot,
    SatelliteFlightSoftware,
    ShutdownEvent,
    StackIdentity,
)
from sim.flight_software.schemas import canonical_json_bytes, from_primitive

MAGIC = b"OEL2"
WIRE_VERSION = 2
HEADER = struct.Struct(">4sBBHII")


class V2PacketKind(IntEnum):
    INPUT_BATCH = 1
    OUTPUT = 2
    BOOT = 3
    SHUTDOWN = 4
    SNAPSHOT_REQUEST = 5
    SNAPSHOT_RESPONSE = 6
    RESTORE = 7


class CfsBridgeFailure(RuntimeError):
    """Transport/process failure; never evidence of a simulated onboard fault."""


@dataclass(frozen=True, slots=True)
class V2WirePacket:
    kind: V2PacketKind
    sequence: int
    payload: bytes


def encode_v2_packet(kind: V2PacketKind, sequence: int, value: object) -> bytes:
    if sequence < 0 or sequence >= 2**32:
        raise ValueError("cFS v2 sequence must fit in uint32")
    payload = canonical_json_bytes(value)
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    return HEADER.pack(MAGIC, WIRE_VERSION, int(kind), 0, sequence, len(payload)) + payload + struct.pack(">I", checksum)


def decode_v2_packet(packet: bytes) -> V2WirePacket:
    if len(packet) < HEADER.size + 4:
        raise ValueError("cFS v2 packet is truncated")
    magic, version, raw_kind, reserved, sequence, payload_size = HEADER.unpack_from(packet)
    if magic != MAGIC or version != WIRE_VERSION or reserved != 0:
        raise ValueError("cFS v2 packet header is unsupported")
    expected = HEADER.size + payload_size + 4
    if len(packet) != expected:
        raise ValueError("cFS v2 packet length does not match its header")
    payload = packet[HEADER.size : HEADER.size + payload_size]
    expected_checksum = struct.unpack_from(">I", packet, HEADER.size + payload_size)[0]
    if zlib.crc32(payload) & 0xFFFFFFFF != expected_checksum:
        raise ValueError("cFS v2 packet checksum mismatch")
    return V2WirePacket(V2PacketKind(raw_kind), sequence, payload)


Exchange = Callable[[bytes, float | None], bytes]


def _packet_evidence(packet: bytes, *, prefix: str) -> dict[str, object]:
    """Return bounded packet provenance without retaining full wire payloads."""

    return {
        f"{prefix}_size_bytes": len(packet),
        f"{prefix}_sha256": sha256(packet).hexdigest(),
    }


class CfsSatelliteFlightSoftwareAdapter(SatelliteFlightSoftware):
    """A cFS endpoint presented through the same in-process stack protocol."""

    def __init__(
        self,
        *,
        satellite_id: str,
        stack_id: str,
        stack_version: str,
        implementation_hash: str,
        exchange: Exchange,
        timeout_s: float | None = None,
        checkpointable: bool = False,
    ) -> None:
        self.satellite_id = satellite_id
        self._identity = StackIdentity(stack_id, stack_version, 1, implementation_hash, checkpointable)
        self.exchange = exchange
        self.timeout_s = timeout_s
        self._sequence = 0
        self._booted = False
        self._transport_events: list[dict[str, object]] = []

    @property
    def identity(self) -> StackIdentity:
        return self._identity

    def boot(self, event: BootEvent) -> None:
        self._ack(V2PacketKind.BOOT, event)
        self._booted = True

    def step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput:
        if not self._booted:
            raise RuntimeError("cFS flight-software endpoint is not booted")
        raw = self._exchange(V2PacketKind.INPUT_BATCH, batch, expected=V2PacketKind.OUTPUT)
        try:
            output = from_primitive(FlightSoftwareOutput, raw)
        except Exception as exc:
            raise self._response_failure(
                f"cFS endpoint returned an invalid output payload: {exc}",
                failure_type=type(exc).__name__,
            ) from exc
        if output.satellite_id != batch.satellite_id or output.invocation_id != batch.invocation_id:
            raise self._response_failure(
                "cFS endpoint returned mismatched output identity",
                failure_type="output_identity_mismatch",
            )
        return output

    def snapshot(self) -> FlightSoftwareSnapshot:
        if not self.identity.checkpointable:
            raise CfsBridgeFailure("cFS endpoint did not declare snapshot capability")
        raw = self._exchange(V2PacketKind.SNAPSHOT_REQUEST, {}, expected=V2PacketKind.SNAPSHOT_RESPONSE)
        try:
            return from_primitive(FlightSoftwareSnapshot, raw)
        except Exception as exc:
            raise self._response_failure(
                f"cFS endpoint returned an invalid snapshot payload: {exc}",
                failure_type=type(exc).__name__,
            ) from exc

    def restore(self, snapshot: FlightSoftwareSnapshot) -> None:
        if not self.identity.checkpointable:
            raise CfsBridgeFailure("cFS endpoint did not declare restore capability")
        self._ack(V2PacketKind.RESTORE, snapshot)

    def shutdown(self, event: ShutdownEvent) -> None:
        self._ack(V2PacketKind.SHUTDOWN, event)
        self._booted = False

    def transport_evidence(self) -> tuple[dict[str, object], ...]:
        """Return transport-only evidence, separate from modeled onboard faults."""

        return tuple(dict(item) for item in self._transport_events)

    def _failure(self, message: str) -> CfsBridgeFailure:
        error = CfsBridgeFailure(message)
        error.bridge_failure_evidence = {
            "satellite_id": self.satellite_id,
            "bridge_transport": self.transport_evidence(),
        }
        return error

    def _response_failure(self, message: str, *, failure_type: str) -> CfsBridgeFailure:
        # The wire packet and JSON parsed, but the typed contract did not.
        # Reclassify that exact response while retaining its packet hashes.
        if self._transport_events and self._transport_events[-1].get("status") == "ok":
            self._transport_events[-1] = {
                **self._transport_events[-1],
                "status": "bridge_failure",
                "failure_type": failure_type,
            }
        return self._failure(message)

    def _ack(self, kind: V2PacketKind, value: object) -> None:
        payload = self._exchange(kind, value, expected=kind)
        if payload != {}:
            if self._transport_events and self._transport_events[-1].get("status") == "ok":
                self._transport_events[-1] = {
                    **self._transport_events[-1],
                    "status": "bridge_failure",
                    "failure_type": "ack_payload_invalid",
                }
            raise self._failure(f"cFS {kind.name.lower()} acknowledgement must be an empty object")

    def _exchange(self, kind: V2PacketKind, value: object, *, expected: V2PacketKind) -> object:
        request_sequence = self._sequence
        self._sequence += 1
        request = encode_v2_packet(kind, request_sequence, value)
        try:
            response = self.exchange(request, self.timeout_s)
        except (OSError, TimeoutError) as exc:
            self._transport_events.append(
                {
                    "sequence": request_sequence,
                    "request_kind": kind.name,
                    "response_kind": None,
                    "status": "bridge_failure",
                    "failure_type": type(exc).__name__,
                    **_packet_evidence(request, prefix="request_packet"),
                }
            )
            raise self._failure(f"cFS bridge transport failure: {type(exc).__name__}: {exc}") from exc
        try:
            decoded = decode_v2_packet(response)
        except Exception as exc:
            self._transport_events.append(
                {
                    "sequence": request_sequence,
                    "request_kind": kind.name,
                    "response_kind": None,
                    "status": "bridge_failure",
                    "failure_type": type(exc).__name__,
                    **_packet_evidence(request, prefix="request_packet"),
                    **_packet_evidence(response, prefix="response_packet"),
                }
            )
            raise self._failure(f"cFS bridge returned a malformed v2 packet: {exc}") from exc
        if decoded.sequence != request_sequence or decoded.kind is not expected:
            self._transport_events.append(
                {
                    "sequence": request_sequence,
                    "request_kind": kind.name,
                    "response_kind": decoded.kind.name,
                    "status": "bridge_failure",
                    "failure_type": "response_identity_mismatch",
                    "response_sequence": decoded.sequence,
                    **_packet_evidence(request, prefix="request_packet"),
                    **_packet_evidence(response, prefix="response_packet"),
                }
            )
            raise self._failure("cFS bridge response kind or sequence mismatch")
        import json
        try:
            payload = json.loads(decoded.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._transport_events.append(
                {
                    "sequence": request_sequence,
                    "request_kind": kind.name,
                    "response_kind": decoded.kind.name,
                    "response_sequence": decoded.sequence,
                    "status": "bridge_failure",
                    "failure_type": type(exc).__name__,
                    **_packet_evidence(request, prefix="request_packet"),
                    **_packet_evidence(response, prefix="response_packet"),
                }
            )
            raise self._failure(f"cFS bridge returned an invalid JSON payload: {exc}") from exc
        self._transport_events.append(
            {
                "sequence": request_sequence,
                "request_kind": kind.name,
                "response_kind": decoded.kind.name,
                "response_sequence": decoded.sequence,
                "status": "ok",
                **_packet_evidence(request, prefix="request_packet"),
                **_packet_evidence(response, prefix="response_packet"),
            }
        )
        return payload


class CfsV2UdpFlightSoftwareAdapter(CfsSatelliteFlightSoftwareAdapter):
    """Configuration-friendly UDP adapter for the maintained live SIL mission."""

    def __init__(
        self,
        *,
        satellite_id: str,
        implementation_hash: str,
        stack_id: str = "cfs.reference_adcs",
        stack_version: str = "2.0.0",
        bind_host: str = "127.0.0.1",
        bind_port: int = 50100,
        cfs_host: str = "127.0.0.1",
        cfs_port: int = 50101,
        timeout_s: float | None = 1.0,
        checkpointable: bool = False,
        actuator_id: str = "wrench",
    ) -> None:
        loopback = {"127.0.0.1", "localhost"}
        if bind_host not in loopback or cfs_host not in loopback:
            raise ValueError("the maintained cFS v2 UDP adapter is restricted to loopback")
        if str(actuator_id) != "wrench":
            raise ValueError(
                "the maintained cFS bridge emits only actuator_id='wrench'; custom actuator ids are unsupported"
            )
        self.actuator_id = "wrench"
        # The maintained bridge uses AF_INET/sockaddr_in.  Resolve localhost
        # explicitly as IPv4 so a host preferring IPv6 cannot create a
        # transport that the bridge can never receive.
        bind_info = socket.getaddrinfo(
            bind_host, int(bind_port), family=socket.AF_INET, type=socket.SOCK_DGRAM
        )[0]
        peer_info = socket.getaddrinfo(
            cfs_host, int(cfs_port), family=socket.AF_INET, type=socket.SOCK_DGRAM
        )[0]
        self._socket = socket.socket(bind_info[0], socket.SOCK_DGRAM)
        try:
            self._socket.bind(bind_info[4])
        except Exception:
            self._socket.close()
            raise
        self._peer = peer_info[4]
        self._peer_ip = peer_info[4][0]

        def exchange(packet: bytes, request_timeout_s: float | None) -> bytes:
            self._socket.settimeout(request_timeout_s)
            self._socket.sendto(packet, self._peer)
            response, source = self._socket.recvfrom(1 << 20)
            if source[0] != self._peer_ip or int(source[1]) != int(self._peer[1]):
                raise OSError("cFS v2 response arrived from an unexpected endpoint")
            return response

        try:
            super().__init__(
                satellite_id=satellite_id,
                stack_id=stack_id,
                stack_version=stack_version,
                implementation_hash=implementation_hash,
                exchange=exchange,
                timeout_s=timeout_s,
                checkpointable=checkpointable,
            )
        except Exception:
            self._socket.close()
            raise

    def shutdown(self, event: ShutdownEvent) -> None:
        try:
            super().shutdown(event)
        finally:
            self._socket.close()
