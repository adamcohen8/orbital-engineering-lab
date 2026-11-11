# OEL cFS/SIL Flight-Software Boundary ICD v2

The Python v2 codec transports canonical `sim.flight_software` input and output
records. The maintained C reference-ADCS bridge implements a strict subset: one
valid canonical `ideal_own_state.v1` measurement event per input batch and one
`ideal_wrench.v1` output. Other typed events are rejected. It is not a generic
projection layer and must not replace mission-specific input mapping.

Each packet contains a 16-byte network-order header (`OEL2`, wire version,
packet kind, reserved zero, uint32 sequence, uint32 payload length), canonical
UTF-8 JSON, and a network-order CRC-32. Packet kinds cover boot, input batch,
output, snapshot, restore, shutdown, and their responses. The payload meaning,
SI units, frames, clocks, ordering, and schema versions are defined by
`GNC_v2_spec.md`; this envelope does not reinterpret them.

| Kind | Value | Request/response rule | Maintained C bridge |
| --- | ---: | --- | --- |
| input batch | 1 | request; response must be kind 2 with the same sequence | accepted, one canonical `ideal_own_state.v1` event |
| output | 2 | response to kind 1 | emitted as one canonical `ideal_wrench.v1` output |
| boot | 3 | request and empty-object acknowledgement both use kind 3 | accepted |
| shutdown | 4 | request and empty-object acknowledgement both use kind 4 | accepted |
| snapshot request | 5 | request; response must be kind 6 | rejected by maintained non-checkpointable C profile |
| snapshot response | 6 | response to kind 5 | not emitted by maintained C profile |
| restore | 7 | request and acknowledgement both use kind 7 | rejected by maintained non-checkpointable C profile |

The CRC-32 covers only the canonical UTF-8 JSON payload; the four-byte CRC is
network order and follows the payload. The maintained C configuration caps a
whole UDP datagram at 16,384 bytes, leaving at most 16,364 payload bytes after
the 16-byte header and four-byte CRC. Python decoders require the received
length to equal the declared payload length plus header and trailer, reject
unknown kinds/reserved bits, and bind every response to the request sequence.

Checkpoint capability is declared out of band by the trusted scenario/external
profile (`checkpointable: true|false`) before execution; there is no wire-level
capability handshake. The maintained live config declares `false`. An adapter
may send kinds 5-7 only when its declared identity is checkpointable and the
endpoint implements the required pairings.

Transport timeouts, process exits, malformed packets, and sequence mismatches
are recorded as bridge failures. They must never be emitted as modeled sensor,
actuator, or onboard software faults. An endpoint declares whether snapshot and
restore are supported before a run begins. Byte-level normative examples and
rejection cases live in `sim/tests/test_cfs_sil_v2_golden_packets.py` and
`sim/tests/test_cfs_sil_v2_mock_endpoint.py`.
