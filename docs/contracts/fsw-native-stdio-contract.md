# Public native flight-software protocol

Contract: `oel.fsw.stdio.v1`. Scope: trusted local C, C++, or Rust executables
inside the public FSW Authoring Kit. OEL owns simulation time, dynamics,
measurements, actuator validation/realization, and review evidence.

## Transport and lifecycle

The host launches exactly one declared executable without a shell. Requests and
responses are UTF-8 JSON objects, each terminated by one LF on stdin/stdout.
Each complete line, including LF, is at most 262144 bytes. Stdout is reserved
for protocol messages; diagnostics use stderr. Stderr retention is bounded.

A request has `protocol`, a strictly increasing integer `sequence` in
`[1, 2^64 - 1]`, `operation`, and object `payload`. A response repeats the exact protocol, sequence, and operation and
adds boolean `ok` plus `payload`. Unknown protocol versions, failed responses,
wrong identities, duplicate response fields, nonfinite JSON, oversized messages, premature EOF, process
exit, and response deadlines fail the bridge. They are host integration
failures, not modeled onboard faults. Host response deadlines include both
writing a request and reading a response. They do not advance simulation time.

The first exchange is `hello`:

```json
{"protocol":"oel.fsw.stdio.v1","sequence":1,"operation":"hello","payload":{"boundary":"oel.fsw.boundary.v1"}}
```

```json
{"protocol":"oel.fsw.stdio.v1","sequence":1,"operation":"hello","ok":true,"payload":{"boundary":"oel.fsw.boundary.v1","checkpointable":false}}
```

Capabilities must match the manifest. A single `boot` follows `hello` before
any step, checkpoint, or shutdown operation. Subsequent operations are:

| Operation | Request payload | Successful response payload |
| --- | --- | --- |
| `boot` | Ordinary `BootEvent` | `{}` |
| `step` | `FlightSoftwareInputBatch` | `FlightSoftwareOutput` |
| `shutdown` | Ordinary `ShutdownEvent` | `{}` |
| `snapshot` | `{}` | `FlightSoftwareSnapshot` |
| `restore` | `FlightSoftwareSnapshot` | `{}` |

Snapshots/restores are available only when checkpointability is explicitly
declared by both candidate and endpoint. The shipped educational starters are
non-checkpointable and reject those operations. Stateful integrations must
implement and test their own snapshot semantics before declaring support.

The endpoint response remains the public
`oel.flight_software.snapshot.v1` record. After decoding that response, the
OEL host wraps it as `oel.flight_software.native_snapshot.v1` and adds
`host_implementation_hash`. The host computes that digest from the verified
source, manifest/contract, verification inputs, and executable bytes in the
trusted build receipt. A native endpoint cannot assert or replace this value.
The envelope is retained in runtime checkpoint evidence and must match the
current host-derived `StackIdentity.implementation_hash` before restore. The
host strips the envelope metadata when sending the restore request, so native
SDKs continue to consume the v1 payload.

Python stack checkpoints stay `oel.flight_software.snapshot.v1` and do not gain
native envelope fields. A pre-envelope native checkpoint is therefore a
legacy migration input. `restore()` rejects it even when
`allow_legacy_snapshot_restore: true`; the flag only enables the explicit
migration path. A caller must invoke
`migrate_legacy_snapshot(snapshot, expected_executable_sha256=...)` with the
old executable digest from the enclosing trusted checkpoint evidence. The host
compares that independently supplied digest with the currently verified build,
returns a new host-bound envelope, and only then permits `restore()`. The
runtime wrapper performs the same promotion after checking the outer checkpoint
identity and runtime state identity. New native checkpoints should be captured
after migration.

Payloads retain the canonical typed records in `sim.flight_software.contracts`
and their serialization in `sim.flight_software.schemas`: SI units, explicit
frames/clocks, schema and contract versions, event identities and ordering,
command validity, telemetry, and task-release requests. No Python object or
simulator truth is sent across the process boundary. Returned records pass the
ordinary decoder and actuator boundary. Output satellite and invocation IDs
must match the input. SDK callbacks receive the complete JSON payload; helper
functions cover ideal state and wrench examples, not every possible mission
mapping. Unsupported mappings must be handled explicitly by candidate code.

## SDKs and examples

`sim/fsw_authoring/sdk/c` owns one bounded JSON codec and stdio runtime. The C++
header adds `oel::Input` and a `FlightSoftware` callback class. The Rust crate
adds a lifetime-bound input view and a `FlightSoftware` trait; it builds and
links the same C runtime using CMake. C callbacks must write a NUL-terminated
JSON object within the supplied capacity and return zero on success. The SDK
checks termination, JSON validity, and the complete response bound before
emitting a frame. Rust therefore also needs a C compiler
and CMake. The bundled SDKs need no third-party registry dependencies.

All three language starters implement the same educational control law: ADCS
quaternion feedback plus rate damping, or ECI relative-position/velocity
feedback rotated into the body frame. These are editable controller examples,
not replacement orbit dynamics, a general rendezvous planner, or operational
flight software. Ideal navigation and a single tracked reference are assumed;
mission-specific sensing, stale-data policy, and fault response need explicit
candidate implementation and evidence. Wrench helpers preserve the input clock
and issue commands with one-second validity.

## Identity, builds, tests, and trust

Python `candidate.v1` remains unchanged. Native `candidate.v2` adds
`kind: native_stack`, a source language (`c`, `cpp`, `rust`), response timeout,
and checkpoint declaration. Inspection reads/hashes inputs without launching
compilers or candidates. Native validation checks the current build receipt and
smoke binding without executing the candidate.

`build --trusted-build` copies and verifies source, runs fixed CMake or Cargo
commands, and records source/config/test hashes, executable hash, toolchain
versions, build environment, platform, command arguments, and logs. C/C++ tests
use CTest; Rust tests use Cargo. Builds are serial, Cargo is offline/locked,
and starters need no downloads. Approved custom build scripts still execute
arbitrary host code and can perform their own IO. These settings do not sandbox
hostile source or enforce OS-level network isolation.

Source/header/build-script/dependency-lock, scenario, test, receipt, and binary
changes invalidate previously bound candidate identity or build validation.
Build, test, and smoke outputs must be fresh directories outside the candidate
source tree. Missing binaries or malformed retained receipts can be repaired
with a trusted rebuild. Component tests use the fixed toolchain command and
the hash-verified build source snapshot; receipt-supplied arbitrary commands
are refused. Test and smoke completion recheck candidate identity so an edit
during execution cannot produce a passed receipt for the earlier inputs.
The public authoring kit stops at authoring, bounded component checks, and one
deterministic serial smoke run. The local FSWDK covers Controller Bench,
tuning, qualification, baseline promotion, review packaging, and cFS/SIL;
restricted inputs and generated qualification/governance evidence retain their
private boundary.

## Platform and evidence limits

The implementation uses portable CMake/Cargo builds, LF framing, binary stdio
on Windows, and platform-specific process termination. A declared code path is
not host qualification. Native acceptance must be run on each compiler/OS row
before claiming that row supported. The implementation development evidence is
macOS arm64; Windows and Linux execution are not established by that evidence.
Run `python -m pytest sim/tests/test_fsw_native.py` on the target host. The suite
builds all three languages, runs both scenarios through OEL, checks command
parity, stale identities, and bridge failures. Simulation evidence is not
flight, real-time, hardware, or certification evidence.
