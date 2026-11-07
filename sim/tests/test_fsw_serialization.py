from __future__ import annotations

from hashlib import sha256

import pytest

from sim.flight_software import (
    ActuatorCommand,
    ActuatorTelemetryPayload,
    AerodynamicEffectorPositionCommand,
    ClockScale,
    ClockTag,
    CmgGimbalRateCommand,
    ContinuousEngineCommand,
    FlightSoftwareInputBatch,
    FlightSoftwareOutput,
    FlightSoftwareSnapshot,
    FrameId,
    GnssOwnStateMeasurement,
    GyroMeasurement,
    IdealOwnStateMeasurement,
    IdealWrenchCommand,
    InputEvent,
    InputKind,
    MagnetometerMeasurement,
    MagnetorquerDipoleCommand,
    MeasurementEvent,
    NativeFlightSoftwareSnapshot,
    PacketId,
    Quality,
    ReactionWheelTorqueCommand,
    RelativeObservationMeasurement,
    StarTrackerMeasurement,
    SunVectorMeasurement,
    TelemetryField,
    ThrusterOnOffCommand,
    ThrusterPulseCommand,
    ValidityInterval,
    canonical_json_bytes,
    canonical_loads,
    from_primitive,
    to_primitive,
)
from sim.flight_software.schemas import (
    _canonical_json_bytes_trusted,
    _to_primitive_trusted,
    registered_schemas,
)


def _time() -> ClockTag:
    return ClockTag("clock", 12, 1_000, ClockScale.ONBOARD)


@pytest.mark.parametrize(
    "payload",
    (
        GyroMeasurement((0.1, 0.2, 0.3)),
        StarTrackerMeasurement((1.0, 0.0, 0.0, 0.0)),
        SunVectorMeasurement((1.0, 0.0, 0.0), irradiance_w_m2=1361.0),
        MagnetometerMeasurement((1.0e-5, 2.0e-5, 3.0e-5)),
        GnssOwnStateMeasurement((7.0e6, 0.0, 0.0), (0.0, 7.5e3, 0.0)),
        RelativeObservationMeasurement(range_m=12.0, los_unit=(1.0, 0.0, 0.0)),
        IdealOwnStateMeasurement(position_m=(7.0e6, 0.0, 0.0), mass_kg=50.0),
        ActuatorTelemetryPayload("wheel", (TelemetryField("speed", 20.0, "rad/s"),)),
        ReactionWheelTorqueCommand((0.1, -0.1, 0.0)),
        MagnetorquerDipoleCommand((0.2, 0.0, -0.2)),
        CmgGimbalRateCommand((0.01, 0.02, 0.03, 0.04)),
        ThrusterPulseCommand("rcs-1", _time(), 0.05),
        ThrusterOnOffCommand("rcs-1", True),
        ContinuousEngineCommand(0.5, (0.01, -0.01)),
        AerodynamicEffectorPositionCommand("flap", 0.25, "rad"),
        IdealWrenchCommand((1.0, 2.0, 3.0), (0.1, 0.2, 0.3)),
    ),
)
def test_every_required_payload_has_a_golden_round_trip(payload: object) -> None:
    wire = canonical_json_bytes(payload)
    assert canonical_loads(wire) == payload
    assert canonical_json_bytes(canonical_loads(wire)) == wire
    assert _to_primitive_trusted(payload) == to_primitive(payload)
    assert _canonical_json_bytes_trusted(payload) == wire


def test_canonical_json_has_stable_field_order_and_no_whitespace() -> None:
    payload = GyroMeasurement((1.0, 2.0, 3.0))
    assert canonical_json_bytes(payload) == (
        b'{"angular_rate_rad_s":[1.0,2.0,3.0],"covariance_rad2_s2":null,"schema":"gyro.v1"}'
    )


def test_snapshot_bytes_use_canonical_base64_round_trip() -> None:
    state = b"\x00state\xff"
    snapshot = FlightSoftwareSnapshot(
        "stack",
        "2.0.0",
        "boot",
        1,
        None,
        None,
        state,
        sha256(state).hexdigest(),
    )
    wire = canonical_json_bytes(snapshot)
    assert b'"$bytes_base64"' in wire
    assert canonical_loads(wire) == snapshot


def test_native_snapshot_schema_dispatch_retains_host_implementation_identity() -> None:
    state = b"native state"
    snapshot = NativeFlightSoftwareSnapshot(
        stack_id="stack",
        stack_version="2.0.0",
        boot_id="boot",
        invocation_id=1,
        active_load_id=None,
        active_load_revision=None,
        state_bytes=state,
        state_hash_sha256=sha256(state).hexdigest(),
        host_implementation_hash="a" * 64,
    )
    assert registered_schemas()["oel.flight_software.native_snapshot.v1"] is NativeFlightSoftwareSnapshot
    decoded = canonical_loads(canonical_json_bytes(snapshot))
    assert decoded == snapshot
    assert decoded.host_implementation_hash == "a" * 64


def test_nested_input_and_output_boundary_batches_round_trip() -> None:
    time = _time()
    sensor_frame = FrameId("OEL/SENSOR/sat/gyro", "frames-v1")
    measurement = MeasurementEvent("gyro", "gyro.v1", time, sensor_frame, GyroMeasurement((0.1, 0.2, 0.3)))
    event = InputEvent(PacketId("gyro", "boot", 1), InputKind.MEASUREMENT, time, time, Quality(), measurement)
    input_batch = FlightSoftwareInputBatch("sat", 1, time, (event,))
    assert canonical_loads(canonical_json_bytes(input_batch)) == input_batch

    command = ActuatorCommand(
        PacketId("fsw", "boot", 1),
        "sat",
        "wheel",
        time,
        ValidityInterval(time),
        FrameId("OEL/ACTUATOR/sat/wheel", "frames-v1"),
        ReactionWheelTorqueCommand((0.1, 0.0, 0.0)),
    )
    output = FlightSoftwareOutput("sat", 1, (command,))
    assert canonical_loads(canonical_json_bytes(output)) == output
    assert _to_primitive_trusted(input_batch) == to_primitive(input_batch)
    assert _to_primitive_trusted(output) == to_primitive(output)


def test_decode_rejects_unknown_fields_nonfinite_json_and_wrong_literals() -> None:
    primitive = to_primitive(GyroMeasurement((1.0, 2.0, 3.0)))
    assert isinstance(primitive, dict)
    primitive["truth"] = 3
    with pytest.raises(ValueError, match="unknown fields"):
        from_primitive(GyroMeasurement, primitive)
    with pytest.raises(ValueError, match="non-finite"):
        canonical_loads('{"schema":"gyro.v1","angular_rate_rad_s":[NaN,0,0]}')
    with pytest.raises(ValueError, match="must equal"):
        from_primitive(GyroMeasurement, {"schema": "wrong", "angular_rate_rad_s": [0.0, 0.0, 0.0]})
