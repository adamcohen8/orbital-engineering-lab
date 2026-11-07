from __future__ import annotations

from hashlib import sha256

import pytest

from sim.flight_software.contracts import (
    ActuatorCommand,
    ClockScale,
    ClockTag,
    FlightSoftwareSnapshot,
    FrameId,
    GyroMeasurement,
    IdealWrenchCommand,
    PacketId,
    ReactionWheelTorqueCommand,
    ValidityInterval,
)


def _time(ticks: int = 10) -> ClockTag:
    return ClockTag("sat-clock", ticks, 1_000_000, ClockScale.ONBOARD)


def test_contracts_enforce_units_shapes_ranges_and_identity() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        GyroMeasurement((1.0, 2.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        GyroMeasurement((1.0, float("nan"), 2.0))
    with pytest.raises(ValueError, match="at least one wheel"):
        ReactionWheelTorqueCommand(())
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        PacketId("sensor", "boot", 2**64)


def test_covariance_must_be_symmetric_positive_semidefinite() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        GyroMeasurement((0.0, 0.0, 0.0), ((1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="positive semidefinite"):
        GyroMeasurement((0.0, 0.0, 0.0), ((1.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_actuator_command_preserves_si_values_and_native_frame() -> None:
    command = ActuatorCommand(
        command_id=PacketId("fsw", "boot", 7),
        satellite_id="sat",
        actuator_id="ideal-wrench",
        issued_at=_time(),
        validity=ValidityInterval(_time(), _time(20)),
        frame=FrameId("OEL/ACTUATOR/sat/ideal-wrench", "frames-v1"),
        payload=IdealWrenchCommand((1.0, 2.0, 3.0), (0.1, 0.2, 0.3)),
    )
    assert command.payload.force_n == (1.0, 2.0, 3.0)
    assert command.frame.name == "OEL/ACTUATOR/sat/ideal-wrench"


def test_snapshot_hash_is_verified() -> None:
    state = b"deterministic-state"
    snapshot = FlightSoftwareSnapshot(
        stack_id="fsw.reference",
        stack_version="2.0.0",
        boot_id="boot",
        invocation_id=4,
        active_load_id=None,
        active_load_revision=None,
        state_bytes=state,
        state_hash_sha256=sha256(state).hexdigest(),
    )
    assert snapshot.state_bytes == state
    with pytest.raises(ValueError, match="does not match"):
        FlightSoftwareSnapshot(
            stack_id="fsw.reference",
            stack_version="2.0.0",
            boot_id="boot",
            invocation_id=4,
            active_load_id=None,
            active_load_revision=None,
            state_bytes=state,
            state_hash_sha256="0" * 64,
        )


@pytest.mark.parametrize("revision", [True, 1.5, "1"])
def test_snapshot_rejects_non_integer_active_load_revision(revision: object) -> None:
    state = b"deterministic-state"
    with pytest.raises(TypeError, match="active_load_revision must be an integer"):
        FlightSoftwareSnapshot(
            stack_id="fsw.reference",
            stack_version="2.0.0",
            boot_id="boot",
            invocation_id=4,
            active_load_id="load-1",
            active_load_revision=revision,  # type: ignore[arg-type]
            state_bytes=state,
            state_hash_sha256=sha256(state).hexdigest(),
        )
