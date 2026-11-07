"""Physical clipping and FDIR must agree at a floating-point wrench limit."""
from __future__ import annotations

from math import inf, sqrt

import pytest

from sim.actuators.command_bus import ActuatorDemand, DemandMode
from sim.actuators.physical import IdealWrenchHardware
from sim.flight_software.contracts import (
    ActuatorTelemetryPayload,
    ClockScale,
    ClockTag,
    IdealWrenchCommand,
    InputEvent,
    InputKind,
    PacketId,
    Quality,
    TelemetryField,
)
from sim.gnc.operations_v2 import HealthManagerConfig, StackHealthManager

# Recorded RIC request: the controller has already limited this torque to
# 0.05 N m, but its computed norm is one representable float above the limit.
RIC_LIMIT_TORQUE = (-0.02332781441093313, -0.04422457530204804, 3.7633155414476205e-06)


def hardware(backend, maximum):
    result = IdealWrenchHardware("attitude", max_force_n=maximum, max_torque_n_m=maximum)
    if backend == "rust":
        pytest.importorskip("oel_rust_orbit")
        from sim.actuators.rust_physical import enable_hardware

        enable_hardware(result)
    return result


def advance(device, vector, channel="torque", index=0):
    zero = (0.0, 0.0, 0.0)
    payload = IdealWrenchCommand(vector if channel == "force" else zero,
                                vector if channel == "torque" else zero)
    demand = ActuatorDemand("attitude", DemandMode.COMMANDED, None, payload)
    return device.advance(demand, start_time_ns=index * 100_000_000, end_time_ns=(index + 1) * 100_000_000)


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("channel", ["force", "torque"])
def test_already_limited_ric_wrench_is_clipped_without_a_false_saturation(backend, channel):
    result = advance(hardware(backend, 0.05), RIC_LIMIT_TORQUE, channel)
    squared_norm = 0.0
    for value in RIC_LIMIT_TORQUE:
        squared_norm += value * value
    expected = tuple(value * (0.05 / sqrt(squared_norm)) for value in RIC_LIMIT_TORQUE)
    actual = result.realized_force_n if channel == "force" else result.realized_torque_n_m
    # Preserve the physical limiter's exact output; only classification changes.
    assert actual == expected
    assert result.saturated is False


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("maximum", [1e-100, 0.05, 1e100])
def test_roundoff_classification_scales_with_the_hardware_limit(backend, maximum):
    device = hardware(backend, maximum)
    close = advance(device, (maximum * (1.0 + 4.0e-16), 0.0, 0.0))
    assert close.saturated is False
    assert close.realized_torque_n_m[0] == pytest.approx(maximum, rel=1e-15, abs=0.0)
    excess = advance(device, (maximum * 1.000001, 0.0, 0.0))
    assert excess.saturated is True
    assert excess.realized_torque_n_m[0] == pytest.approx(maximum, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_zero_limit_has_no_absolute_noise_allowance_and_unbounded_limit_is_preserved(backend):
    disabled = hardware(backend, 0.0)
    assert advance(disabled, (1e-100, 0.0, 0.0)).saturated is True
    assert disabled.realized_torque_n_m == (0.0, 0.0, 0.0)
    assert advance(disabled, (0.0, 0.0, 0.0)).saturated is False
    unlimited = advance(hardware(backend, inf), (1.0, 2.0, 3.0))
    assert unlimited.realized_torque_n_m == (1.0, 2.0, 3.0)
    assert unlimited.saturated is False


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("genuine_excess", [False, True])
def test_fdir_ignores_limit_roundoff_but_still_isolates_persistent_physical_saturation(backend, genuine_excess):
    device = hardware(backend, 0.05)
    health = StackHealthManager(HealthManagerConfig(saturation_limit=5))
    vector = (0.051, 0.0, 0.0) if genuine_excess else RIC_LIMIT_TORQUE
    for index in range(5):
        realization = advance(device, vector, index=index)
        now = ClockTag("clock", index + 1, 100_000_000, ClockScale.ONBOARD)
        event = InputEvent(
            PacketId("attitude/telemetry", "boot", index), InputKind.ACTUATOR_TELEMETRY,
            now, now, Quality(),
            ActuatorTelemetryPayload("attitude", (TelemetryField("saturated", realization.saturated),)),
        )
        assessment = health.update(now, (event,))
        if index < 4:
            assert assessment.isolated_components == ()
    assert assessment.isolated_components == (("attitude",) if genuine_excess else ())
    assert assessment.active_faults == ((("attitude", "persistent_saturation"),) if genuine_excess else ())
