"""Lane 06 audit regression for RCS duty-cycle and pulse quantization."""

from __future__ import annotations

import numpy as np
import pytest

from sim.actuators.orbital import OrbitalActuator, OrbitalActuatorLimits, RcsClusterLimits, RcsThruster
from sim.core.models import Command


def test_rcs_pulse_quantization_never_exceeds_duty_limited_thrust() -> None:
    cluster = RcsClusterLimits(
        thrusters=(
            RcsThruster(
                name="jet",
                position_body_m=np.zeros(3),
                force_direction_body=np.array([1.0, 0.0, 0.0]),
                max_thrust_n=1.0,
                isp_s=220.0,
            ),
        ),
        allocation_mode="force_only",
        duty_cycle=0.7,
        pulse_quantum_s=0.4,
    )
    output = OrbitalActuator().apply(
        Command(
            thrust_eci_km_s2=np.array([1.0e-3, 0.0, 0.0]),
            mode_flags={"current_mass_kg": 1.0},
        ),
        {"orbital": OrbitalActuatorLimits(max_accel_km_s2=1.0, rcs_cluster=cluster)},
        dt_s=1.0,
    )

    assert output.mode_flags["rcs_thruster_forces_n"][0] <= 0.7 + 1.0e-15


def _rcs_output(*, duty: float, quantum_s: float, position_y_m: float = 0.0, dt_s: float = 1.0):
    cluster = RcsClusterLimits(
        thrusters=(
            RcsThruster(
                name="jet",
                position_body_m=np.array([0.0, position_y_m, 0.0]),
                force_direction_body=np.array([1.0, 0.0, 0.0]),
                max_thrust_n=1.0,
                isp_s=220.0,
            ),
        ),
        allocation_mode="force_only",
        duty_cycle=duty,
        pulse_quantum_s=quantum_s,
    )
    return OrbitalActuator().apply(
        Command(
            thrust_eci_km_s2=np.array([1.0e-3, 0.0, 0.0]),
            mode_flags={"current_mass_kg": 1.0},
        ),
        {"orbital": OrbitalActuatorLimits(max_accel_km_s2=1.0, rcs_cluster=cluster)},
        dt_s=dt_s,
    )


def test_rcs_quantization_returns_zero_when_duty_window_is_shorter_than_quantum() -> None:
    output = _rcs_output(duty=0.7, quantum_s=0.8)

    assert output.mode_flags["rcs_thruster_forces_n"] == [0.0]
    assert output.mode_flags["rcs_force_body_n"] == [0.0, 0.0, 0.0]
    assert output.mode_flags["rcs_torque_body_nm"] == [0.0, 0.0, 0.0]
    assert output.mode_flags["delta_mass_kg"] == 0.0


def test_rcs_quantization_keeps_integer_pulses_at_float_boundary_and_preserves_ledgers() -> None:
    # 0.3 / 0.1 is represented as 2.9999999999999996 on this platform.
    output = _rcs_output(duty=0.3, quantum_s=0.1, position_y_m=0.5)

    assert output.mode_flags["rcs_thruster_forces_n"][0] == pytest.approx(0.3)
    assert output.mode_flags["rcs_force_body_n"] == pytest.approx([0.3, 0.0, 0.0])
    assert output.mode_flags["rcs_torque_body_nm"] == pytest.approx([0.0, 0.0, -0.15])
    assert output.mode_flags["delta_mass_kg"] == pytest.approx(0.3 / (220.0 * 9.80665))


def test_rcs_quantization_rejects_nonrepresentable_finite_ratio() -> None:
    # The scalar inputs are finite, but their pulse-count ratio overflows a
    # float. Returning inf here would poison realized force and mass flow.
    with pytest.raises(ValueError, match="quantization ratio is not representable"):
        _rcs_output(duty=1.0, quantum_s=1.0e-308, dt_s=1.0e308)


@pytest.mark.parametrize("quantum_s", [float("nan"), float("inf"), -0.1])
def test_rcs_quantization_rejects_nonfinite_or_negative_quantum(quantum_s: float) -> None:
    with pytest.raises(ValueError, match="pulse_quantum_s"):
        _rcs_output(duty=1.0, quantum_s=quantum_s)


def test_rcs_actuator_rejects_negative_direct_step_but_keeps_zero_step_defined() -> None:
    with pytest.raises(ValueError, match="dt_s must be nonnegative"):
        _rcs_output(duty=1.0, quantum_s=0.1, dt_s=-1.0)

    output = _rcs_output(duty=0.7, quantum_s=0.4, dt_s=0.0)
    assert np.all(np.isfinite(np.asarray(output.thrust_eci_km_s2, dtype=float)))
    assert output.mode_flags["delta_mass_kg"] == 0.0


def test_rcs_large_finite_ratio_does_not_round_past_duty_cap() -> None:
    # The prior scale-dependent 8-ULP allowance admitted one extra pulse here
    # (old realized force was 1.0 N vs. a 0.999999999999999 N cap).
    duty = 0.999999999999999
    output = _rcs_output(duty=duty, quantum_s=0.01, dt_s=1.0e11)

    force_n = float(output.mode_flags["rcs_thruster_forces_n"][0])
    assert force_n <= duty
    assert force_n == pytest.approx(0.9999999999999)
