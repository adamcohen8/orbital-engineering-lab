from __future__ import annotations

import numpy as np
import pytest

from sim.actuators.attitude import AttitudeActuator, ReactionWheelLimits
from sim.actuators.physical import IdealWrenchHardware
from sim.core.models import Command
from sim.dynamics.spacecraft_geometry import GeometryAreaProfile
from sim.estimation.attitude_ekf import AttitudeEKFEstimator


def test_attitude_ekf_requires_a_positive_finite_sample_period() -> None:
    kwargs = {
        "inertia_kg_m2": np.diag([2.0, 3.0, 4.0]),
        "process_noise_diag": np.zeros(7),
        "meas_noise_diag": np.ones(7),
    }
    for dt_s in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="dt_s must be positive and finite"):
            AttitudeEKFEstimator(dt_s=dt_s, **kwargs)


def test_attitude_actuator_rejects_reverse_or_nonfinite_physics_steps() -> None:
    actuator = AttitudeActuator(
        reaction_wheels=ReactionWheelLimits(
            max_torque_nm=np.ones(3),
            max_momentum_nms=np.ones(3),
        )
    )
    command = Command(torque_body_nm=np.array([0.1, 0.0, 0.0]))
    initial_momentum = np.zeros(3)
    actuator.apply(command, limits={}, dt_s=-1.0)
    np.testing.assert_array_equal(actuator.wheel_momentum_wheels_nms, initial_momentum)
    for dt_s in (float("nan"), float("inf")):
        before = actuator.wheel_momentum_wheels_nms.copy()
        with pytest.raises(ValueError, match="dt_s must be finite"):
            actuator.apply(command, limits={}, dt_s=dt_s)
        np.testing.assert_array_equal(actuator.wheel_momentum_wheels_nms, before)
    before = actuator.wheel_momentum_wheels_nms.copy()
    with pytest.raises(ValueError, match="finite three-vector"):
        actuator.apply(Command(torque_body_nm=np.array([float("nan"), 0.0, 0.0])), limits={}, dt_s=0.1)
    np.testing.assert_array_equal(actuator.wheel_momentum_wheels_nms, before)


def test_ideal_wrench_rejects_nonfinite_limits_and_checkpoint_state() -> None:
    with pytest.raises(ValueError, match="hardware limits"):
        IdealWrenchHardware("wrench", max_force_n=float("nan"))
    with pytest.raises(ValueError, match="hardware limits"):
        IdealWrenchHardware("wrench", response_time_constant_s=float("inf"))

    hardware = IdealWrenchHardware("wrench")
    before = hardware.snapshot_state()
    with pytest.raises(ValueError, match="three finite components"):
        hardware.restore_state({"realized_force_n": [float("nan"), 0.0, 0.0], "realized_torque_n_m": [0.0] * 3})
    assert hardware.snapshot_state() == before


def test_geometry_profile_reports_malformed_direction_shape() -> None:
    triangles = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    with pytest.raises(ValueError, match="Profile directions"):
        GeometryAreaProfile.from_triangles(triangles, directions_body=np.array([1.0, 0.0, 0.0]))
