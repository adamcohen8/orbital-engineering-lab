from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sim.actuators.attitude import AttitudeActuator, ControlMomentGyroLimits, MagnetorquerLimits, ReactionWheelLimits
from sim.control.attitude.baseline import QuaternionPDController, SmallAngleLQRController
from sim.control.attitude.bdot_magnetorquer import MagnetorquerBdotController
from sim.core.models import Command, StateTruth
from sim.dynamics import model as dynamics_model_module
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.propagator import OrbitPropagator


def _state() -> StateTruth:
    return StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0]),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.array([0.0, 0.0, 0.01]),
        mass_kg=100.0,
        t_s=0.0,
    )


class TestControlActuationContracts(unittest.TestCase):
    def test_python_attitude_backend_is_forwarded_on_ordinary_path(self) -> None:
        dynamics = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=np.diag([100.0, 90.0, 80.0]),
            orbit_propagator=OrbitPropagator(numeric_backend="python"),
            attitude_numeric_backend="python",
        )
        original = dynamics_model_module.propagate_attitude_exponential_map
        seen_backends: list[str | None] = []

        def wrapped(*args, **kwargs):
            seen_backends.append(kwargs.get("numeric_backend"))
            return original(*args, **kwargs)

        with patch.object(dynamics_model_module, "propagate_attitude_exponential_map", side_effect=wrapped):
            following = dynamics.step(_state(), Command.zero(), env={}, dt_s=0.25)

        self.assertTrue(seen_backends)
        self.assertEqual(set(seen_backends), {"python"})
        self.assertTrue(np.all(np.isfinite(following.attitude_quat_bn)))
        self.assertTrue(np.all(np.isfinite(following.angular_rate_body_rad_s)))

    def test_ordinary_dynamics_rejects_non_positive_or_nonfinite_timestep(self) -> None:
        dynamics = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=np.eye(3),
            orbit_propagator=OrbitPropagator(numeric_backend="python"),
        )
        for dt_s in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(dt_s=dt_s), self.assertRaisesRegex(ValueError, "timestep"):
                dynamics.step(_state(), Command.zero(), env={}, dt_s=dt_s)

    def test_nonfinite_controller_and_actuator_configuration_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "kp"):
            QuaternionPDController(kp=np.nan)
        with self.assertRaisesRegex(ValueError, "inertia_kg_m2"):
            SmallAngleLQRController(
                inertia_kg_m2=np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                numeric_backend="python",
            )
        with self.assertRaisesRegex(ValueError, "gain"):
            MagnetorquerBdotController(gain=np.nan)
        with self.assertRaisesRegex(ValueError, "max_torque_nm"):
            ReactionWheelLimits(max_torque_nm=np.array([np.nan]), max_momentum_nms=np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "max_dipole_a_m2"):
            MagnetorquerLimits(max_dipole_a_m2=np.array([np.nan, 1.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "gimbal_rate_limit_rad_s"):
            ControlMomentGyroLimits(
                max_torque_nm=np.ones(3),
                momentum_nms=np.ones(3),
                gimbal_rate_limit_rad_s=np.array([np.nan, 0.1, 0.1]),
            )

        # Zero limits disable a device, and positive-infinity CMG rate remains
        # the documented unlimited default.
        limits = ControlMomentGyroLimits(
            max_torque_nm=np.zeros(3),
            momentum_nms=np.ones(3),
            gimbal_rate_limit_rad_s=np.inf,
        )
        actuator = AttitudeActuator(control_moment_gyros=limits)
        output = actuator.apply(Command(torque_body_nm=np.ones(3)), limits={}, dt_s=1.0)
        self.assertTrue(np.allclose(output.torque_body_nm, np.zeros(3)))


if __name__ == "__main__":
    unittest.main()
