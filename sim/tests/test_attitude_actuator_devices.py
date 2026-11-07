from __future__ import annotations

import unittest

import numpy as np

from sim.actuators.attitude import (
    AttitudeActuator,
    ControlMomentGyroLimits,
    MagnetorquerLimits,
    ReactionWheelLimits,
    ThrusterPulseLimits,
    WheelDesaturationLimits,
)
from sim.control.attitude.wheel_desaturation import WheelDesaturationController
from sim.core.models import Command, StateBelief


class TestAttitudeActuatorDevices(unittest.TestCase):
    def test_reaction_wheel_rejects_nonfinite_axes(self):
        actuator = AttitudeActuator(
            reaction_wheels=ReactionWheelLimits(
                max_torque_nm=np.ones(3),
                max_momentum_nms=np.ones(3),
                wheel_axes_body=np.array(
                    [[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=float,
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            actuator.apply(Command.zero(), limits={}, dt_s=0.0)

    def test_magnetorquer_uses_b_cross_m_physical_torque(self):
        actuator = AttitudeActuator(magnetorquers=MagnetorquerLimits(max_dipole_a_m2=np.array([10.0, 10.0, 10.0])))
        command = Command(
            torque_body_nm=np.array([1.0e-5, 0.0, 0.0], dtype=float),
            mode_flags={"magnetic_field_body_t": np.array([0.0, 0.0, 5.0e-5], dtype=float)},
        )

        out = actuator.apply(command, limits={}, dt_s=1.0)

        self.assertTrue(np.allclose(out.torque_body_nm, command.torque_body_nm, atol=1e-12))
        self.assertEqual(out.mode_flags["magnetorquer_mode"], "physical_b_cross_m")
        self.assertIn("magnetorquer_dipole_cmd_a_m2", out.mode_flags)

    def test_magnetorquer_without_b_field_has_no_proxy_torque(self):
        actuator = AttitudeActuator(magnetorquers=MagnetorquerLimits(max_dipole_a_m2=np.array([10.0, 10.0, 10.0])))
        command = Command(torque_body_nm=np.array([1.0e-5, 2.0e-5, -3.0e-5], dtype=float))

        out = actuator.apply(command, limits={}, dt_s=1.0)

        self.assertTrue(np.allclose(out.torque_body_nm, np.zeros(3)))
        self.assertEqual(out.mode_flags["magnetorquer_mode"], "no_b_field_zero_torque")

    def test_wheel_and_magnetorquer_torques_are_aggregated(self):
        actuator = AttitudeActuator(
            reaction_wheels=ReactionWheelLimits(
                max_torque_nm=np.full(3, 0.1),
                max_momentum_nms=np.ones(3),
                wheel_axes_body=np.eye(3),
            ),
            magnetorquers=MagnetorquerLimits(max_dipole_a_m2=np.full(3, 10.0)),
        )
        command = Command(
            torque_body_nm=np.array([0.05, 0.0, 0.0]),
            mode_flags={"magnetic_field_body_t": np.array([1.0e-5, 0.0, 0.0])},
        )

        out = actuator.apply(command, limits={}, dt_s=1.0)

        self.assertTrue(np.allclose(out.torque_body_nm, command.torque_body_nm, atol=1.0e-12))
        self.assertTrue(np.allclose(out.mode_flags["rw_body_torque_applied_nm"], command.torque_body_nm))
        self.assertTrue(np.allclose(out.mode_flags["magnetorquer_torque_body_nm"], np.zeros(3)))

    def test_reaction_wheel_allocator_matrix_is_reused_for_fixed_geometry(self):
        actuator = AttitudeActuator(
            reaction_wheels=ReactionWheelLimits(
                max_torque_nm=np.full(3, 0.1),
                max_momentum_nms=np.ones(3),
                wheel_axes_body=np.eye(3),
            )
        )
        actuator.apply(Command(torque_body_nm=np.array([0.01, 0.02, 0.03])), limits={}, dt_s=1.0)
        cached = actuator._wheel_allocator_pinv
        actuator.apply(Command(torque_body_nm=np.array([0.02, 0.01, 0.03])), limits={}, dt_s=1.0)

        assert cached is actuator._wheel_allocator_pinv

    def test_control_moment_gyro_limits_torque_by_momentum_and_gimbal_rate(self):
        actuator = AttitudeActuator(
            control_moment_gyros=ControlMomentGyroLimits(
                max_torque_nm=np.array([1.0, 1.0, 1.0], dtype=float),
                momentum_nms=np.array([2.0, 2.0, 2.0], dtype=float),
                gimbal_rate_limit_rad_s=np.array([0.1, 0.1, 0.1], dtype=float),
            )
        )
        command = Command(torque_body_nm=np.array([0.5, 0.0, 0.0], dtype=float))

        out = actuator.apply(command, limits={}, dt_s=1.0)

        self.assertTrue(np.allclose(out.torque_body_nm, np.array([0.2, 0.0, 0.0], dtype=float)))
        self.assertTrue(np.allclose(out.mode_flags["cmg_torque_cap_nm"], [0.2, 0.2, 0.2]))

    def test_control_moment_gyro_zero_momentum_has_zero_cap_not_nan(self):
        actuator = AttitudeActuator(
            control_moment_gyros=ControlMomentGyroLimits(
                max_torque_nm=np.array([1.0, 1.0, 1.0], dtype=float),
                momentum_nms=np.zeros(3, dtype=float),
                gimbal_rate_limit_rad_s=np.inf,
            )
        )

        out = actuator.apply(Command(torque_body_nm=np.array([0.5, 0.0, 0.0], dtype=float)), limits={}, dt_s=1.0)

        self.assertTrue(np.all(np.isfinite(out.torque_body_nm)))
        self.assertTrue(np.allclose(out.torque_body_nm, np.zeros(3)))
        self.assertTrue(np.allclose(out.mode_flags["cmg_torque_cap_nm"], np.zeros(3)))

    def test_wheel_desaturation_adds_external_unload_torque(self):
        actuator = AttitudeActuator(
            reaction_wheels=ReactionWheelLimits(
                max_torque_nm=np.array([0.1, 0.1, 0.1], dtype=float),
                max_momentum_nms=np.array([1.0, 1.0, 1.0], dtype=float),
                wheel_axes_body=np.eye(3),
            ),
            wheel_desaturation=WheelDesaturationLimits(
                momentum_fraction_threshold=0.1,
                unload_gain_s_inv=1.0,
                max_unload_torque_nm=0.05,
            ),
        )
        actuator.wheel_momentum_nms = np.array([1.0, 0.0, 0.0], dtype=float)

        out = actuator.apply(Command(torque_body_nm=np.zeros(3)), limits={}, dt_s=1.0)

        self.assertTrue(bool(out.mode_flags["wheel_desaturation_active"]))
        self.assertLess(float(out.torque_body_nm[0]), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(out.torque_body_nm)), 0.05, places=8)

    def test_thruster_pulse_quantization_never_exceeds_torque_limit(self):
        actuator = AttitudeActuator(
            thruster_pulse=ThrusterPulseLimits(
                max_torque_nm=np.array([0.1, 0.2, 0.3], dtype=float),
                pulse_quantum_s=0.02,
            )
        )

        out = actuator.apply(
            Command(torque_body_nm=np.array([1.0, -1.0, 1.0], dtype=float)),
            limits={},
            dt_s=0.03,
        )

        self.assertTrue(np.all(np.abs(out.torque_body_nm) <= np.array([0.1, 0.2, 0.3]) + 1.0e-15))

    def test_zero_desaturation_limit_produces_zero_torque(self):
        actuator = AttitudeActuator(
            reaction_wheels=ReactionWheelLimits(
                max_torque_nm=np.full(3, 0.1),
                max_momentum_nms=np.ones(3),
                wheel_axes_body=np.eye(3),
            ),
            wheel_desaturation=WheelDesaturationLimits(
                momentum_fraction_threshold=0.1,
                unload_gain_s_inv=1.0,
                max_unload_torque_nm=0.0,
            ),
        )
        actuator.wheel_momentum_nms = np.array([1.0, 0.0, 0.0], dtype=float)

        out = actuator.apply(Command.zero(), limits={}, dt_s=1.0)

        self.assertTrue(bool(out.mode_flags["wheel_desaturation_active"]))
        self.assertTrue(np.allclose(out.torque_body_nm, np.zeros(3)))

        controller = WheelDesaturationController(
            wheel_momentum_body_nms=np.array([1.0, 0.0, 0.0], dtype=float),
            momentum_threshold_nms=0.1,
            unload_gain_s_inv=1.0,
            max_unload_torque_nm=0.0,
        )
        command = controller.act(
            StateBelief(state=np.zeros(1), covariance=np.eye(1), last_update_t_s=0.0),
            t_s=0.0,
            budget_ms=1.0,
        )
        self.assertTrue(bool(command.mode_flags["wheel_desaturation_requested"]))
        self.assertTrue(np.allclose(command.torque_body_nm, np.zeros(3)))


if __name__ == "__main__":
    unittest.main()
