import unittest

import numpy as np

from sim.core.models import Command, StateBelief, StateTruth
from sim.mission.modules import (
    BudgetedEndStateExecution,
    ControllerPointingExecution,
    DesiredStateMissionStrategy,
    ImpulsiveExecution,
    IntegratedCommandExecution,
    PredictiveBurnExecution,
)
from sim.utils.quaternion import quaternion_to_dcm_bn


def _truth() -> StateTruth:
    return StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )


class _ConstantOrbitController:
    def __init__(self, thrust_eci_km_s2: list[float]):
        self.thrust = np.array(thrust_eci_km_s2, dtype=float)
        self.target_state = None

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        return Command(thrust_eci_km_s2=self.thrust.copy(), torque_body_nm=np.zeros(3), mode_flags={"mode": "constant"})


class _ZeroAttitudeController:
    def __init__(self):
        self.target = None

    def set_target(self, q):
        self.target = np.array(q, dtype=float)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        return Command.zero()


class _RecordingAttitudeController(_ZeroAttitudeController):
    def __init__(self):
        super().__init__()
        self.belief_state = None
        self.belief_covariance = None

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        self.belief_state = np.array(belief.state, dtype=float)
        self.belief_covariance = np.array(belief.covariance, dtype=float)
        return Command(torque_body_nm=np.array([0.25, 0.0, 0.0], dtype=float))


class MissionArchitectureMigrationTests(unittest.TestCase):
    def test_desired_state_strategy_emits_explicit_state_intent(self) -> None:
        s = DesiredStateMissionStrategy(
            desired_state_source="explicit",
            desired_position_eci_km=np.array([7100.0, 1.0, 2.0]),
            desired_velocity_eci_km_s=np.array([0.1, 7.4, 0.0]),
        )
        out = s.update(own_knowledge={})
        self.assertTrue(
            np.allclose(np.array(out["desired_state_eci_6"], dtype=float), np.array([7100.0, 1.0, 2.0, 0.1, 7.4, 0.0]))
        )

    def test_integrated_command_execution_consumes_desired_state_intent(self) -> None:
        e = IntegratedCommandExecution(alignment_tolerance_deg=180.0)
        controller = _ConstantOrbitController([1.0e-5, 0.0, 0.0])
        att = _ZeroAttitudeController()
        truth = _truth()
        orb_belief = StateBelief(
            state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)), covariance=np.eye(6), last_update_t_s=0.0
        )
        att_belief = StateBelief(
            state=np.hstack((truth.attitude_quat_bn, truth.angular_rate_body_rad_s)),
            covariance=np.eye(7),
            last_update_t_s=0.0,
        )
        out = e.update(
            intent={"desired_state_eci_6": np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0])},
            truth=truth,
            orbit_controller=controller,
            attitude_controller=att,
            orb_belief=orb_belief,
            att_belief=att_belief,
            t_s=0.0,
            env={"attitude_disabled": False},
        )
        self.assertTrue(
            np.allclose(np.array(controller.target_state, dtype=float), np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0]))
        )
        self.assertGreater(float(np.linalg.norm(np.array(out["thrust_eci_km_s2"], dtype=float))), 0.0)

    def test_budgeted_end_state_execution_reduces_delta_v_budget(self) -> None:
        e = BudgetedEndStateExecution(
            strategy="thrust_limited",
            max_thrust_n=10.0,
            min_thrust_n=0.0,
            burn_dt_s=1.0,
            available_delta_v_km_s=0.1,
            require_attitude_alignment=False,
            alignment_tolerance_deg=180.0,
        )
        truth = _truth()
        out = e.update(
            intent={"desired_state_eci_6": np.array([7000.0, 0.0, 0.0, 0.01, 7.5, 0.0])},
            truth=truth,
            env={"attitude_disabled": True},
        )
        self.assertIn("remaining_delta_v_km_s", out["mission_mode"])
        self.assertLess(float(out["mission_mode"]["remaining_delta_v_km_s"]), 0.1)

    def test_budgeted_end_state_execution_uses_mass_as_delta_v_truth_source(self) -> None:
        e = BudgetedEndStateExecution(
            strategy="thrust_limited",
            max_thrust_n=10.0,
            min_thrust_n=0.0,
            burn_dt_s=1.0,
            available_delta_v_km_s=0.0,
            require_attitude_alignment=False,
            alignment_tolerance_deg=180.0,
        )
        truth = _truth()
        truth.mass_kg = 120.0
        out = e.update(
            intent={"desired_state_eci_6": np.array([7000.0, 0.0, 0.0, 0.01, 7.5, 0.0])},
            truth=truth,
            dt_s=1.0,
            dry_mass_kg=100.0,
            orbital_isp_s=220.0,
            env={"attitude_disabled": True},
        )
        self.assertGreater(float(out["mission_mode"]["remaining_delta_v_km_s"]), 0.0)

    def test_budgeted_end_state_execution_uses_command_hold_interval_for_accel(self) -> None:
        e = BudgetedEndStateExecution(
            strategy="thrust_limited",
            max_thrust_n=10.0,
            min_thrust_n=0.0,
            burn_dt_s=0.25,
            available_delta_v_km_s=0.1,
            require_attitude_alignment=False,
            alignment_tolerance_deg=180.0,
        )
        truth = _truth()
        out = e.update(
            intent={"desired_state_eci_6": np.array([7000.0, 0.0, 0.0, 0.01, 7.5, 0.0])},
            truth=truth,
            dt_s=0.25,
            orbit_command_period_s=2.0,
            env={"attitude_disabled": True},
        )
        accel_mag = float(np.linalg.norm(np.array(out["thrust_eci_km_s2"], dtype=float)))
        applied_dv = float(out["mission_mode"]["applied_delta_v_km_s"])
        self.assertAlmostEqual(accel_mag, applied_dv / 2.0, places=12)

    def test_predictive_burn_execution_accepts_desired_state_intent(self) -> None:
        e = PredictiveBurnExecution(lead_time_s=0.0, alignment_tolerance_deg=180.0)
        controller = _ConstantOrbitController([2.0e-5, 0.0, 0.0])
        att = _ZeroAttitudeController()
        truth = _truth()
        out = e.update(
            intent={"desired_state_eci_6": np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0])},
            truth=truth,
            own_knowledge={},
            orbit_controller=controller,
            attitude_controller=att,
            att_belief=None,
            t_s=0.0,
            dt_s=1.0,
            env={"attitude_disabled": True},
        )
        self.assertGreater(float(np.linalg.norm(np.array(out["thrust_eci_km_s2"], dtype=float))), 0.0)

    def test_predictive_burn_fallback_attitude_belief_uses_canonical_state(self) -> None:
        execution = PredictiveBurnExecution(lead_time_s=0.0, alignment_tolerance_deg=180.0)
        attitude = _RecordingAttitudeController()
        truth = _truth()
        execution.update(
            intent={"desired_state_eci_6": np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0])},
            truth=truth,
            own_knowledge={},
            orbit_controller=_ConstantOrbitController([2.0e-5, 0.0, 0.0]),
            attitude_controller=attitude,
            att_belief=None,
            t_s=0.0,
            dt_s=1.0,
            env={"attitude_disabled": False},
        )

        self.assertIsNotNone(attitude.belief_state)
        self.assertIsNotNone(attitude.belief_covariance)
        self.assertEqual(attitude.belief_state.shape, (13,))
        self.assertEqual(attitude.belief_covariance.shape, (13, 13))
        self.assertTrue(np.allclose(attitude.belief_state[:6], np.hstack((truth.position_eci_km, truth.velocity_eci_km_s))))
        self.assertTrue(np.allclose(attitude.belief_state[6:10], truth.attitude_quat_bn))
        self.assertTrue(np.allclose(attitude.belief_state[10:13], truth.angular_rate_body_rad_s))

    def test_predictive_burn_respects_positive_lead_time_at_step_boundary(self) -> None:
        execution = PredictiveBurnExecution(lead_time_s=1.0, alignment_tolerance_deg=180.0)
        controller = _ConstantOrbitController([2.0e-5, 0.0, 0.0])
        truth = _truth()
        first = execution.update(
            intent={"desired_state_eci_6": np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0])},
            truth=truth,
            own_knowledge={},
            orbit_controller=controller,
            attitude_controller=_ZeroAttitudeController(),
            t_s=0.0,
            dt_s=1.0,
            env={"attitude_disabled": True},
        )
        truth.t_s = 1.0
        second = execution.update(
            intent={"desired_state_eci_6": np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0])},
            truth=truth,
            own_knowledge={},
            orbit_controller=controller,
            attitude_controller=_ZeroAttitudeController(),
            t_s=1.0,
            dt_s=1.0,
            env={"attitude_disabled": True},
        )

        self.assertTrue(np.allclose(first["thrust_eci_km_s2"], np.zeros(3)))
        self.assertGreater(float(np.linalg.norm(second["thrust_eci_km_s2"])), 0.0)

    def test_impulsive_execution_averages_pulse_over_step_interval(self) -> None:
        execution = ImpulsiveExecution(
            require_attitude_alignment=False,
            pulse_period_s=2.0,
            pulse_width_s=0.25,
            pulse_phase_s=0.0,
        )
        truth = _truth()
        commanded_accel = np.array([2.0e-5, 0.0, 0.0], dtype=float)
        out = execution.update(
            intent={},
            truth=truth,
            orbit_controller=_ConstantOrbitController(commanded_accel.tolist()),
            orb_belief=StateBelief(
                state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
                covariance=np.eye(6),
                last_update_t_s=0.0,
            ),
            t_s=0.0,
            dt_s=1.0,
        )

        self.assertTrue(np.allclose(out["thrust_eci_km_s2"], commanded_accel * 0.25))
        self.assertAlmostEqual(float(out["mission_mode"]["pulse_active_duration_s"]), 0.25)
        self.assertAlmostEqual(float(out["mission_mode"]["pulse_duty_fraction"]), 0.25)
        self.assertTrue(out["mission_bypass_orbital_command_latch"])

    def test_impulsive_execution_integrates_wrapped_and_multiple_pulses(self) -> None:
        execution = ImpulsiveExecution(pulse_period_s=2.0, pulse_width_s=0.25, pulse_phase_s=0.0)

        self.assertAlmostEqual(execution._pulse_active_duration_s(1.9, 0.5), 0.25)
        self.assertAlmostEqual(execution._pulse_active_duration_s(0.0, 5.0), 0.75)

    def test_controller_pointing_execution_targets_thruster_opposite_commanded_delta_v(self) -> None:
        e = ControllerPointingExecution(
            alignment_tolerance_deg=180.0,
            thruster_direction_body=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        controller = _ConstantOrbitController([1.0e-5, 0.0, 0.0])
        att = _ZeroAttitudeController()
        truth = _truth()
        orb_belief = StateBelief(
            state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)), covariance=np.eye(6), last_update_t_s=0.0
        )
        att_belief = StateBelief(
            state=np.hstack((truth.attitude_quat_bn, truth.angular_rate_body_rad_s)),
            covariance=np.eye(7),
            last_update_t_s=0.0,
        )

        out = e.update(
            intent={},
            truth=truth,
            t_s=0.0,
            orbit_controller=controller,
            attitude_controller=att,
            orb_belief=orb_belief,
            att_belief=att_belief,
        )

        self.assertGreater(float(np.linalg.norm(np.array(out["thrust_eci_km_s2"], dtype=float))), 0.0)
        c_bn = quaternion_to_dcm_bn(np.array(att.target, dtype=float))
        plume_axis_eci = c_bn.T @ np.array([0.0, 0.0, 1.0], dtype=float)
        self.assertTrue(np.allclose(plume_axis_eci, np.array([-1.0, 0.0, 0.0], dtype=float), atol=1e-8))


if __name__ == "__main__":
    unittest.main()
