from __future__ import annotations

import numpy as np

from sim.control.orbit.gimbaled_thruster import GimbaledThrusterController
from sim.core.models import Command, SimLog, StateBelief
from sim.metrics.scoring import compute_scores


class _FixedThrustController:
    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        del belief, t_s, budget_ms
        return Command(
            thrust_eci_km_s2=np.array([1.0e-5, 0.0, 0.0]),
            torque_body_nm=np.zeros(3),
            mode_flags={},
        )


def test_gimbal_limit_applies_to_compact_attitude_belief() -> None:
    controller = GimbaledThrusterController(
        base_controller=_FixedThrustController(),
        neutral_direction_body=np.array([0.0, 1.0, 0.0]),
        max_gimbal_angle_rad=0.0,
    )
    # The configured quaternion slice is [6:10]. A compact 10-state belief is
    # sufficient for this controller and must not bypass the gimbal constraint.
    belief = StateBelief(
        state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(10),
        last_update_t_s=0.0,
    )

    command = controller.act(belief, t_s=0.0, budget_ms=1.0)

    np.testing.assert_allclose(command.thrust_eci_km_s2, np.zeros(3))
    assert command.mode_flags["gimbal_angle_request_rad"] == np.pi / 2.0


def test_scores_are_finite_for_single_row_run() -> None:
    zeros = np.zeros((1, 3))
    truth = np.array(
        [[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]]
    )
    log = SimLog(
        t_s=np.array([0.0]),
        truth_by_object={"sat": truth},
        belief_by_object={"sat": truth[:, :6]},
        applied_thrust_by_object={"sat": zeros},
        applied_torque_by_object={"sat": zeros},
        controller_runtime_ms_by_object={"sat": np.array([0.0])},
        controller_skipped_by_object={"sat": np.array([False])},
    )

    scores = compute_scores(log)

    assert scores.mean_runtime_ms["sat"] == 0.0
    assert scores.p95_runtime_ms["sat"] == 0.0
    assert scores.jitter_ms["sat"] == 0.0
    assert scores.controller_overruns["sat"] == 0
