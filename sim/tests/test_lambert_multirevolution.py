import math

import numpy as np
import pytest

from sim.dynamics.orbit import solve_multi_revolution_lambert
from sim.dynamics.orbit.two_body import propagate_two_body_rk4


@pytest.mark.parametrize("turns,short", [(1, True), (1, False), (3, True), (3, False)])
def test_each_branch_arrives_with_requested_winding(turns, short):
    start = np.array([1.0, 0.0, 0.0])
    end = np.array([0.0, 0.8, 0.6])
    duration = math.tau * (turns + (0.25 if short else 0.75))
    result = solve_multi_revolution_lambert(
        start, end, duration, revolutions=turns, short_way=short, mu_km3_s2=1.0, tolerance_s=1e-12
    )
    left, right = result.branches
    assert (left.branch, right.branch) == ("lower_phase", "higher_phase")
    assert left.phase_rad < right.phase_rad and result.minimum_time_s < duration
    expected_circular_velocity = end if short else -end
    assert min(np.linalg.norm(b.v1_km_s - expected_circular_velocity) for b in result.branches) < 1e-12
    for branch in result.branches:
        state = np.r_[start, branch.v1_km_s]
        angles = [0.0]
        steps = 8192 * turns
        for _ in range(steps):
            state = propagate_two_body_rk4(state, duration / steps, 1.0, np.zeros(3))
            angles.append(math.atan2(state[:3] @ end, state[0]))
        np.testing.assert_allclose(state[:3], end, rtol=0, atol=2e-7)
        np.testing.assert_allclose(state[3:], branch.v2_km_s, rtol=0, atol=2e-7)
        winding = np.unwrap(angles)[-1] / math.tau
        assert winding == pytest.approx(turns + 0.25 if short else -turns - 0.75, abs=2e-7)


def test_minimum_and_coalescence_band():
    def solve(time):
        return solve_multi_revolution_lambert(
            [3, 1, 2], [0, 2, -1], time, revolutions=2, mu_km3_s2=2.0, tolerance_s=1e-9
        )

    lower = solve(1)
    assert lower.branches == ()
    minimum = lower.minimum_time_s
    for delta in (-0.5e-9, 0, 0.5e-9):
        (tangent,) = solve(minimum + delta).branches
        assert tangent.branch == "minimum" and abs(tangent.residual_s) <= 1e-9
    assert not solve(minimum - 1e-7).branches
    a, b = solve(minimum + 1e-7).branches
    assert a.phase_rad < tangent.phase_rad < b.phase_rad


@pytest.mark.parametrize(
    "changes",
    [
        {"revolutions": 0},
        {"revolutions": True},
        {"revolutions": 1.5},
        {"r2_km": [-2, 0, 0]},
        {"r1_km": [0, 0, 0]},
        {"time_of_flight_s": float("nan")},
        {"short_way": "false"},
        {"tolerance_s": 0},
        {"mu_km3_s2": -1},
    ],
)
def test_invalid_inputs(changes):
    args = dict(r1_km=[1, 0, 0], r2_km=[0, 1, 0], time_of_flight_s=12, revolutions=1, mu_km3_s2=1)
    with pytest.raises(ValueError):
        solve_multi_revolution_lambert(**{**args, **changes})
