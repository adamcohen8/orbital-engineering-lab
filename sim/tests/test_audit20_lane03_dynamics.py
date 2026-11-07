"""Boundary regressions found during the v0.36 dynamics audit."""

import numpy as np
import pytest

from sim.dynamics.orbit.eclipse import ellipsoid_limb_angle
from sim.dynamics.orbit.frames import FrameContext
from sim.dynamics.orbit.integrators import integrate_adaptive


def test_frame_context_rejects_nonfinite_time_inputs() -> None:
    with pytest.raises(ValueError, match="jd_utc_start must be finite"):
        FrameContext(jd_utc_start=np.nan)
    with pytest.raises(ValueError, match="dut1_s must be finite"):
        FrameContext(dut1_s=np.inf)
    with pytest.raises(ValueError, match="elapsed time must be finite"):
        FrameContext().at(np.nan)


def test_adaptive_integrator_rejects_fractional_limits_and_nonfinite_state() -> None:
    def derivative(_t, state):
        return np.ones_like(state)
    with pytest.raises(ValueError, match="max_substeps must be a positive integer"):
        integrate_adaptive(derivative, 0.0, np.zeros(1), 1.0, max_substeps=1.5)
    with pytest.raises(ValueError, match="state must be finite"):
        integrate_adaptive(derivative, 0.0, np.array([np.nan]), 1.0)


def test_ellipsoid_limb_angle_rejects_degenerate_geometry() -> None:
    with pytest.raises(ValueError, match="observer position norm"):
        ellipsoid_limb_angle(np.zeros(3), np.array([1.0, 0.0, 0.0]), 6378.0)
    with pytest.raises(ValueError, match="Sun position norm"):
        ellipsoid_limb_angle(np.array([7000.0, 0.0, 0.0]), np.zeros(3), 6378.0)


def test_adaptive_integrator_preserves_state_array_shape() -> None:
    initial = np.zeros((2, 2))

    def derivative(_time, state):
        return np.ones_like(state)

    result = integrate_adaptive(derivative, 0.0, initial, 1.0, method="dopri5")
    assert result.shape == initial.shape
    np.testing.assert_allclose(result, np.ones_like(initial), rtol=0.0, atol=1e-14)
