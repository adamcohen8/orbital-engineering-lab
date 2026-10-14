"""Focused propagation audit regressions for the 2026-10-09 review."""

import numpy as np
import pytest

from sim.dynamics.orbit.integrators import integrate_adaptive, integrate_rkf78_hpop


def test_adaptive_integrator_validates_method_and_tolerances_for_zero_duration() -> None:
    state = np.array([1.0], dtype=float)

    with pytest.raises(ValueError, match="Unknown adaptive integrator method"):
        integrate_adaptive(lambda _t, value: value, 0.0, state, 0.0, method="typo")

    with pytest.raises(ValueError, match="tolerances must be finite and non-negative"):
        integrate_adaptive(lambda _t, value: value, 0.0, state, 0.0, method="rkf78", rtol=-1.0)
    assert np.allclose(
        integrate_adaptive(lambda _t, value: value, 0.0, state, 0.0, method="rkf78", rtol=0.0),
        state,
    )
    with pytest.raises(ValueError, match="HPOP RKF78 tolerance must be positive and finite"):
        integrate_rkf78_hpop(lambda _t, value: value, 0.0, state, 0.0, tolerance=0.0)


def test_adaptive_integrator_rejects_nonfinite_nonzero_tolerance_and_keeps_one_sided_modes() -> None:
    state = np.array([1.0], dtype=float)

    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="tolerances must be finite and non-negative"):
            integrate_adaptive(lambda _t, value: value, 0.0, state, 0.1, atol=bad)

    absolute_only = integrate_adaptive(
        lambda _t, value: value,
        0.0,
        state,
        0.1,
        atol=0.0,
        rtol=1.0e-8,
    )
    relative_only = integrate_adaptive(
        lambda _t, value: value,
        0.0,
        state,
        0.1,
        atol=1.0e-8,
        rtol=0.0,
    )
    assert np.all(np.isfinite(absolute_only)) and float(absolute_only[0]) > 1.0
    assert np.all(np.isfinite(relative_only)) and float(relative_only[0]) > 1.0
