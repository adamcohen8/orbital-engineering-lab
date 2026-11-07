"""CR3BP adaptive dispatch, state/STM accuracy, and compatibility checks."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.cr3bp import (
    CR3BPSystem,
    cr3bp_derivative_physical,
    cr3bp_halo_seed_state_km_s,
    propagate_cr3bp_reference_stm,
    propagate_cr3bp_state,
)
from sim.dynamics.orbit.cr3bp_research import cr3bp_jacobi_constant
from sim.dynamics.orbit.integrators import rk4_step_state
from sim.dynamics.orbit.propagator import OrbitPropagator


def test_rk4_python_reference_preserves_state_and_stm():
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    expected = rk4_step_state(lambda t, x: cr3bp_derivative_physical(x), 0.0, state, 60.0)
    np.testing.assert_array_equal(propagate_cr3bp_state(state, 60.0, 0.0, numeric_backend="python"), expected)
    out, phi = propagate_cr3bp_reference_stm(state, np.eye(6), 60.0, 0.0, numeric_backend="python")
    np.testing.assert_array_equal(out, expected)
    assert phi.shape == (6, 6)
    assert propagate_cr3bp_state(state, 60.0, 0.0, return_info=True)[1] is None


@pytest.mark.parametrize("method", ["rkf78", "adaptive", "dopri5"])
def test_runtime_dispatch_and_accounting(method):
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    propagator = OrbitPropagator(model="cr3bp", integrator=method, adaptive_atol=1e-12, adaptive_rtol=1e-11)
    ctx = OrbitContext(mu_km3_s2=398600.0, mass_kg=100.0)
    thrust = np.array([1e-8, -2e-8, 3e-8])
    expected = propagate_cr3bp_state(
        state, 3600.0, 0.0, thrust, integrator=method, adaptive_atol=1e-12, adaptive_rtol=1e-11
    )
    actual = propagator.propagate(state, 3600.0, 0.0, thrust, {}, ctx)
    np.testing.assert_array_equal(actual, expected)
    assert propagator.last_adaptive_step_info.method == ("rkf78" if method == "adaptive" else method)
    first_count = propagator.adaptive_step_info.accepted_steps
    assert first_count > 1
    assert propagator._rkf78_h_next > 0
    propagator.propagate(actual, 3600.0, 3600.0, thrust, {}, ctx)
    assert propagator.adaptive_step_info.accepted_steps > first_count
    # Rewind restarts the adaptive step suggestion deterministically.
    repeated = propagator.propagate(state, 3600.0, 0.0, thrust, {}, ctx)
    np.testing.assert_array_equal(repeated, expected)


def test_rkf78_nrho_tolerance_convergence_against_dop853():
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    duration = 300000.0
    reference = solve_ivp(
        lambda t, x: cr3bp_derivative_physical(x),
        (0.0, duration),
        state,
        method="DOP853",
        rtol=3e-14,
        atol=1e-13,
        max_step=300.0,
    ).y[:, -1]
    errors, changes = [], []
    for tolerance in [1e-7, 1e-11]:
        result, info = propagate_cr3bp_state(
            state,
            duration,
            0.0,
            integrator="rkf78",
            adaptive_atol=tolerance * 0.01,
            adaptive_rtol=tolerance,
            return_info=True,
        )
        errors.append(np.linalg.norm(result[:3] - reference[:3]))
        changes.append(abs(cr3bp_jacobi_constant(result) - cr3bp_jacobi_constant(state)))
        assert info.accepted_steps > 1
    assert errors[1] < errors[0] / 100
    assert errors[1] < 0.001
    assert changes[1] < changes[0] / 100


def test_rkf78_forced_state_and_stm_finite_difference():
    state = cr3bp_halo_seed_state_km_s(family="l1_northern_large")
    duration = 21600.0
    opts = dict(integrator="rkf78", adaptive_atol=1e-13, adaptive_rtol=1e-12)
    command = np.array([1e-7, -2e-7, 3e-7])
    reference = solve_ivp(
        lambda t, x: cr3bp_derivative_physical(x, command_accel_km_s2=command),
        (0.0, duration),
        state,
        method="DOP853",
        rtol=3e-14,
        atol=1e-13,
    ).y[:, -1]
    result = propagate_cr3bp_state(state, duration, 0.0, command, **opts)
    np.testing.assert_allclose(result, reference, rtol=1e-11, atol=1e-8)
    reference, phi, info = propagate_cr3bp_reference_stm(state, np.eye(6), duration, 0.0, return_info=True, **opts)
    assert info.method == "rkf78"
    np.testing.assert_allclose(reference, propagate_cr3bp_state(state, duration, 0.0, **opts), rtol=1e-11, atol=1e-8)
    for axis in range(6):
        delta = np.zeros(6)
        delta[axis] = 0.001 if axis < 3 else 1e-7
        plus = propagate_cr3bp_state(state + delta, duration, 0.0, **opts)
        minus = propagate_cr3bp_state(state - delta, duration, 0.0, **opts)
        np.testing.assert_allclose(phi[:, axis], (plus - minus) / (2 * delta[axis]), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("function", [propagate_cr3bp_state, propagate_cr3bp_reference_stm])
def test_invalid_integrator_and_tolerances_rejected(function):
    state = cr3bp_halo_seed_state_km_s()
    args = (state, np.eye(6), 60.0, 0.0) if function is propagate_cr3bp_reference_stm else (state, 60.0, 0.0)
    with pytest.raises(ValueError, match="Unsupported"):
        function(*args, integrator="typo")
    for bad in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="tolerances"):
            function(*args, integrator="rkf78", adaptive_rtol=bad)


def test_cr3bp_rejects_invalid_state_time_and_system_before_backend_dispatch():
    state = cr3bp_halo_seed_state_km_s()
    with pytest.raises(ValueError, match="state_km_s"):
        propagate_cr3bp_state(np.array([np.nan] * 6), 60.0, 0.0)
    with pytest.raises(ValueError, match="dt_s"):
        propagate_cr3bp_state(state, -1.0, 0.0, integrator="rkf78")
    with pytest.raises(ValueError, match="t_s"):
        propagate_cr3bp_state(state, 60.0, np.inf)
    with pytest.raises(ValueError, match="stm"):
        propagate_cr3bp_reference_stm(state, np.full((6, 6), np.nan), 60.0, 0.0)
    with pytest.raises(ValueError, match="mu"):
        CR3BPSystem(mu=np.nan)


def test_configured_cr3bp_integrator_reaches_runtime():
    from sim.config import scenario_config_from_dict
    from sim.runtime_support import _build_orbit_propagator

    config = scenario_config_from_dict(
        {
            "scenario_name": "cr3bp_adaptive_config",
            "simulator": {
                "duration_s": 3600.0,
                "dt_s": 3600.0,
                "dynamics": {
                    "orbit": {
                        "model": "cr3bp",
                        "cr3bp_system": "earth_moon",
                        "integrator": "rkf78",
                        "adaptive_atol": 1e-12,
                        "adaptive_rtol": 1e-11,
                    }
                },
            },
        }
    )
    propagator = _build_orbit_propagator(config)
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    result = propagator.propagate(state, 3600.0, 0.0, np.zeros(3), {}, OrbitContext(mu_km3_s2=398600.0, mass_kg=100.0))
    assert np.isfinite(result).all()
    assert propagator.last_adaptive_step_info.method == "rkf78"
    assert propagator.adaptive_atol == 1e-12
    assert propagator.adaptive_rtol == 1e-11
