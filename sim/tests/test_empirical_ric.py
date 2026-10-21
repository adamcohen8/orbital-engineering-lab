"""Stage-dependent empirical force geometry and native parity."""

import numpy as np
import pytest

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.pro_perturbations.empirical import EmpiricalRICAcceleration, empirical_ric_acceleration


def test_ric_axes_are_radial_transverse_and_right_handed():
    x = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]
    np.testing.assert_array_equal(empirical_ric_acceleration(x, [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])
    # Transverse excludes the radial velocity component.
    np.testing.assert_array_equal(
        empirical_ric_acceleration([7000.0, 0.0, 0.0, 2.0, 7.5, 0.0], [0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]
    )
    np.testing.assert_array_equal(
        empirical_ric_acceleration([0.0, 7000.0, 0.0, -7.5, 0.0, 0.0], [1.0, 2.0, 3.0]), [-2.0, 1.0, 3.0]
    )


@pytest.mark.parametrize(
    "state,coefficients",
    [
        ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([7000.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([7000.0, 0.0, 0.0, 0.0, 7.0, 0.0], [float("nan"), 0.0, 0.0]),
        ([1.0, 2.0], [0.0, 0.0, 0.0]),
    ],
)
def test_singular_or_invalid_empirical_frame_fails(state, coefficients):
    with pytest.raises(ValueError):
        empirical_ric_acceleration(state, coefficients)


def test_native_empirical_force_matches_reference_at_every_stage():
    native = pytest.importorskip("oel_rust_orbit")
    assert getattr(native, "ONP_EMPIRICAL_RIC_VERSION", 0) == 1
    model = EmpiricalRICAcceleration((1.0e-9, -2.0e-9, 3.0e-9))
    prop = OrbitPropagator(numeric_backend="rust", plugins=[model])
    reference = OrbitPropagator(numeric_backend="rust", plugins=[model])
    ctx = OrbitContext(398600.4415, 100.0)
    x = np.array([12200.0, 300.0, 800.0, -0.1, 5.7, 0.5])
    y = x.copy()
    for i in range(20):
        x = prop.propagate(x, 10.0, i * 10.0, np.zeros(3), {}, ctx)
        y = reference.propagate(y, 10.0, i * 10.0, np.zeros(3), {"_rust_precision_force_plan_disabled": True}, ctx)
    assert prop.last_numeric_path == "rust_native_force_plan"
    assert reference.last_numeric_path == "rust_python_force_callback"
    np.testing.assert_allclose(x, y, rtol=0.0, atol=2.0e-11)


def test_old_native_does_not_select_new_descriptor(monkeypatch):
    from sim.dynamics.orbit.rust_precision_force_plan import precision_code

    monkeypatch.setattr("sim.rust_orbit_backend._extension", lambda: object())
    assert precision_code(EmpiricalRICAcceleration()) is None


def test_empirical_harmonic_phase_and_native_stage_times():
    model = EmpiricalRICAcceleration(
        (1.0e-10, 0.0, 0.0),
        cosine_km_s2=(0.0, 2.0e-10, 0.0),
        sine_km_s2=(0.0, 0.0, 3.0e-10),
        angular_frequency_rad_s=0.01,
        reference_time_s=10.0,
    )
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    np.testing.assert_allclose(model(10.0, state, {}, None), [1.0e-10, 2.0e-10, 0.0], atol=1.0e-25, rtol=0.0)
    np.testing.assert_allclose(
        model(10.0 + np.pi / 0.02, state, {}, None), [1.0e-10, 0.0, 3.0e-10], atol=1.0e-25, rtol=0.0
    )
    extension = pytest.importorskip("oel_rust_orbit")
    assert getattr(extension, "ONP_EMPIRICAL_RIC_HARMONIC_VERSION", 0) == 1
    a = OrbitPropagator(numeric_backend="rust", plugins=[model])
    b = OrbitPropagator(numeric_backend="rust", plugins=[model])
    ctx = OrbitContext(398600.4415, 100.0)
    x, y = state.copy(), state.copy()
    for i in range(20):
        x = a.propagate(x, 10.0, i * 10.0, np.zeros(3), {}, ctx)
        y = b.propagate(y, 10.0, i * 10.0, np.zeros(3), {"_rust_precision_force_plan_disabled": True}, ctx)
    np.testing.assert_allclose(x, y, rtol=0.0, atol=2.0e-11)
