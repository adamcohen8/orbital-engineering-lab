"""Independent frame geometry, eclipse and native/callback ECOM2 parity."""

import numpy as np
import pytest

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator, third_body_sun_plugin
from sim.dynamics.orbit.rust_force_plan import make_plan
from sim.pro_perturbations.ecom import ECOM2Radiation, ecom2_acceleration


def test_axes_phase_and_inverse_square():
    x = np.array([28000.0, 0.0, 0.0, 0.0, 3.8, 0.0])
    sun = np.array([28000.0, 149597870.7, 0.0])
    # D=+Y, Y=-Z, B=-X; delta-u=-pi/2.
    np.testing.assert_allclose(ecom2_acceleration(x, sun, [1, 2, 3, 4, 5, 6, 7]), [4, -3, -2], atol=1e-14)
    sun2 = np.array([28000.0, 2 * 149597870.7, 0.0])
    np.testing.assert_allclose(
        ecom2_acceleration(x, sun2, [1, 2, 3, 4, 5, 6, 7]), np.array([4, -3, -2]) / 4, atol=1e-14
    )
    assert np.linalg.norm(ecom2_acceleration(x, [-149597870.7, 1000.0, 0.0], [1] * 7)) == 0


@pytest.mark.parametrize(
    "state,sun,c",
    [
        ([0] * 6, [1, 2, 3], [0] * 7),
        ([28000, 0, 0, 0, 0, 0], [1, 2, 3], [0] * 7),
        ([28000, 0, 0, 0, 3.8, 0], [28000, 0, 1e8], [0] * 7),
        ([28000, 0, 0, 0, 3.8, 0], [1e8, 2e7, 0], [0] * 6),
        ([28000, 0, 0, 0, 3.8, 0], [1e8, 2e7, 0], [np.nan] * 7),
    ],
)
def test_invalid_geometry_fails(state, sun, c):
    with pytest.raises(ValueError):
        ecom2_acceleration(state, sun, c)


def test_native_forces_and_trajectories_match_python_reference():
    c = (-1.2e-10, 2e-12, -3e-12, 4e-12, 5e-12, -6e-12, 7e-12)
    env = dict(sun_pos_eci_km=np.array([1.4e8, 2e7, -1e7]))
    ctx = OrbitContext(398600.4418, 1.0)
    for x in [np.array([26000.0, 1000, 1000, 0, 2.7, 2.4]), np.array([-28000.0, 1000, 0, 0, -3, 1.0])]:
        prop = OrbitPropagator(plugins=[third_body_sun_plugin, ECOM2Radiation(c)])
        ref = OrbitPropagator(plugins=[third_body_sun_plugin, ECOM2Radiation(c)])
        *_, stage, context = make_plan(prop, x, 0.0, dict(env), ctx)
        actual = np.array(context.acceleration(x.tolist(), stage(0.0, x), [0.0, 0.0, 0.0]))
        expected = ref.acceleration_at(t_s=0.0, x_eci=x, command_accel_eci_km_s2=np.zeros(3), env=dict(env), ctx=ctx)
        np.testing.assert_allclose(actual, expected, atol=2e-19, rtol=0)
        a, b = x.copy(), x.copy()
        for i in range(6):
            a = prop.propagate(a, 30.0, 30.0 * i, np.zeros(3), dict(env), ctx)
            b = ref.propagate(b, 30.0, 30.0 * i, np.zeros(3), dict(env, _rust_precision_force_plan_disabled=True), ctx)
        np.testing.assert_allclose(a, b, atol=2e-11, rtol=0)


def test_old_binary_declines_descriptor(monkeypatch):
    from sim.dynamics.orbit.rust_precision_force_plan import precision_code

    monkeypatch.setattr("sim.rust_orbit_backend._extension", lambda: object())
    assert precision_code(ECOM2Radiation()) is None


def test_standalone_native_descriptor_resolves_sun_without_builtin_body_force():
    x=np.array([26000.,1000.,1000.,-.2,2.7,2.4])
    c=(-1.2e-10,2e-12,-3e-12,4e-12,5e-12,-6e-12,7e-12)
    ctx=OrbitContext(398600.4418,1.)
    env=dict(jd_utc_start=2460364.5,ephemeris_mode='analytic_enhanced')
    prop=OrbitPropagator(plugins=[ECOM2Radiation(c)])
    ref=OrbitPropagator(plugins=[ECOM2Radiation(c)])
    a=prop.propagate(x,60.,0.,np.zeros(3),dict(env),ctx)
    b=ref.propagate(x,60.,0.,np.zeros(3),dict(env,_rust_precision_force_plan_disabled=True),ctx)
    np.testing.assert_allclose(a,b,atol=2e-11,rtol=0)


def test_tide_coefficient_cache_keys_conventions_bodies_and_gradients():
    from sim.dynamics.orbit.frames import FrameContext
    from sim.pro_perturbations.solid_earth_tides import SolidEarthTides
    frame = FrameContext(model='iau76_80_eop',jd_utc_start=2460364.5,dut1_s=.05,
                         dat_s=37.,xp_arcsec=.1,yp_arcsec=.2)
    for zero,pole,scale in [(False,True,1.),(True,True,1.),(False,False,1.),(False,True,1.01),(False,True,1.)]:
        plugin = SolidEarthTides(frame,'zero_tide' if zero else 'tide_free',pole_tide=pole)
        env = dict(jd_utc_start=frame.jd_utc_start,sun_pos_eci_km=np.array([1.4e8,2e7,-1e7])*scale,
                   moon_pos_eci_km=np.array([3.7e5,-8e4,3e4])/scale)
        ctx = OrbitContext(398600.4418*scale,1.)
        for radius in [26000.,27000.]:
            x = np.array([radius,1000.,1000.,-.2,2.7,2.4])
            prop = OrbitPropagator(plugins=[plugin])
            *_,stage,context = make_plan(prop,x,0.,dict(env),ctx)
            actual = context.acceleration(x.tolist(),stage(0.,x),[0.,0.,0.])
            expected = prop.acceleration_at(t_s=0.,x_eci=x,command_accel_eci_km_s2=np.zeros(3),env=dict(env),ctx=ctx)
            np.testing.assert_allclose(actual,expected,atol=2e-18,rtol=0)
