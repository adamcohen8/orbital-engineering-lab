"""Physical frame corrections and opt-in oblate eclipse geometry."""

import numpy as np
import pytest

from sim.dynamics.orbit.eclipse import ellipsoid_limb_angle, srp_shadow_factor
from sim.dynamics.orbit.frames import (
    FrameContext,
    PreparedFrameEvaluator,
    _interp_eop,
    _interp_eop_nutation,
    _load_eop_stage_table,
    eci_to_ecef_rotation_context,
)


def table(path, dp=0.04, de=-0.02):
    path.write_text(
        "NUM_OBSERVED_POINTS 2\n"
        f"2022 03 31 59669 0.1 0.2 -0.1 0 {dp} {de} 0 0 37\n"
        f"2022 04 01 59670 0.2 0.3 -0.2 0 {dp * 3} {de * 3} 0 0 37\n"
    )


def test_eop_nutation_interpolates_arcseconds_and_preserves_existing_contract(tmp_path):
    p = tmp_path / "eop.txt"
    table(p)
    assert len(_interp_eop(59669.5, str(p))) == 4
    np.testing.assert_allclose(_interp_eop_nutation(59669.5, str(p)), np.deg2rad([0.08, -0.04]) / 3600, rtol=1e-15)
    with pytest.raises(ValueError, match="outside EOP coverage"):
        _interp_eop_nutation(59668, str(p))
    np.testing.assert_allclose(
        _interp_eop_nutation(59668, str(p), extrapolation="hold"), np.deg2rad([0.04, -0.02]) / 3600
    )


def test_file_corrections_equal_manual_offsets_and_invalidate_prepared_cache(tmp_path):
    p = tmp_path / "eop.txt"
    table(p)
    ctx = FrameContext(model="iau76_80_eop", jd_utc_start=2459669.5, eop_path=str(p), ddpsi_rad=1e-9)
    xp, yp, dut, dat = _interp_eop(59669.5, str(p))
    dp, de = _interp_eop_nutation(59669.5, str(p))
    manual = FrameContext(
        model="iau76_80_eop",
        jd_utc_start=2459669.5,
        xp_arcsec=xp,
        yp_arcsec=yp,
        dut1_s=dut,
        dat_s=dat,
        ddpsi_rad=dp + 1e-9,
        ddeps_rad=de,
    )
    np.testing.assert_array_equal(eci_to_ecef_rotation_context(43200, ctx), eci_to_ecef_rotation_context(43200, manual))
    prepared = PreparedFrameEvaluator(ctx)
    old = prepared.rotation(43200).copy()
    table(p, 0.08, -0.04)
    prepared.refresh()
    assert np.max(abs(old - prepared.rotation(43200))) > 1e-8
    metadata = ctx.metadata(sample_t_s=43200)
    assert metadata["nutation_corrections_applied"]
    assert metadata["effective_ddpsi_rad"] == pytest.approx(2 * dp + 1e-9)
    # Sampling metadata must not add file corrections a second time.
    assert ctx.at(43200).metadata(sample_t_s=43200)["effective_ddpsi_rad"] == metadata["effective_ddpsi_rad"]


def test_prepared_native_rotation_uses_supplied_nutation(tmp_path):
    native = pytest.importorskip("oel_rust_orbit")
    if not hasattr(native.ONPStageContext, "supports_eop_nutation"):
        pytest.skip("requires rebuilt EOP correction candidate")
    from sim.dynamics.orbit.propagator import OrbitPropagator
    from sim.dynamics.orbit.rust_force_plan import _configure_native_environment_context

    p = tmp_path / "eop.txt"
    table(p)
    env = {"jd_utc_start": 2459669.5}
    owner = _configure_native_environment_context(OrbitPropagator(), env, need_de440=False)
    ctx = FrameContext(model="iau76_80_eop", jd_utc_start=2459669.5, eop_path=str(p))
    prepared = PreparedFrameEvaluator(ctx, owner)
    np.testing.assert_allclose(prepared.rotation(43200), eci_to_ecef_rotation_context(43200, ctx), atol=3e-15, rtol=0)
    assert len(_load_eop_stage_table(str(p))) == 7


@pytest.mark.parametrize(
    "r,sun",
    [
        (np.array([7000.0, 300.0, 900.0]), np.array([1.5e8, 2e7, 3e7])),
        (np.array([0.0, 0.0, 7000.0]), np.array([0.0, 0.0, 1.5e8])),
    ],
)
def test_limb_reduces_to_sphere_and_handles_collinear_geometry(r, sun):
    angle = ellipsoid_limb_angle(r, sun, 6378.137, flattening=0)
    assert angle == pytest.approx(np.arcsin(6378.137 / np.linalg.norm(r)), abs=2e-15)
    assert np.isfinite(ellipsoid_limb_angle(r, sun, 6378.137))


def test_oblate_limb_changes_penumbra_without_changing_spherical_option():
    radius = 7000.0
    alpha = np.arcsin(6378.137 / radius)
    r = np.array([-radius * np.cos(alpha), 0.0, radius * np.sin(alpha)])
    env = {"sun_pos_eci_km": np.array([1.5e8, 0.0, 0.0]), "srp_shadow_model": "conical"}
    sphere = srp_shadow_factor(r, 0, env)
    oblate = srp_shadow_factor(r, 0, {**env, "srp_shadow_model": "conical_ellipsoid"})
    assert 0 < sphere < 1 and sphere < oblate <= 1
    assert srp_shadow_factor(r, 0, env) == sphere


@pytest.mark.parametrize("harmonics", [False, True])
def test_native_stage_interpolates_eop_and_prepares_shadow_rotation(tmp_path, harmonics):
    native = pytest.importorskip("oel_rust_orbit")
    if not hasattr(native.ONPStageContext, "supports_eop_nutation"):
        pytest.skip("requires rebuilt EOP correction candidate")
    from sim.dynamics.orbit.frames import _load_nut80_table
    from sim.dynamics.orbit.rust_stage_context import _frame_spec

    p = tmp_path / "eop.txt"
    table(p)
    ctx = FrameContext(model="iau76_80_eop", jd_utc_start=2459669.5, eop_path=str(p))
    coefficients, terms = _load_nut80_table()
    owner = native.ONPEnvironmentContext(coefficients.ravel().tolist(), terms.ravel().tolist())
    simple = (0, None, [0.0] * 7, [], "")
    stage = native.ONPStageContext(
        owner,
        [_frame_spec(ctx)] + [simple] * 4,
        [1] if harmonics else [3],
        2,
        [0.0] * 6,
        False,
        None,
        None,
        4,
        None,
        None,
        None,
        [],
        lambda *_: pytest.fail("fallback"),
        shadow_frame_inputs=True,
    )
    for t in [0.0, 37.25, 43200.0, 86400.0]:
        rotation = np.array(stage(t, [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])[:9]).reshape(3, 3)
        np.testing.assert_allclose(rotation, eci_to_ecef_rotation_context(t, ctx), atol=3e-15, rtol=0)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_ellipsoid_srp_plugin_and_native_plan_agree_in_penumbra(backend):
    native = pytest.importorskip("oel_rust_orbit")
    if not hasattr(native.ONPStageContext, "supports_shadow_frame_inputs"):
        pytest.skip("requires rebuilt oblate shadow candidate")
    from sim.dynamics.orbit.accelerations import OrbitContext
    from sim.dynamics.orbit.propagator import OrbitPropagator, srp_plugin
    from sim.dynamics.orbit.rust_force_plan import make_plan

    prop = OrbitPropagator(numeric_backend="rust", plugins=[srp_plugin])
    alpha = np.arcsin(6378.137 / 7000.0) - 0.001
    state = np.array([-7000 * np.cos(alpha), 0.0, 7000 * np.sin(alpha), 0.0, 7.5, 0.0])
    sun = np.array([1.5e8, 0.0, 0.0])
    env = {"srp_shadow_model": "conical_ellipsoid", "sun_pos_eci_km": sun, "_rust_numeric_backend": backend}
    ctx = OrbitContext(398600.4415, 300.0, 4.0, 2.2, 1.2)
    *_, stage, context = make_plan(prop, state, 0.0, env, ctx)
    actual = np.array(context.acceleration(state.tolist(), stage(0.0, state), [0.0, 0.0, 0.0]))
    central = -ctx.mu_km3_s2 * state[:3] / np.linalg.norm(state[:3]) ** 3
    expected = central + srp_plugin(0.0, state, env, ctx)
    np.testing.assert_allclose(actual, expected, atol=2e-18, rtol=0)
