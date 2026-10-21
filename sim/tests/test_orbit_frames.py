from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sim.config.scenario_yaml import scenario_config_from_dict
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.epoch import resolved_jd_utc
from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    FRAME_MODEL_SIMPLE_GMST,
    _interp_eop,
    _nutation_iau1980_vallado_matrix,
    _precession_iau1976_matrix,
    _precession_nutation_matrix_approx,
    eci_to_ecef_harmonic,
    eci_to_ecef_rotation_context,
    eci_to_ecef_rotation_hpop_like,
    frame_context_from_mapping,
    normalize_frame_model,
    teme_to_eci_matrix_vallado_iau80,
    transform_position,
    transform_state,
)
from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin


def _write_minimal_eop(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "VERSION test",
                "NUM_OBSERVED_POINTS 2",
                "2024 01 01 60310.0 0.10 0.20 0.30 0 0 0 0 0 37",
                "2024 01 02 60311.0 0.11 0.21 0.31 0 0 0 0 0 37",
            ]
        ),
        encoding="utf-8",
    )


def _write_eop_with_dat_step(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "VERSION test",
                "NUM_OBSERVED_POINTS 3",
                "2024 01 01 60310.0 0.10 0.20 0.30 0 0 0 0 0 36",
                "2024 01 02 60311.0 0.11 0.21 0.31 0 0 0 0 0 37",
                "2024 01 03 60312.0 0.12 0.22 0.32 0 0 0 0 0 37",
            ]
        ),
        encoding="utf-8",
    )


def test_frame_model_aliases_are_canonical() -> None:
    assert normalize_frame_model("simple") == FRAME_MODEL_SIMPLE_GMST
    assert normalize_frame_model("simple_earth_rotation") == FRAME_MODEL_SIMPLE_GMST
    assert normalize_frame_model("hpop_like") == FRAME_MODEL_IAU76_80_EOP
    assert normalize_frame_model("iau76_80_eop") == FRAME_MODEL_IAU76_80_EOP


def test_teme_to_eci_rotation_cache_is_bounded_and_immutable() -> None:
    teme_to_eci_matrix_vallado_iau80.cache_clear()
    first = teme_to_eci_matrix_vallado_iau80(2460310.5)
    second = teme_to_eci_matrix_vallado_iau80(2460310.5)
    distinct = teme_to_eci_matrix_vallado_iau80(2460310.5, ddpsi_rad=1.0e-12)

    assert first is second
    assert first is not distinct
    assert first.flags.writeable is False
    assert teme_to_eci_matrix_vallado_iau80.cache_info().maxsize == 8192
    assert teme_to_eci_matrix_vallado_iau80.cache_info().hits == 1
    with pytest.raises(ValueError, match="read-only"):
        first[0, 0] = 0.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        first.setflags(write=True)
    np.testing.assert_array_equal(
        teme_to_eci_matrix_vallado_iau80(2460310.5),
        second,
    )
    teme_to_eci_matrix_vallado_iau80.cache_clear()


def test_eop_out_of_range_requires_explicit_hold_policy(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)

    with pytest.raises(ValueError, match="outside EOP coverage"):
        _interp_eop(60309.0, str(eop_path))

    held = _interp_eop(60309.0, str(eop_path), extrapolation="hold")
    assert held == pytest.approx((0.10, 0.20, 0.30, 37.0))


def test_eop_dat_step_fails_closed_for_interpolation_and_elapsed_time(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_eop_with_dat_step(eop_path)
    jd_start = 2400000.5 + 60310.0

    with pytest.raises(ValueError, match="DAT changes within the requested interpolation bracket"):
        _interp_eop(60310.5, str(eop_path))
    assert _interp_eop(60311.5, str(eop_path))[3] == pytest.approx(37.0)

    with pytest.raises(ValueError, match="crosses a DAT leap-second boundary"):
        eci_to_ecef_rotation_hpop_like(86400.0, jd_utc_start=jd_start, eop_path=str(eop_path))
    with pytest.raises(ValueError, match="crosses a DAT leap-second boundary"):
        frame_context_from_mapping(
            {"model": "iau76_80_eop", "eop_path": str(eop_path)}, jd_utc_start=jd_start
        ).at(86400.0)
    with pytest.raises(ValueError, match="crosses a DAT leap-second boundary"):
        resolved_jd_utc(
            {"jd_utc_start": jd_start, "spherical_harmonics_frame_model": "iau76_80_eop",
             "spherical_harmonics_eop_path": str(eop_path)}, 86400.0
        )


def test_onp_compiled_rotation_honors_eop_hold_policy(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    env = {
        "spherical_harmonics_frame_model": "iau76_80_eop",
        "spherical_harmonics_eop_path": str(eop_path),
        "jd_utc_start": 2460310.5,
    }
    kwargs = {
        "model_key": "spherical_harmonics_frame_model",
        "path_key": "spherical_harmonics_eop_path",
    }

    with pytest.raises(ValueError, match="outside EOP coverage"):
        OrbitPropagator._compiled_rotation(env, -86400.0, **kwargs)

    held = OrbitPropagator._compiled_rotation(
        {**env, "eop_extrapolation": "hold"},
        -86400.0,
        **kwargs,
    )
    accelerated = OrbitPropagator._accelerated_compiled_rotation(
        {**env, "eop_extrapolation": "hold"},
        -86400.0,
        **kwargs,
    )
    assert np.all(np.isfinite(held))
    np.testing.assert_allclose(accelerated, held, rtol=0.0, atol=1.0e-14)


def test_iau76_80_eop_alias_matches_legacy_hpop_like_rotation(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    r_eci = np.array([7000.0, 120.0, 30.0], dtype=float)

    legacy = eci_to_ecef_harmonic(
        r_eci,
        60.0,
        jd_utc_start=2460310.5,
        frame_model="hpop_like",
        eop_path=str(eop_path),
    )
    canonical = eci_to_ecef_harmonic(
        r_eci,
        60.0,
        jd_utc_start=2460310.5,
        frame_model="iau76_80_eop",
        eop_path=str(eop_path),
    )

    np.testing.assert_allclose(canonical, legacy, rtol=0.0, atol=0.0)


def test_direct_eop_harmonic_transform_requires_absolute_epoch(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)

    with pytest.raises(ValueError, match="require jd_utc_start"):
        eci_to_ecef_harmonic(
            np.array([7000.0, 0.0, 0.0]),
            0.0,
            frame_model="iau76_80_eop",
            eop_path=str(eop_path),
        )


def test_frame_context_records_eop_time_scale_provenance(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    ctx = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path)},
        jd_utc_start=2460310.5,
    )

    meta = ctx.metadata(sample_t_s=0.0)

    assert meta["model"] == "iau76_80_eop"
    assert meta["legacy_frame_model"] == "hpop_like"
    assert meta["time_scale_model"] == "eop_utc_ut1_tt"
    assert meta["eop_path"] == str(eop_path.resolve())
    assert len(meta["eop_table_sha256"]) == 64
    assert meta["dut1_s"] == 0.30
    assert meta["tt_minus_utc_s"] == 69.184
    assert meta["polar_motion_applied"] is True


def test_manual_eop_frame_context_affects_rotation_without_eop_path() -> None:
    base = frame_context_from_mapping({"model": "iau76_80_eop"}, jd_utc_start=2460310.5)
    manual = frame_context_from_mapping(
        {
            "model": "iau76_80_eop",
            "dut1_s": 0.30,
            "xp_arcsec": 0.10,
            "yp_arcsec": 0.20,
            "dat_s": 37.0,
        },
        jd_utc_start=2460310.5,
    )
    r_eci = np.array([7000.0, 120.0, 30.0], dtype=float)

    base_ecef = transform_position(r_eci, "eci", "ecef", t_s=120.0, context=base)
    manual_ecef = transform_position(r_eci, "eci", "ecef", t_s=120.0, context=manual)

    assert manual.metadata()["polar_motion_applied"] is True
    assert manual.metadata()["time_scale_model"] == "eop_utc_ut1_tt"
    assert np.linalg.norm(manual_ecef - base_ecef) > 1.0e-4


def test_eop_rotation_and_digest_refresh_when_file_at_same_path_changes(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    first = eci_to_ecef_rotation_hpop_like(
        0.0,
        jd_utc_start=2460310.5,
        eop_path=str(eop_path),
    )
    first_meta = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path)},
        jd_utc_start=2460310.5,
    ).metadata()
    eop_path.write_text(
        eop_path.read_text(encoding="utf-8").replace("0.30", "0.80"),
        encoding="utf-8",
    )
    second = eci_to_ecef_rotation_hpop_like(
        0.0,
        jd_utc_start=2460310.5,
        eop_path=str(eop_path),
    )
    second_meta = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path)},
        jd_utc_start=2460310.5,
    ).metadata()

    assert not np.array_equal(first, second)
    assert first_meta["eop_table_sha256"] != second_meta["eop_table_sha256"]


def test_eop_cache_refreshes_after_symlink_retarget(tmp_path: Path) -> None:
    first_file, second_file = tmp_path / "first.txt", tmp_path / "second.txt"
    _write_minimal_eop(first_file)
    _write_eop_with_dat_step(second_file)
    link = tmp_path / "current.txt"
    try:
        link.symlink_to(first_file)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this host")
    first = _interp_eop(60310.0, str(link))
    rotation = eci_to_ecef_rotation_hpop_like(0.0, jd_utc_start=2460310.5, eop_path=str(link))
    link.unlink()
    link.symlink_to(second_file)
    assert first[3] == 37.0
    assert _interp_eop(60310.0, str(link))[3] == 36.0
    assert not np.array_equal(
        rotation, eci_to_ecef_rotation_hpop_like(0.0, jd_utc_start=2460310.5, eop_path=str(link)),
    )
    with pytest.raises(ValueError, match="crosses a DAT leap-second boundary"):
        eci_to_ecef_rotation_hpop_like(86400.0, jd_utc_start=2460310.5, eop_path=str(link))


def test_eop_cache_observes_atomic_replacement_and_deletion(tmp_path: Path) -> None:
    eop_path = tmp_path / "eop.txt"
    _write_minimal_eop(eop_path)
    ctx = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path)},
        jd_utc_start=2460310.5,
    )
    original = eci_to_ecef_rotation_context(0.0, ctx)
    assert ctx.at(0.0).dut1_s == 0.30
    metadata = eop_path.stat()

    # Replacement retains size and mtime; the file identity must still refresh
    # interpolation and rotation caches at the same requested epoch.
    replacement = tmp_path / "replacement.txt"
    replacement.write_text(eop_path.read_text().replace("0.30", "0.80"))
    os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    replacement.replace(eop_path)
    assert _interp_eop(60310.0, str(eop_path))[2] == 0.80
    assert ctx.at(0.0).dut1_s == 0.80
    assert not np.array_equal(eci_to_ecef_rotation_context(0.0, ctx), original)

    eop_path.unlink()
    with pytest.raises(FileNotFoundError):
        _interp_eop(60310.0, str(eop_path))
    with pytest.raises(FileNotFoundError):
        ctx.at(0.0)
    with pytest.raises(FileNotFoundError):
        eci_to_ecef_rotation_context(0.0, ctx)


def test_relative_eop_cache_path_tracks_current_directory(tmp_path: Path, monkeypatch) -> None:
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    _write_minimal_eop(first_dir / "eop.txt")
    _write_eop_with_dat_step(second_dir / "eop.txt")
    monkeypatch.chdir(first_dir)
    assert _interp_eop(60310.0, "eop.txt")[3] == 37.0
    monkeypatch.chdir(second_dir)
    assert _interp_eop(60310.0, "eop.txt")[3] == 36.0


def test_eop_cache_preserves_parent_components_after_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    child = target / "child"
    child.mkdir(parents=True)
    _write_minimal_eop(tmp_path / "eop.txt")
    _write_eop_with_dat_step(target / "eop.txt")
    link = tmp_path / "link"
    try:
        link.symlink_to(child, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this host")
    # The filesystem follows the link before '..', selecting target/eop.txt.
    assert _interp_eop(60310.0, str(link / ".." / "eop.txt"))[3] == 36.0


def test_accelerated_iau_frame_rotation_is_numerically_equivalent() -> None:
    env = {
        "spherical_harmonics_frame_model": "iau76_80_eop",
        "jd_utc_start": 2460310.5,
        "dut1_s": 0.30,
        "xp_arcsec": 0.10,
        "yp_arcsec": 0.20,
        "dat_s": 37.0,
        "ddpsi_rad": 1.0e-8,
        "ddeps_rad": -2.0e-8,
    }
    kwargs = {
        "model_key": "spherical_harmonics_frame_model",
        "path_key": "spherical_harmonics_eop_path",
    }

    reference = OrbitPropagator._compiled_rotation(env, 43210.5, **kwargs)
    accelerated = OrbitPropagator._accelerated_compiled_rotation(env, 43210.5, **kwargs)

    np.testing.assert_allclose(accelerated, reference, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(accelerated @ accelerated.T, np.eye(3), rtol=0.0, atol=2.0e-15)


def test_iau1982_split_sidereal_time_matches_sofa_reference_and_keeps_microseconds() -> None:
    from sim.dynamics.orbit.frames import _gmst_iau1982_split

    # SOFA/ERFA t_sofa_c.c / t_erfa_c.c gmst82 reference, UT1 MJD 53736.
    actual = _gmst_iau1982_split(2453736.5, 0., 0.)
    assert actual == pytest.approx(1.754174981860675096, abs=1.e-13)
    # A one-microsecond interval is below the scalar JD resolution here.
    later = _gmst_iau1982_split(2453736.5, 1.e-6, 0.)
    assert later - actual == pytest.approx(7.2921158553e-11, abs=1.e-14)
    assert _gmst_iau1982_split(2453736.5, 86400., .2) == pytest.approx(
        _gmst_iau1982_split(2453737.5, 0., .2), abs=1.e-13)


def test_complementary_equinox_terms_start_at_1997_utc_midnight() -> None:
    from sim.dynamics.orbit.frames import _equation_of_equinoxes_iau1994

    # At the convention boundary UTC=TAI-30s and TT=TAI+32.184s.
    start_tt = 2450506.5 + 62.184/86400.
    assert _equation_of_equinoxes_iau1994(start_tt-15./86400., 0.) == 0.
    assert _equation_of_equinoxes_iau1994(start_tt+15./86400., 0.) != 0.


def test_eop_frame_context_without_epoch_does_not_claim_or_apply_eop(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    ctx = frame_context_from_mapping({"model": "iau76_80_eop", "eop_path": str(eop_path)})

    assert ctx.metadata()["polar_motion_applied"] is False
    with pytest.raises(ValueError, match="requires simulator.initial_jd_utc"):
        eci_to_ecef_rotation_context(0.0, ctx)


def test_scenario_frames_eop_path_requires_initial_epoch(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)

    with pytest.raises(ValueError, match="simulator.frames EOP settings require simulator.initial_jd_utc"):
        scenario_config_from_dict(
            {
                "scenario_name": "missing_frame_epoch",
                "simulator": {
                    "frames": {
                        "model": "iau76_80_eop",
                        "eop_path": str(eop_path),
                    }
                },
                "objects": {},
            }
        )


def test_scenario_frames_manual_eop_requires_initial_epoch() -> None:
    with pytest.raises(ValueError, match="simulator.frames EOP settings require simulator.initial_jd_utc"):
        scenario_config_from_dict(
            {
                "scenario_name": "missing_manual_frame_epoch",
                "simulator": {
                    "frames": {
                        "model": "iau76_80_eop",
                        "dut1_s": 0.2,
                    }
                },
                "objects": {},
            }
        )


def test_scenario_iau_frame_model_without_eop_does_not_require_initial_epoch() -> None:
    cfg = scenario_config_from_dict(
        {
            "scenario_name": "iau_frame_no_eop",
            "simulator": {"frames": {"model": "iau76_80_eop"}},
            "objects": {},
        }
    )

    assert cfg.simulator.frames.model == "iau76_80_eop"
    assert cfg.simulator.initial_jd_utc is None


def test_frame_context_nutation_corrections_affect_rotation_and_provenance(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    base = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path)},
        jd_utc_start=2460310.5,
    )
    corrected = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path), "ddpsi_rad": 1.0e-6, "ddeps_rad": -2.0e-6},
        jd_utc_start=2460310.5,
    )

    r_eci = np.array([7000.0, 120.0, 30.0], dtype=float)
    base_ecef = transform_position(r_eci, "eci", "ecef", t_s=120.0, context=base)
    corrected_ecef = transform_position(r_eci, "eci", "ecef", t_s=120.0, context=corrected)

    assert corrected.metadata()["nutation_corrections_applied"] is True
    assert np.linalg.norm(corrected_ecef - base_ecef) > 1.0e-4


def test_iau76_80_celestial_factor_inverts_tod_nutation() -> None:
    jd_tt = 2460310.5 + 69.184 / 86400.0
    expected = _nutation_iau1980_vallado_matrix(jd_tt)[4].T @ _precession_iau1976_matrix(jd_tt)
    actual = _precession_nutation_matrix_approx(jd_tt)[0]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)


def test_context_transform_matches_canonical_harmonic_helper(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    ctx = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path)},
        jd_utc_start=2460310.5,
    )
    r_eci = np.array([7000.0, 120.0, 30.0], dtype=float)

    from_context = transform_position(r_eci, "eci", "ecef", t_s=120.0, context=ctx)
    direct = eci_to_ecef_harmonic(
        r_eci,
        120.0,
        jd_utc_start=2460310.5,
        frame_model="iau76_80_eop",
        eop_path=str(eop_path),
    )

    np.testing.assert_allclose(from_context, direct, rtol=0.0, atol=0.0)


def test_transform_state_applies_rotating_frame_velocity_term() -> None:
    ctx = frame_context_from_mapping({"model": "simple_gmst"})
    r_eci = np.array([7000.0, 0.0, 0.0], dtype=float)
    v_eci = np.array([0.0, 7.5, 0.0], dtype=float)

    r_ecef, v_ecef = transform_state(r_eci, v_eci, "eci", "ecef", t_s=0.0, context=ctx)
    expected_v_ecef = np.array([0.0, 7.5 - EARTH_ROT_RATE_RAD_S * 7000.0, 0.0], dtype=float)
    roundtrip_r, roundtrip_v = transform_state(r_ecef, v_ecef, "ecef", "eci", t_s=0.0, context=ctx)

    np.testing.assert_allclose(r_ecef, r_eci, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(v_ecef, expected_v_ecef, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(roundtrip_r, r_eci, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(roundtrip_v, v_eci, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("elapsed_s", [0.0, 10.0, 420.0, 86400.0])
def test_simple_frame_station_speed_has_no_julian_date_quantization(elapsed_s: float) -> None:
    from sim.dynamics.orbit.frames import PreparedFrameEvaluator, eci_to_ecef_rotation_derivative_context

    context = frame_context_from_mapping({"model": "simple_gmst"}, jd_utc_start=2460000.5)
    station_ecef = np.array([6378.137, 0.0, 0.0])
    position, velocity = transform_state(station_ecef, np.zeros(3), "ecef", "eci",
                                         t_s=elapsed_s, context=context)
    # GMST polynomial speed at JD 2460000.5, including its secular terms.
    # Variation over this one-day fixture is below 1e-15 km/s.
    expected_speed = 0.46510113945651635
    assert np.linalg.norm(velocity) == pytest.approx(expected_speed, abs=1e-12)
    assert abs(float(position @ velocity)) < 1e-12
    evaluator = PreparedFrameEvaluator(context)
    np.testing.assert_array_equal(evaluator.derivative(elapsed_s),
                                eci_to_ecef_rotation_derivative_context(elapsed_s, context))


def test_manual_eop_fields_propagate_to_drag_relative_velocity() -> None:
    class _Ctx:
        mass_kg = 100.0
        cd = 2.2
        area_m2 = 1.0

    x_eci = np.array([7000.0, 120.0, 30.0, 0.0, 7.5, 0.0], dtype=float)
    base_env = {
        "density_kg_m3": 1.0e-12,
        "drag_frame_model": "hpop_like",
        "jd_utc_start": 2460310.5,
    }
    manual_env = {
        **base_env,
        "dut1_s": 30.0,
        "xp_arcsec": 100.0,
        "yp_arcsec": 200.0,
        "dat_s": 37.0,
        "ddpsi_rad": 1.0e-3,
        "ddeps_rad": -2.0e-3,
    }

    base_drag = drag_plugin(120.0, x_eci, base_env, _Ctx())
    manual_drag = drag_plugin(120.0, x_eci, manual_env, _Ctx())

    assert np.linalg.norm(manual_drag - base_drag) > 1.0e-14
