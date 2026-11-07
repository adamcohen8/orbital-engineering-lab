from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from sim.data_acquisition.contracts import AcquisitionError, digest
from sim.data_acquisition.inspection import inspect_bytes
from sim.dynamics.orbit.eop import encode_eop_all, parse_celestrak_eop
from sim.dynamics.orbit.frames import _load_eop_table
from sim.dynamics.orbit.space_weather_data import encode_msis_table, parse_celestrak_weather
from sim.tests.data_acquisition_helpers import GRAVITY, eop, service, weather


def registered(tmp_path, raw, kind, format):
    svc = service(tmp_path)
    source = tmp_path / "source"
    source.write_bytes(raw)
    return svc, svc.register(source, kind=kind, format=format)


def gravity_requirements(**overrides):
    return {"consumer": "earth-gravity", "degree": 2, "order": 0,
            "normalization": "fully_normalized", "tide_system": "tide_free", **overrides}


def environmental_requirements(consumer="runtime-frames", **overrides):
    return {"consumer": consumer, "start": "2024-01-10T00:00:00Z", "stop": "2024-01-10T12:00:00Z", **overrides}


@pytest.mark.parametrize("raw", [
    GRAVITY.replace(b"fully_normalized", b"schmidt"), GRAVITY.replace(b"tide_free", b"unknown"),
    GRAVITY.replace(b"radius 6378136.3\n", b""), GRAVITY.replace(b"3.986004415e14", b"NaN"),
    GRAVITY + b"gfc 2 0 1 0\n", GRAVITY.replace(b"-4.84165143790815e-4", b"NaN"), b"geoid-height-grid",
])
def test_invalid_gravity_conventions_are_never_qualified(raw):
    value = inspect_bytes(raw, kind="gravity", format="icgem")
    assert not value["compatible"] and not value["parsed"]


def test_temporal_terms_are_preserved_as_a_blocking_finding(tmp_path):
    svc, artifact = registered(tmp_path, GRAVITY + b"gfct 2 0 1 0\n", "gravity", "icgem")
    assert artifact["inspection"]["temporal_terms"] == ["gfct"]
    with pytest.raises(AcquisitionError):
        svc.check(artifact["artifact_id"], gravity_requirements())


def test_only_an_exact_pinned_legacy_model_receives_missing_normalization_context(monkeypatch):
    from dataclasses import replace

    from sim.dynamics.orbit.gravity_assets import REAL_MODEL_DOWNLOADS
    raw = GRAVITY.replace(b"norm fully_normalized\n", b"")
    assert not inspect_bytes(raw, kind="gravity", format="icgem")["parsed"]
    monkeypatch.setitem(REAL_MODEL_DOWNLOADS, "EGM96", replace(REAL_MODEL_DOWNLOADS["EGM96"], sha256=digest(raw)))
    value = inspect_bytes(raw, kind="gravity", format="icgem")
    assert value["compatible"] and "digest-pinned" in value["source_profile_context"]["norm"]
    assert not inspect_bytes(raw + b"\n", kind="gravity", format="icgem")["parsed"]


@pytest.mark.parametrize("requirements", [gravity_requirements(degree=3), gravity_requirements(order=2),
    gravity_requirements(tide_system="zero_tide"), gravity_requirements(normalization="unnormalized")])
def test_gravity_requires_available_terms_and_matching_context(tmp_path, requirements):
    svc, artifact = registered(tmp_path, GRAVITY, "gravity", "icgem")
    with pytest.raises(AcquisitionError):
        svc.check(artifact["artifact_id"], requirements)


def test_gravity_materialization_preserves_exact_coefficients_and_acceleration(tmp_path):
    from sim.dynamics.orbit.spherical_harmonics import accel_spherical_harmonics_terms, load_icgem_gfc_terms
    svc, artifact = registered(tmp_path, GRAVITY, "gravity", "icgem")
    destination = tmp_path / "inputs" / "gravity.gfc"
    svc.materialize(artifact["artifact_id"], destination, requirements=gravity_requirements())
    assert destination.read_bytes() == GRAVITY
    original = load_icgem_gfc_terms(tmp_path / "source", 2, 0)
    copied = load_icgem_gfc_terms(destination, 2, 0)
    assert original == copied
    state = np.array([7000., 100., 200.])
    assert np.array_equal(accel_spherical_harmonics_terms(state, 0, original),
                          accel_spherical_harmonics_terms(state, 0, copied))


def test_eop_units_and_equinox_fields_survive_the_existing_runtime_reader(tmp_path):
    raw = eop()
    series = parse_celestrak_eop(raw.decode(), source_label="synthetic", source_sha256=digest(raw))
    first = series.records[0]
    assert first.dx_mas == 6 and first.ddpsi_mas == 40 and first.lod_ms == 1 and first.dat_s == 37
    path = tmp_path / "eop.txt"
    path.write_bytes(encode_eop_all(series))
    grid, xp, yp, dut1, dat = _load_eop_table(str(path))
    assert grid[0] == 60319 and xp[0] == .1 and yp[0] == .2 and dut1[0] == .03 and dat[0] == 37
    from sim.dynamics.orbit.frames import _interp_eop_nutation
    dpsi, deps = _interp_eop_nutation(60319, str(path))
    np.testing.assert_allclose([dpsi, deps], np.radians(np.array([.04, .05]) / 3600), rtol=1e-14)


def test_eop_predictions_require_an_explicit_choice(tmp_path):
    svc, artifact = registered(tmp_path, eop(), "eop", "celestrak-eop-csv")
    requirements = environmental_requirements(stop="2024-01-14T00:00:00Z")
    with pytest.raises(AcquisitionError) as error:
        svc.check(artifact["artifact_id"], requirements)
    assert error.value.code == "PREDICTED_DATA"
    assert svc.check(artifact["artifact_id"], dict(requirements, allow_predictions=True))["compatible"]


def test_eop_coverage_includes_interpolation_neighbors_and_rejects_gaps(tmp_path):
    svc, artifact = registered(tmp_path, eop(), "eop", "celestrak-eop-csv")
    with pytest.raises(AcquisitionError):
        svc.check(artifact["artifact_id"], environmental_requirements(start="2024-01-09T12:00:00Z"))
    lines = eop().decode().splitlines()
    gap = inspect_bytes(("\n".join(lines[:2] + lines[3:]) + "\n").encode(), kind="eop", format="celestrak-eop-csv")
    assert gap["gaps_after"] and not gap["compatible"]


def test_eop_csv_not_mistaken_for_c04_units():
    raw = eop().replace(b"60319", b"60318")
    assert not inspect_bytes(raw, kind="eop", format="celestrak-eop-csv")["parsed"]


def test_weather_mapping_matches_the_existing_msis_consumer(tmp_path):
    from datetime import datetime, timezone

    from sim.dynamics.orbit.nrlmsise00_backend import _solar_geomagnetic_inputs
    rows = parse_celestrak_weather(weather().decode())
    path = tmp_path / "sw.txt"
    path.write_bytes(encode_msis_table(rows))
    table = np.loadtxt(path)
    assert table.shape == (7, 32)
    assert table[3, 27] == 153 and table[2, 29] == 142
    f107a, f107, ap, ap_a = _solar_geomagnetic_inputs(datetime(2024, 1, 10, tzinfo=timezone.utc), {"nrlmsise00_sw_path": str(path)})
    assert (f107a, f107, ap) == (153, 142, 7)
    assert ap_a[1] == 7 and ap_a[2] == table[3, 14]


@pytest.mark.parametrize("raw,requirements", [
    (weather(), environmental_requirements("nrlmsise00", start="2024-01-08T00:00:00Z")),
    (weather(missing=(date(2024, 1, 9), "AP3")), environmental_requirements("nrlmsise00")),
    (weather(predicted=True), environmental_requirements("nrlmsise00")),
    (weather(), environmental_requirements("jb2008")),
])
def test_weather_rejects_missing_history_missing_indices_predictions_and_wrong_models(tmp_path, raw, requirements):
    svc, artifact = registered(tmp_path, raw, "space-weather", "celestrak-weather-csv")
    with pytest.raises(AcquisitionError):
        svc.check(artifact["artifact_id"], requirements)


def test_weather_materialization_preserves_parent_and_selects_required_history(tmp_path):
    svc, artifact = registered(tmp_path, weather(), "space-weather", "celestrak-weather-csv")
    output = tmp_path / "sw.txt"
    result = svc.materialize(artifact["artifact_id"], output, requirements=environmental_requirements("msis86"))
    values = np.loadtxt(output)
    assert values.shape[0] == 4 and values[0, 2] == 7 and values[-1, 2] == 10
    assert result["format"] == "msis-sw" and result["source_sha256"] == artifact["sha256"]


def test_historical_coverage_is_not_rejected_due_to_current_date(tmp_path):
    svc, artifact = registered(tmp_path, weather(), "space-weather", "celestrak-weather-csv")
    assert svc.check(artifact["artifact_id"], environmental_requirements("nrlmsise00"))["compatible"]
