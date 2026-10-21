from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.ccsds import compare_odm, compare_oem
from sim.core.models import Command, StateTruth
from sim.data_io.api import encode_data, read_data
from sim.data_io.attitude import parse_adm
from sim.data_io.catalog import encode_catalog, from_gp_fields, propagate_catalog, read_catalog
from sim.data_io.common import publish, strict_json, xml_root
from sim.data_io.contracts import AttitudeHistory, DataIOError, Trajectory
from sim.data_io.navigation import encode_navigation, oem_from_trajectory, parse_navigation, read_navigation
from sim.data_io.precise import encode_sp3, read_sp3
from sim.data_io.registry import capability_catalog
from sim.data_io.rinex import encode_rinex, parse_rinex
from sim.data_io.time import parse_time
from sim.data_io.tracking import encode_tracking, parse_tracking, read_tracking
from sim.data_io.trajectory import TrajectorySampler, transform_trajectory
from sim.dynamics.ephemeris_replay import PrescribedTrajectoryDynamics

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "sim/interchange/examples"


@pytest.fixture
def trajectory():
    return Trajectory(
        "SAT",
        "EARTH",
        "EME2000",
        parse_time("2024-01-01T00:00:00"),
        [0, 30, 60],
        [[7000, 0, 0, 0, 7.5, 0], [7000, 225, 0, 0, 7.5, 0], [7000, 450, 0, 0, 7.5, 0]],
    )


@pytest.fixture
def attitude(trajectory):
    angle = np.array([0, 0.2, 0.4])
    q = np.column_stack((np.cos(angle / 2), np.zeros((3, 2)), np.sin(angle / 2)))
    return AttitudeHistory(
        "SAT", "EME2000", "SC_BODY_1", trajectory.epoch, trajectory.times_s, q, np.tile([0, 0, 0.2 / 30], (3, 1))
    )


@pytest.mark.parametrize(
    "format", ["trajectory-json", "trajectory-csv", "trajectory-parquet", "oem-kvn", "oem-xml", "stk-ephemeris"]
)
def test_trajectory_sample_roundtrips(tmp_path, trajectory, format):
    if format.endswith("parquet"):
        pytest.importorskip("pyarrow")
    raw, extra = encode_data(trajectory, format=format)
    path = tmp_path / "input"
    publish(path, raw, receipt=extra)
    result = read_data(path, format=format, options={"object_id": "SAT"})
    if format.startswith("oem"):
        from sim.data_io.navigation import trajectory_from_oem

        result = trajectory_from_oem(result)
    np.testing.assert_allclose(result.states, trajectory.states, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(result.times_s, trajectory.times_s)
    assert result.frame == trajectory.frame
    assert result.epoch.tai_seconds == trajectory.epoch.tai_seconds


@pytest.mark.parametrize("format", ["oem-xml", "opm-xml", "omm-xml"])
def test_existing_officially_cross_read_profiles_preserve_semantics_in_xml(format):
    names = {
        "oem-xml": "ccsds_502_0_b_3_g13_covariance_excerpt.oem",
        "opm-xml": "oel_earth_eme2000_utc_v3.opm",
        "omm-xml": "oel_sgp4_mean_elements_v3.omm",
    }
    filename = EXAMPLES / names[format]
    if not filename.exists() and format == "opm-xml":
        filename = next(EXAMPLES.glob("*.opm"))
    original = read_navigation(filename, format=format.replace("xml", "kvn"))
    encoded = encode_navigation(original, format=format)
    recovered = parse_navigation(encoded.decode(), format=format)
    compare = compare_oem if format.startswith("oem") else compare_odm
    assert compare(original, recovered)["status"] == "equivalent"


def test_oem_xml_preserves_comments_and_rejects_unknown_structure():
    message = read_navigation(EXAMPLES / "oel_earth_eme2000_utc_v3.oem", format="oem-kvn")
    xml = encode_navigation(message, format="oem-xml").decode()
    result = parse_navigation(xml, format="oem-xml")
    assert result.header.comments == message.header.comments
    assert result.segments[0].comments == message.segments[0].comments
    with pytest.raises(DataIOError, match="Unsupported XML"):
        parse_navigation(xml.replace("</metadata>", "<UNKNOWN>1</UNKNOWN></metadata>"), format="oem-xml")


def test_oem_v2_is_explicitly_supported_without_v3_field_loss(trajectory):
    message = oem_from_trajectory(trajectory)
    raw = encode_navigation(message, format="oem-kvn", version="2.0")
    assert raw.startswith(b"CCSDS_OEM_VERS = 2.0")
    assert len(parse_navigation(raw.decode(), format="oem-kvn").segments[0].states) == 3
    with pytest.raises(DataIOError):
        encode_navigation(
            replace(message, header=replace(message.header, message_id="v3")), format="oem-kvn", version="2.0"
        )


def test_covariance_correlation_survives_json_csv_and_oem(tmp_path, trajectory):
    matrix = np.eye(6) * 1e-6
    matrix[0, 3] = matrix[3, 0] = 2e-7
    trajectory = replace(trajectory, covariance_times_s=[0, 60], covariances=[matrix, matrix * 2])
    for format in ("trajectory-json", "trajectory-csv", "oem-kvn", "oem-xml"):
        raw, extra = encode_data(trajectory, format=format)
        path = tmp_path / format
        publish(path, raw, receipt=extra)
        result = read_data(path, format=format)
        if format.startswith("oem"):
            from sim.data_io.navigation import trajectory_from_oem

            result = trajectory_from_oem(result)
        np.testing.assert_array_equal(result.covariances, trajectory.covariances)
    with pytest.raises(DataIOError, match="covariance"):
        encode_data(trajectory, format="stk-ephemeris")


def test_gp_catalog_handles_large_ids_and_fails_unrepresentable_tle(tmp_path):
    record = read_catalog(EXAMPLES / "oel_sgp4_mean_elements_v3.omm", format="omm-kvn")[0]
    fields = {**record.fields, "NORAD_CAT_ID": "123456789"}
    record = from_gp_fields(fields)
    for fmt in ("gp-json", "gp-csv", "omm-kvn", "omm-xml"):
        raw = encode_catalog([record], format=fmt)
        path = tmp_path / fmt
        path.write_bytes(raw)
        parsed = read_catalog(path, format=fmt)[0]
        assert parsed.elements.norad_number == "123456789"
        assert parsed.elements.mean_motion_rev_per_day == record.elements.mean_motion_rev_per_day
    with pytest.raises(DataIOError, match="five-digit"):
        encode_catalog([record], format="tle")


def test_gp_provider_defaults_are_named_and_mean_elements_reach_ogp():
    record = read_catalog(EXAMPLES / "oel_sgp4_mean_elements_v3.omm", format="omm-kvn")[0]
    fields = {
        k: v
        for k, v in record.fields.items()
        if k not in {"CENTER_NAME", "REF_FRAME", "TIME_SYSTEM", "MEAN_ELEMENT_THEORY"}
    }
    with pytest.raises(DataIOError, match="Missing OMM"):
        from_gp_fields(fields, provider_profile="generic-omm")
    result = propagate_catalog(from_gp_fields(fields), start_epoch=fields["EPOCH"], duration_s=120, step_s=60)
    assert result.frame == "TEME" and result.states.shape == (3, 6)
    assert np.linalg.norm(result.states[0, :3]) > 6000


def test_tle_output_cross_reads_with_independent_sgp4(tmp_path):
    from sgp4.api import Satrec

    record = read_catalog(EXAMPLES / "oel_sgp4_mean_elements_v3.omm", format="omm-kvn")[0]
    path = tmp_path / "encoded.tle"
    path.write_bytes(encode_catalog([record], format="tle"))
    reparsed = read_catalog(path, format="tle")[0]
    sat = Satrec.twoline2rv(*path.read_text().splitlines())
    error, position, velocity = sat.sgp4(reparsed.elements.epoch_jd_utc, 0)
    assert error == 0
    from sim.dynamics.orbit.ogp import ogp_propagate_teme

    state = ogp_propagate_teme(reparsed.elements, 0, backend="python")
    # The independent library and OEL's qualified reference differ at the
    # millimetre level; encoding parity is separately tested with identical elements.
    np.testing.assert_allclose(state.position_teme_km, position, atol=5e-6, rtol=0)
    np.testing.assert_allclose(state.velocity_teme_km_s, velocity, atol=5e-9, rtol=0)


@pytest.mark.parametrize("format", ["aem-kvn", "stk-attitude"])
def test_attitude_quaternion_direction_and_rate_units(tmp_path, attitude, format):
    raw, _ = encode_data(attitude, format=format)
    if format == "stk-attitude":
        # Independent of our reader: STK specifies degrees/second in .a files.
        assert float(raw.decode().splitlines()[-2].split()[-1]) == pytest.approx(
            np.rad2deg(attitude.rates_rad_s[-1, 2])
        )
    path = tmp_path / "attitude"
    path.write_bytes(raw)
    parsed = read_data(path, format=format, options={"object_id": "SAT"})
    np.testing.assert_allclose(parsed.quaternions, attitude.quaternions, atol=1e-14)
    np.testing.assert_allclose(parsed.rates_rad_s, attitude.rates_rad_s, atol=1e-14)
    from sim.utils.quaternion import quaternion_to_dcm_bn

    mapped = quaternion_to_dcm_bn(parsed.quaternions[-1]) @ np.array([1, 0, 0])
    np.testing.assert_allclose(mapped, [np.cos(0.4), -np.sin(0.4), 0], atol=1e-14)


def test_apm_requires_an_exact_sample(attitude):
    with pytest.raises(DataIOError, match="exactly one"):
        encode_data(attitude, format="apm-kvn")
    raw, _ = encode_data(attitude, format="apm-kvn", options={"sample_index": 2})
    parsed = parse_adm(raw.decode(), kind="apm")
    np.testing.assert_allclose(parsed.quaternions[0], attitude.quaternions[-1])
    assert parsed.epoch.tai_seconds == attitude.epoch.tai_seconds + 60


def test_sp3_velocity_units_and_clock_zero_roundtrip(tmp_path, trajectory):
    fixed = replace(
        trajectory,
        frame="ITRF",
        metadata={"sp3_clock_records": [{"clock_microseconds": 0, "clock_rate_1e4_microseconds_s": 0}] * 3},
    )
    path = tmp_path / "orbit.sp3"
    path.write_bytes(encode_sp3(fixed, satellite_id="G01"))
    result = read_sp3(path, frame="ITRF", object_id="G01")
    np.testing.assert_allclose(result.states, fixed.states, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(result.times_s, fixed.times_s)
    assert result.epoch.tai_seconds == fixed.epoch.tai_seconds
    assert result.metadata["sp3_clock_records"][0]["clock_rate_1e4_microseconds_s"] == 0


def test_sp3_padded_terminator_is_valid_but_missing_terminator_is_not(tmp_path, trajectory):
    path = tmp_path / "esa.sp3"
    raw = encode_sp3(replace(trajectory, frame="ITRF"), satellite_id="G01").decode()
    path.write_text(raw.replace("EOF", "EOF" + " " * 77))
    np.testing.assert_allclose(read_sp3(path, frame="ITRF").states, trajectory.states, atol=1e-8)
    path.write_text(raw.replace("EOF", ""))
    with pytest.raises(DataIOError, match="missing EOF"):
        read_sp3(path, frame="ITRF")


def test_sp3_does_not_invent_velocity_or_replay_discontinuities(tmp_path, trajectory):
    path = tmp_path / "orbit.sp3"
    raw = encode_sp3(replace(trajectory, frame="ITRF"), satellite_id="G01").decode()
    path.write_text("\n".join(line for line in raw.splitlines() if not line.startswith("VG01")))
    with pytest.raises(DataIOError, match="lacks velocity"):
        read_sp3(path, frame="ITRF")
    parsed = read_sp3(path, frame="ITRF", derive_velocity=True)
    np.testing.assert_allclose(parsed.states[:, 3:], trajectory.states[:, 3:], atol=1e-12)


def test_replay_is_passive_checks_continuity_and_never_extrapolates(trajectory):
    sampler = TrajectorySampler(trajectory, max_gap_s=30)
    np.testing.assert_allclose(sampler.state_at(15), [7000, 112.5, 0, 0, 7.5, 0])
    dynamics = PrescribedTrajectoryDynamics(sampler, simulation_epoch=trajectory.epoch)
    initial = StateTruth(
        trajectory.states[0, :3], trajectory.states[0, 3:], np.array([1.0, 0, 0, 0]), np.zeros(3), 100, 0
    )
    result = dynamics.step(initial, Command.zero(), {}, 15)
    np.testing.assert_allclose(result.position_eci_km, sampler.state_at(15)[:3])
    with pytest.raises(DataIOError, match="passive"):
        dynamics.step(initial, Command(np.array([1.0, 0, 0])), {}, 15)
    with pytest.raises(DataIOError, match="outside"):
        sampler.state_at(61)
    with pytest.raises(DataIOError, match="max_gap"):
        TrajectorySampler(trajectory, max_gap_s=20).state_at(15)
    with pytest.raises(DataIOError, match="discontinuity"):
        TrajectorySampler(replace(trajectory, metadata={"discontinuity_sample_indices": [1]}), max_gap_s=30).state_at(
            15
        )


def test_canonical_frames_transform_both_state_and_correlated_covariance(trajectory):
    matrix = np.eye(6) * 1e-7
    matrix[0, 4] = matrix[4, 0] = 3e-8
    t = replace(trajectory, covariance_times_s=[0], covariances=[matrix])
    converted = transform_trajectory(t, "GCRF")
    recovered = transform_trajectory(converted, "EME2000")
    np.testing.assert_allclose(recovered.states, t.states, atol=1e-10)
    np.testing.assert_allclose(recovered.covariances, t.covariances, atol=1e-20)


def test_raw_doppler_and_radec_tdm_are_preserved_without_fake_od_readiness(tmp_path):
    raw = """CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2024-01-01T00:00:00
ORIGINATOR = OEL
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = STATION
PARTICIPANT_2 = SAT
MODE = SEQUENTIAL
PATH = 2,1
ANGLE_TYPE = RADEC
INTEGRATION_INTERVAL = 1
META_STOP
DATA_START
ANGLE_1 = 2024-01-01T00:00:00 10
ANGLE_2 = 2024-01-01T00:00:00 20
DOPPLER_INTEGRATED = 2024-01-01T00:00:01 -123.5
DATA_STOP
"""
    value = parse_tracking(raw)
    assert value.segments[0]["instantaneous_od_ready"] is False
    for fmt in ("tdm-kvn", "tdm-xml"):
        path = tmp_path / fmt
        path.write_bytes(encode_tracking(value, format=fmt))
        parsed = read_tracking(path, format=fmt)
        assert parsed.segments == value.segments


def _rinex_header(value, label):
    return f"{value:<60}{label}"


def test_rinex_observation_values_and_quality_indicators_roundtrip():
    raw = (
        "\n".join(
            [
                _rinex_header("     3.05           O                   G", "RINEX VERSION / TYPE"),
                _rinex_header("G    2 C1C L1C", "SYS / # / OBS TYPES"),
                _rinex_header("", "END OF HEADER"),
                "> 2024 01 01 00 00  0.0000000  0  1",
                "G01" + f"{20000000.0:14.3f}15{123456.0:14.3f} 7",
            ]
        )
        + "\n"
    )
    value = parse_rinex(raw)
    obs = value.records[0]["satellites"][0]["observations"]["C1C"]
    assert obs == {"value": 20000000, "lli": "1", "ssi": "5"}
    assert parse_rinex(encode_rinex(value).decode()).records == value.records


def test_rinex_navigation_clock_and_ephemeris_fields_roundtrip():
    def f(v):
        return f"{v:19.12E}".replace("E", "D")

    raw = (
        "\n".join(
            [
                _rinex_header("     3.05           N                   G", "RINEX VERSION / TYPE"),
                _rinex_header("", "END OF HEADER"),
                "G01 2024 01 01 00 00 00" + "".join(f(v) for v in [1e-4, 1e-12, 0]),
                *["    " + "".join(f(v) for v in range(4)) for _ in range(7)],
            ]
        )
        + "\n"
    )
    value = parse_rinex(raw)
    assert value.records[0]["clock_coefficients"][0] == 1e-4
    assert parse_rinex(encode_rinex(value).decode()).records == value.records


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'])
def test_json_rejects_ambiguous_or_nonfinite_values(raw):
    with pytest.raises(DataIOError):
        strict_json(raw)


def test_xml_dtd_unknown_attributes_and_source_overwrite_are_rejected(tmp_path):
    for raw in ('<!DOCTYPE a [<!ENTITY x "unsafe">]><a>&x;</a>', '<oem script="x"/>'):
        with pytest.raises(DataIOError):
            xml_root(raw)
    path = tmp_path / "source"
    path.write_text("unchanged")
    with pytest.raises(DataIOError):
        publish(path, b"replace", receipt={}, sources=(path,))
    assert path.read_text() == "unchanged"


def test_csv_tampering_invalidates_its_receipt(tmp_path, trajectory):
    raw, context = encode_data(trajectory, format="trajectory-csv")
    path = tmp_path / "table.csv"
    publish(path, raw, receipt=context)
    path.write_bytes(raw.replace(b"7000", b"7001"))
    with pytest.raises(DataIOError, match="receipt"):
        read_data(path, format="trajectory-csv")


def test_registry_separates_serialization_from_fitting_and_execution():
    catalog = capability_catalog()
    assert catalog["operations"]["fit-review"]["source"] == "completed review store"
    assert len({r["format"] for r in catalog["formats"]}) == len(catalog["formats"])
    assert all("limits" in row and "profiles" in row for row in catalog["formats"])


def test_opm_cartesian_conversion_uses_trajectory_owner_and_retains_annotations(tmp_path, trajectory):
    from sim.data_io.api import convert
    from sim.data_io.products import opm_from_trajectory

    message = opm_from_trajectory(trajectory, sample_index=1)
    source = tmp_path / "source.opm"
    source.write_bytes(encode_navigation(message, format="opm-kvn"))
    result = convert(source, tmp_path / "state.oem", input_format="opm-kvn", output_format="oem-xml")
    output = read_data(tmp_path / "state.oem", format="oem-xml")
    assert len(output.segments[0].states) == 1
    assert "source_annotations" in result
    assert output.segments[0].states[0].position_km[0] == trajectory.states[1, 0]


def test_gp_refuses_other_theories_boolean_numbers_and_message_marking_loss():
    record = read_catalog(EXAMPLES / "oel_sgp4_mean_elements_v3.omm", format="omm-kvn")[0]
    for field, value in (("EPHEMERIS_TYPE", 4), ("GM", 398600.4418), ("MEAN_MOTION", True)):
        with pytest.raises(DataIOError):
            from_gp_fields({**record.fields, field: value})
    marked = from_gp_fields({**record.fields, "CLASSIFICATION": "SYNTHETIC TEST MARKING"})
    with pytest.raises(DataIOError, match="classification banner"):
        encode_catalog([marked], format="tle")


def test_sentinel_units_and_auxiliary_data_are_explicit(tmp_path):
    from sim.data_io.precise import read_sentinel_eof

    raw = '<Earth_Explorer_File><Data_Block><List_of_OSVs count="1"><OSV><UTC>UTC=2024-01-01T00:00:00</UTC><X unit="m">7000000</X><Y unit="m">0</Y><Z unit="m">0</Z><VX unit="m/s">0</VX><VY unit="m/s">7500</VY><VZ unit="m/s">0</VZ><Quality>NOMINAL</Quality></OSV></List_of_OSVs></Data_Block></Earth_Explorer_File>'
    path = tmp_path / "orbit.eof"
    path.write_text(raw)
    value = read_sentinel_eof(path, frame="ITRF", object_id="SAT")
    np.testing.assert_array_equal(value.states[0], [7000, 0, 0, 0, 7.5, 0])
    assert value.metadata["source_auxiliary_records"][0]["Quality"] == "NOMINAL"
    path.write_text(raw.replace('unit="m"', 'unit="km"'))
    with pytest.raises(DataIOError, match="requires m"):
        read_sentinel_eof(path, frame="ITRF", object_id="SAT")


def test_rinex_native_precision_and_normalized_json_roundtrip(tmp_path):
    raw = (
        "\n".join(
            [
                _rinex_header("     3.05           O                   G", "RINEX VERSION / TYPE"),
                _rinex_header("G    1 C1C", "SYS / # / OBS TYPES"),
                _rinex_header("", "END OF HEADER"),
                "> 2024 01 01 00 00  0.0000001  0  1",
                "G01" + f"{20000000.0:14.3f}15",
                "> 2024 01 01 00 00  0.0000002  0  1",
                "G01" + f"{20000001.0:14.3f}15",
            ]
        )
        + "\n"
    )
    value = parse_rinex(raw)
    data, _ = encode_data(value, format="normalized-json")
    path = tmp_path / "observations.json"
    path.write_bytes(data)
    recovered = read_data(path, format="normalized-json")
    assert recovered.records == value.records
    with pytest.raises(DataIOError, match="seven fractional"):
        parse_rinex(raw.replace("0.0000001", "0.00000001"))


def test_visualization_conventions_are_explicit(trajectory):
    from sim.data_io.visualization import encode_visualization

    fixed = replace(trajectory, frame="ITRF", object_id="document")
    czml, _ = encode_visualization(fixed, format="czml")
    packets = strict_json(czml.decode())
    assert packets[1]["id"] != "document"
    assert packets[1]["position"]["referenceFrame"] == "FIXED"
    points, _ = encode_visualization(fixed, format="geojson")
    first = strict_json(points.decode())["features"][0]["geometry"]["coordinates"]
    np.testing.assert_allclose(first, [0, 0, (7000 - 6378.137) * 1000], atol=1e-6)
    kml, receipt = encode_visualization(fixed, format="kml")
    assert b"clampToGround" in kml and "ground-track" in receipt["visualization_omissions"][-1]


def test_parquet_rejects_variable_size_columns_before_materializing_rows(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from sim.data_io.tabular import COLUMNS

    path = tmp_path / "unexpected.parquet"
    pq.write_table(pa.table({name: ["0"] for name in COLUMNS}), path)
    with pytest.raises(DataIOError, match="floating-point types"):
        read_data(path, format="trajectory-parquet")
