from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import erfa
import numpy as np
import pytest

from sim.analysis.tracking_od import TrackingOdError, TrackingOdProblem, assess_tdm_orbit_determination
from sim.estimation.ground_station_od import _state_covariance_eci_km_s
from sim.estimation.partitioning import partition_time_arc
from sim.interchange.ccsds_tdm import (
    CcsdsTdmError,
    TdmMessage,
    compare_tdm,
    parse_tdm_kvn,
    read_tdm_kvn,
    serialize_tdm_kvn,
    write_tdm_kvn,
)
from sim.tracking_data import normalize_tdm_tracking_dataset
from sim.tracking_od import main

ROOT = Path(__file__).resolve().parents[2]
TDM = ROOT / "examples" / "tracking_od" / "public_reduced_geometric_azel_range.tdm"
PROBLEM = ROOT / "examples" / "tracking_od" / "public_tdm_fit_holdout_problem.json"
TRUTH_STATE = np.array([6878.137, 0.0, 0.0, 0.0, 6.592712067068919, 3.806304086611934])


def _problem() -> dict:
    return json.loads(PROBLEM.read_text(encoding="utf-8"))


def _station_mapping() -> list[dict]:
    station = _problem()["stations"][0]
    return [
        {
            "id": station["station_id"],
            "lat_deg": station["latitude_deg"],
            "lon_deg": station["longitude_deg"],
            "alt_km": station["altitude_km"],
        }
    ]


def test_tdm_parses_roundtrips_and_retains_source_identity(tmp_path: Path) -> None:
    message = read_tdm_kvn(TDM)
    assert message.source_sha256 == hashlib.sha256(TDM.read_bytes()).hexdigest()
    assert len(message.segments) == 1
    assert len(message.segments[0].observations) == 39
    assert message.segments[0].metadata.station_id == "OEL-STATION"
    assert message.segments[0].metadata.object_id == "OEL-SAT"

    canonical = parse_tdm_kvn(serialize_tdm_kvn(message))
    assert compare_tdm(message, canonical)["equivalent"] is True
    destination = tmp_path / "canonical.tdm"
    assert main(["roundtrip-tdm", str(TDM), str(destination)]) == 0
    assert compare_tdm(message, destination)["status"] == "equivalent"


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("PATH = 2,1", "PATH = 1,2,1", "PATH = 2,1"),
        ("RANGE_UNITS = km", "RANGE_UNITS = s", "RANGE_UNITS = km"),
        ("RANGE_MODULUS = 0", "RANGE_MODULUS = 100", "ambiguous range"),
        ("TIME_SYSTEM = UTC", "TIME_SYSTEM = TAI", "TIME_SYSTEM = UTC"),
        (
            "RANGE = 2000-01-01T12:00:00Z 1251.430312759146",
            "DOPPLER_INTEGRATED = 2000-01-01T12:00:00Z 1.0",
            "signal, path, count/integration, and sign semantics",
        ),
        ("DATA_QUALITY = VALIDATED", "DATA_QUALITY = DEGRADED", "DATA_QUALITY = VALIDATED"),
        ("CORRECTIONS_APPLIED = YES", "CORRECTIONS_APPLIED = NO", "CORRECTIONS_APPLIED = YES"),
        ("MODE = SEQUENTIAL", "TIMETAG_REF = TRANSMIT\nMODE = SEQUENTIAL", "TIMETAG_REF = RECEIVE"),
        ("MODE = SEQUENTIAL", "PARTICIPANT_3 = EXTRA\nMODE = SEQUENTIAL", "extra participants"),
    ],
)
def test_tdm_profile_fails_closed_on_unsupported_semantics(old: str, new: str, match: str) -> None:
    text = TDM.read_text(encoding="utf-8").replace(old, new, 1)
    with pytest.raises(CcsdsTdmError, match=match):
        parse_tdm_kvn(text)


def test_tdm_rejects_incomplete_segment_and_non_ascii() -> None:
    text = TDM.read_text(encoding="utf-8")
    with pytest.raises(CcsdsTdmError, match="incomplete segment"):
        parse_tdm_kvn(text.replace("DATA_STOP\n", "", 1))
    with pytest.raises(CcsdsTdmError, match="printable ASCII"):
        parse_tdm_kvn(text.replace("ORIGINATOR = OEL", "ORIGINATOR = OÉL", 1))


def test_tdm_writer_validates_before_preserving_or_atomically_replacing_target(tmp_path: Path) -> None:
    target = tmp_path / "output.tdm"
    target.write_text("prior evidence\n", encoding="utf-8")
    message = read_tdm_kvn(TDM)
    invalid = TdmMessage(
        header=message.header,
        segments=(),
        header_comments=message.header_comments,
        source_sha256=message.source_sha256,
    )
    with pytest.raises(CcsdsTdmError, match="at least one segment"):
        write_tdm_kvn(invalid, target)
    assert target.read_text(encoding="utf-8") == "prior evidence\n"

    write_tdm_kvn(message, target)
    assert compare_tdm(message, target)["equivalent"] is True
    assert not list(tmp_path.glob(".output.tdm.*.tmp"))


@pytest.mark.parametrize("value", [float("nan"), "not-a-number"])
def test_tdm_validation_rejects_nonfinite_or_nonnumeric_programmatic_records(value: object) -> None:
    message = read_tdm_kvn(TDM)
    first = message.segments[0].observations[0]
    invalid = replace(
        message,
        segments=(
            replace(
                message.segments[0],
                observations=(replace(first, value=value),) + message.segments[0].observations[1:],
            ),
        ),
    )
    with pytest.raises(CcsdsTdmError, match="finite numeric"):
        serialize_tdm_kvn(invalid)


def test_tdm_validation_rejects_unsupported_programmatic_record_keywords() -> None:
    message = read_tdm_kvn(TDM)
    first = message.segments[0].observations[0]
    invalid = replace(
        message,
        segments=(
            replace(
                message.segments[0],
                observations=(replace(first, keyword="UNSUPPORTED"),) + message.segments[0].observations[1:],
            ),
        ),
    )
    with pytest.raises(CcsdsTdmError, match="unsupported tracking-data keyword"):
        serialize_tdm_kvn(invalid)


def test_tdm_rejects_correction_metadata_bad_digest_and_invalid_record_order() -> None:
    text = TDM.read_text(encoding="utf-8")
    with pytest.raises(CcsdsTdmError, match="correction metadata"):
        parse_tdm_kvn(text.replace("CORRECTIONS_APPLIED = YES", "CORRECTION_RANGE = 1\nCORRECTIONS_APPLIED = YES"))
    with pytest.raises(CcsdsTdmError, match="source_sha256"):
        parse_tdm_kvn(text, source_sha256="0" * 64)

    first = "ANGLE_1 = 2000-01-01T12:00:00Z 180.000000000000"
    second = "ANGLE_1 = 2000-01-01T12:01:00Z 159.169948204004"
    reversed_angles = text.replace(first, "@@", 1).replace(second, first, 1).replace("@@", second, 1)
    with pytest.raises(CcsdsTdmError, match="must be chronological"):
        parse_tdm_kvn(reversed_angles)
    with pytest.raises(CcsdsTdmError, match="duplicate ANGLE_1 timetag"):
        parse_tdm_kvn(text.replace(first, f"{first}\n{first}", 1))

    segment = text[text.index("META_START") :]
    with pytest.raises(CcsdsTdmError, match="duplicate observation across segments"):
        parse_tdm_kvn(f"{text.rstrip()}\n\n{segment}")


def test_normalized_dataset_is_content_bound_and_requires_reduced_geometry() -> None:
    message = read_tdm_kvn(TDM)
    first = normalize_tdm_tracking_dataset(
        message,
        stations=_station_mapping(),
        measurement_semantics="reduced_geometric",
        angle_sigma_deg=0.005,
        range_sigma_km=0.01,
        expected_object_id="OEL-SAT",
    )
    second = normalize_tdm_tracking_dataset(
        message,
        stations=_station_mapping(),
        measurement_semantics="reduced_geometric",
        angle_sigma_deg=0.01,
        range_sigma_km=0.01,
        expected_object_id="OEL-SAT",
    )
    assert first["measurement_epoch_count"] == 13
    assert first["observable_record_count"] == 39
    assert first["measurement_rows"][0]["components"] == ["azimuth_deg", "elevation_deg", "range_km"]
    assert first["dataset_sha256"] != second["dataset_sha256"]
    with pytest.raises(ValueError, match="reduced_geometric"):
        normalize_tdm_tracking_dataset(
            message,
            stations=_station_mapping(),
            measurement_semantics="raw_radiometric",
            angle_sigma_deg=0.005,
            range_sigma_km=0.01,
        )


def test_normalization_preserves_microsecond_epoch_identity_and_accepts_ordinal_epochs() -> None:
    text = TDM.read_text(encoding="utf-8")
    first_epoch_records = "\n".join(
        line for line in text.splitlines() if "= 2000-01-01T12:00:00Z " in line
    )
    shifted_records = first_epoch_records.replace(
        "2000-01-01T12:00:00Z", "2000-01-01T12:00:00.000010Z"
    )
    shifted = text.replace(first_epoch_records, f"{first_epoch_records}\n{shifted_records}", 1)
    dataset = normalize_tdm_tracking_dataset(
        parse_tdm_kvn(shifted),
        stations=_station_mapping(),
        measurement_semantics="reduced_geometric",
        angle_sigma_deg=0.005,
        range_sigma_km=0.01,
    )
    assert dataset["measurement_epoch_count"] == 14
    assert dataset["measurement_rows"][1]["time_tai_seconds"] - dataset["measurement_rows"][0]["time_tai_seconds"] == pytest.approx(1.0e-5, abs=2.0e-7)

    ordinal = parse_tdm_kvn(text.replace("2000-01-01", "2000-001"))
    ordinal_dataset = normalize_tdm_tracking_dataset(
        ordinal,
        stations=_station_mapping(),
        measurement_semantics="reduced_geometric",
        angle_sigma_deg=0.005,
        range_sigma_km=0.01,
    )
    assert ordinal_dataset["measurement_epoch_count"] == 13


def test_tracking_partition_uses_the_declared_strict_boundary() -> None:
    partition = partition_time_arc(
        np.array([0.0, 420.0, 420.00005, 720.0]),
        fit_duration_s=420.0,
        holdout_duration_s=300.0,
        allow_repeated_epochs=True,
        boundary_tolerance_s=0.0,
    )
    assert partition.fit_mask.tolist() == [True, True, False, False]
    assert partition.holdout_mask.tolist() == [False, False, True, True]


def _independent_azel_range(time_s: float) -> tuple[float, float, float]:
    radius_km = 6878.137
    inclination_rad = math.radians(30.0)
    mean_motion_rad_s = math.sqrt(398600.4418 / radius_km**3)
    argument = mean_motion_rad_s * time_s
    target_eci = np.array(
        [
            radius_km * math.cos(argument),
            radius_km * math.sin(argument) * math.cos(inclination_rad),
            radius_km * math.sin(argument) * math.sin(inclination_rad),
        ]
    )
    latitude = math.radians(10.0)
    longitude = math.radians(79.53938163)
    flattening = 1.0 / 298.257223563
    eccentricity_squared = flattening * (2.0 - flattening)
    prime_vertical = 6378.137 / math.sqrt(1.0 - eccentricity_squared * math.sin(latitude) ** 2)
    station_ecef = np.array(
        [
            prime_vertical * math.cos(latitude) * math.cos(longitude),
            prime_vertical * math.cos(latitude) * math.sin(longitude),
            prime_vertical * (1.0 - eccentricity_squared) * math.sin(latitude),
        ]
    )
    gmst = float(erfa.gmst82(2451545.0 + time_s / 86400.0, 0.0))
    cosine = math.cos(gmst)
    sine = math.sin(gmst)
    target_ecef = np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]]) @ target_eci
    east_north_up = np.array(
        [
            [-math.sin(longitude), math.cos(longitude), 0.0],
            [
                -math.sin(latitude) * math.cos(longitude),
                -math.sin(latitude) * math.sin(longitude),
                math.cos(latitude),
            ],
            [
                math.cos(latitude) * math.cos(longitude),
                math.cos(latitude) * math.sin(longitude),
                math.sin(latitude),
            ],
        ]
    ) @ (target_ecef - station_ecef)
    slant_range = float(np.linalg.norm(east_north_up))
    azimuth = math.degrees(math.atan2(east_north_up[0], east_north_up[1])) % 360.0
    elevation = math.degrees(math.asin(east_north_up[2] / slant_range))
    return azimuth, elevation, slant_range


def test_synthetic_fixture_matches_independent_erfa_and_closed_form_oracle() -> None:
    message = read_tdm_kvn(TDM)
    by_epoch: dict[str, dict[str, float]] = {}
    for observation in message.segments[0].observations:
        by_epoch.setdefault(observation.epoch_utc, {})[observation.keyword] = observation.value
    for time_s, epoch in (
        (0.0, "2000-01-01T12:00:00Z"),
        (60.0, "2000-01-01T12:01:00Z"),
        (720.0, "2000-01-01T12:12:00Z"),
    ):
        azimuth, elevation, slant_range = _independent_azel_range(time_s)
        retained = by_epoch[epoch]
        assert retained["ANGLE_1"] == pytest.approx(azimuth, abs=0.002)
        assert retained["ANGLE_2"] == pytest.approx(elevation, abs=0.002)
        assert retained["RANGE"] == pytest.approx(slant_range, abs=0.01)


def test_end_to_end_tdm_fit_retains_holdout_prediction_and_artifact_receipts(tmp_path: Path) -> None:
    evidence = assess_tdm_orbit_determination(
        read_tdm_kvn(TDM),
        TrackingOdProblem.from_mapping(_problem()),
        output_dir=tmp_path / "tdm_od",
    )
    assert evidence["status"] == "completed"
    assert evidence["input"]["measurement_epoch_count"] == 13
    assert evidence["partition"]["fit_observation_count"] == 8
    assert evidence["partition"]["holdout_observation_count"] == 5
    assert evidence["estimator"]["solver"]["success"] is True
    assert evidence["estimator"]["quality_gates"]["data_full_rank"] is True
    assert evidence["estimator"]["fit_metrics"]["weighted_rms"] < 0.01
    assert evidence["estimator"]["holdout_metrics"]["weighted_rms"] < 0.05
    fitted = np.asarray(evidence["estimator"]["fitted_state_eci_km_km_s"], dtype=float)
    assert np.linalg.norm(fitted[:3] - TRUTH_STATE[:3]) < 0.001
    assert np.linalg.norm(fitted[3:] - TRUTH_STATE[3:]) < 1.0e-5
    covariance = np.asarray(evidence["estimator"]["state_covariance_eci"], dtype=float)
    assert covariance.shape == (6, 6)
    assert np.all(np.diag(covariance) >= 0.0)
    prediction = evidence["authoritative_holdout_prediction"]
    assert prediction["component_row_count"] == 15
    assert prediction["components"] == ["azimuth_deg", "elevation_deg", "range_km"]
    assert "not state-error truth" in prediction["claim_boundary"]
    assert evidence["estimator"]["state_epoch_utc"] == "2000-01-01T12:00:00Z"
    assert evidence["estimator"]["state_frame"] == "ECI"
    assert evidence["estimator"]["frame_provenance"]["model"] == "simple_gmst"
    output_root = tmp_path / "tdm_od"
    canonical_receipt = evidence["artifacts"]["canonical_tdm"]
    assert evidence["input"]["tdm_source_sha256"] == canonical_receipt["sha256"]
    fitted_packet = json.loads(
        (output_root / evidence["artifacts"]["fitted_state_packet"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert fitted_packet["frame_provenance"]["model"] == "simple_gmst"
    assert fitted_packet["objects"]["OEL-SAT"]["provenance"]["frame_model"] == "simple_gmst"
    for receipt in evidence["artifacts"].values():
        artifact = output_root / receipt["path"]
        assert artifact.is_file()
        assert artifact.stat().st_size == receipt["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == receipt["sha256"]


def test_tracking_od_refuses_to_mix_existing_output_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    (output_dir / "prior.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(TrackingOdError, match="absent or empty"):
        assess_tdm_orbit_determination(
            read_tdm_kvn(TDM),
            TrackingOdProblem.from_mapping(_problem()),
            output_dir=output_dir,
        )


def test_tracking_od_refuses_symlink_output_directory(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-output"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(TrackingOdError, match="must not be a symlink"):
        assess_tdm_orbit_determination(
            read_tdm_kvn(TDM),
            TrackingOdProblem.from_mapping(_problem()),
            output_dir=link,
        )
    assert not any(target.iterdir())


def test_tracking_od_preserves_symlinked_parent_aliases(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "workspace-alias"
    alias.symlink_to(target, target_is_directory=True)
    evidence = assess_tdm_orbit_determination(
        read_tdm_kvn(TDM),
        TrackingOdProblem.from_mapping(_problem()),
        output_dir=alias / "child-output",
    )
    assert evidence["status"] == "completed"
    assert (target / "child-output" / "tracking_od_evidence.json").is_file()


def test_tdm_reader_refuses_symlink_input(tmp_path: Path) -> None:
    local_source = tmp_path / "input.tdm"
    local_source.write_bytes(TDM.read_bytes())
    linked = tmp_path / "linked.tdm"
    linked.symlink_to(local_source.name)
    with pytest.raises(CcsdsTdmError, match="symbolic links|symlink"):
        read_tdm_kvn(linked)


def test_ground_od_covariance_is_converted_from_parameter_to_state_units() -> None:
    converted = np.asarray(_state_covariance_eci_km_s(np.eye(6) * 1.0e6), dtype=float)
    assert np.diag(converted) == pytest.approx([1.0, 1.0, 1.0, 1.0e-6, 1.0e-6, 1.0e-6])


def test_problem_rejects_raw_semantics_and_mismatched_object() -> None:
    raw = _problem()
    raw["measurement_semantics"] = "raw_radiometric"
    with pytest.raises(ValueError, match="raw radiometric"):
        TrackingOdProblem.from_mapping(raw)
    with pytest.raises(ValueError, match="does not match expected object_id"):
        normalize_tdm_tracking_dataset(
            read_tdm_kvn(TDM),
            stations=_station_mapping(),
            measurement_semantics="reduced_geometric",
            angle_sigma_deg=0.005,
            range_sigma_km=0.01,
            expected_object_id="OTHER-SAT",
        )


def test_problem_schema_rejects_unknown_fields_wrong_types_and_mismatched_epoch(tmp_path: Path) -> None:
    raw = _problem()
    raw["propagation"]["j22"] = True
    with pytest.raises(TrackingOdError, match="unknown fields"):
        TrackingOdProblem.from_mapping(raw)

    raw = _problem()
    raw["propagation"]["j2"] = "false"
    with pytest.raises(TrackingOdError, match="JSON boolean"):
        TrackingOdProblem.from_mapping(raw)

    raw = _problem()
    raw["initial_state_epoch_utc"] = "2000-001T11:59:00Z"
    output = tmp_path / "mismatched_epoch"
    with pytest.raises(TrackingOdError, match="exactly match"):
        assess_tdm_orbit_determination(
            read_tdm_kvn(TDM),
            TrackingOdProblem.from_mapping(raw),
            output_dir=output,
        )
    assert not output.exists()


def test_tracking_od_removes_partial_output_after_runtime_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_solve(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("synthetic estimator failure")

    monkeypatch.setattr("sim.analysis.tracking_od.solve_ground_station_measurement_od", fail_solve)
    output = tmp_path / "failed_run"
    with pytest.raises(RuntimeError, match="synthetic estimator failure"):
        assess_tdm_orbit_determination(
            read_tdm_kvn(TDM),
            TrackingOdProblem.from_mapping(_problem()),
            output_dir=output,
        )
    assert not output.exists()
