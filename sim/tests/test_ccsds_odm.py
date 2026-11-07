from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from sim.ccsds import (
    CcsdsOdmError,
    OmmMessage,
    OpmMessage,
    compare_odm,
    inspect_odm,
    opm_to_mission_input_packet,
    parse_odm_kvn,
    read_odm_kvn,
    serialize_odm_kvn,
)
from tools.validate_ccsds_odm import _load_data_receipt as load_odm_data_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]
OPM = REPO_ROOT / "sim/interchange/examples/oel_earth_eme2000_utc_v3.opm"
OMM = REPO_ROOT / "sim/interchange/examples/oel_sgp4_mean_elements_v3.omm"
HISTORICAL_MANIFEST = REPO_ROOT / "sim/interchange/examples/ccsds_odm_validation_manifest.json"
CURRENT_MANIFEST = REPO_ROOT / "sim/interchange/examples/ccsds_odm_validation_manifest_v036.json"
# The source-bound v0.36 receipt is private audit evidence.  Public exports
# deliberately omit it and continue to exercise the retained public manifest
# and receipt instead of relabelling the local replay as independent evidence.
MANIFEST = CURRENT_MANIFEST if CURRENT_MANIFEST.is_file() else HISTORICAL_MANIFEST


def test_opm_parses_round_trips_and_is_ready_for_bounded_import() -> None:
    message = read_odm_kvn(OPM)
    inspection = inspect_odm(message)

    assert isinstance(message, OpmMessage)
    assert inspection["valid"] is True
    assert inspection["mission_input_ready"] is True
    assert inspection["maneuver_count"] == 1
    assert inspection["covariance_present"] is True
    assert message.source_sha256 == hashlib.sha256(OPM.read_bytes()).hexdigest()
    assert compare_odm(message, parse_odm_kvn(serialize_odm_kvn(message)))["status"] == "equivalent"


def test_omm_preserves_mean_elements_without_silent_state_conversion() -> None:
    message = read_odm_kvn(OMM)
    inspection = inspect_odm(message)

    assert isinstance(message, OmmMessage)
    assert inspection["mean_element_theory"] == "SGP4"
    assert inspection["mission_input_ready"] is False
    assert message.mean_elements["MEAN_MOTION"] == "15.25"
    assert message.tle_parameters["BSTAR"] == "1.2e-5"
    assert compare_odm(message, parse_odm_kvn(serialize_odm_kvn(message)))["status"] == "equivalent"


def test_opm_import_preserves_maneuver_and_covariance_as_nonexecuted_provenance() -> None:
    packet = opm_to_mission_input_packet(OPM).to_dict()
    source = packet["objects"]["2024-001A"]["ephemeris"]["source_metadata"]

    assert source["maneuvers_preserved_not_scheduled"][0]["MAN_DV_2"] == "0.001"
    assert source["covariance_preserved_not_calibrated"][0][0] == pytest.approx(1.0e-6)
    assert packet["objects"]["2024-001A"]["initial_state"]["position_eci_km"] == [7000.0, 120.0, 30.0]


@pytest.mark.parametrize("unit", ["m", "bananas"])
def test_opm_import_rejects_noncanonical_state_units(unit: str) -> None:
    text = OPM.read_text(encoding="utf-8").replace("X = 7000 [km]", f"X = 7000 [{unit}]")

    with pytest.raises(CcsdsOdmError, match="X unit must be km"):
        parse_odm_kvn(text)


def test_covariance_units_are_validated_and_preserved_by_round_trip() -> None:
    text = OPM.read_text(encoding="utf-8").replace("CX_X = 1e-6", "CX_X = 1e-6 [km**2]")
    message = parse_odm_kvn(text)
    serialized = serialize_odm_kvn(message)

    assert "CX_X = 9.9999999999999995e-07 [km**2]" in serialized
    assert compare_odm(message, parse_odm_kvn(serialized))["status"] == "equivalent"
    with pytest.raises(CcsdsOdmError, match=r"CX_X unit must be km\*\*2"):
        parse_odm_kvn(text.replace("[km**2]", "[m**2]", 1))


def test_user_defined_units_are_preserved_and_compared_semantically() -> None:
    text = OPM.read_text(encoding="utf-8").replace(
        "USER_DEFINED_EARTH_MODEL = WGS-84",
        "USER_DEFINED_EARTH_MODEL = 1 [model]",
    )
    message = parse_odm_kvn(text)
    serialized = serialize_odm_kvn(message)

    assert "USER_DEFINED_EARTH_MODEL = 1 [model]" in serialized
    assert compare_odm(message, parse_odm_kvn(serialized))["status"] == "equivalent"
    changed = parse_odm_kvn(serialized.replace("1 [model]", "1 [other-model]"))
    assert compare_odm(message, changed)["status"] == "different"


def test_numeric_formatting_does_not_change_semantic_comparison() -> None:
    changed = parse_odm_kvn(OPM.read_text(encoding="utf-8").replace("X = 7000 [km]", "X = 7.000e3 [km]"))

    assert compare_odm(read_odm_kvn(OPM), changed)["status"] == "equivalent"


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ("X = NaN [km]", "finite decimal"),
        ("UNSUPPORTED_KEY = 1", "Unsupported OPM keywords"),
        ("CY_Y =", "empty ODM key or value"),
        ("TIME_SYSTEM = TDB", "supports UTC, TAI, TT, and UT1"),
        ("OUT_OF_ORDER_COVARIANCE", "out-of-order OPM keyword COV_REF_FRAME"),
    ],
)
def test_malformed_or_out_of_profile_opm_fails_closed(replacement: str, match: str) -> None:
    text = OPM.read_text(encoding="utf-8")
    if replacement.startswith("X ="):
        text = text.replace("X = 7000 [km]", replacement)
    elif replacement.startswith("CY_Y"):
        text = text.replace("CY_Y = 2e-6", replacement)
    elif replacement.startswith("TIME_SYSTEM"):
        text = text.replace("TIME_SYSTEM = UTC", replacement)
    elif replacement == "OUT_OF_ORDER_COVARIANCE":
        covariance = text[text.index("COV_REF_FRAME = EME2000") : text.index("MAN_EPOCH_IGNITION")]
        text = text.replace(covariance, "").replace("USER_DEFINED_EARTH_MODEL", covariance + "USER_DEFINED_EARTH_MODEL")
    else:
        text = text.replace("USER_DEFINED_SOURCE_NOTE = PUBLIC VALIDATION FIXTURE", replacement)
    with pytest.raises(CcsdsOdmError, match=match):
        parse_odm_kvn(text)


def test_non_eme2000_opm_is_inspectable_but_not_import_ready() -> None:
    message = parse_odm_kvn(OPM.read_text(encoding="utf-8").replace("REF_FRAME = EME2000", "REF_FRAME = GCRF"))

    assert inspect_odm(message)["mission_input_ready"] is False
    with pytest.raises(CcsdsOdmError, match="explicit conversion"):
        opm_to_mission_input_packet(message)


def test_tiny_negative_odm_variance_fails_closed() -> None:
    text = OPM.read_text(encoding="utf-8").replace("CX_X = 1e-6", "CX_X = -1e-13")

    with pytest.raises(CcsdsOdmError, match="diagonal variances"):
        parse_odm_kvn(text)


def _assert_retained_selected_field_replay(historical: dict[str, object]) -> None:
    """Replay the public parser against the retained selected-field vector.

    The historical report is an input vector, not a reason to skip current
    parser execution.  This keeps the public export useful after the
    audit-only current-source receipt is excluded.
    """

    orekit = historical["orekit"]
    assert isinstance(orekit, dict)
    opm = read_odm_kvn(OPM)
    omm = read_odm_kvn(OMM)

    assert isinstance(opm, OpmMessage)
    assert isinstance(omm, OmmMessage)
    assert math.isclose(float(opm.header.version), float(orekit["opm_version"]), abs_tol=1e-15)
    assert opm.metadata.object_id == orekit["opm_object_id"]
    assert opm.metadata.ref_frame == orekit["opm_frame"]
    assert len(opm.maneuvers) == int(orekit["opm_maneuver_count"])
    assert opm.covariance is not None
    assert orekit["opm_covariance_present"] == "true"
    expected_position = [float(value) for value in str(orekit["opm_position_m"]).split(",")]
    expected_velocity = [float(value) for value in str(orekit["opm_velocity_m_s"]).split(",")]
    actual_position = [float(opm.state[key]) * 1000.0 for key in ("X", "Y", "Z")]
    actual_velocity = [float(opm.state[key]) * 1000.0 for key in ("X_DOT", "Y_DOT", "Z_DOT")]
    assert all(math.isclose(actual, expected, abs_tol=1e-9) for actual, expected in zip(actual_position, expected_position))
    assert all(math.isclose(actual, expected, abs_tol=1e-12) for actual, expected in zip(actual_velocity, expected_velocity))

    assert math.isclose(float(omm.header.version), float(orekit["omm_version"]), abs_tol=1e-15)
    assert omm.metadata.object_id == orekit["omm_object_id"]
    assert omm.metadata.ref_frame == orekit["omm_frame"]
    assert omm.metadata.mean_element_theory == orekit["omm_theory"]
    assert math.isclose(
        float(omm.mean_elements["MEAN_MOTION"]) * 2.0 * math.pi / 86400.0,
        float(orekit["omm_mean_motion_rad_s"]),
        abs_tol=1e-18,
    )
    assert math.isclose(float(omm.mean_elements["ECCENTRICITY"]), float(orekit["omm_eccentricity"]), abs_tol=1e-15)
    assert math.isclose(
        math.radians(float(omm.mean_elements["INCLINATION"])),
        float(orekit["omm_inclination_rad"]),
        abs_tol=1e-15,
    )
    assert omm.tle_parameters["NORAD_CAT_ID"] == orekit["omm_norad_id"]
    assert compare_odm(opm, parse_odm_kvn(serialize_odm_kvn(opm)))["status"] == "equivalent"
    assert compare_odm(omm, parse_odm_kvn(serialize_odm_kvn(omm)))["status"] == "equivalent"


def test_odm_validation_manifest_is_hash_bound_and_retained_report_passes() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = (MANIFEST.parent / artifact["path"]).resolve()
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    if CURRENT_MANIFEST.is_file():
        local_report = json.loads((MANIFEST.parent / "ccsds_odm_v036_local_replay.json").read_text(encoding="utf-8"))
        historical_path = MANIFEST.parent / local_report["historical_receipt"]["path"]
    else:
        # Public exports retain only the historical receipt; the replay above
        # still exercises the current exported parser against its vectors.
        local_report = None
        historical_path = MANIFEST.parent / "ccsds_odm_orekit_13_1_7_acceptance.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    _assert_retained_selected_field_replay(historical)
    assert historical["status"] == "pass"
    assert historical["runtime"]["orekit_version"] == "13.1.7"
    if local_report is None:
        return
    assert hashlib.sha256(historical_path.read_bytes()).hexdigest() == local_report["historical_receipt"]["sha256"]
    assert local_report["status"] == "pass"
    assert local_report["schema"] == "oel.ccsds-opm-omm-local-replay.v1"
    assert local_report["sources"]["implementation"]["sha256"] == hashlib.sha256(
        (MANIFEST.parent / local_report["sources"]["implementation"]["path"]).read_bytes()
    ).hexdigest()
    assert all(local_report["checks"].values())
    assert local_report["retained_external_fields"] == {
        "opm": [
            "version", "object_id", "frame", "maneuver_count", "position",
            "velocity", "covariance_presence",
        ],
        "omm": [
            "version", "object_id", "frame", "mean_element_theory",
            "mean_motion", "eccentricity", "inclination", "norad_id",
        ],
    }
    mapped_checks = {
        "opm_version", "opm_object_id", "opm_frame", "opm_maneuver_count",
        "opm_position", "opm_velocity", "opm_covariance_presence",
        "omm_version", "omm_object_id", "omm_frame", "omm_theory",
        "omm_mean_motion", "omm_eccentricity", "omm_inclination", "omm_norad_id",
    }
    assert set(local_report["checks"]) - mapped_checks == {"opm_roundtrip", "omm_roundtrip"}


def test_odm_reference_refresh_rejects_stale_orekit_data_receipt(tmp_path: Path) -> None:
    data_dir = tmp_path / "orekit-data"
    data_dir.mkdir()
    (data_dir / "tai-utc.dat").write_text("retained data\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"revision": "fixture", "source": "fixture", "tree_sha256": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="data tree does not match its receipt"):
        load_odm_data_receipt(receipt, data_dir=data_dir)
