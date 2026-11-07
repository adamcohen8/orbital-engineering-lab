from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim.analysis.conjunction_workflow import assess_cdm_message
from sim.interchange.ccsds_cdm import (
    MAX_CDM_BYTES,
    CcsdsCdmError,
    compare_cdm,
    inspect_cdm,
    parse_cdm_kvn,
    serialize_cdm_kvn,
)


def _cdm() -> str:
    covariance_keys = (
        "CR_R",
        "CT_R",
        "CT_T",
        "CN_R",
        "CN_T",
        "CN_N",
        "CRDOT_R",
        "CRDOT_T",
        "CRDOT_N",
        "CRDOT_RDOT",
        "CTDOT_R",
        "CTDOT_T",
        "CTDOT_N",
        "CTDOT_RDOT",
        "CTDOT_TDOT",
        "CNDOT_R",
        "CNDOT_T",
        "CNDOT_N",
        "CNDOT_RDOT",
        "CNDOT_TDOT",
        "CNDOT_NDOT",
    )
    diagonal = {"CR_R", "CT_T", "CN_N", "CRDOT_RDOT", "CTDOT_TDOT", "CNDOT_NDOT"}
    lines = [
        "CCSDS_CDM_VERS = 1.0",
        "CREATION_DATE = 2026-01-01T00:00:00",
        "ORIGINATOR = OEL",
        "TCA = 2026-01-02T00:00:00",
        "MISS_DISTANCE = 1000 [m]",
        "RELATIVE_SPEED = 10 [m/s]",
        "RELATIVE_POSITION_R = -1000 [m]",
        "RELATIVE_POSITION_T = 0 [m]",
        "RELATIVE_POSITION_N = 0 [m]",
        "RELATIVE_VELOCITY_R = 0 [m/s]",
        "RELATIVE_VELOCITY_T = -10 [m/s]",
        "RELATIVE_VELOCITY_N = 0 [m/s]",
    ]
    for number in (1, 2):
        lines += [
            f"OBJECT = OBJECT{number}",
            f"OBJECT_DESIGNATOR = 9000{number}",
            "CATALOG_NAME = SATCAT",
            f"OBJECT_NAME = SYNTHETIC-{number}",
            f"INTERNATIONAL_DESIGNATOR = 2026-00{number}A",
            "EPHEMERIS_NAME = NONE",
            "COVARIANCE_METHOD = CALCULATED",
            "MANEUVERABLE = YES",
            "REF_FRAME = EME2000",
            f"X = {7000 + number} [km]",
            "Y = 0 [km]",
            "Z = 0 [km]",
            "X_DOT = 0 [km/s]",
            f"Y_DOT = {7.49 + 0.01 * number} [km/s]",
            "Z_DOT = 0 [km/s]",
        ]
        lines += [f"{key} = {100.0 if key in diagonal else 0.0}" for key in covariance_keys]
    return "\n".join(lines) + "\n"


def test_cdm_round_trip_is_semantically_stable() -> None:
    message = parse_cdm_kvn(_cdm())
    reparsed = parse_cdm_kvn(serialize_cdm_kvn(message))
    assert compare_cdm(message, reparsed)["equivalent"] is True
    inspection = inspect_cdm(reparsed)
    assert inspection["analysis_ready"] is True
    assert inspection["semantic_checks"]["miss_distance_minus_position_norm_m"] == pytest.approx(0.0)
    assert np.asarray(message.objects[0].covariance_rtn_si).shape == (6, 6)


def test_direct_text_parser_enforces_byte_ceiling() -> None:
    with pytest.raises(CcsdsCdmError, match="byte limit"):
        parse_cdm_kvn(" " * (MAX_CDM_BYTES + 1))


def test_cdm_assessment_recomputes_geometry_and_probability() -> None:
    evidence = assess_cdm_message(parse_cdm_kvn(_cdm()), primary_radius_m=5.0, secondary_radius_m=5.0)
    assert evidence["computed"]["miss_distance_m"] == pytest.approx(1000.0)
    assert evidence["computed"]["relative_speed_m_s"] == pytest.approx(10.0)
    assert 0.0 <= evidence["computed"]["probability"]["collision_probability"] <= 1.0


def test_cdm_rejects_wrong_declared_units() -> None:
    with pytest.raises(CcsdsCdmError, match="unit"):
        parse_cdm_kvn(_cdm().replace("MISS_DISTANCE = 1000 [m]", "MISS_DISTANCE = 1 [km]"))


def test_cdm_rejects_extended_covariance_and_non_psd_covariance() -> None:
    with pytest.raises(CcsdsCdmError, match="outside the public 6x6"):
        parse_cdm_kvn(_cdm() + "CDRG_R = 1\n")
    bad = _cdm().replace("CT_T = 100.0", "CT_T = -100.0", 1)
    with pytest.raises(CcsdsCdmError, match="not positive semidefinite"):
        parse_cdm_kvn(bad)


def test_cdm_validation_rejects_nonfinite_programmatic_scalars_and_states() -> None:
    message = parse_cdm_kvn(_cdm())
    invalid_scalar = replace(
        message,
        relative=replace(message.relative, miss_distance_m=float("nan")),
    )
    with pytest.raises(CcsdsCdmError, match="MISS_DISTANCE.*finite"):
        serialize_cdm_kvn(invalid_scalar)

    first = message.objects[0]
    invalid_state = replace(
        message,
        objects=(replace(first, state_eci_km_km_s=(float("nan"), *first.state_eci_km_km_s[1:])), message.objects[1]),
    )
    with pytest.raises(CcsdsCdmError, match="state.*finite"):
        serialize_cdm_kvn(invalid_state)
