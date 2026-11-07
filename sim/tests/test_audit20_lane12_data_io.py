from __future__ import annotations

from pathlib import Path

import pytest

from sim.data_io.api import encode_data
from sim.data_io.contracts import DataIOError
from sim.data_io.products import tdm_from_measurements
from sim.interchange.ccsds_cdm import CcsdsCdmError, read_cdm_kvn
from sim.interchange.ccsds_odm import CcsdsOdmError, read_odm_kvn
from sim.interchange.ccsds_tdm import CcsdsTdmError, read_tdm_kvn


def test_direct_maneuver_plan_encoding_rejects_execution_authorization() -> None:
    plan = {
        "schema": "oel.maneuver-plan.v1",
        "object_id": "chaser",
        "center": "EARTH",
        "maneuvers": [],
        "execution_authorized": True,
    }

    with pytest.raises(DataIOError, match="execution_authorized=false"):
        encode_data(plan, format="maneuver-plan-json")


def test_direct_maneuver_plan_encoding_preserves_non_execution_contract() -> None:
    plan = {
        "schema": "oel.maneuver-plan.v1",
        "object_id": "chaser",
        "center": "EARTH",
        "maneuvers": [],
        "execution_authorized": False,
    }

    encoded, metadata = encode_data(plan, format="maneuver-plan-json")

    assert b'"execution_authorized": false' in encoded
    assert metadata == {"execution_authorized": False}


def test_generic_measurement_bundle_cannot_relabel_explicit_row_identity() -> None:
    rows = {
        "rows": [
            {
                "epoch_utc": "2024-01-01T00:00:00Z",
                "observable": "RANGE",
                "value": 100.0,
                "measurement_semantics": "reduced_geometric",
                "station_id": "OTHER_SITE",
                "object_id": "OTHER_SAT",
            }
        ]
    }

    with pytest.raises(DataIOError, match="station identity"):
        tdm_from_measurements(rows, station_id="SITE", object_id="SAT")


@pytest.mark.parametrize(
    ("reader", "error"),
    [(read_odm_kvn, CcsdsOdmError), (read_cdm_kvn, CcsdsCdmError), (read_tdm_kvn, CcsdsTdmError)],
)
def test_ccsds_file_readers_use_bounded_reads(reader, error, tmp_path, monkeypatch) -> None:
    oversized = tmp_path / "oversized.ccsds"
    oversized.write_bytes(b"x")
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda *_args, **_kwargs: pytest.fail("CCSDS reader used unbounded Path.read_bytes()."),
    )

    with pytest.raises(error, match="exceeds the"):
        reader(oversized, max_bytes=0)
