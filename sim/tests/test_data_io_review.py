from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from sim.data_io.api import convert, export_review
from sim.data_io.catalog import read_catalog
from sim.data_io.cli import main
from sim.data_io.contracts import DataIOError
from sim.data_io.materialize import materialize_scenario
from sim.data_io.products import opm_from_trajectory, tdm_from_measurements, trajectory_from_state_product
from sim.data_io.review import read_review
from sim.data_io.tables import export_review_table
from sim.data_io.time import parse_time
from sim.tests.data_io_helpers import make_review

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "format",
    [
        "oem-kvn",
        "oem-xml",
        "opm-kvn",
        "opm-xml",
        "trajectory-json",
        "trajectory-csv",
        "trajectory-parquet",
        "stk-ephemeris",
        "aem-kvn",
        "apm-kvn",
        "stk-attitude",
        "czml",
    ],
)
def test_exports_read_saved_evidence_without_modifying_it(tmp_path, format):
    if format.endswith("parquet"):
        pytest.importorskip("pyarrow")
    run = make_review(tmp_path / "run")
    db = run / "review/run.sqlite"
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    options = {"sample_index": 1}
    if format == "czml":
        options["target_frame"] = "GCRF"
    result = export_review(run, tmp_path / "output", format=format, options=options)
    assert result["source"]["source_run_id"] == "io-test-run"
    assert result["simulation_executed"] is False
    assert result["output"]["sha256"] == hashlib.sha256((tmp_path / "output").read_bytes()).hexdigest()
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_review_covariance_and_exact_selection_preserve_cross_terms(tmp_path):
    run = make_review(tmp_path / "run", covariance=True)
    selection = read_review(run, start_s=30, stop_s=60)
    assert selection.receipt["sample_indices"] == [1, 2]
    assert selection.trajectory.covariances[0, 0, 3] == 2e-7
    assert not any(row["calibrated"] for row in selection.receipt["covariance_dispositions"])
    message = opm_from_trajectory(selection.trajectory, sample_index=1)
    assert message.covariance.matrix[0][3] == 2e-7


@pytest.mark.parametrize("fault", ["hash", "epoch", "frame", "order", "nan"])
def test_review_fails_closed_on_invalid_source_semantics(tmp_path, fault):
    run = make_review(tmp_path / "run")
    with sqlite3.connect(run / "review/run.sqlite") as conn:
        if fault == "hash":
            conn.execute("UPDATE run_metadata SET config_sha256='changed'")
        elif fault == "epoch":
            raw = "{}"
            conn.execute(
                "UPDATE run_metadata SET config_json=?, config_sha256=?",
                (raw, hashlib.sha256(raw.encode()).hexdigest()),
            )
        elif fault == "frame":
            conn.execute("UPDATE object_state_frame SET state_frame='cr3bp_rotating'")
        elif fault == "order":
            conn.execute("UPDATE object_state SET time_s=0 WHERE sample_index=1")
        else:
            conn.execute("UPDATE object_state SET pos_x_eci_km=NULL WHERE sample_index=1")
    with pytest.raises((DataIOError, TypeError, ValueError)):
        read_review(run)


def test_ambiguous_object_selection_is_never_silently_first(tmp_path):
    run = make_review(tmp_path / "run")
    with sqlite3.connect(run / "review/run.sqlite") as conn:
        conn.execute("INSERT INTO object_state (sample_index,time_s,object_id) VALUES (0,0,'OTHER')")
    with pytest.raises(DataIOError, match="Explicit object"):
        read_review(run)
    assert read_review(run, object_id="SAT").trajectory.object_id == "SAT"


@pytest.mark.parametrize("format", ["csv", "json", "parquet"])
def test_review_table_exports_accessible_to_other_tools_with_provenance(tmp_path, format):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    run = make_review(tmp_path / "run")
    query = "SELECT time_s, object_id, pos_x_eci_km FROM object_state ORDER BY time_s"
    result = export_review_table(run, tmp_path / "table", query=query, format=format)
    assert result["row_count"] == 3
    assert result["query"] == query
    with pytest.raises(DataIOError, match="truncated"):
        export_review_table(run, tmp_path / "short", query=query, max_rows=1)


def test_export_table_uses_review_read_only_query_enforcement(tmp_path):
    run = make_review(tmp_path / "run")
    with pytest.raises(ValueError):
        export_review_table(run, tmp_path / "unsafe", query="DELETE FROM object_state")
    assert not (tmp_path / "unsafe").exists()


def test_catalog_materialization_validates_native_mean_elements_without_simulating(tmp_path):
    record = read_catalog(ROOT / "sim/interchange/examples/oel_sgp4_mean_elements_v3.omm", format="omm-kvn")[0]
    result = materialize_scenario(
        [record],
        tmp_path / "scenario.yaml",
        scenario_name="io_materialization",
        output_dir=tmp_path / "simulation",
        duration_s=60,
        step_s=10,
        start_epoch=parse_time("2024-01-01T00:00:00"),
    )
    assert result["validation"]["ok"]
    assert result["simulation_executed"] is False
    assert "ogp_mean_elements:" in (tmp_path / "scenario.yaml").read_text()
    assert not (tmp_path / "simulation").exists()


def test_review_state_can_seed_a_separate_validated_onp_study(tmp_path):
    run = make_review(tmp_path / "run")
    result = materialize_scenario(
        read_review(run).trajectory,
        tmp_path / "scenario.yaml",
        scenario_name="io_onp",
        output_dir=tmp_path / "new-run",
        duration_s=60,
        step_s=10,
    )
    assert result["validation"]["ok"]
    assert not (tmp_path / "new-run").exists()


def test_cli_exposes_the_same_separate_operations(tmp_path, capsys):
    assert main(["formats"]) == 0
    assert "export-review" in json.loads(capsys.readouterr().out)["operations"]
    run = make_review(tmp_path / "run")
    assert main(["export-review", str(run), str(tmp_path / "orbit.oem"), "--format", "oem-kvn"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["simulation_executed"] is False
    assert main(["export-review", str(run), str(tmp_path / "orbit.oem"), "--format", "oem-kvn"]) == 2
    assert "Refusing" in capsys.readouterr().err


def test_state_product_exports_preserve_product_quality_without_reexecution(tmp_path):
    product = ROOT / "sim/interchange/examples/state_estimate_accepted_current.json"
    trajectory = trajectory_from_state_product(product)
    assert trajectory.metadata["quality"]["disposition"] == "accepted"
    result = convert(
        product,
        tmp_path / "state.opm",
        input_format="oel-state-product",
        output_format="opm-kvn",
        options={"sample_index": 0},
    )
    assert result["simulation_executed"] is False


def test_synthetic_measurements_serialize_with_explicit_reduction_semantics():
    rows = [
        {
            "epoch_utc": "2024-01-01T00:00:00",
            "observable": keyword,
            "value": value,
            "measurement_semantics": "reduced_geometric",
        }
        for keyword, value in (("ANGLE_1", 20), ("ANGLE_2", 30), ("RANGE", 100))
    ]
    message = tdm_from_measurements(rows, station_id="SITE", object_id="SAT")
    assert len(message.segments[0].observations) == 3
    rows[0]["measurement_semantics"] = "raw"
    with pytest.raises(DataIOError, match="reduced-geometric"):
        tdm_from_measurements(rows, station_id="SITE", object_id="SAT")
