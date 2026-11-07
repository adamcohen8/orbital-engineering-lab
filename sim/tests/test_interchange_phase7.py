from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import yaml

import sim.handoff as handoff
from sim import SimulationConfig, SimulationSession
from sim.api import SimulationWorkspace
from sim.interchange.adapters.review_store import CompletedRunStateExportError
from sim.interchange.cli import main as handoff_main
from sim.interchange.materialization import materialize_onp
from sim.interchange.validation import validate_document, validate_product
from sim.review import ReviewWorkspace
from sim.security.output_writes import staged_output


def _run_config(
    output_dir: Path,
    *,
    two_objects: bool = False,
    absolute_epoch: bool = True,
    duration_s: float = 2.0,
    dt_s: float = 1.0,
    orbit_substep_s: float | None = None,
) -> dict:
    objects = {
        "target": {
            "kind": "satellite",
            "enabled": True,
            "role": "target",
            "specs": {"mass_kg": 120.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
        }
    }
    if two_objects:
        objects["deputy"] = {
            "kind": "satellite",
            "enabled": True,
            "role": "chaser",
            "specs": {"mass_kg": 90.0},
            "initial_state": {
                "position_eci_km": [7001.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.499, 0.0],
            },
        }
    simulator = {
        "duration_s": duration_s,
        "dt_s": dt_s,
        "dynamics": {
            "orbit": {
                "model": "two_body",
                "integrator": "rk4",
                **({"orbit_substep_s": orbit_substep_s} if orbit_substep_s is not None else {}),
            },
            "attitude": {"enabled": False},
            "rocket": {"enabled": False},
        },
        "termination": {"earth_impact_enabled": False},
    }
    if absolute_epoch:
        simulator["initial_jd_utc"] = 2461254.5
    return {
        "scenario_name": "phase7_completed_run",
        "scenario_description": "Completed-run continuation regression fixture",
        "metadata": {
            "owner": "public",
            "public_surface": "public-agent-workflow",
            "export_review": {
                "classification": "public-educational-synthetic",
                "provenance": "synthetic-no-customer-data",
                "approved_for_public_export": True,
            },
        },
        "objects": objects,
        "simulator": simulator,
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": True, "save_full_log": False},
            "review": {"enabled": True, "detail": "standard"},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def _run(
    output_dir: Path,
    *,
    two_objects: bool = False,
    absolute_epoch: bool = True,
    duration_s: float = 2.0,
    dt_s: float = 1.0,
    orbit_substep_s: float | None = None,
) -> Path:
    config = SimulationConfig.from_dict(
        _run_config(
            output_dir,
            two_objects=two_objects,
            absolute_epoch=absolute_epoch,
            duration_s=duration_s,
            dt_s=dt_s,
            orbit_substep_s=orbit_substep_s,
        )
    )
    result = SimulationSession.from_config(config).run()
    return Path(result.summary["review_outputs"]["sqlite"])


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_final_completed_run_state_seeds_a_new_validated_onp_study(tmp_path: Path) -> None:
    source_output = tmp_path / "source_run"
    _run(source_output)
    product_path = tmp_path / "final_state_product.json"

    emission = handoff.export_completed_run_state(
        source_output,
        output_path=product_path,
        object_id="target",
        selector="final",
    )
    product = _load(product_path)
    validation = validate_product(product, source_path=product_path)

    assert emission["status"] == "exported"
    assert emission["execution_occurred"] is False
    assert validation.valid is True
    assert validation.promotable is True
    assert product["product_kind"] == "oel.completed_run_state"
    assert product["payload"]["selection"]["sample_index"] == 2
    assert product["payload"]["selection"]["time_s"] == 2.0
    assert product["payload"]["state"]["epoch"]["value"] == pytest.approx(2461254.5 + 2.0 / 86400.0)
    assert product["payload"]["covariance"] == {
        "present": False,
        "reason": "No full state covariance matches the selected object and sample.",
    }
    assert product["data_markings"]["scope"] == "public"
    assert product["data_markings"]["contains_hidden_truth"] is True

    scenario_path = tmp_path / "continued.yaml"
    materialized = materialize_onp(
        product_path,
        scenario_name="phase7_continuation",
        scenario_path=scenario_path,
        output_dir=tmp_path / "continued_run",
        duration_s=2.0,
        dt_s=1.0,
        trust_plugins=True,
    )
    assert materialized["status"] == "materialized"
    assert materialized["execution_occurred"] is False
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    continuation = scenario["metadata"]["handoff"]["completed_run_continuation"]
    assert continuation["source_run"]["run_id"] == "phase7_completed_run"
    assert continuation["selection"]["sample_index"] == 2
    manifest = _load(Path(materialized["manifest_path"]))
    assert validate_document(manifest).valid is True
    assert manifest["materialization_options"]["completed_run_continuation"]["selection"]["sample_index"] == 2
    assert manifest["source_hashes"]["completed_run_review_store"] == product["payload"]["source_run"][
        "review_db_sha256"
    ]

    SimulationWorkspace().run(scenario_path)
    first = ReviewWorkspace.open(tmp_path / "continued_run").query(
        "SELECT pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
        "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s "
        "FROM object_state WHERE object_id = 'target' AND sample_index = 0"
    )
    assert first.row_count == 1
    assert list(first.rows[0].values()) == pytest.approx(product["payload"]["state"]["values"])


def test_completed_run_export_separates_export_time_from_missing_source_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OEL_GENERATED_UTC", raising=False)
    source_output = tmp_path / "source_run"
    _run(source_output)
    product_path = tmp_path / "state.json"
    export_time = "2026-10-23T00:00:00Z"

    emission = handoff.export_completed_run_state(
        source_output,
        output_path=product_path,
        object_id="target",
        selector="final",
        export_created_utc=export_time,
    )
    product = _load(product_path)

    assert emission["valid"] is True
    assert product["created_utc"] == export_time
    assert product["payload"]["source_run"]["generated_utc"] is None
    assert product["freshness"]["age_status"] == "not_evaluated"
    assert product["freshness"]["evaluated_utc"] is None
    assert product["quality"]["producer_status"] == "completed_run_sample_selected_timestamp_unavailable"
    assert any("producer timestamp is unavailable" in warning for warning in product["quality"]["warnings"])
    assert validate_product(product, source_path=product_path).promotable is True


def test_staged_completed_run_export_validates_logical_provenance_before_publish(tmp_path: Path) -> None:
    source_output = tmp_path / "source_run"
    _run(source_output)
    approved_root = tmp_path / "approved"
    logical_product_path = approved_root / "state.json"

    with staged_output(approved_root, approved_roots=(tmp_path,), require_empty=True) as transaction:
        staged_product_path = transaction.staging_path / logical_product_path.name
        emission = handoff.export_completed_run_state(
            source_output,
            output_path=staged_product_path,
            logical_output_path=logical_product_path,
            writer=transaction.staging,
            object_id="target",
            selector="final",
        )
        assert emission["valid"] is True
        product = _load(staged_product_path)
        assert validate_product(product, source_path=logical_product_path).promotable is True

    assert logical_product_path.is_file()


def test_completed_run_export_creates_new_destination_parent_before_provenance_validation(tmp_path: Path) -> None:
    source_output = tmp_path / "source_run"
    _run(source_output)
    product_path = tmp_path / "new" / "nested" / "final_state_product.json"

    assert product_path.parent.exists() is False
    emission = handoff.export_completed_run_state(
        source_output,
        output_path=product_path,
        object_id="target",
        selector="final",
    )

    assert emission["status"] == "exported"
    assert product_path.is_file()
    assert validate_product(_load(product_path), source_path=product_path).promotable is True


def test_sample_time_event_and_cli_selections_are_exact_and_reproducible(tmp_path: Path, capsys) -> None:
    source_output = tmp_path / "source_run"
    db_path = _run(source_output)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("analyst_marker:1", 1.0, 1, "target", "analyst_marker", "info", "selected sample", "test"),
        )
        conn.commit()

    by_index_path = tmp_path / "by_index.json"
    by_time_path = tmp_path / "by_time.json"
    by_event_path = tmp_path / "by_event.json"
    final_cli_path = tmp_path / "by_cli.json"
    by_index = handoff.export_completed_run_state(
        source_output,
        output_path=by_index_path,
        object_id="target",
        selector="sample_index",
        sample_index=1,
    )
    by_time = handoff.export_completed_run_state(
        source_output,
        output_path=by_time_path,
        object_id="target",
        selector="time_s",
        time_s=1.0,
    )
    by_event = handoff.export_completed_run_state(
        source_output,
        output_path=by_event_path,
        selector="event",
        event_id="analyst_marker:1",
    )

    assert by_index["selection"]["sample_index"] == 1
    assert by_time["selection"]["sample_index"] == 1
    assert by_event["selection"]["sample_index"] == 1
    assert by_event["selection"]["associated_event"]["event_id"] == "analyst_marker:1"
    assert len({by_index["product_id"], by_time["product_id"], by_event["product_id"]}) == 3

    exit_code = handoff_main(
        [
            "export-state",
            str(source_output),
            "--output",
            str(final_cli_path),
            "--object-id",
            "target",
            "--sample",
            "final",
            "--json",
        ]
    )
    cli_result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert cli_result["selection"]["sample_index"] == 2
    assert final_cli_path.is_file()


def test_completed_run_export_fails_closed_on_ambiguous_or_missing_evidence(tmp_path: Path) -> None:
    multi_output = tmp_path / "multi"
    _run(multi_output, two_objects=True)
    with pytest.raises(CompletedRunStateExportError, match="Object selection is ambiguous"):
        handoff.export_completed_run_state(
            multi_output,
            output_path=tmp_path / "ambiguous.json",
            selector="final",
        )
    with pytest.raises(CompletedRunStateExportError, match="Expected exactly one sample_index 99"):
        handoff.export_completed_run_state(
            multi_output,
            output_path=tmp_path / "missing.json",
            object_id="target",
            selector="sample_index",
            sample_index=99,
        )

    no_epoch_output = tmp_path / "no_epoch"
    _run(no_epoch_output, absolute_epoch=False)
    with pytest.raises(CompletedRunStateExportError, match="initial_jd_utc"):
        handoff.export_completed_run_state(
            no_epoch_output,
            output_path=tmp_path / "no_epoch.json",
            object_id="target",
            selector="final",
        )

    override_path = tmp_path / "explicit_epoch.json"
    exported = handoff.export_completed_run_state(
        no_epoch_output,
        output_path=override_path,
        object_id="target",
        selector="final",
        epoch_jd_utc=2461254.5,
    )
    override_product = _load(override_path)
    assert exported["status"] == "exported"
    assert override_product["payload"]["source_run"]["initial_jd_utc_source"] == (
        "explicit_export_override"
    )
    assert override_product["payload"]["state"]["epoch"]["value"] == pytest.approx(
        2461254.5 + 2.0 / 86400.0
    )

    wrong_frame_output = tmp_path / "wrong_frame"
    db_path = _run(wrong_frame_output)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE object_state_frame SET state_frame = 'cr3bp_rotating' WHERE object_id = 'target'")
        conn.commit()
    with pytest.raises(CompletedRunStateExportError, match="requires canonical ECI"):
        handoff.export_completed_run_state(
            wrong_frame_output,
            output_path=tmp_path / "wrong_frame.json",
            object_id="target",
            selector="final",
        )


def test_matching_full_covariance_is_bound_to_the_selected_sample(tmp_path: Path) -> None:
    source_output = tmp_path / "source_run"
    db_path = _run(source_output)
    matrix = np.diag([1e-4, 2e-4, 3e-4, 1e-8, 2e-8, 3e-8])
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO object_state_covariance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                1.0,
                "target",
                "ECI",
                json.dumps(["x", "y", "z", "vx", "vy", "vz"]),
                json.dumps(["km", "km", "km", "km/s", "km/s", "km/s"]),
                json.dumps(matrix.tolist()),
                1,
                0,
                None,
                "phase7_test",
            ),
        )
        conn.commit()

    product_path = tmp_path / "with_covariance.json"
    emission = handoff.export_completed_run_state(
        source_output,
        output_path=product_path,
        object_id="target",
        selector="sample_index",
        sample_index=1,
    )
    product = _load(product_path)

    assert emission["covariance_present"] is True
    covariance = product["payload"]["covariance"]
    assert covariance["epoch_jd_utc"] == product["payload"]["state"]["epoch"]["value"]
    np.testing.assert_allclose(covariance["matrix"], matrix)
    assert validate_product(product, source_path=product_path).promotable is True


def test_completed_run_product_detects_source_review_store_mutation(tmp_path: Path) -> None:
    source_output = tmp_path / "source_run"
    db_path = _run(source_output)
    product_path = tmp_path / "state.json"
    handoff.export_completed_run_state(
        source_output,
        output_path=product_path,
        object_id="target",
        selector="final",
    )
    product = _load(product_path)
    assert validate_product(product, source_path=product_path).valid is True

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE events SET message = message")
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("post_export", 2.0, 2, "target", "marker", "info", "changed", "test"),
        )
        conn.commit()

    validation = validate_product(product, source_path=product_path)
    assert validation.valid is False
    assert any(issue.code == "provenance.source_hash_mismatch" for issue in validation.errors)


def test_completed_run_continuation_bounds_source_substep_to_requested_consumer_cadence(
    tmp_path: Path,
) -> None:
    source_output = tmp_path / "source_run"
    _run(source_output, duration_s=240.0, dt_s=120.0, orbit_substep_s=120.0)
    product_path = tmp_path / "state.json"
    handoff.export_completed_run_state(
        source_output,
        output_path=product_path,
        object_id="target",
        selector="final",
    )

    materialized = materialize_onp(
        product_path,
        scenario_name="bounded_substep_continuation",
        scenario_path=tmp_path / "continuation.yaml",
        output_dir=tmp_path / "continuation_run",
        duration_s=60.0,
        dt_s=30.0,
        trust_plugins=True,
    )

    assert materialized["status"] == "materialized"
    scenario = yaml.safe_load(Path(materialized["scenario_path"]).read_text(encoding="utf-8"))
    assert scenario["simulator"]["dynamics"]["orbit"]["orbit_substep_s"] == 30.0
    manifest = _load(Path(materialized["manifest_path"]))
    assert {
        "field": "simulator.dynamics.orbit.orbit_substep_s",
        "source_value": 120.0,
        "output_value": 30.0,
        "reason": "Bound the integration substep to the explicitly requested consumer dt_s.",
    } in manifest["overrides"]
    assert handoff.compare_handoff(product_path, materialized["scenario_path"])["status"] == "equivalent"


def test_phase7_schema_is_closed_and_facade_exports_focused_owner() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "interchange" / "schemas" / "oel-completed-run-state-v1.schema.json"
    schema = _load(schema_path)

    assert schema["additionalProperties"] is False
    assert schema["properties"]["state"]["properties"]["frame"]["const"] == "ECI"
    assert "explicit_export_override" in schema["properties"]["source_run"]["properties"][
        "initial_jd_utc_source"
    ]["enum"]
    assert handoff.export_completed_run_state.__module__ == "sim.interchange.adapters.review_store"
