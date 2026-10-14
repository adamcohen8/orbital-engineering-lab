from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.dynamics.orbit.elements import coe_to_rv_eci, rv_to_coe_eci
from sim.plotting.summary_outputs import render_summary_outputs
from sim.reporting.review_store import (
    _batched_orbital_elements,
    _create_schema,
    _insert_object_orbital_elements,
    _insert_run_metadata,
)
from sim.reporting.single_run_artifacts import _previous_owned_artifacts
from sim.review.plot_planning import review_plot_plan_id
from sim.review.plotting import EvidencePlotter, ReviewPlotSpec, save_review_plot
from sim.review.workspace import ReviewWorkspace
from sim.utils.io import json_safe, sha256_file, write_json


def test_streaming_json_matches_historical_sanitized_bytes(tmp_path: Path) -> None:
    payload = {
        "path": tmp_path / "evidence.json",
        "values": [np.int64(7), np.float32(1.25), np.nan, np.inf, -np.inf],
        "nested": {3: (True, None, "OEL \u2604")},
        "collision": {1: "integer", "1": "string", "tail": "preserved"},
    }
    expected = json.dumps(json_safe(payload), indent=2, allow_nan=False)
    output = tmp_path / "output.json"

    write_json(str(output), payload)

    assert output.read_text(encoding="utf-8") == expected
    assert output.read_text(encoding="utf-8").count('"1":') == 1


def test_public_plot_api_cannot_accept_unverified_query_results() -> None:
    assert "result" not in inspect.signature(save_review_plot).parameters
    assert "result" not in inspect.signature(EvidencePlotter.dry_run).parameters


def test_sha256_file_matches_whole_file_digest(tmp_path: Path) -> None:
    path = tmp_path / "evidence.bin"
    payload = bytes(range(251)) * 123
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=97) == hashlib.sha256(payload).hexdigest()


def test_review_identity_includes_uncheckpointed_wal_content(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    db_path = review_dir / "run.sqlite"
    with sqlite3.connect(db_path) as writer:
        assert writer.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
        writer.execute("CREATE TABLE evidence(value TEXT)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        workspace = ReviewWorkspace.open(tmp_path)
        initial_raw_hash = sha256_file(db_path)
        initial_identity = workspace.evidence_identity()
        spec = ReviewPlotSpec(sql="SELECT rowid, value FROM evidence", x_column="rowid", y_columns=[])
        initial_plan_id = review_plot_plan_id(workspace, spec)

        writer.execute("INSERT INTO evidence VALUES ('new-wal-row')")
        writer.commit()
        wal_path = db_path.with_name(db_path.name + "-wal")
        wal_hash = sha256_file(wal_path)
        updated_identity = workspace.evidence_identity()

        assert sha256_file(db_path) == initial_raw_hash
        assert sha256_file(wal_path) == wal_hash
        assert updated_identity["sha256"] != initial_identity["sha256"]
        assert review_plot_plan_id(workspace, spec) != initial_plan_id
        assert workspace.evidence_identity() == updated_identity
        assert workspace.query("SELECT value FROM evidence").rows == [{"value": "new-wal-row"}]
        workspace.close()


def test_precomputed_config_identity_must_match_configuration(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        scenario_name="actual",
        scenario_description="",
        to_dict=lambda: {"scenario_name": "actual"},
    )
    forged_json = '{"scenario_name":"forged"}'
    forged_sha256 = hashlib.sha256(forged_json.encode("utf-8")).hexdigest()

    with sqlite3.connect(":memory:") as conn:
        _create_schema(conn)
        with pytest.raises(ValueError, match="config_json does not match"):
            _insert_run_metadata(
                conn,
                cfg=cfg,
                summary={"scenario_name": "actual"},
                outdir=tmp_path,
                generated_utc="2026-01-01T00:00:00Z",
                config_json=forged_json,
                config_sha256=forged_sha256,
            )

        canonical_json = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with pytest.raises(ValueError, match="config_sha256 does not match"):
            _insert_run_metadata(
                conn,
                cfg=cfg,
                summary={"scenario_name": "actual"},
                outdir=tmp_path,
                generated_utc="2026-01-01T00:00:00Z",
                config_json=canonical_json,
                config_sha256="0" * 64,
            )


def test_batched_orbital_elements_are_scalar_exact_for_ordinary_states() -> None:
    coes = [
        (7000.0, 0.001, 51.6, 32.0, 18.0, 4.0),
        (42164.0, 0.15, 0.25, 241.0, 73.0, 215.0),
        (12000.0, 0.4, 89.0, 179.0, 301.0, 97.0),
        (7200.0, 0.0, 28.5, 80.0, 0.0, 190.0),
        (7000.0, 0.0, 0.0, 0.0, 0.0, 123.0),
        (9000.0, 0.2, 0.0, 0.0, 37.0, 100.0),
        (9000.0, 0.2, 180.0, 0.0, 37.0, 100.0),
        (7000.0, 0.0, 180.0, 0.0, 0.0, 123.0),
    ]
    states = []
    expected = []
    for a_km, ecc, inc_deg, raan_deg, argp_deg, true_anomaly_deg in coes:
        r, v = coe_to_rv_eci(
            a_km=a_km,
            ecc=ecc,
            inc_deg=inc_deg,
            raan_deg=raan_deg,
            argp_deg=argp_deg,
            true_anomaly_deg=true_anomaly_deg,
        )
        states.append(np.concatenate((r, v)))
        scalar = rv_to_coe_eci(r, v)
        expected.append(
            (
                scalar.a_km,
                scalar.ecc,
                scalar.inc_deg,
                scalar.raan_deg,
                scalar.argp_deg,
                scalar.true_anomaly_deg,
            )
        )

    assert _batched_orbital_elements(np.asarray(states)) == expected


def test_batched_orbital_elements_leave_edge_cases_for_scalar_fallback() -> None:
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7000.0, 0.0, 0.0, 12.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0, 0.0, 7.5, 0.0],
        ]
    )

    assert _batched_orbital_elements(states) == [None, None, None]


@pytest.mark.parametrize("ecc, inc_deg", [(0.05, 51.6), (0.2, 0.0), (0.2, 180.0), (0.0, 180.0)])
def test_batched_orbital_elements_preserve_sqlite_bytes(tmp_path: Path, monkeypatch, ecc: float, inc_deg: float) -> None:
    states = []
    for true_anomaly_deg in np.linspace(0.0, 359.0, 128):
        r, v = coe_to_rv_eci(
            a_km=7200.0,
            ecc=ecc,
            inc_deg=inc_deg,
            raan_deg=23.0,
            argp_deg=147.0,
            true_anomaly_deg=float(true_anomaly_deg),
        )
        states.append(np.concatenate((r, v)))
    truth_hist = {"satellite": np.asarray(states)}
    t_s = np.arange(len(states), dtype=float)
    baseline = tmp_path / "baseline.sqlite"
    optimized = tmp_path / "optimized.sqlite"

    with monkeypatch.context() as patcher:
        patcher.setattr(
            "sim.reporting.review_store._batched_orbital_elements",
            lambda values: [None] * len(values),
        )
        with sqlite3.connect(baseline) as conn:
            _create_schema(conn)
            _insert_object_orbital_elements(
                conn,
                t_s=t_s,
                truth_hist=truth_hist,
                object_state_frames={"satellite": "eci"},
            )
    with sqlite3.connect(optimized) as conn:
        _create_schema(conn)
        _insert_object_orbital_elements(
            conn,
            t_s=t_s,
            truth_hist=truth_hist,
            object_state_frames={"satellite": "eci"},
        )

    assert sha256_file(optimized) == sha256_file(baseline)


def test_previous_artifacts_skip_fixed_hash_but_verify_dynamic_hash(
    tmp_path: Path, monkeypatch
) -> None:
    fixed = tmp_path / "master_run_log.json"
    dynamic = tmp_path / "custom_plot.png"
    fixed.write_bytes(b"fixed")
    dynamic.write_bytes(b"dynamic")
    (tmp_path / ".oel_run_artifacts.json").write_text(
        json.dumps(
            {
                "version": 2,
                "paths": [
                    {"path": fixed.name, "sha256": hashlib.sha256(b"fixed").hexdigest()},
                    {"path": dynamic.name, "sha256": hashlib.sha256(b"dynamic").hexdigest()},
                ],
            }
        ),
        encoding="utf-8",
    )
    hashed: list[Path] = []

    def tracked_hash(path: str | Path) -> str:
        resolved = Path(path)
        hashed.append(resolved)
        return hashlib.sha256(resolved.read_bytes()).hexdigest()

    monkeypatch.setattr("sim.reporting.single_run_artifacts.sha256_file", tracked_hash)

    assert _previous_owned_artifacts(tmp_path) == {fixed, dynamic}
    assert hashed == [dynamic]


def test_summary_plot_reuses_payload_ground_access(tmp_path: Path, monkeypatch) -> None:
    access = {"station": {"object": {"access": np.array([True, False])}}}
    captured: dict[str, object] = {}

    def fail_recompute(**_kwargs):
        raise AssertionError("ground access should not be recomputed")

    def capture_plot(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("sim.plotting.summary_outputs.evaluate_ground_station_access", fail_recompute)
    monkeypatch.setattr("sim.plotting.plot_ground_station_access", capture_plot)
    context = SimpleNamespace(
        mode="save",
        cfg=SimpleNamespace(
            ground_stations=[],
            simulator=SimpleNamespace(initial_jd_utc=0.0, dynamics={}),
            outputs=SimpleNamespace(plots={}),
        ),
        t_s=np.array([0.0, 1.0]),
        truth_hist={"object": np.zeros((2, 6))},
        target_reference_orbit_truth=None,
        thrust_hist={},
        desired_attitude_hist=None,
        knowledge_hist={},
        outdir=tmp_path,
        belief_hist=None,
        knowledge_measurement_hist=None,
        bridge_hist=None,
        reentry_metrics=None,
        figure_ids=("ground_station_access",),
        frame_context=None,
        reference_object_id="object",
        reference_object_label=None,
        reference_truth=np.zeros((2, 6)),
        ric_truth_hist={},
        dpi=72,
        show=False,
        close=True,
        save_enabled=True,
        draw_ground_track_map=False,
        plot_fns={"plot_orbit_eci": None, "plot_attitude_tumble": None},
        object_state_frames={"object": "eci"},
        ground_station_access=access,
    )

    outputs = render_summary_outputs(context)

    assert captured["ground_station_access"] is access
    assert outputs == {"ground_station_access": str(tmp_path / "ground_station_access.png")}
