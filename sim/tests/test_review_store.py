from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import sim.reporting.review_store as review_store_module
import sim.review.plotting as review_plotting_module
from sim import ReviewWorkspace as TopLevelReviewWorkspace
from sim import SimulationConfig, SimulationSession
from sim.config import scenario_config_from_dict
from sim.execution import run_simulation_config_file
from sim.reporting.review_store import (
    REVIEW_SCHEMA_COMPATIBILITY_POLICY,
    REVIEW_SCHEMA_STABLE_TABLES,
    REVIEW_SCHEMA_VERSION,
    _assess_safety_requirement,
    _create_schema,
    _insert_events,
    _insert_flight_software_evidence,
    _insert_ground_access,
)
from sim.review import (
    EVIDENCE_PLOT_RECIPES,
    SAVED_QUERY_MATURITY_LEVELS,
    SAVED_REVIEW_QUERIES,
    EvidencePlotter,
    ReviewPlotSpec,
    ReviewQueryError,
    ReviewQueryResult,
    ReviewStoreNotFoundError,
    ReviewWorkspace,
    SavedReviewQuery,
    load_workflow_manifest,
    numeric_columns,
    plan_review_plot,
    plot_spec_from_mapping,
    render_review_plot,
    save_review_plot,
    write_workflow_review,
)
from sim.review.evidence_capsule import create_evidence_capsule

ISS_LINE1 = "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003"
ISS_LINE2 = "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004"


def test_review_store_records_mission_recovery_candidate_rejections_as_warning_events() -> None:
    summary = {
        "mission_recovery": {
            "object_id": "target",
            "assessment_time_s": 45.0,
            "planner": {
                "candidate_rejections": {
                    "total": 3,
                    "by_reason": {
                        "lambert_solver": 1,
                        "unsupported_post_transfer_orbit": 2,
                    },
                }
            },
        }
    }
    with sqlite3.connect(":memory:") as conn:
        _create_schema(conn)
        _insert_events(
            conn,
            t_s=np.array([0.0, 45.0]),
            summary=summary,
            thrust_hist={},
        )
        row = conn.execute(
            "SELECT time_s, sample_index, object_id, event_type, severity, message, source FROM events"
        ).fetchone()

    assert row is not None
    assert row[:5] == (
        45.0,
        1,
        "target",
        "mission_recovery_candidate_rejection",
        "warning",
    )
    assert "rejected 3 sampled candidate grid points" in row[5]
    assert "unsupported_post_transfer_orbit=2" in row[5]
    assert row[6] == "mission_recovery_planner"


def test_safety_requirement_assessment_records_quantitative_violations_and_qualitative_review() -> None:
    time_s = np.array([0.0, 1.0, 2.0])
    truth = np.zeros((3, 14), dtype=float)
    truth[:, 13] = [10.0, 9.0, 8.0]
    quantitative = {
        "evaluation": "quantitative",
        "parameters": [
            {"name": "metric", "value": "mass_kg"},
            {"name": "operator", "value": ">="},
            {"name": "threshold", "value": 9.0},
        ],
    }

    satisfied, source, assessment = _assess_safety_requirement(
        "sat",
        quantitative,
        t_s=time_s,
        truth_hist={"sat": truth},
    )

    assert satisfied == 0
    assert source == "truth_evaluator"
    assert assessment["violation_count"] == 1
    assert assessment["first_violation_time_s"] == 2.0
    assert _assess_safety_requirement(
        "sat",
        {"evaluation": "qualitative"},
        t_s=time_s,
        truth_hist={"sat": truth},
    ) == (None, "qualitative_review_required", {"status": "not_machine_assessable"})


def test_safety_review_uses_only_accepted_load_activation_interval() -> None:
    def packet(sequence: int) -> dict[str, object]:
        return {"source_id": "loader", "boot_id": "boot", "sequence": sequence}

    def clock(ticks: int) -> dict[str, object]:
        return {"clock_id": "clock", "ticks": ticks, "tick_period_ns": 1_000_000_000}

    def mission_event(sequence: int, ticks: int, load_id: str, requirement_id: str) -> dict[str, object]:
        return {
            "packet_id": packet(sequence),
            "kind": "mission_load",
            "source_time": clock(ticks),
            "delivery_time": clock(ticks),
            "payload": {
                "manifest": {"load_id": load_id, "revision": 1},
                "safety_requirements": [
                    {
                        "requirement_id": requirement_id,
                        "evaluation": "quantitative",
                        "parameters": [
                            {"name": "metric", "value": "mass_kg"},
                            {"name": "operator", "value": ">="},
                            {"name": "threshold", "value": 5.0},
                        ],
                    }
                ],
            },
        }

    def output(invocation_id: int, load_id: str, disposition: str) -> dict[str, object]:
        return {
            "invocation_id": invocation_id,
            "commands": [],
            "telemetry": [
                {
                    "topic": "fsw.status",
                    "generated_at": clock(invocation_id),
                    "fields": [
                        {"name": "mission_load_id", "value": load_id},
                        {"name": "mission_load_revision", "value": 1},
                        {"name": "mission_load_disposition", "value": disposition},
                    ],
                }
            ],
        }

    truth = np.zeros((4, 14), dtype=float)
    truth[:, 13] = [1.0, 10.0, 10.0, 10.0]
    evidence = {
        "invocations": [
            {"invocation_id": 1, "input_packet_ids": [packet(0)]},
            {"invocation_id": 2, "input_packet_ids": [packet(1)]},
        ],
        "input_events": [
            mission_event(0, 1, "accepted", "active-window"),
            mission_event(1, 2, "rejected", "rejected-load"),
        ],
        "outputs": [output(1, "accepted", "accepted"), output(2, "rejected", "rejected_by_stack")],
    }
    with sqlite3.connect(":memory:") as conn:
        _create_schema(conn)
        _insert_flight_software_evidence(
            conn,
            payload={"flight_software_evidence_by_object": {"sat": evidence}},
            t_s=np.array([0.0, 1.0, 2.0, 3.0]),
            truth_hist={"sat": truth},
        )
        load_rows = conn.execute(
            "SELECT load_id, disposition FROM fsw_load_events ORDER BY invocation_id"
        ).fetchall()
        safety_rows = conn.execute(
            "SELECT requirement_id, satisfied, source, detail_json "
            "FROM safety_requirement_evidence ORDER BY requirement_id"
        ).fetchall()

    assert load_rows == [("accepted", "accepted"), ("rejected", "rejected_by_stack")]
    by_id = {row[0]: row[1:] for row in safety_rows}
    assert by_id["active-window"][0:2] == (1, "truth_evaluator")
    active_detail = json.loads(by_id["active-window"][2])
    assert active_detail["assessment"]["sample_count"] == 3
    assert active_detail["assessment"]["activation_start_s"] == 1.0
    assert by_id["rejected-load"][0:2] == (None, "load_not_accepted")


def _review_store_config(output_dir: Path) -> dict:
    return {
        "scenario_name": "review_store_smoke",
        "scenario_description": "Review store smoke test",
        "objects": {
            "target": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
            "chaser": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "relative_to": "target",
                    "relative_ric_rect": [1.0, 0.0, 0.0, -0.001, 0.0, 0.0],
                },
                "flight_software": {
                    "profile": "fsw.profile.coast_monitor.v1",
                },
            },
        },
        "simulator": {
            "duration_s": 2.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {
                "print_summary": False,
                "save_json": True,
                "save_full_log": True,
            },
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
            "review": {"enabled": True, "detail": "standard"},
        },
    }


def test_single_run_review_store_writes_queryable_sqlite(tmp_path: Path) -> None:
    result = SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()

    review_outputs = dict(result.summary.get("review_outputs", {}) or {})
    db_path = Path(review_outputs["sqlite"])
    schema_path = Path(review_outputs["schema_json"])

    assert db_path.is_file()
    assert schema_path.is_file()
    assert db_path.parent == tmp_path / "review"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["schema_version"] == REVIEW_SCHEMA_VERSION
    assert schema["compatibility"]["policy"] == REVIEW_SCHEMA_COMPATIBILITY_POLICY == "stable_1_x_additive"
    assert schema["compatibility"]["breaking_change_requires_schema_version_bump"] is True
    assert tuple(schema["compatibility"]["stable_tables"]) == REVIEW_SCHEMA_STABLE_TABLES
    assert set(REVIEW_SCHEMA_STABLE_TABLES).issubset(schema["tables"])

    with sqlite3.connect(db_path) as conn:
        scenario_name, config_sha256, config_json = conn.execute(
            "SELECT scenario_name, config_sha256, config_json FROM run_metadata"
        ).fetchone()
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        object_count = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
        sample_count = conn.execute("SELECT COUNT(*) FROM time_samples").fetchone()[0]
        state_count = conn.execute("SELECT COUNT(*) FROM object_state").fetchone()[0]
        relative_count = conn.execute("SELECT COUNT(*) FROM relative_state").fetchone()[0]
        element_count = conn.execute("SELECT COUNT(*) FROM object_orbital_elements").fetchone()[0]
        decision_count = conn.execute("SELECT COUNT(*) FROM controller_decisions").fetchone()[0]
        mission_mode_count = conn.execute("SELECT COUNT(*) FROM mission_modes").fetchone()[0]
        command_gate_count = conn.execute("SELECT COUNT(*) FROM command_gates").fetchone()[0]
        fsw_invocation_count = conn.execute("SELECT COUNT(*) FROM fsw_invocations").fetchone()[0]
        fsw_identities = conn.execute(
            "SELECT DISTINCT stack_id, stack_version, profile_id FROM fsw_invocations WHERE object_id = 'chaser'"
        ).fetchall()
        standard_debug_detail = conn.execute(
            "SELECT detail_json FROM fsw_invocations ORDER BY object_id, invocation_id LIMIT 1"
        ).fetchone()[0]
        min_range = conn.execute("SELECT MIN(range_km) FROM relative_state").fetchone()[0]
        artifact_paths = [row[0] for row in conn.execute("SELECT path FROM artifacts ORDER BY artifact_id")]

    assert scenario_name == "review_store_smoke"
    assert set(REVIEW_SCHEMA_STABLE_TABLES).issubset(table_names)
    assert config_sha256 == hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    assert json.loads(config_json)["objects"]["target"]["enabled"] is True
    assert object_count == 2
    assert sample_count == 3
    assert state_count == 6
    assert relative_count == 3
    assert element_count == state_count
    assert decision_count == 0
    assert mission_mode_count == 0
    assert command_gate_count == 0
    assert fsw_invocation_count == 6
    assert fsw_identities == [("fsw.passive", "2.0.0", "fsw.profile.coast_monitor.v1")]
    assert standard_debug_detail is None
    assert min_range == pytest.approx(result.min_range("chaser", "target"))
    assert {
        "index.md",
        "master_run_log.json",
        "master_run_summary.json",
        "review/run.sqlite",
        "review/schema.json",
    }.issubset(set(artifact_paths))


def test_full_review_store_retains_fsw_debug_detail(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["outputs"]["review"]["detail"] = "full"

    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    db_path = Path(result.summary["review_outputs"]["sqlite"])

    with sqlite3.connect(db_path) as conn:
        detail_json = conn.execute(
            "SELECT detail_json FROM fsw_invocations ORDER BY object_id, invocation_id LIMIT 1"
        ).fetchone()[0]

    detail = json.loads(detail_json)
    assert detail["invocation_id"] >= 1
    assert "input_packet_ids" in detail


def test_v2_review_store_links_invocation_command_receipt_and_realization(tmp_path: Path) -> None:
    raw = SimulationConfig.from_yaml("sim/game/configs/game_mode_basic.yaml").scenario.to_dict()
    raw["scenario_name"] = "v2_review_linkage"
    raw["simulator"]["duration_s"] = raw["simulator"]["dt_s"]
    raw["outputs"]["output_dir"] = str(tmp_path)
    raw["outputs"]["mode"] = "save"
    raw["outputs"]["review"] = {"enabled": True, "detail": "standard"}
    raw["outputs"]["plots"] = {"enabled": False, "figure_ids": []}
    raw["outputs"]["animations"] = {"enabled": False, "types": []}

    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    db_path = Path(result.summary["review_outputs"]["sqlite"])

    with sqlite3.connect(db_path) as conn:
        command_links = conn.execute(
            """
            SELECT COUNT(*)
            FROM actuator_commands AS command
            JOIN fsw_invocations AS invocation
              ON invocation.object_id = command.object_id
             AND invocation.invocation_id = command.invocation_id
            JOIN actuator_command_receipts AS receipt
              ON receipt.object_id = command.object_id
             AND receipt.command_source_id = command.command_source_id
             AND receipt.command_boot_id = command.command_boot_id
             AND receipt.command_sequence = command.command_sequence
            """
        ).fetchone()[0]
        realization_links = conn.execute(
            """
            SELECT COUNT(*)
            FROM actuator_realization AS realization
            JOIN actuator_commands AS command
              ON command.object_id = realization.object_id
             AND command.command_source_id = realization.command_source_id
             AND command.command_boot_id = realization.command_boot_id
             AND command.command_sequence = realization.command_sequence
            """
        ).fetchone()[0]
        input_links = conn.execute(
            """
            SELECT COUNT(*)
            FROM fsw_input_events AS input
            JOIN fsw_invocations AS invocation
              ON invocation.object_id = input.object_id
             AND invocation.invocation_id = input.invocation_id
            """
        ).fetchone()[0]
        normalized_diagnostic_fields = conn.execute(
            "SELECT COUNT(*) FROM fsw_diagnostic_fields"
        ).fetchone()[0]
        standard_diagnostic_detail = conn.execute(
            "SELECT COUNT(*) FROM fsw_diagnostics WHERE detail_json IS NOT NULL"
        ).fetchone()[0]
        standard_realization_detail = conn.execute(
            "SELECT COUNT(*) FROM actuator_realization WHERE detail_json IS NOT NULL"
        ).fetchone()[0]
        standard_snapshot_detail = conn.execute(
            "SELECT COUNT(*) FROM fsw_snapshots WHERE detail_json IS NOT NULL"
        ).fetchone()[0]
        compressed_snapshot_detail = conn.execute(
            "SELECT COUNT(*) FROM fsw_snapshots WHERE detail_gzip IS NOT NULL"
        ).fetchone()[0]
        normalized_realization_count = conn.execute(
            "SELECT COUNT(*) FROM actuator_realization "
            "WHERE realized_force_x_n IS NOT NULL AND mass_flow_kg_s IS NOT NULL"
        ).fetchone()[0]

    assert command_links > 0
    assert realization_links > 0
    assert input_links > 0
    assert normalized_diagnostic_fields > 0
    assert normalized_realization_count > 0
    assert standard_diagnostic_detail == 0
    assert standard_realization_detail == 0
    assert standard_snapshot_detail == 0
    assert compressed_snapshot_detail > 0
def test_review_store_closes_sqlite_before_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_connect = sqlite3.connect
    real_replace = Path.replace
    state = {"closed": False, "replace_observed": False}

    class TrackedConnection:
        def __init__(self, path: Path) -> None:
            self.connection = real_connect(path)

        def __getattr__(self, name: str):
            return getattr(self.connection, name)

        def close(self) -> None:
            self.connection.close()
            state["closed"] = True

    def tracked_connect(path: Path) -> TrackedConnection:
        return TrackedConnection(path)

    def checked_replace(path: Path, target: Path) -> Path:
        if path.name == "run.sqlite.tmp":
            assert state["closed"], "Windows cannot replace an open SQLite file"
            state["replace_observed"] = True
        return real_replace(path, target)

    monkeypatch.setattr(review_store_module.sqlite3, "connect", tracked_connect)
    monkeypatch.setattr(Path, "replace", checked_replace)

    result = SimulationSession.from_config(
        SimulationConfig.from_dict(_review_store_config(tmp_path))
    ).run()

    assert state["replace_observed"]
    assert Path(result.summary["review_outputs"]["sqlite"]).is_file()


def test_review_store_supports_workspace_paths_with_spaces(tmp_path: Path) -> None:
    output_dir = tmp_path / "Orbital Engineering Lab" / "Windows Review Output"
    SimulationSession.from_config(
        SimulationConfig.from_dict(_review_store_config(output_dir))
    ).run()

    workspace = ReviewWorkspace.open(output_dir)
    result = workspace.query("SELECT scenario_name FROM run_metadata")

    assert result.rows == [{"scenario_name": "review_store_smoke"}]
    with pytest.raises(ReviewQueryError, match="read-only"):
        workspace.query("UPDATE run_metadata SET scenario_name = 'changed'")


def test_single_run_review_store_writes_all_configured_relative_pairs(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["scenario_name"] = "review_store_multi_pair"
    raw["simulator"]["duration_s"] = 1.0
    raw["objects"]["chaser_a"] = raw["objects"].pop("chaser")
    raw["objects"]["chaser_b"] = {
        "enabled": True,
        "specs": {"mass_kg": 100.0},
        "initial_state": {
            "relative_to": "target",
            "relative_ric_rect": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    }

    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    db_path = Path(result.summary["review_outputs"]["sqlite"])

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT deputy_id, chief_id, COUNT(*)
            FROM relative_state
            GROUP BY deputy_id, chief_id
            ORDER BY deputy_id, chief_id
            """
        ).fetchall()
        final_rows = conn.execute(SAVED_REVIEW_QUERIES["relative_final_state"].sql).fetchall()

    assert rows == [
        ("chaser_a", "target", 2),
        ("chaser_b", "target", 2),
    ]
    assert [(row[1], row[2]) for row in final_rows] == [
        ("chaser_a", "target"),
        ("chaser_b", "target"),
    ]


def test_saved_final_queries_bound_each_object_and_relative_pair() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(
            """
            CREATE TABLE object_state (
                sample_index INTEGER, time_s REAL, object_id TEXT,
                pos_x_eci_km REAL, pos_y_eci_km REAL, pos_z_eci_km REAL,
                vel_x_eci_km_s REAL, vel_y_eci_km_s REAL, vel_z_eci_km_s REAL
            );
            CREATE TABLE object_orbital_elements (
                sample_index INTEGER, time_s REAL, object_id TEXT,
                radius_km REAL, speed_km_s REAL
            );
            CREATE TABLE relative_state (
                sample_index INTEGER, time_s REAL, deputy_id TEXT, chief_id TEXT,
                r_radial_km REAL, i_intrack_km REAL, c_crosstrack_km REAL,
                range_km REAL, range_rate_km_s REAL
            );
            """
        )
        conn.executemany(
            "INSERT INTO object_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (0, 0.0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (2, 2.0, "a", 2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0, 0.0, "b", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (1, 1.0, "b", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ],
        )
        conn.executemany(
            "INSERT INTO object_orbital_elements VALUES (?, ?, ?, ?, ?)",
            [(2, 2.0, "a", 2.0, 2.0), (1, 1.0, "b", 1.0, 1.0)],
        )
        conn.executemany(
            "INSERT INTO relative_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (0, 0.0, "a", "chief", 0.0, 0.0, 0.0, 4.0, 0.0),
                (2, 2.0, "a", "chief", 2.0, 0.0, 0.0, 2.0, 0.0),
                (0, 0.0, "b", "chief", 0.0, 0.0, 0.0, 5.0, 0.0),
                (1, 1.0, "b", "chief", 1.0, 0.0, 0.0, 1.0, 0.0),
            ],
        )

        passive_rows = conn.execute(SAVED_REVIEW_QUERIES["passive_final_state"].sql).fetchall()
        object_rows = conn.execute(SAVED_REVIEW_QUERIES["object_final_state"].sql).fetchall()
        relative_rows = conn.execute(SAVED_REVIEW_QUERIES["relative_final_state"].sql).fetchall()

    assert [(row[0], row[1]) for row in passive_rows] == [("a", 2.0), ("b", 1.0)]
    assert [(row[0], row[1], row[8]) for row in object_rows] == [("a", 2.0, 2.0), ("b", 1.0, 1.0)]
    assert [(row[1], row[0], row[6]) for row in relative_rows] == [
        ("a", 2.0, 2.0),
        ("b", 1.0, 1.0),
    ]


def test_saved_review_queries_have_machine_readable_contract() -> None:
    assert SAVED_REVIEW_QUERIES
    for key, query in SAVED_REVIEW_QUERIES.items():
        assert key == query.name
        assert query.description
        assert query.sql.lstrip().upper().startswith(("SELECT", "WITH"))
        assert query.source_tables, key
        assert query.maturity in SAVED_QUERY_MATURITY_LEVELS
        assert query.max_vm_steps > 0

    assert SAVED_REVIEW_QUERIES["burn_events"].allow_empty is True
    try:
        SavedReviewQuery(name="bad", description="bad", sql="DELETE FROM metrics")
    except ValueError as exc:
        assert "read-only SELECT/WITH" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("SavedReviewQuery accepted mutating SQL")


def test_review_burn_events_use_applied_interval_boundaries() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                time_s REAL,
                sample_index INTEGER,
                object_id TEXT,
                event_type TEXT,
                severity TEXT,
                message TEXT,
                source TEXT
            )
            """
        )
        _insert_events(
            conn,
            t_s=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            summary={},
            thrust_hist={
                "chaser": np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [1.0e-6, 0.0, 0.0],
                        [1.0e-6, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
            },
        )
        rows = conn.execute(
            "SELECT event_type, time_s, sample_index FROM events ORDER BY time_s, event_type"
        ).fetchall()

    assert rows == [("burn_start", 0.0, 0), ("burn_end", 2.0, 2)]


def test_flagship_scale_burn_activity_uses_vetted_saved_query_budget(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    db_path = review_dir / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE thrust (
                sample_index INTEGER,
                time_s REAL,
                object_id TEXT,
                accel_x_eci_km_s2 REAL,
                accel_y_eci_km_s2 REAL,
                accel_z_eci_km_s2 REAL,
                accel_norm_km_s2 REAL,
                burn_active INTEGER
            );
            CREATE INDEX idx_thrust_object_time ON thrust(object_id, time_s);
            """
        )
        rows = (
            (index, float(index), object_id, 0.0, 0.0, 0.0, 1.0e-6 if object_id == "chaser" else 0.0, int(object_id == "chaser"))
            for object_id in ("chaser", "target")
            for index in range(12_001)
        )
        conn.executemany("INSERT INTO thrust VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    workspace = ReviewWorkspace.open(tmp_path)
    saved = SAVED_REVIEW_QUERIES["burn_activity"]
    with pytest.raises(ReviewQueryError, match="step budget"):
        workspace.query(saved.sql)

    result = workspace.query(saved.sql, max_vm_steps=saved.max_vm_steps)

    assert result.row_count == 2
    assert result.rows[0]["object_id"] == "chaser"
    assert result.rows[0]["samples"] == 12_001


def test_review_events_include_confirmed_maneuver_detection() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                time_s REAL,
                sample_index INTEGER,
                object_id TEXT,
                event_type TEXT,
                severity TEXT,
                message TEXT,
                source TEXT
            )
            """
        )
        _insert_events(
            conn,
            t_s=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            summary={
                "knowledge_consistency_by_observer": {
                    "chaser": {
                        "target": {
                            "maneuver_confirmed_event_count": 1,
                            "maneuver_first_confirmed_t_s": 2.0,
                            "maneuver_max_nis": 42.0,
                        }
                    }
                }
            },
            thrust_hist={},
        )
        row = conn.execute(
            "SELECT event_id, time_s, sample_index, object_id, event_type, severity, source "
            "FROM events"
        ).fetchone()

    assert row == (
        "maneuver_detection_confirmed:chaser:target:2",
        2.0,
        2,
        "target",
        "maneuver_detection_confirmed",
        "warning",
        "knowledge_maneuver_detector",
    )


def test_relative_evidence_plot_recipes_group_by_pair_id() -> None:
    for recipe_id in ("relative_range", "relative_range_rate"):
        recipe = EVIDENCE_PLOT_RECIPES[recipe_id]
        assert recipe.group_column == "pair_id"
        assert "pair_id" in recipe.sql


def test_saved_review_query_source_tables_match_review_schema(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)
    available_tables = set(workspace.tables())
    single_run_queries = [
        "run_metadata",
        "objects",
        "artifacts",
        "passive_final_state",
        "rendezvous_metrics",
        "rendezvous_closest_approach",
        "relative_final_state",
        "burn_activity",
        "burn_events",
        "attitude_rates_first_last",
        "attitude_state_first_last",
    ]

    for query_name in single_run_queries:
        query = SAVED_REVIEW_QUERIES[query_name]
        assert set(query.source_tables).issubset(available_tables), query_name


def test_review_store_writes_object_propagation_metadata_for_sgp4(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["objects"]["chaser"]["enabled"] = False
    raw["objects"]["target"]["propagation_method"] = "general"
    raw["objects"]["target"]["general"] = {"model": "sgp4", "output_frame": "eci", "frame_transform": "teme_as_eci"}
    raw["objects"]["target"]["initial_state"] = {"tle": {"line1": ISS_LINE1, "line2": ISS_LINE2}}
    raw["simulator"]["initial_jd_utc"] = 2460310.5
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    db_path = Path(dict(result.summary.get("review_outputs", {}) or {})["sqlite"])
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT object_id, propagation_method, general_model, native_frame, output_frame, frame_transform
            FROM object_propagation
            """
        ).fetchone()

    assert row == ("target", "general", "sgp4", "teme", "eci", "teme_as_eci")


def test_review_store_writes_attitude_error_history_and_saved_query(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    raw = yaml.safe_load((root / "examples/configs/public_attitude_hold_disturbance.yaml").read_text(encoding="utf-8"))
    raw["scenario_name"] = "attitude_error_review_smoke"
    raw["simulator"]["duration_s"] = 2.0
    raw["outputs"]["output_dir"] = str(tmp_path)
    raw["outputs"]["plots"] = {"enabled": False, "figure_ids": []}
    raw["outputs"]["animations"] = {"enabled": False, "types": []}
    raw["outputs"]["review"] = {"enabled": True, "detail": "standard"}

    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    workspace = ReviewWorkspace.open(result.summary["review_outputs"]["sqlite"])
    rows = workspace.query(SAVED_REVIEW_QUERIES["attitude_error_first_last"].sql).rows

    assert len(rows) == 2
    assert rows[0]["quat_error_angle_deg"] is not None
    assert rows[-1]["time_s"] == pytest.approx(2.0)
    with sqlite3.connect(result.summary["review_outputs"]["sqlite"]) as conn:
        desired = conn.execute(
            "SELECT desired_quat_w, desired_quat_x, desired_quat_y, desired_quat_z "
            "FROM attitude_error ORDER BY time_s LIMIT 1"
        ).fetchone()
    assert desired == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_review_store_writes_frame_provenance(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["simulator"]["frames"] = {"model": "simple_gmst"}
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    db_path = Path(dict(result.summary.get("review_outputs", {}) or {})["sqlite"])
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT scope, model, legacy_frame_model, time_scale_model,
                   polar_motion_applied, nutation_corrections_applied
            FROM frame_provenance
            """
        ).fetchone()

    assert row == ("scenario", "simple_gmst", "simple", "utc_only", 0, 0)
    assert result.payload["frame_provenance"]["model"] == "simple_gmst"


def test_review_store_writes_tle_initialization_metadata_for_special_propagation(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["objects"]["chaser"]["enabled"] = False
    raw["objects"]["target"]["initial_state"] = {"tle": {"line1": ISS_LINE1, "line2": ISS_LINE2}}
    raw["simulator"]["initial_jd_utc"] = 2460310.75
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    db_path = Path(dict(result.summary.get("review_outputs", {}) or {})["sqlite"])
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT object_id, source, initialization_model, initialization_propagator_name,
                   handoff_propagation_method, native_frame, output_frame, frame_transform,
                   tle_age_initialization_days
            FROM object_initialization
            """
        ).fetchone()

    assert row[:8] == (
        "target",
        "tle",
        "ogp",
        "OGP-SGP4",
        "special",
        "teme",
        "eci",
        "teme_to_eci_iau80",
    )
    assert row[8] == pytest.approx(0.25)


def test_review_store_records_canonical_eci_state_frame_for_sgp4(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["objects"]["chaser"]["enabled"] = False
    raw["objects"]["target"]["propagation_method"] = "general"
    raw["objects"]["target"]["general"] = {"model": "sgp4", "output_frame": "teme"}
    raw["objects"]["target"]["initial_state"] = {"tle": {"line1": ISS_LINE1, "line2": ISS_LINE2}}
    raw["simulator"]["initial_jd_utc"] = 2460310.5
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    assert result.payload["object_state_frames"]["target"] == "eci"
    db_path = Path(dict(result.summary.get("review_outputs", {}) or {})["sqlite"])
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT p.output_frame, p.frame_transform, f.state_frame
            FROM object_propagation p
            JOIN object_state_frame f USING (object_id)
            """
        ).fetchone()

    assert row == ("teme", "native", "eci")


def test_review_store_config_defaults_disabled_and_validates_detail(tmp_path: Path) -> None:
    cfg = scenario_config_from_dict(_review_store_config(tmp_path))

    assert cfg.outputs.review.enabled is True
    assert cfg.outputs.review.detail == "standard"
    assert cfg.outputs.review.strict is False

    raw = _review_store_config(tmp_path)
    raw["outputs"]["review"]["detail"] = "dense"
    with pytest.raises(ValueError, match="outputs.review.detail"):
        scenario_config_from_dict(raw)


def test_review_store_writes_mission_recovery_tables(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["objects"]["target"]["specs"]["isp_s"] = 220.0
    raw["analysis"] = {
        "mission_recovery": {
            "enabled": True,
            "object_id": "target",
            "goal": "orbit_shape",
            "element_tolerances": {"a_km": 1.0, "ecc": 0.01},
            "planner": {
                "enabled": True,
                "modes": ["min_delta_v", "min_time", "constrained"],
                "candidate_count": 4,
                "max_recovery_time_s": 7200.0,
                "max_recovery_delta_v_m_s": 20.0,
            },
        }
    }
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    db_path = Path(result.summary["review_outputs"]["sqlite"])

    with sqlite3.connect(db_path) as conn:
        summary_rows = conn.execute(
            "SELECT object_id, goal, method, recovery_delta_v_m_s FROM mission_recovery_summary"
        ).fetchall()
        element_rows = conn.execute(
            "SELECT state_label, a_km, ecc FROM mission_recovery_elements ORDER BY state_label"
        ).fetchall()
        metric_rows = conn.execute(
            "SELECT metric_name, units FROM metrics WHERE metric_name = 'recovery_delta_v_m_s'"
        ).fetchall()
        candidate_rows = conn.execute(
            "SELECT candidate_id, source, source_family, target_basis, feasible, verified "
            "FROM mission_recovery_candidates"
        ).fetchall()
        burn_rows = conn.execute(
            "SELECT candidate_id, burn_index, delta_v_m_s FROM mission_recovery_burns"
        ).fetchall()
        candidate_element_rows = conn.execute(
            "SELECT candidate_id, a_km, ecc FROM mission_recovery_candidate_elements"
        ).fetchall()

    assert len(summary_rows) == 1
    assert summary_rows[0][0] == "target"
    assert summary_rows[0][1] == "orbit_shape"
    assert summary_rows[0][2] in {"sim_state_inferred_intrack_impulse", "local_orbit_shape_velocity_match"}
    assert summary_rows[0][3] is not None
    assert [row[0] for row in element_rows] == ["final", "initial", "target"]
    assert metric_rows == [("recovery_delta_v_m_s", "m/s")]
    assert candidate_rows
    assert burn_rows
    assert candidate_element_rows
    assert all(row[2] == "analytic_reconstitution" for row in candidate_rows)
    assert all(row[3] == "initial_orbit" for row in candidate_rows)
    assert all(row[4] in (0, 1) for row in candidate_rows)
    assert all(row[5] in (0, 1) for row in candidate_rows)


@pytest.mark.parametrize("policy", ["pre_1_0_additive", "stable_1_x_additive"])
def test_review_workspace_reads_policy_metadata_without_rewriting(tmp_path: Path, policy: str) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    db_path = review_dir / "run.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("CREATE TABLE run_metadata (scenario_name TEXT, review_schema_version TEXT)")
        connection.execute("INSERT INTO run_metadata VALUES ('saved_run', '0.13')")
        connection.commit()
    finally:
        connection.close()
    schema_path = review_dir / "schema.json"
    schema_path.write_text(json.dumps({
        "schema_version": "0.13",
        "compatibility": {"policy": policy, "stable_tables": ["run_metadata"]},
    }), encoding="utf-8")
    original_bytes = {path: path.read_bytes() for path in (db_path, schema_path)}

    workspace = ReviewWorkspace.open(tmp_path)
    try:
        assert workspace.schema()["compatibility"]["policy"] == policy
        rows = workspace.query("SELECT scenario_name, review_schema_version FROM run_metadata").rows
        assert rows == [{"scenario_name": "saved_run", "review_schema_version": "0.13"}]
    finally:
        workspace.close()
    assert {path: path.read_bytes() for path in original_bytes} == original_bytes


def test_review_workspace_query_api_allows_safe_selects(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()

    workspace = ReviewWorkspace.open(tmp_path)
    top_level_workspace = TopLevelReviewWorkspace.open(tmp_path / "review" / "run.sqlite")

    assert workspace.db_path == tmp_path / "review" / "run.sqlite"
    assert top_level_workspace.output_dir == tmp_path
    assert "relative_state" in workspace.tables()
    assert "object_state" in workspace.schema()["columns"]
    assert workspace.saved_views() == []

    result = workspace.query(
        """
        -- safe review query
        SELECT time_s, range_km
        FROM relative_state
        WHERE deputy_id = ?
        ORDER BY time_s;
        """,
        ("chaser",),
        max_rows=2,
    )

    assert result.columns == ["time_s", "range_km"]
    assert result.row_count == 2
    assert result.truncated is True
    assert result.rows[0]["time_s"] == 0.0

    aggregate = workspace.query(
        "WITH ranges AS (SELECT range_km FROM relative_state) SELECT MIN(range_km) AS min_range FROM ranges"
    )
    assert 0.9 < float(aggregate.rows[0]["min_range"]) < 1.0


def test_review_workspace_rejects_unsafe_queries(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)

    unsafe_queries = [
        "",
        "DELETE FROM metrics",
        "DROP TABLE metrics",
        "PRAGMA table_info(metrics)",
        "SELECT 1; SELECT 2",
        "WITH selected AS (SELECT 1) DELETE FROM metrics",
    ]

    for sql in unsafe_queries:
        with pytest.raises(ReviewQueryError):
            workspace.query(sql)


def test_review_workspace_rejects_expensive_queries(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)

    with pytest.raises(ReviewQueryError, match="step budget"):
        workspace.query(
            "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt LIMIT 1000000) SELECT max(x) FROM cnt",
            max_vm_steps=1,
        )


def test_review_plot_creator_saves_styled_figure_with_provenance(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)
    result = workspace.query("SELECT time_s, range_km FROM relative_state ORDER BY time_s")

    assert numeric_columns(result) == ["time_s", "range_km"]

    artifact = save_review_plot(
        workspace,
        ReviewPlotSpec(
            sql="SELECT time_s, range_km FROM relative_state ORDER BY time_s",
            x_column="time_s",
            y_columns=["range_km"],
            plot_type="line",
            style_name="oel_light",
            title="Range Over Time",
            x_label="Time (s)",
            y_label="Range (km)",
            artifact_id="range_over_time",
        ),
    )

    assert artifact.path.is_file()
    assert artifact.relative_path == "review/figures/range_over_time.png"

    manifest = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    row = manifest["artifacts"][-1]
    assert row["artifact_id"] == "range_over_time"
    assert row["source"] == "oel_review_plot_api"
    assert row["plot_type"] == "line"
    assert row["style_name"] == "oel_light"
    assert row["x_column"] == "time_s"
    assert row["y_columns"] == ["range_km"]
    assert "FROM relative_state" in row["source_query"]
    output_index = (tmp_path / "index.md").read_text(encoding="utf-8")
    assert "OEL_REVIEW_GENERATED_ARTIFACTS_START" in output_index
    assert "[`review/figures/range_over_time.png`](review/figures/range_over_time.png)" in output_index


def test_evidence_plotter_api_creates_multi_series_recipe_and_custom_plots(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)
    plotter = EvidencePlotter(workspace)

    range_artifact = plotter.line(
        sql="SELECT time_s, range_km, range_rate_km_s FROM relative_state ORDER BY time_s",
        x="time_s",
        y=["range_km", "range_rate_km_s"],
        title="Range and rate",
        y_label="Value",
        artifact_id="api_range_and_rate",
        style="light",
    )

    assert range_artifact.relative_path == "review/figures/api_range_and_rate.png"
    assert range_artifact.path.is_file()

    recipe_artifact = plotter.relative_range_rate(artifact_id="api_range_rate_recipe")
    assert recipe_artifact.relative_path == "review/figures/api_range_rate_recipe.png"

    manifest = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][-2]["source"] == "oel_review_plot_api"
    assert manifest["artifacts"][-2]["style_name"] == "oel_light"
    assert manifest["artifacts"][-2]["y_columns"] == ["range_km", "range_rate_km_s"]
    assert manifest["artifacts"][-1]["extra"]["recipe_id"] == "relative_range_rate"


def test_review_plotter_creates_professional_rectangular_ric_recipe(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()

    artifact = EvidencePlotter(tmp_path).relative_position_ric_2d(style="light")

    assert artifact.path.is_file()
    assert artifact.spec.renderer_id == "ric_rectangular_2d"
    assert artifact.qa["automated_status"] == "passed"
    assert artifact.qa["visual_qa_status"] == "pending_agent_review"
    assert artifact.qa["presentation_quality"]["policy_id"] == "oel.agent_strict"
    assert artifact.qa["presentation_quality"]["policy_version"] == 1
    manifest = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    row = manifest["artifacts"][-1]
    assert row["extra"] == {"source": "oel_review_plot_api", "recipe_id": "relative_position_ric_2d", "recipe_version": 1}
    assert row["renderer_id"] == "ric_rectangular_2d"
    assert len(row["query_sha256"]) == 64
    assert row["qa"]["automated_status"] == "passed"
    assert row["qa"]["presentation_quality"]["numeric_formatting"]


def test_typed_review_plot_plan_is_content_bound_before_render(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    arguments = {
        "sql": "SELECT time_s, range_km FROM relative_state ORDER BY time_s",
        "x_column": "time_s",
        "y_columns": ["range_km"],
        "plot_type": "line",
        "artifact_id": "typed_range",
        "style": "oel_light",
        "format": "png",
    }
    spec = plot_spec_from_mapping(arguments, source="test_typed_review_plot_v2")
    plan = plan_review_plot(tmp_path, spec)

    assert plan["status"] == "planned"
    assert plan["render_authorized"] is False
    assert plan["columns"] == ["time_s", "range_km"]
    artifact = render_review_plot(
        tmp_path,
        spec,
        plot_plan_id=plan["plot_plan_id"],
        path=tmp_path / "review" / "figures" / "typed_range.png",
    )
    assert artifact.path.is_file()
    assert artifact.qa["automated_status"] == "passed"
    assert artifact.qa["presentation_quality"]["automated_status"] == "passed"

    changed = plot_spec_from_mapping({**arguments, "title": "Changed"}, source="test_typed_review_plot_v2")
    with pytest.raises(ValueError, match="stale or does not match"):
        render_review_plot(
            tmp_path,
            changed,
            plot_plan_id=plan["plot_plan_id"],
            path=tmp_path / "review" / "figures" / "changed.png",
        )


def test_capsule_only_plot_plan_and_evidence_manifest_use_logical_identity(tmp_path: Path) -> None:
    result = SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    database = tmp_path / "review" / "run.sqlite"
    create_evidence_capsule(database, remove_original=True)
    spec = plot_spec_from_mapping(
        {
            "sql": "SELECT time_s, range_km FROM relative_state ORDER BY time_s",
            "x_column": "time_s",
            "y_columns": ["range_km"],
            "plot_type": "line",
            "artifact_id": "capsule_range",
        },
        source="test_capsule_plot",
    )

    first = plan_review_plot(tmp_path, spec)
    second = plan_review_plot(tmp_path, spec)
    assert first["plot_plan_id"] == second["plot_plan_id"]
    artifact = render_review_plot(
        tmp_path,
        spec,
        plot_plan_id=first["plot_plan_id"],
        path=tmp_path / "review" / "figures" / "capsule_range.png",
    )
    assert artifact.path.is_file()
    generated = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    assert generated["artifacts"][-1]["review_store"]["relative_path"] == "review/run.sqlite"
    assert generated["artifacts"][-1]["review_store"]["storage"] == "sqlite_gzip_capsule"

    manifest = result.evidence_manifest()["review"]
    assert manifest["db_exists"] is False
    assert manifest["evidence_exists"] is True
    assert manifest["compressed_exists"] is True
    assert manifest["capsule_manifest_exists"] is True
    assert len(manifest["logical_sha256"]) == 64


def test_evidence_plotter_histogram_heatmap_and_helpful_errors(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    with sqlite3.connect(review_dir / "run.sqlite") as conn:
        conn.execute(
            "CREATE TABLE sample_grid (region TEXT, bucket TEXT, value REAL, label TEXT)"
        )
        conn.executemany(
            "INSERT INTO sample_grid VALUES (?, ?, ?, ?)",
            [
                ("A", "early", 1.0, "one"),
                ("A", "late", 2.0, "two"),
                ("B", "early", 3.0, "three"),
                ("B", "late", 4.0, "four"),
                ("C", "invalid", float("inf"), "infinite"),
            ],
        )

    plotter = EvidencePlotter(tmp_path)
    hist = plotter.histogram(
        sql="SELECT value FROM sample_grid",
        y="value",
        artifact_id="api_value_hist",
    )
    heatmap = plotter.heatmap(
        sql="SELECT bucket, region, value FROM sample_grid ORDER BY region, bucket",
        x="bucket",
        y="region",
        value="value",
        artifact_id="api_value_heatmap",
    )

    assert hist.path.is_file()
    assert heatmap.path.is_file()
    with pytest.raises(ValueError, match="Available columns"):
        plotter.line(sql="SELECT value FROM sample_grid", x="time_s", y="value")
    with pytest.raises(ValueError, match="Numeric columns"):
        plotter.line(sql="SELECT bucket, label FROM sample_grid", x="bucket", y="label")


def test_review_plot_preview_can_skip_provenance_manifest(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)
    preview_path = tmp_path / "preview.png"

    artifact = save_review_plot(
        workspace,
        ReviewPlotSpec(
            sql="SELECT time_s, range_km FROM relative_state ORDER BY time_s",
            x_column="time_s",
            y_columns=["range_km"],
            plot_type="scatter",
            style_name="oel_dark",
        ),
        path=preview_path,
        record=False,
    )

    assert artifact.path == preview_path
    assert preview_path.is_file()
    assert not (tmp_path / "review" / "generated_artifacts.json").exists()


def test_review_cli_queries_output_folder(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review",
            str(tmp_path),
            "--query",
            "SELECT scenario_name FROM run_metadata",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["rows"][0]["scenario_name"] == "review_store_smoke"

    unsafe = subprocess.run(
        [sys.executable, "-m", "sim.review", str(tmp_path), "--query", "DELETE FROM metrics"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert unsafe.returncode == 2
    assert "must start with SELECT or WITH" in unsafe.stderr


def test_review_cli_json_represents_blob_metadata_without_exposing_content(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    database = tmp_path / "review" / "run.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE opaque_test (payload BLOB)")
        connection.execute("INSERT INTO opaque_test VALUES (?)", (b"opaque-review-payload",))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review",
            str(tmp_path),
            "--query",
            "SELECT payload FROM opaque_test",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)["rows"][0]["payload"]
    assert payload == {
        "type": "sqlite_blob",
        "byte_count": 21,
        "sha256": hashlib.sha256(b"opaque-review-payload").hexdigest(),
    }


def test_review_plot_cli_dry_run_and_creates_artifact(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()

    dry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review.plot",
            str(tmp_path),
            "--sql",
            "SELECT time_s, range_km FROM relative_state ORDER BY time_s",
            "--x",
            "time_s",
            "--y",
            "range_km",
            "--title",
            "CLI range",
            "--artifact-id",
            "cli_range",
            "--dry-run",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert dry_run.returncode == 0, dry_run.stderr
    dry_payload = json.loads(dry_run.stdout)
    assert dry_payload["spec"]["artifact_id"] == "cli_range"
    assert dry_payload["row_count"] == 3
    assert not (tmp_path / "review" / "figures" / "cli_range.png").exists()

    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review",
            "plot",
            str(tmp_path),
            "--recipe",
            "relative_range",
            "--artifact-id",
            "cli_recipe_range",
            "--style",
            "light",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    assert payload["relative_path"] == "review/figures/cli_recipe_range.png"
    assert Path(payload["path"]).is_file()
    manifest = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    row = manifest["artifacts"][-1]
    assert row["source"] == "oel_review_plot_cli"
    assert row["extra"]["recipe_id"] == "relative_range"


def test_workflow_review_manifest_writes_queryable_tables_and_cli_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OEL_GENERATED_UTC", raising=False)
    (tmp_path / "controller_bench_summary.json").write_text("{}\n", encoding="utf-8")
    outputs = write_workflow_review(
        output_dir=tmp_path,
        workflow_type="controller_bench",
        title="Demo Bench",
        scenario_name="demo_bench",
        status="complete",
        summary={"run_count": 2, "passed_runs": 1},
        artifacts={"summary_json": str(tmp_path / "controller_bench_summary.json")},
        recommended_queries=[
            {
                "name": "controller_bench_runs",
                "description": "Run rows.",
                "sql": "SELECT variant_name, case_name, passed FROM bench_runs",
            }
        ],
        recommended_review_order=["Query bench_runs."],
        source_config="bench.yaml",
        tables={
            "bench_runs": [
                {"variant_name": "a", "case_name": "nominal", "passed": True, "failure_count": 0},
                {"variant_name": "b", "case_name": "nominal", "passed": False, "failure_count": 1},
            ]
        },
    )

    assert Path(outputs["workflow_manifest_json"]).is_file()
    assert Path(outputs["sqlite"]).is_file()
    manifest = load_workflow_manifest(tmp_path)
    assert manifest["workflow_type"] == "controller_bench"
    assert manifest["generated_utc"] == ""

    workspace = ReviewWorkspace.open(tmp_path)
    assert "bench_runs" in workspace.tables()
    query = workspace.query("SELECT variant_name, passed FROM bench_runs ORDER BY variant_name")
    assert query.rows == [{"variant_name": "a", "passed": 1}, {"variant_name": "b", "passed": 0}]

    proc = subprocess.run(
        [sys.executable, "-m", "sim.review", str(tmp_path), "--manifest"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "workflow_type: controller_bench" in proc.stdout


def test_workflow_review_manifest_removes_stale_store_when_tables_are_absent(tmp_path: Path) -> None:
    write_workflow_review(
        output_dir=tmp_path,
        workflow_type="validation",
        tables={
            "validation_benchmarks": [
                {
                    "benchmark_name": "old",
                    "kind": "smoke",
                    "passed": True,
                    "duration_s": 1.0,
                    "output_dir": "old",
                }
            ]
        },
        recommended_queries=[{"name": "old", "sql": "SELECT benchmark_name FROM validation_benchmarks"}],
    )

    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    outputs = write_workflow_review(
        output_dir=tmp_path,
        workflow_type="validation",
        artifacts={"summary_json": str(tmp_path / "summary.json")},
        tables={},
        recommended_queries=[],
    )
    manifest = load_workflow_manifest(tmp_path)

    assert "sqlite" not in outputs
    assert manifest["sqlite"] == ""
    assert manifest["schema_json"] == ""
    assert manifest["saved_views_json"] == ""
    assert not (tmp_path / "review" / "run.sqlite").exists()
    assert not (tmp_path / "review" / "schema.json").exists()
    assert not (tmp_path / "review" / "saved_views.json").exists()
    with pytest.raises(ReviewStoreNotFoundError):
        ReviewWorkspace.open(tmp_path)


def test_quickstart_config_can_emit_review_store_tables(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs" / "quickstart_5min.yaml").read_text(encoding="utf-8"))
    outdir = tmp_path / "quickstart_review"
    config["outputs"]["output_dir"] = str(outdir)
    config["outputs"]["stats"]["print_summary"] = False
    config["simulator"]["duration_s"] = 5.0
    config["outputs"].setdefault("review", {})
    config["outputs"]["review"] = {"enabled": True, "detail": "standard"}

    cfg_path = tmp_path / "quickstart_review.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_simulation_config_file(cfg_path)

    summary = json.loads((outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    db_path = outdir / "review" / "run.sqlite"
    assert db_path.is_file()
    assert Path(summary["review_outputs"]["sqlite"]) == db_path

    counts = _review_table_counts(db_path)
    assert counts["time_samples"] == summary["samples"]
    assert counts["objects"] == 2
    assert counts["object_state"] == summary["samples"] * 2
    assert counts["relative_state"] == summary["samples"]
    assert counts["thrust"] == summary["samples"] * 2
    assert counts["metrics"] >= 4
    assert counts["artifacts"] >= 1
    with sqlite3.connect(db_path) as conn:
        artifact_paths = [row[0] for row in conn.execute("SELECT path FROM artifacts ORDER BY artifact_id")]
        config_path = conn.execute("SELECT config_path FROM run_metadata").fetchone()[0]
    assert {
        "index.md",
        "master_run_summary.json",
        "review/run.sqlite",
        "review/schema.json",
    }.issubset(set(artifact_paths))
    assert config_path == str(cfg_path.resolve())


def test_plotting_config_review_store_indexes_plot_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs" / "plotting_rendezvous_demo.yaml").read_text(encoding="utf-8"))
    outdir = tmp_path / "plotting_review"
    config["outputs"]["output_dir"] = str(outdir)
    config["outputs"]["stats"]["print_summary"] = False
    config["simulator"]["duration_s"] = 5.0
    config["outputs"]["plots"]["enabled"] = True
    config["outputs"]["plots"]["preset"] = []
    config["outputs"]["plots"]["figure_ids"] = ["relative_range"]
    config["outputs"]["animations"]["enabled"] = False
    config["outputs"].setdefault("review", {})
    config["outputs"]["review"] = {"enabled": True, "detail": "standard"}

    cfg_path = tmp_path / "plotting_review.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_simulation_config_file(cfg_path)

    summary = json.loads((outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    db_path = outdir / "review" / "run.sqlite"
    assert db_path.is_file()
    assert (outdir / "relative_ranges.png").is_file()
    assert "relative_ranges" in dict(summary.get("plot_outputs", {}) or {})

    with sqlite3.connect(db_path) as conn:
        scenario_name = conn.execute("SELECT scenario_name FROM run_metadata").fetchone()[0]
        artifact_rows = conn.execute(
            "SELECT artifact_id, artifact_type, path FROM artifacts ORDER BY artifact_id"
        ).fetchall()
        relative_rows = conn.execute("SELECT COUNT(*) FROM relative_state").fetchone()[0]
        ground_access_rows = conn.execute("SELECT COUNT(*) FROM ground_access").fetchone()[0]

    assert scenario_name == "plotting_rendezvous_demo"
    assert relative_rows == int(summary["samples"])
    assert ground_access_rows > 0
    assert ("plots:relative_ranges", "plots", "relative_ranges.png") in artifact_rows


def test_ground_access_windows_distinguish_sample_span_and_credited_duration() -> None:
    payload = {
        "ground_station_access": {
            "station-a": {
                "targets": {
                    "sat-a": {
                        "access": [False, True, True, False],
                        "line_of_sight": [False, True, True, True],
                        "range_km": [1200.0, 900.0, 800.0, 1000.0],
                        "elevation_deg": [0.0, 30.0, 40.0, 10.0],
                        "reason": ["below_horizon", "available", "available", "below_mask"],
                    }
                }
            }
        }
    }
    with sqlite3.connect(":memory:") as connection:
        _create_schema(connection)
        _insert_ground_access(connection, t_s=np.array([0.0, 10.0, 20.0, 30.0]), payload=payload)
        row = connection.execute(
            "SELECT start_s, end_s, duration_s, last_access_sample_s, first_no_access_sample_s, "
            "sample_span_s, credited_duration_s, refinement_status FROM ground_access_windows"
        ).fetchone()

    assert row[:-1] == pytest.approx((10.0, 20.0, 20.0, 20.0, 30.0, 10.0, 20.0))
    assert row[-1] == "sampled_no_interpolation"


def test_auto_plot_reuses_its_bounded_query_result(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    class FakeWorkspace:
        output_dir = tmp_path

        def query(self, sql: str, *, max_rows: int) -> ReviewQueryResult:
            calls.append(sql)
            assert max_rows == 5000
            return ReviewQueryResult(
                columns=["time_s", "range_km"],
                rows=[{"time_s": 0.0, "range_km": 1.0}],
                row_count=1,
            )

    captured: dict[str, object] = {}

    def capture(workspace, spec, *, result, path=None, record=True):
        captured["result"] = result
        captured["path"] = path
        return "plot"

    monkeypatch.setattr(review_plotting_module, "_save_review_plot_from_result", capture)
    plotter = EvidencePlotter.__new__(EvidencePlotter)
    plotter.workspace = FakeWorkspace()

    assert plotter.auto(sql="SELECT time_s, range_km FROM relative_state") == "plot"
    assert calls == ["SELECT time_s, range_km FROM relative_state"]
    assert captured["result"].row_count == 1


def _review_table_counts(db_path: Path) -> dict[str, int]:
    tables = [
        "run_metadata",
        "objects",
        "time_samples",
        "object_state",
        "relative_state",
        "thrust",
        "ground_access",
        "events",
        "metrics",
        "artifacts",
    ]
    with sqlite3.connect(db_path) as conn:
        return {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}


@pytest.fixture
def column_integrity_workspace(tmp_path: Path):
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    with sqlite3.connect(review_dir / "run.sqlite") as conn:
        conn.execute("CREATE TABLE evidence (value REAL)")
        conn.execute("INSERT INTO evidence VALUES (2.0)")
    with ReviewWorkspace.open(tmp_path) as opened:
        yield opened


def test_case_distinct_aliases_preserve_their_own_values(column_integrity_workspace):
    result = column_integrity_workspace.query("SELECT value AS Range, value * 3 AS range FROM evidence")
    assert result.columns == ["Range", "range"]
    assert result.rows == [{"Range": 2.0, "range": 6.0}]


@pytest.mark.parametrize("where", ["", " WHERE 0"])
def test_duplicate_names_require_explicit_unique_aliases(column_integrity_workspace, where):
    with pytest.raises(ReviewQueryError, match="unique.*aliases"):
        column_integrity_workspace.query("SELECT a.value, b.value FROM evidence a JOIN evidence b" + where)


def test_review_line_keeps_gaps_for_nonfinite_measurements_and_abscissas() -> None:
    from matplotlib import pyplot as plt
    from matplotlib.path import Path as FigurePath

    figure, axis = plt.subplots()
    try:
        review_plotting_module._draw_series(
            axis,
            [{"t": 0.0, "y": 0.0}, {"t": 1.0, "y": float("inf")},
             {"t": 2.0, "y": 2.0}, {"t": 3.0, "y": 3.0},
             {"t": float("inf"), "y": 4.0}, {"t": 5.0, "y": 5.0}],
            "t", "y", "line",
        )
        codes = [code for _, code in axis.lines[0].get_path().iter_segments()]
        assert codes.count(FigurePath.MOVETO) == 3
        assert codes.count(FigurePath.LINETO) == 1
    finally:
        plt.close(figure)
