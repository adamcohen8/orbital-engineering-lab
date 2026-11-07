from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import sim.agent_task.runner as agent_task_runner
from sim.agent_task import compare_configs, create_plot, inspect_output, list_plot_recipes, run_recipe
from sim.agent_task.failures import diagnose_failure
from sim.agent_task.models import (
    AGENT_TASK_MATURITY_LEVELS,
    AgentPlotRecipe,
    AgentTaskRecipe,
    EvidencePacket,
    SemanticMetric,
)
from sim.agent_task.plot_recipes import get_plot_recipe
from sim.agent_task.recipes import list_recipes
from sim.agent_task.semantics import list_semantic_metrics
from sim.execution import run_simulation_config_file
from sim.review.plot_recipes import REVIEW_PLOT_RECIPES
from sim.review.queries import SAVED_REVIEW_QUERIES, SavedReviewQuery

ROOT = Path(__file__).resolve().parents[2]


def _review_store_config(output_dir: Path) -> dict:
    return {
        "scenario_name": "agent_task_smoke",
        "scenario_description": "Agent task smoke test",
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
                    "stack": "fsw.passive",
                    "hardware_profile": "hardware.passive.v1",
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


def _write_config(path: Path, output_dir: Path) -> Path:
    path.write_text(yaml.safe_dump(_review_store_config(output_dir), sort_keys=False), encoding="utf-8")
    return path


class _FakeOutputWriter:
    def __init__(self, physical_root: Path, logical_root: Path) -> None:
        self.path = physical_root.resolve()
        self.logical_root = logical_root.resolve()
        self.writes: list[Path] = []

    def write_bytes(self, relative: str | Path, payload: bytes, **_kwargs) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AssertionError(f"writer received an unsafe relative path: {relative}")
        target = self.path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        target.write_bytes(payload)
        self.writes.append(relative_path)
        return target


def test_agent_task_writer_publishes_logical_packet_paths(tmp_path: Path) -> None:
    physical = tmp_path / "staging"
    logical = tmp_path / "published"
    physical.mkdir()
    writer = _FakeOutputWriter(physical, logical)

    payload = run_recipe(
        "quickstart_review",
        output_dir=logical,
        dry_run=True,
        output_writer=writer,
    )

    assert payload["status"] == "validated"
    assert payload["packet_path"] == str(logical / "agent_evidence_packet.json")
    assert payload["configs"][0]["config_path"] == str(logical / "agent_task_config.yaml")
    assert payload["configs"][0]["output_dir"] == str(logical)
    assert writer.writes == [Path("agent_task_config.yaml"), Path("agent_evidence_packet.json")]
    packet_path = physical / "agent_evidence_packet.json"
    assert packet_path.is_file()
    assert str(physical) not in packet_path.read_text(encoding="utf-8")


def test_agent_task_recipe_dry_run_writes_evidence_packet(tmp_path: Path) -> None:
    payload = run_recipe("quickstart_review", output_root=tmp_path, dry_run=True)

    assert payload["status"] == "validated"
    assert payload["recipe"]["recipe_id"] == "quickstart_review"
    assert payload["configs"][0]["review_enabled"] is True
    assert payload["caveats"] == ["Dry run requested: scenario was validated but not executed."]
    assert Path(payload["packet_path"]).is_file()

    config = yaml.safe_load(Path(payload["configs"][0]["config_path"]).read_text(encoding="utf-8"))
    assert config["outputs"]["review"]["enabled"] is True
    assert Path(config["outputs"]["output_dir"]) == tmp_path / "quickstart_review"
    assert payload["evidence_summary"] == {
        "status": "validated",
        "validation_ok": True,
        "review_evidence_complete": None,
        "artifacts_complete": None,
        "plots_complete": None,
        "comparison_complete": None,
        "failure_hint_count": 0,
        "caveat_count": 1,
        "ready_to_cite": False,
        "packet_mode": "inspection_only",
        "readiness_blockers": ["status:validated", "run_evidence_absent"],
    }


def test_agent_task_prepared_config_cache_requires_exact_file_bytes(tmp_path: Path) -> None:
    prepared = agent_task_runner._prepare_config(
        ROOT / "configs" / "quickstart_5min.yaml",
        tmp_path / "prepared",
        label="guarded",
    )
    task_path = Path(prepared["config_path"])

    assert agent_task_runner._guarded_prepared_config(task_path) is not None

    task_path.write_bytes(task_path.read_bytes() + b"\n# changed after preparation\n")

    with pytest.raises(RuntimeError, match="changed after validation"):
        agent_task_runner._guarded_prepared_config(task_path)

    with pytest.raises(RuntimeError, match="changed after validation"):
        agent_task_runner._run_prepared_config(task_path)


def test_agent_task_recipe_with_plots_writes_plot_summary(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _write_config(config_dir / "quickstart_5min.yaml", tmp_path / "unused")
    monkeypatch.setattr(agent_task_runner, "_repo_root", lambda: tmp_path)

    payload = run_recipe("quickstart_review", output_root=tmp_path, make_plots=True)

    assert payload["status"] == "completed"
    assert payload["plots"]
    assert payload["plot_summary"] == {
        "total": 2,
        "ok": 2,
        "failed": 0,
        "missing": 0,
        "truncated": 0,
        "failed_plots": [],
        "missing_plots": [],
        "truncated_plots": [],
        "plots_complete": True,
    }
    assert all(plot["path_exists"] is True for plot in payload["plots"])
    assert payload["evidence_summary"] == {
        "status": "completed",
        "validation_ok": True,
        "review_evidence_complete": True,
        "artifacts_complete": True,
        "plots_complete": True,
        "comparison_complete": None,
        "failure_hint_count": 0,
        "caveat_count": 0,
        "ready_to_cite": True,
        "packet_mode": "execution",
        "readiness_blockers": [],
    }


def test_agent_task_ogp_sgp4_recipe_packages_provenance_state_and_plot(tmp_path: Path) -> None:
    payload = run_recipe("ogp_sgp4_review", output_root=tmp_path, make_plots=True)

    assert payload["status"] == "completed"
    assert payload["recipe"]["maturity"] == "supported"
    queries = {row["name"]: row for row in payload["review"]["queries"]}
    propagation = queries["ogp_propagation_contract"]
    assert propagation["status"] == "ok"
    assert propagation["rows"] == [
        {
                "object_id": "catalog_object",
                "propagation_method": "general",
                "propagator_family": "OGP",
                "propagator_name": "OGP-SGP4",
                "general_model": "sgp4",
                "ogp_regime": "near_earth",
                "orbital_period_min": pytest.approx(92.9135521221465),
                "native_frame": "teme",
                "output_frame": "teme",
                "state_history_frame": "eci",
                "frame_transform": "native",
                "state_frame": "eci",
                "tle_epoch_jd_utc": 2460310.5,
                "tle_age_start_days": 0.0,
                "tle_age_end_days": pytest.approx(7200.0 / 86400.0),
                "max_tle_age_days_warning": None,
                "tle_age_warning": 0,
        }
    ]
    assert queries["passive_final_state"]["status"] == "ok"
    assert len(queries["passive_final_state"]["rows"]) == 1
    assert payload["plots"][0]["recipe_id"] == "object_eci_radius"
    assert payload["plots"][0]["status"] == "ok"
    assert Path(payload["plots"][0]["path"]).is_file()
    assert payload["evidence_summary"]["ready_to_cite"] is True


def test_agent_task_inspects_completed_run_and_creates_plot(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    cfg_path = _write_config(tmp_path / "scenario.yaml", outdir)
    run_simulation_config_file(cfg_path)

    payload = inspect_output(
        outdir,
        query_names=("run_metadata", "rendezvous_closest_approach", "artifacts"),
        semantic_metric_names=("closest_approach_km", "not_recorded_metric"),
    )

    assert payload["status"] == "completed"
    assert payload["semantic_metrics"][0]["name"] == "closest_approach_km"
    assert payload["semantic_metrics"][0]["maturity"] == "supported"
    assert "relative_state" in payload["semantic_metrics"][0]["source_tables"]
    assert payload["semantic_metric_requests"] == [
        {
            "name": "closest_approach_km",
            "known": True,
            "maturity": "supported",
            "source_tables": ["metrics", "relative_state"],
            "saved_query": "rendezvous_closest_approach",
        },
        {
            "name": "not_recorded_metric",
            "known": False,
            "reason": "unknown_semantic_metric",
        },
    ]
    assert (outdir / "agent_evidence_packet.json").is_file()
    query_statuses = {query["name"]: query["status"] for query in payload["review"]["queries"]}
    assert query_statuses["run_metadata"] == "ok"
    assert query_statuses["rendezvous_closest_approach"] == "ok"
    assert payload["review"]["query_summary"] == {
        "total": 3,
        "ok": 3,
        "failed": 0,
        "unknown": 0,
        "unexpected_empty": 0,
        "truncated": 0,
        "failed_queries": [],
        "unknown_queries": [],
        "unexpected_empty_queries": [],
        "truncated_queries": [],
        "evidence_complete": True,
    }
    queries_by_name = {query["name"]: query for query in payload["review"]["queries"]}
    closest_query = queries_by_name["rendezvous_closest_approach"]
    assert closest_query["known"] is True
    assert closest_query["maturity"] == "supported"
    assert closest_query["source_tables"] == ["relative_state"]
    assert closest_query["allow_empty"] is False
    assert closest_query["empty_result"] is False
    assert closest_query["empty_result_allowed"] is False
    assert closest_query["empty_result_unexpected"] is False
    artifact_by_id = {artifact["artifact_id"]: artifact for artifact in payload["artifacts"]}
    assert artifact_by_id["summary_json"]["path"] == "master_run_summary.json"
    assert artifact_by_id["summary_json"]["path_exists"] is True
    assert Path(artifact_by_id["summary_json"]["resolved_path"]).is_file()
    assert artifact_by_id["run_log_json"]["path_exists"] is True
    assert artifact_by_id["effective_config_json"]["path"] == "effective_config.json"
    assert artifact_by_id["effective_config_json"]["path_exists"] is True
    assert artifact_by_id["output_index_md"]["path"] == "index.md"
    assert artifact_by_id["output_index_md"]["path_exists"] is True
    assert artifact_by_id["review_store:sqlite"]["path"] == "review/run.sqlite"
    assert artifact_by_id["review_store:sqlite"]["path_exists"] is True
    assert artifact_by_id["review_store:schema_json"]["path"] == "review/schema.json"
    assert artifact_by_id["review_store:schema_json"]["path_exists"] is True
    assert payload["artifact_summary"] == {
        "total": 6,
        "existing": 6,
        "missing": 0,
        "path_status_unknown": 0,
        "missing_artifacts": [],
        "path_status_unknown_artifacts": [],
        "artifacts_complete": True,
    }

    plot = create_plot(outdir, "relative_range", style_name="oel_light")
    assert plot["status"] == "ok"
    assert plot["recipe_maturity"] == "supported"
    assert plot["source_tables"] == ["relative_state"]
    assert plot["semantic_metric_names"] == ["closest_approach_km", "final_range_km"]
    assert plot["path_exists"] is True
    assert plot["resolved_path"] == plot["path"]
    assert Path(plot["path"]).is_file()
    generated = json.loads((outdir / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    assert generated["artifacts"][-1]["source"] == "oel_review_plot_api"


def test_agent_task_compare_configs_writes_metric_deltas(tmp_path: Path) -> None:
    base_cfg = _write_config(tmp_path / "base.yaml", tmp_path / "base_source")
    candidate_cfg = _write_config(tmp_path / "candidate.yaml", tmp_path / "candidate_source")

    payload = compare_configs(
        base_cfg,
        candidate_cfg,
        output_dir=tmp_path / "compare",
        metric_names=("closest_approach_km", "not_recorded_metric"),
    )

    assert payload["status"] == "completed"
    assert payload["evidence_summary"]["comparison_complete"] is False
    assert payload["evidence_summary"]["ready_to_cite"] is False
    assert payload["comparison"]["summary"] == {
        "total": 2,
        "unknown_metrics": ["not_recorded_metric"],
        "missing_value_metrics": [],
        "missing_delta_metrics": [],
        "partial_inspections": [],
        "complete": False,
    }
    assert payload["comparison"]["query_names"] == [
        "run_metadata",
        "artifacts",
        "rendezvous_closest_approach",
    ]
    assert payload["comparison"]["metrics"]["base"]["closest_approach_km"] is not None
    assert payload["comparison"]["metrics"]["candidate"]["closest_approach_km"] is not None
    assert "closest_approach_time_s" not in payload["comparison"]["metrics"]["base"]
    assert "closest_approach_time_s" not in payload["comparison"]["metrics"]["candidate"]
    assert payload["comparison"]["deltas"]["closest_approach_km"] == 0.0
    assert set(payload["comparison"]["deltas"]) == {"closest_approach_km"}
    status_by_name = {row["name"]: row for row in payload["comparison"]["metric_status"]}
    assert status_by_name["closest_approach_km"] == {
        "name": "closest_approach_km",
        "base_available": True,
        "candidate_available": True,
        "delta_available": True,
        "semantic_metric_known": True,
        "maturity": "supported",
        "source_tables": ["metrics", "relative_state"],
        "saved_query": "rendezvous_closest_approach",
        "query_status_by_label": {"base": "ok", "candidate": "ok"},
    }
    assert status_by_name["not_recorded_metric"] == {
        "name": "not_recorded_metric",
        "base_available": False,
        "candidate_available": False,
        "delta_available": False,
        "semantic_metric_known": False,
        "reason": "unknown_semantic_metric",
    }
    request_by_name = {row["name"]: row for row in payload["semantic_metric_requests"]}
    assert request_by_name["closest_approach_km"]["known"] is True
    assert request_by_name["not_recorded_metric"] == {
        "name": "not_recorded_metric",
        "known": False,
        "reason": "unknown_semantic_metric",
    }
    assert (tmp_path / "compare" / "agent_evidence_packet.json").is_file()


def test_agent_task_compare_runs_requested_semantic_query_but_flags_non_scalar_metric(tmp_path: Path) -> None:
    base_cfg = _write_config(tmp_path / "base.yaml", tmp_path / "base_source")
    candidate_cfg = _write_config(tmp_path / "candidate.yaml", tmp_path / "candidate_source")

    payload = compare_configs(
        base_cfg,
        candidate_cfg,
        output_dir=tmp_path / "compare",
        metric_names=("burn_activity",),
    )

    assert payload["status"] == "completed"
    assert payload["evidence_summary"]["comparison_complete"] is False
    assert payload["evidence_summary"]["ready_to_cite"] is False
    assert "burn_activity" in payload["comparison"]["query_names"]
    assert [query["name"] for query in payload["review"]["base"]["queries"]] == [
        "run_metadata",
        "artifacts",
        "burn_activity",
    ]
    status = payload["comparison"]["metric_status"][0]
    assert status["name"] == "burn_activity"
    assert status["semantic_metric_known"] is True
    assert status["query_status_by_label"] == {"base": "ok", "candidate": "ok"}
    assert status["base_available"] is False
    assert status["candidate_available"] is False
    assert status["delta_available"] is False
    assert status["reason"] == "no_scalar_reducer"
    assert payload["comparison"]["summary"] == {
        "total": 1,
        "unknown_metrics": [],
        "missing_value_metrics": ["burn_activity"],
        "missing_delta_metrics": [],
        "partial_inspections": [],
        "complete": False,
    }


def test_agent_task_compare_propagates_partial_inspection(monkeypatch, tmp_path: Path) -> None:
    base_cfg = _write_config(tmp_path / "base.yaml", tmp_path / "base_source")
    candidate_cfg = _write_config(tmp_path / "candidate.yaml", tmp_path / "candidate_source")

    monkeypatch.setattr(
        agent_task_runner,
        "run_simulation_config_file",
        lambda _path: {"summary": {"scenario_name": "stub"}},
    )

    def fake_inspect(output_dir: Path, **_kwargs) -> dict:
        label = Path(output_dir).name
        if label == "candidate":
            return {
                "status": "partial",
                "review": {"output_dir": str(output_dir), "error": "Review store not found"},
                "artifacts": [],
                "artifact_summary": {"artifacts_complete": True},
                "failure_hints": [{"code": "review_store_missing", "next_step": "rerun with review"}],
            }
        return {
            "status": "completed",
            "review": {"query_summary": {"evidence_complete": True}, "queries": []},
            "artifacts": [],
            "artifact_summary": {"artifacts_complete": True},
            "failure_hints": [],
        }

    monkeypatch.setattr(agent_task_runner, "inspect_output", fake_inspect)

    payload = agent_task_runner.compare_configs(
        base_cfg,
        candidate_cfg,
        output_dir=tmp_path / "compare",
        metric_names=("closest_approach_km",),
    )

    assert payload["status"] == "partial"
    assert payload["failure_hints"] == [
        {"label": "candidate", "code": "review_store_missing", "next_step": "rerun with review"}
    ]
    assert payload["comparison"]["inspection_statuses"] == {"base": "completed", "candidate": "partial"}
    assert payload["comparison"]["summary"]["partial_inspections"] == ["candidate"]
    assert payload["evidence_summary"]["comparison_complete"] is False
    assert payload["evidence_summary"]["ready_to_cite"] is False


def test_saved_query_rows_flag_empty_result_policy(monkeypatch) -> None:
    class EmptyWorkspace:
        def query(self, _sql: str, *, max_rows: int, max_vm_steps: int) -> SimpleNamespace:
            assert max_rows == 5
            assert max_vm_steps == 250_000
            return SimpleNamespace(columns=[], rows=[], row_count=0, truncated=False)

    def saved_query(name: str) -> SavedReviewQuery:
        return SavedReviewQuery(
            name=name,
            description=f"{name} query",
            sql="SELECT event_id FROM events WHERE event_type = 'not_present'",
            source_tables=("events",),
            allow_empty=name == "allowed_empty",
        )

    monkeypatch.setattr(agent_task_runner, "get_saved_review_query", saved_query)

    rows = agent_task_runner._run_saved_queries(EmptyWorkspace(), ("unexpected_empty", "allowed_empty"), max_rows=5)

    by_name = {row["name"]: row for row in rows}
    assert by_name["unexpected_empty"]["empty_result"] is True
    assert by_name["unexpected_empty"]["empty_result_allowed"] is False
    assert by_name["unexpected_empty"]["empty_result_unexpected"] is True
    assert by_name["allowed_empty"]["empty_result"] is True
    assert by_name["allowed_empty"]["empty_result_allowed"] is True
    assert by_name["allowed_empty"]["empty_result_unexpected"] is False


def test_default_inspection_queries_include_present_coverage_and_link_evidence() -> None:
    class DomainWorkspace:
        def tables(self) -> list[str]:
            return ["coverage_summary", "link_summary"]

        def query(self, sql: str, *, max_rows: int) -> SimpleNamespace:
            assert max_rows == 1
            assert "coverage_summary" in sql or "link_summary" in sql
            return SimpleNamespace(rows=[{"present": 1}])

    assert agent_task_runner._default_inspection_queries(DomainWorkspace()) == (
        "run_metadata",
        "objects",
        "artifacts",
        "coverage_summary",
        "coverage_transition_summary",
        "directed_link_summary",
        "directed_link_windows",
    )


def test_saved_query_rows_flag_unknown_query() -> None:
    rows = agent_task_runner._run_saved_queries(object(), ("not_a_saved_query",), max_rows=5)

    assert rows == [
        {
            "name": "not_a_saved_query",
            "known": False,
            "reason": "unknown_saved_query",
            "status": "unknown_query",
        }
    ]


def test_query_summary_aggregates_incomplete_evidence() -> None:
    summary = agent_task_runner._summarize_query_rows(
        [
            {"name": "ok_query", "status": "ok", "empty_result_unexpected": False, "truncated": False},
            {"name": "failed_query", "status": "failed"},
            {"name": "unknown_query", "status": "unknown_query"},
            {"name": "empty_query", "status": "ok", "empty_result_unexpected": True},
            {"name": "truncated_query", "status": "ok", "truncated": True},
        ]
    )

    assert summary == {
        "total": 5,
        "ok": 3,
        "failed": 1,
        "unknown": 1,
        "unexpected_empty": 1,
        "truncated": 1,
        "failed_queries": ["failed_query"],
        "unknown_queries": ["unknown_query"],
        "unexpected_empty_queries": ["empty_query"],
        "truncated_queries": ["truncated_query"],
        "evidence_complete": False,
    }


def test_query_summary_treats_truncation_as_incomplete_evidence() -> None:
    summary = agent_task_runner._summarize_query_rows(
        [
            {
                "name": "large_query",
                "status": "ok",
                "empty_result_unexpected": False,
                "truncated": True,
            }
        ]
    )

    assert summary == {
        "total": 1,
        "ok": 1,
        "failed": 0,
        "unknown": 0,
        "unexpected_empty": 0,
        "truncated": 1,
        "failed_queries": [],
        "unknown_queries": [],
        "unexpected_empty_queries": [],
        "truncated_queries": ["large_query"],
        "evidence_complete": False,
    }


def test_artifact_rows_include_path_existence(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("ok", encoding="utf-8")

    rows = agent_task_runner._artifact_rows(
        [
            {
                "name": "artifacts",
                "status": "ok",
                "rows": [
                    {"artifact_id": "existing", "path": "existing.txt"},
                    {"artifact_id": "missing", "path": "missing.txt"},
                    {"artifact_id": "empty", "path": ""},
                ],
            }
        ],
        output_dir=tmp_path,
    )

    by_id = {row["artifact_id"]: row for row in rows}
    assert by_id["existing"]["path_exists"] is True
    assert by_id["existing"]["resolved_path"] == str(existing)
    assert by_id["missing"]["path_exists"] is False
    assert by_id["missing"]["resolved_path"] == str(tmp_path / "missing.txt")
    assert by_id["empty"]["path_exists"] is False
    assert by_id["empty"]["resolved_path"] == ""


def test_artifact_summary_flags_missing_and_unknown_paths() -> None:
    summary = agent_task_runner._summarize_artifacts(
        [
            {"artifact_id": "existing", "path_exists": True},
            {"artifact_id": "missing", "path_exists": False},
            {"artifact_id": "unknown"},
        ]
    )

    assert summary == {
        "total": 3,
        "existing": 1,
        "missing": 1,
        "path_status_unknown": 1,
        "missing_artifacts": ["missing"],
        "path_status_unknown_artifacts": ["unknown"],
        "artifacts_complete": False,
    }


def test_plot_summary_flags_failed_missing_and_truncated_plots() -> None:
    summary = agent_task_runner._summarize_plots(
        [
            {"recipe_id": "ok_recipe", "artifact_id": "ok_plot", "status": "ok", "path_exists": True},
            {"recipe_id": "failed_recipe", "status": "failed"},
            {"recipe_id": "missing_recipe", "artifact_id": "missing_plot", "status": "ok", "path_exists": False},
            {"recipe_id": "truncated_recipe", "artifact_id": "truncated_plot", "status": "ok", "path_exists": True, "truncated": True},
        ]
    )

    assert summary == {
        "total": 4,
        "ok": 3,
        "failed": 1,
        "missing": 1,
        "truncated": 1,
        "failed_plots": ["failed_recipe"],
        "missing_plots": ["missing_plot"],
        "truncated_plots": ["truncated_plot"],
        "plots_complete": False,
    }


def test_packet_evidence_summary_flags_incomplete_components() -> None:
    packet = EvidencePacket(
        task_id="summary",
        status="completed",
        generated_utc="2026-06-23T00:00:00Z",
        validation={"base": {"ok": True}, "candidate": {"ok": False}},
        review={"base": {"query_summary": {"evidence_complete": True}}, "candidate": {"query_summary": {"evidence_complete": False}}},
        artifact_summary={"artifacts_complete": False},
        plot_summary={"plots_complete": False},
        failure_hints=[{"code": "missing"}],
        caveats=["review needed"],
    )

    assert agent_task_runner._summarize_packet_evidence(packet) == {
        "status": "completed",
        "validation_ok": False,
        "review_evidence_complete": False,
        "artifacts_complete": False,
        "plots_complete": False,
        "comparison_complete": None,
        "failure_hint_count": 1,
        "caveat_count": 1,
        "ready_to_cite": False,
        "packet_mode": "inspection_only",
        "readiness_blockers": [
            "run_evidence_absent",
            "validation_failed",
            "review_evidence_incomplete",
            "artifacts_incomplete",
            "plots_incomplete",
            "failure_hints_present",
        ],
    }


def test_agent_task_cli_lists_recipes_as_json() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "sim.agent_task", "list", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert any(item["recipe_id"] == "quickstart_review" for item in payload["items"])
    by_id = {item["recipe_id"]: item for item in payload["items"]}
    assert by_id["quickstart_review"]["maturity"] == "supported"


def test_agent_task_json_run_keeps_stdout_machine_parseable(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.agent_task",
            "run",
            "quickstart_review",
            "--output-root",
            str(tmp_path),
            "--dry-run",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["task_id"] == "quickstart_review"
    assert payload["status"] == "validated"


def test_agent_task_plain_list_surfaces_recipe_maturity() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "sim.agent_task", "list"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "quickstart_review: Quickstart Review Evidence [supported]" in proc.stdout


def test_agent_task_recipe_maturity_policy_is_explicit() -> None:
    recipes = list_recipes()

    assert recipes
    assert {recipe.maturity for recipe in recipes}.issubset(AGENT_TASK_MATURITY_LEVELS)
    public_recipes = [recipe for recipe in recipes if "public" in recipe.tags]
    assert public_recipes
    assert all(recipe.maturity == "supported" for recipe in public_recipes)


def test_agent_task_recipe_rejects_unknown_maturity() -> None:
    try:
        AgentTaskRecipe(
            recipe_id="bad",
            title="Bad",
            description="Bad maturity",
            config_path="configs/quickstart_5min.yaml",
            maturity="maybe",
        )
    except ValueError as exc:
        assert "Unknown agent task recipe maturity" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("AgentTaskRecipe accepted an unknown maturity value")


def test_agent_plot_recipe_maturity_policy_is_explicit() -> None:
    plot_recipes = list_plot_recipes()

    assert plot_recipes
    assert {recipe.maturity for recipe in plot_recipes}.issubset(AGENT_TASK_MATURITY_LEVELS)
    assert all(recipe.maturity == "supported" for recipe in plot_recipes)
    assert all(recipe.supported_tables for recipe in plot_recipes)
    assert all(recipe.sql.lstrip().upper().startswith(("SELECT", "WITH")) for recipe in plot_recipes)
    for recipe_id in ("relative_range", "relative_range_rate"):
        recipe = get_plot_recipe(recipe_id)
        assert recipe is not None
        assert recipe.group_column == "pair_id"
        assert "pair_id" in recipe.sql


def test_agent_task_uses_authoritative_review_plot_recipe_registry() -> None:
    from sim.agent_task.plot_recipes import PLOT_RECIPES

    assert PLOT_RECIPES is REVIEW_PLOT_RECIPES
    assert "relative_position_ric_2d" in PLOT_RECIPES
    assert PLOT_RECIPES["relative_position_ric_2d"].renderer_id == "ric_rectangular_2d"


def test_agent_plot_recipe_rejects_unknown_maturity_or_missing_tables() -> None:
    try:
        AgentPlotRecipe(
            recipe_id="bad_maturity",
            title="Bad",
            description="Bad maturity",
            sql="SELECT time_s FROM relative_state",
            x_column="time_s",
            y_columns=("time_s",),
            maturity="maybe",
            supported_tables=("relative_state",),
        )
    except ValueError as exc:
        assert "Unknown agent plot recipe maturity" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("AgentPlotRecipe accepted an unknown maturity value")

    try:
        AgentPlotRecipe(
            recipe_id="bad_tables",
            title="Bad",
            description="Missing tables",
            sql="SELECT time_s FROM relative_state",
            x_column="time_s",
            y_columns=("time_s",),
        )
    except ValueError as exc:
        assert "must declare supported_tables" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("AgentPlotRecipe accepted missing supported_tables")


def test_semantic_metric_contract_is_explicit() -> None:
    metrics = list_semantic_metrics()

    assert metrics
    assert {metric.maturity for metric in metrics}.issubset(AGENT_TASK_MATURITY_LEVELS)
    for metric in metrics:
        assert metric.source_tables
        if metric.table:
            assert metric.table in metric.source_tables
        if metric.saved_query:
            assert metric.saved_query in SAVED_REVIEW_QUERIES
            query = SAVED_REVIEW_QUERIES[metric.saved_query]
            assert query.maturity in AGENT_TASK_MATURITY_LEVELS
            assert set(metric.source_tables) & set(query.source_tables)
        if metric.sql:
            assert metric.sql.lstrip().upper().startswith(("SELECT", "WITH"))


def test_semantic_metric_rejects_unknown_maturity_mutating_sql_or_missing_evidence() -> None:
    try:
        SemanticMetric(name="bad_maturity", description="Bad", table="metrics", maturity="maybe")
    except ValueError as exc:
        assert "Unknown semantic metric maturity" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("SemanticMetric accepted an unknown maturity value")

    try:
        SemanticMetric(name="bad_sql", description="Bad", sql="DELETE FROM metrics")
    except ValueError as exc:
        assert "read-only SELECT/WITH SQL" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("SemanticMetric accepted mutating SQL")

    try:
        SemanticMetric(name="bad_evidence", description="Bad")
    except ValueError as exc:
        assert "must declare table, saved_query, or SQL evidence" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("SemanticMetric accepted missing evidence")


def test_agent_task_failure_hints_include_review_store_missing(tmp_path: Path) -> None:
    hints = diagnose_failure("Review store not found", output_dir=tmp_path / "missing")

    codes = {hint.code for hint in hints}
    assert "review_store_missing" in codes
    assert "review_db_absent_after_run" in codes
