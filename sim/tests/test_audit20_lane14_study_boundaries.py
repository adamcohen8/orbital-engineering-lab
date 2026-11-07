from __future__ import annotations

import json
from pathlib import Path

import integrations.oel_mcp.study_handlers as study_handlers
from integrations.oel_mcp.study_handlers import StudyPlanningMCPHandlers
from sim.study_planning.config_validation import validate_config_path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "study_planning"
HANDLING = {"marking": "PUBLIC TEST FIXTURE", "release_scope": "public", "owner": "OEL"}


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_config_validation_uses_descriptor_bound_content_snapshot(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "configs" / "automation_smoke.yaml"
    snapshot = source.read_bytes()
    selected = tmp_path / "selected.yaml"
    selected.write_bytes(snapshot)

    # A caller that already authenticated the bytes must not have them
    # replaced by a later path reopen. The path remains provenance for
    # relative references, while parsing and hashing use ``snapshot``.
    selected.write_text("this: is not a scenario\n", encoding="utf-8")
    receipt, normalized = validate_config_path(
        "scenario_main",
        selected,
        workspace_root=tmp_path,
        content=snapshot,
    )

    assert receipt["source_sha256"]
    assert receipt["valid"] is True
    assert normalized is not None
    assert normalized["scenario_name"] == "automation_smoke"


def test_study_mcp_preflight_keeps_authorized_config_snapshot(monkeypatch, tmp_path: Path) -> None:
    selected = tmp_path / "selected.yaml"
    selected.write_bytes((ROOT / "configs" / "automation_smoke.yaml").read_bytes())
    handler = StudyPlanningMCPHandlers(read_roots=(tmp_path,))
    original = study_handlers.validate_config_path

    def replace_visible_path(config_ref: str, path: Path, **kwargs):
        path.write_text("this: is not a scenario\n", encoding="utf-8")
        return original(config_ref, path, **kwargs)

    monkeypatch.setattr(study_handlers, "validate_config_path", replace_visible_path)
    response = handler.call(
        "oel.study.preflight.v1",
        {
            "request": _load_fixture("passive_request.json"),
            "plan": _load_fixture("passive_plan.json"),
            "configs": [{"config_ref": "scenario_main", "path": str(selected)}],
            "handling": HANDLING,
        },
    )

    assert response["status"] == "completed", response["error"]
    assert response["result"]["status"] == "PLAN_VALID"
