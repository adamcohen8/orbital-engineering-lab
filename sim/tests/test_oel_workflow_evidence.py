from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.evidence import artifact_reference, build_workflow_evidence, load_workflow_evidence, write_workflow_evidence

ROOT = Path(__file__).resolve().parents[2]


def test_workflow_evidence_tracks_required_artifact_integrity_and_matches_schema(tmp_path: Path) -> None:
    present = tmp_path / "present.json"
    present.write_text("{}\n", encoding="utf-8")
    payload = build_workflow_evidence(
        workflow_id="test_workflow",
        status="completed",
        disposition="ready",
        artifacts=[
            artifact_reference(present, artifact_id="present", media_type="application/json"),
            artifact_reference(
                tmp_path / "missing.json",
                artifact_id="missing",
                media_type="application/json",
            ),
        ],
        quality_gates={"test_passed": True},
        non_claims=["Test evidence only."],
    )
    path = write_workflow_evidence(tmp_path / "oel_workflow_evidence.json", payload)
    loaded = load_workflow_evidence(path)
    schema = json.loads(
        (ROOT / "docs/contracts/schemas/oel-workflow-evidence-v1.schema.json").read_text(encoding="utf-8")
    )

    assert schema["properties"]["schema_id"]["const"] == loaded["schema_id"]
    assert set(schema["required"]) <= set(loaded)
    assert loaded["artifact_integrity"] == {
        "required_artifacts_present": False,
        "missing_required_artifact_ids": ["missing"],
    }
    assert json.loads(path.read_text(encoding="utf-8"))["workflow_id"] == "test_workflow"


def test_workflow_evidence_rederives_artifact_integrity_and_rejects_tampering(tmp_path: Path) -> None:
    missing = tmp_path / "not-there.json"
    payload = build_workflow_evidence(
        workflow_id="forgery_check",
        status="failed",
        disposition="missing_artifact",
        artifacts=[
            {
                "artifact_id": "forged",
                "path": str(missing),
                "media_type": "application/json",
                "required": True,
                "exists": True,
                "sha256": "a" * 64,
            }
        ],
    )
    assert payload["artifacts"][0]["exists"] is False
    path = write_workflow_evidence(tmp_path / "evidence.json", payload)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["artifacts"][0]["exists"] = True
    raw["artifacts"][0]["sha256"] = "a" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity mismatch"):
        load_workflow_evidence(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("required", "false"), ("required", 0), ("exists", "false"), ("exists", 1)],
)
def test_workflow_evidence_requires_boolean_artifact_flags(
    tmp_path: Path, field: str, value: object,
) -> None:
    present = tmp_path / "present.json"
    missing = tmp_path / "missing.json"
    present.write_text("{}\n", encoding="utf-8")
    payload = build_workflow_evidence(
        workflow_id="typed_flags",
        status="completed",
        disposition="ready",
        artifacts=[
            artifact_reference(
                present, artifact_id="present", media_type="application/json", required=False,
            ),
            artifact_reference(
                missing, artifact_id="missing", media_type="application/json", required=False,
            ),
        ],
    )
    path = write_workflow_evidence(tmp_path / "evidence.json", payload)
    loaded = load_workflow_evidence(path)
    assert loaded["artifacts"][0]["required"] is False
    assert loaded["artifacts"][0]["exists"] is True
    assert loaded["artifacts"][1]["required"] is False
    assert loaded["artifacts"][1]["exists"] is False

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["artifacts"][0][field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must be booleans"):
        load_workflow_evidence(path)
