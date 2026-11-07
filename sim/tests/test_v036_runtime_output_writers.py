"""Focused regressions for descriptor-bound report writers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from sim.evidence.workflow import (
    artifact_reference,
    build_workflow_evidence,
    load_workflow_evidence,
    write_workflow_evidence,
)
from sim.reporting.output_index import write_output_index
from sim.reporting.single_run_artifacts import SingleRunArtifactContext, _write_history_csv
from sim.security.output_writes import SecureWriteRoot, bind_output_writer


def _context(outdir: Path) -> SingleRunArtifactContext:
    return SingleRunArtifactContext(
        cfg=object(),  # The focused history writer only consumes the fields below.
        outdir=outdir,
        t_s=np.asarray([0.0, 1.0]),
        truth_hist={"target": np.zeros((2, 14), dtype=float)},
        target_reference_orbit_truth=None,
        belief_hist={},
        thrust_hist={},
        torque_hist={},
        desired_attitude_hist={},
        knowledge_hist={},
        knowledge_measurement_hist={},
        rocket_metrics={},
        reentry_metrics={},
        bridge_hist={},
        object_state_frames={},
    )


def test_public_run_retains_logical_output_dir_in_materialized_config(tmp_path: Path) -> None:
    """The executed config and retained effective config share the published path."""

    from sim.tests.test_oel_mcp_m4 import (
        EXECUTION_APPROVAL,
        TRUST_APPROVAL,
        _handlers,
        _scenario_arguments,
    )

    arguments = _scenario_arguments(tmp_path, "logical-config")
    handlers = _handlers(tmp_path)
    validation = handlers.validate_scenario(
        **arguments,
        trust_plugins=True,
        trust_approval=TRUST_APPROVAL,
    )
    result = handlers.run_scenario(
        **arguments,
        validation_id=validation["result"]["identity"]["validation_id"],
        trust_approval=TRUST_APPROVAL,
        approval=EXECUTION_APPROVAL,
    )

    assert result["status"] == "completed"
    output_dir = Path(str(arguments["output_dir"]))
    materialized = yaml.safe_load((output_dir / "mcp_execution_config.yaml").read_text(encoding="utf-8"))
    effective = json.loads((output_dir / "effective_config.json").read_text(encoding="utf-8"))
    assert materialized["outputs"]["output_dir"] == str(output_dir)
    assert effective["outputs"]["output_dir"] == str(output_dir)


def test_public_run_survives_visible_parent_replacement_after_open(tmp_path: Path) -> None:
    """The actual MCP run keeps evidence on its retained destination descriptor."""

    from sim.reporting import run_payload_assembly
    from sim.tests.test_oel_mcp_m4 import (
        EXECUTION_APPROVAL,
        ROOT,
        TRUST_APPROVAL,
        _handlers,
    )

    config = yaml.safe_load((ROOT / "configs" / "automation_smoke.yaml").read_text(encoding="utf-8"))
    config["outputs"]["review"] = {"enabled": True, "detail": "standard"}
    config_path = tmp_path / "reviewed.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "replaced-parent"
    arguments = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "resource_profile": "laptop-safe",
        "handling": {"marking": "PUBLIC", "release_scope": "public"},
    }
    handlers = _handlers(tmp_path)
    validation = handlers.validate_scenario(
        **arguments,
        trust_plugins=True,
        trust_approval=TRUST_APPROVAL,
    )
    moved = tmp_path / "replaced-parent-retained"
    attacker = tmp_path / "replaced-parent-attacker"
    attacker.mkdir()
    original_writer = run_payload_assembly.write_single_run_artifacts

    def replace_visible_parent(*args: object, **kwargs: object) -> dict[str, object]:
        output_dir.rename(moved)
        output_dir.symlink_to(attacker, target_is_directory=True)
        return original_writer(*args, **kwargs)

    with patch.object(
        run_payload_assembly,
        "write_single_run_artifacts",
        side_effect=replace_visible_parent,
    ):
        result = handlers.run_scenario(
            **arguments,
            validation_id=validation["result"]["identity"]["validation_id"],
            trust_approval=TRUST_APPROVAL,
            approval=EXECUTION_APPROVAL,
        )

    assert result["status"] == "completed"
    assert output_dir.is_symlink()
    assert list(attacker.iterdir()) == []
    manifest = json.loads((moved / "mcp_execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts_complete"] is True
    assert (moved / "review" / "run.sqlite").is_file()
    materialized = yaml.safe_load((moved / "mcp_execution_config.yaml").read_text(encoding="utf-8"))
    effective = json.loads((moved / "effective_config.json").read_text(encoding="utf-8"))
    assert materialized["outputs"]["output_dir"] == str(output_dir)
    assert effective["outputs"]["output_dir"] == str(output_dir)
    for record in manifest["artifact_records"]:
        logical = Path(str(record["path"]))
        relative = logical.relative_to(output_dir)
        retained = moved / relative
        data = retained.read_bytes()
        assert record["bytes"] == len(data)
        assert record["sha256"] == hashlib.sha256(data).hexdigest()


def test_history_writer_survives_parent_rename_and_rejects_symlink_path(tmp_path: Path) -> None:
    approved = tmp_path / "approved"
    moved = tmp_path / "moved"
    attacker = tmp_path / "attacker"
    attacker.mkdir()

    with SecureWriteRoot.open(
        approved,
        approved_roots=(tmp_path,),
        create=True,
        require_empty=True,
    ) as writer:
        approved.rename(moved)
        approved.symlink_to(attacker, target_is_directory=True)
        with bind_output_writer(writer):
            logical = _write_history_csv(context=_context(approved))

    assert logical == approved / "master_run_history.csv"
    assert (moved / "master_run_history.csv").is_file()
    assert not (attacker / "master_run_history.csv").exists()


def test_output_index_keeps_logical_path_after_parent_rename(tmp_path: Path) -> None:
    approved = tmp_path / "approved"
    moved = tmp_path / "moved"

    with SecureWriteRoot.open(
        approved,
        approved_roots=(tmp_path,),
        create=True,
        require_empty=True,
    ) as writer:
        approved.rename(moved)
        with bind_output_writer(writer):
            index = write_output_index(
                outdir=approved,
                workflow="other",
                title="rename-regression",
                artifacts={"summary_json": str(approved / "summary.json")},
            )

    assert index == approved / "index.md"
    assert (moved / "index.md").is_file()
    assert "summary.json" in (moved / "index.md").read_text(encoding="utf-8")


def test_workflow_evidence_uses_descriptor_hashes_after_parent_rename(tmp_path: Path) -> None:
    approved = tmp_path / "approved"
    moved = tmp_path / "moved"
    attacker = tmp_path / "attacker"
    attacker.mkdir()

    with SecureWriteRoot.open(
        approved,
        approved_roots=(tmp_path,),
        create=True,
        require_empty=True,
    ) as writer:
        writer.write_bytes("input.txt", b"payload")
        approved.rename(moved)
        approved.symlink_to(attacker, target_is_directory=True)
        with bind_output_writer(writer):
            payload = build_workflow_evidence(
                workflow_id="rename-regression",
                status="completed",
                disposition="accepted",
                artifacts=[
                    {
                        "artifact_id": "input",
                        "path": str(approved / "input.txt"),
                        "media_type": "text/plain",
                        "required": True,
                    }
                ],
            )
            evidence_path = write_workflow_evidence(approved / "evidence.json", payload)
            loaded = load_workflow_evidence(evidence_path)

    assert evidence_path == approved / "evidence.json"
    assert (moved / "evidence.json").is_file()
    assert not (attacker / "evidence.json").exists()
    assert loaded["artifacts"][0]["path"] == str(approved / "input.txt")
    assert loaded["artifacts"][0]["sha256"] == artifact_reference(
        moved / "input.txt",
        artifact_id="input",
        media_type="text/plain",
    )["sha256"]
