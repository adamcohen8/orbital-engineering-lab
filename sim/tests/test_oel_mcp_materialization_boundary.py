from __future__ import annotations

import json
from pathlib import Path

import yaml

import integrations.oel_mcp.public_handlers as public_handlers_module
from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
from sim.interchange.materialization import _manifest_base, materialize_scenario_document
from sim.security.config_paths import ConfigPathPolicy

HANDLING = {"marking": "PUBLIC", "release_scope": "public"}
WRITE_APPROVAL = {"approval_id": "test-write", "scope": "write"}
TRUST_APPROVAL = {"approval_id": "test-trust", "scope": "trust"}


def test_public_onp_materialization_binds_logical_paths_and_blocks_raw_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    product_path = tmp_path / "state-product.json"
    product_path.write_text(json.dumps({"product_kind": "oel.state_estimate"}), encoding="utf-8")
    scenario_path = tmp_path / "handoff" / "continued.yaml"
    run_output_dir = tmp_path / "handoff" / "continued-run"
    observed: dict[str, object] = {}

    def fake_materialize_onp(_product_path: Path, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        policy = kwargs["validation_path_policy"]
        assert policy.allow_external_config_paths is False
        assert policy.config_path == scenario_path
        assert tuple(policy.read_roots) == tuple(handler.path_policy.read_roots)
        assert tuple(policy.write_roots) == tuple(handler.path_policy.write_roots)

        physical_scenario = Path(str(kwargs["scenario_path"]))
        physical_scenario.parent.mkdir(parents=True, exist_ok=True)
        physical_scenario.write_text(
            "outputs:\n"
            f"  output_dir: {kwargs['logical_output_dir']}\n"
            "metadata:\n"
            "  handoff:\n"
            f"    manifest_path: {kwargs['logical_manifest_path']}\n",
            encoding="utf-8",
        )
        physical_manifest = physical_scenario.with_name(
            f"{physical_scenario.stem}.handoff_manifest.json"
        )
        physical_manifest.write_text("{}", encoding="utf-8")
        return {
            "status": "failed",
            "scenario_path": str(physical_scenario),
            "manifest_path": str(physical_manifest),
            "manifest_id": "oel.handoff_manifest:" + "0" * 64,
            "failures": [{"code": "validation.failed"}],
            "recommended_next_action": "Review the generated scenario.",
            "safe_validation": {"ok": False},
            "ordinary_validation": {"ok": None},
            "product_validation": {"valid": True},
        }

    monkeypatch.setenv("OEL_ALLOW_EXTERNAL_CONFIG_PATHS", "1")
    monkeypatch.setattr(public_handlers_module, "materialize_onp", fake_materialize_onp)
    handler = PublicOELMCPHandlers(
        read_roots=(tmp_path,),
        write_roots=(tmp_path,),
        approval_policy=ExecutionApprovalPolicy(
            write_approval_ids=frozenset({"test-write"}),
            trust_approval_ids=frozenset({"test-trust"}),
        ),
    )

    result = handler.materialize_onp_handoff(
        product_path=str(product_path),
        scenario_name="continued",
        scenario_path=str(scenario_path),
        run_output_dir=str(run_output_dir),
        duration_s=30.0,
        dt_s=10.0,
        trust_plugins=False,
        trust_approval=TRUST_APPROVAL,
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )

    assert result["status"] == "partial"
    assert result["result"]["status"] == "blocked"
    assert observed["logical_scenario_path"] == scenario_path
    assert observed["logical_manifest_path"] == scenario_path.with_name("continued.handoff_manifest.json")
    assert observed["logical_output_dir"] == run_output_dir
    assert Path(result["result"]["scenario_path"]) == scenario_path
    assert Path(result["result"]["manifest_path"]) == scenario_path.with_name(
        "continued.handoff_manifest.json"
    )
    assert scenario_path.read_text(encoding="utf-8")


def test_materialization_validation_does_not_honor_external_path_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    scenario = yaml.safe_load((root / "configs" / "automation_smoke.yaml").read_text(encoding="utf-8"))
    outside = tmp_path.parent / "materialization-external-eop.txt"
    outside.write_text("external input", encoding="utf-8")
    scenario.setdefault("simulator", {}).setdefault("frames", {})["eop_path"] = str(outside)
    destination = tmp_path / "generated.yaml"
    manifest_target = tmp_path / "generated.handoff_manifest.json"
    policy = ConfigPathPolicy(
        config_path=destination,
        workspace_root=tmp_path,
        read_roots=(tmp_path,),
        write_roots=(tmp_path,),
        allow_external_config_paths=False,
        allow_external_ai_prompt_files=False,
        restrict_ai_prompt_to_read_roots=True,
    )
    base_manifest = _manifest_base(
        created_utc="2026-10-22T00:00:00Z",
        source_id="source-product",
        source_hash="0" * 64,
        product={},
        scenario_name="generated",
        destination=destination,
        output_dir=tmp_path / "run",
        duration_s=30.0,
        dt_s=10.0,
        overwrite=False,
    )

    monkeypatch.setenv("OEL_ALLOW_EXTERNAL_CONFIG_PATHS", "1")
    result = materialize_scenario_document(
        scenario=scenario,
        destination=destination,
        manifest_target=manifest_target,
        logical_destination=destination,
        logical_manifest_target=manifest_target,
        logical_output_dir=tmp_path / "run",
        validation_path_policy=policy,
        base_manifest=base_manifest,
        product_report={},
        output_kind="onp_scenario",
        trust_plugins=False,
        overwrite=False,
    )

    assert result["status"] == "failed"
    assert result["safe_validation"]["ok"] is False
    assert "outside allowed config roots" in " ".join(result["safe_validation"]["errors"])
    manifest = json.loads(manifest_target.read_text(encoding="utf-8"))
    assert manifest["output"]["status"] == "validation_failed"
