"""Transport-independent handlers for public FSW authoring MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from integrations.oel_mcp.base_handlers import require_file_size
from integrations.oel_mcp.contracts import MAX_MANIFEST_BYTES, ToolContract
from integrations.oel_mcp.execution import secure_staged_output
from sim.fsw_authoring import inspect_candidate
from sim.fsw_authoring.services import (
    describe_capabilities,
    init_candidate,
    plan_workflow,
    run_contract_tests,
    run_smoke,
    validate_candidate_service,
    verify_receipt,
)

from .fsw_authoring_registry import FSW_AUTHORING_TOOL_IDS
from .secure_writes import SecureWriteRoot, bind_output_writer


def call_fsw_authoring_tool(
    handler: Any,
    contract: ToolContract,
    arguments: dict[str, Any],
) -> dict[str, Any] | None:
    if contract.tool_id not in FSW_AUTHORING_TOOL_IDS:
        return None

    def operation() -> dict[str, Any]:
        result = _run(handler, contract.tool_id, arguments)
        return {
            **_project_result(result),
            "hidden_truth_visible": False,
            "private_orchestration_used": False,
            "external_communication": False,
        }

    return handler._envelope(
        contract=contract,
        arguments=arguments,
        operation=operation,
        outcome_status=lambda result: "completed" if str(result.get("status")) in {"ready", "passed"} else "partial",
        evidence=lambda result: {
            "complete": str(result.get("status")) in {"ready", "passed", "failed"},
            "empty": False,
            "truncated": False,
        },
    )


def _run(handler: Any, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_id == FSW_AUTHORING_TOOL_IDS[0]:
        return describe_capabilities()
    if tool_id == FSW_AUTHORING_TOOL_IDS[3]:
        output = handler.path_policy.resolve_write(arguments["output_dir"])
        workspace = _authorized_workspace(handler, output, write=True)
        with SecureWriteRoot.open(
            workspace,
            approved_roots=handler.path_policy.write_roots,
            create=False,
            require_empty=False,
            logical_root=workspace,
        ) as workspace_writer:
            relative_output = output.relative_to(workspace)
            target_writer = (
                workspace_writer.child(relative_output, create=True)
                if relative_output.parts
                else workspace_writer
            )
            target_writer.ensure_empty()
            try:
                with bind_output_writer(target_writer):
                    result = init_candidate(
                        str(arguments["name"]),
                        template=str(arguments["template"]),
                        workspace_root=workspace,
                        output_dir=output,
                        class_name=arguments.get("class_name"),
                        language=str(arguments.get("language", "python")),
                        writer=target_writer,
                    )
                    return result
            finally:
                if target_writer is not workspace_writer:
                    target_writer.close()
    if tool_id == FSW_AUTHORING_TOOL_IDS[7]:
        receipt = handler.path_policy.resolve_read(arguments["receipt_path"], kind="file")
        workspace = _authorized_workspace(handler, receipt)
        return verify_receipt(receipt, workspace_root=workspace)
    manifest = handler.path_policy.resolve_read(arguments["manifest_path"], kind="file")
    require_file_size(manifest, maximum=MAX_MANIFEST_BYTES)
    workspace = _authorized_workspace(handler, manifest)
    if tool_id == FSW_AUTHORING_TOOL_IDS[1]:
        return inspect_candidate(manifest, workspace_root=workspace)
    if tool_id == FSW_AUTHORING_TOOL_IDS[2]:
        output = arguments.get("output_dir")
        if output:
            output = handler.path_policy.resolve_write(output)
            _require_same_workspace(output, workspace)
        return plan_workflow(
            manifest,
            str(arguments["operation"]),
            workspace_root=workspace,
            output_dir=output,
        )
    output = handler.path_policy.resolve_write(arguments["output_dir"])
    _require_same_workspace(output, workspace)
    with secure_staged_output(handler.path_policy, output, require_empty=True) as transaction:
        with bind_output_writer(transaction.staging):
            if tool_id == FSW_AUTHORING_TOOL_IDS[8]:
                from sim.fsw_authoring import build_candidate

                result = build_candidate(
                    manifest,
                    workspace_root=workspace,
                    output_dir=output,
                    trusted_build=bool(arguments["trusted_build"]),
                    writer=transaction.staging,
                )
                return transaction.rewrite(result)
            if tool_id == FSW_AUTHORING_TOOL_IDS[4]:
                result = validate_candidate_service(
                    manifest,
                    workspace_root=workspace,
                    trusted_import=True,
                    receipt_dir=output,
                    writer=transaction.staging,
                )
                return transaction.rewrite(result)
            common = {
                "workspace_root": workspace,
                "output_dir": output,
                "validation_id": str(arguments["validation_id"]),
                "writer": transaction.staging,
            }
            if tool_id == FSW_AUTHORING_TOOL_IDS[5]:
                return transaction.rewrite(run_contract_tests(manifest, **common))
            if tool_id == FSW_AUTHORING_TOOL_IDS[6]:
                return transaction.rewrite(run_smoke(manifest, **common))
    raise PermissionError("Unsupported public FSW authoring tool contract.")


def _authorized_workspace(handler: Any, path: Path, *, write: bool = False) -> Path:
    roots = handler.path_policy.write_roots if write else handler.path_policy.read_roots
    candidates = [root for root in roots if _is_relative_to(path, root)]
    if not candidates:
        raise PermissionError("Path is not inside an authorized public FSW workspace.")
    return max(candidates, key=lambda item: len(item.parts))


def _require_same_workspace(path: Path, workspace: Path) -> None:
    if not _is_relative_to(path, workspace):
        raise PermissionError("Public FSW inputs and outputs must remain inside one authorized workspace.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _project_result(result: dict[str, Any]) -> dict[str, Any]:
    projected = dict(result)
    summary = projected.get("summary")
    if projected.get("schema") == "oel.fsw_authoring.run_manifest.v1" and isinstance(summary, dict):
        projected["summary"] = {
            key: summary.get(key)
            for key in (
                "scenario_name",
                "duration_s",
                "dt_s",
                "samples",
                "objects",
                "terminated_early",
                "termination_reason",
                "review_sqlite_path",
            )
            if key in summary
        }
    execution = projected.get("execution")
    if isinstance(execution, dict):
        bounded = dict(execution)
        for key in ("stdout", "stderr"):
            if isinstance(bounded.get(key), str) and len(bounded[key]) > 10_000:
                bounded[key] = bounded[key][-10_000:]
                bounded[f"{key}_truncated"] = True
        projected["execution"] = bounded
    return projected


__all__ = ["call_fsw_authoring_tool"]
