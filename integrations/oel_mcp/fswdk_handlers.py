"""Transport-independent execution adapter for FSWDK MCP contracts."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event
from typing import Any, Callable

from integrations.oel_mcp.base_handlers import require_file_size
from integrations.oel_mcp.contracts import MAX_MANIFEST_BYTES, ToolContract, legacy_local_tool_id
from integrations.oel_mcp.execution import cancellation_callback, secure_staged_output
from sim.fswdk.candidate import inspect_candidate
from sim.fswdk.services import (
    describe_capabilities,
    init_candidate,
    inspect_evidence,
    plan_workflow,
    prepare_package,
    run_bench,
    run_contract_tests,
    run_qualification,
    run_smoke,
    validate_candidate_service,
)

from .fswdk_registry import FSWDK_TOOL_IDS, FSWDK_TRUST_TOOL_IDS
from .secure_writes import SecureWriteRoot, bind_output_writer


def call_fswdk_tool(
    handler: Any,
    contract: ToolContract,
    arguments: dict[str, Any],
    *,
    cancel_event: Event | None,
    progress: Callable[[float, float | None, str], None] | None,
) -> dict[str, Any] | None:
    """Dispatch one FSWDK tool, returning ``None`` for non-FSWDK tools."""

    legacy_id = legacy_local_tool_id(contract.tool_id)
    if legacy_id not in FSWDK_TOOL_IDS:
        return None
    if legacy_id == FSWDK_TOOL_IDS[2]:
        return inspect_fswdk_evidence(handler, contract, arguments)

    def operation() -> dict[str, Any]:
        check_cancelled = cancellation_callback(cancel_event)
        check_cancelled()
        if legacy_id in FSWDK_TRUST_TOOL_IDS and os.environ.get(
            "OEL_FSWDK_HOST_SANDBOX_ACTIVE", ""
        ).strip().lower() not in {"1", "true", "yes"}:
            raise PermissionError(
                "FSWDK trusted candidate execution requires an operator-provided host sandbox; "
                "start the MCP server inside that boundary and set OEL_FSWDK_HOST_SANDBOX_ACTIVE=1."
            )
        if progress is not None:
            progress(0.0, 1.0, f"Starting {contract.tool_id}")
        result = _project_fswdk_result(
            _run(handler, legacy_id, arguments, cancellation_check=check_cancelled)
        )
        check_cancelled()
        if progress is not None:
            progress(1.0, 1.0, f"Completed {contract.tool_id}")
        return {
            **result,
            "data_label": str(arguments["handling"]["data_label"]),
            "hidden_truth_visible": False,
            "external_communication": False,
            "host_sandbox_asserted": legacy_id in FSWDK_TRUST_TOOL_IDS,
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


def _run(
    handler: Any,
    tool_id: str,
    arguments: dict[str, Any],
    *,
    cancellation_check: Callable[[], None],
) -> dict[str, Any]:
    if tool_id == FSWDK_TOOL_IDS[0]:
        return describe_capabilities()
    if tool_id == FSWDK_TOOL_IDS[4]:
        output = handler.path_policy.resolve_write(arguments["output_dir"])
        project_root = _authorized_workspace(handler, output, write=True)
        with SecureWriteRoot.open(
            project_root,
            approved_roots=handler.path_policy.write_roots,
            create=False,
            require_empty=False,
            logical_root=project_root,
        ) as project_writer:
            relative_output = output.relative_to(project_root)
            target_writer = (
                project_writer.child(relative_output, create=True)
                if relative_output.parts
                else project_writer
            )
            target_writer.ensure_empty()
            try:
                with bind_output_writer(target_writer):
                    result = init_candidate(
                        str(arguments["name"]),
                        template=str(arguments["template"]),
                        project_root=project_root,
                        output_dir=output,
                        class_name=arguments.get("class_name"),
                        force=False,
                        writer=target_writer,
                    )
                    return result
            finally:
                if target_writer is not project_writer:
                    target_writer.close()

    manifest = handler.path_policy.resolve_read(arguments["manifest_path"], kind="file")
    require_file_size(manifest, maximum=MAX_MANIFEST_BYTES)
    workspace = _authorized_workspace(handler, manifest)
    if tool_id == FSWDK_TOOL_IDS[1]:
        return inspect_candidate(manifest, workspace_root=workspace)
    if tool_id == FSWDK_TOOL_IDS[3]:
        output = arguments.get("output_dir")
        if output:
            output = handler.path_policy.resolve_write(output)
        return plan_workflow(
            manifest,
            str(arguments["operation"]),
            workspace_root=workspace,
            output_dir=output,
        )
    if tool_id == FSWDK_TOOL_IDS[5]:
        output = handler.path_policy.resolve_write(arguments["output_dir"])
        _require_same_workspace(output, workspace)
        with secure_staged_output(handler.path_policy, output, require_empty=True) as transaction:
            with bind_output_writer(transaction.staging):
                result = validate_candidate_service(
                    manifest,
                    workspace_root=workspace,
                    trusted_import=True,
                    receipt_dir=output,
                    writer=transaction.staging,
                )
                cancellation_check()
                return transaction.rewrite(result)
    if tool_id in FSWDK_TOOL_IDS[6:10]:
        output = handler.path_policy.resolve_write(arguments["output_dir"])
        _require_same_workspace(output, workspace)
        with secure_staged_output(handler.path_policy, output, require_empty=True) as transaction:
            with bind_output_writer(transaction.staging):
                common = {
                    "workspace_root": workspace,
                    "output_dir": output,
                    "validation_id": str(arguments["validation_id"]),
                    "writer": transaction.staging,
                }
                if tool_id == FSWDK_TOOL_IDS[6]:
                    return transaction.rewrite(
                        run_contract_tests(manifest, cancellation_check=cancellation_check, **common)
                    )
                if tool_id == FSWDK_TOOL_IDS[7]:
                    return transaction.rewrite(run_smoke(manifest, cancellation_check=cancellation_check, **common))
                if tool_id == FSWDK_TOOL_IDS[8]:
                    return transaction.rewrite(
                        run_bench(manifest, show_progress=False, cancellation_check=cancellation_check, **common)
                    )
                return transaction.rewrite(
                    run_qualification(
                        manifest,
                        include_bench=bool(arguments["include_bench"]),
                        cancellation_check=cancellation_check,
                        **common,
                    )
                )
    if tool_id == FSWDK_TOOL_IDS[10]:
        evidence = handler.path_policy.resolve_read(arguments["evidence_dir"], kind="directory")
        output = handler.path_policy.resolve_write(arguments["output_dir"])
        _require_same_workspace(evidence, workspace)
        _require_same_workspace(output, workspace)
        with secure_staged_output(handler.path_policy, output, require_empty=True) as transaction:
            with bind_output_writer(transaction.staging):
                result = prepare_package(
                    manifest,
                    evidence,
                    workspace_root=workspace,
                    output_dir=output,
                    writer=transaction.staging,
                )
                return transaction.rewrite(result)
    raise PermissionError("Unsupported FSWDK tool contract.")


def inspect_fswdk_evidence(handler: Any, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle the evidence inspection tool, which has no candidate manifest input."""

    def operation() -> dict[str, Any]:
        source = handler.path_policy.resolve_read(arguments["evidence_path"])
        result = inspect_evidence(source)
        return {
            **result,
            "data_label": str(arguments["handling"]["data_label"]),
            "hidden_truth_visible": False,
            "external_communication": False,
        }

    return handler._envelope(
        contract=contract,
        arguments=arguments,
        operation=operation,
        outcome_status=lambda result: "completed" if bool(result.get("complete")) else "partial",
        evidence=lambda result: {
            "complete": bool(result.get("complete")),
            "empty": False,
            "truncated": False,
        },
    )


def _authorized_workspace(handler: Any, path: Path, *, write: bool = False) -> Path:
    roots = handler.path_policy.write_roots if write else handler.path_policy.read_roots
    candidates = [root for root in roots if _is_relative_to(path, root)]
    if not candidates:
        raise PermissionError("Path is not inside an authorized FSWDK workspace.")
    return max(candidates, key=lambda item: len(item.parts))


def _project_fswdk_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep agent responses bounded while leaving full evidence on disk."""

    projected = dict(result)
    summary = projected.get("summary")
    if projected.get("schema") == "oel.fswdk.run_manifest.v1" and isinstance(summary, dict):
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
    if projected.get("schema") == "oel.fswdk.qualification_packet.v1":
        projected["gates"] = [
            {
                key: gate.get(key)
                for key in (
                    "id",
                    "kind",
                    "required",
                    "requirement_ids",
                    "status",
                    "evidence_citations",
                )
            }
            for gate in list(projected.get("gates", []) or [])
        ]
    files = list(projected.get("files", []) or [])
    if files:
        projected["files"] = files[:500]
        projected["files_truncated"] = len(files) > 500
        projected["file_count"] = len(files)
    return projected


def _require_same_workspace(path: Path, workspace: Path) -> None:
    if not _is_relative_to(path, workspace):
        raise PermissionError("FSWDK inputs and outputs must remain inside the candidate workspace.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = ["call_fswdk_tool", "inspect_fswdk_evidence"]
