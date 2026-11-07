"""Transport adapter for the local BYO frontier study-planning pilot."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import Any

from integrations.oel_mcp.base_handlers import BaseOELMCPHandlers
from integrations.oel_mcp.contracts import MAX_RESPONSE_BYTES, ToolContract
from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.frontier_projection import frontier_safe_value
from integrations.oel_mcp.study_registry import PROFILE, study_contracts
from sim.hosted_client import validate_hosted_execution_package
from sim.project_version import installed_project_version, source_project_version
from sim.study_planning import (
    build_capability_manifest,
    build_plan_review,
    discovery_capability_catalog,
    preflight_study_plan,
    prepare_capability_request,
)
from sim.study_planning.config_validation import validate_config_path


class StudyPlanningMCPHandlers(BaseOELMCPHandlers):
    def __init__(
        self,
        *,
        read_roots: tuple[str | Path, ...] | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        approval_policy: ExecutionApprovalPolicy | None = None,
    ) -> None:
        super().__init__(
            profile=PROFILE,
            contracts=study_contracts(),
            read_roots=read_roots,
            write_roots=(),
            max_response_bytes=max_response_bytes,
            approval_policy=approval_policy,
        )

    def _call_contract(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        del cancel_event, progress
        if contract.tool_id == "oel.study.capabilities.v1":
            return self._envelope(contract=contract, arguments=arguments, operation=self._capability_manifest)
        if contract.tool_id == "oel.study.preflight.v1":
            return self._envelope(
                contract=contract,
                arguments=arguments,
                operation=lambda: self._preflight(arguments),
            )
        if contract.tool_id == "oel.study.plan_review.v1":
            return self._envelope(
                contract=contract,
                arguments=arguments,
                operation=lambda: build_plan_review(arguments["planning_result"]),
            )
        if contract.tool_id == "oel.hosted.validate_package.v1":
            return self._envelope(
                contract=contract,
                arguments=arguments,
                operation=lambda: self._validate_package(arguments),
            )
        if contract.tool_id == "oel.study.prepare_capability_request.v1":
            return self._envelope(
                contract=contract,
                arguments=arguments,
                operation=lambda: self._feedback(arguments),
            )
        raise PermissionError("Tool is not available in the study-planning profile.")

    def _project_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Project local paths while preserving RFC 6901 constraint pointers.

        The frontier projection treats every string beginning with ``/`` as a
        possible absolute local path.  StudyPlan constraint paths also begin
        with ``/`` by contract, but they are JSON pointers into the normalized
        configuration and must remain valid for the result schema and the
        subsequent plan-review call.  Temporarily remove those pointers from
        the generic projection boundary, then restore the exact bound values.
        """

        if self.profile != PROFILE:
            return result

        projected_input = deepcopy(result)
        pointers: dict[str, str] = {}
        pointer_index = 0
        for container in (projected_input, projected_input.get("plan")):
            if not isinstance(container, dict):
                continue
            constraints = container.get("config_constraints")
            if not isinstance(constraints, list):
                continue
            for constraint in constraints:
                if not isinstance(constraint, dict) or not isinstance(constraint.get("path"), str):
                    continue
                sentinel = f"oel-json-pointer:{pointer_index}"
                pointer_index += 1
                pointers[sentinel] = constraint["path"]
                constraint["path"] = sentinel

        projected = frontier_safe_value(
            projected_input,
            authorized_roots=self._authorized_roots(),
        )
        self._restore_json_pointer_fields(projected, pointers)
        return projected

    @staticmethod
    def _restore_json_pointer_fields(value: Any, pointers: dict[str, str]) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and value["path"] in pointers:
                value["path"] = pointers[value["path"]]
            for item in value.values():
                StudyPlanningMCPHandlers._restore_json_pointer_fields(item, pointers)
        elif isinstance(value, list):
            for item in value:
                StudyPlanningMCPHandlers._restore_json_pointer_fields(item, pointers)

    @staticmethod
    def _capability_manifest() -> dict[str, Any]:
        return build_capability_manifest(discovery_capability_catalog())

    def _preflight(self, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, dict[str, Any]] = {}
        receipts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in arguments.get("configs", []) or []:
            config_ref = str(item["config_ref"])
            if config_ref in seen:
                raise ValueError("Configuration references must be unique.")
            seen.add(config_ref)
            path, content = self.path_policy.read_authorized_bytes(
                item["path"],
                maximum=2_000_000,
            )
            root = self._containing_read_root(path)
            receipt, document = validate_config_path(
                config_ref,
                path,
                workspace_root=root,
                content=content,
            )
            receipts.append(receipt)
            if document is not None:
                normalized[config_ref] = document
        return preflight_study_plan(
            arguments["request"],
            arguments["plan"],
            catalog=discovery_capability_catalog(),
            normalized_configs=normalized,
            config_receipts=receipts,
        )

    def _feedback(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return prepare_capability_request(
            arguments["planning_result"],
            user_summary=str(arguments["user_summary"]),
            engine_version=source_project_version() or installed_project_version() or "unknown",
            would_pay=arguments["would_pay"],
            contact_permission=bool(arguments["contact_permission"]),
            contact=arguments["contact"],
            attachments=arguments["attachments"],
        )

    def _validate_package(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self.path_policy.resolve_read(arguments["package_path"], kind="any")
        root = self._containing_read_root(path)
        return validate_hosted_execution_package(path, workspace_root=root)

    def _containing_read_root(self, path: Path) -> Path:
        for root in self.path_policy.read_roots:
            try:
                path.relative_to(root)
            except ValueError:
                continue
            return root
        raise PermissionError("Path is not authorized for study planning.")


__all__ = ["StudyPlanningMCPHandlers"]
