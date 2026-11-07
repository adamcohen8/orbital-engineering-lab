from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, local
from typing import Any, Callable

from integrations.oel_mcp.agent_guidance import BOOTSTRAP_URI, WORKFLOWS_URI
from integrations.oel_mcp.approval_grants import (
    ApprovalDecision,
    DurableOperationLedger,
    canonical_request_digest,
    derive_operation_id,
)
from integrations.oel_mcp.contracts import MAX_RESPONSE_BYTES, TOOL_CONTRACT_VERSION, ToolContract, effects
from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.frontier_projection import frontier_safe_value
from integrations.oel_mcp.policy import MCPPathPolicy, validate_handling
from integrations.oel_mcp.recovery import annotate_error, recovery_details


class BaseOELMCPHandlers:
    """Shared transport-independent contract, policy, audit, and envelope behavior."""

    def __init__(
        self,
        *,
        profile: str,
        contracts: dict[str, ToolContract],
        read_roots: tuple[str | Path, ...] | None = None,
        write_roots: tuple[str | Path, ...] | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        approval_policy: ExecutionApprovalPolicy | None = None,
        operation_ledger: DurableOperationLedger | None = None,
        operation_ledger_root: str | Path | None = None,
    ) -> None:
        self.profile = profile
        self.contracts = dict(contracts)
        self.path_policy = MCPPathPolicy.configured(read_roots=read_roots, write_roots=write_roots)
        self.max_response_bytes = max(1, min(int(max_response_bytes), MAX_RESPONSE_BYTES))
        self.approval_policy = approval_policy or ExecutionApprovalPolicy.configured()
        self.operation_ledger = operation_ledger
        self.operation_ledger_root = (
            Path(operation_ledger_root).expanduser() if operation_ledger_root is not None else None
        )
        self._approval_context = local()

    def describe_capabilities(self) -> dict[str, Any]:
        contract = self.contracts["oel.describe_capabilities.v1"]
        result = {
            "status": "available",
            "integration": "oel_mcp_v1",
            "transport": "stdio",
            "deployment_profile": self.profile,
            "capabilities": [item.capability() for item in self.contracts.values()],
            "agent_guide_uri": BOOTSTRAP_URI,
            "workflow_routes_uri": WORKFLOWS_URI,
            "readiness": {
                "semantics": "Process configuration only; inputs, handling, trust and exact approvals are checked on every call.",
                "execution_authorized": False,
                "tools": [self._tool_readiness(item) for item in self.contracts.values()],
            },
            "dependency_direction": "mcp_consumes_oel",
            "compatibility": {
                "additive_optional_fields_allowed": True,
                "new_major_required_for": [
                    "required_argument_change",
                    "unit_or_field_meaning_change",
                    "risk_or_effect_change",
                    "hidden_truth_boundary_change",
                    "quality_gate_weakening",
                ],
            },
            "non_claims": [
                "The MCP adapter does not implement physics, private scoring workflows, or dataset validation.",
                "MCP discovery and transport do not replace deployment authorization or release policy.",
                "Supported execution is local, explicitly operator-approved, and never communicates externally.",
            ],
        }
        return self._envelope(contract=contract, arguments={}, operation=lambda: result)

    def _tool_readiness(self, contract: ToolContract) -> dict[str, Any]:
        blockers = []
        scope = "execute" if contract.executes else "write" if contract.writes else None
        if scope and not self._scope_configuration_ready(contract, scope):
            blockers.append(f"{scope}_approval_not_configured")
        needs_write_root = contract.writes or contract.executes or contract.tool_id in {
            "oel.plan_run.v1", "oel.validate_scenario.v1"
        }
        if needs_write_root and not self.path_policy.write_roots:
            blockers.append("write_root_not_configured")
        if (contract.writes or contract.executes) and self._bounded_grant_for_contract(contract) and (
            self._operation_ledger_root_candidate() is None
        ):
            blockers.append("operation_ledger_not_configured")
        required = set(contract.input_schema.get("required", ()))
        trust_required = "trust_approval" in required or bool(contract.limits.get("source_trust_approval_required"))
        if trust_required and not self._scope_configuration_ready(contract, "trust"):
            blockers.append("trust_approval_not_configured")
        entitlement = self._entitlement_readiness(contract)
        if entitlement not in {"not_required", "available"}:
            blockers.append("entitlement_" + entitlement)
        return {
            "tool_id": contract.tool_id, "registered": True,
            "configuration_ready": not blockers, "blockers": blockers,
            "entitlement_status": entitlement,
            "trust_approval_required": trust_required,
            "conditional_trust_available": self._scope_configuration_ready(contract, "trust")
            if "trust_approval" in contract.input_schema.get("properties", {}) else None,
            "execution_authorized": False,
        }

    def _entitlement_readiness(self, contract: ToolContract) -> str:
        return "not_checked" if contract.required_entitlement or contract.required_entitlements else "not_required"

    def _bounded_grant_for_contract(self, contract: ToolContract) -> bool:
        policy = self.approval_policy
        store = policy.grant_store
        if store is None:
            return False
        scope = "execute" if contract.executes else "write" if contract.writes else "trust"
        now = datetime.now(timezone.utc)
        return any(
            scope in grant.scopes
            and contract.tool_id in grant.tool_ids
            and grant.bounded
            and grant.grant_id not in store.revoked_ids
            and not grant.revoked
            and (grant.expires_at is None or now < grant.expires_at.astimezone(timezone.utc))
            for grant in store.grants
        )

    def _scope_configuration_ready(self, contract: ToolContract, scope: str) -> bool:
        policy = self.approval_policy
        standing = (
            bool(policy.execution_approval_ids) if scope == "execute"
            else bool(policy.write_approval_ids) if scope == "write"
            else bool(policy.trust_approval_ids)
        )
        if standing:
            return True
        store = policy.grant_store
        if store is None:
            return False
        now = datetime.now(timezone.utc)
        for grant in store.grants:
            if scope not in grant.scopes or contract.tool_id not in grant.tool_ids:
                continue
            if grant.grant_id in store.revoked_ids or grant.revoked:
                continue
            if grant.expires_at is not None and now >= grant.expires_at.astimezone(timezone.utc):
                continue
            return True
        return False

    def call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        cancel_event: Event | None = None,
        progress: Callable[[float, float | None, str], None] | None = None,
    ) -> dict[str, Any]:
        tool_id = str(tool_name)
        contract = self.contracts.get(tool_id)
        if contract is None:
            raise annotate_error(PermissionError("Tool is not available in this deployment profile."), "tool.unavailable")
        args = dict(arguments or {})
        if tool_id == "oel.describe_capabilities.v1":
            _validate_arguments(contract, args)
            return self.describe_capabilities()
        try:
            validate_handling(self.profile, args.get("handling"))
        except PermissionError as exc:
            annotate_error(exc, "handling.review_required")
            raise
        _validate_arguments(contract, args)
        approval_decision: ApprovalDecision | None = None
        trust_decision: ApprovalDecision | None = None
        if contract.writes or contract.executes:
            approval_decision = self.approval_policy.authorize(
                args.get("approval"),
                executes=contract.executes,
                tool_id=tool_id,
                arguments=args,
                path_policy=self.path_policy,
            )
        if self._requires_plugin_trust(contract, args):
            trust_decision = self.approval_policy.authorize_trust(
                args.get("trust_approval"),
                tool_id=tool_id,
                arguments=args,
                path_policy=self.path_policy,
            )
        approval_stack = self._approval_stack()
        trust_stack = self._trust_stack()
        if approval_decision is not None:
            approval_stack.append(approval_decision)
        if trust_decision is not None:
            trust_stack.append(trust_decision)
        try:
            return self._call_contract(contract, args, cancel_event=cancel_event, progress=progress)
        finally:
            if trust_decision is not None:
                trust_stack.pop()
            if approval_decision is not None:
                approval_stack.pop()

    def _call_contract(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None = None,
        progress: Callable[[float, float | None, str], None] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _requires_plugin_trust(self, contract: ToolContract, arguments: dict[str, Any]) -> bool:
        tool_id = contract.tool_id
        return bool(contract.limits.get("source_trust_approval_required", False)) or (
            (tool_id == "oel.validate_scenario.v1" and bool(arguments.get("trust_plugins", False)))
            or tool_id == "oel.run_scenario.v1"
            or (
                tool_id in {"oel.materialize_onp_handoff.v1", "oel.materialize_scenario_patch.v1"}
                and bool(arguments.get("trust_plugins", False))
            )
        )

    def _approval_stack(self) -> list[ApprovalDecision]:
        stack = getattr(self._approval_context, "approval_stack", None)
        if stack is None:
            stack = []
            self._approval_context.approval_stack = stack
        return stack

    def _trust_stack(self) -> list[ApprovalDecision]:
        stack = getattr(self._approval_context, "trust_stack", None)
        if stack is None:
            stack = []
            self._approval_context.trust_stack = stack
        return stack

    def _current_approval(self) -> ApprovalDecision | None:
        stack = self._approval_stack()
        return stack[-1] if stack else None

    def _current_trust(self) -> ApprovalDecision | None:
        stack = self._trust_stack()
        return stack[-1] if stack else None

    @staticmethod
    def _decision_matches(
        decision: ApprovalDecision | None,
        contract: ToolContract,
        arguments: dict[str, Any],
    ) -> bool:
        if decision is None or decision.tool_id != contract.tool_id:
            return False
        return decision.request_digest == canonical_request_digest(contract.tool_id, arguments)

    @property
    def active_operation_id(self) -> str | None:
        """Return the operation reserved for the current effect, if any."""

        stack = getattr(self._approval_context, "operation_stack", None)
        return stack[-1] if stack else None

    def _operation_stack(self) -> list[str]:
        stack = getattr(self._approval_context, "operation_stack", None)
        if stack is None:
            stack = []
            self._approval_context.operation_stack = stack
        return stack

    def _operation_ledger_root_candidate(self) -> Path | None:
        if self.operation_ledger is not None:
            # An injected host-owned ledger is already the durable authority;
            # readiness must account for it just like an explicit root.
            candidate = Path(self.operation_ledger.root).expanduser()
        else:
            configured_root = self.operation_ledger_root
            if configured_root is None:
                environment_root = os.environ.get("OEL_MCP_OPERATION_LEDGER_ROOT", "").strip()
                if environment_root:
                    configured_root = Path(environment_root).expanduser()
            if configured_root is not None:
                candidate = configured_root
            else:
                # Durable effects require an explicit host-private root.  Never
                # invent a ledger directory inside a model-authorized workspace.
                return None
        if not candidate.is_absolute():
            return None
        if self._ledger_overlaps_authorized_roots(candidate):
            return None
        return candidate

    def _ledger_overlaps_authorized_roots(self, candidate: Path) -> bool:
        candidate = Path(candidate).expanduser()
        roots = (*self.path_policy.read_roots, *self.path_policy.write_roots)
        for root in roots:
            try:
                candidate.relative_to(root)
                return True
            except ValueError:
                pass
            try:
                root.relative_to(candidate)
                return True
            except ValueError:
                pass
        return False

    def _get_operation_ledger(self) -> DurableOperationLedger:
        if self.operation_ledger is not None:
            if self._ledger_overlaps_authorized_roots(self.operation_ledger.root):
                raise PermissionError("The MCP operation ledger must be outside authorized data roots.")
            return self.operation_ledger
        configured_root = self._operation_ledger_root_candidate()
        if configured_root is None:
            raise PermissionError(
                "Durable MCP effects require a host-private ledger outside authorized data roots."
            )
        self.operation_ledger = DurableOperationLedger(
            configured_root,
            artifact_roots=(*self.path_policy.read_roots, *self.path_policy.write_roots),
        )
        return self.operation_ledger

    def _envelope(
        self,
        *,
        contract: ToolContract,
        arguments: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
        evidence: Callable[[dict[str, Any]], dict[str, bool]] | None = None,
        outcome_status: Callable[[dict[str, Any]], str] | None = None,
    ) -> dict[str, Any]:
        argument_digest = _sha256_json(_redacted_arguments(arguments))
        approval_decision = self._current_approval()
        if not self._decision_matches(approval_decision, contract, arguments):
            approval_decision = None
        if contract.writes or contract.executes:
            if approval_decision is None:
                approval_decision = self.approval_policy.authorize(
                    arguments.get("approval"),
                    executes=contract.executes,
                    tool_id=contract.tool_id,
                    arguments=arguments,
                    path_policy=self.path_policy,
                )
        trust_decision = self._current_trust()
        if self._requires_plugin_trust(contract, arguments) and not self._decision_matches(
            trust_decision, contract, arguments
        ):
            self.approval_policy.authorize_trust(
                arguments.get("trust_approval"),
                tool_id=contract.tool_id,
                arguments=arguments,
                path_policy=self.path_policy,
            )

        ledger: DurableOperationLedger | None = None
        operation_id: str | None = None
        authority_id: str | None = None
        request_digest: str | None = None
        operation_started = False
        durable_effect = bool(
            (contract.writes or contract.executes)
            and approval_decision is not None
            and (approval_decision.bounded or approval_decision.operation_id)
        )
        if durable_effect:
            if approval_decision is None:
                raise PermissionError("MCP effect approval could not be established.")
            authority_id = approval_decision.grant_id or approval_decision.approval_id
            request_digest = approval_decision.request_digest or canonical_request_digest(contract.tool_id, arguments)
            operation_id = approval_decision.operation_id or derive_operation_id(
                tool_id=contract.tool_id,
                authority_id=authority_id,
                request_digest=request_digest,
            )
            ledger = self._get_operation_ledger()
            begun = ledger.begin(
                operation_id=operation_id,
                tool_id=contract.tool_id,
                request_digest=request_digest,
                authority_id=authority_id,
            )
            if begun.replay_envelope is not None:
                return dict(begun.replay_envelope)
            self._operation_stack().append(operation_id)
            operation_started = True

        try:
            try:
                raw_result = operation()
                status = outcome_status(raw_result) if outcome_status else "completed"
                evidence_status = evidence(raw_result) if evidence else _evidence(complete=status == "completed")
                result = _json_safe_value(self._project_result(raw_result))
                _validate_value(result, contract.result_schema, path="tool result")
                _enforce_response_size(result, self.max_response_bytes)
                error = None
            except Exception as exc:
                result = None
                status = "failed"
                evidence_status = _evidence(complete=False)
                error = {
                    "type": type(exc).__name__,
                    "message": _safe_error_message(exc, authorized_roots=self._authorized_roots()),
                }
                recovery = recovery_details(exc)
                if recovery is not None:
                    error["recovery"] = recovery
            envelope = {
                "tool_contract_version": TOOL_CONTRACT_VERSION,
                "tool_id": contract.tool_id,
                "risk_class": contract.risk_class,
                "status": status,
                "effects": effects(writes=contract.writes, executes=contract.executes),
                "evidence": evidence_status,
                "error": error,
                "audit": {
                    "schema_version": 1,
                    "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "deployment_profile": self.profile,
                    "tool_id": contract.tool_id,
                    "status": status,
                    "arguments_sha256": argument_digest,
                    "arguments_sha256_semantics": "argument_names_and_handling_labels_only",
                    "argument_values_retained": False,
                    "payload_retained": False,
                },
                "result": result,
            }
            if operation_started and ledger is not None and operation_id is not None:
                envelope["audit"]["operation_id"] = operation_id
                ledger.finish(
                    operation_id=operation_id,
                    tool_id=contract.tool_id,
                    request_digest=request_digest or "",
                    authority_id=authority_id or "",
                    envelope=envelope,
                    status=status,
                )
            return envelope
        finally:
            if operation_started:
                self._operation_stack().pop()

    def _project_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if self.profile != "direct_frontier_restricted":
            return result
        return _frontier_safe_value(result, authorized_roots=self._authorized_roots())

    def _authorized_roots(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(self.path_policy.read_roots + self.path_policy.write_roots))


def require_file_size(path: Path, *, maximum: int) -> None:
    if path.stat().st_size > maximum:
        raise ValueError("Authorized input exceeds the tool file-size budget.")


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}
    return value


def _validate_arguments(contract: ToolContract, arguments: dict[str, Any]) -> None:
    try:
        _validate_value(arguments, contract.input_schema, path="tool arguments")
    except ValueError as exc:
        annotate_error(exc, "tool.arguments_invalid")
        raise


def _validate_value(value: Any, schema: dict[str, Any], *, path: str) -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        if value is None and "null" in expected:
            return
        candidates = [item for item in expected if item != "null"]
        if len(candidates) != 1:
            raise ValueError(f"{path} has an unsupported union schema.")
        schema = {**schema, "type": candidates[0]}
        expected = candidates[0]
    if expected == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object.")
        properties = dict(schema.get("properties", {}) or {})
        required = set(schema.get("required", []) or [])
        missing = sorted(required - set(value))
        unexpected = sorted(set(value) - set(properties)) if schema.get("additionalProperties") is False else []
        if missing:
            raise ValueError(f"Missing required {path}: {', '.join(missing)}")
        if unexpected:
            raise ValueError(f"Unexpected {path}: {', '.join(unexpected)}")
        for key, item in value.items():
            if key in properties:
                _validate_value(item, dict(properties[key]), path=f"{path}.{key}")
    elif expected == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array.")
        if len(value) < int(schema.get("minItems", len(value))) or len(value) > int(schema.get("maxItems", len(value))):
            raise ValueError(f"{path} is outside its supported item count.")
        item_schema = dict(schema.get("items", {}) or {})
        for index, item in enumerate(value):
            _validate_value(item, item_schema, path=f"{path}[{index}]")
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer.")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if minimum is not None and value < int(minimum):
            raise ValueError(f"{path} is outside its supported range.")
        if maximum is not None and value > int(maximum):
            raise ValueError(f"{path} is outside its supported range.")
        if exclusive_minimum is not None and float(value) <= float(exclusive_minimum):
            raise ValueError(f"{path} is outside its supported range.")
        if exclusive_maximum is not None and float(value) >= float(exclusive_maximum):
            raise ValueError(f"{path} is outside its supported range.")
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{path} must be a finite number.")
        numeric_value = float(value)
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if minimum is not None and numeric_value < float(minimum):
            raise ValueError(f"{path} is outside its supported range.")
        if maximum is not None and numeric_value > float(maximum):
            raise ValueError(f"{path} is outside its supported range.")
        if exclusive_minimum is not None and numeric_value <= float(exclusive_minimum):
            raise ValueError(f"{path} is outside its supported range.")
        if exclusive_maximum is not None and numeric_value >= float(exclusive_maximum):
            raise ValueError(f"{path} is outside its supported range.")
    elif expected == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean.")
    elif expected == "string":
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{path} must be a string.")
        text = str(value)
        if len(text) < int(schema.get("minLength", len(text))) or len(text) > int(schema.get("maxLength", len(text))):
            raise ValueError(f"{path} is outside its supported length.")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not one of the supported values.")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} does not match the required value.")


def _redacted_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "argument_names": sorted(arguments),
        "handling": {
            "marking": str(dict(arguments.get("handling", {}) or {}).get("marking", "")),
            "release_scope": str(dict(arguments.get("handling", {}) or {}).get("release_scope", "")),
            "data_label": str(dict(arguments.get("handling", {}) or {}).get("data_label", "")),
            "hidden_truth_access": dict(arguments.get("handling", {}) or {}).get("hidden_truth_access"),
            "provider_exposure": dict(arguments.get("handling", {}) or {}).get("provider_exposure"),
            "operational_authority": dict(arguments.get("handling", {}) or {}).get("operational_authority"),
        },
    }


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _enforce_response_size(result: dict[str, Any], maximum: int) -> None:
    size = len(json.dumps(result, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    if size > maximum:
        raise ValueError("Tool result exceeds the configured response-size budget.")


def _safe_error_message(exc: Exception, *, authorized_roots: tuple[Path, ...]) -> str:
    if not isinstance(exc, (PermissionError, FileNotFoundError, ValueError)):
        return "Tool operation failed without disclosing local details."
    message = str(exc)
    if any(str(root) in message for root in authorized_roots):
        return "Tool operation failed without disclosing local details."
    return message


def _frontier_safe_value(value: Any, *, authorized_roots: tuple[Path, ...]) -> Any:
    return frontier_safe_value(value, authorized_roots=authorized_roots)


def _evidence(*, complete: bool, empty: bool = False, truncated: bool = False) -> dict[str, bool]:
    return {"complete": complete, "empty": empty, "truncated": truncated}


__all__ = ["BaseOELMCPHandlers", "require_file_size"]
