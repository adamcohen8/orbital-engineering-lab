from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from integrations.oel_mcp.recovery import RECOVERY_SCHEMA

TOOL_CONTRACT_VERSION = 1
MAX_ROWS = 1000
MAX_VM_STEPS = 2_000_000
MAX_MANIFEST_BYTES = 5_000_000
MAX_REVIEW_STORE_BYTES = 1_000_000_000
MAX_RESPONSE_BYTES = 1_000_000
MAX_REVIEW_VALUE_BYTES = 250_000
MAX_REVIEW_RESULT_BYTES = 750_000

DEPLOYMENT_PROFILES = frozenset(
    {
        "public_local",
        "pro_local",
        "direct_frontier_restricted",
    }
)

HANDLING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "marking": {"type": "string", "minLength": 1, "maxLength": 120},
        "release_scope": {
            "type": "string",
            "enum": ["public", "local_only", "frontier_eligible"],
        },
        "owner": {"type": "string", "maxLength": 200, "default": ""},
    },
    "required": ["marking", "release_scope"],
    "additionalProperties": False,
}

ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_contract_version": {"const": TOOL_CONTRACT_VERSION},
        "tool_id": {"type": "string"},
        "risk_class": {"enum": ["R0_read", "R1_write", "R2_execute", "R3_sensitive"]},
        "status": {"enum": ["completed", "partial", "failed"]},
        "effects": {
            "type": "object",
            "properties": {
                "reads": {"type": "boolean"},
                "writes": {"type": "boolean"},
                "executes": {"type": "boolean"},
                "external_communication": {"const": False},
            },
            "required": ["reads", "writes", "executes", "external_communication"],
            "additionalProperties": False,
        },
        "evidence": {
            "type": "object",
            "properties": {
                "complete": {"type": "boolean"},
                "empty": {"type": "boolean"},
                "truncated": {"type": "boolean"},
            },
            "required": ["complete", "empty", "truncated"],
            "additionalProperties": False,
        },
        "error": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"}, "message": {"type": "string"},
                        "recovery": RECOVERY_SCHEMA,
                    },
                    "required": ["type", "message"],
                    "additionalProperties": False,
                },
            ]
        },
        "audit": {
            "type": "object",
            "properties": {
                "schema_version": {"const": 1},
                "generated_utc": {"type": "string"},
                "deployment_profile": {"type": "string"},
                "tool_id": {"type": "string"},
                "status": {"type": "string"},
                "arguments_sha256": {"type": "string"},
                "arguments_sha256_semantics": {
                    "const": "argument_names_and_handling_labels_only"
                },
                "argument_values_retained": {"const": False},
                "payload_retained": {"const": False},
                "operation_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 160,
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$",
                },
            },
            "required": [
                "schema_version",
                "generated_utc",
                "deployment_profile",
                "tool_id",
                "status",
                "arguments_sha256",
                "arguments_sha256_semantics",
                "argument_values_retained",
                "payload_retained",
            ],
            "additionalProperties": False,
        },
        "result": {},
    },
    "required": [
        "tool_contract_version",
        "tool_id",
        "risk_class",
        "status",
        "effects",
        "evidence",
        "error",
        "audit",
        "result",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ToolContract:
    tool_id: str
    title: str
    description: str
    risk_class: str
    oel_api: str
    maturity: str
    install_profile: str
    deployment_profiles: tuple[str, ...]
    data_classes: tuple[str, ...]
    writes: bool
    input_schema: dict[str, Any]
    result_schema: dict[str, Any]
    executes: bool = False
    required_entitlement: str = ""
    required_entitlements: tuple[str, ...] = ()
    limits: dict[str, Any] = field(default_factory=dict)
    deprecated: bool = False
    replacement: str = ""
    deprecated_since: str = ""
    removal_after: str = ""
    migration_guide: str = ""

    def capability(self) -> dict[str, Any]:
        capability = {
            "tool_id": self.tool_id,
            "title": self.title,
            "risk_class": self.risk_class,
            "oel_api": self.oel_api,
            "maturity": self.maturity,
            "required_install_profile": self.install_profile,
            "deployment_profiles": list(self.deployment_profiles),
            "data_classes": list(self.data_classes),
            "effects": effects(writes=self.writes, executes=self.executes),
            "deprecated": self.deprecated,
            "replacement": self.replacement,
            "input_schema_sha256": _schema_sha256(self.input_schema),
            "result_schema_sha256": _schema_sha256(self.result_schema),
            "limits": {**_tool_limits(self.tool_id), **deepcopy(self.limits)},
        }
        if self.required_entitlement:
            capability["required_entitlement"] = self.required_entitlement
        entitlements = tuple(
            dict.fromkeys(
                item
                for item in (self.required_entitlement, *self.required_entitlements)
                if item
            )
        )
        if entitlements:
            capability["required_entitlements"] = list(entitlements)
        if self.deprecated:
            capability.update(
                {
                    "deprecated_since": self.deprecated_since,
                    "removal_after": self.removal_after,
                    "migration_guide": self.migration_guide,
                }
            )
        return capability

    def mcp_definition(self) -> dict[str, Any]:
        output_schema = deepcopy(ENVELOPE_SCHEMA)
        output_schema["properties"]["tool_id"] = {"const": self.tool_id}
        output_schema["properties"]["risk_class"] = {"const": self.risk_class}
        output_schema["properties"]["effects"] = object_schema(
            {
                "reads": {"const": True},
                "writes": {"const": self.writes},
                "executes": {"const": self.executes},
                "external_communication": {"const": False},
            },
            required=("reads", "writes", "executes", "external_communication"),
        )
        output_schema["properties"]["result"] = {
            "anyOf": [deepcopy(self.result_schema), {"type": "null"}]
        }
        return {
            "name": self.tool_id,
            "title": self.title,
            "description": self.description,
            "inputSchema": deepcopy(self.input_schema),
            "outputSchema": output_schema,
            "annotations": {
                "readOnlyHint": not self.writes and not self.executes,
                "destructiveHint": self.writes or self.executes,
                "idempotentHint": not self.writes and not self.executes,
                "openWorldHint": False,
            },
        }


def object_schema(
    properties: dict[str, Any],
    *,
    required: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": deepcopy(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def handling_properties(properties: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(properties)
    out["handling"] = deepcopy(HANDLING_SCHEMA)
    return out


def effects(*, writes: bool, executes: bool = False) -> dict[str, bool]:
    return {
        "reads": True,
        "writes": writes,
        "executes": executes,
        "external_communication": False,
    }


def _schema_sha256(schema: dict[str, Any]) -> str:
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tool_limits(tool_id: str) -> dict[str, Any]:
    limits: dict[str, Any] = {"max_response_bytes": MAX_RESPONSE_BYTES}
    if tool_id == "oel.inspect_run.v1":
        limits.update({"max_rows_per_saved_query": MAX_ROWS, "max_review_store_bytes": MAX_REVIEW_STORE_BYTES})
    elif tool_id == "oel.query_review.v1":
        limits.update(
            {
                "max_rows": MAX_ROWS,
                "max_vm_steps": MAX_VM_STEPS,
                "max_review_store_bytes": MAX_REVIEW_STORE_BYTES,
            }
        )
    elif tool_id in {"oel.plan_run.v1", "oel.validate_scenario.v1", "oel.run_scenario.v1"}:
        limits.update(
            {
                "max_scenario_bytes": 2_000_000,
                "resource_profiles": ["laptop-safe", "standard"],
                "new_or_empty_output_required": tool_id == "oel.run_scenario.v1",
                "operator_approval_required": tool_id == "oel.run_scenario.v1",
            }
        )
    elif tool_id == "oel.compare_runs.v1":
        limits.update({"max_metrics": 20, "max_rows_per_query": 200})
    elif tool_id == "oel.plot_evidence.v1":
        limits.update({"one_plot_per_call": True, "operator_approval_required": True})
    elif tool_id == "oel.plan_review_plot.v1":
        limits.update({"max_rows": 5000, "read_only_query": True, "render_authorized": False})
    elif tool_id == "oel.render_review_plot.v2":
        limits.update(
            {
                "max_rows": 5000,
                "one_plot_per_call": True,
                "operator_approval_required": True,
                "content_bound_plan": True,
            }
        )
    elif tool_id == "oel.plan_review_animation.v1":
        limits.update(
            {
                "max_rows": 5000,
                "max_frames": 600,
                "max_duration_s": 30,
                "read_only_query": True,
                "render_authorized": False,
            }
        )
    elif tool_id == "oel.render_review_animation.v1":
        limits.update(
            {
                "max_rows": 5000,
                "max_frames": 600,
                "max_duration_s": 30,
                "max_file_bytes": 100_000_000,
                "one_animation_per_call": True,
                "operator_approval_required": True,
                "content_bound_plan": True,
            }
        )
    elif tool_id == "oel.run_agent_task.v1":
        limits.update(
            {
                "public_supported_recipes_only": True,
                "max_rows_per_query": 200,
                "resource_profiles": ["laptop-safe", "standard"],
                "operator_approval_required": True,
            }
        )
    elif tool_id == "oel.prepare_report_packet.v1":
        limits.update(
            {
                "max_artifacts": 100,
                "max_artifact_bytes": 1_000_000_000,
                "max_total_artifact_bytes": 2_000_000_000,
                "max_rows_per_query": 100,
                "operator_approval_required": True,
                "provider_call_made": False,
            }
        )
    elif tool_id == "oel.audit_report.v1":
        limits.update(
            {
                "max_report_bytes": 2_000_000,
                "max_packet_bytes": 2_000_000,
                "operator_approval_required": True,
                "provider_call_made": False,
                "semantic_claim_review": False,
            }
        )
    return limits


def legacy_local_tool_id(tool_id: str) -> str:
    """Resolve the legacy dispatch key for an approved public tool alias."""
    if tool_id.startswith("oel.pro."):
        return tool_id
    if tool_id.startswith("oel.fswdk."):
        return tool_id.replace("oel.fswdk.", "oel.pro.fsw.", 1)
    return "oel.pro." + tool_id.removeprefix("oel.")
