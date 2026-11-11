"""Small shared Pro MCP registry constants used by focused contract modules."""

from __future__ import annotations

from typing import Any

from integrations.oel_mcp.contracts import object_schema

PRO_EXECUTION_PROFILES = ("pro_local",)

PRO_M5_HANDLING_SCHEMA: dict[str, Any] = object_schema(
    {
        "marking": {"type": "string", "minLength": 1, "maxLength": 120},
        "release_scope": {"const": "local_only"},
        "data_label": {
            "type": "string",
            "enum": ["oel_pro_internal", "public_source_local_analysis"],
        },
        "hidden_truth_access": {"const": False},
        "owner": {"type": "string", "maxLength": 200, "default": ""},
    },
    required=("marking", "release_scope", "data_label", "hidden_truth_access"),
)


__all__ = ["PRO_EXECUTION_PROFILES", "PRO_M5_HANDLING_SCHEMA"]
