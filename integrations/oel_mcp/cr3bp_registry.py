"""MCP contracts for the experimental CR3BP review renderer.

The contracts in this module describe a narrow public-local plan/render surface.
The deterministic CR3BP equations and evidence checks remain in
``sim.review.cr3bp``; this module only defines the adapter boundary.
"""

from __future__ import annotations

from typing import Any

from integrations.oel_mcp.contracts import (
    MAX_RESPONSE_BYTES,
    MAX_REVIEW_STORE_BYTES,
    ToolContract,
    handling_properties,
    object_schema,
)

CR3BP_TOOL_IDS = (
    "oel.plan_cr3bp_render.v1",
    "oel.render_cr3bp.v1",
)

CR3BP_PLAN_TOOL_ID, CR3BP_RENDER_TOOL_ID = CR3BP_TOOL_IDS

# These bounds mirror the existing deterministic renderer's safe envelope.  A
# caller can request less, but the MCP adapter never expands the renderer's
# input range or lets a remote/frontier profile reach this experimental path.
MAX_CR3BP_ROWS = 100_000
MAX_CR3BP_GRID_RESOLUTION = 301
MAX_CR3BP_FRAMES = 600
MAX_CR3BP_FPS = 60
MAX_CR3BP_REVIEW_STORE_BYTES = MAX_REVIEW_STORE_BYTES

CR3BP_RENDER_SPEC_PROPERTIES: dict[str, Any] = {
    "output_dir": {"type": "string", "minLength": 1},
    "object_id": {"type": "string", "minLength": 1, "maxLength": 160},
    "kind": {
        "type": "string",
        "enum": ["trajectory", "zero_velocity", "jacobi"],
        "default": "trajectory",
    },
    "axes": {"type": "string", "enum": ["rotating", "inertial"], "default": "rotating"},
    "origin": {
        "type": "string",
        "enum": ["barycenter", "p1", "p2"],
        "default": "barycenter",
    },
    "plane": {"type": "string", "enum": ["xy", "xz", "yz"], "default": "xy"},
    "reference_time_s": {"type": "number", "default": 0.0},
    "reference_angle_rad": {"type": "number", "default": 0.0},
    "slice_km": {"type": "number", "default": 0.0},
    "jacobi": {"type": ["number", "null"], "default": None},
    "jacobi_mode": {"type": "string", "enum": ["fixed", "instantaneous"], "default": "fixed"},
    "bounds_km": {
        "type": ["array", "null"],
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "default": None,
    },
    "resolution": {
        "type": "integer",
        "minimum": 16,
        "maximum": MAX_CR3BP_GRID_RESOLUTION,
        "default": MAX_CR3BP_GRID_RESOLUTION,
    },
    "movie_format": {"type": ["string", "null"], "enum": ["gif", "mp4", None], "default": None},
    "frames": {
        "type": "integer",
        "minimum": 2,
        "maximum": MAX_CR3BP_FRAMES,
        "default": 120,
    },
    "fps": {"type": "integer", "minimum": 1, "maximum": MAX_CR3BP_FPS, "default": 20},
    "style": {"type": "string", "enum": ["oel_light", "oel_dark"], "default": "oel_dark"},
}

CR3BP_PLAN_RESULT_SCHEMA = object_schema(
    {
        "status": {"const": "planned"},
        "output_dir": {"type": "string"},
        "review_store": {"const": "review/run.sqlite"},
        "cr3bp_plan_id": {"type": "string"},
        "spec": {"type": "object"},
        "evidence_identity": {"type": "object"},
        "system": {"type": "object"},
        "row_count": {"type": "integer", "minimum": 2, "maximum": MAX_CR3BP_ROWS},
        "time_range_s": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "limits": {"type": "object"},
        "render_authorized": {"const": False},
        "visual_review_required": {"const": True},
    },
    required=(
        "status",
        "output_dir",
        "review_store",
        "cr3bp_plan_id",
        "spec",
        "evidence_identity",
        "system",
        "row_count",
        "time_range_s",
        "warnings",
        "limits",
        "render_authorized",
        "visual_review_required",
    ),
)

CR3BP_RENDER_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial"]},
        "output_dir": {"type": "string"},
        "cr3bp_plan_id": {"type": "string"},
        "artifact": {"type": "object"},
        "manifest_path": {"type": "string"},
    },
    required=("status", "output_dir", "cr3bp_plan_id", "artifact", "manifest_path"),
)


def _contracts() -> tuple[ToolContract, ...]:
    return (
        ToolContract(
            tool_id=CR3BP_PLAN_TOOL_ID,
            title="Plan an experimental CR3BP review render",
            description=(
                "Validate a bounded CR3BP review-store render and bind its specification to the exact "
                "evidence identity without writing an artifact. CR3BP model inertial axes are not J2000."
            ),
            risk_class="R0_read",
            oel_api="sim.review.cr3bp.load_cr3bp_history",
            maturity="experimental",
            install_profile="mcp",
            deployment_profiles=("public_local",),
            data_classes=("run_evidence", "cr3bp_render_specification"),
            writes=False,
            input_schema=object_schema(
                handling_properties(CR3BP_RENDER_SPEC_PROPERTIES),
                required=("output_dir", "object_id", "handling"),
            ),
            result_schema=CR3BP_PLAN_RESULT_SCHEMA,
            limits={
                "read_only_query": True,
                "render_authorized": False,
                "max_review_store_bytes": MAX_CR3BP_REVIEW_STORE_BYTES,
                "max_rows": MAX_CR3BP_ROWS,
                "max_grid_resolution": MAX_CR3BP_GRID_RESOLUTION,
                "max_frames": MAX_CR3BP_FRAMES,
            },
        ),
        ToolContract(
            tool_id=CR3BP_RENDER_TOOL_ID,
            title="Render a planned experimental CR3BP review figure or movie",
            description=(
                "Render one content-bound CR3BP review figure or movie through the existing deterministic "
                "research renderer. Writes are local-only, bounded, operator-approved, and experimental."
            ),
            risk_class="R1_write",
            oel_api="sim.review.cr3bp.render_cr3bp_research",
            maturity="experimental",
            install_profile="mcp",
            deployment_profiles=("public_local",),
            data_classes=("run_evidence", "cr3bp_render_specification", "cr3bp_render_artifact"),
            writes=True,
            input_schema=object_schema(
                handling_properties(
                    {
                        **CR3BP_RENDER_SPEC_PROPERTIES,
                        "cr3bp_plan_id": {"type": "string", "minLength": 1, "maxLength": 160},
                        "approval": {
                            "type": "object",
                            "properties": {
                                "approval_id": {"type": "string", "minLength": 1, "maxLength": 120},
                                "scope": {"type": "string", "enum": ["write"]},
                            },
                            "required": ["approval_id", "scope"],
                            "additionalProperties": False,
                        },
                    }
                ),
                required=("output_dir", "object_id", "cr3bp_plan_id", "approval", "handling"),
            ),
            result_schema=CR3BP_RENDER_RESULT_SCHEMA,
            limits={
                "one_render_per_call": True,
                "operator_approval_required": True,
                "content_bound_plan": True,
                "max_review_store_bytes": MAX_CR3BP_REVIEW_STORE_BYTES,
                "max_rows": MAX_CR3BP_ROWS,
                "max_grid_resolution": MAX_CR3BP_GRID_RESOLUTION,
                "max_frames": MAX_CR3BP_FRAMES,
                "max_response_bytes": MAX_RESPONSE_BYTES,
            },
        ),
    )


CR3BP_TOOL_CONTRACTS = _contracts()


def cr3bp_contracts_for_profile(profile: str) -> tuple[ToolContract, ...]:
    """Return contracts exposed by the requested deployment profile."""

    return tuple(contract for contract in CR3BP_TOOL_CONTRACTS if profile in contract.deployment_profiles)


def cr3bp_contract_map(profile: str) -> dict[str, ToolContract]:
    return {contract.tool_id: contract for contract in cr3bp_contracts_for_profile(profile)}


__all__ = [
    "CR3BP_PLAN_TOOL_ID",
    "CR3BP_RENDER_TOOL_ID",
    "CR3BP_RENDER_SPEC_PROPERTIES",
    "CR3BP_PLAN_RESULT_SCHEMA",
    "CR3BP_RENDER_RESULT_SCHEMA",
    "CR3BP_TOOL_IDS",
    "CR3BP_TOOL_CONTRACTS",
    "MAX_CR3BP_FRAMES",
    "MAX_CR3BP_FPS",
    "MAX_CR3BP_GRID_RESOLUTION",
    "MAX_CR3BP_REVIEW_STORE_BYTES",
    "MAX_CR3BP_ROWS",
    "cr3bp_contract_map",
    "cr3bp_contracts_for_profile",
]
