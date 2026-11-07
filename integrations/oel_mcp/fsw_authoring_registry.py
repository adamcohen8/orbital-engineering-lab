"""MCP contracts for the public OEL FSW Authoring Kit."""

from __future__ import annotations

from typing import Any

from integrations.oel_mcp.contracts import ToolContract, handling_properties, object_schema

FSW_AUTHORING_TOOL_IDS = (
    "oel.fsw.describe.v1",
    "oel.fsw.inspect_candidate.v1",
    "oel.fsw.plan_candidate.v1",
    "oel.fsw.scaffold_candidate.v1",
    "oel.fsw.validate_candidate.v1",
    "oel.fsw.run_candidate_tests.v1",
    "oel.fsw.run_candidate_smoke.v1",
    "oel.fsw.verify_receipt.v1",
    "oel.fsw.build_candidate.v1",
)

FSW_AUTHORING_TRUST_TOOL_IDS = frozenset(
    {
        "oel.fsw.validate_candidate.v1",
        "oel.fsw.run_candidate_tests.v1",
        "oel.fsw.run_candidate_smoke.v1",
        "oel.fsw.build_candidate.v1",
    }
)

_PATH = {"type": "string", "minLength": 1}
_RESULT: dict[str, Any] = {"type": "object"}
_APPROVAL = object_schema(
    {
        "approval_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "scope": {"type": "string", "enum": ["trust", "write", "execute"]},
        "operation_id": {
            "type": "string", "minLength": 1, "maxLength": 160,
            "pattern": "^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$",
        },
    },
    required=("approval_id", "scope"),
)


def _contract(
    *,
    tool_id: str,
    title: str,
    description: str,
    api: str,
    properties: dict[str, Any],
    required: tuple[str, ...],
    writes: bool = False,
    executes: bool = False,
) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        title=title,
        description=description,
        risk_class="R2_execute" if executes else "R1_write" if writes else "R0_read",
        oel_api=api,
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=("public_local",),
        data_classes=("flight_software_candidate", "local_run_evidence"),
        writes=writes,
        executes=executes,
        input_schema=object_schema(handling_properties(properties), required=(*required, "handling")),
        result_schema=_RESULT,
        limits={
            "public_python_stack_only": False,
            "candidate_kinds": ["python_stack", "native_stack"],
            "hidden_truth_access": False,
            "private_orchestration": False,
            "new_output_required": writes,
            "operator_approval_required": writes or executes,
            "source_trust_approval_required": tool_id in FSW_AUTHORING_TRUST_TOOL_IDS,
            "external_communication": False,
            "workers": 1,
            "component_test_timeout_s": 300,
        },
    )


FSW_AUTHORING_TOOL_CONTRACTS: tuple[ToolContract, ...] = (
    _contract(
        tool_id=FSW_AUTHORING_TOOL_IDS[0],
        title="Describe public FSW authoring",
        description="Describe public candidate templates, contracts, bounded operations, and explicit private boundaries.",
        api="sim.fsw_authoring.describe_capabilities",
        properties={},
        required=(),
    ),
    _contract(
        tool_id=FSW_AUTHORING_TOOL_IDS[1],
        title="Inspect a public FSW candidate",
        description="Inspect a candidate manifest and content identity without importing or executing candidate code.",
        api="sim.fsw_authoring.inspect_candidate",
        properties={"manifest_path": _PATH},
        required=("manifest_path",),
    ),
    _contract(
        tool_id=FSW_AUTHORING_TOOL_IDS[2],
        title="Plan a public FSW candidate operation",
        description="Return a content-bound validate, test, or deterministic-smoke plan without causing effects.",
        api="sim.fsw_authoring.plan_workflow",
        properties={
            "manifest_path": _PATH,
            "operation": {"enum": ["validate", "test", "smoke", "build"]},
            "output_dir": _PATH,
        },
        required=("manifest_path", "operation"),
    ),
    _contract(
        tool_id=FSW_AUTHORING_TOOL_IDS[3],
        title="Scaffold a public FSW candidate",
        description="Create one dependency-clean ADCS or RPO complete-stack starter inside an approved workspace.",
        api="sim.fsw_authoring.init_candidate",
        properties={
            "name": {"type": "string", "minLength": 1, "maxLength": 80},
            "template": {"enum": ["adcs", "rpo"]},
            "language": {"enum": ["python", "c", "cpp", "rust"]},
            "output_dir": _PATH,
            "class_name": {"type": "string", "minLength": 1, "maxLength": 120},
            "approval": _APPROVAL,
        },
        required=("name", "template", "output_dir", "approval"),
        writes=True,
    ),
    _contract(
        tool_id=FSW_AUTHORING_TOOL_IDS[4],
        title="Validate a public FSW candidate",
        description="Import an explicitly trusted candidate, verify its lifecycle and smoke contract, and write a receipt.",
        api="sim.fsw_authoring.validate_candidate_service",
        properties={
            "manifest_path": _PATH,
            "output_dir": _PATH,
            "trusted_import": {"const": True},
            "trust_approval": _APPROVAL,
            "approval": _APPROVAL,
        },
        required=("manifest_path", "output_dir", "trusted_import", "trust_approval", "approval"),
        writes=True,
    ),
    _contract(
        tool_id=FSW_AUTHORING_TOOL_IDS[5],
        title="Run public FSW component tests",
        description="Run the exactly validated candidate's declared component suite and preserve a bounded receipt.",
        api="sim.fsw_authoring.run_contract_tests",
        properties={
            "manifest_path": _PATH,
            "output_dir": _PATH,
            "validation_id": _PATH,
            "trust_approval": _APPROVAL,
            "approval": _APPROVAL,
        },
        required=("manifest_path", "output_dir", "validation_id", "trust_approval", "approval"),
        writes=True,
        executes=True,
    ),
    _contract(
        tool_id=FSW_AUTHORING_TOOL_IDS[6],
        title="Run a public FSW deterministic smoke",
        description="Run one exactly validated serial smoke scenario and preserve review-backed run evidence.",
        api="sim.fsw_authoring.run_smoke",
        properties={
            "manifest_path": _PATH,
            "output_dir": _PATH,
            "validation_id": _PATH,
            "trust_approval": _APPROVAL,
            "approval": _APPROVAL,
        },
        required=("manifest_path", "output_dir", "validation_id", "trust_approval", "approval"),
        writes=True,
        executes=True,
    ),
    _contract(
        tool_id=FSW_AUTHORING_TOOL_IDS[7],
        title="Verify a public FSW receipt",
        description="Recompute candidate and artifact identity for one public authoring receipt without executing code.",
        api="sim.fsw_authoring.verify_receipt",
        properties={"receipt_path": _PATH},
        required=("receipt_path",),
    ),
    _contract(
        tool_id=FSW_AUTHORING_TOOL_IDS[8],
        title="Build a trusted public native FSW candidate",
        description="Build reviewed native source offline with fixed local toolchain commands and bind the executable identity.",
        api="sim.fsw_authoring.build_candidate",
        properties={
            "manifest_path": _PATH,
            "output_dir": _PATH,
            "trusted_build": {"const": True},
            "trust_approval": _APPROVAL,
            "approval": _APPROVAL,
        },
        required=("manifest_path", "output_dir", "trusted_build", "trust_approval", "approval"),
        writes=True,
        executes=True,
    ),
)


__all__ = ["FSW_AUTHORING_TOOL_CONTRACTS", "FSW_AUTHORING_TOOL_IDS", "FSW_AUTHORING_TRUST_TOOL_IDS"]
