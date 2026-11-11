"""Agent-native MCP contracts for the private OEL FSWDK surface."""

from __future__ import annotations

from typing import Any

from integrations.oel_mcp.contracts import ToolContract, object_schema
from integrations.oel_mcp.public_registry import APPROVAL_SCHEMA
from sim.pro_features import (
    FEATURE_CONTROLLER_BENCH,
    FEATURE_CUSTOMER_PACKAGES,
    FEATURE_GNC_WORKBENCH,
    FEATURE_VALIDATION_EVIDENCE,
)

from .pro_registry_types import PRO_EXECUTION_PROFILES, PRO_M5_HANDLING_SCHEMA

FSWDK_TOOL_IDS = (
    "oel.pro.fsw.describe_capabilities.v1",
    "oel.pro.fsw.inspect_candidate.v1",
    "oel.pro.fsw.inspect_evidence.v1",
    "oel.pro.fsw.plan_workflow.v1",
    "oel.pro.fsw.scaffold_candidate.v1",
    "oel.pro.fsw.validate_candidate.v1",
    "oel.pro.fsw.run_contract_tests.v1",
    "oel.pro.fsw.run_smoke.v1",
    "oel.pro.fsw.run_bench.v1",
    "oel.pro.fsw.run_qualification.v1",
    "oel.pro.fsw.prepare_package.v1",
)

FSWDK_TRUST_TOOL_IDS = frozenset(
    {
        "oel.pro.fsw.validate_candidate.v1",
        "oel.pro.fsw.run_contract_tests.v1",
        "oel.pro.fsw.run_smoke.v1",
        "oel.pro.fsw.run_bench.v1",
        "oel.pro.fsw.run_qualification.v1",
    }
)

_PATH = {"type": "string", "minLength": 1}
_RESULT: dict[str, Any] = {"type": "object"}


def _properties(values: dict[str, Any]) -> dict[str, Any]:
    return {**values, "handling": PRO_M5_HANDLING_SCHEMA}


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
    entitlement: str = FEATURE_GNC_WORKBENCH,
    entitlements: tuple[str, ...] = (),
) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        title=title,
        description=description,
        risk_class="R3_sensitive",
        oel_api=api,
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=("flight_software_candidate", "local_only_labeled_evidence"),
        writes=writes,
        executes=executes,
        input_schema=object_schema(
            _properties(properties),
            required=(*required, "handling"),
        ),
        result_schema=_RESULT,
        required_entitlement=entitlement,
        required_entitlements=entitlements,
        limits={
            "local_only": True,
            "hidden_truth_access": False,
            "entitlement_required": True,
            "new_output_required": writes,
            "operator_approval_required": writes or executes,
            "source_trust_approval_required": tool_id in FSWDK_TRUST_TOOL_IDS,
            "external_communication": False,
            "max_bench_runs": 32,
            "component_test_timeout_s": 300,
            "max_projected_files": 500,
            "trusted_candidate_code_requires_host_sandbox": True,
        },
    )


FSWDK_TOOL_CONTRACTS: tuple[ToolContract, ...] = (
    _contract(
        tool_id=FSWDK_TOOL_IDS[0],
        title="Describe FSWDK capabilities",
        description="Describe the local candidate lifecycle, templates, contracts, and non-claims.",
        api="sim.fswdk.describe_capabilities",
        properties={},
        required=(),
    ),
    _contract(
        tool_id=FSWDK_TOOL_IDS[1],
        title="Inspect an FSW candidate",
        description="Safely inspect a candidate manifest and source identity without importing candidate code.",
        api="sim.fswdk.inspect_candidate",
        properties={"manifest_path": _PATH},
        required=("manifest_path",),
    ),
    _contract(
        tool_id=FSWDK_TOOL_IDS[2],
        title="Inspect FSW evidence",
        description="Inspect one bounded FSWDK evidence packet without authorizing promotion.",
        api="sim.fswdk.inspect_evidence",
        properties={"evidence_path": _PATH},
        required=("evidence_path",),
        entitlement=FEATURE_VALIDATION_EVIDENCE,
    ),
    _contract(
        tool_id=FSWDK_TOOL_IDS[3],
        title="Plan an FSWDK workflow",
        description="Return a content-bound execution plan without writing or executing the workflow.",
        api="sim.fswdk.plan_workflow",
        properties={
            "manifest_path": _PATH,
            "operation": {"enum": ["validate", "test", "simulate", "bench", "qualify", "package"]},
            "output_dir": _PATH,
        },
        required=("manifest_path", "operation"),
    ),
    _contract(
        tool_id=FSWDK_TOOL_IDS[4],
        title="Scaffold an FSW candidate",
        description="Create a bounded ADCS or RPO candidate workspace from maintained FSWDK templates.",
        api="sim.fswdk.init_candidate",
        properties={
            "name": {"type": "string", "minLength": 1, "maxLength": 80},
            "template": {"enum": ["adcs", "rpo"]},
            "output_dir": _PATH,
            "class_name": {"type": "string", "minLength": 1, "maxLength": 120},
            "approval": APPROVAL_SCHEMA,
        },
        required=("name", "template", "output_dir", "approval"),
        writes=True,
    ),
    _contract(
        tool_id=FSWDK_TOOL_IDS[5],
        title="Validate an FSW candidate",
        description="Import an explicitly trusted candidate, verify its lifecycle contract, and write a validation receipt.",
        api="sim.fswdk.validate_candidate_service",
        properties={
            "manifest_path": _PATH,
            "output_dir": _PATH,
            "trusted_import": {"const": True},
            "trust_approval": APPROVAL_SCHEMA,
            "approval": APPROVAL_SCHEMA,
        },
        required=("manifest_path", "output_dir", "trusted_import", "trust_approval", "approval"),
        writes=True,
    ),
    _contract(
        tool_id=FSWDK_TOOL_IDS[6],
        title="Run FSW contract tests",
        description="Execute the exactly validated candidate component contract suite and preserve test evidence.",
        api="sim.fswdk.run_contract_tests",
        properties={
            "manifest_path": _PATH,
            "output_dir": _PATH,
            "validation_id": _PATH,
            "trust_approval": APPROVAL_SCHEMA,
            "approval": APPROVAL_SCHEMA,
        },
        required=("manifest_path", "output_dir", "validation_id", "trust_approval", "approval"),
        writes=True,
        executes=True,
    ),
    _contract(
        tool_id=FSWDK_TOOL_IDS[7],
        title="Run an FSW deterministic smoke scenario",
        description="Run the exactly validated candidate in OEL's deterministic simulation engine and preserve review evidence.",
        api="sim.fswdk.run_smoke",
        properties={
            "manifest_path": _PATH,
            "output_dir": _PATH,
            "validation_id": _PATH,
            "trust_approval": APPROVAL_SCHEMA,
            "approval": APPROVAL_SCHEMA,
        },
        required=("manifest_path", "output_dir", "validation_id", "trust_approval", "approval"),
        writes=True,
        executes=True,
    ),
    _contract(
        tool_id=FSWDK_TOOL_IDS[8],
        title="Run an FSW Controller Bench comparison",
        description="Run bounded candidate-versus-reference Controller Bench cases and return engineering evidence.",
        api="sim.fswdk.run_bench",
        properties={
            "manifest_path": _PATH,
            "output_dir": _PATH,
            "validation_id": _PATH,
            "trust_approval": APPROVAL_SCHEMA,
            "approval": APPROVAL_SCHEMA,
        },
        required=("manifest_path", "output_dir", "validation_id", "trust_approval", "approval"),
        writes=True,
        executes=True,
        entitlements=(FEATURE_CONTROLLER_BENCH,),
    ),
    _contract(
        tool_id=FSWDK_TOOL_IDS[9],
        title="Run FSW qualification",
        description="Run declared contract, smoke, benchmark, and artifact gates and preserve a qualification packet.",
        api="sim.fswdk.run_qualification",
        properties={
            "manifest_path": _PATH,
            "output_dir": _PATH,
            "validation_id": _PATH,
            "include_bench": {"type": "boolean", "default": True},
            "trust_approval": APPROVAL_SCHEMA,
            "approval": APPROVAL_SCHEMA,
        },
        required=(
            "manifest_path",
            "output_dir",
            "validation_id",
            "include_bench",
            "trust_approval",
            "approval",
        ),
        writes=True,
        executes=True,
        entitlement=FEATURE_VALIDATION_EVIDENCE,
        entitlements=(FEATURE_GNC_WORKBENCH, FEATURE_CONTROLLER_BENCH),
    ),
    _contract(
        tool_id=FSWDK_TOOL_IDS[10],
        title="Prepare an FSW evidence package",
        description="Create an evidence-only local package that excludes candidate source and flight binaries.",
        api="sim.fswdk.prepare_package",
        properties={
            "manifest_path": _PATH,
            "evidence_dir": _PATH,
            "output_dir": _PATH,
            "approval": APPROVAL_SCHEMA,
        },
        required=("manifest_path", "evidence_dir", "output_dir", "approval"),
        writes=True,
        entitlement=FEATURE_CUSTOMER_PACKAGES,
        entitlements=(FEATURE_GNC_WORKBENCH,),
    ),
)


__all__ = ["FSWDK_TOOL_CONTRACTS", "FSWDK_TOOL_IDS", "FSWDK_TRUST_TOOL_IDS"]
