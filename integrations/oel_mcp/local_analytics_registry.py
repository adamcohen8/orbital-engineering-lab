"""Contracts for local OEL analytical workflows."""

from typing import Any

from integrations.oel_mcp.contracts import MAX_MANIFEST_BYTES, MAX_REVIEW_STORE_BYTES, ToolContract, object_schema
from integrations.oel_mcp.pro_registry_types import PRO_EXECUTION_PROFILES, PRO_M5_HANDLING_SCHEMA
from integrations.oel_mcp.public_registry import APPROVAL_SCHEMA, DYNAMICS_OD_RESULT_SCHEMA
from sim.pro_features import FEATURE_SCALE_ANALYTICS, FEATURE_TRAJECTORY_OPTIMIZATION


def _m5_properties(properties: dict[str, Any]) -> dict[str, Any]:
    out = dict(properties)
    out["handling"] = PRO_M5_HANDLING_SCHEMA
    return out

def _pro_limits(**extra: Any) -> dict[str, Any]:
    return {
        "local_only": True,
        "hidden_truth_access": False,
        "entitlement_required": True,
        **extra,
    }

INSPECT_SCALE_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"const": "completed"},
        "schema_version": {"type": "string"},
        "store_snapshot_sha256": {"type": "string"},
        "catalog_object_count": {"type": "integer", "minimum": 0},
        "current_object_state_count": {"type": "integer", "minimum": 0},
        "conjunction_candidate_count": {"type": "integer", "minimum": 0},
        "open_invalidation_count": {"type": "integer", "minimum": 0},
        "invalidation_by_reason": {"type": "object"},
        "table_counts": {"type": "object"},
        "hidden_truth_visible": {"const": False},
        "data_label": {"type": "string"},
        "entitlement": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "schema_version",
        "store_snapshot_sha256",
        "catalog_object_count",
        "current_object_state_count",
        "conjunction_candidate_count",
        "open_invalidation_count",
        "invalidation_by_reason",
        "table_counts",
        "hidden_truth_visible",
        "data_label",
        "entitlement",
        "non_claims",
    ),
)

PLAN_SCALE_RECOMPUTE_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"enum": ["completed", "partial", "cancelled"]},
        "workflow": {"type": "object"},
        "selection": {"type": "object"},
        "invalidation": {"type": "object"},
        "work_counts": {"type": "object"},
        "planned_jobs": {"type": "array", "items": {"type": "object"}},
        "artifacts": {"type": "array", "items": {"type": "string"}},
        "manifest_path": {"type": "string"},
        "hidden_truth_visible": {"const": False},
        "data_label": {"type": "string"},
        "entitlement": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "workflow",
        "selection",
        "invalidation",
        "work_counts",
        "planned_jobs",
        "artifacts",
        "manifest_path",
        "hidden_truth_visible",
        "data_label",
        "entitlement",
        "non_claims",
    ),
)

INSPECT_SCALE_PLAN_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"enum": ["completed", "partial", "failed", "cancelled"]},
        "workflow": {"type": "object"},
        "selection": {"type": "object"},
        "invalidation": {"type": "object"},
        "work_counts": {"type": "object"},
        "planned_jobs": {"type": "array", "items": {"type": "object"}},
        "manifest": {"type": "object"},
        "hidden_truth_visible": {"const": False},
        "data_label": {"type": "string"},
        "entitlement": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "workflow",
        "selection",
        "invalidation",
        "work_counts",
        "planned_jobs",
        "manifest",
        "hidden_truth_visible",
        "data_label",
        "entitlement",
        "non_claims",
    ),
)

VALIDATE_SCALE_REFRESH_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"enum": ["validated", "blocked"]},
        "validation_id": {"type": "string"},
        "execution_ready": {"type": "boolean"},
        "execution_authorized": {"const": False},
        "selection": {"type": "object"},
        "work_counts": {"type": "object"},
        "resource_plan": {"type": "object"},
        "artifacts": {"type": "array", "items": {"type": "string"}},
        "manifest_path": {"type": "string"},
        "hidden_truth_visible": {"const": False},
        "data_label": {"type": "string"},
        "entitlement": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status", "validation_id", "execution_ready", "execution_authorized", "selection",
        "work_counts", "resource_plan", "artifacts", "manifest_path", "hidden_truth_visible",
        "data_label", "entitlement", "non_claims",
    ),
)

RUN_SCALE_REFRESH_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"enum": ["completed", "partial", "failed", "cancelled"]},
        "validation_id": {"type": "string"},
        "workflow": {"type": "object"},
        "steps": {"type": "array", "items": {"type": "object"}},
        "final_health": {"type": "object"},
        "comparison": {"type": "object"},
        "artifacts": {"type": "array", "items": {"type": "string"}},
        "manifest_path": {"type": "string"},
        "hidden_truth_visible": {"const": False},
        "data_label": {"type": "string"},
        "entitlement": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status", "validation_id", "workflow", "steps", "final_health", "comparison",
        "artifacts", "manifest_path", "hidden_truth_visible", "data_label", "entitlement",
        "non_claims",
    ),
)

INSPECT_SCALE_REFRESH_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        **RUN_SCALE_REFRESH_RESULT_SCHEMA["properties"],
        "manifest": {"type": "object"},
    },
    required=tuple(RUN_SCALE_REFRESH_RESULT_SCHEMA["required"]) + ("manifest",),
)

VALIDATE_TRAJECTORY_OPTIMIZATION_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"const": "validated"},
        "problem": {"type": "object"},
        "problem_sha256": {"type": "string"},
        "resource_plan": {"type": "object"},
        "execution_ready": {"type": "boolean"},
        "execution_authorized": {"const": False},
        "hidden_truth_visible": {"const": False},
        "data_label": {"type": "string"},
        "entitlement": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "problem",
        "problem_sha256",
        "resource_plan",
        "execution_ready",
        "execution_authorized",
        "hidden_truth_visible",
        "data_label",
        "entitlement",
        "non_claims",
    ),
)

RUN_TRAJECTORY_OPTIMIZATION_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"enum": ["completed", "partial", "cancelled"]},
        "solution_status": {"type": "string"},
        "converged": {"type": "boolean"},
        "problem_sha256": {"type": "string"},
        "output_dir": {"type": "string"},
        "summary": {"type": "object"},
        "manifest": {"type": "object"},
        "artifacts": {"type": "array", "items": {"type": "string"}},
        "authoritative_replay_required": {"const": True},
        "hidden_truth_visible": {"const": False},
        "data_label": {"type": "string"},
        "entitlement": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "solution_status",
        "converged",
        "problem_sha256",
        "output_dir",
        "summary",
        "manifest",
        "artifacts",
        "authoritative_replay_required",
        "hidden_truth_visible",
        "data_label",
        "entitlement",
        "non_claims",
    ),
)

REPLAY_TRAJECTORY_OPTIMIZATION_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"const": "verified"},
        "problem_sha256": {"type": "string"},
        "solution_status": {"type": "string"},
        "file_count": {"type": "integer", "minimum": 0},
        "output_dir": {"type": "string"},
        "hidden_truth_visible": {"const": False},
        "data_label": {"type": "string"},
        "entitlement": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "problem_sha256",
        "solution_status",
        "file_count",
        "output_dir",
        "hidden_truth_visible",
        "data_label",
        "entitlement",
        "non_claims",
    ),
)


LOCAL_ANALYTICS_CONTRACTS = (
    ToolContract(
        tool_id="oel.pro.inspect_scale_catalog.v1",
        title="Inspect entitled local Scale catalog",
        description="Inspect bounded aggregate health from a labeled local OEL Scale catalog store.",
        risk_class="R3_sensitive",
        oel_api="sim.scale.summarize_store",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=("pro_scale_catalog_aggregate", "local_only_labeled_evidence"),
        writes=False,
        input_schema=object_schema(
            _m5_properties({"store_path": {"type": "string", "minLength": 1}}),
            required=("store_path", "handling"),
        ),
        result_schema=INSPECT_SCALE_RESULT_SCHEMA,
        required_entitlement=FEATURE_SCALE_ANALYTICS,
        limits=_pro_limits(max_store_bytes=MAX_REVIEW_STORE_BYTES),
    ),
    ToolContract(
        tool_id="oel.pro.plan_scale_recompute.v1",
        title="Plan entitled local Scale targeted recompute",
        description="Write a bounded targeted-recompute plan without running propagation, screening, or refinement.",
        risk_class="R3_sensitive",
        oel_api="sim.scale.run_recompute_plan",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=("pro_scale_config", "pro_scale_plan", "local_only_labeled_evidence"),
        writes=True,
        input_schema=object_schema(
            _m5_properties(
                {
                    "config_path": {"type": "string", "minLength": 1},
                    "output_dir": {"type": "string", "minLength": 1},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("config_path", "output_dir", "handling", "approval"),
        ),
        result_schema=PLAN_SCALE_RECOMPUTE_RESULT_SCHEMA,
        required_entitlement=FEATURE_SCALE_ANALYTICS,
        limits=_pro_limits(
            max_input_file_bytes=MAX_MANIFEST_BYTES,
            max_store_bytes=MAX_REVIEW_STORE_BYTES,
            max_selected_objects=1000,
            new_or_empty_output_required=True,
            operator_approval_required=True,
        ),
    ),
    ToolContract(
        tool_id="oel.pro.inspect_scale_recompute_plan.v1",
        title="Inspect an entitled local Scale recompute plan",
        description="Inspect aggregate counts, planned job classes, and manifest identity from one bounded Scale recompute plan without returning object identities.",
        risk_class="R3_sensitive",
        oel_api="sim.scale recompute-plan evidence contract",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=("pro_scale_plan", "local_only_labeled_evidence"),
        writes=False,
        input_schema=object_schema(
            _m5_properties({"output_dir": {"type": "string", "minLength": 1}}),
            required=("output_dir", "handling"),
        ),
        result_schema=INSPECT_SCALE_PLAN_RESULT_SCHEMA,
        required_entitlement=FEATURE_SCALE_ANALYTICS,
        limits=_pro_limits(max_input_file_bytes=MAX_MANIFEST_BYTES),
    ),
    ToolContract(
        tool_id="oel.run_dynamics_od.v1",
        title="Run bounded Pro dynamics orbit determination",
        description="Fit a state from an OEL Pro observation packet and emit reviewable fit, holdout, and typed state-product evidence.",
        risk_class="R3_sensitive",
        oel_api="sim.estimation.solve_dynamics_orbit_determination",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=("observation_packet", "od_evidence", "state_estimate_product"),
        writes=True,
        executes=True,
        input_schema=object_schema(
            _m5_properties({
                "observation_packet_path": {"type": "string", "minLength": 1},
                "output_dir": {"type": "string", "minLength": 1},
                "object_id": {"type": "string", "minLength": 1},
                "fit_duration_s": {"type": "number", "minimum": 0.0, "maximum": 604800.0},
                "holdout_duration_s": {"type": "number", "minimum": 0.0, "maximum": 604800.0},
                "dt_s": {"type": "number", "minimum": 0.01, "maximum": 300.0},
                "dynamics_model": {"enum": ["two_body", "special_perturbations"]},
                "j2": {"type": "boolean"},
                "max_nfev": {"type": "integer", "minimum": 1, "maximum": 24},
                "detect_maneuvers": {"type": "boolean"},
                "approval": APPROVAL_SCHEMA,
            }),
            required=(
                "observation_packet_path", "output_dir", "fit_duration_s", "holdout_duration_s",
                "dt_s", "dynamics_model", "j2", "max_nfev", "detect_maneuvers",
                "approval", "handling",
            ),
        ),
        result_schema=DYNAMICS_OD_RESULT_SCHEMA,
        required_entitlement=FEATURE_SCALE_ANALYTICS,
        limits=_pro_limits(
            max_input_file_bytes=MAX_MANIFEST_BYTES,
            max_nfev=24,
            max_observations=2000,
            max_arc_duration_s=604800,
            maneuver_detection_max_observations=500,
            new_output_required=True,
            operator_approval_required=True,
            execution_authorized_outputs=False,
        ),
    ),
    ToolContract(
        tool_id="oel.pro.validate_scale_refresh.v1",
        title="Validate a bounded Scale refresh-to-ONP workflow",
        description="Content-bind and resource-gate a labeled Scale refresh without executing propagation or ONP.",
        risk_class="R3_sensitive",
        oel_api="sim.scale.run_recompute_plan",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=("pro_scale_config", "pro_scale_plan", "local_only_labeled_evidence"),
        writes=True,
        input_schema=object_schema(
            _m5_properties({
                "config_path": {"type": "string", "minLength": 1},
                "output_dir": {"type": "string", "minLength": 1},
                "resource_profile": {"const": "laptop-safe"},
                "approval": APPROVAL_SCHEMA,
            }),
            required=("config_path", "output_dir", "resource_profile", "handling", "approval"),
        ),
        result_schema=VALIDATE_SCALE_REFRESH_RESULT_SCHEMA,
        required_entitlement=FEATURE_SCALE_ANALYTICS,
        limits=_pro_limits(max_selected_objects=1000, workers=[1], new_or_empty_output_required=True),
    ),
    ToolContract(
        tool_id="oel.pro.run_scale_refresh_to_onp.v1",
        title="Run a bounded Scale refresh through ONP",
        description="Execute one exactly validated, serial Scale refresh-to-ONP workflow and preserve comparison evidence.",
        risk_class="R3_sensitive",
        oel_api="sim.scale.run_refresh_to_onp_workflow",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=("pro_scale_config", "pro_scale_evidence", "local_only_labeled_evidence"),
        writes=True,
        executes=True,
        input_schema=object_schema(
            _m5_properties({
                "config_path": {"type": "string", "minLength": 1},
                "output_dir": {"type": "string", "minLength": 1},
                "resource_profile": {"const": "laptop-safe"},
                "workers": {"const": 1},
                "validation_id": {"type": "string", "minLength": 1, "maxLength": 160},
                "approval": APPROVAL_SCHEMA,
            }),
            required=(
                "config_path", "output_dir", "resource_profile", "workers", "validation_id",
                "handling", "approval",
            ),
        ),
        result_schema=RUN_SCALE_REFRESH_RESULT_SCHEMA,
        required_entitlement=FEATURE_SCALE_ANALYTICS,
        limits=_pro_limits(
            max_selected_objects=1000, workers=[1], new_or_empty_output_required=True,
            operator_approval_required=True,
        ),
    ),
    ToolContract(
        tool_id="oel.pro.inspect_scale_refresh.v1",
        title="Inspect completed Scale refresh-to-ONP evidence",
        description="Inspect aggregate health and comparison evidence without returning catalog object identities.",
        risk_class="R3_sensitive",
        oel_api="sim.scale refresh-to-ONP evidence contract",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=("pro_scale_evidence", "local_only_labeled_evidence"),
        writes=False,
        input_schema=object_schema(
            _m5_properties({"output_dir": {"type": "string", "minLength": 1}}),
            required=("output_dir", "handling"),
        ),
        result_schema=INSPECT_SCALE_REFRESH_RESULT_SCHEMA,
        required_entitlement=FEATURE_SCALE_ANALYTICS,
        limits=_pro_limits(max_input_file_bytes=MAX_MANIFEST_BYTES),
    ),
    ToolContract(
        tool_id="oel.pro.validate_trajectory_optimization.v1",
        title="Validate a bounded Pro trajectory-optimization problem",
        description="Parse, normalize, and resource-check one entitled local trajectory-optimization problem without solving it.",
        risk_class="R3_sensitive",
        oel_api="sim.pro_trajectory_optimization.TrajectoryOptimizationProblem",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=("pro_trajectory_optimization_problem", "local_only_labeled_evidence"),
        writes=False,
        input_schema=object_schema(
            _m5_properties(
                {
                    "problem_path": {"type": "string", "minLength": 1},
                    "resource_profile": {"const": "laptop-safe"},
                }
            ),
            required=("problem_path", "resource_profile", "handling"),
        ),
        result_schema=VALIDATE_TRAJECTORY_OPTIMIZATION_RESULT_SCHEMA,
        required_entitlement=FEATURE_TRAJECTORY_OPTIMIZATION,
        limits=_pro_limits(
            max_input_file_bytes=5_000_000,
            max_evaluation_budget=20_000,
            max_derived_propagation_steps=20_000_000,
            resource_profiles=["laptop-safe"],
            execution_authorized=False,
        ),
    ),
    ToolContract(
        tool_id="oel.pro.run_trajectory_optimization.v1",
        title="Run bounded Pro trajectory optimization",
        description="Solve one entitled, bounded local trajectory problem and write replayable authoritative evidence.",
        risk_class="R3_sensitive",
        oel_api="sim.pro_trajectory_optimization.solve_trajectory_optimization; write_evidence",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=(
            "pro_trajectory_optimization_problem",
            "pro_trajectory_optimization_evidence",
            "local_only_labeled_evidence",
        ),
        writes=True,
        executes=True,
        input_schema=object_schema(
            _m5_properties(
                {
                    "problem_path": {"type": "string", "minLength": 1},
                    "output_dir": {"type": "string", "minLength": 1},
                    "resource_profile": {"const": "laptop-safe"},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("problem_path", "output_dir", "resource_profile", "approval", "handling"),
        ),
        result_schema=RUN_TRAJECTORY_OPTIMIZATION_RESULT_SCHEMA,
        required_entitlement=FEATURE_TRAJECTORY_OPTIMIZATION,
        limits=_pro_limits(
            max_input_file_bytes=5_000_000,
            max_evaluation_budget=20_000,
            max_derived_propagation_steps=20_000_000,
            resource_profiles=["laptop-safe"],
            new_output_required=True,
            operator_approval_required=True,
            authoritative_onp_repropagation_required=True,
        ),
    ),
    ToolContract(
        tool_id="oel.pro.replay_trajectory_optimization.v1",
        title="Replay Pro trajectory-optimization evidence",
        description="Digest-check, resolve, and authoritatively repropagate one entitled local optimization evidence packet.",
        risk_class="R3_sensitive",
        oel_api="sim.pro_trajectory_optimization.replay_evidence",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=PRO_EXECUTION_PROFILES,
        data_classes=("pro_trajectory_optimization_evidence", "local_only_labeled_evidence"),
        writes=False,
        executes=True,
        input_schema=object_schema(
            _m5_properties(
                {
                    "output_dir": {"type": "string", "minLength": 1},
                    "resource_profile": {"const": "laptop-safe"},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("output_dir", "resource_profile", "approval", "handling"),
        ),
        result_schema=REPLAY_TRAJECTORY_OPTIMIZATION_RESULT_SCHEMA,
        required_entitlement=FEATURE_TRAJECTORY_OPTIMIZATION,
        limits=_pro_limits(
            max_packet_bytes=100_000_000,
            resource_profiles=["laptop-safe"],
            operator_approval_required=True,
            authoritative_repropagation=True,
        ),
    ),
)
LOCAL_ANALYTICS_BY_ID = {contract.tool_id: contract for contract in LOCAL_ANALYTICS_CONTRACTS}
