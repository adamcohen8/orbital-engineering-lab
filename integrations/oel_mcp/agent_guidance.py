"""Small routing catalog derived from the active registry, without authorizing effects."""

from __future__ import annotations

from collections.abc import Iterable

BOOTSTRAP_URI = "oel://agent/bootstrap/v1"
WORKFLOWS_URI = "oel://agent/workflows/v1"


_OPTIONAL_ROUTE_DEFINITIONS = (
    ("study_evidence", "Inspect, replay, and compare completed study bundles", (
        "oel.inspect_study.v1", "oel.replay_study.v1", "oel.compare_studies.v1")),
    ("interchange", "Inspect CCSDS data and convert supported frames and times", (
        "oel.inspect_ccsds.v1", "oel.convert_frame_time.v1")),
    ("animation", "Plan and render a supported review animation", (
        "oel.plan_review_animation.v1", "oel.render_review_animation.v1")),
    ("tasks", "Run a checked-in public OEL task recipe", ("oel.run_agent_task.v1",)),
    ("report_evidence", "Prepare and audit a provider-neutral evidence packet", (
        "oel.prepare_report_packet.v1", "oel.audit_report.v1")),
    ("handoff", "Inspect, export, materialize, and compare typed OEL handoffs", (
        "oel.inspect_handoff.v1", "oel.export_run_product.v1", "oel.emit_scenario_overlay.v1",
        "oel.materialize_onp_handoff.v1", "oel.materialize_scenario_patch.v1", "oel.compare_handoff.v1")),
    ("maneuver_readiness", "Assess bounded maneuver-readiness evidence", (
        "oel.assess_maneuver_readiness.v1",)),
    ("public_fsw", "Author and smoke-test a public flight-software candidate", (
        "oel.fsw.describe.v1", "oel.fsw.inspect_candidate.v1", "oel.fsw.plan_candidate.v1",
        "oel.fsw.scaffold_candidate.v1", "oel.fsw.validate_candidate.v1",
        "oel.fsw.run_candidate_tests.v1", "oel.fsw.run_candidate_smoke.v1",
        "oel.fsw.verify_receipt.v1")),
    ("pro_analytics", "Run bounded entitled Pro analytical workflows", (
        "oel.pro.inspect_ihe_dataset.v1", "oel.pro.validate_hypothesis_study.v1",
        "oel.pro.inspect_ihe_dataset.v2", "oel.pro.validate_hypothesis_study.v2",
        "oel.pro.run_hypothesis_study.v1", "oel.pro.inspect_scale_catalog.v1",
        "oel.pro.plan_scale_recompute.v1", "oel.pro.inspect_hypothesis_study.v1",
        "oel.pro.inspect_scale_recompute_plan.v1", "oel.pro.validate_scale_refresh.v1",
        "oel.pro.run_scale_refresh_to_onp.v1", "oel.pro.inspect_scale_refresh.v1",
        "oel.pro.prepare_detection_ihe_handoff.v1")),
    ("pro_fsw", "Run the private FSWDK lifecycle", (
        "oel.pro.fsw.describe_capabilities.v1", "oel.pro.fsw.inspect_candidate.v1",
        "oel.pro.fsw.inspect_evidence.v1", "oel.pro.fsw.plan_workflow.v1",
        "oel.pro.fsw.scaffold_candidate.v1", "oel.pro.fsw.validate_candidate.v1",
        "oel.pro.fsw.run_contract_tests.v1", "oel.pro.fsw.run_smoke.v1",
        "oel.pro.fsw.run_bench.v1", "oel.pro.fsw.run_qualification.v1",
        "oel.pro.fsw.prepare_package.v1")),
    ("trajectory_optimization", "Validate, run, and replay entitled trajectory optimization", (
        "oel.pro.validate_trajectory_optimization.v1", "oel.pro.run_trajectory_optimization.v1",
        "oel.pro.replay_trajectory_optimization.v1")),
)

_CR3BP_ROUTE = ("cr3bp_render", "Render frame-checked CR3BP research evidence", (
    "oel.plan_cr3bp_render.v1", "oel.render_cr3bp.v1"))

_OD_ROUTE_PREFIXES = ("oel.od.", "oel.pro.od.")
_OD_ROUTE_TOOL_IDS = ("oel.run_dynamics_od.v1",)


def _is_od_tool(tool_id: str) -> bool:
    """Recognize registered OD contracts without advertising guessed IDs."""
    if not tool_id.startswith("oel."):
        return False
    if tool_id.startswith(_OD_ROUTE_PREFIXES):
        return True
    return any(
        "orbit_determination" in token
        or token.startswith("od_")
        or token.endswith("_od")
        or "_od_" in token
        for token in tool_id.split(".")
    )


def _active_od_tools(active: frozenset[str]) -> tuple[str, ...]:
    ordered = [tool_id for tool_id in _OD_ROUTE_TOOL_IDS if tool_id in active]
    ordered.extend(sorted(tool_id for tool_id in active if _is_od_tool(tool_id) and tool_id not in ordered))
    return tuple(ordered)


def workflow_routes(tool_ids: Iterable[str]) -> dict:
    active = frozenset(tool_ids)
    routes = (
        ("scenario", "Validate and execute an existing scenario", (
            "oel.plan_run.v1", "oel.validate_scenario.v1", "oel.run_scenario.v1", "oel.inspect_run.v1")),
        ("study_plan", "Resolve a typed study proposal before execution", (
            "oel.study.capabilities.v1", "oel.study.preflight.v1", "oel.study.plan_review.v1")),
        ("review", "Answer a question from completed evidence", (
            "oel.inspect_run.v1", "oel.query_review.v1")),
        ("comparison", "Compare completed runs", ("oel.compare_runs.v1",)),
        ("plot", "Render a standard recipe or plan and render a custom evidence plot", (
            "oel.plot_evidence.v1", "oel.plan_review_plot.v1", "oel.render_review_plot.v2")),
    )

    routes += tuple(
        route for route in _OPTIONAL_ROUTE_DEFINITIONS
        if any(tool_id in active for tool_id in route[2])
    )
    if any(tool_id in active for tool_id in _CR3BP_ROUTE[2]):
        routes += (_CR3BP_ROUTE,)
    od_tools = _active_od_tools(active)
    if od_tools:
        routes += (("orbit_determination", "Run bounded entitled orbit-determination workflows", od_tools),)
    if "oel.pro.twin.capabilities.v1" in active:
        routes += (("spacecraft_twin", "Build a source-backed spacecraft model from engineering files", (
            "oel.pro.twin.capabilities.v1", "oel.pro.twin.intake.v1", "oel.pro.twin.geometry.v1",
            "oel.pro.twin.propose.v1", "oel.pro.twin.inspect.v1", "oel.pro.twin.accept.v1",
            "oel.pro.twin.verify.v1", "oel.pro.twin.diff.v1", "oel.pro.twin.materialize.v1",
            "oel.pro.twin.migrate_v0.v1")),)
    return {
        "schema_version": 1,
        "bootstrap_uri": BOOTSTRAP_URI,
        "routes": [
            {"route_id": name, "purpose": purpose, "tools": list(steps),
             "all_tools_registered_here": all(step in active for step in steps),
             "missing_tools": [step for step in steps if step not in active]}
            for name, purpose, steps in routes
        ],
        "study_planner": {
            "maturity": "prototype", "separate_server_command": "oel-study-mcp",
            "registered_here": "oel.study.preflight.v1" in active,
            "connection_action": "A host operator configures the separate local planner when needed.",
            "outcomes": ["PLAN_VALID", "CLARIFICATION_REQUIRED", "UNSUPPORTED"],
            "execution_authorized": False,
        },
        "authoring": {
            "main_scenario_input": "existing config_path",
            "route": "Documented scenario YAML or sim.api.ScenarioBuilder through authorized host file tools.",
            "arbitrary_file_write_tool": False,
        },
        "local_workflows": {
            "host_authorization_required": True,
            "discovery_authorizes_effects": False,
            "workflows": [
                {
                    "workflow_id": "modular_data_io",
                    "interfaces": ["oel io", "python -m sim.data_io", "sim.data_io"],
                    "documentation": "docs/data-io.md",
                    "dedicated_mcp_tools": [],
                    "purpose": (
                        "Inspect or convert offline files, compose objects into a new validated scenario, "
                        "or export completed evidence with receipts."
                    ),
                    "boundaries": [
                        "Discover exact profiles and optional dependencies with oel io formats.",
                        "Object import and materialization do not execute scenarios or replay source histories.",
                        "Export uses saved review evidence; serializers do not propagate or fit an orbit.",
                        "Review-to-OGP/TLE fitting is a separate Pro tracking_od operation.",
                    ],
                },
                {
                    "workflow_id": "full_state_innovation_recovery",
                    "interfaces": ["scenario YAML", "sim.estimation"],
                    "documentation": "docs/innovation-recovery.md",
                    "dedicated_mcp_tools": [],
                    "purpose": (
                        "Opt into experimental full-state knowledge-estimator recovery "
                        "through an existing scenario or estimator API."
                    ),
                    "boundaries": [
                        "Disabled by default; requires full-state observations and positive-definite covariance.",
                        "Use ordinary scenario validation/run and review-query routes when registered and authorized.",
                        "This is separate from flight-software navigation and Pro sequential/continuous tracking OD.",
                    ],
                },
            ],
        },
        "rules": [
            "Registered tools may still require configured operator approvals and entitlements.",
            "Inspect describe_capabilities readiness; it never authorizes an exact input or run.",
            "A missing route does not authorize changing servers or using a more permissive data boundary.",
            "CLI/Python run lifecycle v1 is separate from MCP execution; retain exact run identity.",
        ],
    }
