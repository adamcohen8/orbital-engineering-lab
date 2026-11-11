from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Any, Iterable

from integrations.oel_mcp.agent_guidance import BOOTSTRAP_URI, WORKFLOWS_URI, workflow_routes
from integrations.oel_mcp.contracts import DEPLOYMENT_PROFILES, ToolContract
from sim.installation.resources import agent_bootstrap_text

RESOURCE_SCHEMA_VERSION = 1
MAX_RESOURCE_BYTES = 500_000
PUBLIC_RESOURCE_URIS = (
    "oel://capabilities/tools/v1",
    "oel://review/saved-queries/v1",
    "oel://agent/tasks/v1",
    "oel://docs/operator-guide/v1",
    "oel://handoff/product-kinds/v1",
    "oel://review/plot-recipes/v1",
    "oel://review/animation-recipes/v1",
    "oel://analysis/workflows/v1",
    BOOTSTRAP_URI,
    WORKFLOWS_URI,
)


def _public_saved_query_names() -> frozenset[str]:
    """Return the public registry names while preserving the legacy export."""
    from sim.review.queries import list_saved_review_queries

    return frozenset(query.name for query in list_saved_review_queries())


PUBLIC_SAVED_QUERY_NAMES = _public_saved_query_names()


@dataclass(frozen=True)
class ResourceContract:
    uri: str
    name: str
    title: str
    description: str
    mime_type: str
    source: str
    maturity: str = "supported"


@dataclass(frozen=True)
class PublishedResource:
    contract: ResourceContract
    text: str

    @property
    def size(self) -> int:
        return len(self.text.encode("utf-8"))


PUBLIC_RESOURCE_CONTRACTS = (
    # Preserve existing resource ordering; append new resources below.
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[0],
        name="oel-public-tool-schemas-v1",
        title="OEL public MCP tool schemas",
        description="Versioned schemas and effect annotations for the public OEL MCP tool registry.",
        mime_type="application/json",
        source="integrations.oel_mcp.public_registry",
    ),
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[1],
        name="oel-public-saved-review-queries-v1",
        title="OEL public saved review queries",
        description="Bounded metadata for supported public read-only review queries.",
        mime_type="application/json",
        source="sim.review.queries",
    ),
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[2],
        name="oel-public-agent-tasks-v1",
        title="OEL public agent task definitions",
        description="Public agent task definitions suitable for deterministic OEL workflow routing.",
        mime_type="application/json",
        source="sim.agent_task.recipes",
    ),
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[3],
        name="oel-mcp-operator-guide-v1",
        title="OEL MCP operator guide",
        description="Packaged guidance for the supported local stdio MCP surface.",
        mime_type="text/markdown",
        source="integrations.oel_mcp.resource_data/operator-guide.md",
    ),
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[4],
        name="oel-handoff-product-kinds-v1",
        title="OEL handoff product kinds",
        description="Public-safe routing metadata for typed OEL products and supported MCP next actions.",
        mime_type="application/json",
        source="sim.handoff",
    ),
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[5],
        name="oel-public-review-plot-recipes-v1",
        title="OEL review plot recipes",
        description="Supported plot recipes, required evidence, renderers, and natural-language routing triggers.",
        mime_type="application/json",
        source="sim.review.plot_recipes",
    ),
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[6],
        name="oel-public-review-animation-recipes-v1",
        title="OEL review animation recipes",
        description="Supported animation recipes, evidence requirements, renderers, and quality policy.",
        mime_type="application/json",
        source="sim.review.animation_recipes",
    ),
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[7],
        name="oel-public-analysis-workflows-v1",
        title="OEL public orbital-analysis workflows",
        description="Versioned routing contracts for standalone public orbital-analysis problems, evidence, replay, and MCP support.",
        mime_type="application/json",
        source="docs/agent-capability-routing.md; sim.analysis",

    ),
    ResourceContract(
        uri=BOOTSTRAP_URI, name="oel-agent-bootstrap-v1", title="Start here: operating OEL",
        description="Shared agent operating procedure, planning routes, evidence and recovery rules.",
        mime_type="text/markdown", source="sim.installation/data/agent-bootstrap.md",
    ),
    ResourceContract(
        uri=WORKFLOWS_URI, name="oel-agent-workflows-v1", title="OEL agent workflow routes",
        description="Compact workflow routes and separate prototype planner, matched to the active tool registry.",
        mime_type="application/json", source="integrations.oel_mcp.agent_guidance",
    ),
)


def build_public_resource_catalog(
    *,
    profile: str,
    tool_contracts: Iterable[ToolContract],
) -> tuple[PublishedResource, ...]:
    if profile not in DEPLOYMENT_PROFILES:
        raise ValueError("Unknown OEL MCP deployment profile.")
    contracts = tuple(tool_contracts)
    # The main local server marks its contracts ``mcp`` while the separate
    # public study-planning server marks its contracts ``public``.  Both are
    # safe public MCP surfaces; private/pro registries are passed through
    # neither server and therefore remain excluded by the handler contract
    # set itself.  Filtering to only ``mcp`` made the study server advertise
    # tools through ``tools/list`` while publishing an empty tool-schema
    # resource.
    public_tools = tuple(contract for contract in contracts if contract.install_profile in {"mcp", "public"})
    loaders = {
        PUBLIC_RESOURCE_URIS[0]: lambda: json.dumps(
            _tool_schema_payload(profile, public_tools), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) + "\n",
        PUBLIC_RESOURCE_URIS[1]: lambda: _json_text(_saved_query_payload()),
        PUBLIC_RESOURCE_URIS[2]: lambda: _json_text(_agent_task_payload()),
        PUBLIC_RESOURCE_URIS[3]: _operator_guide_text,
        PUBLIC_RESOURCE_URIS[4]: lambda: _json_text(_handoff_product_payload()),
        PUBLIC_RESOURCE_URIS[5]: lambda: _json_text(_plot_recipe_payload()),
        PUBLIC_RESOURCE_URIS[6]: lambda: _json_text(_animation_recipe_payload()),
        PUBLIC_RESOURCE_URIS[7]: lambda: _json_text(_analysis_workflow_payload()),
        BOOTSTRAP_URI: agent_bootstrap_text,
        WORKFLOWS_URI: lambda: _json_text(workflow_routes(contract.tool_id for contract in contracts)),
    }
    published: list[PublishedResource] = []
    for contract in PUBLIC_RESOURCE_CONTRACTS:
        text = loaders[contract.uri]()
        size = len(text.encode("utf-8"))
        if size > MAX_RESOURCE_BYTES:
            raise ValueError(f"Packaged MCP resource exceeds the {MAX_RESOURCE_BYTES}-byte limit: {contract.uri}")
        published.append(PublishedResource(contract=contract, text=text))
    return tuple(published)


def public_resource_map(resources: Iterable[PublishedResource]) -> dict[str, PublishedResource]:
    return {resource.contract.uri: resource for resource in resources}


def _tool_schema_payload(profile: str, contracts: tuple[ToolContract, ...]) -> dict[str, Any]:
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "resource_uri": PUBLIC_RESOURCE_URIS[0],
        "deployment_profile": profile,
        "source": "integrations.oel_mcp.public_registry",
        "tools": [contract.mcp_definition() for contract in contracts],
        "non_claims": [
            "Resource discovery is not execution authority.",
            "Only the listed deployment registry is available through this resource.",
        ],
    }


def _saved_query_payload() -> dict[str, Any]:
    from sim.review.queries import list_saved_review_queries

    queries = [
        {
            "name": query.name,
            "description": query.description,
            "sql": query.sql,
            "source_tables": list(query.source_tables),
            "maturity": query.maturity,
            "allow_empty": query.allow_empty,
        }
        for query in list_saved_review_queries()
    ]
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "resource_uri": PUBLIC_RESOURCE_URIS[1],
        "source": "sim.review.queries",
        "queries": queries,
        "query_count": len(queries),
        "rules": {
            "read_only_prefixes": ["SELECT", "WITH"],
            "units_are_never_inferred": True,
            "review_database_mutation_allowed": False,
        },
    }


def _agent_task_payload() -> dict[str, Any]:
    from sim.agent_task.recipes import is_public_mcp_recipe, list_recipes

    recipes = [asdict(recipe) for recipe in list_recipes() if is_public_mcp_recipe(recipe)]
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "resource_uri": PUBLIC_RESOURCE_URIS[2],
        "source": "sim.agent_task.recipes",
        "tasks": recipes,
        "task_count": len(recipes),
        "non_claims": [
            "Task resources describe documented workflows; reading them does not run a task.",
            "Examples and recipes do not replace scenario validation or deterministic OEL evidence.",
        ],
    }


def _operator_guide_text() -> str:
    resource = files("integrations.oel_mcp").joinpath("resource_data/operator-guide.md")
    return resource.read_text(encoding="utf-8")


def _handoff_product_payload() -> dict[str, Any]:
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "resource_uri": PUBLIC_RESOURCE_URIS[4],
        "source": "sim.handoff",
        "product_kinds": [
            {
                "product_kind": "oel.state_estimate",
                "producer_tools": [],
                "next_actions": ["oel.inspect_handoff.v1", "oel.materialize_onp_handoff.v1"],
            },
            {
                "product_kind": "oel.completed_run_state",
                "producer_tools": ["oel.export_run_product.v1"],
                "next_actions": ["oel.inspect_handoff.v1", "oel.materialize_onp_handoff.v1"],
            },
            {
                "product_kind": "oel.completed_run_snapshot",
                "producer_tools": ["oel.export_run_product.v1"],
                "next_actions": ["oel.inspect_handoff.v1", "oel.materialize_onp_handoff.v1"],
            },
            {
                "product_kind": "oel.maneuver_detection",
                "producer_tools": ["oel.export_run_product.v1"],
                "next_actions": ["oel.inspect_handoff.v1"],
            },
            {
                "product_kind": "oel.scenario_patch",
                "producer_tools": ["oel.emit_scenario_overlay.v1"],
                "next_actions": ["oel.inspect_handoff.v1", "oel.materialize_scenario_patch.v1"],
            },
        ],
        "rules": {
            "materialization_executes": False,
            "generated_scenarios_execution_authorized": False,
            "inspect_before_consume": True,
        },
    }


def _plot_recipe_payload() -> dict[str, Any]:
    from sim.review.plot_recipes import PLOT_RECIPE_SCHEMA_VERSION, list_review_plot_recipes

    recipes = [recipe.to_dict() for recipe in list_review_plot_recipes() if recipe.maturity == "supported"]
    return {
        "schema_version": PLOT_RECIPE_SCHEMA_VERSION,
        "resource_uri": PUBLIC_RESOURCE_URIS[5],
        "source": "sim.review.plot_recipes",
        "recipes": recipes,
        "recipe_count": len(recipes),
        "routing": {
            "oel_review_evidence_plotter_is_authoritative": True,
            "prefer_recipe_before_custom_spec": True,
            "custom_spec_tool": "oel.render_review_plot.v2",
            "visual_inspection_required": True,
        },
        "non_claims": [
            "A plot visualizes recorded review evidence; it does not validate physics accuracy.",
            "Missing review columns are not reconstructed or approximated by the plotting layer.",
        ],
    }


def _animation_recipe_payload() -> dict[str, Any]:
    from sim.plotting.animation_quality import STRICT_AGENT_ANIMATION_QUALITY
    from sim.review.animation_recipes import ANIMATION_RECIPE_SCHEMA_VERSION, list_review_animation_recipes

    recipes = [recipe.to_dict() for recipe in list_review_animation_recipes() if recipe.maturity == "supported"]
    return {
        "schema_version": ANIMATION_RECIPE_SCHEMA_VERSION,
        "resource_uri": PUBLIC_RESOURCE_URIS[6],
        "source": "sim.review.animation_recipes",
        "recipes": recipes,
        "recipe_count": len(recipes),
        "quality_policy": {
            "policy_id": STRICT_AGENT_ANIMATION_QUALITY.policy_id,
            "policy_version": STRICT_AGENT_ANIMATION_QUALITY.version,
            "max_frames": STRICT_AGENT_ANIMATION_QUALITY.max_frames,
            "max_duration_s": STRICT_AGENT_ANIMATION_QUALITY.max_duration_s,
            "contact_sheet_required": True,
            "visual_inspection_required": True,
        },
        "routing": {
            "prefer_supported_recipe": True,
            "plan_tool": "oel.plan_review_animation.v1",
            "render_tool": "oel.render_review_animation.v1",
        },
        "non_claims": [
            "An animation visualizes recorded review evidence; it does not validate physics accuracy.",
            "Automated frame and encoding checks do not replace inspection of the movie and contact sheet.",
        ],
    }


def _analysis_workflow_payload() -> dict[str, Any]:
    workflows = [
        {
            "workflow_id": "ccsds_interchange",
            "product_boundary": "public",
            "interfaces": ["python -m sim.ccsds", "sim.ccsds"],
            "evidence": "typed inspection or conversion receipt",
            "authoritative_replay": "reparse or reconvert from the retained source",
            "mcp_tools": ["oel.inspect_ccsds.v1"],
        },
        {
            "workflow_id": "frame_time_conversion",
            "product_boundary": "public",
            "interfaces": ["python -m sim.frame_time", "sim.frame_time"],
            "evidence": "epoch, EOP, or frame-transform receipt",
            "authoritative_replay": "repeat the content-bound conversion with the retained EOP source",
            "mcp_tools": ["oel.convert_frame_time.v1"],
        },
        {
            "workflow_id": "trajectory_targeting",
            "product_boundary": "public",
            "interfaces": ["python -m sim.trajectory_design solve", "sim.trajectory_design"],
            "evidence": "oel.trajectory_targeting_evidence.v1",
            "authoritative_replay": "python -m sim.trajectory_design replay",
            "mcp_tools": [],
        },
        {
            "workflow_id": "constellation_design",
            "product_boundary": "public",
            "interfaces": ["python -m sim.constellation_design solve", "sim.constellation_design"],
            "evidence": "oel.constellation_design_evidence.v1",
            "authoritative_replay": "python -m sim.constellation_design replay",
            "mcp_tools": [],
        },
        {
            "workflow_id": "conjunction_assessment",
            "product_boundary": "public",
            "interfaces": ["python -m sim.conjunction assess", "sim.conjunction"],
            "evidence": "oel.conjunction_assessment_evidence.v1",
            "authoritative_replay": "python -m sim.conjunction replay",
            "mcp_tools": [],
        },
        {
            "workflow_id": "collection_analysis",
            "product_boundary": "public",
            "interfaces": ["python -m sim.collection", "sim.collection"],
            "evidence": "typed collection opportunity or capacity evidence",
            "authoritative_replay": "workflow-specific replay over retained typed inputs",
            "mcp_tools": [],
        },
        {
            "workflow_id": "tracking_data_orbit_determination",
            "product_boundary": "public",
            "interfaces": ["python -m sim.tracking_od", "sim.tracking_od"],
            "evidence": "typed OD fit, holdout, and state-product evidence",
            "authoritative_replay": "workflow-specific replay over retained TDM and problem inputs",
            "mcp_tools": [],
        },
        {
            "workflow_id": "mission_scheduling",
            "product_boundary": "public",
            "interfaces": ["python -m sim.mission_scheduling solve", "sim.mission_scheduling"],
            "evidence": "oel.mission_scheduling_evidence.v1",
            "authoritative_replay": "python -m sim.mission_scheduling replay",
            "mcp_tools": [],
        },
        {
            "workflow_id": "spacecraft_power",
            "product_boundary": "public",
            "interfaces": ["python -m sim.spacecraft_power analyze", "sim.spacecraft_power"],
            "evidence": "oel.spacecraft_power_evidence.v1",
            "authoritative_replay": "python -m sim.spacecraft_power replay",
            "mcp_tools": [],
        },
        {
            "workflow_id": "orbit_lifetime",
            "product_boundary": "public",
            "interfaces": ["python -m sim.orbit_lifetime analyze", "sim.orbit_lifetime"],
            "evidence": "oel.orbit_lifetime_evidence.v1",
            "authoritative_replay": "python -m sim.orbit_lifetime replay",
            "mcp_tools": [],
        },
        {
            "workflow_id": "study_lifecycle",
            "product_boundary": "public",
            "interfaces": ["python -m sim.study", "sim.study"],
            "evidence": "content-bound study request, plan, run, evidence, claims, and receipt",
            "authoritative_replay": "identity and citation replay over retained completed evidence",
            "mcp_tools": ["oel.inspect_study.v1", "oel.replay_study.v1", "oel.compare_studies.v1"],
        },
        {
            "workflow_id": "source_schedule_power_study",
            "product_boundary": "public",
            "interfaces": [
                "python -m sim.study build-source-schedule-power",
                "python -m sim.study inspect-source-schedule-power",
            ],
            "evidence": "oel.source_schedule_power_study.v1 with retained collection/link, schedule, power, and study packets",
            "authoritative_replay": "python -m sim.study inspect-source-schedule-power",
            "mcp_tools": [],
        },
        {
            "workflow_id": "orbit_bound_source_study",
            "product_boundary": "public",
            "interfaces": [
                "python -m sim.study export-orbit-history",
                "python -m sim.collection <problem> --orbit-history-dir <product>",
                "python -m sim.study build-orbit-bound-link",
                "python -m sim.study build-orbit-bound-source-study",
                "python -m sim.study inspect-orbit-bound-source-study",
            ],
            "evidence": "oel.orbit_bound_source_study.v1 with one retained parent ECI history",
            "authoritative_replay": "python -m sim.study inspect-orbit-bound-source-study",
            "mcp_tools": [],
        },
    ]
    for workflow_id, interfaces, evidence, replay, tools in (
        ("trajectory_optimization", ["python -m sim.pro_trajectory_optimization"],
         "bounded optimization result", "python -m sim.pro_trajectory_optimization replay",
         ["oel.validate_trajectory_optimization.v1", "oel.run_trajectory_optimization.v1",
          "oel.replay_trajectory_optimization.v1"]),
        ("constellation_optimization", ["python -m sim.pro_constellation_optimization"],
         "bounded design search with ranked candidates", "python -m sim.pro_constellation_optimization replay", []),
        ("scale_catalog", ["python -m sim.scale"],
         "local catalog, screening, and refresh artifacts", "workflow-specific inspection of retained artifacts",
         ["oel.inspect_scale_catalog.v1", "oel.plan_scale_recompute.v1", "oel.inspect_scale_recompute_plan.v1",
          "oel.validate_scale_refresh.v1", "oel.run_scale_refresh_to_onp.v1", "oel.inspect_scale_refresh.v1"]),
        ("local_tracking_and_slr", ["python -m sim.pro_tracking_od", "python -m sim.pro_slr_od"],
         "tracking fit, holdout, and SLR evidence", "workflow-specific replay of retained observations",
         ["oel.run_dynamics_od.v1", "oel.run_ground_sensor_od.v1", "oel.run_ogp_od.v1",
          "oel.run_sequential_od.v1", "oel.run_relative_od.v1"]),
        ("local_campaigns", ["oel-sim --config", "oel-sim --campaign"],
         "Monte Carlo, sensitivity, covariance, and config queue summaries", "repeat with the same inputs and seed", []),
        ("controller_bench", ["oel-sim --controller-bench", "python -m sim.controller_lab"],
         "bench summary, criteria, and review evidence", "rerun the same declared cases and variants", []),
    ):
        workflows.append({
            "workflow_id": workflow_id,
            "product_boundary": "public",
            "interfaces": interfaces,
            "evidence": evidence,
            "authoritative_replay": replay,
            "mcp_tools": tools,
        })
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "resource_uri": PUBLIC_RESOURCE_URIS[7],
        "source": "docs/agent-capability-routing.md; sim.analysis",
        "workflow_count": len(workflows),
        "workflows": workflows,
        "cross_cutting_pro_escalations": [],
        "routing": {
            "common_loop": (
                "request -> route -> scenario YAML or typed orbital-analysis problem -> validate -> execute -> "
                "authoritative replay -> inspect -> bounded claim"
            ),
            "unlisted_mcp_execution_tools_are_not_available": True,
            "cli_and_python_api_remain_foundational": True,
            "pro_recommendations_are_not_execution_authority": True,
            "hosted_execution_package_validator": "oel.hosted.validate_package.v1",
            "hosted_pro_package_validator": "oel.hosted.validate_package.v1",
            "hosted_pro_package_validation_is_not_execution": True,
        },
        "non_claims": [
            "Workflow discovery is not execution authority.",
            "An empty mcp_tools list means use the documented CLI or Python API; it does not authorize approximation.",
            "Public evidence does not establish operational qualification, maneuver authority, or global optimality.",
            "A locally valid Hosted execution package still requires authoritative Hosted preflight and explicit approval.",
        ],
    }


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


__all__ = [
    "MAX_RESOURCE_BYTES",
    "PUBLIC_RESOURCE_CONTRACTS",
    "PUBLIC_RESOURCE_URIS",
    "PUBLIC_SAVED_QUERY_NAMES",
    "PublishedResource",
    "ResourceContract",
    "build_public_resource_catalog",
    "public_resource_map",
]
