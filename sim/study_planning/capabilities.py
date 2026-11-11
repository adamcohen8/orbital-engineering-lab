"""Explicit composable capability vocabulary for study-plan compilation.

Availability is derived from an authoritative executor binding.  A capability
without a binding remains discoverable, but cannot be selected by preflight.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from .contracts import STUDY_CAPABILITY_SCHEMA_ID, require_valid_document


def _parameters(properties: dict[str, Any], *, required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


PARAMETER_SCHEMAS: dict[str, dict[str, Any]] = {
    "oel.config.validate.v1": _parameters({"safe_only": {"const": True}}, required=("safe_only",)),
    "oel.scenario.execute.v1": _parameters(
        {"approval_required": {"const": True}}, required=("approval_required",)
    ),
    "oel.run.inspect.v1": _parameters({}),
    "oel.review.query.v1": _parameters(
        {
            "query": {"type": "string", "minLength": 1, "maxLength": 100_000},
            "max_rows": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        required=("query",),
    ),
    "oel.run.compare.v1": _parameters(
        {"metric_names": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 64}},
        required=("metric_names",),
    ),
    "oel.review.plot.v1": _parameters(
        {
            "recipe": {"type": "string", "minLength": 1, "maxLength": 160},
            "title": {"type": "string", "maxLength": 300},
        },
        required=("recipe",),
    ),
    "oel.tracking_od.fit.v1": _parameters(
        {"authoritative_replay_required": {"const": True}}, required=("authoritative_replay_required",)
    ),
    "oel.mission_scheduling.solve.v1": _parameters(
        {"authoritative_replay_required": {"const": True}}, required=("authoritative_replay_required",)
    ),
    "oel.spacecraft_power.analyze.v1": _parameters(
        {"authoritative_replay_required": {"const": True}}, required=("authoritative_replay_required",)
    ),
    "oel.orbit_lifetime.analyze.v1": _parameters(
        {"authoritative_replay_required": {"const": True}}, required=("authoritative_replay_required",)
    ),
    "oel.pro.campaign.monte_carlo.v1": _parameters(
        {
            "samples": {"type": "integer", "minimum": 2, "maximum": 1_000_000},
            "seed": {"type": "integer", "minimum": 0, "maximum": 4_294_967_295},
            "distributions": {"type": "array", "items": {"type": "object"}, "minItems": 1, "maxItems": 128},
        },
        required=("samples", "seed", "distributions"),
    ),
    "oel.pro.campaign.sensitivity.v1": _parameters(
        {
            "design": {"type": "string", "enum": ["one_at_a_time", "grid", "latin_hypercube"]},
            "parameters": {"type": "array", "items": {"type": "object"}, "minItems": 1, "maxItems": 128},
            "cases": {"type": "integer", "minimum": 2, "maximum": 1_000_000},
        },
        required=("design", "parameters", "cases"),
    ),
    "oel.pro.controller.benchmark.v1": _parameters(
        {
            "controller_variants": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 64,
                "uniqueItems": True,
            },
            "objective_metrics": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 64,
                "uniqueItems": True,
            },
        },
        required=("controller_variants", "objective_metrics"),
    ),
    "oel.pro.trajectory_optimization.v1": _parameters(
        {"authoritative_replay_required": {"const": True}},
        required=("authoritative_replay_required",),
    ),
    "orbit_determination.reduced_tracking": _parameters(
        {"authoritative_replay_required": {"const": True}},
        required=("authoritative_replay_required",),
    ),
    "orbit_determination.ilrs_slr": _parameters(
        {"authoritative_replay_required": {"const": True}},
        required=("authoritative_replay_required",),
    ),
    "oel.pro.scale.screening.v1": _parameters(
        {
            "mode": {"type": "string", "enum": ["inspect", "screen", "refine"]},
            "maximum_candidates": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
        },
        required=("mode", "maximum_candidates"),
    ),
    "oel.pro.report.packet.v1": _parameters(
        {
            "sections": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 32,
                "uniqueItems": True,
            }
        },
        required=("sections",),
    ),
}


@dataclass(frozen=True, slots=True)
class CapabilityExecutorBinding:
    capability_id: str
    status: str
    executor_contract_ids: tuple[str, ...]
    adapter_id: str | None


@dataclass(frozen=True, slots=True)
class CapabilityPort:
    port_id: str
    semantic_role: str
    artifact_kind: str
    schema_ids: tuple[str, ...]
    cardinality: str = "exactly_one"
    required_metadata: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if self.cardinality not in {"exactly_one", "optional", "many"}:
            raise ValueError(f"Invalid port cardinality for {self.port_id!r}.")
        if not self.schema_ids:
            raise ValueError(f"Capability port {self.port_id!r} requires at least one schema id.")
        return {
            "port_id": self.port_id,
            "semantic_role": self.semantic_role,
            "artifact_kind": self.artifact_kind,
            "schema_ids": list(self.schema_ids),
            "cardinality": self.cardinality,
            "required_metadata": list(self.required_metadata),
        }


PUBLIC_DOCUMENTED_EXECUTOR_MODULES: dict[str, str] = {
    "python.module.sim.tracking_od": "sim.tracking_od",
    "python.module.sim.mission_scheduling": "sim.mission_scheduling",
    "python.module.sim.spacecraft_power": "sim.spacecraft_power",
    "python.module.sim.orbit_lifetime": "sim.orbit_lifetime",
}


def _port(
    port_id: str,
    artifact_kind: str,
    *schema_ids: str,
    semantic_role: str | None = None,
    cardinality: str = "exactly_one",
    required_metadata: tuple[str, ...] = (),
) -> CapabilityPort:
    return CapabilityPort(
        port_id=port_id,
        semantic_role=semantic_role or port_id,
        artifact_kind=artifact_kind,
        schema_ids=tuple(schema_ids),
        cardinality=cardinality,
        required_metadata=required_metadata,
    )


EXECUTOR_BINDING_REGISTRY: dict[str, CapabilityExecutorBinding] = {
    binding.capability_id: binding
    for binding in (
        CapabilityExecutorBinding(
            "oel.config.validate.v1",
            "bound",
            ("oel.validate_scenario.v1",),
            "oel.study.adapter.public_config_validate.v1",
        ),
        CapabilityExecutorBinding(
            "oel.scenario.execute.v1",
            "bound",
            ("oel.run_scenario.v1",),
            "oel.study.adapter.public_scenario_execute.v1",
        ),
        CapabilityExecutorBinding(
            "oel.run.inspect.v1",
            "bound",
            ("oel.inspect_run.v1",),
            "oel.study.adapter.public_run_inspect.v1",
        ),
        CapabilityExecutorBinding(
            "oel.review.query.v1",
            "bound",
            ("oel.query_review.v1",),
            "oel.study.adapter.public_review_query.v1",
        ),
        CapabilityExecutorBinding(
            "oel.run.compare.v1",
            "bound",
            ("oel.compare_runs.v1",),
            "oel.study.adapter.public_run_compare.v1",
        ),
        CapabilityExecutorBinding(
            "oel.review.plot.v1",
            "bound",
            ("oel.plan_review_plot.v1", "oel.render_review_plot.v2"),
            "oel.study.adapter.public_review_plot.v1",
        ),
        CapabilityExecutorBinding(
            "oel.tracking_od.fit.v1", "bound", ("python.module.sim.tracking_od",),
            "python.module.sim.tracking_od",
        ),
        CapabilityExecutorBinding(
            "oel.mission_scheduling.solve.v1", "bound", ("python.module.sim.mission_scheduling",),
            "python.module.sim.mission_scheduling",
        ),
        CapabilityExecutorBinding(
            "oel.spacecraft_power.analyze.v1", "bound", ("python.module.sim.spacecraft_power",),
            "python.module.sim.spacecraft_power",
        ),
        CapabilityExecutorBinding(
            "oel.orbit_lifetime.analyze.v1", "bound", ("python.module.sim.orbit_lifetime",),
            "python.module.sim.orbit_lifetime",
        ),
        *(
            CapabilityExecutorBinding(capability_id, "unbound", (), None)
            for capability_id in (
                "oel.pro.campaign.monte_carlo.v1",
                "oel.pro.campaign.sensitivity.v1",
                "oel.pro.controller.benchmark.v1",
                "oel.pro.trajectory_optimization.v1",
                "orbit_determination.reduced_tracking",
                "orbit_determination.ilrs_slr",
                "oel.pro.scale.screening.v1",
                "oel.pro.report.packet.v1",
            )
        ),
    )
}


def _executor_binding_fields(capability_id: str) -> dict[str, Any]:
    binding = EXECUTOR_BINDING_REGISTRY[capability_id]
    return {
        "binding_status": binding.status,
        "executor_contract_ids": binding.executor_contract_ids,
        "binding_adapter_id": binding.adapter_id,
    }


@dataclass(frozen=True, slots=True)
class StudyCapability:
    capability_id: str
    title: str
    description: str
    edition: str
    maturity: str
    binding_status: str
    executor_contract_ids: tuple[str, ...]
    binding_adapter_id: str | None
    writes: bool
    executes: bool
    input_kinds: tuple[str, ...]
    output_kinds: tuple[str, ...]
    input_ports: tuple[CapabilityPort, ...] = ()
    output_ports: tuple[CapabilityPort, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def availability(self) -> str:
        return "available" if self.binding_status == "bound" else "unavailable"

    @property
    def parameter_schema(self) -> dict[str, Any]:
        return deepcopy(PARAMETER_SCHEMAS[self.capability_id])

    @property
    def compiled_input_ports(self) -> tuple[CapabilityPort, ...]:
        if self.input_ports:
            return self.input_ports
        cardinality = "exactly_one" if len(self.input_kinds) == 1 else "optional"
        return tuple(
            _port(
                f"input_{kind}",
                kind,
                f"oel.artifact.{kind}.v1",
                semantic_role=kind,
                cardinality=cardinality,
            )
            for kind in self.input_kinds
        )

    @property
    def compiled_output_ports(self) -> tuple[CapabilityPort, ...]:
        return self.output_ports or tuple(
            _port(
                f"output_{kind}",
                kind,
                f"oel.artifact.{kind}.v1",
                semantic_role=kind,
            )
            for kind in self.output_kinds
        )

    def to_dict(self) -> dict[str, Any]:
        if self.binding_status not in {"bound", "unbound"}:
            raise ValueError(f"Invalid executor binding status for {self.capability_id!r}.")
        if self.binding_status == "bound" and (
            not self.executor_contract_ids or not self.binding_adapter_id
        ):
            raise ValueError(f"Bound capability {self.capability_id!r} requires an executor and adapter.")
        if self.binding_status == "unbound" and self.binding_adapter_id is not None:
            raise ValueError(f"Unbound capability {self.capability_id!r} cannot declare a binding adapter.")
        public_edition = self.edition == "public"
        public_available = public_edition and self.availability == "available"
        payload = {
            "schema": STUDY_CAPABILITY_SCHEMA_ID,
            "capability_id": self.capability_id,
            "title": self.title,
            "description": self.description,
            "edition": self.edition,
            "maturity": self.maturity,
            "availability": self.availability,
            "executor_binding": {
                "status": self.binding_status,
                "executor_contract_ids": list(self.executor_contract_ids),
                "adapter_id": self.binding_adapter_id,
            },
            "access": {
                "discoverable_in_public": True,
                "available_in_public_install": public_available,
                "access_mode": "public_local" if public_edition else "hosted_pro_only",
                "hosted_pro_required": not public_edition,
            },
            "effects": {
                "reads": True,
                "writes": self.writes,
                "executes": self.executes,
                "external_communication": False,
            },
            "input_kinds": list(self.input_kinds),
            "output_kinds": list(self.output_kinds),
            "input_ports": [item.to_dict() for item in self.compiled_input_ports],
            "output_ports": [item.to_dict() for item in self.compiled_output_ports],
            "parameter_schema": self.parameter_schema,
            "limitations": list(self.limitations),
        }
        return require_valid_document(payload, expected_schema=STUDY_CAPABILITY_SCHEMA_ID)


class CapabilityCatalog:
    def __init__(self, capabilities: Iterable[StudyCapability], *, scope: str) -> None:
        items = tuple(capabilities)
        identifiers = [item.capability_id for item in items]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Study capability identifiers must be unique.")
        for item in items:
            binding = EXECUTOR_BINDING_REGISTRY.get(item.capability_id)
            if binding is None or (
                item.binding_status,
                item.executor_contract_ids,
                item.binding_adapter_id,
            ) != (binding.status, binding.executor_contract_ids, binding.adapter_id):
                raise ValueError(
                    f"Capability {item.capability_id!r} does not match the authoritative executor registry."
                )
            item.to_dict()
        self._items = items
        self._by_id = {item.capability_id: item for item in items}
        self.scope = str(scope)

    def get(self, capability_id: str) -> StudyCapability | None:
        return self._by_id.get(str(capability_id))

    def require(self, capability_id: str) -> StudyCapability:
        capability = self.get(capability_id)
        if capability is None:
            raise KeyError(str(capability_id))
        return capability

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]

    def available_ids(self) -> tuple[str, ...]:
        return tuple(item.capability_id for item in self._items if item.availability == "available")


PUBLIC_STUDY_CAPABILITIES = (
    StudyCapability(
        capability_id="oel.config.validate.v1",
        title="Validate an OEL configuration",
        description="Safely parse, normalize, and validate one supported OEL scenario without advancing simulation state.",
        edition="public",
        maturity="supported",
        **_executor_binding_fields("oel.config.validate.v1"),
        writes=False,
        executes=False,
        input_kinds=("scenario_config", "study_intent"),
        output_kinds=(
            "validation_receipt",
            "validated_scenario",
            "normalized_config",
            "resource_estimate",
        ),
        limitations=("Trusted plugin import requires separate host authority.", "Validation is not physics qualification."),
    ),
    StudyCapability(
        capability_id="oel.scenario.execute.v1",
        title="Execute a deterministic OEL scenario",
        description="Run one validated OEL scenario through the canonical deterministic engine and retain review evidence.",
        edition="public",
        maturity="supported",
        **_executor_binding_fields("oel.scenario.execute.v1"),
        writes=True,
        executes=True,
        input_kinds=("validated_scenario",),
        output_kinds=("completed_run", "review_store", "artifact_manifest"),
        limitations=("Execution requires separate user approval.", "Completion is not scientific qualification."),
    ),
    StudyCapability(
        capability_id="oel.run.inspect.v1",
        title="Inspect completed run evidence",
        description="Inspect durable status, provenance, review evidence, and artifact disposition for an OEL run.",
        edition="public",
        maturity="supported",
        **_executor_binding_fields("oel.run.inspect.v1"),
        writes=False,
        executes=False,
        input_kinds=("completed_run",),
        output_kinds=("run_inspection",),
    ),
    StudyCapability(
        capability_id="oel.review.query.v1",
        title="Query OEL review evidence",
        description="Run a bounded read-only query over a completed OEL review store.",
        edition="public",
        maturity="supported",
        **_executor_binding_fields("oel.review.query.v1"),
        writes=False,
        executes=False,
        input_kinds=("review_store",),
        output_kinds=("query_evidence",),
        limitations=("Only bounded read-only queries are supported.",),
    ),
    StudyCapability(
        capability_id="oel.run.compare.v1",
        title="Compare completed OEL runs",
        description="Compare allowlisted metrics from identity-bound completed runs.",
        edition="public",
        maturity="supported",
        **_executor_binding_fields("oel.run.compare.v1"),
        writes=False,
        executes=False,
        input_kinds=("completed_run",),
        output_kinds=("comparison_evidence",),
    ),
    StudyCapability(
        capability_id="oel.review.plot.v1",
        title="Render OEL review evidence",
        description="Plan or render a supported plot from completed OEL review evidence.",
        edition="public",
        maturity="supported",
        **_executor_binding_fields("oel.review.plot.v1"),
        writes=True,
        executes=False,
        input_kinds=("review_store", "plot_specification"),
        output_kinds=("plot_artifact", "plot_receipt"),
        limitations=("The plot must be derived from recorded review evidence.",),
    ),
    StudyCapability(
        capability_id="oel.tracking_od.fit.v1",
        title="Fit one bounded public CCSDS TDM tracking arc",
        description="Run the public reduced-geometric UTC AZEL/range batch fit with a mandatory untouched holdout.",
        edition="public", maturity="experimental",
        **_executor_binding_fields("oel.tracking_od.fit.v1"),
        writes=True, executes=True,
        input_kinds=("tracking_od_problem", "ccsds_tdm_kvn"),
        output_kinds=("tracking_od_evidence", "authoritative_replay_receipt"),
        input_ports=(
            _port("problem", "tracking_od_problem", "oel.tracking_od_problem.v1", required_metadata=("content_sha256", "arc_duration_s")),
            _port("observations", "ccsds_tdm_kvn", "oel.ccsds-tdm-kvn.v0.1", required_metadata=("content_sha256", "observation_count")),
        ),
        output_ports=(
            _port("evidence", "tracking_od_evidence", "oel.tracking_od_evidence.v1"),
            _port("replay", "authoritative_replay_receipt", "oel.tracking_od_replay_receipt.v1"),
        ),
        limitations=("No raw radiometric reduction, calibrated predicted accuracy, custody, or operational authority.",),
    ),
    StudyCapability(
        capability_id="oel.mission_scheduling.solve.v1",
        title="Solve one bounded mission schedule",
        description="Select an exact feasible subset from at most eighteen supplied opportunities and emit replayable scheduling evidence.",
        edition="public", maturity="experimental",
        **_executor_binding_fields("oel.mission_scheduling.solve.v1"),
        writes=True, executes=True,
        input_kinds=("mission_scheduling_problem",),
        output_kinds=("mission_scheduling_evidence", "authoritative_replay_receipt"),
        input_ports=(_port("problem", "mission_scheduling_problem", "oel.mission_scheduling_problem.v1", required_metadata=("content_sha256",)),),
        output_ports=(
            _port("evidence", "mission_scheduling_evidence", "oel.mission_scheduling_evidence.v1"),
            _port("replay", "authoritative_replay_receipt", "oel.mission_scheduling_replay_receipt.v1"),
        ),
        limitations=("Source opportunities and battery, thermal, routing, and command feasibility are not created or qualified by scheduling.",),
    ),
    StudyCapability(
        capability_id="oel.spacecraft_power.analyze.v1",
        title="Analyze bounded spacecraft power feasibility",
        description="Evaluate a declared load timeline, array, and lumped battery against one retained orbit history.",
        edition="public", maturity="experimental",
        **_executor_binding_fields("oel.spacecraft_power.analyze.v1"),
        writes=True, executes=True,
        input_kinds=("spacecraft_power_problem", "orbit_history"),
        output_kinds=("spacecraft_power_evidence", "authoritative_replay_receipt"),
        input_ports=(
            _port("problem", "spacecraft_power_problem", "oel.spacecraft_power_problem.v1", required_metadata=("content_sha256",)),
            _port("history", "orbit_history", "oel.spacecraft_power_history.v1", required_metadata=("content_sha256", "sample_count")),
        ),
        output_ports=(
            _port("evidence", "spacecraft_power_evidence", "oel.spacecraft_power_evidence.v1"),
            _port("replay", "authoritative_replay_receipt", "oel.spacecraft_power_replay_receipt.v1"),
        ),
        limitations=("This is not detailed EPS, thermal, degradation, uncertainty, or flight qualification.",),
    ),
    StudyCapability(
        capability_id="oel.orbit_lifetime.analyze.v1",
        title="Analyze bounded deterministic orbit lifetime",
        description="Propagate one declared ONP drag case to a horizon or threshold and return replayable decay evidence.",
        edition="public", maturity="experimental",
        **_executor_binding_fields("oel.orbit_lifetime.analyze.v1"),
        writes=True, executes=True,
        input_kinds=("orbit_lifetime_problem",),
        output_kinds=("orbit_lifetime_evidence", "authoritative_replay_receipt"),
        input_ports=(_port("problem", "orbit_lifetime_problem", "oel.orbit_lifetime_problem.v1", required_metadata=("content_sha256", "arc_duration_s")),),
        output_ports=(
            _port("evidence", "orbit_lifetime_evidence", "oel.orbit_lifetime_evidence.v1"),
            _port("replay", "authoritative_replay_receipt", "oel.orbit_lifetime_replay_receipt.v1"),
        ),
        limitations=("A horizon-complete run is not an extrapolated lifetime or operational disposal conclusion.",),
    ),
)

PRO_STUDY_CAPABILITIES = (
    StudyCapability(
        capability_id="oel.pro.campaign.monte_carlo.v1",
        title="Run a bounded Monte Carlo campaign",
        description="Execute OEL's checked-in Monte Carlo orchestration over a validated scenario and declared distributions.",
        edition="pro",
        maturity="supported",
        **_executor_binding_fields("oel.pro.campaign.monte_carlo.v1"),
        writes=True,
        executes=True,
        input_kinds=("validated_scenario", "completed_run", "distribution_specification"),
        output_kinds=("campaign_evidence", "review_artifacts"),
        limitations=("Statistical claims are bounded to the declared samples and distributions.",),
    ),
    StudyCapability(
        capability_id="oel.pro.campaign.sensitivity.v1",
        title="Run a bounded sensitivity study",
        description="Execute supported one-at-a-time, grid, or Latin-hypercube sensitivity analysis over declared parameters.",
        edition="pro",
        maturity="supported",
        **_executor_binding_fields("oel.pro.campaign.sensitivity.v1"),
        writes=True,
        executes=True,
        input_kinds=("validated_scenario", "sensitivity_specification"),
        output_kinds=("sensitivity_evidence", "review_artifacts"),
        limitations=("Sensitivity evidence does not establish global causality or optimality.",),
    ),
    StudyCapability(
        capability_id="oel.pro.controller.benchmark.v1",
        title="Benchmark supported controllers",
        description="Compare supported controller variants against explicit objectives and evidence criteria.",
        edition="pro",
        maturity="supported",
        **_executor_binding_fields("oel.pro.controller.benchmark.v1"),
        writes=True,
        executes=True,
        input_kinds=("controller_benchmark_specification",),
        output_kinds=("controller_comparison_evidence",),
        limitations=("Results apply only to the declared cases, models, objectives, and controller variants.",),
    ),
    StudyCapability(
        capability_id="oel.pro.trajectory_optimization.v1",
        title="Run bounded trajectory optimization",
        description="Solve one typed bounded trajectory-optimization problem and require authoritative replay evidence.",
        edition="pro",
        maturity="experimental",
        **_executor_binding_fields("oel.pro.trajectory_optimization.v1"),
        writes=True,
        executes=True,
        input_kinds=("trajectory_optimization_problem",),
        output_kinds=("trajectory_optimization_evidence", "authoritative_replay_receipt"),
        input_ports=(
            _port(
                "problem",
                "trajectory_optimization_problem",
                "oel.trajectory_optimization_problem.v1",
                semantic_role="optimization_problem",
                required_metadata=("content_sha256",),
            ),
        ),
        output_ports=(
            _port("evidence", "trajectory_optimization_evidence", "oel.trajectory_optimization_evidence.v1"),
            _port("replay", "authoritative_replay_receipt", "oel.trajectory_optimization_replay_receipt.v1"),
        ),
        limitations=("Global optimality and maneuver authority are not established.",),
    ),
    StudyCapability(
        capability_id="orbit_determination.reduced_tracking",
        title="Run sequential orbit determination over reduced tracking data",
        description="Run the entitled reduced ground or optical EKF/RTS workflow with an untouched holdout and authoritative replay.",
        edition="pro",
        maturity="supported",
        **_executor_binding_fields("orbit_determination.reduced_tracking"),
        writes=True,
        executes=True,
        input_kinds=("pro_tracking_od_problem",),
        output_kinds=("pro_tracking_od_evidence", "authoritative_replay_receipt"),
        input_ports=(
            _port(
                "problem",
                "pro_tracking_od_problem",
                "oel.pro_tracking_od_problem.v1",
                semantic_role="tracking_od_problem",
                required_metadata=(
                    "content_sha256",
                    "input_bytes",
                    "measurement_model",
                    "observation_count",
                    "arc_duration_s",
                    "station_count",
                ),
            ),
        ),
        output_ports=(
            _port("evidence", "pro_tracking_od_evidence", "oel.pro_tracking_od_evidence.v1"),
            _port("replay", "authoritative_replay_receipt", "oel.pro_tracking_od_replay_receipt.v1"),
        ),
        limitations=(
            "Inputs must already be reduced geometric ground or inertial optical observations.",
            "Covariance is not calibrated predicted accuracy, custody, or operational authority.",
        ),
    ),
    StudyCapability(
        capability_id="orbit_determination.ilrs_slr",
        title="Run bounded ILRS/SLR orbit determination",
        description="Run the entitled CRD-v2 normal-point batch plus EKF/RTS workflow with validation, untouched holdout, and replay.",
        edition="pro",
        maturity="supported",
        **_executor_binding_fields("orbit_determination.ilrs_slr"),
        writes=True,
        executes=True,
        input_kinds=("pro_slr_od_problem", "ilrs_crd_v2_normal_points"),
        output_kinds=("pro_slr_od_evidence", "authoritative_replay_receipt"),
        input_ports=(
            _port(
                "problem",
                "pro_slr_od_problem",
                "oel.pro_slr_od_problem.v1",
                semantic_role="slr_od_problem",
                required_metadata=(
                    "content_sha256",
                    "input_bytes",
                    "arc_duration_s",
                    "batch_evaluations",
                ),
            ),
            _port(
                "normal_points",
                "ilrs_crd_v2_normal_points",
                "oel.ilrs_crd_v2_normal_points.v1",
                semantic_role="slr_normal_points",
                required_metadata=("content_sha256", "input_bytes", "normal_point_count", "station_count"),
            ),
        ),
        output_ports=(
            _port("evidence", "pro_slr_od_evidence", "oel.pro_slr_od_evidence.v1"),
            _port("replay", "authoritative_replay_receipt", "oel.pro_slr_od_replay_receipt.v1"),
        ),
        limitations=(
            "This is not an ILRS analysis-center or geodetic orbit solution.",
            "Media, target-center, calibration, association, custody, and operational authority remain unsupported.",
        ),
    ),
    StudyCapability(
        capability_id="oel.pro.scale.screening.v1",
        title="Run bounded OEL Scale analysis",
        description="Inspect or refine supported catalog-screening and targeted recompute products through OEL Scale.",
        edition="pro",
        maturity="experimental",
        **_executor_binding_fields("oel.pro.scale.screening.v1"),
        writes=True,
        executes=True,
        input_kinds=("scale_store", "screening_specification"),
        output_kinds=("screening_evidence", "handoff_product"),
        limitations=("Generic operational catalog authority and unrestricted campaign execution are not supported.",),
    ),
    StudyCapability(
        capability_id="oel.pro.report.packet.v1",
        title="Prepare a review-ready report packet",
        description="Assemble provider-neutral report inputs and audit evidence from completed OEL artifacts.",
        edition="pro",
        maturity="supported",
        **_executor_binding_fields("oel.pro.report.packet.v1"),
        writes=True,
        executes=False,
        input_kinds=("run_inspection", "campaign_evidence", "study_evidence", "study_claims"),
        output_kinds=("report_packet", "report_audit"),
        limitations=("Report assembly does not upgrade the underlying evidence or claims.",),
    ),
)


def public_capability_catalog() -> CapabilityCatalog:
    return CapabilityCatalog(PUBLIC_STUDY_CAPABILITIES, scope="public_only")


def discovery_capability_catalog() -> CapabilityCatalog:
    """Return public-safe descriptors for public and Pro planning surfaces.

    The Pro entries are contracts for agent discovery only. Their presence in
    this catalog does not make the underlying capability available locally.
    """

    return CapabilityCatalog(
        (*PUBLIC_STUDY_CAPABILITIES, *PRO_STUDY_CAPABILITIES),
        scope="public_agent_discovery",
    )


__all__ = [
    "CapabilityCatalog",
    "CapabilityExecutorBinding",
    "CapabilityPort",
    "EXECUTOR_BINDING_REGISTRY",
    "PRO_STUDY_CAPABILITIES",
    "PARAMETER_SCHEMAS",
    "PUBLIC_DOCUMENTED_EXECUTOR_MODULES",
    "PUBLIC_STUDY_CAPABILITIES",
    "StudyCapability",
    "discovery_capability_catalog",
    "public_capability_catalog",
]
