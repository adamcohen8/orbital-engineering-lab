"""Typed, local-only MCP contracts for orbit-determination workflows.

The registry deliberately contains no solver logic.  It describes a small
set of closed input boundaries around the deterministic OEL estimation
engines.  ``pro_registry`` owns profile selection and entitlement wiring;
this module can be imported by it without importing the parent registry.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from integrations.oel_mcp.contracts import ToolContract, object_schema
from integrations.oel_mcp.pro_registry_types import PRO_EXECUTION_PROFILES, PRO_M5_HANDLING_SCHEMA
from integrations.oel_mcp.public_registry import APPROVAL_SCHEMA
from sim.pro_features import FEATURE_SCALE_ANALYTICS

OD_TOOL_IDS = (
    "oel.pro.run_ground_sensor_od.v1",
    "oel.pro.run_ogp_od.v1",
    "oel.pro.run_sequential_od.v1",
    "oel.pro.run_relative_od.v1",
    "oel.pro.inspect_od_json.v1",
)

OD_EXECUTION_TOOL_IDS = OD_TOOL_IDS[:4]
OD_INSPECTION_TOOL_IDS = (OD_TOOL_IDS[4],)

_PATH = {"type": "string", "minLength": 1, "maxLength": 2048}
_ID = {"type": "string", "minLength": 1, "maxLength": 160}
_STATE6 = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 6,
    "maxItems": 6,
}
_COVARIANCE6 = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 6,
        "maxItems": 6,
    },
    "minItems": 6,
    "maxItems": 6,
}
_NONNEGATIVE_SECONDS = {"type": "number", "minimum": 0.0, "maximum": 604800.0}
_POSITIVE_SECONDS = {"type": "number", "exclusiveMinimum": 0.0, "maximum": 86400.0}
_SIGMA = {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0e6}


OD_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"enum": ["completed", "partial", "cancelled"]},
        "method": {"type": "string"},
        "object_id": {"type": "string"},
        "observation_count": {"type": "integer", "minimum": 0},
        "fit_observation_count": {"type": "integer", "minimum": 0},
        "holdout_observation_count": {"type": "integer", "minimum": 0},
        "quality_gates": {"type": "object"},
        "solver": {"type": "object"},
        "verdict": {"type": "object"},
        "output_dir": {"type": "string"},
        "report_path": {"type": "string"},
        "artifacts": {"type": "array", "items": {"type": "string"}},
        "resource_plan": {"type": "object"},
        "operation_manifest_path": {"type": "string"},
        "execution_authorized_outputs": {"const": False},
        "hidden_truth_visible": {"const": False},
        "local_only": {"const": True},
        "data_label": {"type": "string"},
        "entitlement": {"type": "object"},
        "deferred_capabilities": {"type": "array", "items": {"type": "string"}},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "method",
        "object_id",
        "observation_count",
        "fit_observation_count",
        "holdout_observation_count",
        "quality_gates",
        "solver",
        "verdict",
        "output_dir",
        "report_path",
        "artifacts",
        "resource_plan",
        "operation_manifest_path",
        "execution_authorized_outputs",
        "hidden_truth_visible",
        "local_only",
        "data_label",
        "entitlement",
        "deferred_capabilities",
        "non_claims",
    ),
)

OD_INSPECT_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"const": "completed"},
        "kind": {"type": "string"},
        "schema_version": {"type": ["integer", "null"]},
        "top_level_keys": {"type": "array", "items": {"type": "string"}},
        "safe_summary": {"type": "object"},
        "file_bytes": {"type": "integer", "minimum": 0},
        "hidden_truth_visible": {"const": False},
        "local_only": {"const": True},
        "data_label": {"type": "string"},
        "entitlement": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "kind",
        "schema_version",
        "top_level_keys",
        "safe_summary",
        "file_bytes",
        "hidden_truth_visible",
        "local_only",
        "data_label",
        "entitlement",
        "non_claims",
    ),
)


def _with_handling(
    properties: Mapping[str, Any],
    *,
    handling_schema: Mapping[str, Any],
    property_builder: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any]:
    values = deepcopy(dict(properties))
    if property_builder is not None:
        values = deepcopy(property_builder(values))
    values.setdefault("handling", deepcopy(dict(handling_schema)))
    return values


def _limits(
    *,
    writes: bool,
    limits_factory: Callable[..., dict[str, Any]] | Mapping[str, Any] | None,
    **extra: Any,
) -> dict[str, Any]:
    defaults = {
        "local_only": True,
        "hidden_truth_access": False,
        "entitlement_required": True,
        "external_communication": False,
        "new_output_required": writes,
        "operator_approval_required": writes,
    }
    defaults.update(extra)
    if limits_factory is None:
        return defaults
    if callable(limits_factory):
        return {**dict(limits_factory(**defaults)), **defaults}
    return {**dict(limits_factory), **defaults}


def _contract(
    *,
    tool_id: str,
    title: str,
    description: str,
    api: str,
    properties: Mapping[str, Any],
    required: tuple[str, ...],
    result_schema: dict[str, Any],
    writes: bool,
    executes: bool,
    handling_schema: Mapping[str, Any],
    approval_schema: Mapping[str, Any],
    property_builder: Callable[[dict[str, Any]], dict[str, Any]] | None,
    deployment_profiles: tuple[str, ...],
    entitlement: str,
    limits_factory: Callable[..., dict[str, Any]] | Mapping[str, Any] | None,
    limits: Mapping[str, Any],
) -> ToolContract:
    contract_properties = _with_handling(
        properties,
        handling_schema=handling_schema,
        property_builder=property_builder,
    )
    required_fields = [*required, "handling"]
    if writes:
        contract_properties["approval"] = deepcopy(dict(approval_schema))
        required_fields.append("approval")
    return ToolContract(
        tool_id=tool_id,
        title=title,
        description=description,
        risk_class="R3_sensitive",
        oel_api=api,
        maturity="supported",
        install_profile="pro",
        deployment_profiles=deployment_profiles,
        data_classes=("orbit_determination", "local_only_labeled_evidence"),
        writes=writes,
        executes=executes,
        input_schema=object_schema(contract_properties, required=tuple(required_fields)),
        result_schema=result_schema,
        required_entitlement=entitlement,
        limits=_limits(writes=writes, limits_factory=limits_factory, **dict(limits)),
    )


def od_contracts(
    *,
    handling_schema: Mapping[str, Any] | None = None,
    approval_schema: Mapping[str, Any] | None = None,
    property_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    deployment_profiles: tuple[str, ...] = PRO_EXECUTION_PROFILES,
    inspection_profiles: tuple[str, ...] | None = None,
    entitlement: str = FEATURE_SCALE_ANALYTICS,
    limits_factory: Callable[..., dict[str, Any]] | Mapping[str, Any] | None = None,
) -> tuple[ToolContract, ...]:
    """Return OD contracts for the requested Pro profile.

    ``pro_registry`` can provide its exact M5 handling and limit builders.  A
    safe default is retained for focused imports and SDK tests.  The factory
    does not import ``pro_registry`` so parent registry construction remains
    acyclic.
    """

    handling = deepcopy(dict(handling_schema or PRO_M5_HANDLING_SCHEMA))
    approval = deepcopy(dict(approval_schema or APPROVAL_SCHEMA))
    inspect_profiles = tuple(inspection_profiles or deployment_profiles)
    common_numeric = {
        # The MCP surface deliberately caps solver evaluations at the
        # laptop-safe first-pass budget.  Larger campaigns remain a native
        # Python/API workflow until a separately measured resource contract
        # exists.
        "max_nfev": {"type": "integer", "minimum": 1, "maximum": 24},
        "robust_loss": {"type": "string", "enum": ["linear", "soft_l1", "huber"]},
        "robust_f_scale": _SIGMA,
        "sigma_clip_threshold": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0e6},
    }
    ground_properties = {
        "measurements_path": _PATH,
        "output_dir": _PATH,
        "object_id": _ID,
        "initial_state_eci_km_s": _STATE6,
        "initial_state_source": {"type": "string", "maxLength": 160},
        "fit_duration_s": _NONNEGATIVE_SECONDS,
        "holdout_duration_s": _NONNEGATIVE_SECONDS,
        "dt_s": _POSITIVE_SECONDS,
        "dynamics_model": {"type": "string", "enum": ["two_body", "j2"]},
        "numeric_backend": {"type": "string", "enum": ["python", "rust"]},
        "exclude_station_ids": {"type": "array", "items": _ID, "maxItems": 64},
        "holdout_station_ids": {"type": "array", "items": _ID, "maxItems": 64},
        "scenario_name": {"type": "string", "minLength": 1, "maxLength": 160},
        **common_numeric,
    }
    ogp_properties = {
        "observations_path": _PATH,
        "output_dir": _PATH,
        "position_sigma_km": _SIGMA,
        "velocity_sigma_km_s": _SIGMA,
        "fit_until_jd_utc": {"type": ["number", "null"], "minimum": 0.0},
        "parameterization": {
            "type": "string",
            "enum": ["auto", "classical_delta", "equinoctial_nonsingular"],
        },
        "estimate_bstar": {"type": "boolean"},
        "bstar_prior_sigma": {"type": ["number", "null"], "exclusiveMinimum": 0.0, "maximum": 1.0e6},
        "estimate_epoch_offset": {"type": "boolean"},
        "epoch_offset_prior_sigma_s": {
            "type": ["number", "null"],
            "exclusiveMinimum": 0.0,
            "maximum": 604800.0,
        },
        "covariance_blocks": {
            "type": ["array", "null"],
            "maxItems": 2000,
            "items": {
                "type": "array",
                "minItems": 6,
                "maxItems": 6,
                "items": {"type": "array", "minItems": 6, "maxItems": 6, "items": {"type": "number"}},
            },
        },
        "propagation_backend": {"type": "string", "enum": ["scalar_reference", "accelerated"]},
        **common_numeric,
    }
    sequential_properties = {
        "observation_packet_path": _PATH,
        "output_dir": _PATH,
        "initial_state_eci_km_s": _STATE6,
        "initial_covariance": _COVARIANCE6,
        "estimator": {"type": "string", "enum": ["ekf", "ukf"]},
        "process_noise_model": {"type": "string", "enum": ["white_acceleration", "state_random_walk"]},
        "process_noise_spectral_density": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0e6},
            "minItems": 3,
            "maxItems": 6,
        },
        "innovation_gate_probability": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
        "integration_substep_s": _POSITIVE_SECONDS,
        "checkpoint_interval": {"type": "integer", "minimum": 1, "maximum": 1000},
    }
    relative_properties = {
        "observations_path": _PATH,
        "output_dir": _PATH,
        "dynamics_model": {"type": "string", "enum": ["hcw", "ss_j2", "th", "ya"]},
        "measurement_model": {
            "type": "string",
            "enum": [
                "relative_state",
                "relative_range",
                "relative_range_rate",
                "relative_angles",
                "relative_angles_range",
                "relative_angles_range_rate",
            ],
        },
        "measurement_origin": {"type": "string", "enum": ["chief", "deputy"]},
        "initial_guess_ric": _STATE6,
        "mean_motion_rad_s": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
        "chief_initial_state_eci_km_s": _STATE6,
        "fit_duration_s": _NONNEGATIVE_SECONDS,
        "holdout_duration_s": _NONNEGATIVE_SECONDS,
        "integration_substep_s": _POSITIVE_SECONDS,
        "transition_evaluation_method": {
            "type": "string",
            "enum": ["direct", "sequential_composition"],
        },
        "numeric_backend": {"type": "string", "enum": ["python", "rust"]},
        **common_numeric,
    }
    inspect_properties = {
        "path": _PATH,
        "max_file_bytes": {"type": "integer", "minimum": 1, "maximum": 5_000_000},
    }
    return (
        _contract(
            tool_id=OD_TOOL_IDS[0],
            title="Run bounded ground-sensor orbit determination",
            description="Fit a checked-in OEL numerical dynamics model to typed ground-station measurements.",
            api="sim.estimation.ground_station_od.solve_ground_station_measurement_od",
            properties=ground_properties,
            required=("measurements_path", "output_dir", "object_id"),
            result_schema=OD_RESULT_SCHEMA,
            writes=True,
            executes=True,
            handling_schema=handling,
            approval_schema=approval,
            property_builder=property_builder,
            deployment_profiles=deployment_profiles,
            entitlement=entitlement,
            limits_factory=limits_factory,
            limits={"max_observations": 2_000, "max_arc_duration_s": 604_800.0},
        ),
        _contract(
            tool_id=OD_TOOL_IDS[1],
            title="Run bounded OGP TEME PV orbit determination",
            description="Fit OGP-SGP4 or OGP-SDP4 mean elements to native TEME position and velocity observations.",
            api="sim.estimation.ogp_od.fit_ogp_mean_elements_to_observations",
            properties=ogp_properties,
            required=("observations_path", "output_dir"),
            result_schema=OD_RESULT_SCHEMA,
            writes=True,
            executes=True,
            handling_schema=handling,
            approval_schema=approval,
            property_builder=property_builder,
            deployment_profiles=deployment_profiles,
            entitlement=entitlement,
            limits_factory=limits_factory,
            limits={"max_observations": 2_000, "max_arc_duration_s": 604_800.0},
        ),
        _contract(
            tool_id=OD_TOOL_IDS[2],
            title="Run sequential EKF or UKF orbit determination",
            description="Run the deterministic OEL sequential ECI estimator and fixed-interval smoother.",
            api="sim.estimation.sequential_od.run_sequential_orbit_od",
            properties=sequential_properties,
            required=("observation_packet_path", "output_dir", "initial_state_eci_km_s", "initial_covariance"),
            result_schema=OD_RESULT_SCHEMA,
            writes=True,
            executes=True,
            handling_schema=handling,
            approval_schema=approval,
            property_builder=property_builder,
            deployment_profiles=deployment_profiles,
            entitlement=entitlement,
            limits_factory=limits_factory,
            limits={"max_observations": 500, "max_arc_duration_s": 604_800.0},
        ),
        _contract(
            tool_id=OD_TOOL_IDS[3],
            title="Run bounded relative orbit determination",
            description="Fit a batch HCW, SS-J2, TH, or YA relative state model with typed RIC measurements.",
            api="sim.estimation.relative_od.solve_batch_relative_od",
            properties=relative_properties,
            required=("observations_path", "output_dir", "dynamics_model", "measurement_model", "initial_guess_ric"),
            result_schema=OD_RESULT_SCHEMA,
            writes=True,
            executes=True,
            handling_schema=handling,
            approval_schema=approval,
            property_builder=property_builder,
            deployment_profiles=deployment_profiles,
            entitlement=entitlement,
            limits_factory=limits_factory,
            limits={"max_observations": 500, "max_arc_duration_s": 604_800.0},
        ),
        _contract(
            tool_id=OD_TOOL_IDS[4],
            title="Inspect an OD JSON artifact safely",
            description="Read bounded JSON metadata and counts without importing, executing, or exposing truth payloads.",
            api="integrations.oel_mcp.od_handlers.inspect_od_json",
            properties=inspect_properties,
            required=("path",),
            result_schema=OD_INSPECT_RESULT_SCHEMA,
            writes=False,
            executes=False,
            handling_schema=handling,
            approval_schema=approval,
            property_builder=property_builder,
            deployment_profiles=inspect_profiles,
            entitlement=entitlement,
            limits_factory=limits_factory,
            limits={"max_file_bytes": 5_000_000, "read_only": True},
        ),
    )


OD_TOOL_CONTRACTS = od_contracts()
OD_TOOL_CONTRACT_MAP = {contract.tool_id: contract for contract in OD_TOOL_CONTRACTS}

__all__ = [
    "OD_EXECUTION_TOOL_IDS",
    "OD_INSPECTION_TOOL_IDS",
    "OD_INSPECT_RESULT_SCHEMA",
    "OD_RESULT_SCHEMA",
    "OD_TOOL_CONTRACT_MAP",
    "OD_TOOL_CONTRACTS",
    "OD_TOOL_IDS",
    "od_contracts",
]
