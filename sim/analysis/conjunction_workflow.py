"""Public deterministic conjunction assessment and avoidance-candidate evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.analysis.conjunction_geometry import (
    StateHistory,
    encounter_frame,
    interpolate_history,
    refine_time_of_closest_approach,
)
from sim.analysis.conjunction_probability import (
    collision_probability_2d,
    covariance_rtn_si_to_eci_km,
    project_combined_covariance,
    validate_covariance,
)
from sim.analysis.trajectory_targeting import PropagationSettings, solve_trajectory_target
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin, j3_plugin, j4_plugin

CONJUNCTION_ASSESSMENT_PROBLEM_SCHEMA = "oel.conjunction_assessment_problem.v1"
CONJUNCTION_ASSESSMENT_EVIDENCE_SCHEMA = "oel.conjunction_assessment_evidence.v1"
_FORCE_PLUGINS = {"j2": j2_plugin, "j3": j3_plugin, "j4": j4_plugin}
_ZERO_ACCELERATION = np.zeros(3, dtype=float)
_PROBLEM_FIELDS = {
    "schema_version",
    "name",
    "duration_s",
    "propagation",
    "primary",
    "secondary",
    "screening_objects",
    "avoidance_candidates",
}
_OBJECT_FIELDS = {
    "object_id",
    "initial_state_eci_km_km_s",
    "covariance_eci_km_at_tca",
    "hard_body_radius_m",
}
_CANDIDATE_FIELDS = {
    "name",
    "burn_time_s",
    "frame",
    "burn_component",
    "terminal_quantity",
    "target_offset",
    "tolerance",
    "initial_delta_v_m_s",
    "perturbation_m_s",
    "max_abs_delta_v_m_s",
}
_PROPAGATION_FIELDS = {
    "step_s",
    "integrator",
    "force_model",
    "mu_km3_s2",
    "central_body_radius_km",
    "mass_kg",
    "event_time_tolerance_s",
    "event_value_tolerance",
    "event_max_iterations",
}
_PROPAGATION_NUMERIC_FIELDS = {
    "step_s",
    "mu_km3_s2",
    "central_body_radius_km",
    "mass_kg",
    "event_time_tolerance_s",
    "event_value_tolerance",
    "event_max_iterations",
}


class ConjunctionAssessmentError(ValueError):
    """Raised when a conjunction-assessment problem is invalid."""


def _mapping_with_known_fields(
    value: Any, *, field: str, allowed_fields: set[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConjunctionAssessmentError(f"{field} must be a mapping.")
    raw = dict(value)
    unknown = sorted(set(raw) - allowed_fields, key=str)
    if unknown:
        raise ConjunctionAssessmentError(
            f"{field} contains unknown fields: {', '.join(str(item) for item in unknown)}."
        )
    return raw


def _reject_boolean_values(value: Any, field: str) -> None:
    """Reject JSON/Python booleans before numeric conversion erases their type."""

    if isinstance(value, (bool, np.bool_)):
        raise ConjunctionAssessmentError(f"{field} must contain numeric values, not booleans.")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            raise ConjunctionAssessmentError(f"{field} must contain numeric values, not booleans.")
        if value.dtype.kind == "O":
            for item in value.flat:
                _reject_boolean_values(item, field)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_boolean_values(item, field)


def _physical_float(value: Any, field: str) -> float:
    _reject_boolean_values(value, field)
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConjunctionAssessmentError(f"{field} must be numeric.") from exc


@dataclass(frozen=True)
class ConjunctionObject:
    object_id: str
    initial_state_eci_km_km_s: tuple[float, float, float, float, float, float]
    covariance_eci_km_at_tca: tuple[tuple[float, ...], ...]
    hard_body_radius_m: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ConjunctionObject:
        raw = _mapping_with_known_fields(data, field="conjunction object", allowed_fields=_OBJECT_FIELDS)
        object_id = str(raw.get("object_id", "")).strip()
        raw_state = raw.get("initial_state_eci_km_km_s", [])
        raw_covariance = raw.get("covariance_eci_km_at_tca", [])
        raw_radius = raw.get("hard_body_radius_m", 0.0)
        _reject_boolean_values(raw_state, "initial_state_eci_km_km_s")
        _reject_boolean_values(raw_covariance, "covariance_eci_km_at_tca")
        _reject_boolean_values(raw_radius, "hard_body_radius_m")
        try:
            state = np.asarray(raw_state, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConjunctionAssessmentError(
                "initial_state_eci_km_km_s must contain numeric values."
            ) from exc
        covariance = validate_covariance(raw_covariance, dimension=6)
        radius = _physical_float(raw_radius, "hard_body_radius_m")
        if not object_id:
            raise ConjunctionAssessmentError("Every conjunction object requires object_id.")
        if state.shape != (6,) or not np.all(np.isfinite(state)) or float(np.linalg.norm(state[:3])) <= 0.0:
            raise ConjunctionAssessmentError(
                f"{object_id} initial state must contain six finite values and nonzero position."
            )
        if not math.isfinite(radius) or radius <= 0.0:
            raise ConjunctionAssessmentError(f"{object_id} hard_body_radius_m must be positive and finite.")
        return cls(
            object_id=object_id,
            initial_state_eci_km_km_s=tuple(float(value) for value in state),
            covariance_eci_km_at_tca=tuple(tuple(float(value) for value in row) for row in covariance),
            hard_body_radius_m=radius,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "initial_state_eci_km_km_s": list(self.initial_state_eci_km_km_s),
            "covariance_eci_km_at_tca": [list(row) for row in self.covariance_eci_km_at_tca],
            "hard_body_radius_m": self.hard_body_radius_m,
        }


@dataclass(frozen=True)
class AvoidanceCandidate:
    name: str
    burn_time_s: float
    frame: str
    burn_component: str
    terminal_quantity: str
    target_offset: float
    tolerance: float
    initial_delta_v_m_s: float
    perturbation_m_s: float
    max_abs_delta_v_m_s: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AvoidanceCandidate:
        raw = _mapping_with_known_fields(data, field="avoidance candidate", allowed_fields=_CANDIDATE_FIELDS)
        item = cls(
            name=str(raw.get("name", "")).strip(),
            burn_time_s=_physical_float(raw.get("burn_time_s", 0.0), "burn_time_s"),
            frame=str(raw.get("frame", "ric")).strip().lower(),
            burn_component=str(raw.get("burn_component", "i")).strip().lower(),
            terminal_quantity=str(raw.get("terminal_quantity", "position_x_km")).strip().lower(),
            target_offset=_physical_float(raw.get("target_offset", 0.0), "target_offset"),
            tolerance=_physical_float(raw.get("tolerance", 1.0e-5), "tolerance"),
            initial_delta_v_m_s=_physical_float(
                raw.get("initial_delta_v_m_s", 0.0), "initial_delta_v_m_s"
            ),
            perturbation_m_s=_physical_float(raw.get("perturbation_m_s", 1.0e-3), "perturbation_m_s"),
            max_abs_delta_v_m_s=_physical_float(
                raw.get("max_abs_delta_v_m_s", 10.0), "max_abs_delta_v_m_s"
            ),
        )
        allowed_components = {"eci": {"x", "y", "z"}, "ric": {"r", "i", "c"}}
        if not item.name:
            raise ConjunctionAssessmentError("Every avoidance candidate requires name.")
        if item.frame not in allowed_components or item.burn_component not in allowed_components[item.frame]:
            raise ConjunctionAssessmentError("Avoidance frame/component must be ECI x/y/z or RIC r/i/c.")
        if item.terminal_quantity not in {
            "position_x_km",
            "position_y_km",
            "position_z_km",
            "velocity_x_km_s",
            "velocity_y_km_s",
            "velocity_z_km_s",
        }:
            raise ConjunctionAssessmentError(
                "Avoidance terminal_quantity must be one Cartesian position or velocity component."
            )
        finite_positive = (item.tolerance, item.perturbation_m_s, item.max_abs_delta_v_m_s)
        if item.burn_time_s < 0.0 or not all(math.isfinite(value) and value > 0.0 for value in finite_positive):
            raise ConjunctionAssessmentError("Avoidance timing and solver magnitudes are invalid.")
        if not math.isfinite(item.target_offset) or not math.isfinite(item.initial_delta_v_m_s):
            raise ConjunctionAssessmentError("Avoidance candidate values must be finite.")
        return item


@dataclass(frozen=True)
class ConjunctionAssessmentProblem:
    name: str
    primary: ConjunctionObject
    secondary: ConjunctionObject
    screening_objects: tuple[ConjunctionObject, ...]
    duration_s: float
    propagation: PropagationSettings
    avoidance_candidates: tuple[AvoidanceCandidate, ...]
    schema_version: str = CONJUNCTION_ASSESSMENT_PROBLEM_SCHEMA

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ConjunctionAssessmentProblem:
        raw = _mapping_with_known_fields(data, field="conjunction assessment problem", allowed_fields=_PROBLEM_FIELDS)
        schema = str(raw.get("schema_version", CONJUNCTION_ASSESSMENT_PROBLEM_SCHEMA)).strip()
        if schema != CONJUNCTION_ASSESSMENT_PROBLEM_SCHEMA:
            raise ConjunctionAssessmentError(f"Unsupported schema {schema!r}.")
        duration = _physical_float(raw.get("duration_s", 0.0), "duration_s")
        propagation_raw = raw.get("propagation")
        if propagation_raw is None:
            propagation_raw = {}
        propagation_values = _mapping_with_known_fields(
            propagation_raw, field="propagation", allowed_fields=_PROPAGATION_FIELDS
        )
        for field in sorted(_PROPAGATION_NUMERIC_FIELDS & propagation_values.keys()):
            _reject_boolean_values(propagation_values[field], f"propagation.{field}")
        propagation = PropagationSettings.from_mapping(propagation_values)
        if not math.isfinite(duration) or duration <= propagation.step_s:
            raise ConjunctionAssessmentError("duration_s must be finite and greater than propagation.step_s.")
        problem = cls(
            schema_version=schema,
            name=str(raw.get("name", "conjunction_assessment")).strip() or "conjunction_assessment",
            primary=ConjunctionObject.from_mapping(raw.get("primary", {})),
            secondary=ConjunctionObject.from_mapping(raw.get("secondary", {})),
            screening_objects=tuple(ConjunctionObject.from_mapping(item) for item in raw.get("screening_objects", [])),
            duration_s=duration,
            propagation=propagation,
            avoidance_candidates=tuple(
                AvoidanceCandidate.from_mapping(item) for item in raw.get("avoidance_candidates", [])
            ),
        )
        identifiers = [
            problem.primary.object_id,
            problem.secondary.object_id,
            *(item.object_id for item in problem.screening_objects),
        ]
        if len(set(identifiers)) != len(identifiers):
            raise ConjunctionAssessmentError("Primary, secondary, and screening object identifiers must be unique.")
        return problem

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "duration_s": self.duration_s,
            "propagation": self.propagation.to_dict(),
            "primary": self.primary.to_dict(),
            "secondary": self.secondary.to_dict(),
            "screening_objects": [item.to_dict() for item in self.screening_objects],
            "avoidance_candidates": [item.__dict__ for item in self.avoidance_candidates],
        }


def _propagator(settings: PropagationSettings) -> OrbitPropagator:
    return OrbitPropagator(
        model="two_body",
        integrator=settings.integrator,
        plugins=[_FORCE_PLUGINS[name] for name in settings.force_model],
    )


def _ric_basis(state: np.ndarray) -> np.ndarray:
    """Return the RIC basis, rejecting a degenerate angular-momentum vector."""

    position = np.asarray(state[:3], dtype=float)
    velocity = np.asarray(state[3:], dtype=float)
    radius = float(np.linalg.norm(position))
    angular_momentum = np.cross(position, velocity)
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    if (
        not math.isfinite(radius)
        or radius <= 1.0e-12
        or not np.all(np.isfinite(angular_momentum))
        or not math.isfinite(angular_momentum_norm)
        or angular_momentum_norm <= 1.0e-12
    ):
        raise ConjunctionAssessmentError(
            "RIC burn frame requires finite position and nonzero angular momentum."
        )
    radial = position / radius
    cross_track = angular_momentum / angular_momentum_norm
    in_track = np.cross(cross_track, radial)
    return np.column_stack((radial, in_track, cross_track))


def propagate_history(
    initial_state: Sequence[float],
    duration_s: float,
    settings: PropagationSettings,
    *,
    burn_time_s: float | None = None,
    burn_frame: str = "eci",
    delta_v_m_s: Sequence[float] | None = None,
) -> StateHistory:
    """Propagate one authoritative OEL history, optionally with one impulse."""

    state = np.asarray(initial_state, dtype=float).copy()
    time_s = 0.0
    burn_applied = burn_time_s is None
    propagator = _propagator(settings)
    context = OrbitContext(mu_km3_s2=settings.mu_km3_s2, mass_kg=settings.mass_kg)
    burn_vector = np.asarray(delta_v_m_s if delta_v_m_s is not None else [0.0, 0.0, 0.0], dtype=float)
    if burn_time_s is not None and abs(float(burn_time_s)) <= 1.0e-12:
        if burn_frame == "ric":
            basis = _ric_basis(state)
        else:
            basis = np.eye(3)
        state[3:] += basis @ (burn_vector / 1000.0)
        burn_applied = True
    times = [0.0]
    states = [state.copy()]
    incoming_velocities = [state[3:].copy()]
    while time_s < duration_s:
        next_boundary = min(time_s + settings.step_s, duration_s)
        if not burn_applied and burn_time_s is not None and time_s < burn_time_s < next_boundary:
            next_boundary = float(burn_time_s)
        step = next_boundary - time_s
        state = propagator.propagate(state, step, time_s, _ZERO_ACCELERATION, {}, context)
        time_s = next_boundary
        incoming_velocity = state[3:].copy()
        if not burn_applied and burn_time_s is not None and abs(time_s - burn_time_s) <= 1.0e-10:
            if burn_frame == "ric":
                basis = _ric_basis(state)
            else:
                basis = np.eye(3)
            state = state.copy()
            state[3:] += basis @ (burn_vector / 1000.0)
            burn_applied = True
        times.append(time_s)
        states.append(state.copy())
        incoming_velocities.append(incoming_velocity)
    if not burn_applied:
        raise ConjunctionAssessmentError("burn_time_s must lie inside the propagation horizon.")
    return StateHistory.from_arrays(
        times,
        states,
        incoming_velocities_eci_km_s=incoming_velocities,
    )


def assess_histories(
    primary: ConjunctionObject,
    secondary: ConjunctionObject,
    primary_history: StateHistory,
    secondary_history: StateHistory,
    *,
    numeric_backend: str = "rust",
    probability_numeric_backend: str = "rust",
) -> dict[str, Any]:
    _require_probability_backend(probability_numeric_backend)
    closest = refine_time_of_closest_approach(primary_history, secondary_history, numeric_backend=numeric_backend)
    if closest["at_search_boundary"]:
        return {
            "primary_id": primary.object_id,
            "secondary_id": secondary.object_id,
            "status": "incomplete_search_window",
            "disposition": "closest_approach_at_search_boundary",
            "closest_approach": closest,
            "encounter_frame": None,
            "covariance_projection": None,
            "probability": None,
            "limitations": [
                "Encounter-plane covariance and Pc are withheld because the supplied window does not bracket TCA."
            ],
        }
    frame = encounter_frame(closest["relative_position_eci_km"], closest["relative_velocity_eci_km_s"])
    projected = project_combined_covariance(
        primary.covariance_eci_km_at_tca, secondary.covariance_eci_km_at_tca, frame["basis_rows_eci"]
    )
    probability = collision_probability_2d(
        frame["plane_miss_km"],
        projected["encounter_plane_covariance_km2"],
        (primary.hard_body_radius_m + secondary.hard_body_radius_m) / 1000.0,
        numeric_backend=probability_numeric_backend,
    )
    return {
        "primary_id": primary.object_id,
        "secondary_id": secondary.object_id,
        "status": "completed",
        "disposition": "interior_tca_assessed",
        "closest_approach": closest,
        "encounter_frame": frame,
        "covariance_projection": projected,
        "probability": probability,
    }


def assess_cdm_message(message_or_path: Any, *, primary_radius_m: float, secondary_radius_m: float,
                       numeric_backend: str = "rust") -> dict[str, Any]:
    """Independently recompute one CDM's instantaneous geometry and educational Pc."""

    from sim.interchange.ccsds_cdm import CdmMessage, inspect_cdm, read_cdm_kvn

    _require_probability_backend(numeric_backend)

    message = read_cdm_kvn(message_or_path) if not isinstance(message_or_path, CdmMessage) else message_or_path
    inspection = inspect_cdm(message)
    if not inspection["analysis_ready"]:
        raise ConjunctionAssessmentError(f"CDM is valid but not analysis-ready: {inspection['analysis_ready_issues']}")
    if (
        not math.isfinite(primary_radius_m)
        or not math.isfinite(secondary_radius_m)
        or primary_radius_m <= 0.0
        or secondary_radius_m <= 0.0
    ):
        raise ConjunctionAssessmentError("CDM assessment requires positive finite primary and secondary radii.")
    primary_state = np.asarray(message.objects[0].state_eci_km_km_s, dtype=float)
    secondary_state = np.asarray(message.objects[1].state_eci_km_km_s, dtype=float)
    relative = primary_state - secondary_state
    frame = encounter_frame(relative[:3], relative[3:])
    primary_covariance = covariance_rtn_si_to_eci_km(message.objects[0].covariance_rtn_si, primary_state)
    secondary_covariance = covariance_rtn_si_to_eci_km(message.objects[1].covariance_rtn_si, secondary_state)
    projected = project_combined_covariance(primary_covariance, secondary_covariance, frame["basis_rows_eci"])
    probability = collision_probability_2d(
        frame["plane_miss_km"],
        projected["encounter_plane_covariance_km2"],
        (float(primary_radius_m) + float(secondary_radius_m)) / 1000.0,
        numeric_backend=numeric_backend,
    )
    computed_miss_m = float(np.linalg.norm(relative[:3])) * 1000.0
    computed_speed_m_s = float(np.linalg.norm(relative[3:])) * 1000.0
    reported_pc = message.relative.collision_probability
    return {
        "schema_version": "oel.cdm_assessment_evidence.v1",
        "status": "completed",
        "inspection": inspection,
        "computed": {
            "miss_distance_m": computed_miss_m,
            "relative_speed_m_s": computed_speed_m_s,
            "encounter_frame": frame,
            "covariance_projection": projected,
            "probability": probability,
        },
        "reported_minus_computed": {
            "miss_distance_m": message.relative.miss_distance_m - computed_miss_m,
            "relative_speed_m_s": message.relative.relative_speed_m_s - computed_speed_m_s,
            "collision_probability": None
            if reported_pc is None
            else reported_pc - probability["collision_probability"],
        },
        "limitations": [
            "This recomputes instantaneous TCA geometry from the two CDM states; it does not refine TCA without ephemeris histories.",
            "The supplied hard-body radii are analyst inputs because the bounded CDM profile does not infer radii from AREA_PC.",
            "The educational probability comparison does not validate or replace an originating agency's operational process.",
        ],
    }


def _cartesian_quantity(state: Sequence[float], quantity: str) -> float:
    index = {
        "position_x_km": 0,
        "position_y_km": 1,
        "position_z_km": 2,
        "velocity_x_km_s": 3,
        "velocity_y_km_s": 4,
        "velocity_z_km_s": 5,
    }[quantity]
    return float(state[index])


def _targeter_problem(
    problem: ConjunctionAssessmentProblem, candidate: AvoidanceCandidate, baseline: Mapping[str, Any]
) -> dict[str, Any]:
    tca = float(baseline["closest_approach"]["time_s"])
    if not candidate.burn_time_s < tca:
        raise ConjunctionAssessmentError(f"Candidate {candidate.name!r} burn_time_s must precede baseline TCA.")
    segments: list[dict[str, Any]] = []
    if candidate.burn_time_s > 0.0:
        segments.append({"name": "coast_to_burn", "type": "coast", "duration_s": candidate.burn_time_s})
    components = [0.0, 0.0, 0.0]
    component_index = {"x": 0, "y": 1, "z": 2, "r": 0, "i": 1, "c": 2}[candidate.burn_component]
    components[component_index] = candidate.initial_delta_v_m_s
    segments.append(
        {"name": "avoidance_burn", "type": "impulsive_burn", "frame": candidate.frame, "delta_v_m_s": components}
    )
    segments.append({"name": "coast_to_baseline_tca", "type": "coast", "duration_s": tca - candidate.burn_time_s})
    field = f"delta_v_{candidate.burn_component}_m_s"
    target_state = baseline["closest_approach"]["primary_state_eci_km_km_s"]
    return {
        "schema_version": "oel.trajectory_targeting_problem.v1",
        "name": f"{problem.name}:{candidate.name}",
        "initial_state_eci_km_km_s": list(problem.primary.initial_state_eci_km_km_s),
        "propagation": problem.propagation.to_dict(),
        "segments": segments,
        "variables": [
            {
                "name": "avoidance_delta_v",
                "segment": "avoidance_burn",
                "field": field,
                "initial": candidate.initial_delta_v_m_s,
                "perturbation": candidate.perturbation_m_s,
            }
        ],
        "constraints": [
            {
                "name": "terminal_offset",
                "quantity": candidate.terminal_quantity,
                "target": _cartesian_quantity(target_state, candidate.terminal_quantity) + candidate.target_offset,
                "tolerance": candidate.tolerance,
            }
        ],
        "solver": {"max_iterations": 12, "correction_limit": candidate.max_abs_delta_v_m_s},
    }


def _require_probability_backend(backend: str) -> None:
    backend = str(backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ConjunctionAssessmentError("probability numeric backend must be python or rust.")
    if backend == "rust":
        from sim.rust_conjunction_probability_backend import integrand

        integrand()


def assess_conjunction(problem: ConjunctionAssessmentProblem | Mapping[str, Any], *,
                       numeric_backend: str = "rust", probability_numeric_backend: str = "rust") -> dict[str, Any]:
    _require_probability_backend(probability_numeric_backend)
    numeric_backend = str(numeric_backend).strip().lower()
    if numeric_backend not in {"python", "rust"}:
        raise ConjunctionAssessmentError("numeric_backend must be 'python' or 'rust'.")
    if numeric_backend == "rust":
        from sim.rust_conjunction_backend import _kernels

        _kernels()
    parsed = ConjunctionAssessmentProblem.from_mapping(
        problem.to_dict() if isinstance(problem, ConjunctionAssessmentProblem) else problem
    )
    primary_history = propagate_history(parsed.primary.initial_state_eci_km_km_s, parsed.duration_s, parsed.propagation)
    secondary_history = propagate_history(
        parsed.secondary.initial_state_eci_km_km_s, parsed.duration_s, parsed.propagation
    )
    screening_histories = {
        item.object_id: propagate_history(item.initial_state_eci_km_km_s, parsed.duration_s, parsed.propagation)
        for item in parsed.screening_objects
    }
    baseline = assess_histories(parsed.primary, parsed.secondary, primary_history, secondary_history, numeric_backend=numeric_backend, probability_numeric_backend=probability_numeric_backend)
    if baseline["status"] != "completed":
        canonical_problem = parsed.to_dict()
        return {
            "schema_version": CONJUNCTION_ASSESSMENT_EVIDENCE_SCHEMA,
            "problem_name": parsed.name,
            "problem_sha256": hashlib.sha256(
                json.dumps(canonical_problem, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest(),
            "status": "incomplete_search_window",
            "baseline": baseline,
            "avoidance_candidates": [
                {
                    "name": candidate.name,
                    "assessment_completed": False,
                    "risk_disposition": "not_evaluated_no_acceptance_criteria",
                    "disposition": "baseline_tca_incomplete",
                }
                for candidate in parsed.avoidance_candidates
            ],
            "resources": {
                "primary_samples": len(primary_history.times_s),
                "secondary_samples": len(secondary_history.times_s),
                "screening_object_count": len(parsed.screening_objects),
                "candidate_count": len(parsed.avoidance_candidates),
            },
            "limitations": [
                "The search window did not bracket the baseline TCA, so encounter-plane Pc and avoidance assessment were withheld.",
                "This deterministic public workflow screens one primary-secondary pair plus an explicit small secondary list; it is not catalog-scale screening.",
                "Input covariances are declared at TCA and are not propagated or estimated by this workflow.",
                "The educational 2D Pc and deterministic repropagation are analysis evidence, not an operational collision-avoidance recommendation.",
            ],
        }
    candidates: list[dict[str, Any]] = []
    for candidate in parsed.avoidance_candidates:
        try:
            targeter_problem = _targeter_problem(parsed, candidate, baseline)
            targeter = solve_trajectory_target(targeter_problem)
            row: dict[str, Any] = {
                "name": candidate.name,
                "targeter": targeter,
                "assessment_completed": False,
                "risk_disposition": "not_evaluated_no_acceptance_criteria",
            }
            if not targeter.get("converged"):
                row["disposition"] = "targeter_not_converged"
                candidates.append(row)
                continue
            delta_v = float(targeter["decision_values"][0])
            if abs(delta_v) > candidate.max_abs_delta_v_m_s:
                row["disposition"] = "delta_v_limit_exceeded"
                candidates.append(row)
                continue
            burn_vector = np.zeros(3)
            burn_vector[{"x": 0, "y": 1, "z": 2, "r": 0, "i": 1, "c": 2}[candidate.burn_component]] = delta_v
            maneuvered_history = propagate_history(
                parsed.primary.initial_state_eci_km_km_s,
                parsed.duration_s,
                parsed.propagation,
                burn_time_s=candidate.burn_time_s,
                burn_frame=candidate.frame,
                delta_v_m_s=burn_vector,
            )
            assessment = assess_histories(parsed.primary, parsed.secondary, maneuvered_history, secondary_history, numeric_backend=numeric_backend, probability_numeric_backend=probability_numeric_backend)
            targeter_final = np.asarray(
                targeter["authoritative_repropagation"]["execution"]["final_state_eci_km_km_s"], dtype=float
            )
            # The candidate TCA can move, so continuity is checked against an independent
            # history interpolation at the original targeter's terminal epoch below.
            history_at_target_epoch = interpolate_history(
                maneuvered_history, float(baseline["closest_approach"]["time_s"])
            )
            continuity_error = float(np.max(np.abs(history_at_target_epoch - targeter_final)))
            rescreen = [
                assess_histories(parsed.primary, item, maneuvered_history, screening_histories[item.object_id], numeric_backend=numeric_backend, probability_numeric_backend=probability_numeric_backend)
                for item in parsed.screening_objects
            ]
            row.update(
                {
                    "assessment_completed": True,
                    "disposition": "repropagated_and_rescreened",
                    "delta_v_m_s": delta_v,
                    "assessment": assessment,
                    "secondary_rescreen": rescreen,
                    "authoritative_history_continuity": {
                        "target_epoch_s": float(baseline["closest_approach"]["time_s"]),
                        "max_abs_state_difference": continuity_error,
                        "acceptance_tolerance": 1.0e-6,
                        "units": "mixed km and km/s",
                    },
                }
            )
            if continuity_error > 1.0e-6:
                row["assessment_completed"] = False
                row["disposition"] = "authoritative_history_mismatch"
            candidates.append(row)
        except (ValueError, RuntimeError) as exc:
            candidates.append(
                {
                    "name": candidate.name,
                    "assessment_completed": False,
                    "risk_disposition": "not_evaluated_no_acceptance_criteria",
                    "disposition": "invalid_candidate",
                    "message": str(exc),
                }
            )
    canonical_problem = parsed.to_dict()
    return {
        "schema_version": CONJUNCTION_ASSESSMENT_EVIDENCE_SCHEMA,
        "problem_name": parsed.name,
        "problem_sha256": hashlib.sha256(
            json.dumps(canonical_problem, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "status": "completed",
        "baseline": baseline,
        "avoidance_candidates": candidates,
        "resources": {
            "primary_samples": len(primary_history.times_s),
            "secondary_samples": len(secondary_history.times_s),
            "screening_object_count": len(parsed.screening_objects),
            "candidate_count": len(parsed.avoidance_candidates),
        },
        "limitations": [
            "This deterministic public workflow screens one primary-secondary pair plus an explicit small secondary list; it is not catalog-scale screening.",
            "Input covariances are declared at TCA and are not propagated or estimated by this workflow.",
            "Avoidance candidates solve one equality constraint with one impulsive-burn component; they are not globally optimized.",
            "The educational 2D Pc and deterministic repropagation are analysis evidence, not an operational collision-avoidance recommendation.",
        ],
    }


def write_conjunction_evidence(evidence: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(evidence), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return target


__all__ = [
    "CONJUNCTION_ASSESSMENT_EVIDENCE_SCHEMA",
    "CONJUNCTION_ASSESSMENT_PROBLEM_SCHEMA",
    "AvoidanceCandidate",
    "ConjunctionAssessmentError",
    "ConjunctionAssessmentProblem",
    "ConjunctionObject",
    "assess_conjunction",
    "assess_cdm_message",
    "assess_histories",
    "propagate_history",
    "write_conjunction_evidence",
]
