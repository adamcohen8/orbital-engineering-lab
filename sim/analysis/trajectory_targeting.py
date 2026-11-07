"""Deterministic event-driven trajectory propagation and single shooting.

This module deliberately provides one inspectable public targeting primitive.
It reuses :class:`sim.dynamics.orbit.propagator.OrbitPropagator` for every
trajectory evaluation; it does not carry a second or approximate dynamics
implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin, j3_plugin, j4_plugin

TRAJECTORY_TARGETING_PROBLEM_SCHEMA = "oel.trajectory_targeting_problem.v1"
TRAJECTORY_TARGETING_EVIDENCE_SCHEMA = "oel.trajectory_targeting_evidence.v1"

_ZERO_ACCELERATION = np.zeros(3, dtype=float)
_FORCE_PLUGINS = {"j2": j2_plugin, "j3": j3_plugin, "j4": j4_plugin}
_STATE_QUANTITIES = {
    "position_x_km",
    "position_y_km",
    "position_z_km",
    "velocity_x_km_s",
    "velocity_y_km_s",
    "velocity_z_km_s",
    "radius_km",
    "altitude_km",
    "speed_km_s",
    "radial_velocity_km_s",
    "semi_major_axis_km",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "argument_of_periapsis_deg",
    "true_anomaly_deg",
    "elapsed_time_s",
}
_ANGULAR_QUANTITIES = {
    "inclination_deg",
    "raan_deg",
    "argument_of_periapsis_deg",
    "true_anomaly_deg",
}
_TARGETING_PROBLEM_FIELDS = {
    "schema_version",
    "name",
    "initial_state_eci_km_km_s",
    "propagation",
    "segments",
    "variables",
    "constraints",
    "solver",
}
_TARGETING_PROPAGATION_FIELDS = {
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
_TARGETING_VARIABLE_FIELDS = {"name", "segment", "field", "initial", "perturbation"}
_TARGETING_CONSTRAINT_FIELDS = {"name", "quantity", "target", "tolerance"}
_TARGETING_SOLVER_FIELDS = {
    "max_iterations",
    "rank_rcond",
    "minimum_line_search_scale",
    "correction_limit",
}
_TARGETING_IMPULSIVE_SEGMENT_FIELDS = {"type", "name", "frame", "delta_v_m_s"}
_TARGETING_COAST_SEGMENT_FIELDS = {"type", "name", "duration_s", "stop"}
_TARGETING_STOP_FIELDS = {"quantity", "target", "direction", "max_duration_s", "minimum_elapsed_s"}


class TrajectoryTargetingError(ValueError):
    """Raised when a targeting problem or execution is invalid."""


class MissedEventError(TrajectoryTargetingError):
    """Raised when an event coast reaches its declared search horizon."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.receipt = dict(receipt or {})


class EventRefinementError(TrajectoryTargetingError):
    """Raised when a bracketed event cannot be refined to a declared tolerance."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.receipt = dict(receipt or {})


def _mapping_with_known_fields(
    value: Any, *, field: str, allowed_fields: set[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrajectoryTargetingError(f"{field} must be a mapping.")
    raw = dict(value)
    unknown = sorted(set(raw) - allowed_fields, key=str)
    if unknown:
        raise TrajectoryTargetingError(
            f"{field} contains unknown fields: {', '.join(str(item) for item in unknown)}."
        )
    return raw


def _reject_boolean_values(value: Any, field: str) -> None:
    """Reject booleans before Python/NumPy numeric conversion erases their type."""

    if isinstance(value, (bool, np.bool_)):
        raise TrajectoryTargetingError(f"{field} must not contain boolean values.")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            raise TrajectoryTargetingError(f"{field} must not contain boolean values.")
        if value.dtype.kind == "O":
            for item in value.flat:
                _reject_boolean_values(item, field)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_boolean_values(item, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_boolean_values(item, field)


def _reject_boolean_fields(raw: Mapping[str, Any], field: str) -> None:
    for key, value in raw.items():
        _reject_boolean_values(value, f"{field}.{key}")


@dataclass(frozen=True)
class PropagationSettings:
    step_s: float = 10.0
    integrator: str = "rk4"
    force_model: tuple[str, ...] = ()
    mu_km3_s2: float = EARTH_MU_KM3_S2
    central_body_radius_km: float = EARTH_RADIUS_KM
    mass_kg: float = 100.0
    event_time_tolerance_s: float = 1.0e-6
    event_value_tolerance: float = 1.0e-10
    event_max_iterations: int = 80

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> PropagationSettings:
        raw = _mapping_with_known_fields(
            {} if data is None else data,
            field="propagation",
            allowed_fields=_TARGETING_PROPAGATION_FIELDS,
        )
        _reject_boolean_fields(raw, "propagation")
        settings = cls(
            step_s=float(raw.get("step_s", 10.0)),
            integrator=str(raw.get("integrator", "rk4")).strip().lower(),
            force_model=tuple(str(item).strip().lower() for item in raw.get("force_model", [])),
            mu_km3_s2=float(raw.get("mu_km3_s2", EARTH_MU_KM3_S2)),
            central_body_radius_km=float(raw.get("central_body_radius_km", EARTH_RADIUS_KM)),
            mass_kg=float(raw.get("mass_kg", 100.0)),
            event_time_tolerance_s=float(raw.get("event_time_tolerance_s", 1.0e-6)),
            event_value_tolerance=float(raw.get("event_value_tolerance", 1.0e-10)),
            event_max_iterations=int(raw.get("event_max_iterations", 80)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not math.isfinite(self.step_s) or self.step_s <= 0.0:
            raise TrajectoryTargetingError("propagation.step_s must be positive and finite.")
        if self.integrator not in {"rk4", "rkf78", "dopri5", "adaptive"}:
            raise TrajectoryTargetingError("propagation.integrator must be rk4, rkf78, dopri5, or adaptive.")
        unsupported = sorted(set(self.force_model) - set(_FORCE_PLUGINS))
        if unsupported:
            raise TrajectoryTargetingError(
                f"Unsupported public targeting force_model entries: {unsupported}; supported: {sorted(_FORCE_PLUGINS)}."
            )
        if (
            not math.isfinite(self.mu_km3_s2)
            or not math.isfinite(self.central_body_radius_km)
            or not math.isfinite(self.mass_kg)
            or self.mu_km3_s2 <= 0.0
            or self.central_body_radius_km <= 0.0
            or self.mass_kg <= 0.0
        ):
            raise TrajectoryTargetingError(
                "Propagation mu, central-body radius, and mass must be positive and finite."
            )
        if (
            not math.isfinite(self.event_time_tolerance_s)
            or not math.isfinite(self.event_value_tolerance)
            or self.event_time_tolerance_s <= 0.0
            or self.event_value_tolerance <= 0.0
        ):
            raise TrajectoryTargetingError("Event tolerances must be positive and finite.")
        if self.event_max_iterations <= 0:
            raise TrajectoryTargetingError("event_max_iterations must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_s": self.step_s,
            "integrator": self.integrator,
            "force_model": list(self.force_model),
            "mu_km3_s2": self.mu_km3_s2,
            "central_body_radius_km": self.central_body_radius_km,
            "mass_kg": self.mass_kg,
            "event_time_tolerance_s": self.event_time_tolerance_s,
            "event_value_tolerance": self.event_value_tolerance,
            "event_max_iterations": self.event_max_iterations,
        }


@dataclass(frozen=True)
class DecisionVariable:
    name: str
    segment: str
    field: str
    initial: float
    perturbation: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DecisionVariable:
        raw = _mapping_with_known_fields(
            data,
            field="decision variable",
            allowed_fields=_TARGETING_VARIABLE_FIELDS,
        )
        _reject_boolean_fields(raw, "decision variable")
        variable = cls(
            name=str(raw.get("name", "")).strip(),
            segment=str(raw.get("segment", "")).strip(),
            field=str(raw.get("field", "")).strip().lower(),
            initial=float(raw.get("initial", 0.0)),
            perturbation=float(raw.get("perturbation", 0.0)),
        )
        if not variable.name or not variable.segment or not variable.field:
            raise TrajectoryTargetingError("Every variable requires non-empty name, segment, and field values.")
        if not math.isfinite(variable.initial):
            raise TrajectoryTargetingError(f"Variable {variable.name!r} initial value must be finite.")
        if not math.isfinite(variable.perturbation) or variable.perturbation <= 0.0:
            raise TrajectoryTargetingError(f"Variable {variable.name!r} perturbation must be positive and finite.")
        return variable

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "segment": self.segment,
            "field": self.field,
            "initial": self.initial,
            "perturbation": self.perturbation,
        }


@dataclass(frozen=True)
class TerminalConstraint:
    name: str
    quantity: str
    target: float
    tolerance: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TerminalConstraint:
        raw = _mapping_with_known_fields(
            data,
            field="terminal constraint",
            allowed_fields=_TARGETING_CONSTRAINT_FIELDS,
        )
        _reject_boolean_fields(raw, "terminal constraint")
        constraint = cls(
            name=str(raw.get("name", "")).strip(),
            quantity=str(raw.get("quantity", "")).strip().lower(),
            target=float(raw.get("target", 0.0)),
            tolerance=float(raw.get("tolerance", 0.0)),
        )
        if not constraint.name:
            raise TrajectoryTargetingError("Every terminal constraint requires a non-empty name.")
        if constraint.quantity not in _STATE_QUANTITIES:
            raise TrajectoryTargetingError(
                f"Unsupported terminal quantity {constraint.quantity!r}; supported: {sorted(_STATE_QUANTITIES)}."
            )
        if not math.isfinite(constraint.target):
            raise TrajectoryTargetingError(f"Constraint {constraint.name!r} target must be finite.")
        if not math.isfinite(constraint.tolerance) or constraint.tolerance <= 0.0:
            raise TrajectoryTargetingError(f"Constraint {constraint.name!r} tolerance must be positive and finite.")
        return constraint

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quantity": self.quantity,
            "target": self.target,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class SolverSettings:
    max_iterations: int = 12
    rank_rcond: float = 1.0e-10
    minimum_line_search_scale: float = 1.0 / 128.0
    correction_limit: float = 1000.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> SolverSettings:
        raw = _mapping_with_known_fields(
            {} if data is None else data,
            field="solver",
            allowed_fields=_TARGETING_SOLVER_FIELDS,
        )
        _reject_boolean_fields(raw, "solver")
        settings = cls(
            max_iterations=int(raw.get("max_iterations", 12)),
            rank_rcond=float(raw.get("rank_rcond", 1.0e-10)),
            minimum_line_search_scale=float(raw.get("minimum_line_search_scale", 1.0 / 128.0)),
            correction_limit=float(raw.get("correction_limit", 1000.0)),
        )
        if settings.max_iterations <= 0:
            raise TrajectoryTargetingError("solver.max_iterations must be positive.")
        if not 0.0 < settings.rank_rcond < 1.0:
            raise TrajectoryTargetingError("solver.rank_rcond must lie between zero and one.")
        if not 0.0 < settings.minimum_line_search_scale <= 1.0:
            raise TrajectoryTargetingError("solver.minimum_line_search_scale must lie in (0, 1].")
        if not math.isfinite(settings.correction_limit) or settings.correction_limit <= 0.0:
            raise TrajectoryTargetingError("solver.correction_limit must be positive and finite.")
        return settings

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "rank_rcond": self.rank_rcond,
            "minimum_line_search_scale": self.minimum_line_search_scale,
            "correction_limit": self.correction_limit,
        }


@dataclass(frozen=True)
class TrajectoryTargetingProblem:
    initial_state_eci_km_km_s: tuple[float, float, float, float, float, float]
    segments: tuple[dict[str, Any], ...]
    variables: tuple[DecisionVariable, ...]
    constraints: tuple[TerminalConstraint, ...]
    propagation: PropagationSettings
    solver: SolverSettings
    schema_version: str = TRAJECTORY_TARGETING_PROBLEM_SCHEMA
    name: str = "trajectory_targeting_problem"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrajectoryTargetingProblem:
        raw = _mapping_with_known_fields(
            data,
            field="trajectory targeting problem",
            allowed_fields=_TARGETING_PROBLEM_FIELDS,
        )
        _reject_boolean_fields(raw, "trajectory targeting problem")
        schema = str(raw.get("schema_version", TRAJECTORY_TARGETING_PROBLEM_SCHEMA)).strip()
        if schema != TRAJECTORY_TARGETING_PROBLEM_SCHEMA:
            raise TrajectoryTargetingError(
                f"Unsupported trajectory-targeting schema {schema!r}; expected {TRAJECTORY_TARGETING_PROBLEM_SCHEMA!r}."
            )
        state = np.asarray(raw.get("initial_state_eci_km_km_s", []), dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)) or float(np.linalg.norm(state[:3])) <= 0.0:
            raise TrajectoryTargetingError(
                "initial_state_eci_km_km_s must contain six finite values and nonzero position."
            )
        segments_raw = raw.get("segments", [])
        if not isinstance(segments_raw, list) or not segments_raw:
            raise TrajectoryTargetingError("segments must be a non-empty list.")
        segments = tuple(_validate_segment(item) for item in segments_raw)
        variables = tuple(DecisionVariable.from_mapping(item) for item in raw.get("variables", []))
        constraints = tuple(TerminalConstraint.from_mapping(item) for item in raw.get("constraints", []))
        if not constraints:
            raise TrajectoryTargetingError("constraints must contain at least one terminal constraint.")
        problem = cls(
            schema_version=schema,
            name=str(raw.get("name", "trajectory_targeting_problem")).strip() or "trajectory_targeting_problem",
            initial_state_eci_km_km_s=tuple(float(value) for value in state),
            segments=segments,
            variables=variables,
            constraints=constraints,
            propagation=PropagationSettings.from_mapping(raw.get("propagation")),
            solver=SolverSettings.from_mapping(raw.get("solver")),
        )
        _validate_problem_links(problem)
        return problem

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "initial_state_eci_km_km_s": list(self.initial_state_eci_km_km_s),
            "propagation": self.propagation.to_dict(),
            "segments": [dict(segment) for segment in self.segments],
            "variables": [variable.to_dict() for variable in self.variables],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "solver": self.solver.to_dict(),
        }


def _validate_segment(data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TrajectoryTargetingError("Every segment must be a JSON object.")
    segment_type = str(data.get("type", "")).strip().lower()
    if segment_type == "impulsive_burn":
        raw = _mapping_with_known_fields(
            data,
            field="impulsive burn segment",
            allowed_fields=_TARGETING_IMPULSIVE_SEGMENT_FIELDS,
        )
    elif segment_type == "coast":
        raw = _mapping_with_known_fields(
            data,
            field="coast segment",
            allowed_fields=_TARGETING_COAST_SEGMENT_FIELDS,
        )
    else:
        raw = _mapping_with_known_fields(
            data,
            field="segment",
            allowed_fields=_TARGETING_IMPULSIVE_SEGMENT_FIELDS | _TARGETING_COAST_SEGMENT_FIELDS,
        )
    _reject_boolean_fields(raw, f"{segment_type or 'segment'} segment")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise TrajectoryTargetingError("Every segment requires a non-empty name.")
    if segment_type == "impulsive_burn":
        frame = str(raw.get("frame", "eci")).strip().lower()
        if frame not in {"eci", "ric"}:
            raise TrajectoryTargetingError(f"Burn segment {name!r} frame must be 'eci' or 'ric'.")
        delta_v = np.asarray(raw.get("delta_v_m_s", [0.0, 0.0, 0.0]), dtype=float)
        if delta_v.shape != (3,) or not np.all(np.isfinite(delta_v)):
            raise TrajectoryTargetingError(f"Burn segment {name!r} delta_v_m_s must contain three finite values.")
        return {"type": segment_type, "name": name, "frame": frame, "delta_v_m_s": delta_v.tolist()}
    if segment_type == "coast":
        has_duration = raw.get("duration_s") is not None
        has_stop = raw.get("stop") is not None
        if has_duration == has_stop:
            raise TrajectoryTargetingError(f"Coast segment {name!r} must declare exactly one of duration_s or stop.")
        if has_duration:
            duration = float(raw["duration_s"])
            if not math.isfinite(duration) or duration <= 0.0:
                raise TrajectoryTargetingError(f"Coast segment {name!r} duration_s must be positive and finite.")
            return {"type": segment_type, "name": name, "duration_s": duration}
        stop = _validate_stop_event(raw["stop"], segment_name=name)
        return {"type": segment_type, "name": name, "stop": stop}
    raise TrajectoryTargetingError(f"Unsupported segment type {segment_type!r}; expected coast or impulsive_burn.")


def _validate_stop_event(data: Mapping[str, Any], *, segment_name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TrajectoryTargetingError(f"Coast segment {segment_name!r} stop must be a JSON object.")
    raw = _mapping_with_known_fields(
        data,
        field=f"Coast segment {segment_name!r} stop",
        allowed_fields=_TARGETING_STOP_FIELDS,
    )
    _reject_boolean_fields(raw, f"Coast segment {segment_name!r} stop")
    quantity = str(raw.get("quantity", "")).strip().lower()
    if quantity not in _STATE_QUANTITIES:
        raise TrajectoryTargetingError(f"Coast segment {segment_name!r} has unsupported stop quantity {quantity!r}.")
    direction = str(raw.get("direction", "any")).strip().lower()
    if direction not in {"any", "increasing", "decreasing"}:
        raise TrajectoryTargetingError(
            f"Coast segment {segment_name!r} stop direction must be any, increasing, or decreasing."
        )
    target = float(raw.get("target", 0.0))
    max_duration = float(raw.get("max_duration_s", 0.0))
    minimum_elapsed = float(raw.get("minimum_elapsed_s", 0.0))
    if not math.isfinite(target):
        raise TrajectoryTargetingError(f"Coast segment {segment_name!r} stop target must be finite.")
    if not math.isfinite(max_duration) or max_duration <= 0.0:
        raise TrajectoryTargetingError(
            f"Coast segment {segment_name!r} stop.max_duration_s must be positive and finite."
        )
    if not math.isfinite(minimum_elapsed) or minimum_elapsed < 0.0 or minimum_elapsed >= max_duration:
        raise TrajectoryTargetingError(
            f"Coast segment {segment_name!r} stop.minimum_elapsed_s must be finite and inside the search horizon."
        )
    return {
        "quantity": quantity,
        "target": target,
        "direction": direction,
        "max_duration_s": max_duration,
        "minimum_elapsed_s": minimum_elapsed,
    }


def _validate_problem_links(problem: TrajectoryTargetingProblem) -> None:
    segment_by_name = {segment["name"]: segment for segment in problem.segments}
    if len(segment_by_name) != len(problem.segments):
        raise TrajectoryTargetingError("Segment names must be unique.")
    variable_names: set[str] = set()
    variable_targets: set[tuple[str, str]] = set()
    for variable in problem.variables:
        if variable.name in variable_names:
            raise TrajectoryTargetingError(f"Variable name {variable.name!r} is duplicated.")
        variable_names.add(variable.name)
        if variable.segment not in segment_by_name:
            raise TrajectoryTargetingError(
                f"Variable {variable.name!r} refers to unknown segment {variable.segment!r}."
            )
        segment = segment_by_name[variable.segment]
        allowed = _variable_fields_for_segment(segment)
        if variable.field not in allowed:
            raise TrajectoryTargetingError(
                f"Variable {variable.name!r} field {variable.field!r} is invalid for segment {variable.segment!r}; "
                f"supported: {sorted(allowed)}."
            )
        target = (variable.segment, variable.field)
        if target in variable_targets:
            raise TrajectoryTargetingError(f"Multiple variables target the same segment field {target}.")
        variable_targets.add(target)
    constraint_names = [constraint.name for constraint in problem.constraints]
    if len(set(constraint_names)) != len(constraint_names):
        raise TrajectoryTargetingError("Terminal constraint names must be unique.")
    if len(problem.variables) > len(problem.constraints):
        raise TrajectoryTargetingError(
            "The public single-shooting targeter requires at least as many constraints as decision variables."
        )


def _variable_fields_for_segment(segment: Mapping[str, Any]) -> set[str]:
    if segment["type"] == "coast" and "duration_s" in segment:
        return {"duration_s"}
    if segment["type"] == "impulsive_burn":
        if segment["frame"] == "eci":
            return {"delta_v_x_m_s", "delta_v_y_m_s", "delta_v_z_m_s"}
        return {"delta_v_r_m_s", "delta_v_i_m_s", "delta_v_c_m_s"}
    return set()


def _propagator(settings: PropagationSettings, *, numeric_backend: str) -> OrbitPropagator:
    return OrbitPropagator(
        model="two_body",
        integrator=settings.integrator,
        plugins=[_FORCE_PLUGINS[name] for name in settings.force_model],
        numeric_backend=numeric_backend,
    )


def _quantity_value(
    quantity: str,
    state: np.ndarray,
    elapsed_time_s: float,
    settings: PropagationSettings,
) -> float:
    r = np.asarray(state[:3], dtype=float)
    v = np.asarray(state[3:], dtype=float)
    r_norm = float(np.linalg.norm(r))
    if quantity == "position_x_km":
        return float(r[0])
    if quantity == "position_y_km":
        return float(r[1])
    if quantity == "position_z_km":
        return float(r[2])
    if quantity == "velocity_x_km_s":
        return float(v[0])
    if quantity == "velocity_y_km_s":
        return float(v[1])
    if quantity == "velocity_z_km_s":
        return float(v[2])
    if quantity == "radius_km":
        return r_norm
    if quantity == "altitude_km":
        return r_norm - settings.central_body_radius_km
    if quantity == "speed_km_s":
        return float(np.linalg.norm(v))
    if quantity == "radial_velocity_km_s":
        return float(np.dot(r, v) / r_norm)
    if quantity == "elapsed_time_s":
        return float(elapsed_time_s)
    coes = rv_to_coe_eci(r, v, mu_km3_s2=settings.mu_km3_s2)
    values = {
        "semi_major_axis_km": coes.a_km,
        "eccentricity": coes.ecc,
        "inclination_deg": coes.inc_deg,
        "raan_deg": coes.raan_deg,
        "argument_of_periapsis_deg": coes.argp_deg,
        "true_anomaly_deg": coes.true_anomaly_deg,
    }
    return float(values[quantity])


def _signed_residual(quantity: str, actual: float, target: float) -> float:
    residual = float(actual - target)
    if quantity in _ANGULAR_QUANTITIES:
        residual = (residual + 180.0) % 360.0 - 180.0
    return residual


def _ric_basis(state: np.ndarray) -> np.ndarray:
    r = np.asarray(state[:3], dtype=float)
    v = np.asarray(state[3:], dtype=float)
    r_hat = r / float(np.linalg.norm(r))
    c = np.cross(r, v)
    c_norm = float(np.linalg.norm(c))
    if c_norm <= 0.0:
        raise TrajectoryTargetingError("Cannot construct a RIC burn frame from zero angular momentum.")
    c_hat = c / c_norm
    i_hat = np.cross(c_hat, r_hat)
    return np.column_stack((r_hat, i_hat, c_hat))


def _materialized_segments(
    problem: TrajectoryTargetingProblem,
    decision_values: Sequence[float],
) -> list[dict[str, Any]]:
    values = np.asarray(decision_values, dtype=float)
    if values.shape != (len(problem.variables),) or not np.all(np.isfinite(values)):
        raise TrajectoryTargetingError("decision_values must match variables and contain only finite values.")
    segments = [dict(segment) for segment in problem.segments]
    by_name = {segment["name"]: segment for segment in segments}
    for variable, value in zip(problem.variables, values, strict=True):
        segment = by_name[variable.segment]
        if variable.field == "duration_s":
            if value <= 0.0:
                raise TrajectoryTargetingError(f"Variable {variable.name!r} produced a non-positive coast duration.")
            segment["duration_s"] = float(value)
            continue
        components = list(segment["delta_v_m_s"])
        component_index = {
            "delta_v_x_m_s": 0,
            "delta_v_y_m_s": 1,
            "delta_v_z_m_s": 2,
            "delta_v_r_m_s": 0,
            "delta_v_i_m_s": 1,
            "delta_v_c_m_s": 2,
        }[variable.field]
        components[component_index] = float(value)
        segment["delta_v_m_s"] = components
    return segments


def _propagate_duration(
    state: np.ndarray,
    *,
    start_time_s: float,
    duration_s: float,
    settings: PropagationSettings,
    numeric_backend: str = "rust",
) -> tuple[np.ndarray, int]:
    if numeric_backend == "rust":
        from sim.rust_targeting_backend import zonal_coast_history

        widths = []
        remaining = float(duration_s)
        while remaining > 0.0:
            width = min(settings.step_s, remaining)
            widths.append(width)
            remaining -= width
        rows = zonal_coast_history(state, widths, mu_km3_s2=settings.mu_km3_s2,
                                   force_model=settings.force_model)
        return rows[-1], len(widths)
    current = np.asarray(state, dtype=float).copy()
    t_s = float(start_time_s)
    remaining = float(duration_s)
    steps = 0
    propagator = _propagator(settings, numeric_backend=numeric_backend)
    context = OrbitContext(mu_km3_s2=settings.mu_km3_s2, mass_kg=settings.mass_kg)
    while remaining > 0.0:
        step = min(settings.step_s, remaining)
        current = propagator.propagate(current, step, t_s, _ZERO_ACCELERATION, {}, context)
        t_s += step
        remaining -= step
        steps += 1
    return current, steps


def _crossed(left: float, right: float, direction: str, tolerance: float) -> bool:
    if direction == "increasing":
        return left < -tolerance and right >= -tolerance
    if direction == "decreasing":
        return left > tolerance and right <= tolerance
    return (left < -tolerance and right >= -tolerance) or (left > tolerance and right <= tolerance)


def _unwrap_angle_near(angle_deg: float, reference_deg: float) -> float:
    """Return the branch of an angle nearest a continuous reference angle."""

    return float(angle_deg + 360.0 * round((reference_deg - angle_deg) / 360.0))


def _crossing_target_value(
    *,
    quantity: str,
    left_value: float,
    right_value: float,
    target: float,
    direction: str,
    tolerance: float,
) -> float | None:
    """Return the continuous target branch crossed by one propagation step."""

    if quantity not in _ANGULAR_QUANTITIES:
        return target if _crossed(left_value - target, right_value - target, direction, tolerance) else None

    candidates: list[float] = []
    if direction in {"any", "increasing"} and right_value >= left_value:
        level = target + 360.0 * (math.floor((left_value - target) / 360.0) + 1)
        if _crossed(left_value - level, right_value - level, "increasing", tolerance):
            candidates.append(level)
    if direction in {"any", "decreasing"} and right_value <= left_value:
        level = target + 360.0 * (math.ceil((left_value - target) / 360.0) - 1)
        if _crossed(left_value - level, right_value - level, "decreasing", tolerance):
            candidates.append(level)
    if not candidates:
        return None
    return min(candidates, key=lambda value: abs(value - left_value))


def _refine_event(
    *,
    left_state: np.ndarray,
    left_time_s: float,
    right_state: np.ndarray,
    right_time_s: float,
    left_value: float,
    right_value: float,
    quantity: str,
    target: float,
    continuous_target: float,
    settings: PropagationSettings,
    numeric_backend: str = "rust",
) -> tuple[np.ndarray, float, int, tuple[float, float], int]:
    lower_state = np.asarray(left_state, dtype=float).copy()
    lower_time = float(left_time_s)
    upper_time = float(right_time_s)
    lower_value = float(left_value)
    upper_value = float(right_value)
    iterations = 0
    propagation_steps = 0
    while upper_time - lower_time > settings.event_time_tolerance_s and iterations < settings.event_max_iterations:
        midpoint_time = 0.5 * (lower_time + upper_time)
        midpoint_state, midpoint_steps = _propagate_duration(
            lower_state,
            start_time_s=lower_time,
            duration_s=midpoint_time - lower_time,
            settings=settings, numeric_backend=numeric_backend,
        )
        propagation_steps += midpoint_steps
        midpoint_value = _quantity_value(quantity, midpoint_state, midpoint_time, settings)
        if quantity in _ANGULAR_QUANTITIES:
            midpoint_value = _unwrap_angle_near(midpoint_value, 0.5 * (lower_value + upper_value))
        midpoint_residual = midpoint_value - continuous_target
        if abs(midpoint_residual) <= settings.event_value_tolerance:
            lower_state = midpoint_state
            lower_time = midpoint_time
            upper_time = midpoint_time
            lower_value = midpoint_value
            upper_value = midpoint_value
            break
        if np.signbit(midpoint_residual) == np.signbit(lower_value - continuous_target):
            lower_state = midpoint_state
            lower_time = midpoint_time
            lower_value = midpoint_value
        else:
            upper_time = midpoint_time
            upper_value = midpoint_value
        iterations += 1
    event_time = 0.5 * (lower_time + upper_time)
    if upper_time == lower_time:
        event_state = lower_state
        event_value = lower_value
    else:
        event_state, event_steps = _propagate_duration(
            lower_state,
            start_time_s=lower_time,
            duration_s=event_time - lower_time,
            settings=settings, numeric_backend=numeric_backend,
        )
        propagation_steps += event_steps
        event_value = _quantity_value(quantity, event_state, event_time, settings)
        if quantity in _ANGULAR_QUANTITIES:
            event_value = _unwrap_angle_near(event_value, 0.5 * (lower_value + upper_value))
    bracket = (lower_time, upper_time)
    time_satisfied = upper_time - lower_time <= settings.event_time_tolerance_s
    value_satisfied = abs(event_value - continuous_target) <= settings.event_value_tolerance
    if not (time_satisfied or value_satisfied):
        actual = _quantity_value(quantity, event_state, event_time, settings)
        raise EventRefinementError(
            f"Event {quantity}={target} was bracketed but did not refine to its declared tolerances.",
            receipt={
                "status": "refinement_failed",
                "quantity": quantity,
                "target": target,
                "actual": actual,
                "residual": _signed_residual(quantity, actual, target),
                "time_s": event_time,
                "bracket_start_s": lower_time,
                "bracket_end_s": upper_time,
                "refinement_iterations": iterations,
                "event_time_tolerance_s": settings.event_time_tolerance_s,
                "event_value_tolerance": settings.event_value_tolerance,
                "propagation_steps": propagation_steps,
            },
        )
    return event_state, event_time, iterations, bracket, propagation_steps


def _propagate_to_event(
    state: np.ndarray,
    *,
    start_time_s: float,
    stop: Mapping[str, Any],
    settings: PropagationSettings,
    numeric_backend: str = "rust",
) -> tuple[np.ndarray, float, int, dict[str, Any]]:
    quantity = str(stop["quantity"])
    target = float(stop["target"])
    direction = str(stop["direction"])
    max_duration = float(stop["max_duration_s"])
    minimum_elapsed = float(stop["minimum_elapsed_s"])
    current = np.asarray(state, dtype=float).copy()
    current_time = float(start_time_s)
    elapsed = 0.0
    steps = 0
    if minimum_elapsed > 0.0:
        current, minimum_steps = _propagate_duration(
            current,
            start_time_s=current_time,
            duration_s=minimum_elapsed,
            settings=settings, numeric_backend=numeric_backend,
        )
        current_time += minimum_elapsed
        elapsed = minimum_elapsed
        steps += minimum_steps
    current_value = _quantity_value(quantity, current, current_time, settings)
    armed = abs(_signed_residual(quantity, current_value, target)) > settings.event_value_tolerance
    forecast_rows = None
    forecast_index = 0
    while elapsed < max_duration:
        step = min(settings.step_s, max_duration - elapsed)
        if numeric_backend == "rust":
            if forecast_rows is None or forecast_index == len(forecast_rows):
                from sim.rust_targeting_backend import zonal_coast_history

                widths = []
                future_elapsed = elapsed
                for _ in range(256):
                    if future_elapsed >= max_duration:
                        break
                    width = min(settings.step_s, max_duration - future_elapsed)
                    widths.append(width)
                    future_elapsed += width
                try:
                    forecast_rows = zonal_coast_history(
                        current, widths, mu_km3_s2=settings.mu_km3_s2,
                        force_model=settings.force_model,
                    )[1:]
                    forecast_index = 0
                except (ValueError, ArithmeticError):
                    # A speculative failure must not precede an earlier event.
                    forecast_rows = None
            if forecast_rows is not None:
                next_state = forecast_rows[forecast_index]
                forecast_index += 1
                duration_steps = 1
            else:
                next_state, duration_steps = _propagate_duration(
                    current, start_time_s=current_time, duration_s=step,
                    settings=settings, numeric_backend=numeric_backend,
                )
        else:
            next_state, duration_steps = _propagate_duration(
                current, start_time_s=current_time, duration_s=step, settings=settings,
                numeric_backend=numeric_backend,
            )
        next_time = current_time + step
        next_elapsed = elapsed + step
        next_value = _quantity_value(quantity, next_state, next_time, settings)
        if quantity in _ANGULAR_QUANTITIES:
            next_value = _unwrap_angle_near(next_value, current_value)
        steps += duration_steps
        if not armed and abs(_signed_residual(quantity, next_value, target)) > settings.event_value_tolerance:
            armed = True
        continuous_target = (
            _crossing_target_value(
                quantity=quantity,
                left_value=current_value,
                right_value=next_value,
                target=target,
                direction=direction,
                tolerance=settings.event_value_tolerance,
            )
            if armed
            else None
        )
        if continuous_target is not None:
            try:
                event_state, event_time, iterations, bracket, refinement_steps = _refine_event(
                    left_state=current,
                    left_time_s=current_time,
                    right_state=next_state,
                    right_time_s=next_time,
                    left_value=current_value,
                    right_value=next_value,
                    quantity=quantity,
                    target=target,
                    continuous_target=continuous_target,
                    settings=settings, numeric_backend=numeric_backend,
                )
            except EventRefinementError as exc:
                exc.receipt["direction"] = direction
                exc.receipt["elapsed_in_segment_s"] = float(exc.receipt.get("time_s", current_time)) - start_time_s
                exc.receipt["propagation_steps"] = steps + int(exc.receipt.get("propagation_steps", 0))
                raise
            steps += refinement_steps
            actual = _quantity_value(quantity, event_state, event_time, settings)
            receipt = {
                "status": "found",
                "quantity": quantity,
                "target": target,
                "actual": actual,
                "residual": _signed_residual(quantity, actual, target),
                "direction": direction,
                "time_s": event_time,
                "elapsed_in_segment_s": event_time - start_time_s,
                "bracket_start_s": bracket[0],
                "bracket_end_s": bracket[1],
                "refinement_iterations": iterations,
                "refinement_propagation_steps": refinement_steps,
                "propagation_steps": steps,
            }
            return event_state, event_time, steps, receipt
        current = next_state
        current_time = next_time
        elapsed = next_elapsed
        current_value = next_value
    receipt = {
        "status": "missed",
        "quantity": quantity,
        "target": target,
        "direction": direction,
        "max_duration_s": max_duration,
        "final_value": _quantity_value(quantity, current, current_time, settings),
        "final_time_s": current_time,
        "propagation_steps": steps,
    }
    raise MissedEventError(
        f"Event {quantity}={target} was not found within {max_duration} s.",
        receipt=receipt,
    )


def execute_trajectory(
    problem: TrajectoryTargetingProblem | Mapping[str, Any],
    decision_values: Sequence[float] | None = None,
    *, numeric_backend: str = "rust",
) -> dict[str, Any]:
    """Execute one materialized event/burn trajectory through the OEL propagator."""

    parsed = (
        problem if isinstance(problem, TrajectoryTargetingProblem) else TrajectoryTargetingProblem.from_mapping(problem)
    )
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise TrajectoryTargetingError("numeric_backend must be python or rust.")
    if backend == "rust":
        _validate_native_rust_integrator(parsed, operation="direct trajectory execution")
    values = (
        np.asarray([variable.initial for variable in parsed.variables], dtype=float)
        if decision_values is None
        else np.asarray(decision_values, dtype=float)
    )
    segments = _materialized_segments(parsed, values)
    state = np.asarray(parsed.initial_state_eci_km_km_s, dtype=float)
    time_s = 0.0
    segment_receipts: list[dict[str, Any]] = []
    propagation_steps = 0
    total_delta_v_m_s = 0.0
    for segment in segments:
        start_state = state.copy()
        start_time = time_s
        if segment["type"] == "impulsive_burn":
            commanded = np.asarray(segment["delta_v_m_s"], dtype=float)
            basis = np.eye(3) if segment["frame"] == "eci" else _ric_basis(state)
            delta_v_eci_km_s = basis @ (commanded / 1000.0)
            state = state.copy()
            state[3:] += delta_v_eci_km_s
            magnitude_m_s = float(np.linalg.norm(commanded))
            total_delta_v_m_s += magnitude_m_s
            segment_receipts.append(
                {
                    "name": segment["name"],
                    "type": segment["type"],
                    "start_time_s": start_time,
                    "end_time_s": time_s,
                    "start_state_eci_km_km_s": start_state.tolist(),
                    "end_state_eci_km_km_s": state.tolist(),
                    "command_frame": segment["frame"],
                    "delta_v_command_m_s": commanded.tolist(),
                    "delta_v_eci_m_s": (delta_v_eci_km_s * 1000.0).tolist(),
                    "delta_v_magnitude_m_s": magnitude_m_s,
                }
            )
            continue
        if "duration_s" in segment:
            duration = float(segment["duration_s"])
            state, steps = _propagate_duration(
                state,
                start_time_s=time_s,
                duration_s=duration,
                settings=parsed.propagation, numeric_backend=backend,
            )
            time_s += duration
            event_receipt = None
        else:
            try:
                state, time_s, steps, event_receipt = _propagate_to_event(
                    state,
                    start_time_s=time_s,
                    stop=segment["stop"],
                    settings=parsed.propagation, numeric_backend=backend,
                )
            except (MissedEventError, EventRefinementError) as exc:
                exc.receipt["propagation_steps"] = propagation_steps + int(
                    exc.receipt.get("propagation_steps", 0)
                )
                exc.receipt["completed_segments"] = len(segment_receipts)
                raise
        propagation_steps += steps
        segment_receipts.append(
            {
                "name": segment["name"],
                "type": segment["type"],
                "start_time_s": start_time,
                "end_time_s": time_s,
                "duration_s": time_s - start_time,
                "start_state_eci_km_km_s": start_state.tolist(),
                "end_state_eci_km_km_s": state.tolist(),
                "propagation_steps": steps,
                "stop_event": event_receipt,
            }
        )
    return {
        "status": "completed",
        "decision_values": values.tolist(),
        "final_state_eci_km_km_s": state.tolist(),
        "elapsed_time_s": time_s,
        "segments": segment_receipts,
        "resources": {
            "segment_count": len(segments),
            "coast_segment_count": sum(segment["type"] == "coast" for segment in segments),
            "burn_count": sum(segment["type"] == "impulsive_burn" for segment in segments),
            "propagation_steps": propagation_steps,
            "total_delta_v_m_s": total_delta_v_m_s,
            "coast_time_s": sum(
                float(receipt.get("duration_s", 0.0)) for receipt in segment_receipts if receipt["type"] == "coast"
            ),
        },
        "numeric_backend": backend,
        "propagator": {
            **_propagator(parsed.propagation, numeric_backend=backend).propagation_metadata(),
            **parsed.propagation.to_dict(),
        },
    }


def evaluate_terminal_constraints(
    problem: TrajectoryTargetingProblem,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    state = np.asarray(execution["final_state_eci_km_km_s"], dtype=float)
    elapsed = float(execution["elapsed_time_s"])
    rows: list[dict[str, Any]] = []
    raw: list[float] = []
    normalized: list[float] = []
    for constraint in problem.constraints:
        actual = _quantity_value(constraint.quantity, state, elapsed, problem.propagation)
        residual = _signed_residual(constraint.quantity, actual, constraint.target)
        normalized_residual = residual / constraint.tolerance
        raw.append(residual)
        normalized.append(normalized_residual)
        rows.append(
            {
                **constraint.to_dict(),
                "actual": actual,
                "residual": residual,
                "normalized_residual": normalized_residual,
                "satisfied": abs(normalized_residual) <= 1.0,
            }
        )
    maximum = max(abs(value) for value in normalized)
    return {
        "rows": rows,
        "raw_residuals": raw,
        "normalized_residuals": normalized,
        "max_abs_normalized_residual": maximum,
        "all_satisfied": maximum <= 1.0,
    }


def finite_difference_jacobian(
    problem: TrajectoryTargetingProblem | Mapping[str, Any],
    decision_values: Sequence[float],
    *, numeric_backend: str = "rust",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a central finite-difference Jacobian and evaluation accounting."""

    parsed = (
        problem if isinstance(problem, TrajectoryTargetingProblem) else TrajectoryTargetingProblem.from_mapping(problem)
    )
    values = np.asarray(decision_values, dtype=float)
    if values.shape != (len(parsed.variables),):
        raise TrajectoryTargetingError("decision_values must match the number of variables.")
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise TrajectoryTargetingError("numeric_backend must be python or rust.")
    if backend == "rust":
        _validate_native_jacobian_envelope(parsed)
    batch_fixed = backend == "rust" and all(
        segment["type"] == "impulsive_burn" or "duration_s" in segment for segment in parsed.segments
    )
    jacobian = np.empty((len(parsed.constraints), len(parsed.variables)), dtype=float)
    evaluations = 0
    propagation_steps = 0
    effective_perturbations: list[float] = []
    native_trials: list[list[dict[str, Any]]] = []
    for column, variable in enumerate(parsed.variables):
        perturbation = float(variable.perturbation)
        if variable.field == "duration_s":
            perturbation = min(perturbation, 0.5 * float(values[column]))
        if not math.isfinite(perturbation) or perturbation <= 0.0:
            raise TrajectoryTargetingError(
                f"Variable {variable.name!r} has no positive central-difference neighborhood at the current value."
            )
        effective_perturbations.append(perturbation)
        plus = values.copy()
        minus = values.copy()
        plus[column] += perturbation
        minus[column] -= perturbation
        if batch_fixed:
            native_trials.append(_materialized_segments(parsed, plus))
            native_trials.append(_materialized_segments(parsed, minus))
            continue
        try:
            plus_execution = execute_trajectory(parsed, plus, numeric_backend=backend)
        except (MissedEventError, EventRefinementError) as exc:
            exc.receipt["trajectory_evaluations"] = evaluations + 1
            exc.receipt["propagation_steps"] = propagation_steps + int(
                exc.receipt.get("propagation_steps", 0)
            )
            raise
        evaluations += 1
        propagation_steps += int(plus_execution["resources"]["propagation_steps"])
        try:
            minus_execution = execute_trajectory(parsed, minus, numeric_backend=backend)
        except (MissedEventError, EventRefinementError) as exc:
            exc.receipt["trajectory_evaluations"] = evaluations + 1
            exc.receipt["propagation_steps"] = propagation_steps + int(
                exc.receipt.get("propagation_steps", 0)
            )
            raise
        evaluations += 1
        propagation_steps += int(minus_execution["resources"]["propagation_steps"])
        plus_residuals = np.asarray(
            evaluate_terminal_constraints(parsed, plus_execution)["normalized_residuals"], dtype=float
        )
        minus_residuals = np.asarray(
            evaluate_terminal_constraints(parsed, minus_execution)["normalized_residuals"], dtype=float
        )
        jacobian[:, column] = (plus_residuals - minus_residuals) / (2.0 * perturbation)
    if batch_fixed:
        from sim.rust_targeting_backend import fixed_duration_rollouts

        rows = fixed_duration_rollouts(
            np.asarray(parsed.initial_state_eci_km_km_s, dtype=float), native_trials,
            step_s=parsed.propagation.step_s, mu_km3_s2=parsed.propagation.mu_km3_s2,
            force_model=parsed.propagation.force_model,
        )
        evaluations = len(rows)
        propagation_steps = int(np.sum(rows[:, 7]))
        for column, perturbation in enumerate(effective_perturbations):
            residuals = []
            for row in rows[2 * column:2 * column + 2]:
                execution = {"final_state_eci_km_km_s": row[:6], "elapsed_time_s": row[6]}
                residuals.append(np.asarray(
                    evaluate_terminal_constraints(parsed, execution)["normalized_residuals"], dtype=float,
                ))
            jacobian[:, column] = (residuals[0] - residuals[1]) / (2.0 * perturbation)
    return jacobian, {
        "trajectory_evaluations": evaluations,
        "propagation_steps": propagation_steps,
        "effective_perturbations": effective_perturbations,
    }


def _validate_native_rust_integrator(problem: TrajectoryTargetingProblem, *, operation: str) -> None:
    """Reject unsupported targeting integrators before native selection or execution."""

    integrator = problem.propagation.integrator
    if integrator != "rk4":
        raise TrajectoryTargetingError(
            f"Rust {operation} supports only RK4 (propagation.integrator='rk4'); received {integrator!r}. "
            "Use numeric_backend='python' for explicit RKF78, DOPRI5, or adaptive propagation."
        )


def _validate_native_jacobian_envelope(problem: TrajectoryTargetingProblem) -> None:
    _validate_native_rust_integrator(problem, operation="targeting Jacobian evaluation")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_failed_evaluation(
    resources: dict[str, int],
    exc: TrajectoryTargetingError,
    *,
    default_evaluations: int = 1,
) -> None:
    receipt = getattr(exc, "receipt", {})
    resources["trajectory_evaluations"] += int(receipt.get("trajectory_evaluations", default_evaluations))
    resources["propagation_steps"] += int(receipt.get("propagation_steps", 0))


def _failure_result(
    problem: TrajectoryTargetingProblem,
    *,
    problem_sha256: str,
    numeric_backend: str,
    status: str,
    message: str,
    history: list[dict[str, Any]],
    resources: dict[str, int],
    best_execution: Mapping[str, Any] | None,
    best_constraints: Mapping[str, Any] | None,
    decision_values: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": TRAJECTORY_TARGETING_EVIDENCE_SCHEMA,
        "problem_name": problem.name,
        "problem_sha256": problem_sha256,
        "numeric_backend": numeric_backend,
        "execution_sha256": _canonical_sha256({"problem_sha256": problem_sha256, "numeric_backend": numeric_backend}),
        "status": status,
        "converged": False,
        "message": message,
        "variables": [variable.to_dict() for variable in problem.variables],
        "constraints": [constraint.to_dict() for constraint in problem.constraints],
        "decision_values": decision_values.tolist(),
        "convergence_history": history,
        "best_execution": None if best_execution is None else dict(best_execution),
        "best_constraint_evaluation": None if best_constraints is None else dict(best_constraints),
        "authoritative_repropagation": None,
        "resources": resources,
        "limitations": [
            "Deterministic single shooting is local and does not establish global optimality.",
            "This public primitive has no bounds, inequality constraints, multi-start search, uncertainty campaign, or finite burns.",
            "A converged deterministic result is engineering evidence, not operational maneuver authorization or flight qualification.",
        ],
    }


def solve_trajectory_target(
    problem: TrajectoryTargetingProblem | Mapping[str, Any],
    *, numeric_backend: str = "rust",
) -> dict[str, Any]:
    """Solve a public single-shooting problem and independently repropagate it."""

    parsed = (
        problem if isinstance(problem, TrajectoryTargetingProblem) else TrajectoryTargetingProblem.from_mapping(problem)
    )
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise TrajectoryTargetingError("numeric_backend must be python or rust.")
    if backend == "rust" and parsed.variables:
        _validate_native_jacobian_envelope(parsed)
    problem_sha256 = _canonical_sha256(parsed.to_dict())
    decision_values = np.asarray([variable.initial for variable in parsed.variables], dtype=float)
    history: list[dict[str, Any]] = []
    resources = {"trajectory_evaluations": 0, "jacobian_evaluations": 0, "propagation_steps": 0}
    best_execution: dict[str, Any] | None = None
    best_constraints: dict[str, Any] | None = None

    if not parsed.variables:
        try:
            execution = execute_trajectory(parsed, decision_values, numeric_backend=backend)
        except (MissedEventError, EventRefinementError) as exc:
            _record_failed_evaluation(resources, exc)
            status = "event_refinement_failed" if isinstance(exc, EventRefinementError) else "missed_event"
            return _failure_result(
                parsed,
                problem_sha256=problem_sha256,
                status=status,
                message=str(exc),
                history=history,
                resources=resources,
                best_execution={"status": status, "event": exc.receipt},
                best_constraints=None,
                decision_values=decision_values,
                numeric_backend=backend,
            )
        resources["trajectory_evaluations"] += 1
        resources["propagation_steps"] += int(execution["resources"]["propagation_steps"])
        constraints = evaluate_terminal_constraints(parsed, execution)
        if not constraints["all_satisfied"]:
            return _failure_result(
                parsed,
                problem_sha256=problem_sha256,
                status="infeasible",
                message="The fixed trajectory does not satisfy its terminal constraints and has no decision variables.",
                history=history,
                resources=resources,
                best_execution=execution,
                best_constraints=constraints,
                decision_values=decision_values,
                numeric_backend=backend,
            )
        return _successful_result(parsed, problem_sha256, decision_values, history, resources, execution, constraints, numeric_backend=backend)

    for iteration in range(parsed.solver.max_iterations + 1):
        try:
            execution = execute_trajectory(parsed, decision_values, numeric_backend=backend)
        except (MissedEventError, EventRefinementError) as exc:
            _record_failed_evaluation(resources, exc)
            status = "event_refinement_failed" if isinstance(exc, EventRefinementError) else "missed_event"
            return _failure_result(
                parsed,
                problem_sha256=problem_sha256,
                status=status,
                message=str(exc),
                history=history,
                resources=resources,
                best_execution={"status": status, "event": exc.receipt},
                best_constraints=best_constraints,
                decision_values=decision_values,
                numeric_backend=backend,
            )
        resources["trajectory_evaluations"] += 1
        resources["propagation_steps"] += int(execution["resources"]["propagation_steps"])
        constraint_evaluation = evaluate_terminal_constraints(parsed, execution)
        best_execution = execution
        best_constraints = constraint_evaluation
        residuals = np.asarray(constraint_evaluation["normalized_residuals"], dtype=float)
        history_row: dict[str, Any] = {
            "iteration": iteration,
            "decision_values": decision_values.tolist(),
            "raw_residuals": list(constraint_evaluation["raw_residuals"]),
            "normalized_residuals": residuals.tolist(),
            "max_abs_normalized_residual": float(np.max(np.abs(residuals))),
        }
        if constraint_evaluation["all_satisfied"]:
            history_row["disposition"] = "converged"
            history.append(history_row)
            return _successful_result(
                parsed,
                problem_sha256,
                decision_values,
                history,
                resources,
                execution,
                constraint_evaluation,
                numeric_backend=backend,
            )
        if iteration >= parsed.solver.max_iterations:
            history_row["disposition"] = "iteration_limit"
            history.append(history_row)
            break
        try:
            if backend == "python":
                jacobian, jacobian_resources = finite_difference_jacobian(parsed, decision_values, numeric_backend=backend)
            else:
                jacobian, jacobian_resources = finite_difference_jacobian(
                    parsed, decision_values, numeric_backend=backend,
                )
        except (MissedEventError, EventRefinementError) as exc:
            _record_failed_evaluation(resources, exc, default_evaluations=0)
            status = "event_refinement_failed" if isinstance(exc, EventRefinementError) else "missed_event"
            history_row["disposition"] = f"{status}_during_jacobian"
            history.append(history_row)
            return _failure_result(
                parsed,
                problem_sha256=problem_sha256,
                status=status,
                message=f"A Jacobian perturbation could not complete its event evaluation: {exc}",
                history=history,
                resources=resources,
                best_execution=execution,
                best_constraints=constraint_evaluation,
                decision_values=decision_values,
                numeric_backend=backend,
            )
        except TrajectoryTargetingError as exc:
            history_row["disposition"] = "jacobian_failed"
            history.append(history_row)
            return _failure_result(
                parsed,
                problem_sha256=problem_sha256,
                status="jacobian_failed",
                message=f"The Jacobian could not be evaluated: {exc}",
                history=history,
                resources=resources,
                best_execution=execution,
                best_constraints=constraint_evaluation,
                decision_values=decision_values,
                numeric_backend=backend,
            )
        resources["trajectory_evaluations"] += int(jacobian_resources["trajectory_evaluations"])
        resources["jacobian_evaluations"] += 1
        resources["propagation_steps"] += int(jacobian_resources["propagation_steps"])
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        threshold = parsed.solver.rank_rcond * float(singular_values[0]) if singular_values.size else 0.0
        rank = int(np.sum(singular_values > threshold))
        history_row["jacobian"] = jacobian.tolist()
        history_row["effective_perturbations"] = list(jacobian_resources["effective_perturbations"])
        history_row["jacobian_rank"] = rank
        history_row["jacobian_singular_values"] = singular_values.tolist()
        if rank < len(parsed.variables):
            history_row["disposition"] = "rank_deficient"
            history.append(history_row)
            return _failure_result(
                parsed,
                problem_sha256=problem_sha256,
                status="rank_deficient",
                message="The normalized constraint Jacobian is rank deficient for the declared decision variables.",
                history=history,
                resources=resources,
                best_execution=execution,
                best_constraints=constraint_evaluation,
                decision_values=decision_values,
                numeric_backend=backend,
            )
        correction, *_ = np.linalg.lstsq(jacobian, -residuals, rcond=parsed.solver.rank_rcond)
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > parsed.solver.correction_limit:
            correction *= parsed.solver.correction_limit / correction_norm
        history_row["proposed_correction"] = correction.tolist()
        history_row["correction_norm"] = correction_norm
        current_norm = float(np.linalg.norm(residuals))
        scale = 1.0
        accepted = False
        while scale >= parsed.solver.minimum_line_search_scale:
            candidate_values = decision_values + scale * correction
            try:
                candidate_execution = execute_trajectory(parsed, candidate_values, numeric_backend=backend)
                candidate_constraints = evaluate_terminal_constraints(parsed, candidate_execution)
            except TrajectoryTargetingError as exc:
                _record_failed_evaluation(resources, exc)
                scale *= 0.5
                continue
            resources["trajectory_evaluations"] += 1
            resources["propagation_steps"] += int(candidate_execution["resources"]["propagation_steps"])
            candidate_norm = float(np.linalg.norm(candidate_constraints["normalized_residuals"]))
            if candidate_norm < current_norm:
                decision_values = candidate_values
                history_row["accepted_line_search_scale"] = scale
                history_row["accepted_residual_norm"] = candidate_norm
                history_row["disposition"] = "correction_accepted"
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            history_row["disposition"] = "no_improving_correction"
            history.append(history_row)
            return _failure_result(
                parsed,
                problem_sha256=problem_sha256,
                status="non_convergent",
                message="The line search found no correction that reduced the normalized residual norm.",
                history=history,
                resources=resources,
                best_execution=execution,
                best_constraints=constraint_evaluation,
                decision_values=decision_values,
                numeric_backend=backend,
            )
        history.append(history_row)
    return _failure_result(
        parsed,
        problem_sha256=problem_sha256,
        status="non_convergent",
        message=f"The targeter reached its {parsed.solver.max_iterations}-iteration limit.",
        history=history,
        resources=resources,
        best_execution=best_execution,
        best_constraints=best_constraints,
        decision_values=decision_values,
        numeric_backend=backend,
    )


def _successful_result(
    problem: TrajectoryTargetingProblem,
    problem_sha256: str,
    decision_values: np.ndarray,
    history: list[dict[str, Any]],
    resources: dict[str, int],
    solve_execution: Mapping[str, Any],
    solve_constraints: Mapping[str, Any],
    *, numeric_backend: str,
) -> dict[str, Any]:
    repropagation = execute_trajectory(problem, decision_values, numeric_backend=numeric_backend)
    repropagation_constraints = evaluate_terminal_constraints(problem, repropagation)
    resources["trajectory_evaluations"] += 1
    resources["propagation_steps"] += int(repropagation["resources"]["propagation_steps"])
    solve_state = np.asarray(solve_execution["final_state_eci_km_km_s"], dtype=float)
    repropagated_state = np.asarray(repropagation["final_state_eci_km_km_s"], dtype=float)
    state_difference = repropagated_state - solve_state
    verified = bool(repropagation_constraints["all_satisfied"])
    return {
        "schema_version": TRAJECTORY_TARGETING_EVIDENCE_SCHEMA,
        "problem_name": problem.name,
        "problem_sha256": problem_sha256,
        "numeric_backend": numeric_backend,
        "execution_sha256": _canonical_sha256({"problem_sha256": problem_sha256, "numeric_backend": numeric_backend}),
        "status": "converged" if verified else "repropagation_failed",
        "converged": verified,
        "message": (
            "The terminal constraints converged and passed an independent authoritative OEL repropagation."
            if verified
            else "The shooting pass converged, but authoritative repropagation did not satisfy the constraints."
        ),
        "variables": [variable.to_dict() for variable in problem.variables],
        "constraints": [constraint.to_dict() for constraint in problem.constraints],
        "decision_values": decision_values.tolist(),
        "convergence_history": history,
        "solution_execution": dict(solve_execution),
        "solution_constraint_evaluation": dict(solve_constraints),
        "authoritative_repropagation": {
            "status": "verified" if verified else "failed",
            "execution": repropagation,
            "constraint_evaluation": repropagation_constraints,
            "final_state_difference_km_km_s": state_difference.tolist(),
            "final_position_difference_norm_km": float(np.linalg.norm(state_difference[:3])),
            "final_velocity_difference_norm_km_s": float(np.linalg.norm(state_difference[3:])),
        },
        "resources": resources,
        "limitations": [
            "Deterministic single shooting is local and does not establish global optimality.",
            "This public primitive has no bounds, inequality constraints, multi-start search, uncertainty campaign, or finite burns.",
            "A converged deterministic result is engineering evidence, not operational maneuver authorization or flight qualification.",
        ],
    }


def write_trajectory_targeting_evidence(path: str | Path, evidence: Mapping[str, Any]) -> Path:
    """Write canonical, human-inspectable targeting evidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(evidence), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return destination


__all__ = [
    "TRAJECTORY_TARGETING_EVIDENCE_SCHEMA",
    "TRAJECTORY_TARGETING_PROBLEM_SCHEMA",
    "DecisionVariable",
    "EventRefinementError",
    "MissedEventError",
    "PropagationSettings",
    "SolverSettings",
    "TerminalConstraint",
    "TrajectoryTargetingError",
    "TrajectoryTargetingProblem",
    "evaluate_terminal_constraints",
    "execute_trajectory",
    "finite_difference_jacobian",
    "solve_trajectory_target",
    "write_trajectory_targeting_evidence",
]
