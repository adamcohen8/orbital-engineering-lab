"""License-gated constrained trajectory optimization on the authoritative ONP.

This package owns optimization and evidence orchestration only.  Every coast
and finite-burn integration step is delegated to :class:`OrbitPropagator`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin, j3_plugin, j4_plugin
from sim.pro_features import FEATURE_TRAJECTORY_OPTIMIZATION, require_pro_feature
from sim.security.output_writes import (
    OutputWriter,
    bind_output_writer,
    current_output_writer,
    mkdir_output,
    read_output_bytes,
    sha256_output,
    staged_output_file,
    write_output_bytes,
)
from sim.utils.io import SafeReadError, read_regular_file_nofollow

PROBLEM_SCHEMA = "oel.trajectory_optimization_problem.v1"
EVIDENCE_SCHEMA = "oel.trajectory_optimization_evidence.v1"
MANIFEST_SCHEMA = "oel.trajectory_optimization_manifest.v1"
G0_M_S2 = 9.80665
_FORCES = {"j2": j2_plugin, "j3": j3_plugin, "j4": j4_plugin}
_VECTOR_FIELDS = {
    "delta_v_x_m_s": ("delta_v_m_s", 0),
    "delta_v_y_m_s": ("delta_v_m_s", 1),
    "delta_v_z_m_s": ("delta_v_m_s", 2),
    "delta_v_r_m_s": ("delta_v_m_s", 0),
    "delta_v_i_m_s": ("delta_v_m_s", 1),
    "delta_v_c_m_s": ("delta_v_m_s", 2),
    "direction_x": ("direction", 0),
    "direction_y": ("direction", 1),
    "direction_z": ("direction", 2),
    "direction_r": ("direction", 0),
    "direction_i": ("direction", 1),
    "direction_c": ("direction", 2),
}
_STATE_METRICS = {
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
    "mass_kg",
}
_RESOURCE_METRICS = {
    "total_delta_v_m_s",
    "impulsive_delta_v_m_s",
    "finite_burn_delta_v_m_s",
    "propellant_used_kg",
    "final_mass_kg",
    "elapsed_time_s",
    "propagation_steps",
}
_PATH_METRICS = {
    "minimum_altitude_km",
    "maximum_altitude_km",
    "minimum_mass_kg",
    "maximum_thrust_n",
}
_ANGULAR_METRICS = {"raan_deg", "argument_of_periapsis_deg", "true_anomaly_deg"}
_MAX_PROBLEM_BYTES = 5_000_000
_MAX_SEGMENTS = 128
_MAX_VARIABLES = 128
_MAX_OBJECTIVES = 64
_MAX_CONSTRAINTS = 256
_MAX_SHOOTING_NODES = 32
_MAX_EVALUATION_BUDGET = 20_000
_MAX_SOLVER_ITERATIONS = 5_000
_MAX_EVENT_ITERATIONS = 200
_MAX_DERIVED_PROPAGATION_STEPS = 20_000_000
_MIN_PROPAGATION_STEP_S = 1.0e-3


class TrajectoryOptimizationError(ValueError):
    """Raised when a problem, evaluation, or evidence packet is invalid."""


class EvaluationBudgetExceeded(TrajectoryOptimizationError):
    """Raised when a solve exceeds its declared authoritative evaluation budget."""


class _FeasibleConvergence(RuntimeError):
    """Internal stop when OEL's declared feasibility and stagnation tests pass."""

    def __init__(self, scaled: np.ndarray, evaluation: dict[str, Any]):
        super().__init__("OEL feasibility and stagnation tolerances satisfied.")
        self.scaled = scaled.copy()
        self.evaluation = evaluation


def _unknown(raw: Mapping[str, Any], allowed: set[str], context: str) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        raise TrajectoryOptimizationError(f"Unknown {context} fields: {extra}.")


def _finite(value: Any, context: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TrajectoryOptimizationError(f"{context} must be finite numeric data, not boolean.")
    result = float(value)
    if not math.isfinite(result):
        raise TrajectoryOptimizationError(f"{context} must be finite.")
    return result


def _positive(value: Any, context: str) -> float:
    result = _finite(value, context)
    if result <= 0.0:
        raise TrajectoryOptimizationError(f"{context} must be positive.")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TrajectoryOptimizationError(f"{context} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrajectoryOptimizationError(f"{context} must be an integer.") from exc
    if result != value:
        raise TrajectoryOptimizationError(f"{context} must be an integer.")
    return result


def _vector(value: Any, size: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise TrajectoryOptimizationError(f"{context} must contain exactly {size} values.")
    result = tuple(_finite(item, context) for item in value)
    return result


def _sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(read_regular_file_nofollow(Path(__file__), min_bytes=1, max_bytes=5_000_000)).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise TrajectoryOptimizationError(f"Non-finite JSON constant {value!r} is not permitted.")


def _load_json(path: Path | str, *, max_bytes: int = _MAX_PROBLEM_BYTES) -> Any:
    source = Path(path)
    try:
        content = read_regular_file_nofollow(source, max_bytes=max_bytes)
        return json.loads(content.decode("utf-8"), parse_constant=_reject_json_constant)
    except SafeReadError as exc:
        raise TrajectoryOptimizationError(f"Could not safely read JSON input: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrajectoryOptimizationError(f"Invalid JSON: {exc}") from exc


@dataclass(frozen=True)
class Variable:
    name: str
    segment: str
    field: str
    initial: float
    lower: float
    upper: float
    scale: float

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> Variable:
        raw = dict(value)
        _unknown(raw, {"name", "segment", "field", "initial", "lower", "upper", "scale"}, "variable")
        item = cls(
            name=str(raw.get("name", "")).strip(),
            segment=str(raw.get("segment", "")).strip(),
            field=str(raw.get("field", "")).strip().lower(),
            initial=_finite(raw.get("initial"), "variable.initial"),
            lower=_finite(raw.get("lower"), "variable.lower"),
            upper=_finite(raw.get("upper"), "variable.upper"),
            scale=_positive(raw.get("scale"), "variable.scale"),
        )
        if not item.name or not item.segment or not item.field:
            raise TrajectoryOptimizationError("Every variable requires name, segment, and field.")
        if item.lower > item.initial or item.initial > item.upper or item.lower == item.upper:
            raise TrajectoryOptimizationError(
                f"Variable {item.name!r} must satisfy lower <= initial <= upper and span a range."
            )
        return item

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Constraint:
    name: str
    scope: str
    quantity: str
    equal: float | None
    lower: float | None
    upper: float | None
    tolerance: float
    scale: float

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> Constraint:
        raw = dict(value)
        _unknown(raw, {"name", "scope", "quantity", "equal", "lower", "upper", "tolerance", "scale"}, "constraint")
        scope = str(raw.get("scope", "terminal")).strip().lower()
        quantity = str(raw.get("quantity", "")).strip().lower()
        supported = {"terminal": _STATE_METRICS, "resource": _RESOURCE_METRICS, "path": _PATH_METRICS}
        if scope not in supported or quantity not in supported[scope]:
            raise TrajectoryOptimizationError(f"Unsupported {scope!r} constraint quantity {quantity!r}.")
        equal = None if raw.get("equal") is None else _finite(raw["equal"], "constraint.equal")
        lower = None if raw.get("lower") is None else _finite(raw["lower"], "constraint.lower")
        upper = None if raw.get("upper") is None else _finite(raw["upper"], "constraint.upper")
        if equal is not None and (lower is not None or upper is not None):
            raise TrajectoryOptimizationError("A constraint cannot combine equal with lower/upper.")
        if equal is None and lower is None and upper is None:
            raise TrajectoryOptimizationError("A constraint requires equal, lower, or upper.")
        if lower is not None and upper is not None and lower > upper:
            raise TrajectoryOptimizationError("Constraint lower cannot exceed upper.")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise TrajectoryOptimizationError("Every constraint requires a name.")
        return cls(
            name,
            scope,
            quantity,
            equal,
            lower,
            upper,
            _positive(raw.get("tolerance", 1e-6), "constraint.tolerance"),
            _positive(raw.get("scale", 1.0), "constraint.scale"),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Objective:
    name: str
    scope: str
    quantity: str
    weight: float
    scale: float
    goal: str

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> Objective:
        raw = dict(value)
        _unknown(raw, {"name", "scope", "quantity", "weight", "scale", "goal"}, "objective")
        scope = str(raw.get("scope", "resource")).strip().lower()
        quantity = str(raw.get("quantity", "")).strip().lower()
        supported = _STATE_METRICS if scope == "terminal" else _RESOURCE_METRICS if scope == "resource" else set()
        if quantity not in supported:
            raise TrajectoryOptimizationError(f"Unsupported {scope!r} objective quantity {quantity!r}.")
        goal = str(raw.get("goal", "minimize")).strip().lower()
        if goal not in {"minimize", "maximize"}:
            raise TrajectoryOptimizationError("objective.goal must be minimize or maximize.")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise TrajectoryOptimizationError("Every objective requires a name.")
        return cls(
            name,
            scope,
            quantity,
            _finite(raw.get("weight", 1.0), "objective.weight"),
            _positive(raw.get("scale", 1.0), "objective.scale"),
            goal,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ShootingNode:
    name: str
    before_segment: str
    state: tuple[float, ...]
    mass_kg: float
    position_bound_km: float
    velocity_bound_km_s: float
    mass_bound_kg: float
    position_scale_km: float
    velocity_scale_km_s: float
    mass_scale_kg: float
    position_tolerance_km: float
    velocity_tolerance_km_s: float
    mass_tolerance_kg: float

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> ShootingNode:
        raw = dict(value)
        allowed = {
            "name",
            "before_segment",
            "state_eci_km_km_s",
            "mass_kg",
            "position_bound_km",
            "velocity_bound_km_s",
            "mass_bound_kg",
            "position_scale_km",
            "velocity_scale_km_s",
            "mass_scale_kg",
            "position_tolerance_km",
            "velocity_tolerance_km_s",
            "mass_tolerance_kg",
        }
        _unknown(raw, allowed, "shooting node")
        name, before = str(raw.get("name", "")).strip(), str(raw.get("before_segment", "")).strip()
        if not name or not before:
            raise TrajectoryOptimizationError("Every shooting node requires name and before_segment.")
        return cls(
            name,
            before,
            _vector(raw.get("state_eci_km_km_s"), 6, "shooting node state"),
            _positive(raw.get("mass_kg"), "shooting node mass"),
            _positive(raw.get("position_bound_km", 100.0), "position bound"),
            _positive(raw.get("velocity_bound_km_s", 0.1), "velocity bound"),
            _positive(raw.get("mass_bound_kg", 10.0), "mass bound"),
            _positive(raw.get("position_scale_km", 1.0), "position scale"),
            _positive(raw.get("velocity_scale_km_s", 0.001), "velocity scale"),
            _positive(raw.get("mass_scale_kg", 1.0), "mass scale"),
            _positive(raw.get("position_tolerance_km", 1e-3), "position tolerance"),
            _positive(raw.get("velocity_tolerance_km_s", 1e-6), "velocity tolerance"),
            _positive(raw.get("mass_tolerance_kg", 1e-6), "mass tolerance"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = self.__dict__.copy()
        result["state_eci_km_km_s"] = list(result.pop("state"))
        return result


@dataclass(frozen=True)
class TrajectoryOptimizationProblem:
    name: str
    initial_state: tuple[float, ...]
    wet_mass_kg: float
    dry_mass_kg: float
    impulsive_isp_s: float
    propagation: dict[str, Any]
    segments: tuple[dict[str, Any], ...]
    variables: tuple[Variable, ...]
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...]
    shooting_nodes: tuple[ShootingNode, ...]
    solver: dict[str, Any]
    robustness: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrajectoryOptimizationProblem:
        raw = dict(value)
        allowed = {
            "schema_version",
            "name",
            "initial_state_eci_km_km_s",
            "spacecraft",
            "propagation",
            "segments",
            "variables",
            "objectives",
            "constraints",
            "shooting_nodes",
            "solver",
            "robustness",
        }
        _unknown(raw, allowed, "problem")
        if raw.get("schema_version") != PROBLEM_SCHEMA:
            raise TrajectoryOptimizationError(f"schema_version must be {PROBLEM_SCHEMA!r}.")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise TrajectoryOptimizationError("Problem name is required.")
        spacecraft = dict(raw.get("spacecraft") or {})
        _unknown(spacecraft, {"wet_mass_kg", "dry_mass_kg", "impulsive_isp_s"}, "spacecraft")
        wet = _positive(spacecraft.get("wet_mass_kg"), "spacecraft.wet_mass_kg")
        dry = _positive(spacecraft.get("dry_mass_kg"), "spacecraft.dry_mass_kg")
        if dry >= wet:
            raise TrajectoryOptimizationError("spacecraft.dry_mass_kg must be below wet_mass_kg.")
        propagation = _parse_propagation(raw.get("propagation") or {})
        raw_segments = raw.get("segments", [])
        raw_variables = raw.get("variables", [])
        raw_objectives = raw.get("objectives", [])
        raw_constraints = raw.get("constraints", [])
        raw_nodes = raw.get("shooting_nodes", [])
        for label, items, limit in (
            ("segments", raw_segments, _MAX_SEGMENTS),
            ("variables", raw_variables, _MAX_VARIABLES),
            ("objectives", raw_objectives, _MAX_OBJECTIVES),
            ("constraints", raw_constraints, _MAX_CONSTRAINTS),
            ("shooting_nodes", raw_nodes, _MAX_SHOOTING_NODES),
        ):
            if not isinstance(items, list):
                raise TrajectoryOptimizationError(f"{label} must be a JSON array.")
            if len(items) > limit:
                raise TrajectoryOptimizationError(f"{label} exceeds the bounded limit of {limit}.")
        segments = tuple(_parse_segment(item) for item in raw_segments)
        if not segments:
            raise TrajectoryOptimizationError("At least one segment is required.")
        names = [item["name"] for item in segments]
        if len(set(names)) != len(names):
            raise TrajectoryOptimizationError("Segment names must be unique.")
        variables = tuple(Variable.parse(item) for item in raw_variables)
        if len({item.name for item in variables}) != len(variables):
            raise TrajectoryOptimizationError("Variable names must be unique.")
        by_name = {item["name"]: item for item in segments}
        for variable in variables:
            if variable.segment not in by_name:
                raise TrajectoryOptimizationError(f"Variable {variable.name!r} references an unknown segment.")
            _validate_variable_field(variable, by_name[variable.segment])
        variable_targets = [(item.segment, item.field) for item in variables]
        if len(variable_targets) != len(set(variable_targets)):
            raise TrajectoryOptimizationError("Variables must target unique segment fields.")
        objectives = tuple(Objective.parse(item) for item in raw_objectives)
        if not objectives:
            raise TrajectoryOptimizationError("At least one objective is required.")
        if len({item.name for item in objectives}) != len(objectives):
            raise TrajectoryOptimizationError("Objective names must be unique.")
        constraints = tuple(Constraint.parse(item) for item in raw_constraints)
        if len({item.name for item in constraints}) != len(constraints):
            raise TrajectoryOptimizationError("Constraint names must be unique.")
        nodes = tuple(ShootingNode.parse(item) for item in raw_nodes)
        if len({item.name for item in nodes}) != len(nodes):
            raise TrajectoryOptimizationError("Shooting-node names must be unique.")
        for node in nodes:
            if node.before_segment not in by_name or names.index(node.before_segment) == 0:
                raise TrajectoryOptimizationError("A shooting node must precede a known non-first segment.")
            if not dry <= node.mass_kg <= wet:
                raise TrajectoryOptimizationError(f"Shooting node {node.name!r} mass is outside spacecraft limits.")
        before_segments = [item.before_segment for item in nodes]
        if len(before_segments) != len(set(before_segments)):
            raise TrajectoryOptimizationError("Only one shooting node may precede a segment.")
        solver = _parse_solver(raw.get("solver") or {})
        robustness = _parse_robustness(raw.get("robustness") or {})
        duration_bound_s = 0.0
        for segment in segments:
            duration_bound_s += float(segment.get("duration_s", segment.get("stop", {}).get("max_duration_s", 0.0)))
        derived_steps = math.ceil(duration_bound_s / propagation["step_s"])
        derived_steps *= solver["evaluation_budget"] + robustness["sample_count"] + 1
        if derived_steps > _MAX_DERIVED_PROPAGATION_STEPS:
            raise TrajectoryOptimizationError(
                "Problem exceeds the bounded derived propagation workload limit."
            )
        return cls(
            name,
            _vector(raw.get("initial_state_eci_km_km_s"), 6, "initial state"),
            wet,
            dry,
            _positive(spacecraft.get("impulsive_isp_s", 300.0), "spacecraft.impulsive_isp_s"),
            propagation,
            segments,
            variables,
            objectives,
            constraints,
            nodes,
            solver,
            robustness,
        )

    @classmethod
    def from_json(cls, path: Path | str) -> TrajectoryOptimizationProblem:
        return cls.from_mapping(_load_json(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROBLEM_SCHEMA,
            "name": self.name,
            "initial_state_eci_km_km_s": list(self.initial_state),
            "spacecraft": {
                "wet_mass_kg": self.wet_mass_kg,
                "dry_mass_kg": self.dry_mass_kg,
                "impulsive_isp_s": self.impulsive_isp_s,
            },
            "propagation": self.propagation,
            "segments": list(self.segments),
            "variables": [item.to_dict() for item in self.variables],
            "objectives": [item.to_dict() for item in self.objectives],
            "constraints": [item.to_dict() for item in self.constraints],
            "shooting_nodes": [item.to_dict() for item in self.shooting_nodes],
            "solver": self.solver,
            "robustness": self.robustness,
        }


def _parse_propagation(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    _unknown(
        raw,
        {
            "step_s",
            "integrator",
            "force_model",
            "mu_km3_s2",
            "central_body_radius_km",
            "event_time_tolerance_s",
            "event_value_tolerance",
            "event_max_iterations",
            "numeric_backend",
        },
        "propagation",
    )
    backend = str(raw.get("numeric_backend", "rust")).strip().lower()
    if backend not in {"python", "rust"}:
        raise TrajectoryOptimizationError("propagation.numeric_backend must be python or rust.")
    integrator = str(raw.get("integrator", "rk4")).strip().lower()
    if integrator not in {"rk4", "rkf78", "dopri5", "adaptive"}:
        raise TrajectoryOptimizationError("Unsupported propagation integrator.")
    force_model = [str(item).strip().lower() for item in raw.get("force_model", [])]
    if set(force_model) - set(_FORCES):
        raise TrajectoryOptimizationError("Only j2, j3, and j4 force-model plugins are supported.")
    if len(force_model) != len(set(force_model)):
        raise TrajectoryOptimizationError("propagation.force_model entries must be unique.")
    event_max_iterations = _integer(raw.get("event_max_iterations", 80), "propagation.event_max_iterations")
    if not 1 <= event_max_iterations <= _MAX_EVENT_ITERATIONS:
        raise TrajectoryOptimizationError(
            f"propagation.event_max_iterations must be 1..{_MAX_EVENT_ITERATIONS}."
        )
    step_s = _positive(raw.get("step_s", 10.0), "propagation.step_s")
    if step_s < _MIN_PROPAGATION_STEP_S:
        raise TrajectoryOptimizationError(
            f"propagation.step_s must be at least {_MIN_PROPAGATION_STEP_S:g} seconds."
        )
    return {
        "step_s": step_s,
        "integrator": integrator,
        **{"numeric_backend": backend},
        "force_model": force_model,
        "mu_km3_s2": _positive(raw.get("mu_km3_s2", EARTH_MU_KM3_S2), "mu"),
        "central_body_radius_km": _positive(raw.get("central_body_radius_km", EARTH_RADIUS_KM), "radius"),
        "event_time_tolerance_s": _positive(raw.get("event_time_tolerance_s", 1e-6), "event time tolerance"),
        "event_value_tolerance": _positive(raw.get("event_value_tolerance", 1e-10), "event value tolerance"),
        "event_max_iterations": event_max_iterations,
    }


def _parse_segment(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    kind = str(raw.get("type", "")).strip().lower()
    common = {"name", "type"}
    if kind == "coast":
        _unknown(raw, common | {"duration_s", "stop"}, "coast segment")
        if ("duration_s" in raw) == ("stop" in raw):
            raise TrajectoryOptimizationError("A coast requires exactly one of duration_s or stop.")
        result = {"name": str(raw.get("name", "")).strip(), "type": kind}
        if "duration_s" in raw:
            result["duration_s"] = _positive(raw["duration_s"], "coast.duration_s")
        else:
            stop = dict(raw["stop"])
            _unknown(stop, {"quantity", "target", "direction", "max_duration_s", "minimum_elapsed_s"}, "event stop")
            quantity = str(stop.get("quantity", "")).strip().lower()
            if quantity not in _STATE_METRICS - {"mass_kg"}:
                raise TrajectoryOptimizationError(f"Unsupported event quantity {quantity!r}.")
            direction = str(stop.get("direction", "either")).strip().lower()
            if direction not in {"either", "increasing", "decreasing"}:
                raise TrajectoryOptimizationError("Event direction must be either, increasing, or decreasing.")
            result["stop"] = {
                "quantity": quantity,
                "target": _finite(stop.get("target"), "event target"),
                "direction": direction,
                "max_duration_s": _positive(stop.get("max_duration_s"), "event horizon"),
                "minimum_elapsed_s": _finite(stop.get("minimum_elapsed_s", 0.0), "minimum elapsed"),
            }
            if (
                result["stop"]["minimum_elapsed_s"] < 0.0
                or result["stop"]["minimum_elapsed_s"] >= result["stop"]["max_duration_s"]
            ):
                raise TrajectoryOptimizationError(
                    "Event minimum_elapsed_s must be nonnegative and below max_duration_s."
                )
        if not result["name"]:
            raise TrajectoryOptimizationError("Every segment requires a name.")
        return result
    if kind == "impulsive_burn":
        _unknown(raw, common | {"frame", "delta_v_m_s"}, "impulsive-burn segment")
        frame = str(raw.get("frame", "eci")).strip().lower()
        if frame not in {"eci", "ric"}:
            raise TrajectoryOptimizationError("Burn frame must be eci or ric.")
        result = {
            "name": str(raw.get("name", "")).strip(),
            "type": kind,
            "frame": frame,
            "delta_v_m_s": list(_vector(raw.get("delta_v_m_s"), 3, "delta-v")),
        }
    elif kind == "finite_burn":
        _unknown(raw, common | {"frame", "direction", "thrust_n", "duration_s", "isp_s"}, "finite-burn segment")
        frame = str(raw.get("frame", "eci")).strip().lower()
        if frame not in {"eci", "ric"}:
            raise TrajectoryOptimizationError("Finite-burn frame must be eci or ric.")
        direction = np.asarray(_vector(raw.get("direction"), 3, "finite-burn direction"), dtype=float)
        if np.linalg.norm(direction) <= 0.0:
            raise TrajectoryOptimizationError("Finite-burn direction cannot be zero.")
        result = {
            "name": str(raw.get("name", "")).strip(),
            "type": kind,
            "frame": frame,
            "direction": direction.tolist(),
            "thrust_n": _positive(raw.get("thrust_n"), "thrust"),
            "duration_s": _positive(raw.get("duration_s"), "burn duration"),
            "isp_s": _positive(raw.get("isp_s"), "finite-burn Isp"),
        }
    else:
        raise TrajectoryOptimizationError(f"Unsupported segment type {kind!r}.")
    if not result["name"]:
        raise TrajectoryOptimizationError("Every segment requires a name.")
    return result


def _validate_variable_field(variable: Variable, segment: Mapping[str, Any]) -> None:
    supported = {
        "coast": {"duration_s"},
        "impulsive_burn": set(_VECTOR_FIELDS) & {f"delta_v_{x}_m_s" for x in "xyzric"},
        "finite_burn": {"duration_s", "thrust_n"} | (set(_VECTOR_FIELDS) - {f"delta_v_{x}_m_s" for x in "xyzric"}),
    }
    if variable.field not in supported[segment["type"]]:
        raise TrajectoryOptimizationError(f"Variable {variable.name!r} field is invalid for {segment['type']}.")
    if variable.field == "duration_s" and "duration_s" not in segment:
        raise TrajectoryOptimizationError("Event-coast duration cannot be a direct variable.")
    if variable.field in {"duration_s", "thrust_n"} and variable.lower <= 0.0:
        raise TrajectoryOptimizationError(
            f"Variable {variable.name!r} requires a strictly positive lower bound."
        )
    if segment["type"] == "impulsive_burn":
        suffix = variable.field.removeprefix("delta_v_").removesuffix("_m_s")
        if (segment["frame"] == "eci" and suffix not in {"x", "y", "z"}) or (
            segment["frame"] == "ric" and suffix not in {"r", "i", "c"}
        ):
            raise TrajectoryOptimizationError(f"Variable {variable.name!r} does not match the burn frame.")
    if segment["type"] == "finite_burn" and variable.field.startswith("direction_"):
        suffix = variable.field.removeprefix("direction_")
        if (segment["frame"] == "eci" and suffix not in {"x", "y", "z"}) or (
            segment["frame"] == "ric" and suffix not in {"r", "i", "c"}
        ):
            raise TrajectoryOptimizationError(f"Variable {variable.name!r} does not match the finite-burn frame.")


def _parse_solver(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    _unknown(
        raw,
        {
            "method",
            "max_iterations",
            "ftol",
            "finite_difference_rel_step",
            "evaluation_budget",
            "multi_start_count",
            "seed",
        },
        "solver",
    )
    method = str(raw.get("method", "slsqp")).strip().lower()
    if method not in {"slsqp", "trust-constr"}:
        raise TrajectoryOptimizationError("solver.method must be slsqp or trust-constr.")
    result = {
        "method": method,
        "max_iterations": _integer(raw.get("max_iterations", 100), "solver.max_iterations"),
        "ftol": _positive(raw.get("ftol", 1e-9), "solver.ftol"),
        "finite_difference_rel_step": _positive(raw.get("finite_difference_rel_step", 1e-6), "finite difference step"),
        "evaluation_budget": _integer(raw.get("evaluation_budget", 5000), "solver.evaluation_budget"),
        "multi_start_count": _integer(raw.get("multi_start_count", 1), "solver.multi_start_count"),
        "seed": _integer(raw.get("seed", 0), "solver.seed"),
    }
    if (
        not 1 <= result["max_iterations"] <= _MAX_SOLVER_ITERATIONS
        or not 1 <= result["evaluation_budget"] <= _MAX_EVALUATION_BUDGET
        or not 1 <= result["multi_start_count"] <= 64
    ):
        raise TrajectoryOptimizationError(
            "Solver max_iterations must be 1..5000, evaluation_budget 1..20000, "
            "and multi_start_count 1..64."
        )
    return result


def _parse_robustness(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    _unknown(
        raw, {"sample_count", "seed", "position_sigma_km", "velocity_sigma_km_s", "thrust_scale_sigma"}, "robustness"
    )
    result = {
        "sample_count": _integer(raw.get("sample_count", 0), "robustness.sample_count"),
        "seed": _integer(raw.get("seed", 0), "robustness.seed"),
        "position_sigma_km": _finite(raw.get("position_sigma_km", 0.0), "position sigma"),
        "velocity_sigma_km_s": _finite(raw.get("velocity_sigma_km_s", 0.0), "velocity sigma"),
        "thrust_scale_sigma": _finite(raw.get("thrust_scale_sigma", 0.0), "thrust sigma"),
    }
    if (
        not 0 <= result["sample_count"] <= 64
        or min(result[k] for k in ("position_sigma_km", "velocity_sigma_km_s", "thrust_scale_sigma")) < 0
    ):
        raise TrajectoryOptimizationError("Robustness sample count must be 0..64 and sigmas nonnegative.")
    return result


def _ric_basis(state: np.ndarray) -> np.ndarray:
    r, v = state[:3], state[3:]
    r_hat = r / np.linalg.norm(r)
    c = np.cross(r, v)
    if np.linalg.norm(c) <= 0.0:
        raise TrajectoryOptimizationError("Cannot construct RIC frame from zero angular momentum.")
    c_hat = c / np.linalg.norm(c)
    return np.column_stack((r_hat, np.cross(c_hat, r_hat), c_hat))


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _angular_event_target_at_boundary(
    target_deg: float,
    boundary_deg: float,
    direction: str,
    motion_delta_deg: float,
    tolerance_deg: float,
) -> float:
    """Return the first eligible angular branch at the event-search boundary."""
    turns = (boundary_deg - target_deg) / 360.0
    nearest_turn = round(turns)
    nearest_target = target_deg + 360.0 * nearest_turn
    if abs(boundary_deg - nearest_target) <= tolerance_deg:
        return nearest_target
    if direction == "decreasing" or (direction == "either" and motion_delta_deg < 0.0):
        return target_deg + 360.0 * math.floor(turns)
    return target_deg + 360.0 * math.ceil(turns)


def _state_metrics(
    state: np.ndarray, elapsed: float, mass: float, problem: TrajectoryOptimizationProblem
) -> dict[str, float]:
    r, v = state[:3], state[3:]
    coe = rv_to_coe_eci(r, v, mu_km3_s2=problem.propagation["mu_km3_s2"])
    return {
        "position_x_km": float(r[0]),
        "position_y_km": float(r[1]),
        "position_z_km": float(r[2]),
        "velocity_x_km_s": float(v[0]),
        "velocity_y_km_s": float(v[1]),
        "velocity_z_km_s": float(v[2]),
        "radius_km": float(np.linalg.norm(r)),
        "altitude_km": float(np.linalg.norm(r) - problem.propagation["central_body_radius_km"]),
        "speed_km_s": float(np.linalg.norm(v)),
        "radial_velocity_km_s": float(np.dot(r, v) / np.linalg.norm(r)),
        "semi_major_axis_km": float(coe.a_km),
        "eccentricity": float(coe.ecc),
        "inclination_deg": float(coe.inc_deg),
        "raan_deg": float(coe.raan_deg),
        "argument_of_periapsis_deg": float(coe.argp_deg),
        "true_anomaly_deg": float(coe.true_anomaly_deg),
        "elapsed_time_s": float(elapsed),
        "mass_kg": float(mass),
    }


def _materialize(
    problem: TrajectoryOptimizationProblem, physical: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, float]]]:
    segments = [json.loads(json.dumps(item)) for item in problem.segments]
    by_name = {item["name"]: item for item in segments}
    for variable, value in zip(problem.variables, physical[: len(problem.variables)], strict=True):
        segment = by_name[variable.segment]
        if variable.field in {"duration_s", "thrust_n"}:
            segment[variable.field] = float(value)
        else:
            field, index = _VECTOR_FIELDS[variable.field]
            segment[field][index] = float(value)
    for segment in segments:
        if "duration_s" in segment and float(segment["duration_s"]) <= 0.0:
            raise TrajectoryOptimizationError(f"Segment {segment['name']!r} duration must remain positive.")
        if segment["type"] == "finite_burn":
            if float(segment["thrust_n"]) <= 0.0:
                raise TrajectoryOptimizationError(f"Finite burn {segment['name']!r} thrust must remain positive.")
            direction = np.asarray(segment["direction"], dtype=float)
            if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 0.0:
                raise TrajectoryOptimizationError(
                    f"Finite burn {segment['name']!r} direction must remain finite and nonzero."
                )
    nodes: dict[str, tuple[np.ndarray, float]] = {}
    cursor = len(problem.variables)
    for node in problem.shooting_nodes:
        nodes[node.before_segment] = (
            np.asarray(physical[cursor : cursor + 6], dtype=float),
            float(physical[cursor + 6]),
        )
        cursor += 7
    return segments, nodes


def _decision_contract(
    problem: TrajectoryOptimizationProblem,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = [item.name for item in problem.variables]
    initial = [item.initial for item in problem.variables]
    lower = [item.lower for item in problem.variables]
    upper = [item.upper for item in problem.variables]
    scale = [item.scale for item in problem.variables]
    for node in problem.shooting_nodes:
        values = list(node.state) + [node.mass_kg]
        spans = [node.position_bound_km] * 3 + [node.velocity_bound_km_s] * 3 + [node.mass_bound_kg]
        scales = [node.position_scale_km] * 3 + [node.velocity_scale_km_s] * 3 + [node.mass_scale_kg]
        suffixes = ["rx_km", "ry_km", "rz_km", "vx_km_s", "vy_km_s", "vz_km_s", "mass_kg"]
        names.extend(f"node.{node.name}.{suffix}" for suffix in suffixes)
        initial.extend(values)
        node_lower = [v - s for v, s in zip(values, spans)]
        node_upper = [v + s for v, s in zip(values, spans)]
        node_lower[-1] = max(node_lower[-1], problem.dry_mass_kg)
        node_upper[-1] = min(node_upper[-1], problem.wet_mass_kg)
        lower.extend(node_lower)
        upper.extend(node_upper)
        scale.extend(scales)
    return names, np.asarray(initial), np.asarray(lower), np.asarray(upper), np.asarray(scale)


def _require_numeric_backend(problem: TrajectoryOptimizationProblem) -> None:
    backend = problem.propagation.get("numeric_backend", "rust")
    if backend not in {"python", "rust"}:
        raise TrajectoryOptimizationError("propagation.numeric_backend must be python or rust.")
    if backend == "rust":
        from sim.rust_orbit_backend import _extension

        if not callable(getattr(_extension(), "orbit_zonal_history_bytes", None)):
            raise RuntimeError("Rust trajectory optimization requires native zonal coast histories")


def _sample_state_fields(state: np.ndarray, problem: TrajectoryOptimizationProblem) -> dict[str, float]:
    """Retain the COE owner's domain guards without computing unused angles."""
    x, y, z, vx, vy, vz = map(float, state)
    r_norm = float(np.linalg.norm(state[:3]))
    v_norm = float(np.linalg.norm(state[3:]))
    if r_norm <= 0.0:
        raise ValueError("Position norm must be positive.")
    h = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    if float(np.linalg.norm(h)) <= 0.0:
        raise ValueError("Angular momentum norm must be positive.")
    mu = problem.propagation["mu_km3_s2"]
    e = (
        np.asarray((vy * h[2] - vz * h[1], vz * h[0] - vx * h[2], vx * h[1] - vy * h[0])) / mu
        - state[:3] / r_norm
    )
    energy = 0.5 * v_norm * v_norm - mu / r_norm
    if abs(energy) <= 1e-15:
        raise ValueError("Parabolic COEs are not supported.")
    if -mu / (2.0 * energy) <= 0.0 or float(np.linalg.norm(e)) >= 1.0:
        raise ValueError("Only elliptical Earth-centered COEs are supported.")
    return {
        "position_x_km": x, "position_y_km": y, "position_z_km": z,
        "velocity_x_km_s": vx, "velocity_y_km_s": vy, "velocity_z_km_s": vz,
        "altitude_km": float(r_norm - problem.propagation["central_body_radius_km"]),
    }


def _coast_samples(
    problem: TrajectoryOptimizationProblem, segment: Mapping[str, Any],
    state: np.ndarray, remaining: float,
):
    """Batch only uninterrupted RK4 coasts; retain scalar event/burn owners."""
    if (
        problem.propagation.get("numeric_backend", "rust") != "rust"
        or problem.propagation["integrator"] != "rk4"
        or segment["type"] != "coast" or "stop" in segment
    ):
        return None
    from sim.rust_targeting_backend import zonal_coast_history

    # Bounded chunks avoid retaining a second large history alongside samples.
    def samples():
        nonlocal remaining
        current = state.copy()
        while remaining > 1e-12:
            widths = []
            while remaining > 1e-12 and len(widths) < 4096:
                width = min(problem.propagation["step_s"], remaining)
                widths.append(width)
                remaining -= width
            history = zonal_coast_history(
                current, widths, mu_km3_s2=problem.propagation["mu_km3_s2"],
                force_model=problem.propagation["force_model"],
            )
            current = history[-1].copy()
            yield from history[1:]

    return samples()


def _evaluate_trajectory(
    problem: TrajectoryOptimizationProblem,
    physical: Sequence[float],
    *,
    initial_perturbation: np.ndarray | None = None,
    thrust_scale: float = 1.0,
) -> dict[str, Any]:
    _require_numeric_backend(problem)
    values = np.asarray(physical, dtype=float)
    names, _, lower, upper, _ = _decision_contract(problem)
    if values.shape != (len(names),) or not np.all(np.isfinite(values)):
        raise TrajectoryOptimizationError("Decision vector has the wrong shape or nonfinite values.")
    if np.any(values < lower - 1e-12) or np.any(values > upper + 1e-12):
        raise TrajectoryOptimizationError("Decision vector violates declared bounds.")
    segments, nodes = _materialize(problem, values)
    state = np.asarray(problem.initial_state, dtype=float).copy()
    if initial_perturbation is not None:
        state += np.asarray(initial_perturbation, dtype=float)
    mass, elapsed = problem.wet_mass_kg, 0.0
    propagator = OrbitPropagator(
        integrator=problem.propagation["integrator"],
        plugins=[_FORCES[x] for x in problem.propagation["force_model"]],
        numeric_backend=problem.propagation.get("numeric_backend", "rust"),
    )
    samples: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    defects: list[dict[str, Any]] = []
    resources = {
        "impulsive_delta_v_m_s": 0.0,
        "finite_burn_delta_v_m_s": 0.0,
        "propellant_used_kg": 0.0,
        "final_mass_kg": mass,
        "elapsed_time_s": 0.0,
        "propagation_steps": 0.0,
        "event_refinement_steps": 0.0,
    }

    def sample(segment: str, thrust_n: float = 0.0) -> None:
        metrics = _sample_state_fields(state, problem)
        samples.append(
            {
                "sample_index": len(samples),
                "segment": segment,
                "time_s": elapsed,
                "mass_kg": mass,
                "thrust_n": thrust_n,
                **{
                    key: metrics[key]
                    for key in (
                        "position_x_km",
                        "position_y_km",
                        "position_z_km",
                        "velocity_x_km_s",
                        "velocity_y_km_s",
                        "velocity_z_km_s",
                        "altitude_km",
                    )
                },
            }
        )

    sample("initial")
    for segment in segments:
        if segment["name"] in nodes:
            node_state, node_mass = nodes[segment["name"]]
            node = next(item for item in problem.shooting_nodes if item.before_segment == segment["name"])
            delta = np.concatenate((state - node_state, [mass - node_mass]))
            tolerances = np.asarray(
                [node.position_tolerance_km] * 3 + [node.velocity_tolerance_km_s] * 3 + [node.mass_tolerance_kg]
            )
            defects.append({"node": node.name, "raw": delta.tolist(), "normalized": (delta / tolerances).tolist()})
            state, mass = node_state.copy(), node_mass
            sample(f"node:{node.name}")
        start_time, start_mass, start_state = elapsed, mass, state.copy()
        if segment["type"] == "impulsive_burn":
            command = np.asarray(segment["delta_v_m_s"], dtype=float)
            dv_eci = (np.eye(3) if segment["frame"] == "eci" else _ric_basis(state)) @ (command / 1000.0)
            magnitude = float(np.linalg.norm(command))
            new_mass = mass * math.exp(-magnitude / (problem.impulsive_isp_s * G0_M_S2))
            if new_mass < problem.dry_mass_kg - 1e-12:
                raise TrajectoryOptimizationError(f"Impulsive burn {segment['name']!r} violates dry mass.")
            state[3:] += dv_eci
            mass = new_mass
            resources["impulsive_delta_v_m_s"] += magnitude
            sample(segment["name"])
        else:
            duration = float(segment.get("duration_s", segment.get("stop", {}).get("max_duration_s", 0.0)))
            stop = segment.get("stop")
            prior_actual = None
            prior_unwrapped = None
            event_target = None
            if stop:
                prior_actual = _state_metrics(state, elapsed, mass, problem)[stop["quantity"]]
                if stop["quantity"] in _ANGULAR_METRICS:
                    prior_unwrapped = prior_actual
                    angular_delta = _wrap_degrees(stop["target"] - prior_actual)
                    if stop["direction"] == "increasing" and angular_delta < 0.0:
                        angular_delta += 360.0
                    elif stop["direction"] == "decreasing" and angular_delta > 0.0:
                        angular_delta -= 360.0
                    event_target = prior_actual + angular_delta
                else:
                    event_target = stop["target"]
            remaining = duration
            found_event = False
            event_search_started = False
            coast_samples = _coast_samples(problem, segment, state, remaining)
            while remaining > 1e-12:
                dt = min(problem.propagation["step_s"], remaining)
                thrust = float(segment.get("thrust_n", 0.0)) * float(thrust_scale)
                acceleration = np.zeros(3)
                if thrust:
                    mdot = thrust / (float(segment["isp_s"]) * G0_M_S2)
                    next_mass = mass - mdot * dt
                    if next_mass < problem.dry_mass_kg - 1e-12:
                        raise TrajectoryOptimizationError(f"Finite burn {segment['name']!r} violates dry mass.")
                    direction = np.asarray(segment["direction"], dtype=float)
                    direction /= np.linalg.norm(direction)
                    direction_eci = (np.eye(3) if segment["frame"] == "eci" else _ric_basis(state)) @ direction
                    acceleration = direction_eci * thrust / (0.5 * (mass + next_mass) * 1000.0)
                context = OrbitContext(mu_km3_s2=problem.propagation["mu_km3_s2"], mass_kg=mass)
                previous_state, previous_time, previous_mass = state.copy(), elapsed, mass
                if coast_samples is None:
                    state = propagator.propagate(state, dt, elapsed, acceleration, {}, context)
                else:
                    state = next(coast_samples).copy()
                    propagator.last_numeric_path = "rust_native_zonal_coast_history"
                elapsed += dt
                remaining -= dt
                resources["propagation_steps"] += 1
                if thrust:
                    mass = next_mass
                    resources["finite_burn_delta_v_m_s"] += thrust / (0.5 * (previous_mass + mass)) * dt
                sample(segment["name"], thrust)
                if stop and elapsed - start_time >= stop["minimum_elapsed_s"]:
                    first_search_step = not event_search_started
                    actual = _state_metrics(state, elapsed, mass, problem)[stop["quantity"]]
                    if stop["quantity"] in _ANGULAR_METRICS:
                        current_unwrapped = prior_unwrapped + _wrap_degrees(actual - prior_actual)
                    else:
                        current_unwrapped = actual
                    left_state, left_time = previous_state, previous_time
                    left_actual, left_unwrapped = prior_actual, prior_unwrapped
                    minimum_time = start_time + stop["minimum_elapsed_s"]
                    if left_time < minimum_time:
                        left_state = propagator.propagate(
                            previous_state,
                            minimum_time - previous_time,
                            previous_time,
                            np.zeros(3),
                            {},
                            OrbitContext(mu_km3_s2=problem.propagation["mu_km3_s2"], mass_kg=mass),
                        )
                        resources["event_refinement_steps"] += 1
                        left_time = minimum_time
                        left_actual = _state_metrics(left_state, left_time, mass, problem)[stop["quantity"]]
                        if stop["quantity"] in _ANGULAR_METRICS:
                            left_unwrapped = prior_unwrapped + _wrap_degrees(left_actual - prior_actual)
                        else:
                            left_unwrapped = left_actual
                    if stop["quantity"] in _ANGULAR_METRICS:
                        if stop["minimum_elapsed_s"] > 0.0 and first_search_step:
                            event_target = _angular_event_target_at_boundary(
                                stop["target"],
                                left_unwrapped,
                                stop["direction"],
                                current_unwrapped - left_unwrapped,
                                problem.propagation["event_value_tolerance"],
                            )
                        elif stop["direction"] == "either":
                            turns = (prior_unwrapped - stop["target"]) / 360.0
                            branch = math.ceil(turns) if current_unwrapped >= prior_unwrapped else math.floor(turns)
                            event_target = stop["target"] + 360.0 * branch
                        left_value = left_unwrapped - event_target
                        current_value = current_unwrapped - event_target
                    else:
                        left_value = left_actual - event_target
                        current_value = current_unwrapped - event_target
                    event_search_started = True
                    crossed = (
                        abs(left_value) <= problem.propagation["event_value_tolerance"]
                        or (stop["direction"] in {"either", "increasing"} and left_value < 0 <= current_value)
                        or (stop["direction"] in {"either", "decreasing"} and left_value > 0 >= current_value)
                    )
                    if crossed:
                        right_state, right_time, right_value = state, elapsed, current_value
                        for _ in range(problem.propagation["event_max_iterations"]):
                            if right_time - left_time <= problem.propagation["event_time_tolerance_s"]:
                                break
                            midpoint_time = 0.5 * (left_time + right_time)
                            midpoint_state = propagator.propagate(
                                left_state,
                                midpoint_time - left_time,
                                left_time,
                                np.zeros(3),
                                {},
                                OrbitContext(mu_km3_s2=problem.propagation["mu_km3_s2"], mass_kg=mass),
                            )
                            resources["event_refinement_steps"] += 1
                            midpoint_actual = _state_metrics(midpoint_state, midpoint_time, mass, problem)[
                                stop["quantity"]
                            ]
                            if stop["quantity"] in _ANGULAR_METRICS:
                                midpoint_unwrapped = left_unwrapped + _wrap_degrees(midpoint_actual - left_actual)
                                midpoint_value = midpoint_unwrapped - event_target
                            else:
                                midpoint_unwrapped = midpoint_actual
                                midpoint_value = midpoint_actual - event_target
                            if abs(midpoint_value) <= problem.propagation["event_value_tolerance"]:
                                left_state = right_state = midpoint_state
                                left_time = right_time = midpoint_time
                                left_value = right_value = midpoint_value
                                break
                            if np.signbit(midpoint_value) == np.signbit(left_value):
                                left_state, left_time, left_value = midpoint_state, midpoint_time, midpoint_value
                                left_actual, left_unwrapped = midpoint_actual, midpoint_unwrapped
                            else:
                                right_state, right_time, right_value = midpoint_state, midpoint_time, midpoint_value
                        if (
                            right_time - left_time > problem.propagation["event_time_tolerance_s"]
                            and min(abs(left_value), abs(right_value))
                            > problem.propagation["event_value_tolerance"]
                        ):
                            raise TrajectoryOptimizationError(
                                f"Event refinement in segment {segment['name']!r} did not converge."
                            )
                        if abs(left_value) <= abs(right_value):
                            state, elapsed = left_state, left_time
                        else:
                            state, elapsed = right_state, right_time
                        mass = previous_mass
                        refined = _state_metrics(state, elapsed, mass, problem)
                        samples[-1].update(
                            sample_index=len(samples) - 1,
                            time_s=elapsed,
                            mass_kg=mass,
                            **{k: v for k, v in refined.items() if k in samples[-1] and k != "mass_kg"},
                        )
                        remaining = 0.0
                        found_event = True
                        break
                    prior_actual = actual
                    prior_unwrapped = current_unwrapped
                elif stop:
                    actual = _state_metrics(state, elapsed, mass, problem)[stop["quantity"]]
                    if stop["quantity"] in _ANGULAR_METRICS:
                        prior_unwrapped = prior_unwrapped + _wrap_degrees(actual - prior_actual)
                    else:
                        prior_unwrapped = actual
                    prior_actual = actual
            if stop and not found_event:
                raise TrajectoryOptimizationError(f"Event in segment {segment['name']!r} was not found.")
        receipts.append(
            {
                "name": segment["name"],
                "type": segment["type"],
                "start_time_s": start_time,
                "end_time_s": elapsed,
                "start_mass_kg": start_mass,
                "end_mass_kg": mass,
                "start_state_eci_km_km_s": start_state.tolist(),
                "end_state_eci_km_km_s": state.tolist(),
            }
        )
    resources["total_delta_v_m_s"] = resources["impulsive_delta_v_m_s"] + resources["finite_burn_delta_v_m_s"]
    resources["propellant_used_kg"] = problem.wet_mass_kg - mass
    resources["final_mass_kg"] = mass
    resources["elapsed_time_s"] = elapsed
    terminal = _state_metrics(state, elapsed, mass, problem)
    path = {
        "minimum_altitude_km": min(item["altitude_km"] for item in samples),
        "maximum_altitude_km": max(item["altitude_km"] for item in samples),
        "minimum_mass_kg": min(item["mass_kg"] for item in samples),
        "maximum_thrust_n": max(item["thrust_n"] for item in samples),
    }
    constraints = _constraint_rows(problem, terminal, resources, path, defects)
    objective_terms = []
    objective = 0.0
    for item in problem.objectives:
        actual = terminal[item.quantity] if item.scope == "terminal" else resources[item.quantity]
        signed = actual if item.goal == "minimize" else -actual
        contribution = item.weight * signed / item.scale
        objective += contribution
        objective_terms.append({**item.to_dict(), "actual": actual, "contribution": contribution})
    return {
        "status": "completed",
        "decision_values": values.tolist(),
        "objective": float(objective),
        "objective_terms": objective_terms,
        "terminal": terminal,
        "path": path,
        "resources": resources,
        "constraints": constraints,
        "defects": defects,
        "segments": receipts,
        "trajectory": samples,
        "propagator": {**propagator.propagation_metadata(), **problem.propagation},
    }


def _constraint_rows(
    problem: TrajectoryOptimizationProblem,
    terminal: Mapping[str, float],
    resources: Mapping[str, float],
    path: Mapping[str, float],
    defects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    eq, lower = [], []
    for item in problem.constraints:
        source = terminal if item.scope == "terminal" else resources if item.scope == "resource" else path
        actual = float(source[item.quantity])
        if item.equal is not None:
            residual = actual - item.equal
            if item.scope == "terminal" and item.quantity in _ANGULAR_METRICS:
                residual = (residual + 180.0) % 360.0 - 180.0
            normalized = residual / item.tolerance
            eq.append(normalized)
            margin = 1.0 - abs(normalized)
            satisfied = abs(normalized) <= 1.0
        else:
            margins = []
            if item.lower is not None:
                margins.append((actual - item.lower) / item.scale)
            if item.upper is not None:
                margins.append((item.upper - actual) / item.scale)
            lower.extend(margins)
            margin = min(margins)
            normalized = margin
            satisfied = margin >= -item.tolerance / item.scale
        rows.append(
            {
                **item.to_dict(),
                "actual": actual,
                "normalized": float(normalized),
                "margin": float(margin),
                "satisfied": bool(satisfied),
            }
        )
    for defect in defects:
        for index, value in enumerate(defect["normalized"]):
            eq.append(float(value))
            rows.append(
                {
                    "name": f"defect.{defect['node']}.{index}",
                    "scope": "defect",
                    "quantity": "continuity",
                    "equal": 0.0,
                    "lower": None,
                    "upper": None,
                    "tolerance": 1.0,
                    "scale": 1.0,
                    "actual": float(value),
                    "normalized": float(value),
                    "margin": 1.0 - abs(float(value)),
                    "satisfied": abs(float(value)) <= 1.0,
                }
            )
    return {
        "rows": rows,
        "equality": eq,
        "inequality": lower,
        "all_satisfied": all(item["satisfied"] for item in rows),
        "max_abs_equality": max([abs(x) for x in eq], default=0.0),
        "min_inequality_margin": min(lower, default=0.0),
    }


class _Evaluator:
    def __init__(self, problem: TrajectoryOptimizationProblem, initial: np.ndarray, scale: np.ndarray):
        self.problem, self.initial, self.scale = problem, initial, scale
        self.cache: dict[tuple[float, ...], dict[str, Any]] = {}
        self.calls = 0
        self.equality_count = sum(item.equal is not None for item in problem.constraints) + 7 * len(
            problem.shooting_nodes
        )
        self.inequality_count = sum(
            int(item.lower is not None) + int(item.upper is not None)
            for item in problem.constraints
            if item.equal is None
        )

    def physical(self, scaled: Sequence[float]) -> np.ndarray:
        return self.initial + np.asarray(scaled, dtype=float) * self.scale

    def __call__(self, scaled: Sequence[float]) -> dict[str, Any]:
        key = tuple(float(x) for x in scaled)
        if key not in self.cache:
            if self.calls >= self.problem.solver["evaluation_budget"]:
                raise EvaluationBudgetExceeded("Authoritative trajectory evaluation budget exhausted.")
            self.calls += 1
            try:
                self.cache[key] = _evaluate_trajectory(self.problem, self.physical(scaled))
            except TrajectoryOptimizationError as exc:
                self.cache[key] = {
                    "status": "invalid",
                    "message": str(exc),
                    "objective": 1e12,
                    "constraints": {
                        "equality": [1e6] * self.equality_count,
                        "inequality": [-1e6] * self.inequality_count,
                        "all_satisfied": False,
                        "rows": [],
                    },
                }
        return self.cache[key]

    @property
    def propagation_steps(self) -> int:
        return int(
            sum(
                float(item.get("resources", {}).get("propagation_steps", 0.0))
                + float(item.get("resources", {}).get("event_refinement_steps", 0.0))
                for item in self.cache.values()
            )
        )


def _central_gradient(function, x: np.ndarray, lower: np.ndarray, upper: np.ndarray, rel_step: float) -> np.ndarray:
    gradient = np.empty_like(x, dtype=float)
    for index in range(len(x)):
        h = rel_step * max(1.0, abs(float(x[index])))
        plus, minus = x.copy(), x.copy()
        plus[index] = min(upper[index], x[index] + h)
        minus[index] = max(lower[index], x[index] - h)
        denominator = plus[index] - minus[index]
        if denominator <= 0.0:
            gradient[index] = 0.0
        else:
            gradient[index] = (float(function(plus)) - float(function(minus))) / denominator
    return gradient


def _central_jacobian(function, x: np.ndarray, lower: np.ndarray, upper: np.ndarray, rel_step: float) -> np.ndarray:
    baseline = np.asarray(function(x), dtype=float)
    jacobian = np.empty((len(baseline), len(x)), dtype=float)
    for index in range(len(x)):
        h = rel_step * max(1.0, abs(float(x[index])))
        plus, minus = x.copy(), x.copy()
        plus[index] = min(upper[index], x[index] + h)
        minus[index] = max(lower[index], x[index] - h)
        denominator = plus[index] - minus[index]
        if denominator <= 0.0:
            jacobian[:, index] = 0.0
        else:
            jacobian[:, index] = (np.asarray(function(plus)) - np.asarray(function(minus))) / denominator
    return jacobian


def _constraint_vector(evaluator: _Evaluator, key: str, x: Sequence[float]) -> np.ndarray:
    return np.asarray(evaluator(x)["constraints"][key], dtype=float)


def _constraint_jacobian(
    evaluator: _Evaluator,
    key: str,
    lower: np.ndarray,
    upper: np.ndarray,
    rel_step: float,
    x: Sequence[float],
) -> np.ndarray:
    function = partial(_constraint_vector, evaluator, key)
    return _central_jacobian(function, np.asarray(x), lower, upper, rel_step)


def _objective_value(evaluator: _Evaluator, x: Sequence[float]) -> float:
    return float(evaluator(x)["objective"])


def _objective_gradient(
    evaluator: _Evaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    rel_step: float,
    x: Sequence[float],
) -> np.ndarray:
    return _central_gradient(partial(_objective_value, evaluator), np.asarray(x), lower, upper, rel_step)


def _record_iteration(
    evaluator: _Evaluator,
    convergence_history: list[dict[str, Any]],
    ftol: float,
    x: Sequence[float],
    _solver_state: Any = None,
) -> None:
    current = evaluator(x)
    convergence_history.append(_history_row(len(convergence_history), evaluator.physical(x), current))
    recent = convergence_history[-3:]
    if len(recent) == 3 and all(row["feasible"] for row in recent):
        objectives = np.asarray([row["objective"] for row in recent], dtype=float)
        decisions = np.asarray([row["decision_values"] for row in recent], dtype=float)
        objective_stable = float(np.ptp(objectives)) <= ftol * max(1.0, float(np.max(np.abs(objectives))))
        decision_stable = float(np.max(np.ptp(decisions, axis=0), initial=0.0)) <= math.sqrt(ftol)
        if objective_stable and decision_stable:
            raise _FeasibleConvergence(np.asarray(x, dtype=float), current)


def _finite_difference_derivatives(
    problem: TrajectoryOptimizationProblem | Mapping[str, Any], physical: Sequence[float]
) -> dict[str, Any]:
    """Return bounded central-difference derivatives in physical decision units."""

    parsed = (
        problem
        if isinstance(problem, TrajectoryOptimizationProblem)
        else TrajectoryOptimizationProblem.from_mapping(problem)
    )
    _require_numeric_backend(parsed)
    _, initial, lower, upper, scale = _decision_contract(parsed)
    values = np.asarray(physical, dtype=float)
    if values.shape != initial.shape:
        raise TrajectoryOptimizationError("Decision vector has the wrong shape.")
    evaluator = _Evaluator(parsed, initial, scale)
    scaled = (values - initial) / scale
    scaled_lower, scaled_upper = (lower - initial) / scale, (upper - initial) / scale
    step = parsed.solver["finite_difference_rel_step"]
    equality = partial(_constraint_vector, evaluator, "equality")
    inequality = partial(_constraint_vector, evaluator, "inequality")
    objective_gradient_scaled = _objective_gradient(evaluator, scaled_lower, scaled_upper, step, scaled)
    equality_scaled = _central_jacobian(equality, scaled, scaled_lower, scaled_upper, step)
    inequality_scaled = _central_jacobian(inequality, scaled, scaled_lower, scaled_upper, step)
    return {
        "objective_gradient": (objective_gradient_scaled / scale).tolist(),
        "equality_jacobian": (equality_scaled / scale[np.newaxis, :]).tolist(),
        "inequality_jacobian": (inequality_scaled / scale[np.newaxis, :]).tolist(),
        "trajectory_evaluations": evaluator.calls,
    }


def _solve_trajectory_optimization(
    problem: TrajectoryOptimizationProblem | Mapping[str, Any],
) -> dict[str, Any]:
    parsed = (
        problem
        if isinstance(problem, TrajectoryOptimizationProblem)
        else TrajectoryOptimizationProblem.from_mapping(problem)
    )
    _require_numeric_backend(parsed)
    names, initial, lower, upper, scale = _decision_contract(parsed)
    evaluator = _Evaluator(parsed, initial, scale)
    x0 = np.zeros_like(initial)
    bounds = Bounds((lower - initial) / scale, (upper - initial) / scale)
    scaled_lower, scaled_upper = np.asarray(bounds.lb), np.asarray(bounds.ub)
    fd_step = parsed.solver["finite_difference_rel_step"]
    seed = np.random.default_rng(parsed.solver["seed"])
    starts = [x0] + [seed.uniform(bounds.lb, bounds.ub) for _ in range(parsed.solver["multi_start_count"] - 1)]
    candidates: list[dict[str, Any]] = []
    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
    for index, start in enumerate(starts):
        baseline: dict[str, Any] | None = None
        convergence_history: list[dict[str, Any]] = []
        evaluations_at_start = evaluator.calls
        try:
            baseline = evaluator(start)
            convergence_history.append(_history_row(0, evaluator.physical(start), baseline))
            eq_count = len(baseline["constraints"]["equality"])
            ineq_count = len(baseline["constraints"]["inequality"])
            nonlinear = []
            if eq_count:
                eq_function = partial(_constraint_vector, evaluator, "equality")
                eq_jacobian = partial(
                    _constraint_jacobian,
                    evaluator,
                    "equality",
                    scaled_lower,
                    scaled_upper,
                    fd_step,
                )
                nonlinear.append(NonlinearConstraint(eq_function, 0.0, 0.0, jac=eq_jacobian))
            if ineq_count:
                ineq_function = partial(_constraint_vector, evaluator, "inequality")
                ineq_jacobian = partial(
                    _constraint_jacobian,
                    evaluator,
                    "inequality",
                    scaled_lower,
                    scaled_upper,
                    fd_step,
                )
                nonlinear.append(NonlinearConstraint(ineq_function, 0.0, np.inf, jac=ineq_jacobian))
            options = {"maxiter": parsed.solver["max_iterations"]}
            if parsed.solver["method"] == "slsqp":
                options["ftol"] = parsed.solver["ftol"]
            else:
                options["gtol"] = parsed.solver["ftol"]
                options["factorization_method"] = "SVDFactorization"
            objective_function = partial(_objective_value, evaluator)
            objective_jacobian = partial(
                _objective_gradient,
                evaluator,
                scaled_lower,
                scaled_upper,
                fd_step,
            )
            record_iteration = partial(_record_iteration, evaluator, convergence_history, parsed.solver["ftol"])
            result = minimize(
                objective_function,
                start,
                method=parsed.solver["method"],
                bounds=bounds,
                constraints=nonlinear,
                jac=objective_jacobian,
                callback=record_iteration,
                options=options,
            )
            evaluated = evaluator(result.x)
            feasible = evaluated["status"] == "completed" and evaluated["constraints"]["all_satisfied"]
            candidate = {
                "start_index": index,
                "solver_success": bool(result.success),
                "feasible": feasible,
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "objective": float(evaluated["objective"]),
                "decision_values": evaluator.physical(result.x).tolist(),
                "trajectory_evaluations": evaluator.calls - evaluations_at_start,
                "convergence_history": convergence_history,
            }
            candidates.append(candidate)
            if result.success and feasible and (best is None or candidate["objective"] < best[0]):
                best = candidate["objective"], evaluator.physical(result.x), evaluated
        except _FeasibleConvergence as exc:
            evaluated = exc.evaluation
            candidate = {
                "start_index": index,
                "solver_success": True,
                "feasible": True,
                "status": 100,
                "message": str(exc),
                "iterations": max(0, len(convergence_history) - 1),
                "objective": float(evaluated["objective"]),
                "decision_values": evaluator.physical(exc.scaled).tolist(),
                "trajectory_evaluations": evaluator.calls - evaluations_at_start,
                "convergence_history": convergence_history,
            }
            candidates.append(candidate)
            if best is None or candidate["objective"] < best[0]:
                best = candidate["objective"], evaluator.physical(exc.scaled), evaluated
        except EvaluationBudgetExceeded as exc:
            candidates.append(
                {
                    "start_index": index,
                    "solver_success": False,
                    "feasible": False,
                    "status": -2,
                    "message": str(exc),
                    "iterations": 0,
                    "objective": None,
                    "decision_values": [],
                    "trajectory_evaluations": evaluator.calls - evaluations_at_start,
                    "convergence_history": convergence_history,
                }
            )
            break
        except Exception as exc:
            candidate = {
                "start_index": index,
                "solver_success": False,
                "feasible": False,
                "status": -1,
                "message": f"{type(exc).__name__}: {exc}",
                "iterations": 0,
                "objective": None,
                "decision_values": [],
                "trajectory_evaluations": evaluator.calls - evaluations_at_start,
                "convergence_history": convergence_history,
            }
            candidates.append(candidate)
    if best is None:
        status = (
            "budget_exhausted"
            if candidates and candidates[-1]["status"] == -2
            else "non_convergent"
            if candidates and all(not item["solver_success"] for item in candidates)
            else "infeasible"
        )
        return {
            "schema_version": EVIDENCE_SCHEMA,
            "problem_sha256": _sha(parsed.to_dict()),
            "problem": parsed.to_dict(),
            "status": status,
            "converged": False,
            "message": "No authoritatively feasible candidate was found.",
            "decision_names": names,
            "candidates": candidates,
            "solution": None,
            "authoritative_repropagation": None,
            "robustness": [],
            "robustness_summary": _robustness_summary([]),
            "resources": {"trajectory_evaluations": evaluator.calls, "propagation_steps": evaluator.propagation_steps},
        }
    _, decision, _ = best
    reprop = _evaluate_trajectory(parsed, decision)
    if not reprop["constraints"]["all_satisfied"]:
        status, converged = "repropagation_failed", False
    else:
        status, converged = "converged", True
    robustness = _robustness(parsed, decision) if converged else []
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "problem_sha256": _sha(parsed.to_dict()),
        "problem": parsed.to_dict(),
        "status": status,
        "converged": converged,
        "message": "Fresh authoritative ONP repropagation accepted." if converged else "Fresh repropagation failed.",
        "decision_names": names,
        "candidates": candidates,
        "solution": {"decision_values": decision.tolist(), "objective": reprop["objective"]},
        "authoritative_repropagation": reprop,
        "robustness": robustness,
        "robustness_summary": _robustness_summary(robustness),
        "resources": {
            "trajectory_evaluations": evaluator.calls,
            "multi_start_count": len(candidates),
            "robustness_evaluations": len(robustness),
            "propagation_steps": evaluator.propagation_steps,
            "robustness_propagation_steps": int(sum(row.get("propagation_steps", 0) for row in robustness)),
        },
    }


def _history_row(iteration: int, decision: np.ndarray, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    constraints = evaluation.get("constraints", {})
    return {
        "iteration": iteration,
        "status": evaluation.get("status"),
        "objective": evaluation.get("objective"),
        "decision_values": decision.tolist(),
        "max_abs_equality": constraints.get("max_abs_equality"),
        "min_inequality_margin": constraints.get("min_inequality_margin"),
        "feasible": bool(constraints.get("all_satisfied", False)),
    }


def _robustness(problem: TrajectoryOptimizationProblem, decision: np.ndarray) -> list[dict[str, Any]]:
    settings = problem.robustness
    rng = np.random.default_rng(settings["seed"])
    rows = []
    for index in range(settings["sample_count"]):
        perturb = np.concatenate(
            (rng.normal(0.0, settings["position_sigma_km"], 3), rng.normal(0.0, settings["velocity_sigma_km_s"], 3))
        )
        thrust_scale = max(0.0, 1.0 + rng.normal(0.0, settings["thrust_scale_sigma"]))
        try:
            result = _evaluate_trajectory(
                problem, decision, initial_perturbation=perturb, thrust_scale=thrust_scale
            )
            rows.append(
                {
                    "sample_index": index,
                    "feasible": result["constraints"]["all_satisfied"],
                    "objective": result["objective"],
                    "max_abs_equality": result["constraints"]["max_abs_equality"],
                    "min_inequality_margin": result["constraints"]["min_inequality_margin"],
                    "propagation_steps": int(
                        result["resources"]["propagation_steps"] + result["resources"]["event_refinement_steps"]
                    ),
                    "position_perturbation_km": perturb[:3].tolist(),
                    "velocity_perturbation_km_s": perturb[3:].tolist(),
                    "thrust_scale": thrust_scale,
                }
            )
        except TrajectoryOptimizationError as exc:
            rows.append(
                {
                    "sample_index": index,
                    "feasible": False,
                    "objective": None,
                    "max_abs_equality": 1e300,
                    "min_inequality_margin": -1e300,
                    "propagation_steps": 0,
                    "position_perturbation_km": perturb[:3].tolist(),
                    "velocity_perturbation_km_s": perturb[3:].tolist(),
                    "thrust_scale": thrust_scale,
                    "error": str(exc),
                }
            )
    return rows


def _robustness_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    feasible = sum(bool(row.get("feasible")) for row in rows)
    objectives = [float(row["objective"]) for row in rows if row.get("objective") is not None]
    return {
        "sample_count": len(rows),
        "feasible_count": feasible,
        "feasible_fraction": feasible / len(rows) if rows else None,
        "objective_min": min(objectives) if objectives else None,
        "objective_max": max(objectives) if objectives else None,
        "worst_max_abs_equality": max((float(row["max_abs_equality"]) for row in rows), default=None),
        "worst_min_inequality_margin": min((float(row["min_inequality_margin"]) for row in rows), default=None),
    }


def _write_evidence(
    result: Mapping[str, Any],
    output_dir: Path | str,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    if writer is not None and current_output_writer() is None:
        with bind_output_writer(writer):
            return _write_evidence(result, output_dir)
    if writer is not None and current_output_writer() is not writer:
        raise TrajectoryOptimizationError("A different secure output writer is already bound.")
    if result.get("schema_version") != EVIDENCE_SCHEMA:
        raise TrajectoryOptimizationError("Result is not trajectory-optimization evidence.")
    problem = TrajectoryOptimizationProblem.from_mapping(result.get("problem") or {})
    if _sha(problem.to_dict()) != result.get("problem_sha256"):
        raise TrajectoryOptimizationError("Result problem digest is inconsistent.")
    reproduced = _solve_trajectory_optimization(problem)
    if _sha(reproduced) != _sha(dict(result)):
        raise TrajectoryOptimizationError("Result does not match a fresh authoritative optimization replay.")
    active_writer = current_output_writer()
    destination = _output_path(output_dir, resolve=active_writer is None)
    if active_writer is not None:
        if _output_path(active_writer.logical_root, resolve=False) != destination:
            raise TrajectoryOptimizationError(
                "Trajectory evidence logical output_dir must match the secure writer root."
            )
        payloads = _trajectory_evidence_payloads(result)
        files = _write_payloads(destination, payloads)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "problem_sha256": result["problem_sha256"],
            "implementation_sha256": _implementation_sha256(),
            "status": result["status"],
            "files": files,
        }
        _write_payload(destination / "trajectory_optimization_manifest.json", _json_bytes(manifest))
        return manifest

    if destination.exists():
        raise TrajectoryOptimizationError("Evidence output directory must not already exist.")
    mkdir_output(destination.parent)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        payloads = _trajectory_evidence_payloads(result)
        files = _write_payloads(temp, payloads)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "problem_sha256": result["problem_sha256"],
            "implementation_sha256": _implementation_sha256(),
            "status": result["status"],
            "files": files,
        }
        _write_payload(temp / "trajectory_optimization_manifest.json", _json_bytes(manifest))
        os.replace(temp, destination)
        return manifest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _replay_evidence(output_dir: Path | str) -> dict[str, Any]:
    active_writer = current_output_writer()
    directory = _output_path(output_dir, resolve=active_writer is None)
    physical_directory = (
        _physical_output_path(directory, active_writer) if active_writer is not None else directory
    )
    expected = {
        "normalized_problem.json",
        "trajectory_optimization_summary.json",
        "trajectory_optimization_trajectory.csv",
        "trajectory_optimization_constraints.csv",
        "trajectory_optimization_candidates.csv",
        "trajectory_optimization_robustness.csv",
        "trajectory_optimization_manifest.json",
    }
    actual = {path.name for path in physical_directory.iterdir()}
    if actual != expected or any(path.is_symlink() or not path.is_file() for path in physical_directory.iterdir()):
        raise TrajectoryOptimizationError("Evidence packet has unexpected, missing, non-file, or symlink entries.")
    if (
        any(path.stat().st_size > 50_000_000 for path in physical_directory.iterdir())
        or sum(path.stat().st_size for path in physical_directory.iterdir()) > 100_000_000
    ):
        raise TrajectoryOptimizationError("Evidence packet exceeds bounded replay size limits.")
    manifest = _load_output_json(directory / "trajectory_optimization_manifest.json", max_bytes=50_000_000)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise TrajectoryOptimizationError("Evidence manifest schema is invalid.")
    if manifest.get("implementation_sha256") != _implementation_sha256():
        raise TrajectoryOptimizationError("Evidence implementation identity does not match this runtime.")
    manifest_paths = [str(row.get("path", "")) for row in manifest.get("files", [])]
    if len(manifest_paths) != len(set(manifest_paths)) or set(manifest_paths) != expected - {
        "trajectory_optimization_manifest.json"
    }:
        raise TrajectoryOptimizationError("Evidence manifest inventory is incomplete or duplicated.")
    for row in manifest["files"]:
        path = directory / row["path"]
        content = read_output_bytes(path, max_bytes=50_000_000)
        if len(content) != row["bytes"] or sha256_output(path) != row["sha256"]:
            raise TrajectoryOptimizationError(f"Evidence digest mismatch for {path.name}.")
    problem = TrajectoryOptimizationProblem.from_mapping(
        _load_output_json(directory / "normalized_problem.json", max_bytes=50_000_000)
    )
    summary = _load_output_json(directory / "trajectory_optimization_summary.json", max_bytes=50_000_000)
    if summary.get("schema_version") != EVIDENCE_SCHEMA:
        raise TrajectoryOptimizationError("Evidence summary schema is invalid.")
    if _sha(problem.to_dict()) != summary["problem_sha256"]:
        raise TrajectoryOptimizationError("Normalized problem digest mismatch.")
    if _sha(summary.get("problem")) != summary["problem_sha256"]:
        raise TrajectoryOptimizationError("Evidence summary problem does not match its normalized problem.")
    if manifest.get("problem_sha256") != summary.get("problem_sha256") or manifest.get("status") != summary.get(
        "status"
    ):
        raise TrajectoryOptimizationError("Evidence manifest and summary claims do not agree.")
    converged = summary.get("status") == "converged" and summary.get("converged") is True
    if converged != bool(summary.get("solution") and summary.get("authoritative_repropagation")):
        raise TrajectoryOptimizationError("Evidence convergence, solution, and repropagation claims are inconsistent.")
    reproduced = _solve_trajectory_optimization(problem)
    if _sha(reproduced) != _sha(summary):
        raise TrajectoryOptimizationError("Fresh optimization replay does not reproduce the saved summary.")
    if summary.get("solution"):
        reprop = _evaluate_trajectory(problem, summary["solution"]["decision_values"])
        if _sha(reprop) != _sha(summary["authoritative_repropagation"]):
            raise TrajectoryOptimizationError("Authoritative replay does not reproduce saved evidence.")
        robustness = _robustness(problem, np.asarray(summary["solution"]["decision_values"], dtype=float))
        if _sha(robustness) != _sha(summary.get("robustness", [])):
            raise TrajectoryOptimizationError("Robustness replay does not reproduce saved evidence.")
        if _sha(_robustness_summary(robustness)) != _sha(summary.get("robustness_summary")):
            raise TrajectoryOptimizationError("Robustness summary does not reproduce saved evidence.")
    derived_rows = {
        "trajectory_optimization_trajectory.csv": (summary.get("authoritative_repropagation") or {}).get(
            "trajectory", []
        ),
        "trajectory_optimization_constraints.csv": (
            (summary.get("authoritative_repropagation") or {}).get("constraints") or {}
        ).get("rows", []),
        "trajectory_optimization_candidates.csv": summary.get("candidates", []),
        "trajectory_optimization_robustness.csv": summary.get("robustness", []),
    }
    for name, rows in derived_rows.items():
        if read_output_bytes(directory / name, max_bytes=50_000_000) != _csv_bytes(rows):
            raise TrajectoryOptimizationError(f"Derived evidence file {name} does not match the saved summary.")
    return {
        "status": "verified",
        "problem_sha256": summary["problem_sha256"],
        "solution_status": summary["status"],
        "file_count": len(expected),
    }


def _trajectory_evidence_payloads(result: Mapping[str, Any]) -> list[tuple[str, bytes]]:
    reprop = result.get("authoritative_repropagation") or {}
    return [
        ("normalized_problem.json", _json_bytes(result["problem"])),
        ("trajectory_optimization_summary.json", _json_bytes(dict(result))),
        ("trajectory_optimization_trajectory.csv", _csv_bytes(reprop.get("trajectory", []))),
        (
            "trajectory_optimization_constraints.csv",
            _csv_bytes((reprop.get("constraints") or {}).get("rows", [])),
        ),
        ("trajectory_optimization_candidates.csv", _csv_bytes(result.get("candidates", []))),
        ("trajectory_optimization_robustness.csv", _csv_bytes(result.get("robustness", []))),
    ]


def _output_path(value: Path | str, *, resolve: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve() if resolve else path


def _physical_output_path(logical_path: Path, writer: OutputWriter) -> Path:
    logical_root = _output_path(writer.logical_root, resolve=False)
    try:
        relative = logical_path.relative_to(logical_root)
    except ValueError as exc:
        raise TrajectoryOptimizationError(
            "Trajectory evidence path is outside the secure writer logical root."
        ) from exc
    return Path(writer.root).joinpath(relative)


def _write_payload(root: Path, payload: bytes) -> None:
    with staged_output_file(root) as staged:
        staged.write_bytes(payload)


def _write_payloads(root: Path, payloads: Sequence[tuple[str, bytes]]) -> list[dict[str, Any]]:
    mkdir_output(root)
    files = []
    for name, payload in sorted(payloads):
        path = root / name
        _write_payload(path, payload)
        content = read_output_bytes(path, max_bytes=50_000_000)
        if content != payload:
            raise TrajectoryOptimizationError(f"Evidence output changed while writing {name}.")
        files.append(
            {
                "path": name,
                "sha256": sha256_output(path),
                "bytes": len(content),
            }
        )
    return files


def _load_output_json(path: Path, *, max_bytes: int) -> Any:
    try:
        content = read_output_bytes(path, max_bytes=max_bytes)
        return json.loads(content.decode("utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrajectoryOptimizationError(f"Invalid evidence JSON: {exc}") from exc


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    write_output_bytes(path, _json_bytes(value))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_output_bytes(path, _csv_bytes(rows))


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b"\n"
    fields = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    return stream.getvalue().encode("utf-8")


def decision_contract(
    problem: TrajectoryOptimizationProblem,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the entitled Pro decision-vector contract."""

    require_pro_feature(FEATURE_TRAJECTORY_OPTIMIZATION)
    return _decision_contract(problem)


def evaluate_trajectory(
    problem: TrajectoryOptimizationProblem,
    physical: Sequence[float],
    *,
    initial_perturbation: np.ndarray | None = None,
    thrust_scale: float = 1.0,
) -> dict[str, Any]:
    """Evaluate one entitled Pro trajectory candidate."""

    require_pro_feature(FEATURE_TRAJECTORY_OPTIMIZATION)
    return _evaluate_trajectory(
        problem,
        physical,
        initial_perturbation=initial_perturbation,
        thrust_scale=thrust_scale,
    )


def finite_difference_derivatives(
    problem: TrajectoryOptimizationProblem | Mapping[str, Any], physical: Sequence[float]
) -> dict[str, Any]:
    """Return entitled bounded finite-difference derivatives."""

    require_pro_feature(FEATURE_TRAJECTORY_OPTIMIZATION)
    return _finite_difference_derivatives(problem, physical)


def solve_trajectory_optimization(
    problem: TrajectoryOptimizationProblem | Mapping[str, Any],
) -> dict[str, Any]:
    """Solve an entitled Pro trajectory-optimization problem."""

    require_pro_feature(FEATURE_TRAJECTORY_OPTIMIZATION)
    return _solve_trajectory_optimization(problem)


def write_evidence(
    result: Mapping[str, Any],
    output_dir: Path | str,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    """Write an entitled Pro evidence packet."""

    require_pro_feature(FEATURE_TRAJECTORY_OPTIMIZATION)
    return _write_evidence(result, output_dir, writer=writer)


def replay_evidence(output_dir: Path | str) -> dict[str, Any]:
    """Verify and repropagate an entitled Pro evidence packet."""

    require_pro_feature(FEATURE_TRAJECTORY_OPTIMIZATION)
    return _replay_evidence(output_dir)
