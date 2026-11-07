"""Deterministic optical collection-opportunity workflow and evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.analysis.collection_opportunity_resources import (
    CollectionResources,
    screen_collection_resources,
)
from sim.analysis.conjunction_geometry import StateHistory, _interpolate_prepared
from sim.analysis.conjunction_workflow import propagate_history
from sim.analysis.coverage_tasking import TaskOpportunity
from sim.analysis.event_refinement import availability_intervals, refine_availability_transitions
from sim.analysis.history_adapters import AnalysisHistory
from sim.analysis.optical_collection import (
    OPTICAL_COLLECTION_MODEL,
    CollectionConstraints,
    GroundTarget,
    OpticalPayload,
    evaluate_collection_sample,
    footprint_boundary_evidence,
    local_nadir_frame_sensor_from_eci,
    sensor_frame_and_gimbal_vector,
)
from sim.analysis.orbit_history_product import orbit_history_semantic_sha256
from sim.analysis.rust_numeric import AnalyticSunEvaluator, OpticalPointingEvaluator, require_native
from sim.analysis.rust_numeric import numeric_backend as validate_numeric_backend
from sim.analysis.trajectory_targeting import PropagationSettings
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.frame_time import TimeScale, epoch_julian_date, parse_epoch
from sim.utils.vector_math import dot3, norm3

COLLECTION_OPPORTUNITY_PROBLEM_SCHEMA = "oel.collection_opportunity_problem.v1"
COLLECTION_OPPORTUNITY_EVIDENCE_SCHEMA = "oel.collection_opportunity_evidence.v1"
MAX_COLLECTION_SAMPLES = 200_000
MAX_COLLECTION_DURATION_S = 7.0 * 86400.0
INTERIOR_DISCOVERY_SUBDIVISIONS = 16


class CollectionOpportunityError(ValueError):
    """Raised when a collection-opportunity problem is invalid."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CollectionOpportunityError(f"{field} must be a JSON object.")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise CollectionOpportunityError(f"{field} contains unknown fields: {', '.join(unknown)}.")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectionOpportunityError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise CollectionOpportunityError(f"{field} must be a finite number.")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CollectionOpportunityError(f"{field} must be an integer.")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CollectionOpportunityError(f"{field} must be a non-empty string.")
    return value.strip()


def _validated_propagation(value: Any) -> PropagationSettings:
    raw = {} if value is None else _mapping(value, "propagation")
    _reject_unknown(
        raw,
        {
            "step_s",
            "integrator",
            "force_model",
            "mu_km3_s2",
            "central_body_radius_km",
            "mass_kg",
            "event_time_tolerance_s",
            "event_value_tolerance",
            "event_max_iterations",
        },
        "propagation",
    )
    for key in (
        "step_s",
        "mu_km3_s2",
        "central_body_radius_km",
        "mass_kg",
        "event_time_tolerance_s",
        "event_value_tolerance",
    ):
        if key in raw:
            _finite_number(raw[key], f"propagation.{key}")
    if "event_max_iterations" in raw:
        _integer(raw["event_max_iterations"], "propagation.event_max_iterations")
    if "integrator" in raw and not isinstance(raw["integrator"], str):
        raise CollectionOpportunityError("propagation.integrator must be a string.")
    if "force_model" in raw:
        force_model = raw["force_model"]
        if not isinstance(force_model, list) or any(not isinstance(item, str) for item in force_model):
            raise CollectionOpportunityError("propagation.force_model must be an array of strings.")
    return PropagationSettings.from_mapping(raw)


@dataclass(frozen=True)
class SpacecraftSource:
    asset_id: str
    initial_state_eci_km_km_s: tuple[float, float, float, float, float, float]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SpacecraftSource:
        raw = _mapping(data, "spacecraft")
        _reject_unknown(raw, {"asset_id", "initial_state_eci_km_km_s"}, "spacecraft")
        asset_id = _required_text(raw.get("asset_id"), "spacecraft.asset_id")
        state_raw = raw.get("initial_state_eci_km_km_s")
        if not isinstance(state_raw, list) or len(state_raw) != 6:
            raise CollectionOpportunityError(
                "spacecraft.initial_state_eci_km_km_s must contain six finite numbers."
            )
        state = np.asarray(
            [
                _finite_number(value, f"spacecraft.initial_state_eci_km_km_s[{index}]")
                for index, value in enumerate(state_raw)
            ],
            dtype=float,
        )
        if state.shape != (6,) or not np.all(np.isfinite(state)) or float(np.linalg.norm(state[:3])) <= 0.0:
            raise CollectionOpportunityError(
                "spacecraft.initial_state_eci_km_km_s must contain six finite values and nonzero position."
            )
        return cls(asset_id, tuple(float(value) for value in state))


@dataclass(frozen=True)
class CollectionOpportunityProblem:
    name: str
    epoch_utc: str
    spacecraft: SpacecraftSource
    target: GroundTarget
    sensor: OpticalPayload
    constraints: CollectionConstraints
    resources: CollectionResources
    duration_s: float
    propagation: PropagationSettings
    transition_time_tolerance_s: float
    transition_max_iterations: int
    schema_version: str = COLLECTION_OPPORTUNITY_PROBLEM_SCHEMA
    numeric_backend: str = "rust"

    def __post_init__(self) -> None:
        try:
            validate_numeric_backend(self.numeric_backend)
        except ValueError as exc:
            raise CollectionOpportunityError(str(exc)) from exc

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CollectionOpportunityProblem:
        raw = _mapping(data, "collection problem")
        _reject_unknown(
            raw,
            {
                "schema_version",
                "numeric_backend",
                "name",
                "epoch_utc",
                "duration_s",
                "spacecraft",
                "target",
                "sensor",
                "constraints",
                "resources",
                "propagation",
                "transition_time_tolerance_s",
                "transition_max_iterations",
            },
            "collection problem",
        )
        schema = _required_text(
            raw.get("schema_version", COLLECTION_OPPORTUNITY_PROBLEM_SCHEMA), "schema_version"
        )
        if schema != COLLECTION_OPPORTUNITY_PROBLEM_SCHEMA:
            raise CollectionOpportunityError(f"Unsupported collection schema {schema!r}.")
        epoch_text = _required_text(raw.get("epoch_utc"), "epoch_utc")
        parse_epoch(epoch_text, TimeScale.UTC)
        duration = _finite_number(raw.get("duration_s", 0.0), "duration_s")
        propagation = _validated_propagation(raw.get("propagation"))
        tolerance = _finite_number(raw.get("transition_time_tolerance_s", 0.1), "transition_time_tolerance_s")
        iterations = _integer(raw.get("transition_max_iterations", 60), "transition_max_iterations")
        if not math.isfinite(duration) or not propagation.step_s < duration <= MAX_COLLECTION_DURATION_S:
            raise CollectionOpportunityError(
                f"duration_s must exceed propagation.step_s and not exceed {MAX_COLLECTION_DURATION_S}."
            )
        propagation_intervals = int(math.ceil(duration / propagation.step_s))
        sample_count = propagation_intervals + 1
        minimum_discovery_samples = 2 * propagation_intervals + 1
        if sample_count > MAX_COLLECTION_SAMPLES or minimum_discovery_samples > MAX_COLLECTION_SAMPLES:
            raise CollectionOpportunityError(
                f"The requested cadence plus required interior discovery would exceed the "
                f"{MAX_COLLECTION_SAMPLES} sample limit."
            )
        if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= propagation.step_s:
            raise CollectionOpportunityError(
                "transition_time_tolerance_s must be positive and smaller than propagation.step_s."
            )
        if iterations <= 0:
            raise CollectionOpportunityError("transition_max_iterations must be positive.")
        resources = CollectionResources.from_mapping(raw.get("resources"))
        if any(item.start_s < 0.0 or item.end_s > duration for item in resources.downlink_windows):
            raise CollectionOpportunityError("Downlink windows must lie inside [0, duration_s].")
        return cls(
            numeric_backend=raw.get("numeric_backend", "rust"),
            schema_version=schema,
            name=_required_text(raw.get("name", "collection_opportunity"), "name"),
            epoch_utc=epoch_text,
            spacecraft=SpacecraftSource.from_mapping(raw.get("spacecraft", {})),
            target=GroundTarget.from_mapping(raw.get("target", {})),
            sensor=OpticalPayload.from_mapping(raw.get("sensor", {})),
            constraints=CollectionConstraints.from_mapping(raw.get("constraints")),
            resources=resources,
            duration_s=duration,
            propagation=propagation,
            transition_time_tolerance_s=tolerance,
            transition_max_iterations=iterations,
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "name": self.name,
            "epoch_utc": self.epoch_utc,
            "duration_s": self.duration_s,
            "spacecraft": {
                "asset_id": self.spacecraft.asset_id,
                "initial_state_eci_km_km_s": list(self.spacecraft.initial_state_eci_km_km_s),
            },
            "target": asdict(self.target),
            "sensor": self.sensor.to_dict(),
            "constraints": self.constraints.to_dict(),
            "resources": self.resources.to_dict(),
            "propagation": self.propagation.to_dict(),
            "transition_time_tolerance_s": self.transition_time_tolerance_s,
            "transition_max_iterations": self.transition_max_iterations,
        }
        value["numeric_backend"] = self.numeric_backend
        return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _frame_context(problem: CollectionOpportunityProblem) -> FrameContext:
    epoch = parse_epoch(problem.epoch_utc, TimeScale.UTC)
    return FrameContext(
        model="simple_gmst",
        jd_utc_start=epoch_julian_date(epoch, TimeScale.UTC),
        time_scale_model="utc_leap_second_aware_epoch",
        source="collection_opportunity_problem",
    )


def _angle_rate(left: np.ndarray, right: np.ndarray, duration_s: float) -> float:
    if duration_s <= 0.0:
        return 0.0
    angle = math.atan2(norm3(np.cross(left, right)), dot3(left, right))
    return angle / duration_s


def _interior_discovery_times(history_times_s: Sequence[float]) -> tuple[np.ndarray, int]:
    """Return a bounded sub-cadence grid and its subdivisions per propagation interval."""

    base = np.asarray(history_times_s, dtype=float)
    intervals = base.size - 1
    maximum_subdivisions = max(1, (MAX_COLLECTION_SAMPLES - 1) // intervals)
    subdivisions = min(INTERIOR_DISCOVERY_SUBDIVISIONS, maximum_subdivisions)
    pieces = [
        np.linspace(left, right, subdivisions + 1, endpoint=True)[:-1]
        for left, right in zip(base[:-1], base[1:], strict=True)
    ]
    return np.concatenate((*pieces, base[-1:])), subdivisions


class _CollectionEvaluator:
    def __init__(
        self,
        problem: CollectionOpportunityProblem,
        history: StateHistory,
        frame_context: FrameContext,
    ) -> None:
        self.problem = problem
        self.history = history
        self.frame_context = frame_context
        self._times, self._states = history.arrays()
        self._sun = AnalyticSunEvaluator(problem.numeric_backend, "analytic_enhanced")
        self._native_pointing = OpticalPointingEvaluator() if problem.numeric_backend == "rust" else None
        self.start_s = float(history.times_s[0])
        self.stop_s = float(history.times_s[-1])

    def _pointing(self, time_s: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        state = _interpolate_prepared(self.history, self._times, self._states, float(time_s))
        ecef_from_eci = eci_to_ecef_rotation_context(float(time_s), self.frame_context)
        target_eci = ecef_from_eci.T @ self.problem.target.ecef_km
        pointing = sensor_frame_and_gimbal_vector if self._native_pointing is None else self._native_pointing.pointing
        sensor_from_eci, gimbal_vector, gimbal_angle = pointing(
            state,
            target_eci,
            pointing_mode=self.problem.sensor.pointing_mode,
        )
        sensor_from_ecef = sensor_from_eci @ ecef_from_eci.T
        observer_ecef = ecef_from_eci @ state[:3]
        sun_eci = self._sun.at_jd(self.frame_context.jd_utc_start + float(time_s) / 86400.0)
        return state, gimbal_vector, gimbal_angle, sensor_from_ecef, observer_ecef, ecef_from_eci @ sun_eci

    def _gimbal_vector(self, time_s: float) -> np.ndarray:
        state = _interpolate_prepared(self.history, self._times, self._states, time_s)
        rotation = eci_to_ecef_rotation_context(time_s, self.frame_context)
        target = rotation.T @ self.problem.target.ecef_km
        mode = self.problem.sensor.pointing_mode
        if self._native_pointing is not None:
            return self._native_pointing.gimbal(state, target, pointing_mode=mode)
        local = local_nadir_frame_sensor_from_eci(state)
        if mode == "nadir_fixed":
            return np.array([0.0, 0.0, 1.0])
        los = target - state[:3]
        return local @ (los / np.linalg.norm(los))

    def required_slew_rate(self, time_s: float) -> float:
        query = float(time_s)
        span = min(0.5, 0.05 * self.problem.propagation.step_s)
        left = max(self.start_s, query - span)
        right = min(self.stop_s, query + span)
        if not right > left:
            return 0.0
        left_vector = self._gimbal_vector(left)
        right_vector = self._gimbal_vector(right)
        return _angle_rate(left_vector, right_vector, right - left)

    def sample(self, time_s: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        state, _gimbal_vector, gimbal_angle, sensor_from_ecef, observer_ecef, sun_ecef = self._pointing(time_s)
        sun_eci = eci_to_ecef_rotation_context(float(time_s), self.frame_context).T @ sun_ecef
        geometry = resolve_srp_geometry(state[:3], float(time_s), {"sun_pos_eci_km": sun_eci})
        illumination = srp_shadow_factor(
            state[:3],
            float(time_s),
            {"sun_pos_eci_km": sun_eci, "srp_shadow_model": "conical"},
            srp_geometry=geometry,
        )
        row = evaluate_collection_sample(
            observer_ecef_km=observer_ecef,
            target=self.problem.target,
            dcm_sensor_from_ecef=sensor_from_ecef,
            payload=self.problem.sensor,
            constraints=self.problem.constraints,
            sun_ecef_km=sun_ecef,
            gimbal_off_nadir_rad=gimbal_angle,
            required_slew_rate_rad_s=self.required_slew_rate(time_s),
            spacecraft_illumination_fraction=illumination,
        )
        row["time_s"] = float(time_s)
        return row, observer_ecef, sensor_from_ecef

    def availability(self, time_s: float) -> tuple[bool, str]:
        row, _observer, _rotation = self.sample(time_s)
        return bool(row["available"]), str(row["reason"])


def _window_sample_rows(rows: Sequence[Mapping[str, Any]], start_s: float, end_s: float) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if start_s - 1.0e-12 <= float(row["time_s"]) <= end_s + 1.0e-12]
    if selected:
        return selected
    return [min(rows, key=lambda row: abs(float(row["time_s"]) - 0.5 * (start_s + end_s)))]


def assess_collection_opportunities(
    problem: CollectionOpportunityProblem | Mapping[str, Any],
    *,
    orbit_history: AnalysisHistory | None = None,
) -> dict[str, Any]:
    parsed = (
        problem
        if isinstance(problem, CollectionOpportunityProblem)
        else CollectionOpportunityProblem.from_mapping(problem)
    )
    if parsed.numeric_backend == "rust":
        require_native("optical_pointing", "optical_gimbal_vector", "ONPEnvironmentContext.analytic_sun_km")
    normalized_problem = parsed.to_dict()
    problem_sha256 = _canonical_sha256(normalized_problem)
    if orbit_history is None:
        history = propagate_history(
            parsed.spacecraft.initial_state_eci_km_km_s,
            parsed.duration_s,
            parsed.propagation,
        )
    else:
        frame_epoch = _frame_context(parsed).jd_utc_start
        if (orbit_history.object_id != parsed.spacecraft.asset_id
                or abs(orbit_history.initial_jd_utc - frame_epoch) > 1.0e-12
                or orbit_history.times_s[0] != 0.0
                or orbit_history.times_s[-1] < parsed.duration_s):
            raise CollectionOpportunityError("Orbit history asset, epoch, or horizon differs from collection problem.")
        initial = np.hstack((orbit_history.position_eci_km[0], orbit_history.velocity_eci_km_s[0]))
        if not np.array_equal(initial, np.asarray(parsed.spacecraft.initial_state_eci_km_km_s)):
            raise CollectionOpportunityError("Collection initial state differs from retained orbit history.")
        relevant = orbit_history.times_s <= parsed.duration_s
        if orbit_history.times_s[relevant][-1] != parsed.duration_s:
            raise CollectionOpportunityError("Collection duration must be a retained orbit-history sample.")
        history = StateHistory.from_arrays(
            orbit_history.times_s[relevant],
            np.hstack((orbit_history.position_eci_km[relevant], orbit_history.velocity_eci_km_s[relevant])),
        )
    frame_context = _frame_context(parsed)
    evaluator = _CollectionEvaluator(parsed, history, frame_context)
    discovery_times, discovery_subdivisions = _interior_discovery_times(history.times_s)
    sample_rows: list[dict[str, Any]] = []
    for time_s in discovery_times:
        row, _observer, _rotation = evaluator.sample(float(time_s))
        sample_rows.append(row)
    times = discovery_times
    available = np.asarray([row["available"] for row in sample_rows], dtype=bool)
    reasons = tuple(str(row["reason"]) for row in sample_rows)
    transitions = refine_availability_transitions(
        times,
        available,
        reasons,
        evaluator_at_time=evaluator.availability,
        time_tolerance_s=parsed.transition_time_tolerance_s,
        max_iterations=parsed.transition_max_iterations,
    )
    intervals = availability_intervals(times, available, reasons, transitions=transitions)
    candidates: list[dict[str, Any]] = []
    task_opportunities: list[dict[str, Any]] = []
    for interval in intervals:
        collection_start = interval.start_s + parsed.sensor.settling_time_s
        collection_end = interval.end_s
        duration = max(0.0, collection_end - collection_start)
        if duration > 0.0:
            supporting = _window_sample_rows(sample_rows, collection_start, collection_end)
            midpoint = 0.5 * (collection_start + collection_end)
        else:
            supporting = _window_sample_rows(sample_rows, interval.start_s, interval.end_s)
            midpoint = 0.5 * (interval.start_s + interval.end_s)
        midpoint_row, midpoint_observer, midpoint_rotation = evaluator.sample(midpoint)
        footprint = footprint_boundary_evidence(
            observer_ecef_km=midpoint_observer,
            dcm_sensor_from_ecef=midpoint_rotation,
            payload=parsed.sensor,
            target=parsed.target,
        )
        generated_bytes = duration * parsed.sensor.data_generation_rate_bps / 8.0
        resource_screen = screen_collection_resources(
            parsed.resources,
            collection_end_s=collection_end,
            generated_data_bytes=generated_bytes,
        )
        duration_pass = duration + 1.0e-12 >= parsed.sensor.minimum_collection_duration_s
        disposition = (
            "accepted"
            if duration_pass and resource_screen["resource_feasible"]
            else "insufficient_collection_duration"
            if not duration_pass
            else str(resource_screen["reason"])
        )
        candidate = {
            "opportunity_id": f"{parsed.name}:{len(candidates):04d}",
            "raw_geometry_start_s": interval.start_s,
            "raw_geometry_end_s": interval.end_s,
            "collection_start_s": collection_start,
            "collection_end_s": collection_end,
            "collection_duration_s": duration,
            "settling_time_s": parsed.sensor.settling_time_s,
            "accepted": disposition == "accepted",
            "disposition": disposition,
            "generated_data_bytes": generated_bytes,
            "minimum_effective_resolution_m": min(float(row["effective_resolution_m"]) for row in supporting),
            "maximum_effective_resolution_m": max(float(row["effective_resolution_m"]) for row in supporting),
            "maximum_off_nadir_angle_deg": max(float(row["off_nadir_angle_deg"]) for row in supporting),
            "maximum_incidence_angle_deg": max(float(row["incidence_angle_deg"]) for row in supporting),
            "minimum_sun_elevation_deg": min(float(row["sun_elevation_deg"]) for row in supporting),
            "maximum_required_slew_rate_deg_s": max(float(row["required_slew_rate_deg_s"]) for row in supporting),
            "midpoint_sample": midpoint_row,
            "midpoint_footprint": footprint,
            "resource_screen": resource_screen,
            "boundary_evidence": {
                "start_censored": interval.start_censored,
                "end_censored": interval.end_censored,
                "acquisition_disposition": interval.acquisition_disposition,
                "loss_disposition": interval.loss_disposition,
                "acquisition_reason": interval.acquisition_reason,
                "loss_reason": interval.loss_reason,
            },
        }
        candidates.append(candidate)
        if candidate["accepted"]:
            pointing_state = _interpolate_prepared(history, evaluator._times, evaluator._states, midpoint)
            target_eci = eci_to_ecef_rotation_context(midpoint, frame_context).T @ parsed.target.ecef_km
            pointing = target_eci - pointing_state[:3]
            pointing /= float(np.linalg.norm(pointing))
            task = TaskOpportunity(
                opportunity_id=candidate["opportunity_id"],
                source_product_sha256=problem_sha256,
                asset_id=parsed.spacecraft.asset_id,
                kind="observation",
                start_s=collection_start,
                end_s=collection_end,
                objective_value=duration,
                storage_delta_bytes=generated_bytes,
                energy_cost_wh=0.0,
                pointing_unit_eci=tuple(float(value) for value in pointing),
                target_id=parsed.target.target_id,
            )
            task_opportunities.append(asdict(task))
    reason_counts = {reason: reasons.count(reason) for reason in sorted(set(reasons))}
    evidence = {
        "schema_version": COLLECTION_OPPORTUNITY_EVIDENCE_SCHEMA,
        "problem_name": parsed.name,
        "problem_sha256": problem_sha256,
        "normalized_problem": normalized_problem,
        "status": "completed",
        "model": OPTICAL_COLLECTION_MODEL,
        "frame_time_provenance": frame_context.metadata(sample_t_s=0.0),
        "solar_ephemeris_provenance": {
            "provider": "oel_analytic_enhanced",
            "epoch_utc": parsed.epoch_utc,
            "qualification": "bounded analytic Sun geometry; not precision ephemeris",
        },
        "sample_ledger": sample_rows,
        "transitions": [asdict(item) for item in transitions],
        "opportunity_candidates": candidates,
        "task_opportunities": task_opportunities,
        "summary": {
            "sample_count": len(sample_rows),
            "available_sample_count": int(np.count_nonzero(available)),
            "reason_counts": reason_counts,
            "raw_interval_count": len(intervals),
            "accepted_opportunity_count": len(task_opportunities),
        },
        "resources": {
            "propagation_samples": len(history.times_s),
            "interior_discovery_samples": len(discovery_times),
            "interior_discovery_subdivisions_per_propagation_interval": discovery_subdivisions,
            "maximum_interior_discovery_step_s": float(np.max(np.diff(discovery_times))),
            "transition_evaluations_are_bounded_by": len(transitions) * parsed.transition_max_iterations,
            "footprint_boundary_rays_per_candidate": 4 * parsed.sensor.boundary_samples_per_edge,
        },
        "limitations": [
            "The v1 workflow evaluates one WGS84 surface target and one optical payload over one deterministic ONP history.",
            "The default Earth rotation is simple GMST and the Sun provider is bounded analytic; higher-accuracy EOP/SPICE evidence is a later slice.",
            "Target-track pointing is ideal within declared gimbal angle, slew-rate, and settling limits; actuator torque and flexible-body dynamics are not modeled.",
            "Diffraction, detector sampling, and tangent-plane footprint metrics are transparent first-order calculations, not calibrated image-quality claims.",
            "Downlink windows are content-identified external evidence supplied by the analyst; this workflow does not invent RF availability.",
            "Accepted opportunities are engineering evidence, not collection authorization or operational availability.",
            "Opportunity discovery is bounded by the recorded interior-discovery cadence; windows narrower than that cadence require a smaller propagation step.",
        ],
    }
    if orbit_history is not None:
        evidence["orbit_history_semantic_sha256"] = orbit_history_semantic_sha256(orbit_history)
        evidence["orbit_history_derivation"] = {
            "method": "exact_prefix_samples_with_existing_hermite_refinement",
            "sample_count": len(history.times_s),
            "end_s": parsed.duration_s,
        }
        evidence["limitations"].append(
            "Collection geometry uses the retained ECI history; declared propagation settings were not executed in this mode."
        )
    return evidence


def write_collection_evidence(evidence: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(evidence), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return target


__all__ = [
    "COLLECTION_OPPORTUNITY_EVIDENCE_SCHEMA", "COLLECTION_OPPORTUNITY_PROBLEM_SCHEMA", "CollectionOpportunityError",
    "CollectionOpportunityProblem", "SpacecraftSource", "assess_collection_opportunities", "write_collection_evidence",
]
