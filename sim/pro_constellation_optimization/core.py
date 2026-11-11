"""Entitled deterministic constellation and ground-network design search."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sim.analysis.constellation_design import (
    CONSTELLATION_DESIGN_PROBLEM_SCHEMA,
    MAX_PUBLIC_GROUND_SITES,
    MAX_PUBLIC_SELECTED_GROUND_SITES,
    ConstellationDesignProblem,
    GroundSite,
    _require_numeric_backend,
    solve_constellation_design,
)
from sim.pro_features import FEATURE_CONSTELLATION_OPTIMIZATION, require_pro_feature
from sim.utils.io import SafeReadError, read_regular_file_nofollow

PROBLEM_SCHEMA = "oel.pro_constellation_optimization_problem.v1"
EVIDENCE_SCHEMA = "oel.pro_constellation_optimization_evidence.v1"
MANIFEST_SCHEMA = "oel.pro_constellation_optimization_manifest.v1"
CHECKPOINT_SCHEMA = "oel.pro_constellation_optimization_checkpoint.v1"
ALGORITHM_ID = "deterministic_grid_or_seeded_reservoir_pareto.v1"

MAX_SEARCH_VALUES_PER_FIELD = 16
MAX_GENERATION_COMBINATIONS = 1_000_000
MAX_GENERATED_CANDIDATES = 100_000
MAX_AUTHORITATIVE_EVALUATIONS = 64
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_EVIDENCE_BYTES = 160 * 1024 * 1024


class ProConstellationOptimizationError(ValueError):
    """Raised when a Pro constellation search or evidence packet is invalid."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProConstellationOptimizationError(f"{field} must be a JSON object.")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ProConstellationOptimizationError(f"{field} must be a JSON array.")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ProConstellationOptimizationError(f"{field} is missing required fields: {', '.join(missing)}.")
    if unknown:
        raise ProConstellationOptimizationError(f"{field} contains unknown fields: {', '.join(unknown)}.")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProConstellationOptimizationError(f"{field} must be a non-empty string.")
    normalized = value.strip()
    if len(normalized) > 128:
        raise ProConstellationOptimizationError(f"{field} exceeds the 128-character bound.")
    return normalized


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ProConstellationOptimizationError(f"{field} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProConstellationOptimizationError(f"{field} must be finite.") from exc
    if not math.isfinite(result):
        raise ProConstellationOptimizationError(f"{field} must be finite.")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ProConstellationOptimizationError(f"{field} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProConstellationOptimizationError(f"{field} must be an integer.") from exc
    if result != value:
        raise ProConstellationOptimizationError(f"{field} must be an integer.")
    return result


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProConstellationOptimizationError(f"Evidence must contain finite JSON values: {exc}") from exc


def _semantic_sha256(value: Any) -> str:
    try:
        content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProConstellationOptimizationError(f"Value must contain finite JSON values: {exc}") from exc
    return hashlib.sha256(content).hexdigest()


def _normalized_values(
    value: Any,
    field: str,
    *,
    integer: bool,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[int | float, ...]:
    raw = _sequence(value, field)
    if not 1 <= len(raw) <= MAX_SEARCH_VALUES_PER_FIELD:
        raise ProConstellationOptimizationError(f"{field} must contain 1-{MAX_SEARCH_VALUES_PER_FIELD} values.")
    normalized: list[int | float] = []
    for index, item in enumerate(raw):
        parsed: int | float = _integer(item, f"{field}[{index}]") if integer else _finite(item, f"{field}[{index}]")
        if minimum is not None and parsed < minimum:
            raise ProConstellationOptimizationError(f"{field}[{index}] must be at least {minimum}.")
        if maximum is not None and parsed > maximum:
            raise ProConstellationOptimizationError(f"{field}[{index}] must not exceed {maximum}.")
        normalized.append(parsed)
    return tuple(sorted(set(normalized)))


@dataclass(frozen=True)
class ProConstellationOptimizationProblem:
    analysis_id: str
    public_evaluation: dict[str, Any]
    design_space: dict[str, Any]
    search: dict[str, Any]
    cost_model: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProConstellationOptimizationProblem:
        raw = _mapping(value, "Pro constellation-optimization problem")
        _exact_fields(
            raw,
            {"schema_version", "analysis_id", "public_evaluation", "design_space", "search", "cost_model"},
            "Pro constellation-optimization problem",
        )
        if raw["schema_version"] != PROBLEM_SCHEMA:
            raise ProConstellationOptimizationError(f"Unsupported problem schema: {raw['schema_version']!r}.")
        analysis_id = _text(raw["analysis_id"], "analysis_id")

        evaluation_raw = _mapping(raw["public_evaluation"], "public_evaluation")
        evaluation_fields = {
            "initial_jd_utc",
            "duration_s",
            "sample_step_s",
            "propagation",
            "coverage",
            "ground_sites",
            "link_budget",
            "objective",
        }
        _exact_fields(evaluation_raw, evaluation_fields, "public_evaluation")
        sites = tuple(
            GroundSite.from_mapping(item)
            for item in _sequence(evaluation_raw["ground_sites"], "public_evaluation.ground_sites")
        )
        if not 1 <= len(sites) <= MAX_PUBLIC_GROUND_SITES:
            raise ProConstellationOptimizationError(
                f"public_evaluation.ground_sites must contain 1-{MAX_PUBLIC_GROUND_SITES} sites."
            )
        site_ids = [site.site_id for site in sites]
        if len(site_ids) != len(set(site_ids)):
            raise ProConstellationOptimizationError("public_evaluation ground-site IDs must be unique.")
        public_evaluation = {
            "initial_jd_utc": _finite(evaluation_raw["initial_jd_utc"], "public_evaluation.initial_jd_utc"),
            "duration_s": _finite(evaluation_raw["duration_s"], "public_evaluation.duration_s"),
            "sample_step_s": _finite(evaluation_raw["sample_step_s"], "public_evaluation.sample_step_s"),
            "propagation": dict(_mapping(evaluation_raw["propagation"], "public_evaluation.propagation")),
            "coverage": dict(_mapping(evaluation_raw["coverage"], "public_evaluation.coverage")),
            "ground_sites": [asdict(site) for site in sorted(sites, key=lambda item: item.site_id)],
            "link_budget": dict(_mapping(evaluation_raw["link_budget"], "public_evaluation.link_budget")),
            "objective": dict(_mapping(evaluation_raw["objective"], "public_evaluation.objective")),
        }

        space_raw = _mapping(raw["design_space"], "design_space")
        space_fields = {
            "patterns",
            "satellite_counts",
            "plane_counts",
            "phasing_values",
            "altitude_km_values",
            "inclination_deg_values",
            "raan_start_deg_values",
            "initial_phase_deg_values",
            "shell_raan_span_deg_values",
            "ground_site_pool",
            "minimum_ground_sites",
            "maximum_ground_sites",
        }
        _exact_fields(space_raw, space_fields, "design_space")
        patterns = tuple(
            sorted(
                {
                    _text(item, "design_space.patterns item").lower()
                    for item in _sequence(space_raw["patterns"], "design_space.patterns")
                }
            )
        )
        if not patterns or set(patterns) - {"walker_delta", "walker_star", "shell"}:
            raise ProConstellationOptimizationError(
                "design_space.patterns must contain walker_delta, walker_star, or shell."
            )
        site_pool = tuple(
            sorted(
                {
                    _text(item, "design_space.ground_site_pool item")
                    for item in _sequence(space_raw["ground_site_pool"], "design_space.ground_site_pool")
                }
            )
        )
        if not site_pool or set(site_pool) - set(site_ids):
            raise ProConstellationOptimizationError(
                "design_space.ground_site_pool must contain known public_evaluation ground-site IDs."
            )
        minimum_sites = _integer(space_raw["minimum_ground_sites"], "design_space.minimum_ground_sites")
        maximum_sites = _integer(space_raw["maximum_ground_sites"], "design_space.maximum_ground_sites")
        if not 1 <= minimum_sites <= maximum_sites <= min(len(site_pool), MAX_PUBLIC_SELECTED_GROUND_SITES):
            raise ProConstellationOptimizationError(
                f"Ground-site selection bounds must satisfy 1 <= minimum <= maximum <= {MAX_PUBLIC_SELECTED_GROUND_SITES}."
            )
        design_space = {
            "patterns": list(patterns),
            "satellite_counts": list(
                _normalized_values(
                    space_raw["satellite_counts"], "design_space.satellite_counts", integer=True, minimum=2, maximum=24
                )
            ),
            "plane_counts": list(
                _normalized_values(
                    space_raw["plane_counts"], "design_space.plane_counts", integer=True, minimum=1, maximum=24
                )
            ),
            "phasing_values": list(
                _normalized_values(
                    space_raw["phasing_values"], "design_space.phasing_values", integer=True, minimum=0, maximum=23
                )
            ),
            "altitude_km_values": list(
                _normalized_values(
                    space_raw["altitude_km_values"],
                    "design_space.altitude_km_values",
                    integer=False,
                    minimum=100.0,
                    maximum=100_000.0,
                )
            ),
            "inclination_deg_values": list(
                _normalized_values(
                    space_raw["inclination_deg_values"],
                    "design_space.inclination_deg_values",
                    integer=False,
                    minimum=0.0,
                    maximum=180.0,
                )
            ),
            "raan_start_deg_values": list(
                _normalized_values(
                    space_raw["raan_start_deg_values"], "design_space.raan_start_deg_values", integer=False
                )
            ),
            "initial_phase_deg_values": list(
                _normalized_values(
                    space_raw["initial_phase_deg_values"], "design_space.initial_phase_deg_values", integer=False
                )
            ),
            "shell_raan_span_deg_values": list(
                _normalized_values(
                    space_raw["shell_raan_span_deg_values"],
                    "design_space.shell_raan_span_deg_values",
                    integer=False,
                    minimum=1.0e-12,
                    maximum=360.0,
                )
            ),
            "ground_site_pool": list(site_pool),
            "minimum_ground_sites": minimum_sites,
            "maximum_ground_sites": maximum_sites,
        }
        site_group_count = sum(math.comb(len(site_pool), count) for count in range(minimum_sites, maximum_sites + 1))
        pattern_span_count = sum(
            len(design_space["shell_raan_span_deg_values"]) if pattern == "shell" else 1 for pattern in patterns
        )
        raw_combination_count = (
            math.prod(
                len(design_space[field])
                for field in (
                    "satellite_counts",
                    "plane_counts",
                    "phasing_values",
                    "altitude_km_values",
                    "inclination_deg_values",
                    "raan_start_deg_values",
                    "initial_phase_deg_values",
                )
            )
            * site_group_count
            * pattern_span_count
        )
        if raw_combination_count > MAX_GENERATION_COMBINATIONS:
            raise ProConstellationOptimizationError(
                f"Raw design-space product exceeds the {MAX_GENERATION_COMBINATIONS}-combination generation bound."
            )

        search_raw = _mapping(raw["search"], "search")
        _exact_fields(search_raw, {"method", "seed", "max_generated_candidates", "max_evaluations"}, "search")
        method = _text(search_raw["method"], "search.method").lower()
        if method not in {"exhaustive", "seeded_sample"}:
            raise ProConstellationOptimizationError("search.method must be exhaustive or seeded_sample.")
        seed = _integer(search_raw["seed"], "search.seed")
        max_generated = _integer(search_raw["max_generated_candidates"], "search.max_generated_candidates")
        max_evaluations = _integer(search_raw["max_evaluations"], "search.max_evaluations")
        if not 1 <= max_generated <= MAX_GENERATED_CANDIDATES:
            raise ProConstellationOptimizationError(
                f"search.max_generated_candidates must be within [1, {MAX_GENERATED_CANDIDATES}]."
            )
        if not 1 <= max_evaluations <= MAX_AUTHORITATIVE_EVALUATIONS:
            raise ProConstellationOptimizationError(
                f"search.max_evaluations must be within [1, {MAX_AUTHORITATIVE_EVALUATIONS}]."
            )
        search = {
            "method": method,
            "seed": seed,
            "max_generated_candidates": max_generated,
            "max_evaluations": max_evaluations,
        }

        cost_raw = _mapping(raw["cost_model"], "cost_model")
        cost_fields = {
            "cost_unit",
            "fixed_cost",
            "satellite_unit_cost",
            "ground_site_costs",
            "maximum_total_cost",
            "cost_penalty_per_unit",
        }
        _exact_fields(cost_raw, cost_fields, "cost_model")
        site_costs_raw = _mapping(cost_raw["ground_site_costs"], "cost_model.ground_site_costs")
        if set(site_costs_raw) != set(site_ids):
            raise ProConstellationOptimizationError(
                "cost_model.ground_site_costs must contain exactly every public_evaluation site ID."
            )
        site_costs = {
            site_id: _finite(site_costs_raw[site_id], f"ground_site_costs.{site_id}") for site_id in sorted(site_ids)
        }
        fixed_cost = _finite(cost_raw["fixed_cost"], "cost_model.fixed_cost")
        satellite_cost = _finite(cost_raw["satellite_unit_cost"], "cost_model.satellite_unit_cost")
        cost_penalty = _finite(cost_raw["cost_penalty_per_unit"], "cost_model.cost_penalty_per_unit")
        if (
            fixed_cost < 0.0
            or satellite_cost < 0.0
            or cost_penalty < 0.0
            or any(value < 0.0 for value in site_costs.values())
        ):
            raise ProConstellationOptimizationError("All cost-model amounts and penalties must be nonnegative.")
        maximum_cost_raw = cost_raw["maximum_total_cost"]
        maximum_cost = None if maximum_cost_raw is None else _finite(maximum_cost_raw, "cost_model.maximum_total_cost")
        if maximum_cost is not None and maximum_cost < 0.0:
            raise ProConstellationOptimizationError("cost_model.maximum_total_cost must be nonnegative or null.")
        cost_model = {
            "cost_unit": _text(cost_raw["cost_unit"], "cost_model.cost_unit"),
            "fixed_cost": fixed_cost,
            "satellite_unit_cost": satellite_cost,
            "ground_site_costs": site_costs,
            "maximum_total_cost": maximum_cost,
            "cost_penalty_per_unit": cost_penalty,
        }

        result = cls(analysis_id, public_evaluation, design_space, search, cost_model)
        selected, generated_count = _select_candidate_specs(result)
        if not selected:
            raise ProConstellationOptimizationError("The declared design space contains no valid candidates.")
        if method == "exhaustive" and generated_count > max_evaluations:
            raise ProConstellationOptimizationError(
                "Exhaustive search requires max_evaluations to cover every valid generated candidate."
            )
        public_problem = _validated_public_problem(result, selected[0])
        # Canonicalize the optional selection without changing older public
        # evaluation payload spelling or normalized default evidence.
        propagation = result.public_evaluation["propagation"]
        propagation["numeric_backend"] = public_problem.propagation.get("numeric_backend", "rust")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> ProConstellationOptimizationProblem:
        source = Path(path).expanduser()
        try:
            content = read_regular_file_nofollow(source, min_bytes=1, max_bytes=MAX_ARTIFACT_BYTES)
            payload = _strict_json(content, "problem JSON")
        except (OSError, SafeReadError) as exc:
            raise ProConstellationOptimizationError(f"Could not read problem JSON {source}: {exc}") from exc
        return cls.from_mapping(_mapping(payload, "problem JSON"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROBLEM_SCHEMA,
            "analysis_id": self.analysis_id,
            "public_evaluation": json.loads(json.dumps(self.public_evaluation)),
            "design_space": json.loads(json.dumps(self.design_space)),
            "search": dict(self.search),
            "cost_model": json.loads(json.dumps(self.cost_model)),
        }


@dataclass(frozen=True)
class ProConstellationOptimizationResult:
    problem: ProConstellationOptimizationProblem
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ProConstellationOptimizationArtifacts:
    output_dir: Path
    manifest_json: Path
    problem_json: Path
    summary_json: Path
    candidates_csv: Path
    pareto_csv: Path


def _iter_candidate_specs(problem: ProConstellationOptimizationProblem) -> Iterable[dict[str, Any]]:
    space = problem.design_space
    site_groups = tuple(
        combination
        for count in range(space["minimum_ground_sites"], space["maximum_ground_sites"] + 1)
        for combination in itertools.combinations(space["ground_site_pool"], count)
    )
    for pattern in space["patterns"]:
        spans: Sequence[float | None] = space["shell_raan_span_deg_values"] if pattern == "shell" else (None,)
        for (
            satellite_count,
            plane_count,
            phasing,
            altitude_km,
            inclination_deg,
            raan_start_deg,
            initial_phase_deg,
            raan_span_deg,
            ground_site_ids,
        ) in itertools.product(
            space["satellite_counts"],
            space["plane_counts"],
            space["phasing_values"],
            space["altitude_km_values"],
            space["inclination_deg_values"],
            space["raan_start_deg_values"],
            space["initial_phase_deg_values"],
            spans,
            site_groups,
        ):
            if satellite_count % plane_count or phasing >= plane_count:
                continue
            identity = {
                "pattern": pattern,
                "satellite_count": satellite_count,
                "plane_count": plane_count,
                "phasing": phasing,
                "altitude_km": altitude_km,
                "inclination_deg": inclination_deg,
                "raan_start_deg": float(raan_start_deg) % 360.0,
                "initial_phase_deg": float(initial_phase_deg) % 360.0,
                "raan_span_deg": raan_span_deg,
                "ground_site_ids": list(ground_site_ids),
            }
            yield {"design_id": f"opt-{_semantic_sha256(identity)[:16]}", **identity}


def _select_candidate_specs(
    problem: ProConstellationOptimizationProblem,
) -> tuple[list[dict[str, Any]], int]:
    limit = int(problem.search["max_evaluations"])
    maximum_generated = int(problem.search["max_generated_candidates"])
    method = str(problem.search["method"])
    reservoir: list[dict[str, Any]] = []
    rng = random.Random(int(problem.search["seed"]))
    generated = 0
    seen_ids: set[str] = set()
    for generated, spec in enumerate(_iter_candidate_specs(problem), start=1):
        if generated > maximum_generated:
            raise ProConstellationOptimizationError("Generated design space exceeds search.max_generated_candidates.")
        if spec["design_id"] in seen_ids:
            raise ProConstellationOptimizationError("Generated candidate identity collision detected.")
        seen_ids.add(spec["design_id"])
        if method == "exhaustive":
            reservoir.append(spec)
        elif len(reservoir) < limit:
            reservoir.append(spec)
        else:
            replacement = rng.randrange(generated)
            if replacement < limit:
                reservoir[replacement] = spec
    reservoir.sort(key=lambda item: item["design_id"])
    return reservoir, generated


def _public_problem_mapping(
    problem: ProConstellationOptimizationProblem,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CONSTELLATION_DESIGN_PROBLEM_SCHEMA,
        "analysis_id": f"{problem.analysis_id}.{spec['design_id']}",
        **json.loads(json.dumps(problem.public_evaluation)),
        "designs": [dict(spec)],
    }


def _validated_public_problem(
    problem: ProConstellationOptimizationProblem,
    spec: Mapping[str, Any],
) -> ConstellationDesignProblem:
    try:
        return ConstellationDesignProblem.from_mapping(_public_problem_mapping(problem, spec))
    except ValueError as exc:
        raise ProConstellationOptimizationError(
            f"Generated candidate {spec.get('design_id')!r} violates the public evaluator contract: {exc}"
        ) from exc


def _candidate_total_cost(
    problem: ProConstellationOptimizationProblem,
    spec: Mapping[str, Any],
) -> float:
    model = problem.cost_model
    return math.fsum(
        (
            float(model["fixed_cost"]),
            int(spec["satellite_count"]) * float(model["satellite_unit_cost"]),
            *(float(model["ground_site_costs"][site_id]) for site_id in spec["ground_site_ids"]),
        )
    )


def _evaluate_candidate(
    problem: ProConstellationOptimizationProblem,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    public_result = solve_constellation_design(_validated_public_problem(problem, spec))
    candidate = public_result.evidence["candidate_results"][0]
    total_cost = _candidate_total_cost(problem, spec)
    maximum_cost = problem.cost_model["maximum_total_cost"]
    within_cost = maximum_cost is None or total_cost <= float(maximum_cost) + 1.0e-12
    feasible = bool(candidate["feasible"] and within_cost)
    utility = float(candidate["score"]) - float(problem.cost_model["cost_penalty_per_unit"]) * total_cost
    record = {
        "design_id": spec["design_id"],
        "design": dict(spec),
        "feasible": feasible,
        "public_service_feasible": bool(candidate["feasible"]),
        "within_cost_cap": within_cost,
        "coverage_fraction": float(candidate["coverage"]["time_weighted_mean_covered_fraction"]),
        "network_availability_fraction": float(candidate["network"]["union_sampled_available_fraction"]),
        "satellite_count": int(spec["satellite_count"]),
        "ground_site_count": len(spec["ground_site_ids"]),
        "total_cost": total_cost,
        "cost_unit": problem.cost_model["cost_unit"],
        "public_score": float(candidate["score"]),
        "selection_utility": utility,
        "public_input_semantic_sha256": public_result.evidence["input_semantic_sha256"],
        "public_result_semantic_sha256": public_result.evidence["result_semantic_sha256"],
        "public_candidate_result": candidate,
    }
    record["evaluation_semantic_sha256"] = _semantic_sha256(record)
    return record


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = (
        float(left["coverage_fraction"]) >= float(right["coverage_fraction"])
        and float(left["network_availability_fraction"]) >= float(right["network_availability_fraction"])
        and float(left["total_cost"]) <= float(right["total_cost"])
        and int(left["satellite_count"]) <= int(right["satellite_count"])
        and int(left["ground_site_count"]) <= int(right["ground_site_count"])
    )
    strictly_better = (
        float(left["coverage_fraction"]) > float(right["coverage_fraction"])
        or float(left["network_availability_fraction"]) > float(right["network_availability_fraction"])
        or float(left["total_cost"]) < float(right["total_cost"])
        or int(left["satellite_count"]) < int(right["satellite_count"])
        or int(left["ground_site_count"]) < int(right["ground_site_count"])
    )
    return no_worse and strictly_better


def _pareto_frontier(evaluations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    feasible = [dict(item) for item in evaluations if bool(item["feasible"])]
    frontier = [
        candidate
        for candidate in feasible
        if not any(other["design_id"] != candidate["design_id"] and _dominates(other, candidate) for other in feasible)
    ]
    frontier.sort(
        key=lambda item: (
            -float(item["coverage_fraction"]),
            -float(item["network_availability_fraction"]),
            float(item["total_cost"]),
            str(item["design_id"]),
        )
    )
    return frontier


def _ranking_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(item["feasible"])),
        -float(item["selection_utility"]),
        -float(item["coverage_fraction"]),
        -float(item["network_availability_fraction"]),
        float(item["total_cost"]),
        str(item["design_id"]),
    )


def _build_result(
    problem: ProConstellationOptimizationProblem,
    specs: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    *,
    generated_count: int,
) -> ProConstellationOptimizationResult:
    if len(specs) != len(evaluations):
        raise ProConstellationOptimizationError("Completed evaluation count does not match selected candidates.")
    by_id = {str(item["design_id"]): dict(item) for item in evaluations}
    if set(by_id) != {str(item["design_id"]) for item in specs}:
        raise ProConstellationOptimizationError("Completed evaluations do not match the selected candidate identities.")
    ranked = sorted(by_id.values(), key=_ranking_key)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    frontier = _pareto_frontier(ranked)
    frontier_ids = [str(item["design_id"]) for item in frontier]
    for candidate in ranked:
        candidate["pareto_optimal"] = candidate["design_id"] in frontier_ids
    recommended = min(frontier, key=_ranking_key)["design_id"] if frontier else None

    reevaluations: list[dict[str, Any]] = []
    for frontier_candidate in frontier:
        reproduced = _evaluate_candidate(problem, frontier_candidate["design"])
        original_digest = frontier_candidate["evaluation_semantic_sha256"]
        reproduced_digest = reproduced["evaluation_semantic_sha256"]
        if original_digest != reproduced_digest:
            raise ProConstellationOptimizationError(
                f"Authoritative reevaluation mismatch for promoted design {frontier_candidate['design_id']!r}."
            )
        reevaluations.append(
            {
                "design_id": frontier_candidate["design_id"],
                "status": "verified",
                "original_evaluation_semantic_sha256": original_digest,
                "reevaluation_semantic_sha256": reproduced_digest,
                "public_result_semantic_sha256": reproduced["public_result_semantic_sha256"],
            }
        )

    problem_digest = _semantic_sha256(problem.to_dict())
    search_summary = {
        "algorithm_id": ALGORITHM_ID,
        "method": problem.search["method"],
        "seed": problem.search["seed"],
        "generated_candidate_count": generated_count,
        "selected_candidate_count": len(specs),
        "evaluated_candidate_count": len(evaluations),
        "feasible_candidate_count": sum(bool(item["feasible"]) for item in ranked),
        "pareto_candidate_count": len(frontier),
        "exhaustive_over_generated_space": problem.search["method"] == "exhaustive",
    }
    digest_payload = {
        "problem_semantic_sha256": problem_digest,
        "search": search_summary,
        "candidates": ranked,
        "pareto_design_ids": frontier_ids,
        "recommended_design_id": recommended,
        "authoritative_reevaluations": reevaluations,
    }
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "analysis_id": problem.analysis_id,
        "status": "complete",
        **digest_payload,
        "result_semantic_sha256": _semantic_sha256(digest_payload),
        "claim_limits": [
            "Exhaustive means exhaustive only over the finite declared generated design space.",
            "Seeded sampling does not prove global optimality or complete Pareto coverage.",
            "Every evaluated and promoted design uses the public deterministic constellation evaluator.",
            "Cost inputs are analyst-declared units and are not independently calibrated estimates.",
            "Ground-site placement selects from the declared site catalog; it is not continuous geospatial optimization.",
            "The public evaluator's ideal-attitude, free-space-link, scheduling, and operational non-claims remain in force.",
            "Evidence is engineering analysis support, not operational qualification or authorization.",
        ],
    }
    return ProConstellationOptimizationResult(problem, evidence)


def validate_problem(
    value: ProConstellationOptimizationProblem | Mapping[str, Any],
) -> ProConstellationOptimizationProblem:
    """Normalize and resource-check a problem without requiring an entitlement."""

    return (
        value
        if isinstance(value, ProConstellationOptimizationProblem)
        else ProConstellationOptimizationProblem.from_mapping(value)
    )


def solve_pro_constellation_optimization(
    value: ProConstellationOptimizationProblem | Mapping[str, Any],
) -> ProConstellationOptimizationResult:
    """Run one entitled deterministic exhaustive or seeded design search."""

    require_pro_feature(FEATURE_CONSTELLATION_OPTIMIZATION)
    problem = validate_problem(value)
    _require_numeric_backend(problem.public_evaluation["propagation"].get("numeric_backend", "rust"))
    specs, generated_count = _select_candidate_specs(problem)
    evaluations = [_evaluate_candidate(problem, spec) for spec in specs]
    return _build_result(problem, specs, evaluations, generated_count=generated_count)


def _candidate_csv_bytes(evidence: Mapping[str, Any], *, pareto_only: bool) -> bytes:
    fields = (
        "rank",
        "design_id",
        "pareto_optimal",
        "feasible",
        "public_service_feasible",
        "within_cost_cap",
        "coverage_fraction",
        "network_availability_fraction",
        "satellite_count",
        "ground_site_count",
        "total_cost",
        "cost_unit",
        "public_score",
        "selection_utility",
        "evaluation_semantic_sha256",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for candidate in evidence["candidates"]:
        if pareto_only and not candidate["pareto_optimal"]:
            continue
        writer.writerow({field: candidate[field] for field in fields})
    return stream.getvalue().encode("utf-8")


def _render_artifacts(result: ProConstellationOptimizationResult) -> dict[str, bytes]:
    rendered = {
        "normalized_problem.json": _json_bytes(result.problem.to_dict()),
        "constellation_optimization_summary.json": _json_bytes(result.evidence),
        "constellation_optimization_candidates.csv": _candidate_csv_bytes(result.evidence, pareto_only=False),
        "constellation_optimization_pareto.csv": _candidate_csv_bytes(result.evidence, pareto_only=True),
    }
    oversized = sorted(name for name, content in rendered.items() if len(content) > MAX_ARTIFACT_BYTES)
    if oversized:
        raise ProConstellationOptimizationError(
            "Constellation-optimization artifacts exceed the per-file bound: " + ", ".join(oversized)
        )
    if sum(map(len, rendered.values())) > MAX_TOTAL_EVIDENCE_BYTES:
        raise ProConstellationOptimizationError("Constellation-optimization evidence exceeds its total size bound.")
    return rendered


def _manifest(result: ProConstellationOptimizationResult, rendered: Mapping[str, bytes]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": result.problem.analysis_id,
        "status": result.evidence["status"],
        "problem_semantic_sha256": result.evidence["problem_semantic_sha256"],
        "result_semantic_sha256": result.evidence["result_semantic_sha256"],
        "recommended_design_id": result.evidence["recommended_design_id"],
        "pareto_design_ids": result.evidence["pareto_design_ids"],
        "artifacts": [
            {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in rendered.items()
        ],
        "claim_limits": result.evidence["claim_limits"],
    }


def write_evidence(
    result: ProConstellationOptimizationResult,
    output_dir: str | Path,
) -> ProConstellationOptimizationArtifacts:
    """Atomically publish one completed Pro optimization evidence packet."""

    destination = Path(output_dir).expanduser()
    if destination.is_symlink():
        raise ProConstellationOptimizationError("output_dir must not be a symbolic link.")
    destination = destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise ProConstellationOptimizationError(f"output_dir must be absent; refusing to replace {destination}.")
    rendered = _render_artifacts(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.staging-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for name, content in rendered.items():
            (staging / name).write_bytes(content)
        (staging / "constellation_optimization_manifest.json").write_bytes(_json_bytes(_manifest(result, rendered)))
        try:
            os.rename(staging, destination)
        except FileExistsError as exc:
            raise ProConstellationOptimizationError(f"output_dir appeared during publication: {destination}.") from exc
    return ProConstellationOptimizationArtifacts(
        destination,
        destination / "constellation_optimization_manifest.json",
        destination / "normalized_problem.json",
        destination / "constellation_optimization_summary.json",
        destination / "constellation_optimization_candidates.csv",
        destination / "constellation_optimization_pareto.csv",
    )


def _strict_json(content: bytes, field: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ProConstellationOptimizationError(f"{field} contains forbidden non-finite constant {value}.")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ProConstellationOptimizationError(f"{field} contains duplicate field {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(content.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProConstellationOptimizationError(f"Could not parse {field}: {exc}") from exc


def _read_json_file(path: Path, field: str) -> dict[str, Any]:
    try:
        content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=MAX_ARTIFACT_BYTES)
    except (OSError, SafeReadError) as exc:
        raise ProConstellationOptimizationError(f"Could not read {field}: {exc}") from exc
    value = _strict_json(content, field)
    if not isinstance(value, dict):
        raise ProConstellationOptimizationError(f"{field} must be a JSON object.")
    return value


def replay_evidence(evidence_dir: str | Path) -> dict[str, Any]:
    """Entitlement-check, verify receipts, rerun the search, and compare exact artifacts."""

    require_pro_feature(FEATURE_CONSTELLATION_OPTIMIZATION)
    source = Path(evidence_dir).expanduser()
    if source.is_symlink():
        raise ProConstellationOptimizationError("evidence_dir must be a real directory.")
    root = source.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ProConstellationOptimizationError("evidence_dir must be a real directory.")
    required = {
        "normalized_problem.json",
        "constellation_optimization_summary.json",
        "constellation_optimization_candidates.csv",
        "constellation_optimization_pareto.csv",
        "constellation_optimization_manifest.json",
    }
    actual = {path.name for path in root.iterdir()}
    if actual != required:
        raise ProConstellationOptimizationError("Constellation-optimization evidence inventory is not exact.")
    manifest = _read_json_file(root / "constellation_optimization_manifest.json", "manifest")
    manifest_fields = {
        "schema_version",
        "analysis_id",
        "status",
        "problem_semantic_sha256",
        "result_semantic_sha256",
        "recommended_design_id",
        "pareto_design_ids",
        "artifacts",
        "claim_limits",
    }
    if set(manifest) != manifest_fields or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ProConstellationOptimizationError("Constellation-optimization manifest contract is invalid.")
    receipts = manifest.get("artifacts")
    if not isinstance(receipts, list) or len(receipts) != 4:
        raise ProConstellationOptimizationError("Manifest requires exactly four artifact receipts.")
    content_by_name: dict[str, bytes] = {}
    total_bytes = 0
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != {"path", "bytes", "sha256"}:
            raise ProConstellationOptimizationError("Artifact receipt contract is invalid.")
        name = receipt.get("path")
        if name not in required - {"constellation_optimization_manifest.json"} or name in content_by_name:
            raise ProConstellationOptimizationError("Artifact receipt inventory is invalid.")
        size = receipt.get("bytes")
        digest = receipt.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_ARTIFACT_BYTES:
            raise ProConstellationOptimizationError("Artifact receipt size is invalid.")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ProConstellationOptimizationError("Artifact receipt SHA-256 is invalid.")
        try:
            content = read_regular_file_nofollow(root / str(name), min_bytes=1, max_bytes=MAX_ARTIFACT_BYTES)
        except (OSError, SafeReadError) as exc:
            raise ProConstellationOptimizationError(f"Could not read received artifact {name}: {exc}") from exc
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise ProConstellationOptimizationError(f"Artifact receipt mismatch for {name}.")
        total_bytes += len(content)
        content_by_name[str(name)] = content
    if total_bytes > MAX_TOTAL_EVIDENCE_BYTES or set(content_by_name) != required - {
        "constellation_optimization_manifest.json"
    }:
        raise ProConstellationOptimizationError("Constellation-optimization evidence exceeds its bounded inventory.")
    problem_payload = _strict_json(content_by_name["normalized_problem.json"], "normalized problem")
    if not isinstance(problem_payload, dict):
        raise ProConstellationOptimizationError("Normalized problem must be a JSON object.")
    problem = ProConstellationOptimizationProblem.from_mapping(problem_payload)
    reproduced = solve_pro_constellation_optimization(problem)
    expected = _render_artifacts(reproduced)
    if expected != content_by_name:
        raise ProConstellationOptimizationError("Retained evidence differs from authoritative deterministic replay.")
    expected_manifest = _manifest(reproduced, expected)
    if manifest != expected_manifest:
        raise ProConstellationOptimizationError("Manifest claims differ from authoritative deterministic replay.")
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "analysis_id": problem.analysis_id,
        "status": "verified",
        "recommended_design_id": reproduced.evidence["recommended_design_id"],
        "pareto_design_ids": reproduced.evidence["pareto_design_ids"],
        "problem_semantic_sha256": reproduced.evidence["problem_semantic_sha256"],
        "result_semantic_sha256": reproduced.evidence["result_semantic_sha256"],
    }


def _work_dir(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.constellation-optimization-work"


def _atomic_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    content = _json_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ProConstellationOptimizationError("Refusing to replace an unexpected checkpoint temporary path.")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _checkpoint_payload(
    problem: ProConstellationOptimizationProblem,
    specs: Sequence[Mapping[str, Any]],
    completed: Sequence[Mapping[str, Any]],
    generated_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "analysis_id": problem.analysis_id,
        "problem_semantic_sha256": _semantic_sha256(problem.to_dict()),
        "generated_candidate_count": generated_count,
        "selected_candidates": [dict(spec) for spec in specs],
        "completed_evaluations": [dict(item) for item in completed],
    }


def _load_checkpoint(
    problem: ProConstellationOptimizationProblem,
    work: Path,
    specs: Sequence[Mapping[str, Any]],
    generated_count: int,
) -> list[dict[str, Any]]:
    checkpoint = _read_json_file(work / "checkpoint.json", "optimization checkpoint")
    expected_fields = {
        "schema_version",
        "analysis_id",
        "problem_semantic_sha256",
        "generated_candidate_count",
        "selected_candidates",
        "completed_evaluations",
    }
    if set(checkpoint) != expected_fields or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ProConstellationOptimizationError("Optimization checkpoint contract is invalid.")
    if (
        checkpoint.get("analysis_id") != problem.analysis_id
        or checkpoint.get("problem_semantic_sha256") != _semantic_sha256(problem.to_dict())
        or checkpoint.get("generated_candidate_count") != generated_count
        or checkpoint.get("selected_candidates") != list(specs)
    ):
        raise ProConstellationOptimizationError("Optimization checkpoint does not match the requested problem/search.")
    completed = checkpoint.get("completed_evaluations")
    if not isinstance(completed, list) or len(completed) > len(specs):
        raise ProConstellationOptimizationError("Optimization checkpoint completed-evaluation inventory is invalid.")
    for index, evaluation in enumerate(completed):
        if not isinstance(evaluation, dict) or evaluation.get("design_id") != specs[index]["design_id"]:
            raise ProConstellationOptimizationError("Optimization checkpoint evaluation order is invalid.")
        retained_digest = evaluation.get("evaluation_semantic_sha256")
        without_digest = dict(evaluation)
        without_digest.pop("evaluation_semantic_sha256", None)
        if retained_digest != _semantic_sha256(without_digest):
            raise ProConstellationOptimizationError("Optimization checkpoint evaluation digest is invalid.")
    return [dict(item) for item in completed]


def run_pro_constellation_optimization(
    value: ProConstellationOptimizationProblem | Mapping[str, Any],
    output_dir: str | Path,
    *,
    resume: bool = False,
    max_new_evaluations: int | None = None,
) -> dict[str, Any]:
    """Run or resume a bounded search, retaining a private checkpoint until completion."""

    require_pro_feature(FEATURE_CONSTELLATION_OPTIMIZATION)
    problem = validate_problem(value)
    _require_numeric_backend(problem.public_evaluation["propagation"].get("numeric_backend", "rust"))
    destination = Path(output_dir).expanduser()
    if destination.is_symlink():
        raise ProConstellationOptimizationError("output_dir must not be a symbolic link.")
    destination = destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise ProConstellationOptimizationError(f"output_dir must be absent; refusing to replace {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = _work_dir(destination)
    specs, generated_count = _select_candidate_specs(problem)
    if max_new_evaluations is not None:
        max_new_evaluations = _integer(max_new_evaluations, "max_new_evaluations")
        if max_new_evaluations <= 0:
            raise ProConstellationOptimizationError("max_new_evaluations must be positive when supplied.")
    if resume:
        if not work.is_dir() or work.is_symlink():
            raise ProConstellationOptimizationError("No matching real checkpoint directory exists to resume.")
        completed = _load_checkpoint(problem, work, specs, generated_count)
    else:
        if work.exists() or work.is_symlink():
            raise ProConstellationOptimizationError(
                f"Checkpoint directory already exists; use resume or inspect {work}."
            )
        work.mkdir()
        (work / "normalized_problem.json").write_bytes(_json_bytes(problem.to_dict()))
        completed = []
        _atomic_checkpoint(work / "checkpoint.json", _checkpoint_payload(problem, specs, completed, generated_count))
    stop = len(specs) if max_new_evaluations is None else min(len(specs), len(completed) + max_new_evaluations)
    for spec in specs[len(completed) : stop]:
        completed.append(_evaluate_candidate(problem, spec))
        _atomic_checkpoint(work / "checkpoint.json", _checkpoint_payload(problem, specs, completed, generated_count))
    if len(completed) < len(specs):
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "analysis_id": problem.analysis_id,
            "status": "in_progress",
            "completed_evaluations": len(completed),
            "total_evaluations": len(specs),
            "checkpoint_dir": str(work),
        }
    result = _build_result(problem, specs, completed, generated_count=generated_count)
    artifacts = write_evidence(result, destination)
    shutil.rmtree(work)
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "analysis_id": problem.analysis_id,
        "status": "complete",
        "recommended_design_id": result.evidence["recommended_design_id"],
        "pareto_design_ids": result.evidence["pareto_design_ids"],
        "manifest": str(artifacts.manifest_json),
    }


__all__ = [
    "ALGORITHM_ID",
    "CHECKPOINT_SCHEMA",
    "EVIDENCE_SCHEMA",
    "MANIFEST_SCHEMA",
    "PROBLEM_SCHEMA",
    "ProConstellationOptimizationArtifacts",
    "ProConstellationOptimizationError",
    "ProConstellationOptimizationProblem",
    "ProConstellationOptimizationResult",
    "replay_evidence",
    "run_pro_constellation_optimization",
    "solve_pro_constellation_optimization",
    "validate_problem",
    "write_evidence",
]
