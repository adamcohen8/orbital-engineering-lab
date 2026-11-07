"""Bounded exact multi-asset mission scheduling with authoritative replay."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sim.utils.io import SafeReadError, read_regular_file_nofollow

MISSION_SCHEDULING_PROBLEM_SCHEMA = "oel.mission_scheduling_problem.v1"
MISSION_SCHEDULING_EVIDENCE_SCHEMA = "oel.mission_scheduling_evidence.v1"
MAX_PUBLIC_MISSION_OPPORTUNITIES = 18
_EPS = 1.0e-12
_MAX_PUBLIC_TEXT_LENGTH = 256
_MAX_EVIDENCE_ARTIFACT_BYTES = 4 * 1024 * 1024
_MAX_EVIDENCE_TOTAL_BYTES = 8 * 1024 * 1024


class MissionSchedulingError(ValueError):
    """Raised when a bounded mission-scheduling problem or replay is invalid."""


def _required(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise MissionSchedulingError(f"{field} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise MissionSchedulingError(f"{field} must be a non-empty string.")
    if len(normalized) > _MAX_PUBLIC_TEXT_LENGTH:
        raise MissionSchedulingError(
            f"{field} exceeds the {_MAX_PUBLIC_TEXT_LENGTH}-character public bound."
        )
    return normalized


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise MissionSchedulingError(f"{field} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MissionSchedulingError(f"{field} must be finite.") from exc
    if not math.isfinite(result):
        raise MissionSchedulingError(f"{field} must be finite.")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise MissionSchedulingError(f"{field} must be an integer.")
    return int(value)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MissionSchedulingError(f"{field} must be a JSON object.")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise MissionSchedulingError(f"{field} must be a JSON array.")
    return value


def _bounded_iterable(value: Any, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise MissionSchedulingError(f"{field} must be an iterable of values.")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise MissionSchedulingError(f"{field} must be an iterable of values.") from exc
    if len(result) > MAX_PUBLIC_MISSION_OPPORTUNITIES:
        raise MissionSchedulingError(
            f"{field} exceeds the bounded public inventory of {MAX_PUBLIC_MISSION_OPPORTUNITIES}."
        )
    return result


def _reject_unknown_fields(data: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(str(key) for key in set(data) - allowed)
    if unknown:
        raise MissionSchedulingError(f"{field} contains unknown fields: {', '.join(unknown)}.")


def _finite_sum(values: Iterable[float], field: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise MissionSchedulingError(f"{field} must have a finite aggregate.") from exc
    if not math.isfinite(result):
        raise MissionSchedulingError(f"{field} must have a finite aggregate.")
    return result


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise MissionSchedulingError(f"JSON contains forbidden non-finite constant {value}.")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MissionSchedulingError(f"JSON object contains duplicate field {key!r}.")
        result[key] = value
    return result


def _parse_json_bytes(content: bytes, field: str) -> Any:
    try:
        return json.loads(
            content.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MissionSchedulingError(f"Could not parse {field} as strict UTF-8 JSON: {exc}") from exc


def _validate_sha256(value: Any, field: str) -> str:
    digest = _required(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MissionSchedulingError(f"{field} must be a lowercase SHA-256 digest.")
    return digest


@dataclass(frozen=True)
class MissionOpportunity:
    opportunity_id: str
    source_product_sha256: str
    asset_id: str
    kind: str
    start_s: float
    end_s: float
    objective_value: float
    energy_cost_wh: float
    data_volume_bytes: float = 0.0
    downlink_capacity_bytes: float = 0.0
    station_id: str | None = None
    pointing_unit_eci: tuple[float, float, float] | None = None
    target_id: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MissionOpportunity:
        data = _mapping(data, "opportunity")
        _reject_unknown_fields(
            data,
            {
                "opportunity_id",
                "source_product_sha256",
                "asset_id",
                "kind",
                "start_s",
                "end_s",
                "objective_value",
                "energy_cost_wh",
                "data_volume_bytes",
                "downlink_capacity_bytes",
                "station_id",
                "pointing_unit_eci",
                "target_id",
            },
            "opportunity",
        )
        pointing = data.get("pointing_unit_eci")
        if pointing is not None and not isinstance(pointing, (list, tuple)):
            raise MissionSchedulingError("pointing_unit_eci must be a three-value JSON array.")
        return cls(
            opportunity_id=data.get("opportunity_id", ""),
            source_product_sha256=data.get("source_product_sha256", ""),
            asset_id=data.get("asset_id", ""),
            kind=data.get("kind", ""),
            start_s=data.get("start_s", float("nan")),
            end_s=data.get("end_s", float("nan")),
            objective_value=data.get("objective_value", 0.0),
            energy_cost_wh=data.get("energy_cost_wh", 0.0),
            data_volume_bytes=data.get("data_volume_bytes", 0.0),
            downlink_capacity_bytes=data.get("downlink_capacity_bytes", 0.0),
            station_id=data.get("station_id"),
            pointing_unit_eci=None if pointing is None else tuple(pointing),
            target_id=data.get("target_id"),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "opportunity_id", _required(self.opportunity_id, "opportunity_id"))
        object.__setattr__(
            self,
            "source_product_sha256",
            _validate_sha256(self.source_product_sha256, "source_product_sha256"),
        )
        object.__setattr__(self, "asset_id", _required(self.asset_id, "asset_id"))
        kind = _required(self.kind, "kind").lower()
        if kind not in {"observation", "downlink", "other"}:
            raise MissionSchedulingError("kind must be observation, downlink, or other.")
        object.__setattr__(self, "kind", kind)
        for field in (
            "start_s",
            "end_s",
            "objective_value",
            "energy_cost_wh",
            "data_volume_bytes",
            "downlink_capacity_bytes",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        duration = self.end_s - self.start_s
        if not math.isfinite(duration):
            raise MissionSchedulingError("Opportunity duration must be finite.")
        if self.end_s <= self.start_s:
            raise MissionSchedulingError("end_s must be greater than start_s.")
        if min(
            self.objective_value,
            self.energy_cost_wh,
            self.data_volume_bytes,
            self.downlink_capacity_bytes,
        ) < 0.0:
            raise MissionSchedulingError("Objective, energy, data, and downlink capacity must be nonnegative.")
        station = None if self.station_id is None else _required(self.station_id, "station_id")
        target = None if self.target_id is None else _required(self.target_id, "target_id")
        object.__setattr__(self, "station_id", station)
        object.__setattr__(self, "target_id", target)
        if kind == "observation":
            if self.data_volume_bytes <= 0.0 or self.downlink_capacity_bytes != 0.0 or station is not None:
                raise MissionSchedulingError(
                    "Observation opportunities require positive data_volume_bytes and no downlink capacity or station."
                )
        elif kind == "downlink":
            if self.downlink_capacity_bytes <= 0.0 or self.data_volume_bytes != 0.0 or station is None:
                raise MissionSchedulingError(
                    "Downlink opportunities require positive downlink_capacity_bytes, station_id, and no data volume."
                )
            if self.objective_value != 0.0:
                raise MissionSchedulingError("Downlink objective_value must be zero; observation value is counted once.")
        elif self.data_volume_bytes != 0.0 or self.downlink_capacity_bytes != 0.0 or station is not None:
            raise MissionSchedulingError("Other opportunities cannot produce data, downlink data, or reserve a station.")
        if self.pointing_unit_eci is not None:
            try:
                vector = np.asarray(self.pointing_unit_eci, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MissionSchedulingError("pointing_unit_eci must contain three finite values.") from exc
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise MissionSchedulingError("pointing_unit_eci must contain three finite values.")
            norm = float(np.linalg.norm(vector))
            if abs(norm - 1.0) > 1.0e-10:
                raise MissionSchedulingError("pointing_unit_eci must be normalized within 1e-10.")
            object.__setattr__(self, "pointing_unit_eci", tuple(float(value) for value in vector))


@dataclass(frozen=True)
class AssetScheduleConstraints:
    asset_id: str
    storage_capacity_bytes: float
    initial_storage_bytes: float
    energy_budget_wh: float
    maximum_payload_duty_cycle: float = 1.0
    maximum_slew_rate_rad_s: float | None = None
    settling_time_s: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssetScheduleConstraints:
        data = _mapping(data, "asset constraint")
        _reject_unknown_fields(
            data,
            {
                "asset_id",
                "storage_capacity_bytes",
                "initial_storage_bytes",
                "energy_budget_wh",
                "maximum_payload_duty_cycle",
                "maximum_slew_rate_rad_s",
                "settling_time_s",
            },
            "asset constraint",
        )
        return cls(
            asset_id=data.get("asset_id", ""),
            storage_capacity_bytes=data.get("storage_capacity_bytes", float("nan")),
            initial_storage_bytes=data.get("initial_storage_bytes", 0.0),
            energy_budget_wh=data.get("energy_budget_wh", float("nan")),
            maximum_payload_duty_cycle=data.get("maximum_payload_duty_cycle", 1.0),
            maximum_slew_rate_rad_s=data.get("maximum_slew_rate_rad_s"),
            settling_time_s=data.get("settling_time_s", 0.0),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_id", _required(self.asset_id, "asset_id"))
        for field in (
            "storage_capacity_bytes",
            "initial_storage_bytes",
            "energy_budget_wh",
            "maximum_payload_duty_cycle",
            "settling_time_s",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if self.storage_capacity_bytes < 0.0:
            raise MissionSchedulingError("storage_capacity_bytes must be nonnegative.")
        if not 0.0 <= self.initial_storage_bytes <= self.storage_capacity_bytes:
            raise MissionSchedulingError("initial_storage_bytes must lie within storage capacity.")
        if self.energy_budget_wh < 0.0 or self.settling_time_s < 0.0:
            raise MissionSchedulingError("Energy budget and settling time must be nonnegative.")
        if not 0.0 <= self.maximum_payload_duty_cycle <= 1.0:
            raise MissionSchedulingError("maximum_payload_duty_cycle must lie within [0, 1].")
        if self.maximum_slew_rate_rad_s is not None:
            rate = _finite(self.maximum_slew_rate_rad_s, "maximum_slew_rate_rad_s")
            if rate <= 0.0:
                raise MissionSchedulingError("maximum_slew_rate_rad_s must be positive when supplied.")
            object.__setattr__(self, "maximum_slew_rate_rad_s", rate)


@dataclass(frozen=True)
class MissionSchedulingProblem:
    analysis_id: str
    horizon_start_s: float
    horizon_end_s: float
    assets: tuple[AssetScheduleConstraints, ...]
    opportunities: tuple[MissionOpportunity, ...]
    require_observation_delivery_by_horizon: bool = True
    minimum_selected_observations: int = 1
    maximum_candidates: int = MAX_PUBLIC_MISSION_OPPORTUNITIES
    schema_version: str = MISSION_SCHEDULING_PROBLEM_SCHEMA

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MissionSchedulingProblem:
        data = _mapping(data, "mission-scheduling problem")
        _reject_unknown_fields(
            data,
            {
                "schema_version",
                "analysis_id",
                "horizon_start_s",
                "horizon_end_s",
                "assets",
                "opportunities",
                "require_observation_delivery_by_horizon",
                "minimum_selected_observations",
                "maximum_candidates",
            },
            "mission-scheduling problem",
        )
        assets = _sequence(data.get("assets", ()), "assets")
        opportunities = _sequence(data.get("opportunities", ()), "opportunities")
        if len(assets) > MAX_PUBLIC_MISSION_OPPORTUNITIES:
            raise MissionSchedulingError(
                f"Asset count exceeds the bounded public inventory of {MAX_PUBLIC_MISSION_OPPORTUNITIES}."
            )
        if len(opportunities) > MAX_PUBLIC_MISSION_OPPORTUNITIES:
            raise MissionSchedulingError(
                f"Opportunity count exceeds the bounded public inventory of {MAX_PUBLIC_MISSION_OPPORTUNITIES}."
            )
        schema = str(data.get("schema_version", MISSION_SCHEDULING_PROBLEM_SCHEMA)).strip()
        return cls(
            schema_version=schema,
            analysis_id=data.get("analysis_id", ""),
            horizon_start_s=data.get("horizon_start_s", float("nan")),
            horizon_end_s=data.get("horizon_end_s", float("nan")),
            assets=tuple(AssetScheduleConstraints.from_mapping(item) for item in assets),
            opportunities=tuple(MissionOpportunity.from_mapping(item) for item in opportunities),
            require_observation_delivery_by_horizon=data.get(
                "require_observation_delivery_by_horizon", True
            ),
            minimum_selected_observations=data.get("minimum_selected_observations", 1),
            maximum_candidates=data.get("maximum_candidates", MAX_PUBLIC_MISSION_OPPORTUNITIES),
        )

    def __post_init__(self) -> None:
        if self.schema_version != MISSION_SCHEDULING_PROBLEM_SCHEMA:
            raise MissionSchedulingError(f"Unsupported mission-scheduling schema {self.schema_version!r}.")
        object.__setattr__(self, "analysis_id", _required(self.analysis_id, "analysis_id"))
        object.__setattr__(self, "horizon_start_s", _finite(self.horizon_start_s, "horizon_start_s"))
        object.__setattr__(self, "horizon_end_s", _finite(self.horizon_end_s, "horizon_end_s"))
        horizon = self.horizon_end_s - self.horizon_start_s
        if not math.isfinite(horizon):
            raise MissionSchedulingError("The scheduling horizon duration must be finite.")
        if self.horizon_end_s <= self.horizon_start_s:
            raise MissionSchedulingError("The scheduling horizon end must be after its start.")
        if not self.assets:
            raise MissionSchedulingError("At least one asset constraint is required.")
        if not self.opportunities:
            raise MissionSchedulingError("At least one opportunity is required.")
        if not isinstance(self.require_observation_delivery_by_horizon, bool):
            raise MissionSchedulingError("require_observation_delivery_by_horizon must be boolean.")
        if any(not isinstance(item, AssetScheduleConstraints) for item in self.assets):
            raise MissionSchedulingError("assets must contain validated AssetScheduleConstraints values.")
        if any(not isinstance(item, MissionOpportunity) for item in self.opportunities):
            raise MissionSchedulingError("opportunities must contain validated MissionOpportunity values.")
        asset_ids = tuple(item.asset_id for item in self.assets)
        opportunity_ids = tuple(item.opportunity_id for item in self.opportunities)
        if len(set(asset_ids)) != len(asset_ids):
            raise MissionSchedulingError("Asset IDs must be unique.")
        if len(set(opportunity_ids)) != len(opportunity_ids):
            raise MissionSchedulingError("Opportunity IDs must be unique.")
        maximum = _integer(self.maximum_candidates, "maximum_candidates")
        if not 1 <= maximum <= MAX_PUBLIC_MISSION_OPPORTUNITIES:
            raise MissionSchedulingError(
                f"maximum_candidates must lie within [1, {MAX_PUBLIC_MISSION_OPPORTUNITIES}]."
            )
        if len(self.opportunities) > maximum:
            raise MissionSchedulingError("Opportunity count exceeds maximum_candidates.")
        object.__setattr__(self, "maximum_candidates", maximum)
        minimum = _integer(self.minimum_selected_observations, "minimum_selected_observations")
        if minimum < 0:
            raise MissionSchedulingError("minimum_selected_observations must be a nonnegative integer.")
        object.__setattr__(self, "minimum_selected_observations", minimum)
        known_assets = set(asset_ids)
        slew_assets = {item.asset_id for item in self.assets if item.maximum_slew_rate_rad_s is not None}
        for item in self.opportunities:
            if item.asset_id not in known_assets:
                raise MissionSchedulingError(
                    f"Opportunity {item.opportunity_id!r} names unknown asset {item.asset_id!r}."
                )
            if item.start_s < self.horizon_start_s or item.end_s > self.horizon_end_s:
                raise MissionSchedulingError(f"Opportunity {item.opportunity_id!r} is outside the horizon.")
            if item.asset_id in slew_assets and item.pointing_unit_eci is None:
                raise MissionSchedulingError(
                    f"Slew-constrained asset {item.asset_id!r} requires pointing on every opportunity."
                )
        _finite_sum((item.objective_value for item in self.opportunities), "objective values")
        for asset_id in asset_ids:
            owned = tuple(item for item in self.opportunities if item.asset_id == asset_id)
            _finite_sum((item.energy_cost_wh for item in owned), f"energy costs for asset {asset_id!r}")
            _finite_sum((item.data_volume_bytes for item in owned), f"data volumes for asset {asset_id!r}")
            _finite_sum(
                (item.downlink_capacity_bytes for item in owned),
                f"downlink capacities for asset {asset_id!r}",
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["assets"] = sorted(payload["assets"], key=lambda item: item["asset_id"])
        payload["opportunities"] = sorted(
            payload["opportunities"], key=lambda item: item["opportunity_id"]
        )
        return payload


@dataclass(frozen=True)
class ScheduledActivity:
    sequence: int
    opportunity_id: str
    asset_id: str
    kind: str
    station_id: str | None
    target_id: str | None
    start_s: float
    end_s: float
    objective_value: float
    storage_after_bytes: float
    cumulative_energy_wh: float
    data_produced_bytes: float
    data_downlinked_bytes: float
    source_product_sha256: str


@dataclass(frozen=True)
class ObservationDelivery:
    opportunity_id: str
    asset_id: str
    produced_bytes: float
    delivered_bytes: float
    undelivered_bytes: float
    fully_delivered: bool


@dataclass(frozen=True)
class _Evaluation:
    feasible: bool
    reason: str | None
    activities: tuple[ScheduledActivity, ...]
    deliveries: tuple[ObservationDelivery, ...]
    objective_value: float
    resource_summary: dict[str, dict[str, float]]


@dataclass(frozen=True)
class MissionSchedulingResult:
    problem: MissionSchedulingProblem
    status: str
    selected_opportunity_ids: tuple[str, ...]
    activities: tuple[ScheduledActivity, ...]
    deliveries: tuple[ObservationDelivery, ...]
    rejected_opportunity_reasons: dict[str, str]
    objective_value: float
    evaluated_subset_count: int
    feasible_subset_count: int
    resource_summary: dict[str, dict[str, float]]
    input_semantic_sha256: str
    schedule_semantic_sha256: str
    summary: dict[str, Any]


@dataclass(frozen=True)
class MissionSchedulingArtifacts:
    output_dir: Path
    manifest_json: Path
    problem_json: Path
    summary_json: Path
    schedule_csv: Path
    rejected_csv: Path
    resources_csv: Path
    deliveries_csv: Path


def _transition_required_s(
    previous: MissionOpportunity,
    following: MissionOpportunity,
    constraint: AssetScheduleConstraints,
) -> float:
    if constraint.maximum_slew_rate_rad_s is None:
        return constraint.settling_time_s
    first = np.asarray(previous.pointing_unit_eci, dtype=float)
    second = np.asarray(following.pointing_unit_eci, dtype=float)
    angle = float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))
    return angle / constraint.maximum_slew_rate_rad_s + constraint.settling_time_s


def _evaluate_selection(
    problem: MissionSchedulingProblem,
    selected: Iterable[MissionOpportunity],
    *,
    enforce_minimum: bool = True,
) -> _Evaluation:
    chosen = tuple(sorted(selected, key=lambda item: (item.start_s, item.end_s, item.opportunity_id)))
    if enforce_minimum and sum(item.kind == "observation" for item in chosen) < problem.minimum_selected_observations:
        return _Evaluation(False, "minimum_observations_not_met", (), (), 0.0, {})
    constraints = {item.asset_id: item for item in problem.assets}
    station_tasks: dict[str, list[MissionOpportunity]] = {}
    asset_tasks: dict[str, list[MissionOpportunity]] = {item.asset_id: [] for item in problem.assets}
    for item in chosen:
        asset_tasks[item.asset_id].append(item)
        if item.kind == "downlink":
            station_tasks.setdefault(str(item.station_id), []).append(item)
    for station_id, items in station_tasks.items():
        ordered = sorted(items, key=lambda item: (item.start_s, item.end_s, item.opportunity_id))
        for previous, following in zip(ordered, ordered[1:]):
            if following.start_s < previous.end_s - _EPS:
                return _Evaluation(
                    False,
                    f"station_contention:{station_id}:{previous.opportunity_id}:{following.opportunity_id}",
                    (),
                    (),
                    0.0,
                    {},
                )

    activities_by_id: dict[str, ScheduledActivity] = {}
    deliveries: list[ObservationDelivery] = []
    resources: dict[str, dict[str, float]] = {}
    horizon = problem.horizon_end_s - problem.horizon_start_s
    for asset_id in sorted(asset_tasks):
        items = sorted(asset_tasks[asset_id], key=lambda item: (item.start_s, item.end_s, item.opportunity_id))
        constraint = constraints[asset_id]
        for previous, following in zip(items, items[1:]):
            if following.start_s < previous.end_s - _EPS:
                return _Evaluation(
                    False,
                    f"asset_overlap:{asset_id}:{previous.opportunity_id}:{following.opportunity_id}",
                    (),
                    (),
                    0.0,
                    {},
                )
            if following.start_s - previous.end_s + _EPS < _transition_required_s(
                previous, following, constraint
            ):
                return _Evaluation(
                    False,
                    f"slew_or_settling:{asset_id}:{previous.opportunity_id}:{following.opportunity_id}",
                    (),
                    (),
                    0.0,
                    {},
                )
        storage = constraint.initial_storage_bytes
        initial_remaining = constraint.initial_storage_bytes
        energy = 0.0
        observation_duration = 0.0
        generated: list[dict[str, Any]] = []
        data_downlinked = 0.0
        peak_storage = storage
        for item in items:
            energy += item.energy_cost_wh
            if energy > constraint.energy_budget_wh + _EPS:
                return _Evaluation(False, f"energy_budget:{asset_id}:{item.opportunity_id}", (), (), 0.0, {})
            transferred = 0.0
            if item.kind == "observation":
                observation_duration += item.end_s - item.start_s
                if observation_duration / horizon > constraint.maximum_payload_duty_cycle + _EPS:
                    return _Evaluation(False, f"payload_duty_cycle:{asset_id}:{item.opportunity_id}", (), (), 0.0, {})
                storage += item.data_volume_bytes
                generated.append({"opportunity": item, "remaining": item.data_volume_bytes})
                if storage > constraint.storage_capacity_bytes + _EPS:
                    return _Evaluation(False, f"storage_capacity:{asset_id}:{item.opportunity_id}", (), (), 0.0, {})
            elif item.kind == "downlink":
                transferred = min(item.downlink_capacity_bytes, storage)
                storage -= transferred
                remaining_transfer = transferred
                from_initial = min(initial_remaining, remaining_transfer)
                initial_remaining -= from_initial
                remaining_transfer -= from_initial
                for record in generated:
                    amount = min(float(record["remaining"]), remaining_transfer)
                    record["remaining"] = float(record["remaining"]) - amount
                    remaining_transfer -= amount
                    if remaining_transfer <= _EPS:
                        break
                data_downlinked += transferred
            peak_storage = max(peak_storage, storage)
            activities_by_id[item.opportunity_id] = ScheduledActivity(
                sequence=-1,
                opportunity_id=item.opportunity_id,
                asset_id=item.asset_id,
                kind=item.kind,
                station_id=item.station_id,
                target_id=item.target_id,
                start_s=item.start_s,
                end_s=item.end_s,
                objective_value=item.objective_value,
                storage_after_bytes=max(0.0, storage),
                cumulative_energy_wh=energy,
                data_produced_bytes=item.data_volume_bytes,
                data_downlinked_bytes=transferred,
                source_product_sha256=item.source_product_sha256,
            )
        for record in generated:
            opportunity = record["opportunity"]
            remaining = max(0.0, float(record["remaining"]))
            delivered = opportunity.data_volume_bytes - remaining
            deliveries.append(
                ObservationDelivery(
                    opportunity_id=opportunity.opportunity_id,
                    asset_id=asset_id,
                    produced_bytes=opportunity.data_volume_bytes,
                    delivered_bytes=delivered,
                    undelivered_bytes=remaining,
                    fully_delivered=remaining <= _EPS,
                )
            )
        if problem.require_observation_delivery_by_horizon:
            undelivered = next((item for item in deliveries if item.asset_id == asset_id and not item.fully_delivered), None)
            if undelivered is not None:
                return _Evaluation(False, f"observation_not_delivered:{undelivered.opportunity_id}", (), (), 0.0, {})
        resources[asset_id] = {
            "initial_storage_bytes": constraint.initial_storage_bytes,
            "final_storage_bytes": max(0.0, storage),
            "peak_storage_bytes": peak_storage,
            "storage_capacity_bytes": constraint.storage_capacity_bytes,
            "energy_used_wh": energy,
            "energy_budget_wh": constraint.energy_budget_wh,
            "payload_duty_cycle": observation_duration / horizon,
            "data_downlinked_bytes": data_downlinked,
        }
    ordered_activities = tuple(
        ScheduledActivity(**{**asdict(activities_by_id[item.opportunity_id]), "sequence": sequence})
        for sequence, item in enumerate(chosen)
    )
    return _Evaluation(
        True,
        None,
        ordered_activities,
        tuple(sorted(deliveries, key=lambda item: item.opportunity_id)),
        math.fsum(item.objective_value for item in chosen),
        resources,
    )


def solve_mission_schedule(
    problem: MissionSchedulingProblem | Mapping[str, Any], *, numeric_backend: str = "rust",
) -> MissionSchedulingResult:
    """Exactly maximize a bounded multi-asset opportunity set."""

    parsed = problem if isinstance(problem, MissionSchedulingProblem) else MissionSchedulingProblem.from_mapping(problem)
    candidates = tuple(sorted(parsed.opportunities, key=lambda item: item.opportunity_id))
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise MissionSchedulingError("numeric_backend must be python or rust.")
    best_ids: tuple[str, ...] | None = None
    best_evaluation: _Evaluation | None = None
    evaluated = 0
    feasible = 0
    if backend == "rust":
        from sim.rust_mission_schedule_backend import search

        try:
            best_mask, feasible = search(parsed, candidates, _transition_required_s, epsilon=_EPS)
        except ValueError as exc:
            raise MissionSchedulingError(str(exc)) from exc
        evaluated = 1 << len(candidates)
        if best_mask is not None:
            selected = tuple(candidates[index] for index in range(len(candidates)) if best_mask & (1 << index))
            best_ids = tuple(item.opportunity_id for item in selected)
            best_evaluation = _evaluate_selection(parsed, selected)
            if not best_evaluation.feasible:
                raise RuntimeError("Rust mission schedule selected an infeasible subset")
    else:
        for mask in range(1 << len(candidates)):
            selected = tuple(candidates[index] for index in range(len(candidates)) if mask & (1 << index))
            evaluation = _evaluate_selection(parsed, selected)
            evaluated += 1
            if not evaluation.feasible:
                continue
            feasible += 1
            selected_ids = tuple(item.opportunity_id for item in selected)
            if (
                best_evaluation is None
                or evaluation.objective_value > best_evaluation.objective_value
                or (
                    evaluation.objective_value == best_evaluation.objective_value
                    and (len(selected_ids), selected_ids) < (len(best_ids or ()), best_ids or ())
                )
            ):
                best_ids = selected_ids
                best_evaluation = evaluation
    status = "complete" if best_evaluation is not None else "infeasible"
    if best_evaluation is None:
        best_ids = ()
        best_evaluation = _Evaluation(False, "no_feasible_schedule", (), (), 0.0, {})
    by_id = {item.opportunity_id: item for item in candidates}
    selected_set = set(best_ids)
    selected_opportunities = tuple(by_id[item] for item in best_ids)
    rejected: dict[str, str] = {}
    for candidate in candidates:
        if candidate.opportunity_id in selected_set:
            continue
        trial = _evaluate_selection(parsed, (*selected_opportunities, candidate))
        rejected[candidate.opportunity_id] = trial.reason or "not_selected_by_global_objective"
    normalized = parsed.to_dict()
    input_hash = _sha256(normalized)
    schedule_hash = _sha256(
        {
            "schema_version": MISSION_SCHEDULING_EVIDENCE_SCHEMA,
            "input_semantic_sha256": input_hash,
            "status": status,
            "selected_opportunity_ids": best_ids,
            "objective_value": best_evaluation.objective_value,
            "activities": [asdict(item) for item in best_evaluation.activities],
            "deliveries": [asdict(item) for item in best_evaluation.deliveries],
            "resource_summary": best_evaluation.resource_summary,
        }
    )
    summary = {
        "schema_version": MISSION_SCHEDULING_EVIDENCE_SCHEMA,
        "analysis_id": parsed.analysis_id,
        "status": status,
        "solver": "deterministic_exact_exhaustive_enumeration",
        "candidate_count": len(candidates),
        "asset_count": len(parsed.assets),
        "station_count": len({item.station_id for item in candidates if item.station_id}),
        "evaluated_subset_count": evaluated,
        "feasible_subset_count": feasible,
        "selected_count": len(best_ids),
        "selected_observation_count": sum(by_id[item].kind == "observation" for item in best_ids),
        "objective_value": best_evaluation.objective_value,
        "input_semantic_sha256": input_hash,
        "schedule_semantic_sha256": schedule_hash,
        "source_product_sha256s": sorted({item.source_product_sha256 for item in candidates}),
        "claim_limits": [
            "Caller-supplied opportunities and hashes are trusted inputs; the scheduler does not invent access windows.",
            "Slew uses a direct angular-rate bound plus settling, not full attitude dynamics.",
            "Energy is a horizon budget; storage is event-based; battery, thermal, routing, and packet dynamics are excluded.",
            "The exact public solver is bounded to 18 candidates and is not an operational-scale replanner.",
        ],
    }
    return MissionSchedulingResult(
        problem=parsed,
        status=status,
        selected_opportunity_ids=best_ids,
        activities=best_evaluation.activities,
        deliveries=best_evaluation.deliveries,
        rejected_opportunity_reasons=rejected,
        objective_value=best_evaluation.objective_value,
        evaluated_subset_count=evaluated,
        feasible_subset_count=feasible,
        resource_summary=best_evaluation.resource_summary,
        input_semantic_sha256=input_hash,
        schedule_semantic_sha256=schedule_hash,
        summary=summary,
    )


def _authoritative_replay_result(
    problem: MissionSchedulingProblem | Mapping[str, Any],
    *,
    selected_opportunity_ids: Iterable[str],
    expected_input_semantic_sha256: str,
    expected_schedule_semantic_sha256: str,
    expected_status: str = "complete",
) -> MissionSchedulingResult:
    """Return the authoritative result after verifying claimed identities."""

    parsed = problem if isinstance(problem, MissionSchedulingProblem) else MissionSchedulingProblem.from_mapping(problem)
    input_hash = _sha256(parsed.to_dict())
    if input_hash != _validate_sha256(expected_input_semantic_sha256, "expected_input_semantic_sha256"):
        raise MissionSchedulingError("Replay input semantic SHA-256 does not match the normalized problem.")
    selected_values = _bounded_iterable(selected_opportunity_ids, "selected_opportunity_ids")
    ids = tuple(sorted(_required(item, "selected_opportunity_id") for item in selected_values))
    if len(ids) != len(set(ids)):
        raise MissionSchedulingError("Replay selected opportunity IDs must be unique.")
    by_id = {item.opportunity_id: item for item in parsed.opportunities}
    missing = sorted(set(ids) - set(by_id))
    if missing:
        raise MissionSchedulingError(f"Replay names unknown opportunities: {', '.join(missing)}.")
    if expected_status not in {"complete", "infeasible"}:
        raise MissionSchedulingError("expected_status must be complete or infeasible.")
    authoritative = solve_mission_schedule(parsed)
    if expected_status != authoritative.status or ids != authoritative.selected_opportunity_ids:
        raise MissionSchedulingError("Replay selection is not the authoritative exact optimum for this problem.")
    if expected_status == "infeasible" and ids:
        raise MissionSchedulingError("An infeasible evidence packet cannot name selected opportunities.")
    schedule_hash = _validate_sha256(expected_schedule_semantic_sha256, "expected_schedule_semantic_sha256")
    if authoritative.schedule_semantic_sha256 != schedule_hash:
        raise MissionSchedulingError("Authoritative replay schedule SHA-256 mismatch.")
    return authoritative


def _verified_payload(result: MissionSchedulingResult) -> dict[str, Any]:
    return {
        "schema_version": MISSION_SCHEDULING_EVIDENCE_SCHEMA,
        "status": "verified",
        "analysis_id": result.problem.analysis_id,
        "selected_opportunity_ids": list(result.selected_opportunity_ids),
        "input_semantic_sha256": result.input_semantic_sha256,
        "schedule_semantic_sha256": result.schedule_semantic_sha256,
        "objective_value": result.objective_value,
        "activities": [asdict(item) for item in result.activities],
        "deliveries": [asdict(item) for item in result.deliveries],
        "resource_summary": result.resource_summary,
        "summary": result.summary,
    }


def replay_mission_schedule(
    problem: MissionSchedulingProblem | Mapping[str, Any],
    *,
    selected_opportunity_ids: Iterable[str],
    expected_input_semantic_sha256: str,
    expected_schedule_semantic_sha256: str,
    expected_status: str = "complete",
) -> dict[str, Any]:
    """Recompute one selected schedule and verify its content identities."""

    return _verified_payload(
        _authoritative_replay_result(
            problem,
            selected_opportunity_ids=selected_opportunity_ids,
            expected_input_semantic_sha256=expected_input_semantic_sha256,
            expected_schedule_semantic_sha256=expected_schedule_semantic_sha256,
            expected_status=expected_status,
        )
    )


def verify_mission_scheduling_artifacts(evidence_dir: str | Path) -> dict[str, Any]:
    """Verify receipts and every deterministic artifact against authoritative replay."""

    requested_root = Path(evidence_dir).expanduser()
    if requested_root.is_symlink():
        raise MissionSchedulingError("Mission-scheduling evidence directory must not be a symbolic link.")
    root = requested_root.resolve()
    required_artifacts = {
        "normalized_problem.json",
        "mission_schedule_summary.json",
        "mission_schedule.csv",
        "mission_schedule_rejections.csv",
        "mission_resource_summary.csv",
        "mission_data_delivery.csv",
    }
    required_inventory = {*required_artifacts, "mission_schedule_manifest.json"}
    if not root.is_dir():
        raise MissionSchedulingError(f"Mission-scheduling evidence directory does not exist: {root}.")
    inventory = {item.name for item in root.iterdir()}
    if inventory != required_inventory:
        raise MissionSchedulingError("Mission-scheduling evidence directory inventory is not exact.")
    manifest_path = root / "mission_schedule_manifest.json"
    problem_path = root / "normalized_problem.json"
    try:
        total_bytes = 0
        for inventory_path in (root / name for name in sorted(required_inventory)):
            if inventory_path.is_symlink() or not inventory_path.is_file():
                raise MissionSchedulingError(
                    f"Mission-scheduling evidence artifact {inventory_path.name} must be a regular file, not a symbolic link."
                )
            size = inventory_path.stat().st_size
            if size > _MAX_EVIDENCE_ARTIFACT_BYTES:
                raise MissionSchedulingError(
                    f"Mission-scheduling evidence artifact {inventory_path.name} exceeds the public size bound."
                )
            total_bytes += size
        if total_bytes > _MAX_EVIDENCE_TOTAL_BYTES:
            raise MissionSchedulingError("Mission-scheduling evidence exceeds the aggregate public size bound.")
        manifest = _parse_json_bytes(
            read_regular_file_nofollow(manifest_path, min_bytes=1, max_bytes=_MAX_EVIDENCE_ARTIFACT_BYTES),
            "mission-scheduling manifest",
        )
        problem_payload = _parse_json_bytes(
            read_regular_file_nofollow(problem_path, min_bytes=1, max_bytes=_MAX_EVIDENCE_ARTIFACT_BYTES),
            "normalized problem",
        )
    except OSError as exc:
        raise MissionSchedulingError(f"Could not read mission-scheduling evidence in {root}: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(problem_payload, dict):
        raise MissionSchedulingError("Mission-scheduling manifest and normalized problem must be JSON objects.")
    required_manifest_fields = {
        "schema_version",
        "analysis_id",
        "status",
        "selected_opportunity_ids",
        "objective_value",
        "input_semantic_sha256",
        "schedule_semantic_sha256",
        "source_product_sha256s",
        "artifacts",
        "claim_limits",
    }
    if set(manifest) != required_manifest_fields:
        raise MissionSchedulingError("Mission-scheduling manifest field inventory is not exact.")
    if manifest.get("schema_version") != MISSION_SCHEDULING_EVIDENCE_SCHEMA:
        raise MissionSchedulingError("Mission-scheduling manifest has an unsupported schema version.")
    receipts = manifest.get("artifacts")
    if not isinstance(receipts, list) or not receipts:
        raise MissionSchedulingError("Mission-scheduling manifest requires artifact receipts.")
    received_artifacts: set[str] = set()
    artifact_contents: dict[str, bytes] = {}
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise MissionSchedulingError("Every artifact receipt must be a JSON object.")
        if set(receipt) != {"path", "bytes", "sha256"}:
            raise MissionSchedulingError("Artifact receipt field inventory is not exact.")
        relative = Path(str(receipt.get("path", "")))
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
            raise MissionSchedulingError("Artifact receipt paths must be simple relative filenames.")
        if relative.name in received_artifacts:
            raise MissionSchedulingError(f"Duplicate artifact receipt for {relative}.")
        received_artifacts.add(relative.name)
        path = root / relative
        expected_bytes = receipt.get("bytes")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise MissionSchedulingError(f"Artifact receipt size for {relative} must be a nonnegative integer.")
        expected_sha256 = _validate_sha256(receipt.get("sha256"), f"artifact receipt SHA-256 for {relative}")
        try:
            if expected_bytes > _MAX_EVIDENCE_ARTIFACT_BYTES:
                raise MissionSchedulingError(f"Received artifact {relative} exceeds the public size bound.")
            content = read_regular_file_nofollow(path, max_bytes=_MAX_EVIDENCE_ARTIFACT_BYTES)
        except (OSError, SafeReadError) as exc:
            raise MissionSchedulingError(f"Could not read received artifact {relative}: {exc}") from exc
        if len(content) != expected_bytes or hashlib.sha256(content).hexdigest() != expected_sha256:
            raise MissionSchedulingError(f"Artifact receipt mismatch for {relative}.")
        artifact_contents[relative.name] = content
    if received_artifacts != required_artifacts:
        raise MissionSchedulingError("Mission-scheduling manifest does not contain the exact required artifact set.")
    problem = MissionSchedulingProblem.from_mapping(problem_payload)
    authoritative = _authoritative_replay_result(
        problem,
        selected_opportunity_ids=_sequence(manifest.get("selected_opportunity_ids"), "selected_opportunity_ids"),
        expected_input_semantic_sha256=manifest.get("input_semantic_sha256", ""),
        expected_schedule_semantic_sha256=manifest.get("schedule_semantic_sha256", ""),
        expected_status=str(manifest.get("status", "")),
    )
    replay = _verified_payload(authoritative)
    expected_sources = authoritative.summary["source_product_sha256s"]
    if (
        manifest.get("analysis_id") != replay["analysis_id"]
        or manifest.get("objective_value") != replay["objective_value"]
        or manifest.get("source_product_sha256s") != expected_sources
        or manifest.get("claim_limits") != authoritative.summary["claim_limits"]
    ):
        raise MissionSchedulingError("Mission-scheduling manifest claims differ from authoritative replay.")
    expected_artifacts = _render_mission_scheduling_artifacts(authoritative)
    for name, expected_content in expected_artifacts.items():
        if artifact_contents.get(name) != expected_content:
            raise MissionSchedulingError(
                f"Mission-scheduling artifact {name} differs from authoritative deterministic replay."
            )
    return replay


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _dataclass_csv_bytes(values: Iterable[Any], fields: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for value in values:
        writer.writerow(asdict(value))
    return stream.getvalue().encode("utf-8")


def _rejections_csv_bytes(values: Mapping[str, str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("opportunity_id", "reason"))
    writer.writerows(sorted(values.items()))
    return stream.getvalue().encode("utf-8")


_RESOURCE_FIELDS = (
    "asset_id",
    "initial_storage_bytes",
    "final_storage_bytes",
    "peak_storage_bytes",
    "storage_capacity_bytes",
    "energy_used_wh",
    "energy_budget_wh",
    "payload_duty_cycle",
    "data_downlinked_bytes",
)


def _resources_csv_bytes(values: Mapping[str, Mapping[str, float]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_RESOURCE_FIELDS, lineterminator="\n")
    writer.writeheader()
    for asset_id, resource_values in sorted(values.items()):
        writer.writerow({"asset_id": asset_id, **resource_values})
    return stream.getvalue().encode("utf-8")


def _render_mission_scheduling_artifacts(result: MissionSchedulingResult) -> dict[str, bytes]:
    return {
        "normalized_problem.json": _json_bytes(result.problem.to_dict()),
        "mission_schedule_summary.json": _json_bytes(result.summary),
        "mission_schedule.csv": _dataclass_csv_bytes(
            result.activities, tuple(ScheduledActivity.__dataclass_fields__)
        ),
        "mission_schedule_rejections.csv": _rejections_csv_bytes(result.rejected_opportunity_reasons),
        "mission_resource_summary.csv": _resources_csv_bytes(result.resource_summary),
        "mission_data_delivery.csv": _dataclass_csv_bytes(
            result.deliveries, tuple(ObservationDelivery.__dataclass_fields__)
        ),
    }


def _receipt(path: Path, root: Path, content: bytes) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def write_mission_scheduling_artifacts(
    result: MissionSchedulingResult,
    output_dir: str | Path,
) -> MissionSchedulingArtifacts:
    """Atomically publish content-bound JSON/CSV evidence to an absent directory."""

    requested_destination = Path(output_dir).expanduser()
    if requested_destination.is_symlink():
        raise MissionSchedulingError("output_dir must not be a symbolic link.")
    destination = requested_destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise MissionSchedulingError(
            f"output_dir must be absent; refusing to mix evidence or replace {destination}."
        )
    rendered = _render_mission_scheduling_artifacts(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.staging-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for name, content in rendered.items():
            (staging / name).write_bytes(content)
        artifact_receipts = [
            _receipt(staging / name, staging, content) for name, content in rendered.items()
        ]
        manifest_content = _json_bytes(
            {
                "schema_version": MISSION_SCHEDULING_EVIDENCE_SCHEMA,
                "analysis_id": result.problem.analysis_id,
                "status": result.status,
                "selected_opportunity_ids": list(result.selected_opportunity_ids),
                "objective_value": result.objective_value,
                "input_semantic_sha256": result.input_semantic_sha256,
                "schedule_semantic_sha256": result.schedule_semantic_sha256,
                "source_product_sha256s": result.summary["source_product_sha256s"],
                "artifacts": artifact_receipts,
                "claim_limits": result.summary["claim_limits"],
            }
        )
        (staging / "mission_schedule_manifest.json").write_bytes(manifest_content)
        try:
            os.rename(staging, destination)
        except FileExistsError as exc:
            raise MissionSchedulingError(f"output_dir appeared during publication; refusing to replace {destination}.") from exc
    manifest = destination / "mission_schedule_manifest.json"
    problem = destination / "normalized_problem.json"
    summary = destination / "mission_schedule_summary.json"
    schedule = destination / "mission_schedule.csv"
    rejected = destination / "mission_schedule_rejections.csv"
    resources = destination / "mission_resource_summary.csv"
    deliveries = destination / "mission_data_delivery.csv"
    return MissionSchedulingArtifacts(
        output_dir=destination,
        manifest_json=manifest,
        problem_json=problem,
        summary_json=summary,
        schedule_csv=schedule,
        rejected_csv=rejected,
        resources_csv=resources,
        deliveries_csv=deliveries,
    )


__all__ = [
    "MAX_PUBLIC_MISSION_OPPORTUNITIES",
    "MISSION_SCHEDULING_EVIDENCE_SCHEMA",
    "MISSION_SCHEDULING_PROBLEM_SCHEMA",
    "AssetScheduleConstraints",
    "MissionOpportunity",
    "MissionSchedulingArtifacts",
    "MissionSchedulingError",
    "MissionSchedulingProblem",
    "MissionSchedulingResult",
    "ObservationDelivery",
    "ScheduledActivity",
    "replay_mission_schedule",
    "solve_mission_schedule",
    "verify_mission_scheduling_artifacts",
    "write_mission_scheduling_artifacts",
]
