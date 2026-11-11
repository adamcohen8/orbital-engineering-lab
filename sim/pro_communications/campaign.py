"""Bounded reproducible Pro communications sensitivity and uncertainty campaigns."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

import numpy as np

from sim.pro_communications.contracts import LinkBudgetProfile, ResourceLimits
from sim.pro_communications.link_budget import (
    ProLinkBudgetResult,
    _prepare_pro_link_inputs,
    _PreparedProLinkInputs,
    evaluate_pro_link_budget,
)

PRO_CAMPAIGN_CONTRACT_VERSION = "oel.pro-communications-campaign.v0.1"
_SUPPORTED_PARAMETERS = {
    "transmitter.power_w",
    "receiver.system_noise_temperature_k",
    "waveform.data_rate_bps",
    "waveform.required_eb_n0_db",
    "atmosphere.rainfall_rate_mm_h",
    "atmosphere.rain_exceedance_percentage",
    "atmosphere.integrated_cloud_liquid_water_kg_m2",
    "polarization_loss_db",
    "pointing_loss_db",
    "implementation_loss_db",
}


def _immutable_array_copy(values: np.ndarray) -> np.ndarray:
    """Return a C-order array whose ultimate backing store is immutable bytes."""

    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class LinkGeometryEvidence:
    source_id: str
    times_s: np.ndarray
    range_km: np.ndarray
    elevation_rad: np.ndarray
    tx_off_axis_rad: np.ndarray | None = None
    rx_off_axis_rad: np.ndarray | None = None
    range_rate_km_s: np.ndarray | None = None
    earth_clear: np.ndarray | None = None

    def __post_init__(self) -> None:
        source = str(self.source_id or "").strip()
        if not source:
            raise ValueError("source_id must be a non-empty string.")
        object.__setattr__(self, "source_id", source)
        times = np.array(self.times_s, dtype=float, copy=True)
        if times.ndim != 1 or times.size < 2 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be a strictly increasing finite vector with at least two samples.")
        times = _immutable_array_copy(times)
        object.__setattr__(self, "times_s", times)
        for field_name in ("range_km", "elevation_rad", "tx_off_axis_rad", "rx_off_axis_rad", "range_rate_km_s"):
            raw = getattr(self, field_name)
            if raw is None:
                continue
            values = np.array(raw, dtype=float, copy=True)
            if values.shape != times.shape or np.any(~np.isfinite(values)):
                raise ValueError(f"{field_name} must match times_s and contain finite values.")
            values = _immutable_array_copy(values)
            object.__setattr__(self, field_name, values)
        if np.any(np.asarray(self.range_km) <= 0.0):
            raise ValueError("range_km must be positive.")
        if self.earth_clear is not None:
            clear = np.array(self.earth_clear, dtype=bool, copy=True)
            if clear.shape != times.shape:
                raise ValueError("earth_clear must match times_s.")
            clear = _immutable_array_copy(clear)
            object.__setattr__(self, "earth_clear", clear)

    @cached_property
    def semantic_sha256(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {"contract": "oel.pro-link-geometry.v1", "source_id": self.source_id},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for field_name in (
            "times_s",
            "range_km",
            "elevation_rad",
            "tx_off_axis_rad",
            "rx_off_axis_rad",
            "range_rate_km_s",
        ):
            values = getattr(self, field_name)
            digest.update(field_name.encode())
            digest.update(b"none" if values is None else np.ascontiguousarray(values, dtype="<f8").tobytes())
        digest.update(b"earth_clear")
        digest.update(
            b"none" if self.earth_clear is None else np.ascontiguousarray(self.earth_clear, dtype=np.uint8).tobytes()
        )
        return digest.hexdigest()


@dataclass(frozen=True)
class TradeDimension:
    parameter: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        parameter = str(self.parameter or "").strip()
        if parameter not in _SUPPORTED_PARAMETERS:
            raise ValueError(f"Unsupported campaign parameter {parameter!r}.")
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or not 1 <= values.size <= 16 or np.any(~np.isfinite(values)):
            raise ValueError("Trade values must contain 1 to 16 finite scalars.")
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "values", tuple(float(value) for value in values))


@dataclass(frozen=True)
class UncertaintyDimension:
    parameter: str
    mean: float
    standard_deviation: float

    def __post_init__(self) -> None:
        parameter = str(self.parameter or "").strip()
        if parameter not in _SUPPORTED_PARAMETERS:
            raise ValueError(f"Unsupported campaign parameter {parameter!r}.")
        mean = float(self.mean)
        deviation = float(self.standard_deviation)
        if not np.isfinite(mean) or not np.isfinite(deviation) or deviation < 0.0:
            raise ValueError("Uncertainty mean and standard deviation must be finite, with nonnegative deviation.")
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", deviation)


@dataclass(frozen=True)
class ProCampaignCaseResult:
    case_id: str
    parameter_values: dict[str, float]
    profile_sha256: str
    link_semantic_sha256: str
    minimum_margin_db: float
    maximum_margin_db: float
    available_duration_s: float
    estimated_delivered_data_bits: float


@dataclass(frozen=True)
class ProCampaignResult:
    campaign_id: str
    kind: str
    geometry_sha256: str
    seed: int | None
    correlation_matrix: tuple[tuple[float, ...], ...] | None
    cases: tuple[ProCampaignCaseResult, ...]
    summary: dict[str, Any]
    semantic_sha256: str


def _replace_parameter(profile: LinkBudgetProfile, parameter: str, value: float) -> LinkBudgetProfile:
    owner, separator, field_name = parameter.partition(".")
    if not separator:
        return replace(profile, **{field_name or owner: value})
    nested = getattr(profile, owner)
    if owner == "transmitter" and field_name == "power_w":
        nested = replace(
            nested,
            power_w=value,
            commanded_power_w=None,
            available_power_w=None,
            amplifier_backoff_db=0.0,
        )
    elif owner == "receiver" and field_name == "system_noise_temperature_k" and nested.g_over_t_db_per_k is not None:
        nested = replace(nested, g_over_t_db_per_k=None, system_noise_temperature_k=value)
    else:
        nested = replace(nested, **{field_name: value})
    return replace(profile, **{owner: nested})


def _evaluate_case(
    case_id: str,
    base_profile: LinkBudgetProfile,
    geometry: LinkGeometryEvidence,
    values: dict[str, float],
    limits: ResourceLimits,
    prepared_inputs: _PreparedProLinkInputs,
) -> ProCampaignCaseResult:
    profile = base_profile
    for parameter, value in sorted(values.items()):
        profile = _replace_parameter(profile, parameter, value)
    link: ProLinkBudgetResult = evaluate_pro_link_budget(
        profile,
        times_s=geometry.times_s,
        range_km=geometry.range_km,
        elevation_rad=geometry.elevation_rad,
        tx_off_axis_rad=geometry.tx_off_axis_rad,
        rx_off_axis_rad=geometry.rx_off_axis_rad,
        range_rate_km_s=geometry.range_rate_km_s,
        earth_clear=geometry.earth_clear,
        resource_limits=limits,
        _prepared_inputs=prepared_inputs,
    )
    return ProCampaignCaseResult(
        case_id=case_id,
        parameter_values=dict(sorted(values.items())),
        profile_sha256=profile.semantic_sha256,
        link_semantic_sha256=link.semantic_sha256,
        minimum_margin_db=float(link.summary["minimum_margin_db"]),
        maximum_margin_db=float(link.summary["maximum_margin_db"]),
        available_duration_s=float(link.summary["available_duration_s"]),
        estimated_delivered_data_bits=float(link.summary["estimated_delivered_data_bits"]),
    )


def _finish_campaign(
    *,
    campaign_id: str,
    kind: str,
    geometry: LinkGeometryEvidence,
    seed: int | None,
    correlation_matrix: tuple[tuple[float, ...], ...] | None,
    cases: tuple[ProCampaignCaseResult, ...],
    input_contract: dict[str, Any],
) -> ProCampaignResult:
    payload = {
        "contract_version": PRO_CAMPAIGN_CONTRACT_VERSION,
        "campaign_id": campaign_id,
        "kind": kind,
        "geometry_sha256": geometry.semantic_sha256,
        "seed": seed,
        "correlation_matrix": correlation_matrix,
        "input_contract": input_contract,
        "cases": [case.__dict__ for case in cases],
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    margins = np.asarray([case.minimum_margin_db for case in cases])
    delivery = np.asarray([case.estimated_delivered_data_bits for case in cases])
    summary = {
        "contract_version": PRO_CAMPAIGN_CONTRACT_VERSION,
        "campaign_id": campaign_id,
        "kind": kind,
        "case_count": len(cases),
        "geometry_sha256": geometry.semantic_sha256,
        "minimum_case_margin_db": float(np.min(margins)),
        "maximum_case_margin_db": float(np.max(margins)),
        "minimum_estimated_delivered_data_bits": float(np.min(delivery)),
        "maximum_estimated_delivered_data_bits": float(np.max(delivery)),
        "semantic_sha256": semantic_sha256,
        "claims": ["Reproducible bounded trade or uncertainty results for the declared inputs."],
        "non_claims": [
            "No convergence claim beyond the retained cases.",
            "No operational probability, calibration, or unconstrained optimization claim.",
        ],
    }
    return ProCampaignResult(
        campaign_id=campaign_id,
        kind=kind,
        geometry_sha256=geometry.semantic_sha256,
        seed=seed,
        correlation_matrix=correlation_matrix,
        cases=cases,
        summary=summary,
        semantic_sha256=semantic_sha256,
    )


def run_trade_campaign(
    campaign_id: str,
    base_profile: LinkBudgetProfile,
    geometry: LinkGeometryEvidence,
    dimensions: tuple[TradeDimension, ...],
    *,
    resource_limits: ResourceLimits | None = None,
) -> ProCampaignResult:
    if not dimensions:
        raise ValueError("A trade campaign requires at least one dimension.")
    parameters = [dimension.parameter for dimension in dimensions]
    if len(set(parameters)) != len(parameters):
        raise ValueError("Trade campaign parameters must be unique.")
    cases_raw = list(itertools.product(*(dimension.values for dimension in dimensions)))
    limits = resource_limits or ResourceLimits()
    limits.enforce(campaign_cases=len(cases_raw))
    prepared_inputs = _prepare_pro_link_inputs(
        times_s=geometry.times_s,
        range_km=geometry.range_km,
        elevation_rad=geometry.elevation_rad,
        tx_off_axis_rad=geometry.tx_off_axis_rad,
        rx_off_axis_rad=geometry.rx_off_axis_rad,
        range_rate_km_s=geometry.range_rate_km_s,
        earth_clear=geometry.earth_clear,
        minimum_elevation_rad=np.deg2rad(5.0),
    )
    cases = tuple(
        _evaluate_case(
            f"{campaign_id}:{index:04d}",
            base_profile,
            geometry,
            dict(zip(parameters, values)),
            limits,
            prepared_inputs,
        )
        for index, values in enumerate(cases_raw)
    )
    return _finish_campaign(
        campaign_id=campaign_id,
        kind="deterministic_cartesian_trade",
        geometry=geometry,
        seed=None,
        correlation_matrix=None,
        cases=cases,
        input_contract={dimension.parameter: list(dimension.values) for dimension in dimensions},
    )


def run_uncertainty_campaign(
    campaign_id: str,
    base_profile: LinkBudgetProfile,
    geometry: LinkGeometryEvidence,
    dimensions: tuple[UncertaintyDimension, ...],
    *,
    case_count: int,
    seed: int,
    correlation_matrix: np.ndarray | None = None,
    resource_limits: ResourceLimits | None = None,
) -> ProCampaignResult:
    if not dimensions:
        raise ValueError("An uncertainty campaign requires at least one dimension.")
    parameters = [dimension.parameter for dimension in dimensions]
    if len(set(parameters)) != len(parameters):
        raise ValueError("Uncertainty parameters must be unique.")
    if isinstance(case_count, (bool, np.bool_)) or int(case_count) != case_count or int(case_count) < 1:
        raise ValueError("case_count must be a positive integer.")
    limits = resource_limits or ResourceLimits()
    limits.enforce(campaign_cases=case_count)
    count = int(case_count)
    normalized_seed = int(seed)
    dimension_count = len(dimensions)
    if correlation_matrix is None:
        correlation = np.eye(dimension_count)
    else:
        correlation = np.asarray(correlation_matrix, dtype=float)
        if correlation.shape != (dimension_count, dimension_count) or np.any(~np.isfinite(correlation)):
            raise ValueError("correlation_matrix must be a finite square matrix matching dimensions.")
        if not np.allclose(correlation, correlation.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("correlation_matrix must be symmetric.")
        if not np.allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("correlation_matrix must have a unit diagonal.")
        if float(np.min(np.linalg.eigvalsh(correlation))) < -1.0e-12:
            raise ValueError("correlation_matrix must be positive semidefinite.")
    rng = np.random.default_rng(normalized_seed)
    samples = rng.multivariate_normal(np.zeros(dimension_count), correlation, size=count, check_valid="raise")
    means = np.asarray([dimension.mean for dimension in dimensions])
    deviations = np.asarray([dimension.standard_deviation for dimension in dimensions])
    samples = means + samples * deviations
    prepared_inputs = _prepare_pro_link_inputs(
        times_s=geometry.times_s,
        range_km=geometry.range_km,
        elevation_rad=geometry.elevation_rad,
        tx_off_axis_rad=geometry.tx_off_axis_rad,
        rx_off_axis_rad=geometry.rx_off_axis_rad,
        range_rate_km_s=geometry.range_rate_km_s,
        earth_clear=geometry.earth_clear,
        minimum_elevation_rad=np.deg2rad(5.0),
    )
    cases = tuple(
        _evaluate_case(
            f"{campaign_id}:{index:04d}",
            base_profile,
            geometry,
            dict(zip(parameters, row)),
            limits,
            prepared_inputs,
        )
        for index, row in enumerate(samples)
    )
    frozen_correlation = tuple(tuple(float(value) for value in row) for row in correlation)
    return _finish_campaign(
        campaign_id=campaign_id,
        kind="seeded_correlated_normal_uncertainty",
        geometry=geometry,
        seed=normalized_seed,
        correlation_matrix=frozen_correlation,
        cases=cases,
        input_contract={
            dimension.parameter: {
                "distribution": "normal",
                "mean": dimension.mean,
                "standard_deviation": dimension.standard_deviation,
            }
            for dimension in dimensions
        },
    )


__all__ = [
    "LinkGeometryEvidence",
    "PRO_CAMPAIGN_CONTRACT_VERSION",
    "ProCampaignCaseResult",
    "ProCampaignResult",
    "TradeDimension",
    "UncertaintyDimension",
    "run_trade_campaign",
    "run_uncertainty_campaign",
]
