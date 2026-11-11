"""Bounded ground-network and constellation design-trade automation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.pro_communications.campaign import LinkGeometryEvidence
from sim.pro_communications.constellation import evaluate_pro_constellation_service
from sim.pro_communications.contracts import LinkBudgetProfile, ResourceLimits
from sim.pro_communications.coverage import ProCommunicationsCoverageResult
from sim.pro_communications.link_budget import evaluate_pro_link_budget

PRO_NETWORK_TRADE_VERSION = "oel.pro-communications-network-trade.v0.1"


def _required(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


@dataclass(frozen=True)
class GroundNetworkSite:
    geometry: LinkGeometryEvidence
    minimum_elevation_rad: float = np.deg2rad(5.0)
    maximum_range_km: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, LinkGeometryEvidence):
            raise ValueError("Ground-network sites require LinkGeometryEvidence.")
        elevation = float(self.minimum_elevation_rad)
        if not np.isfinite(elevation) or not -0.5 * np.pi <= elevation <= 0.5 * np.pi:
            raise ValueError("minimum_elevation_rad must be finite within [-pi/2, pi/2].")
        object.__setattr__(self, "minimum_elevation_rad", elevation)
        if self.maximum_range_km is not None:
            maximum_range = float(self.maximum_range_km)
            if not np.isfinite(maximum_range) or maximum_range <= 0.0:
                raise ValueError("maximum_range_km must be positive and finite when supplied.")
            object.__setattr__(self, "maximum_range_km", maximum_range)


@dataclass(frozen=True)
class GroundNetworkAlternative:
    alternative_id: str
    sites: Mapping[str, GroundNetworkSite | LinkGeometryEvidence]

    def __post_init__(self) -> None:
        object.__setattr__(self, "alternative_id", _required(self.alternative_id, "alternative_id"))
        if not self.sites:
            raise ValueError("Ground-network alternatives require one or more sites.")
        normalized = {
            str(key).strip(): value if isinstance(value, GroundNetworkSite) else GroundNetworkSite(value)
            for key, value in sorted(self.sites.items())
        }
        if any(not key for key in normalized) or len(normalized) != len(self.sites):
            raise ValueError("Ground-network site IDs must be unique non-empty strings.")
        object.__setattr__(self, "sites", normalized)


@dataclass(frozen=True)
class ProDesignTradeCase:
    rank: int
    alternative_id: str
    source_semantic_sha256: tuple[str, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ProDesignTradeResult:
    trade_id: str
    kind: str
    objective: str
    cases: tuple[ProDesignTradeCase, ...]
    recommended_alternative_id: str
    summary: Mapping[str, Any]
    semantic_sha256: str


def _finish(
    *,
    trade_id: str,
    kind: str,
    objective: str,
    raw_cases: list[tuple[str, tuple[str, ...], dict[str, float]]],
) -> ProDesignTradeResult:
    ranked = sorted(raw_cases, key=lambda value: (-value[2][objective], value[0]))
    cases = tuple(
        ProDesignTradeCase(
            rank=index + 1,
            alternative_id=alternative_id,
            source_semantic_sha256=hashes,
            metrics=dict(sorted(metrics.items())),
        )
        for index, (alternative_id, hashes, metrics) in enumerate(ranked)
    )
    payload = {
        "contract_version": PRO_NETWORK_TRADE_VERSION,
        "trade_id": trade_id,
        "kind": kind,
        "objective": objective,
        "decision_rule": f"maximize {objective}; break exact ties by lexicographically smallest alternative_id",
        "cases": [case.__dict__ for case in cases],
        "recommended_alternative_id": cases[0].alternative_id,
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    summary = {
        **payload,
        "semantic_sha256": semantic_sha256,
        "claims": ["Deterministic bounded comparison and recommendation under the declared objective."],
        "non_claims": [
            "The recommendation is conditional on supplied alternatives and does not claim unconstrained optimization, cost, staffing, or operational suitability."
        ],
    }
    return ProDesignTradeResult(
        trade_id=trade_id,
        kind=kind,
        objective=objective,
        cases=cases,
        recommended_alternative_id=cases[0].alternative_id,
        summary=summary,
        semantic_sha256=semantic_sha256,
    )


def run_ground_network_trade(
    trade_id: str,
    profile: LinkBudgetProfile,
    alternatives: tuple[GroundNetworkAlternative, ...],
    *,
    resource_limits: ResourceLimits | None = None,
) -> ProDesignTradeResult:
    """Compare declared ground networks by summed contact delivery without contention."""

    identity = _required(trade_id, "trade_id")
    if not alternatives or len({value.alternative_id for value in alternatives}) != len(alternatives):
        raise ValueError("Ground-network alternatives must be non-empty with unique IDs.")
    limits = resource_limits or ResourceLimits()
    limits.enforce(
        campaign_cases=len(alternatives),
        terminals=max(len(value.sites) for value in alternatives),
    )
    raw_cases = []
    for alternative in sorted(alternatives, key=lambda value: value.alternative_id):
        links = [
            evaluate_pro_link_budget(
                profile,
                times_s=site.geometry.times_s,
                range_km=site.geometry.range_km,
                elevation_rad=site.geometry.elevation_rad,
                tx_off_axis_rad=site.geometry.tx_off_axis_rad,
                rx_off_axis_rad=site.geometry.rx_off_axis_rad,
                range_rate_km_s=site.geometry.range_rate_km_s,
                earth_clear=site.geometry.earth_clear,
                minimum_elevation_rad=site.minimum_elevation_rad,
                maximum_range_km=site.maximum_range_km,
                resource_limits=limits,
            )
            for site in alternative.sites.values()
        ]
        raw_cases.append(
            (
                alternative.alternative_id,
                tuple(value.semantic_sha256 for value in links),
                {
                    "site_count": float(len(links)),
                    "summed_available_duration_s": float(sum(value.summary["available_duration_s"] for value in links)),
                    "summed_estimated_delivered_data_bits": float(
                        sum(value.summary["estimated_delivered_data_bits"] for value in links)
                    ),
                    "worst_minimum_margin_db": float(min(value.summary["minimum_margin_db"] for value in links)),
                    "best_maximum_margin_db": float(max(value.summary["maximum_margin_db"] for value in links)),
                },
            )
        )
    return _finish(
        trade_id=identity,
        kind="bounded_ground_network_trade",
        objective="summed_estimated_delivered_data_bits",
        raw_cases=raw_cases,
    )


def run_constellation_service_trade(
    trade_id: str,
    alternatives: Mapping[str, tuple[ProCommunicationsCoverageResult, ...]],
    *,
    required_multiplicity: int = 1,
    resource_limits: ResourceLimits | None = None,
) -> ProDesignTradeResult:
    """Compare supplied synchronized constellations by mean RF-qualified coverage."""

    identity = _required(trade_id, "trade_id")
    if not alternatives:
        raise ValueError("At least one constellation alternative is required.")
    limits = resource_limits or ResourceLimits()
    limits.enforce(
        campaign_cases=len(alternatives),
        constellation_members=max(len(value) for value in alternatives.values()),
    )
    raw_cases = []
    for alternative_id, members in sorted(alternatives.items()):
        result = evaluate_pro_constellation_service(
            analysis_id=f"{identity}:{alternative_id}",
            members=members,
            required_multiplicity=required_multiplicity,
            resource_limits=limits,
        )
        duration = float(result.times_s[-1] - result.times_s[0])
        mean_fraction = float(np.dot(result.instantaneous_covered_fraction[:-1], np.diff(result.times_s)) / duration)
        raw_cases.append(
            (
                str(alternative_id),
                (result.interval_semantic_sha256,),
                {
                    "member_count": float(len(members)),
                    "time_weighted_mean_covered_fraction": mean_fraction,
                    "ever_covered_fraction": float(
                        np.count_nonzero(result.cell_metrics.interval_count) / result.cell_metrics.interval_count.size
                    ),
                    "maximum_multiplicity": float(np.max(result.max_multiplicity_per_cell)),
                },
            )
        )
    return _finish(
        trade_id=identity,
        kind="bounded_constellation_service_trade",
        objective="time_weighted_mean_covered_fraction",
        raw_cases=raw_cases,
    )


def write_design_trade_packet(result: ProDesignTradeResult, path: str | Path) -> Path:
    """Write a fail-on-existing review-ready design-trade packet."""

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "PRO_NETWORK_TRADE_VERSION",
    "GroundNetworkAlternative",
    "GroundNetworkSite",
    "ProDesignTradeCase",
    "ProDesignTradeResult",
    "run_constellation_service_trade",
    "run_ground_network_trade",
    "write_design_trade_packet",
]
