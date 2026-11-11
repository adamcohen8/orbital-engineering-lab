"""Content-bound opportunity generation, scheduling, and delivery estimates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.analysis.coverage_tasking import (
    CoverageTaskingConfig,
    CoverageTaskingResult,
    TaskingConstraints,
    TaskOpportunity,
    optimize_coverage_tasking,
)
from sim.pro_communications.contracts import ResourceLimits
from sim.pro_communications.link_budget import ProLinkBudgetResult

PRO_SCHEDULING_CONTRACT_VERSION = "oel.pro-communications-scheduling.v0.3"


@dataclass(frozen=True)
class ProDownlinkOpportunity:
    opportunity: TaskOpportunity
    source_link_semantic_sha256: str
    data_rate_bps: float
    estimated_data_bits: float
    rate_semantics: str


@dataclass(frozen=True)
class ProScheduleResult:
    tasking: CoverageTaskingResult
    opportunities: tuple[ProDownlinkOpportunity, ...]
    selected_estimated_data_bits: float
    selected_estimated_data_bytes: float
    excluded_by_candidate_bound: tuple[str, ...]
    summary: dict[str, Any]
    semantic_sha256: str


def generate_downlink_opportunities(
    link: ProLinkBudgetResult,
    *,
    asset_id: str,
    energy_cost_w: float = 0.0,
    opportunity_namespace: str | None = None,
    pointing_unit_eci: tuple[float, float, float] | None = None,
    target_id: str | None = None,
) -> tuple[ProDownlinkOpportunity, ...]:
    """Convert sample-bounded link intervals into declared-rate opportunities."""

    if not isinstance(link, ProLinkBudgetResult):
        raise ValueError("link must be a ProLinkBudgetResult.")
    normalized_asset = str(asset_id or "").strip()
    if not normalized_asset:
        raise ValueError("asset_id must be a non-empty string.")
    power_w = float(energy_cost_w)
    if power_w < 0.0:
        raise ValueError("energy_cost_w must be nonnegative.")
    namespace = str(opportunity_namespace or link.profile.profile_id).strip()
    if not namespace:
        raise ValueError("opportunity_namespace must be non-empty when supplied.")
    times = link.samples.time_s
    selected_rate = link.samples.selected_data_rate_bps
    records: list[ProDownlinkOpportunity] = []
    for interval in link.intervals:
        overlap = np.maximum(
            0.0,
            np.minimum(times[1:], interval.end_s) - np.maximum(times[:-1], interval.start_s),
        )
        bits = float(np.dot(selected_rate[:-1], overlap))
        average_rate = 0.0 if interval.duration_s == 0.0 else bits / interval.duration_s
        task = TaskOpportunity(
            opportunity_id=f"{namespace}:downlink:{interval.interval_index:04d}",
            source_product_sha256=link.semantic_sha256,
            asset_id=normalized_asset,
            kind="downlink",
            start_s=interval.start_s,
            end_s=interval.end_s,
            objective_value=bits,
            storage_delta_bytes=-bits / 8.0,
            energy_cost_wh=power_w * interval.duration_s / 3600.0,
            pointing_unit_eci=pointing_unit_eci,
            target_id=target_id,
        )
        records.append(
            ProDownlinkOpportunity(
                opportunity=task,
                source_link_semantic_sha256=link.semantic_sha256,
                data_rate_bps=average_rate,
                estimated_data_bits=bits,
                rate_semantics=(
                    "declared constant user-data rate integrated over a sample-bounded interval"
                    if link.profile.waveform.rate_policy == "fixed"
                    else "profile-selected left-sample-held adaptive user-data rate integrated over the interval"
                ),
            )
        )
    return tuple(records)


def schedule_downlink_opportunities(
    opportunities: tuple[ProDownlinkOpportunity, ...],
    *,
    analysis_id: str,
    asset_id: str,
    constraints: TaskingConstraints,
    resource_limits: ResourceLimits | None = None,
    numeric_backend: str = "rust",
) -> ProScheduleResult:
    """Run the public exact bounded scheduler over content-bound Pro opportunities."""

    if not opportunities:
        raise ValueError("At least one Pro downlink opportunity is required.")
    if any(not isinstance(value, ProDownlinkOpportunity) for value in opportunities):
        raise ValueError("opportunities must contain ProDownlinkOpportunity records.")
    limits = resource_limits or ResourceLimits()
    limits.enforce(schedule_opportunities=len(opportunities))
    source_hashes = {value.source_link_semantic_sha256 for value in opportunities}
    if any(
        value.opportunity.source_product_sha256 != value.source_link_semantic_sha256
        for value in opportunities
    ):
        raise ValueError("Every opportunity task must bind to its typed source-link semantic hash.")
    if constraints.maximum_candidates > 24:
        raise ValueError("The authoritative exact scheduler supports at most 24 candidates.")
    ranked = sorted(
        opportunities,
        key=lambda value: (
            -value.opportunity.objective_value,
            value.opportunity.start_s,
            value.opportunity.opportunity_id,
        ),
    )
    pool = tuple(ranked[: constraints.maximum_candidates])
    excluded = tuple(value.opportunity.opportunity_id for value in ranked[constraints.maximum_candidates :])
    tasking = optimize_coverage_tasking(
        CoverageTaskingConfig(
            analysis_id=analysis_id,
            asset_id=asset_id,
            constraints=constraints,
        ),
        (value.opportunity for value in pool),
        numeric_backend=numeric_backend,
    )
    by_id = {value.opportunity.opportunity_id: value for value in opportunities}
    selected_bits = sum(by_id[identifier].estimated_data_bits for identifier in tasking.selected_opportunity_ids)
    payload = {
        "contract_version": PRO_SCHEDULING_CONTRACT_VERSION,
        "source_link_semantic_sha256": sorted(source_hashes),
        "schedule_semantic_sha256": tasking.schedule_semantic_sha256,
        "selected_estimated_data_bits": selected_bits,
        "rate_semantics": sorted({value.rate_semantics for value in opportunities}),
        "excluded_by_candidate_bound": excluded,
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    summary = {
        **payload,
        "semantic_sha256": semantic_sha256,
        "opportunity_count": len(opportunities),
        "exact_solver_candidate_count": len(pool),
        "selected_count": len(tasking.selected_opportunity_ids),
        "selected_estimated_data_bytes": selected_bits / 8.0,
        "selection_policy": "deterministic value-ranked bound followed by exact public tasking solver",
        "claims": [
            "Deterministic bounded selection over one or more content-bound, sample-bounded link products.",
            "Declared fixed- or profile-selected-rate engineering estimate over selected intervals.",
        ],
        "non_claims": [
            "Candidate preselection above the exact-solver bound is not a globally optimal schedule claim.",
            "No packet overhead, retransmission, routing, buffering, latency, or delivery assurance.",
        ],
    }
    return ProScheduleResult(
        tasking=tasking,
        opportunities=opportunities,
        selected_estimated_data_bits=selected_bits,
        selected_estimated_data_bytes=selected_bits / 8.0,
        excluded_by_candidate_bound=excluded,
        summary=summary,
        semantic_sha256=semantic_sha256,
    )


__all__ = [
    "PRO_SCHEDULING_CONTRACT_VERSION",
    "ProDownlinkOpportunity",
    "ProScheduleResult",
    "generate_downlink_opportunities",
    "schedule_downlink_opportunities",
]
