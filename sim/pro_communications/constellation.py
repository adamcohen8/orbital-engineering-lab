"""Pro RF-service constellation aggregation over validated member products."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Iterable

from sim.analysis.coverage_aggregation import (
    ConstellationCoverageConfig,
    ConstellationCoverageResult,
    evaluate_constellation_coverage,
)
from sim.pro_communications.contracts import ResourceLimits
from sim.pro_communications.coverage import ProCommunicationsCoverageResult

PRO_CONSTELLATION_SERVICE_VERSION = "oel.pro-communications-constellation-service.v0.1"


def _service_definition(members: tuple[ProCommunicationsCoverageResult, ...]) -> tuple[str, dict[str, object]]:
    records = []
    for member in members:
        record = asdict(member.config)
        for field_name in (
            "analysis_id",
            "source_asset_id",
            "state_provider_id",
            "attitude_provider_id",
            "chunk_size",
        ):
            record.pop(field_name)
        record["link_profile_sha256"] = member.config.link_profile.semantic_sha256
        record.pop("link_profile")
        records.append(record)
    if any(record != records[0] for record in records[1:]):
        raise ValueError("Constellation members must share one exact Pro RF service definition.")
    payload = {
        "contract_version": PRO_CONSTELLATION_SERVICE_VERSION,
        "scientific_service_definition": records[0],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return digest, payload


def evaluate_pro_constellation_service(
    *,
    analysis_id: str,
    members: Iterable[ProCommunicationsCoverageResult],
    required_multiplicity: int = 1,
    resource_limits: ResourceLimits | None = None,
) -> ConstellationCoverageResult:
    """Aggregate synchronized RF-qualified members under one hashed service definition."""

    products = tuple(sorted(tuple(members), key=lambda value: value.config.analysis_id))
    if len(products) < 2 or any(not isinstance(value, ProCommunicationsCoverageResult) for value in products):
        raise ValueError("Pro constellation service requires at least two validated Pro coverage members.")
    limits = resource_limits or ResourceLimits()
    limits.enforce(constellation_members=len(products))
    service_definition_id, service_payload = _service_definition(products)
    result = evaluate_constellation_coverage(
        ConstellationCoverageConfig(
            analysis_id=analysis_id,
            member_analysis_ids=tuple(value.config.analysis_id for value in products),
            order=products[0].config.order,
            service_definition_id=service_definition_id,
            required_multiplicity=required_multiplicity,
        ),
        products,
    )
    return replace(
        result,
        summary={
            **result.summary,
            "pro_contract_version": PRO_CONSTELLATION_SERVICE_VERSION,
            "pro_service_definition": service_payload,
            "claims": [
                "Deterministic synchronized multiplicity for RF-qualified member coverage under one exact service definition."
            ],
            "non_claims": [
                "No terminal contention, routing, probabilistic availability, or unconstrained constellation optimization."
            ],
        },
    )


__all__ = [
    "PRO_CONSTELLATION_SERVICE_VERSION",
    "evaluate_pro_constellation_service",
]
