"""Deterministic split-half stability evidence for Pro uncertainty campaigns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sim.pro_communications.campaign import ProCampaignResult

PRO_CAMPAIGN_CONVERGENCE_VERSION = "oel.pro-communications-campaign-convergence.v0.1"


@dataclass(frozen=True)
class CampaignConvergenceCriteria:
    maximum_mean_margin_absolute_delta_db: float
    maximum_p10_margin_absolute_delta_db: float
    maximum_mean_delivery_relative_delta: float
    maximum_delivery_probability_absolute_delta: float
    minimum_cases_per_half: int = 8

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_mean_margin_absolute_delta_db",
            "maximum_p10_margin_absolute_delta_db",
            "maximum_mean_delivery_relative_delta",
            "maximum_delivery_probability_absolute_delta",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative.")
            object.__setattr__(self, field_name, value)
        count = self.minimum_cases_per_half
        if isinstance(count, (bool, np.bool_)) or int(count) != count or int(count) < 2:
            raise ValueError("minimum_cases_per_half must be an integer of at least two.")
        object.__setattr__(self, "minimum_cases_per_half", int(count))


@dataclass(frozen=True)
class CampaignConvergenceResult:
    campaign_id: str
    campaign_semantic_sha256: str
    method: str
    criteria: CampaignConvergenceCriteria
    metrics: dict[str, float]
    passed: bool
    disposition: str
    summary: dict[str, Any]
    semantic_sha256: str


def _mean_delivery_relative_delta(first: np.ndarray, second: np.ndarray) -> float:
    first_mean = float(np.mean(first))
    second_mean = float(np.mean(second))
    denominator = max(abs(first_mean), abs(second_mean), 1.0)
    return abs(first_mean - second_mean) / denominator


def evaluate_campaign_convergence(
    campaign: ProCampaignResult,
    criteria: CampaignConvergenceCriteria,
) -> CampaignConvergenceResult:
    """Evaluate deterministic odd/even split stability without asserting asymptotic convergence."""

    if not isinstance(campaign, ProCampaignResult) or campaign.kind != "seeded_correlated_normal_uncertainty":
        raise ValueError("Campaign convergence requires a seeded correlated uncertainty campaign.")
    if not isinstance(criteria, CampaignConvergenceCriteria):
        raise ValueError("criteria must be CampaignConvergenceCriteria.")
    odd_cases = campaign.cases[0::2]
    even_cases = campaign.cases[1::2]
    if min(len(odd_cases), len(even_cases)) < criteria.minimum_cases_per_half:
        raise ValueError("Campaign does not contain enough cases for the declared minimum_cases_per_half.")
    odd_margin = np.asarray([value.minimum_margin_db for value in odd_cases], dtype=float)
    even_margin = np.asarray([value.minimum_margin_db for value in even_cases], dtype=float)
    odd_delivery = np.asarray([value.estimated_delivered_data_bits for value in odd_cases], dtype=float)
    even_delivery = np.asarray([value.estimated_delivered_data_bits for value in even_cases], dtype=float)
    metrics = {
        "mean_margin_absolute_delta_db": abs(float(np.mean(odd_margin)) - float(np.mean(even_margin))),
        "p10_margin_absolute_delta_db": abs(float(np.quantile(odd_margin, 0.1)) - float(np.quantile(even_margin, 0.1))),
        "mean_delivery_relative_delta": _mean_delivery_relative_delta(odd_delivery, even_delivery),
        "delivery_probability_absolute_delta": abs(
            float(np.mean(odd_delivery > 0.0)) - float(np.mean(even_delivery > 0.0))
        ),
        "odd_case_count": float(len(odd_cases)),
        "even_case_count": float(len(even_cases)),
    }
    passed = (
        metrics["mean_margin_absolute_delta_db"] <= criteria.maximum_mean_margin_absolute_delta_db
        and metrics["p10_margin_absolute_delta_db"] <= criteria.maximum_p10_margin_absolute_delta_db
        and metrics["mean_delivery_relative_delta"] <= criteria.maximum_mean_delivery_relative_delta
        and metrics["delivery_probability_absolute_delta"] <= criteria.maximum_delivery_probability_absolute_delta
    )
    payload = {
        "contract_version": PRO_CAMPAIGN_CONVERGENCE_VERSION,
        "campaign_id": campaign.campaign_id,
        "campaign_semantic_sha256": campaign.semantic_sha256,
        "method": "deterministic odd-even split-half stability",
        "criteria": asdict(criteria),
        "metrics": metrics,
        "passed": passed,
        "disposition": "within_declared_limits" if passed else "outside_declared_limits",
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    summary = {
        **payload,
        "semantic_sha256": semantic_sha256,
        "claims": ["Deterministic split-half stability result for the exact retained seeded campaign and criteria."],
        "non_claims": [
            "Passing split-half limits is not proof of asymptotic convergence, calibrated probability, or operational assurance."
        ],
    }
    return CampaignConvergenceResult(
        campaign_id=campaign.campaign_id,
        campaign_semantic_sha256=campaign.semantic_sha256,
        method=payload["method"],
        criteria=criteria,
        metrics=metrics,
        passed=passed,
        disposition=payload["disposition"],
        summary=summary,
        semantic_sha256=semantic_sha256,
    )


__all__ = [
    "PRO_CAMPAIGN_CONVERGENCE_VERSION",
    "CampaignConvergenceCriteria",
    "CampaignConvergenceResult",
    "evaluate_campaign_convergence",
]
