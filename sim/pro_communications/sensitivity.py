"""Pro coverage cadence and resolution sensitivity adapters."""

from __future__ import annotations

from sim.analysis.coverage_sensitivity import (
    CoverageSensitivityCriteria,
    CoverageSensitivityResult,
    evaluate_coverage_sensitivity,
)
from sim.pro_communications.coverage import ProCommunicationsCoverageResult


def evaluate_pro_coverage_sensitivity(
    *,
    comparison_id: str,
    comparison_kind: str,
    baseline: ProCommunicationsCoverageResult,
    refined: ProCommunicationsCoverageResult,
    criteria: CoverageSensitivityCriteria,
) -> CoverageSensitivityResult:
    """Evaluate source-bound sensitivity for structurally valid Pro coverage products."""

    if not isinstance(baseline, ProCommunicationsCoverageResult) or not isinstance(
        refined, ProCommunicationsCoverageResult
    ):
        raise ValueError("Pro sensitivity requires ProCommunicationsCoverageResult inputs.")
    return evaluate_coverage_sensitivity(
        comparison_id=comparison_id,
        comparison_kind=comparison_kind,
        baseline=baseline,
        refined=refined,
        criteria=criteria,
    )


__all__ = [
    "CoverageSensitivityCriteria",
    "CoverageSensitivityResult",
    "evaluate_pro_coverage_sensitivity",
]
