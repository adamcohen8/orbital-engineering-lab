from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .manifests import bind_output_path
from .scenario_patches import (
    ScenarioPatchError,
    build_scenario_patch_product,
    write_scenario_patch_product,
)

SCENARIO_OVERLAY_ADAPTER_ID = "oel.scenario_capability_overlay"
SCENARIO_OVERLAY_ADAPTER_VERSION = "1"
_OBJECT_FIELDS = {
    "base_guidance",
    "flight_software",
    "knowledge",
}
_ANALYSIS_FIELDS = {
    "enabled",
    "study_type",
    "execution",
    "metrics",
    "baseline",
    "monte_carlo",
    "sensitivity",
    "covariance",
    "mission_recovery",
    "orbital_delivery",
}


class ScenarioOverlayError(ScenarioPatchError):
    """Raised when a capability overlay exceeds the bounded scenario surface."""


def load_scenario_overlay(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScenarioOverlayError("Scenario overlay must contain a YAML or JSON mapping.")
    return raw


def emit_scenario_overlay(
    source_scenario: str | Path,
    overlay: Mapping[str, Any],
    *,
    overlay_id: str,
    output_path: str | Path,
    logical_output_path: str | Path | None = None,
    writer: Any | None = None,
    rationale: str,
    data_markings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    overlay_key = str(overlay_id or "").strip()
    if not overlay_key:
        raise ScenarioOverlayError("overlay_id must be non-empty.")
    reason = str(rationale or "").strip()
    if not reason:
        raise ScenarioOverlayError("rationale must be non-empty.")
    operations = _overlay_operations(overlay, rationale=reason)
    target, logical_target = bind_output_path(
        output_path,
        logical_output_path,
        writer=writer,
    )
    try:
        product = build_scenario_patch_product(
            source_scenario,
            patch_type="scenario_capability_overlay",
            selection={
                "selection_id": overlay_key,
                "selection_kind": "scenario_capability_overlay",
                "rank": 1,
                "recommended_modes": ["explicit_overlay"],
            },
            operations=operations,
            evidence={
                "objective": {"overlay_id": overlay_key, "rationale": reason},
                "constraints": {
                    "bounded_top_level_fields": [
                        "analysis",
                        "ground_stations",
                        "objects",
                        "outputs.review",
                        "simulator.termination",
                    ]
                },
                "producer_evidence": {"overlay": deepcopy(dict(overlay))},
            },
            producer_capability_id=SCENARIO_OVERLAY_ADAPTER_ID,
            producer_run_id=overlay_key,
            disposition="accepted",
            producer_status="explicit_overlay_validated",
            non_claims=[
                "The overlay changes scenario configuration but does not execute the scenario.",
                "Ordinary scenario validation remains required after materialization.",
            ],
            data_markings=data_markings,
            output_path=target,
            logical_output_path=logical_target,
            writer=writer,
        )
        result = write_scenario_patch_product(
            product,
            target,
            logical_output_path=logical_target,
            writer=writer,
        )
    except ScenarioOverlayError:
        raise
    except ScenarioPatchError as exc:
        raise ScenarioOverlayError(str(exc)) from exc
    return {"status": "emitted", **result, "operation_count": len(operations)}


def _overlay_operations(overlay: Mapping[str, Any], *, rationale: str) -> list[dict[str, Any]]:
    raw = deepcopy(dict(overlay))
    unknown = sorted(set(raw) - {"ground_stations", "objects", "simulator", "outputs", "analysis"})
    if unknown:
        raise ScenarioOverlayError(f"Unsupported overlay top-level fields: {unknown}.")
    operations: list[dict[str, Any]] = []
    if "ground_stations" in raw:
        if not isinstance(raw["ground_stations"], list):
            raise ScenarioOverlayError("ground_stations overlay must be a list.")
        operations.append(_operation("ground_stations", raw["ground_stations"], rationale))
    objects = raw.get("objects", {})
    if not isinstance(objects, Mapping):
        raise ScenarioOverlayError("objects overlay must be a mapping.")
    for object_id, object_overlay_raw in sorted(objects.items()):
        if not str(object_id).strip() or not isinstance(object_overlay_raw, Mapping):
            raise ScenarioOverlayError("Each object overlay must be a named mapping.")
        object_overlay = dict(object_overlay_raw)
        unsupported = sorted(set(object_overlay) - _OBJECT_FIELDS)
        if unsupported:
            raise ScenarioOverlayError(
                f"Unsupported overlay fields for object {object_id!r}: {unsupported}."
            )
        for field, value in sorted(object_overlay.items()):
            operations.append(_operation(f"objects.{object_id}.{field}", value, rationale))
    simulator = raw.get("simulator", {})
    if not isinstance(simulator, Mapping) or set(simulator) - {"termination"}:
        raise ScenarioOverlayError("simulator overlay may contain only termination.")
    if "termination" in simulator:
        operations.append(_operation("simulator.termination", simulator["termination"], rationale))
    outputs = raw.get("outputs", {})
    if not isinstance(outputs, Mapping) or set(outputs) - {"review"}:
        raise ScenarioOverlayError("outputs overlay may contain only review.")
    if "review" in outputs:
        operations.append(_operation("outputs.review", outputs["review"], rationale))
    analysis = raw.get("analysis", {})
    if not isinstance(analysis, Mapping):
        raise ScenarioOverlayError("analysis overlay must be a mapping.")
    unsupported_analysis = sorted(set(analysis) - _ANALYSIS_FIELDS)
    if unsupported_analysis:
        raise ScenarioOverlayError(
            f"Unsupported analysis overlay fields: {unsupported_analysis}."
        )
    for name, value in sorted(analysis.items()):
        if not str(name).strip():
            raise ScenarioOverlayError("analysis overlay keys must be non-empty.")
        operations.append(_operation(f"analysis.{name}", value, rationale))
    if not operations:
        raise ScenarioOverlayError("Scenario overlay must contain at least one supported field.")
    return operations


def _operation(path: str, value: Any, rationale: str) -> dict[str, Any]:
    return {
        "op": "upsert",
        "kind": "scenario_override",
        "path": path,
        "value": deepcopy(value),
        "reason": rationale,
    }


__all__ = [
    "SCENARIO_OVERLAY_ADAPTER_ID",
    "SCENARIO_OVERLAY_ADAPTER_VERSION",
    "ScenarioOverlayError",
    "emit_scenario_overlay",
    "load_scenario_overlay",
]
