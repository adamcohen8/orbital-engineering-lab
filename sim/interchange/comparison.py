from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .manifests import write_json_document
from .materialization import canonical_scenario_digest
from .provenance import canonical_json_bytes, sha256_file
from .validation import load_interchange_document, validate_document, validate_product

HANDOFF_COMPARISON_SCHEMA_ID = "oel-handoff-comparison-v1"
HANDOFF_COMPARISON_SCHEMA_VERSION = 1


class HandoffComparisonError(ValueError):
    """Raised when handoff comparison inputs cannot be inspected safely."""


def compare_handoff(
    product_path: str | Path,
    scenario_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    run_output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    writer: Any | None = None,
) -> dict[str, Any]:
    """Compare producer semantics with a materialized scenario and optional run.

    This function is read-only except for the optional comparison-packet write.
    It never validates by importing plugins and never executes a scenario.
    """

    product_source = Path(product_path).expanduser().resolve()
    scenario_source = Path(scenario_path).expanduser().resolve()
    manifest_source = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else scenario_source.with_name(f"{scenario_source.stem}.handoff_manifest.json")
    )
    product = load_interchange_document(product_source)
    manifest = load_interchange_document(manifest_source)
    scenario = _load_yaml_mapping(scenario_source)
    product_report = validate_product(product, source_path=product_source)
    manifest_report = validate_document(manifest)
    checks: list[dict[str, Any]] = []

    _check(checks, "product.valid", "contract", True, product_report.valid)
    _check(checks, "product.promotable", "contract", True, product_report.promotable)
    _check(checks, "manifest.valid", "contract", True, manifest_report.valid)
    _check(checks, "manifest.execution_boundary", "execution", False, manifest.get("execution_occurred"))

    product_id = str(product.get("product_id", "") or "")
    product_sha = sha256_file(product_source)
    _check(
        checks,
        "manifest.source_product_id",
        "provenance",
        True,
        product_id in list(manifest.get("source_product_ids", []) or []),
    )
    _check(
        checks,
        "manifest.source_product_sha256",
        "provenance",
        product_sha,
        dict(manifest.get("source_hashes", {}) or {}).get(product_id),
    )
    _check(
        checks,
        "manifest.scenario_digest",
        "identity",
        str(dict(manifest.get("output", {}) or {}).get("digest", "") or ""),
        canonical_scenario_digest(scenario),
    )
    _check(
        checks,
        "manifest.output_markings",
        "markings",
        dict(product.get("data_markings", {}) or {}),
        dict(manifest.get("output_markings", {}) or {}),
    )
    _check(
        checks,
        "scenario.source_product_metadata",
        "provenance",
        True,
        _scenario_cites_product(scenario, product_id),
    )

    product_kind = str(product.get("product_kind", "") or "")
    if product_kind in {"oel.state_estimate", "oel.completed_run_state"}:
        _compare_absolute_state(checks, product, scenario, manifest=manifest)
    elif product_kind == "oel.ogp_mean_element_product":
        _compare_ogp_mean_elements(checks, product, scenario)
    elif product_kind == "oel.relative_state_estimate":
        _compare_relative_state(checks, product, scenario)
    elif product_kind == "oel.scenario_patch":
        _compare_scenario_patch(checks, product, scenario)
    elif product_kind == "oel.satellite_checkpoint":
        _compare_satellite_checkpoint(checks, product, scenario)
    elif product_kind == "oel.completed_run_snapshot":
        _compare_completed_run_snapshot(checks, product, scenario)
    else:
        _check(checks, "product.kind_supported", "contract", True, False, observed=product_kind)

    execution = _execution_comparison(
        checks,
        product=product,
        scenario=scenario,
        scenario_source=scenario_source,
        run_output_dir=run_output_dir,
    )
    failed = [item["check_id"] for item in checks if item["passed"] is not True]
    packet: dict[str, Any] = {
        "schema_id": HANDOFF_COMPARISON_SCHEMA_ID,
        "schema_version": HANDOFF_COMPARISON_SCHEMA_VERSION,
        "comparison_id": "oel.handoff_comparison:" + "0" * 64,
        "created_utc": str(manifest.get("created_utc", "") or ""),
        "status": "equivalent" if not failed else "failed",
        "source": {
            "product_id": product_id,
            "product_kind": product_kind,
            "product_sha256": product_sha,
            "path": str(product_source),
        },
        "materialization": {
            "manifest_id": str(manifest.get("manifest_id", "") or ""),
            "manifest_path": str(manifest_source),
            "adapter": deepcopy(dict(manifest.get("adapter", {}) or {})),
            "scenario_path": str(scenario_source),
            "scenario_sha256": sha256_file(scenario_source),
            "scenario_digest": canonical_scenario_digest(scenario),
            "execution_occurred": False,
        },
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "passed_count": len(checks) - len(failed),
            "failed_count": len(failed),
            "failed_check_ids": failed,
        },
        "execution_evidence": execution,
        "data_markings": deepcopy(dict(product.get("data_markings", {}) or {})),
        "non_claims": [
            "Semantic parity does not establish producer-model accuracy or operational suitability.",
            "A compared execution proves only the recorded initial consumer state, not full trajectory equivalence.",
        ],
    }
    packet["comparison_id"] = _comparison_id(packet)
    if output_path is not None:
        target = Path(output_path).expanduser().resolve()
        write_json_document(packet, target, writer=writer, overwrite=True)
    return packet


def _compare_absolute_state(
    checks: list[dict[str, Any]],
    product: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> None:
    payload = dict(product.get("payload", {}) or {})
    obj = dict(payload.get("object", {}) or {})
    state = dict(payload.get("state", {}) or {})
    object_id = str(obj.get("object_id", "") or "")
    scenario_obj = dict(dict(scenario.get("objects", {}) or {}).get(object_id, {}) or {})
    initial = dict(scenario_obj.get("initial_state", {}) or {})
    values = list(state.get("values", []) or [])
    _check(checks, "state.object_id", "identity", True, bool(scenario_obj), observed=object_id)
    _check(checks, "state.position_eci_km", "state", values[:3], initial.get("position_eci_km"))
    _check(checks, "state.velocity_eci_km_s", "state", values[3:], initial.get("velocity_eci_km_s"))
    _check(
        checks,
        "state.epoch_jd_utc",
        "epoch",
        dict(state.get("epoch", {}) or {}).get("value"),
        dict(scenario.get("simulator", {}) or {}).get("initial_jd_utc"),
    )
    expected_specs = deepcopy(dict(payload.get("object_specs", {}) or {}))
    resource = dict(payload.get("resource_state", {}) or {})
    if resource:
        expected_specs["mass_kg"] = float(resource["mass_kg"])
        if resource.get("propellant_state") == "tracked":
            expected_specs["dry_mass_kg"] = float(resource["dry_mass_kg"])
            expected_specs["fuel_mass_kg"] = float(resource["fuel_mass_kg"])
        else:
            expected_specs.pop("dry_mass_kg", None)
            expected_specs.pop("fuel_mass_kg", None)
    _check(checks, "state.object_specs", "object", expected_specs, scenario_obj.get("specs", {}))
    expected_force = deepcopy(
        dict(dict(payload.get("model_assumptions", {}) or {}).get("orbit_force_model", {}) or {})
    )
    expected_environment = dict(expected_force.pop("environment", {}) or {})
    substep_override = _manifest_override(
        manifest,
        field="simulator.dynamics.orbit.orbit_substep_s",
        source_value=expected_force.get("orbit_substep_s"),
    )
    if substep_override is not None:
        expected_force["orbit_substep_s"] = substep_override
    observed_force = dict(dict(dict(scenario.get("simulator", {}) or {}).get("dynamics", {}) or {}).get("orbit", {}) or {})
    _check(checks, "state.force_model", "model", True, _mapping_contains(observed_force, expected_force), observed=observed_force)
    if expected_environment:
        observed_environment = dict(dict(scenario.get("simulator", {}) or {}).get("environment", {}) or {})
        _check(
            checks,
            "state.environment",
            "model",
            True,
            _mapping_contains(observed_environment, expected_environment),
            observed=observed_environment,
        )
    covariance = dict(payload.get("covariance", {}) or {})
    analysis_covariance = dict(
        dict(dict(dict(scenario.get("analysis", {}) or {}).get("covariance", {}) or {}).get("objects", {}) or {}).get(object_id, {}) or {}
    )
    if analysis_covariance:
        _check(checks, "state.covariance_matrix", "covariance", covariance.get("matrix"), analysis_covariance.get("covariance"))


def _compare_ogp_mean_elements(checks: list[dict[str, Any]], product: Mapping[str, Any], scenario: Mapping[str, Any]) -> None:
    payload = dict(product.get("payload", {}) or {})
    obj = dict(payload.get("object", {}) or {})
    elements = dict(payload.get("mean_elements", {}) or {})
    object_id = str(obj.get("object_id", "") or "")
    scenario_obj = dict(dict(scenario.get("objects", {}) or {}).get(object_id, {}) or {})
    initial = dict(scenario_obj.get("initial_state", {}) or {})
    _check(checks, "ogp.object_id", "identity", True, bool(scenario_obj), observed=object_id)
    _check(checks, "ogp.mean_elements", "state", dict(elements.get("values", {}) or {}), initial.get("ogp_mean_elements"))
    _check(
        checks,
        "ogp.epoch_jd_utc",
        "epoch",
        dict(elements.get("epoch", {}) or {}).get("value"),
        dict(scenario.get("simulator", {}) or {}).get("initial_jd_utc"),
    )
    _check(checks, "ogp.propagation_method", "model", "general", scenario_obj.get("propagation_method"))
    _check(checks, "ogp.general_model", "model", "sgp4", dict(scenario_obj.get("general", {}) or {}).get("model"))
    propagation = dict(payload.get("propagation", {}) or {})
    handoff = dict(dict(scenario.get("metadata", {}) or {}).get("handoff", {}) or {})
    _check(checks, "ogp.source_regime", "provenance", propagation.get("regime"), handoff.get("source_ogp_regime"))
    _check(
        checks,
        "ogp.source_propagator_name",
        "provenance",
        propagation.get("propagator_name"),
        handoff.get("source_ogp_propagator_name"),
    )


def _compare_relative_state(checks: list[dict[str, Any]], product: Mapping[str, Any], scenario: Mapping[str, Any]) -> None:
    payload = dict(product.get("payload", {}) or {})
    chief_id = str(dict(payload.get("chief", {}) or {}).get("object_id", "") or "")
    deputy_id = str(dict(payload.get("deputy", {}) or {}).get("object_id", "") or "")
    state = dict(payload.get("relative_state", {}) or {})
    deputy = dict(dict(scenario.get("objects", {}) or {}).get(deputy_id, {}) or {})
    initial = dict(deputy.get("initial_state", {}) or {})
    _check(checks, "relative.chief_id", "identity", chief_id, initial.get("relative_to"))
    _check(checks, "relative.deputy_id", "identity", True, bool(deputy), observed=deputy_id)
    _check(checks, "relative.rectangular_ric_state", "state", state.get("values"), initial.get("relative_ric_rect"))
    _check(
        checks,
        "relative.epoch_jd_utc",
        "epoch",
        dict(state.get("epoch", {}) or {}).get("value"),
        dict(scenario.get("simulator", {}) or {}).get("initial_jd_utc"),
    )


def _compare_scenario_patch(checks: list[dict[str, Any]], product: Mapping[str, Any], scenario: Mapping[str, Any]) -> None:
    payload = dict(product.get("payload", {}) or {})
    patch = dict(payload.get("patch", {}) or {})
    selection = dict(payload.get("selection", {}) or {})
    handoff = dict(dict(scenario.get("metadata", {}) or {}).get("handoff", {}) or {})
    _check(checks, "patch.selection", "selection", selection, handoff.get("selection"))
    for index, raw in enumerate(list(patch.get("operations", []) or [])):
        operation = dict(raw or {})
        path = str(operation.get("path", "") or "")
        observed = _path_value(scenario, path)
        if operation.get("op") == "append":
            passed = isinstance(observed, list) and operation.get("value") in observed
            _check(checks, f"patch.operation.{index}", "patch", True, passed, observed=observed)
        elif patch.get("patch_type") == "scenario_capability_overlay":
            _check(
                checks,
                f"patch.operation.{index}",
                "patch",
                True,
                _overlay_value_matches(observed, operation.get("value")),
                observed=observed,
            )
        else:
            _check(checks, f"patch.operation.{index}", "patch", operation.get("value"), observed)


def _compare_satellite_checkpoint(
    checks: list[dict[str, Any]], product: Mapping[str, Any], scenario: Mapping[str, Any]
) -> None:
    payload = dict(product.get("payload", {}) or {})
    obj = dict(payload.get("object", {}) or {})
    object_id = str(obj.get("object_id", "") or "")
    scenario_obj = dict(dict(scenario.get("objects", {}) or {}).get(object_id, {}) or {})
    initial = dict(scenario_obj.get("initial_state", {}) or {})
    state = dict(payload.get("state", {}) or {})
    values = list(state.get("values", []) or [])
    attitude = dict(payload.get("attitude", {}) or {})
    _check(checks, "satellite.object_id", "identity", True, bool(scenario_obj), observed=object_id)
    _check(checks, "satellite.position_eci_km", "state", values[:3], initial.get("position_eci_km"))
    _check(checks, "satellite.velocity_eci_km_s", "state", values[3:], initial.get("velocity_eci_km_s"))
    _check(
        checks,
        "satellite.attitude_quat_bn",
        "state",
        attitude.get("quaternion_bn"),
        initial.get("attitude_quat_bn"),
    )
    _check(
        checks,
        "satellite.angular_rate_body_rad_s",
        "state",
        attitude.get("angular_rate_body_rad_s"),
        initial.get("angular_rate_body_rad_s"),
    )
    expected_fsw = deepcopy(dict(payload.get("flight_software", {}) or {}))
    expected_fsw["checkpoint"] = deepcopy(dict(payload.get("checkpoint", {}) or {}))
    _check(checks, "satellite.flight_software", "checkpoint", expected_fsw, scenario_obj.get("flight_software"))
    _check(
        checks,
        "satellite.mass_kg",
        "state",
        dict(payload.get("resource_state", {}) or {}).get("mass_kg"),
        dict(scenario_obj.get("specs", {}) or {}).get("mass_kg"),
    )


def _compare_completed_run_snapshot(
    checks: list[dict[str, Any]], product: Mapping[str, Any], scenario: Mapping[str, Any]
) -> None:
    payload = dict(product.get("payload", {}) or {})
    states = list(payload.get("states", []) or [])
    scenario_objects = dict(scenario.get("objects", {}) or {})
    for item_raw in states:
        item = dict(item_raw)
        obj = dict(item.get("object", {}) or {})
        object_id = str(obj.get("object_id", "") or "")
        scenario_obj = dict(scenario_objects.get(object_id, {}) or {})
        initial = dict(scenario_obj.get("initial_state", {}) or {})
        values = list(dict(item.get("state", {}) or {}).get("values", []) or [])
        _check(checks, f"snapshot.{object_id}.object_id", "identity", True, bool(scenario_obj))
        _check(
            checks,
            f"snapshot.{object_id}.position_eci_km",
            "state",
            values[:3],
            initial.get("position_eci_km"),
        )
        _check(
            checks,
            f"snapshot.{object_id}.velocity_eci_km_s",
            "state",
            values[3:],
            initial.get("velocity_eci_km_s"),
        )
        resource_state = dict(item.get("resource_state", {}) or {})
        object_specs = dict(item.get("object_specs", {}) or {})
        scenario_specs = dict(scenario_obj.get("specs", {}) or {})
        if resource_state.get("mass_kg") is not None:
            _check(
                checks,
                f"snapshot.{object_id}.mass_kg",
                "state",
                resource_state.get("mass_kg"),
                scenario_specs.get("mass_kg"),
            )
        _check(
            checks,
            f"snapshot.{object_id}.object_specs",
            "model",
            True,
            _mapping_contains(scenario_specs, object_specs),
        )
    if states:
        epoch = dict(dict(states[0]).get("state", {}) or {}).get("epoch", {})
        _check(
            checks,
            "snapshot.epoch_jd_utc",
            "time",
            dict(epoch or {}).get("value"),
            dict(scenario.get("simulator", {}) or {}).get("initial_jd_utc"),
        )


def _overlay_value_matches(observed: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(observed, Mapping) and _mapping_contains(observed, expected)
    if isinstance(expected, list):
        if not expected:
            return observed in (None, [])
        if not isinstance(observed, list) or len(observed) != len(expected):
            return False
        return all(_overlay_value_matches(actual, wanted) for actual, wanted in zip(observed, expected, strict=True))
    return observed == expected


def _execution_comparison(
    checks: list[dict[str, Any]],
    *,
    product: Mapping[str, Any],
    scenario: Mapping[str, Any],
    scenario_source: Path,
    run_output_dir: str | Path | None,
) -> dict[str, Any]:
    if run_output_dir is None:
        return {"status": "not_supplied", "execution_occurred": False}
    root = Path(run_output_dir).expanduser().resolve()
    db = root / "review" / "run.sqlite"
    if not db.is_file():
        raise HandoffComparisonError(f"Run output has no review store: {db}")
    product_kind = str(product.get("product_kind", "") or "")
    payload = dict(product.get("payload", {}) or {})
    compared_row: dict[str, Any] | None = None
    with sqlite3.connect(db) as connection:
        metadata_rows = connection.execute(
            "SELECT config_sha256, config_json FROM run_metadata LIMIT 2"
        ).fetchall()
        if len(metadata_rows) != 1:
            raise HandoffComparisonError("Run review store must contain exactly one run_metadata row.")
        config_sha256, config_json = metadata_rows[0]
        config_text = str(config_json or "")
        recorded_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
        try:
            recorded_config = json.loads(config_text)
        except json.JSONDecodeError as exc:
            raise HandoffComparisonError("Run review store config_json is invalid.") from exc
        if not isinstance(recorded_config, Mapping):
            raise HandoffComparisonError("Run review store config_json must contain an object.")
        _check(checks, "execution.config_sha256_integrity", "execution", str(config_sha256), recorded_hash)
        _check(
            checks,
            "execution.config_sha256",
            "execution",
            _normalized_config_sha256(scenario, source_path=scenario_source),
            str(config_sha256),
        )
        if product_kind in {"oel.state_estimate", "oel.completed_run_state", "oel.satellite_checkpoint"}:
            object_id = str(dict(payload.get("object", {}) or {}).get("object_id", "") or "")
            row = connection.execute(
                "SELECT pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
                "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s, mass_kg "
                "FROM object_state WHERE object_id = ? ORDER BY sample_index LIMIT 1",
                (object_id,),
            ).fetchone()
            observed = list(row[:6]) if row is not None else None
            _check(checks, "execution.initial_absolute_state", "execution", dict(payload.get("state", {}) or {}).get("values"), observed)
            if row is not None and payload.get("resource_state"):
                _check(
                    checks,
                    "execution.initial_mass_kg",
                    "execution",
                    dict(payload.get("resource_state", {}) or {}).get("mass_kg"),
                    row[6],
                )
            compared_row = {"object_id": object_id, "values": observed, "mass_kg": None if row is None else row[6]}
        elif product_kind == "oel.relative_state_estimate":
            chief_id = str(dict(payload.get("chief", {}) or {}).get("object_id", "") or "")
            deputy_id = str(dict(payload.get("deputy", {}) or {}).get("object_id", "") or "")
            row = connection.execute(
                "SELECT r_radial_km, i_intrack_km, c_crosstrack_km, "
                "vr_radial_km_s, vi_intrack_km_s, vc_crosstrack_km_s "
                "FROM relative_state WHERE chief_id = ? AND deputy_id = ? ORDER BY sample_index LIMIT 1",
                (chief_id, deputy_id),
            ).fetchone()
            observed = list(row) if row is not None else None
            _check(checks, "execution.initial_relative_state", "execution", dict(payload.get("relative_state", {}) or {}).get("values"), observed)
            compared_row = {"chief_id": chief_id, "deputy_id": deputy_id, "values": observed}
        elif product_kind == "oel.completed_run_snapshot":
            rows = []
            for item_raw in list(payload.get("states", []) or []):
                item = dict(item_raw)
                object_id = str(dict(item.get("object", {}) or {}).get("object_id", "") or "")
                row = connection.execute(
                    "SELECT pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
                    "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s, mass_kg "
                    "FROM object_state WHERE object_id = ? ORDER BY sample_index LIMIT 1",
                    (object_id,),
                ).fetchone()
                observed = None if row is None else list(row[:6])
                expected = list(dict(item.get("state", {}) or {}).get("values", []) or [])
                _check(checks, f"execution.snapshot.{object_id}.initial_state", "execution", expected, observed)
                expected_mass = dict(item.get("resource_state", {}) or {}).get("mass_kg")
                if expected_mass is not None:
                    _check(
                        checks,
                        f"execution.snapshot.{object_id}.initial_mass_kg",
                        "execution",
                        expected_mass,
                        None if row is None else row[6],
                    )
                rows.append({"object_id": object_id, "values": observed, "mass_kg": None if row is None else row[6]})
            compared_row = {"states": rows} if rows else None
    return {
        "status": "compared" if compared_row is not None else "not_applicable",
        "execution_occurred": True,
        "run_output_dir": str(root),
        "review_db_sha256": sha256_file(db),
        "compared_row": compared_row,
    }


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    category: str,
    expected: Any,
    actual: Any,
    *,
    observed: Any | None = None,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "category": category,
            "passed": actual == expected,
            "expected": deepcopy(expected),
            "observed": deepcopy(actual if observed is None else observed),
        }
    )


def _normalized_config_sha256(
    scenario: Mapping[str, Any], *, source_path: str | Path | None = None
) -> str:
    from sim import SimulationConfig

    normalized = SimulationConfig.from_dict(
        deepcopy(dict(scenario)),
        source_path=source_path,
    ).to_dict()
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _scenario_cites_product(scenario: Mapping[str, Any], product_id: str) -> bool:
    handoff = dict(dict(scenario.get("metadata", {}) or {}).get("handoff", {}) or {})
    return handoff.get("source_product_id") == product_id or product_id in list(handoff.get("source_product_ids", []) or [])


def _mapping_contains(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if key not in observed:
            if isinstance(value, Mapping) and not value:
                continue
            return False
        if isinstance(value, Mapping):
            if not isinstance(observed[key], Mapping) or not _mapping_contains(observed[key], value):
                return False
        elif observed[key] != value:
            return False
    return True


def _manifest_override(
    manifest: Mapping[str, Any],
    *,
    field: str,
    source_value: Any,
) -> Any | None:
    matches = [
        dict(item or {})
        for item in list(manifest.get("overrides", []) or [])
        if isinstance(item, Mapping)
        and item.get("field") == field
        and item.get("source_value") == source_value
        and "output_value" in item
    ]
    if len(matches) != 1:
        return None
    return deepcopy(matches[0]["output_value"])


def _path_value(root: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = root
    for token in [item for item in dotted_path.split(".") if item]:
        if not isinstance(current, Mapping) or token not in current:
            return None
        current = current[token]
    return current


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HandoffComparisonError(f"Materialized scenario does not exist: {path}") from exc
    if not isinstance(value, dict):
        raise HandoffComparisonError("Materialized scenario YAML must contain a mapping.")
    return value


def _comparison_id(packet: Mapping[str, Any]) -> str:
    semantic = {
        "source_product_id": dict(packet.get("source", {}) or {}).get("product_id"),
        "source_product_sha256": dict(packet.get("source", {}) or {}).get("product_sha256"),
        "manifest_id": dict(packet.get("materialization", {}) or {}).get("manifest_id"),
        "scenario_digest": dict(packet.get("materialization", {}) or {}).get("scenario_digest"),
        "checks": list(packet.get("checks", []) or []),
        "execution_review_db_sha256": dict(packet.get("execution_evidence", {}) or {}).get("review_db_sha256"),
        "data_markings": dict(packet.get("data_markings", {}) or {}),
    }
    digest = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
    return f"oel.handoff_comparison:{digest}"


__all__ = [
    "HANDOFF_COMPARISON_SCHEMA_ID",
    "HANDOFF_COMPARISON_SCHEMA_VERSION",
    "HandoffComparisonError",
    "compare_handoff",
]
