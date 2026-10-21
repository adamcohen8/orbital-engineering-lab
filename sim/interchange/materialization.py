from __future__ import annotations

import hashlib
import math
import os
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sim.scenarios import ScenarioArtifact

from .manifests import finalize_handoff_manifest, write_handoff_manifest
from .provenance import canonical_json_bytes, sha256_file
from .validation import load_interchange_document, validate_product

ONP_MATERIALIZATION_ADAPTER_ID = "oel.state_estimate_to_onp"
ONP_MATERIALIZATION_ADAPTER_VERSION = "1"
OGP_MATERIALIZATION_ADAPTER_ID = "oel.ogp_mean_elements_to_ogp"
OGP_MATERIALIZATION_ADAPTER_VERSION = "1"


def _ensure_distinct_materialization_paths(
    paths: tuple[tuple[str, Path], ...],
    error_type: type[ValueError] = ValueError,
) -> None:
    for index, (left_label, left_path) in enumerate(paths):
        for right_label, right_path in paths[index + 1 :]:
            if left_path == right_path:
                raise error_type(
                    f"Materialization targets must be distinct: {left_label} and {right_label} both resolve to {left_path}."
                )


class ONPMaterializationError(ValueError):
    """Raised when a state product cannot be safely materialized into ONP YAML."""


class OGPMaterializationError(ValueError):
    """Raised when native mean elements cannot be safely materialized into OGP YAML."""


def inspect_state_materialization(product: Mapping[str, Any], *, source_path: str | Path) -> dict[str, Any]:
    """Read-only shared promotion/compatibility decision for state consumers."""
    report = validate_product(product, source_path=source_path)
    specs = None
    blockers = [issue.to_dict() for issue in report.issues if issue.severity in {"error", "blocker"}]
    if product.get("product_kind") not in {"oel.state_estimate", "oel.completed_run_state"}:
        blockers.append(
            {"code": "compatibility.product_kind", "message": "Use the dedicated adapter for this product kind."}
        )
    if report.valid:
        blockers.extend(_onp_compatibility_errors(product))
        try:
            specs = _continued_object_specs(dict(product.get("payload", {}) or {}))
        except ValueError as exc:
            blockers.append({"code": "compatibility.resource_state", "message": str(exc)})
    return {
        "valid": report.valid,
        "promotable": report.promotable and not blockers,
        "blockers": blockers,
        "object_specs": specs,
    }



def materialize_onp(
    state_product: str | Path,
    *,
    scenario_name: str,
    scenario_path: str | Path,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    manifest_path: str | Path | None = None,
    trust_plugins: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    product_path = Path(state_product).expanduser().resolve()
    destination = Path(scenario_path).expanduser().resolve()
    manifest_target = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else destination.with_name(f"{destination.stem}.handoff_manifest.json")
    )
    _ensure_distinct_materialization_paths(
        (("source product", product_path), ("scenario", destination), ("manifest", manifest_target)),
        ONPMaterializationError,
    )
    if not str(scenario_name or "").strip():
        raise ONPMaterializationError("scenario_name must be non-empty.")
    if float(duration_s) <= 0.0 or float(dt_s) <= 0.0:
        raise ONPMaterializationError("duration_s and dt_s must be positive.")
    product = load_interchange_document(product_path)
    product_report = validate_product(product, source_path=product_path)
    created_utc = _now_utc()
    source_id = str(product.get("product_id", "") or "")
    source_hash = sha256_file(product_path)
    base_manifest = _manifest_base(
        created_utc=created_utc,
        source_id=source_id,
        source_hash=source_hash,
        product=product,
        scenario_name=scenario_name,
        destination=destination,
        output_dir=output_dir,
        duration_s=duration_s,
        dt_s=dt_s,
        overwrite=overwrite,
    )
    cadence_override = _onp_cadence_override(product, dt_s=float(dt_s))
    if cadence_override:
        base_manifest["overrides"].append(cadence_override)
    continuation = _completed_run_context(product)
    if continuation:
        base_manifest["materialization_options"]["completed_run_continuation"] = continuation
        review_hash = str(dict(continuation.get("source_run", {}) or {}).get("review_db_sha256", "") or "")
        if review_hash:
            base_manifest["source_hashes"]["completed_run_review_store"] = review_hash
    if not product_report.valid or not product_report.promotable:
        failures = [issue.to_dict() for issue in product_report.issues if issue.severity in {"error", "blocker"}]
        manifest = _failed_manifest(
            base_manifest,
            output_status="blocked",
            failures=failures,
            next_action="Resolve product validation, quality, freshness, and compatibility blockers before materialization.",
        )
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, product_report.to_dict())

    supported_kinds = {"oel.state_estimate", "oel.completed_run_state"}
    if product.get("product_kind") not in supported_kinds:
        issue = {
            "code": "compatibility.product_kind",
            "path": "$.product_kind",
            "message": (
                "Generic ONP materialization accepts only state estimates and completed-run states; "
                "use the dedicated adapter for this product kind."
            ),
        }
        manifest = _failed_manifest(
            base_manifest,
            output_status="blocked",
            failures=[issue],
            next_action=issue["message"],
        )
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, product_report.to_dict())

    compatibility_errors = _onp_compatibility_errors(product)
    if compatibility_errors:
        manifest = _failed_manifest(
            base_manifest,
            output_status="blocked",
            failures=compatibility_errors,
            next_action="Use a dedicated adapter that preserves the incompatible frame, force-model, or attitude semantics.",
        )
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, product_report.to_dict())

    scenario = build_onp_scenario(
        product,
        scenario_name=str(scenario_name).strip(),
        output_dir=output_dir,
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        manifest_path=manifest_target,
    )
    return materialize_scenario_document(
        scenario=scenario,
        destination=destination,
        manifest_target=manifest_target,
        base_manifest=base_manifest,
        product_report=product_report.to_dict(),
        output_kind="onp_scenario",
        trust_plugins=trust_plugins,
        overwrite=overwrite,
    )


def materialize_ogp(
    mean_element_product: str | Path,
    *,
    scenario_name: str,
    scenario_path: str | Path,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    manifest_path: str | Path | None = None,
    trust_plugins: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize a passive OGP scenario without encoding fitted elements as a TLE."""

    product_path = Path(mean_element_product).expanduser().resolve()
    destination = Path(scenario_path).expanduser().resolve()
    manifest_target = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else destination.with_name(f"{destination.stem}.handoff_manifest.json")
    )
    _ensure_distinct_materialization_paths(
        (("source product", product_path), ("scenario", destination), ("manifest", manifest_target)),
        OGPMaterializationError,
    )
    if not str(scenario_name or "").strip() or float(duration_s) <= 0.0 or float(dt_s) <= 0.0:
        raise OGPMaterializationError("scenario_name must be non-empty and duration_s/dt_s must be positive.")
    product = load_interchange_document(product_path)
    report = validate_product(product, source_path=product_path)
    created = _now_utc()
    source_id = str(product.get("product_id", "") or "")
    base = _manifest_base(
        created_utc=created,
        source_id=source_id,
        source_hash=sha256_file(product_path),
        product=product,
        scenario_name=scenario_name,
        destination=destination,
        output_dir=output_dir,
        duration_s=duration_s,
        dt_s=dt_s,
        overwrite=overwrite,
    )
    base["adapter"] = {
        "adapter_id": OGP_MATERIALIZATION_ADAPTER_ID,
        "adapter_version": OGP_MATERIALIZATION_ADAPTER_VERSION,
    }
    base["defaults_applied"] = {
        "propagation_method": "general",
        "general_model": "sgp4",
        "output_frame": "teme",
        "review_detail": "standard",
    }
    base["output"]["kind"] = "ogp_scenario"
    if product.get("product_kind") != "oel.ogp_mean_element_product":
        report_issue = {
            "code": "compatibility.product_kind",
            "path": "$.product_kind",
            "message": "OGP materialization requires oel.ogp_mean_element_product.",
        }
        manifest = _failed_manifest(base, output_status="blocked", failures=[report_issue], next_action=report_issue["message"])
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, report.to_dict())
    if not report.valid or not report.promotable:
        failures = [item.to_dict() for item in report.issues if item.severity in {"error", "blocker"}]
        manifest = _failed_manifest(base, output_status="blocked", failures=failures, next_action="Resolve product validation blockers before OGP materialization.")
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, report.to_dict())
    scenario = build_ogp_scenario(
        product,
        scenario_name=str(scenario_name).strip(),
        output_dir=output_dir,
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        manifest_path=manifest_target,
    )
    return materialize_scenario_document(
        scenario=scenario,
        destination=destination,
        manifest_target=manifest_target,
        base_manifest=base,
        product_report=report.to_dict(),
        output_kind="ogp_scenario",
        trust_plugins=trust_plugins,
        overwrite=overwrite,
    )


def materialize_scenario_document(
    *,
    scenario: Mapping[str, Any],
    destination: str | Path,
    manifest_target: str | Path,
    base_manifest: Mapping[str, Any],
    product_report: Mapping[str, Any],
    output_kind: str,
    trust_plugins: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Write and validate a normal OEL scenario while preserving the execution boundary."""

    destination = Path(destination).expanduser().resolve()
    manifest_target = Path(manifest_target).expanduser().resolve()
    _ensure_distinct_materialization_paths((("scenario", destination), ("manifest", manifest_target)))
    created_utc = str(base_manifest.get("created_utc", "") or _now_utc())
    try:
        artifact = ScenarioArtifact.from_dict(scenario)
    except (TypeError, ValueError) as exc:
        manifest = _failed_manifest(
            base_manifest,
            output_status="not_written",
            failures=[
                {
                    "code": "materialization.scenario_invalid",
                    "path": "$.output",
                    "message": str(exc),
                }
            ],
            next_action="Resolve the source model assumptions or use a compatible dedicated adapter.",
        )
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, product_report)
    yaml_text = artifact.to_yaml_text()
    existing_identical = destination.is_file() and destination.read_text(encoding="utf-8") == yaml_text
    if destination.exists() and not existing_identical and not overwrite:
        manifest = _failed_manifest(
            base_manifest,
            output_status="not_written",
            failures=[
                {
                    "code": "output.exists",
                    "path": str(destination),
                    "message": "Destination exists with different content; pass overwrite=True explicitly to replace it.",
                }
            ],
            next_action="Choose a new scenario path or explicitly authorize overwrite.",
        )
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, product_report)
    if not existing_identical:
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending_manifest = deepcopy(base_manifest)
        pending_manifest["output"] = {
            "kind": str(output_kind),
            "path": str(destination),
            "status": "materializing",
        }
        write_handoff_manifest(pending_manifest, manifest_target)
        temporary = _write_materialization_temp(destination, yaml_text)
        try:
            if overwrite:
                temporary.replace(destination)
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    existing_identical = _destination_matches_text(destination, yaml_text)
                    if not existing_identical:
                        manifest = _failed_manifest(
                            base_manifest,
                            output_status="not_written",
                            failures=[
                                {
                                    "code": "output.exists",
                                    "path": str(destination),
                                    "message": (
                                        "Destination appeared during materialization with different content; "
                                        "pass overwrite=True explicitly to replace it."
                                    ),
                                }
                            ],
                            next_action="Choose a new scenario path or explicitly authorize overwrite.",
                        )
                        write_handoff_manifest(manifest, manifest_target)
                        return _result("blocked", destination, manifest_target, manifest, product_report)
        finally:
            temporary.unlink(missing_ok=True)

    from sim.api import SimulationWorkspace

    workspace = SimulationWorkspace()
    safe_validation = workspace.validate_safe(destination)
    if trust_plugins and bool(safe_validation.get("ok", False)):
        ordinary_validation = workspace.validate(destination, import_plugins=True)
    else:
        ordinary_validation = {
            "ok": None,
            "status": "trust_required" if safe_validation.get("ok") else "not_run",
            "reason": "trust_not_granted" if safe_validation.get("ok") else "safe_validation_failed",
        }
    destination_matches_generated = _destination_matches_text(destination, yaml_text)
    validation_ok = bool(safe_validation.get("ok", False)) and (
        not trust_plugins or bool(ordinary_validation.get("ok", False))
    ) and destination_matches_generated
    scenario_digest = canonical_scenario_digest(artifact.to_artifact_dict())
    output_status = "validated" if validation_ok else "validation_failed"
    failures = []
    if not safe_validation.get("ok"):
        failures.extend(_validation_failures("safe_validation", safe_validation))
    if trust_plugins and not ordinary_validation.get("ok"):
        failures.extend(_validation_failures("ordinary_validation", ordinary_validation))
    if not destination_matches_generated:
        failures.append(
            {
                "code": "output.changed_during_validation",
                "path": str(destination),
                "message": "Scenario content changed during validation; materialization is not reported as successful.",
            }
        )
    manifest = deepcopy(base_manifest)
    manifest["output"] = {
        "kind": str(output_kind),
        "path": str(destination),
        "digest": scenario_digest,
        "status": output_status,
    }
    manifest["validation"] = {
        "safe_validation_result": safe_validation,
        "ordinary_validation_result": ordinary_validation,
        "validated_utc": created_utc,
    }
    manifest["failures"] = failures
    manifest["recommended_next_action"] = (
        f"Review the scenario, then execute separately with: .venv/bin/python run_simulation.py --config {destination}"
        if validation_ok
        else "Correct the generated scenario validation failures before execution."
    )
    manifest = finalize_handoff_manifest(manifest)
    write_handoff_manifest(manifest, manifest_target)
    if validation_ok and not _destination_matches_text(destination, yaml_text):
        manifest["output"]["status"] = "validation_failed"
        manifest["failures"].append(
            {
                "code": "output.changed_during_validation",
                "path": str(destination),
                "message": "Scenario content changed before materialization completed; success is withdrawn.",
            }
        )
        manifest["recommended_next_action"] = "Restore or review the scenario content before treating it as materialized."
        manifest = finalize_handoff_manifest(manifest)
        write_handoff_manifest(manifest, manifest_target)
        validation_ok = False
    return _result(
        "materialized" if validation_ok else "failed",
        destination,
        manifest_target,
        manifest,
        product_report,
    )


def _write_materialization_temp(destination: Path, yaml_text: str) -> Path:
    """Create and fully write a unique sibling file using exclusive creation."""

    for _ in range(10):
        temporary = destination.with_name(f".{destination.name}.materializing.{uuid.uuid4().hex}")
        try:
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            continue
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(yaml_text)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return temporary
    raise FileExistsError(f"Could not allocate a unique materialization file beside {destination}.")


def _destination_matches_text(destination: Path, expected_text: str) -> bool:
    try:
        return destination.is_file() and destination.read_text(encoding="utf-8") == expected_text
    except (OSError, UnicodeDecodeError):
        return False


def canonical_scenario_digest(scenario: Mapping[str, Any]) -> str:
    semantic = deepcopy(dict(scenario))
    outputs = semantic.get("outputs")
    if isinstance(outputs, dict):
        outputs.pop("output_dir", None)
    metadata = semantic.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("handoff"), dict):
        metadata["handoff"].pop("manifest_path", None)
    return hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()


def build_onp_scenario(
    product: Mapping[str, Any],
    *,
    scenario_name: str,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    manifest_path: Path,
    adapter_id: str = ONP_MATERIALIZATION_ADAPTER_ID,
    adapter_version: str = ONP_MATERIALIZATION_ADAPTER_VERSION,
    scenario_description: str | None = None,
) -> dict[str, Any]:
    payload = dict(product["payload"])
    obj = dict(payload["object"])
    state = dict(payload["state"])
    epoch = dict(state["epoch"])
    values = [float(item) for item in state["values"]]
    assumptions = dict(payload["model_assumptions"])
    force_model = deepcopy(dict(assumptions.get("orbit_force_model", {}) or {}))
    environment = deepcopy(dict(force_model.pop("environment", {}) or {}))
    environment.setdefault("ephemeris_mode", "analytic_simple")
    environment.setdefault("atmosphere_env", {})
    force_model.setdefault("model", "two_body")
    source_substep_s = force_model.get("orbit_substep_s")
    if source_substep_s is None or float(source_substep_s) > float(dt_s):
        force_model["orbit_substep_s"] = float(dt_s)
    source_hashes = {
        str(item.get("artifact_id", "")): str(item.get("sha256", ""))
        for item in list(dict(product["provenance"]).get("source_artifacts", []) or [])
        if isinstance(item, Mapping)
    }
    object_id = str(obj["object_id"])
    handoff_metadata = {
        "source_product_id": product["product_id"],
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "source_hashes": source_hashes,
        "quality_disposition": dict(product["quality"])["disposition"],
        "manifest_path": str(manifest_path),
        "execution_occurred": False,
    }
    continuation = _completed_run_context(product)
    if continuation:
        handoff_metadata["completed_run_continuation"] = continuation
    return {
        "scenario_name": scenario_name,
        "scenario_description": scenario_description
        or f"ONP passive continuation materialized from {product['product_id']}.",
        "metadata": {
            "owner": "oel-handoff",
            "handoff": handoff_metadata,
        },
        "objects": {
            object_id: {
                "enabled": True,
                "role": str(obj["role"]),
                "kind": str(obj["kind"]),
                "specs": _continued_object_specs(payload),
                "initial_state": {
                    "position_eci_km": values[:3],
                    "velocity_eci_km_s": values[3:],
                },
                "flight_software": {
                    "stack": "fsw.passive",
                    "hardware_profile": "hardware.passive.v1",
                },
            }
        },
        "simulator": {
            "initial_jd_utc": float(epoch["value"]),
            "duration_s": duration_s,
            "dt_s": dt_s,
            "dynamics": {
                "orbit": force_model,
                "attitude": {"enabled": False},
                "rocket": {"enabled": False},
            },
            "environment": environment,
            "plugin_validation": {"strict": True},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {
                "enabled": True,
                "print_summary": False,
                "save_json": True,
                "save_csv": False,
                "save_full_log": False,
            },
            "review": {"enabled": True, "detail": "standard"},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def build_ogp_scenario(
    product: Mapping[str, Any],
    *,
    scenario_name: str,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    manifest_path: Path,
) -> dict[str, Any]:
    payload = dict(product["payload"])
    obj = dict(payload["object"])
    state = dict(payload["mean_elements"])
    values = deepcopy(dict(state["values"]))
    object_id = str(obj["object_id"])
    propagation = dict(payload.get("propagation", {}) or {})
    return {
        "scenario_name": scenario_name,
        "scenario_description": f"Passive OGP continuation materialized from {product['product_id']}.",
        "metadata": {
            "owner": "oel-handoff",
            "handoff": {
                "source_product_id": product["product_id"],
                "adapter_id": OGP_MATERIALIZATION_ADAPTER_ID,
                "adapter_version": OGP_MATERIALIZATION_ADAPTER_VERSION,
                "source_epoch_jd_utc": float(dict(state["epoch"])["value"]),
                "quality_disposition": dict(product["quality"])["disposition"],
                "manifest_path": str(manifest_path),
                "execution_occurred": False,
                "source_ogp_regime": str(propagation.get("regime", "")),
                "source_ogp_propagator_name": str(propagation.get("propagator_name", "")),
            },
        },
        "objects": {
            object_id: {
                "enabled": True,
                "role": "catalog_object",
                "kind": "satellite",
                "propagation_method": "general",
                "general": {"model": "sgp4", "output_frame": "teme", "frame_transform": "native"},
                "specs": {"mass_kg": 300.0},
                "initial_state": {"ogp_mean_elements": values},
            }
        },
        "simulator": {
            "initial_jd_utc": float(dict(state["epoch"])["value"]),
            "duration_s": duration_s,
            "dt_s": dt_s,
            "dynamics": {
                "orbit": {"model": "two_body", "integrator": "rk4", "orbit_substep_s": dt_s},
                "attitude": {"enabled": False},
                "rocket": {"enabled": False},
            },
            "environment": {"ephemeris_mode": "analytic_simple", "atmosphere_env": {}},
            "plugin_validation": {"strict": True},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"enabled": True, "print_summary": False, "save_json": True, "save_csv": False, "save_full_log": False},
            "review": {"enabled": True, "detail": "standard"},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def _onp_compatibility_errors(product: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = dict(product.get("payload", {}) or {})
    state = dict(payload.get("state", {}) or {})
    assumptions = dict(payload.get("model_assumptions", {}) or {})
    attitude = dict(assumptions.get("attitude", {}) or {})
    force = dict(assumptions.get("orbit_force_model", {}) or {})
    errors: list[dict[str, str]] = []
    if state.get("frame") != "ECI":
        errors.append(
            {
                "code": "compatibility.frame",
                "path": "$.payload.state.frame",
                "message": "The first ONP materializer requires canonical ECI state input.",
            }
        )
    if str(attitude.get("source", "none") or "none") != "none":
        errors.append(
            {
                "code": "compatibility.attitude",
                "path": "$.payload.model_assumptions.attitude.source",
                "message": "Attitude-dependent OD assumptions cannot be dropped by the first ONP materializer.",
            }
        )
    if str(force.get("model", "two_body") or "two_body").lower() == "cr3bp":
        errors.append(
            {
                "code": "compatibility.force_model",
                "path": "$.payload.model_assumptions.orbit_force_model.model",
                "message": "Batch OD ECI continuation cannot be relabeled as a CR3BP rotating-frame scenario.",
            }
        )
    orbit_substep_s = force.get("orbit_substep_s")
    if orbit_substep_s is not None and (
        isinstance(orbit_substep_s, bool)
        or not isinstance(orbit_substep_s, (int, float))
        or not math.isfinite(float(orbit_substep_s))
        or float(orbit_substep_s) <= 0.0
    ):
        errors.append(
            {
                "code": "compatibility.orbit_substep",
                "path": "$.payload.model_assumptions.orbit_force_model.orbit_substep_s",
                "message": "ONP orbit_substep_s must be a positive finite number when supplied.",
            }
        )
    return errors


def _continued_object_specs(payload: Mapping[str, Any]) -> dict[str, Any]:
    specs = deepcopy(dict(payload.get("object_specs", {}) or {}))
    if any(dict(specs.get(name, {}) or {}).get("enabled", False) for name in ("thermal", "power")):
        raise ValueError("Thermal/power continuation requires a resource-state handoff; this product cannot restore it.")
    resource = dict(payload.get("resource_state", {}) or {})
    if not resource:
        return specs
    mass_kg = float(resource["mass_kg"])
    specs["mass_kg"] = mass_kg
    if resource.get("propellant_state") == "tracked":
        specs["dry_mass_kg"] = float(resource["dry_mass_kg"])
        specs["fuel_mass_kg"] = float(resource["fuel_mass_kg"])
    else:
        specs.pop("dry_mass_kg", None)
        specs.pop("fuel_mass_kg", None)
    return specs


def _onp_cadence_override(product: Mapping[str, Any], *, dt_s: float) -> dict[str, Any] | None:
    assumptions = dict(dict(product.get("payload", {}) or {}).get("model_assumptions", {}) or {})
    force = dict(assumptions.get("orbit_force_model", {}) or {})
    source_substep_s = force.get("orbit_substep_s")
    if (
        isinstance(source_substep_s, (int, float))
        and not isinstance(source_substep_s, bool)
        and math.isfinite(float(source_substep_s))
        and float(source_substep_s) > float(dt_s)
    ):
        return {
            "field": "simulator.dynamics.orbit.orbit_substep_s",
            "source_value": float(source_substep_s),
            "output_value": float(dt_s),
            "reason": "Bound the integration substep to the explicitly requested consumer dt_s.",
        }
    return None


def _completed_run_context(product: Mapping[str, Any]) -> dict[str, Any]:
    if product.get("product_kind") != "oel.completed_run_state":
        return {}
    payload = dict(product.get("payload", {}) or {})
    return {
        "source_run": deepcopy(dict(payload.get("source_run", {}) or {})),
        "selection": deepcopy(dict(payload.get("selection", {}) or {})),
    }


def _manifest_base(
    *,
    created_utc: str,
    source_id: str,
    source_hash: str,
    product: Mapping[str, Any],
    scenario_name: str,
    destination: Path,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    overwrite: bool,
) -> dict[str, Any]:
    markings = deepcopy(dict(product.get("data_markings", {}) or {}))
    return {
        "schema_id": "oel-handoff-manifest-v1",
        "schema_version": 1,
        "manifest_id": "oel.handoff_manifest:" + "0" * 64,
        "created_utc": created_utc,
        "source_product_ids": [source_id],
        "source_hashes": {source_id: source_hash},
        "adapter": {
            "adapter_id": ONP_MATERIALIZATION_ADAPTER_ID,
            "adapter_version": ONP_MATERIALIZATION_ADAPTER_VERSION,
        },
        "materialization_options": {
            "scenario_name": str(scenario_name),
            "scenario_path": str(destination),
            "output_dir": str(output_dir),
            "duration_s": float(duration_s),
            "dt_s": float(dt_s),
        },
        "defaults_applied": {
            "flight_software": "fsw.passive@hardware.passive.v1",
            "attitude_enabled": False,
            "review_detail": "standard",
            "plots_enabled": False,
            "animations_enabled": False,
        },
        "overrides": ([{"field": "scenario_path", "reason": "explicit overwrite authorization"}] if overwrite else []),
        "source_markings": markings,
        "output_markings": markings,
        "output": {"kind": "onp_scenario", "path": str(destination), "digest": "", "status": "pending"},
        "validation": {
            "safe_validation_result": {"status": "not_run"},
            "ordinary_validation_result": {"status": "not_run"},
        },
        "warnings": [],
        "failures": [],
        "recommended_next_action": "Review product validation before materialization.",
        "execution_occurred": False,
    }


def _failed_manifest(
    base: Mapping[str, Any],
    *,
    output_status: str,
    failures: list[dict[str, Any]],
    next_action: str,
) -> dict[str, Any]:
    manifest = deepcopy(dict(base))
    manifest["output"]["status"] = output_status
    manifest["failures"] = deepcopy(failures)
    manifest["recommended_next_action"] = next_action
    return finalize_handoff_manifest(manifest)


def _validation_failures(stage: str, validation: Mapping[str, Any]) -> list[dict[str, str]]:
    errors = list(validation.get("errors", []) or [])
    if not errors:
        errors = [f"{stage} did not report success"]
    return [
        {"code": f"validation.{stage}", "path": "$.validation", "message": str(error)} for error in errors
    ]


def _result(
    status: str,
    scenario_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    product_validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "scenario_path": str(scenario_path),
        "manifest_path": str(manifest_path),
        "manifest_id": manifest.get("manifest_id"),
        "execution_occurred": False,
        "product_validation": dict(product_validation),
        "safe_validation": dict(dict(manifest.get("validation", {}) or {}).get("safe_validation_result", {}) or {}),
        "ordinary_validation": dict(
            dict(manifest.get("validation", {}) or {}).get("ordinary_validation_result", {}) or {}
        ),
        "failures": deepcopy(list(manifest.get("failures", []) or [])),
        "recommended_next_action": manifest.get("recommended_next_action"),
    }


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "ONP_MATERIALIZATION_ADAPTER_ID",
    "ONP_MATERIALIZATION_ADAPTER_VERSION",
    "ONPMaterializationError",
    "OGPMaterializationError",
    "OGP_MATERIALIZATION_ADAPTER_ID",
    "OGP_MATERIALIZATION_ADAPTER_VERSION",
    "build_ogp_scenario",
    "build_onp_scenario",
    "canonical_scenario_digest",
    "materialize_onp",
    "materialize_ogp",
    "materialize_scenario_document",
]
