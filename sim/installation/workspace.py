"""User-owned OEL workspace manifests, audits, and explicit migrations."""

from __future__ import annotations

import difflib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from sim.project_version import source_project_version
from sim.schema_versions import LEGACY_SCENARIO_SCHEMA_VERSION, SCENARIO_SCHEMA_VERSION

from .contracts import (
    COMPATIBILITY_REPORT_SCHEMA,
    COMPATIBILITY_STATUSES,
    MIGRATION_PLAN_SCHEMA,
    TEMPLATE_MANIFEST_SCHEMA,
    WORKSPACE_SCHEMA,
    ContractError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_version,
    version_satisfies,
)
from .state import StateLock, atomic_write_json, atomic_write_text, read_state

WORKSPACE_FILENAME = "oel-workspace.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_engine_requirement(version: str) -> str:
    parts = [int(item) for item in version.split("-", 1)[0].split(".")]
    if len(parts) < 2:
        return f"=={version}"
    return f">={parts[0]}.{parts[1]},<{parts[0]}.{parts[1] + 3}"


def _inside(path: Path, root: Path, *, label: str, must_exist: bool = True) -> Path:
    lexical = path.expanduser()
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        relative = None
    if relative is not None:
        current = root
        for part in relative.parts:
            current = current / part
            # ``Path.exists()`` is deliberately not part of this guard: a
            # dangling link is still a routing decision and must be rejected
            # before resolution.
            if current.is_symlink():
                if current == lexical:
                    raise ContractError(f"{label} may not be a symbolic link: {path}")
                raise ContractError(f"{label} may not contain symbolic links: {path}")
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{label} must remain inside the workspace: {path}") from exc
    if must_exist and not resolved.exists():
        raise ContractError(f"{label} does not exist: {resolved}")
    return resolved


def workspace_manifest_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.name == WORKSPACE_FILENAME else path / WORKSPACE_FILENAME


def load_workspace(value: str | Path) -> dict[str, Any]:
    manifest_path = workspace_manifest_path(value).expanduser()
    if manifest_path.is_symlink():
        raise ContractError(f"Workspace manifest may not be a symbolic link: {manifest_path}")
    manifest_path = manifest_path.resolve()
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContractError(f"Workspace manifest YAML is invalid: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractError("Workspace manifest must contain a mapping/object.")
    return validate_workspace_manifest(raw, root=manifest_path.parent)


def validate_workspace_manifest(value: Mapping[str, Any], *, root: str | Path) -> dict[str, Any]:
    workspace_root = Path(root).expanduser().resolve()
    allowed = {"schema_version", "workspace_id", "created_with", "engine", "contracts", "paths", "dependencies", "policy"}
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ContractError(f"Workspace manifest has unsupported field(s): {', '.join(unknown)}.")
    required = {"schema_version", "workspace_id", "created_with", "engine", "contracts", "paths", "dependencies", "policy"}
    missing = sorted(required - set(value))
    if missing:
        raise ContractError(f"Workspace manifest is missing field(s): {', '.join(missing)}.")
    if value["schema_version"] != WORKSPACE_SCHEMA:
        raise ContractError(f"Workspace schema_version must be {WORKSPACE_SCHEMA!r}.")
    workspace_id = str(value["workspace_id"] or "").strip()
    if not workspace_id or not workspace_id.replace("_", "").replace("-", "").isalnum():
        raise ContractError("workspace_id must contain letters, digits, underscores, or hyphens.")
    engine = value["engine"]
    contracts = value["contracts"]
    paths = value["paths"]
    dependencies = value["dependencies"]
    policy = value["policy"]
    if not all(isinstance(item, Mapping) for item in (engine, contracts, paths, dependencies, policy)):
        raise ContractError("Workspace engine, contracts, paths, dependencies, and policy must be objects.")
    locked_version = validate_version(engine.get("locked_version"), label="engine.locked_version")
    created_with = validate_version(value["created_with"], label="created_with")
    requirement = str(engine.get("requirement", f"=={locked_version}") or "").strip()
    if not version_satisfies(locked_version, requirement):
        raise ContractError(f"Locked engine {locked_version} does not satisfy engine.requirement {requirement!r}.")
    resolved_paths: dict[str, str] = {}
    for key in ("configs", "flight_software", "tests", "outputs"):
        text = str(paths.get(key, "") or "").strip()
        if not text:
            raise ContractError(f"Workspace paths.{key} must be non-empty.")
        resolved = _inside(workspace_root / text, workspace_root, label=f"paths.{key}", must_exist=False)
        resolved_paths[key] = os.path.relpath(resolved, workspace_root)
    lock_text = str(dependencies.get("lock", "") or "").strip()
    if lock_text:
        _inside(workspace_root / lock_text, workspace_root, label="dependencies.lock", must_exist=False)
    normalized = {
        "schema_version": WORKSPACE_SCHEMA,
        "workspace_id": workspace_id,
        "created_with": created_with,
        "engine": {
            "channel": str(engine.get("channel", "stable") or "stable"),
            "requirement": requirement,
            "locked_version": locked_version,
        },
        "contracts": {
            "scenario": str(contracts.get("scenario", SCENARIO_SCHEMA_VERSION)),
            "fsw": str(contracts.get("fsw", "oel.fsw.boundary.v2")),
            "candidate": str(contracts.get("candidate", "oel.fswdk.candidate.v1")),
        },
        "paths": resolved_paths,
        "dependencies": {"lock": lock_text},
        "policy": {
            "network": str(policy.get("network", "explicit_only")),
            "trust_user_code": bool(policy.get("trust_user_code", False)),
        },
        "root": str(workspace_root),
        "manifest_path": str(workspace_root / WORKSPACE_FILENAME),
    }
    normalized["manifest_sha256"] = sha256_bytes(canonical_json_bytes({key: item for key, item in normalized.items() if key not in {"root", "manifest_path"}}))
    return normalized


def init_workspace(
    destination: str | Path,
    *,
    workspace_id: str | None = None,
    engine_version: str | None = None,
    engine_requirement: str | None = None,
    quickstart_config: str | Path | None = None,
    example_configs: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    root = Path(destination).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to initialize a non-empty workspace: {root}")
    root.mkdir(parents=True, exist_ok=True)
    version = engine_version or source_project_version() or "0.0.0"
    validate_version(version, label="engine version")
    identifier = workspace_id or root.name.lower().replace(" ", "_").replace("-", "_") or "oel_workspace"
    for relative in ("configs", "fsw", "tests", "outputs", ".oel/compatibility", ".oel/migrations", ".oel/receipts"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": WORKSPACE_SCHEMA,
        "workspace_id": identifier,
        "created_with": version,
        "engine": {
            "channel": "stable",
            "requirement": engine_requirement or _default_engine_requirement(version),
            "locked_version": version,
        },
        "contracts": {
            "scenario": SCENARIO_SCHEMA_VERSION,
            "fsw": "oel.fsw.boundary.v2",
            "candidate": "oel.fswdk.candidate.v1",
        },
        "paths": {"configs": "configs", "flight_software": "fsw", "tests": "tests", "outputs": "outputs"},
        "dependencies": {"lock": "requirements.lock"},
        "policy": {"network": "explicit_only", "trust_user_code": False},
    }
    atomic_write_text(root / WORKSPACE_FILENAME, yaml.safe_dump(manifest, sort_keys=False))
    atomic_write_text(root / "requirements.lock", "# Workspace-specific dependencies. Keep hashes when adding packages.\n")
    from .resources import agent_bootstrap_text

    atomic_write_text(root / "AGENTS.md", agent_bootstrap_text())
    generated: list[dict[str, Any]] = [
        {"path": "AGENTS.md", "sha256": sha256_file(root / "AGENTS.md"), "user_editable": True}
    ]
    if quickstart_config is not None:
        source = Path(quickstart_config).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Quickstart config was not found: {source}")
        target = root / "configs" / "quickstart_5min.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ContractError("Quickstart config must contain a mapping/object.")
        raw["schema_version"] = SCENARIO_SCHEMA_VERSION
        raw.setdefault("outputs", {})["output_dir"] = "outputs/quickstart_5min"
        atomic_write_text(target, yaml.safe_dump(raw, sort_keys=False))
        generated.append({"path": target.relative_to(root).as_posix(), "sha256": sha256_file(target), "user_editable": True})
    for example in example_configs:
        source = Path(example).expanduser().resolve()
        target = root / "configs" / source.name
        if target.exists():
            raise FileExistsError(f"Workspace template config names must be unique: {source.name}")
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ContractError("Example config must contain a mapping/object.")
        raw["schema_version"] = SCENARIO_SCHEMA_VERSION
        atomic_write_text(target, yaml.safe_dump(raw, sort_keys=False))
        generated.append({"path": target.relative_to(root).as_posix(), "sha256": sha256_file(target), "user_editable": True})
    template = {
        "schema_version": TEMPLATE_MANIFEST_SCHEMA,
        "template_id": "oel.workspace.default.v1",
        "template_version": "1",
        "oel_version": version,
        "generated_at": utc_now(),
        "parameters": {"workspace_id": identifier},
        "files": generated,
    }
    atomic_write_json(root / ".oel" / "template-manifest.json", template)
    return {"schema_version": "oel.workspace-init-receipt.v1", "status": "ready", "workspace": load_workspace(root)}


def register_workspace(
    value: str | Path,
    *,
    registry_path: str | Path,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    registry_target = Path(registry_path)
    transaction_lock = Path(lock_path) if lock_path is not None else registry_target.parent / "update.lock"
    with StateLock(transaction_lock, operation="register-workspace"):
        return _register_workspace_unlocked(value, registry_path=registry_target)


def _register_workspace_unlocked(value: str | Path, *, registry_path: str | Path) -> dict[str, Any]:
    workspace = load_workspace(value)
    registry = read_state(registry_path, default={"schema_version": "oel.workspace-registry.v1", "workspaces": {}})
    items = dict(registry.get("workspaces", {}) or {})
    items[workspace["workspace_id"]] = {
        "path": workspace["root"],
        "manifest_sha256": workspace["manifest_sha256"],
        "locked_version": workspace["engine"]["locked_version"],
        "registered_at": utc_now(),
    }
    registry["schema_version"] = "oel.workspace-registry.v1"
    registry["workspaces"] = items
    atomic_write_json(registry_path, registry)
    return {"status": "ready", "workspace_id": workspace["workspace_id"], "path": workspace["root"]}


def _scenario_files(workspace: Mapping[str, Any]) -> list[Path]:
    root = Path(str(workspace["root"]))
    config_root = _inside(root / workspace["paths"]["configs"], root, label="configs")
    return sorted({*config_root.rglob("*.yaml"), *config_root.rglob("*.yml")})


def audit_workspace(
    value: str | Path,
    *,
    target_version: str,
    release_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = load_workspace(value)
    issues: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    target = validate_version(target_version, label="target version")
    requirement = workspace["engine"]["requirement"]
    if not version_satisfies(target, requirement):
        issues.append(
            {
                "code": "engine_requirement",
                "severity": "blocked",
                "message": f"Target OEL {target} does not satisfy workspace requirement {requirement!r}.",
            }
        )
    supported_contracts = dict((release_manifest or {}).get("contracts", {}) or {})
    target_scenario_contract = str(supported_contracts.get("scenario", SCENARIO_SCHEMA_VERSION))
    if target_scenario_contract != SCENARIO_SCHEMA_VERSION:
        issues.append(
            {
                "code": "target_scenario_contract",
                "severity": "blocked",
                "message": f"Target release advertises unsupported scenario contract {target_scenario_contract!r}.",
            }
        )
    for path in _scenario_files(workspace):
        relative = path.relative_to(Path(workspace["root"])).as_posix()
        if path.is_symlink():
            issues.append(
                {
                    "code": "config_symlink",
                    "severity": "invalid",
                    "path": relative,
                    "message": "Scenario config may not be a symbolic link.",
                }
            )
            configs.append({"path": relative, "status": "invalid", "error": "symbolic link rejected"})
            continue
        row: dict[str, Any] = {"path": relative, "sha256": sha256_file(path), "status": "compatible"}
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ContractError("YAML root must be a mapping/object.")
            schema = str(raw.get("schema_version", LEGACY_SCENARIO_SCHEMA_VERSION))
            row["schema_version"] = schema
            if schema == LEGACY_SCENARIO_SCHEMA_VERSION:
                row["status"] = "migration_available"
                issues.append(
                    {
                        "code": "legacy_scenario_schema",
                        "severity": "migration",
                        "path": relative,
                        "message": f"Add schema_version: {SCENARIO_SCHEMA_VERSION}.",
                    }
                )
            elif schema != SCENARIO_SCHEMA_VERSION:
                row["status"] = "blocked"
                issues.append(
                    {
                        "code": "unsupported_scenario_schema",
                        "severity": "blocked",
                        "path": relative,
                        "message": f"Unsupported scenario schema {schema!r}.",
                    }
                )
            else:
                from sim.config import load_simulation_yaml, validate_scenario_plugins
                from sim.security import ConfigPathPolicy

                policy = ConfigPathPolicy.default(config_path=path, workspace_root=workspace["root"])
                config = load_simulation_yaml(path, path_policy=policy)
                plugin_issues = list(validate_scenario_plugins(config, import_plugins=False))
                if plugin_issues:
                    row["status"] = "manual_review"
                    issues.extend(
                        {
                            "code": "plugin_pointer",
                            "severity": "manual_review",
                            "path": relative,
                            "message": str(item),
                        }
                        for item in plugin_issues
                    )
            if sha256_file(path) != row["sha256"]:
                row["status"] = "cancelled"
                issues.append(
                    {
                        "code": "config_changed_during_audit",
                        "severity": "cancelled",
                        "path": relative,
                        "message": "Scenario config changed during compatibility audit; rerun the audit.",
                    }
                )
        except Exception as exc:
            row["status"] = "invalid"
            row["error"] = str(exc)
            issues.append({"code": "config_invalid", "severity": "invalid", "path": relative, "message": str(exc)})
        configs.append(row)
    candidate_rows = _inspect_candidate_manifests(workspace)
    for row in candidate_rows:
        if row["status"] != "compatible":
            issues.append(
                {
                    "code": "candidate_contract",
                    "severity": row["status"],
                    "path": row["path"],
                    "message": row["message"],
                }
            )
    lock_text = str(workspace["dependencies"].get("lock", "") or "")
    dependency = {"path": lock_text, "status": "not_declared"}
    if lock_text:
        lock_path = Path(workspace["root"]) / lock_text
        lock_lines = [
            line.strip()
            for line in lock_path.read_text(encoding="utf-8").splitlines()
            if lock_path.is_file() and line.strip() and not line.lstrip().startswith("#")
        ] if lock_path.is_file() else []
        unhashed = [line for line in lock_lines if "--hash=" not in line]
        dependency = {
            "path": lock_text,
            "status": "compatible" if lock_path.is_file() else "blocked",
            "sha256": sha256_file(lock_path) if lock_path.is_file() else None,
            "entries": len(lock_lines),
            "unhashed_entries": len(unhashed),
        }
        if not lock_path.is_file():
            issues.append(
                {
                    "code": "dependency_lock_missing",
                    "severity": "blocked",
                    "path": lock_text,
                    "message": "Workspace dependency lock is missing.",
                }
            )
        elif unhashed:
            dependency["status"] = "manual_review"
            issues.append(
                {
                    "code": "dependency_lock_unhashed",
                    "severity": "manual_review",
                    "path": lock_text,
                    "message": "Workspace dependency entries must be reviewed because one or more are not hash-locked.",
                }
            )
    evidence = _inspect_run_evidence(workspace, target_version=target)
    current_workspace = load_workspace(workspace["root"])
    if current_workspace["manifest_sha256"] != workspace["manifest_sha256"]:
        issues.append(
            {
                "code": "workspace_changed_during_audit",
                "severity": "cancelled",
                "message": "Workspace manifest changed during compatibility audit; rerun the audit.",
            }
        )
    severities = {str(item["severity"]) for item in issues}
    if "cancelled" in severities:
        status = "cancelled"
    elif "invalid" in severities:
        status = "invalid"
    elif "blocked" in severities:
        status = "blocked"
    elif "manual_review" in severities:
        status = "manual_review"
    elif "migration" in severities:
        status = "migration_available"
    elif issues:
        status = "compatible_with_warnings"
    else:
        status = "compatible"
    assert status in COMPATIBILITY_STATUSES
    report = {
        "schema_version": COMPATIBILITY_REPORT_SCHEMA,
        "status": status,
        "generated_at": utc_now(),
        "workspace": {
            "id": workspace["workspace_id"],
            "root": workspace["root"],
            "manifest_sha256": workspace["manifest_sha256"],
            "current_version": workspace["engine"]["locked_version"],
        },
        "target": {
            "version": target,
            "release_manifest_sha256": (
                sha256_bytes(canonical_json_bytes(release_manifest)) if release_manifest is not None else None
            ),
        },
        "configs": configs,
        "candidates": candidate_rows,
        "dependencies": dependency,
        "evidence": evidence,
        "issues": issues,
        "effects": {
            "user_source_modified": False,
            "workspace_metadata_written": True,
            "code_executed": False,
            "network_used": False,
        },
    }
    output = Path(workspace["root"]) / ".oel" / "compatibility" / f"{target}.json"
    atomic_write_json(output, report)
    report["report_path"] = str(output)
    return report


def _inspect_run_evidence(workspace: Mapping[str, Any], *, target_version: str) -> dict[str, Any]:
    root = Path(str(workspace["root"]))
    outputs = _inside(root / workspace["paths"]["outputs"], root, label="outputs", must_exist=False)
    rows: list[dict[str, Any]] = []
    if outputs.is_dir():
        for path in sorted(outputs.rglob("master_run_summary.json")):
            if path.is_symlink():
                rows.append({"path": path.relative_to(root).as_posix(), "status": "invalid"})
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                provenance = dict(payload.get("installation_provenance", {}) or {}) if isinstance(payload, dict) else {}
                engine = str(provenance.get("engine_version", "") or "unknown")
                rows.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                        "engine_version": engine,
                        "status": "current" if engine == target_version else "prior_or_unknown",
                    }
                )
            except (OSError, ValueError):
                rows.append({"path": path.relative_to(root).as_posix(), "status": "invalid"})
    return {
        "status": "current" if rows and all(row["status"] == "current" for row in rows) else ("not_present" if not rows else "stale"),
        "runs": rows,
        "non_claim": "Prior evidence is retained and is not requalified by a compatibility audit.",
    }


def plan_template_sync(
    value: str | Path,
    *,
    target_template_manifest: str | Path,
    template_root: str | Path,
) -> dict[str, Any]:
    """Classify template changes without overwriting user-owned workspace files."""

    workspace = load_workspace(value)
    root = Path(workspace["root"])
    origin_path = root / ".oel" / "template-manifest.json"
    origin = json.loads(origin_path.read_text(encoding="utf-8"))
    target = json.loads(Path(target_template_manifest).expanduser().read_text(encoding="utf-8"))
    if origin.get("schema_version") != TEMPLATE_MANIFEST_SCHEMA or target.get("schema_version") != TEMPLATE_MANIFEST_SCHEMA:
        raise ContractError("Template sync requires oel.template-manifest.v1 manifests.")
    new_root = Path(template_root).expanduser().resolve()
    old_files = {str(item["path"]): item for item in origin.get("files", []) if isinstance(item, dict)}
    new_files = {str(item["path"]): item for item in target.get("files", []) if isinstance(item, dict)}
    changes: list[dict[str, Any]] = []
    for relative in sorted(set(old_files) | set(new_files)):
        current = _inside(root / relative, root, label="template file", must_exist=False)
        old_digest = str(old_files.get(relative, {}).get("sha256", ""))
        current_digest = sha256_file(current) if current.is_file() else None
        new_digest = str(new_files.get(relative, {}).get("sha256", "")) or None
        if relative not in new_files:
            classification = "upstream_removed" if current_digest == old_digest else "conflict"
        elif relative not in old_files:
            classification = "new" if not current.exists() else "conflict"
        elif current_digest != old_digest and new_digest != old_digest:
            classification = "conflict"
        elif current_digest != old_digest:
            classification = "user_modified"
        elif new_digest != old_digest:
            classification = "upstream_changed"
        else:
            classification = "unchanged"
        proposed = new_root / relative
        if new_digest and (not proposed.is_file() or sha256_file(proposed) != new_digest):
            raise ContractError(f"Target template content does not match its manifest: {relative}")
        changes.append(
            {
                "path": relative,
                "classification": classification,
                "original_sha256": old_digest or None,
                "current_sha256": current_digest,
                "target_sha256": new_digest,
                "user_editable": bool(new_files.get(relative, old_files.get(relative, {})).get("user_editable", True)),
            }
        )
    return {
        "schema_version": "oel.template-sync-plan.v1",
        "status": "manual_review" if any(item["classification"] == "conflict" for item in changes) else "ready",
        "workspace": {"id": workspace["workspace_id"], "manifest_sha256": workspace["manifest_sha256"]},
        "origin_template": origin.get("template_id"),
        "target_template": target.get("template_id"),
        "changes": changes,
        "effects": {"user_source_modified": False, "code_executed": False, "network_used": False},
    }


def _inspect_candidate_manifests(workspace: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(str(workspace["root"]))
    fsw_root = _inside(root / workspace["paths"]["flight_software"], root, label="flight_software")
    rows: list[dict[str, Any]] = []
    candidates = sorted({*fsw_root.rglob("candidate.yaml"), *fsw_root.rglob("candidate.yml"), *fsw_root.rglob("candidate.json")})
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append({"path": relative, "status": "invalid", "message": "Candidate manifest symlink rejected."})
            continue
        try:
            initial_sha256 = sha256_file(path)
            raw = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ContractError("Candidate manifest must contain an object.")
            schema = str(raw.get("schema_version", ""))
            interfaces = raw.get("interfaces", {}) if isinstance(raw.get("interfaces"), dict) else {}
            onboard = str(interfaces.get("onboard_contract", ""))
            compatible = schema == workspace["contracts"]["candidate"] and onboard == workspace["contracts"]["fsw"]
            rows.append(
                {
                    "path": relative,
                    "sha256": initial_sha256,
                    "schema_version": schema,
                    "fsw_contract": onboard,
                    "status": "compatible" if compatible else "manual_review",
                    "message": (
                        "Candidate manifest and FSW contract match the workspace."
                        if compatible
                        else "Candidate manifest or FSW contract does not match the workspace contract."
                    ),
                }
            )
            if sha256_file(path) != initial_sha256:
                rows[-1].update(status="cancelled", message="Candidate manifest changed during audit; rerun.")
        except Exception as exc:
            rows.append({"path": relative, "status": "invalid", "message": str(exc)})
    return rows


def plan_migration(value: str | Path, *, target_version: str, release_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    workspace = load_workspace(value)
    report = audit_workspace(workspace["root"], target_version=target_version, release_manifest=release_manifest)
    if report["status"] in {"blocked", "invalid", "manual_review", "incomplete", "cancelled"}:
        return {
            "schema_version": MIGRATION_PLAN_SCHEMA,
            "status": report["status"],
            "workspace": report["workspace"],
            "target": report["target"],
            "compatibility_report": report["report_path"],
            "changes": [],
            "effects": {"user_source_modified": False},
        }
    migration_id = str(uuid.uuid4())
    migration_root = Path(workspace["root"]) / ".oel" / "migrations" / migration_id
    proposed_root = migration_root / "proposed"
    changes: list[dict[str, Any]] = []
    for row in report["configs"]:
        if row["status"] != "migration_available":
            continue
        original = Path(workspace["root"]) / row["path"]
        raw = yaml.safe_load(original.read_text(encoding="utf-8"))
        raw = dict(raw)
        raw["schema_version"] = SCENARIO_SCHEMA_VERSION
        proposed_text = yaml.safe_dump(raw, sort_keys=False)
        proposed = proposed_root / row["path"]
        atomic_write_text(proposed, proposed_text)
        original_text = original.read_text(encoding="utf-8")
        changes.append(
            {
                "kind": "scenario_schema",
                "path": row["path"],
                "original_sha256": sha256_file(original),
                "proposed_sha256": sha256_file(proposed),
                "proposed_path": str(proposed),
                "diff": "".join(
                    difflib.unified_diff(
                        original_text.splitlines(keepends=True),
                        proposed_text.splitlines(keepends=True),
                        fromfile=row["path"],
                        tofile=f"{row['path']} ({SCENARIO_SCHEMA_VERSION})",
                    )
                ),
            }
        )
    manifest_path = Path(workspace["manifest_path"])
    raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    proposed_manifest = dict(raw_manifest)
    proposed_manifest["engine"] = dict(proposed_manifest["engine"])
    proposed_manifest["engine"]["locked_version"] = target_version
    proposed_manifest_path = proposed_root / WORKSPACE_FILENAME
    proposed_manifest_text = yaml.safe_dump(proposed_manifest, sort_keys=False)
    atomic_write_text(proposed_manifest_path, proposed_manifest_text)
    changes.append(
        {
            "kind": "engine_pin",
            "path": WORKSPACE_FILENAME,
            "original_sha256": sha256_file(manifest_path),
            "proposed_sha256": sha256_file(proposed_manifest_path),
            "proposed_path": str(proposed_manifest_path),
            "diff": "".join(
                difflib.unified_diff(
                    manifest_path.read_text(encoding="utf-8").splitlines(keepends=True),
                    proposed_manifest_text.splitlines(keepends=True),
                    fromfile=WORKSPACE_FILENAME,
                    tofile=f"{WORKSPACE_FILENAME} ({target_version})",
                )
            ),
        }
    )
    plan = {
        "schema_version": MIGRATION_PLAN_SCHEMA,
        "migration_id": migration_id,
        "status": "ready",
        "generated_at": utc_now(),
        "workspace": report["workspace"],
        "target": report["target"],
        "compatibility_report": report["report_path"],
        "changes": changes,
        "effects": {"user_source_modified": False, "code_executed": False},
    }
    plan_path = migration_root / "migration-plan.json"
    atomic_write_json(plan_path, plan)
    plan["plan_path"] = str(plan_path)
    return plan


def apply_migration(plan_path: str | Path) -> dict[str, Any]:
    source = Path(plan_path).expanduser().resolve()
    existing_receipt = source.parent / "migration-receipt.json"
    if existing_receipt.is_file():
        existing = json.loads(existing_receipt.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and existing.get("status") == "ready":
            return {**existing, "receipt_path": str(existing_receipt), "idempotent": True}
    plan = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != MIGRATION_PLAN_SCHEMA or plan.get("status") != "ready":
        raise ContractError("Migration plan is not a ready OEL migration plan.")
    workspace_root = Path(plan["workspace"]["root"]).resolve()
    current = load_workspace(workspace_root)
    if current["manifest_sha256"] != plan["workspace"]["manifest_sha256"]:
        raise ContractError("Workspace manifest changed after the migration plan was created.")
    backup_root = source.parent / "originals"
    applied: list[dict[str, Any]] = []
    staged: list[tuple[Path, Path, Path, Mapping[str, Any]]] = []
    for change in plan["changes"]:
        target = _inside(workspace_root / change["path"], workspace_root, label="migration target")
        if sha256_file(target) != change["original_sha256"]:
            raise ContractError(f"Migration target changed after planning: {change['path']}")
        proposed = Path(change["proposed_path"]).resolve()
        if sha256_file(proposed) != change["proposed_sha256"]:
            raise ContractError(f"Proposed migration content changed: {proposed}")
        backup = backup_root / change["path"]
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        staged.append((target, proposed, backup, change))
    try:
        for target, proposed, backup, change in staged:
            atomic_write_text(target, proposed.read_text(encoding="utf-8"))
            applied.append({"path": change["path"], "backup": str(backup), "sha256": sha256_file(target)})
    except Exception:
        for target, _, backup, _change in reversed(staged[: len(applied)]):
            if backup.is_file():
                atomic_write_text(target, backup.read_text(encoding="utf-8"))
        raise
    receipt = {
        "schema_version": "oel.workspace-migration-receipt.v1",
        "status": "ready",
        "migration_id": plan["migration_id"],
        "applied_at": utc_now(),
        "workspace": {"id": current["workspace_id"], "root": str(workspace_root)},
        "target": plan["target"],
        "applied": applied,
        "effects": {"user_source_modified": bool(applied), "code_executed": False, "network_used": False},
        "idempotent": False,
    }
    receipt_path = source.parent / "migration-receipt.json"
    atomic_write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt
