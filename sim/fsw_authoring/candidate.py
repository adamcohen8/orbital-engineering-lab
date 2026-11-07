"""Safe inspection and explicitly trusted validation for public FSW candidates."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import math
import re
import sys
import sysconfig
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml
from jsonschema import Draft202012Validator

from sim.config import load_simulation_yaml, validate_scenario_plugins

from .contracts import (
    CANDIDATE_SCHEMA_ID,
    AuthoringIssue,
    CandidateEntrypoint,
    CandidateSource,
    CandidateVerification,
    FlightSoftwareCandidate,
    canonical_json,
    sha256_tree,
    sha256_value,
)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SCHEMA_PATH = Path(__file__).with_name("schemas") / "candidate.schema.json"
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_CLASS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_IMPORT_PREFIXES = (
    "sim.actuators",
    "sim.core",
    "sim.dynamics",
    "sim.runtime",
    "sim.sensors",
)


class CandidateValidationError(ValueError):
    def __init__(self, issues: list[AuthoringIssue] | tuple[AuthoringIssue, ...]):
        self.issues = tuple(issues)
        text = "\n- ".join(issue.message for issue in self.issues)
        super().__init__(f"FSW candidate validation failed:\n- {text}")


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.suffix.lower() == ".json"
            else yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CandidateValidationError(
            [AuthoringIssue("candidate_unreadable", f"Candidate manifest could not be read: {exc}", str(path))]
        ) from exc
    if not isinstance(raw, dict):
        raise CandidateValidationError(
            [AuthoringIssue("candidate_not_object", "Candidate manifest must be a YAML or JSON object.", str(path))]
        )
    return raw


def _schema() -> dict[str, Any]:
    return json.loads(CANDIDATE_SCHEMA_PATH.read_text(encoding="utf-8"))


def _inside(path: Path, root: Path, *, label: str, must_exist: bool = True) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CandidateValidationError(
            [
                AuthoringIssue(
                    "path_outside_workspace", f"{label} must remain inside the authorized workspace.", str(path)
                )
            ]
        ) from exc
    if must_exist and not resolved.exists():
        raise CandidateValidationError([AuthoringIssue("path_missing", f"{label} was not found.", str(resolved))])
    return resolved


def _candidate_path(value: Any, *, candidate_root: Path, workspace_root: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise CandidateValidationError([AuthoringIssue("path_missing", f"{label} must be declared.")])
    raw = Path(text)
    path = raw if raw.is_absolute() else candidate_root / raw
    if path.is_symlink():
        raise CandidateValidationError(
            [AuthoringIssue("symlink_input", f"{label} may not be a symbolic link.", str(path))]
        )
    return _inside(path, workspace_root, label=label)


def _entrypoint_source_path(module: str, *, workspace_root: Path) -> Path:
    module_path = workspace_root.joinpath(*module.split("."))
    file_path = module_path.with_suffix(".py")
    package_path = module_path / "__init__.py"
    if file_path.is_file():
        return file_path.resolve()
    if package_path.is_file():
        return package_path.resolve()
    raise CandidateValidationError(
        [
            AuthoringIssue(
                "entrypoint_source_missing",
                f"Candidate entrypoint module {module!r} does not resolve to Python source.",
                str(file_path),
            )
        ]
    )


def _candidate_source_modules(module_name: str, root: Path) -> dict[str, Path]:
    """Find local source imports without importing packages or scanning the repository."""
    runtime_paths = sysconfig.get_paths()
    runtime_roots = tuple(
        Path(runtime_paths[key]).resolve()
        for key in ("stdlib", "platstdlib", "purelib", "platlib")
        if runtime_paths.get(key)
    )
    pending = [module_name]
    found: dict[str, Path] = {}
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        # OEL framework imports are not editable candidate helper modules.
        if name != module_name and (name == "sim" or name.startswith("sim.")):
            continue
        source = None
        for entry in sys.path:
            base = Path(entry or ".").joinpath(*name.split("."))
            # Python resolves a regular package before a same-named module file.
            choices = (base / "__init__.py", base.with_suffix(".py"))
            existing = next((path for path in choices if path.is_file()), None)
            if existing is not None:
                resolved = existing.resolve()
                installed_runtime = (
                    any(resolved.is_relative_to(path) for path in runtime_roots)
                    or "site-packages" in resolved.parts
                    or "dist-packages" in resolved.parts
                )
                if resolved.is_relative_to(root) and not installed_runtime:
                    source = resolved
                break
        if source is None:
            continue
        found[name] = source
        package = name if source.name == "__init__.py" else name.rpartition(".")[0]
        if package and package != name:
            pending.append(package)
        for node in ast.walk(ast.parse(source.read_bytes(), filename=str(source))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = ("." * node.level) + (node.module or "")
                if node.level:
                    if not package:
                        continue
                    target = importlib.util.resolve_name(target, package)
                if target:
                    pending.append(target)
                    pending.extend(f"{target}.{alias.name}" for alias in node.names if alias.name != "*")
    return found


def clear_candidate_imports(module_name: str, *, source_root: Path | None = None) -> None:
    """Evict the candidate and its statically imported local helpers before trusted reload."""
    sources = {} if source_root is None else _candidate_source_modules(module_name, Path(source_root).resolve())
    for source in sources.values():
        cache = Path(importlib.util.cache_from_source(str(source)))
        cache.resolve().relative_to(Path(source_root).resolve())
        cache.unlink(missing_ok=True)
    parts = module_name.split(".")
    prefixes = {".".join(parts[:index]) for index in range(1, len(parts) + 1)}
    for name in list(sys.modules):
        if name in sources or name in prefixes or name == module_name or name.startswith(f"{module_name}."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def load_candidate(manifest: str | Path, *, workspace_root: str | Path = ROOT) -> FlightSoftwareCandidate:
    workspace = Path(workspace_root).expanduser().resolve()
    unresolved_manifest = Path(manifest).expanduser()
    if unresolved_manifest.is_symlink():
        raise CandidateValidationError(
            [AuthoringIssue("symlink_manifest", "Candidate manifests may not be symbolic links.", str(manifest))]
        )
    manifest_path = _inside(unresolved_manifest, workspace, label="candidate manifest")
    raw = _load_mapping(manifest_path)
    if raw.get("kind") == "native_stack":
        from .native import load_native_candidate

        return load_native_candidate(raw, manifest_path, workspace)
    schema_errors = sorted(Draft202012Validator(_schema()).iter_errors(raw), key=lambda item: list(item.path))
    if schema_errors:
        raise CandidateValidationError(
            [
                AuthoringIssue(
                    "schema_invalid",
                    error.message,
                    ".".join(str(part) for part in error.absolute_path),
                    next_step="Repair the public candidate manifest and rerun inspect.",
                )
                for error in schema_errors
            ]
        )
    if raw["schema_version"] != CANDIDATE_SCHEMA_ID or raw["kind"] != "python_stack":
        raise CandidateValidationError(
            [
                AuthoringIssue(
                    "public_contract_required", "The public authoring kit accepts Python stack candidates only."
                )
            ]
        )
    candidate_id = str(raw["candidate_id"])
    if not _ID_RE.fullmatch(candidate_id):
        raise CandidateValidationError(
            [AuthoringIssue("candidate_id", "candidate_id must use lowercase snake_case and contain 2-64 characters.")]
        )
    source_raw = dict(raw["source"])
    source_root_value = Path(str(source_raw.get("root", ".") or "."))
    unresolved_source_root = (
        source_root_value if source_root_value.is_absolute() else manifest_path.parent / source_root_value
    )
    if unresolved_source_root.is_symlink():
        raise CandidateValidationError(
            [AuthoringIssue("symlink_source", "Candidate source roots may not be symbolic links.")]
        )
    source_root = _inside(unresolved_source_root, workspace, label="candidate source root")
    entry_raw = dict(source_raw["entrypoint"])
    module = str(entry_raw["module"])
    class_name = str(entry_raw["class_name"])
    if not _MODULE_RE.fullmatch(module) or not _CLASS_RE.fullmatch(class_name):
        raise CandidateValidationError(
            [AuthoringIssue("entrypoint_invalid", "Python module or class_name is not a valid identifier.")]
        )
    source_path = _entrypoint_source_path(module, workspace_root=workspace)
    _inside(source_path, source_root, label="candidate entrypoint")
    verification_raw = dict(raw["verification"])
    verification = CandidateVerification(
        component_suite=_candidate_path(
            verification_raw["component_suite"],
            candidate_root=manifest_path.parent,
            workspace_root=workspace,
            label="component test suite",
        ),
        smoke_case=_candidate_path(
            verification_raw["smoke_case"],
            candidate_root=manifest_path.parent,
            workspace_root=workspace,
            label="smoke scenario",
        ),
    )
    interfaces = dict(raw["interfaces"])
    task_period_s = float(interfaces["task_period_s"])
    if not math.isfinite(task_period_s) or task_period_s <= 0.0:
        raise CandidateValidationError(
            [AuthoringIssue("task_period", "interfaces.task_period_s must be finite and greater than zero.")]
        )
    normalized = json.loads(canonical_json(raw))
    manifest_sha256 = sha256_value(normalized)
    try:
        source_sha256 = sha256_tree(
            source_root,
            suffixes=frozenset({".py", ".json", ".yaml", ".yml"}),
        )
        verification_sha256 = sha256_value(
            {
                "component_suite": sha256_tree(verification.component_suite),
                "smoke_case": sha256_tree(verification.smoke_case),
            }
        )
    except ValueError as exc:
        raise CandidateValidationError([AuthoringIssue("symlink_tree", str(exc), str(source_root))]) from exc
    candidate_sha256 = sha256_value(
        {
            "manifest_sha256": manifest_sha256,
            "source_sha256": source_sha256,
            "verification_sha256": verification_sha256,
            "contract_version": interfaces["onboard_contract"],
        }
    )
    return FlightSoftwareCandidate(
        candidate_id=candidate_id,
        revision=str(raw["revision"]),
        manifest_path=manifest_path,
        workspace_root=workspace,
        source=CandidateSource(source_root, str(source_raw["revision_id"]), CandidateEntrypoint(module, class_name)),
        onboard_contract=str(interfaces["onboard_contract"]),
        hardware_profile=str(interfaces["hardware_profile"]),
        task_period_s=task_period_s,
        intended_use=str(dict(raw["claims"])["intended_use"]),
        verification=verification,
        handling=dict(raw["handling"]),
        normalized_manifest=normalized,
        manifest_sha256=manifest_sha256,
        source_sha256=source_sha256,
        verification_sha256=verification_sha256,
        candidate_sha256=candidate_sha256,
    )


def inspect_candidate(manifest: str | Path, *, workspace_root: str | Path = ROOT) -> dict[str, Any]:
    candidate = load_candidate(manifest, workspace_root=workspace_root)
    return {
        "schema": "oel.fsw_authoring.inspection.v1",
        "status": "ready",
        "candidate": candidate.identity(),
        "intended_use": candidate.intended_use,
        "handling": dict(candidate.handling),
        "paths": {
            "manifest": str(candidate.manifest_path),
            "source_root": str(candidate.source.root),
            "component_suite": str(candidate.verification.component_suite),
            "smoke_case": str(candidate.verification.smoke_case),
        },
        "entrypoint": candidate.source.entrypoint.to_dict() if candidate.source.entrypoint else None,
        "native": dict(candidate.native) if candidate.native else None,
        "safe_inspection": True,
        "candidate_code_imported": False,
        "candidate_code_executed": False,
        "private_operations_available": False,
    }


@contextmanager
def _workspace_import_path(workspace_root: Path) -> Iterator[None]:
    value = str(workspace_root)
    added = value not in sys.path
    if added:
        sys.path.insert(0, value)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        if added:
            sys.path.remove(value)
        importlib.invalidate_caches()


def validate_candidate(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    trusted_import: bool = False,
) -> tuple[FlightSoftwareCandidate, list[AuthoringIssue], dict[str, Any]]:
    candidate = load_candidate(manifest, workspace_root=workspace_root)
    if candidate.native is not None:
        from .native import validate_native

        issues, checks = validate_native(candidate, trusted_plugins=trusted_import)
        return candidate, issues, checks
    issues: list[AuthoringIssue] = []
    checks: dict[str, Any] = {
        "manifest_schema": "passed",
        "path_policy": "passed",
        "source_identity": "passed",
        "truth_firewall": "not_run",
        "candidate_import": "not_run",
        "lifecycle": "not_run",
        "smoke_config": "not_run",
        "public_execution_boundary": "not_run",
    }
    violations = _candidate_truth_import_violations(candidate.source.root)
    if violations:
        issues.extend(
            AuthoringIssue(
                "truth_boundary_import",
                f"Candidate source imports simulator-owned module {module!r}.",
                f"{path}:{line}",
                next_step="Use typed sim.flight_software inputs and public sim.gnc components only.",
            )
            for path, line, module in violations
        )
        checks["truth_firewall"] = "failed"
    else:
        checks["truth_firewall"] = "passed"
    try:
        cfg = load_simulation_yaml(candidate.verification.smoke_case)
        checks["smoke_config"] = "schema_valid"
        if bool(cfg.analysis.enabled) or bool(cfg.monte_carlo.enabled):
            raise ValueError("Public candidate smoke scenarios may not enable analysis or Monte Carlo workflows.")
        object_policy = str(getattr(cfg.simulator, "object_execution_policy", "serial") or "serial")
        if object_policy not in {"serial", "configured"}:
            raise ValueError("Public candidate smoke scenarios must use deterministic serial object execution.")
        checks["public_execution_boundary"] = "passed"
        if trusted_import:
            with _workspace_import_path(candidate.workspace_root):
                clear_candidate_imports(candidate.source.entrypoint.module, source_root=candidate.source.root)
                plugin_errors = validate_scenario_plugins(cfg)
            if plugin_errors:
                issues.extend(
                    AuthoringIssue("smoke_plugin_invalid", message, str(candidate.verification.smoke_case))
                    for message in plugin_errors
                )
                checks["smoke_config"] = "failed"
            else:
                checks["smoke_config"] = "passed"
    except Exception as exc:
        issues.append(AuthoringIssue("smoke_config_invalid", str(exc), str(candidate.verification.smoke_case)))
        checks["smoke_config"] = "failed"
        if checks["public_execution_boundary"] == "not_run":
            checks["public_execution_boundary"] = "failed"
    if trusted_import:
        try:
            with _workspace_import_path(candidate.workspace_root):
                clear_candidate_imports(candidate.source.entrypoint.module, source_root=candidate.source.root)
                module = importlib.import_module(candidate.source.entrypoint.module)
                imported_path = Path(str(module.__file__ or "")).resolve()
                _inside(imported_path, candidate.source.root, label="imported candidate module")
                stack_type = getattr(module, candidate.source.entrypoint.class_name)
                stack = stack_type()
            checks["candidate_import"] = "passed"
            missing = [
                method
                for method in ("boot", "step", "shutdown", "snapshot", "restore")
                if not callable(getattr(stack, method, None))
            ]
            if missing:
                raise TypeError("Candidate is missing required lifecycle methods: " + ", ".join(missing))
            checks["lifecycle"] = "passed"
        except Exception as exc:
            issues.append(AuthoringIssue("candidate_contract_invalid", str(exc), str(candidate.source.root)))
            checks["candidate_import"] = "failed"
            checks["lifecycle"] = "failed"
    return candidate, issues, checks


def _candidate_truth_import_violations(source_root: Path) -> list[tuple[str, int, str]]:
    violations: list[tuple[str, int, str]] = []
    for source_path in sorted(source_root.rglob("*.py")):
        relative = source_path.relative_to(source_root)
        if "tests" in relative.parts or "__pycache__" in relative.parts:
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            for module in modules:
                if module.startswith(_FORBIDDEN_IMPORT_PREFIXES):
                    violations.append((str(source_path), int(node.lineno), module))
    return violations


__all__ = [
    "CANDIDATE_SCHEMA_PATH",
    "CandidateValidationError",
    "ROOT",
    "clear_candidate_imports",
    "inspect_candidate",
    "load_candidate",
    "validate_candidate",
]
