"""Safe candidate inspection and trusted candidate validation."""

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
    CandidateEntrypoint,
    CandidateSource,
    CandidateVerification,
    FlightSoftwareCandidate,
    FSWDKIssue,
    canonical_json,
    sha256_file,
    sha256_tree,
    sha256_value,
)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SCHEMA_PATH = Path(__file__).with_name("schemas") / "candidate.schema.json"
QUALIFICATION_SCHEMA_PATH = CANDIDATE_SCHEMA_PATH.with_name("qualification.schema.json")
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_CLASS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_CANDIDATE_IMPORT_PREFIXES = (
    "sim.actuators",
    "sim.core",
    "sim.dynamics",
    "sim.runtime",
    "sim.sensors",
)


class CandidateValidationError(ValueError):
    def __init__(self, issues: list[FSWDKIssue] | tuple[FSWDKIssue, ...]):
        self.issues = tuple(issues)
        text = "\n- ".join(issue.message for issue in self.issues)
        super().__init__(f"FSWDK candidate validation failed:\n- {text}")


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CandidateValidationError(
            [FSWDKIssue("candidate_not_object", "Candidate manifest must be a YAML or JSON object.", str(path))]
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
            [FSWDKIssue("path_outside_workspace", f"{label} must remain inside the authorized workspace.", str(path))]
        ) from exc
    if must_exist and not resolved.exists():
        raise CandidateValidationError([FSWDKIssue("path_missing", f"{label} was not found.", str(resolved))])
    return resolved


def _candidate_path(value: Any, *, candidate_root: Path, workspace_root: Path, label: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    raw = Path(text)
    path = raw if raw.is_absolute() else candidate_root / raw
    if path.is_symlink():
        raise CandidateValidationError([FSWDKIssue("symlink_input", f"{label} may not be a symbolic link.", str(path))])
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
            FSWDKIssue(
                "entrypoint_source_missing",
                f"Candidate entrypoint module {module!r} does not resolve to a Python source file.",
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


def load_candidate(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
) -> FlightSoftwareCandidate:
    workspace = Path(workspace_root).expanduser().resolve()
    unresolved_manifest = Path(manifest).expanduser()
    if unresolved_manifest.is_symlink():
        raise CandidateValidationError(
            [
                FSWDKIssue(
                    "symlink_manifest",
                    "Candidate manifests may not be symbolic links.",
                    str(unresolved_manifest),
                )
            ]
        )
    manifest_path = _inside(unresolved_manifest, workspace, label="candidate manifest")
    raw = _load_mapping(manifest_path)
    schema_errors = sorted(Draft202012Validator(_schema()).iter_errors(raw), key=lambda item: list(item.path))
    if schema_errors:
        issues = [
            FSWDKIssue(
                "schema_invalid",
                error.message,
                ".".join(str(part) for part in error.absolute_path),
                next_step="Repair the candidate manifest and rerun inspect or validate.",
            )
            for error in schema_errors
        ]
        raise CandidateValidationError(issues)
    if raw["schema_version"] != CANDIDATE_SCHEMA_ID:
        raise CandidateValidationError(
            [FSWDKIssue("schema_version", f"schema_version must be {CANDIDATE_SCHEMA_ID!r}.", "schema_version")]
        )
    candidate_id = str(raw["candidate_id"])
    if not _ID_RE.fullmatch(candidate_id):
        raise CandidateValidationError(
            [FSWDKIssue("candidate_id", "candidate_id must use lowercase snake_case and contain 2-64 characters.")]
        )
    source_raw = dict(raw["source"])
    source_root_text = str(source_raw.get("root", ".") or ".")
    source_root_value = Path(source_root_text)
    unresolved_source_root = (
        source_root_value if source_root_value.is_absolute() else manifest_path.parent / source_root_value
    )
    if unresolved_source_root.is_symlink():
        raise CandidateValidationError(
            [
                FSWDKIssue(
                    "symlink_source",
                    "Candidate source roots may not be symbolic links.",
                    str(unresolved_source_root),
                )
            ]
        )
    source_root = _inside(
        unresolved_source_root,
        workspace,
        label="candidate source root",
    )
    kind = str(raw["kind"])
    entrypoint: CandidateEntrypoint | None = None
    artifact: Path | None = None
    if kind == "python_stack":
        entry_raw = dict(source_raw["entrypoint"])
        module = str(entry_raw["module"])
        class_name = str(entry_raw["class_name"])
        if not _MODULE_RE.fullmatch(module) or not _CLASS_RE.fullmatch(class_name):
            raise CandidateValidationError(
                [FSWDKIssue("entrypoint_invalid", "Python module or class_name is not a valid identifier.")]
            )
        entrypoint = CandidateEntrypoint(module, class_name)
        source_path = _entrypoint_source_path(module, workspace_root=workspace)
        _inside(source_path, source_root, label="candidate entrypoint")
    else:
        artifact = _candidate_path(
            source_raw.get("artifact"),
            candidate_root=source_root,
            workspace_root=workspace,
            label="candidate artifact",
        )
        if artifact is None or not artifact.is_file():
            raise CandidateValidationError(
                [FSWDKIssue("artifact_missing", "External and cFS candidates require source.artifact.")]
            )
    verification_raw = dict(raw["verification"])
    verification = CandidateVerification(
        component_suite=_candidate_path(
            verification_raw.get("component_suite"),
            candidate_root=manifest_path.parent,
            workspace_root=workspace,
            label="component test suite",
        ),
        smoke_case=_candidate_path(
            verification_raw.get("smoke_case"),
            candidate_root=manifest_path.parent,
            workspace_root=workspace,
            label="smoke scenario",
        ),
        bench_suite=_candidate_path(
            verification_raw.get("bench_suite"),
            candidate_root=manifest_path.parent,
            workspace_root=workspace,
            label="Controller Bench suite",
        ),
        qualification_spec=_candidate_path(
            verification_raw.get("qualification_spec"),
            candidate_root=manifest_path.parent,
            workspace_root=workspace,
            label="qualification specification",
        ),
    )
    claims = dict(raw["claims"])
    declared_envelope = _candidate_path(
        claims.get("declared_envelope"),
        candidate_root=manifest_path.parent,
        workspace_root=workspace,
        label="declared envelope",
    )
    interfaces = dict(raw["interfaces"])
    task_period_s = float(interfaces["task_period_s"])
    if not math.isfinite(task_period_s) or task_period_s <= 0.0:
        raise CandidateValidationError(
            [FSWDKIssue("task_period", "interfaces.task_period_s must be finite and greater than zero.")]
        )
    normalized = json.loads(canonical_json(raw))
    manifest_sha256 = sha256_value(normalized)
    try:
        source_sha256 = (
            sha256_tree(source_root)
            if kind == "python_stack"
            else sha256_file(artifact)
        )
        verification_sha256 = sha256_value(
            {
                "component_suite": sha256_tree(verification.component_suite) if verification.component_suite else "",
                "smoke_case": sha256_tree(verification.smoke_case) if verification.smoke_case else "",
                "bench_suite": sha256_tree(verification.bench_suite) if verification.bench_suite else "",
                "qualification_spec": sha256_tree(verification.qualification_spec)
                if verification.qualification_spec
                else "",
                "declared_envelope": sha256_tree(declared_envelope) if declared_envelope else "",
            }
        )
    except ValueError as exc:
        raise CandidateValidationError([FSWDKIssue("symlink_tree", str(exc), str(source_root))]) from exc
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
        kind=kind,
        manifest_path=manifest_path,
        workspace_root=workspace,
        source=CandidateSource(
            source_root,
            str(source_raw["revision_id"]),
            entrypoint=entrypoint,
            artifact=artifact,
        ),
        onboard_contract=str(interfaces["onboard_contract"]),
        hardware_profile=str(interfaces["hardware_profile"]),
        task_period_s=task_period_s,
        intended_use=str(claims["intended_use"]),
        declared_envelope=declared_envelope,
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
    paths = {
        "manifest": candidate.manifest_path,
        "source_root": candidate.source.root,
        "component_suite": candidate.verification.component_suite,
        "smoke_case": candidate.verification.smoke_case,
        "bench_suite": candidate.verification.bench_suite,
        "qualification_spec": candidate.verification.qualification_spec,
        "declared_envelope": candidate.declared_envelope,
    }
    return {
        "status": "ready",
        "candidate": candidate.identity(),
        "intended_use": candidate.intended_use,
        "handling": dict(candidate.handling),
        "paths": {key: str(value) if value is not None else "" for key, value in paths.items()},
        "entrypoint": candidate.source.entrypoint.to_dict() if candidate.source.entrypoint else None,
        "safe_inspection": True,
        "candidate_code_imported": False,
        "candidate_code_executed": False,
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
) -> tuple[FlightSoftwareCandidate, list[FSWDKIssue], dict[str, Any]]:
    candidate = load_candidate(manifest, workspace_root=workspace_root)
    issues: list[FSWDKIssue] = []
    checks: dict[str, Any] = {
        "manifest_schema": "passed",
        "path_policy": "passed",
        "source_identity": "passed",
        "truth_firewall": "not_run",
        "candidate_import": "not_run",
        "lifecycle": "not_run",
        "smoke_config": "not_run",
        "bench_config": "not_run",
        "qualification_spec": "not_run",
    }
    if candidate.kind == "python_stack":
        violations = _candidate_truth_import_violations(candidate.source.root)
        if violations:
            issues.extend(
                FSWDKIssue(
                    "truth_boundary_import",
                    f"Candidate source imports simulator-owned module {module!r}.",
                    f"{path}:{line}",
                    next_step="Use typed sim.flight_software inputs and sim.gnc components only.",
                )
                for path, line, module in violations
            )
            checks["truth_firewall"] = "failed"
        else:
            checks["truth_firewall"] = "passed"
    smoke = candidate.verification.smoke_case
    if smoke is not None:
        try:
            cfg = load_simulation_yaml(smoke)
            checks["smoke_config"] = "schema_valid"
            if trusted_import:
                with _workspace_import_path(candidate.workspace_root):
                    if candidate.source.entrypoint is not None:
                        clear_candidate_imports(candidate.source.entrypoint.module, source_root=candidate.source.root)
                    plugin_errors = validate_scenario_plugins(cfg)
                if plugin_errors:
                    for message in plugin_errors:
                        issues.append(FSWDKIssue("smoke_plugin_invalid", message, str(smoke)))
                    checks["smoke_config"] = "failed"
                else:
                    checks["smoke_config"] = "passed"
        except Exception as exc:
            issues.append(FSWDKIssue("smoke_config_invalid", str(exc), str(smoke)))
            checks["smoke_config"] = "failed"
    if candidate.verification.bench_suite is not None:
        try:
            raw = yaml.safe_load(candidate.verification.bench_suite.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not raw.get("cases") or not raw.get("variants"):
                raise ValueError("Controller Bench suite must declare non-empty cases and variants.")
            checks["bench_config"] = "structurally_valid"
        except Exception as exc:
            issues.append(FSWDKIssue("bench_config_invalid", str(exc), str(candidate.verification.bench_suite)))
            checks["bench_config"] = "failed"
    if candidate.verification.qualification_spec is not None:
        try:
            raw = yaml.safe_load(candidate.verification.qualification_spec.read_text(encoding="utf-8"))
            qualification_schema = json.loads(QUALIFICATION_SCHEMA_PATH.read_text(encoding="utf-8"))
            qualification_errors = sorted(
                Draft202012Validator(qualification_schema).iter_errors(raw),
                key=lambda item: list(item.path),
            )
            if qualification_errors:
                raise ValueError(
                    "Qualification schema failed: " + "; ".join(error.message for error in qualification_errors)
                )
            requirement_ids = {str(item["id"]) for item in raw["requirements"]}
            referenced = {str(requirement_id) for gate in raw["gates"] for requirement_id in gate["requirement_ids"]}
            unknown = sorted(referenced - requirement_ids)
            if unknown:
                raise ValueError("Qualification gates reference unknown requirements: " + ", ".join(unknown))
            if str(raw["candidate_id"]) != candidate.candidate_id:
                raise ValueError("Qualification candidate_id does not match the candidate manifest.")
            declared_envelope = candidate.declared_envelope
            spec_envelope = (candidate.verification.qualification_spec.parent / str(raw["envelope"])).resolve()
            if declared_envelope is None or spec_envelope != declared_envelope.resolve():
                raise ValueError("Qualification envelope does not match the candidate declared_envelope.")
            checks["qualification_spec"] = "passed"
        except Exception as exc:
            issues.append(
                FSWDKIssue(
                    "qualification_spec_invalid",
                    str(exc),
                    str(candidate.verification.qualification_spec),
                )
            )
            checks["qualification_spec"] = "failed"
    if trusted_import and candidate.kind == "python_stack" and candidate.source.entrypoint is not None:
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
            issues.append(FSWDKIssue("candidate_contract_invalid", str(exc), str(candidate.source.root)))
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
                if module.startswith(_FORBIDDEN_CANDIDATE_IMPORT_PREFIXES):
                    violations.append((str(source_path), int(node.lineno), module))
    return violations
