"""Transport-neutral orchestration services for OEL FSWDK.

Every service delegates simulation, benchmark, and review behavior to existing
OEL owners.  This module owns candidate lifecycle and evidence composition.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from xml.etree import ElementTree

import yaml

from sim.api import SimulationWorkspace
from sim.config import load_simulation_yaml
from sim.controller_lab import run_controller_bench, validate_controller_bench_config
from sim.controller_lab.agent import inspect_controller_bench_output
from sim.installation.provenance import execution_provenance
from sim.installation.workspace import WORKSPACE_FILENAME, load_workspace
from sim.resource_limits import estimate_resource_requirements
from sim.security.output_writes import (
    SecureWriteConflict,
    bind_output_writer,
    current_output_writer,
    mkdir_output,
    rewrite_staged_paths,
    sha256_output,
    write_output_bytes,
)

from .candidate import (
    ROOT,
    CandidateValidationError,
    clear_candidate_imports,
    load_candidate,
    validate_candidate,
)
from .contracts import (
    AGENT_EVIDENCE_PACKET_SCHEMA_ID,
    COMPARISON_PACKET_SCHEMA_ID,
    EXECUTION_PLAN_SCHEMA_ID,
    FSWDK_NON_CLAIMS,
    PACKAGE_MANIFEST_SCHEMA_ID,
    QUALIFICATION_PACKET_SCHEMA_ID,
    RUN_MANIFEST_SCHEMA_ID,
    TEST_RESULT_SCHEMA_ID,
    VALIDATION_RECEIPT_SCHEMA_ID,
    WORK_ORDER_SCHEMA_ID,
    FSWDKIssue,
    FSWDKReceipt,
    effects,
    generated_utc,
    sha256_file,
    sha256_value,
)
from .scaffold import scaffold_candidate


class _OutputBinding:
    """Bind logical FSWDK evidence paths to a descriptor-backed writer."""

    def __init__(self, writer: Any, logical_root: Path) -> None:
        self.writer = writer
        self.logical_root = logical_root

    @property
    def path(self) -> Path:
        return Path(self.writer.path)

    def physical(self, logical_path: Path) -> Path:
        logical_path = Path(logical_path)
        if not logical_path.is_absolute():
            logical_path = self.logical_root / logical_path
        try:
            relative = logical_path.relative_to(self.logical_root)
        except ValueError as exc:
            raise PermissionError("Output path is outside the bound logical output root.") from exc
        return self.path / relative

    def ensure_directory(self, logical_path: Path) -> Path:
        logical_path = Path(logical_path)
        if not logical_path.is_absolute():
            logical_path = self.logical_root / logical_path
        with self.bound():
            mkdir_output(logical_path)
        physical = self.physical(logical_path)
        if not physical.is_dir():
            raise PermissionError("Secure output writer did not provide a directory-backed destination.")
        return physical

    def write_bytes(self, logical_path: Path, payload: bytes) -> Path:
        logical_path = Path(logical_path)
        if not logical_path.is_absolute():
            logical_path = self.logical_root / logical_path
        with self.bound():
            write_output_bytes(logical_path, payload)
        return logical_path

    def update_existing_bytes(self, logical_path: Path, payload: bytes) -> Path:
        logical_path = Path(logical_path)
        if not logical_path.is_absolute():
            logical_path = self.logical_root / logical_path
        relative = logical_path.relative_to(self.logical_root)
        self.writer.update_existing_bytes(relative, payload)
        return logical_path

    @contextmanager
    def bound(self) -> Iterator[None]:
        """Run one producer operation against this writer's logical root."""

        if current_output_writer() is self.writer:
            yield
            return
        with bind_output_writer(self.writer):
            yield


def _bind_writer(writer: Any | None, logical_root: Path) -> _OutputBinding | None:
    if writer is None:
        writer = current_output_writer()
    if writer is None:
        return None
    if isinstance(writer, _OutputBinding):
        return writer
    return _OutputBinding(writer, logical_root)


def _logicalize(value: Any, binding: _OutputBinding | None) -> Any:
    if binding is None:
        return value
    return rewrite_staged_paths(value, binding.path, binding.logical_root)


def _write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    writer: _OutputBinding | Any | None = None,
    logical_root: Path | None = None,
) -> Path:
    serialized = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")
    if writer is not None:
        binding = _bind_writer(writer, logical_root or path.parent)
        assert binding is not None
        try:
            return binding.write_bytes(path, serialized)
        except SecureWriteConflict:
            return binding.update_existing_bytes(path, serialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _artifact(
    path: Path,
    *,
    artifact_id: str,
    writer: _OutputBinding | Any | None = None,
    logical_root: Path | None = None,
) -> dict[str, Any]:
    binding = _bind_writer(writer, logical_root or path.parent)
    physical = binding.physical(path) if binding is not None else path
    if binding is not None:
        with binding.bound():
            exists = physical.is_file()
            digest = sha256_output(path) if exists else ""
    else:
        exists = physical.is_file()
        digest = sha256_file(physical) if exists else ""
    return {
        "artifact_id": artifact_id,
        "path": str(path),
        "exists": exists,
        "sha256": digest,
    }


def _declared_packet_artifact(path: Path, *, artifact_id: str) -> dict[str, Any]:
    """Describe a packet within itself without an impossible self-referential digest."""

    return {"artifact_id": artifact_id, "path": str(path), "exists": True, "digest_scope": "external"}


def _execution_identity(candidate: Any) -> dict[str, Any]:
    payload = execution_provenance()
    manifest = candidate.workspace_root / WORKSPACE_FILENAME
    if payload.get("workspace") is None and manifest.is_file():
        workspace = load_workspace(manifest)
        payload["workspace"] = {
            "workspace_id": workspace["workspace_id"],
            "manifest_sha256": workspace["manifest_sha256"],
            "locked_version": workspace["engine"]["locked_version"],
            "contracts": workspace["contracts"],
        }
    return payload


def _default_output(candidate_id: str, operation: str, *, workspace_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return workspace_root / "outputs" / "fswdk" / candidate_id / f"{timestamp}_{operation}"


def _new_output(
    path: str | Path | None,
    *,
    candidate_id: str,
    operation: str,
    workspace_root: Path,
    writer: _OutputBinding | Any | None = None,
) -> Path:
    if writer is None:
        writer = current_output_writer()
    output = (
        Path(path).expanduser()
        if path is not None
        else _default_output(candidate_id, operation, workspace_root=workspace_root)
    )
    if not output.is_absolute():
        output = workspace_root / output
    output = output.resolve()
    try:
        output.relative_to(workspace_root)
    except ValueError as exc:
        raise PermissionError("FSWDK output must remain inside the authorized workspace.") from exc
    if writer is None:
        if output.exists():
            raise FileExistsError(f"FSWDK output directory already exists: {output}")
        output.mkdir(parents=True)
    else:
        binding = _bind_writer(writer, output)
        assert binding is not None
        physical = binding.ensure_directory(output)
        if any(physical.iterdir()):
            raise FileExistsError(f"FSWDK output directory must be new or empty: {output}")
    return output


def describe_capabilities() -> dict[str, Any]:
    return {
        "schema": "oel.fswdk.capabilities.v1",
        "status": "ready",
        "product": "OEL Flight Software Development and Verification Kit",
        "templates": [
            {"id": "adcs", "maturity": "supported_vertical_slice", "external_network": False},
            {"id": "rpo", "maturity": "experimental", "external_network": False},
        ],
        "candidate_kinds": ["python_stack", "external_process", "cfs"],
        "operations": [
            "describe",
            "doctor",
            "init",
            "inspect",
            "plan",
            "validate",
            "test",
            "simulate",
            "validate-external",
            "simulate-external",
            "bench",
            "compare",
            "qualify",
            "promote-baseline",
            "package",
            "verify-package",
        ],
        "contracts": {
            "candidate": "oel.fswdk.candidate.v1",
            "work_order": WORK_ORDER_SCHEMA_ID,
            "execution_plan": EXECUTION_PLAN_SCHEMA_ID,
            "validation_receipt": VALIDATION_RECEIPT_SCHEMA_ID,
            "agent_evidence_packet": AGENT_EVIDENCE_PACKET_SCHEMA_ID,
        },
        "non_claims": list(FSWDK_NON_CLAIMS),
    }


def doctor(*, workspace_root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(workspace_root).expanduser().resolve()
    checks = {
        "workspace_exists": root.is_dir(),
        "workspace_readable": os.access(root, os.R_OK),
        "workspace_writable": os.access(root, os.W_OK),
        "python_supported": sys.version_info >= (3, 11),
        "candidate_schema": (Path(__file__).with_name("schemas") / "candidate.schema.json").is_file(),
        "qualification_schema": (Path(__file__).with_name("schemas") / "qualification.schema.json").is_file(),
        "work_order_schema": (Path(__file__).with_name("schemas") / "work_order.schema.json").is_file(),
        "execution_plan_schema": (Path(__file__).with_name("schemas") / "execution_plan.schema.json").is_file(),
        "receipt_schema": (Path(__file__).with_name("schemas") / "receipt.schema.json").is_file(),
        "controller_bench": (ROOT / "sim/controller_lab").is_dir(),
        "cfs_sil": (ROOT / "integrations/cfs_sil/ICD_v2.md").is_file(),
    }
    return {
        "schema": "oel.fswdk.doctor.v1",
        "status": "ready" if all(checks.values()) else "failed",
        "workspace_root": str(root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "network_required": False,
        },
        "checks": checks,
        "non_claims": list(FSWDK_NON_CLAIMS),
    }


def init_candidate(
    name: str,
    *,
    template: str = "adcs",
    project_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    class_name: str | None = None,
    force: bool = False,
    writer: _OutputBinding | Any | None = None,
) -> dict[str, Any]:
    result = scaffold_candidate(
        name,
        template=template,
        project_root=project_root,
        output_dir=output_dir,
        class_name=class_name,
        force=force,
        writer=writer,
    )
    artifact_binding = _bind_writer(writer, result.root_dir)
    receipt = FSWDKReceipt(
        schema="oel.fswdk.scaffold_receipt.v1",
        status="ready",
        operation="init",
        candidate=dict(result.candidate["candidate"]),
        effects=effects(writes=True),
        artifacts=tuple(
            _artifact(
                path,
                artifact_id=path.name,
                writer=writer,
                logical_root=result.root_dir,
            )
            for path in result.files_written
            if (artifact_binding.physical(path) if artifact_binding is not None else path).is_file()
        ),
        result={"root_dir": str(result.root_dir), "manifest_path": str(result.manifest_path)},
    ).to_dict()
    receipt_path = (result.root_dir / ".oel" / "fsw_scaffold_receipt.json").resolve()
    try:
        receipt_path.relative_to(Path(project_root).expanduser().resolve())
    except ValueError as exc:
        raise PermissionError("FSWDK scaffold receipt must remain inside the authorized workspace.") from exc
    _write_json(receipt_path, receipt, writer=writer, logical_root=result.root_dir)
    return receipt


def plan_workflow(
    manifest: str | Path,
    operation: str,
    *,
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    candidate = load_candidate(manifest, workspace_root=workspace_root)
    supported = {"validate", "test", "simulate", "bench", "qualify", "package"}
    if operation not in supported:
        raise ValueError(f"Unsupported FSWDK operation {operation!r}; choose one of {sorted(supported)}")
    writes = operation in supported
    executes = operation in {"test", "simulate", "bench", "qualify"}
    destination_path = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else _default_output(candidate.candidate_id, operation, workspace_root=candidate.workspace_root)
    )
    if not destination_path.is_absolute():
        destination_path = candidate.workspace_root / destination_path
    destination_path = destination_path.resolve()
    try:
        destination_path.relative_to(candidate.workspace_root)
    except ValueError as exc:
        raise PermissionError("FSWDK planned output must remain inside the authorized workspace.") from exc
    destination = str(destination_path)
    required_inputs = {
        "validate": [candidate.manifest_path],
        "test": [candidate.manifest_path, candidate.verification.component_suite],
        "simulate": [candidate.manifest_path, candidate.verification.smoke_case],
        "bench": [candidate.manifest_path, candidate.verification.bench_suite],
        "qualify": [candidate.manifest_path, candidate.verification.qualification_spec],
        "package": [candidate.manifest_path],
    }[operation]
    work_order = {
        "schema": WORK_ORDER_SCHEMA_ID,
        "status": "ready",
        "candidate": candidate.identity(),
        "operation": operation,
        "requested_output_dir": destination,
        "constraints": {
            "deterministic": True,
            "workers": 1,
            "network": False,
            "hosted_ai": False,
            "new_output_required": True,
        },
        "work_order_id": "",
    }
    work_order["work_order_id"] = sha256_value(
        {key: value for key, value in work_order.items() if key != "work_order_id"}
    )
    plan = {
        "schema": EXECUTION_PLAN_SCHEMA_ID,
        "status": "ready",
        "operation": operation,
        "candidate": candidate.identity(),
        "required_inputs": [str(item) for item in required_inputs if item is not None],
        "work_order": work_order,
        "output_dir": destination,
        "effects": effects(writes=writes, executes=executes),
        "approval_required": executes,
        "source_trust_required": executes,
        "hosted_ai": False,
        "network": False,
        "resource_plan": {
            "profile": "laptop-safe",
            "workers": 1,
            "bounded": True,
            "action": "proceed",
        },
        "plan_id": "",
        "non_claims": list(FSWDK_NON_CLAIMS),
    }
    plan["plan_id"] = sha256_value({key: value for key, value in plan.items() if key != "plan_id"})
    return plan


def validate_candidate_service(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    trusted_import: bool = False,
    write_receipt: bool = True,
    receipt_dir: str | Path | None = None,
    writer: _OutputBinding | Any | None = None,
) -> dict[str, Any]:
    try:
        candidate, issues, checks = validate_candidate(
            manifest,
            workspace_root=workspace_root,
            trusted_import=trusted_import,
        )
    except CandidateValidationError as exc:
        receipt = FSWDKReceipt(
            schema=VALIDATION_RECEIPT_SCHEMA_ID,
            status="invalid",
            operation="validate",
            effects=effects(writes=bool(write_receipt and receipt_dir is not None)),
            issues=exc.issues,
            result={"validation_id": "", "trusted_import": trusted_import},
        ).to_dict()
        if write_receipt and receipt_dir is not None:
            root = Path(workspace_root).expanduser().resolve()
            path = (Path(receipt_dir).expanduser() / "fsw_validation_receipt.json").resolve()
            try:
                path.relative_to(root)
            except ValueError as path_exc:
                raise PermissionError(
                    "FSWDK validation receipt must remain inside the authorized workspace."
                ) from path_exc
            logical_root = Path(receipt_dir).expanduser().resolve()
            _write_json(path, receipt, writer=writer, logical_root=logical_root)
            receipt["artifacts"] = [
                _artifact(path, artifact_id="validation_receipt", writer=writer, logical_root=logical_root)
            ]
        return receipt
    status = "ready" if not issues else "invalid"
    validation_id = sha256_value(
        {
            "candidate_sha256": candidate.candidate_sha256,
            "checks": checks,
            "issues": [item.to_dict() for item in issues],
        }
    )
    receipt = FSWDKReceipt(
        schema=VALIDATION_RECEIPT_SCHEMA_ID,
        status=status,
        operation="validate",
        candidate=candidate.identity(),
        effects=effects(writes=write_receipt),
        issues=tuple(issues),
        result={
            "validation_id": validation_id,
            "trusted_import": trusted_import,
            "execution_authorized": False,
            "checks": checks,
        },
    ).to_dict()
    if write_receipt:
        path = (
            Path(receipt_dir).expanduser().resolve() / "fsw_validation_receipt.json"
            if receipt_dir is not None
            else candidate.manifest_path.parent / ".oel" / "fsw_validation_receipt.json"
        )
        path = path.resolve()
        try:
            path.relative_to(candidate.workspace_root)
        except ValueError as exc:
            raise PermissionError("FSWDK validation receipt must remain inside the authorized workspace.") from exc
        logical_root = Path(receipt_dir).expanduser().resolve() if receipt_dir is not None else path.parent
        _write_json(path, receipt, writer=writer, logical_root=logical_root)
        receipt["artifacts"] = [
            _artifact(path, artifact_id="validation_receipt", writer=writer, logical_root=logical_root)
        ]
    return receipt


def _require_validated(
    manifest: str | Path,
    *,
    workspace_root: str | Path,
    validation_id: str | None,
) -> tuple[Any, dict[str, Any]]:
    receipt = validate_candidate_service(
        manifest,
        workspace_root=workspace_root,
        trusted_import=True,
        write_receipt=False,
    )
    if receipt["status"] != "ready":
        raise CandidateValidationError(
            [
                FSWDKIssue(str(item["code"]), str(item["message"]), str(item.get("path", "")))
                for item in receipt["issues"]
            ]
        )
    actual = str(receipt["result"]["validation_id"])
    if validation_id not in (None, "") and validation_id != actual:
        raise PermissionError("The supplied validation_id is stale or does not match the current candidate.")
    return load_candidate(manifest, workspace_root=workspace_root), receipt


def _require_candidate_current(candidate: Any) -> None:
    """Reject evidence if candidate content changed while an operation ran."""

    try:
        current = load_candidate(candidate.manifest_path, workspace_root=candidate.workspace_root)
    except Exception as exc:
        raise PermissionError("FSWDK candidate identity changed during execution.") from exc
    if current.candidate_sha256 != candidate.candidate_sha256:
        raise PermissionError("FSWDK candidate identity changed during execution.")


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_s: float = 300.0,
    cancellation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if cancellation_check is not None:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env or os.environ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        started = time.monotonic()
        try:
            while True:
                cancellation_check()
                try:
                    stdout, stderr = process.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started > timeout_s:
                        process.kill()
                        stdout, stderr = process.communicate()
                        return {
                            "command": command, "returncode": 124,
                            "stdout": stdout[-200_000:],
                            "stderr": (stderr + f"\nFSWDK component test timeout after {timeout_s:.1f} s.")[-200_000:],
                            "timed_out": True,
                        }
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
            raise
        return {
            "command": command,
            "returncode": process.returncode,
            "stdout": stdout[-200_000:],
            "stderr": stderr[-200_000:],
            "output_truncated": len(stdout) > 200_000 or len(stderr) > 200_000,
        }
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env or os.environ),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": str(exc.stdout or "")[-200_000:],
            "stderr": (str(exc.stderr or "") + f"\nFSWDK component test timeout after {timeout_s:.1f} s.")[-200_000:],
            "timed_out": True,
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-200_000:],
        "stderr": completed.stderr[-200_000:],
        "output_truncated": len(completed.stdout) > 200_000 or len(completed.stderr) > 200_000,
    }


def run_contract_tests(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    validation_id: str | None = None,
    cancellation_check: Callable[[], None] | None = None,
    writer: _OutputBinding | Any | None = None,
) -> dict[str, Any]:
    candidate, validation = _require_validated(
        manifest,
        workspace_root=workspace_root,
        validation_id=validation_id,
    )
    if candidate.verification.component_suite is None:
        raise ValueError("Candidate does not declare a component test suite.")
    output = _new_output(
        output_dir,
        candidate_id=candidate.candidate_id,
        operation="test",
        workspace_root=candidate.workspace_root,
        writer=writer,
    )
    binding = _bind_writer(writer, output)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(candidate.workspace_root), env.get("PYTHONPATH", "")) if value
    )
    if binding is not None:
        temporary = output / "subprocess_tmp"
        physical_temporary = binding.ensure_directory(temporary)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TMPDIR"] = str(physical_temporary)
        env["TEMP"] = str(physical_temporary)
        env["TMP"] = str(physical_temporary)
        env["PYTEST_ADDOPTS"] = (
            f"{env.get('PYTEST_ADDOPTS', '').strip()} --basetemp={shlex.quote(str(physical_temporary))} "
            "-p no:cacheprovider"
        ).strip()
    execution = _run_subprocess(
        [sys.executable, "-m", "pytest", str(candidate.verification.component_suite), "-q"],
        cwd=candidate.workspace_root,
        env=env,
        cancellation_check=cancellation_check,
    )
    _require_candidate_current(candidate)
    status = "passed" if execution["returncode"] == 0 else "failed"
    packet = {
        "schema": TEST_RESULT_SCHEMA_ID,
        "status": status,
        "generated_utc": generated_utc(),
        "candidate": candidate.identity(),
        "execution_provenance": _execution_identity(candidate),
        "validation_id": validation["result"]["validation_id"],
        "execution": execution,
        "effects": effects(writes=True, executes=True),
        "non_claims": list(FSWDK_NON_CLAIMS),
    }
    path = output / "fsw_test_results.json"
    packet["artifacts"] = [_declared_packet_artifact(path, artifact_id="test_results")]
    _write_json(path, packet, writer=binding, logical_root=output)
    return packet


def _materialize_scenario(
    candidate: Any,
    output: Path,
    *,
    writer: _OutputBinding | Any | None = None,
) -> tuple[Path, Path, Path | None]:
    binding = _bind_writer(writer, output)
    physical_output = binding.physical(output) if binding is not None else output
    if binding is not None:
        binding.ensure_directory(output)
    source = candidate.verification.smoke_case
    if source is None:
        raise ValueError("Candidate does not declare a smoke scenario.")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Candidate smoke scenario must be a YAML object.")
    raw.setdefault("outputs", {})["output_dir"] = str(output / "run")
    raw["outputs"].setdefault("review", {"enabled": True, "detail": "standard"})
    generated = output / "materialized_smoke.yaml"
    serialized = yaml.safe_dump(raw, sort_keys=False).encode("utf-8")
    if binding is not None:
        binding.write_bytes(generated, serialized)
    else:
        generated.write_bytes(serialized)
    if binding is None:
        return generated, generated, None
    execution_raw = yaml.safe_load(serialized)
    execution_raw["outputs"]["output_dir"] = str(physical_output / "run")
    with tempfile.NamedTemporaryFile(
        "wb", prefix=".oel-fswdk-smoke-", suffix=".yaml", dir=tempfile.gettempdir(), delete=False
    ) as handle:
        handle.write(yaml.safe_dump(execution_raw, sort_keys=False).encode("utf-8"))
        execution_path = Path(handle.name)
    return generated, execution_path, execution_path


def run_smoke(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    validation_id: str | None = None,
    cancellation_check: Callable[[], None] | None = None,
    writer: _OutputBinding | Any | None = None,
) -> dict[str, Any]:
    candidate, validation = _require_validated(
        manifest,
        workspace_root=workspace_root,
        validation_id=validation_id,
    )
    if candidate.verification.smoke_case is None:
        raise ValueError("Candidate does not declare a smoke scenario.")
    resource_estimate = asdict(
        estimate_resource_requirements(load_simulation_yaml(candidate.verification.smoke_case))
    )
    resource_estimate["action"] = "refuse" if resource_estimate["risk"] == "unsafe" else "proceed"
    if resource_estimate["action"] == "refuse":
        raise ValueError("Candidate smoke failed the OEL resource-safety preflight.")
    output = _new_output(
        output_dir,
        candidate_id=candidate.candidate_id,
        operation="smoke",
        workspace_root=candidate.workspace_root,
        writer=writer,
    )
    binding = _bind_writer(writer, output)
    scenario, physical_scenario, execution_cleanup = _materialize_scenario(candidate, output, writer=binding)
    physical_output = binding.physical(output) if binding is not None else output
    workspace = SimulationWorkspace(
        workspace_root=candidate.workspace_root,
        read_roots=(candidate.workspace_root, physical_output, physical_scenario.parent),
        write_roots=(candidate.workspace_root, physical_output),
    )
    sys.path.insert(0, str(candidate.workspace_root))
    try:
        if candidate.source.entrypoint is not None:
            clear_candidate_imports(candidate.source.entrypoint.module, source_root=candidate.source.root)
        # The engine receives a private physical staging config and writes its
        # derived run tree below that staging root.  Suspend the logical writer
        # only for this path-based engine call; transaction publication remains
        # descriptor-bound after the run completes.
        with bind_output_writer(None):
            result = workspace.run(
                physical_scenario,
                step_callback=(None if cancellation_check is None else lambda *_args: cancellation_check()),
            )
    finally:
        sys.path.remove(str(candidate.workspace_root))
        if execution_cleanup is not None:
            execution_cleanup.unlink(missing_ok=True)
    _require_candidate_current(candidate)
    run_output = Path(str(result.payload.get("output_dir") or physical_output / "run"))
    summary = _logicalize(dict(result.summary), binding)
    run_output = _logicalize(run_output, binding)
    terminated_early = bool(summary.get("terminated_early", False))
    status = "failed" if terminated_early else "passed"
    if binding is not None:
        with binding.bound():
            scenario_sha256 = sha256_output(scenario)
    else:
        scenario_sha256 = sha256_file(physical_scenario)
    packet = {
        "schema": RUN_MANIFEST_SCHEMA_ID,
        "status": status,
        "generated_utc": generated_utc(),
        "candidate": candidate.identity(),
        "execution_provenance": _execution_identity(candidate),
        "validation_id": validation["result"]["validation_id"],
        "scenario_path": str(scenario),
        "scenario_sha256": scenario_sha256,
        "run_output_dir": str(run_output),
        "summary": summary,
        "resource_plan": resource_estimate,
        "effects": effects(writes=True, executes=True),
        "non_claims": list(FSWDK_NON_CLAIMS),
    }
    path = output / "fsw_run_manifest.json"
    packet["artifacts"] = [_declared_packet_artifact(path, artifact_id="run_manifest")]
    _write_json(path, packet, writer=binding, logical_root=output)
    return packet


def _materialize_bench(
    candidate: Any,
    output: Path,
    *,
    writer: _OutputBinding | Any | None = None,
) -> tuple[Path, Path]:
    binding = _bind_writer(writer, output)
    physical_output = binding.physical(output) if binding is not None else output
    if binding is not None:
        binding.ensure_directory(output)
    source = candidate.verification.bench_suite
    if source is None:
        raise ValueError("Candidate does not declare a Controller Bench suite.")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Candidate Controller Bench suite must be a YAML object.")
    for case in list(raw.get("cases", []) or []):
        if not isinstance(case, dict) or not case.get("config_path"):
            continue
        case_path = Path(str(case["config_path"]))
        if not case_path.is_absolute():
            case["config_path"] = str((source.parent / case_path).resolve())
    raw["output_dir"] = str(output / "bench")
    generated = output / "materialized_bench.yaml"
    serialized = yaml.safe_dump(raw, sort_keys=False).encode("utf-8")
    if binding is not None:
        binding.write_bytes(generated, serialized)
    else:
        generated.write_bytes(serialized)
    if binding is None:
        return generated, generated, None
    execution_raw = yaml.safe_load(serialized)
    execution_raw["output_dir"] = str(physical_output / "bench")
    with tempfile.NamedTemporaryFile(
        "wb", prefix=".oel-fswdk-bench-", suffix=".yaml", dir=tempfile.gettempdir(), delete=False
    ) as handle:
        handle.write(yaml.safe_dump(execution_raw, sort_keys=False).encode("utf-8"))
        execution_path = Path(handle.name)
    return generated, execution_path, execution_path


def run_bench(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    validation_id: str | None = None,
    show_progress: bool = False,
    cancellation_check: Callable[[], None] | None = None,
    writer: _OutputBinding | Any | None = None,
) -> dict[str, Any]:
    candidate, validation = _require_validated(
        manifest,
        workspace_root=workspace_root,
        validation_id=validation_id,
    )
    if candidate.verification.bench_suite is None:
        raise ValueError("Candidate does not declare a Controller Bench suite.")
    run_count = _bounded_bench_run_count(candidate.verification.bench_suite)
    if run_count > 32:
        raise ValueError("FSWDK Controller Bench plans are limited to 32 candidate runs.")
    output = _new_output(
        output_dir,
        candidate_id=candidate.candidate_id,
        operation="bench",
        workspace_root=candidate.workspace_root,
        writer=writer,
    )
    binding = _bind_writer(writer, output)
    suite, physical_suite, execution_cleanup = _materialize_bench(candidate, output, writer=binding)
    sys.path.insert(0, str(candidate.workspace_root))
    try:
        with bind_output_writer(None):
            preflight = validate_controller_bench_config(physical_suite)
            if not preflight.get("passed", False):
                raise ValueError("Controller Bench preflight failed: " + "; ".join(preflight.get("errors", [])))
            summary = run_controller_bench(
                physical_suite,
                show_progress=show_progress,
                step_callback=(None if cancellation_check is None else lambda *_args: cancellation_check()),
            )
    finally:
        sys.path.remove(str(candidate.workspace_root))
        if execution_cleanup is not None:
            execution_cleanup.unlink(missing_ok=True)
    _require_candidate_current(candidate)
    bench_output = Path(str(summary.get("output_dir") or (binding.physical(output) if binding else output) / "bench"))
    with bind_output_writer(None):
        evidence = inspect_controller_bench_output(bench_output, write_packet=True)
    evidence = _logicalize(evidence, binding)
    logical_bench_output = _logicalize(bench_output, binding)
    packet = {
        "schema": AGENT_EVIDENCE_PACKET_SCHEMA_ID,
        "status": "passed"
        if int(dict(summary.get("constraint_summary", {}) or {}).get("failed_run_count", 0) or 0) == 0
        else "failed",
        "generated_utc": generated_utc(),
        "candidate": candidate.identity(),
        "execution_provenance": _execution_identity(candidate),
        "validation_id": validation["result"]["validation_id"],
        "bench_output_dir": str(logical_bench_output),
        "summary": dict(evidence.get("evidence_summary", {}) or {}),
        "resource_plan": {"workers": 1, "planned_runs": run_count, "max_runs": 32, "action": "proceed"},
        "source_evidence_packet": str(evidence.get("agent_evidence_packet", "")),
        "effects": effects(writes=True, executes=True),
        "non_claims": list(FSWDK_NON_CLAIMS),
    }
    path = output / "fsw_agent_evidence_packet.json"
    packet["artifacts"] = [_declared_packet_artifact(path, artifact_id="fswdk_agent_evidence")]
    _write_json(path, packet, writer=binding, logical_root=output)
    return packet


def _bounded_bench_run_count(path: Path) -> int:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Controller Bench plan must be an object.")
    parallel = dict(raw.get("parallel", {}) or {})
    if bool(parallel.get("enabled", False)):
        raise ValueError("FSWDK Controller Bench execution is serial; parallel bench plans are not supported.")
    cases = list(raw.get("cases", []) or [])
    variants = list(raw.get("variants", []) or [])
    variant_runs = 0
    for variant in variants:
        combinations = 1
        for values in dict(dict(variant or {}).get("sweep", {}) or {}).values():
            combinations *= max(1, len(values) if isinstance(values, list) else 1)
        variant_runs += combinations
    envelope = dict(raw.get("envelope", {}) or {})
    generated_cases = 0
    if bool(envelope.get("enabled", False)):
        base_count = len(list(envelope.get("base_cases", []) or [])) or len(cases)
        combinations = 1
        for variation in list(envelope.get("variations", []) or []):
            combinations *= max(1, len(list(dict(variation or {}).get("values", []) or [])))
        generated_cases = base_count * combinations
        maximum = int(envelope.get("max_cases", 0) or 0)
        if maximum > 0:
            generated_cases = min(generated_cases, maximum)
    total_cases = len(cases) + generated_cases
    optimization = dict(raw.get("optimization", {}) or {})
    if bool(optimization.get("enabled", False)):
        particles = int(optimization.get("particles", 0) or 0)
        iterations = int(optimization.get("iterations", 0) or 0)
        return total_cases * particles * (iterations + 1)
    return total_cases * variant_runs


def promote_baseline(
    manifest: str | Path,
    evidence_packet: str | Path,
    name: str,
    *,
    workspace_root: str | Path = ROOT,
    registry_dir: str | Path | None = None,
    approved: bool = False,
    writer: _OutputBinding | Any | None = None,
) -> dict[str, Any]:
    if not approved:
        raise PermissionError("Baseline promotion requires explicit approved=True authority.")
    candidate = load_candidate(manifest, workspace_root=workspace_root)
    evidence_path = Path(evidence_packet).expanduser().resolve()
    evidence_path.relative_to(candidate.workspace_root)
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    slug = name.strip().lower().replace(" ", "_")
    if not slug or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in slug):
        raise ValueError("Baseline name must use lowercase letters, digits, underscores, or hyphens.")
    target_root = (
        Path(registry_dir).resolve() if registry_dir is not None else candidate.manifest_path.parent / "baselines"
    )
    target_root.relative_to(candidate.workspace_root)
    target = target_root / f"{slug}.json"
    if target.exists():
        raise FileExistsError("Immutable FSWDK baselines may not be overwritten.")
    payload = {
        "schema": "oel.fswdk.baseline.v1",
        "name": slug,
        "promoted_utc": generated_utc(),
        "candidate": candidate.identity(),
        "execution_provenance": _execution_identity(candidate),
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "promotion_approved": True,
        "non_claims": list(FSWDK_NON_CLAIMS),
    }
    _write_json(target, payload, writer=writer, logical_root=target_root)
    return {**payload, "baseline_path": str(target)}


def compare_evidence(
    baseline: str | Path,
    candidate_packet: str | Path,
    *,
    output_path: str | Path | None = None,
    writer: _OutputBinding | Any | None = None,
) -> dict[str, Any]:
    baseline_path = Path(baseline).expanduser().resolve()
    baseline_raw = json.loads(baseline_path.read_text(encoding="utf-8"))
    packet_path = Path(candidate_packet).expanduser().resolve()
    packet_raw = json.loads(packet_path.read_text(encoding="utf-8"))
    expected_hash = str(baseline_raw.get("evidence_sha256", ""))
    baseline_evidence = Path(str(baseline_raw.get("evidence_path", ""))).expanduser().resolve()
    baseline_current = baseline_evidence.is_file() and sha256_file(baseline_evidence) == expected_hash
    baseline_candidate = dict(baseline_raw.get("candidate", {}) or {})
    packet_candidate = dict(packet_raw.get("candidate", {}) or {})
    comparability_fields = (
        "candidate_id",
        "onboard_contract",
        "hardware_profile",
        "task_period_s",
        "verification_sha256",
    )
    mismatched_fields = [
        field for field in comparability_fields if baseline_candidate.get(field) != packet_candidate.get(field)
    ]
    comparable = baseline_current and not mismatched_fields
    payload = {
        "schema": COMPARISON_PACKET_SCHEMA_ID,
        "status": "passed" if comparable else "invalid",
        "generated_utc": generated_utc(),
        "comparable": comparable,
        "comparability_fields": list(comparability_fields),
        "mismatched_fields": mismatched_fields,
        "baseline_current": baseline_current,
        "baseline": baseline_raw,
        "candidate_packet": str(packet_path),
        "candidate_packet_sha256": sha256_file(packet_path),
        "candidate_status": packet_raw.get("status"),
        "non_claims": list(FSWDK_NON_CLAIMS),
    }
    if output_path is not None:
        target = Path(output_path).expanduser().resolve()
        _write_json(target, payload, writer=writer, logical_root=target.parent)
    return payload


def run_qualification(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    validation_id: str | None = None,
    include_bench: bool = True,
    cancellation_check: Callable[[], None] | None = None,
    writer: _OutputBinding | Any | None = None,
) -> dict[str, Any]:
    candidate, validation = _require_validated(
        manifest,
        workspace_root=workspace_root,
        validation_id=validation_id,
    )
    spec_path = candidate.verification.qualification_spec
    if spec_path is None:
        raise ValueError("Candidate does not declare a qualification specification.")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("Qualification specification must be an object.")
    output = _new_output(
        output_dir,
        candidate_id=candidate.candidate_id,
        operation="qualification",
        workspace_root=candidate.workspace_root,
        writer=writer,
    )
    binding = _bind_writer(writer, output)
    gate_results: list[dict[str, Any]] = []
    requirement_ids = {str(item["id"]) for item in list(spec.get("requirements", []) or [])}
    for gate in list(spec.get("gates", []) or []):
        if cancellation_check is not None:
            cancellation_check()
        gate_id = str(gate.get("id", ""))
        kind = str(gate.get("kind", ""))
        required = bool(gate.get("required", True))
        gate_requirement_ids = [str(item) for item in list(gate.get("requirement_ids", []) or [])]
        try:
            if kind == "contract_test":
                result = run_contract_tests(
                    manifest,
                    workspace_root=workspace_root,
                    output_dir=output / gate_id,
                    validation_id=validation["result"]["validation_id"],
                    cancellation_check=cancellation_check,
                    writer=binding,
                )
            elif kind == "smoke":
                result = run_smoke(
                    manifest,
                    workspace_root=workspace_root,
                    output_dir=output / gate_id,
                    validation_id=validation["result"]["validation_id"],
                    cancellation_check=cancellation_check,
                    writer=binding,
                )
            elif kind == "controller_bench" and include_bench:
                result = run_bench(
                    manifest,
                    workspace_root=workspace_root,
                    output_dir=output / gate_id,
                    validation_id=validation["result"]["validation_id"],
                    cancellation_check=cancellation_check,
                    writer=binding,
                )
            elif kind == "controller_bench":
                result = {"status": "incomplete", "reason": "bench gate was explicitly skipped"}
            elif kind == "artifact":
                citations = [citation for item in gate_results for citation in item["evidence_citations"]]
                current = [
                    citation
                    for citation in citations
                    if (
                        (binding.physical(Path(str(citation["path"]))) if binding else Path(str(citation["path"]))).is_file()
                        and sha256_file(
                            binding.physical(Path(str(citation["path"])))
                            if binding
                            else Path(str(citation["path"]))
                        )
                        == str(citation["sha256"])
                    )
                ]
                result = {
                    "status": "passed" if citations and len(current) == len(citations) else "failed",
                    "verified_artifact_count": len(current),
                    "declared_artifact_count": len(citations),
                    "artifacts": current,
                }
            else:
                result = {"status": "invalid", "reason": f"unsupported qualification gate kind: {kind}"}
        except Exception as exc:
            if cancellation_check is not None:
                cancellation_check()
            result = {"status": "failed", "error": str(exc)}
        citations = _qualification_citations(result, binding=binding)
        gate_results.append(
            {
                "id": gate_id,
                "kind": kind,
                "required": required,
                "requirement_ids": gate_requirement_ids,
                "status": result.get("status"),
                "evidence_citations": citations,
                "result": result,
            }
        )
    failed = [item for item in gate_results if item["required"] and item["status"] != "passed"]
    invalid = [item for item in failed if item["status"] == "invalid"]
    insufficient = [item for item in failed if item["status"] == "incomplete"]
    verdict = "invalid" if invalid else "insufficient_evidence" if insufficient else "fail" if failed else "pass"
    requirements = []
    for requirement in list(spec.get("requirements", []) or []):
        requirement_id = str(requirement["id"])
        linked = [item for item in gate_results if requirement_id in item["requirement_ids"]]
        requirements.append(
            {
                **dict(requirement),
                "verdict": "pass" if linked and all(item["status"] == "passed" for item in linked) else "fail",
                "gate_ids": [item["id"] for item in linked],
                "evidence_citations": [citation for item in linked for citation in item["evidence_citations"]],
            }
        )
    unknown_requirements = sorted(
        {
            requirement_id
            for item in gate_results
            for requirement_id in item["requirement_ids"]
            if requirement_id not in requirement_ids
        }
    )
    if unknown_requirements:
        verdict = "invalid"
    _require_candidate_current(candidate)
    junit_path = output / "fsw_qualification_junit.xml"
    _write_qualification_junit(junit_path, gate_results, writer=binding, logical_root=output)
    traceability_path = output / "fsw_qualification_traceability.md"
    traceability_path = _write_qualification_traceability(
        traceability_path,
        candidate_id=candidate.candidate_id,
        verdict=verdict,
        requirements=requirements,
        writer=binding,
        logical_root=output,
    )
    packet = {
        "schema": QUALIFICATION_PACKET_SCHEMA_ID,
        "status": "passed" if verdict == "pass" else "failed",
        "verdict": verdict,
        "generated_utc": generated_utc(),
        "candidate": candidate.identity(),
        "execution_provenance": _execution_identity(candidate),
        "validation_id": validation["result"]["validation_id"],
        "specification_path": str(spec_path),
        "specification_sha256": sha256_file(spec_path),
        "envelope_path": str(candidate.declared_envelope or ""),
        "envelope_sha256": sha256_file(candidate.declared_envelope) if candidate.declared_envelope else "",
        "compatibility": {
            "candidate_identity_matches": str(spec.get("candidate_id")) == candidate.candidate_id,
            "onboard_contract": candidate.onboard_contract,
            "hardware_profile": candidate.hardware_profile,
            "task_period_s": candidate.task_period_s,
        },
        "evidence_freshness": {
            "current": True,
            "max_age_s": float(spec.get("max_evidence_age_s", 86400.0)),
            "semantics": "All gate evidence was generated in this immutable qualification run.",
        },
        "requirements": requirements,
        "gates": gate_results,
        "failed_required_gate_count": len(failed),
        "unknown_requirement_ids": unknown_requirements,
        "junit_path": str(junit_path),
        "traceability_path": str(traceability_path),
        "promotion_authorized": False,
        "effects": effects(writes=True, executes=True),
        "non_claims": list(FSWDK_NON_CLAIMS),
    }
    path = output / "fsw_qualification_packet.json"
    packet["artifacts"] = [
        _declared_packet_artifact(path, artifact_id="qualification_packet"),
        _artifact(junit_path, artifact_id="qualification_junit", writer=binding, logical_root=output),
        _artifact(traceability_path, artifact_id="qualification_traceability", writer=binding, logical_root=output),
    ]
    _write_json(path, packet, writer=binding, logical_root=output)
    return packet


def _qualification_citations(
    result: Mapping[str, Any], *, binding: _OutputBinding | None = None
) -> list[dict[str, Any]]:
    citations = []
    for artifact in list(result.get("artifacts", []) or []):
        logical_path = Path(str(artifact.get("path", ""))).expanduser().resolve()
        path = binding.physical(logical_path) if binding else logical_path
        if not path.is_file():
            continue
        citations.append(
            {
                "artifact_id": str(artifact.get("artifact_id", logical_path.name)),
                "path": str(logical_path),
                "sha256": sha256_file(path),
            }
        )
    return citations


def _write_qualification_junit(
    path: Path,
    gates: list[dict[str, Any]],
    *,
    writer: _OutputBinding | Any | None = None,
    logical_root: Path | None = None,
) -> Path:
    failures = sum(1 for gate in gates if gate["required"] and gate["status"] != "passed")
    suite = ElementTree.Element(
        "testsuite",
        name="oel.fswdk.qualification",
        tests=str(len(gates)),
        failures=str(failures),
        errors="0",
    )
    for gate in gates:
        case = ElementTree.SubElement(suite, "testcase", name=str(gate["id"]), classname=str(gate["kind"]))
        if gate["required"] and gate["status"] != "passed":
            failure = ElementTree.SubElement(case, "failure", message=f"gate status: {gate['status']}")
            failure.text = json.dumps(gate["result"], sort_keys=True)
    if writer is not None:
        buffer = io.BytesIO()
        ElementTree.ElementTree(suite).write(buffer, encoding="utf-8", xml_declaration=True)
        binding = _bind_writer(writer, logical_root or path.parent)
        assert binding is not None
        try:
            binding.write_bytes(path, buffer.getvalue())
        except SecureWriteConflict:
            binding.update_existing_bytes(path, buffer.getvalue())
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        ElementTree.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _write_qualification_traceability(
    path: Path,
    *,
    candidate_id: str,
    verdict: str,
    requirements: list[dict[str, Any]],
    writer: _OutputBinding | Any | None = None,
    logical_root: Path | None = None,
) -> Path:
    lines = [
        f"# FSWDK Qualification Traceability: {candidate_id}",
        "",
        f"Verdict: `{verdict}`",
        "",
        "| Requirement | Verdict | Verification gates | Evidence artifacts |",
        "| --- | --- | --- | --- |",
    ]
    for requirement in requirements:
        artifacts = ", ".join(str(item["artifact_id"]) for item in requirement["evidence_citations"]) or "none"
        lines.append(
            f"| {requirement['id']} | {requirement['verdict']} | "
            f"{', '.join(requirement['gate_ids']) or 'none'} | {artifacts} |"
        )
    lines.extend(
        [
            "",
            "This is simulation-verification evidence inside the declared envelope, not flight qualification or certification.",
            "",
        ]
    )
    payload = "\n".join(lines).encode("utf-8")
    if writer is not None:
        binding = _bind_writer(writer, logical_root or path.parent)
        assert binding is not None
        try:
            binding.write_bytes(path, payload)
        except SecureWriteConflict:
            binding.update_existing_bytes(path, payload)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return path


def inspect_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        candidates = sorted(source.rglob("fsw_*packet.json")) + sorted(source.rglob("fsw_run_manifest.json"))
        if not candidates:
            raise FileNotFoundError("No FSWDK evidence packet was found in the requested directory.")
        source = candidates[-1]
    raw = json.loads(source.read_text(encoding="utf-8"))
    return {
        "schema": AGENT_EVIDENCE_PACKET_SCHEMA_ID,
        "status": raw.get("status", "incomplete"),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "candidate": dict(raw.get("candidate", {}) or {}),
        "facts": {
            "schema": raw.get("schema"),
            "status": raw.get("status"),
            "failed_required_gate_count": raw.get("failed_required_gate_count"),
            "summary": dict(raw.get("summary", {}) or {}),
        },
        "complete": raw.get("status") in {"passed", "failed"},
        "promotion_authorized": False,
        "non_claims": list(FSWDK_NON_CLAIMS),
    }


def prepare_package(
    manifest: str | Path,
    evidence_dir: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    writer: _OutputBinding | Any | None = None,
) -> dict[str, Any]:
    candidate = load_candidate(manifest, workspace_root=workspace_root)
    evidence_root = Path(evidence_dir).expanduser().resolve()
    evidence_root.relative_to(candidate.workspace_root)
    if not evidence_root.is_dir():
        raise FileNotFoundError(evidence_root)
    output = _new_output(
        output_dir,
        candidate_id=candidate.candidate_id,
        operation="package",
        workspace_root=candidate.workspace_root,
        writer=writer,
    )
    binding = _bind_writer(writer, output)
    try:
        output.relative_to(evidence_root)
    except ValueError:
        pass
    else:
        raise ValueError("FSWDK package output may not be nested inside its source evidence directory.")
    allowed_suffixes = {
        ".json",
        ".md",
        ".csv",
        ".png",
        ".svg",
        ".pdf",
        ".sqlite",
        ".yaml",
        ".yml",
        ".xml",
    }
    source_files = []
    for path in sorted(evidence_root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in allowed_suffixes:
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(evidence_root)
        except ValueError as exc:
            raise PermissionError(
                "Evidence packages may not traverse symbolic links outside the evidence root."
            ) from exc
        source_files.append(resolved)
    archive = output / f"{candidate.candidate_id}_fswdk_evidence.zip"
    physical_archive = binding.physical(archive) if binding else archive
    rows: list[dict[str, Any]] = []
    captured_sources: list[tuple[Path, str, int]] = []
    with zipfile.ZipFile(physical_archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in source_files:
            digest = hashlib.sha256()
            size = 0
            member = str(Path("evidence") / path.relative_to(evidence_root))
            with path.open("rb") as source, bundle.open(member, "w") as archived:
                while chunk := source.read(1024 * 1024):
                    archived.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            source_digest = digest.hexdigest()
            if not _package_source_unchanged(path, source_digest, size):
                raise RuntimeError(f"Evidence source changed while creating FSWDK package: {path}")
            rows.append({"path": str(path.relative_to(evidence_root)), "sha256": source_digest, "size_bytes": size})
            captured_sources.append((path, source_digest, size))
        for path, digest, size in captured_sources:
            if not _package_source_unchanged(path, digest, size):
                raise RuntimeError(f"Evidence source changed while creating FSWDK package: {path}")
        manifest_payload = {
            "schema": PACKAGE_MANIFEST_SCHEMA_ID,
            "status": "passed",
            "generated_utc": generated_utc(),
            "candidate": candidate.identity(),
            "execution_provenance": _execution_identity(candidate),
            "handling": dict(candidate.handling),
            "source_evidence_root": str(evidence_root),
            "source_included": False,
            "flight_binary_included": False,
            "external_release_authorized": False,
            "files": rows,
            "archive_name": f"{candidate.candidate_id}_fswdk_evidence.zip",
            "archive_digest_location": "external_manifest",
            "non_claims": list(FSWDK_NON_CLAIMS),
        }
        embedded_manifest = (json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        bundle.writestr("fsw_package_manifest.json", embedded_manifest)
    package_manifest = output / "fsw_package_manifest.json"
    manifest_payload["package_path"] = str(archive)
    manifest_payload["package_sha256"] = sha256_file(physical_archive)
    _write_json(package_manifest, manifest_payload, writer=binding, logical_root=output)
    return manifest_payload


def _package_source_unchanged(path: Path, digest: str, size: int) -> bool:
    """Check that the source still matches the bytes written to the archive."""

    try:
        return path.stat().st_size == size and sha256_file(path) == digest
    except OSError:
        return False


def verify_package(package_manifest: str | Path) -> dict[str, Any]:
    manifest_path = Path(package_manifest).expanduser().resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema") != PACKAGE_MANIFEST_SCHEMA_ID:
        raise ValueError("Not an FSWDK package manifest.")
    archive = Path(str(raw.get("package_path", ""))).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError("The FSWDK package archive was not found.")
    archive_hash_ok = sha256_file(archive) == str(raw.get("package_sha256", ""))
    verified_rows = []
    with zipfile.ZipFile(archive, "r") as bundle:
        member_names = bundle.namelist()
        archive_member_names_unique = len(member_names) == len(set(member_names))
        names = set(member_names)
        rows = list(raw.get("files", []) or [])
        relative_rows = []
        for row in rows:
            relative = Path(str(row.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("FSWDK package manifest contains an unsafe artifact path.")
            relative_rows.append(relative)
        manifest_row_paths = [str(relative) for relative in relative_rows]
        manifest_rows_unique = len(manifest_row_paths) == len(set(manifest_row_paths))
        expected_names = {"fsw_package_manifest.json"} | {
            str(Path("evidence") / relative) for relative in relative_rows
        }
        exact_member_set = archive_member_names_unique and names == expected_names
        try:
            embedded_manifest = json.loads(bundle.read("fsw_package_manifest.json").decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
            embedded_manifest = None
        external_common = {key: value for key, value in raw.items() if key not in {"package_path", "package_sha256"}}
        embedded_manifest_matches = embedded_manifest == external_common
        for row, relative in zip(rows, relative_rows, strict=True):
            member = str(Path("evidence") / relative)
            present = member in names
            digest = hashlib.sha256(bundle.read(member)).hexdigest() if present else ""
            verified_rows.append(
                {
                    "path": str(relative),
                    "present": present,
                    "sha256_matches": present and digest == str(row.get("sha256", "")),
                }
            )
    passed = (
        archive_hash_ok
        and exact_member_set
        and manifest_rows_unique
        and embedded_manifest_matches
        and all(item["present"] and item["sha256_matches"] for item in verified_rows)
    )
    return {
        "schema": "oel.fswdk.package_verification.v1",
        "status": "passed" if passed else "failed",
        "manifest_path": str(manifest_path),
        "archive_path": str(archive),
        "archive_sha256_matches": archive_hash_ok,
        "archive_member_set_matches": exact_member_set,
        "archive_member_names_unique": archive_member_names_unique,
        "manifest_rows_unique": manifest_rows_unique,
        "embedded_manifest_matches": embedded_manifest_matches,
        "verified_files": verified_rows,
        "source_included": bool(raw.get("source_included", False)),
        "flight_binary_included": bool(raw.get("flight_binary_included", False)),
        "external_release_authorized": False,
        "non_claims": list(FSWDK_NON_CLAIMS),
    }
