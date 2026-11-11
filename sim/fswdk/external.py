"""External-process and cFS integration for the FSWDK candidate lifecycle."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from sim.fsw_authoring.native_transport import start_local_process, stop_local_process

from .candidate import ROOT, load_candidate
from .contracts import FSWDK_NON_CLAIMS, effects, generated_utc, sha256_file
from .services import _artifact, _bind_writer, _new_output, _require_candidate_current, _write_json, run_smoke

SUPPORTED_CFS_PROFILE = {
    "profile_id": "oel.cfs_sil.reference_adcs.v2",
    "cfs_release": "v7.0.1",
    "target": "native_linux",
    "wire_protocol_version": 2,
    "timing_mode": "deterministic_step",
    "transport_scope": "loopback_only",
}
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_CFS_ADAPTER_MODULE = "integrations.cfs_sil.v2_codec"
_CFS_ADAPTER_CLASS = "CfsV2UdpFlightSoftwareAdapter"


def validate_external_compatibility(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Validate an external candidate without starting or importing it."""

    candidate = load_candidate(manifest, workspace_root=workspace_root)
    if candidate.kind not in {"external_process", "cfs"}:
        raise ValueError("External compatibility validation requires an external_process or cfs candidate.")
    artifact = candidate.source.artifact
    if artifact is None or not artifact.is_file():
        raise FileNotFoundError("The candidate executable artifact was not found.")
    if artifact.is_symlink():
        raise PermissionError("External candidate artifacts may not be symbolic links.")
    executable = os.access(artifact, os.X_OK)
    if not executable:
        raise PermissionError("The external candidate artifact is not executable.")
    hosts = _declared_transport_hosts(candidate.verification.smoke_case)
    non_loopback = sorted(host for host in hosts if host not in _LOOPBACK_HOSTS)
    if non_loopback:
        raise PermissionError(
            "FSWDK external execution is loopback-only; non-loopback host declarations were found: "
            + ", ".join(non_loopback)
        )
    if candidate.kind == "cfs":
        # The maintained cFS v2 adapter binds AF_INET sockets and deliberately
        # accepts only 127.0.0.1/localhost.  Treating ::1 as compatible here
        # would let validation report ``ready`` and defer a deterministic
        # endpoint rejection until the simulation starts.
        unsupported_cfs_hosts = sorted(host for host in hosts if host not in {"127.0.0.1", "localhost"})
        if unsupported_cfs_hosts:
            raise PermissionError(
                "The maintained cFS v2 profile is IPv4 loopback-only; unsupported host declarations were found: "
                + ", ".join(unsupported_cfs_hosts)
            )
        if candidate.onboard_contract != "oel.cfs_sil.v2":
            raise ValueError("The supported cFS profile requires onboard contract oel.cfs_sil.v2.")
        if candidate.hardware_profile != "hardware.ideal_wrench.v1":
            raise ValueError("The supported cFS reference profile requires hardware.ideal_wrench.v1.")
        sections = _declared_flight_software_sections(candidate.verification.smoke_case)
        compatible = [
            section
            for section in sections
            if section.get("module") == _CFS_ADAPTER_MODULE and section.get("class_name") == _CFS_ADAPTER_CLASS
        ]
        if len(compatible) != 1:
            raise ValueError("The supported cFS smoke must declare exactly one OEL cFS v2 adapter.")
        if abs(float(compatible[0].get("task_period_s", 0.0)) - candidate.task_period_s) > 1.0e-12:
            raise ValueError("The cFS candidate cadence does not match its smoke adapter cadence.")
    compatibility = (
        dict(SUPPORTED_CFS_PROFILE)
        if candidate.kind == "cfs"
        else {
            "profile_id": "oel.external_process.loopback.v1",
            "target": "local_process",
            "timing_mode": "candidate_declared",
            "transport_scope": "loopback_only",
        }
    )
    return {
        "schema": "oel.fswdk.external_compatibility.v1",
        "status": "ready",
        "candidate": candidate.identity(),
        "artifact_path": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "artifact_executable": executable,
        "declared_transport_hosts": sorted(hosts),
        "compatibility": compatibility,
        "execution_authorized": False,
        "effects": effects(),
        "non_claims": [
            *FSWDK_NON_CLAIMS,
            "Compatibility validation does not build, start, certify, or deploy the external candidate.",
        ],
    }


def run_external_smoke(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    validation_id: str | None = None,
    startup_wait_s: float = 2.0,
    writer: Any | None = None,
) -> dict[str, Any]:
    """Start one trusted local executable and run its declared OEL smoke case."""

    compatibility = validate_external_compatibility(manifest, workspace_root=workspace_root)
    candidate = load_candidate(manifest, workspace_root=workspace_root)
    wait_s = float(startup_wait_s)
    if not 0.0 <= wait_s <= 15.0:
        raise ValueError("startup_wait_s must be between 0 and 15 seconds.")
    output = _new_output(
        output_dir,
        candidate_id=candidate.candidate_id,
        operation="external_smoke",
        workspace_root=candidate.workspace_root,
        writer=writer,
    )
    binding = _bind_writer(writer, output)
    artifact = candidate.source.artifact
    assert artifact is not None
    log_path = output / "external_process.log"
    physical_log_path = binding.physical(log_path) if binding is not None else log_path
    process: subprocess.Popen[str] | None = None
    smoke: dict[str, Any] = {"status": "incomplete"}
    failure: dict[str, str] | None = None
    try:
        _require_declared_endpoint_available(candidate.verification.smoke_case)
        # Recheck the content-bound candidate immediately before launch.  The
        # external process is trusted-host code rather than a sandbox, but this
        # closes the deterministic window between compatibility validation and
        # Popen for edits that persist until launch.  The post-run check below
        # remains authoritative for edits during execution.
        _require_candidate_current(candidate)
        physical_log_path.parent.mkdir(parents=True, exist_ok=True)
        with physical_log_path.open("w", encoding="utf-8") as log:
            process = start_local_process(
                artifact,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                # The candidate process is allowed to resolve resources beside
                # its declared artifact.  Keep that contract independent of
                # the operation's logical/physical output binding.
                cwd=artifact.parent,
            )
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"External candidate exited during startup with code {process.returncode}.")
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            smoke = run_smoke(
                manifest,
                workspace_root=candidate.workspace_root,
                output_dir=output / "smoke",
                validation_id=validation_id,
                writer=binding,
            )
            if process.poll() is not None:
                raise RuntimeError(f"External candidate exited during smoke with code {process.returncode}.")
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        smoke = {"status": "failed", "error": failure}
    finally:
        _stop_process(process)
    try:
        _require_candidate_current(candidate)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        smoke = {"status": "failed", "error": failure}
    packet = {
        "schema": "oel.fswdk.external_run.v1",
        "status": smoke["status"],
        "generated_utc": generated_utc(),
        "candidate": candidate.identity(),
        "compatibility": compatibility["compatibility"],
        "artifact_sha256": compatibility["artifact_sha256"],
        "process_returncode": process.returncode if process is not None else None,
        "process_pid": process.pid if process is not None else None,
        "smoke": smoke,
        "effects": effects(writes=True, executes=True, external=False),
        "external_communication": False,
        "failure": failure,
        "non_claims": [
            *FSWDK_NON_CLAIMS,
            "Execution evidence applies only to the pinned local artifact, interface, scenario, and host environment.",
        ],
    }
    path = output / "fsw_external_run_packet.json"
    packet["artifacts"] = [
        {"artifact_id": "external_run_packet", "path": str(path), "exists": True, "digest_scope": "external"},
        _artifact(log_path, artifact_id="external_process_log", writer=binding, logical_root=output),
    ]
    _write_json(path, packet, writer=binding, logical_root=output)
    return packet


def _declared_transport_hosts(smoke_case: Path | None) -> set[str]:
    if smoke_case is None:
        raise ValueError("External candidates must declare a smoke scenario.")
    raw = yaml.safe_load(smoke_case.read_text(encoding="utf-8"))
    hosts: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"bind_host", "cfs_host", "remote_host"} and isinstance(child, str):
                    hosts.add(child.strip().lower())
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(raw)
    return hosts


def _require_declared_endpoint_available(smoke_case: Path | None) -> None:
    """Reject a stale responder before starting the declared external artifact."""

    sections = _declared_flight_software_sections(smoke_case)
    for section in sections:
        params = dict(section.get("params", {}) or {})
        host = str(params.get("cfs_host", params.get("remote_host", ""))).strip()
        raw_port = params.get("cfs_port", params.get("remote_port"))
        if not host or raw_port is None:
            continue
        port = int(raw_port)
        info = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0]
        probe = socket.socket(info[0], socket.SOCK_DGRAM)
        try:
            probe.bind(info[4])
        except OSError as exc:
            raise RuntimeError(f"Declared external endpoint is already in use before launch: {host}:{port}") from exc
        finally:
            probe.close()


def _declared_flight_software_sections(smoke_case: Path | None) -> list[dict[str, Any]]:
    if smoke_case is None:
        return []
    raw = yaml.safe_load(smoke_case.read_text(encoding="utf-8"))
    sections: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            candidate = value.get("flight_software")
            if isinstance(candidate, dict):
                sections.append(dict(candidate))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(raw)
    return sections


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    stop_local_process(process)
    return


__all__ = ["SUPPORTED_CFS_PROFILE", "run_external_smoke", "validate_external_compatibility"]
