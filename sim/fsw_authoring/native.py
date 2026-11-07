"""Public native candidate identity, offline builds, and native test services."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from .contracts import (
    AuthoringIssue,
    CandidateSource,
    CandidateVerification,
    FlightSoftwareCandidate,
    effects,
    generated_utc,
    sha256_file,
    sha256_tree,
    sha256_value,
)

SCHEMA = Path(__file__).with_name("schemas") / "native-candidate.schema.json"
BUILD_SCHEMA = "oel.fsw_authoring.build_receipt.v1"
_CONTENT_POINTER_PREFIX = "native_build_pointer_"
_CONTENT_POINTER_SUFFIX = ".json"


def _native_build_binding(candidate: FlightSoftwareCandidate) -> dict[str, Any]:
    """Return the immutable identity used by MCP build pointers.

    The public CLI historically keeps one mutable ``.oel/native_build.json``
    pointer next to the candidate manifest.  A descriptor-bound MCP operation
    cannot safely update that path because it is outside the operation's
    declared output root.  MCP builds therefore retain their pointer inside
    the new output and address it by the exact source, ABI, and hardware/native
    profile identity below.
    """

    native = dict(candidate.native or {})
    return {
        "source_sha256": candidate.source_sha256,
        "onboard_contract": candidate.onboard_contract,
        "hardware_profile": candidate.hardware_profile,
        "native_profile": {
            "language": str(native.get("language", "")),
            "checkpointable": bool(native.get("checkpointable", False)),
            "timeout_s": float(native.get("timeout_s", 0.0)),
            "task_period_s": float(candidate.task_period_s),
        },
    }


def _content_pointer_identity(candidate: FlightSoftwareCandidate) -> str:
    return sha256_value(_native_build_binding(candidate))


def _content_pointer_candidate_identity(candidate: FlightSoftwareCandidate) -> dict[str, Any]:
    """Return the immutable candidate identity mapping stored in an MCP pointer.

    The legacy CLI receipt is part of ``candidate.identity()`` so that its
    repair-on-rebuild lifecycle remains visible to existing callers.  An MCP
    content pointer must remain addressable when that mutable receipt is
    replaced, so its candidate binding retains the public mapping shape but
    derives ``candidate_sha256`` only from source, manifest, and verification
    inputs.
    """

    identity = candidate.identity()
    identity["candidate_sha256"] = sha256_value(
        {
            "manifest": candidate.manifest_sha256,
            "source": candidate.source_sha256,
            "verification": candidate.verification_sha256,
        }
    )
    return identity


def _content_pointer_name(candidate: FlightSoftwareCandidate) -> str:
    return f"{_CONTENT_POINTER_PREFIX}{_content_pointer_identity(candidate)}{_CONTENT_POINTER_SUFFIX}"


def _find_content_pointer(candidate: FlightSoftwareCandidate) -> Path | None:
    """Find one exact MCP pointer without following workspace symlinks.

    A build output is deliberately portable within the authorized workspace,
    so its content-addressed pointer is not tied to a single output directory.
    Multiple matching pointers are ambiguous and fail closed instead of making
    the adapter choose an arbitrary executable.
    """

    expected_name = _content_pointer_name(candidate)
    matches: list[Path] = []

    def fail(error: OSError) -> None:
        raise error

    for directory, directories, files in os.walk(candidate.workspace_root, followlinks=False, onerror=fail):
        directories[:] = [
            name
            for name in directories
            if name not in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
        ]
        if expected_name not in files:
            continue
        pointer = Path(directory) / expected_name
        if pointer.is_symlink():
            raise PermissionError("Native build pointer files may not be symlinks.")
        matches.append(pointer)
    if len(matches) > 1:
        raise PermissionError("Multiple native build pointers match the current source/ABI/profile identity.")
    return matches[0] if matches else None


def native_source_digest(root: Path) -> str:
    """Bind every native build input, including dotfiles and build scripts."""
    rows = []

    def fail(error: OSError) -> None:
        raise error

    for directory, directories, files in os.walk(root, followlinks=False, onerror=fail):
        for name in directories + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise PermissionError(f"Native source inputs may not be symlinks: {path}")
            if path.is_file():
                rows.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    return sha256_value(sorted(rows, key=lambda row: row["path"]))


def _build_pointer(manifest: Path) -> Path:
    return manifest.parent / ".oel" / "native_build.json"


def load_native_candidate(raw: dict, manifest: Path, workspace: Path) -> FlightSoftwareCandidate:
    from .candidate import CandidateValidationError, _candidate_path, _inside

    errors = list(Draft202012Validator(json.loads(SCHEMA.read_text())).iter_errors(raw))
    if errors:
        raise CandidateValidationError([AuthoringIssue("schema_invalid", e.message) for e in errors])
    source = _candidate_path(
        raw["source"]["root"], candidate_root=manifest.parent, workspace_root=workspace, label="Native source"
    )
    verification = CandidateVerification(
        *[
            _candidate_path(
                raw["verification"][name], candidate_root=manifest.parent, workspace_root=workspace, label=name
            )
            for name in ("component_suite", "smoke_case")
        ]
    )
    if verification.component_suite != source:
        raise ValueError("Native component_suite must be the source root owning CMake/Cargo tests.")
    if not source.is_dir():
        raise ValueError("Native source must be a directory.")
    normalized = json.loads(json.dumps(raw, sort_keys=True, allow_nan=False))
    manifest_hash = sha256_value(normalized)
    source_hash = native_source_digest(source)
    verification_hash = sha256_value(
        {
            "component_suite": sha256_tree(verification.component_suite),
            "smoke_case": sha256_file(verification.smoke_case),
        }
    )
    native = {**raw["native"], "language": raw["source"]["language"]}
    build_identity = None
    pointer = _build_pointer(manifest)
    if pointer.parent.is_symlink() or pointer.is_symlink():
        raise PermissionError("Native build pointer files and directories may not be symlinks.")
    _inside(pointer, workspace, label="Build pointer", must_exist=False)
    if pointer.exists():
        # A missing/invalid retained build must not prevent inspection or a
        # trusted rebuild. Preserve its digest so old validation IDs go stale.
        build_identity = {"receipt_sha256": sha256_file(pointer), "executable_sha256": ""}
        try:
            receipt = json.loads(pointer.read_text())
            artifact_path = Path(receipt["executable"])
            if not artifact_path.is_absolute():
                raise ValueError("Native executable paths must be absolute.")
            if not artifact_path.resolve().is_relative_to(workspace):
                raise PermissionError("Native executable must remain inside the authorized workspace.")
            artifact = _inside(artifact_path, workspace, label="Native executable", must_exist=False)
            if artifact_path.is_symlink():
                raise PermissionError("Native executables may not be symlinks.")
            if artifact.is_file():
                build_identity["executable_sha256"] = sha256_file(artifact)
        except (KeyError, TypeError, ValueError):
            # Escape/symlink denials remain fail-closed, rather than repairable.
            pass

    identity = sha256_value(
        {"manifest": manifest_hash, "source": source_hash, "verification": verification_hash, "build": build_identity}
    )
    return FlightSoftwareCandidate(
        str(raw["candidate_id"]),
        str(raw["revision"]),
        manifest,
        workspace,
        CandidateSource(source, raw["source"]["revision_id"], None),
        raw["interfaces"]["onboard_contract"],
        raw["interfaces"]["hardware_profile"],
        float(raw["interfaces"]["task_period_s"]),
        raw["claims"]["intended_use"],
        verification,
        raw["handling"],
        normalized,
        manifest_hash,
        source_hash,
        verification_hash,
        identity,
        native,
    )


def inspect_native_build(candidate: FlightSoftwareCandidate) -> dict:
    from .candidate import load_candidate

    current = load_candidate(candidate.manifest_path, workspace_root=candidate.workspace_root)
    # An MCP build may coexist with an older CLI receipt next to the manifest.
    # Resolve the exact current content pointer first so a stale legacy pointer
    # cannot shadow a valid immutable build.  With no matching MCP pointer,
    # retain the historical CLI path and its repair-on-rebuild behavior.
    pointer = _find_content_pointer(current)
    pointer_mode = "content_addressed" if pointer is not None else "legacy"
    if pointer is None:
        pointer = _build_pointer(candidate.manifest_path)
        if pointer.parent.is_symlink() or pointer.is_symlink():
            raise PermissionError("Native build pointer files and directories may not be symlinks.")
        if not pointer.exists():
            raise ValueError("Native build receipt pointer is missing; rebuild the trusted candidate.")
    try:
        receipt = json.loads(pointer.read_text())
        required = ("executable", "source_sha256", "manifest_sha256", "verification_sha256", "executable_sha256")
        if not isinstance(receipt, dict) or any(not isinstance(receipt.get(key), str) for key in required):
            raise ValueError("Native build receipt has invalid fields.")
    except (ValueError, TypeError) as exc:
        raise ValueError("Native build receipt is malformed; rebuild the trusted candidate.") from exc
    executable_path = Path(receipt["executable"]).expanduser()
    if not executable_path.is_absolute():
        raise ValueError("Native executable paths must be absolute.")
    if executable_path.is_symlink():
        raise PermissionError("Native executables may not be symlinks.")
    executable = executable_path.resolve()
    if not executable.is_relative_to(current.workspace_root):
        raise PermissionError("Native executable must remain inside the authorized workspace.")
    if not os.access(executable, os.X_OK):
        raise PermissionError("Native artifact is not executable on this host.")
    if pointer_mode == "content_addressed":
        expected_pointer_name = _content_pointer_name(current)
        expected_binding = _native_build_binding(current)
        if pointer.name != expected_pointer_name:
            raise PermissionError("Native build pointer name does not match the current source/ABI/profile identity.")
        if receipt.get("pointer_identity") != _content_pointer_identity(current):
            raise PermissionError("Native build pointer identity does not match the current source/ABI/profile.")
        if receipt.get("build_binding") != expected_binding:
            raise PermissionError("Native build pointer profile does not match the current candidate.")
        candidate_identity = receipt.get("candidate")
        if (
            candidate_identity != _content_pointer_candidate_identity(current)
            or receipt.get("pointer_path") != str(pointer)
        ):
            raise PermissionError("Native build pointer candidate identity is stale.")
    if (
        receipt.get("schema") != BUILD_SCHEMA
        or receipt.get("status") != "passed"
        or receipt["source_sha256"] != current.source_sha256
        or receipt["manifest_sha256"] != current.manifest_sha256
        or receipt["verification_sha256"] != current.verification_sha256
        or receipt["executable_sha256"] != sha256_file(executable)
    ):
        raise PermissionError("Native build is stale; rebuild the current candidate.")
    receipt["executable"] = str(executable)
    receipt["pointer_path"] = str(pointer)
    receipt["pointer_mode"] = pointer_mode
    return receipt


def validate_native(candidate: FlightSoftwareCandidate, *, trusted_plugins: bool = False) -> tuple[list, dict]:
    from .candidate import load_simulation_yaml

    issues = []
    checks = {
        "manifest_schema": "passed",
        "source_identity": "passed",
        "path_policy": "passed",
        "candidate_import": "not_applicable",
        "lifecycle": "declared",
        "truth_firewall": "typed_boundary_not_security_sandbox",
    }
    try:
        cfg = load_simulation_yaml(candidate.verification.smoke_case)
        if cfg.analysis.enabled or cfg.monte_carlo.enabled:
            raise ValueError("Native smoke must use one deterministic serial scenario.")
        if str(getattr(cfg.simulator, "object_execution_policy", "serial")) not in {"serial", "configured"}:
            raise ValueError("Native smoke must use deterministic serial stepping.")
        raw = yaml.safe_load(candidate.verification.smoke_case.read_text())
        sections = [obj.get("flight_software", {}) for obj in raw["objects"].values() if obj.get("enabled", True)]
        selected = [s for s in sections if s.get("module") == "sim.fsw_authoring.native_transport"]
        if len(selected) != 1 or selected[0].get("class_name") != "NativeFlightSoftwareAdapter":
            raise ValueError("Native smoke must declare exactly one public native adapter.")
        params = selected[0].get("params", {})
        if (
            Path(params.get("manifest_path", "")).resolve() != candidate.manifest_path
            or Path(params.get("workspace_root", "")).resolve() != candidate.workspace_root
            or selected[0].get("hardware_profile") != candidate.hardware_profile
            or float(selected[0].get("task_period_s", 0)) != candidate.task_period_s
        ):
            raise ValueError("Native smoke adapter does not match candidate identity, hardware, or cadence.")
        if not cfg.outputs.review.enabled:
            raise ValueError("Native smoke must retain the standard review store.")
        if trusted_plugins:
            from sim.config import validate_scenario_plugins

            errors = validate_scenario_plugins(cfg)
            if errors:
                raise ValueError("; ".join(errors))
        checks["smoke_config"] = "passed"
        checks["public_execution_boundary"] = "passed"
        inspect_native_build(candidate)
        checks["native_build"] = "passed"
    except (OSError, ValueError, KeyError, PermissionError) as exc:
        issues.append(
            AuthoringIssue("native_candidate_invalid", str(exc), next_step="Build the trusted native candidate.")
        )
        checks["native_build"] = "failed"
    return issues, checks


def toolchains() -> dict:
    return {name: shutil.which(name) for name in ("cmake", "cc", "c++", "cl", "cargo", "rustc")}


def _execute(command: list[str], cwd: Path, log: Path, *, env: dict | None = None) -> dict:
    # Disk-backed bounded output avoids unbounded capture_output memory.
    from .native_transport import stop_local_process

    counters = [0]
    with log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            bufsize=0,
        )

        cancelled = threading.Event()

        def drain():
            try:
                while not cancelled.is_set():
                    chunk = process.stdout.read(8192)
                    if not chunk or cancelled.is_set():
                        return
                    remaining = max(0, 2_000_000 - counters[0])
                    output.write(chunk[:remaining])
                    counters[0] += len(chunk)
            except (OSError, ValueError):
                return

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            code = 124
        reader.join(timeout=0.1)
        stop_local_process(process)
        reader.join(timeout=1)
        if reader.is_alive():
            cancelled.set()
            code = 125

    return {"command": command, "returncode": code, "log": str(log), "output_truncated": counters[0] > 2_000_000}


def build_candidate(
    manifest: str | Path,
    *,
    workspace_root: str | Path,
    output_dir: str | Path | None = None,
    trusted_build: bool = False,
    writer: Any | None = None,
) -> dict:
    from .candidate import load_candidate
    from .services import _bind_writer, _logicalize, _new_output, _write_json

    candidate = load_candidate(manifest, workspace_root=workspace_root)
    if candidate.native is None:
        raise ValueError("build accepts native_stack candidates only.")
    if not trusted_build:
        raise PermissionError("Native build scripts execute code; use --trusted-build for reviewed source.")
    from .services import _default_output

    output_path = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else _default_output(candidate.candidate_id, "build", workspace_root=candidate.workspace_root)
    )
    if not output_path.is_absolute():
        output_path = candidate.workspace_root / output_path
    # Validate the destination before checking optional toolchains.  A caller
    # must receive the path-policy error even on a host that cannot build; the
    # source/output boundary is a safety invariant, not a compiler-dependent
    # convenience check.
    require_native_output(candidate, output_path)
    tools = toolchains()
    required = ("cmake", "cargo", "rustc") if candidate.native["language"] == "rust" else ("cmake",)
    missing = [name for name in required if tools[name] is None]
    if missing:
        raise ValueError("Missing native build tools: " + ", ".join(missing))
    output = _new_output(
        output_path,
        candidate_id=candidate.candidate_id,
        operation="build",
        workspace_root=candidate.workspace_root,
        writer=writer,
    )
    binding = _bind_writer(writer, output)
    physical_output = binding.physical(output) if binding is not None else output
    source = physical_output / "source"
    # Preserve a concurrently inserted link as a link so the digest refuses it
    # before compilation, rather than copying bytes from its target.
    shutil.copytree(candidate.source.root, source, symlinks=True)
    # Snapshot is hash-verified and contains no symlinks (load_native_candidate refused them).
    if native_source_digest(source) != candidate.source_sha256:
        raise PermissionError("Native source changed while copying build inputs.")
    # Native toolchains are path-facing producers.  Keep their source copy,
    # build tree, executable, and bounded logs in the private physical stage;
    # only the final receipt exposes the logical output paths.
    build = physical_output / "build"
    env = dict(os.environ)
    env["CARGO_NET_OFFLINE"] = "true"
    env["CARGO_BUILD_JOBS"] = "1"
    env["CARGO_TARGET_DIR"] = str(build)
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"
    if candidate.native["language"] == "rust":
        commands = [[tools["cargo"], "build", "--offline", "--locked", "--manifest-path", str(source / "Cargo.toml")]]
        artifact = build / "debug" / ("oel_candidate.exe" if os.name == "nt" else "oel_candidate")
        test_command = [
            tools["cargo"],
            "test",
            "--offline",
            "--locked",
            "--manifest-path",
            str(source / "Cargo.toml"),
            "--",
            "--test-threads=1",
        ]
    else:
        commands = [
            [tools["cmake"], "-S", str(source), "-B", str(build)],
            [tools["cmake"], "--build", str(build), "--config", "Release", "--parallel", "1"],
        ]
        artifact = build / "bin" / ("oel_candidate.exe" if os.name == "nt" else "oel_candidate")
        test_command = [
            str(Path(tools["cmake"]).with_name("ctest.exe" if os.name == "nt" else "ctest")),
            "--test-dir",
            str(build),
            "-C",
            "Release",
            "--output-on-failure",
            "--no-tests=error",
        ]

    runs = []
    for i, command in enumerate(commands):
        runs.append(_execute(command, physical_output, physical_output / f"build-{i}.log", env=env))
        if runs[-1]["returncode"]:
            break
    current = load_candidate(manifest, workspace_root=workspace_root)
    passed = (
        all(r["returncode"] == 0 for r in runs)
        and artifact.is_file()
        and current.source_sha256 == candidate.source_sha256
        and current.verification_sha256 == candidate.verification_sha256
        and current.manifest_sha256 == candidate.manifest_sha256
        and native_source_digest(source) == candidate.source_sha256
    )
    versions = {}
    for name in required:
        run = subprocess.run([tools[name], "--version"], capture_output=True, text=True, timeout=10, check=False)
        versions[name] = run.stdout.splitlines()[0] if run.stdout else run.stderr[:200]
    for cache in build.rglob("CMakeCache.txt") if build.is_dir() else ():
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith(("CMAKE_C_COMPILER:FILEPATH=", "CMAKE_CXX_COMPILER:FILEPATH=")):
                name, compiler = line.split("=", 1)
                run = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=10, check=False)
                versions[name] = {
                    "executable": compiler,
                    "version": (run.stdout or run.stderr)[:2000],
                    "cmake_cache_sha256": sha256_file(cache),
                }
    logical_artifact = _logicalize(artifact, binding)
    logical_source = _logicalize(source, binding)
    logical_build = _logicalize(build, binding)
    receipt = {
        "schema": BUILD_SCHEMA,
        "status": "passed" if passed else "failed",
        "generated_utc": generated_utc(),
        "manifest_path": str(candidate.manifest_path),
        "source_sha256": candidate.source_sha256,
        "manifest_sha256": candidate.manifest_sha256,
        "verification_sha256": candidate.verification_sha256,
        "executable": str(logical_artifact),
        "executable_sha256": sha256_file(artifact) if artifact.is_file() else "",
        "platform": platform.platform(),
        "toolchains": versions,
        "builds": _logicalize(runs, binding),
        "build_environment": {
            key: env.get(key, "")
            for key in (
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "CPPFLAGS",
                "LDFLAGS",
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "CMAKE_GENERATOR",
            )
        },
        "test_command": _logicalize(test_command, binding),
        "test_env": {"CARGO_TARGET_DIR": str(logical_build), "CARGO_NET_OFFLINE": "true"},
        "test_cwd": str(logical_source),
        "effects": effects(writes=True, executes=True),
    }
    pointer: Path | None = None
    if passed:
        current = load_candidate(manifest, workspace_root=workspace_root)
        receipt["candidate"] = (
            current.identity() if binding is None else _content_pointer_candidate_identity(current)
        )
        receipt["build_binding"] = _native_build_binding(current)
        receipt["pointer_identity"] = _content_pointer_identity(current)
        if binding is None:
            # Preserve the historical CLI pointer and its repair-on-rebuild
            # behavior.  Only MCP's active descriptor writer uses the
            # immutable output-local pointer below.
            pointer = _build_pointer(candidate.manifest_path)
            if pointer.parent.is_symlink() or pointer.is_symlink():
                raise PermissionError("Native build pointer files and directories may not be symlinks.")
            receipt["pointer_mode"] = "legacy"
            receipt["pointer_path"] = str(pointer)
            _write_json(pointer, receipt)
        else:
            pointer = output / _content_pointer_name(current)
            receipt["pointer_mode"] = "content_addressed"
            receipt["pointer_path"] = str(pointer)
            # Do not use _write_json here: its manifest-finalization fallback
            # intentionally updates an operation-owned file, while this
            # content-addressed pointer must be created exactly once.
            serialized_pointer = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
            binding.write_bytes(pointer, serialized_pointer)
    receipt["artifacts"] = [
        {
            "artifact_id": "native_executable",
            "path": str(logical_artifact),
            "sha256": receipt["executable_sha256"],
            "exists": artifact.is_file(),
        }
    ]
    if pointer is not None:
        if binding is not None:
            from sim.security.output_writes import sha256_output

            pointer_digest = sha256_output(pointer)
        else:
            pointer_digest = sha256_file(pointer)
        receipt["artifacts"].append(
            {
                "artifact_id": "native_build_pointer",
                "path": str(pointer),
                "sha256": pointer_digest,
                "exists": pointer.is_file(),
            }
        )
    _write_json(output / "fsw_build_receipt.json", receipt, writer=binding, logical_root=output)
    return receipt


def require_native_output(candidate: FlightSoftwareCandidate, path: Path) -> None:
    if path.expanduser().resolve().is_relative_to(candidate.source.root):
        raise PermissionError("Native output must be outside the candidate source tree.")


def require_native_current(candidate: FlightSoftwareCandidate) -> None:
    from .candidate import load_candidate

    current = load_candidate(candidate.manifest_path, workspace_root=candidate.workspace_root)
    if current.candidate_sha256 != candidate.candidate_sha256:
        raise PermissionError("Native candidate identity changed during execution.")
    inspect_native_build(current)


def run_native_tests(candidate: FlightSoftwareCandidate, output: Path) -> dict:
    build = inspect_native_build(candidate)
    artifact = Path(build["executable"])
    build_root = artifact.parent.parent
    source = build_root.parent / "source"
    if (
        not source.resolve().is_relative_to(candidate.workspace_root)
        or build_root.name != "build"
        or build.get("test_cwd") != str(source)
        or native_source_digest(source) != candidate.source_sha256
    ):
        raise PermissionError("Native test snapshot no longer matches the bound build.")
    tools = toolchains()
    if candidate.native["language"] == "rust":
        if not tools["cargo"]:
            raise ValueError("Cargo is unavailable for the native component suite.")
        command = [
            tools["cargo"],
            "test",
            "--offline",
            "--locked",
            "--manifest-path",
            str(source / "Cargo.toml"),
            "--",
            "--test-threads=1",
        ]
    else:
        if not tools["cmake"]:
            raise ValueError("CMake/CTest is unavailable for the native component suite.")
        command = [
            str(Path(tools["cmake"]).with_name("ctest.exe" if os.name == "nt" else "ctest")),
            "--test-dir",
            str(build_root),
            "-C",
            "Release",
            "--output-on-failure",
            "--no-tests=error",
        ]
    environment = {"CARGO_TARGET_DIR": str(build_root), "CARGO_NET_OFFLINE": "true"}
    if build.get("test_command") != command or build.get("test_env") != environment:
        raise PermissionError("Native test receipt does not match the fixed toolchain profile.")
    env = {**os.environ, **environment, "CARGO_BUILD_JOBS": "1"}
    result = _execute(command, source, output / "native-tests.log", env=env)
    require_native_current(candidate)
    if native_source_digest(source) != candidate.source_sha256:
        raise PermissionError("Native test snapshot changed during execution.")
    return result
