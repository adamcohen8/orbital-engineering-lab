"""Regressions for native authoring audit findings, using real local artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim.flight_software import (
    FlightSoftwareSnapshot,
    NativeFlightSoftwareSnapshot,
    StackIdentity,
    canonical_json_bytes,
    canonical_loads,
    from_primitive,
    to_primitive,
)
from sim.fsw_authoring import (
    build_candidate,
    init_candidate,
    load_candidate,
    native,
    native_transport,
    plan_workflow,
    run_contract_tests,
    run_smoke,
    services,
)
from sim.fsw_authoring.native import inspect_native_build
from sim.fsw_authoring.native_transport import NativeBridgeFailure, NativeFlightSoftwareAdapter, StdioEndpoint
from sim.tests.test_fsw_native import _endpoint


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not shutil.which("cmake"):
        pytest.skip("CMake and a C compiler are required.")
    root = tmp_path_factory.mktemp("native_audit")
    manifest = Path(init_candidate("audit_c", language="c", workspace_root=root)["result"]["manifest_path"])
    result = build_candidate(manifest, workspace_root=root, trusted_build=True)
    assert result["status"] == "passed", result
    return root, manifest


def test_build_output_cannot_recurse_into_source(tmp_path):
    manifest = Path(init_candidate("overlap", language="c", workspace_root=tmp_path)["result"]["manifest_path"])
    output = manifest.parent / "src" / "build"
    with pytest.raises(PermissionError, match="outside.*source"):
        build_candidate(manifest, workspace_root=tmp_path, output_dir=output, trusted_build=True)
    with pytest.raises(PermissionError, match="outside.*source"):
        plan_workflow(manifest, "build", workspace_root=tmp_path, output_dir=output)
    assert not output.exists()


def test_deleted_binary_can_be_rebuilt(built):
    root, manifest = built
    receipt = inspect_native_build(load_candidate(manifest, workspace_root=root))
    Path(receipt["executable"]).unlink()
    result = build_candidate(manifest, workspace_root=root, trusted_build=True)
    assert result["status"] == "passed"
    assert Path(result["executable"]).is_file()


def test_malformed_receipt_can_be_repaired_by_trusted_build(built):
    root, manifest = built
    pointer = manifest.parent / ".oel/native_build.json"
    saved = pointer.read_bytes()
    try:
        pointer.write_text("[]")
        assert load_candidate(manifest, workspace_root=root).kind == "native_stack"
        with pytest.raises(ValueError, match="malformed"):
            inspect_native_build(load_candidate(manifest, workspace_root=root))
        assert build_candidate(manifest, workspace_root=root, trusted_build=True)["status"] == "passed"
    finally:
        if not pointer.is_file() or not isinstance(json.loads(pointer.read_text()), dict):
            pointer.write_bytes(saved)


def test_content_addressed_mcp_pointer_resolves_current_candidate(built, tmp_path):
    root, manifest = built
    candidate = load_candidate(manifest, workspace_root=root)
    legacy = manifest.parent / ".oel/native_build.json"
    saved_legacy = legacy.read_bytes()
    receipt = json.loads(saved_legacy)
    pointer_dir = root / "mcp-pointer-output"
    pointer_dir.mkdir()
    pointer = pointer_dir / native._content_pointer_name(candidate)
    legacy.unlink()
    current = load_candidate(manifest, workspace_root=root)
    receipt.update(
        {
            "candidate": native._content_pointer_candidate_identity(current),
            "build_binding": native._native_build_binding(current),
            "pointer_identity": native._content_pointer_identity(current),
            "pointer_mode": "content_addressed",
            "pointer_path": str(pointer),
        }
    )
    pointer.write_text(json.dumps(receipt), encoding="utf-8")
    try:
        resolved = inspect_native_build(candidate)
        assert resolved["pointer_mode"] == "content_addressed"
        assert resolved["pointer_path"] == str(pointer)
        tampered = dict(receipt)
        tampered["build_binding"] = {**receipt["build_binding"], "hardware_profile": "other-profile"}
        pointer.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(PermissionError, match="profile"):
            inspect_native_build(candidate)
    finally:
        pointer.unlink(missing_ok=True)
        legacy.write_bytes(saved_legacy)


def test_current_content_pointer_precedes_stale_legacy_pointer(built):
    root, manifest = built
    legacy = manifest.parent / ".oel/native_build.json"
    saved_legacy = legacy.read_bytes()
    receipt = json.loads(saved_legacy)
    pointer_dir = root / "mcp-current-pointer"
    pointer_dir.mkdir()
    try:
        # A prior CLI receipt can remain next to the manifest after its source
        # identity has changed.  The current MCP pointer must win if it binds
        # the candidate now loaded from disk.
        legacy.write_text(json.dumps({**receipt, "status": "failed"}), encoding="utf-8")
        current = load_candidate(manifest, workspace_root=root)
        pointer = pointer_dir / native._content_pointer_name(current)
        receipt.update(
            {
                "candidate": native._content_pointer_candidate_identity(current),
                "build_binding": native._native_build_binding(current),
                "pointer_identity": native._content_pointer_identity(current),
                "pointer_mode": "content_addressed",
                "pointer_path": str(pointer),
            }
        )
        pointer.write_text(json.dumps(receipt), encoding="utf-8")
        resolved = inspect_native_build(current)
        assert resolved["pointer_mode"] == "content_addressed"
        assert resolved["pointer_path"] == str(pointer)
    finally:
        for path in pointer_dir.glob("*"):
            path.unlink()
        pointer_dir.rmdir()
        legacy.write_bytes(saved_legacy)


def test_content_pointer_survives_mutable_legacy_receipt_replacement(built):
    root, manifest = built
    legacy = manifest.parent / ".oel/native_build.json"
    saved_legacy = legacy.read_bytes()
    pointer_dir = root / "mcp-stable-pointer"
    pointer_dir.mkdir()
    try:
        current = load_candidate(manifest, workspace_root=root)
        pointer = pointer_dir / native._content_pointer_name(current)
        receipt = json.loads(saved_legacy)
        receipt.update(
            {
                "candidate": native._content_pointer_candidate_identity(current),
                "build_binding": native._native_build_binding(current),
                "pointer_identity": native._content_pointer_identity(current),
                "pointer_mode": "content_addressed",
                "pointer_path": str(pointer),
            }
        )
        pointer.write_text(json.dumps(receipt), encoding="utf-8")
        assert receipt["candidate"]["kind"] == "native_stack"
        # A legacy CLI receipt is mutable state and may be rewritten after the
        # MCP pointer was issued.  That replacement must not stale the MCP
        # pointer's immutable candidate binding.
        legacy.write_text(json.dumps({**json.loads(saved_legacy), "status": "failed"}), encoding="utf-8")
        resolved = inspect_native_build(current)
        assert resolved["pointer_mode"] == "content_addressed"
        assert resolved["pointer_path"] == str(pointer)
    finally:
        for path in pointer_dir.glob("*"):
            path.unlink()
        pointer_dir.rmdir()
        legacy.write_bytes(saved_legacy)


def test_native_snapshot_requires_active_matching_boot():
    adapter = object.__new__(NativeFlightSoftwareAdapter)
    adapter._identity = StackIdentity("stack", "1.0", 1, "0" * 64, True)
    state = b"state"
    snapshot = FlightSoftwareSnapshot(
        "stack", "1.0", "boot-a", 0, None, None, state, hashlib.sha256(state).hexdigest()
    )

    adapter._active_boot_id = None
    with pytest.raises(NativeBridgeFailure, match="active boot"):
        adapter._check_snapshot_identity(snapshot)

    adapter._active_boot_id = "boot-b"
    with pytest.raises(NativeBridgeFailure, match="different boot"):
        adapter._check_snapshot_identity(snapshot)

    adapter._active_boot_id = "boot-a"
    adapter._check_snapshot_identity(snapshot)


def _host_bound_snapshot(*, implementation_hash: str = "a" * 64, boot_id: str = "boot-a") -> NativeFlightSoftwareSnapshot:
    state = b"native-state"
    return NativeFlightSoftwareSnapshot(
        stack_id="stack",
        stack_version="1.0",
        boot_id=boot_id,
        invocation_id=0,
        active_load_id=None,
        active_load_revision=None,
        state_bytes=state,
        state_hash_sha256=hashlib.sha256(state).hexdigest(),
        host_implementation_hash=implementation_hash,
    )


def _snapshot_test_adapter(*, implementation_hash: str = "a" * 64, allow_legacy: bool = False):
    adapter = object.__new__(NativeFlightSoftwareAdapter)
    adapter._identity = StackIdentity("stack", "1.0", 1, implementation_hash, True)
    adapter._implementation_hash = implementation_hash
    adapter._legacy_implementation_hash = "c" * 64
    adapter._allow_legacy_snapshot_restore = allow_legacy
    adapter._active_boot_id = "boot-a"
    calls = []
    adapter._ack = lambda operation, payload: calls.append((operation, payload))
    return adapter, calls


def test_native_snapshot_binding_rejects_tamper_cross_binary_and_cross_boot():
    adapter, calls = _snapshot_test_adapter()
    snapshot = _host_bound_snapshot()
    adapter.restore(snapshot)
    assert calls == [("restore", snapshot.as_flight_software_snapshot())]

    with pytest.raises(NativeBridgeFailure, match="verified artifact"):
        adapter.restore(replace(snapshot, host_implementation_hash="b" * 64))
    with pytest.raises(NativeBridgeFailure, match="different boot"):
        adapter.restore(replace(snapshot, boot_id="boot-b"))
    assert len(calls) == 1


def test_native_snapshot_binding_requires_metadata_and_explicit_legacy_migration():
    snapshot = _host_bound_snapshot()
    assert canonical_loads(canonical_json_bytes(snapshot)) == snapshot
    primitive = dict(to_primitive(snapshot))
    primitive.pop("host_implementation_hash")
    with pytest.raises(TypeError, match="host_implementation_hash"):
        from_primitive(NativeFlightSoftwareSnapshot, primitive)

    adapter, calls = _snapshot_test_adapter()
    legacy = snapshot.as_flight_software_snapshot()
    with pytest.raises(NativeBridgeFailure, match="Legacy native checkpoint restore is blocked"):
        adapter.restore(legacy)
    with pytest.raises(NativeBridgeFailure, match="requires allow_legacy_snapshot_restore"):
        adapter.migrate_legacy_snapshot(legacy, expected_executable_sha256="c" * 64)
    adapter._allow_legacy_snapshot_restore = True
    with pytest.raises(NativeBridgeFailure, match="Legacy native checkpoint restore is blocked"):
        adapter.restore(legacy)
    with pytest.raises(NativeBridgeFailure, match="expected executable digest"):
        adapter.migrate_legacy_snapshot(legacy, expected_executable_sha256="")
    with pytest.raises(NativeBridgeFailure, match="does not match"):
        adapter.migrate_legacy_snapshot(legacy, expected_executable_sha256="d" * 64)
    bound = adapter.migrate_legacy_snapshot(legacy, expected_executable_sha256="c" * 64)
    assert bound.host_implementation_hash == adapter._implementation_hash
    adapter.restore(bound)
    assert calls == [("restore", legacy)]


def test_native_snapshot_host_hash_is_derived_from_verified_receipt():
    candidate = SimpleNamespace(
        candidate_id="stack",
        revision="1.0",
        onboard_contract="oel.fsw.boundary.v1",
        hardware_profile="hardware.ideal_wrench.v1",
        normalized_manifest={"source": {"language": "c"}},
    )
    build = {
        "source_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "verification_sha256": "3" * 64,
        "executable_sha256": "4" * 64,
    }
    first = native_transport._verified_native_implementation_hash(candidate, build)
    second = native_transport._verified_native_implementation_hash(
        candidate, {**build, "executable_sha256": "5" * 64}
    )
    assert len(first) == 64 and first != second
    python_snapshot = FlightSoftwareSnapshot(
        "stack", "1.0", "boot-a", 0, None, None, b"python", hashlib.sha256(b"python").hexdigest()
    )
    assert "host_implementation_hash" not in to_primitive(python_snapshot)


def test_multiple_current_content_pointers_fail_closed(built):
    root, manifest = built
    legacy = manifest.parent / ".oel/native_build.json"
    saved_legacy = legacy.read_bytes()
    pointer_dirs = (root / "mcp-pointer-a", root / "mcp-pointer-b")
    try:
        legacy.unlink()
        current = load_candidate(manifest, workspace_root=root)
        pointer_name = native._content_pointer_name(current)
        for pointer_dir in pointer_dirs:
            pointer_dir.mkdir()
            (pointer_dir / pointer_name).write_text("{}", encoding="utf-8")
        with pytest.raises(PermissionError, match="Multiple native build pointers"):
            inspect_native_build(current)
    finally:
        for pointer_dir in pointer_dirs:
            pointer = next(pointer_dir.glob("*"), None)
            if pointer is not None:
                pointer.unlink()
            pointer_dir.rmdir()
        legacy.write_bytes(saved_legacy)


def test_receipt_cannot_inject_component_test_command(built, tmp_path):
    root, manifest = built
    pointer = manifest.parent / ".oel/native_build.json"
    saved = pointer.read_bytes()
    marker = root / "injected"
    try:
        receipt = json.loads(saved)
        receipt["test_command"] = [sys.executable, "-c", f'open({str(marker)!r},"w").write("executed")']
        pointer.write_text(json.dumps(receipt))
        with pytest.raises(PermissionError, match="fixed toolchain"):
            run_contract_tests(manifest, workspace_root=root, output_dir=root / "audit-test-command")
        assert not marker.exists()
    finally:
        pointer.write_bytes(saved)


def test_receipt_executable_must_remain_inside_workspace(built, tmp_path):
    root, manifest = built
    pointer = manifest.parent / ".oel/native_build.json"
    saved = pointer.read_bytes()
    candidate = load_candidate(manifest, workspace_root=root)
    try:
        receipt = json.loads(saved)
        outside = tmp_path / "outside-executable"
        outside.write_text("#!/bin/sh\n", encoding="utf-8")
        outside.chmod(0o755)
        receipt["executable"] = str(outside)
        receipt["executable_sha256"] = native.sha256_file(outside)
        pointer.write_text(json.dumps(receipt))
        with pytest.raises(PermissionError, match="remain inside the authorized workspace"):
            inspect_native_build(candidate)
    finally:
        pointer.write_bytes(saved)


def test_smoke_rejects_identity_change_during_execution(built, tmp_path, monkeypatch):
    root, manifest = built
    source = manifest.parent / "src/main.c"
    before = source.read_bytes()

    def changed_run(*args, **kwargs):
        source.write_bytes(before + b"\n/* concurrent edit */\n")
        return SimpleNamespace(payload={}, summary={})

    monkeypatch.setattr(services.SimulationWorkspace, "run", changed_run)
    output = root / "audit-smoke"
    try:
        with pytest.raises(PermissionError, match="identity changed"):
            run_smoke(manifest, workspace_root=root, output_dir=output)
        assert not (output / "fsw_run_manifest.json").exists()
    finally:
        source.write_bytes(before)


def test_component_suite_rejects_identity_change_during_execution(built, tmp_path, monkeypatch):
    root, manifest = built
    source = manifest.parent / "src/main.c"
    before = source.read_bytes()

    def changed_execute(*args, **kwargs):
        source.write_bytes(before + b"\n/* changed by component suite */\n")
        return {"returncode": 0}

    monkeypatch.setattr(native, "_execute", changed_execute)
    try:
        with pytest.raises(PermissionError, match="identity changed"):
            run_contract_tests(manifest, workspace_root=root, output_dir=root / "audit-component")
    finally:
        source.write_bytes(before)


@pytest.mark.parametrize("language", ["python", "c", "cpp", "rust"])
def test_short_candidate_names_are_valid(tmp_path, language):
    receipt = init_candidate("C", language=language, workspace_root=tmp_path)
    assert receipt["status"] == "ready"
    assert receipt["candidate"]["candidate_id"] == "c_stack"


def test_python_build_plan_is_rejected(tmp_path):
    manifest = Path(init_candidate("python_only", workspace_root=tmp_path)["result"]["manifest_path"])
    with pytest.raises(ValueError, match="native_stack"):
        plan_workflow(manifest, "build", workspace_root=tmp_path)


def test_partial_pipe_writes_preserve_complete_request(tmp_path):
    program = 'import sys,json\np=json.loads(sys.stdin.readline());p["ok"]=True;print(json.dumps(p),flush=True)\nsys.stdin.read()\n'
    endpoint = StdioEndpoint(_endpoint(tmp_path, program), timeout_s=2)
    original = endpoint.process.stdin

    class PartialWriter:
        def write(self, data):
            return original.write(data[:7])

        def close(self):
            return original.close()

    endpoint.process.stdin = PartialWriter()
    try:
        value = {"message": "a" * 1000}
        assert endpoint.exchange("hello", value) == value
    finally:
        endpoint.close()


def test_duplicate_response_fields_fail_closed(tmp_path):
    program = 'import sys\nsys.stdin.readline()\nprint(\'{"protocol":"oel.fsw.stdio.v1","sequence":0,"sequence":1,"operation":"hello","ok":true,"payload":{}}\',flush=True)\nsys.stdin.read()\n'
    endpoint = StdioEndpoint(_endpoint(tmp_path, program), timeout_s=1)
    with pytest.raises(NativeBridgeFailure, match="Duplicate"):
        endpoint.exchange("hello", {})
    assert endpoint.process.poll() is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX descendant fixture; Windows needs native qualification.")
def test_response_timeout_does_not_block_on_inherited_pipe(tmp_path, monkeypatch):
    child_pid = tmp_path / "child.pid"
    program = f"""import subprocess,sys,time
p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(3)"])
open({str(child_pid)!r},"w").write(str(p.pid))
time.sleep(3)
"""
    endpoint = StdioEndpoint(_endpoint(tmp_path, program), timeout_s=0.15)

    # Exercise the fallback used when a host denies Unix group signaling.
    def direct_only(process, force=False):
        if process.poll() is None:
            process.kill() if force else process.terminate()

    monkeypatch.setattr(native_transport, "_signal_process", direct_only)
    start = time.monotonic()
    try:
        with pytest.raises(NativeBridgeFailure):
            endpoint.exchange("hello", {"large_request": "x" * 150000})
        assert time.monotonic() - start < 1.0
    finally:
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass
        endpoint.close()


@pytest.mark.parametrize("sequence", [True, 0, -1, 1.5, "1", 2**64])
def test_sdk_rejects_invalid_sequence(built, sequence):
    root, manifest = built
    executable = inspect_native_build(load_candidate(manifest, workspace_root=root))["executable"]
    request = {
        "protocol": "oel.fsw.stdio.v1",
        "sequence": sequence,
        "operation": "hello",
        "payload": {"boundary": "oel.fsw.boundary.v1"},
    }
    result = subprocess.run([executable], input=json.dumps(request) + "\n", text=True, capture_output=True, timeout=3)
    assert result.returncode != 0
    assert not result.stdout


def test_sdk_rejects_unsupported_handshake_and_unbooted_step(built):
    root, manifest = built
    executable = inspect_native_build(load_candidate(manifest, workspace_root=root))["executable"]
    requests = [
        {"protocol": "oel.fsw.stdio.v1", "sequence": 1, "operation": "hello", "payload": {"boundary": "unsupported"}}
    ]
    result = subprocess.run(
        [executable], input=json.dumps(requests[0]) + "\n", text=True, capture_output=True, timeout=3
    )
    assert json.loads(result.stdout)["ok"] is False
    requests[0]["payload"]["boundary"] = "oel.fsw.boundary.v1"
    requests.append({"protocol": "oel.fsw.stdio.v1", "sequence": 2, "operation": "step", "payload": {}})
    result = subprocess.run(
        [executable], input="".join(json.dumps(r) + "\n" for r in requests), text=True, capture_output=True, timeout=3
    )
    assert json.loads(result.stdout.splitlines()[1])["ok"] is False


def test_scaffold_copies_only_maintained_sdk_sources(tmp_path, monkeypatch):
    from sim.fsw_authoring import native_scaffold

    sdk = tmp_path / "sdk"
    shutil.copytree(Path(native_scaffold.__file__).with_name("sdk"), sdk)
    (sdk / "c/build").mkdir()
    (sdk / "c/build/opaque.bin").write_bytes(b"not source")
    # Override only the module-local SDK root via its source location.
    monkeypatch.setattr(native_scaffold, "__file__", str(tmp_path / "native_scaffold.py"))
    manifest = Path(init_candidate("inventory", language="c", workspace_root=tmp_path)["result"]["manifest_path"])
    assert not list((manifest.parent / "src").rglob("opaque.bin"))


@pytest.mark.parametrize("response", ["malformed", "unterminated"])
def test_sdk_refuses_invalid_callback_payload(tmp_path, response):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("SDK codec fixture requires a C compiler.")
    sdk = Path(native.__file__).with_name("sdk") / "c"
    body = "memset(out, 65, capacity);" if response == "unterminated" else 'strcpy(out, "not JSON");'
    source = tmp_path / "callback.c"
    source.write_text(
        '#include "oel_fsw.h"\n#include <string.h>\n'
        "static int callback(const char *op,const oel_json *input,char *out,size_t capacity,void *context){"
        "(void)op;(void)input;(void)context;(void)capacity;" + body + "return 0;}\n"
        "int main(void){return oel_run(callback,0,0);}\n"
    )
    executable = tmp_path / "callback"
    subprocess.run(
        [compiler, "-I", str(sdk), str(source), str(sdk / "oel_fsw.c"), "-lm", "-o", str(executable)], check=True
    )
    requests = [
        {
            "protocol": "oel.fsw.stdio.v1",
            "sequence": 1,
            "operation": "hello",
            "payload": {"boundary": "oel.fsw.boundary.v1"},
        },
        {"protocol": "oel.fsw.stdio.v1", "sequence": 2, "operation": "boot", "payload": {}},
    ]
    result = subprocess.run(
        [str(executable)],
        input="".join(json.dumps(r) + "\n" for r in requests),
        text=True,
        capture_output=True,
        timeout=3,
    )
    assert result.returncode == 4
    assert len(result.stdout.splitlines()) == 1  # Only the successful SDK handshake.


@pytest.mark.parametrize("dangling", [False, True])
def test_build_pointer_directory_cannot_redirect_writes(tmp_path, dangling):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = Path(init_candidate("pointer_link", language="c", workspace_root=workspace)["result"]["manifest_path"])
    outside = tmp_path / "outside"
    if not dangling:
        outside.mkdir()
    directory = manifest.parent / ".oel"
    saved = manifest.parent / ".oel.saved"
    directory.rename(saved)  # Preserve the scaffold receipt while exercising the link.
    directory.symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(PermissionError, match="symlink"):
            build_candidate(manifest, workspace_root=workspace, trusted_build=True)
        assert not (outside / "native_build.json").exists()
    finally:
        directory.unlink()
        saved.rename(directory)
