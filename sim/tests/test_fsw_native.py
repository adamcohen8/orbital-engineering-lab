"""Native public authoring acceptance: actual compilers and OEL closed-loop runs."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sim.flight_software import BootEvent, ShutdownEvent
from sim.flight_software.schemas import to_primitive
from sim.fsw_authoring import (
    build_candidate,
    init_candidate,
    load_candidate,
    run_contract_tests,
    run_smoke,
    validate_candidate_service,
    verify_receipt,
)
from sim.fsw_authoring.native import inspect_native_build
from sim.fsw_authoring.native_transport import NativeBridgeFailure, NativeFlightSoftwareAdapter, StdioEndpoint
from sim.review import ReviewWorkspace
from sim.tests.fsw_v2_helpers import batch, ideal_event
from sim.tests.fsw_v2_helpers import clock as tag


@pytest.fixture(scope="module")
def candidates(tmp_path_factory):
    if not shutil.which("cmake") or not shutil.which("cargo"):
        pytest.skip("Native acceptance needs CMake, C/C++, and Rust toolchains.")
    root = (
        Path(os.environ["OEL_NATIVE_ACCEPTANCE_ROOT"])
        if "OEL_NATIVE_ACCEPTANCE_ROOT" in os.environ
        else tmp_path_factory.mktemp("native_fsw")
    )
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for language in ("c", "cpp", "rust"):
        for template in ("adcs", "rpo"):
            manifest = Path(
                init_candidate(f"{language}_{template}", language=language, template=template, workspace_root=root)[
                    "result"
                ]["manifest_path"]
            )
            try:
                build = inspect_native_build(load_candidate(manifest, workspace_root=root))
                if not any(name.startswith("CMAKE_C_COMPILER") for name in build["toolchains"]):
                    raise ValueError("Refresh compiler provenance")
            except (OSError, ValueError, PermissionError):
                build = build_candidate(manifest, workspace_root=root, trusted_build=True)
            assert build["status"] == "passed", build
            result[language, template] = manifest
    return root, result


@pytest.mark.parametrize("language", ["c", "cpp", "rust"])
@pytest.mark.parametrize("template", ["adcs", "rpo"])
def test_native_candidate_build_test_and_oel_smoke(candidates, language, template):
    root, rows = candidates
    manifest = rows[language, template]
    validated = validate_candidate_service(manifest, workspace_root=root)
    assert validated["status"] == "ready", validated
    tested = run_contract_tests(manifest, workspace_root=root, validation_id=validated["result"]["validation_id"])
    assert tested["status"] == "passed", tested
    smoked = run_smoke(manifest, workspace_root=root, validation_id=validated["result"]["validation_id"])
    assert smoked["status"] == "passed"
    assert smoked["summary"]["samples"] == 21
    assert Path(smoked["summary"]["review_sqlite_path"]).is_file()
    assert smoked["summary"]["runtime_profile"]["executor"]["object_step_backend"] == "serial"
    assert verify_receipt(smoked["artifacts"][0]["path"], workspace_root=root)["status"] == "passed"
    review = ReviewWorkspace.open(smoked["run_output_dir"])
    field = "torque" if template == "adcs" else "force"
    unit = "n_m" if template == "adcs" else "n"
    query = f"SELECT MAX(ABS(realized_{field}_x_{unit})+ABS(realized_{field}_y_{unit})+ABS(realized_{field}_z_{unit})) AS peak FROM actuator_realization"
    peak = review.query(query).rows[0]["peak"]
    assert peak > 0, "Smoke must realize a nonzero native control command."
    disposition_query = (
        "SELECT COUNT(*) AS rejected FROM actuator_command_receipts WHERE disposition <> 'CommandDisposition.ACCEPTED'"
    )
    rejected = review.query(disposition_query).rows[0]["rejected"]
    assert rejected == 0
    smoked["acceptance_checks"] = {
        "peak_effort": peak,
        "query": query,
        "rejected_commands": rejected,
        "disposition_query": disposition_query,
    }
    # Record this host's empirical acceptance separately from platform claims.
    (root / f"{language}-{template}-acceptance.json").write_text(json.dumps(smoked, indent=2, default=str))


def test_same_boundary_input_produces_matching_native_commands(candidates):
    root, rows = candidates
    outputs = []
    for language in ("c", "cpp", "rust"):
        manifest = rows[language, "adcs"]
        stack = NativeFlightSoftwareAdapter(str(manifest), str(root), satellite_id="sat")
        stack.boot(BootEvent("sat", "conformance-boot", tag(0)))
        try:
            output = stack.step(batch(1, ideal_event(0, 1, quaternion=(0.8, 0.6, 0.0, 0.0), rate=(0.01, 0.02, 0.03))))
            assert len(output.commands) == 1
            assert output.commands[0].validity.expires_at.ticks > output.commands[0].issued_at.ticks
            assert any(abs(v) > 0 for v in output.commands[0].payload.torque_n_m)
            outputs.append(to_primitive(output))
            with pytest.raises(NativeBridgeFailure, match="checkpointable=false"):
                stack.snapshot()
        finally:
            stack.shutdown(ShutdownEvent("sat", tag(1), "test_complete"))
    for output in outputs[1:]:
        for field in ("force_n", "torque_n_m"):
            assert output["commands"][0]["payload"][field] == pytest.approx(
                outputs[0]["commands"][0]["payload"][field], rel=1e-12, abs=1e-14
            )
            output["commands"][0]["payload"][field] = outputs[0]["commands"][0]["payload"][field]
        assert output == outputs[0]


def test_native_build_binds_source_and_binary(candidates):
    root, rows = candidates
    manifest = rows["c", "adcs"]
    candidate = load_candidate(manifest, workspace_root=root)
    build = inspect_native_build(candidate)
    source = candidate.source.root / "main.c"
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n/* changed source */\n")
        assert load_candidate(manifest, workspace_root=root).candidate_sha256 != candidate.candidate_sha256
        with pytest.raises(PermissionError, match="stale"):
            inspect_native_build(candidate)
    finally:
        source.write_bytes(original)
    executable = Path(build["executable"])
    original_binary = executable.read_bytes()
    try:
        executable.write_bytes(original_binary + b"changed")
        with pytest.raises(PermissionError, match="stale"):
            inspect_native_build(candidate)
    finally:
        executable.write_bytes(original_binary)
    if os.name != "nt":
        original_mode = executable.stat().st_mode
        try:
            executable.chmod(0o600)
            with pytest.raises(PermissionError, match="not executable"):
                inspect_native_build(candidate)
        finally:
            executable.chmod(original_mode)


def test_build_requires_explicit_source_trust(tmp_path):
    manifest = Path(init_candidate("trust", language="c", workspace_root=tmp_path)["result"]["manifest_path"])
    with pytest.raises(PermissionError, match="trusted-build"):
        build_candidate(manifest, workspace_root=tmp_path)
    assert not (manifest.parent / ".oel" / "native_build.json").exists()


def _endpoint(tmp_path, program):
    launcher = tmp_path / "endpoint"
    if os.name == "nt":
        pytest.skip("POSIX executable fixture; native lifecycle acceptance is portable.")
    source = tmp_path / "endpoint.py"
    source.write_text(program)
    # Use exec so signals reach the Python process, and quote the interpreter
    # and fixture paths because POSIX shebangs cannot encode spaces in paths.
    launcher.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(source))} \"$@\"\n")
    launcher.chmod(0o700)
    return launcher


@pytest.mark.parametrize("behavior", ["hang", "crash", "wrong_sequence", "oversize", "invalid_json"])
def test_transport_failures_are_bounded_and_reap_process(tmp_path, behavior):
    programs = {
        "hang": "import time\ntime.sleep(10)\n",
        "crash": "raise SystemExit(2)\n",
        "wrong_sequence": "import json,sys\np=json.loads(sys.stdin.readline());p.update(sequence=999,ok=True);print(json.dumps(p),flush=True)\n",
        "oversize": 'print("x"*300000,flush=True)\n',
        "invalid_json": 'print("not JSON",flush=True)\n',
    }
    endpoint = StdioEndpoint(_endpoint(tmp_path, programs[behavior]), timeout_s=0.25)
    with pytest.raises(NativeBridgeFailure):
        endpoint.exchange("hello", {})
    assert endpoint.process.poll() is not None


def test_native_sdk_rejects_malformed_protocol(candidates):
    root, rows = candidates
    artifact = inspect_native_build(load_candidate(rows["c", "adcs"], workspace_root=root))["executable"]
    for malformed in ("{}\n", "{broken\n", "[]\n", "x" * 262145 + "\n"):
        completed = subprocess.run([artifact], input=malformed, text=True, capture_output=True, timeout=3)
        assert completed.returncode != 0
        assert not completed.stdout


def test_native_closed_loop_histories_agree(candidates):
    root, _ = candidates
    query = "SELECT pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s, quat_w, quat_x, quat_y, quat_z FROM object_state ORDER BY object_id, sample_index"
    for template in ("adcs", "rpo"):
        histories = []
        for language in ("c", "cpp", "rust"):
            record = json.loads((root / f"{language}-{template}-acceptance.json").read_text())
            histories.append(ReviewWorkspace.open(record["run_output_dir"]).query(query).rows)
        for history in histories[1:]:
            assert len(history) == len(histories[0])
            for row, reference in zip(history, histories[0]):
                assert row == pytest.approx(reference, rel=1e-12, abs=1e-10)


def test_native_hidden_build_inputs_are_content_bound(tmp_path):
    from sim.fsw_authoring.native import native_source_digest

    source = tmp_path / "src"
    source.mkdir()
    (source / "main.c").write_text("int main(void){return 0;}")
    original = native_source_digest(source)
    hidden = source / ".oel"
    hidden.mkdir()
    (hidden / "compile-flags").write_text("-O2")
    assert native_source_digest(source) != original
    original = native_source_digest(source)
    (hidden / "compile-flags").write_text("-O3")
    assert native_source_digest(source) != original
    (source / "escaped.h").symlink_to(tmp_path / "outside.h")
    with pytest.raises(PermissionError, match="symlink"):
        native_source_digest(source)


def test_checkpointable_native_candidate_round_trips_state(tmp_path):
    import hashlib

    import yaml

    from sim.flight_software import FlightSoftwareSnapshot

    if not shutil.which("cmake"):
        pytest.skip("Native checkpoint acceptance requires CMake and C.")
    manifest = Path(init_candidate("checkpoint", language="c", workspace_root=tmp_path)["result"]["manifest_path"])
    raw = yaml.safe_load(manifest.read_text())
    raw["native"]["checkpointable"] = True
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False))
    snapshots = []
    for value in (0, 1):
        state = str(value).encode()
        snapshot = FlightSoftwareSnapshot(
            "checkpoint", "0.1.0", "checkpoint", value, None, None, state, hashlib.sha256(state).hexdigest()
        )
        snapshots.append(json.dumps(json.dumps(to_primitive(snapshot))))
    source = r"""#include "oel_fsw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int count=0;
static int callback(const char* op,const oel_json* input,char* out,size_t capacity,void* context){
    (void)context;
    if(!strcmp(op,"boot")){count=0;snprintf(out,capacity,"{}");return 0;}
    if(!strcmp(op,"shutdown")){snprintf(out,capacity,"{}");return 0;}
    if(!strcmp(op,"snapshot")){snprintf(out,capacity,"%s",count?@ONE@:@ZERO@);return 0;}
    if(!strcmp(op,"restore")){
        char invocation[32];int token=oel_field(input,-1,"invocation_id");
        if(token<0||oel_raw(input,token,invocation,sizeof invocation))return -1;
        count=atoi(invocation);snprintf(out,capacity,"{}");return 0;
    }
    if(!strcmp(op,"step")){
        double f[3]={0,0,0},t[3]={0.001,0,0};++count;t[0]*=count;
        return oel_wrench(input,"\"checkpoint\"",f,t,out,capacity);
    }
    return -1;
}
int main(void){return oel_run(callback,NULL,1);}
"""
    (manifest.parent / "src/main.c").write_text(source.replace("@ZERO@", snapshots[0]).replace("@ONE@", snapshots[1]))
    built = build_candidate(manifest, workspace_root=tmp_path, trusted_build=True)
    assert built["status"] == "passed", built
    stack = NativeFlightSoftwareAdapter(str(manifest), str(tmp_path), satellite_id="sat")
    stack.boot(BootEvent("sat", "checkpoint", tag(0)))
    try:
        before = stack.snapshot()
        first = stack.step(batch(1, ideal_event(0, 1)))
        after = stack.snapshot()
        assert before.state_bytes == b"0" and after.state_bytes == b"1"
        stack.restore(before)
        repeated = stack.step(batch(1, ideal_event(0, 1)))
        assert first == repeated
    finally:
        stack.shutdown(ShutdownEvent("sat", tag(2), "test_complete"))


def test_build_refuses_a_symlink_inserted_during_source_snapshot(tmp_path, monkeypatch):
    from sim.fsw_authoring import native

    manifest = Path(init_candidate("source_race", language="c", workspace_root=tmp_path)["result"]["manifest_path"])
    original = shutil.copytree
    inserted = False

    def racing_copy(*args, **kwargs):
        nonlocal inserted
        if not inserted:
            inserted = True
            (manifest.parent / "src" / "late-link.h").symlink_to(tmp_path / "outside.h")
        return original(*args, **kwargs)

    monkeypatch.setattr(native.shutil, "copytree", racing_copy)
    # Source snapshot rejection happens before any compiler invocation.
    monkeypatch.setattr(native, "toolchains", lambda: {"cmake": "cmake"})
    with pytest.raises(PermissionError, match="symlink"):
        build_candidate(manifest, workspace_root=tmp_path, trusted_build=True)
