from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sim.execution.run_lifecycle.models import (
    InvalidTransitionError,
    RunIdentity,
    RunPhase,
    RunPolicyError,
    RunState,
    local_host_sha256,
)
from sim.execution.run_lifecycle.runner import prepare_foreground_run, run_foreground
from sim.execution.run_lifecycle.service import await_run, inspect_run, reconcile_stale_run
from sim.execution.run_lifecycle.store import LifecycleStore, _RunLock
from sim.resource_limits import ResourceEstimate

ROOT = Path(__file__).resolve().parents[2]


def test_heartbeat_start_failure_preserves_original_error_and_exited_owner(monkeypatch):
    from types import SimpleNamespace

    from sim.execution.run_lifecycle import runner

    statuses = []
    store = SimpleNamespace(
        create_owner=lambda *args, **kwargs: None,
        heartbeat_owner=lambda *args, **kwargs: statuses.append(kwargs["status"]),
    )

    class UnstartableThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("no thread slot")

        def join(self, **kwargs):
            raise AssertionError("must not join a thread that never started")

    monkeypatch.setattr(runner.threading, "Thread", UnstartableThread)
    heartbeat = runner._OwnerHeartbeat(store, "test-run")
    with pytest.raises(RuntimeError, match="no thread slot"):
        heartbeat.start()
    heartbeat.stop()
    assert statuses == ["exited"]


def _store(tmp_path: Path) -> LifecycleStore:
    return LifecycleStore(
        state_root=tmp_path / "state",
        allowed_output_roots=(tmp_path / "outputs",),
    )


def _identity(store: LifecycleStore, output_dir: Path) -> RunIdentity:
    run_id = str(uuid.uuid4())
    digest = "a" * 64
    return RunIdentity(
        run_id=run_id,
        manifest_ref=store.manifest_ref(run_id, output_dir),
        normalized_config_sha256=digest,
        validation_id=f"test:{digest}",
        source_config_sha256="b" * 64,
        source_config_ref="test.yaml",
        output_dir=str(output_dir),
        resource_profile="laptop-safe",
        resource_plan_sha256="c" * 64,
        engine_version="0.28.0",
        engine_edition="public",
        installation_disposition="developer",
        source_revision="test",
        authorization_disposition="explicit_local_invocation",
        plugin_trust_disposition="trusted_local_cli",
    )


def _create(store: LifecycleStore, output_name: str = "run") -> tuple[str, Path]:
    output = store.prepare_output_dir(output_name)
    manifest = store.create(_identity(store, output))
    return manifest.identity.run_id, output


def _complete(store: LifecycleStore, run_id: str) -> None:
    store.transition(
        run_id,
        RunState.STARTING,
        phase=RunPhase.MATERIALIZING,
        event="starting",
    )
    store.transition(
        run_id,
        RunState.RUNNING,
        phase=RunPhase.EXECUTING,
        event="running",
    )
    store.transition(
        run_id,
        RunState.FINALIZING,
        phase=RunPhase.COLLECTING_ARTIFACTS,
        event="finalizing",
    )
    store.transition(
        run_id,
        RunState.COMPLETED,
        phase=RunPhase.COMMITTING_TERMINAL_STATE,
        event="completed",
    )


def _safe_estimate(_: object) -> ResourceEstimate:
    return ResourceEstimate(
        profile="laptop-safe",
        study_type="single",
        runs=1,
        steps_per_run=120,
        requested_workers=1,
        effective_workers=1,
        active_objects=1,
        plots_enabled=False,
        checkpoint_enabled=True,
        estimated_history_mb_per_run=1.0,
        estimated_parallel_history_mb=1.0,
        estimated_incremental_memory_mb=1.0,
        current_available_memory_mb=1024.0,
        projected_available_memory_mb=1023.0,
        memory_pressure_free_percent=50.0,
        load_per_cpu=0.1,
        acceleration_mode="auto",
        acceleration_backend="numpy",
        risk="safe",
    )


def test_manifest_transitions_are_ordered_and_terminal_is_immutable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, _ = _create(store)
    _complete(store, run_id)

    manifest = store.read_manifest(run_id)
    assert manifest.state is RunState.COMPLETED
    assert manifest.sequence == 5
    assert manifest.terminal_at
    events = store.read_events(run_id)
    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]
    assert [event.event for event in events] == [
        "accepted",
        "starting",
        "running",
        "finalizing",
        "completed",
    ]
    with pytest.raises(InvalidTransitionError):
        store.transition(
            run_id,
            RunState.FAILED,
            phase=RunPhase.COMMITTING_TERMINAL_STATE,
            event="failed",
        )


def test_invalid_transition_is_rejected_without_mutation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, _ = _create(store)
    before = store.read_manifest(run_id)
    with pytest.raises(InvalidTransitionError):
        store.transition(
            run_id,
            RunState.COMPLETED,
            phase=RunPhase.COMMITTING_TERMINAL_STATE,
            event="completed",
        )
    after = store.read_manifest(run_id)
    assert after.manifest_sha256 == before.manifest_sha256
    assert len(store.read_events(run_id)) == 1


def test_await_timeout_is_bounded_and_does_not_mutate_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, _ = _create(store)
    before = store.read_manifest(run_id)
    result = await_run(store, run_id, timeout_s=0.02, poll_interval_s=0.01)
    after = store.read_manifest(run_id)
    assert result.outcome == "still_running"
    assert result.wait_observed_s >= 0.0
    assert before.manifest_sha256 == after.manifest_sha256
    with pytest.raises(ValueError, match="between 0 and 3600"):
        await_run(store, run_id, timeout_s=3600.1)


def test_await_wakes_after_terminal_manifest_commit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, _ = _create(store)

    def finish() -> None:
        time.sleep(0.05)
        _complete(store, run_id)

    worker = threading.Thread(target=finish)
    worker.start()
    result = await_run(store, run_id, timeout_s=2.0, poll_interval_s=0.01)
    worker.join(timeout=2.0)
    assert result.outcome == "terminal"
    assert result.state is not None
    assert result.state["state"] == "completed"
    assert result.event is not None
    assert result.event["manifest_sha256"] == result.state["manifest_sha256"]


def test_any_event_uses_sequence_cursor(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, _ = _create(store)
    store.transition(
        run_id,
        RunState.STARTING,
        phase=RunPhase.MATERIALIZING,
        event="starting",
    )
    result = await_run(
        store,
        run_id,
        timeout_s=0.0,
        wake_condition="any_event",
        after_sequence=1,
    )
    assert result.outcome == "event"
    assert result.event is not None
    assert result.event["sequence"] == 2


def test_lost_owner_wakes_read_only_await_then_reconciles_interrupted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, _ = _create(store)
    store.create_owner(
        run_id,
        owner_token="test-owner",
        pid=999999,
        host_sha256=local_host_sha256(),
        heartbeat_interval_s=1.0,
        stale_after_s=15.0,
    )
    store.transition(
        run_id,
        RunState.STARTING,
        phase=RunPhase.MATERIALIZING,
        event="starting",
    )
    store.transition(
        run_id,
        RunState.RUNNING,
        phase=RunPhase.EXECUTING,
        event="running",
    )

    awakened = await_run(
        store,
        run_id,
        timeout_s=10.0,
        process_alive=lambda _: False,
    )
    assert awakened.outcome == "owner_lost"
    assert awakened.owner is not None
    assert store.read_manifest(run_id).state is RunState.RUNNING

    reconciled = reconcile_stale_run(store, run_id, process_alive=lambda _: False)
    assert reconciled.outcome == "interrupted"
    assert reconciled.run_state_changed is True
    assert reconciled.state is not None
    assert reconciled.state["state"] == "interrupted"
    repeated = reconcile_stale_run(store, run_id, process_alive=lambda _: False)
    assert repeated.outcome == "already_terminal"
    assert repeated.run_state_changed is False


def test_malformed_owner_nonfinite_heartbeat_is_reported_without_raw_json_error(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, output = _create(store)
    store.create_owner(
        run_id,
        owner_token="malformed-owner",
        pid=os.getpid(),
        host_sha256=local_host_sha256(),
        heartbeat_interval_s=1.0,
        stale_after_s=15.0,
    )
    owner_path = output / "lifecycle" / "execution_owner.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    owner["heartbeat_interval_s"] = float("nan")
    owner_path.write_text(json.dumps(owner), encoding="utf-8")

    result = await_run(store, run_id, timeout_s=0.0)

    assert result.outcome == "malformed_state"


def test_reconcile_preserves_state_while_owner_is_alive(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, _ = _create(store)
    store.create_owner(
        run_id,
        owner_token="test-owner",
        pid=123,
        host_sha256=local_host_sha256(),
        heartbeat_interval_s=1.0,
        stale_after_s=15.0,
    )
    result = reconcile_stale_run(store, run_id, process_alive=lambda _: True)
    assert result.outcome == "owner_active"
    assert result.run_state_changed is False
    assert store.read_manifest(run_id).state is RunState.ACCEPTED


def test_reconcile_reclaims_orphaned_transition_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sim.execution.run_lifecycle.store._process_alive", lambda _: False)
    store = _store(tmp_path)
    run_id, output = _create(store)
    store.create_owner(
        run_id,
        owner_token="lost-owner",
        pid=999999,
        host_sha256=local_host_sha256(),
        heartbeat_interval_s=1.0,
        stale_after_s=15.0,
    )
    store.transition(
        run_id,
        RunState.STARTING,
        phase=RunPhase.MATERIALIZING,
        event="starting",
    )
    store.transition(
        run_id,
        RunState.RUNNING,
        phase=RunPhase.EXECUTING,
        event="running",
    )
    lock_path = output / "lifecycle" / ".run.lock"
    lock_path.write_text(
        json.dumps(
            {
                "token": "orphaned-transition",
                "pid": 999999,
                "host_sha256": local_host_sha256(),
                "created_at": "2026-08-27T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    reconciled = reconcile_stale_run(store, run_id, process_alive=lambda _: False)

    assert reconciled.outcome == "interrupted"
    assert reconciled.run_state_changed is True
    assert store.read_manifest(run_id).state is RunState.INTERRUPTED
    assert not lock_path.exists()


def test_live_transition_lock_is_not_reclaimed(tmp_path: Path) -> None:
    lock_path = tmp_path / ".run.lock"
    lock_path.write_text(
        json.dumps(
            {
                "token": "live-transition",
                "pid": os.getpid(),
                "host_sha256": local_host_sha256(),
                "created_at": "2026-08-27T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RunPolicyError, match="Timed out waiting"):
        with _RunLock(lock_path, timeout_s=0.0):
            pass

    assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == "live-transition"


def test_torn_transition_lock_is_reclaimed_after_stable_grace(tmp_path: Path) -> None:
    lock_path = tmp_path / ".run.lock"
    lock_path.write_bytes(b'{"token":"torn"')
    stale = time.time() - 2.0
    os.utime(lock_path, (stale, stale))

    with _RunLock(lock_path, timeout_s=0.5):
        assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] != "torn"

    assert not lock_path.exists()


def test_tampered_manifest_is_reported_as_malformed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, output = _create(store)
    path = output / "lifecycle" / "run_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["state"] = "completed"
    path.write_text(json.dumps(value), encoding="utf-8")
    result = inspect_run(store, run_id)
    assert result.outcome == "malformed_state"
    assert result.state is None


def test_manifest_nonfinite_value_is_reported_as_malformed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, output = _create(store)
    path = output / "lifecycle" / "run_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["execution"]["nonfinite"] = float("nan")
    path.write_text(json.dumps(value), encoding="utf-8")

    result = inspect_run(store, run_id)

    assert result.outcome == "malformed_state"
    assert result.state is None


def test_manifest_invalid_utf8_is_reported_as_malformed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, output = _create(store)
    (output / "lifecycle" / "run_manifest.json").write_bytes(b"\xff\xfe")

    result = inspect_run(store, run_id)

    assert result.outcome == "malformed_state"
    assert result.state is None


def test_event_log_invalid_utf8_is_reported_as_malformed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, output = _create(store)
    (output / "lifecycle" / "run_events.jsonl").write_bytes(b"\xff\xfe")

    result = await_run(store, run_id, timeout_s=0.0, wake_condition="any_event")

    assert result.outcome == "malformed_state"
    assert result.state is None


def test_locator_nonfinite_value_is_reported_as_malformed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, _ = _create(store)
    path = store.locator_root / f"{run_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["updated_at"] = float("nan")
    path.write_text(json.dumps(value), encoding="utf-8")

    result = inspect_run(store, run_id)

    assert result.outcome == "malformed_state"
    assert result.state is None


def test_identity_expectations_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id, _ = _create(store)
    result = inspect_run(store, run_id, expected_normalized_config_sha256="f" * 64)
    assert result.outcome == "identity_mismatch"


def test_output_policy_requires_empty_child_and_rejects_symlinks(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.prepare_output_dir("one")
    with pytest.raises(FileExistsError):
        store.prepare_output_dir("one")
    with pytest.raises(RunPolicyError):
        store.prepare_output_dir(tmp_path / "outside")

    output_root = tmp_path / "outputs"
    real = output_root / "real"
    real.mkdir(parents=True)
    alias = output_root / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(RunPolicyError, match="symlink"):
        store.prepare_output_dir(alias / "run")

    external_alias = tmp_path / "external-alias"
    external_alias.symlink_to(output_root, target_is_directory=True)
    with pytest.raises(RunPolicyError, match="symlink"):
        store.prepare_output_dir(external_alias / "run")


class _SuccessfulExecution:
    def study_type(self, config: Any) -> str:
        return "single_run"

    def run_single(self, config: Any, *, step_callback: Any = None) -> dict[str, object]:
        output = Path(str(config.outputs.output_dir))
        (output / "review").mkdir(parents=True)
        (output / "review" / "run.sqlite").write_bytes(b"sqlite evidence")
        (output / "summary.json").write_text('{"ok": true}\n', encoding="utf-8")
        return {"scenario_name": "automation_smoke", "duration_s": 120.0, "samples": 121}

    def wrap_single_file_payload(
        self,
        *,
        payload: dict[str, object],
        cfg: Any,
        config_path: str | Path,
    ) -> dict[str, object]:
        return payload


class _FailingExecution:
    def study_type(self, config: Any) -> str:
        return "single_run"

    def run_single(self, config: Any, *, step_callback: Any = None) -> dict[str, object]:
        raise RuntimeError("private absolute path must not enter the manifest")

    def wrap_single_file_payload(
        self,
        *,
        payload: dict[str, object],
        cfg: Any,
        config_path: str | Path,
    ) -> dict[str, object]:
        return payload


def _prepared(tmp_path: Path, name: str) -> tuple[LifecycleStore, object]:
    output_root = tmp_path / "outputs"
    store = LifecycleStore(state_root=tmp_path / "state", allowed_output_roots=(output_root,))
    prepared = prepare_foreground_run(
        config_path=ROOT / "configs" / "automation_smoke.yaml",
        output_dir=name,
        output_root=output_root,
        workspace_root=ROOT,
        resource_estimator=_safe_estimate,
    )
    return store, prepared


def test_foreground_runner_commits_handle_artifacts_and_review(tmp_path: Path) -> None:
    store, prepared = _prepared(tmp_path, "success")
    handles = []
    result = run_foreground(
        prepared,
        store=store,
        on_handle=handles.append,
        execution_service=_SuccessfulExecution(),
    )
    assert result.succeeded
    assert handles == [result.handle]
    assert handles[0].state is RunState.ACCEPTED
    assert result.terminal_manifest.sequence == 5
    assert result.terminal_manifest.review["available"] is True
    assert result.terminal_manifest.artifacts["complete"] is True
    assert "config_path" not in result.terminal_manifest.execution
    assert {item["path"] for item in result.terminal_manifest.artifacts["files"]} == {
        "lifecycle/execution_owner.json",
        "review/run.sqlite",
        "summary.json",
    }


def test_foreground_runner_commits_sanitized_failure(tmp_path: Path) -> None:
    store, prepared = _prepared(tmp_path, "failure")
    result = run_foreground(prepared, store=store, execution_service=_FailingExecution())
    assert not result.succeeded
    assert result.terminal_manifest.state is RunState.FAILED
    error = result.terminal_manifest.error
    assert error is not None
    assert error["code"] == "execution_failed"
    assert "private absolute path" not in json.dumps(error)


def test_resource_refusal_happens_before_output_creation(tmp_path: Path) -> None:
    unsafe = replace(_safe_estimate(object()), risk="unsafe")
    output_root = tmp_path / "outputs"
    with pytest.raises(RunPolicyError, match="unsafe"):
        prepare_foreground_run(
            config_path=ROOT / "configs" / "automation_smoke.yaml",
            output_dir="refused",
            output_root=output_root,
            workspace_root=ROOT,
            resource_estimator=lambda _: unsafe,
        )
    assert not (output_root / "refused").exists()


def test_only_laptop_safe_is_supported(tmp_path: Path) -> None:
    with pytest.raises(RunPolicyError, match="laptop-safe"):
        prepare_foreground_run(
            config_path=ROOT / "configs" / "automation_smoke.yaml",
            output_dir="run",
            output_root=tmp_path / "outputs",
            workspace_root=ROOT,
            resource_profile="aggressive",
        )
