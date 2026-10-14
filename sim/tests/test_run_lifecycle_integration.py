from __future__ import annotations

import ast
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sim.execution.run_lifecycle.models import RunPhase, RunState, local_host_sha256
from sim.execution.run_lifecycle.runner import prepare_foreground_run, run_foreground
from sim.execution.run_lifecycle.service import await_run, reconcile_stale_run
from sim.execution.run_lifecycle.store import LifecycleStore
from sim.execution.service import SimulationExecutionService
from sim.installation.cli import _dispatch_commands
from sim.resource_limits import ResourceEstimate

ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE = ROOT / "sim" / "execution" / "run_lifecycle"
SCHEMAS = LIFECYCLE / "schemas"
CONFIG = ROOT / "configs" / "automation_smoke.yaml"


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


def _schema(name: str) -> dict[str, object]:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def _run_metadata(path: Path) -> tuple[object, ...]:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT scenario_name, duration_s, dt_s, samples FROM run_metadata"
        ).fetchone()
    assert row is not None
    return tuple(row)


def test_real_engine_lifecycle_matches_direct_execution_evidence(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    store = LifecycleStore(state_root=tmp_path / "state", allowed_output_roots=(output_root,))
    lifecycle = prepare_foreground_run(
        config_path=CONFIG,
        output_dir="lifecycle",
        output_root=output_root,
        workspace_root=ROOT,
        resource_estimator=_safe_estimate,
    )
    result = run_foreground(lifecycle, store=store, execution_service=SimulationExecutionService())
    assert result.succeeded

    direct = prepare_foreground_run(
        config_path=CONFIG,
        output_dir="direct",
        output_root=output_root,
        workspace_root=ROOT,
        resource_estimator=_safe_estimate,
    )
    direct.output_dir.mkdir(parents=True)
    SimulationExecutionService().run_single(direct.config)

    lifecycle_db = lifecycle.output_dir / "review" / "run.sqlite"
    direct_db = direct.output_dir / "review" / "run.sqlite"
    assert _run_metadata(lifecycle_db) == _run_metadata(direct_db)
    assert result.terminal_manifest.review["available"] is True
    owner = json.loads(
        (lifecycle.output_dir / "lifecycle" / "execution_owner.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(_schema("execution_owner.schema.json")).validate(owner)
    assert owner["status"] == "exited"


def test_real_process_loss_wakes_and_reconciles(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    store = LifecycleStore(state_root=tmp_path / "state", allowed_output_roots=(output_root,))
    from sim.tests.test_run_lifecycle import _identity

    output = store.prepare_output_dir("lost-process")
    manifest = store.create(_identity(store, output))
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        store.create_owner(
            manifest.identity.run_id,
            owner_token="real-process-owner",
            pid=sleeper.pid,
            host_sha256=local_host_sha256(),
            heartbeat_interval_s=1.0,
            stale_after_s=15.0,
        )
        store.transition(
            manifest.identity.run_id,
            RunState.STARTING,
            phase=RunPhase.MATERIALIZING,
            event="starting",
        )
        store.transition(
            manifest.identity.run_id,
            RunState.RUNNING,
            phase=RunPhase.EXECUTING,
            event="running",
        )
        sleeper.terminate()
        sleeper.wait(timeout=10.0)
        awakened = await_run(store, manifest.identity.run_id, timeout_s=2.0)
        assert awakened.outcome == "owner_lost"
        reconciled = reconcile_stale_run(store, manifest.identity.run_id)
        assert reconciled.outcome == "interrupted"
        Draft202012Validator(_schema("reconcile_result.schema.json")).validate(
            reconciled.to_dict()
        )
    finally:
        if sleeper.poll() is None:
            sleeper.terminate()
            sleeper.wait(timeout=10.0)


@pytest.mark.parametrize("config_name", ["automation_smoke.yaml", "mars_reference_drag.yaml"])
def test_cli_jsonl_start_await_inspect_and_events(tmp_path: Path, config_name: str) -> None:
    output_root = tmp_path / "outputs"
    state_root = tmp_path / "state"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.execution.run_lifecycle",
            "start",
            "--config",
            str(ROOT / "configs" / config_name),
            "--output-dir",
            "cli-run",
            "--output-root",
            str(output_root),
            "--state-root",
            str(state_root),
            "--jsonl",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(records) == 2
    handle, terminal = records
    Draft202012Validator(_schema("run_handle.schema.json")).validate(handle)
    Draft202012Validator(_schema("run_handle.schema.json")).validate(terminal["handle"])
    Draft202012Validator(_schema("run_state.schema.json")).validate(terminal["state"])
    assert terminal["state"]["state"] == "completed"
    if config_name == "mars_reference_drag.yaml":
        with sqlite3.connect(output_root / "cli-run" / "review" / "run.sqlite") as connection:
            assert connection.execute("SELECT COUNT(*) FROM planetary_state").fetchone()[0] > 0
            assert connection.execute("SELECT COUNT(*) FROM object_state").fetchone()[0] == 0

    common = ["--output-root", str(output_root), "--state-root", str(state_root)]
    await_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.execution.run_lifecycle",
            "await",
            handle["run_id"],
            "--timeout",
            "0",
            "--expect-config-sha256",
            handle["normalized_config_sha256"],
            "--expect-manifest-ref",
            handle["manifest_ref"],
            *common,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert await_completed.returncode == 0
    awaited = json.loads(await_completed.stdout)
    Draft202012Validator(_schema("await_result.schema.json")).validate(awaited)
    assert awaited["outcome"] == "terminal"
    assert awaited["run_state_changed"] is False

    for command in ("inspect", "events", "reconcile"):
        observed = subprocess.run(
            [
                sys.executable,
                "-m",
                "sim.execution.run_lifecycle",
                command,
                handle["run_id"],
                *common,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert observed.returncode == 0, observed.stderr or observed.stdout
        value = json.loads(observed.stdout)
        assert value["run_state_changed"] is False
    assert value["outcome"] == "already_terminal"

    events_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.execution.run_lifecycle",
            "events",
            handle["run_id"],
            *common,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert len(json.loads(events_completed.stdout)["events"]) == 5


def test_provider_neutral_reference_client_accepts_and_wakes(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "python" / "run_lifecycle_client.py"),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(tmp_path / "outputs" / "reference-run"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--state-root",
            str(tmp_path / "state"),
            "--timeout",
            "30",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60.0,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [record["event"] for record in records] == ["accepted", "wake"]
    assert records[1]["result"]["outcome"] == "terminal"


def test_lifecycle_core_has_no_model_provider_or_transport_dependency() -> None:
    prohibited = {"anthropic", "openai", "mcp", "grok", "subprocess"}
    imported: set[str] = set()
    for path in LIFECYCLE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint(prohibited)
    assert "runs" in _dispatch_commands()


def test_all_lifecycle_schemas_are_valid_draft_2020_12() -> None:
    paths = sorted(SCHEMAS.glob("*.json"))
    assert len(paths) == 7
    for path in paths:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))
