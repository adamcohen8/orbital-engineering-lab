"""Focused runtime/config audit regressions for the ten-agent review."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from sim.config import scenario_config_from_dict
from sim.execution.run_lifecycle import store as lifecycle_store
from sim.execution.run_lifecycle.models import local_host_sha256
from sim.execution.run_lifecycle.store import LifecycleStore


def _store(tmp_path: Path) -> LifecycleStore:
    return LifecycleStore(
        state_root=tmp_path / "state",
        allowed_output_roots=(tmp_path / "outputs",),
    )


def _create_run(store: LifecycleStore) -> str:
    output = store.prepare_output_dir("run")
    run_id = "4b6f4f4f-9e3f-4e84-8ad6-8fd584d2d7f7"
    digest = "a" * 64
    from sim.execution.run_lifecycle.models import RunIdentity

    store.create(
        RunIdentity(
            run_id=run_id,
            manifest_ref=store.manifest_ref(run_id, output),
            normalized_config_sha256=digest,
            validation_id=f"test:{digest}",
            source_config_sha256="b" * 64,
            source_config_ref="test.yaml",
            output_dir=str(output),
            resource_profile="laptop-safe",
            resource_plan_sha256="c" * 64,
            engine_version="0.28.0",
            engine_edition="public",
            installation_disposition="developer",
            source_revision="test",
            authorization_disposition="explicit_local_invocation",
            plugin_trust_disposition="trusted_local_cli",
        )
    )
    return run_id


@pytest.mark.parametrize("field", ["bundles", "constraints", "correlated_normals"])
def test_monte_carlo_collection_fields_reject_mappings(field: str) -> None:
    raw = {"analysis": {"enabled": True, "monte_carlo": {field: {}}}}
    with pytest.raises(ValueError, match="must be lists"):
        scenario_config_from_dict(raw)


def test_monte_carlo_variation_options_reject_mapping() -> None:
    raw = {
        "analysis": {
            "enabled": True,
            "monte_carlo": {
                "variations": [{
                    "parameter_path": "simulator.dt_s",
                    "mode": "choice",
                    "options": {},
                }],
            },
        }
    }
    with pytest.raises(ValueError, match="options must be a list"):
        scenario_config_from_dict(raw)


def test_monte_carlo_optional_collections_keep_null_as_empty() -> None:
    cfg = scenario_config_from_dict({
        "analysis": {
            "enabled": True,
            "monte_carlo": {
                "bundles": None,
                "constraints": None,
                "correlated_normals": None,
            },
        }
    })
    assert cfg.monte_carlo.bundles == []
    assert cfg.monte_carlo.constraints == []
    assert cfg.monte_carlo.correlated_normals == []


def test_owner_creation_is_single_winner_when_first_write_is_paused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    run_id = _create_run(store)
    first_write_started = threading.Event()
    second_lock_attempted = threading.Event()
    release_first_write = threading.Event()
    counter_lock = threading.Lock()
    lock_attempts = 0
    results: dict[str, str] = {}

    original_write = lifecycle_store._atomic_write_json
    original_enter = lifecycle_store._RunLock.__enter__

    def paused_write(path: Path, payload: dict[str, object]) -> None:
        if path.name == lifecycle_store.OWNER_NAME:
            first_write_started.set()
            assert release_first_write.wait(timeout=5.0)
        original_write(path, payload)

    def observed_enter(lock: lifecycle_store._RunLock) -> lifecycle_store._RunLock:
        nonlocal lock_attempts
        with counter_lock:
            lock_attempts += 1
            if lock_attempts == 2:
                second_lock_attempted.set()
        return original_enter(lock)

    monkeypatch.setattr(lifecycle_store, "_atomic_write_json", paused_write)
    monkeypatch.setattr(lifecycle_store._RunLock, "__enter__", observed_enter)

    def create_owner(token: str) -> str:
        try:
            store.create_owner(
                run_id,
                owner_token=token,
                pid=os.getpid(),
                host_sha256=local_host_sha256(),
                heartbeat_interval_s=1.0,
                stale_after_s=15.0,
            )
        except FileExistsError:
            return "exists"
        return "created"

    def run_owner(token: str) -> None:
        try:
            results[token] = create_owner(token)
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            results[token] = f"error:{type(exc).__name__}"

    first = threading.Thread(target=run_owner, args=("owner-a",), daemon=True)
    first.start()
    assert first_write_started.wait(timeout=2.0)

    second = threading.Thread(target=run_owner, args=("owner-b",), daemon=True)
    second.start()
    try:
        assert second_lock_attempted.wait(timeout=2.0)
    finally:
        release_first_write.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert results == {"owner-a": "created", "owner-b": "exists"}
    assert store.read_owner(run_id).owner_token in {"owner-a", "owner-b"}
