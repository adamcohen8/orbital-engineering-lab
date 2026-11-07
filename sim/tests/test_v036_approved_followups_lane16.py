from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from sim.execution.run_lifecycle.models import (
    RunIdentity,
    RunPolicyError,
)
from sim.execution.run_lifecycle.store import LifecycleStore
from sim.security.input_reads import secure_read_primitives_available

pytestmark = pytest.mark.skipif(
    not secure_read_primitives_available(),
    reason="descriptor-bound lifecycle paths require the supported POSIX primitives",
)


def _identity(store: LifecycleStore, output: Path) -> RunIdentity:
    run_id = str(uuid.uuid4())
    config_sha256 = "a" * 64
    return RunIdentity(
        run_id=run_id,
        manifest_ref=store.manifest_ref(run_id, output),
        normalized_config_sha256=config_sha256,
        validation_id=f"test:{config_sha256}",
        source_config_sha256="b" * 64,
        source_config_ref="test.yaml",
        output_dir=str(output),
        resource_profile="laptop-safe",
        resource_plan_sha256="c" * 64,
        engine_version="0.36.0",
        engine_edition="public",
        installation_disposition="developer",
        source_revision="test",
        authorization_disposition="explicit_local_invocation",
        plugin_trust_disposition="trusted_local_cli",
    )


def test_store_fails_closed_when_visible_output_root_is_replaced(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    store = LifecycleStore(state_root=tmp_path / "state", allowed_output_roots=(output_root,))
    try:
        os.rename(output_root, tmp_path / "outputs-old")
        output_root.mkdir()
        with pytest.raises(RunPolicyError, match="root identity"):
            store.resolve_output_dir("replacement")
    finally:
        store.close()


def test_store_fails_closed_when_retained_lifecycle_directory_is_replaced(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    store = LifecycleStore(state_root=tmp_path / "state", allowed_output_roots=(output_root,))
    try:
        output = store.prepare_output_dir("run")
        manifest = store.create(_identity(store, output))
        lifecycle = output / "lifecycle"
        os.rename(lifecycle, output / "lifecycle-old")
        lifecycle.mkdir(mode=0o700)
        with pytest.raises(RunPolicyError, match="directory identity"):
            store.read_manifest(manifest.identity.run_id)
    finally:
        store.close()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_store_rejects_symlinked_locator_root_after_open(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    store = LifecycleStore(state_root=tmp_path / "state", allowed_output_roots=(output_root,))
    try:
        output = store.prepare_output_dir("run")
        manifest = store.create(_identity(store, output))
        locator_root = store.locator_root
        os.rename(locator_root, tmp_path / "locators-old")
        locator_root.symlink_to(tmp_path / "locators-old", target_is_directory=True)
        with pytest.raises(RunPolicyError, match="locator root identity"):
            store.read_manifest(manifest.identity.run_id)
    finally:
        store.close()


def test_unbound_lifecycle_json_write_is_rejected() -> None:
    from sim.execution.run_lifecycle import store as lifecycle_store

    with pytest.raises(RunPolicyError, match="retained directory descriptor"):
        lifecycle_store._atomic_write_json(Path("/tmp/unbound-lifecycle.json"), {"ok": True})
