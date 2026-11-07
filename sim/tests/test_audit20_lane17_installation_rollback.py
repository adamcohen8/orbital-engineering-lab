"""Audit-20 lane 17 regressions for installer rollback safety."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.installation import manager, state
from sim.installation.desktop import publish_trainer
from sim.installation.paths import InstallationPaths


def test_activate_rollback_does_not_write_through_preexisting_verifier_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locations = InstallationPaths(tmp_path / "data", tmp_path / "config")
    locations.ensure()
    external = tmp_path / "outside.txt"
    external.write_text("outside", encoding="utf-8")
    bundle_file = locations.launcher / "trusted-verifier" / "sim" / "__init__.py"
    bundle_file.parent.mkdir(parents=True)
    bundle_file.symlink_to(external)

    monkeypatch.setattr(
        manager,
        "verify_installation",
        lambda version, *, paths, full: {"status": "official", "version": version},
    )

    def fail_after_snapshot(version: str, *, paths: InstallationPaths) -> dict[str, str]:
        raise RuntimeError("simulated launcher failure")

    monkeypatch.setattr(manager, "write_launchers", fail_after_snapshot)

    with pytest.raises(RuntimeError, match="simulated launcher failure"):
        manager.activate("1.2.3", paths=locations)

    assert external.read_text(encoding="utf-8") == "outside"
    assert not bundle_file.is_symlink()
    assert bundle_file.read_text(encoding="utf-8") == "outside"


def test_desktop_publish_rejects_linked_application_parent(tmp_path: Path) -> None:
    locations = InstallationPaths(tmp_path / "data", tmp_path / "config")
    locations.ensure()
    version_root = locations.version_root("1.2.3")
    version_root.mkdir(parents=True)
    (version_root / "installation-record.json").write_text(
        json.dumps({"current": "1.2.3", "profile": "game"}), encoding="utf-8"
    )
    locations.current_state.write_text(json.dumps({"current": "1.2.3"}), encoding="utf-8")

    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside-applications"
    outside.mkdir()
    (home / "Applications").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="linked desktop launcher parent"):
        publish_trainer(locations, home=home, system="darwin")
    assert not (outside / "RPO Trainer.app").exists()


def test_desktop_publish_rejects_linked_launcher_before_script_write(tmp_path: Path) -> None:
    locations = InstallationPaths(tmp_path / "data", tmp_path / "config")
    locations.ensure()
    version_root = locations.version_root("1.2.3")
    version_root.mkdir(parents=True)
    (version_root / "installation-record.json").write_text(
        json.dumps({"current": "1.2.3", "profile": "game"}), encoding="utf-8"
    )
    locations.current_state.write_text(json.dumps({"current": "1.2.3"}), encoding="utf-8")

    outside = tmp_path / "outside-launcher"
    outside.mkdir()
    locations.launcher.rmdir()
    locations.launcher.symlink_to(outside, target_is_directory=True)

    with pytest.raises((OSError, PermissionError, RuntimeError)):
        publish_trainer(locations, home=tmp_path, system="darwin")
    assert not (outside / "rpo-trainer.py").exists()


def test_desktop_publish_keeps_launcher_script_on_open_directory_after_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locations = InstallationPaths(tmp_path / "data", tmp_path / "config")
    locations.ensure()
    version_root = locations.version_root("1.2.3")
    version_root.mkdir(parents=True)
    (version_root / "installation-record.json").write_text(
        json.dumps({"current": "1.2.3", "profile": "game"}), encoding="utf-8"
    )
    locations.current_state.write_text(json.dumps({"current": "1.2.3"}), encoding="utf-8")

    moved = tmp_path / "launcher-retained"
    outside = tmp_path / "outside-launcher"
    outside.mkdir()
    real_open = state.SecureWriteRoot.open
    swapped = False

    def open_and_swap(cls, path, **kwargs):
        nonlocal swapped
        writer = real_open(path, **kwargs)
        if not swapped and Path(path) == locations.launcher:
            locations.launcher.rename(moved)
            locations.launcher.symlink_to(outside, target_is_directory=True)
            swapped = True
        return writer

    monkeypatch.setattr(state.SecureWriteRoot, "open", classmethod(open_and_swap))
    publish_trainer(locations, home=tmp_path, system="darwin")

    assert swapped
    assert (moved / "rpo-trainer.py").is_file()
    assert not (outside / "rpo-trainer.py").exists()


def test_descriptor_bound_state_write_keeps_parent_identity_after_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    launcher = data_root / "launcher"
    launcher.mkdir(parents=True)
    moved = tmp_path / "launcher-retained"
    outside = tmp_path / "outside-launcher"
    outside.mkdir()
    real_open = state.SecureWriteRoot.open
    swapped = False

    def open_and_swap(cls, path, **kwargs):
        nonlocal swapped
        writer = real_open(path, **kwargs)
        if not swapped and Path(path) == launcher:
            launcher.rename(moved)
            launcher.symlink_to(outside, target_is_directory=True)
            swapped = True
        return writer

    monkeypatch.setattr(state.SecureWriteRoot, "open", classmethod(open_and_swap))
    state.atomic_write_text_descriptor_bound(
        launcher / "oel",
        "stable",
        approved_root=data_root,
    )

    assert swapped
    assert (moved / "oel").read_text(encoding="utf-8") == "stable"
    assert not (outside / "oel").exists()


def test_activate_rollback_does_not_write_through_current_selector_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locations = InstallationPaths(tmp_path / "data", tmp_path / "config")
    locations.ensure()
    external = tmp_path / "current-state.json"
    original = '{"schema_version":"oel.current-installation.v1","current":"0.9.0"}\n'
    external.write_text(original, encoding="utf-8")
    locations.current_state.symlink_to(external)
    monkeypatch.setattr(
        manager,
        "verify_installation",
        lambda version, *, paths, full: {"status": "official", "version": version},
    )
    monkeypatch.setattr(
        manager,
        "write_launchers",
        lambda version, *, paths: (_ for _ in ()).throw(RuntimeError("simulated launcher failure")),
    )

    with pytest.raises(RuntimeError, match="simulated launcher failure"):
        manager.activate("1.2.3", paths=locations)

    assert external.read_text(encoding="utf-8") == original
    assert not locations.current_state.is_symlink()
    assert locations.current_state.read_text(encoding="utf-8") == original
