from __future__ import annotations

from pathlib import Path

import sim.project_version as project_version

ROOT = Path(__file__).resolve().parents[2]


class _DistributionFixture:
    def __init__(self, version: str, path: str) -> None:
        self.version = version
        self._path = Path(path)


def test_source_project_version_finds_repository_by_default() -> None:
    assert project_version.source_project_version() == project_version.source_project_version(
        source_root=ROOT
    )


def test_installed_version_prefers_dist_info_over_stale_checkout_egg_info(monkeypatch) -> None:
    monkeypatch.setattr(
        project_version.metadata,
        "distributions",
        lambda **_kwargs: [
            _DistributionFixture("0.25.0", "orbital_engineering_lab.egg-info"),
            _DistributionFixture("0.26.0", "/venv/site-packages/orbital_engineering_lab-0.26.0.dist-info"),
        ],
    )

    assert project_version.installed_project_version() == "0.26.0"


def test_installed_version_preserves_first_dist_info_discovery_order(monkeypatch) -> None:
    monkeypatch.setattr(
        project_version.metadata,
        "distributions",
        lambda **_kwargs: [
            _DistributionFixture("0.30.0", "orbital_engineering_lab.egg-info"),
            _DistributionFixture(
                "0.35.0",
                "/Users/adam/build/worktrees/v0.36.0/.venv/lib/python3.14/site-packages/"
                "orbital_engineering_lab-0.35.0.dist-info",
            ),
            _DistributionFixture(
                "0.34.0",
                "/Users/adam/.venv/lib/python3.14/site-packages/"
                "orbital_engineering_lab-0.34.0.dist-info",
            ),
        ],
    )

    # The first modern entry is the one discovered on the active sys.path;
    # do not replace that precedence with lexicographic path sorting.
    assert project_version.installed_project_version() == "0.35.0"


def test_project_version_status_accepts_matching_source_and_metadata(monkeypatch) -> None:
    source_version = project_version.source_project_version(source_root=ROOT)
    assert source_version
    monkeypatch.setattr(project_version, "installed_project_version", lambda **_kwargs: source_version)

    status = project_version.inspect_project_version(source_root=ROOT)

    assert status.ok
    assert status.required
    assert status.source_version == source_version
    assert status.installed_version == source_version


def test_project_version_status_fails_stale_installed_metadata(monkeypatch) -> None:
    source_version = project_version.source_project_version(source_root=ROOT)
    assert source_version
    monkeypatch.setattr(project_version, "installed_project_version", lambda **_kwargs: "0.0.1")

    status = project_version.inspect_project_version(source_root=ROOT)

    assert not status.ok
    assert status.required
    assert f"source {source_version}" in status.detail
    assert "installed 0.0.1" in status.detail
    assert "pip install -e ." in status.detail


def test_project_version_status_warns_when_source_is_not_installed(monkeypatch) -> None:
    monkeypatch.setattr(project_version, "installed_project_version", lambda **_kwargs: None)

    status = project_version.inspect_project_version(source_root=ROOT)

    assert not status.ok
    assert not status.required
    assert status.source_version
    assert status.installed_version is None
