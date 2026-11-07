"""Focused release-builder preservation regressions for the v0.36 audit."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("destination", [
    "http://github.com/adamcohen8/orbital-engineering-lab/releases/download/v0.36.0/public-stable.json",
    "https://evil.invalid/public-stable.json",
    "https://github.com/other/repository/releases/download/v0.36.0/public-stable.json",
    "https://github.com/adamcohen8/orbital-engineering-lab/releases/download/v0.36.0/other.json",
    "https://release-assets.githubusercontent.com/github-production-release-asset/99999999/c12c8816-b6bb-44c9-a165-89fc32fb0ff6",
    "https://release-assets.githubusercontent.com:8443/github-production-release-asset/1163745057/c12c8816-b6bb-44c9-a165-89fc32fb0ff6",
    "https://user:secret@release-assets.githubusercontent.com/github-production-release-asset/1163745057/c12c8816-b6bb-44c9-a165-89fc32fb0ff6",
])
def test_release_redirect_allowlist_rejects_unapproved_destinations(destination):
    from sim.installation.manager import _approved_release_redirect

    source = "https://github.com/adamcohen8/orbital-engineering-lab/releases/latest/download/public-stable.json"
    assert not _approved_release_redirect(source, destination)


def test_public_release_redirects_repin_each_hop_without_forwarding_credentials(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from sim.installation import manager

    source = "https://github.com/adamcohen8/orbital-engineering-lab/releases/latest/download/public-stable.json"
    versioned = source.replace("latest/download", "download/v0.36.0")
    asset = "https://release-assets.githubusercontent.com/github-production-release-asset/1163745057/c12c8816-b6bb-44c9-a165-89fc32fb0ff6?fixture=1"
    pinned, opened = [], []
    response = object()

    class Opener:
        def open(self, request, **kwargs):
            opened.append(request)
            destination = {source: versioned, versioned: asset}.get(request.full_url)
            if destination:
                raise urllib.error.HTTPError(request.full_url, 302, "Found", {"Location": destination}, io.BytesIO())
            return response

    monkeypatch.setattr(manager, "_pinned_addresses", lambda url: pinned.append(url) or [object()])
    monkeypatch.setattr(manager, "_PinnedHTTPSHandler", lambda *a, **k: object())
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())
    request = urllib.request.Request(source, headers={"Range": "bytes=3-", "Authorization": "fixture", "Cookie": "fixture"})
    assert manager._open_pinned_url(request, timeout=10) is response
    assert pinned == [source, versioned, asset]
    assert opened[-1].get_header("Range") == "bytes=3-"
    assert opened[-1].get_header("Authorization") is None
    assert opened[-1].get_header("Cookie") is None


def test_release_transport_keeps_a_hard_redirect_limit(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from sim.installation import manager

    opened = []

    class Opener:
        def open(self, request, **kwargs):
            opened.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, 302, "Found", {"Location": request.full_url}, io.BytesIO())

    monkeypatch.setattr(manager, "_pinned_addresses", lambda url: [object()])
    monkeypatch.setattr(manager, "_PinnedHTTPSHandler", lambda *a, **k: object())
    monkeypatch.setattr(manager, "_approved_release_redirect", lambda *a: True)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())
    with pytest.raises(manager.ContractError, match="redirect limit"):
        manager._open_pinned_url(urllib.request.Request("https://github.com/fixture"), timeout=10)
    assert len(opened) == 4


def test_export_without_own_git_metadata_does_not_claim_parent_commit(tmp_path, monkeypatch):
    from tools import build_installable_release as builder

    source = tmp_path / "export"
    source.mkdir()
    monkeypatch.setattr(builder.subprocess, "run", lambda *a, **k: pytest.fail("parent Git discovery is forbidden"))
    assert builder._source_commit(source) is None


@pytest.mark.parametrize("corruption", ["digest", "size", "boolean", "base64", "trailing", "truncated", "limit", "fields"])
def test_inventory_transport_rejects_malformed_or_unbounded_streams(corruption):
    import base64

    from sim.installation.integrity_transport import (
        MAX_INTEGRITY_BYTES,
        pack_installation_integrity,
        unpack_installation_integrity,
    )

    packed = pack_installation_integrity({"inventory": "x" * 10000})
    if corruption == "digest":
        packed["sha256"] = "0" * 64
    elif corruption == "size":
        packed["bytes"] = 1
    elif corruption == "boolean":
        packed["bytes"] = True
    elif corruption == "base64":
        packed["data"] = "!"
    elif corruption == "trailing":
        packed["data"] = base64.b64encode(base64.b64decode(packed["data"]) + b"trailing").decode()
    elif corruption == "truncated":
        packed["data"] = base64.b64encode(base64.b64decode(packed["data"])[:-1]).decode()
    elif corruption == "limit":
        packed["bytes"] = MAX_INTEGRITY_BYTES + 1
    else:
        packed["extra"] = True
    with pytest.raises(ValueError):
        unpack_installation_integrity(packed)


def test_profile_wheel_closure_scopes_extras_and_revisits_expanded_dependencies(tmp_path):
    from tools.build_installable_release import _profile_wheel_closures

    def wheel(name, requirements=()):
        path = tmp_path / f"{name}-1.0-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"{name}-1.0.dist-info/METADATA",
                f"Name: {name}\nVersion: 1.0\n"
                + "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements),
            )
        return path

    project = wheel("orbital_engineering_lab", [
        'bridge; extra == "full"',
        'middleware; extra == "full"',
    ])
    middleware = wheel("middleware", [
        'unrequested; extra == "full"',
        'token; extra == "secure"',
        "common",
    ])
    bridge = wheel("bridge", ["middleware[secure]"])
    common = wheel("common")
    token = wheel("token")

    closures = _profile_wheel_closures([project, middleware, bridge, common, token], ["core", "full"])
    assert closures["core"] == [project]
    assert set(closures["full"]) == {project, middleware, bridge, common, token}

    with pytest.raises(ValueError, match="missing from the signed wheelhouse: token"):
        _profile_wheel_closures([project, middleware, bridge, common], ["full"])


def test_project_wheel_build_uses_source_inventory_stage_without_checkout_writes(tmp_path, monkeypatch):
    from tools import build_installable_release as builder

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname = "orbital-engineering-lab"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    (source / "tracked.py").write_text("tracked = True\n", encoding="utf-8")
    (source / "untracked.py").write_text("should_not_be_staged = True\n", encoding="utf-8")
    build_dir = source / "build"
    build_dir.mkdir()
    sentinel = build_dir / "sentinel.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")

    monkeypatch.setattr(
        builder,
        "_source_files",
        lambda root: [root / "pyproject.toml", root / "tracked.py"],
    )

    observed: dict[str, Path] = {}

    def fake_run(command, *, cwd, **_kwargs):
        stage = Path(cwd)
        observed["stage"] = stage
        assert stage != source
        assert (stage / "pyproject.toml").is_file()
        assert (stage / "tracked.py").is_file()
        assert not (stage / "untracked.py").exists()
        assert not (stage / "build" / "sentinel.txt").exists()
        wheel_dir = Path(command[command.index("--wheel-dir") + 1])
        wheel_dir.mkdir(parents=True, exist_ok=True)
        wheel = wheel_dir / "orbital_engineering_lab-1.0.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                "orbital_engineering_lab-1.0.0.dist-info/METADATA",
                "Name: orbital-engineering-lab\nVersion: 1.0.0\n",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    destination = tmp_path / "release" / "wheelhouse"
    result = builder._build_project_wheel(source, destination, version="1.0.0")

    assert result == destination / "orbital_engineering_lab-1.0.0-py3-none-any.whl"
    assert observed["stage"] != source
    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
    assert (source / "untracked.py").read_text(encoding="utf-8") == "should_not_be_staged = True\n"


def test_official_runtime_provisioning_uses_exact_closed_wheel_command_and_clean_env(tmp_path, monkeypatch):
    from sim.installation import manager

    source = tmp_path / "source"
    source.mkdir()
    runtime = tmp_path / "runtime"
    wheel = tmp_path / "orbital_engineering_lab-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "orbital_engineering_lab-1.0.0.dist-info/METADATA",
            "Name: orbital-engineering-lab\nVersion: 1.0.0\n",
        )

    class FakeEnvBuilder:
        def __init__(self, **_kwargs):
            pass

        def create(self, root):
            python = Path(root) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("provider\n", encoding="utf-8")

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "No broken requirements found.\n", "")

    monkeypatch.setattr(manager.venv, "EnvBuilder", FakeEnvBuilder)
    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "audit-metadata"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "audit-python"))
    monkeypatch.setenv("PIP_INDEX_URL", "https://evil.example/simple")

    result = manager._install_runtime(
        source,
        runtime,
        profile="core",
        constraints=None,
        signed_wheels=[wheel],
        signed_wheel_hashes={wheel.name: manager.sha256_file(wheel)},
    )

    install_command = calls[0][0]
    environment = calls[0][1]["env"]
    assert install_command[1:3] == ["-I", "-B"]
    assert "--no-deps" in install_command
    assert "--no-index" in install_command
    assert "--index-url" not in install_command
    assert "--require-hashes" in install_command
    assert f"{wheel.resolve().as_uri()}#sha256={manager.sha256_file(wheel)}" in install_command
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "PIP_INDEX_URL" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert result["install_command"] == install_command

    with pytest.raises(manager.ContractError, match="wheel closure is empty"):
        manager._install_runtime(
            source,
            tmp_path / "empty-runtime",
            profile="core",
            constraints=None,
            signed_wheels=[],
        )


def test_signed_setuptools_is_preserved_but_venv_seed_is_removed(tmp_path):
    from sim.installation import manager

    def populate(root: Path) -> tuple[Path, Path]:
        site = root / "lib" / "python3.11" / "site-packages"
        site.mkdir(parents=True)
        scripts = root / "bin"
        scripts.mkdir(parents=True)
        for package in ("setuptools", "pkg_resources", "_distutils_hack", "pip"):
            (site / package).mkdir()
            (site / package / "__init__.py").write_text("# package\n", encoding="utf-8")
        for distribution in ("setuptools-83.0.0.dist-info", "pip-25.0.dist-info"):
            metadata = "Name: setuptools\nVersion: 83.0.0\n" if distribution.startswith("setuptools") else "Name: pip\nVersion: 25.0\n"
            path = site / distribution
            path.mkdir()
            (path / "METADATA").write_text(metadata, encoding="utf-8")
        for name in ("distutils-precedence.pth", "easy-install.pth"):
            (site / name).write_text("import _distutils_hack\n", encoding="utf-8")
        for name in ("activate", "pip", "pip3", "easy_install", "easy_install-3.11"):
            (scripts / name).write_text("#!/bin/sh\n", encoding="utf-8")
        return site, scripts

    preserved_root = tmp_path / "preserved"
    site, scripts = populate(preserved_root)
    manager._remove_bootstrap_runtime_packages(
        preserved_root,
        preserve_distributions={"setuptools"},
    )
    manager._remove_unbound_runtime_convenience(
        preserved_root,
        preserve_distributions={"setuptools"},
    )
    assert (site / "setuptools").is_dir()
    assert (site / "pkg_resources").is_dir()
    assert (site / "_distutils_hack").is_dir()
    assert (site / "setuptools-83.0.0.dist-info").is_dir()
    assert (site / "distutils-precedence.pth").is_file()
    assert (site / "easy-install.pth").is_file()
    assert not (site / "pip").exists()
    assert not (site / "pip-25.0.dist-info").exists()
    assert not (scripts / "pip").exists()
    assert (scripts / "easy_install").is_file()
    assert (scripts / "easy_install-3.11").is_file()
    assert not (scripts / "activate").exists()

    seeded_root = tmp_path / "seeded"
    seeded_site, seeded_scripts = populate(seeded_root)
    manager._remove_bootstrap_runtime_packages(seeded_root)
    manager._remove_unbound_runtime_convenience(seeded_root)
    assert not (seeded_site / "setuptools").exists()
    assert not (seeded_site / "pip").exists()
    assert not (seeded_site / "setuptools-83.0.0.dist-info").exists()
    assert not (seeded_scripts / "easy_install").exists()
