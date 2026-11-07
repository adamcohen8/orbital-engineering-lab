"""Transactional side-by-side installation and update management for OEL."""

from __future__ import annotations

import csv
import http.client
import importlib.util
import ipaddress
import json
import os
import platform
import re
import shlex
import shutil
import socket
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import venv
import zipfile
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from sim.project_version import source_project_version

from .archive import safe_extract
from .contracts import (
    CHANNEL_CONFIG_SCHEMA,
    CHANNEL_INDEX_SCHEMA,
    INSTALLATION_RECORD_SCHEMA,
    UPDATE_RECEIPT_SCHEMA,
    ContractError,
    canonical_json_bytes,
    load_json_object,
    reject_unknown_keys,
    release_manifest_digest,
    require_keys,
    sha256_bytes,
    sha256_file,
    sha256_tree,
    validate_release_manifest,
    validate_version,
    version_satisfies,
    version_tuple,
)
from .integrity import (
    _rebind_runtime_scripts,
    validate_installation_integrity,
    verify_runtime_inventory,
    verify_source_inventory,
)
from .paths import InstallationPaths
from .signing import RSAPublicKey, load_release_public_keys, verify_payload
from .state import (
    StateLock,
    atomic_write_json,
    atomic_write_json_descriptor_bound,
    atomic_write_text,
    atomic_write_text_descriptor_bound,
    empty_installation_state,
    read_state,
    recover_stale_lock,
    secure_state_write_available,
    unlink_descriptor_bound,
)

MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_RELEASE_BYTES = 8 * 1024 * 1024 * 1024
TRUSTED_LAUNCHER_MARKER = "oel-trusted-dispatcher:v1"
_PYTHON_PROVISIONING_VARS = {
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONUSERBASE",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
    "PYTHONEXECUTABLE",
    "PYTHONWARNINGS",
    "PYTHONBREAKPOINT",
    "PYTHONPROFILEIMPORTTIME",
}
_TRUSTED_BUNDLE_MODULES = (
    "contracts.py",
    "signing.py",
    "integrity.py",
    "integrity_transport.py",
    "trusted_verifier.py",
)


def _trusted_bundle_paths(locations: InstallationPaths) -> tuple[Path, ...]:
    bundle = locations.launcher / "trusted-verifier"
    return (
        locations.launcher / "trusted-dispatcher.py",
        bundle / "sim" / "__init__.py",
        bundle / "sim" / "schema_versions.py",
        bundle / "sim" / "installation" / "__init__.py",
        *(bundle / "sim" / "installation" / name for name in _TRUSTED_BUNDLE_MODULES),
    )


def _trusted_external_python(locations: InstallationPaths) -> Path:
    """Select the operator-owned Python used before managed imports."""

    candidate = Path(getattr(sys, "_base_executable", sys.executable)).expanduser().absolute()
    try:
        candidate.resolve(strict=False).relative_to(locations.versions.resolve(strict=False))
    except ValueError:
        pass
    else:
        raise ContractError(
            "Cannot write a launcher with an external verifier provider: "
            "the active Python is inside the managed version roots."
        )
    if not candidate.is_file():
        raise ContractError(f"External verifier Python provider is not a regular file: {candidate}")
    return candidate


def _install_trusted_verifier_bundle(locations: InstallationPaths) -> Path:
    """Copy the reviewed verifier helpers outside version roots atomically."""

    bundle = locations.launcher / "trusted-verifier"
    source_installation = Path(__file__).resolve().parent
    source_sim = source_installation.parent
    source_files: dict[Path, Path | None] = {
        bundle / "sim" / "__init__.py": None,
        bundle / "sim" / "schema_versions.py": source_sim / "schema_versions.py",
        bundle / "sim" / "installation" / "__init__.py": None,
    }
    source_files.update(
        {
            bundle / "sim" / "installation" / name: source_installation / name
            for name in _TRUSTED_BUNDLE_MODULES
        }
    )
    for destination, source in source_files.items():
        if destination.is_symlink():
            raise ContractError(f"Trusted verifier bundle may not contain symbolic links: {destination}")
        text = "" if source is None else source.read_text(encoding="utf-8")
        atomic_write_text(destination, text)
    return bundle


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _provisioning_environment(*, offline: bool) -> dict[str, str]:
    """Return an isolated environment for venv/pip provisioning subprocesses."""

    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("PIP_") or name in _PYTHON_PROVISIONING_VARS:
            environment.pop(name, None)
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if offline:
        environment["PIP_NO_INDEX"] = "1"
    return environment


def pro_installation_available() -> bool:
    """Return whether this edition contains the private Pro license verifier."""

    try:
        spec = importlib.util.find_spec("sim.licensing.offline")
    except ModuleNotFoundError:
        return False
    if spec is None or spec.origin is None:
        return False

    # An editable private checkout can coexist with a generated public export.
    # Only advertise Pro when the verifier belongs to this installed OEL tree;
    # never borrow a private module discovered through another import path.
    package_root = Path(__file__).resolve().parents[1]
    try:
        return Path(spec.origin).resolve().is_relative_to(package_root)
    except (OSError, RuntimeError, ValueError):
        return False


def _python_in(runtime: Path) -> Path:
    return runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _find_source_root(extracted: Path) -> Path:
    direct = extracted / "pyproject.toml"
    if direct.is_file():
        return extracted
    candidates = sorted(path.parent for path in extracted.glob("*/pyproject.toml"))
    if len(candidates) != 1:
        raise ContractError("Release artifact must contain exactly one OEL source root with pyproject.toml.")
    return candidates[0]


def _archive_source_version(archive: Path) -> str | None:
    """Read the single source ``pyproject.toml`` without extracting code."""

    raw: bytes | None = None
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as bundle:
            members = [
                member
                for member in bundle.getmembers()
                if member.isfile()
                and PurePosixPath(member.name).name == "pyproject.toml"
                and len(PurePosixPath(member.name).parts) <= 2
            ]
            if len(members) != 1:
                return None
            stream = bundle.extractfile(members[0])
            if stream is not None:
                raw = stream.read(1024 * 1024 + 1)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            names = [
                name
                for name in bundle.namelist()
                if not name.endswith("/")
                and PurePosixPath(name).name == "pyproject.toml"
                and len(PurePosixPath(name).parts) <= 2
            ]
            if len(names) != 1:
                return None
            raw = bundle.read(names[0], 1024 * 1024 + 1)
    if raw is None or len(raw) > 1024 * 1024:
        return None
    in_project = False
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError:
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("version") and "=" in line:
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def _artifact_path(manifest_path: Path, artifact: Mapping[str, Any]) -> Path:
    manifest_root = manifest_path.parent.resolve()
    text = str(artifact.get("path", "") or "").strip()
    candidates: list[Path] = []
    if text:
        path = Path(text)
        if path.is_absolute() or ".." in path.parts:
            raise ContractError("Release artifact paths must be relative to the manifest directory.")
        candidates.append(manifest_root.joinpath(*path.parts))
    name = str(artifact.get("name", "") or "").strip()
    if name:
        if Path(name).name != name:
            raise ContractError("Release artifact name must be a plain file name.")
        candidates.append(manifest_root / name)
    for candidate in candidates:
        # A signed manifest must bind to a regular file in its own directory
        # tree.  Checking the lexical path and every component before
        # resolving prevents a pre-existing symlink from redirecting a wheel
        # or source archive outside the signed bundle.
        try:
            relative = candidate.relative_to(manifest_root)
        except ValueError as exc:
            raise ContractError("Release artifact path escapes the manifest directory.") from exc
        current = manifest_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ContractError(f"Release artifact path may not contain a symbolic link: {candidate}")
        resolved = candidate.resolve()
        if resolved != candidate:
            raise ContractError(f"Release artifact path is not canonical: {candidate}")
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(item) for item in candidates) or "no local candidates"
    raise FileNotFoundError(f"Release artifact was not found ({searched}).")


def _record_path(root: Path, raw: Any, *, label: str) -> Path:
    """Resolve a recorded installation path without following path escapes."""

    text = str(raw or "").strip()
    if not text:
        raise ContractError(f"{label} path is missing.")
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"{label} path must be relative to the installation root.")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{label} path may not contain a symbolic link: {candidate}")
    resolved = candidate.resolve()
    if resolved != candidate:
        raise ContractError(f"{label} path is not canonical: {candidate}")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{label} path escapes the installation root.") from exc
    return candidate


def _select_source_artifact(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    host_platform = platform.system().lower()
    host_machine = platform.machine().lower().replace("amd64", "x86_64").replace("aarch64", "arm64")
    candidates: list[tuple[int, Mapping[str, Any]]] = []
    for item in manifest["artifacts"]:
        if item.get("kind") not in {"source", "source_bundle"}:
            continue
        target_platform = str(item.get("platform", "") or "").lower()
        target_machine = str(item.get("architecture", "") or "").lower().replace("amd64", "x86_64").replace("aarch64", "arm64")
        if target_platform and target_platform != host_platform:
            continue
        if target_machine and target_machine != host_machine:
            continue
        candidates.append((int(bool(target_platform)) + int(bool(target_machine)), item))
    if not candidates:
        raise ContractError(f"Release manifest has no source artifact for {platform.system()}/{platform.machine()}.")
    best_score = max(score for score, _ in candidates)
    selected = [item for score, item in candidates if score == best_score]
    if len(selected) != 1:
        raise ContractError("Release manifest source artifact selection is ambiguous for this host.")
    return selected[0]


def _validate_host_compatibility(manifest: Mapping[str, Any]) -> None:
    supported = {str(item).lower() for item in manifest.get("platforms", [])}
    if supported and platform.system().lower() not in supported:
        raise ContractError(f"OEL {manifest['version']} does not support host platform {platform.system()}.")
    expected_arch = str(manifest.get("architecture", "") or "").lower().replace("amd64", "x86_64").replace("aarch64", "arm64")
    host_arch = platform.machine().lower().replace("amd64", "x86_64").replace("aarch64", "arm64")
    if expected_arch and expected_arch != "any" and expected_arch != host_arch:
        raise ContractError(f"OEL {manifest['version']} targets architecture {expected_arch}, not host {host_arch}.")
    requires = str(dict(manifest.get("python", {}) or {}).get("requires", "") or "")
    host_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if requires and not version_satisfies(host_python, requires):
        raise ContractError(f"OEL {manifest['version']} requires Python {requires}; this interpreter is {host_python}.")


def _validate_offline_runtime_compatibility(manifest: Mapping[str, Any]) -> None:
    qualification = dict(
        dict(manifest.get("supply_chain", {}) or {}).get("offline_runtime_qualification", {}) or {}
    )
    qualified_python = str(qualification.get("python", "") or "")
    if not qualified_python:
        return
    qualified = version_tuple(qualified_python)
    if len(qualified) < 2:
        raise ContractError("Offline bundle runtime qualification has an invalid Python version.")
    required_minor = qualified[:2]
    host_minor = (sys.version_info.major, sys.version_info.minor)
    if required_minor != host_minor:
        required_tag = f"py{required_minor[0]}{required_minor[1]}"
        host_tag = f"py{host_minor[0]}{host_minor[1]}"
        raise ContractError(
            f"Offline bundle was qualified for CPython {required_minor[0]}.{required_minor[1]} "
            f"({required_tag}), not this interpreter's CPython {host_minor[0]}.{host_minor[1]} ({host_tag}). "
            "Use the bundle whose Python tag matches the installing OEL launcher."
        )


def verify_release_manifest(
    value: Mapping[str, Any],
    *,
    public_keys: Mapping[str, RSAPublicKey] | None,
    require_signature: bool = True,
    require_integrity: bool = False,
) -> dict[str, Any]:
    manifest = validate_release_manifest(value)
    signature = manifest.get("signature")
    if require_signature and not signature:
        raise ContractError("Official release manifests must be signed.")
    if signature and (not public_keys or not verify_payload(manifest, public_keys)):
        raise ContractError("Release manifest signature verification failed.")
    if require_integrity:
        try:
            validate_installation_integrity(
                dict(dict(manifest.get("supply_chain", {}) or {}).get("installation_integrity", {}) or {})
            )
        except ContractError as exc:
            raise ContractError(
                "Official release manifest is missing authenticated installation integrity; "
                "legacy releases require explicit re-verification/migration."
            ) from exc
    return manifest


def verify_release_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    actual_size = path.stat().st_size
    expected_size = int(artifact["bytes"])
    if actual_size != expected_size:
        raise ContractError(f"Release artifact size mismatch: expected {expected_size}, got {actual_size}.")
    actual_digest = sha256_file(path)
    if actual_digest != artifact["sha256"]:
        raise ContractError(f"Release artifact SHA-256 mismatch: expected {artifact['sha256']}, got {actual_digest}.")


def _install_runtime(
    source_root: Path,
    runtime_root: Path,
    *,
    profile: str,
    constraints: Path | None,
    offline_wheelhouse: Path | None = None,
    signed_wheels: list[Path] | None = None,
    signed_wheel_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if signed_wheels is not None and not signed_wheels:
        raise ContractError("Signed runtime wheel closure is empty.")
    signed_closure = signed_wheels is not None
    requirements = [str(source_root) if profile == "core" else f"{source_root}[{profile}]"]
    if signed_wheels is not None:
        wheel_names = [wheel.name for wheel in signed_wheels]
        if (
            signed_wheel_hashes is None
            or len(set(wheel_names)) != len(wheel_names)
            or set(signed_wheel_hashes) != set(wheel_names)
        ):
            raise ContractError("Signed runtime wheel hashes must exactly match the selected wheel closure.")
        requirements = []
        for wheel in signed_wheels:
            digest = signed_wheel_hashes[wheel.name]
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ContractError(f"Signed runtime wheel has an invalid SHA-256 digest: {wheel.name!r}")
            # Use the authenticated inventory digest, never a new hash of local
            # bytes: pip must reject wheels changed since release verification.
            requirements.append(f"{wheel.resolve().as_uri()}#sha256={digest}")
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt").create(runtime_root)
    python = _python_in(runtime_root)
    command = [str(python), "-I", "-B", "-m", "pip", "install", "--no-compile", "--only-binary=:all:"]
    if signed_closure:
        # Every selected wheel is already authenticated against the signed
        # manifest before this call.  No resolver or index may introduce
        # another package into an official runtime.
        command.extend(["--no-deps", "--no-index", "--require-hashes"])
    if offline_wheelhouse is not None:
        if not offline_wheelhouse.is_dir() or not any(offline_wheelhouse.iterdir()):
            raise ContractError("Offline installation requires a non-empty verified wheelhouse.")
        command.extend(["--no-index", "--find-links", str(offline_wheelhouse)])
    elif not signed_closure:
        command.extend(["--index-url", "https://pypi.org/simple"])
    # Official runtimes use the exact authenticated, hash-pinned wheel closure
    # with no dependency resolution. Source constraints apply only to developer
    # resolution; they must not substitute another version for a signed wheel.
    if constraints is not None and not signed_closure:
        command.extend(["-c", str(constraints)])
    command.extend(str(requirement) for requirement in requirements)
    environment = _provisioning_environment(offline=offline_wheelhouse is not None or signed_closure)
    completed = subprocess.run(command, cwd=source_root, env=environment, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"OEL runtime installation failed:\n{completed.stdout}\n{completed.stderr}".strip())
    check = subprocess.run(
        [str(python), "-I", "-B", "-m", "pip", "check"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(f"Installed dependency graph failed pip check:\n{check.stdout}\n{check.stderr}".strip())
    signed_distributions = {
        name
        for wheel in (signed_wheels or [])
        if (name := _wheel_distribution_name(wheel)) is not None
    }
    _remove_bootstrap_runtime_packages(runtime_root, preserve_distributions=signed_distributions)
    _remove_unbound_distribution_metadata(runtime_root)
    _remove_bytecode_caches(runtime_root)
    _remove_unbound_runtime_convenience(runtime_root, preserve_distributions=signed_distributions)
    return {
        "python": str(python),
        "install_command": command,
        "pip_check": check.stdout.strip() or "No broken requirements found.",
    }


def _normalized_distribution_name(value: str) -> str:
    """Normalize a wheel distribution identity for closure comparisons."""

    return re.sub(r"[-_.]+", "-", value).lower()


def _wheel_distribution_name(path: Path) -> str | None:
    """Read a wheel's distribution identity when a verified wheel is available.

    Official callers have already authenticated the wheel archive.  Returning
    ``None`` for an invalid direct helper input keeps the cleanup helper usable
    by developer/test fixtures while ensuring no unverified package is retained
    merely because its filename looks like a bootstrap distribution.
    """

    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                return None
            metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError):
        return None
    name = str(metadata.get("Name", "")).strip()
    return _normalized_distribution_name(name) if name else None


def _remove_bootstrap_runtime_packages(
    root: Path,
    *,
    preserve_distributions: Iterable[str] = (),
) -> None:
    """Remove venv-seeded package managers absent from the signed closure.

    A full profile may intentionally carry a signed ``setuptools`` wheel, so
    package-manager files are removed only when their normalized distribution
    is absent from the selected authenticated wheel closure.  The default
    empty closure preserves the historical cleanup behavior for developer
    fixtures and unsigned source installations.
    """

    candidates = sorted((root / "lib").glob("python*/site-packages"))
    candidates.append(root / "Lib" / "site-packages")
    site = next((path for path in candidates if path.is_dir() and not path.is_symlink()), None)
    if site is None:
        return
    preserved = {_normalized_distribution_name(name) for name in preserve_distributions}
    preserve_pip = "pip" in preserved
    preserve_setuptools = bool(
        {"setuptools", "pkg-resources", "-distutils-hack"}.intersection(preserved)
    )
    names = {
        "pip",
        "setuptools",
        "pkg_resources",
        "_distutils_hack",
        "distutils-precedence.pth",
        "easy-install.pth",
    }
    for path in sorted(site.iterdir(), key=lambda item: item.name):
        normalized = _normalized_distribution_name(path.name)
        if path.name.endswith(".dist-info"):
            metadata_name = path / "METADATA"
            if metadata_name.is_file() and not metadata_name.is_symlink():
                try:
                    metadata = Parser().parsestr(metadata_name.read_text(encoding="utf-8"))
                except (OSError, UnicodeError):
                    metadata = None
                if metadata is not None:
                    normalized = _normalized_distribution_name(str(metadata.get("Name", "")).strip())
            if normalized in preserved:
                continue
            if path.name.lower().startswith("pip-") and preserve_pip:
                continue
            if path.name.lower().startswith("setuptools-") and preserve_setuptools:
                continue
        elif normalized in preserved:
            continue
        elif preserve_pip and path.name == "pip":
            continue
        elif preserve_setuptools and path.name in {"setuptools", "pkg_resources", "_distutils_hack"}:
            continue
        elif preserve_setuptools and path.name in {"distutils-precedence.pth", "easy-install.pth"}:
            continue
        if path.name in names or (
            path.name.endswith(".dist-info")
            and (path.name.startswith("pip-") or path.name.startswith("setuptools-"))
        ):
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path)


def _remove_unbound_distribution_metadata(root: Path) -> None:
    """Remove pip-created metadata absent from the signed wheel archive.

    ``INSTALLER``, ``REQUESTED`` and ``direct_url.json`` are generated by pip
    after wheel extraction.  They are not executable runtime inputs and their
    values vary with the provisioning command, so the managed installer
    removes them and removes their rows from the installed RECORD before the
    semantic RECORD verifier is applied.
    """

    candidates = sorted((root / "lib").glob("python*/site-packages"))
    candidates.append(root / "Lib" / "site-packages")
    site = next((path for path in candidates if path.is_dir() and not path.is_symlink()), None)
    if site is None:
        return
    for dist_info in sorted(site.glob("*.dist-info"), key=lambda item: item.name):
        if not dist_info.is_dir() or dist_info.is_symlink():
            continue
        removed = {"INSTALLER", "REQUESTED", "direct_url.json"}
        for name in removed:
            path = dist_info / name
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
        record = dist_info / "RECORD"
        if not record.is_file() or record.is_symlink():
            continue
        try:
            rows = list(csv.reader(record.read_text(encoding="utf-8", errors="strict").splitlines()))
        except (OSError, UnicodeError, csv.Error):
            continue
        kept = [
            row
            for row in rows
            if len(row) != 3
            or not (
                PurePosixPath(row[0].replace("\\", "/")).parent == PurePosixPath(dist_info.name)
                and PurePosixPath(row[0].replace("\\", "/")).name in removed
            )
        ]
        with record.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerows(kept)


def _remove_bytecode_caches(root: Path) -> None:
    """Remove provisioning-time derived caches before signed runtime checks."""

    if not root.is_dir() or root.is_symlink():
        return
    paths = sorted(root.rglob("*"), key=lambda item: (len(item.parts), item.as_posix()), reverse=True)
    for path in paths:
        if path.is_symlink():
            continue
        if path.is_dir() and path.name == "__pycache__":
            shutil.rmtree(path)
        elif path.is_file() and path.suffix.lower() in {".pyc", ".pyo"}:
            path.unlink()


def _remove_unbound_runtime_convenience(
    root: Path,
    *,
    preserve_distributions: Iterable[str] = (),
) -> None:
    """Remove unbound venv convenience files from official runtimes.

    OEL launches through the signed entry-point wrappers and the provider
    alias.  Activation/pip convenience files are host-generated and are not a
    trust root; removing them keeps the signed runtime surface exact.
    """

    if not root.is_dir() or root.is_symlink():
        return
    scripts = root / ("Scripts" if os.name == "nt" else "bin")
    if not scripts.is_dir() or scripts.is_symlink():
        return
    preserved = {_normalized_distribution_name(name) for name in preserve_distributions}
    preserve_pip = "pip" in preserved
    preserve_setuptools = bool(
        {"setuptools", "pkg-resources", "-distutils-hack"}.intersection(preserved)
    )
    for path in sorted(scripts.iterdir(), key=lambda item: item.name):
        name = path.name.lower()
        is_pip = bool(
            re.fullmatch(r"pip(?:3(?:\.\d+)?)?(?:-script\.py|\.(?:exe|cmd|bat))?", name)
        )
        is_easy_install = bool(
            re.fullmatch(
                r"easy_install(?:[-.]\d+(?:\.\d+)?)?(?:-script\.py|\.(?:exe|cmd|bat))?",
                name,
            )
        )
        if name.startswith("activate"):
            if path.is_file() and not path.is_symlink():
                path.unlink()
        elif is_pip:
            if preserve_pip:
                continue
            if path.is_file() and not path.is_symlink():
                path.unlink()
        elif is_easy_install:
            if preserve_setuptools:
                continue
            if path.is_file() and not path.is_symlink():
                path.unlink()


def _constraints_for(source_root: Path) -> Path | None:
    minor = f"py{sys.version_info.major}{sys.version_info.minor}.txt"
    candidate = source_root / "constraints" / minor
    return candidate if candidate.is_file() else None


def _tree_inventory(source_root: Path) -> dict[str, Any]:
    files = [
        {"path": path.relative_to(source_root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
    ]
    return {"schema_version": "oel.installed-tree.v1", "files": files}


def _trusted_anchor_path(locations: InstallationPaths) -> Path:
    """Return the operator/channel trust anchor after boundary checks.

    The anchor must live in the persistent config root, outside every version
    directory.  A symlink here would let a mutable installation redirect the
    release authority, so it is rejected even when its target is readable.
    """

    config_root = locations.config_root.expanduser().resolve(strict=False)
    versions = locations.versions.expanduser().resolve(strict=False)
    try:
        config_resolved = config_root
        versions_resolved = versions
        if config_resolved == versions_resolved or config_resolved.is_relative_to(versions_resolved):
            raise ContractError("Official release trust anchor must live outside managed version roots.")
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"Could not resolve the official release trust-anchor boundary: {exc}") from exc
    anchor = config_root / "trusted-release-keys.json"
    if anchor.is_symlink():
        raise ContractError("Official release trust anchor may not be a symbolic link.")
    if anchor.exists() and not anchor.is_file():
        raise ContractError("Official release trust anchor must be a regular file.")
    return anchor


def _official_manifest_cache_path(locations: InstallationPaths, version: str) -> Path:
    """Return the detached signed-manifest receipt for an official version."""

    cache_root = locations.config_root.expanduser().resolve(strict=False) / "trusted-manifests"
    versions = locations.versions.expanduser().resolve(strict=False)
    try:
        cache_root.resolve(strict=False).relative_to(versions.resolve(strict=False))
    except ValueError:
        pass
    else:
        raise ContractError("Official signed-manifest receipts must live outside managed version roots.")
    if cache_root.is_symlink():
        raise ContractError("Official signed-manifest receipt directory may not be a symbolic link.")
    if cache_root.exists() and not cache_root.is_dir():
        raise ContractError("Official signed-manifest receipt directory must be a directory.")
    path = cache_root / f"{validate_version(version)}.json"
    if path.is_symlink():
        raise ContractError("Official signed-manifest receipt may not be a symbolic link.")
    return path


def _keyset_identity(keys: Mapping[str, RSAPublicKey]) -> dict[str, tuple[Any, ...]]:
    return {
        str(key_id): (
            key.key_id,
            int(key.n),
            int(key.e),
            key.alg,
            key.not_before.isoformat() if key.not_before else None,
            key.expires_at.isoformat() if key.expires_at else None,
            bool(key.revoked),
        )
        for key_id, key in keys.items()
    }


def _official_anchor_keys(locations: InstallationPaths) -> tuple[Path, dict[str, RSAPublicKey]]:
    anchor = _trusted_anchor_path(locations)
    if not anchor.is_file():
        raise ContractError(
            "Official release trust anchor is missing outside the managed version root; "
            "bootstrap or an operator-supplied channel must establish it before activation."
        )
    try:
        keys = load_release_public_keys(anchor)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"Official release trust anchor is invalid: {exc}") from exc
    return anchor, keys


def install_release(
    manifest_path: str | Path,
    *,
    paths: InstallationPaths | None = None,
    public_keys: Mapping[str, RSAPublicKey] | None = None,
    require_signature: bool = True,
    profile: str = "core",
    create_runtime: bool = True,
    license_path: str | Path | None = None,
    license_public_keys: Mapping[str, RSAPublicKey] | None = None,
    offline_wheelhouse: str | Path | None = None,
) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    locations.ensure()
    source_manifest_path = Path(manifest_path).expanduser().resolve()
    anchor_path: Path | None = None
    anchor_keys: dict[str, RSAPublicKey] | None = None
    if require_signature:
        anchor_path, anchor_keys = _official_anchor_keys(locations)
        # The caller-supplied mapping is only a transport hint.  The
        # persistent operator/channel anchor is authoritative; an embedded
        # registry may legitimately describe a future rotation, but it never
        # replaces this anchor implicitly.
        manifest_keys = anchor_keys
    else:
        manifest_keys = public_keys
    manifest = verify_release_manifest(
        load_json_object(source_manifest_path),
        public_keys=manifest_keys,
        require_signature=require_signature,
        require_integrity=require_signature,
    )
    integrity: dict[str, Any] | None = None
    if require_signature:
        integrity = validate_installation_integrity(
            dict(dict(manifest.get("supply_chain", {}) or {}).get("installation_integrity", {}) or {})
        )
        if str(integrity.get("profile", "")) not in {str(item) for item in manifest.get("profiles", []) or []}:
            raise ContractError("Signed installation integrity default profile is not declared by the release manifest.")
    _validate_host_compatibility(manifest)
    if offline_wheelhouse is not None and create_runtime:
        _validate_offline_runtime_compatibility(manifest)
    version = manifest["version"]
    if profile not in manifest["profiles"]:
        raise ContractError(f"Install profile {profile!r} is not declared by OEL release {version}.")
    profile_inventories = dict(integrity.get("runtime_profiles", {}) or {}) if integrity is not None else {}
    profile_inventory = profile_inventories.get(profile) if profile_inventories else None
    if require_signature and not isinstance(profile_inventory, Mapping):
        raise ContractError(
            f"Official release does not contain a signed runtime inventory for selected profile {profile!r}."
        )
    if require_signature and integrity is not None:
        missing_profiles = sorted(
            set(str(item) for item in manifest.get("profiles", []) or []) - set(profile_inventories)
        )
        if missing_profiles:
            raise ContractError(
                "Official release does not contain signed runtime inventories for profiles: "
                + ", ".join(missing_profiles)
            )
    if manifest["edition"] == "pro":
        if not pro_installation_available():
            raise ContractError("Managed OEL Pro installation is unavailable in this edition.")
        from sim.licensing.offline import verify_license_file

        license_status = verify_license_file(
            Path(license_path).expanduser() if license_path is not None else None,
            product="oel-pro",
            version=version,
            public_keys=dict(license_public_keys) if license_public_keys is not None else None,
            allow_owner_bypass=False,
        )
        if not license_status.valid:
            raise ContractError(f"OEL Pro release {version} is not licensed: {license_status.message}")
    artifact = _select_source_artifact(manifest)
    archive = _artifact_path(source_manifest_path, artifact)
    verify_release_artifact(archive, artifact)
    archive_version = _archive_source_version(archive)
    if archive_version != version:
        raise ContractError(
            f"Release artifact source version {archive_version!r} does not match signed manifest version {version!r}."
        )
    if require_signature and not create_runtime:
        raise ContractError(
            "Official managed installations must provision the signed runtime; "
            "use a separate developer installation for an operator-provided runtime."
        )
    if require_signature and create_runtime and offline_wheelhouse is None:
        raise ContractError(
            "Official managed installations require a verified signed wheelhouse; "
            "developer installations may use an operator-provided package source."
        )
    if offline_wheelhouse is not None:
        wheelhouse_root = Path(offline_wheelhouse).expanduser().resolve()
        wheel_artifacts = [item for item in manifest["artifacts"] if item.get("kind") == "wheel"]
        if not wheel_artifacts:
            raise ContractError("Offline release manifest does not declare a wheelhouse.")
        declared_paths: set[Path] = set()
        for wheel in wheel_artifacts:
            wheel_path = _artifact_path(source_manifest_path, wheel)
            try:
                wheel_path.relative_to(wheelhouse_root)
            except ValueError as exc:
                raise ContractError(f"Offline wheel is outside the verified wheelhouse: {wheel_path}") from exc
            verify_release_artifact(wheel_path, wheel)
            declared_paths.add(wheel_path)
        actual_paths = {path.resolve() for path in wheelhouse_root.iterdir() if path.is_file()}
        if actual_paths != declared_paths:
            raise ContractError("Offline wheelhouse contents do not exactly match the signed release manifest.")
    if require_signature and integrity is not None:
        signed_wheels = {
            str(row.get("name")): str(row.get("sha256"))
            for row in list(dict(profile_inventory or {}).get("wheels", []) or [])
            if isinstance(row, Mapping)
        }
        declared_wheels = {
            str(item.get("name")): str(item.get("sha256"))
            for item in manifest.get("artifacts", [])
            if isinstance(item, Mapping) and item.get("kind") == "wheel"
        }
        if signed_wheels != {name: digest for name, digest in declared_wheels.items() if name in signed_wheels}:
            raise ContractError("Signed installation integrity does not match the release wheel artifacts.")
        if not set(signed_wheels).issubset(set(declared_wheels)):
            raise ContractError("Signed profile runtime inventory references an undeclared release wheel.")
    profile_wheel_paths: list[Path] | None = None
    profile_wheel_hashes: dict[str, str] | None = None
    if require_signature and create_runtime:
        # Install the exact wheel closure authenticated by the selected
        # profile inventory.  Installing the source archive here can produce
        # a different project distribution (or a different dependency set)
        # from the wheel bytes that the release signed.
        profile_wheel_paths = []
        profile_wheel_hashes = {}
        for row in list(dict(profile_inventory or {}).get("wheels", []) or []):
            if not isinstance(row, Mapping):
                raise ContractError("Signed profile runtime inventory contains an invalid wheel row.")
            name = str(row.get("name", ""))
            wheel_path = wheelhouse_root / name
            if not name or wheel_path.is_symlink() or not wheel_path.is_file():
                raise ContractError(f"Signed profile wheel is missing from the offline wheelhouse: {name!r}")
            if sha256_file(wheel_path) != str(row.get("sha256", "")):
                raise ContractError(f"Signed profile wheel changed before provisioning: {name!r}")
            profile_wheel_paths.append(wheel_path)
            profile_wheel_hashes[name] = str(row["sha256"])
    final_root = locations.version_root(version)
    with StateLock(locations.transaction_lock, operation=f"install:{version}") as lock:
        if final_root.exists():
            # An existing version is an idempotent success only after the same
            # full source/inventory checks used by activation.  A quick record
            # check would turn an edited installation into a false success.
            existing = verify_installation(version, paths=locations, full=True)
            if existing["status"] == "official":
                state = read_state(locations.installations_state, default=empty_installation_state())
                installations = dict(state.get("installations", {}) or {})
                if version not in installations:
                    record_path = final_root / "installation-record.json"
                    record = load_json_object(record_path)
                    installations[version] = {
                        "path": str(final_root),
                        "record": str(record_path),
                        "status": str(record.get("status", "official")),
                        "installed_at": record.get("installed_at"),
                    }
                    state["installations"] = installations
                    history = list(state.get("history", []) or [])
                    history.append({"operation": "reconcile-install", "version": version, "at": utc_now()})
                    state["history"] = history[-100:]
                    atomic_write_json(locations.installations_state, state)
                return {
                    "schema_version": UPDATE_RECEIPT_SCHEMA,
                    "status": "ready",
                    "operation": "install",
                    "transaction_id": lock.transaction_id,
                    "version": version,
                    "idempotent": True,
                    "installation": existing,
                    "effects": {"workspace_modified": False, "activated": False},
                }
            raise FileExistsError(f"Installation destination already exists but is not verified: {final_root}")
        transaction_root = locations.versions / f".{version}.{lock.transaction_id}.incomplete"
        transaction_root.mkdir(parents=True, exist_ok=False)
        try:
            verified_archive = transaction_root / "verified-source-archive.tar.gz"
            shutil.copyfile(archive, verified_archive)
            verify_release_artifact(verified_archive, artifact)
            extracted = transaction_root / "source"
            safe_extract(verified_archive, extracted, max_bytes=MAX_RELEASE_BYTES)
            source_root = _find_source_root(extracted)
            source_version = source_project_version(source_root=source_root)
            if source_version != version:
                raise ContractError(
                    f"Release artifact source version {source_version!r} does not match signed manifest version {version!r}."
                )
            for name, expected_digest in dict(manifest.get("constraints", {}) or {}).items():
                if Path(str(name)).name != str(name):
                    raise ContractError(f"Release constraint name is unsafe: {name!r}")
                constraint = source_root / "constraints" / str(name)
                if not constraint.is_file() or sha256_file(constraint) != expected_digest:
                    raise ContractError(f"Release constraint is missing or changed: {name}")
            if require_signature and integrity is not None:
                source_issues = verify_source_inventory(source_root, integrity["source"])
                if source_issues:
                    raise ContractError(
                        "Extracted source does not match the signed release inventory: " + "; ".join(source_issues[:8])
                    )
            runtime = transaction_root / "runtime"
            runtime_result: dict[str, Any] = {"created": False, "python": sys.executable}
            if create_runtime:
                wheelhouse = Path(offline_wheelhouse).expanduser().resolve() if offline_wheelhouse is not None else None
                runtime_result = {
                    "created": True,
                    **_install_runtime(
                        source_root,
                        runtime,
                        profile=profile,
                        constraints=_constraints_for(source_root),
                        offline_wheelhouse=wheelhouse,
                        signed_wheels=profile_wheel_paths,
                        signed_wheel_hashes=profile_wheel_hashes,
                    ),
                }
                if require_signature and integrity is not None:
                    runtime_issues = verify_runtime_inventory(
                        runtime,
                        profile_inventory,
                        interpreter=runtime_result.get("python"),
                        trusted_provider=_trusted_external_python(locations),
                    )
                    if runtime_issues:
                        raise ContractError(
                            "Installed runtime does not match the signed wheel inventory: "
                            + "; ".join(runtime_issues[:8])
                        )
                    _rebind_runtime_scripts(runtime, profile_inventory, final_root / "runtime")
                provider = Path(str(runtime_result.get("python") or sys.executable)).expanduser()
                try:
                    provider_resolved = provider.resolve(strict=True)
                except OSError:
                    provider_resolved = provider.absolute()
                try:
                    provider_resolved.relative_to(transaction_root)
                except ValueError:
                    pass
                else:
                    provider_resolved = Path(sys.executable).resolve()
                try:
                    managed_python = provider.relative_to(transaction_root)
                except ValueError:
                    managed_python = provider
                runtime_result["provider"] = {
                    "path": str(provider_resolved),
                    "boundary": "external-interpreter",
                    "trust": "operator-owned",
                }
                # The transaction directory is renamed atomically below. Keep
                # the managed interpreter path relocatable in the record and
                # retain the resolved external provider separately.
                runtime_result["python"] = str(managed_python)
            source_digest = sha256_tree(source_root)
            tree_manifest_path = transaction_root / "installed-tree.json"
            atomic_write_json(tree_manifest_path, _tree_inventory(source_root))
            copied_manifest = transaction_root / "release-manifest.json"
            shutil.copy2(source_manifest_path, copied_manifest)
            record = {
                "schema_version": INSTALLATION_RECORD_SCHEMA,
                "status": "official" if require_signature else "developer",
                "version": version,
                "edition": manifest["edition"],
                "channel": manifest["channel"],
                "installed_at": utc_now(),
                "transaction_id": lock.transaction_id,
                "release_manifest_sha256": release_manifest_digest(manifest),
                "release_artifact": {
                    "name": artifact["name"],
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                },
                "source": {"path": str(source_root.relative_to(transaction_root)), "sha256": source_digest},
                "installed_tree_manifest": "installed-tree.json",
                "runtime": runtime_result,
                "provenance": {
                    "authority": "external-trusted-release-key-registry" if require_signature else "developer",
                    "trust_anchor": str(anchor_path) if anchor_path is not None else None,
                    "trust_anchor_sha256": sha256_file(anchor_path) if anchor_path is not None else None,
                    "manifest_signer_key_id": str(dict(manifest.get("signature", {}) or {}).get("key_id", "")) or None,
                    "installation_integrity_sha256": integrity.get("digest") if integrity is not None else None,
                    "source_inventory_sha256": integrity.get("source", {}).get("digest") if integrity else None,
                    "runtime_inventory_sha256": profile_inventory.get("digest") if profile_inventory else None,
                },
                "profile": profile,
                "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
                "effects": {"workspace_modified": False, "activated": False},
            }
            atomic_write_json(transaction_root / "installation-record.json", record)
            if require_signature:
                cache_path = _official_manifest_cache_path(locations, version)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                if cache_path.exists():
                    cached = load_json_object(cache_path)
                    if release_manifest_digest(cached) != release_manifest_digest(manifest):
                        raise ContractError(
                            "An external signed-manifest receipt already exists for this version and differs from the release."
                        )
                else:
                    atomic_write_json(cache_path, manifest)
            transaction_root.replace(final_root)
            if require_signature and create_runtime and profile_inventory is not None:
                published_runtime = final_root / "runtime"
                runtime_issues = verify_runtime_inventory(
                    published_runtime, profile_inventory,
                    interpreter=published_runtime / str(profile_inventory["layout"]["interpreter"]),
                    trusted_provider=_trusted_external_python(locations),
                )
                if runtime_issues:
                    raise ContractError("Published runtime failed integrity verification: " + "; ".join(runtime_issues[:8]))
            state = read_state(locations.installations_state, default=empty_installation_state())
            installations = dict(state.get("installations", {}) or {})
            installations[version] = {
                "path": str(final_root),
                "record": str(final_root / "installation-record.json"),
                "status": record["status"],
                "installed_at": record["installed_at"],
            }
            state["installations"] = installations
            history = list(state.get("history", []) or [])
            history.append({"operation": "install", "version": version, "transaction_id": lock.transaction_id, "at": utc_now()})
            state["history"] = history[-100:]
            atomic_write_json(locations.installations_state, state)
        except Exception:
            if final_root.exists() and not transaction_root.exists():
                final_root.replace(transaction_root)
            if transaction_root.exists():
                failed_record = {
                    "schema_version": INSTALLATION_RECORD_SCHEMA,
                    "status": "incomplete",
                    "version": version,
                    "transaction_id": lock.transaction_id,
                    "failed_at": utc_now(),
                    "effects": {
                        "workspace_modified": False,
                        "activated": False,
                        "network_used": bool(create_runtime and offline_wheelhouse is None),
                    },
                }
                atomic_write_json(transaction_root / "installation-record.json", failed_record)
            raise
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "install",
        "transaction_id": lock.transaction_id,
        "version": version,
        "idempotent": False,
        "installation": verify_installation(version, paths=locations, full=False),
        "effects": {
            "workspace_modified": False,
            "activated": False,
            "network_used": bool(create_runtime and offline_wheelhouse is None),
        },
    }


def verify_installation(version: str, *, paths: InstallationPaths | None = None, full: bool = False) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    root = locations.version_root(validate_version(version))
    record_path = root / "installation-record.json"
    if root.is_symlink():
        return {
            "status": "incomplete",
            "version": version,
            "path": str(root),
            "issues": ["installation root may not be a symbolic link"],
        }
    if not record_path.is_file() or record_path.is_symlink():
        return {"status": "incomplete", "version": version, "path": str(root), "issues": ["installation record missing"]}
    try:
        record = load_json_object(record_path)
    except (OSError, ContractError) as exc:
        return {
            "status": "incomplete",
            "version": version,
            "path": str(root),
            "record": str(record_path),
            "issues": [f"installation record unreadable: {exc}"],
        }
    issues: list[str] = []
    trust_issues: list[str] = []
    provenance_diagnostics: list[str] = []
    if record.get("schema_version") != INSTALLATION_RECORD_SCHEMA:
        issues.append("unsupported installation record schema")
    if str(record.get("version", "")) != version:
        issues.append("installation record version does not match its directory")
    source_record = record.get("source", {})
    source = None
    if not isinstance(source_record, Mapping):
        issues.append("installed source record is malformed")
    else:
        try:
            source = _record_path(root, source_record.get("path", "source"), label="installed source")
        except ContractError as exc:
            issues.append(str(exc))
    if source is not None and not source.is_dir():
        issues.append("installed source directory missing")
    elif source is not None and full and record.get("status") != "official":
        tree_hash_ok = True
        try:
            actual = sha256_tree(source)
        except (OSError, ContractError) as exc:
            tree_hash_ok = False
            issues.append(f"installed source tree could not be hashed: {exc}")
        else:
            if actual != source_record.get("sha256"):
                issues.append("installed source tree digest mismatch")
        try:
            tree_manifest = _record_path(
                root,
                record.get("installed_tree_manifest", "installed-tree.json"),
                label="installed tree manifest",
            )
        except ContractError as exc:
            issues.append(str(exc))
        else:
            if not tree_manifest.is_file() or tree_manifest.is_symlink():
                issues.append("installed source tree manifest missing")
            elif tree_hash_ok:
                try:
                    inventory = load_json_object(tree_manifest)
                except (OSError, ContractError) as exc:
                    issues.append(f"installed source tree manifest unreadable: {exc}")
                else:
                    if inventory != _tree_inventory(source):
                        issues.append("installed source tree inventory mismatch")
    runtime_record = record.get("runtime", {})
    if not isinstance(runtime_record, Mapping):
        issues.append("installed runtime record is malformed")
    elif runtime_record.get("created"):
        relocated = root / "runtime" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not relocated.is_file():
            issues.append("managed runtime interpreter missing")
    manifest_path = root / "release-manifest.json"
    manifest: dict[str, Any] | None = None
    if not manifest_path.is_file() or manifest_path.is_symlink():
        issues.append("release manifest missing")
    else:
        try:
            manifest = load_json_object(manifest_path)
        except (OSError, ContractError) as exc:
            issues.append(f"release manifest unreadable: {exc}")
        else:
            if sha256_bytes(canonical_json_bytes(manifest)) != record.get("release_manifest_sha256"):
                issues.append("release manifest digest mismatch")
    manifest_is_signed = bool(isinstance(manifest, Mapping) and isinstance(manifest.get("signature"), Mapping))
    # The detached signed-manifest receipt is the external installation-class
    # fact.  If a mutable record is downgraded while that receipt remains, the
    # installation must still take the authenticated path and fail closed.
    cache_path: Path | None = None
    cache_error: str | None = None
    cache_receipt_present = False
    try:
        cache_path = _official_manifest_cache_path(locations, version)
        cache_receipt_present = cache_path.exists() or cache_path.is_symlink()
    except ContractError as exc:
        cache_error = str(exc)
        cache_receipt_present = True
    official_receipt = record.get("status") == "official" or manifest_is_signed or cache_receipt_present
    if official_receipt:
        try:
            _anchor_path, anchor_keys = _official_anchor_keys(locations)
            if manifest is None:
                raise ContractError("release manifest is unavailable")
            verified_manifest = verify_release_manifest(
                manifest,
                public_keys=anchor_keys,
                require_signature=True,
                require_integrity=True,
            )
            if verified_manifest.get("version") != version:
                trust_issues.append("signed release manifest version does not match its managed version root")
            if record.get("status") == "developer":
                trust_issues.append("official installation status was downgraded to developer in mutable state")
            if cache_error is not None:
                trust_issues.append(cache_error)
            for field in ("edition", "channel"):
                if record.get(field) != verified_manifest.get(field):
                    trust_issues.append(f"installation {field} does not match the signed release manifest")
            recorded_artifact = record.get("release_artifact")
            artifact_match = isinstance(recorded_artifact, Mapping) and any(
                isinstance(item, Mapping)
                and item.get("kind") in {"source", "source_bundle"}
                and all(recorded_artifact.get(field) == item.get(field) for field in ("name", "bytes", "sha256"))
                for item in list(verified_manifest.get("artifacts", []) or [])
            )
            if not artifact_match:
                trust_issues.append("installation source artifact does not match the signed release manifest")
            profile = str(record.get("profile", ""))
            if profile not in list(verified_manifest.get("profiles", []) or []):
                trust_issues.append("installation profile is not declared by the signed release manifest")
            integrity = validate_installation_integrity(
                dict(dict(verified_manifest.get("supply_chain", {}) or {}).get("installation_integrity", {}) or {})
            )
            if str(integrity.get("profile", "")) not in {
                str(item) for item in verified_manifest.get("profiles", []) or []
            }:
                trust_issues.append("signed installation integrity default profile is not declared by the release manifest")
            runtime_profiles = dict(integrity.get("runtime_profiles", {}) or {})
            missing_profiles = sorted(set(str(item) for item in verified_manifest.get("profiles", []) or []) - set(runtime_profiles))
            if missing_profiles:
                trust_issues.append(
                    "signed installation integrity lacks profile runtime inventories: " + ", ".join(missing_profiles)
                )
            selected_runtime = runtime_profiles.get(profile)
            if not isinstance(selected_runtime, Mapping):
                trust_issues.append("signed installation integrity lacks the selected profile runtime inventory")
            if full and source is not None and source.is_dir():
                source_issues = verify_source_inventory(source, integrity["source"])
                issues.extend(source_issues)
            if official_receipt and (not isinstance(runtime_record, Mapping) or not runtime_record.get("created")):
                trust_issues.append("official installation is missing its authenticated managed runtime")
            if full and isinstance(runtime_record, Mapping) and runtime_record.get("created"):
                runtime_root = root / "runtime"
                recorded_python = runtime_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                runtime_issues = verify_runtime_inventory(
                    runtime_root,
                    selected_runtime or {},
                    interpreter=recorded_python,
                    trusted_provider=_trusted_external_python(locations),
                )
                issues.extend(runtime_issues)
            provenance = record.get("provenance")
            if not isinstance(provenance, Mapping) or provenance.get("authority") != "external-trusted-release-key-registry":
                trust_issues.append("installation provenance is missing an external trust-anchor binding")
            else:
                if provenance.get("trust_anchor") != str(_anchor_path):
                    trust_issues.append("installation provenance trust-anchor path does not match the external anchor")
                recorded_anchor_digest = provenance.get("trust_anchor_sha256")
                if not isinstance(recorded_anchor_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded_anchor_digest):
                    trust_issues.append("installation provenance historical trust-anchor digest is missing or malformed")
                elif recorded_anchor_digest != sha256_file(_anchor_path):
                    # The digest is retained as installation-time history. A
                    # signed, authorized key rotation legitimately changes
                    # the current anchor bytes; the current anchor signature
                    # check above remains the authority for this install.
                    provenance_diagnostics.append(
                        "installation provenance historical anchor digest differs from the current external key anchor"
                    )
                if provenance.get("manifest_signer_key_id") != str(dict(verified_manifest.get("signature", {}) or {}).get("key_id", "")):
                    trust_issues.append("installation provenance signer does not match the signed manifest")
                if provenance.get("installation_integrity_sha256") != integrity.get("digest"):
                    trust_issues.append("installation provenance does not match the signed integrity digest")
                if selected_runtime is not None and provenance.get("runtime_inventory_sha256") != selected_runtime.get("digest"):
                    trust_issues.append("installation provenance does not match the selected runtime inventory")
            if record.get("status") == "official" or cache_receipt_present:
                if cache_path is None or not cache_path.is_file():
                    trust_issues.append("official signed-manifest receipt is missing; reverification_required")
                else:
                    cached = verify_release_manifest(
                        load_json_object(cache_path),
                        public_keys=anchor_keys,
                        require_signature=True,
                        require_integrity=True,
                    )
                    if release_manifest_digest(cached) != release_manifest_digest(verified_manifest):
                        trust_issues.append("official signed-manifest receipt does not match the managed manifest")
        except (ContractError, OSError, ValueError) as exc:
            message = str(exc)
            if "installation integrity" in message.lower() or "reverification" in message.lower():
                trust_issues.append("reverification_required")
            trust_issues.append(f"official installation is not authenticated: {message}")
    issues.extend(trust_issues)
    if trust_issues:
        # A malformed mutable record can independently prove a local
        # modification (for example, a recorded source symlink).  Preserve
        # that diagnostic disposition while still refusing activation; an
        # otherwise intact legacy/revoked installation is explicitly
        # ``untrusted`` below.
        disposition = "modified" if any("symbolic link" in issue for issue in issues) else "untrusted"
    elif issues and record.get("status") in {"official", "developer"}:
        disposition = "modified"
    else:
        disposition = str(record.get("status", "incomplete"))
    if not issues and disposition not in {"official", "developer", "untrusted"}:
        disposition = "incomplete"
    return {
        "status": disposition,
        "version": version,
        "path": str(root),
        "record": str(record_path),
        "profile": record.get("profile"),
        "full_integrity": bool(full),
        "issues": issues,
        **({"diagnostics": provenance_diagnostics} if provenance_diagnostics else {}),
    }


def activate(version: str, *, paths: InstallationPaths | None = None) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    locations.ensure()
    target = validate_version(version)
    verified = verify_installation(target, paths=locations, full=True)
    if verified["status"] not in {"official", "developer"}:
        raise RuntimeError(f"Cannot activate unverified OEL installation {target}: {verified}")
    with StateLock(locations.transaction_lock, operation=f"activate:{target}") as lock:
        current = read_state(locations.current_state, default={"schema_version": "oel.current-installation.v1"})
        previous = current.get("current")
        payload = {
            "schema_version": "oel.current-installation.v1",
            "current": target,
            "previous": previous if previous != target else current.get("previous"),
            "activated_at": utc_now(),
            "transaction_id": lock.transaction_id,
        }
        old_state = locations.current_state.read_bytes() if locations.current_state.is_file() else None
        old_launchers = {
            path: path.read_bytes() if path.is_file() else None
            for path in (
                locations.launcher / "oel",
                locations.launcher / "oel.cmd",
                *_trusted_bundle_paths(locations),
            )
        }
        try:
            launchers = write_launchers(target, paths=locations)
            atomic_write_json_descriptor_bound(
                locations.current_state,
                payload,
                approved_root=locations.data_root,
            )
        except Exception:
            for path, content in old_launchers.items():
                # ``write_launchers`` normally replaces paths atomically, but
                # a pre-existing linked verifier entry can make it fail before
                # all entries are replaced.  Never restore bytes with
                # ``Path.write_bytes`` in that case: it follows a symlink and
                # could overwrite an unrelated file outside the installation
                # root.  Unlink the current node first, then restore through an
                # atomic sibling replacement.
                if content is None:
                    if secure_state_write_available():
                        unlink_descriptor_bound(path, approved_root=locations.data_root)
                    else:
                        if path.is_symlink() or (path.exists() and not path.is_file()):
                            path.unlink(missing_ok=True)
                        path.unlink(missing_ok=True)
                else:
                    if secure_state_write_available():
                        atomic_write_text_descriptor_bound(
                            path,
                            content.decode("utf-8"),
                            approved_root=locations.data_root,
                        )
                    else:
                        if path.is_symlink() or (path.exists() and not path.is_file()):
                            path.unlink(missing_ok=True)
                        atomic_write_text(path, content.decode("utf-8"))
            if old_state is None:
                if secure_state_write_available():
                    unlink_descriptor_bound(locations.current_state, approved_root=locations.data_root)
                else:
                    locations.current_state.unlink(missing_ok=True)
            else:
                # The selector is mutable state too.  A pre-existing
                # symlink can be readable through ``read_bytes``; restoring
                # it with ``Path.write_bytes`` would follow that link after a
                # launcher failure.  Replace the node atomically instead.
                if secure_state_write_available():
                    atomic_write_text_descriptor_bound(
                        locations.current_state,
                        old_state.decode("utf-8"),
                        approved_root=locations.data_root,
                    )
                else:
                    if locations.current_state.is_symlink() or (
                        locations.current_state.exists() and not locations.current_state.is_file()
                    ):
                        locations.current_state.unlink(missing_ok=True)
                    atomic_write_text(locations.current_state, old_state.decode("utf-8"))
            raise
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "activate",
        "transaction_id": lock.transaction_id,
        "current": target,
        "previous": payload["previous"],
        "launchers": launchers,
        "effects": {"workspace_modified": False, "activated": True},
    }


def write_launchers(version: str, *, paths: InstallationPaths | None = None) -> dict[str, str]:
    locations = paths or InstallationPaths.default()
    locations.ensure()
    source, _managed_python = _source_and_python(version, locations)
    del source, _managed_python
    external_python = _trusted_external_python(locations)
    bundle = _install_trusted_verifier_bundle(locations)
    dispatcher = locations.launcher / "trusted-dispatcher.py"
    dispatcher_source = Path(__file__).with_name("trusted_dispatcher.py")
    atomic_write_text(dispatcher, dispatcher_source.read_text(encoding="utf-8"))
    posix = locations.launcher / "oel"
    windows = locations.launcher / "oel.cmd"
    atomic_write_text(
        posix,
        "#!/bin/sh\n"
        f"# {TRUSTED_LAUNCHER_MARKER}\n"
        f"exec {shlex.quote(str(external_python))} -I {shlex.quote(str(dispatcher))} "
        f"--data-root {shlex.quote(str(locations.data_root))} "
        f"--config-root {shlex.quote(str(locations.config_root))} "
        f"--version {shlex.quote(validate_version(version))} -- \"$@\"\n",
    )
    try:
        posix.chmod(0o755)
    except OSError:
        pass
    atomic_write_text(
        windows,
        (
            f"@echo off\r\nrem {TRUSTED_LAUNCHER_MARKER}\r\n"
            f'"{external_python}" -I "{dispatcher}" '
            f'--data-root "{locations.data_root}" --config-root "{locations.config_root}" '
            f'--version "{validate_version(version)}" -- %*\r\n'
        ),
    )
    return {
        "posix": str(posix),
        "windows": str(windows),
        "trusted_dispatcher": str(dispatcher),
        "trusted_bundle": str(bundle),
    }


def _source_and_python(version: str, paths: InstallationPaths) -> tuple[Path, Path]:
    root = paths.version_root(validate_version(version))
    record = load_json_object(root / "installation-record.json")
    source_record = record.get("source", {})
    if not isinstance(source_record, Mapping):
        raise RuntimeError("Installed source record is malformed.")
    source = _record_path(root, source_record.get("path", "source"), label="installed source")
    if not source.is_dir():
        raise RuntimeError(f"Installed source directory is missing: {source}")
    runtime_record = record.get("runtime", {})
    if not isinstance(runtime_record, Mapping):
        raise RuntimeError("Installed runtime record is malformed.")
    runtime = dict(runtime_record)
    python = Path(str(runtime.get("python") or sys.executable))
    relocated = root / "runtime" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if runtime.get("created"):
        if not relocated.is_file():
            raise RuntimeError(f"Managed runtime interpreter is missing: {relocated}")
        python = relocated
    elif not python.name.lower().startswith("python"):
        raise RuntimeError(f"Recorded runtime interpreter is not a Python executable: {python}")
    # Keep the managed virtualenv interpreter path intact. Resolving the POSIX
    # ``bin/python`` symlink selects the base interpreter and silently drops
    # the virtualenv's installed packages when the launcher runs.
    return source.resolve(), python.expanduser().absolute()


def rollback(*, paths: InstallationPaths | None = None) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    current = read_state(locations.current_state)
    previous = str(current.get("previous", "") or "")
    if not previous:
        raise RuntimeError("No previous OEL installation is recorded for rollback.")
    return activate(previous, paths=locations)


def installation_status(*, paths: InstallationPaths | None = None, full: bool = False) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    locations.ensure()
    state = read_state(locations.installations_state, default=empty_installation_state())
    current = read_state(locations.current_state, default={})
    channel_config = read_state(
        locations.channel_config,
        default={"schema_version": CHANNEL_CONFIG_SCHEMA, "default": None, "channels": {}},
    )
    versions = sorted(dict(state.get("installations", {}) or {}))
    incomplete = [
        {"path": str(path), "record": str(path / "installation-record.json")}
        for path in sorted(locations.versions.glob(".*.incomplete"))
        if path.is_dir()
    ]
    return {
        "schema_version": "oel.update-status.v1",
        "status": "ready",
        "paths": {"data_root": str(locations.data_root), "config_root": str(locations.config_root)},
        "current": current.get("current"),
        "previous": current.get("previous"),
        "update_channels": {
            "default": channel_config.get("default"),
            "configured": sorted(dict(channel_config.get("channels", {}) or {})),
            "path": str(locations.channel_config),
        },
        "installations": [verify_installation(version, paths=locations, full=full) for version in versions],
        "incomplete_transactions": incomplete,
        "lock": str(locations.transaction_lock) if locations.transaction_lock.exists() else None,
    }


def write_support_receipt(output: str | Path, *, paths: InstallationPaths | None = None) -> dict[str, Any]:
    """Write a sanitized support snapshot without workspace paths or user content."""

    locations = paths or InstallationPaths.default()
    status = installation_status(paths=locations, full=False)
    registry = read_state(locations.workspaces_state, default={})
    workspaces = dict(registry.get("workspaces", {}) or {})
    payload = {
        "schema_version": "oel.support-receipt.v1",
        "generated_at": utc_now(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "current": status.get("current"),
        "previous": status.get("previous"),
        "installations": [
            {"version": item["version"], "status": item["status"], "profile": item.get("profile"), "issues": item["issues"]}
            for item in status["installations"]
        ],
        "workspaces": [
            {"workspace_id": workspace_id, "locked_version": item.get("locked_version")}
            for workspace_id, item in sorted(workspaces.items())
            if isinstance(item, dict)
        ],
        "privacy": {
            "workspace_paths_included": False,
            "user_source_included": False,
            "config_contents_included": False,
            "telemetry_sent": False,
        },
    }
    destination = Path(output).expanduser().resolve()
    atomic_write_json(destination, payload)
    return {"status": "ready", "path": str(destination), "receipt": payload}


def cleanup(*, paths: InstallationPaths | None = None, dry_run: bool = True, keep: int = 2) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    status = installation_status(paths=locations)
    current = {item for item in (status.get("current"), status.get("previous")) if item}
    workspace_registry = read_state(locations.workspaces_state, default={})
    referenced = {
        str(item.get("locked_version"))
        for item in dict(workspace_registry.get("workspaces", {}) or {}).values()
        if isinstance(item, dict) and item.get("locked_version")
    }
    all_versions = sorted((item["version"] for item in status["installations"]), key=version_tuple, reverse=True)
    retained = set(all_versions[: max(0, int(keep))]) | current | referenced
    candidates = [version for version in all_versions if version not in retained]
    for version in candidates:
        lease_root = locations.version_root(version) / "leases"
        if lease_root.is_dir() and any(lease_root.iterdir()):
            retained.add(version)
    candidates = [version for version in candidates if version not in retained]
    incomplete_candidates = [Path(item["path"]) for item in status.get("incomplete_transactions", [])]
    removed: list[str] = []
    if not dry_run:
        with StateLock(locations.transaction_lock, operation="cleanup"):
            locked_status = installation_status(paths=locations)
            locked_current = {
                item for item in (locked_status.get("current"), locked_status.get("previous")) if item
            }
            locked_registry = read_state(locations.workspaces_state, default={})
            locked_referenced = {
                str(item.get("locked_version"))
                for item in dict(locked_registry.get("workspaces", {}) or {}).values()
                if isinstance(item, dict) and item.get("locked_version")
            }
            locked_candidates: list[str] = []
            for version in candidates:
                lease_root = locations.version_root(version) / "leases"
                if version in locked_current or version in locked_referenced:
                    continue
                if lease_root.is_dir() and any(lease_root.iterdir()):
                    continue
                locked_candidates.append(version)
            candidates = locked_candidates
            for version in candidates:
                shutil.rmtree(locations.version_root(version))
                removed.append(version)
            for path in incomplete_candidates:
                shutil.rmtree(path)
            state = read_state(locations.installations_state, default=empty_installation_state())
            installations = dict(state.get("installations", {}) or {})
            for version in removed:
                installations.pop(version, None)
            state["installations"] = installations
            atomic_write_json(locations.installations_state, state)
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "cleanup",
        "dry_run": bool(dry_run),
        "candidates": candidates,
        "removed": removed,
        "incomplete_candidates": [str(path) for path in incomplete_candidates],
        "incomplete_removed": [str(path) for path in incomplete_candidates] if not dry_run else [],
        "retained": sorted(retained),
        "recoverable_by_redownload": bool(removed),
    }


def uninstall(
    version: str,
    *,
    paths: InstallationPaths | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Remove one unreferenced managed engine while preserving every workspace."""

    locations = paths or InstallationPaths.default()
    target = validate_version(version)
    current = read_state(locations.current_state, default={})
    protected_by = [name for name in ("current", "previous") if current.get(name) == target]
    registry = read_state(locations.workspaces_state, default={})
    workspace_refs = [
        str(workspace_id)
        for workspace_id, item in dict(registry.get("workspaces", {}) or {}).items()
        if isinstance(item, dict) and item.get("locked_version") == target
    ]
    lease_root = locations.version_root(target) / "leases"
    active_leases = sorted(path.name for path in lease_root.iterdir()) if lease_root.is_dir() else []
    blockers = {
        "selectors": protected_by,
        "workspaces": workspace_refs,
        "active_leases": active_leases,
    }
    blocked = any(blockers.values())
    removed = False
    if not dry_run and blocked:
        raise RuntimeError(f"Cannot uninstall referenced OEL {target}: {blockers}")
    if not dry_run:
        with StateLock(locations.transaction_lock, operation=f"uninstall:{target}"):
            current = read_state(locations.current_state, default={})
            registry = read_state(locations.workspaces_state, default={})
            lease_root = locations.version_root(target) / "leases"
            blockers = {
                "selectors": [name for name in ("current", "previous") if current.get(name) == target],
                "workspaces": [
                    str(workspace_id)
                    for workspace_id, item in dict(registry.get("workspaces", {}) or {}).items()
                    if isinstance(item, dict) and item.get("locked_version") == target
                ],
                "active_leases": sorted(path.name for path in lease_root.iterdir()) if lease_root.is_dir() else [],
            }
            if any(blockers.values()):
                raise RuntimeError(f"Cannot uninstall referenced OEL {target}: {blockers}")
            root = locations.version_root(target)
            if root.is_dir():
                shutil.rmtree(root)
                removed = True
            state = read_state(locations.installations_state, default=empty_installation_state())
            installations = dict(state.get("installations", {}) or {})
            installations.pop(target, None)
            state["installations"] = installations
            history = list(state.get("history", []) or [])
            history.append({"operation": "uninstall", "version": target, "at": utc_now()})
            state["history"] = history[-100:]
            atomic_write_json(locations.installations_state, state)
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "blocked" if blocked else "ready",
        "operation": "uninstall",
        "version": target,
        "dry_run": bool(dry_run),
        "removed": removed,
        "blockers": blockers,
        "retained_workspaces": sorted(dict(registry.get("workspaces", {}) or {})),
        "effects": {"workspace_modified": False, "activated": False},
    }


def recover_lock(*, paths: InstallationPaths | None = None, stale_after_s: float = 3600.0) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    return recover_stale_lock(locations.transaction_lock, stale_after_s=stale_after_s)


def _read_url(url: str, *, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "OEL-Updater/1", "Accept-Encoding": "identity"},
    )
    with _open_pinned_url(request, timeout=30) as response:
        status = getattr(response, "status", None)
        scheme = urllib.parse.urlsplit(url).scheme.lower()
        # ``file://`` fixtures are admitted only after the existing URL
        # policy has accepted the scheme.  urllib's local-file response has
        # no HTTP status; an actual HTTP response must always report 200.
        if (scheme == "file" and status is not None and status != 200) or (
            scheme != "file" and status != 200
        ):
            raise ContractError("Remote metadata endpoint did not return HTTP 200.")
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                declared = int(length)
            except (TypeError, ValueError) as exc:
                raise ContractError("Remote content has an invalid Content-Length header.") from exc
            if declared > max_bytes:
                raise ContractError(f"Remote content exceeds the {max_bytes} byte limit.")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ContractError(f"Remote content exceeds the {max_bytes} byte limit.")
    return payload


def _validate_remote_url(url: str, *, allow_local_file: bool) -> None:
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme == "https":
        if not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ContractError("Online OEL update URLs must include an HTTPS host.")
        try:
            port = parsed.port or 443
        except ValueError as exc:
            raise ContractError("Online OEL update URL has an invalid HTTPS port.") from exc
        if not 1 <= port <= 65535:
            raise ContractError("Online OEL update URL has an invalid HTTPS port.")
        return
    if allow_local_file and scheme == "file" and parsed.path:
        return
    raise ContractError("Online OEL update URLs must use HTTPS.")


def configure_channel(
    channel_url: str,
    *,
    edition: str = "public",
    channel: str = "stable",
    paths: InstallationPaths | None = None,
    source: str = "explicit",
    set_default: bool = True,
    allow_local_file: bool = False,
) -> dict[str, Any]:
    """Persist one trusted-metadata endpoint without weakening signature checks."""

    _validate_remote_url(channel_url, allow_local_file=allow_local_file)
    if edition not in {"public", "pro"}:
        raise ContractError("Configured update channel edition must be public or pro.")
    if channel not in {"stable", "preview"}:
        raise ContractError("Configured update channel must be stable or preview.")
    locations = paths or InstallationPaths.default()
    locations.ensure()
    with StateLock(locations.transaction_lock, operation=f"configure-channel:{edition}:{channel}"):
        config = read_state(
            locations.channel_config,
            default={"schema_version": CHANNEL_CONFIG_SCHEMA, "default": None, "channels": {}},
        )
        reject_unknown_keys(config, {"schema_version", "default", "channels"}, label="update channel configuration")
        if config.get("schema_version") != CHANNEL_CONFIG_SCHEMA:
            raise ContractError("Update channel configuration has an unsupported schema.")
        channels = dict(config.get("channels", {}) or {})
        channel_id = f"{edition}:{channel}"
        channels[channel_id] = {
            "edition": edition,
            "channel": channel,
            "url": channel_url,
            "source": source,
            "configured_at": utc_now(),
        }
        config["channels"] = channels
        if set_default or not config.get("default"):
            config["default"] = channel_id
        atomic_write_json(locations.channel_config, config)
    return {
        "schema_version": CHANNEL_CONFIG_SCHEMA,
        "status": "ready",
        "channel_id": channel_id,
        "channel_url": channel_url,
        "default": config["default"],
        "path": str(locations.channel_config),
    }


def configured_channel_url(
    *,
    edition: str = "public",
    channel: str = "stable",
    paths: InstallationPaths | None = None,
    explicit_url: str | None = None,
    allow_local_file: bool = False,
) -> str:
    if explicit_url:
        _validate_remote_url(explicit_url, allow_local_file=allow_local_file)
        return explicit_url
    locations = paths or InstallationPaths.default()
    config = read_state(locations.channel_config)
    reject_unknown_keys(config, {"schema_version", "default", "channels"}, label="update channel configuration")
    if config.get("schema_version") != CHANNEL_CONFIG_SCHEMA:
        raise ContractError(
            "No supported update channel is configured. Re-run the official installer or provide --channel-url."
        )
    channel_id = f"{edition}:{channel}"
    entry = dict(dict(config.get("channels", {}) or {}).get(channel_id, {}) or {})
    require_keys(entry, {"edition", "channel", "url", "source", "configured_at"}, label=f"configured channel {channel_id}")
    reject_unknown_keys(
        entry,
        {"edition", "channel", "url", "source", "configured_at"},
        label=f"configured channel {channel_id}",
    )
    if entry["edition"] != edition or entry["channel"] != channel:
        raise ContractError(f"Configured update channel identity does not match {channel_id}.")
    url = str(entry["url"])
    _validate_remote_url(url, allow_local_file=allow_local_file)
    return url


def _download_url(url: str, destination: Path, *, max_bytes: int) -> None:
    """Download with a bounded partial-file resume when the server supports ranges."""

    offset = destination.stat().st_size if destination.is_file() else 0
    if offset > max_bytes:
        raise ContractError(f"Partial download exceeds the {max_bytes} byte limit.")
    headers = {"User-Agent": "OEL-Updater/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with _open_pinned_url(request, timeout=60) as response:
        status = getattr(response, "status", None)
        scheme = urllib.parse.urlsplit(url).scheme.lower()
        if (scheme == "file" and status is not None and status not in {200, 206}) or (
            scheme != "file" and status not in {200, 206}
        ):
            raise ContractError("Remote release endpoint did not return HTTP 200 or 206.")
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        total = offset if resumed else 0
        with destination.open(mode) as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ContractError(f"Remote content exceeds the {max_bytes} byte limit.")
                stream.write(chunk)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Keep urllib from following redirects before the updater has checked
        # the destination and pinned its independently resolved TLS endpoint.
        return None


def _approved_release_redirect(source: str, destination: str) -> bool:
    """Permit only the public repository's canonical GitHub asset route."""

    try:
        _validate_remote_url(source, allow_local_file=False)
        _validate_remote_url(destination, allow_local_file=False)
        old, new = urllib.parse.urlsplit(source), urllib.parse.urlsplit(destination)
        if old.hostname != "github.com" or old.port not in {None, 443} or old.query:
            return False
        if new.port not in {None, 443}:
            return False
    except (ValueError, ContractError):
        return False
    match = re.fullmatch(
        r"/adamcohen8/orbital-engineering-lab/releases/(?:latest/download|download/v\d+\.\d+\.\d+)/([A-Za-z0-9_.-]+)",
        old.path,
    )
    if match is None:
        return False
    if new.hostname == "github.com" and "/releases/latest/download/" in old.path and not new.query:
        return bool(re.fullmatch(
            r"/adamcohen8/orbital-engineering-lab/releases/download/v\d+\.\d+\.\d+/" + re.escape(match[1]),
            new.path,
        ))
    return new.hostname == "release-assets.githubusercontent.com" and bool(re.fullmatch(
        r"/github-production-release-asset/1163745057/[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}", new.path,
    ))


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to one DNS result while retaining HTTPS hostname validation."""

    def __init__(self, host: str, *, address: tuple, **kwargs: Any):
        self._pinned_address = address
        super().__init__(host, **kwargs)

    def connect(self) -> None:
        family, socktype, proto, _canonname, sockaddr = self._pinned_address
        self.sock = socket.socket(family, socktype, proto)
        try:
            self.sock.settimeout(self.timeout)
            self.sock.connect(sockaddr)
            if self._tunnel_host:
                self._tunnel()
            server_hostname = self._tunnel_host or self.host
            self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)
        except BaseException:
            self.close()
            raise


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, address: tuple, *, context: ssl.SSLContext):
        super().__init__(context=context, check_hostname=True)
        self._pinned_address = address

    def _connection(self, host: str, **kwargs: Any) -> _PinnedHTTPSConnection:
        return _PinnedHTTPSConnection(host, address=self._pinned_address, **kwargs)

    def https_open(self, req):
        return self.do_open(
            self._connection,
            req,
            context=self._context,
        )


def _pinned_addresses(url: str) -> list[tuple]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return []
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise ContractError("Online OEL update URL has an invalid HTTPS port.") from exc
    try:
        records = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ContractError(f"Could not resolve update host {parsed.hostname!r}.") from exc
    addresses: list[tuple] = []
    seen: set[tuple] = set()
    for record in records:
        if len(record) != 5:
            continue
        family, socktype, proto, canonname, sockaddr = record
        if family not in {socket.AF_INET, socket.AF_INET6} or socktype != socket.SOCK_STREAM:
            continue
        try:
            ipaddress.ip_address(str(sockaddr[0]))
        except ValueError:
            continue
        key = (family, socktype, proto, tuple(sockaddr))
        if key in seen:
            continue
        seen.add(key)
        addresses.append((family, socktype, proto, canonname, sockaddr))
    if not addresses:
        raise ContractError(f"Update host {parsed.hostname!r} has no usable HTTPS address.")
    return addresses


def _open_pinned_url(request: urllib.request.Request, *, timeout: float):
    parsed = urllib.parse.urlsplit(request.full_url)
    if parsed.scheme.lower() != "https":
        return urllib.request.urlopen(request, timeout=timeout)
    context = ssl.create_default_context()
    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        raise ContractError("HTTPS update transport requires certificate and hostname verification.")
    for _hop in range(4):
        last_error: BaseException | None = None
        redirected = None
        for address in _pinned_addresses(request.full_url):
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                _NoRedirectHandler(),
                _PinnedHTTPSHandler(address, context=context),
            )
            try:
                return opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as exc:
                if exc.code not in {301, 302, 303, 307, 308}:
                    last_error = exc
                    continue
                location = (exc.headers or {}).get("Location")
                destination = urllib.parse.urljoin(request.full_url, location or "")
                exc.close()
                if request.get_method() != "GET" or not location or not _approved_release_redirect(request.full_url, destination):
                    raise ContractError("Online OEL update redirect is outside the approved public release route.") from exc
                headers = {"User-Agent": "OEL-Updater/1", "Accept-Encoding": "identity"}
                if request.get_header("Range"):
                    headers["Range"] = request.get_header("Range")
                redirected = urllib.request.Request(destination, headers=headers)
                break
            except (OSError, http.client.HTTPException, urllib.error.URLError) as exc:
                last_error = exc
        if redirected is None:
            raise ContractError("Could not connect to the pinned HTTPS update endpoint.") from last_error
        request = redirected
    raise ContractError("Online OEL update redirect limit exceeded.")


def check_channel(
    channel_url: str | None = None,
    *,
    public_keys: Mapping[str, RSAPublicKey],
    edition: str = "public",
    channel: str = "stable",
    paths: InstallationPaths | None = None,
    allow_feed_rollback: bool = False,
    allow_local_file: bool = False,
) -> dict[str, Any]:
    channel_url = configured_channel_url(
        edition=edition,
        channel=channel,
        paths=paths,
        explicit_url=channel_url,
        allow_local_file=allow_local_file,
    )
    _validate_remote_url(channel_url, allow_local_file=allow_local_file)
    payload = json.loads(_read_url(channel_url, max_bytes=MAX_METADATA_BYTES))
    if not isinstance(payload, dict) or payload.get("schema_version") != CHANNEL_INDEX_SCHEMA:
        raise ContractError("Update channel metadata has an unsupported schema.")
    channel_fields = {
        "schema_version",
        "edition",
        "channel",
        "latest",
        "manifest_url",
        "published_at",
        "signature",
    }
    reject_unknown_keys(payload, channel_fields, label="release channel")
    require_keys(payload, channel_fields, label="release channel")
    if not verify_payload(payload, public_keys):
        raise ContractError("Update channel signature verification failed.")
    if payload.get("edition") != edition or payload.get("channel") != channel:
        raise ContractError("Update channel edition or channel does not match the request.")
    _validate_remote_url(str(payload.get("manifest_url", "")), allow_local_file=allow_local_file)
    latest = validate_version(payload.get("latest"), label="channel latest")
    locations = paths or InstallationPaths.default()
    locations.ensure()
    channel_id = f"{edition}:{channel}"
    with StateLock(locations.transaction_lock, operation=f"check-channel:{channel_id}"):
        state = read_state(
            locations.channel_state,
            default={"schema_version": "oel.channel-state.v1", "channels": {}},
        )
        previous = dict(state.get("channels", {}) or {}).get(channel_id)
        if (
            previous
            and version_tuple(latest) < version_tuple(str(previous.get("latest", "0.0")))
            and not allow_feed_rollback
        ):
            raise ContractError(
                f"Release feed rollback rejected: {channel_id} previously advertised {previous['latest']}, now {latest}."
            )
        if (
            previous
            and version_tuple(latest) == version_tuple(str(previous.get("latest", "0.0")))
            and str(payload.get("published_at", "")) < str(previous.get("published_at", ""))
            and not allow_feed_rollback
        ):
            raise ContractError(f"Release feed timestamp rollback rejected for {channel_id} at version {latest}.")
        channels = dict(state.get("channels", {}) or {})
        channels[channel_id] = {
            "latest": latest,
            "published_at": payload.get("published_at"),
            "checked_at": utc_now(),
        }
        state["channels"] = channels
        atomic_write_json(locations.channel_state, state)
    return {
        "schema_version": "oel.update-check.v1",
        "status": "ready",
        "edition": edition,
        "channel": channel,
        "channel_url": channel_url,
        "latest": latest,
        "manifest_url": payload.get("manifest_url"),
        "published_at": payload.get("published_at"),
        "effects": {"network_used": True, "downloaded_release": False, "workspace_modified": False},
    }


def download_release_wheelhouse(
    manifest: Mapping[str, Any],
    manifest_url: str,
    destination: Path,
    *,
    allow_local_file: bool = False,
) -> Path | None:
    """Fetch wheels for an already verified manifest from its versioned bundle.

    The ZIP is a transport, not a trust root: its manifest must exactly match
    the verified manifest and every installed wheel is checked against that
    manifest's signed size and SHA-256 before it reaches pip.
    """
    wheels = [row for row in manifest["artifacts"] if row.get("kind") == "wheel"]
    if not wheels:
        return None
    _validate_host_compatibility(manifest)
    _validate_offline_runtime_compatibility(manifest)
    qualification = dict(manifest.get("supply_chain", {})).get("offline_runtime_qualification", {})
    python_match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", str(qualification.get("python", "")))
    architecture = str(manifest.get("architecture", "")).lower()
    if python_match is None or re.fullmatch(r"[a-z0-9_]+", architecture) is None:
        raise ContractError("Bundled release wheels require an exact qualified Python minor and architecture.")
    tag = f"py{python_match[1]}{python_match[2]}"
    name = f"oel-{manifest['edition']}-{manifest['version']}-{architecture}-{tag}.bundle.zip"
    url = urllib.parse.urljoin(manifest_url, name)
    _validate_remote_url(url, allow_local_file=allow_local_file)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="oel-release-wheels-", dir=destination) as temporary:
        work = Path(temporary)
        bundle = work / name
        transport_limit = min(
            MAX_RELEASE_BYTES,
            sum(int(row["bytes"]) for row in manifest["artifacts"]) + MAX_METADATA_BYTES,
        )
        _download_url(url, bundle, max_bytes=transport_limit)
        unpacked = safe_extract(bundle, work / "unpacked", max_bytes=MAX_RELEASE_BYTES)
        bundled_manifest = unpacked / "release-manifest.json"
        if canonical_json_bytes(load_json_object(bundled_manifest)) != canonical_json_bytes(manifest):
            raise ContractError("Downloaded bundle manifest does not match the verified release manifest.")
        expected: set[Path] = set()
        for wheel in wheels:
            path = _artifact_path(bundled_manifest, wheel)
            if path.parent != unpacked / "wheelhouse":
                raise ContractError("Bundled release wheel must be directly inside wheelhouse.")
            verify_release_artifact(path, wheel)
            expected.add(path)
        actual = set((unpacked / "wheelhouse").rglob("*"))
        if actual != expected:
            raise ContractError("Downloaded bundle wheelhouse differs from the signed wheel inventory.")
        wheelhouse = destination / "wheelhouse"
        if wheelhouse.exists():
            shutil.rmtree(wheelhouse)
        (unpacked / "wheelhouse").replace(wheelhouse)
    return wheelhouse


def download_release(
    manifest_url: str,
    *,
    paths: InstallationPaths | None = None,
    public_keys: Mapping[str, RSAPublicKey],
    allow_local_file: bool = False,
) -> dict[str, Any]:
    _validate_remote_url(manifest_url, allow_local_file=allow_local_file)
    locations = paths or InstallationPaths.default()
    locations.ensure()
    manifest_payload = json.loads(_read_url(manifest_url, max_bytes=MAX_METADATA_BYTES))
    if not isinstance(manifest_payload, dict):
        raise ContractError("Release manifest response must be an object.")
    manifest = verify_release_manifest(manifest_payload, public_keys=public_keys, require_signature=True)
    artifact = _select_source_artifact(manifest)
    url = str(artifact.get("url", "") or "")
    if not url:
        raise ContractError("Online release artifact is missing url.")
    resolved_url = urllib.parse.urljoin(manifest_url, url)
    _validate_remote_url(resolved_url, allow_local_file=allow_local_file)
    cache_root = (
        locations.cache
        / str(manifest["edition"])
        / str(manifest["channel"])
        / str(manifest["version"])
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / artifact["name"]
    temporary = archive.with_suffix(archive.suffix + ".partial")
    _download_url(resolved_url, temporary, max_bytes=min(MAX_RELEASE_BYTES, int(artifact["bytes"])))
    verify_release_artifact(temporary, artifact)
    temporary.replace(archive)
    manifest_path = cache_root / "release-manifest.json"
    # Keep signed content byte-for-byte equivalent at the contract level. The
    # local installer resolves the downloaded artifact by its signed name.
    atomic_write_json(manifest_path, manifest)
    wheelhouse = download_release_wheelhouse(
        manifest, manifest_url, cache_root, allow_local_file=allow_local_file
    )
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "download",
        "version": manifest["version"],
        "manifest": str(manifest_path),
        "artifact": str(archive),
        "wheelhouse": str(wheelhouse) if wheelhouse is not None else None,
        "effects": {"network_used": True, "workspace_modified": False, "activated": False},
    }


def install_latest_release(
    *,
    paths: InstallationPaths | None = None,
    public_keys: Mapping[str, RSAPublicKey],
    channel_url: str | None = None,
    edition: str = "public",
    channel: str = "stable",
    profile: str = "core",
    create_runtime: bool = True,
    license_path: str | Path | None = None,
    license_public_keys: Mapping[str, RSAPublicKey] | None = None,
    allow_local_file: bool = False,
) -> dict[str, Any]:
    """Check, download, and install the newest signed release without activation."""

    locations = paths or InstallationPaths.default()
    checked = check_channel(
        channel_url,
        public_keys=public_keys,
        edition=edition,
        channel=channel,
        paths=locations,
        allow_local_file=allow_local_file,
    )
    downloaded = download_release(
        str(checked["manifest_url"]),
        paths=locations,
        public_keys=public_keys,
        allow_local_file=allow_local_file,
    )
    downloaded_manifest = load_json_object(downloaded["manifest"])
    expected_identity = (str(checked["latest"]), edition, channel)
    actual_identity = (
        str(downloaded_manifest.get("version", "")),
        str(downloaded_manifest.get("edition", "")),
        str(downloaded_manifest.get("channel", "")),
    )
    if actual_identity != expected_identity:
        raise ContractError(
            "Signed channel and release manifest identities do not agree: "
            f"expected {expected_identity}, got {actual_identity}."
        )
    installed = install_release(
        downloaded["manifest"],
        paths=locations,
        public_keys=public_keys,
        require_signature=True,
        profile=profile,
        create_runtime=create_runtime,
        license_path=license_path,
        license_public_keys=license_public_keys,
        offline_wheelhouse=downloaded.get("wheelhouse"),
    )
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "install-latest",
        "version": checked["latest"],
        "activated": False,
        "workspace_modified": False,
        "check": checked,
        "download": downloaded,
        "installation": installed,
        "next": {
            "global": f"oel update activate {checked['latest']}",
            "workspace": f"oel workspace check <path> --against {checked['latest']}",
        },
    }


def install_bundle(
    bundle: str | Path,
    *,
    paths: InstallationPaths | None = None,
    public_keys: Mapping[str, RSAPublicKey] | None = None,
    require_signature: bool = True,
    profile: str = "core",
    create_runtime: bool = True,
    license_path: str | Path | None = None,
    license_public_keys: Mapping[str, RSAPublicKey] | None = None,
) -> dict[str, Any]:
    source = Path(bundle).expanduser().resolve()
    if source.is_dir():
        manifest = source / "release-manifest.json"
        wheelhouse = source / "wheelhouse"
        offline_wheelhouse = wheelhouse if wheelhouse.is_dir() else None
        if create_runtime and offline_wheelhouse is None:
            raise ContractError(
                "Offline bundle does not contain a wheelhouse; install with create_runtime=False "
                "or provide a complete signed bundle."
            )
        return install_release(
            manifest,
            paths=paths,
            public_keys=public_keys,
            require_signature=require_signature,
            profile=profile,
            create_runtime=create_runtime,
            license_path=license_path,
            license_public_keys=license_public_keys,
            offline_wheelhouse=offline_wheelhouse,
        )
    locations = paths or InstallationPaths.default()
    locations.ensure()
    with tempfile.TemporaryDirectory(prefix="oel-bundle-", dir=locations.cache) as temporary:
        root = Path(temporary)
        safe_extract(source, root, max_bytes=MAX_RELEASE_BYTES)
        manifests = list(root.rglob("release-manifest.json"))
        if len(manifests) != 1:
            raise ContractError("Offline bundle must contain exactly one release-manifest.json.")
        wheelhouse = manifests[0].parent / "wheelhouse"
        offline_wheelhouse = wheelhouse if wheelhouse.is_dir() else None
        if create_runtime and offline_wheelhouse is None:
            raise ContractError(
                "Offline bundle does not contain a wheelhouse; install with create_runtime=False "
                "or provide a complete signed bundle."
            )
        return install_release(
            manifests[0],
            paths=locations,
            public_keys=public_keys,
            require_signature=require_signature,
            profile=profile,
            create_runtime=create_runtime,
            license_path=license_path,
            license_public_keys=license_public_keys,
            offline_wheelhouse=offline_wheelhouse,
        )


def keys_from_path(path: str | Path | None, *, paths: InstallationPaths | None = None) -> dict[str, RSAPublicKey]:
    locations = paths or InstallationPaths.default()
    source = Path(path).expanduser() if path else _trusted_anchor_path(locations)
    if source.is_symlink():
        raise ContractError("Trusted OEL release-key registry may not be a symbolic link.")
    if not source.is_file():
        raise FileNotFoundError(f"Trusted OEL release-key registry was not found: {source}")
    return load_release_public_keys(source)


def rotate_trusted_release_keys(
    registry_path: str | Path,
    *,
    paths: InstallationPaths | None = None,
    current_keys: Mapping[str, RSAPublicKey] | None = None,
) -> dict[str, Any]:
    """Adopt a key registry only when it is authorized by a currently trusted key."""

    locations = paths or InstallationPaths.default()
    locations.ensure()
    anchor = _trusted_anchor_path(locations)
    if not anchor.is_file():
        raise ContractError(
            "A current external trusted-key anchor is required before rotation; "
            "bootstrap the initial operator/channel anchor first."
        )
    payload = load_json_object(registry_path)
    if payload.get("schema_version") != "oel.trusted-key-registry.v1":
        raise ContractError("Trusted release-key registry has an unsupported schema.")
    reject_unknown_keys(
        payload,
        {"schema_version", "published_at", "keys", "signature"},
        label="trusted release-key registry",
    )
    require_keys(payload, {"schema_version", "keys", "signature"}, label="trusted release-key registry")
    if not str(payload.get("published_at", "") or ""):
        raise ContractError("Trusted release-key registry must declare published_at for replay protection.")
    try:
        incoming_published = datetime.fromisoformat(str(payload["published_at"]).replace("Z", "+00:00"))
        if incoming_published.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        incoming_published = incoming_published.astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise ContractError("Trusted release-key registry published_at must be a timezone-aware ISO-8601 timestamp.") from exc
    with StateLock(locations.transaction_lock, operation="rotate-trusted-release-keys"):
        trusted = dict(keys_from_path(None, paths=locations))
        if current_keys is not None and _keyset_identity(current_keys) != _keyset_identity(trusted):
            raise ContractError("Caller-supplied current keys do not match the external trust anchor.")
        if not verify_payload(payload, trusted):
            raise ContractError("Trusted release-key registry signature verification failed.")
        replacement = load_release_public_keys(registry_path)
        active = [key for key in replacement.values() if not key.revoked]
        if not active:
            raise ContractError("Trusted release-key registry must retain at least one non-revoked key.")
        if any(key.alg != "RS256" or key.n.bit_length() < 2048 or key.e != 65537 for key in active):
            raise ContractError("Active trusted release keys must be RS256, RSA-2048 or stronger, with e=65537.")
        current_payload = load_json_object(anchor)
        current_raw = current_payload.get("published_at")
        if current_raw in (None, ""):
            # The original operator bootstrap format predates replay
            # timestamps and contains only ``keys``.  Treat that exact
            # key-only anchor as an explicit epoch baseline; the first signed
            # rotation must still carry a valid timezone-aware timestamp and
            # establishes the timestamped history thereafter.
            if set(current_payload) != {"keys"}:
                raise ContractError("Current trusted release-key registry published_at is invalid.")
            current_published = datetime.min.replace(tzinfo=timezone.utc)
        else:
            try:
                current_published = datetime.fromisoformat(str(current_raw).replace("Z", "+00:00"))
                if current_published.tzinfo is None:
                    raise ValueError("timestamp must be timezone-aware")
                current_published = current_published.astimezone(timezone.utc)
            except (TypeError, ValueError) as exc:
                raise ContractError("Current trusted release-key registry published_at is invalid.") from exc
        if incoming_published <= current_published:
            raise ContractError("Trusted release-key registry replay or timestamp rollback rejected.")
        atomic_write_json(anchor, payload)
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "rotate-trusted-release-keys",
        "key_ids": sorted(replacement),
        "effects": {"workspace_modified": False, "activated": False, "network_used": False},
    }
