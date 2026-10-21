"""Build deterministic, signed OEL source releases and offline bundles."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.installation.contracts import (  # noqa: E402
    CHANNEL_INDEX_SCHEMA,
    RELEASE_MANIFEST_SCHEMA,
    SCENARIO_SCHEMA,
    WORKSPACE_SCHEMA,
    sha256_file,
)
from sim.installation.signing import (  # noqa: E402
    RSAPrivateKey,
    RSAPublicKey,
    load_private_key,
    load_public_keys,
    sign_payload,
    verify_payload,
)
from sim.project_version import source_project_version  # noqa: E402

_EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv_temp",
    "__pycache__",
    "build",
    "dist",
    "outputs",
}
_LOCAL_ONLY_SOURCE_PATTERNS = (
    "examples/experiments/**",
    "sim/game/assets/drag_racing_*.png",
    "sim/game/configs/game_training_rpo_bonus_drag_racing.yaml",
    "sim/tests/test_game_drag_racing_local.py",
)
_MINIMUM_OFFICIAL_RSA_BITS = 2048


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_commit(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    candidates: list[Path]
    if (root / ".git").exists():
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
        if tracked.returncode != 0:
            raise ValueError("Could not resolve the tracked source inventory for the release archive.")
        candidates = [root / item.decode("utf-8") for item in tracked.stdout.split(b"\0") if item]
    else:
        candidates = list(root.rglob("*"))
    for path in candidates:
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        if any(relative.match(pattern) for pattern in _LOCAL_ONLY_SOURCE_PATTERNS):
            continue
        if not path.exists() and not path.is_symlink():
            raise ValueError(f"Tracked release source is missing from the working tree: {path}")
        if path.is_symlink():
            raise ValueError(f"Release source may not contain symbolic links: {path}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def release_source_identity(root: Path) -> dict[str, Any]:
    """Bind an audit to exactly the source inventory shipped in its archive."""
    files = {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "executable": os.access(path, os.X_OK),
        }
        for path in _source_files(root)
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "oel.release-source-identity.v1", "sha256": digest, "files": len(files)}


def _build_source_archive(root: Path, output: Path, *, version: str, epoch: int) -> None:
    prefix = f"orbital-engineering-lab-{version}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in _source_files(root):
                    relative = path.relative_to(root).as_posix()
                    data = path.read_bytes()
                    info = tarfile.TarInfo(f"{prefix}/{relative}")
                    info.size = len(data)
                    info.mtime = epoch
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                    archive.addfile(info, io.BytesIO(data))


def _constraint_digests(root: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted((root / "constraints").glob("py*.txt"))
        if path.is_file()
    }


def _write_bundle(output: Path, files: list[Path], *, epoch: int) -> None:
    stamp = datetime.fromtimestamp(max(epoch, 315532800), tz=timezone.utc).timetuple()[:6]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files, key=lambda item: item.relative_to(output.parent).as_posix()):
            info = zipfile.ZipInfo(path.relative_to(output.parent).as_posix(), date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def _render_installers(
    output: Path,
    *,
    public_keys: Path | None,
    base_url: str | None,
    channel_url: str | None,
) -> list[Path]:
    templates = Path(__file__).resolve().parent / "installers"
    output.mkdir(parents=True, exist_ok=True)
    keys_text = public_keys.read_text(encoding="utf-8") if public_keys is not None else '{"keys": []}\n'
    bootstrap_text = (templates / "bootstrap_install.py").read_text(encoding="utf-8")
    bootstrap_text = bootstrap_text.replace(
        "__OEL_TRUSTED_KEYS_B64__",
        base64.b64encode(keys_text.encode("utf-8")).decode("ascii"),
    )
    bootstrap_text = bootstrap_text.replace("__OEL_TRUSTED_KEYS_RENDERED__", "true")
    bootstrap = output / "bootstrap_install.py"
    bootstrap.write_text(bootstrap_text, encoding="utf-8")
    bootstrap_digest = sha256_file(bootstrap)
    default_url = base_url or "__OEL_DEFAULT_BASE_URL__"
    default_channel_url = channel_url or "__OEL_DEFAULT_CHANNEL_URL__"
    rendered: list[Path] = [bootstrap]
    for name in ("install.sh", "install.ps1"):
        text = (templates / name).read_text(encoding="utf-8")
        text = text.replace("__OEL_BOOTSTRAP_SHA256__", bootstrap_digest)
        text = text.replace("__OEL_DEFAULT_BASE_URL__", default_url)
        text = text.replace("__OEL_DEFAULT_CHANNEL_URL__", default_channel_url)
        text = text.replace("__OEL_INSTALLER_RENDERED__", "true")
        target = output / name
        target.write_text(text, encoding="utf-8")
        if name.endswith(".sh"):
            target.chmod(0o755)
        rendered.append(target)
    return rendered


def _official_signing_material(
    *,
    private_key: Path | None,
    public_keys: Path | None,
) -> tuple[RSAPrivateKey, dict[str, RSAPublicKey]]:
    if private_key is None:
        raise ValueError("A private release-signing key is required unless --developer-unsigned is selected.")
    if public_keys is None:
        raise ValueError("Signed release builds require --public-keys containing the matching trusted key.")
    signing_key = load_private_key(private_key)
    if signing_key.alg != "RS256" or signing_key.n.bit_length() < _MINIMUM_OFFICIAL_RSA_BITS:
        raise ValueError(
            f"Official release-signing keys must be RS256 with at least {_MINIMUM_OFFICIAL_RSA_BITS} RSA bits."
        )
    trusted_keys = load_public_keys(public_keys)
    trusted = trusted_keys.get(signing_key.key_id)
    if trusted is None:
        raise ValueError("The trusted public-key registry does not contain the release-signing key id.")
    if (
        trusted.alg != signing_key.alg
        or trusted.n != signing_key.n
        or trusted.e != 65537
        or trusted.revoked
    ):
        raise ValueError("The trusted public-key registry does not match the active release-signing key.")
    return signing_key, trusted_keys


def _collect_supply_chain_evidence(
    source: Path | None,
    output: Path,
    *,
    version: str,
    required: bool,
    public_source_root: Path | None = None,
) -> dict[str, Any]:
    if source is None:
        if required:
            raise ValueError("Signed release builds require --supply-chain-evidence from a passing exact-profile gate.")
        return {"status": "developer_unqualified", "artifacts": []}
    evidence_root = source.expanduser().resolve()
    gate_path = evidence_root / "supply-chain-gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"Supply-chain gate manifest was not found: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        not isinstance(gate, dict)
        or gate.get("schema_version") != 1
        or gate.get("kind") != "oel_supply_chain_gate"
        or gate.get("product") not in {"oel-pro", "oel-public"}
        or (public_source_root is None and gate.get("product") != "oel-pro")
        or not gate.get("passed")
    ):
        raise ValueError("Supply-chain evidence is not a passing gate.")
    if public_source_root is not None:
        if (
            gate.get("product") != "oel-public"
            or not gate.get("isolated_environment")
            or gate.get("source_identity") != release_source_identity(public_source_root)
        ):
            raise ValueError("Signed public releases require an isolated audit of the exact public source export.")
        from tools.native_source_identity import native_source_identity

        qualified = dict(gate.get("dependency_sources", {}) or {}).get("qualified_native_wheels", [])
        expected = {
            "oel-rust-orbit": native_source_identity(public_source_root / "rust" / "oel-orbit")["sha256"],
            "oel-rust-game": native_source_identity(public_source_root / "rust" / "oel-game")["sha256"],
        }
        if (
            len(qualified) != len(expected)
            or {row.get("package"): row.get("source_identity_sha256") for row in qualified} != expected
        ):
            raise ValueError("Public native wheel qualifications differ from the public crate source.")
    gate_version = str(gate.get("package_version", "") or "")
    if gate_version and gate_version != version:
        raise ValueError(f"Supply-chain evidence version {gate_version!r} does not match release {version!r}.")
    required_artifacts = {
        "pip-install-report.json",
        "build-install-report.json",
        "pip-check.txt",
        "wheel-inventory.json",
        "sbom.cdx.json",
        "python-freeze.txt",
        "pip-audit.json",
    }
    source_files = [gate_path]
    for item in gate.get("artifacts", []):
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = Path(str(item["path"])).expanduser().resolve()
        try:
            path.relative_to(evidence_root)
        except ValueError as exc:
            raise ValueError(f"Supply-chain evidence artifact escapes its evidence root: {path}") from exc
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"Supply-chain evidence artifact is missing or changed: {path}")
        if int(item.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Supply-chain evidence artifact size changed: {path}")
        source_files.append(path)
    if {path.name for path in source_files[1:]} != required_artifacts:
        raise ValueError("Supply-chain gate does not declare the exact required release evidence set.")
    wheel_inventory = json.loads((evidence_root / "wheel-inventory.json").read_text(encoding="utf-8"))
    resolved_wheels = [
        {"name": str(row.get("artifact", "")), "sha256": str(row.get("sha256", ""))}
        for row in list(dict(wheel_inventory).get("packages", []) or [])
        if isinstance(row, dict) and row.get("artifact_type") == "wheel"
    ]
    if not resolved_wheels or any(not row["name"] or len(row["sha256"]) != 64 for row in resolved_wheels):
        raise ValueError("Supply-chain wheel inventory is incomplete or lacks immutable wheel hashes.")
    if public_source_root is not None:
        for row in qualified:
            if {"name": row.get("wheel"), "sha256": row.get("sha256")} not in resolved_wheels:
                raise ValueError("Public native wheel inventory differs from its qualification receipt.")
    source_digests: list[dict[str, Any]] = []
    names: set[str] = set()
    for path in source_files:
        if path.name in names:
            raise ValueError(f"Supply-chain evidence contains duplicate artifact name: {path.name}")
        names.add(path.name)
        source_digests.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    # Retained candidate evidence intentionally records exact commands, source
    # provenance, and absolute paths. Keep that authoritative packet private
    # and publish only a deterministic digest attestation in distributable
    # bundles so internal workspace details cannot cross the release boundary.
    destination = output / "release-evidence"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    attestation = {
        "schema_version": "oel.public-supply-chain-attestation.v1",
        "status": "passed",
        "package_version": version,
        "source_evidence": source_digests,
    }
    attestation_path = destination / "public-supply-chain-attestation.json"
    attestation_path.write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    published = {
        "name": attestation_path.name,
        "bytes": attestation_path.stat().st_size,
        "sha256": sha256_file(attestation_path),
    }
    return {
        "status": "passed",
        "gate": f"release-evidence/{attestation_path.name}",
        "artifacts": [published],
        "resolved_wheels": resolved_wheels,
    }


def _collect_wheelhouse(
    source: Path | None,
    output: Path,
    *,
    required: bool,
    expected_wheels: list[dict[str, Any]] | None = None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    if source is None:
        if required:
            raise ValueError("Signed release builds require --wheelhouse for zero-network offline installation.")
        return [], []
    wheelhouse = source.expanduser().resolve()
    wheels = sorted(wheelhouse.glob("*.whl")) if wheelhouse.is_dir() else []
    if not wheels:
        raise ValueError(f"Release wheelhouse contains no wheels: {wheelhouse}")
    for wheel in wheels:
        if wheel.is_symlink() or not wheel.is_file():
            raise ValueError(f"Wheelhouse entries must be regular files: {wheel}")
        if not zipfile.is_zipfile(wheel):
            raise ValueError(f"Wheelhouse entry is not a valid wheel archive: {wheel}")
    actual_wheels = {wheel.name: sha256_file(wheel) for wheel in wheels}
    expected = {str(row.get("name")): str(row.get("sha256")) for row in list(expected_wheels or [])}
    if required and actual_wheels != expected:
        raise ValueError("Release wheelhouse does not exactly match the passing gate wheel inventory.")
    destination = output / "wheelhouse"
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    artifacts: list[dict[str, Any]] = []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if archive.testzip() is not None:
                raise ValueError(f"Wheelhouse entry failed ZIP integrity validation: {wheel}")
            if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
                raise ValueError(f"Wheelhouse entry contains an unsafe archive path: {wheel}")
            metadata_roots = {
                name.split("/", 1)[0]
                for name in names
                if ".dist-info/" in name and name.split("/", 1)[0].endswith(".dist-info")
            }
            if len(metadata_roots) != 1:
                raise ValueError(f"Wheelhouse entry must contain exactly one dist-info directory: {wheel}")
            metadata_root = next(iter(metadata_roots))
            required_metadata = {
                f"{metadata_root}/METADATA",
                f"{metadata_root}/WHEEL",
                f"{metadata_root}/RECORD",
            }
            if not required_metadata.issubset(names):
                raise ValueError(f"Wheelhouse entry is missing required wheel metadata: {wheel}")
        target = destination / wheel.name
        shutil.copy2(wheel, target)
        copied.append(target)
        artifacts.append(
            {
                "name": target.name,
                "kind": "wheel",
                "path": f"wheelhouse/{target.name}",
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "media_type": "application/zip",
            }
        )
    return copied, artifacts


def _runtime_python(runtime_root: Path) -> Path:
    return runtime_root / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def _verify_offline_runtime_install(
    *,
    archive: Path,
    wheelhouse: Path,
    version: str,
) -> dict[str, Any]:
    """Prove the copied wheelhouse can install and import the full release offline."""

    with tempfile.TemporaryDirectory(prefix="oel-offline-release-smoke-") as temporary:
        runtime_root = Path(temporary) / "runtime"
        create = subprocess.run(
            [sys.executable, "-m", "venv", str(runtime_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        if create.returncode != 0:
            raise ValueError(f"Could not create offline-install smoke runtime: {create.stderr.strip()}")
        python = _runtime_python(runtime_root)
        install = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--find-links",
                str(wheelhouse),
                "--only-binary=:all:",
                f"{archive}[full]",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if install.returncode != 0:
            detail = (install.stderr or install.stdout).strip()
            raise ValueError(f"Offline wheelhouse dependency closure failed: {detail}")
        probe = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata; import sim; import sim.installation.cli; "
                    f"assert importlib.metadata.version('orbital-engineering-lab') == {version!r}"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout).strip()
            raise ValueError(f"Offline installed-runtime import probe failed: {detail}")
    return {
        "status": "passed",
        "network_used": False,
        "profile": "full",
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
    }


def build_release(
    *,
    source_root: Path,
    output_dir: Path,
    edition: str,
    channel: str,
    version: str,
    private_key: Path | None,
    public_keys: Path | None,
    base_url: str | None,
    developer_unsigned: bool,
    epoch: int,
    supply_chain_evidence: Path | None = None,
    wheelhouse: Path | None = None,
    platforms: list[str] | None = None,
    architecture: str | None = None,
    channel_url: str | None = None,
) -> dict[str, Any]:
    root = source_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if edition == "public":
        if (root / "docs" / "operations" / "public_surface_manifest.yaml").is_file():
            raise ValueError("Public installable releases must be built from the generated public export, not the private tree.")
        try:
            from tools.check_public_export import check_public_export
        except ModuleNotFoundError as exc:
            if not developer_unsigned:
                raise ValueError(
                    "Official signed public builds must run from the private release workspace so the "
                    "generated source can pass the complete public-boundary checker."
                ) from exc
        else:
            boundary_errors = check_public_export(root)
            if boundary_errors:
                raise ValueError("Public source boundary check failed:\n- " + "\n- ".join(boundary_errors))
    if edition not in {"public", "pro"} or channel not in {"stable", "preview"}:
        raise ValueError("Unsupported release edition or channel.")
    signing_key: RSAPrivateKey | None = None
    trusted_keys: dict[str, RSAPublicKey] = {}
    if not developer_unsigned:
        signing_key, trusted_keys = _official_signing_material(
            private_key=private_key,
            public_keys=public_keys,
        )
    if not developer_unsigned and (not base_url or not base_url.lower().startswith("https://")):
        raise ValueError("Signed release builds require an explicit HTTPS --base-url.")
    if not developer_unsigned and (not channel_url or not channel_url.lower().startswith("https://")):
        raise ValueError("Signed release builds require an explicit HTTPS --channel-url.")
    if base_url and not re.fullmatch(r"https://[A-Za-z0-9:/._~?&=%+\-]+", base_url):
        raise ValueError("Release base URL contains characters that are unsafe for rendered installer literals.")
    if channel_url and not re.fullmatch(r"https://[A-Za-z0-9:/._~?&=%+\-]+", channel_url):
        raise ValueError("Release channel URL contains characters that are unsafe for rendered installer literals.")
    if source_project_version(source_root=root) != version:
        raise ValueError("Requested version does not match source pyproject.toml.")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        if not any(part in _EXCLUDED_PARTS for part in output.relative_to(root).parts):
            raise ValueError("Release output inside the included source tree would make the artifact self-referential.")
    if private_key is not None:
        signing_key_path = private_key.expanduser().resolve()
        for boundary, label in ((root, "source root"), (output, "release output")):
            try:
                signing_key_path.relative_to(boundary)
            except ValueError:
                continue
            raise ValueError(f"Private release-signing keys may not be stored inside the {label}.")
    output.mkdir(parents=True, exist_ok=True)
    supply_chain = _collect_supply_chain_evidence(
        supply_chain_evidence,
        output,
        version=version,
        required=not developer_unsigned,
        public_source_root=root if edition == "public" and not developer_unsigned else None,
    )
    wheel_files, wheel_artifacts = _collect_wheelhouse(
        wheelhouse,
        output,
        required=not developer_unsigned,
        expected_wheels=list(supply_chain.get("resolved_wheels", []) or []),
    )
    archive = output / f"orbital-engineering-lab-{version}-{edition}.tar.gz"
    _build_source_archive(root, archive, version=version, epoch=epoch)
    offline_runtime_qualification = None
    if not developer_unsigned:
        offline_runtime_qualification = _verify_offline_runtime_install(
            archive=archive,
            wheelhouse=output / "wheelhouse",
            version=version,
        )
    artifact = {
        "name": archive.name,
        "kind": "source",
        "path": archive.name,
        **({"url": f"{base_url.rstrip('/')}/{archive.name}"} if base_url else {}),
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "media_type": "application/gzip",
    }
    published_at = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    manifest: dict[str, Any] = {
        "schema_version": RELEASE_MANIFEST_SCHEMA,
        # Preserve the v1 wire identity for already-installed updaters.
        "product": "orbital-engagement-lab",
        "edition": edition,
        "version": version,
        "source_commit": _source_commit(root),
        "channel": channel,
        "published_at": published_at,
        "artifacts": [artifact, *wheel_artifacts],
        "platforms": platforms or [platform.system()],
        "architecture": architecture or platform.machine(),
        "python": {"requires": ">=3.10,<3.15", "recommended": "3.14"},
        "profiles": ["core", "game", "accel", "validation", "mcp", "full"]
        + (["pro"] if edition == "pro" else []),
        "contracts": {
            "workspace": WORKSPACE_SCHEMA,
            "scenario": SCENARIO_SCHEMA,
            "fsw": "oel.fsw.boundary.v2",
            "candidate": "oel.fswdk.candidate.v1",
        },
        "constraints": _constraint_digests(root),
        "minimum_launcher_version": "0.25.0",
        "release_notes": "CHANGELOG.md",
        "claims": ["Immutable source artifact with content-bound installation metadata."],
        "non_claims": [
            "Installation does not establish physics validation beyond retained release evidence.",
            "OEL is not flight software or an operational decision system.",
        ],
        "supply_chain": {
            **supply_chain,
            **(
                {"offline_runtime_qualification": offline_runtime_qualification}
                if offline_runtime_qualification is not None
                else {}
            ),
        },
    }
    if developer_unsigned:
        signed = manifest
    else:
        assert signing_key is not None
        signed = sign_payload(manifest, signing_key)
        if not verify_payload(signed, trusted_keys):
            raise ValueError("Signed release manifest does not verify against the embedded trusted-key registry.")
    manifest_path = output / "release-manifest.json"
    manifest_path.write_text(json.dumps(signed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_path = output / "SHA256SUMS"
    checksum_path.write_text(f"{sha256_file(archive)}  {archive.name}\n{sha256_file(manifest_path)}  {manifest_path.name}\n")
    installers = _render_installers(output, public_keys=public_keys, base_url=base_url, channel_url=channel_url)
    bundle_files = [
        archive,
        manifest_path,
        checksum_path,
        *installers,
        *sorted((output / "release-evidence").glob("*")),
        *wheel_files,
    ]
    if public_keys is not None:
        trusted = output / "trusted-release-keys.json"
        shutil.copy2(public_keys, trusted)
        bundle_files.append(trusted)
    qualified_python = str(
        dict(offline_runtime_qualification or {}).get("python", platform.python_version())
    )
    qualified_match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", qualified_python)
    if qualified_match is None:
        raise ValueError("Offline runtime qualification has an invalid Python version.")
    python_tag = f"py{qualified_match.group(1)}{qualified_match.group(2)}"
    bundle = output / (
        f"oel-{edition}-{version}-{(architecture or platform.machine()).lower()}-{python_tag}.bundle.zip"
    )
    if wheel_files:
        _write_bundle(bundle, bundle_files, epoch=epoch)
    channel_index: dict[str, Any] = {
        "schema_version": CHANNEL_INDEX_SCHEMA,
        "edition": edition,
        "channel": channel,
        "latest": version,
        "manifest_url": f"{base_url.rstrip('/')}/release-manifest.json" if base_url else "release-manifest.json",
        "published_at": published_at,
    }
    if not developer_unsigned:
        assert signing_key is not None
        channel_index = sign_payload(channel_index, signing_key)
        if not verify_payload(channel_index, trusted_keys):
            raise ValueError("Signed update channel does not verify against the embedded trusted-key registry.")
    channel_path = output / f"{edition}-{channel}.json"
    channel_path.write_text(json.dumps(channel_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    release_asset_paths = [
        archive,
        manifest_path,
        channel_path,
        checksum_path,
        *installers,
        *([bundle] if wheel_files else []),
    ]
    trusted_release_keys = output / "trusted-release-keys.json"
    if trusted_release_keys.is_file():
        release_asset_paths.append(trusted_release_keys)
    receipt = {
        "schema_version": "oel.release-build-receipt.v1",
        "status": "ready",
        "edition": edition,
        "channel": channel,
        "version": version,
        "source_root": str(root),
        "source_commit": manifest["source_commit"],
        "artifact": str(archive),
        "manifest": str(manifest_path),
        "channel_index": str(channel_path),
        "channel_url": channel_url,
        "offline_bundle": str(bundle) if wheel_files else None,
        "installers": [str(path) for path in installers],
        "developer_unsigned": developer_unsigned,
        "supply_chain": supply_chain,
        "offline_runtime_qualification": offline_runtime_qualification,
        "release_assets": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(release_asset_paths, key=lambda item: item.name)
        ],
    }
    (output / "release-build-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an installable OEL release artifact and signed metadata.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edition", choices=("public", "pro"), required=True)
    parser.add_argument("--channel", choices=("stable", "preview"), default="stable")
    parser.add_argument("--version")
    parser.add_argument("--private-key", type=Path)
    parser.add_argument("--public-keys", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--channel-url")
    parser.add_argument("--developer-unsigned", action="store_true")
    parser.add_argument("--supply-chain-evidence", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--platform", action="append", dest="platforms")
    parser.add_argument("--architecture")
    args = parser.parse_args(argv)
    root = args.source_root.expanduser().resolve()
    version = args.version or source_project_version(source_root=root)
    if version is None:
        raise SystemExit("Could not determine the source project version.")
    epoch = max(int(os.environ.get("SOURCE_DATE_EPOCH", "315532800")), 315532800)
    receipt = build_release(
        source_root=root,
        output_dir=args.output_dir,
        edition=args.edition,
        channel=args.channel,
        version=version,
        private_key=args.private_key,
        public_keys=args.public_keys,
        base_url=args.base_url,
        developer_unsigned=bool(args.developer_unsigned),
        epoch=epoch,
        supply_chain_evidence=args.supply_chain_evidence,
        wheelhouse=args.wheelhouse,
        platforms=args.platforms,
        architecture=args.architecture,
        channel_url=args.channel_url,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
