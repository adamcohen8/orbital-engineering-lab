"""Small stdlib-only official-install verifier used by stable launchers.

This module is copied, together with the exact signed-contract, signing, and
inventory helpers, to ``<data-root>/launcher/trusted-verifier``.  The copied
bundle is outside every managed version root.  The stable launcher runs this
module with an operator-owned Python before it imports any module from the
selected managed runtime or source tree.

Keep this module limited to authentication and path selection.  It must not
import the managed ``sim`` package, read mutable installation receipts as
authority, or execute a version's code before the signed manifest, external
key anchor, source inventory, and runtime inventory have passed.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    INSTALLATION_RECORD_SCHEMA,
    ContractError,
    load_json_object,
    release_manifest_digest,
    sha256_file,
    validate_release_manifest,
    validate_version,
)
from .integrity import validate_installation_integrity, verify_runtime_inventory, verify_source_inventory
from .signing import load_release_public_keys, verify_payload

TRUSTED_VERIFIER_SCHEMA = "oel.trusted-verifier.v1"


def _record_path(root: Path, raw: Any, *, label: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ContractError(f"{label} path is missing.")
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"{label} path must be relative to the managed version root.")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{label} path may not contain a symbolic link: {candidate}")
    resolved = candidate.resolve(strict=False)
    if resolved != candidate:
        raise ContractError(f"{label} path is not canonical: {candidate}")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{label} path escapes the managed version root.") from exc
    return candidate


def _anchor_path(data_root: Path, config_root: Path) -> Path:
    """Return the external key anchor, refusing a version-root placement."""

    data_versions = (data_root / "versions").resolve(strict=False)
    config = config_root.resolve(strict=False)
    try:
        config.relative_to(data_versions)
    except ValueError:
        pass
    else:
        raise ContractError("Official release trust anchor must be outside managed version roots.")
    anchor = config / "trusted-release-keys.json"
    if anchor.is_symlink():
        raise ContractError("Official release trust anchor may not be a symbolic link.")
    if not anchor.is_file():
        raise ContractError("Official release trust anchor is missing outside managed version roots.")
    return anchor


def _manifest_cache_path(config_root: Path, version: str) -> Path:
    cache_root = config_root.expanduser().resolve(strict=False) / "trusted-manifests"
    if cache_root.is_symlink():
        raise ContractError("Official signed-manifest receipt directory may not be a symbolic link.")
    if cache_root.exists() and not cache_root.is_dir():
        raise ContractError("Official signed-manifest receipt directory must be a directory.")
    cache = cache_root / f"{version}.json"
    if cache.is_symlink():
        raise ContractError("Official signed-manifest receipt may not be a symbolic link.")
    return cache


def _runtime_python(root: Path, runtime: Mapping[str, Any]) -> Path:
    if not runtime.get("created"):
        raise ContractError("Official managed installations must contain an authenticated managed runtime.")
    raw = str(runtime.get("python") or "").strip()
    if raw and Path(raw).is_absolute():
        raise ContractError("Managed runtime interpreter must be the canonical relative runtime provider alias.")
    python = root / "runtime" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if raw:
        candidate = root / Path(raw)
        if candidate != python:
            raise ContractError("Managed runtime interpreter record does not match the signed runtime layout.")
    if not python.exists() and not python.is_symlink():
        raise ContractError("Managed runtime interpreter is missing.")
    return python


def _trusted_external_provider() -> Path:
    """Return the operator-owned interpreter running this preflight bundle."""

    candidate = Path(getattr(sys, "_base_executable", sys.executable)).expanduser()
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise ContractError("Trusted external verifier Python provider is unavailable.") from exc


def verify_official_installation(
    version: str,
    *,
    data_root: str | Path,
    config_root: str | Path,
) -> dict[str, Any]:
    """Authenticate one managed version before any managed import is allowed."""

    target = validate_version(version)
    data = Path(data_root).expanduser().resolve(strict=False)
    if (data / "versions").is_symlink():
        raise ContractError("Managed version directory may not be a symbolic link.")
    root = data / "versions" / target
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"Managed OEL version root is missing or linked: {root}")
    record_path = root / "installation-record.json"
    manifest_path = root / "release-manifest.json"
    if record_path.is_symlink() or manifest_path.is_symlink():
        raise ContractError("Managed installation authority files may not be symbolic links.")
    record = load_json_object(record_path)
    if record.get("schema_version") != INSTALLATION_RECORD_SCHEMA:
        raise ContractError("Managed installation record schema is unsupported.")
    if str(record.get("version", "")) != target or record.get("status") != "official":
        raise ContractError("Managed installation is not an official release for its selected version.")
    manifest = validate_release_manifest(load_json_object(manifest_path))
    if manifest.get("version") != target:
        raise ContractError("Signed release manifest version does not match its managed version root.")
    for field in ("edition", "channel"):
        if record.get(field) != manifest.get(field):
            raise ContractError(f"Managed installation {field} does not match the signed release manifest.")
    recorded_artifact = record.get("release_artifact")
    if not isinstance(recorded_artifact, Mapping):
        raise ContractError("Managed installation source artifact record is malformed.")
    artifact_match = any(
        isinstance(item, Mapping)
        and item.get("kind") in {"source", "source_bundle"}
        and all(recorded_artifact.get(field) == item.get(field) for field in ("name", "bytes", "sha256"))
        for item in list(manifest.get("artifacts", []) or [])
    )
    if not artifact_match:
        raise ContractError("Managed installation source artifact does not match the signed release manifest.")
    profile = str(record.get("profile", ""))
    if profile not in list(manifest.get("profiles", []) or []):
        raise ContractError("Managed installation profile is not declared by the signed release manifest.")
    if record.get("release_manifest_sha256") != release_manifest_digest(manifest):
        raise ContractError("Managed release manifest digest does not match its installation record.")

    anchor = _anchor_path(data, Path(config_root).expanduser())
    keys = load_release_public_keys(anchor)
    if not verify_payload(manifest, keys):
        raise ContractError("Signed release manifest failed the external trusted-key check.")
    cached_path = _manifest_cache_path(Path(config_root).expanduser(), target)
    if not cached_path.is_file():
        raise ContractError("Official signed-manifest receipt is missing; reverification is required.")
    cached = validate_release_manifest(load_json_object(cached_path))
    if not verify_payload(cached, keys) or release_manifest_digest(cached) != release_manifest_digest(manifest):
        raise ContractError("Official signed-manifest receipt does not match the managed manifest.")
    supply_chain = manifest.get("supply_chain")
    if not isinstance(supply_chain, Mapping):
        raise ContractError("Signed release manifest has no supply-chain integrity payload.")
    integrity = validate_installation_integrity(dict(supply_chain.get("installation_integrity", {}) or {}))
    if str(integrity.get("profile", "")) not in {str(item) for item in manifest.get("profiles", []) or []}:
        raise ContractError("Signed installation integrity default profile is not declared by the release manifest.")
    runtime_profiles = dict(integrity.get("runtime_profiles", {}) or {})
    missing_profiles = sorted(
        set(str(item) for item in manifest.get("profiles", []) or []) - set(runtime_profiles)
    )
    if missing_profiles:
        raise ContractError(
            "Signed installation integrity lacks profile runtime inventories: " + ", ".join(missing_profiles)
        )
    selected_runtime = runtime_profiles.get(profile)
    if not isinstance(selected_runtime, Mapping):
        raise ContractError("Signed installation integrity lacks the selected profile runtime inventory.")

    source_record = record.get("source")
    if not isinstance(source_record, Mapping):
        raise ContractError("Managed installation source record is malformed.")
    source = _record_path(root, source_record.get("path", ""), label="managed source")
    if not source.is_dir():
        raise ContractError("Managed source directory is missing.")
    issues = verify_source_inventory(source, integrity["source"])
    diagnostics: list[str] = []

    runtime_record = record.get("runtime")
    if not isinstance(runtime_record, Mapping):
        raise ContractError("Managed installation runtime record is malformed.")
    python = _runtime_python(root, runtime_record)
    if runtime_record.get("created"):
        trusted_provider = _trusted_external_provider()
        issues.extend(
            verify_runtime_inventory(
                root / "runtime",
                selected_runtime,
                interpreter=python,
                trusted_provider=trusted_provider,
            )
        )

    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        issues.append("managed installation provenance is missing")
    else:
        if provenance.get("authority") != "external-trusted-release-key-registry":
            issues.append("managed installation provenance is not externally anchored")
        if provenance.get("trust_anchor") != str(anchor):
            issues.append("managed installation provenance does not match the external key-anchor path")
        recorded_anchor_digest = provenance.get("trust_anchor_sha256")
        if not isinstance(recorded_anchor_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded_anchor_digest):
            issues.append("managed installation provenance historical trust-anchor digest is missing or malformed")
        elif recorded_anchor_digest != sha256_file(anchor):
            diagnostics.append(
                "managed installation provenance historical anchor digest differs from the current external key anchor"
            )
        if provenance.get("installation_integrity_sha256") != integrity.get("digest"):
            issues.append("managed installation provenance does not match signed installation integrity")
        if provenance.get("runtime_inventory_sha256") != selected_runtime.get("digest"):
            issues.append("managed installation provenance does not match selected runtime inventory")
        signer = str(dict(manifest.get("signature", {}) or {}).get("key_id", ""))
        if provenance.get("manifest_signer_key_id") != signer:
            issues.append("managed installation provenance signer does not match the signed manifest")

    if issues:
        raise ContractError("Official managed installation failed preflight: " + "; ".join(sorted(set(issues))[:12]))
    return {
        "schema_version": TRUSTED_VERIFIER_SCHEMA,
        "status": "official",
        "version": target,
        "root": str(root),
        "source": str(source),
        "python": str(python),
        "anchor": str(anchor),
        **({"diagnostics": diagnostics} if diagnostics else {}),
    }
