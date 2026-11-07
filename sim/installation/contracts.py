"""Versioned contracts and content identity for managed OEL installations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from sim.schema_versions import LEGACY_SCENARIO_SCHEMA_VERSION, SCENARIO_SCHEMA_VERSION, WORKSPACE_SCHEMA_VERSION

RELEASE_MANIFEST_SCHEMA = "oel.release-manifest.v1"
INSTALLATION_RECORD_SCHEMA = "oel.installation-record.v1"
INSTALLATION_STATE_SCHEMA = "oel.installation-state.v1"
WORKSPACE_SCHEMA = WORKSPACE_SCHEMA_VERSION
TEMPLATE_MANIFEST_SCHEMA = "oel.template-manifest.v1"
COMPATIBILITY_REPORT_SCHEMA = "oel.workspace-compatibility.v1"
MIGRATION_PLAN_SCHEMA = "oel.workspace-migration-plan.v1"
UPDATE_RECEIPT_SCHEMA = "oel.update-receipt.v1"
CHANNEL_INDEX_SCHEMA = "oel.release-channel.v1"
CHANNEL_CONFIG_SCHEMA = "oel.update-channels.v1"
SCENARIO_SCHEMA = SCENARIO_SCHEMA_VERSION
LEGACY_SCENARIO_SCHEMA = LEGACY_SCENARIO_SCHEMA_VERSION

SUPPORTED_RELEASE_EDITIONS = frozenset({"public", "pro"})
SUPPORTED_CHANNELS = frozenset({"stable", "preview"})
# The signed v1 product is a wire identity. Existing updaters still require
# the former value; readers accept either spelling.
SUPPORTED_RELEASE_PRODUCTS = frozenset({"orbital-engagement-lab", "orbital-engineering-lab"})
# ``untrusted`` is retained as a distinct disposition for legacy signed
# releases, revoked-key installs, and installations whose external trust
# anchor cannot authenticate the signed inventory.  Their bytes remain
# inspectable but they must not be activated or dispatched as official OEL.
INSTALLATION_DISPOSITIONS = frozenset({"official", "modified", "incomplete", "developer", "untrusted"})
COMPATIBILITY_STATUSES = frozenset(
    {
        "compatible",
        "compatible_with_warnings",
        "migration_available",
        "manual_review",
        "blocked",
        "invalid",
        "incomplete",
        "cancelled",
    }
)

_VERSION_RE = re.compile(r"^(0|[1-9]\d*)(?:\.(0|[1-9]\d*)){1,3}(?:[-+][0-9A-Za-z.-]+)?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when a managed-installation contract is malformed."""


def canonical_json_bytes(value: Any, *, omit_signature: bool = False) -> bytes:
    normalized = value
    if omit_signature and isinstance(value, Mapping):
        normalized = {str(key): item for key, item in value.items() if str(key) != "signature"}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: str | Path, *, excluded_names: Iterable[str] = ()) -> str:
    base = Path(root).expanduser().resolve()
    excluded = set(excluded_names)
    digest = hashlib.sha256()
    for path in sorted(item for item in base.rglob("*") if item.is_file() and item.name not in excluded):
        if path.is_symlink():
            raise ContractError(f"Content-bound trees may not contain symbolic links: {path}")
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"JSON is invalid at {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {source}")
    return value


def require_keys(value: Mapping[str, Any], keys: Iterable[str], *, label: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ContractError(f"{label} is missing required field(s): {', '.join(sorted(missing))}.")


def reject_unknown_keys(value: Mapping[str, Any], keys: Iterable[str], *, label: str) -> None:
    allowed = set(keys)
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ContractError(f"{label} has unsupported field(s): {', '.join(unknown)}.")


def validate_version(value: Any, *, label: str = "version") -> str:
    version = str(value or "").strip()
    if not _VERSION_RE.fullmatch(version):
        raise ContractError(f"{label} must be a dotted version string, got {value!r}.")
    return version


def validate_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest.")
    return digest


def version_tuple(value: str) -> tuple[int, ...]:
    core = str(value).split("+", 1)[0].split("-", 1)[0]
    try:
        return tuple(int(item) for item in core.split("."))
    except ValueError:
        return ()


def version_satisfies(version: str, requirement: str | None) -> bool:
    """Evaluate the deliberately small version-range grammar used by OEL manifests."""

    if requirement in (None, "", "*"):
        return True
    candidate = version_tuple(version)
    if not candidate:
        return False
    for raw_clause in str(requirement).split(","):
        clause = raw_clause.strip()
        if not clause:
            continue
        match = re.fullmatch(r"(==|>=|<=|>|<)?\s*([0-9]+(?:\.[0-9]+){1,3})", clause)
        if match is None:
            return False
        operator = match.group(1) or "=="
        expected = version_tuple(match.group(2))
        width = max(len(candidate), len(expected))
        left = candidate + (0,) * (width - len(candidate))
        right = expected + (0,) * (width - len(expected))
        comparison = (left > right) - (left < right)
        if operator == "==" and comparison != 0:
            return False
        if operator == ">=" and comparison < 0:
            return False
        if operator == "<=" and comparison > 0:
            return False
        if operator == ">" and comparison <= 0:
            return False
        if operator == "<" and comparison >= 0:
            return False
    return True


def validate_artifact(value: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    label = f"release manifest artifacts[{index}]"
    allowed = {"name", "kind", "path", "url", "bytes", "sha256", "media_type", "platform", "architecture"}
    require_keys(value, {"name", "kind", "bytes", "sha256"}, label=label)
    reject_unknown_keys(value, allowed, label=label)
    if not isinstance(value.get("name"), str) or not str(value["name"]).strip():
        raise ContractError(f"{label}.name must be a non-empty string.")
    if not isinstance(value.get("kind"), str) or not str(value["kind"]).strip():
        raise ContractError(f"{label}.kind must be a non-empty string.")
    if not value.get("path") and not value.get("url"):
        raise ContractError(f"{label} must provide path or url.")
    for field in ("path", "url"):
        if field in value and value[field] is not None and not isinstance(value[field], str):
            raise ContractError(f"{label}.{field} must be a string.")
    name = str(value["name"]).strip()
    if not name or Path(name).name != name:
        raise ContractError(f"{label}.name must be a plain file name.")
    if isinstance(value["bytes"], bool) or not isinstance(value["bytes"], int):
        raise ContractError(f"{label}.bytes must be an integer.")
    size = value["bytes"]
    if size < 0:
        raise ContractError(f"{label}.bytes must be non-negative.")
    return {
        "name": name,
        "kind": str(value["kind"]),
        **({"path": str(value["path"])} if value.get("path") else {}),
        **({"url": str(value["url"])} if value.get("url") else {}),
        "bytes": size,
        "sha256": validate_sha256(value["sha256"], label=f"{label}.sha256"),
        **({"media_type": str(value["media_type"])} if value.get("media_type") else {}),
        **({"platform": str(value["platform"])} if value.get("platform") else {}),
        **({"architecture": str(value["architecture"])} if value.get("architecture") else {}),
    }


def validate_release_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "product",
        "edition",
        "version",
        "source_commit",
        "channel",
        "published_at",
        "artifacts",
        "platforms",
        "architecture",
        "python",
        "profiles",
        "contracts",
        "constraints",
        "minimum_launcher_version",
        "release_notes",
        "claims",
        "non_claims",
        "license",
        "supply_chain",
        "signature",
    }
    required = {
        "schema_version",
        "product",
        "edition",
        "version",
        "channel",
        "published_at",
        "artifacts",
        "platforms",
        "python",
        "profiles",
        "contracts",
    }
    require_keys(value, required, label="release manifest")
    reject_unknown_keys(value, allowed, label="release manifest")
    if value["schema_version"] != RELEASE_MANIFEST_SCHEMA:
        raise ContractError(f"release manifest schema_version must be {RELEASE_MANIFEST_SCHEMA!r}.")
    if value["product"] not in SUPPORTED_RELEASE_PRODUCTS:
        raise ContractError(f"release manifest product must be one of {sorted(SUPPORTED_RELEASE_PRODUCTS)}.")
    edition = str(value["edition"])
    if edition not in SUPPORTED_RELEASE_EDITIONS:
        raise ContractError(f"release manifest edition must be one of {sorted(SUPPORTED_RELEASE_EDITIONS)}.")
    channel = str(value["channel"])
    if channel not in SUPPORTED_CHANNELS:
        raise ContractError(f"release manifest channel must be one of {sorted(SUPPORTED_CHANNELS)}.")
    artifacts_value = value["artifacts"]
    if not isinstance(artifacts_value, list) or not artifacts_value:
        raise ContractError("release manifest artifacts must be a non-empty list.")
    artifacts = [validate_artifact(item, index=index) for index, item in enumerate(artifacts_value) if isinstance(item, Mapping)]
    if len(artifacts) != len(artifacts_value):
        raise ContractError("release manifest artifacts must contain only objects.")
    contracts = value["contracts"]
    if not isinstance(contracts, Mapping):
        raise ContractError("release manifest contracts must be an object.")
    if not {"workspace", "scenario"}.issubset(contracts):
        raise ContractError("release manifest contracts must declare workspace and scenario versions.")
    platforms = value["platforms"]
    profiles = value["profiles"]
    python = value["python"]
    if not isinstance(platforms, list) or not platforms or not all(isinstance(item, str) and item for item in platforms):
        raise ContractError("release manifest platforms must be a non-empty string list.")
    if not isinstance(profiles, list) or not profiles or not all(isinstance(item, str) and item for item in profiles):
        raise ContractError("release manifest profiles must be a non-empty string list.")
    if value.get("architecture") is not None and not str(value.get("architecture", "") or "").strip():
        raise ContractError("release manifest architecture must be a non-empty string when present.")
    constraints = value.get("constraints", {})
    if not isinstance(constraints, Mapping):
        raise ContractError("release manifest constraints must be an object when present.")
    for name, digest in constraints.items():
        if Path(str(name)).name != str(name):
            raise ContractError(f"release manifest constraint name is unsafe: {name!r}.")
        validate_sha256(digest, label=f"release manifest constraints.{name}")
    if not isinstance(python, Mapping) or not str(python.get("requires", "") or "").strip():
        raise ContractError("release manifest python.requires must be declared.")
    published_at = str(value["published_at"])
    try:
        datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("release manifest published_at must be an ISO-8601 timestamp.") from exc
    signature = value.get("signature")
    if signature is not None:
        if not isinstance(signature, Mapping):
            raise ContractError("release manifest signature must be an object.")
        reject_unknown_keys(signature, {"alg", "key_id", "value"}, label="release manifest signature")
        require_keys(signature, {"alg", "key_id", "value"}, label="release manifest signature")
        if signature.get("alg") != "RS256" or not signature.get("key_id") or not signature.get("value"):
            raise ContractError("release manifest signature must contain a non-empty RS256 key_id and value.")
    return {**dict(value), "version": validate_version(value["version"]), "edition": edition, "channel": channel, "artifacts": artifacts}


def release_manifest_digest(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))
