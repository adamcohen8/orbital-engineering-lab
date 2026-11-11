"""Governed Pro equipment-library resolution and calibration dispositions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sim.pro_communications.contracts import LinkBudgetProfile
from sim.pro_communications.profiles import load_link_profiles

EQUIPMENT_LIBRARY_SCHEMA_VERSION = "oel.pro-communications-equipment-library.v0.1"
_CALIBRATION_STATUSES = {"not_calibrated", "source_documented", "measured_governed"}
_USE_DISPOSITIONS = {"illustrative_only", "preliminary_design", "calibrated_envelope"}


def _required(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GovernedEquipmentEntry:
    profile: LinkBudgetProfile
    calibration_status: str
    use_disposition: str
    applicability: str
    approved_by: str | None


@dataclass(frozen=True)
class GovernedEquipmentLibrary:
    library_id: str
    library_version: str
    data_classification: str
    source_path: Path
    source_sha256: str
    profile_file: Path
    profile_file_sha256: str
    entries: Mapping[str, GovernedEquipmentEntry]
    semantic_sha256: str


def load_governed_equipment_library(path: str | Path) -> GovernedEquipmentLibrary:
    """Load a strict manifest whose profile hashes and use dispositions are frozen."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Equipment library must be a JSON object.")
    allowed = {
        "schema_version",
        "library_id",
        "library_version",
        "data_classification",
        "profile_file",
        "entries",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown or payload.get("schema_version") != EQUIPMENT_LIBRARY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported or unknown equipment-library fields: {unknown}.")
    profile_file = (source.parent / _required(payload.get("profile_file"), "profile_file")).resolve()
    try:
        profile_file.relative_to(source.parent)
    except ValueError as exc:
        raise ValueError("Equipment-library profile_file must stay inside the library directory.") from exc
    profiles = load_link_profiles(profile_file)
    entries_payload = payload.get("entries")
    if not isinstance(entries_payload, dict) or set(entries_payload) != set(profiles):
        raise ValueError("Equipment-library entries must exactly match the referenced profiles.")
    entries: dict[str, GovernedEquipmentEntry] = {}
    for profile_id, raw in sorted(entries_payload.items()):
        if not isinstance(raw, dict):
            raise ValueError(f"Equipment entry {profile_id!r} must be an object.")
        entry_unknown = sorted(
            set(raw) - {"profile_sha256", "calibration_status", "use_disposition", "applicability", "approved_by"}
        )
        if entry_unknown:
            raise ValueError(f"Unknown equipment entry fields for {profile_id!r}: {entry_unknown}.")
        profile = profiles[profile_id]
        if raw.get("profile_sha256") != profile.semantic_sha256:
            raise ValueError(f"Equipment entry {profile_id!r} does not match its frozen profile SHA-256.")
        calibration = _required(raw.get("calibration_status"), "calibration_status")
        use = _required(raw.get("use_disposition"), "use_disposition")
        if calibration not in _CALIBRATION_STATUSES or use not in _USE_DISPOSITIONS:
            raise ValueError(f"Equipment entry {profile_id!r} has an unsupported governance disposition.")
        approved_by = None if raw.get("approved_by") is None else _required(raw["approved_by"], "approved_by")
        if use == "calibrated_envelope" and (calibration != "measured_governed" or approved_by is None):
            raise ValueError("calibrated_envelope requires measured_governed evidence and approved_by.")
        entries[profile_id] = GovernedEquipmentEntry(
            profile=profile,
            calibration_status=calibration,
            use_disposition=use,
            applicability=_required(raw.get("applicability"), "applicability"),
            approved_by=approved_by,
        )
    identity = {
        "schema_version": EQUIPMENT_LIBRARY_SCHEMA_VERSION,
        "library_id": _required(payload.get("library_id"), "library_id"),
        "library_version": _required(payload.get("library_version"), "library_version"),
        "data_classification": _required(payload.get("data_classification"), "data_classification"),
        "source_sha256": _sha256(source),
        "profile_file_sha256": _sha256(profile_file),
        "profiles": {key: value.profile.semantic_sha256 for key, value in entries.items()},
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return GovernedEquipmentLibrary(
        library_id=identity["library_id"],
        library_version=identity["library_version"],
        data_classification=identity["data_classification"],
        source_path=source,
        source_sha256=identity["source_sha256"],
        profile_file=profile_file,
        profile_file_sha256=identity["profile_file_sha256"],
        entries=entries,
        semantic_sha256=semantic_sha256,
    )


__all__ = [
    "EQUIPMENT_LIBRARY_SCHEMA_VERSION",
    "GovernedEquipmentEntry",
    "GovernedEquipmentLibrary",
    "load_governed_equipment_library",
]
