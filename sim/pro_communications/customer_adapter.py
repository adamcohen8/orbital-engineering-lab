"""Fail-closed customer profile adapter with evidence-safe redaction controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sim.pro_communications.contracts import LinkBudgetProfile
from sim.pro_communications.profiles import link_profile_from_mapping

CUSTOMER_ADAPTER_SCHEMA_VERSION = "oel.pro-communications-customer-adapter.v0.1"
_PROHIBITED_CLASSIFICATIONS = {"classified", "cui", "export_controlled"}
_DEFAULT_REDACTED_FIELDS = frozenset(
    {
        "account_id",
        "contact",
        "customer_id",
        "customer_name",
        "location_address",
        "proprietary_notes",
        "serial_number",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact(value: Any, fields: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if str(key).lower() in fields else _redact(item, fields)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, fields) for item in value]
    return value


@dataclass(frozen=True)
class CustomerAdapterPolicy:
    allowed_classifications: tuple[str, ...] = ("customer_proprietary",)
    redacted_fields: frozenset[str] = _DEFAULT_REDACTED_FIELDS

    def __post_init__(self) -> None:
        allowed = tuple(sorted(str(value or "").strip().lower() for value in self.allowed_classifications))
        if not allowed or any(not value for value in allowed) or set(allowed) & _PROHIBITED_CLASSIFICATIONS:
            raise ValueError("Customer adapter policy must allow only explicit non-CUI classifications.")
        fields = frozenset(str(value or "").strip().lower() for value in self.redacted_fields)
        if not fields or any(not value for value in fields):
            raise ValueError("redacted_fields must contain non-empty field names.")
        object.__setattr__(self, "allowed_classifications", allowed)
        object.__setattr__(self, "redacted_fields", fields)


@dataclass(frozen=True)
class AdaptedCustomerProfiles:
    adapter_id: str
    data_classification: str
    authorization_id_sha256: str
    source_sha256: str
    profiles: Mapping[str, LinkBudgetProfile]
    redacted_receipt: Mapping[str, Any]
    semantic_sha256: str


def adapt_customer_profile_file(
    path: str | Path,
    *,
    policy: CustomerAdapterPolicy | None = None,
) -> AdaptedCustomerProfiles:
    """Resolve authorized profiles while retaining only redacted evidence metadata."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Customer adapter source must be a JSON object.")
    allowed_fields = {
        "schema_version",
        "adapter_id",
        "customer_id",
        "data_classification",
        "authorization_id",
        "profiles",
        "metadata",
    }
    unknown = sorted(set(payload) - allowed_fields)
    if unknown or payload.get("schema_version") != CUSTOMER_ADAPTER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported or unknown customer adapter fields: {unknown}.")
    resolved_policy = policy or CustomerAdapterPolicy()
    classification = str(payload.get("data_classification") or "").strip().lower()
    if classification in _PROHIBITED_CLASSIFICATIONS:
        raise PermissionError(
            "Classified, CUI, and export-controlled inputs require a separately authorized environment."
        )
    if classification not in resolved_policy.allowed_classifications:
        raise PermissionError(f"Customer data classification {classification!r} is not authorized by this policy.")
    authorization = str(payload.get("authorization_id") or "").strip()
    if not authorization:
        raise PermissionError("Customer profile adaptation requires a non-empty authorization_id.")
    adapter_id = str(payload.get("adapter_id") or "").strip()
    customer_id = str(payload.get("customer_id") or "").strip()
    if not adapter_id or not customer_id:
        raise ValueError("adapter_id and customer_id must be non-empty strings.")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("Customer adapter source must contain at least one profile.")
    profiles = {str(key): link_profile_from_mapping(value) for key, value in sorted(raw_profiles.items())}
    if any(key != profile.profile_id for key, profile in profiles.items()):
        raise ValueError("Customer profile keys must equal profile_id.")
    source_sha256 = _sha256(source)
    authorization_sha256 = hashlib.sha256(authorization.encode()).hexdigest()
    customer_sha256 = hashlib.sha256(customer_id.encode()).hexdigest()
    redacted = _redact(payload.get("metadata", {}), resolved_policy.redacted_fields)
    receipt = {
        "schema_version": CUSTOMER_ADAPTER_SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "customer_id_sha256": customer_sha256,
        "data_classification": classification,
        "authorization_id_sha256": authorization_sha256,
        "source_sha256": source_sha256,
        "profile_semantic_sha256": {key: value.semantic_sha256 for key, value in profiles.items()},
        "redacted_metadata": redacted,
        "redacted_fields": sorted(resolved_policy.redacted_fields),
        "claims": ["Profile structure and provenance were validated under the declared local authorization."],
        "non_claims": ["The receipt contains no raw customer identifier, authorization token, or unredacted metadata."],
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return AdaptedCustomerProfiles(
        adapter_id=adapter_id,
        data_classification=classification,
        authorization_id_sha256=authorization_sha256,
        source_sha256=source_sha256,
        profiles=profiles,
        redacted_receipt=receipt,
        semantic_sha256=semantic_sha256,
    )


__all__ = [
    "CUSTOMER_ADAPTER_SCHEMA_VERSION",
    "AdaptedCustomerProfiles",
    "CustomerAdapterPolicy",
    "adapt_customer_profile_file",
]
