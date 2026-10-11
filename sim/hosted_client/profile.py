"""Content-bound local profiles for a Hosted OEL transport and scoped session."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from sim.study_planning import canonical_sha256

HOSTED_PROFILE_SCHEMA = "oel.hosted_client_profile.v1"


def link_profile(
    path: str | Path,
    *,
    service_label: str,
    transport_command: Sequence[str],
    session_token: Mapping[str, Any],
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    label = str(service_label).strip()
    command = [str(item) for item in transport_command]
    if not label or not command or any(not item for item in command):
        raise ValueError("Hosted profile requires a service label and an argv transport command.")
    if float(timeout_s) <= 0:
        raise ValueError("Hosted profile transport timeout must be positive.")
    token = dict(session_token)
    if token.get("schema") != "oel.hosted_session_token.v1":
        raise ValueError("Hosted profile requires a signed Hosted OEL session token.")
    profile = {
        "schema": HOSTED_PROFILE_SCHEMA,
        "profile_id": "pending",
        "service_label": label,
        "transport": {
            "kind": "subprocess_json_v1",
            "command": command,
            "timeout_s": float(timeout_s),
            "shell": False,
        },
        "session_token": token,
        "tenant_id": str(token.get("tenant_id", "")),
        "linked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_credentials_stored": False,
    }
    profile["profile_id"] = f"hosted-profile:{canonical_sha256(profile)[:24]}"
    profile["profile_sha256"] = canonical_sha256(profile)
    _write_private_json(Path(path), profile)
    return profile


def load_profile(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().absolute()
    if any(p.is_symlink() for p in (source, *source.parents)) or not source.is_file():
        raise ValueError("Hosted profile must be a regular file.")
    if os.name == "posix" and stat.S_IMODE(source.stat().st_mode) & 0o077:
        raise PermissionError("Hosted profile must not be readable or writable by group or others.")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != HOSTED_PROFILE_SCHEMA:
        raise ValueError("Unsupported Hosted OEL profile schema.")
    expected = canonical_sha256(
        {key: item for key, item in value.items() if key != "profile_sha256"}
    )
    if value.get("profile_sha256") != expected:
        raise ValueError("Hosted OEL profile failed digest verification.")
    transport = value.get("transport")
    if isinstance(transport, Mapping) and transport.get("kind") == "https_json_v1":
        from .https_transport import validate_https_transport
        validate_https_transport(transport)
        if value.get("tenant_id") != value.get("session_token", {}).get("tenant_id"):
            raise ValueError("Hosted profile tenant differs from its session")
        return value
    if (
        not isinstance(transport, Mapping)
        or transport.get("kind") != "subprocess_json_v1"
        or transport.get("shell") is not False
        or not isinstance(transport.get("command"), list)
        or not transport["command"]
    ):
        raise ValueError("Hosted OEL profile contains an unsupported transport.")
    return value


def link_https_profile(path, *, service_label, service_url, account_key, scopes,
                       ca_file=None, timeout_s=30.0):
    """Exchange a key once; the profile retains only its short-lived session."""
    from .https_transport import HttpsJsonTransport, validate_https_transport
    transport = {"kind": "https_json_v1", "url": service_url, "timeout_s": timeout_s}
    if ca_file:
        ca = Path(ca_file).expanduser().resolve(strict=True)
        transport.update(ca_file=str(ca), ca_sha256=hashlib.sha256(ca.read_bytes()).hexdigest())
    validate_https_transport(transport)
    if not str(service_label).strip():
        raise ValueError("Hosted profile requires a service label")
    token = HttpsJsonTransport({"transport": transport}).exchange(account_key, scopes)
    if token.get("schema") != "oel.hosted_session_token.v1" or not token.get("tenant_id"):
        raise ValueError("Service returned an invalid account session")
    profile = {"schema": HOSTED_PROFILE_SCHEMA, "profile_id": "pending", "service_label": service_label,
        "transport": transport, "session_token": token, "tenant_id": token["tenant_id"],
        "linked_at": datetime.now(timezone.utc).isoformat(), "model_credentials_stored": False}
    profile["profile_id"] = "hosted-profile:" + canonical_sha256(profile)[:24]
    profile["profile_sha256"] = canonical_sha256(profile)
    _write_private_json(Path(path), profile)
    return profile


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["HOSTED_PROFILE_SCHEMA", "link_profile", "link_https_profile", "load_profile"]
