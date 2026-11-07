"""Standalone first-install bootstrap for a verified OEL release.

This file is rendered by ``tools/build_installable_release.py`` with a trusted
release-key registry, then its digest is embedded in the platform bootstrap.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

# __OEL_INTEGRITY_TRANSPORT__

TRUSTED_KEYS_B64 = "__OEL_TRUSTED_KEYS_B64__"
TRUSTED_KEYS_RENDERED = "__OEL_TRUSTED_KEYS_RENDERED__"
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_RELEASE_BYTES = 8 * 1024 * 1024 * 1024
_SHA256_DIGESTINFO_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")


def _fetch(url: str, *, maximum: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "OEL-Bootstrap/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                declared = int(length)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Remote content has an invalid Content-Length header.") from exc
            if declared > maximum:
                raise RuntimeError(f"Remote content exceeds the {maximum} byte safety limit.")
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise RuntimeError(f"Remote content exceeds the {maximum} byte safety limit.")
    return data


def _fetch_to_file(url: str, destination: Path, *, maximum: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "OEL-Bootstrap/1"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as stream:
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                declared = int(length)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Remote content has an invalid Content-Length header.") from exc
            if declared > maximum:
                raise RuntimeError(f"Remote content exceeds the {maximum} byte safety limit.")
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RuntimeError(f"Remote content exceeds the {maximum} byte safety limit.")
            stream.write(chunk)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    raw = text.encode("ascii")
    return int.from_bytes(base64.urlsafe_b64decode(raw + b"=" * ((4 - len(raw) % 4) % 4)), "big")


def _is_os_path_alias(path: Path) -> bool:
    """Allow the standard macOS aliases while rejecting arbitrary link routing."""

    if os.name == "nt":
        return False
    aliases = {Path("/tmp"): Path("/private/tmp"), Path("/var"): Path("/private/var")}
    target = aliases.get(path)
    return target is not None and path.resolve(strict=False) == target


def _validated_destination(destination: str | Path) -> Path:
    requested = Path(destination).expanduser().absolute()
    for node in (requested, *requested.parents):
        if node.is_symlink() and not _is_os_path_alias(node):
            raise RuntimeError(f"Extraction destination may not use a symbolic-link parent: {node}")
    return requested.resolve(strict=False)


def _validate_trusted_keys_payload(value: Any) -> dict[str, Any]:
    """Validate the embedded official release-key registry before use."""

    if not isinstance(value, dict) or not isinstance(value.get("keys"), list):
        raise SystemExit("Embedded trusted release-key registry is malformed.")
    seen: set[str] = set()
    for item in value["keys"]:
        if not isinstance(item, dict):
            raise SystemExit("Embedded trusted release-key registry contains a malformed key.")
        key_id = item.get("key_id")
        if not isinstance(key_id, str) or not key_id.strip() or key_id in seen:
            raise SystemExit("Embedded trusted release-key registry contains duplicate or empty key IDs.")
        seen.add(key_id)
        try:
            modulus = _decode_int(item["n"])
            exponent = _decode_int(item.get("e", "AQAB"))
        except (KeyError, TypeError, ValueError, UnicodeError, binascii.Error) as exc:
            raise SystemExit("Embedded trusted release-key registry contains an invalid RSA key.") from exc
        revoked = item.get("revoked", False)
        if not isinstance(revoked, bool):
            raise SystemExit("Embedded trusted release-key registry contains an invalid revoked flag.")
        if modulus <= 0 or item.get("alg", "RS256") != "RS256" or (
            not revoked and (modulus.bit_length() < 2048 or exponent != 65537)
        ):
            raise SystemExit("Official trusted release keys must be RS256, RSA-2048 or stronger, with e=65537.")
    if not seen:
        raise SystemExit("Embedded trusted release-key registry is empty.")
    return value


def _validate_bootstrap_manifest(value: Any, *, require_integrity: bool = True) -> dict[str, Any]:
    """Validate fields used before the signed artifact is written to disk."""

    if not isinstance(value, dict) or value.get("schema_version") != "oel.release-manifest.v1":
        raise SystemExit("Release manifest has an unsupported schema.")
    required = {
        "schema_version", "product", "edition", "version", "channel", "published_at",
        "artifacts", "platforms", "python", "profiles", "contracts",
    }
    if not required.issubset(value):
        raise SystemExit("Release manifest is missing required fields.")
    allowed = {
        "schema_version", "product", "edition", "version", "source_commit", "channel",
        "published_at", "artifacts", "platforms", "architecture", "python", "profiles",
        "contracts", "constraints", "minimum_launcher_version", "release_notes", "claims",
        "non_claims", "license", "supply_chain", "signature",
    }
    if set(value) - allowed:
        raise SystemExit("Release manifest contains unsupported fields.")
    for field in ("product", "edition", "version", "channel", "published_at"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise SystemExit(f"Release manifest {field} is malformed.")
    try:
        datetime.fromisoformat(value["published_at"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise SystemExit("Release manifest published_at is malformed.") from exc
    for field in ("platforms", "profiles"):
        entries = value.get(field)
        if not isinstance(entries, list) or not entries or not all(isinstance(item, str) and item for item in entries):
            raise SystemExit(f"Release manifest {field} is malformed.")
    python = value.get("python")
    contracts = value.get("contracts")
    if not isinstance(python, dict) or not str(python.get("requires", "") or "").strip():
        raise SystemExit("Release manifest python requirements are missing.")
    if not isinstance(contracts, dict) or not {"workspace", "scenario"}.issubset(contracts):
        raise SystemExit("Release manifest contracts are malformed.")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SystemExit("Release manifest artifacts are malformed.")
    artifact_allowed = {"name", "kind", "path", "url", "bytes", "sha256", "media_type", "platform", "architecture"}
    for item in artifacts:
        if not isinstance(item, dict) or set(item) - artifact_allowed:
            raise SystemExit("Release manifest contains a malformed artifact.")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or Path(name.replace("\\", "/")).name != name:
            raise SystemExit("Release manifest artifact names must be plain file names.")
        if not isinstance(item.get("kind"), str) or not item["kind"].strip():
            raise SystemExit("Release manifest artifact kind is malformed.")
        if not item.get("path") and not item.get("url"):
            raise SystemExit("Release manifest artifacts need a path or URL.")
        if any(item.get(field) is not None and not isinstance(item[field], str) for field in ("path", "url")):
            raise SystemExit("Release manifest artifact locations must be strings.")
        size = item.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0 or size > MAX_RELEASE_BYTES:
            raise SystemExit("Release manifest artifact bytes are malformed.")
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SystemExit("Release manifest artifact SHA-256 is malformed.")
    source_count = sum(item.get("kind") in {"source", "source_bundle"} for item in artifacts if isinstance(item, dict))
    if source_count != 1:
        raise SystemExit("Release manifest must declare exactly one source artifact.")
    if require_integrity and not _validate_bootstrap_integrity(value):
        raise SystemExit(
            "Official release manifest is missing a structurally valid signed installation-integrity payload."
        )
    return value


def _validate_bootstrap_integrity(value: Mapping[str, Any]) -> bool:
    """Check the signed inventory shape before downloading source bytes."""

    supply_chain = value.get("supply_chain")
    integrity = supply_chain.get("installation_integrity") if isinstance(supply_chain, dict) else None
    if isinstance(integrity, dict) and integrity.get("schema_version") == "oel.installation-integrity.xz.v1":
        try:
            integrity = globals()["unpack_installation_integrity"](integrity)
        except (KeyError, ValueError):
            return False
    if not isinstance(integrity, dict) or integrity.get("schema_version") != "oel.installation-integrity.v1":
        return False
    selected_profile = integrity.get("profile")
    if not isinstance(selected_profile, str) or not selected_profile:
        return False
    if not isinstance(integrity.get("digest"), str) or len(integrity["digest"]) != 64:
        return False
    source = integrity.get("source")
    if not isinstance(source, dict) or source.get("schema_version") != "oel.source-inventory.v1":
        return False
    exclusions = source.get("exclusions")
    if not isinstance(exclusions, dict) or exclusions != {
        "directories": [],
        "suffixes": [],
        "globs": ["*.log"],
    }:
        return False
    files = source.get("files")
    if not isinstance(files, list) or not files:
        return False
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, dict):
            return False
        path = str(row.get("path", "")).replace("\\", "/")
        if not path or path.startswith("/") or ".." in PurePosixPath(path).parts or path in seen:
            return False
        seen.add(path)
        if (
            isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] < 0
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            return False
    runtime = integrity.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("schema_version") != "oel.runtime-inventory.v1":
        return False
    if not isinstance(runtime.get("digest"), str) or not re.fullmatch(r"[0-9a-f]{64}", runtime["digest"]):
        return False
    if not isinstance(runtime.get("profile"), str) or not runtime["profile"]:
        return False
    if runtime["profile"] != selected_profile:
        return False
    layout = runtime.get("layout")
    if not isinstance(layout, dict) or not isinstance(layout.get("interpreter"), str):
        return False
    extras = layout.get("extras")
    if not isinstance(extras, list):
        return False
    for extra in extras:
        if not isinstance(extra, dict) or not isinstance(extra.get("path"), str) or not extra.get("path"):
            return False
        if extra.get("kind") not in {"venv-config", "provider-alias", "lib-alias"}:
            return False
    wheels = runtime.get("wheels")
    if not isinstance(wheels, list):
        return False
    wheel_names: set[str] = set()
    for wheel in wheels:
        if not isinstance(wheel, dict) or not isinstance(wheel.get("name"), str):
            return False
        if Path(wheel["name"]).name != wheel["name"] or wheel["name"] in wheel_names:
            return False
        wheel_names.add(wheel["name"])
        if not isinstance(wheel.get("bytes"), int) or not re.fullmatch(r"[0-9a-f]{64}", str(wheel.get("sha256", ""))):
            return False
        if not isinstance(wheel.get("files"), list) or not wheel.get("record"):
            return False
    runtime_profiles = integrity.get("runtime_profiles")
    if not isinstance(runtime_profiles, dict) or not runtime_profiles:
        return False
    advertised_profiles = value.get("profiles")
    if not isinstance(advertised_profiles, list) or any(
        not isinstance(profile, str) or not profile for profile in advertised_profiles
    ):
        return False
    if selected_profile not in set(advertised_profiles):
        return False
    if not set(advertised_profiles).issubset(set(runtime_profiles)):
        return False
    for profile, profile_runtime in runtime_profiles.items():
        if not isinstance(profile, str) or not isinstance(profile_runtime, dict):
            return False
        if profile_runtime.get("schema_version") != "oel.runtime-inventory.v1" or profile_runtime.get("profile") != profile:
            return False
        if not isinstance(profile_runtime.get("digest"), str) or not re.fullmatch(r"[0-9a-f]{64}", profile_runtime["digest"]):
            return False
    return True


def _prepare_external_anchor(path: Path, keys: Mapping[str, Any], *, published_at: str | None = None) -> None:
    """Create the initial anchor exactly once; never replace an existing one."""

    anchor = path.expanduser().absolute()
    for node in (anchor, *anchor.parents):
        if node.is_symlink() and not _is_os_path_alias(node):
            raise SystemExit(f"Official release trust anchor may not use a symbolic-link parent: {node}")
    if anchor.is_symlink():
        raise SystemExit("Official release trust anchor may not be a symbolic link.")
    if anchor.exists():
        if not anchor.is_file():
            raise SystemExit("Official release trust anchor must be a regular file.")
        try:
            current = json.loads(anchor.read_text(encoding="utf-8"))
            _validate_trusted_keys_payload(current)
        except (OSError, TypeError, ValueError, json.JSONDecodeError, SystemExit) as exc:
            raise SystemExit("Existing official release trust anchor is invalid.") from exc
        # Keep the existing operator/channel anchor.  A bundle may carry a
        # future registry during a planned rotation; the manager will accept
        # it only when the incoming manifest is authenticated by this current
        # anchor.  New-key adoption still requires explicit signed rotation.
        return
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor_payload = dict(keys)
    if published_at and not anchor_payload.get("published_at"):
        anchor_payload["published_at"] = published_at
    rendered = json.dumps(anchor_payload, indent=2, sort_keys=True) + "\n"
    try:
        with anchor.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    except FileExistsError:
        _prepare_external_anchor(anchor, keys, published_at=published_at)


def _verify(payload: dict[str, Any], keys_payload: dict[str, Any]) -> bool:
    signature = payload.get("signature")
    if not isinstance(signature, dict) or signature.get("alg") != "RS256":
        return False
    key_id = str(signature.get("key_id", ""))
    items = keys_payload.get("keys", [])
    item = next((entry for entry in items if isinstance(entry, dict) and entry.get("key_id") == key_id), None)
    if item is None:
        return False
    if bool(item.get("revoked", False)):
        return False
    now = datetime.now(timezone.utc)
    for field, is_lower_bound in (("not_before", True), ("expires_at", False)):
        raw_time = item.get(field)
        if raw_time:
            try:
                boundary = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                if boundary.tzinfo is None:
                    boundary = boundary.replace(tzinfo=timezone.utc)
            except ValueError:
                return False
            if (is_lower_bound and now < boundary) or (not is_lower_bound and now >= boundary):
                return False
    try:
        modulus = _decode_int(item["n"])
        exponent = _decode_int(item.get("e", "AQAB"))
    except (KeyError, TypeError, ValueError, UnicodeError, binascii.Error):
        return False
    if modulus.bit_length() < 2048 or exponent != 65537:
        return False
    signature_text = str(signature.get("value", ""))
    try:
        raw = signature_text.encode("ascii")
        signature_bytes = base64.urlsafe_b64decode(raw + b"=" * ((4 - len(raw) % 4) % 4))
    except (TypeError, ValueError, UnicodeError, binascii.Error):
        return False
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    message = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    width = (modulus.bit_length() + 7) // 8
    if len(signature_bytes) != width:
        return False
    encoded = pow(int.from_bytes(signature_bytes, "big"), exponent, modulus).to_bytes(width, "big")
    expected = _SHA256_DIGESTINFO_PREFIX + hashlib.sha256(message).digest()
    if not encoded.startswith(b"\x00\x01"):
        return False
    try:
        separator = encoded.index(b"\x00", 2)
    except ValueError:
        return False
    padding = encoded[2:separator]
    return len(padding) >= 8 and all(byte == 0xFF for byte in padding) and encoded[separator + 1 :] == expected


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination = _validated_destination(destination)
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    names: set[str] = set()
    with tarfile.open(archive, "r:*") as source:
        members = source.getmembers()
        for member in members:
            normalized = member.name.replace("\\", "/")
            relative = PurePosixPath(normalized)
            if (
                not normalized
                or not relative.parts
                or normalized in {".", "./"}
                or relative.is_absolute()
                or ".." in relative.parts
                or ":" in relative.parts[0]
            ):
                raise RuntimeError(f"Unsafe archive path: {member.name!r}")
            canonical_name = relative.as_posix()
            if canonical_name in names:
                raise RuntimeError(f"Duplicate archive member: {canonical_name}")
            names.add(canonical_name)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise RuntimeError(f"Unsupported archive member: {normalized}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsupported archive member type: {normalized}")
            total += max(0, member.size)
            if total > MAX_RELEASE_BYTES:
                raise RuntimeError("Release archive expands beyond the safety limit.")
        for member in members:
            relative = PurePosixPath(member.name.replace("\\", "/"))
            if member.isdir():
                _prepare_member_output(destination, relative, directory=True)
                continue
            output = _prepare_member_output(destination, relative, directory=False)
            stream = source.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Could not read release member: {member.name}")
            with stream, output.open("wb") as sink:
                shutil.copyfileobj(stream, sink)
            output.chmod(0o755 if member.mode & 0o111 else 0o644)
    roots = sorted(path.parent for path in destination.glob("*/pyproject.toml"))
    if len(roots) != 1:
        raise RuntimeError("Release archive must contain exactly one OEL source root.")
    return roots[0]


def _prepare_member_output(destination: Path, relative: PurePosixPath, *, directory: bool) -> Path:
    """Create one bootstrap archive member without following local links."""

    current = destination
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"Archive extraction path contains a symbolic link: {current}")
        final = index == len(relative.parts) - 1
        if current.exists():
            if final:
                if directory and not current.is_dir():
                    raise RuntimeError(f"Archive directory collides with a non-directory: {current}")
                if not directory:
                    raise RuntimeError(f"Archive file would overwrite an existing path: {current}")
            elif not current.is_dir():
                raise RuntimeError(f"Archive extraction parent is not a directory: {current}")
        elif final and directory:
            current.mkdir()
        elif not final:
            current.mkdir()
    return current


def _validate_host_compatibility(manifest: dict[str, Any]) -> None:
    """Reject a signed bundle for another host before interpreter selection."""
    system = platform.system()
    machine = platform.machine()
    supported = {item.lower() for item in manifest["platforms"]}
    aliases = {"amd64": "x86_64", "aarch64": "arm64"}
    expected = str(manifest.get("architecture", "") or "").lower()
    actual = aliases.get(machine.lower(), machine.lower())
    expected = aliases.get(expected, expected)
    if system.lower() not in supported or (expected and expected != "any" and expected != actual):
        raise SystemExit(
            f"This signed OEL {manifest['version']} bundle targets "
            f"{', '.join(manifest['platforms'])}/{expected or 'any architecture'}, "
            f"not this host's {system}/{machine}. "
            "Use the source-build instructions at "
            "https://github.com/adamcohen8/orbital-engineering-lab/blob/main/docs/native-source-build.md. "
            "No release artifacts were downloaded or packages installed."
        )


def _use_qualified_python(manifest: dict[str, Any]) -> None:
    """Use the signed bundle's qualified interpreter before downloading code."""
    if not any(row.get("kind") == "wheel" for row in manifest.get("artifacts", [])):
        return
    qualification = dict(manifest.get("supply_chain", {})).get("offline_runtime_qualification", {})
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", str(qualification.get("python", "")))
    if match is None:
        raise SystemExit("Release wheelhouse is missing its qualified Python minor.")
    required = (int(match[1]), int(match[2]))
    if sys.version_info[:2] == required:
        return
    candidates: list[list[str]] = []
    named = shutil.which(f"python{required[0]}.{required[1]}")
    if named:
        candidates.append([named])
    launcher = shutil.which("py") if sys.platform == "win32" else None
    if launcher:
        candidates.append([launcher, f"-{required[0]}.{required[1]}"])
    for candidate in candidates:
        probe = subprocess.run(
            [*candidate, "-c", f"import sys; raise SystemExit(0 if sys.version_info[:2] == {required!r} else 1)"],
            capture_output=True, check=False, timeout=15,
        )
        if probe.returncode == 0:
            os.execv(candidate[0], [*candidate, str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(
        f"This signed release bundle requires CPython {required[0]}.{required[1]}. "
        "Install that Python minor and rerun the installer; no packages were installed."
    )


def _user_launcher_location() -> tuple[Path, Path]:
    if sys.platform == "win32":
        user_bin = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Programs" / "OEL" / "bin"
        target = user_bin / "oel.cmd"
    else:
        user_bin = Path.home() / ".local" / "bin"
        target = user_bin / "oel"
    return user_bin, target


def _preflight_user_launcher() -> None:
    user_bin, target = _user_launcher_location()
    if target.exists() or target.is_symlink():
        managed = target.is_file() and any(
            marker in target.read_text(encoding="utf-8", errors="ignore")
            for marker in ("oel-trusted-dispatcher:v1", "sim.installation.cli")
        )
        if not managed:
            raise RuntimeError(f"Refusing to replace a non-OEL launcher: {target}")
    existing_parent = next((path for path in (user_bin, *user_bin.parents) if path.exists()), None)
    if existing_parent is not None and not os.access(existing_parent, os.W_OK):
        raise RuntimeError(f"User launcher directory is not writable: {existing_parent}")


def _publish_user_launcher(activation: dict[str, Any]) -> dict[str, Any]:
    user_bin, target = _user_launcher_location()
    source = Path(
        activation["launchers"]["windows" if sys.platform == "win32" else "posix"]
    )
    user_bin.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        managed = (
            target.is_symlink() and target.resolve() == source.resolve()
        ) or (
            target.is_file()
            and any(
                marker in target.read_text(encoding="utf-8", errors="ignore")
                for marker in ("oel-trusted-dispatcher:v1", "sim.installation.cli")
            )
        )
        if not managed:
            raise RuntimeError(f"Refusing to replace a non-OEL launcher: {target}")
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)
        if sys.platform != "win32":
            target.chmod(0o755)
    path_updated = False
    if sys.platform == "win32":
        import winreg

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            try:
                current, _ = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current = ""
            entries = [item for item in str(current).split(";") if item]
            if str(user_bin).lower() not in {item.lower() for item in entries}:
                entries.append(str(user_bin))
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, ";".join(entries))
                path_updated = True
    return {"path": str(target), "directory": str(user_bin), "path_updated": path_updated}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install one signed OEL release.")
    parser.add_argument("--manifest-url", required=True)
    parser.add_argument("--channel-url")
    parser.add_argument("--profile", default="game")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--developer-unsigned", action="store_true")
    args = parser.parse_args(argv)
    if not ((3, 10) <= sys.version_info[:2] < (3, 15)):
        raise SystemExit("OEL requires CPython >=3.10,<3.15.")
    manifest_scheme = urllib.parse.urlparse(args.manifest_url).scheme.lower()
    if manifest_scheme != "https" and not (args.developer_unsigned and manifest_scheme == "file"):
        raise SystemExit("Official OEL bootstrap URLs must use HTTPS.")
    if args.channel_url:
        channel_scheme = urllib.parse.urlparse(args.channel_url).scheme.lower()
        if channel_scheme != "https" and not (args.developer_unsigned and channel_scheme == "file"):
            raise SystemExit("Official OEL update channel URLs must use HTTPS.")
    manifest_data = _fetch(args.manifest_url, maximum=MAX_METADATA_BYTES)
    try:
        manifest = _validate_bootstrap_manifest(
            json.loads(manifest_data), require_integrity=not args.developer_unsigned
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit("Release manifest is not valid JSON.") from exc
    rendered = TRUSTED_KEYS_RENDERED == "true"
    keys: dict[str, Any] = {}
    if rendered:
        try:
            keys = _validate_trusted_keys_payload(json.loads(base64.b64decode(TRUSTED_KEYS_B64).decode("utf-8")))
        except (TypeError, ValueError, json.JSONDecodeError, binascii.Error) as exc:
            raise SystemExit("Embedded trusted release-key registry is not valid JSON.") from exc
    if not args.developer_unsigned and (not rendered or not _verify(manifest, keys)):
        raise SystemExit("Release manifest signature verification failed.")
    _validate_host_compatibility(manifest)
    _use_qualified_python(manifest)
    artifacts = [item for item in manifest.get("artifacts", []) if item.get("kind") in {"source", "source_bundle"}]
    if len(artifacts) != 1:
        raise SystemExit("Release manifest must declare exactly one source artifact.")
    artifact = artifacts[0]
    artifact_location = str(artifact.get("url") or artifact.get("path") or "").strip()
    if not artifact_location:
        raise SystemExit("Release source artifact is missing both url and path.")
    artifact_url = urllib.parse.urljoin(args.manifest_url, artifact_location)
    artifact_scheme = urllib.parse.urlparse(artifact_url).scheme.lower()
    if artifact_scheme != "https" and not (args.developer_unsigned and artifact_scheme == "file"):
        raise SystemExit("Official OEL release artifact URLs must use HTTPS.")
    with tempfile.TemporaryDirectory(prefix="oel-bootstrap-") as temporary:
        root = Path(temporary)
        archive = root / artifact["name"]
        _fetch_to_file(artifact_url, archive, maximum=artifact["bytes"])
        if archive.stat().st_size != artifact["bytes"] or _sha256_file(archive) != artifact["sha256"]:
            raise SystemExit("Release artifact size or SHA-256 verification failed.")
        inspection = root / "inspection"
        source_root = _safe_extract(archive, inspection)
        sys.path.insert(0, str(source_root))
        from sim.installation.manager import activate, configure_channel, download_release_wheelhouse, install_release
        from sim.installation.paths import InstallationPaths
        from sim.installation.signing import load_release_public_keys

        manifest_path = root / "release-manifest.json"
        # Preserve the signed manifest. The managed installer resolves this
        # verified download by the signed artifact name in the same directory.
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        keys_path = root / "trusted-release-keys.json"
        keys_path.write_text(json.dumps(keys, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths = InstallationPaths.default()
        if args.data_root or args.config_root:
            paths = InstallationPaths(
                (args.data_root or paths.data_root).expanduser().resolve(),
                (args.config_root or paths.config_root).expanduser().resolve(),
            )
        if not args.developer_unsigned:
            config_resolved = paths.config_root.resolve(strict=False)
            versions_resolved = paths.versions.resolve(strict=False)
            if config_resolved == versions_resolved or config_resolved.is_relative_to(versions_resolved):
                raise SystemExit("Official release trust anchor must live outside managed version roots.")
            _prepare_external_anchor(
                paths.trusted_release_keys,
                keys,
                published_at=str(manifest.get("published_at", "")),
            )
        # Catch deterministic launcher conflicts before install/activation
        # changes the managed engine or current selector.
        _preflight_user_launcher()
        wheelhouse = download_release_wheelhouse(
            manifest, args.manifest_url, root, allow_local_file=bool(args.developer_unsigned)
        )
        result = install_release(
            manifest_path,
            paths=paths,
            public_keys=None if args.developer_unsigned else load_release_public_keys(keys_path),
            require_signature=not args.developer_unsigned,
            profile=args.profile,
            create_runtime=True,
            offline_wheelhouse=wheelhouse,
        )
        activation = None
        channel_configuration = None
        user_launcher = None
        try:
            activation = activate(str(manifest["version"]), paths=paths)
            if args.channel_url:
                channel_configuration = configure_channel(
                    args.channel_url,
                    edition=str(manifest["edition"]),
                    channel=str(manifest["channel"]),
                    paths=paths,
                    source="official-bootstrap" if not args.developer_unsigned else "developer-bootstrap",
                    allow_local_file=bool(args.developer_unsigned),
                )
            user_launcher = _publish_user_launcher(activation)
            from sim.installation.desktop import publish_trainer

            user_launcher["trainer"] = publish_trainer(paths)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "status": "partial",
                        "error": f"{type(exc).__name__}: {exc}",
                        "installation": result,
                        "activation": activation,
                        "channel_configuration": channel_configuration,
                        "user_launcher": user_launcher,
                        "recovery": (
                            "Resolve the reported launcher/channel problem, then rerun this bootstrap; "
                            "the signed installation step is idempotent. If activation changed current, "
                            "use the managed rollback command to restore the previous selector."
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            raise
        print(
            json.dumps(
                {
                    "status": "ready",
                    "installation": result,
                    "activation": activation,
                    "channel_configuration": channel_configuration,
                    "user_launcher": user_launcher,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
