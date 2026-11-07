"""Dependency-free RS256 signing for OEL release metadata.

Release signing and customer-license signing use separate keys and policies even
though both currently use the same reviewed RS256 primitive.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import canonical_json_bytes

_SHA256_DIGESTINFO_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")


@dataclass(frozen=True)
class RSAPublicKey:
    key_id: str
    n: int
    e: int = 65537
    alg: str = "RS256"
    not_before: datetime | None = None
    expires_at: datetime | None = None
    revoked: bool = False


@dataclass(frozen=True)
class RSAPrivateKey:
    key_id: str
    n: int
    d: int
    alg: str = "RS256"


def generate_rsa_private_key(key_id: str, *, bits: int = 3072) -> RSAPrivateKey:
    if bits < 512 or bits % 2:
        raise ValueError("RSA signing-key size must be an even number of at least 512 bits.")
    half_bits = bits // 2
    p = _generate_prime(half_bits)
    q = _generate_prime(half_bits)
    while q == p:
        q = _generate_prime(half_bits)
    exponent = 65537
    phi = (p - 1) * (q - 1)
    modulus = p * q
    if math.gcd(exponent, phi) != 1 or modulus.bit_length() < bits:
        return generate_rsa_private_key(key_id, bits=bits)
    return RSAPrivateKey(key_id=key_id, n=modulus, d=pow(exponent, -1, phi))


def sign_payload(payload: Mapping[str, Any], private_key: RSAPrivateKey) -> dict[str, Any]:
    signed = dict(payload)
    signed.pop("signature", None)
    signature = _rsa_sign_sha256(canonical_json_bytes(signed), private_key)
    signed["signature"] = {
        "alg": private_key.alg,
        "key_id": private_key.key_id,
        "value": _b64url_encode(signature),
    }
    return signed


def verify_payload(
    payload: Mapping[str, Any],
    public_keys: Mapping[str, RSAPublicKey],
    *,
    now: datetime | None = None,
) -> bool:
    signature = payload.get("signature")
    if not isinstance(signature, Mapping):
        return False
    alg = str(signature.get("alg", ""))
    key_id = str(signature.get("key_id", ""))
    value = str(signature.get("value", ""))
    key = public_keys.get(key_id)
    if alg != "RS256" or key is None or key.alg != alg or not value:
        return False
    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if key.revoked or (key.not_before and checked_at < key.not_before) or (key.expires_at and checked_at >= key.expires_at):
        return False
    try:
        return _rsa_verify_sha256(
            canonical_json_bytes(payload, omit_signature=True),
            _b64url_decode(value),
            key,
        )
    except (binascii.Error, UnicodeError, OverflowError, ValueError):
        return False


def load_public_keys(path: str | Path) -> dict[str, RSAPublicKey]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    items = data.get("keys", data) if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())
    keys: dict[str, RSAPublicKey] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        key = RSAPublicKey(
            key_id=str(item["key_id"]),
            n=_decode_int(item["n"]),
            e=_decode_int(item.get("e", "AQAB")),
            alg=str(item.get("alg", "RS256")),
            not_before=_parse_datetime(item.get("not_before")),
            expires_at=_parse_datetime(item.get("expires_at")),
            revoked=bool(item.get("revoked", False)),
        )
        keys[key.key_id] = key
    return keys


def load_release_public_keys(path: str | Path) -> dict[str, RSAPublicKey]:
    """Load an official release registry with its stronger key policy.

    Community catalogs and test fixtures intentionally use ``load_public_keys``
    with their own trust roots.  Official release registries are stricter:
    they must be an explicit key list, contain unique IDs, and every active key
    must use RS256 with a 2048-bit-or-larger modulus and e=65537. Historical
    revoked entries remain parseable so key-rotation behavior is unchanged.
    """

    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("keys"), list):
        raise ValueError("Official release-key registry must contain a keys list.")
    keys: dict[str, RSAPublicKey] = {}
    for index, item in enumerate(data["keys"]):
        if not isinstance(item, dict):
            raise ValueError(f"Official release-key registry key {index} is malformed.")
        key_id = item.get("key_id")
        if not isinstance(key_id, str) or not key_id.strip():
            raise ValueError(f"Official release-key registry key {index} has an empty key ID.")
        if key_id in keys:
            raise ValueError(f"Official release-key registry contains duplicate key ID: {key_id}.")
        if item.get("alg", "RS256") != "RS256":
            raise ValueError(f"Official release key {key_id} must use RS256.")
        try:
            modulus = _decode_int(item["n"])
            exponent = _decode_int(item.get("e", "AQAB"))
        except (KeyError, TypeError, ValueError, UnicodeError, binascii.Error) as exc:
            raise ValueError(f"Official release key {key_id} has invalid RSA parameters.") from exc
        revoked = item.get("revoked", False)
        if not isinstance(revoked, bool):
            raise ValueError(f"Official release key {key_id} has a malformed revoked flag.")
        if modulus <= 0 or (not revoked and (modulus.bit_length() < 2048 or exponent != 65537)):
            raise ValueError(f"Official release key {key_id} must be RSA-2048 or stronger with e=65537.")
        keys[key_id] = RSAPublicKey(
            key_id=key_id,
            n=modulus,
            e=exponent,
            alg="RS256",
            not_before=_parse_datetime(item.get("not_before")),
            expires_at=_parse_datetime(item.get("expires_at")),
            revoked=revoked,
        )
    if not keys:
        raise ValueError("Official release-key registry must contain at least one key.")
    return keys


def load_private_key(path: str | Path) -> RSAPrivateKey:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Private key file must contain a JSON object.")
    return RSAPrivateKey(
        key_id=str(data["key_id"]),
        n=_decode_int(data["n"]),
        d=_decode_int(data["d"]),
        alg=str(data.get("alg", "RS256")),
    )


def public_key_to_json(key: RSAPublicKey) -> dict[str, Any]:
    return {
        "key_id": key.key_id,
        "alg": key.alg,
        "n": _int_to_b64url(key.n),
        "e": _int_to_b64url(key.e),
        **({"not_before": _format_datetime(key.not_before)} if key.not_before else {}),
        **({"expires_at": _format_datetime(key.expires_at)} if key.expires_at else {}),
        **({"revoked": True} if key.revoked else {}),
    }


def private_key_to_json(key: RSAPrivateKey) -> dict[str, str]:
    return {"key_id": key.key_id, "alg": key.alg, "n": _int_to_b64url(key.n), "d": _int_to_b64url(key.d)}


def write_key_files(
    private_key: RSAPrivateKey,
    *,
    private_output: str | Path,
    public_output: str | Path,
    force: bool = False,
) -> None:
    private_path = Path(private_output).expanduser()
    public_path = Path(public_output).expanduser()
    for path in (private_path, public_path):
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to replace existing key file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    public_key = RSAPublicKey(key_id=private_key.key_id, n=private_key.n, alg=private_key.alg)
    if private_path.resolve(strict=False) == public_path.resolve(strict=False):
        raise ValueError("Private and public key outputs must be different paths.")
    _atomic_key_write(
        private_path,
        json.dumps(private_key_to_json(private_key), indent=2, sort_keys=True) + "\n",
        mode=0o600,
    )
    _atomic_key_write(
        public_path,
        json.dumps({"keys": [public_key_to_json(public_key)]}, indent=2, sort_keys=True) + "\n",
        mode=0o644,
    )


def _atomic_key_write(path: Path, payload: str, *, mode: int) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.chmod(temporary, mode)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rsa_verify_sha256(message: bytes, signature: bytes, key: RSAPublicKey) -> bool:
    width = (key.n.bit_length() + 7) // 8
    if len(signature) != width:
        return False
    encoded = pow(int.from_bytes(signature, "big"), key.e, key.n).to_bytes(width, "big")
    expected = _SHA256_DIGESTINFO_PREFIX + hashlib.sha256(message).digest()
    if not encoded.startswith(b"\x00\x01"):
        return False
    try:
        separator = encoded.index(b"\x00", 2)
    except ValueError:
        return False
    padding = encoded[2:separator]
    return len(padding) >= 8 and all(byte == 0xFF for byte in padding) and encoded[separator + 1 :] == expected


def _rsa_sign_sha256(message: bytes, key: RSAPrivateKey) -> bytes:
    width = (key.n.bit_length() + 7) // 8
    digest_info = _SHA256_DIGESTINFO_PREFIX + hashlib.sha256(message).digest()
    padding_len = width - len(digest_info) - 3
    if padding_len < 8:
        raise ValueError("RSA key is too small for an RS256 signature.")
    encoded = b"\x00\x01" + (b"\xff" * padding_len) + b"\x00" + digest_info
    return pow(int.from_bytes(encoded, "big"), key.d, key.n).to_bytes(width, "big")


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    raw = value.encode("ascii")
    return base64.urlsafe_b64decode(raw + b"=" * ((4 - len(raw) % 4) % 4))


def _int_to_b64url(value: int) -> str:
    width = max(1, (int(value).bit_length() + 7) // 8)
    return _b64url_encode(int(value).to_bytes(width, "big"))


def _decode_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    return int.from_bytes(_b64url_decode(text), "big")


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _generate_prime(bits: int) -> int:
    while True:
        candidate = secrets.randbits(bits)
        candidate |= (1 << (bits - 1)) | 1
        if candidate % 65537 == 1:
            continue
        if _is_probable_prime(candidate):
            return candidate


def _is_probable_prime(value: int, *, rounds: int = 40) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value == prime:
            return True
        if value % prime == 0:
            return False
    odd = value - 1
    shifts = 0
    while odd % 2 == 0:
        shifts += 1
        odd //= 2
    for _ in range(rounds):
        witness = secrets.randbelow(value - 3) + 2
        sample = pow(witness, odd, value)
        if sample in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            sample = pow(sample, 2, value)
            if sample == value - 1:
                break
        else:
            return False
    return True
