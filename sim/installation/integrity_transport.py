"""Bounded, lossless transport for signed installation inventories.

The installer renderer embeds this stdlib-only module in the standalone
bootstrap, so decoding never imports an unverified installation.
"""

import base64 as _transport_base64
import hashlib as _transport_hashlib
import json as _transport_json
import lzma as _transport_lzma

INTEGRITY_TRANSPORT_SCHEMA = "oel.installation-integrity.xz.v1"
MAX_INTEGRITY_BYTES = 32 * 1024 * 1024
MAX_INTEGRITY_ENCODED_BYTES = 4 * 1024 * 1024


def pack_installation_integrity(value):
    raw = _transport_json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if not raw or len(raw) > MAX_INTEGRITY_BYTES:
        raise ValueError("Installation inventory exceeds the decoded transport limit.")
    data = _transport_base64.b64encode(_transport_lzma.compress(raw, preset=6)).decode("ascii")
    if len(data) > MAX_INTEGRITY_ENCODED_BYTES:
        raise ValueError("Installation inventory exceeds the encoded transport limit.")
    return {
        "schema_version": INTEGRITY_TRANSPORT_SCHEMA,
        "bytes": len(raw),
        "sha256": _transport_hashlib.sha256(raw).hexdigest(),
        "data": data,
    }


def unpack_installation_integrity(value):
    if not isinstance(value, dict) or value.get("schema_version") != INTEGRITY_TRANSPORT_SCHEMA:
        return value
    if set(value) != {"schema_version", "bytes", "sha256", "data"}:
        raise ValueError("Compressed installation inventory has unsupported fields.")
    size, digest, data = value["bytes"], value["sha256"], value["data"]
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_INTEGRITY_BYTES:
        raise ValueError("Compressed installation inventory has an invalid decoded size.")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Compressed installation inventory has an invalid digest.")
    if not isinstance(data, str) or len(data) > MAX_INTEGRITY_ENCODED_BYTES:
        raise ValueError("Compressed installation inventory exceeds the encoded transport limit.")
    try:
        compressed = _transport_base64.b64decode(data, validate=True)
        decoder = _transport_lzma.LZMADecompressor(format=_transport_lzma.FORMAT_XZ, memlimit=64 * 1024 * 1024)
        raw = decoder.decompress(compressed, max_length=size + 1)
        if len(raw) != size or not decoder.eof or decoder.unused_data:
            raise ValueError("Compressed installation inventory size or stream boundary is invalid.")
        if _transport_hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Compressed installation inventory digest is invalid.")
        decoded = _transport_json.loads(raw)
    except (ValueError, EOFError, _transport_lzma.LZMAError, RecursionError) as exc:
        raise ValueError("Compressed installation inventory is invalid.") from exc
    if not isinstance(decoded, dict):
        raise ValueError("Compressed installation inventory must contain an object.")
    return decoded
