"""Bounded input, strict JSON/XML, and exclusive artifact/receipt publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .contracts import MAX_BYTES, DataIOError


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: str | Path, *, limit: int = MAX_BYTES) -> bytes:
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise DataIOError(f"Input exceeds {limit} bytes.")
    return raw


def read_text(path: str | Path) -> str:
    try:
        return read_bytes(path).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataIOError("Input must be UTF-8.") from exc


def strict_json(raw: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DataIOError(f"Duplicate JSON key: {key}.")
            result[key] = value
        return result

    def constant(value):
        raise DataIOError(f"Nonfinite JSON number: {value}.")

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise DataIOError(f"Nonfinite JSON number: {value}.")
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise DataIOError("Invalid or excessively nested JSON.") from exc


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def xml_root(raw: str) -> ET.Element:
    if len(raw.encode()) > MAX_BYTES or "<!DOCTYPE" in raw.upper() or "<!ENTITY" in raw.upper():
        raise DataIOError("XML exceeds limits or contains a forbidden DTD/entity declaration.")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise DataIOError("Invalid XML.") from exc
    count = 0
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if depth > 20 or count > 2_000_000:
            raise DataIOError("XML structural limit exceeded.")
        node.tag = node.tag.split("}")[-1]
        if any(
            key.split("}")[-1]
            not in {
                "id",
                "version",
                "units",
                "unit",
                "count",
                "parameter",
                "noNamespaceSchemaLocation",
                "schemaLocation",
            }
            for key in node.attrib
        ):
            raise DataIOError(f"Unsupported XML attributes on {node.tag}.")
        stack.extend((child, depth + 1) for child in node)
    return root


def source_receipt(path: str | Path) -> dict[str, Any]:
    hasher, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            if size > 2 * 1024**3:
                raise DataIOError("Source identity hashing exceeds the 2 GiB per-file limit.")
            hasher.update(chunk)
    return {"path": str(Path(path).resolve()), "sha256": hasher.hexdigest(), "size_bytes": size}


def publish(path: str | Path, data: bytes, *, receipt: dict[str, Any], sources: tuple[str | Path, ...] = ()) -> dict:
    """Publish a new data file and receipt; refuse replacement, including inputs."""
    target = Path(path).expanduser().absolute()
    receipt_path = target.with_name(target.name + ".receipt.json")
    source_paths = {Path(p).resolve() for p in sources}
    for item in (target, receipt_path):
        if item.is_symlink() or item.resolve() in source_paths or item.exists():
            raise DataIOError(f"Refusing to replace existing or source artifact: {item}.")
    payload = {
        "schema": "oel.data-io-receipt.v1",
        "status": "written",
        "simulation_executed": False,
        **receipt,
        "output": {"path": str(target), "sha256": digest(data), "size_bytes": len(data)},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for item, content in ((target, data), (receipt_path, json_bytes(payload))):
            name = None
            try:
                with tempfile.NamedTemporaryFile(dir=item.parent, prefix=".oel-io-", delete=False) as stream:
                    name = Path(stream.name)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Atomically expose complete bytes, exclusively. Never replace
                # a concurrent artifact after the preflight existence check.
                os.link(name, item)
                written.append((item, item.stat().st_ino))
            finally:
                if name is not None:
                    name.unlink(missing_ok=True)
    except BaseException:
        for item, inode in reversed(written):
            if item.exists() and item.stat().st_ino == inode:
                item.unlink()
        raise
    return {**payload, "receipt_path": str(receipt_path)}


def temporary_path(suffix: str):
    """Owned temporary filename for binary libraries requiring a file handle."""
    return tempfile.TemporaryDirectory(prefix="oel-data-io-")
