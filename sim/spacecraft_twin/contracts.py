"""Private, data-only spacecraft authoring contracts and bounded file operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import yaml

from sim.security.output_writes import (
    SecureWriteConflict,
    current_output_writer,
    read_output_bytes,
    sha256_output,
    write_output_text,
)

SCHEMA = "oel.spacecraft_twin.v1"
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_FILES = 500
MAX_TOTAL_BYTES = 256 * 1024 * 1024
# The file and byte caps do not bound work for a tree containing only empty
# directories. Keep the intake walk itself bounded as well.
MAX_TRAVERSAL_ENTRIES = 10_000
MAX_TRAVERSAL_DEPTH = 64


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _absolute_lexical(path):
    """Normalize a path without resolving symlinks or following output roots."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _bound_relative(path):
    writer = current_output_writer()
    if writer is None:
        return None
    candidate = _absolute_lexical(path)
    logical_root = _absolute_lexical(writer.logical_root)
    try:
        return candidate.relative_to(logical_root)
    except ValueError:
        return None


def physical_output_path(path):
    """Map one logical output path to the active writer's physical path."""

    writer = current_output_writer()
    if writer is None:
        return Path(path)
    relative = _bound_relative(path)
    if relative is None:
        return Path(path)
    return Path(writer.path).joinpath(relative)


def file_digest(path):
    if _bound_relative(path) is not None:
        return sha256_output(path)
    path = physical_output_path(path)
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("Identifiers must start with a letter and contain only letters, digits, _, ., or - (80 max).")
    return value


def contained(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Package paths must be relative and cannot traverse parents.")
    result = root / p
    if result.is_symlink() or not result.resolve().is_relative_to(root):
        raise ValueError("Package references cannot escape the package root.")
    return result


def load(path):
    path = Path(path)
    bound = _bound_relative(path)
    if bound is not None:
        text = read_output_bytes(path, max_bytes=MAX_FILE_BYTES).decode("utf-8")
    else:
        path = physical_output_path(path)
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Input exceeds 32 MiB limit.")
        text = path.read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError("Input exceeds 32 MiB limit.")
    value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a mapping.")
    # Reject non-JSON types, NaN and cyclic YAML aliases before downstream processing.
    json.dumps(value, allow_nan=False)
    return value


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    writer = current_output_writer()
    if writer is not None:
        try:
            return write_output_text(path, payload)
        except SecureWriteConflict:
            relative = _bound_relative(path)
            if relative is None:
                raise
            return writer.update_existing_bytes(relative, payload.encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".twin-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def vector(value, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Expected a finite {size}-vector.")
    return result


def positive(value):
    if isinstance(value, bool):
        raise ValueError("Expected a positive number, not boolean.")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Expected a finite positive number.")
    return result


def rotation(value):
    r = np.asarray(value, dtype=float)
    if (
        r.shape != (3, 3)
        or not np.all(np.isfinite(r))
        or not np.allclose(r.T @ r, np.eye(3), atol=1e-9)
        or not np.isclose(np.linalg.det(r), 1, atol=1e-9)
    ):
        raise ValueError("Rotation must be a proper orthonormal 3x3 matrix (local to parent).")
    return r


def frames(model):
    """Return local-to-body transforms. Positions in each parent frame use declared units."""
    raw = model.get("frames", {})
    if len(raw) > 128 or "body" in raw:
        raise ValueError("At most 128 frames; body is the implicit identity frame.")
    resolved = {"body": (np.eye(3), np.zeros(3))}
    visiting = set()

    def resolve(name):
        if name in resolved:
            return resolved[name]
        if name in visiting or name not in raw:
            raise ValueError(f"Frame cycle or unknown frame: {name}")
        visiting.add(name)
        row = raw[name]
        parent_r, parent_t = resolve(row["parent"])
        scale = {"m": 1.0, "mm": 0.001, "cm": 0.01}.get(row["units"])
        if scale is None:
            raise ValueError("Frame position units must be m, cm, or mm.")
        r = parent_r @ rotation(row["rotation_to_parent"])
        t = parent_t + parent_r @ (vector(row["translation"]) * scale)
        resolved[name] = (r, t)
        visiting.remove(name)
        return r, t

    for name in raw:
        identifier(name)
        resolve(name)
    return resolved
