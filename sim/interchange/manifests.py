from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from sim.security.output_writes import current_output_writer

from .provenance import compute_manifest_id


def _active_writer(writer: Any | None) -> Any | None:
    """Use an explicit writer or the operation's descriptor-bound writer."""

    return writer if writer is not None else current_output_writer()


def bind_output_path(
    physical_path: str | Path,
    logical_path: str | Path | None = None,
    *,
    writer: Any | None = None,
) -> tuple[Path, Path]:
    """Return the physical destination and its pre-publication logical path.

    MCP producers may run against a private staging directory.  The producer
    must still describe the eventual approved destination while it builds and
    hashes its document; rewriting a serialized document after publication
    would invalidate that binding.  A descriptor writer can expose the same
    relationship through ``path`` and ``logical_root``.
    """

    writer = _active_writer(writer)
    physical = Path(physical_path).expanduser()
    if writer is None:
        physical = physical.resolve()
    elif not physical.is_absolute():
        physical = Path(writer.path).expanduser() / physical
    if logical_path is not None:
        logical = Path(logical_path).expanduser().resolve()
        if writer is not None:
            logical_root = Path(getattr(writer, "logical_root", writer.path)).expanduser().resolve()
            try:
                logical.relative_to(logical_root)
            except ValueError as exc:
                raise ValueError("Logical output is outside the bound writer root.") from exc
        return physical, logical
    if writer is not None:
        writer_path = Path(writer.path).expanduser()
        try:
            relative = physical.relative_to(writer_path)
        except ValueError as exc:
            raise ValueError("Physical output is outside the bound writer root.") from exc
        logical_root = Path(getattr(writer, "logical_root", writer_path)).expanduser().resolve()
        return physical, logical_root / relative
    return physical, physical.resolve()


def write_json_document(
    document: Mapping[str, Any],
    path: str | Path,
    *,
    writer: Any | None = None,
    overwrite: bool = True,
) -> Path:
    """Serialize one document and publish it through an optional writer.

    ``writer`` is the descriptor-bound production handle used by MCP.  The
    path-only branch remains for the CLI and existing Python callers.  An
    existing entry can be refreshed only when the caller explicitly permits
    it; descriptor writers additionally require that the entry was created by
    the same handle.
    """

    target = Path(path).expanduser()
    writer = _active_writer(writer)
    payload = (json.dumps(dict(document), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if writer is not None:
        writer_root = Path(writer.path).expanduser()
        try:
            relative = target.relative_to(writer_root)
        except ValueError as exc:
            raise ValueError("Output document is outside the bound writer root.") from exc
        try:
            writer.write_bytes(relative, payload)
        except FileExistsError:
            if not overwrite:
                reader = getattr(writer, "read_bytes", None)
                try:
                    if reader is not None and reader(relative) == payload:
                        return target
                except (OSError, PermissionError):
                    pass
                raise
            updater = getattr(writer, "update_existing_bytes", None)
            if updater is None:
                raise
            updater(relative, payload)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            existing = target.read_bytes()
        except OSError:
            existing = None
        if existing == payload:
            return target
        if not overwrite:
            raise FileExistsError(f"Output document exists with different content: {target}")
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(target)
    return target


def finalize_handoff_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(document)
    manifest["manifest_id"] = "oel.handoff_manifest:" + "0" * 64
    manifest["manifest_id"] = compute_manifest_id(manifest)
    return manifest


def write_handoff_manifest(
    document: Mapping[str, Any],
    path: str | Path,
    *,
    writer: Any | None = None,
    overwrite: bool = True,
) -> Path:
    return write_json_document(document, path, writer=writer, overwrite=overwrite)


__all__ = [
    "bind_output_path",
    "finalize_handoff_manifest",
    "write_handoff_manifest",
    "write_json_document",
]
