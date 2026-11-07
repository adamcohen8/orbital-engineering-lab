"""Project MCP results for a direct-frontier deployment.

The direct-frontier profile may expose structured results produced by local
tools.  Authorized-root paths already have a compatibility-sensitive opaque
reference format.  This module preserves that format and adds a conservative
guard for absolute local paths that happen to fall outside the configured
roots.  The latter matters for projected review rows and embedded JSON, where
the path policy cannot prove that every producer used an authorized root.

This is a projection boundary, not a filesystem canonicalizer.  It does not
resolve symlinks, inspect the filesystem, or claim that an opaque reference is
an authenticated identity for the source path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from typing import Any

_JSON_MAX_CHARS = 1_000_000
_JSON_START_CHARS = frozenset('{["')
_URL_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+")
_URL_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_POSIX_ABSOLUTE_RE = re.compile(r"(?<![\w:])/(?!/)[^\s\"'<>/]+(?:/[^\s\"'<>/]*)*")
_WINDOWS_DRIVE_ABSOLUTE_RE = re.compile(
    r"(?<![\w])[A-Za-z]:[\\/][^\s\"'<>]*"
)
_UNC_ABSOLUTE_RE = re.compile(
    r"(?<![\w:])(?:\\\\|//)[^\\/\s\"'<>]+[\\/][^\\/\s\"'<>]+"
    r"(?:[\\/][^\\/\s\"'<>]+)*"
)


def frontier_safe_value(
    value: Any,
    *,
    authorized_roots: Iterable[str | os.PathLike[str]] = (),
) -> Any:
    """Return ``value`` with local path disclosures projected safely.

    Strings containing a configured root retain the historic
    ``oel-local-ref:<digest>`` algorithm.  Other absolute local paths are
    projected to an opaque reference as well.  ``PathLike`` values are
    converted before projection so they cannot bypass the string boundary and
    reach the later JSON-safety pass as an unhandled path object.
    """

    roots = tuple(os.fsdecode(os.fspath(root)) for root in authorized_roots)
    return _project(value, roots)


def project_frontier_value(
    value: Any,
    *,
    authorized_roots: Iterable[str | os.PathLike[str]] = (),
) -> Any:
    """Named alias for callers that describe the operation as projection."""

    return frontier_safe_value(value, authorized_roots=authorized_roots)


def _project(value: Any, roots: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {key: _project(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [_project(item, roots) for item in value]
    if isinstance(value, tuple):
        return [_project(item, roots) for item in value]
    if isinstance(value, os.PathLike):
        return _project_string(os.fsdecode(os.fspath(value)), roots)
    if isinstance(value, str):
        return _project_string(value, roots)
    return value


def _project_string(value: str, roots: tuple[str, ...]) -> str:
    legacy = _legacy_opaque_ref(value, roots)
    if legacy is not None:
        return legacy

    # Parse complete JSON text before testing the outer string for a path.  A
    # query row may carry a JSON document as a string; projecting its values
    # individually avoids replacing an otherwise useful document wholesale.
    if _looks_like_json(value):
        projected_json = _project_embedded_json(value, roots)
        if projected_json is not None:
            return projected_json

    if _contains_absolute_local_path(value):
        return _opaque_ref(value)
    return value


def _legacy_opaque_ref(value: str, roots: tuple[str, ...]) -> str | None:
    if not any(root in value for root in roots):
        return None
    normalized = value
    for index, root in enumerate(roots):
        normalized = normalized.replace(root, f"<AUTHORIZED_ROOT_{index}>")
    normalized = normalized.replace("\\", "/")
    return _opaque_ref(normalized)


def _project_embedded_json(value: str, roots: tuple[str, ...]) -> str | None:
    if len(value) > _JSON_MAX_CHARS:
        return None
    try:
        parsed = json.loads(value)
        projected, changed = _project_json_value(parsed, roots)
    except (TypeError, ValueError, RecursionError):
        return None
    if not changed:
        # Preserve ordinary JSON text exactly when no local path was found.
        return None
    try:
        return json.dumps(projected, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError, RecursionError):
        return None


def _project_json_value(value: Any, roots: tuple[str, ...]) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        projected: dict[Any, Any] = {}
        for key, item in value.items():
            projected_item, item_changed = _project_json_value(item, roots)
            projected[key] = projected_item
            changed = changed or item_changed
        return projected, changed
    if isinstance(value, list):
        changed = False
        projected_items = []
        for item in value:
            projected_item, item_changed = _project_json_value(item, roots)
            projected_items.append(projected_item)
            changed = changed or item_changed
        return projected_items, changed
    if isinstance(value, str):
        projected = _project_string(value, roots)
        return projected, projected != value
    return value, False


def _looks_like_json(value: str) -> bool:
    stripped = value.lstrip()
    return bool(stripped) and stripped[0] in _JSON_START_CHARS


def _contains_absolute_local_path(value: str) -> bool:
    # A complete URL, including file:// URLs, is an intentional non-local-path
    # value at this boundary.  Scheme URLs embedded in prose are masked while
    # looking for a separate local path, so the URL itself is not mistaken for
    # a UNC path.
    if _URL_PREFIX_RE.match(value) and not any(char.isspace() for char in value):
        return False
    if value.strip() == "/":
        return True
    search_value = _URL_TOKEN_RE.sub(" ", value)
    return bool(
        _POSIX_ABSOLUTE_RE.search(search_value)
        or _WINDOWS_DRIVE_ABSOLUTE_RE.search(search_value)
        or _UNC_ABSOLUTE_RE.search(search_value)
    )


def _opaque_ref(value: str) -> str:
    digest = hashlib.sha256(value.replace("\\", "/").encode("utf-8")).hexdigest()[:16]
    return f"oel-local-ref:{digest}"


__all__ = ["frontier_safe_value", "project_frontier_value"]
