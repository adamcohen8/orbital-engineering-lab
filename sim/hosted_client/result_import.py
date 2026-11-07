"""Public verified import of transport-neutral Hosted OEL result transfers."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping

from sim.platform_compat import canonicalize_platform_compatibility_root as _canonicalize_platform_compatibility_root
from sim.study_planning import canonical_sha256

RESULT_TRANSFER_SCHEMA = "oel.hosted_result_transfer.v1"
RESULT_IMPORT_RECEIPT_SCHEMA = "oel.hosted_result_import_receipt.v1"


def import_result_transfer(
    transfer: Mapping[str, Any],
    destination: str | Path,
    *,
    read_file: Callable[[str, str], bytes],
) -> dict[str, Any]:
    """Import a declared transfer into a new directory and verify every byte."""

    document, expected_transfer_sha256 = _validated_transfer(transfer)

    # Keep the target lexical until its components have been checked.  Using
    # resolve() first would turn a symlinked destination into its referent and
    # let the importer write outside the path the caller selected.
    target = _target_path(destination)
    _reject_symlink_components(target)
    if target.exists() or target.is_symlink():
        return _verify_existing_import(document, expected_transfer_sha256, target)
    target.parent.mkdir(parents=True, exist_ok=True)

    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Hosted OEL result transfer declares no artifacts.")

    imported: list[dict[str, Any]] = []
    seen_artifact_ids: set[str] = set()
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as temporary:
        temporary_root = Path(temporary) / "payload"
        temporary_root.mkdir()
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping):
                raise ValueError("Hosted OEL result transfer contains an invalid artifact.")
            artifact_id = str(artifact.get("artifact_id", ""))
            if not artifact_id or artifact_id in seen_artifact_ids:
                raise ValueError("Hosted OEL result artifact ids must be non-empty and unique.")
            seen_artifact_ids.add(artifact_id)
            artifact_root = temporary_root / f"{index:03d}-{_safe_component(str(artifact.get('kind', 'artifact')))}"
            artifact_root.mkdir()
            files = artifact.get("files")
            if not isinstance(files, list) or not files:
                raise ValueError("Hosted OEL result artifact declares no files.")
            imported_files: list[dict[str, Any]] = []
            for relative_path, row in _validated_file_rows(files):
                payload = read_file(artifact_id, relative_path)
                if not isinstance(payload, bytes):
                    raise TypeError("Hosted result transport must return file bytes.")
                observed = hashlib.sha256(payload).hexdigest()
                if observed != row.get("content_sha256") or len(payload) != row.get("bytes"):
                    raise ValueError("Hosted result file did not match its transfer declaration.")
                output = artifact_root / PurePosixPath(relative_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as stream:
                    stream.write(payload)
                imported_files.append(
                    {
                        "relative_path": relative_path,
                        "bytes": len(payload),
                        "content_sha256": observed,
                    }
                )
            declared_artifact_sha256 = artifact.get("content_sha256")
            verified_artifact_sha256 = _verified_aggregate_identity(
                artifact_root,
                declared_artifact_sha256,
            )
            imported.append(
                {
                    "artifact_id": artifact_id,
                    "kind": str(artifact.get("kind", "")),
                    "local_directory": artifact_root.relative_to(temporary_root).as_posix(),
                    # Keep the transfer's verified identity in the receipt.
                    # Directory artifacts use their canonical inventory digest;
                    # a one-file artifact may retain the legacy raw-file digest.
                    "content_sha256": verified_artifact_sha256,
                    "files": imported_files,
                }
            )

        # Verify the materialized bytes, not only each transport response.
        for artifact in imported:
            for row in artifact["files"]:
                output = temporary_root / artifact["local_directory"] / row["relative_path"]
                payload = output.read_bytes()
                if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["content_sha256"]:
                    raise ValueError("Materialized Hosted result did not match its receipt.")

        receipt = {
            "schema": RESULT_IMPORT_RECEIPT_SCHEMA,
            "job_id": document["job_id"],
            "tenant_id": document["tenant_id"],
            "capsule_sha256": document["capsule_sha256"],
            "worker_image_sha256": document["worker_image_sha256"],
            "artifact_bundle_sha256": document["artifact_bundle_sha256"],
            "transfer_sha256": expected_transfer_sha256,
            "artifacts": imported,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        receipt_path = temporary_root / "result_import_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            temporary_root.rename(target)
        except FileExistsError:
            # Another exact retry may have won the destination race after the
            # initial existence check.  Verify that committed import instead of
            # turning a successful concurrent import into a spurious failure.
            return _verify_existing_import(document, expected_transfer_sha256, target)
    return receipt


def verify_result_import(
    transfer: Mapping[str, Any], destination: str | Path
) -> dict[str, Any]:
    """Verify and return an already materialized result import.

    This is used by callers that cache the receipt separately from the public
    importer.  The transfer is revalidated first, then the receipt, file rows,
    artifact tree, and destination identity are checked against it.
    """

    document, expected_transfer_sha256 = _validated_transfer(transfer)
    target = _target_path(destination)
    _reject_symlink_components(target)
    return _verify_existing_import(document, expected_transfer_sha256, target)


def _validated_transfer(
    transfer: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    document = dict(transfer)
    if document.get("schema") != RESULT_TRANSFER_SCHEMA:
        raise ValueError("Unsupported Hosted OEL result-transfer schema.")
    expected_transfer_sha256 = canonical_sha256(
        {key: value for key, value in document.items() if key != "transfer_sha256"}
    )
    if document.get("transfer_sha256") != expected_transfer_sha256:
        raise ValueError("Hosted OEL result transfer failed digest verification.")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Hosted OEL result transfer declares no artifacts.")
    return document, expected_transfer_sha256


def _target_path(destination: str | Path) -> Path:
    return _canonicalize_platform_compatibility_root(
        Path(destination).expanduser().absolute()
    )


def _verify_existing_import(
    document: Mapping[str, Any],
    expected_transfer_sha256: str,
    target: Path,
) -> dict[str, Any]:
    if target.is_symlink() or not target.is_dir():
        raise ValueError("Existing result-import destination is not a regular directory.")
    receipt_path = target / "result_import_receipt.json"
    _reject_symlink_components(receipt_path)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("Existing result-import destination has no verified receipt.")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Existing result-import receipt is unreadable.") from exc
    if not isinstance(receipt, Mapping):
        raise ValueError("Existing result-import receipt must contain an object.")
    receipt = dict(receipt)
    if receipt.get("schema") != RESULT_IMPORT_RECEIPT_SCHEMA:
        raise ValueError("Existing result-import receipt has an unsupported schema.")
    if receipt.get("receipt_sha256") != canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    ):
        raise ValueError("Existing result-import receipt failed digest verification.")
    for key in (
        "job_id",
        "tenant_id",
        "capsule_sha256",
        "worker_image_sha256",
        "artifact_bundle_sha256",
    ):
        if receipt.get(key) != document.get(key):
            raise ValueError("Existing result-import receipt failed transfer identity verification.")
    if receipt.get("transfer_sha256") != expected_transfer_sha256:
        raise ValueError("Existing result-import receipt does not match the current transfer.")

    declared_artifacts = document.get("artifacts")
    imported_artifacts = receipt.get("artifacts")
    if not isinstance(declared_artifacts, list) or not isinstance(imported_artifacts, list):
        raise ValueError("Existing result-import receipt has an invalid artifact inventory.")
    declared_by_id = {
        str(item.get("artifact_id")): item
        for item in declared_artifacts
        if isinstance(item, Mapping) and item.get("artifact_id")
    }
    imported_by_id = {
        str(item.get("artifact_id")): item
        for item in imported_artifacts
        if isinstance(item, Mapping) and item.get("artifact_id")
    }
    if len(declared_by_id) != len(declared_artifacts) or len(imported_by_id) != len(imported_artifacts):
        raise ValueError("Existing result-import receipt contains duplicate or invalid artifacts.")
    if set(declared_by_id) != set(imported_by_id):
        raise ValueError("Existing result-import receipt artifact inventory changed.")

    for artifact_id, declared in declared_by_id.items():
        imported = imported_by_id[artifact_id]
        if imported.get("kind") != declared.get("kind"):
            raise ValueError("Existing result-import receipt changed an artifact kind.")
        local_directory = imported.get("local_directory")
        if not isinstance(local_directory, str):
            raise ValueError("Existing result-import receipt has an unsafe artifact directory.")
        safe_directory = _safe_relative_path(local_directory)
        artifact_root = target / PurePosixPath(safe_directory)
        _reject_symlink_components(artifact_root)
        if artifact_root.is_symlink() or not artifact_root.is_dir():
            raise ValueError("Existing result-import artifact directory is not a regular directory.")

        declared_rows = _validated_file_rows(list(declared.get("files", [])))
        imported_rows = imported.get("files")
        if not isinstance(imported_rows, list):
            raise ValueError("Existing result-import receipt has an invalid file inventory.")
        imported_by_path = {
            str(row.get("relative_path")): row
            for row in imported_rows
            if isinstance(row, Mapping) and row.get("relative_path")
        }
        if len(imported_by_path) != len(imported_rows):
            raise ValueError("Existing result-import receipt contains duplicate file paths.")
        declared_by_path = {relative: row for relative, row in declared_rows}
        if set(declared_by_path) != set(imported_by_path):
            raise ValueError("Existing result-import receipt file inventory changed.")
        for relative, declared_row in declared_by_path.items():
            imported_row = imported_by_path[relative]
            if (
                imported_row.get("bytes") != declared_row.get("bytes")
                or imported_row.get("content_sha256") != declared_row.get("content_sha256")
            ):
                raise ValueError("Existing result-import receipt changed a file declaration.")
            output = artifact_root / PurePosixPath(relative)
            _verify_materialized_file(output, imported_row)

        actual_rows = _file_rows(artifact_root)
        if {
            row["relative_path"]: row for row in actual_rows
        } != {
            str(row.get("relative_path")): row for row in imported_rows
        }:
            raise ValueError("Existing result-import artifact contains unexpected files.")
        declared_artifact_sha256 = declared.get("content_sha256")
        verified_artifact_sha256 = _verified_aggregate_identity(
            artifact_root,
            declared_artifact_sha256,
        )
        if (
            imported.get("content_sha256") is not None
            and imported.get("content_sha256") != verified_artifact_sha256
        ):
            raise ValueError("Existing result-import receipt failed aggregate content verification.")
    return receipt


def _verify_materialized_file(path: Path, row: Mapping[str, Any]) -> None:
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Materialized Hosted result is not a regular file.")
    payload = path.read_bytes()
    if len(payload) != row.get("bytes") or hashlib.sha256(payload).hexdigest() != row.get("content_sha256"):
        raise ValueError("Materialized Hosted result did not match its receipt.")


def _file_rows(path: Path) -> list[dict[str, Any]]:
    _reject_symlink_components(path)
    rows: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Materialized Hosted result contains a symbolic link.")
        if item.is_file():
            payload = item.read_bytes()
            rows.append(
                {
                    "relative_path": item.relative_to(path).as_posix(),
                    "bytes": len(payload),
                    "content_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        elif not item.is_dir():
            raise ValueError("Materialized Hosted result contains a non-regular entry.")
    return rows


def _sha256_path(path: Path) -> str:
    if path.is_symlink():
        raise ValueError("Materialized Hosted result contains a symbolic link.")
    if path.is_file():
        payload = path.read_bytes()
        return hashlib.sha256(payload).hexdigest()
    if path.is_dir():
        return canonical_sha256(
            [
                (row["relative_path"], row["content_sha256"])
                for row in _file_rows(path)
            ]
        )
    raise ValueError("Materialized Hosted result artifact does not exist.")


def _verified_aggregate_identity(
    path: Path,
    declared_sha256: Any,
) -> str:
    """Verify an artifact aggregate while retaining its producer identity.

    Hosted result-transfer v1 has no file-versus-directory discriminator.  A
    producer that returned one file could therefore declare either the raw
    file digest (the legacy file artifact contract) or the canonical inventory
    digest used for directory artifacts.  The canonical tree is always
    computed first.  Only a single, already verified file may additionally
    satisfy the legacy raw-file form; a multi-file artifact must use its
    canonical inventory digest.
    """

    canonical_sha256_value = _sha256_path(path)
    if declared_sha256 is None:
        return canonical_sha256_value
    if declared_sha256 == canonical_sha256_value:
        return str(declared_sha256)

    rows = _file_rows(path)
    if len(rows) == 1 and declared_sha256 == rows[0]["content_sha256"]:
        return str(declared_sha256)
    raise ValueError("Hosted result artifact failed aggregate content verification.")


def _validated_file_rows(files: list[Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Reject portable file and directory aliases before reading any payloads."""
    nodes: dict[tuple[str, ...], tuple[tuple[str, ...], bool]] = {}
    rows = []
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError("Hosted OEL result transfer contains an invalid file row.")
        raw_relative = row.get("relative_path")
        if not isinstance(raw_relative, str):
            raise ValueError("Hosted OEL result transfer contains an invalid file path.")
        raw_bytes = row.get("bytes")
        if type(raw_bytes) is not int or raw_bytes < 0:
            raise ValueError("Hosted OEL result transfer contains an invalid file size.")
        digest = row.get("content_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Hosted OEL result transfer contains an invalid file digest.")
        relative = _safe_relative_path(raw_relative)
        parts = PurePosixPath(relative).parts
        for length in range(1, len(parts) + 1):
            prefix = parts[:length]
            key = tuple(unicodedata.normalize("NFC", part).casefold() for part in prefix)
            is_file = length == len(parts)
            previous = nodes.get(key)
            if previous is not None and (previous[0] != prefix or previous[1] or is_file):
                raise ValueError("Hosted OEL result artifact contains a duplicate or aliasing file path.")
            nodes[key] = (prefix, is_file)
        rows.append((relative, row))
    return rows


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or ":" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
        or any(part.endswith((".", " ")) for part in path.parts)
    ):
        raise ValueError("Unsafe path in Hosted OEL result transfer.")
    return path.as_posix()


def _safe_component(value: str) -> str:
    text = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return text.strip("-")[:64] or "artifact"


def _reject_symlink_components(path: Path) -> None:
    path = _canonicalize_platform_compatibility_root(path)
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError("Result-import destination must not contain symbolic links.")


__all__ = [
    "RESULT_IMPORT_RECEIPT_SCHEMA",
    "RESULT_TRANSFER_SCHEMA",
    "import_result_transfer",
    "verify_result_import",
]
