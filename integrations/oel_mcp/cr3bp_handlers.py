"""Transport-neutral MCP handlers for the experimental CR3BP renderer.

This adapter owns contract, path, approval, and content-bound-plan policy. It
delegates all orbital mechanics and plotting to :mod:`sim.review.cr3bp`.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from integrations.oel_mcp.contracts import MAX_MANIFEST_BYTES
from integrations.oel_mcp.cr3bp_registry import (
    CR3BP_PLAN_TOOL_ID,
    CR3BP_RENDER_TOOL_ID,
    MAX_CR3BP_FPS,
    MAX_CR3BP_FRAMES,
    MAX_CR3BP_GRID_RESOLUTION,
    MAX_CR3BP_REVIEW_STORE_BYTES,
    MAX_CR3BP_ROWS,
)
from integrations.oel_mcp.execution import (
    MCPExecutionCancelled,
    complete_manifest,
    manifest_base,
    safe_artifact_id,
    write_execution_manifest,
)
from sim.review.cr3bp import load_cr3bp_history, render_cr3bp_research
from sim.review.evidence_capsule import evidence_file_mtime_ns
from sim.review.workspace import ReviewWorkspace
from sim.security.input_reads import SecureReadError
from sim.security.output_writes import bind_output_writer

CR3BP_PLAN_SCHEMA_VERSION = 1

_DEFAULTS: dict[str, Any] = {
    "kind": "trajectory",
    "axes": "rotating",
    "origin": "barycenter",
    "plane": "xy",
    "reference_time_s": 0.0,
    "reference_angle_rad": 0.0,
    "slice_km": 0.0,
    "jacobi": None,
    "jacobi_mode": "fixed",
    "bounds_km": None,
    "resolution": MAX_CR3BP_GRID_RESOLUTION,
    "movie_format": None,
    "frames": 120,
    "fps": 20,
    "style": "oel_dark",
}

def cr3bp_plan_id(
    evidence_identity: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> str:
    """Return an immutable id for one evidence identity and normalized spec."""

    payload = {
        "schema_version": CR3BP_PLAN_SCHEMA_VERSION,
        "evidence_identity": dict(evidence_identity),
        "spec": dict(spec),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "oel-cr3bp-render-plan-v1:" + hashlib.sha256(encoded).hexdigest()


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _normalize_spec(arguments: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Normalize the renderer's public arguments without doing physics."""

    values = {key: arguments.get(key, default) for key, default in _DEFAULTS.items()}
    object_id = str(arguments.get("object_id", "")).strip()
    if not object_id:
        raise ValueError("object_id is required.")
    kind = str(values["kind"])
    axes = str(values["axes"])
    origin = str(values["origin"])
    plane = str(values["plane"])
    jacobi_mode = str(values["jacobi_mode"])
    movie_format = values["movie_format"]
    if movie_format is not None:
        movie_format = str(movie_format)
    if kind not in {"trajectory", "zero_velocity", "jacobi"}:
        raise ValueError("Use trajectory, zero_velocity, or jacobi.")
    if axes not in {"rotating", "inertial"}:
        raise ValueError("Axes must be rotating or inertial.")
    if origin not in {"barycenter", "p1", "p2"}:
        raise ValueError("Origin must be barycenter, p1, or p2.")
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("Plane must be xy, xz, or yz.")
    if movie_format not in {None, "gif", "mp4"}:
        raise ValueError("Movie format must be gif, mp4, or null.")
    if jacobi_mode not in {"fixed", "instantaneous"}:
        raise ValueError("Jacobi mode must be fixed or instantaneous.")
    if movie_format and kind == "jacobi":
        raise ValueError("Use a static Jacobi diagnostic.")
    if kind == "zero_velocity" and axes != "rotating":
        raise ValueError("Zero-velocity slices require rotating axes.")
    if jacobi_mode == "instantaneous":
        if kind != "zero_velocity" or not movie_format:
            raise ValueError("Instantaneous Jacobi mode requires a zero_velocity movie.")
        if values["jacobi"] is not None:
            raise ValueError("An explicit fixed Jacobi value conflicts with instantaneous mode.")

    reference_time_s = _finite_number(values["reference_time_s"], "reference_time_s")
    reference_angle_rad = _finite_number(values["reference_angle_rad"], "reference_angle_rad")
    slice_km = _finite_number(values["slice_km"], "slice_km")
    jacobi = values["jacobi"]
    if jacobi is not None:
        jacobi = _finite_number(jacobi, "jacobi")
    resolution = int(values["resolution"])
    frames = int(values["frames"])
    fps = int(values["fps"])
    if not 16 <= resolution <= MAX_CR3BP_GRID_RESOLUTION:
        raise ValueError(f"resolution must be between 16 and {MAX_CR3BP_GRID_RESOLUTION}.")
    if not 2 <= frames <= MAX_CR3BP_FRAMES:
        raise ValueError(f"frames must be between 2 and {MAX_CR3BP_FRAMES}.")
    if not 1 <= fps <= MAX_CR3BP_FPS:
        raise ValueError(f"fps must be between 1 and {MAX_CR3BP_FPS}.")
    if movie_format and frames / fps > 30.0:
        raise ValueError("CR3BP movie duration is limited to 30 seconds (frames / fps).")
    bounds = values["bounds_km"]
    if bounds is not None:
        if len(bounds) != 4:
            raise ValueError("bounds_km must have four values.")
        bounds = [_finite_number(value, "bounds_km") for value in bounds]
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("bounds_km must contain increasing u/v limits.")
    return {
        "output_dir": str(output_dir),
        "object_id": object_id,
        "kind": kind,
        "axes": axes,
        "origin": origin,
        "plane": plane,
        "reference_time_s": reference_time_s,
        "reference_angle_rad": reference_angle_rad,
        "slice_km": slice_km,
        "jacobi": jacobi,
        "jacobi_mode": jacobi_mode,
        "bounds_km": bounds,
        "resolution": resolution,
        "movie_format": movie_format,
        "frames": frames,
        "fps": fps,
        "style": str(values["style"]),
    }


def _reject_symlinked_path(path: Path, *, root: Path, policy: Any, kind: str) -> Path:
    """Resolve a path while rejecting every lexical symlink component."""

    lexical = Path(path).absolute()
    resolved = policy.resolve_write(lexical) if kind == "write" else policy.resolve_read(lexical, kind="any")
    if resolved != lexical:
        raise PermissionError("Path is not authorized for this operation: symlinked path component.")
    current = lexical
    while True:
        if current.is_symlink():
            raise PermissionError("Path is not authorized for this operation: symlinked path component.")
        if current == root:
            break
        if root not in current.parents:
            raise PermissionError("Path is not authorized for this operation: path escaped its intended parent.")
        current = current.parent
    return resolved


def _read_child(handler: Any, output: Path, *parts: str, kind: str = "any") -> Path:
    lexical = output.joinpath(*parts)
    try:
        return handler.path_policy.authorized_read_path(lexical, kind=kind)
    except FileNotFoundError:
        return lexical


def _require_review_input(handler: Any, output: Path) -> None:
    """Authorize the logical review store and enforce its input-size budget."""

    policy = handler.path_policy
    db = _read_child(handler, output, "review", "run.sqlite", kind="file")
    canonical_capsule = _read_child(handler, output, "review", "run.sqlite.gz", kind="file")
    capsule_manifest = _read_child(handler, output, "review", "evidence_capsule.json", kind="file")
    capsule = canonical_capsule
    compressed_capsule: Path | None = None
    capsule_entry: dict[str, Any] | None = None
    manifest_bytes: bytes | None = None
    try:
        _manifest_path, manifest_bytes = _read_authorized_with_budget(
            policy,
            capsule_manifest,
            maximum=MAX_MANIFEST_BYTES,
        )
    except FileNotFoundError:
        manifest_bytes = None
    if manifest_bytes is not None:
        try:
            payload = json.loads(manifest_bytes.decode("utf-8"))
            entries = [
                dict(item)
                for item in list(dict(payload or {}).get("artifacts", []) or [])
                if isinstance(item, dict) and item.get("logical_path") == "run.sqlite"
            ]
            if not entries:
                # A normal, uncompressed review store may share a capsule
                # manifest for another logical artifact.  Only a manifest
                # that claims to describe run.sqlite is relevant here.
                canonical_capsule_record = _authorized_optional_record(
                    policy,
                    canonical_capsule,
                    maximum=MAX_CR3BP_REVIEW_STORE_BYTES,
                )
                db_record = _authorized_optional_record(
                    policy,
                    db,
                    maximum=MAX_CR3BP_REVIEW_STORE_BYTES,
                )
                if canonical_capsule_record is not None or db_record is None:
                    raise ValueError("Evidence capsule manifest must describe exactly one CR3BP SQLite store.")
            elif len(entries) != 1:
                raise ValueError("Evidence capsule manifest must describe exactly one CR3BP SQLite store.")
            if not entries:
                entries = []
            else:
                capsule_entry = entries[0]
            if capsule_entry is None:
                compressed_name = ""
            else:
                compressed_name = str(capsule_entry.get("compressed_path", "") or "")
            compressed_candidate = Path(compressed_name)
            if capsule_entry is not None and (
                not compressed_name
                or compressed_candidate.is_absolute()
                or len(compressed_candidate.parts) != 1
                or compressed_candidate.name != compressed_name
            ):
                raise ValueError("Evidence capsule has an unsafe compressed path.")
            if capsule_entry is not None:
                compressed_capsule = _read_child(
                    handler,
                    output,
                    "review",
                    compressed_name,
                    kind="file",
                )
                try:
                    _compressed_path, compressed_record = policy.authorized_file_record(
                        compressed_capsule,
                        maximum=MAX_CR3BP_REVIEW_STORE_BYTES,
                    )
                except FileNotFoundError as exc:
                    raise FileNotFoundError("Evidence capsule compressed SQLite store is missing.") from exc
                except SecureReadError as exc:
                    if "read budget" in str(exc).lower():
                        raise ValueError("Evidence capsule compressed payload exceeds the file-size budget.") from exc
                    raise
                compressed_bytes = int(capsule_entry.get("compressed_bytes", -1))
                if compressed_bytes != int(compressed_record["bytes"]):
                    raise ValueError("Evidence capsule compressed size does not match its manifest.")
                original_bytes = int(capsule_entry.get("original_bytes", -1))
                if original_bytes < 0 or original_bytes > MAX_CR3BP_REVIEW_STORE_BYTES:
                    raise ValueError("Hydrated CR3BP evidence exceeds the file-size budget.")
                # A manifest may deliberately use a safe, noncanonical basename;
                # ReviewWorkspace follows that content-bound mapping when hydrating.
                capsule = compressed_capsule
        except ValueError as exc:
            # Preserve the established public size-budget contract while
            # collapsing malformed capsule details to one safe diagnostic.
            if "file-size budget" in str(exc).lower():
                raise
            raise ValueError("Evidence capsule has no valid bounded compressed-size declaration.") from exc
        except SecureReadError as exc:
            if "read budget" in str(exc).lower():
                raise ValueError("Authorized input exceeds the file-size budget.") from exc
            raise
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Evidence capsule has no valid bounded compressed-size declaration.") from exc

    db_present = _authorized_optional_record(policy, db, maximum=MAX_CR3BP_REVIEW_STORE_BYTES)
    capsule_present = _authorized_optional_record(policy, capsule, maximum=MAX_CR3BP_REVIEW_STORE_BYTES)
    if not db_present and not capsule_present:
        raise FileNotFoundError("Required CR3BP review evidence is missing.")
    sidecars = [
        _read_child(handler, output, "review", "run.sqlite-wal", kind="file"),
        _read_child(handler, output, "review", "run.sqlite-shm", kind="file"),
    ]
    total_bytes = 0
    counted: set[Path] = set()
    for candidate in (db, canonical_capsule, compressed_capsule):
        if candidate is not None:
            candidate = candidate.absolute()
            if candidate in counted:
                continue
            record = _authorized_optional_record(
                policy,
                candidate,
                maximum=MAX_CR3BP_REVIEW_STORE_BYTES,
            )
            if record is None:
                continue
            total_bytes += int(record["bytes"])
            counted.add(candidate)
    for candidate in sidecars:
        record = _authorized_optional_record(
            policy,
            candidate,
            maximum=MAX_CR3BP_REVIEW_STORE_BYTES,
        )
        if record is not None:
            total_bytes += int(record["bytes"])
    if total_bytes > MAX_CR3BP_REVIEW_STORE_BYTES:
        raise ValueError("CR3BP review store and SQLite sidecars exceed the file-size budget.")
    # The renderer also reads the schema/saved-view sidecars when present.  A
    # sidecar is optional, but resolving it prevents an escaping symlink from
    # becoming an implicit renderer input.
    for name in (
        "schema.json",
        "saved_views.json",
        "evidence_capsule.json",
        "workflow_manifest.json",
        "generated_artifacts.json",
    ):
        sidecar = _read_child(handler, output, "review", name, kind="file")
        _authorized_optional_record(policy, sidecar, maximum=MAX_MANIFEST_BYTES)
    for name in ("mcp_execution_manifest.json", "master_run_summary.json", "agent_evidence_packet.json", "index.md"):
        sidecar = _read_child(handler, output, name, kind="file")
        _authorized_optional_record(policy, sidecar, maximum=MAX_MANIFEST_BYTES)

    # A compressed logical store is accepted only when its declared hydrated
    # size is within the same MCP budget used for an ordinary SQLite file.
    if capsule_entry is not None and compressed_capsule is None:
        raise ValueError("Evidence capsule compressed SQLite store is missing.")


def _authorized_optional_record(policy: Any, path: Path, *, maximum: int) -> dict[str, Any] | None:
    try:
        _logical, record = policy.authorized_file_record(path, maximum=maximum)
    except FileNotFoundError:
        return None
    except SecureReadError as exc:
        if "read budget" in str(exc).lower():
            raise ValueError("Authorized input exceeds the file-size budget.") from exc
        raise
    return dict(record)


def _read_authorized_with_budget(
    policy: Any,
    path: Path,
    *,
    maximum: int,
) -> tuple[Path, bytes]:
    """Read one CR3BP input while retaining its stable size error contract."""

    try:
        return policy.read_authorized_bytes(path, maximum=maximum)
    except SecureReadError as exc:
        if "read budget" in str(exc).lower():
            raise ValueError("Authorized input exceeds the file-size budget.") from exc
        raise


def _same_review_content(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("relative_path", "size_bytes", "sha256", "storage"))


def _source_review_identity(handler: Any, output: Path) -> dict[str, Any]:
    """Return identity for the authorized logical source, not its sealed copy."""

    with ReviewWorkspace.open(output, protected=True) as workspace:
        identity = dict(workspace.evidence_identity())
    # Capsule hydration creates a fresh temporary inode on every open.  Use
    # the content-bound source mtime so a plan remains stable across calls.
    # Direct SQLite identities already include a retained WAL descriptor; the
    # explicit source stat is only needed for a compressed capsule, where the
    # hydrated inode is intentionally ephemeral.
    if identity.get("storage") == "sqlite_gzip_capsule":
        identity["mtime_ns"] = int(
            evidence_file_mtime_ns(output / "review" / "run.sqlite", secure=True)
        )
    return identity


@contextmanager
def _secure_review_input_snapshot(handler: Any, output: Path):
    """Seal the bounded review inputs before handing them to SQLite/plotting."""

    policy = handler.path_policy
    paths = [
        "review/run.sqlite",
        "review/run.sqlite-wal",
        "review/run.sqlite-shm",
        "review/run.sqlite.gz",
        "review/evidence_capsule.json",
        "review/schema.json",
        "review/saved_views.json",
        "review/workflow_manifest.json",
        "review/generated_artifacts.json",
        "mcp_execution_manifest.json",
        "master_run_summary.json",
        "agent_evidence_packet.json",
        "index.md",
    ]
    try:
        _manifest_path, manifest_bytes = policy.read_authorized_bytes(
            output / "review" / "evidence_capsule.json",
            maximum=MAX_MANIFEST_BYTES,
        )
        payload = json.loads(manifest_bytes.decode("utf-8"))
        entries = [
            dict(item)
            for item in list(dict(payload or {}).get("artifacts", []) or [])
            if isinstance(item, dict) and item.get("logical_path") == "run.sqlite"
        ]
        if len(entries) == 1:
            compressed_name = str(entries[0].get("compressed_path", "") or "")
            candidate = Path(compressed_name)
            if compressed_name and not candidate.is_absolute() and len(candidate.parts) == 1 and candidate.name == compressed_name:
                paths.append(str(Path("review") / compressed_name))
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        # _require_review_input reports the authoritative malformed/missing
        # capsule error.  Keep the snapshot allowlist conservative here.
        pass
    with policy.snapshot_read_files(
        output,
        paths,
        maximum_file_bytes=MAX_CR3BP_REVIEW_STORE_BYTES,
        maximum_total_bytes=MAX_CR3BP_REVIEW_STORE_BYTES + 8 * MAX_MANIFEST_BYTES,
    ) as (logical_root, snapshot):
        yield logical_root, snapshot


def _authorize_renderer_writes(handler: Any, output: Path) -> tuple[Path, ...]:
    """Pre-authorize every durable path the existing renderer can update."""

    review = output / "review"
    figures = review / "figures"
    manifest_dir = review / "mcp_cr3bp"
    _reject_symlinked_path(output, root=output, policy=handler.path_policy, kind="write")
    _reject_symlinked_path(review, root=output, policy=handler.path_policy, kind="write")
    _reject_symlinked_path(figures, root=review, policy=handler.path_policy, kind="write")
    _reject_symlinked_path(manifest_dir, root=review, policy=handler.path_policy, kind="write")
    _reject_symlinked_path(review / "generated_artifacts.json", root=review, policy=handler.path_policy, kind="write")
    _reject_symlinked_path(output / "index.md", root=output, policy=handler.path_policy, kind="write")
    return figures, manifest_dir


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _authorize_renderer_result(handler: Any, output: Path, result: Mapping[str, Any]) -> list[Path]:
    """Verify every renderer-returned file stays under an authorized root."""

    figures = (output / "review" / "figures").resolve()
    files: list[Path] = []
    for key, raw in result.items():
        if key in {"diagnostics"} or raw is None:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = output / path
        resolved = handler.path_policy.resolve_write(path)
        if not _is_relative_to(resolved, figures):
            raise PermissionError("CR3BP renderer returned a path outside its authorized artifact directory.")
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError("CR3BP renderer returned a missing or symlinked artifact.")
        files.append(resolved)
    if not files:
        raise ValueError("CR3BP renderer did not produce a durable artifact.")
    return files


def _new_files(figures: Path, before: set[Path], handler: Any) -> list[Path]:
    if not figures.is_dir() or figures.is_symlink():
        return []
    files: list[Path] = []
    for candidate in figures.rglob("*"):
        lexical = candidate.absolute()
        if lexical in before or not candidate.is_file():
            continue
        files.append(handler.path_policy.resolve_write(lexical))
    return files


def _operation_manifest(
    handler: Any,
    *,
    contract: Any,
    arguments: Mapping[str, Any],
    spec: Mapping[str, Any],
    evidence_identity: Mapping[str, Any],
    manifest_dir: Path,
    manifest_writer: Any,
) -> tuple[dict[str, Any], Path]:
    plan_id = str(arguments["cr3bp_plan_id"])
    token = safe_artifact_id("cr3bp_" + plan_id.rsplit(":", 1)[-1][:32])
    target = manifest_dir / f"{token}.manifest.json"
    _reject_symlinked_path(target, root=manifest_dir, policy=handler.path_policy, kind="write")
    manifest = manifest_base(
        tool_id=contract.tool_id,
        approval_id=str(arguments["approval"]["approval_id"]),
    )
    manifest.update(
        {
            "cr3bp_plan_id": plan_id,
            "cr3bp_spec": dict(spec),
            "review_store": dict(evidence_identity),
            "renderer": "sim.review.cr3bp.render_cr3bp_research",
            "inertial_alignment": "CR3BP model axes, not J2000",
            "manifest_filename": target.name,
        }
    )
    serialized = (json.dumps(dict(manifest), indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    # The child writer retains the descriptor for the approved manifest
    # directory.  Its exclusive create both reserves the operation receipt and
    # records the inode that may later be finalized by this same operation.
    written = manifest_writer.write_bytes(target.name, serialized)
    return manifest, written


def _finish_operation_manifest(
    manifest_dir: Path,
    target: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_writer: Any,
    status: str,
    artifacts: list[Path],
    cancelled: bool = False,
    error_type: str = "",
) -> Path:
    completed = complete_manifest(
        dict(manifest),
        status=status,
        cancelled=cancelled,
        artifacts=[str(path) for path in artifacts if path.is_file() and not path.is_symlink()],
        error_type=error_type,
    )
    return write_execution_manifest(
        manifest_dir,
        completed,
        filename=target.name,
        writer=manifest_writer,
    )


def _plan_cr3bp_render(handler: Any, contract: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        output = handler.path_policy.authorized_read_path(arguments["output_dir"], kind="directory")
        _require_review_input(handler, output)
        spec = _normalize_spec(arguments, output_dir=output)
        source_identity = _source_review_identity(handler, output)
        with _secure_review_input_snapshot(handler, output) as (_logical_root, snapshot):
            with ReviewWorkspace.open(
                snapshot,
                protected=True,
                logical_output_dir=output,
            ) as workspace:
                initial_identity = workspace.evidence_identity()
                if not _same_review_content(source_identity, initial_identity):
                    raise ValueError("Review store changed while planning; no CR3BP render plan was issued.")
                times, _states, system = load_cr3bp_history(workspace, spec["object_id"])
                final_identity = workspace.evidence_identity()
                if final_identity != initial_identity:
                    raise ValueError("Review store changed while planning; no CR3BP render plan was issued.")
        return {
            "status": "planned",
            "output_dir": str(output),
            "review_store": "review/run.sqlite",
            "cr3bp_plan_id": cr3bp_plan_id(source_identity, spec),
            "spec": spec,
            "evidence_identity": source_identity,
            "system": asdict(system),
            "row_count": int(times.size),
            "time_range_s": [float(times[0]), float(times[-1])],
            "warnings": [
                "CR3BP MCP rendering is experimental and requires visual review.",
                "Inertial display axes are CR3BP model axes, not J2000.",
            ],
            "limits": {
                "max_review_store_bytes": MAX_CR3BP_REVIEW_STORE_BYTES,
                "max_rows": MAX_CR3BP_ROWS,
                "max_grid_resolution": MAX_CR3BP_GRID_RESOLUTION,
                "max_frames": MAX_CR3BP_FRAMES,
                "max_fps": MAX_CR3BP_FPS,
            },
            "render_authorized": False,
            "visual_review_required": True,
        }

    return handler._envelope(
        contract=contract,
        arguments=arguments,
        operation=operation,
        evidence=lambda result: {
            "complete": True,
            "empty": False,
            "truncated": False,
        },
    )


def _render_cr3bp(
    handler: Any,
    contract: Any,
    arguments: dict[str, Any],
    *,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        output = handler.path_policy.authorized_read_path(arguments["output_dir"], kind="directory")
        _require_review_input(handler, output)
        figures, manifest_dir = _authorize_renderer_writes(handler, output)
        spec = _normalize_spec(arguments, output_dir=output)
        source_identity = _source_review_identity(handler, output)
        manifest: dict[str, Any] | None = None
        manifest_target: Path | None = None
        manifest_path: Path | None = None
        manifest_finalized = False
        rendered: dict[str, Any] = {}
        before_files = {path.absolute() for path in figures.rglob("*")} if figures.is_dir() else set()

        def check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise MCPExecutionCancelled("The CR3BP render was cancelled before its deterministic render boundary.")

        # The input review store remains a read-only path input.  All renderer
        # outputs and the operation receipt share one descriptor-bound writer;
        # no producer can fall back to path-only writes while it is bound.
        with handler.path_policy.open_secure_writer(output, create=True, require_empty=False) as output_writer:
            with output_writer.child(Path("review") / "mcp_cr3bp", create=True) as manifest_writer:
                with bind_output_writer(output_writer):
                    try:
                        with _secure_review_input_snapshot(handler, output) as (_logical_root, snapshot):
                            with ReviewWorkspace.open(
                                snapshot,
                                protected=True,
                                logical_output_dir=output,
                            ) as workspace:
                                initial_identity = workspace.evidence_identity()
                                if not _same_review_content(source_identity, initial_identity):
                                    raise ValueError("Review store changed while rendering; generated artifacts are not valid evidence.")
                                expected_plan = cr3bp_plan_id(source_identity, spec)
                                if str(arguments["cr3bp_plan_id"]) != expected_plan:
                                    raise ValueError("The CR3BP plan is stale or does not match the review store and specification.")
                                manifest, manifest_path = _operation_manifest(
                                    handler,
                                    contract=contract,
                                    arguments=arguments,
                                    spec=spec,
                                    evidence_identity=source_identity,
                                    manifest_dir=manifest_dir,
                                    manifest_writer=manifest_writer,
                                )
                                manifest_target = manifest_path
                                check_cancelled()
                                rendered = render_cr3bp_research(
                                    workspace,
                                    spec["object_id"],
                                    kind=spec["kind"],
                                    axes=spec["axes"],
                                    origin=spec["origin"],
                                    plane=spec["plane"],
                                    reference_time_s=spec["reference_time_s"],
                                    reference_angle_rad=spec["reference_angle_rad"],
                                    slice_km=spec["slice_km"],
                                    jacobi=spec["jacobi"],
                                    jacobi_mode=spec["jacobi_mode"],
                                    bounds_km=spec["bounds_km"],
                                    resolution=spec["resolution"],
                                    movie_format=spec["movie_format"],
                                    frames=spec["frames"],
                                    fps=spec["fps"],
                                    style=spec["style"],
                                    record=False,
                                )
                                files = _authorize_renderer_result(handler, output, rendered)
                                final_identity = workspace.evidence_identity()
                                source_final_identity = _source_review_identity(handler, output)
                                if final_identity != initial_identity or source_final_identity != source_identity:
                                    _finish_operation_manifest(
                                        manifest_dir,
                                        manifest_target,
                                        manifest,
                                        manifest_writer=manifest_writer,
                                        status="failed",
                                        artifacts=files,
                                        error_type="EvidenceChanged",
                                    )
                                    manifest_finalized = True
                                    raise ValueError(
                                        "Review store changed while rendering; generated artifacts are not valid evidence."
                                    )
                                if cancel_event is not None and cancel_event.is_set():
                                    _finish_operation_manifest(
                                        manifest_dir,
                                        manifest_target,
                                        manifest,
                                        manifest_writer=manifest_writer,
                                        status="cancelled",
                                        cancelled=True,
                                        artifacts=files,
                                        error_type="MCPExecutionCancelled",
                                    )
                                    manifest_finalized = True
                                    raise MCPExecutionCancelled("The CR3BP render was cancelled after its render boundary.")

                                artifact_path = Path(str(rendered["figure"]))
                                if not artifact_path.is_absolute():
                                    artifact_path = output / artifact_path
                                artifact_id = safe_artifact_id(artifact_path.stem)
                                manifest.update({"artifact_id": artifact_id})
                                manifest_path = _finish_operation_manifest(
                                    manifest_dir,
                                    manifest_target,
                                    manifest,
                                    manifest_writer=manifest_writer,
                                    status="completed" if all(path.is_file() for path in files) else "partial",
                                    artifacts=files,
                                )
                                manifest_finalized = True
                    except BaseException as exc:
                        if manifest is not None and manifest_target is not None and not manifest_finalized:
                            try:
                                artifacts = _authorize_renderer_result(handler, output, rendered) if rendered else []
                            except Exception:
                                artifacts = _new_files(figures, before_files, handler)
                            try:
                                _finish_operation_manifest(
                                    manifest_dir,
                                    manifest_target,
                                    manifest,
                                    manifest_writer=manifest_writer,
                                    status="cancelled" if isinstance(exc, MCPExecutionCancelled) else "failed",
                                    cancelled=isinstance(exc, MCPExecutionCancelled),
                                    artifacts=artifacts,
                                    error_type=type(exc).__name__,
                                )
                            except Exception:
                                # Preserve the original renderer/error result.
                                # The started receipt remains durable if its
                                # terminalization itself fails.
                                pass
                        raise

        artifact = {
            "artifact_id": artifact_id,
            "path": str(artifact_path),
            "relative_path": str(artifact_path.relative_to(output)),
            "path_exists": artifact_path.is_file(),
            "kind": spec["kind"],
            "movie_format": spec["movie_format"],
            "files": {str(key): str(value) for key, value in rendered.items() if key != "diagnostics"},
            "diagnostics": rendered.get("diagnostics", {}),
            "visual_qa_status": "pending_agent_review",
            "experimental": True,
            "inertial_alignment": "CR3BP model axes, not J2000",
        }
        status = "completed" if all(path.is_file() for path in files) else "partial"
        return {
            "status": status,
            "output_dir": str(output),
            "cr3bp_plan_id": str(arguments["cr3bp_plan_id"]),
            "artifact": artifact,
            "manifest_path": str(manifest_path),
        }

    return handler._envelope(
        contract=contract,
        arguments=arguments,
        operation=operation,
        outcome_status=lambda result: result["status"],
        evidence=lambda result: {
            "complete": result["status"] == "completed",
            "empty": False,
            "truncated": False,
        },
    )


def call_cr3bp_tool(
    handler: Any,
    contract: Any,
    arguments: dict[str, Any],
    *,
    cancel_event: Any | None = None,
    progress: Any | None = None,
) -> dict[str, Any] | None:
    """Dispatch one CR3BP contract from a central OEL MCP handler.

    ``None`` means the caller should continue dispatching its own contracts.
    This keeps public/pro registries and the SDK integration owned by their
    respective central modules.
    """

    if contract.tool_id == CR3BP_PLAN_TOOL_ID:
        return _plan_cr3bp_render(handler, contract, arguments)
    if contract.tool_id == CR3BP_RENDER_TOOL_ID:
        return _render_cr3bp(handler, contract, arguments, cancel_event=cancel_event)
    return None


class CR3BPMCPHandlersMixin:
    """Optional convenience methods for a central handler class."""

    def plan_cr3bp_render(self, **arguments: Any) -> dict[str, Any]:
        return self.call(CR3BP_PLAN_TOOL_ID, arguments)

    def render_cr3bp(self, **arguments: Any) -> dict[str, Any]:
        return self.call(CR3BP_RENDER_TOOL_ID, arguments)


__all__ = [
    "CR3BP_PLAN_SCHEMA_VERSION",
    "CR3BPMCPHandlersMixin",
    "call_cr3bp_tool",
    "cr3bp_plan_id",
]
