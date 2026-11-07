"""Deterministic presentation-quality rules for saved OEL animations.

The animation policy owns display formatting, temporal presentation checks,
encoding verification, and visual-review aids.  It never changes review-store
evidence, simulation samples, or deterministic physics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from sim.plotting.quality import (
    STRICT_AGENT_PLOT_QUALITY,
    NumericFormatDecision,
    PlotQualityIssue,
    apply_plot_quality_policy,
    assess_plot_quality,
)
from sim.plotting.style import OELArtifactMetadata, prepare_oel_animation_figure

ANIMATION_QUALITY_POLICY_VERSION = 1


@dataclass(frozen=True)
class AnimationQualityPolicy:
    """Versioned, bounded rules for one encoded animation artifact."""

    policy_id: str = "oel.agent_animation_strict"
    version: int = ANIMATION_QUALITY_POLICY_VERSION
    max_frames: int = 600
    max_duration_s: float = 30.0
    max_file_bytes: int = 100_000_000
    minimum_file_bytes: int = 1_000
    contact_sheet_frames: int = 9
    contact_sheet_thumbnail_width_px: int = 420
    camera_span_relative_tolerance: float = 1e-6
    max_reported_failed_frames: int = 50

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("Animation quality policy version must be positive.")
        if self.max_frames < 2:
            raise ValueError("Animation quality max_frames must be at least two.")
        if self.max_duration_s <= 0.0:
            raise ValueError("Animation quality max_duration_s must be positive.")
        if self.max_file_bytes <= self.minimum_file_bytes:
            raise ValueError("Animation quality file limits are inconsistent.")
        if self.contact_sheet_frames < 2:
            raise ValueError("Animation contact sheets must include at least two frames.")
        if self.contact_sheet_thumbnail_width_px <= 0:
            raise ValueError("Animation contact sheet thumbnail width must be positive.")
        if not math.isfinite(float(self.camera_span_relative_tolerance)) or self.camera_span_relative_tolerance < 0.0:
            raise ValueError("Animation camera tolerance must be finite and nonnegative.")
        if self.max_reported_failed_frames < 0:
            raise ValueError("Animation max_reported_failed_frames cannot be negative.")


STRICT_AGENT_ANIMATION_QUALITY = AnimationQualityPolicy()


@dataclass(frozen=True)
class AnimationEncodingReport:
    """Portable checks over the encoded GIF or MP4 file."""

    format: str
    exists: bool
    size_bytes: int
    sha256: str
    width_px: int | None
    height_px: int | None
    frame_count: int | None
    fps: float | None
    duration_s: float | None
    decode_ok: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnimationQualityReport:
    """Serializable evidence for presentation, temporal, and file QA."""

    policy_id: str
    policy_version: int
    automated_status: str
    camera_policy: str
    frames_expected: int
    frames_checked: int
    failed_frame_count: int
    failed_frames: tuple[dict[str, Any], ...]
    checks: tuple[dict[str, Any], ...]
    repairs: tuple[str, ...]
    numeric_formatting: tuple[NumericFormatDecision, ...]
    contact_sheet_frame_indices: tuple[int, ...]
    contact_sheet_path: str
    quality_receipt_path: str
    encoding: AnimationEncodingReport
    source: dict[str, Any] = field(default_factory=dict)
    visual_qa_status: str = "pending_agent_review"
    visual_review_required: bool = True
    non_claim: str = "Automated animation checks do not replace agent inspection of the contact sheet and movie."

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "automated_status": self.automated_status,
            "camera_policy": self.camera_policy,
            "frames_expected": self.frames_expected,
            "frames_checked": self.frames_checked,
            "failed_frame_count": self.failed_frame_count,
            "failed_frames": [dict(item) for item in self.failed_frames],
            "checks": [dict(item) for item in self.checks],
            "failed_checks": [str(item["check_id"]) for item in self.checks if not bool(item["passed"])],
            "repairs": list(self.repairs),
            "numeric_formatting": [item.to_dict() for item in self.numeric_formatting],
            "contact_sheet_frame_indices": list(self.contact_sheet_frame_indices),
            "contact_sheet_path": self.contact_sheet_path,
            "quality_receipt_path": self.quality_receipt_path,
            "encoding": self.encoding.to_dict(),
            "source": dict(self.source),
            "visual_qa_status": self.visual_qa_status,
            "visual_review_required": self.visual_review_required,
            "non_claim": self.non_claim,
        }


def animation_time_decimal_places(times_s: Sequence[float], *, maximum: int = 3) -> int:
    """Choose one fixed time resolution for every frame annotation."""

    values = np.asarray(times_s, dtype=float).reshape(-1)
    finite = np.unique(values[np.isfinite(values)])
    if finite.size < 2:
        return 1
    steps = np.diff(finite)
    positive = steps[steps > 0.0]
    if positive.size == 0:
        return 1
    step = float(np.min(positive))
    for decimal_places in range(0, int(maximum) + 1):
        scaled = step * (10.0**decimal_places)
        if abs(scaled - round(scaled)) <= 1e-9 * max(abs(scaled), 1.0):
            return max(1, decimal_places)
    return int(maximum)


def format_animation_time(value_s: float, *, decimal_places: int, width: int | None = None) -> str:
    """Format time without negative zero and optionally reserve a fixed width."""

    value = float(value_s)
    zero_threshold = 0.5 * 10.0 ** (-max(int(decimal_places), 0))
    if abs(value) < zero_threshold:
        value = 0.0
    rendered = f"{value:.{max(int(decimal_places), 0)}f}"
    return rendered.rjust(int(width)) if width is not None else rendered


def fixed_time_text_width(times_s: Sequence[float], *, decimal_places: int) -> int:
    values = np.asarray(times_s, dtype=float).reshape(-1)
    rendered = [format_animation_time(value, decimal_places=decimal_places) for value in values if np.isfinite(value)]
    return max((len(item) for item in rendered), default=3)


def select_contact_sheet_frames(
    frame_count: int,
    *,
    key_frame_indices: Iterable[int] = (),
    maximum: int = 9,
) -> tuple[int, ...]:
    """Select deterministic first/final, stratified, and semantic key frames."""

    count = int(frame_count)
    if count <= 0:
        return ()
    cap = max(int(maximum), 2)
    selected = {0, count - 1}
    for value in key_frame_indices:
        index = int(value)
        if 0 <= index < count:
            selected.add(index)
    if len(selected) < cap:
        for value in np.linspace(0, count - 1, num=cap, endpoint=True):
            selected.add(int(round(float(value))))
    ordered = sorted(selected)
    if len(ordered) <= cap:
        return tuple(ordered)
    mandatory = {0, count - 1}
    mandatory.update(int(value) for value in key_frame_indices if 0 <= int(value) < count)
    if len(mandatory) >= cap:
        return tuple(sorted(mandatory)[: cap - 1] + [count - 1])
    remaining = [value for value in ordered if value not in mandatory]
    slots = cap - len(mandatory)
    if len(remaining) > slots:
        positions = np.linspace(0, len(remaining) - 1, num=slots, endpoint=True)
        remaining = [remaining[int(round(float(position)))] for position in positions]
    return tuple(sorted(mandatory.union(remaining)))


def save_animation_with_quality(
    animation_obj: Any,
    fig: Any,
    path: str | Path,
    *,
    update: Callable[[int], Any],
    frame_values: Sequence[int],
    frame_times_s: Sequence[float],
    fps: float,
    camera_policy: str,
    metadata: OELArtifactMetadata | None = None,
    artifact_id: str = "",
    style_name: str | None = None,
    format_limits: Mapping[tuple[int, str], tuple[float, float]] | None = None,
    key_frame_indices: Iterable[int] = (),
    source: Mapping[str, Any] | None = None,
    policy: AnimationQualityPolicy = STRICT_AGENT_ANIMATION_QUALITY,
    logical_contact_sheet_path: str | Path | None = None,
    logical_quality_receipt_path: str | Path | None = None,
) -> AnimationQualityReport:
    """Inspect, encode, verify, and document one saved animation."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = tuple(int(value) for value in frame_values)
    times = tuple(float(value) for value in frame_times_s)
    if not values:
        raise ValueError("Animation quality requires at least one frame.")
    if len(values) != len(times):
        raise ValueError("Animation frame values and frame times must have identical lengths.")
    if len(values) > policy.max_frames:
        raise ValueError(f"Animation has {len(values)} frames; the strict policy maximum is {policy.max_frames}.")
    fps_value = float(fps)
    if not np.isfinite(fps_value) or fps_value <= 0.0:
        raise ValueError("Animation fps must be positive and finite.")
    duration_s = len(values) / fps_value
    if duration_s > policy.max_duration_s + 1e-9:
        raise ValueError(
            f"Animation duration is {duration_s:.3f}s; the strict policy maximum is {policy.max_duration_s:.3f}s."
        )
    if camera_policy not in {"fixed", "fit_history", "follow"}:
        raise ValueError("camera_policy must be 'fixed', 'fit_history', or 'follow'.")
    finite_times = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(finite_times)):
        raise ValueError("Animation frame times must be finite.")

    prepare_oel_animation_figure(
        fig,
        metadata=metadata,
        artifact_id=artifact_id or target.stem,
        style_name=style_name,
    )
    update(values[0])
    numeric_formatting, setup_repairs = _prepare_stable_animation_layout(
        fig,
        update=update,
        first_frame_value=values[0],
        format_limits=format_limits or {},
    )

    scan = _scan_animation_frames(
        fig,
        update=update,
        frame_values=values,
        camera_policy=camera_policy,
        policy=policy,
        capture_indices=(),
    )
    repairs = list(setup_repairs)
    if scan["failed_frames"]:
        first_failed = int(scan["failed_frames"][0]["frame_ordinal"])
        update(values[first_failed])
        repair_report = apply_plot_quality_policy(
            fig,
            policy=STRICT_AGENT_PLOT_QUALITY,
            format_numeric_axes=False,
        )
        repairs.extend(item for item in repair_report.repairs if item not in repairs)
        scan = _scan_animation_frames(
            fig,
            update=update,
            frame_values=values,
            camera_policy=camera_policy,
            policy=policy,
            capture_indices=(),
        )

    contact_indices = select_contact_sheet_frames(
        len(values),
        key_frame_indices=key_frame_indices,
        maximum=policy.contact_sheet_frames,
    )
    final_scan = _scan_animation_frames(
        fig,
        update=update,
        frame_values=values,
        camera_policy=camera_policy,
        policy=policy,
        capture_indices=contact_indices,
    )
    contact_path = target.with_suffix(".contact-sheet.png")
    _write_contact_sheet(
        final_scan["captures"],
        contact_path,
        frame_times_s=times,
        thumbnail_width_px=policy.contact_sheet_thumbnail_width_px,
    )

    encoding_error = ""
    try:
        _encode_animation(animation_obj, target, fps=fps_value)
    except Exception as exc:  # pragma: no cover - exercised through failure injection
        encoding_error = f"{type(exc).__name__}: {exc}"
    encoding = verify_animation_encoding(
        target,
        expected_frames=len(values),
        expected_fps=fps_value,
        policy=policy,
        encoding_error=encoding_error,
    )

    checks = _animation_checks(
        frame_times_s=times,
        frame_count=len(values),
        camera_policy=camera_policy,
        scan=final_scan,
        encoding=encoding,
        expected_fps=fps_value,
        contact_path=contact_path,
        policy=policy,
    )
    status = "passed" if all(bool(item["passed"]) for item in checks) else "failed"
    receipt_path = target.with_suffix(".quality.json")
    report_contact_path = (
        Path(logical_contact_sheet_path)
        if logical_contact_sheet_path is not None
        else contact_path
    )
    report_receipt_path = (
        Path(logical_quality_receipt_path)
        if logical_quality_receipt_path is not None
        else receipt_path
    )
    report = AnimationQualityReport(
        policy_id=policy.policy_id,
        policy_version=policy.version,
        automated_status=status,
        camera_policy=camera_policy,
        frames_expected=len(values),
        frames_checked=int(final_scan["frames_checked"]),
        failed_frame_count=len(final_scan["failed_frames"]),
        failed_frames=tuple(final_scan["failed_frames"][: policy.max_reported_failed_frames]),
        checks=tuple(checks),
        repairs=tuple(dict.fromkeys(repairs)),
        numeric_formatting=tuple(numeric_formatting),
        contact_sheet_frame_indices=contact_indices,
        contact_sheet_path=str(report_contact_path),
        quality_receipt_path=str(report_receipt_path),
        encoding=encoding,
        source=dict(source or {}),
    )
    receipt_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def verify_animation_encoding(
    path: str | Path,
    *,
    expected_frames: int,
    expected_fps: float,
    policy: AnimationQualityPolicy = STRICT_AGENT_ANIMATION_QUALITY,
    encoding_error: str = "",
) -> AnimationEncodingReport:
    """Decode every encoded frame and verify the expected movie is present."""

    target = Path(path)
    exists = target.is_file()
    size_bytes = int(target.stat().st_size) if exists else 0
    digest = _sha256_file(target) if exists else ""
    suffix = target.suffix.lower().lstrip(".")
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    actual_fps: float | None = None
    duration: float | None = None
    decode_ok = False
    error = encoding_error
    if exists and not error:
        try:
            expected_frame_count = max(int(expected_frames), 1)
            if suffix == "gif":
                from PIL import Image

                with Image.open(target) as image:
                    width, height = (int(value) for value in image.size)
                    frame_count = int(getattr(image, "n_frames", 1))
                    frame_durations_ms: list[float] = []
                    for frame_index in range(frame_count):
                        image.seek(frame_index)
                        decoded = image.convert("RGB")
                        if decoded.size != (width, height):
                            raise ValueError(
                                f"GIF frame {frame_index} has dimensions {decoded.size}, "
                                f"expected {(width, height)}."
                            )
                        frame_durations_ms.append(float(image.info.get("duration", 1000.0 / expected_fps)))
                    total_duration_s = sum(frame_durations_ms) / 1000.0
                    actual_fps = frame_count / total_duration_s if total_duration_s > 0.0 else None
                    duration = total_duration_s if total_duration_s > 0.0 else None
            elif suffix == "mp4":
                import imageio.v2 as imageio

                reader = imageio.get_reader(str(target))
                try:
                    metadata = dict(reader.get_meta_data() or {})
                    first = np.asarray(reader.get_data(0))
                    height, width = (int(first.shape[0]), int(first.shape[1]))
                    frame_count = int(reader.count_frames())
                    for frame_index in range(frame_count):
                        frame = np.asarray(reader.get_data(frame_index))
                        if frame.ndim < 2 or tuple(frame.shape[:2]) != (height, width):
                            raise ValueError(
                                f"MP4 frame {frame_index} has dimensions {tuple(frame.shape[:2])}, "
                                f"expected {(height, width)}."
                            )
                    actual_fps = float(metadata.get("fps", expected_fps))
                    duration = float(metadata.get("duration", frame_count / actual_fps))
                finally:
                    reader.close()
            else:
                raise ValueError("Strict animation quality supports only GIF and MP4 artifacts.")
            decode_ok = bool(
                width
                and height
                and frame_count
                and frame_count >= expected_frame_count
            )
        except Exception as exc:  # pragma: no cover - backend failures vary by platform
            error = f"{type(exc).__name__}: {exc}"
    return AnimationEncodingReport(
        format=suffix,
        exists=exists,
        size_bytes=size_bytes,
        sha256=digest,
        width_px=width,
        height_px=height,
        frame_count=frame_count,
        fps=actual_fps,
        duration_s=duration,
        decode_ok=decode_ok,
        error=error,
    )


def _prepare_stable_animation_layout(
    fig: Any,
    *,
    update: Callable[[int], Any],
    first_frame_value: int,
    format_limits: Mapping[tuple[int, str], tuple[float, float]],
) -> tuple[tuple[NumericFormatDecision, ...], tuple[str, ...]]:
    update(first_frame_value)
    for (axes_index, axis_name), limits in sorted(format_limits.items()):
        axes = fig.get_axes()
        if not 0 <= int(axes_index) < len(axes):
            raise ValueError(f"Unknown animation axes index {axes_index} in format_limits.")
        lower, upper = (float(value) for value in limits)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("Animation format limits must be finite increasing pairs.")
        if axis_name == "x":
            axes[int(axes_index)].set_xlim(lower, upper)
        elif axis_name == "y":
            axes[int(axes_index)].set_ylim(lower, upper)
        else:
            raise ValueError("Animation format_limits axis names must be 'x' or 'y'.")
    report = apply_plot_quality_policy(fig, policy=STRICT_AGENT_PLOT_QUALITY)
    update(first_frame_value)
    fig.canvas.draw()
    return report.numeric_formatting, report.repairs


def _scan_animation_frames(
    fig: Any,
    *,
    update: Callable[[int], Any],
    frame_values: Sequence[int],
    camera_policy: str,
    policy: AnimationQualityPolicy,
    capture_indices: Iterable[int],
) -> dict[str, Any]:
    capture_set = {int(value) for value in capture_indices}
    captures: dict[int, Any] = {}
    failed_frames: list[dict[str, Any]] = []
    formatter_signatures: list[tuple[Any, ...]] = []
    camera_states: list[tuple[tuple[float, float, float, float], ...]] = []
    for ordinal, frame_value in enumerate(frame_values):
        update(int(frame_value))
        fig.canvas.draw()
        issues = assess_plot_quality(fig, policy=STRICT_AGENT_PLOT_QUALITY)
        formatter_signatures.append(_formatter_signature(fig))
        camera_states.append(_camera_state(fig))
        if issues:
            failed_frames.append(
                {
                    "frame_ordinal": ordinal,
                    "frame_value": int(frame_value),
                    "issues": [_issue_payload(issue) for issue in issues],
                }
            )
        if ordinal in capture_set:
            captures[ordinal] = _capture_canvas(fig)
    format_stable = len(set(formatter_signatures)) <= 1
    camera_stable = _camera_is_stable(
        camera_states,
        camera_policy=camera_policy,
        tolerance=policy.camera_span_relative_tolerance,
    )
    return {
        "frames_checked": len(frame_values),
        "failed_frames": failed_frames,
        "format_stable": format_stable,
        "camera_stable": camera_stable,
        "captures": captures,
    }


def _formatter_signature(fig: Any) -> tuple[Any, ...]:
    out: list[Any] = []
    for axes_index, ax in enumerate(fig.get_axes()):
        for axis_name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            formatter = axis.get_major_formatter()
            out.append(
                (
                    axes_index,
                    axis_name,
                    type(formatter).__module__,
                    type(formatter).__qualname__,
                    getattr(formatter, "decimal_places", None),
                    getattr(formatter, "scale_exponent", None),
                )
            )
    return tuple(out)


def _camera_state(fig: Any) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        (
            float(ax.get_xlim()[0]),
            float(ax.get_xlim()[1]),
            float(ax.get_ylim()[0]),
            float(ax.get_ylim()[1]),
        )
        for ax in fig.get_axes()
        if str(getattr(ax, "name", "") or "").lower() != "3d"
    )


def _camera_is_stable(
    states: Sequence[tuple[tuple[float, float, float, float], ...]],
    *,
    camera_policy: str,
    tolerance: float,
) -> bool:
    if len(states) < 2:
        return True
    reference = states[0]
    for state in states[1:]:
        if len(state) != len(reference):
            return False
        for current, initial in zip(state, reference):
            if camera_policy in {"fixed", "fit_history"}:
                comparisons = zip(current, initial)
            else:
                comparisons = (
                    (current[1] - current[0], initial[1] - initial[0]),
                    (current[3] - current[2], initial[3] - initial[2]),
                )
            for value, expected in comparisons:
                scale = max(abs(expected), 1.0)
                if abs(value - expected) > tolerance * scale:
                    return False
    return True


def _animation_checks(
    *,
    frame_times_s: Sequence[float],
    frame_count: int,
    camera_policy: str,
    scan: Mapping[str, Any],
    encoding: AnimationEncodingReport,
    expected_fps: float,
    contact_path: Path,
    policy: AnimationQualityPolicy,
) -> list[dict[str, Any]]:
    times = np.asarray(frame_times_s, dtype=float)
    monotonic = bool(times.size <= 1 or np.all(np.diff(times) >= 0.0))
    duration = float(encoding.duration_s or (frame_count / max(float(encoding.fps or 1.0), 1e-9)))
    fps_tolerance = max(0.1, 0.05 * float(expected_fps))
    return [
        {"check_id": "frame_count_within_policy", "passed": frame_count <= policy.max_frames, "value": frame_count},
        {"check_id": "frame_times_monotonic", "passed": monotonic, "value": monotonic},
        {
            "check_id": "frame_presentation",
            "passed": not bool(scan["failed_frames"]),
            "value": {"failed_frame_count": len(scan["failed_frames"]), "frames_checked": scan["frames_checked"]},
        },
        {"check_id": "numeric_format_stable", "passed": bool(scan["format_stable"]), "value": bool(scan["format_stable"])},
        {
            "check_id": "camera_policy_stable",
            "passed": bool(scan["camera_stable"]),
            "value": {"camera_policy": camera_policy, "stable": bool(scan["camera_stable"])},
        },
        {"check_id": "contact_sheet_exists", "passed": contact_path.is_file(), "value": str(contact_path)},
        {
            "check_id": "artifact_size",
            "passed": policy.minimum_file_bytes <= encoding.size_bytes <= policy.max_file_bytes,
            "value": encoding.size_bytes,
        },
        {"check_id": "artifact_decodes", "passed": encoding.decode_ok, "value": encoding.error or True},
        {
            "check_id": "encoded_frame_count",
            "passed": encoding.frame_count == frame_count,
            "value": {"expected": frame_count, "actual": encoding.frame_count},
        },
        {
            "check_id": "encoded_fps",
            "passed": encoding.fps is not None and abs(float(encoding.fps) - float(expected_fps)) <= fps_tolerance,
            "value": {"expected": expected_fps, "actual": encoding.fps, "tolerance": fps_tolerance},
        },
        {
            "check_id": "encoded_duration",
            "passed": duration <= policy.max_duration_s + 1e-6,
            "value": duration,
        },
    ]


def _write_contact_sheet(
    captures: Mapping[int, Any],
    path: Path,
    *,
    frame_times_s: Sequence[float],
    thumbnail_width_px: int,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    if not captures:
        raise ValueError("Animation contact sheet requires at least one captured frame.")
    tiles: list[Any] = []
    for ordinal, image in sorted(captures.items()):
        tile = image.convert("RGB")
        ratio = min(1.0, float(thumbnail_width_px) / max(tile.width, 1))
        tile = tile.resize(
            (max(1, int(round(tile.width * ratio))), max(1, int(round(tile.height * ratio)))),
            Image.Resampling.LANCZOS,
        )
        label_height = 24
        labeled = Image.new("RGB", (tile.width, tile.height + label_height), "white")
        labeled.paste(tile, (0, label_height))
        draw = ImageDraw.Draw(labeled)
        time_value = float(frame_times_s[min(ordinal, len(frame_times_s) - 1)])
        draw.text((6, 5), f"frame {ordinal}   t={time_value:g} s", fill="#111827")
        tiles.append(ImageOps.expand(labeled, border=1, fill="#94A3B8"))
    columns = min(3, len(tiles))
    rows = int(math.ceil(len(tiles) / columns))
    cell_width = max(tile.width for tile in tiles)
    cell_height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#E5E7EB")
    for index, tile in enumerate(tiles):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(tile, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")


def _capture_canvas(fig: Any) -> Any:
    from PIL import Image

    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    return Image.fromarray(rgba, mode="RGBA")


def _encode_animation(animation_obj: Any, path: Path, *, fps: float) -> None:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        from matplotlib.animation import PillowWriter

        animation_obj.save(str(path), writer=PillowWriter(fps=fps))
        return
    if suffix == ".mp4":
        import imageio_ffmpeg
        import matplotlib
        from matplotlib.animation import FFMpegWriter

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        animation_obj.save(str(path), writer=writer)
        return
    raise ValueError("Strict animation quality supports only GIF and MP4 artifacts.")


def _issue_payload(issue: PlotQualityIssue) -> dict[str, Any]:
    return issue.to_dict()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ANIMATION_QUALITY_POLICY_VERSION",
    "STRICT_AGENT_ANIMATION_QUALITY",
    "AnimationEncodingReport",
    "AnimationQualityPolicy",
    "AnimationQualityReport",
    "animation_time_decimal_places",
    "fixed_time_text_width",
    "format_animation_time",
    "save_animation_with_quality",
    "select_contact_sheet_frames",
    "verify_animation_encoding",
]
