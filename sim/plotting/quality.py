"""Deterministic presentation-quality rules for OEL plot artifacts.

This module owns display formatting and renderer-level presentation checks.  It
does not modify source evidence, simulation results, or plotted data values.
Callers should apply the policy after all titles, labels, legends, annotations,
and artifact footers have been added to a figure.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

from matplotlib.ticker import Formatter

PLOT_QUALITY_POLICY_VERSION = 1
AxisName = Literal["x", "y"]


@dataclass(frozen=True)
class PlotQualityPolicy:
    """Versioned, deterministic presentation rules for one rendered figure."""

    policy_id: str = "oel.agent_strict"
    version: int = PLOT_QUALITY_POLICY_VERSION
    target_major_ticks: int = 6
    minimum_major_ticks: int = 3
    scientific_lower_exponent: int = -3
    scientific_upper_exponent: int = 5
    max_fixed_decimals: int = 6
    minimum_font_size_pt: float = 7.5
    minimum_footer_font_size_pt: float = 6.0
    collision_padding_px: float = 1.0
    canvas_margin_px: float = 1.0
    categorical_rotation_deg: float = 35.0
    categorical_max_rotation_deg: float = 65.0
    max_inside_legend_entries: int = 4
    auto_repair: bool = True

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("Plot quality policy version must be positive.")
        if self.target_major_ticks < self.minimum_major_ticks:
            raise ValueError("target_major_ticks must be greater than or equal to minimum_major_ticks.")
        if self.minimum_major_ticks < 2:
            raise ValueError("minimum_major_ticks must be at least two.")
        if self.scientific_lower_exponent >= self.scientific_upper_exponent:
            raise ValueError("scientific exponent bounds must be strictly increasing.")
        if self.max_fixed_decimals < 0:
            raise ValueError("max_fixed_decimals cannot be negative.")
        if self.minimum_font_size_pt <= 0.0:
            raise ValueError("minimum_font_size_pt must be positive.")
        if not 0.0 < self.minimum_footer_font_size_pt <= self.minimum_font_size_pt:
            raise ValueError("minimum_footer_font_size_pt must be positive and no larger than the text minimum.")
        if not 0.0 <= self.categorical_rotation_deg <= self.categorical_max_rotation_deg <= 90.0:
            raise ValueError("categorical tick rotations must satisfy 0 <= initial <= maximum <= 90 degrees.")


STRICT_AGENT_PLOT_QUALITY = PlotQualityPolicy()


@dataclass(frozen=True)
class NumericFormatDecision:
    """The single display decision applied to every major tick on one axis."""

    axes_index: int
    axis_name: AxisName
    tick_step: float
    decimal_places: int
    scale_exponent: int
    formatter_kind: str
    visible_min: float
    visible_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlotQualityIssue:
    """One deterministic presentation defect found in display coordinates."""

    check_id: str
    severity: str
    message: str
    artists: tuple[str, ...] = ()
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlotQualityReport:
    """Serializable quality evidence for a completed renderer pass."""

    policy_id: str
    policy_version: int
    automated_status: str
    issues: tuple[PlotQualityIssue, ...] = ()
    repairs: tuple[str, ...] = ()
    numeric_formatting: tuple[NumericFormatDecision, ...] = ()
    checks_performed: tuple[str, ...] = (
        "minimum_font_size",
        "text_inside_canvas",
        "text_overlap",
        "ambiguous_numeric_tick_labels",
        "legend_data_obstruction",
        "figure_legend_axes_obstruction",
    )
    visual_qa_status: str = "pending_agent_review"
    visual_review_required: bool = True
    non_claim: str = "Automated presentation checks do not replace agent visual inspection."

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "automated_status": self.automated_status,
            "issues": [issue.to_dict() for issue in self.issues],
            "failed_checks": sorted({issue.check_id for issue in self.issues if issue.severity == "error"}),
            "repairs": list(self.repairs),
            "numeric_formatting": [decision.to_dict() for decision in self.numeric_formatting],
            "checks_performed": list(self.checks_performed),
            "visual_qa_status": self.visual_qa_status,
            "visual_review_required": self.visual_review_required,
            "non_claim": self.non_claim,
        }


class StableEngineeringFormatter(Formatter):
    """Matplotlib-compatible formatter with one shared engineering exponent."""

    def __init__(self, *, decimal_places: int, scale_exponent: int) -> None:
        super().__init__()
        self.decimal_places = max(int(decimal_places), 0)
        self.scale_exponent = int(scale_exponent)

    def __call__(self, value: float, _position: int | None = None) -> str:
        scale = 10.0**self.scale_exponent
        scaled = float(value) / scale
        zero_threshold = 0.5 * 10.0 ** (-self.decimal_places)
        if abs(scaled) < zero_threshold:
            scaled = 0.0
        return f"{scaled:.{self.decimal_places}f}"

    def get_offset(self) -> str:
        if self.scale_exponent == 0:
            return ""
        return rf"$\times 10^{{{self.scale_exponent}}}$"

    def set_locs(self, _locations: Iterable[float]) -> None:
        """Satisfy Matplotlib's Formatter protocol without location-dependent output."""


@dataclass
class _TextBox:
    key: str
    kind: str
    artist: Any
    bbox: Any


def apply_stable_axis_format(
    ax: Any,
    axis_name: AxisName,
    *,
    axes_index: int = 0,
    policy: PlotQualityPolicy = STRICT_AGENT_PLOT_QUALITY,
) -> NumericFormatDecision | None:
    """Apply a stable axis-wide formatter, skipping categorical and nonlinear axes."""

    from matplotlib.ticker import MaxNLocator

    axis = ax.xaxis if axis_name == "x" else ax.yaxis
    if not bool(axis.get_visible()):
        return None
    if str(getattr(ax, "name", "") or "").lower() == "3d":
        return None
    scale_name = ax.get_xscale() if axis_name == "x" else ax.get_yscale()
    get_converter = getattr(axis, "get_converter", None)
    converter = get_converter() if callable(get_converter) else getattr(axis, "converter", None)
    if scale_name != "linear" or converter is not None:
        return None
    visible_min, visible_max = ax.get_xlim() if axis_name == "x" else ax.get_ylim()
    visible_min = float(visible_min)
    visible_max = float(visible_max)
    if not (math.isfinite(visible_min) and math.isfinite(visible_max)) or visible_min == visible_max:
        return None

    locator = MaxNLocator(
        nbins=int(policy.target_major_ticks),
        steps=[1.0, 2.0, 2.5, 5.0, 10.0],
        min_n_ticks=int(policy.minimum_major_ticks),
    )
    axis.set_major_locator(locator)
    ticks = [
        float(value)
        for value in locator.tick_values(visible_min, visible_max)
        if math.isfinite(float(value))
    ]
    tick_step = _minimum_positive_step(ticks)
    max_abs = max(abs(visible_min), abs(visible_max))
    scale_exponent = _shared_engineering_exponent(max_abs, tick_step, policy=policy)
    scaled_step = tick_step / (10.0**scale_exponent)
    decimal_places = _decimal_places_for_step(scaled_step, maximum=policy.max_fixed_decimals)
    formatter_kind = "shared_engineering" if scale_exponent else "fixed_resolution"
    axis.set_major_formatter(
        StableEngineeringFormatter(decimal_places=decimal_places, scale_exponent=scale_exponent)
    )
    return NumericFormatDecision(
        axes_index=int(axes_index),
        axis_name=axis_name,
        tick_step=float(tick_step),
        decimal_places=int(decimal_places),
        scale_exponent=int(scale_exponent),
        formatter_kind=formatter_kind,
        visible_min=visible_min,
        visible_max=visible_max,
    )


def apply_plot_quality_policy(
    fig: Any,
    *,
    policy: PlotQualityPolicy = STRICT_AGENT_PLOT_QUALITY,
    format_numeric_axes: bool = True,
) -> PlotQualityReport:
    """Format, repair, and assess a complete figure without changing plotted data."""

    # Aspect constraints, shared axes, and custom projections can change the
    # visible limits during the first renderer pass. Resolve those constraints
    # before deriving one formatter from the final axis interval.
    _draw_figure(fig)
    decisions: list[NumericFormatDecision] = []
    if format_numeric_axes:
        for axes_index, ax in enumerate(fig.get_axes()):
            for axis_name in ("x", "y"):
                decision = apply_stable_axis_format(
                    ax,
                    axis_name,
                    axes_index=axes_index,
                    policy=policy,
                )
                if decision is not None:
                    decisions.append(decision)

    repairs: list[str] = []
    _draw_figure(fig)
    if policy.auto_repair:
        repairs.extend(_repair_tick_collisions(fig, policy=policy))
        repairs.extend(_repair_legend_obstruction(fig, policy=policy))
        if repairs:
            _apply_bounded_layout(fig, reserve_right=any("legend" in item for item in repairs))
            _draw_figure(fig)
            repairs.extend(item for item in _repair_tick_collisions(fig, policy=policy) if item not in repairs)
            _draw_figure(fig)

    issues = assess_plot_quality(fig, policy=policy)
    status = "failed" if any(issue.severity == "error" for issue in issues) else "passed"
    return PlotQualityReport(
        policy_id=policy.policy_id,
        policy_version=policy.version,
        automated_status=status,
        issues=tuple(issues),
        repairs=tuple(repairs),
        numeric_formatting=tuple(decisions),
    )


def assess_plot_quality(
    fig: Any,
    *,
    policy: PlotQualityPolicy = STRICT_AGENT_PLOT_QUALITY,
) -> list[PlotQualityIssue]:
    """Inspect rendered artist geometry without mutating the figure."""

    _draw_figure(fig)
    renderer = fig.canvas.get_renderer()
    text_boxes = _collect_text_boxes(fig, renderer)
    issues: list[PlotQualityIssue] = []
    issues.extend(_font_size_issues(text_boxes, policy=policy))
    issues.extend(_canvas_issues(fig, text_boxes, policy=policy))
    issues.extend(_text_overlap_issues(text_boxes, policy=policy))
    issues.extend(_ambiguous_numeric_tick_label_issues(fig))
    issues.extend(_legend_obstruction_issues(fig, renderer, policy=policy))
    issues.extend(_figure_legend_obstruction_issues(fig, renderer, policy=policy))
    return _deduplicate_issues(issues)


def _minimum_positive_step(values: list[float]) -> float:
    diffs = [right - left for left, right in zip(values, values[1:]) if right > left]
    return min(diffs) if diffs else 1.0


def _shared_engineering_exponent(
    max_abs: float,
    tick_step: float,
    *,
    policy: PlotQualityPolicy,
) -> int:
    reference = max(float(max_abs), float(tick_step), 0.0)
    if reference == 0.0 or not math.isfinite(reference):
        return 0
    order = int(math.floor(math.log10(reference)))
    if policy.scientific_lower_exponent <= order < policy.scientific_upper_exponent:
        return 0
    # Use an engineering exponent with a readable coefficient range of roughly
    # 0.1 to 100. This avoids flipping 0.8 micro-units into 800 nano-units while
    # remaining stable throughout the same order of magnitude.
    return int(3 * math.floor((order + 1) / 3))


def _decimal_places_for_step(step: float, *, maximum: int) -> int:
    value = abs(float(step))
    if value == 0.0 or not math.isfinite(value):
        return 0
    for decimal_places in range(maximum + 1):
        scaled = value * (10.0**decimal_places)
        if math.isclose(scaled, round(scaled), rel_tol=1.0e-10, abs_tol=1.0e-10):
            return decimal_places
    return int(maximum)


def _draw_figure(fig: Any) -> None:
    canvas = getattr(fig, "canvas", None)
    if canvas is None or not callable(getattr(canvas, "draw", None)):
        raise TypeError("Plot quality checks require a Matplotlib figure with a drawable canvas.")
    canvas.draw()


def _collect_text_boxes(fig: Any, renderer: Any) -> list[_TextBox]:
    boxes: list[_TextBox] = []
    seen: set[int] = set()

    def add(artist: Any, *, key: str, kind: str) -> None:
        if artist is None or id(artist) in seen or not bool(artist.get_visible()):
            return
        text = str(artist.get_text() or "").strip()
        if not text:
            return
        bbox = artist.get_window_extent(renderer=renderer)
        if float(bbox.width) <= 0.0 or float(bbox.height) <= 0.0:
            return
        seen.add(id(artist))
        boxes.append(_TextBox(key=key, kind=kind, artist=artist, bbox=bbox))

    for axes_index, ax in enumerate(fig.get_axes()):
        add(ax.title, key=f"axes[{axes_index}].title", kind="title")
        add(getattr(ax, "_left_title", None), key=f"axes[{axes_index}].left_title", kind="title")
        add(getattr(ax, "_right_title", None), key=f"axes[{axes_index}].right_title", kind="title")
        if ax.xaxis.get_visible():
            add(ax.xaxis.label, key=f"axes[{axes_index}].x_label", kind="axis_label")
            add(ax.xaxis.get_offset_text(), key=f"axes[{axes_index}].x_offset", kind="axis_offset")
        if ax.yaxis.get_visible():
            add(ax.yaxis.label, key=f"axes[{axes_index}].y_label", kind="axis_label")
            add(ax.yaxis.get_offset_text(), key=f"axes[{axes_index}].y_offset", kind="axis_offset")
        if str(getattr(ax, "name", "") or "").lower() != "3d":
            x_limits = tuple(float(value) for value in ax.get_xlim())
            y_limits = tuple(float(value) for value in ax.get_ylim())
            if ax.xaxis.get_visible():
                for index, (value, label) in enumerate(zip(ax.get_xticks(), ax.get_xticklabels())):
                    if _tick_is_in_view(float(value), x_limits):
                        add(label, key=f"axes[{axes_index}].x_tick[{index}]", kind="x_tick")
            if ax.yaxis.get_visible():
                for index, (value, label) in enumerate(zip(ax.get_yticks(), ax.get_yticklabels())):
                    if _tick_is_in_view(float(value), y_limits):
                        add(label, key=f"axes[{axes_index}].y_tick[{index}]", kind="y_tick")
        for index, annotation in enumerate(ax.texts):
            add(annotation, key=f"axes[{axes_index}].annotation[{index}]", kind="annotation")
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            add(legend.get_title(), key=f"axes[{axes_index}].legend_title", kind="legend")
            for index, label in enumerate(legend.get_texts()):
                add(label, key=f"axes[{axes_index}].legend_text[{index}]", kind="legend")

    for index, text in enumerate(getattr(fig, "texts", []) or []):
        gid = str(getattr(text, "get_gid", lambda: "")() or "")
        kind = "footer" if gid == "oel_artifact_footer" else "figure_text"
        add(text, key=f"figure.{kind}[{index}]", kind=kind)
    for legend_index, legend in enumerate(getattr(fig, "legends", []) or []):
        if not legend.get_visible():
            continue
        add(legend.get_title(), key=f"figure.legend[{legend_index}].title", kind="legend")
        for text_index, label in enumerate(legend.get_texts()):
            add(
                label,
                key=f"figure.legend[{legend_index}].text[{text_index}]",
                kind="legend",
            )
    return boxes


def _tick_is_in_view(value: float, limits: tuple[float, float]) -> bool:
    lower, upper = sorted(limits)
    tolerance = max(abs(lower), abs(upper), upper - lower, 1.0e-300) * 1.0e-12
    return lower - tolerance <= value <= upper + tolerance


def _font_size_issues(
    boxes: list[_TextBox],
    *,
    policy: PlotQualityPolicy,
) -> list[PlotQualityIssue]:
    issues: list[PlotQualityIssue] = []
    for item in boxes:
        size = float(item.artist.get_fontsize())
        minimum = policy.minimum_footer_font_size_pt if item.kind == "footer" else policy.minimum_font_size_pt
        if size + 1.0e-9 < minimum:
            issues.append(
                PlotQualityIssue(
                    check_id="minimum_font_size",
                    severity="error",
                    message=(
                        f"{item.key} uses {size:.2f} pt text; the policy minimum is "
                        f"{minimum:.2f} pt."
                    ),
                    artists=(item.key,),
                    value={"font_size_pt": size, "minimum_pt": minimum},
                )
            )
    return issues


def _canvas_issues(
    fig: Any,
    boxes: list[_TextBox],
    *,
    policy: PlotQualityPolicy,
) -> list[PlotQualityIssue]:
    width, height = fig.canvas.get_width_height()
    margin = float(policy.canvas_margin_px)
    issues: list[PlotQualityIssue] = []
    for item in boxes:
        bbox = item.bbox
        outside = (
            float(bbox.x0) < margin
            or float(bbox.y0) < margin
            or float(bbox.x1) > float(width) - margin
            or float(bbox.y1) > float(height) - margin
        )
        if outside:
            issues.append(
                PlotQualityIssue(
                    check_id="text_inside_canvas",
                    severity="error",
                    message=f"{item.key} extends outside the figure canvas.",
                    artists=(item.key,),
                    value={
                        "bbox_px": [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)],
                        "canvas_px": [int(width), int(height)],
                    },
                )
            )
    return issues


def _text_overlap_issues(
    boxes: list[_TextBox],
    *,
    policy: PlotQualityPolicy,
) -> list[PlotQualityIssue]:
    issues: list[PlotQualityIssue] = []
    padding = float(policy.collision_padding_px)
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if _allowed_text_pair(left, right):
                continue
            width = min(float(left.bbox.x1), float(right.bbox.x1)) - max(
                float(left.bbox.x0), float(right.bbox.x0)
            )
            height = min(float(left.bbox.y1), float(right.bbox.y1)) - max(
                float(left.bbox.y0), float(right.bbox.y0)
            )
            if width <= padding or height <= padding:
                continue
            issues.append(
                PlotQualityIssue(
                    check_id="text_overlap",
                    severity="error",
                    message=f"{left.key} overlaps {right.key}.",
                    artists=(left.key, right.key),
                    value={"overlap_width_px": width, "overlap_height_px": height},
                )
            )
    return issues


def _ambiguous_numeric_tick_label_issues(fig: Any) -> list[PlotQualityIssue]:
    issues: list[PlotQualityIssue] = []
    for axes_index, ax in enumerate(fig.get_axes()):
        if str(getattr(ax, "name", "") or "").lower() == "3d":
            continue
        for axis_name in ("x", "y"):
            axis = ax.xaxis if axis_name == "x" else ax.yaxis
            if not axis.get_visible():
                continue
            get_converter = getattr(axis, "get_converter", None)
            converter = get_converter() if callable(get_converter) else getattr(axis, "converter", None)
            scale_name = ax.get_xscale() if axis_name == "x" else ax.get_yscale()
            if converter is not None or scale_name != "linear":
                continue
            values = ax.get_xticks() if axis_name == "x" else ax.get_yticks()
            labels = ax.get_xticklabels() if axis_name == "x" else ax.get_yticklabels()
            limits = tuple(float(value) for value in (ax.get_xlim() if axis_name == "x" else ax.get_ylim()))
            by_label: dict[str, list[float]] = {}
            for value, label in zip(values, labels):
                numeric_value = float(value)
                text = str(label.get_text() or "").strip()
                if not text or not _tick_is_in_view(numeric_value, limits):
                    continue
                by_label.setdefault(text, []).append(numeric_value)
            duplicates = {label: values for label, values in by_label.items() if len(values) > 1}
            if not duplicates:
                continue
            issues.append(
                PlotQualityIssue(
                    check_id="ambiguous_numeric_tick_labels",
                    severity="error",
                    message=f"axes[{axes_index}] {axis_name} axis maps distinct ticks to duplicate labels.",
                    artists=(f"axes[{axes_index}].{axis_name}_axis",),
                    value=duplicates,
                )
            )
    return issues


def _allowed_text_pair(left: _TextBox, right: _TextBox) -> bool:
    # A legend deliberately contains its own labels; their boxes are assessed
    # against one another, but not against axes text behind the opaque legend.
    if (left.kind == "legend") != (right.kind == "legend"):
        return True
    return False


def _legend_obstruction_issues(
    fig: Any,
    renderer: Any,
    *,
    policy: PlotQualityPolicy,
) -> list[PlotQualityIssue]:
    issues: list[PlotQualityIssue] = []
    for axes_index, ax in enumerate(fig.get_axes()):
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        bbox = legend.get_window_extent(renderer=renderer)
        covered = _data_vertices_inside_legend(ax, bbox)
        entry_count = len(legend.get_texts())
        if covered <= 0 and entry_count <= policy.max_inside_legend_entries:
            continue
        reason = "covers plotted data" if covered > 0 else "contains too many entries for an inside legend"
        issues.append(
            PlotQualityIssue(
                check_id="legend_data_obstruction",
                severity="error",
                message=f"axes[{axes_index}] legend {reason}.",
                artists=(f"axes[{axes_index}].legend",),
                value={"covered_vertices": covered, "entry_count": entry_count},
            )
        )
    return issues


def _data_vertices_inside_legend(ax: Any, legend_bbox: Any) -> int:
    covered = 0
    for line in ax.get_lines():
        if not line.get_visible() or str(line.get_label() or "").startswith("_"):
            continue
        try:
            vertices = line.get_path().transformed(line.get_transform()).vertices
        except Exception:
            continue
        for x_value, y_value in vertices:
            if legend_bbox.contains(float(x_value), float(y_value)):
                covered += 1
    for collection in ax.collections:
        if not collection.get_visible():
            continue
        try:
            offsets = collection.get_offset_transform().transform(collection.get_offsets())
        except Exception:
            continue
        for x_value, y_value in offsets:
            if legend_bbox.contains(float(x_value), float(y_value)):
                covered += 1
    return covered


def _figure_legend_obstruction_issues(
    fig: Any,
    renderer: Any,
    *,
    policy: PlotQualityPolicy,
) -> list[PlotQualityIssue]:
    issues: list[PlotQualityIssue] = []
    padding = float(policy.collision_padding_px)
    for legend_index, legend in enumerate(getattr(fig, "legends", []) or []):
        if not legend.get_visible():
            continue
        legend_bbox = legend.get_window_extent(renderer=renderer)
        for axes_index, ax in enumerate(fig.get_axes()):
            axes_bbox = ax.get_window_extent(renderer=renderer)
            width = min(float(legend_bbox.x1), float(axes_bbox.x1)) - max(
                float(legend_bbox.x0), float(axes_bbox.x0)
            )
            height = min(float(legend_bbox.y1), float(axes_bbox.y1)) - max(
                float(legend_bbox.y0), float(axes_bbox.y0)
            )
            if width <= padding or height <= padding:
                continue
            issues.append(
                PlotQualityIssue(
                    check_id="figure_legend_axes_obstruction",
                    severity="error",
                    message=f"figure legend {legend_index} overlaps axes[{axes_index}].",
                    artists=(f"figure.legend[{legend_index}]", f"axes[{axes_index}]"),
                    value={"overlap_width_px": width, "overlap_height_px": height},
                )
            )
    return issues


def _repair_tick_collisions(fig: Any, *, policy: PlotQualityPolicy) -> list[str]:
    repairs: list[str] = []
    renderer = fig.canvas.get_renderer()
    for axes_index, ax in enumerate(fig.get_axes()):
        if str(getattr(ax, "name", "") or "").lower() == "3d":
            continue
        for axis_name in ("x", "y"):
            axis = ax.xaxis if axis_name == "x" else ax.yaxis
            if not axis.get_visible():
                continue
            labels = _visible_tick_labels(ax, axis_name)
            if not _adjacent_artists_overlap(
                labels,
                renderer,
                axis_name=axis_name,
                padding=policy.collision_padding_px,
            ):
                continue
            get_converter = getattr(axis, "get_converter", None)
            converter = get_converter() if callable(get_converter) else getattr(axis, "converter", None)
            if converter is not None and axis_name == "x":
                for label in labels:
                    label.set_rotation(policy.categorical_rotation_deg)
                    label.set_horizontalalignment("right")
                repairs.append(f"axes[{axes_index}]: rotated categorical x tick labels")
                _draw_figure(fig)
                rotated_renderer = fig.canvas.get_renderer()
                if _adjacent_artists_overlap(
                    labels,
                    rotated_renderer,
                    axis_name="x",
                    padding=policy.collision_padding_px,
                ):
                    for label in labels:
                        label.set_rotation(policy.categorical_max_rotation_deg)
                    repairs.append(f"axes[{axes_index}]: increased categorical x tick rotation")
            elif converter is None:
                from matplotlib.ticker import MaxNLocator

                axis.set_major_locator(
                    MaxNLocator(
                        nbins=max(policy.minimum_major_ticks, policy.target_major_ticks - 2),
                        steps=[1.0, 2.0, 2.5, 5.0, 10.0],
                        min_n_ticks=policy.minimum_major_ticks,
                    )
                )
                repairs.append(f"axes[{axes_index}]: reduced numeric {axis_name} tick density")
    return repairs


def _visible_tick_labels(ax: Any, axis_name: AxisName) -> list[Any]:
    if axis_name == "x":
        limits = tuple(float(value) for value in ax.get_xlim())
        values = ax.get_xticks()
        labels = ax.get_xticklabels()
    else:
        limits = tuple(float(value) for value in ax.get_ylim())
        values = ax.get_yticks()
        labels = ax.get_yticklabels()
    return [
        label
        for value, label in zip(values, labels)
        if _tick_is_in_view(float(value), limits) and label.get_visible() and label.get_text().strip()
    ]


def _repair_legend_obstruction(fig: Any, *, policy: PlotQualityPolicy) -> list[str]:
    repairs: list[str] = []
    renderer = fig.canvas.get_renderer()
    for axes_index, ax in enumerate(fig.get_axes()):
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        bbox = legend.get_window_extent(renderer=renderer)
        entry_count = len(legend.get_texts())
        covered = _data_vertices_inside_legend(ax, bbox)
        if covered <= 0 and entry_count <= policy.max_inside_legend_entries:
            continue
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            continue
        legend.remove()
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        repairs.append(f"axes[{axes_index}]: moved legend outside plotting area")
    return repairs


def _adjacent_artists_overlap(
    artists: list[Any],
    renderer: Any,
    *,
    axis_name: AxisName,
    padding: float,
) -> bool:
    boxes = [artist.get_window_extent(renderer=renderer) for artist in artists]
    if axis_name == "x":
        boxes.sort(key=lambda box: float(box.x0))
        return any(float(left.x1) - float(right.x0) > padding for left, right in zip(boxes, boxes[1:]))
    boxes.sort(key=lambda box: float(box.y0))
    return any(float(left.y1) - float(right.y0) > padding for left, right in zip(boxes, boxes[1:]))


def _apply_bounded_layout(fig: Any, *, reserve_right: bool) -> None:
    right = 0.78 if reserve_right else 0.98
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout(rect=(0.02, 0.05, right, 0.96))
        except (AttributeError, ValueError):
            return


def _deduplicate_issues(issues: list[PlotQualityIssue]) -> list[PlotQualityIssue]:
    out: list[PlotQualityIssue] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for issue in issues:
        key = (issue.check_id, issue.artists)
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    return out


__all__ = [
    "PLOT_QUALITY_POLICY_VERSION",
    "STRICT_AGENT_PLOT_QUALITY",
    "NumericFormatDecision",
    "PlotQualityIssue",
    "PlotQualityPolicy",
    "PlotQualityReport",
    "StableEngineeringFormatter",
    "apply_plot_quality_policy",
    "apply_stable_axis_format",
    "assess_plot_quality",
]
