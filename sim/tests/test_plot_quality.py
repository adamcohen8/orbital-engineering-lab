from __future__ import annotations

import pytest

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception as exc:  # pragma: no cover - depends on the optional plotting stack
    pytest.skip(f"matplotlib is not usable in this environment: {exc}", allow_module_level=True)

import matplotlib.pyplot as plt

from sim.plotting.quality import (
    PlotQualityPolicy,
    StableEngineeringFormatter,
    apply_plot_quality_policy,
    apply_stable_axis_format,
    assess_plot_quality,
)
from sim.plotting.style import add_artifact_footer, artifact_metadata


def test_fixed_resolution_formatter_is_axis_wide_and_suppresses_negative_zero() -> None:
    formatter = StableEngineeringFormatter(decimal_places=2, scale_exponent=0)

    assert formatter(1.2) == "1.20"
    assert formatter(1.234) == "1.23"
    assert formatter(-0.0001) == "0.00"
    assert formatter.get_offset() == ""


def test_shared_engineering_formatter_uses_one_axis_exponent() -> None:
    formatter = StableEngineeringFormatter(decimal_places=1, scale_exponent=-6)

    assert formatter(2.5e-6) == "2.5"
    assert formatter(4.0e-6) == "4.0"
    assert formatter.get_offset() == r"$\times 10^{-6}$"


def test_axis_formatting_derives_precision_from_one_nice_tick_interval() -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([0.0, 1.0], [0.0, 0.01])

    decision = apply_stable_axis_format(ax, "y")
    fig.canvas.draw()
    labels = [label.get_text() for label in ax.get_yticklabels() if label.get_text()]
    plt.close(fig)

    assert decision is not None
    assert decision.formatter_kind == "fixed_resolution"
    assert decision.decimal_places >= 2
    assert labels
    assert len({len(label.partition(".")[2]) for label in labels}) == 1


def test_axis_formatting_uses_shared_engineering_scale_for_small_values() -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([0.0, 1.0], [1.0e-7, 8.0e-7])

    decision = apply_stable_axis_format(ax, "y")
    fig.canvas.draw()
    offset = ax.yaxis.get_major_formatter().get_offset()
    plt.close(fig)

    assert decision is not None
    assert decision.formatter_kind == "shared_engineering"
    assert decision.scale_exponent == -6
    assert offset == r"$\times 10^{-6}$"


def test_quality_policy_formats_after_equal_aspect_resolves_final_limits() -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([-3.0, -2.5], [0.0, 0.1])
    ax.set_aspect("equal", adjustable="datalim")

    report = apply_plot_quality_policy(fig)
    labels = [
        label.get_text()
        for value, label in zip(ax.get_yticks(), ax.get_yticklabels())
        if min(ax.get_ylim()) <= float(value) <= max(ax.get_ylim()) and label.get_text()
    ]
    plt.close(fig)

    assert report.automated_status == "passed"
    assert len(labels) == len(set(labels))


def test_quality_assessment_detects_overlapping_and_clipped_text() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    fig.text(0.5, 0.5, "first overlapping label", ha="center")
    fig.text(0.5, 0.5, "second overlapping label", ha="center")
    fig.text(-0.10, 0.8, "clipped label")

    issues = assess_plot_quality(fig)
    plt.close(fig)

    check_ids = {issue.check_id for issue in issues}
    assert "text_overlap" in check_ids
    assert "text_inside_canvas" in check_ids


def test_quality_assessment_detects_below_minimum_font_size() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    fig.text(0.5, 0.5, "too small", fontsize=5)

    issues = assess_plot_quality(fig)
    plt.close(fig)

    assert any(issue.check_id == "minimum_font_size" for issue in issues)


def test_quality_policy_allows_readable_compact_provenance_footer() -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    add_artifact_footer(
        fig,
        metadata=artifact_metadata(scenario_name="quality", version_text="0.25.0"),
        artifact_id="quality_footer",
    )

    issues = assess_plot_quality(fig)
    plt.close(fig)

    assert not any(issue.check_id == "minimum_font_size" and "footer" in issue.artists[0] for issue in issues)


def test_quality_policy_moves_obstructive_legend_and_records_repair() -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    x_values = [0.0, 0.5, 1.0]
    for index in range(5):
        ax.plot(x_values, [0.4 + index * 0.01] * 3, label=f"series {index}")
    ax.legend(loc="center")

    report = apply_plot_quality_policy(fig)
    legend = ax.get_legend()
    plt.close(fig)

    assert legend is not None
    assert any("moved legend outside" in repair for repair in report.repairs)
    assert report.policy_version == 1
    assert report.visual_qa_status == "pending_agent_review"
    assert report.numeric_formatting


def test_quality_assessment_detects_figure_legend_over_axes() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for index, ax in enumerate(axes):
        ax.plot([0.0, 1.0], [float(index), float(index + 1)], label=f"series {index}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center")

    issues = assess_plot_quality(fig)
    plt.close(fig)

    assert any(issue.check_id == "figure_legend_axes_obstruction" for issue in issues)


def test_quality_policy_repairs_long_categorical_tick_labels() -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = [
        "very long category alpha",
        "very long category beta",
        "very long category gamma",
        "very long category delta",
    ]
    ax.bar(labels, [1.0, 2.0, 3.0, 4.0])
    ax.set_title("Long category labels")

    report = apply_plot_quality_policy(fig)
    rotations = {float(label.get_rotation()) for label in ax.get_xticklabels()}
    plt.close(fig)

    assert report.automated_status == "passed"
    assert any("rotated categorical" in repair for repair in report.repairs)
    assert rotations == {65.0}


def test_quality_report_is_manifest_ready_and_preserves_unresolved_failures() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    fig.text(0.5, 0.5, "same location")
    fig.text(0.5, 0.5, "same location")
    policy = PlotQualityPolicy(auto_repair=False)

    payload = apply_plot_quality_policy(fig, policy=policy).to_dict()
    plt.close(fig)

    assert payload["policy_id"] == "oel.agent_strict"
    assert payload["policy_version"] == 1
    assert payload["automated_status"] == "failed"
    assert "text_overlap" in payload["failed_checks"]
    assert payload["visual_review_required"] is True
    assert payload["visual_qa_status"] == "pending_agent_review"


def test_small_density_axis_excludes_outside_ticks():
    from sim.plotting.quality import _tick_is_in_view
    assert not _tick_is_in_view(2.30e-10, (2.25e-10, 2.29e-10))
    assert _tick_is_in_view(2.27e-10, (2.25e-10, 2.29e-10))
