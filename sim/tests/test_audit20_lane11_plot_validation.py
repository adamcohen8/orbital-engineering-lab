from __future__ import annotations

import pytest

from sim.plotting.animation_quality import AnimationQualityPolicy
from sim.plotting.quality import PlotQualityPolicy
from sim.review.plotting import ReviewPlotSpec, _validate_plot_spec
from sim.review.workspace import ReviewQueryResult


def _result(rows: list[dict[str, object]]) -> ReviewQueryResult:
    return ReviewQueryResult(
        columns=list(rows[0]) if rows else [],
        rows=rows,
        row_count=len(rows),
        truncated=False,
    )


def test_plot_validation_rejects_rows_without_plottable_x_values() -> None:
    spec = ReviewPlotSpec(sql="SELECT", x_column="time_s", y_columns=["value"])

    with pytest.raises(ValueError, match="no finite value paired"):
        _validate_plot_spec(spec, _result([{"time_s": None, "value": 1.0}]))


def test_plot_validation_rejects_degenerate_heatmap_categories() -> None:
    spec = ReviewPlotSpec(
        sql="SELECT",
        x_column="time_s",
        y_columns=["value"],
        group_column="series",
        plot_type="heatmap",
    )

    with pytest.raises(ValueError, match="non-null x category"):
        _validate_plot_spec(
            spec,
            _result([{"time_s": None, "series": None, "value": 1.0}]),
        )

    with pytest.raises(ValueError, match="non-null x category"):
        _validate_plot_spec(
            spec,
            _result([{"time_s": float("nan"), "series": "A", "value": 1.0}]),
        )


def test_plot_validation_rejects_nonpositive_render_bounds() -> None:
    result = _result([{"time_s": 0.0, "value": 1.0}])

    with pytest.raises(ValueError, match="max_rows"):
        _validate_plot_spec(
            ReviewPlotSpec(sql="SELECT", x_column="time_s", y_columns=["value"], max_rows=0),
            result,
        )
    with pytest.raises(ValueError, match="dpi"):
        _validate_plot_spec(
            ReviewPlotSpec(sql="SELECT", x_column="time_s", y_columns=["value"], dpi=0),
            result,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"collision_padding_px": -1.0},
        {"canvas_margin_px": float("nan")},
        {"max_inside_legend_entries": -1},
    ],
)
def test_plot_quality_policy_rejects_invalid_geometry_limits(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PlotQualityPolicy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contact_sheet_thumbnail_width_px": 0},
        {"camera_span_relative_tolerance": -1.0},
        {"max_reported_failed_frames": -1},
    ],
)
def test_animation_quality_policy_rejects_invalid_limits(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AnimationQualityPolicy(**kwargs)


def test_categorical_renderers_preserve_display_labels_after_quality_formatting() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from sim.review.plotting import _draw_plot

    bar_fig, bar_ax = plt.subplots()
    bar_spec = ReviewPlotSpec(sql="SELECT", x_column="category", y_columns=["value"], plot_type="bar")
    bar_result = _result([
        {"category": "alpha", "value": 1.0},
        {"category": "beta", "value": 2.0},
    ])
    _draw_plot(bar_ax, bar_result, bar_spec)
    bar_labels_before = [label.get_text() for label in bar_ax.get_xticklabels()]
    from sim.plotting.quality import apply_plot_quality_policy

    apply_plot_quality_policy(bar_fig)
    bar_labels_after = [label.get_text() for label in bar_ax.get_xticklabels()]
    plt.close(bar_fig)

    heat_fig, heat_ax = plt.subplots()
    heat_spec = ReviewPlotSpec(
        sql="SELECT", x_column="time_s", y_columns=["value"], group_column="series", plot_type="heatmap"
    )
    heat_result = _result([
        {"time_s": "t0", "series": "A", "value": 1.0},
        {"time_s": "t0", "series": "B", "value": 2.0},
    ])
    _draw_plot(heat_ax, heat_result, heat_spec)
    apply_plot_quality_policy(heat_fig)
    heat_x_labels = [label.get_text() for label in heat_ax.get_xticklabels()]
    heat_y_labels = [label.get_text() for label in heat_ax.get_yticklabels()]
    plt.close(heat_fig)

    assert bar_labels_before == ["alpha", "beta"]
    assert bar_labels_after == bar_labels_before
    assert heat_x_labels == ["t0"]
    assert heat_y_labels == ["A", "B"]


def test_heatmap_quality_does_not_create_colorbar_x_tick_labels() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from sim.plotting.quality import apply_plot_quality_policy
    from sim.review.plotting import _draw_plot

    result = _result([
        {"time_s": "t0", "series": "A", "value": 1.0},
        {"time_s": "t0", "series": "B", "value": 2.0},
    ])
    spec = ReviewPlotSpec(
        sql="SELECT", x_column="time_s", y_columns=["value"], group_column="series", plot_type="heatmap"
    )
    fig, ax = plt.subplots()
    _draw_plot(ax, result, spec)
    apply_plot_quality_policy(fig)
    colorbar = next(axis for axis in fig.axes if axis.get_label() == "<colorbar>")
    assert not any(label.get_text() for label in colorbar.get_xticklabels())
    plt.close(fig)
