from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.plotting.style import artifact_metadata, save_oel_figure, style_name_from_config
from sim.utils.figure_size import cap_figsize


def _numeric_run_series(run_details: list[dict[str, Any]], key: str) -> np.ndarray:
    from sim.execution.monte_carlo_support import safe_float

    if key.startswith("metric:"):
        source, name = "metrics", key[len("metric:") :]
    elif key.startswith("study_metric:"):
        source, name = "metrics", key[len("study_metric:") :]
    elif key.startswith("gate_metric:"):
        source, name = "gate_metrics", key[len("gate_metric:") :]
    elif key.startswith("study:"):
        # ``study:`` is also a legal user metric name. Treat the exact
        # configured key as authoritative when it is present; the prefixed
        # form is a namespace only for a study/gate name collision emitted by
        # ``_plot_metric_names``.
        if any(key in dict(detail.get("metrics", {}) or {}) for detail in run_details):
            source, name = "metrics", key
        else:
            source, name = "metrics", key[len("study:") :]
    elif key.startswith("gate:"):
        if any(key in dict(detail.get("gate_metrics", {}) or {}) for detail in run_details):
            source, name = "gate_metrics", key
        else:
            source, name = "gate_metrics", key[len("gate:") :]
    else:
        source = "metrics" if any(key in dict(d.get("metrics", {}) or {}) for d in run_details) else "gate_metrics"
        name = key
    return np.array([safe_float(dict(d.get(source, {}) or {}).get(name)) for d in run_details], dtype=float)


def _numeric_sampled_parameter_paths(run_details: list[dict[str, Any]]) -> list[str]:
    from sim.execution.monte_carlo_support import safe_float

    paths = sorted(
        {
            str(path)
            for detail in run_details
            for path in dict(detail.get("sampled_parameters", {}) or {}).keys()
            if str(path)
        }
    )
    out: list[str] = []
    for path in paths:
        vals = np.array(
            [safe_float(dict(detail.get("sampled_parameters", {}) or {}).get(path)) for detail in run_details],
            dtype=float,
        )
        finite = vals[np.isfinite(vals)]
        if finite.size >= 2 and not np.allclose(np.std(finite), 0.0):
            out.append(path)
    return out


def _sampled_parameter_array(run_details: list[dict[str, Any]], path: str) -> np.ndarray:
    from sim.execution.monte_carlo_support import safe_float

    return np.array(
        [safe_float(dict(detail.get("sampled_parameters", {}) or {}).get(path)) for detail in run_details],
        dtype=float,
    )


def _plot_metric_names(run_details: list[dict[str, Any]]) -> list[str]:
    study = {
        str(name) for detail in run_details for name in dict(detail.get("metrics", {}) or {}) if str(name)
    }
    gates = {
        str(name) for detail in run_details for name in dict(detail.get("gate_metrics", {}) or {}) if str(name)
    }
    names = [_plot_metric_key("metrics", name, study=study, gates=gates) for name in sorted(study)]
    names += [_plot_metric_key("gate_metrics", name, study=study, gates=gates) for name in sorted(gates)]
    return [name for name in names if np.isfinite(_numeric_run_series(run_details, name)).any()]


def _plot_metric_key(source: str, name: str, *, study: set[str], gates: set[str]) -> str:
    """Return a collision-free display key for one metric source."""

    if source == "metrics":
        if name in gates:
            # A literal ``study:<name>`` metric would otherwise collide with
            # the namespace used for the plain study metric ``name``.
            return f"study_metric:{name}" if f"study:{name}" in study else f"study:{name}"
        if name.startswith(("study:", "gate:", "metric:", "study_metric:", "gate_metric:")):
            return f"metric:{name}"
        return name
    if name in study:
        return f"gate_metric:{name}" if f"gate:{name}" in gates else f"gate:{name}"
    if name.startswith(("study:", "gate:", "metric:", "study_metric:", "gate_metric:")):
        return f"gate_metric:{name}"
    return name


def _gate_thresholds(run_details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    thresholds: dict[str, dict[str, Any]] = {}
    study_names = {
        str(name) for detail in run_details for name in dict(detail.get("metrics", {}) or {}) if str(name)
    }
    gate_names = {
        str(result.get("name", "") or "")
        for detail in run_details
        for result in list(detail.get("metric_gate_results", []) or [])
        if str(result.get("name", "") or "")
    }
    for detail in run_details:
        for result_raw in list(detail.get("metric_gate_results", []) or []):
            result = dict(result_raw or {})
            name = str(result.get("name", "") or "")
            if name in study_names:
                plot_name = f"gate_metric:{name}" if f"gate:{name}" in gate_names else f"gate:{name}"
            else:
                plot_name = _plot_metric_key("gate_metrics", name, study=study_names, gates=gate_names)
            if name and plot_name not in thresholds:
                thresholds[plot_name] = {
                    "op": str(result.get("op", "") or ""),
                    "value": result.get("value"),
                    "metric": str(result.get("metric", "") or ""),
                }
    return thresholds


def _gate_margin(values: np.ndarray, *, op: str, limit: Any) -> np.ndarray | None:
    from sim.execution.monte_carlo_support import safe_float

    threshold = safe_float(limit)
    if not np.isfinite(threshold):
        return None
    op_norm = str(op or "").strip().lower()
    if op_norm in {"<", "<=", "lt", "lte", "le"}:
        return threshold - values
    if op_norm in {">", ">=", "gt", "gte", "ge"}:
        return values - threshold
    return None


def _write_generic_monte_carlo_metric_plots(
    *,
    cfg: SimulationScenarioConfig,
    outdir: Path,
    agg: dict[str, Any],
    run_details: list[dict[str, Any]],
    save: bool,
    show: bool,
) -> dict[str, Any]:
    if not run_details:
        return agg
    metric_names = _plot_metric_names(run_details)
    if not metric_names:
        return agg

    import matplotlib.pyplot as plt

    artifacts = dict(agg.get("artifacts", {}) or {})
    dpi = int(cfg.outputs.plots.get("dpi", 150))
    metadata = artifact_metadata(scenario_name=str(cfg.scenario_name or "monte_carlo"))
    style_name = style_name_from_config(dict(cfg.outputs.plots or {}))
    pass_color = np.array(
        ["tab:green" if bool(d.get("pass", False)) else "tab:red" for d in run_details],
        dtype=object,
    )
    thresholds = _gate_thresholds(run_details)

    metric_subset = metric_names[:6]
    nplots = len(metric_subset)
    ncols = min(3, max(1, nplots))
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=cap_figsize(5.0 * ncols, 3.6 * nrows), squeeze=False)
    for ax, metric_name in zip(np.ravel(axes), metric_subset):
        values = _numeric_run_series(run_details, metric_name)
        finite = values[np.isfinite(values)]
        bins = int(max(5, min(30, np.sqrt(max(finite.size, 1)))))
        ax.hist(finite, bins=bins, alpha=0.85)
        threshold = thresholds.get(metric_name)
        if threshold:
            margin = _gate_margin(finite, op=str(threshold.get("op")), limit=threshold.get("value"))
            if margin is not None:
                value = float(threshold.get("value"))
                ax.axvline(value, linestyle="--", color="k")
        ax.set_title(metric_name)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
    for ax in list(np.ravel(axes))[nplots:]:
        ax.set_visible(False)
    fig.suptitle("Monte Carlo Study Metric Histograms", fontsize=12)
    fig.tight_layout()
    path = outdir / "master_monte_carlo_study_metric_histograms.png"
    if save:
        save_oel_figure(fig, path, dpi=dpi, metadata=metadata, artifact_id=path.stem, style_name=style_name)
        artifacts["study_metric_histograms_png"] = str(path)
    if show:
        plt.show()
    plt.close(fig)

    fig_cdf, axes_cdf = plt.subplots(nrows, ncols, figsize=cap_figsize(5.0 * ncols, 3.6 * nrows), squeeze=False)
    for ax, metric_name in zip(np.ravel(axes_cdf), metric_subset):
        values = _numeric_run_series(run_details, metric_name)
        finite = np.sort(values[np.isfinite(values)])
        if finite.size:
            y = np.arange(1, finite.size + 1, dtype=float) / float(finite.size)
            ax.step(finite, y, where="post")
        threshold = thresholds.get(metric_name)
        if threshold and _gate_margin(finite, op=str(threshold.get("op")), limit=threshold.get("value")) is not None:
            ax.axvline(float(threshold.get("value")), linestyle="--", color="k")
        ax.set_title(metric_name)
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
    for ax in list(np.ravel(axes_cdf))[nplots:]:
        ax.set_visible(False)
    fig_cdf.suptitle("Monte Carlo Study Metric CDFs", fontsize=12)
    fig_cdf.tight_layout()
    cdf_path = outdir / "master_monte_carlo_study_metric_cdfs.png"
    if save:
        save_oel_figure(fig_cdf, cdf_path, dpi=dpi, metadata=metadata, artifact_id=cdf_path.stem, style_name=style_name)
        artifacts["study_metric_cdfs_png"] = str(cdf_path)
    if show:
        plt.show()
    plt.close(fig_cdf)

    parameter_paths = _numeric_sampled_parameter_paths(run_details)
    if parameter_paths:
        params = parameter_paths[:3]
        metrics = metric_names[:3]
        fig_scatter, axes_scatter = plt.subplots(
            len(metrics),
            len(params),
            figsize=cap_figsize(4.8 * len(params), 3.8 * len(metrics)),
            squeeze=False,
        )
        for row, metric_name in enumerate(metrics):
            y = _numeric_run_series(run_details, metric_name)
            threshold = thresholds.get(metric_name)
            for col, param_path in enumerate(params):
                x = _sampled_parameter_array(run_details, param_path)
                finite = np.isfinite(x) & np.isfinite(y)
                ax = axes_scatter[row, col]
                ax.scatter(x[finite], y[finite], c=pass_color[finite], s=24, alpha=0.9)
                if threshold and _gate_margin(y, op=str(threshold.get("op")), limit=threshold.get("value")) is not None:
                    ax.axhline(float(threshold.get("value")), linestyle="--", color="k")
                ax.set_xlabel(param_path)
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
        fig_scatter.suptitle("Sampled Parameters vs Study Metrics", fontsize=12)
        fig_scatter.tight_layout()
        scatter_path = outdir / "master_monte_carlo_parameter_metric_scatter.png"
        if save:
            save_oel_figure(fig_scatter, scatter_path, dpi=dpi, metadata=metadata, artifact_id=scatter_path.stem, style_name=style_name)
            artifacts["parameter_metric_scatter_png"] = str(scatter_path)
        if show:
            plt.show()
        plt.close(fig_scatter)

    if len(parameter_paths) >= 2:
        x_path = parameter_paths[0]
        y_path = parameter_paths[1]
        x = _sampled_parameter_array(run_details, x_path)
        y = _sampled_parameter_array(run_details, y_path)
        finite = np.isfinite(x) & np.isfinite(y)
        fig_pf, ax_pf = plt.subplots(figsize=cap_figsize(7.5, 5.5))
        ax_pf.scatter(x[finite], y[finite], c=pass_color[finite], s=34, alpha=0.9)
        ax_pf.set_xlabel(x_path)
        ax_pf.set_ylabel(y_path)
        ax_pf.set_title("Pass/Fail by Sampled Parameter Pair")
        ax_pf.grid(True, alpha=0.3)
        pf_path = outdir / "master_monte_carlo_pass_fail_parameter_map.png"
        fig_pf.tight_layout()
        if save:
            save_oel_figure(fig_pf, pf_path, dpi=dpi, metadata=metadata, artifact_id=pf_path.stem, style_name=style_name)
            artifacts["pass_fail_parameter_map_png"] = str(pf_path)
        if show:
            plt.show()
        plt.close(fig_pf)

    margin_rows: list[tuple[str, np.ndarray]] = []
    for metric_name, threshold in thresholds.items():
        values = _numeric_run_series(run_details, metric_name)
        margin = _gate_margin(values, op=str(threshold.get("op")), limit=threshold.get("value"))
        if margin is None:
            continue
        finite_margin = margin[np.isfinite(margin)]
        if finite_margin.size:
            margin_rows.append((metric_name, finite_margin))
    if margin_rows:
        labels = [name for name, _ in margin_rows[:10]]
        data = [vals for _, vals in margin_rows[:10]]
        fig_margin, ax_margin = plt.subplots(figsize=cap_figsize(max(7.5, 0.7 * len(data) + 4.0), 5.0))
        try:
            ax_margin.boxplot(data, tick_labels=labels, showfliers=True)
        except TypeError:
            ax_margin.boxplot(data, labels=labels, showfliers=True)
        ax_margin.axhline(0.0, linestyle="--", color="k")
        ax_margin.set_title("Metric Gate Margins")
        ax_margin.set_ylabel("positive margin passes")
        ax_margin.tick_params(axis="x", labelrotation=30)
        ax_margin.grid(True, axis="y", alpha=0.3)
        fig_margin.tight_layout()
        margin_path = outdir / "master_monte_carlo_metric_gate_margins.png"
        if save:
            save_oel_figure(fig_margin, margin_path, dpi=dpi, metadata=metadata, artifact_id=margin_path.stem, style_name=style_name)
            artifacts["metric_gate_margins_png"] = str(margin_path)
        if show:
            plt.show()
        plt.close(fig_margin)

    agg["artifacts"] = artifacts
    return agg


def write_monte_carlo_plot_artifacts(
    *,
    cfg: SimulationScenarioConfig,
    outdir: Path,
    agg: dict[str, Any],
    runs: list[dict[str, Any]],
    run_details: list[dict[str, Any]],
    relative_range_series_runs: list[dict[str, np.ndarray] | None],
    durations_s: np.ndarray,
    ca_finite: np.ndarray,
    all_obj_ids: list[str],
    dv_by_object: dict[str, list[float]],
    dv_remaining_m_s_by_object: dict[str, list[float]],
    dv_budget_m_s_by_object: dict[str, float],
    failure_mode_counts: dict[str, int],
    keepout_threshold: float,
    gates: Any,
    mc_out_cfg: dict[str, Any],
) -> dict[str, Any]:
    from sim.execution.monte_carlo_support import mc_initial_relative_ric_curv_samples, safe_float

    metadata = artifact_metadata(scenario_name=str(cfg.scenario_name or "monte_carlo"))
    style_name = style_name_from_config(dict(cfg.outputs.plots or {}))
    legacy_gates = dict(gates if isinstance(gates, dict) else {})
    save_hist = bool(cfg.outputs.monte_carlo.get("save_histograms", False))
    show_hist = bool(cfg.outputs.monte_carlo.get("display_histograms", False))
    if (save_hist or show_hist) and runs:
        import matplotlib.pyplot as plt

        plot_series: list[tuple[str, np.ndarray]] = [("Duration (s)", durations_s)]
        if ca_finite.size:
            plot_series.append(("Closest Approach (km)", ca_finite))
        for oid in all_obj_ids:
            dv_arr = np.array(dv_by_object.get(oid, []), dtype=float)
            if dv_arr.size:
                plot_series.append((f"{oid} Total dV (m/s)", dv_arr))
        for oid in sorted(dv_remaining_m_s_by_object.keys()):
            rem_arr = np.array(dv_remaining_m_s_by_object.get(oid, []), dtype=float)
            if rem_arr.size:
                plot_series.append((f"{oid} dV Remaining (m/s)", rem_arr))
        nplots = len(plot_series)
        if nplots == 6:
            nrows, ncols = 3, 2
        else:
            ncols = min(3, max(1, nplots))
            nrows = int(np.ceil(nplots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=cap_figsize(5.2 * ncols, 3.8 * nrows), squeeze=False)
        axes_flat = list(np.ravel(axes))
        for ax, (title, arr) in zip(axes_flat, plot_series):
            bins = int(max(5, min(30, np.sqrt(max(arr.size, 1)))))
            ax.hist(arr, bins=bins, alpha=0.85)
            ax.set_title(title)
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[nplots:]:
            ax.set_visible(False)
        fig.tight_layout()
        if save_hist:
            hist_path = outdir / "master_monte_carlo_histograms.png"
            save_oel_figure(fig, hist_path, dpi=int(cfg.outputs.plots.get("dpi", 150)), metadata=metadata, artifact_id=hist_path.stem, style_name=style_name)
            agg["artifacts"]["histograms_png"] = str(hist_path)
        if show_hist:
            plt.show()
        plt.close(fig)

        range_series_available = [s for s in relative_range_series_runs if isinstance(s, dict)]
        if range_series_available:
            fig_rr, ax_rr = plt.subplots(figsize=cap_figsize(10, 6))
            for idx, series in enumerate(relative_range_series_runs):
                if not isinstance(series, dict):
                    continue
                t_rr = np.array(series.get("time_s", []), dtype=float)
                r_rr = np.array(series.get("range_km", []), dtype=float)
                if t_rr.size == 0 or r_rr.size == 0:
                    continue
                ax_rr.plot(t_rr, r_rr, linewidth=1.0, alpha=0.65, label=f"run {idx}")
            ax_rr.set_title("Chaser-Target Relative Range by Iteration")
            ax_rr.set_xlabel("Time (s)")
            ax_rr.set_ylabel("Range (km)")
            ax_rr.grid(True, alpha=0.3)
            fig_rr.tight_layout()
            rr_path = outdir / "master_monte_carlo_relative_range_timeseries.png"
            if save_hist:
                save_oel_figure(fig_rr, rr_path, dpi=int(cfg.outputs.plots.get("dpi", 150)), metadata=metadata, artifact_id=rr_path.stem, style_name=style_name)
                agg["artifacts"]["relative_range_timeseries_png"] = str(rr_path)
            if show_hist:
                plt.show()
            plt.close(fig_rr)

    save_ops_dashboard = bool(mc_out_cfg.get("save_ops_dashboard", True))
    show_ops_dashboard = bool(mc_out_cfg.get("display_ops_dashboard", False))
    agg = _write_generic_monte_carlo_metric_plots(
        cfg=cfg,
        outdir=outdir,
        agg=agg,
        run_details=run_details,
        save=bool(save_hist or save_ops_dashboard),
        show=bool(show_hist or show_ops_dashboard),
    )
    if (save_ops_dashboard or show_ops_dashboard) and run_details:
        import matplotlib.pyplot as plt

        ric_initial_samples = mc_initial_relative_ric_curv_samples(cfg, run_details)
        pass_color = np.array(
            ["tab:green" if bool(d.get("pass", False)) else "tab:red" for d in run_details], dtype=object
        )
        idx_arr = np.arange(len(run_details), dtype=float)
        ca_run_arr = np.array([safe_float(d.get("closest_approach_km")) for d in run_details], dtype=float)
        dur_run_arr = np.array([safe_float(d.get("duration_s"), default=0.0) for d in run_details], dtype=float)
        dv_run_arr = np.array([safe_float(d.get("total_dv_m_s_total"), default=0.0) for d in run_details], dtype=float)

        fig, axes = plt.subplots(2, 3, figsize=cap_figsize(14, 8))
        bins = int(max(5, min(30, np.sqrt(max(len(run_details), 1)))))

        finite_ca = ca_run_arr[np.isfinite(ca_run_arr)]
        axes[0, 0].hist(finite_ca, bins=bins, alpha=0.85)
        if np.isfinite(keepout_threshold):
            axes[0, 0].axvline(keepout_threshold, linestyle="--", color="k")
        axes[0, 0].set_title("Closest Approach (km)")
        axes[0, 0].set_ylabel("count")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(dur_run_arr[np.isfinite(dur_run_arr)], bins=bins, alpha=0.85)
        max_duration_s = safe_float(legacy_gates.get("max_duration_s"))
        if np.isfinite(max_duration_s):
            axes[0, 1].axvline(max_duration_s, linestyle="--", color="k")
        axes[0, 1].set_title("Duration (s)")
        axes[0, 1].set_ylabel("count")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].hist(dv_run_arr[np.isfinite(dv_run_arr)], bins=bins, alpha=0.85, color="tab:orange")
        max_total_dv_m_s = safe_float(legacy_gates.get("max_total_dv_m_s"))
        if np.isfinite(max_total_dv_m_s):
            axes[0, 2].axvline(max_total_dv_m_s, linestyle="--", color="k")
        axes[0, 2].set_title("Total dV (m/s)")
        axes[0, 2].set_ylabel("count")
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].scatter(idx_arr, ca_run_arr, c=pass_color, s=22, alpha=0.9)
        if np.isfinite(keepout_threshold):
            axes[1, 0].axhline(keepout_threshold, linestyle="--", color="k")
        axes[1, 0].set_title("Closest Approach by Run")
        axes[1, 0].set_xlabel("run index")
        axes[1, 0].set_ylabel("km")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(idx_arr, dv_run_arr, c=pass_color, s=22, alpha=0.9)
        if np.isfinite(max_total_dv_m_s):
            axes[1, 1].axhline(max_total_dv_m_s, linestyle="--", color="k")
        axes[1, 1].set_title("Total dV by Run")
        axes[1, 1].set_xlabel("run index")
        axes[1, 1].set_ylabel("m/s")
        axes[1, 1].grid(True, alpha=0.3)

        top_fail_pairs = sorted(failure_mode_counts.items(), key=lambda kv: int(kv[1]), reverse=True)[:6]
        if top_fail_pairs:
            labels = [k for k, _ in top_fail_pairs]
            vals = [int(v) for _, v in top_fail_pairs]
            axes[1, 2].bar(np.arange(len(vals)), vals, color="tab:red", alpha=0.85)
            axes[1, 2].set_xticks(np.arange(len(vals)))
            axes[1, 2].set_xticklabels(labels, rotation=30, ha="right")
        axes[1, 2].set_title("Failure Mode Counts")
        axes[1, 2].set_ylabel("count")
        axes[1, 2].grid(True, alpha=0.3)

        fig.suptitle("Monte Carlo Ops Dashboard", fontsize=12)
        fig.tight_layout()
        dashboard_path = outdir / "master_monte_carlo_ops_dashboard.png"
        if save_ops_dashboard:
            save_oel_figure(fig, dashboard_path, dpi=int(cfg.outputs.plots.get("dpi", 150)), metadata=metadata, artifact_id=dashboard_path.stem, style_name=style_name)
            agg["artifacts"]["ops_dashboard_png"] = str(dashboard_path)
        if show_ops_dashboard:
            plt.show()
        plt.close(fig)

        if ric_initial_samples:
            fig_ic, axes_ic = plt.subplots(3, 2, figsize=cap_figsize(12, 9), squeeze=False)
            scatter_specs = [
                ("radial_sep_km", "Initial Radial Separation (km)"),
                ("radial_vel_km_s", "Initial Radial Velocity (km/s)"),
                ("in_track_sep_km", "Initial In-Track Separation (km)"),
                ("in_track_vel_km_s", "Initial In-Track Velocity (km/s)"),
                ("cross_track_sep_km", "Initial Cross-Track Separation (km)"),
                ("cross_track_vel_km_s", "Initial Cross-Track Velocity (km/s)"),
            ]
            axes_ic_flat = [
                axes_ic[0, 0],
                axes_ic[0, 1],
                axes_ic[1, 0],
                axes_ic[1, 1],
                axes_ic[2, 0],
                axes_ic[2, 1],
            ]
            for ax, (key, xlabel) in zip(axes_ic_flat, scatter_specs):
                x = np.array(ric_initial_samples.get(key, []), dtype=float)
                finite = np.isfinite(x) & np.isfinite(ca_run_arr)
                ax.scatter(x[finite], ca_run_arr[finite], c=pass_color[finite], s=24, alpha=0.85)
                if np.isfinite(keepout_threshold):
                    ax.axhline(keepout_threshold, linestyle="--", color="k")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Closest Approach (km)")
                ax.grid(True, alpha=0.3)
            fig_ic.suptitle("Initial Relative RIC State vs Closest Approach", fontsize=12)
            fig_ic.tight_layout()
            ic_path = outdir / "master_monte_carlo_initial_relative_state_vs_closest_approach.png"
            if save_ops_dashboard:
                save_oel_figure(fig_ic, ic_path, dpi=int(cfg.outputs.plots.get("dpi", 150)), metadata=metadata, artifact_id=ic_path.stem, style_name=style_name)
                agg["artifacts"]["initial_relative_state_vs_closest_approach_png"] = str(ic_path)
            if show_ops_dashboard:
                plt.show()
            plt.close(fig_ic)

        rem_obj_ids = sorted(str(oid) for oid in dv_budget_m_s_by_object.keys())
        if rem_obj_ids:
            fig_rem, axes_rem = plt.subplots(
                2, len(rem_obj_ids), figsize=cap_figsize(5.0 * len(rem_obj_ids), 7.0), squeeze=False
            )
            for j, oid in enumerate(rem_obj_ids):
                rem_arr = np.array(
                    [
                        safe_float(dict(d.get("delta_v_remaining_m_s_by_object", {}) or {}).get(oid))
                        for d in run_details
                    ],
                    dtype=float,
                )
                finite_rem = rem_arr[np.isfinite(rem_arr)]
                bins_rem = int(max(5, min(30, np.sqrt(max(finite_rem.size, 1)))))
                axes_rem[0, j].hist(finite_rem, bins=bins_rem, alpha=0.85, color="tab:blue")
                axes_rem[0, j].set_title(f"{oid} dV Remaining (m/s)")
                axes_rem[0, j].set_ylabel("count")
                axes_rem[0, j].grid(True, alpha=0.3)

                axes_rem[1, j].scatter(idx_arr, rem_arr, c=pass_color, s=22, alpha=0.9)
                axes_rem[1, j].set_title(f"{oid} dV Remaining by Run")
                axes_rem[1, j].set_xlabel("run index")
                axes_rem[1, j].set_ylabel("m/s")
                axes_rem[1, j].grid(True, alpha=0.3)
            fig_rem.suptitle("Monte Carlo Delta-V Remaining", fontsize=12)
            fig_rem.tight_layout()
            rem_path = outdir / "master_monte_carlo_delta_v_remaining.png"
            if save_ops_dashboard:
                save_oel_figure(fig_rem, rem_path, dpi=int(cfg.outputs.plots.get("dpi", 150)), metadata=metadata, artifact_id=rem_path.stem, style_name=style_name)
                agg["artifacts"]["delta_v_remaining_png"] = str(rem_path)
            if show_ops_dashboard:
                plt.show()
            plt.close(fig_rem)

    return agg
