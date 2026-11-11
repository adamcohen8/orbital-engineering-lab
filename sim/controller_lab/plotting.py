from __future__ import annotations

import contextlib
import csv
import io
import os
from pathlib import Path
from typing import Any

import numpy as np

from sim.plotting.style import OELArtifactMetadata, artifact_metadata, show_save_close_oel
from sim.utils.figure_size import cap_figsize

PlotMode = str


def _extract_linear_feedback_debug(entry: dict[str, Any]) -> dict[str, Any]:
    command_raw = dict(entry.get("command_raw", {}) or {})
    raw_mode_flags = dict(command_raw.get("mode_flags", {}) or {})
    debug = dict(raw_mode_flags.get("linear_feedback_debug", {}) or {})
    if debug:
        return debug
    mode_flags = dict(entry.get("mode_flags", {}) or {})
    return dict(mode_flags.get("linear_feedback_debug", {}) or {})


def extract_linear_feedback_trace(payload: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    debug_entries = list(dict(payload.get("controller_debug_by_object", {}) or {}).get(object_id, []) or [])
    if not debug_entries:
        return None

    times: list[float] = []
    control_pre_limit: list[np.ndarray] = []
    control_post_limit: list[np.ndarray] = []
    term_contributions_pre_limit: list[np.ndarray] = []
    term_contributions_post_limit: list[np.ndarray] = []
    state_rect_hist: list[np.ndarray] = []
    state_effective_hist: list[np.ndarray] = []
    gain_matrix: np.ndarray | None = None
    control_axes: list[str] | None = None
    state_labels: list[str] | None = None
    law_label = ""

    for entry in debug_entries:
        debug = _extract_linear_feedback_debug(dict(entry or {}))
        if not debug:
            continue
        try:
            t_s = float(entry.get("t_s"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(t_s):
            continue

        axes = [str(axis) for axis in list(debug.get("control_axes", []) or [])]
        labels = [str(label) for label in list(debug.get("state_labels", []) or [])]
        if not axes or not labels:
            continue

        pre = np.array(debug.get("control_pre_limit", []), dtype=float).reshape(-1)
        post = np.array(debug.get("control_post_limit", []), dtype=float).reshape(-1)
        contrib_pre = np.array(debug.get("term_contributions_pre_limit", []), dtype=float)
        contrib_post = np.array(debug.get("term_contributions_post_limit", []), dtype=float)
        state_rect = np.array(debug.get("state_rect", []), dtype=float).reshape(-1)
        state_effective = np.array(debug.get("state_effective", []), dtype=float).reshape(-1)
        if (
            pre.size != len(axes)
            or post.size != len(axes)
            or contrib_pre.shape != (len(axes), len(labels))
            or contrib_post.shape != (len(axes), len(labels))
            or state_rect.size != len(labels)
            or state_effective.size != len(labels)
        ):
            continue

        gain_here = np.array(debug.get("gain_matrix", []), dtype=float)
        if gain_here.shape != (len(axes), len(labels)):
            gain_here = None

        if control_axes is None:
            control_axes = axes
            state_labels = labels
            law_label = str(debug.get("law_label", "") or "").strip()
            if gain_here is not None:
                gain_matrix = gain_here
        elif axes != control_axes or labels != state_labels:
            continue

        times.append(t_s)
        control_pre_limit.append(pre)
        control_post_limit.append(post)
        term_contributions_pre_limit.append(contrib_pre)
        term_contributions_post_limit.append(contrib_post)
        state_rect_hist.append(state_rect)
        state_effective_hist.append(state_effective)

    if not times or control_axes is None or state_labels is None:
        return None

    return {
        "object_id": object_id,
        "law_label": law_label or "-Kx",
        "control_axes": control_axes,
        "state_labels": state_labels,
        "time_s": np.array(times, dtype=float),
        "control_pre_limit": np.vstack(control_pre_limit),
        "control_post_limit": np.vstack(control_post_limit),
        "term_contributions_pre_limit": np.stack(term_contributions_pre_limit),
        "term_contributions_post_limit": np.stack(term_contributions_post_limit),
        "state_rect": np.vstack(state_rect_hist),
        "state_effective": np.vstack(state_effective_hist),
        "gain_matrix": None if gain_matrix is None else np.array(gain_matrix, dtype=float),
    }


def _write_axis_csv(trace: dict[str, Any], axis_idx: int, axis_label: str, out_path: Path) -> None:
    time_s = np.array(trace["time_s"], dtype=float)
    control_pre = np.array(trace["control_pre_limit"], dtype=float)[:, axis_idx]
    control_post = np.array(trace["control_post_limit"], dtype=float)[:, axis_idx]
    contrib_pre = np.array(trace["term_contributions_pre_limit"], dtype=float)[:, axis_idx, :]
    contrib_post = np.array(trace["term_contributions_post_limit"], dtype=float)[:, axis_idx, :]
    state_rect = np.array(trace["state_rect"], dtype=float)
    state_effective = np.array(trace["state_effective"], dtype=float)
    state_labels = [str(label) for label in list(trace["state_labels"])]

    fieldnames = [
        "time_s",
        "command_pre_limit_km_s2",
        "command_post_limit_km_s2",
        *[f"contribution_pre_limit_{label}" for label in state_labels],
        *[f"contribution_post_limit_{label}" for label in state_labels],
        *[f"state_rect_{label}" for label in state_labels],
        *[f"state_effective_{label}" for label in state_labels],
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, t_s in enumerate(time_s):
            row: dict[str, float] = {
                "time_s": float(t_s),
                "command_pre_limit_km_s2": float(control_pre[idx]),
                "command_post_limit_km_s2": float(control_post[idx]),
            }
            for label_idx, label in enumerate(state_labels):
                row[f"contribution_pre_limit_{label}"] = float(contrib_pre[idx, label_idx])
                row[f"contribution_post_limit_{label}"] = float(contrib_post[idx, label_idx])
                row[f"state_rect_{label}"] = float(state_rect[idx, label_idx])
                row[f"state_effective_{label}"] = float(state_effective[idx, label_idx])
            writer.writerow(row)


def _try_import_pyplot(*, prefer_interactive: bool = False):
    if not prefer_interactive:
        os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _read_axis_csv(csv_path: Path) -> dict[str, np.ndarray] | None:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            try:
                value_f = float(value) if value is not None and value != "" else float("nan")
            except (TypeError, ValueError):
                value_f = float("nan")
            columns.setdefault(str(key), []).append(value_f)
    return {key: np.array(values, dtype=float) for key, values in columns.items()}


def _controller_feedback_show_save_close(
    fig,
    *,
    plt: Any,
    mode: PlotMode,
    out_path: Path | None,
    metadata: OELArtifactMetadata | None = None,
) -> None:
    show_save_close_oel(
        fig,
        mode=mode,
        out_path=out_path,
        dpi=150,
        metadata=metadata,
        style_name="oel_dark",
        plt_module=plt,
        show_block=False,
    )


def _plot_axis_feedback_from_csv(
    *,
    plt: Any,
    csv_path: Path,
    object_id: str,
    axis_label: str,
    law_label: str,
    mode: PlotMode,
    outdir: Path | None,
    metadata: OELArtifactMetadata | None = None,
) -> dict[str, str]:
    data = _read_axis_csv(csv_path)
    if data is None:
        return {}
    time_s = np.array(data.get("time_s", []), dtype=float)
    control_pre = np.array(data.get("command_pre_limit_km_s2", []), dtype=float)
    control_post = np.array(data.get("command_post_limit_km_s2", []), dtype=float)
    state_labels = [
        key[len("contribution_post_limit_") :] for key in data.keys() if key.startswith("contribution_post_limit_")
    ]
    state_labels.sort(
        key=lambda label: (
            ["R", "I", "C", "dR", "dI", "dC"].index(label) if label in {"R", "I", "C", "dR", "dI", "dC"} else 99
        )
    )
    out: dict[str, str] = {}

    fig, (ax_cmd, ax_terms) = plt.subplots(2, 1, figsize=cap_figsize(10, 8.0), sharex=True)
    ax_cmd.plot(time_s, control_pre, linewidth=2.0, label=f"{axis_label} raw command")
    if not np.allclose(control_pre, control_post, equal_nan=True):
        ax_cmd.plot(
            time_s,
            control_post,
            linestyle="--",
            linewidth=1.8,
            label=f"{axis_label} actuator-limited",
        )
    else:
        ax_cmd.plot(
            time_s, control_post, linestyle="--", linewidth=1.2, alpha=0.7, label=f"{axis_label} actuator-limited"
        )
    ax_cmd.set_title(f"{object_id} {axis_label}-axis control command ({law_label})")
    ax_cmd.set_ylabel("Accel (km/s^2)")
    ax_cmd.grid(True, alpha=0.3)
    ax_cmd.legend(loc="best")
    for state_label in state_labels:
        key = f"contribution_post_limit_{state_label}"
        ax_terms.plot(time_s, np.array(data.get(key, []), dtype=float), linewidth=1.5, label=state_label)
    ax_terms.plot(time_s, control_post, color="black", linewidth=2.2, label="sum")
    ax_terms.set_title(f"{object_id} {axis_label}-axis term contributions ({law_label})")
    ax_terms.set_xlabel("Time (s)")
    ax_terms.set_ylabel("Accel Contribution (km/s^2)")
    ax_terms.grid(True, alpha=0.3)
    ax_terms.legend(loc="best", ncol=2)
    fig.tight_layout()
    combined_path = None if outdir is None else outdir / f"{object_id}_{axis_label.lower()}_feedback.png"
    _controller_feedback_show_save_close(fig, plt=plt, mode=mode, out_path=combined_path, metadata=metadata)
    if combined_path is not None:
        out["combined_plot_png"] = str(combined_path)
        out["command_plot_png"] = str(combined_path)
        out["terms_plot_png"] = str(combined_path)
    return out


def _plot_variant_pass_rates(
    *, plt: Any, result: dict[str, Any], mode: PlotMode, outdir: Path | None, metadata: OELArtifactMetadata | None = None
) -> dict[str, str]:
    summaries = list(result.get("variant_summaries", []) or [])
    if not summaries:
        return {}
    variant_names = [str(summary.get("variant_name", "")) for summary in summaries]
    pass_rates = [100.0 * float(summary.get("pass_rate", 0.0)) for summary in summaries]
    fig, ax = plt.subplots(figsize=cap_figsize(max(8.0, 1.6 * max(len(variant_names), 1)), 4.8))
    x = np.arange(len(variant_names), dtype=float)
    bars = ax.bar(x, pass_rates, color="#355C7D")
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=20, ha="right")
    ax.set_ylim(0.0, max(100.0, max(pass_rates) * 1.1 if pass_rates else 100.0))
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title(f"{str(result.get('suite_name', 'controller_bench'))} Variant Pass Rates")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path = None if outdir is None else outdir / "controller_bench_variant_pass_rates.png"
    _controller_feedback_show_save_close(fig, plt=plt, mode=mode, out_path=out_path, metadata=metadata)
    return {} if out_path is None else {"pass_rate_plot_png": str(out_path)}


def write_linear_feedback_diagnostics(payload: dict[str, Any], object_id: str, outdir: Path) -> dict[str, Any]:
    trace = extract_linear_feedback_trace(payload, object_id)
    if trace is None:
        return {}

    diag_outdir = outdir / "controller_feedback"
    diag_outdir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {
        "output_dir": str(diag_outdir),
        "object_id": str(object_id),
        "law_label": str(trace["law_label"]),
        "control_axes": list(trace["control_axes"]),
        "state_labels": list(trace["state_labels"]),
        "axes": {},
    }
    for axis_idx, axis_label in enumerate(list(trace["control_axes"])):
        csv_path = diag_outdir / f"{object_id}_{str(axis_label).lower()}_feedback.csv"
        _write_axis_csv(trace, axis_idx, str(axis_label), csv_path)
        artifacts["axes"][str(axis_label)] = {"csv": str(csv_path)}
    return artifacts


def render_controller_bench_visualizations(result: dict[str, Any], outdir: Path, mode: PlotMode) -> dict[str, Any]:
    plot_mode = str(mode or "save").strip().lower()
    if plot_mode not in {"interactive", "save", "both"}:
        raise ValueError("controller bench plot mode must be one of: interactive, save, both.")

    plt = _try_import_pyplot(prefer_interactive=plot_mode in ("interactive", "both"))
    if plt is None:
        return {}

    plot_outdir = outdir if plot_mode in ("save", "both") else None
    artifacts: dict[str, Any] = {}
    metadata = artifact_metadata(scenario_name=str(result.get("suite_name", "controller_bench") or "controller_bench"))
    pass_rate_plot = _plot_variant_pass_rates(
        plt=plt,
        result=result,
        mode=plot_mode,
        outdir=plot_outdir,
        metadata=metadata,
    )
    if pass_rate_plot:
        artifacts.update(pass_rate_plot)

    feedback_artifacts: dict[str, Any] = {}
    for run in list(result.get("runs", []) or []):
        run_key = f"{str(run.get('case_name', 'case'))}::{str(run.get('variant_name', 'variant'))}"
        run_feedback = dict(dict(run.get("artifacts", {}) or {}).get("linear_feedback", {}) or {})
        if not run_feedback:
            continue
        axes = dict(run_feedback.get("axes", {}) or {})
        object_id = str(run_feedback.get("object_id", "controller"))
        law_label = str(run_feedback.get("law_label", "-Kx"))
        run_entry: dict[str, Any] = {"output_dir": run_feedback.get("output_dir", ""), "axes": {}}
        run_outdir = None if plot_outdir is None else Path(str(run_feedback.get("output_dir", "") or ""))
        if run_outdir is not None:
            run_outdir.mkdir(parents=True, exist_ok=True)
        for axis_label, axis_data in axes.items():
            csv_path = Path(str(dict(axis_data or {}).get("csv", "") or ""))
            if not csv_path.exists():
                continue
            axis_artifacts = dict(axis_data or {})
            axis_artifacts.update(
                _plot_axis_feedback_from_csv(
                    plt=plt,
                    csv_path=csv_path,
                    object_id=object_id,
                    axis_label=str(axis_label),
                    law_label=law_label,
                    mode=plot_mode,
                    outdir=run_outdir,
                    metadata=metadata,
                )
            )
            run_entry["axes"][str(axis_label)] = axis_artifacts
        if run_entry["axes"]:
            feedback_artifacts[run_key] = run_entry
            run.setdefault("artifacts", {})["linear_feedback"] = {
                **run_feedback,
                "axes": run_entry["axes"],
            }
    if feedback_artifacts:
        artifacts["linear_feedback_runs"] = feedback_artifacts

    if plot_mode in ("interactive", "both"):
        plt.show()
        plt.close("all")
    return artifacts
