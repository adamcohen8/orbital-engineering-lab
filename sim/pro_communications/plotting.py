"""Standard analyst-facing plots for Pro communications evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim.analysis.coverage_aggregation import ConstellationCoverageResult
from sim.plotting.style import OELArtifactMetadata, get_oel_version, oel_plot_context, role_color, save_oel_figure
from sim.pro_communications.availability import (
    StatisticalAggregateAvailabilityResult,
    StatisticalRainAvailabilityResult,
)
from sim.pro_communications.coverage import ProCommunicationsCoverageResult
from sim.pro_communications.link_budget import ProLinkBudgetResult


def _metadata(study_id: str, artifact_id: str) -> OELArtifactMetadata:
    return OELArtifactMetadata(
        scenario_name=study_id,
        generated_utc="",
        version=get_oel_version(),
        artifact_id=artifact_id,
    )


def _save(figure: plt.Figure, path: Path, metadata: OELArtifactMetadata) -> None:
    figure.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.14)
    save_oel_figure(
        figure,
        path,
        dpi=180,
        metadata=metadata,
        artifact_id=metadata.artifact_id,
        style_name="oel_dark",
    )
    plt.close(figure)


def write_margin_comparison_plot(
    output_path: str | Path,
    *,
    study_id: str,
    title: str,
    links: Mapping[str, ProLinkBudgetResult],
) -> Path:
    path = Path(output_path).expanduser().resolve()
    artifact_id = path.stem
    metadata = _metadata(study_id, artifact_id)
    colors = [role_color("actual"), role_color("desired"), role_color("warning"), "#ad7bee", "#5fd3bc"]
    with oel_plot_context(style_name="oel_dark", metadata=metadata):
        figure, axes = plt.subplots(figsize=(11.0, 5.2))
        for (label, result), color in zip(links.items(), colors, strict=False):
            axes.plot(result.samples.time_s, result.samples.margin_db, linewidth=2.0, label=label, color=color)
        axes.axhline(0.0, color=role_color("warning"), linewidth=1.2, linestyle="--", label="closure threshold")
        axes.set_title(title)
        axes.set_xlabel("Scenario time (s)")
        axes.set_ylabel("Link margin (dB)")
        axes.grid(True, alpha=0.3)
        axes.legend(loc="best", fontsize=9)
        _save(figure, path, metadata)
    return path


def write_coverage_fraction_plot(
    output_path: str | Path,
    *,
    study_id: str,
    coverages: Mapping[str, ProCommunicationsCoverageResult],
) -> Path:
    path = Path(output_path).expanduser().resolve()
    metadata = _metadata(study_id, path.stem)
    colors = [role_color("actual"), role_color("desired"), role_color("warning"), "#ad7bee"]
    with oel_plot_context(style_name="oel_dark", metadata=metadata):
        figure, axes = plt.subplots(figsize=(11.0, 5.2))
        for (label, result), color in zip(coverages.items(), colors, strict=False):
            axes.step(
                result.times_s,
                100.0 * result.instantaneous_covered_fraction,
                where="post",
                linewidth=2.0,
                label=label,
                color=color,
            )
        axes.set_title("RF-Qualified Whole-Earth Cell-Center Coverage")
        axes.set_xlabel("Scenario time (s)")
        axes.set_ylabel("Represented cells closing link (%)")
        axes.grid(True, alpha=0.3)
        axes.legend(loc="best", fontsize=9)
        _save(figure, path, metadata)
    return path


def write_coverage_margin_map(
    output_path: str | Path,
    *,
    study_id: str,
    title: str,
    coverage: ProCommunicationsCoverageResult,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    metadata = _metadata(study_id, path.stem)
    with oel_plot_context(style_name="oel_dark", metadata=metadata):
        figure, axes = plt.subplots(figsize=(11.0, 5.6))
        finite = np.isfinite(coverage.cell_best_margin_db)
        lower = float(np.percentile(coverage.cell_best_margin_db[finite], 2.0))
        upper = float(np.percentile(coverage.cell_best_margin_db[finite], 98.0))
        image = axes.scatter(
            coverage.cell_longitude_deg[finite],
            coverage.cell_geodetic_latitude_deg[finite],
            c=coverage.cell_best_margin_db[finite],
            s=3.0,
            cmap="coolwarm",
            vmin=lower,
            vmax=upper,
            linewidths=0.0,
        )
        axes.set_title(title)
        axes.set_xlabel("Longitude (deg)")
        axes.set_ylabel("WGS84 geodetic latitude (deg)")
        axes.set_xlim(-180.0, 180.0)
        axes.set_ylim(-90.0, 90.0)
        axes.grid(True, alpha=0.18)
        colorbar = figure.colorbar(image, ax=axes, pad=0.02)
        colorbar.set_label("Best sampled margin (dB)")
        figure.subplots_adjust(left=0.07, right=0.92, top=0.88, bottom=0.14)
        save_oel_figure(
            figure,
            path,
            dpi=180,
            metadata=metadata,
            artifact_id=metadata.artifact_id,
            style_name="oel_dark",
        )
        plt.close(figure)
    return path


def write_constellation_coverage_plot(
    output_path: str | Path,
    *,
    study_id: str,
    constellations: Mapping[str, ConstellationCoverageResult],
) -> Path:
    """Write synchronized RF-qualified constellation coverage fractions."""

    path = Path(output_path).expanduser().resolve()
    metadata = _metadata(study_id, path.stem)
    colors = [role_color("actual"), role_color("desired"), role_color("warning"), "#ad7bee"]
    with oel_plot_context(style_name="oel_dark", metadata=metadata):
        figure, axes = plt.subplots(figsize=(11.0, 5.2))
        for (label, result), color in zip(constellations.items(), colors, strict=False):
            axes.step(
                result.times_s,
                100.0 * result.instantaneous_covered_fraction,
                where="post",
                linewidth=2.0,
                label=label,
                color=color,
            )
        axes.set_title("RF-Qualified Constellation Service Coverage")
        axes.set_xlabel("Scenario time (s)")
        axes.set_ylabel("Represented cells closing link (%)")
        axes.grid(True, alpha=0.3)
        axes.legend(loc="best", fontsize=9)
        _save(figure, path, metadata)
    return path


def write_statistical_availability_plot(
    output_path: str | Path,
    *,
    study_id: str,
    result: StatisticalRainAvailabilityResult | StatisticalAggregateAvailabilityResult,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    metadata = _metadata(study_id, path.stem)
    aggregate = isinstance(result, StatisticalAggregateAvailabilityResult)
    with oel_plot_context(style_name="oel_dark", metadata=metadata):
        figure, axes = plt.subplots(figsize=(9.5, 5.2))
        attenuation = result.total_attenuation_db if aggregate else result.rain_attenuation_db
        axes.semilogx(
            result.exceedance_percentage,
            attenuation,
            color=role_color("actual"),
            linewidth=2.2,
            label=("P.618-14 aggregate attenuation" if aggregate else "P.618-14 declared-R0.01 rain attenuation"),
        )
        axes.axhline(
            (result.clear_path_margin_db if aggregate else result.clear_sky_margin_db),
            color=role_color("warning"),
            linestyle="--",
            linewidth=1.4,
            label="clear-path closure margin" if aggregate else "clear-sky closure margin",
        )
        axes.set_title(
            "Ka-Band Aggregate Attenuation Exceedance Curve"
            if aggregate
            else "Ka-Band Rain Attenuation Exceedance Curve"
        )
        axes.set_xlabel("Percentage of time attenuation is exceeded (%)")
        axes.set_ylabel("Aggregate attenuation (dB)" if aggregate else "Rain attenuation (dB)")
        axes.grid(True, which="both", alpha=0.3)
        axes.legend(loc="best", fontsize=9)
        _save(figure, path, metadata)
    return path


__all__ = [
    "write_constellation_coverage_plot",
    "write_coverage_fraction_plot",
    "write_coverage_margin_map",
    "write_margin_comparison_plot",
    "write_statistical_availability_plot",
]
