from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim.plotting.style import OELArtifactMetadata, artifact_metadata, save_oel_figure, style_name_from_config
from sim.utils.figure_size import cap_figsize


class MonteCarloRelativeRangePlotWriter:
    """Bounded parent-process writer for campaign relative-range time series."""

    def __init__(
        self,
        *,
        outdir: Path,
        save: bool,
        show: bool,
        dpi: int = 150,
        max_runs: int = 200,
        max_points_per_run: int = 1000,
        style_name: str = "oel_dark",
        metadata: OELArtifactMetadata | None = None,
    ) -> None:
        self.outdir = Path(outdir)
        self.save = bool(save)
        self.show = bool(show)
        self.dpi = int(dpi)
        self.max_runs = int(max(max_runs, 0))
        self.max_points_per_run = int(max(max_points_per_run, 2))
        self.style_name = str(style_name or "oel_dark")
        self.metadata = metadata or artifact_metadata(scenario_name="monte_carlo")
        self._fig: Any | None = None
        self._ax: Any | None = None
        self._plt: Any | None = None
        self.plotted_runs = 0
        self.missing_checkpoint_runs: list[int] = []

    @classmethod
    def from_config(
        cls,
        *,
        outdir: Path,
        plots_cfg: dict[str, Any],
        mc_out_cfg: dict[str, Any],
        scenario_name: str = "",
    ) -> MonteCarloRelativeRangePlotWriter:
        save_hist = bool(mc_out_cfg.get("save_histograms", False))
        show_hist = bool(mc_out_cfg.get("display_histograms", False))
        save = bool(mc_out_cfg.get("save_relative_range_timeseries", save_hist))
        show = bool(mc_out_cfg.get("display_relative_range_timeseries", show_hist))
        return cls(
            outdir=outdir,
            save=save,
            show=show,
            dpi=int(plots_cfg.get("dpi", 150)),
            max_runs=int(mc_out_cfg.get("relative_range_max_runs", 200) or 200),
            max_points_per_run=int(mc_out_cfg.get("relative_range_max_points_per_run", 1000) or 1000),
            style_name=style_name_from_config(plots_cfg),
            metadata=artifact_metadata(scenario_name=str(scenario_name or "monte_carlo")),
        )

    @property
    def enabled(self) -> bool:
        return bool((self.save or self.show) and self.max_runs > 0)

    def should_collect_iteration(self, iteration: int) -> bool:
        return bool(self.enabled and int(iteration) < self.max_runs)

    def add_run(self, *, iteration: int, series: dict[str, Any] | None) -> None:
        if not self.should_collect_iteration(iteration) or not isinstance(series, dict):
            return
        t_rr = np.array(series.get("time_s", []), dtype=float).reshape(-1)
        r_rr = np.array(series.get("range_km", []), dtype=float).reshape(-1)
        n = int(min(t_rr.size, r_rr.size))
        if n <= 0:
            return
        t_rr = t_rr[:n]
        r_rr = r_rr[:n]
        finite_mask = np.isfinite(t_rr) & np.isfinite(r_rr)
        if not np.any(finite_mask):
            return
        t_rr, r_rr = self._downsample(t_rr[finite_mask], r_rr[finite_mask])
        self._ensure_figure()
        assert self._ax is not None
        self._ax.plot(t_rr, r_rr, linewidth=1.0, alpha=0.65, label=f"run {int(iteration)}")
        self.plotted_runs += 1

    def mark_checkpoint_omitted(self, iteration: int) -> None:
        if self.should_collect_iteration(iteration):
            self.missing_checkpoint_runs.append(int(iteration))

    def finalize(self) -> dict[str, str]:
        if self.missing_checkpoint_runs:
            if self._fig is not None and self._plt is not None:
                self._plt.close(self._fig)
            self._fig = None
            self._ax = None
            return {
                "relative_range_timeseries_status": (
                    "skipped_incomplete_resume:"
                    + ",".join(str(i) for i in sorted(set(self.missing_checkpoint_runs)))
                )
            }
        if self._fig is None or self._ax is None or self._plt is None or self.plotted_runs <= 0:
            return {}
        self._ax.set_title("Chaser-Target Relative Range by Iteration")
        self._ax.set_xlabel("Time (s)")
        self._ax.set_ylabel("Range (km)")
        self._ax.grid(True, alpha=0.3)
        self._fig.tight_layout()
        artifacts: dict[str, str] = {}
        if self.save:
            path = self.outdir / "master_monte_carlo_relative_range_timeseries.png"
            save_oel_figure(
                self._fig,
                path,
                dpi=self.dpi,
                metadata=self.metadata,
                artifact_id=path.stem,
                style_name=self.style_name,
            )
            artifacts["relative_range_timeseries_png"] = str(path)
        if self.show:
            self._plt.show()
        self._plt.close(self._fig)
        self._fig = None
        self._ax = None
        self.missing_checkpoint_runs = []
        return artifacts

    def reset(self) -> None:
        if self._fig is not None and self._plt is not None:
            self._plt.close(self._fig)
        self._fig = None
        self._ax = None
        self.plotted_runs = 0

    def _ensure_figure(self) -> None:
        if self._fig is not None and self._ax is not None:
            return
        import matplotlib.pyplot as plt

        self._plt = plt
        self._fig, self._ax = plt.subplots(figsize=cap_figsize(10, 6))

    def _downsample(self, t_rr: np.ndarray, r_rr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if t_rr.size <= self.max_points_per_run:
            return t_rr, r_rr
        idx = np.linspace(0, t_rr.size - 1, num=self.max_points_per_run, dtype=int)
        return t_rr[idx], r_rr[idx]
