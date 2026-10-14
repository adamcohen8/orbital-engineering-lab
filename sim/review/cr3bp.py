"""Experimental CR3BP research figures and movies from frame-checked review rows.

Shared Python/CLI renderer used by the dedicated MCP CR3BP plan/render tools.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.cr3bp import cr3bp_system
from sim.dynamics.orbit.cr3bp_research import (
    cr3bp_jacobi_constant,
    cr3bp_jacobi_diagnostics,
    cr3bp_libration_points,
    cr3bp_zero_velocity_grid,
    transform_cr3bp_state,
)
from sim.review.workspace import ReviewWorkspace

STATE_COLUMNS = [f"{kind}_{axis}_eci_{unit}" for kind, unit in (("pos", "km"), ("vel", "km_s")) for axis in "xyz"]
STATE_SQL = "SELECT time_s, " + ", ".join(STATE_COLUMNS) + " FROM object_state WHERE object_id = ? ORDER BY time_s"


def load_cr3bp_history(workspace: ReviewWorkspace, object_id: str) -> tuple:
    """Fail closed on missing/wrong frames, system, truncation, or duplicate times."""
    metadata = workspace.query(
        "SELECT f.state_frame, p.propagator_family, p.propagator_name FROM object_state_frame f "
        "JOIN object_propagation p USING(object_id) WHERE f.object_id = ?",
        [object_id],
        max_rows=2,
    )
    if len(metadata.rows) != 1:
        raise ValueError("One object with explicit CR3BP frame and propagation metadata is required.")
    row = metadata.rows[0]
    if row["state_frame"] != "cr3bp_rotating" or row["propagator_family"] != "CR3BP":
        raise ValueError("Research plots require cr3bp_rotating evidence; ECI rows cannot be relabeled.")
    name = str(row["propagator_name"])
    if not name.endswith(" CR3BP"):
        raise ValueError("Missing CR3BP system identity.")
    system = cr3bp_system(name.removesuffix(" CR3BP"))
    result = workspace.query(STATE_SQL, [object_id], max_rows=100000, max_vm_steps=10000000)
    if result.truncated or result.row_count < 2:
        raise ValueError("Need 2 to 100000 complete state samples; truncated histories are not plotted.")
    times = np.array([r["time_s"] for r in result.rows], dtype=float)
    states = np.array([[r[c] for c in STATE_COLUMNS] for r in result.rows], dtype=float)
    if not np.isfinite(times).all() or not np.isfinite(states).all() or np.any(np.diff(times) <= 0):
        raise ValueError("History must have finite states and strictly increasing times.")
    return times, states, system


class _ZeroVelocityContours:
    """Reuse the potential mesh and replace only contour artists on updates."""

    def __init__(self, ax, u, v, potential):
        from sim.plotting.style import role_color

        self.color = role_color("warning")
        self.ax, self.u, self.v, self.potential = ax, u, v, potential
        self.artists = []
        self.jacobi = None

    def update(self, jacobi):
        if self.jacobi == jacobi:
            return tuple(self.artists)
        for artist in self.artists:
            artist.remove()
        self.artists = []
        allowed = self.potential - jacobi
        self.artists.append(
            self.ax.contourf(
                self.u,
                self.v,
                (allowed < 0).astype(float),
                levels=[0.5, 1.5],
                colors=[self.color],
                alpha=0.3,
            )
        )
        if np.nanmin(allowed) < 0 < np.nanmax(allowed):
            self.artists.append(
                self.ax.contour(
                    self.u,
                    self.v,
                    allowed,
                    levels=[0],
                    colors=[self.color],
                    linewidths=1.0,
                )
            )
        self.jacobi = jacobi
        return tuple(self.artists)


def render_cr3bp_research(
    workspace: ReviewWorkspace | str | Path,
    object_id: str,
    *,
    kind: str = "trajectory",
    axes: str = "rotating",
    origin: str = "barycenter",
    plane: str = "xy",
    reference_time_s: float = 0.0,
    reference_angle_rad: float = 0.0,
    slice_km: float = 0.0,
    jacobi: float | None = None,
    jacobi_mode: str = "fixed",
    bounds_km: tuple | None = None,
    resolution: int = 301,
    movie_format: str | None = None,
    frames: int = 120,
    fps: int = 20,
    style: str = "oel_dark",
) -> dict:
    """Render trajectory, zero_velocity slice/overlay, or Jacobi-change diagnostic.

    Movie frames select nearest recorded samples at uniform requested times,
    without interpolation or propagation. Exact selected times are recorded.
    """
    if not isinstance(workspace, ReviewWorkspace):
        with ReviewWorkspace.open(workspace) as opened:
            return render_cr3bp_research(
                opened,
                object_id,
                kind=kind,
                axes=axes,
                origin=origin,
                plane=plane,
                reference_time_s=reference_time_s,
                reference_angle_rad=reference_angle_rad,
                slice_km=slice_km,
                jacobi=jacobi,
                jacobi_mode=jacobi_mode,
                bounds_km=bounds_km,
                resolution=resolution,
                movie_format=movie_format,
                frames=frames,
                fps=fps,
                style=style,
            )
    if kind not in {"trajectory", "zero_velocity", "jacobi"} or plane not in {"xy", "xz", "yz"}:
        raise ValueError("Use trajectory/zero_velocity/jacobi and xy/xz/yz.")
    if style not in {"oel_light", "oel_dark"}:
        raise ValueError("Style must be oel_light or oel_dark.")
    if movie_format not in {None, "gif", "mp4"} or not 2 <= frames <= 600 or not 1 <= fps <= 60:
        raise ValueError("Movie format must be gif/mp4; frames 2..600, fps 1..60.")
    if movie_format and kind == "jacobi":
        raise ValueError("Use a static Jacobi diagnostic.")
    if kind == "zero_velocity" and axes != "rotating":
        raise ValueError("Zero-velocity slices require rotating axes.")
    if jacobi_mode not in {"fixed", "instantaneous"}:
        raise ValueError("Jacobi mode must be fixed or instantaneous.")
    if jacobi_mode == "instantaneous":
        if kind != "zero_velocity" or not movie_format:
            raise ValueError("Instantaneous Jacobi mode requires a zero_velocity movie.")
        if jacobi is not None:
            raise ValueError("An explicit fixed Jacobi value conflicts with instantaneous mode.")
    identity = workspace.evidence_identity()
    times, states, system = load_cr3bp_history(workspace, object_id)
    positions = transform_cr3bp_state(
        states,
        times,
        target_axes=axes,
        target_origin=origin,
        reference_time_s=reference_time_s,
        reference_angle_rad=reference_angle_rad,
        system=system,
    )[:, :3]
    c = cr3bp_jacobi_constant(states, system=system)
    if not np.isfinite(c).all():
        raise ValueError("History intersects a singular primary.")
    reference_c = float(c[0] if jacobi is None else jacobi)
    if not np.isfinite(reference_c):
        raise ValueError("Jacobi reference must be finite.")
    i, j = ["xyz".index(a) for a in plane]
    primary_states = np.zeros((2, 6))
    primary_states[:, 0] = np.array([-system.mu, 1 - system.mu]) * system.distance_km
    primary_tracks = [
        transform_cr3bp_state(
            p,
            times,
            target_axes=axes,
            target_origin=origin,
            reference_time_s=reference_time_s,
            reference_angle_rad=reference_angle_rad,
            system=system,
        )[:, :3]
        for p in primary_states
    ]
    if bounds_km is None:
        data = np.concatenate([positions[:, [i, j]], *[p[:, [i, j]] for p in primary_tracks]])
        lo, hi = data.min(axis=0), data.max(axis=0)
        span = max(float(np.max(hi - lo)), 0.05 * system.distance_km)
        center = (lo + hi) / 2
        bounds_km = (center[0] - 0.6 * span, center[0] + 0.6 * span, center[1] - 0.6 * span, center[1] + 0.6 * span)
    bounds = np.asarray(bounds_km, dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("Bounds must be finite increasing u/v limits.")
    from sim.runtime_environment import configure_headless_runtime

    configure_headless_runtime()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    from sim.plotting.quality import STRICT_AGENT_PLOT_QUALITY, apply_plot_quality_policy
    from sim.plotting.style import add_artifact_footer, artifact_metadata, oel_plot_context, role_color, save_oel_figure
    from sim.review.plotting import ReviewPlotArtifact, ReviewPlotSpec, record_generated_artifact

    if movie_format == "mp4" and not FFMpegWriter.isAvailable():
        raise ValueError("MP4 rendering requires FFmpeg; install it or select GIF.")
    artifact_id = "cr3bp_" + kind + "_" + uuid.uuid4().hex[:10]
    folder = workspace.review_dir / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (artifact_id + ".png")
    metadata = artifact_metadata(scenario_name=workspace.output_dir.name, artifact_id=artifact_id)
    provenance = {
        "source_query": STATE_SQL,
        "query_parameters": [object_id],
        "review_store": identity,
        "system": asdict(system),
        "system_parameter_source": "Maintained named-system constants in this OEL installation",
        "axes": axes,
        "origin": origin,
        "units": "km, km/s, seconds",
        "reference_time_s": reference_time_s,
        "reference_angle_rad": reference_angle_rad,
        "inertial_alignment": "CR3BP model axes, not J2000",
        "plane": plane,
        "slice_km": slice_km,
        "jacobi_mode": jacobi_mode,
        "jacobi_reference": reference_c,
        "static_preview_sample_index": 0 if jacobi_mode == "instantaneous" else None,
        "boundary_interpretation": "Instantaneous coast accessibility on the selected slice; continued thrust changes C.",
        "jacobi_reference_source": "first_sample" if jacobi is None else "explicit",
        "bounds_km": list(map(float, bounds)),
        "grid_resolution": resolution,
        "diagnostics": cr3bp_jacobi_diagnostics(states, system=system),
        "visual_qa_status": "pending_agent_review",
        "maturity": "experimental",
    }
    with oel_plot_context(style_name=style, metadata=metadata):
        fig, ax = plt.subplots(figsize=(10, 7))
        if movie_format:
            fig.suptitle("CR3BP research", x=0.02, y=0.985, ha="left", va="top", fontsize=12)
        moving = []
        if kind == "jacobi":
            ax.plot(times, c - c[0])
            change = float(np.max(np.abs(c - c[0])))
            # Stable symmetric limits avoid spurious off-canvas ticks near roundoff.
            exponent = 10.0 ** np.floor(np.log10(max(change, np.finfo(float).eps)))
            limit = np.ceil(max(change, np.finfo(float).eps) / exponent) * exponent
            ax.set_ylim(-limit, limit)
            ax.set(
                xlabel="Elapsed run time (s)",
                ylabel="C(t) - C(first), dimensionless",
                title="Jacobi change — conservation only for unforced CR3BP",
            )
        else:
            if kind == "zero_velocity":
                # Translate the selected-origin grid into barycentric coordinates.
                offset = {"barycenter": 0.0, "p1": -system.mu, "p2": 1 - system.mu}[origin] * system.distance_km
                shift = np.array([offset, 0.0, 0.0])
                b = bounds + np.array([shift[i], shift[i], shift[j], shift[j]])
                k = ({0, 1, 2} - {i, j}).pop()
                u, v, potential = cr3bp_zero_velocity_grid(
                    0.0,
                    plane=plane,
                    slice_km=slice_km + shift[k],
                    bounds_km=tuple(b),
                    resolution=resolution,
                    system=system,
                )
                u, v = u - shift[i], v - shift[j]
                contours = _ZeroVelocityContours(ax, u, v, potential)
                contours.update(reference_c)
                for name, pos in cr3bp_libration_points(system=system).items():
                    pos = pos - shift
                    if (
                        abs(pos[k] - slice_km) < 1e-8
                        and bounds[0] <= pos[i] <= bounds[1]
                        and bounds[2] <= pos[j] <= bounds[3]
                    ):
                        ax.plot(pos[i], pos[j], "+", color=role_color("warning"))
                        ax.annotate(name, (pos[i], pos[j]), xytext=(5, 5), textcoords="offset points")
                ax.set_title(f"Zero-velocity {plane} slice; C = {reference_c:.8f}; fixed {'xyz'[k]} = {slice_km:g} km")
                provenance["overlay_note"] = (
                    "Trajectory is projected; forbidden shading applies only on the labeled slice."
                )
            else:
                ax.set_title(f"CR3BP trajectory — {axes} axes, {origin} origin")
            ax.plot(positions[:, i], positions[:, j], color=role_color("actual"), label=f"{object_id} (projection)")
            ax.plot(positions[0, i], positions[0, j], "o", label="Start", markersize=5, color=role_color("desired"))
            (trail,) = ax.plot([], [], linewidth=2.0, color=role_color("chaser"))
            (marker,) = ax.plot([], [], "o", markersize=5, color=role_color("chaser"))
            for label, p in zip(("P1", "P2"), primary_tracks):
                (artist,) = ax.plot(
                    [p[-1, i]],
                    [p[-1, j]],
                    "o",
                    label=label,
                    markersize=7,
                    color=role_color("actual" if label == "P1" else "target"),
                )
                moving.append((artist, p))
            ax.set(xlabel=f"{plane[0]} (km)", ylabel=f"{plane[1]} (km)", xlim=bounds[:2], ylim=bounds[2:])
            ax.set_aspect("equal", adjustable="box")
            handles, labels = ax.get_legend_handles_labels()
            if kind == "zero_velocity":
                ax.text(
                    0.0,
                    -0.22,
                    f"Rotating axes; {origin} origin. Shading: forbidden on slice.\n"
                    + (
                        "Instantaneous coast boundary; continued thrust changes C. Trajectory projected."
                        if jacobi_mode == "instantaneous"
                        else "Trajectory is projected; it need not lie on this slice."
                    ),
                    transform=ax.transAxes,
                    fontsize=9,
                )
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax.grid(alpha=0.25)
        add_artifact_footer(fig, metadata=metadata, artifact_id=artifact_id)
        fig.tight_layout(rect=(0, 0.05, 1, 0.92 if movie_format else 0.96))
        qa = apply_plot_quality_policy(fig, policy=STRICT_AGENT_PLOT_QUALITY).to_dict()
        save_oel_figure(fig, path, dpi=150, metadata=metadata, artifact_id=artifact_id, style_name=style)
        files = {"figure": str(path)}
        if movie_format:
            targets = np.linspace(times[0], times[-1], frames)
            right = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
            indices = np.where(targets - times[right - 1] <= times[right] - targets, right - 1, right)
            from sim.plotting.animation_quality import (
                animation_time_decimal_places,
                fixed_time_text_width,
                format_animation_time,
            )

            decimals = animation_time_decimal_places(times[indices])
            time_width = fixed_time_text_width(times[indices], decimal_places=decimals)
            timestamp = fig.text(
                0.98, 0.985, "", ha="right", va="top", fontsize=9, family="monospace", gid="oel_animation_time"
            )

            def update(frame: int) -> tuple:
                n = int(indices[frame])
                trail.set_data(positions[: n + 1, i], positions[: n + 1, j])
                marker.set_data([positions[n, i]], [positions[n, j]])
                for artist, p in moving:
                    artist.set_data([p[n, i]], [p[n, j]])
                if jacobi_mode == "instantaneous":
                    contours.update(float(c[n]))
                    ax.set_title(f"Zero-velocity {plane} slice; C(t) = {c[n]:.8f}; fixed {'xyz'[k]} = {slice_km:g} km")
                timestamp.set_text(
                    "Sim time: " + format_animation_time(times[n], decimal_places=decimals, width=time_width) + " s"
                )
                return (trail, marker, timestamp, *[a for a, _ in moving])

            movie = folder / (artifact_id + "." + movie_format)
            from sim.plotting.animation_quality import save_animation_with_quality

            animation = FuncAnimation(fig, update, frames=frames, blit=False)
            report = save_animation_with_quality(
                animation,
                fig,
                movie,
                update=update,
                frame_values=tuple(range(frames)),
                frame_times_s=tuple(times[indices]),
                fps=fps,
                camera_policy="fixed",
                metadata=metadata,
                artifact_id=artifact_id,
                style_name=style,
                format_limits={(0, "x"): tuple(bounds[:2]), (0, "y"): tuple(bounds[2:])},
                source=provenance,
            )
            files.update(
                movie=str(movie),
                contact_sheet=report.contact_sheet_path,
                movie_quality_receipt=report.quality_receipt_path,
            )
            provenance.update(
                movie_frame_times_s=times[indices].tolist(),
                movie_sample_indices=indices.tolist(),
                movie_jacobi_values=(
                    c[indices].tolist() if jacobi_mode == "instantaneous" else [reference_c] * len(indices)
                )
                if kind == "zero_velocity"
                else None,
                fps=fps,
                sampling="Nearest recorded sample to uniform times; no interpolation",
                movie_qa=report.to_dict(),
            )
        plt.close(fig)
    if workspace.evidence_identity() != identity:
        raise ValueError("Review store changed during rendering; generated artifacts are not valid evidence.")
    provenance.update(files=files, presentation_quality=qa)
    from sim.utils.io import sha256_file

    provenance["artifact_sha256"] = {key: sha256_file(Path(value)) for key, value in files.items()}
    receipt = folder / (artifact_id + ".json")
    receipt.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    spec = ReviewPlotSpec(
        sql=STATE_SQL,
        x_column="time_s",
        y_columns=STATE_COLUMNS,
        artifact_id=artifact_id,
        renderer_id="cr3bp_research",
        style_name=style,
        extra=provenance,
    )
    artifact = ReviewPlotArtifact(
        artifact_id,
        path,
        str(path.relative_to(workspace.output_dir)),
        len(times),
        False,
        spec,
        {"presentation_quality": qa, "visual_qa_status": "pending_agent_review"},
    )
    record_generated_artifact(workspace, artifact, review_store_identity=identity)
    return {**files, "receipt": str(receipt), "diagnostics": provenance["diagnostics"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--object", required=True, dest="object_id")
    parser.add_argument("--kind", choices=["trajectory", "zero_velocity", "jacobi"], default="trajectory")
    parser.add_argument("--axes", choices=["rotating", "inertial"], default="rotating")
    parser.add_argument("--origin", choices=["barycenter", "p1", "p2"], default="barycenter")
    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--reference-time-s", type=float, default=0.0)
    parser.add_argument("--reference-angle-rad", type=float, default=0.0)
    parser.add_argument("--slice-km", type=float, default=0.0)
    parser.add_argument("--jacobi", type=float)
    parser.add_argument("--jacobi-mode", choices=["fixed", "instantaneous"], default="fixed")
    parser.add_argument("--bounds-km", type=float, nargs=4)
    parser.add_argument("--resolution", type=int, default=301)
    parser.add_argument("--movie-format", choices=["gif", "mp4"])
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--style", choices=["oel_light", "oel_dark"], default="oel_dark")
    args = vars(parser.parse_args(argv))
    output = args.pop("output_dir")
    print(json.dumps(render_cr3bp_research(output, **args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
