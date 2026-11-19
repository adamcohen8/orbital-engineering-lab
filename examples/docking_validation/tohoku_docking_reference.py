"""Correlate OEL's normal contact model with Zebenay et al., Figure 13(b)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sim.config.scenario.docking import normalize_docking
from sim.core.models import StateTruth
from sim.execution.docking import DockingStepper
from sim.execution.object_workers import ObjectStepInput

PDF_SHA256 = "5b704cd3ff56d4463c642a73a52a59babe8535ea1e6740d1c9818a1abcb08761"
SOURCE = "https://arxiv.org/pdf/1309.3512#page=14"
MASS_KG = 63.2
STIFFNESS_N_M = 1000.
DAMPING_N_S_M = 47.5
CLOSING_M_S = .020
REBOUND_M_S = .015
PEAK_N = 5.
THRESHOLD_N = .5
# The PDF stores this figure as calibrated vector paths. Inclusive path start
# and exclusive end indices identify the strokes in the SHA-pinned page stream.
FORCE_STROKES = ((3370, 3576), (3066, 3366), (2762, 3062))
SPEED_STROKES = ((4074, 4171), (4175, 4200))


def extract_traces(pdf: Path):
    from pypdf import PdfReader

    if sha256(pdf) != PDF_SHA256:
        raise ValueError("Reference PDF does not match the pinned arXiv v1 file.")
    ops = PdfReader(pdf).pages[13].get_contents().operations

    def trace(strokes, y_zero, y_tick, increment):
        points = {}
        for start, end in strokes:
            for args, op in ops[start:end]:
                if op not in (b"m", b"l"):
                    raise ValueError("Unexpected PDF path operator.")
                x, y = map(float, args)
                points[x] = (y - y_zero) * increment / (y_tick - y_zero)
        x = np.array(sorted(points))
        times = 2.8 + (x - 3831.32) * 1.2 / (5687.24 - 3831.32)
        return times, np.array([points[k] for k in x])

    # Force is plotted compressive-negative. Flip the sign for its magnitude
    # convention while preserving baseline oscillation. Velocity is plotted
    # about zero pre-impact and becomes negative as the free body accelerates.
    force = trace(FORCE_STROKES, -3107.91, -2921.12, 2.)
    speed = trace(SPEED_STROKES, -2157.50, -2021.43, -.010)
    return force, speed


class AirBearingStepper(DockingStepper):
    """Bench boundary condition for a force-free body and a prescribed receiver."""

    def _gravity(self, time_s, pair):
        acceleration = np.zeros((2, 3))
        if self.phase == "contact":
            geometry = self.interface.geometry(pair)
            forces, *_ = self.interface.wrenches(pair, geometry, captured=False)
            # In the constant-speed target's frame the receiver is stationary.
            # Its drive supplies the opposing force at every integration stage.
            acceleration[1] = -forces[1] / self.masses[1] / 1000.
        return acceleration


def simulate(step_s):
    # Four coaxial sites are in parallel. Axial symmetry makes their torques
    # cancel. Placeholder receiver inertia/mass have no effect on this case.
    config = normalize_docking({
        "enabled": True,
        "probe": {"object_id": "probe", "offset_body_m": [0, 0, 0], "radius_m": .1},
        "receiver": {"object_id": "receiver", "offset_body_m": [0, 0, 0], "radius_m": .2},
        "max_step_s": step_s,
        "contact_stiffness_n_m": STIFFNESS_N_M / 4,
        "contact_damping_n_s_m": DAMPING_N_S_M / 4,
        "max_compression_m": .02,
    })
    states = {
        "probe": StateTruth(np.array([0., 0., 2e-6]), np.array([0., 0., -CLOSING_M_S / 1000]),
                            np.array([0., 1., 0., 0.]), np.zeros(3), MASS_KG, 0.),
        "receiver": StateTruth(np.zeros(3), np.zeros(3), np.array([1., 0., 0., 0.]), np.zeros(3), 1., 0.),
    }
    agents = {oid: SimpleNamespace(truth=state, dynamics=SimpleNamespace(
        inertia_kg_m2=np.eye(3), mu_km3_s2=0., area_m2=1., cd=2.2, cr=1.2,
        orbit_substep_s=None, attitude_substep_s=None,
    )) for oid, state in states.items()}
    stepper = AirBearingStepper(config, agents, {}, step_s)
    for index in range(round(1.2 / step_s)):
        start, end = index * step_s, (index + 1) * step_s
        results = stepper.step_objects([
            ObjectStepInput(oid, agents[oid], states[oid], states, start, end, index)
            for oid in states
        ])
        states = {result.object_id: result.truth for result in results}
    if [row["event"] for row in stepper.events] != ["first_contact", "rebound"]:
        raise ValueError("Expected one centered impact followed by rebound.")
    receiver = states["receiver"]
    if (np.linalg.norm(receiver.position_eci_km) > 1e-13
            or np.linalg.norm(receiver.velocity_eci_km_s) > 1e-13):
        raise ValueError("Prescribed receiver drifted.")
    first, rebound = stepper.events
    times, forces, speeds = np.array([
        [row["time_s"], row["normal_load_n"], row["normal_speed_m_s"]]
        for row in stepper.samples
    ]).T
    return times - first["time_s"], forces, speeds, rebound["time_s"] - first["time_s"]


def threshold_crossings(t, y, threshold):
    inside = np.flatnonzero(y >= threshold)
    if not len(inside) or inside[0] == 0 or inside[-1] == len(t) - 1:
        raise ValueError("Force trace must bracket contact at both ends.")
    left, right = inside[0], inside[-1]
    return (float(np.interp(threshold, y[left-1:left+1], t[left-1:left+1])),
            float(np.interp(threshold, y[right:right+2][::-1], t[right:right+2][::-1])))


def write_trace(path, columns, *arrays):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*arrays))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(pdf, output):
    output.mkdir(parents=True, exist_ok=False)
    (force_t, raw_force), (speed_t, raw_speed) = extract_traces(pdf)
    baseline_n = float(np.mean(raw_force[force_t < 2.98]))
    baseline_m_s = float(np.mean(raw_speed[speed_t < 2.98]))
    force = raw_force - baseline_n
    # The plotted body accelerates in the negative direction. OEL reports
    # positive opening speed in the constant-speed target's frame.
    signed_speed = -(raw_speed - baseline_m_s) - CLOSING_M_S
    begin, end = threshold_crossings(force_t, force, THRESHOLD_N)
    # Use the same source-clock window for impulse and overlays. A small
    # allowance includes low-force tails without integrating the whole plot.
    grid = np.linspace(-.050, .850, 901)
    reference_force = np.interp(grid + begin, force_t, force)
    reference_speed = np.interp(grid + begin, speed_t, signed_speed)
    reference_impulse = float(np.trapezoid(reference_force, grid))
    repository = Path(__file__).resolve().parents[2]
    report = {
        "status": "correlation_only",
        "oel_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        ).strip() if (repository / ".git").exists() else None,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "reference": {
            "source": SOURCE, "section": "3.4, Table 3 and Figure 13(b)",
            "pdf_sha256": PDF_SHA256,
            "raw_reference": "Vector digitization of published curves; raw telemetry unavailable.",
            "stiffness_n_m": STIFFNESS_N_M, "damping_n_s_m_fitted_by_authors": DAMPING_N_S_M,
            "moving_body_mass_kg": MASS_KG, "closing_speed_m_s": CLOSING_M_S,
            "rebound_speed_m_s": REBOUND_M_S, "restitution": REBOUND_M_S / CLOSING_M_S,
            "table_peak_force_n": PEAK_N, "trace_peak_force_n": float(force.max()),
            "peak_force_time_from_onset_s": float(force_t[np.argmax(force)] - begin),
            "force_baseline_n": baseline_n, "velocity_baseline_m_s": baseline_m_s,
            "force_threshold_n": THRESHOLD_N, "force_threshold_onset_plot_time_s": begin,
            "force_threshold_duration_s": end - begin,
            "trace_impulse_n_s": reference_impulse,
            "momentum_inferred_impulse_n_s": MASS_KG * (CLOSING_M_S + REBOUND_M_S),
            "trace_impulse_window_from_onset_s": [float(grid[0]), float(grid[-1])],
        },
        "setup": "One freely translating body, one prescribed constant-speed boundary. "
                 "OEL paired stepper and per-site force law with stiffness and damping divided across four sites.",
        "source_sha256": {p: sha256(repository / p) for p in (
            Path(__file__).resolve().relative_to(repository).as_posix(), "sim/execution/docking.py",
            "sim/dynamics/docking.py", "sim/dynamics/orbit/integrators.py",
            "sim/dynamics/attitude/rigid_body.py", "sim/config/scenario/docking.py",
        )},
        "runs": [],
    }
    write_trace(output / "figure13b_force.csv", ("source_time_s", "force_n"), force_t, raw_force)
    write_trace(output / "figure13b_velocity.csv", ("source_time_s", "plotted_velocity_m_s"), speed_t, raw_speed)
    for step in (.002, .001, .0005):
        t, f, v, geometric_duration = simulate(step)
        start, stop = threshold_crossings(t, f, THRESHOLD_N)
        predicted_force = np.interp(grid + start, t, f, left=0., right=0.)
        predicted_speed = np.interp(grid + start, t, v)
        active_window = (grid >= 0) & (grid <= end - begin)
        impulse = float(np.trapezoid(f, t))
        run = {
            "step_s": step, "peak_force_n": float(f.max()),
            "rebound_speed_m_s": float(v[-1]), "restitution": float(v[-1] / CLOSING_M_S),
            "force_threshold_duration_s": stop - start, "geometric_contact_duration_s": geometric_duration,
            "impulse_n_s": impulse,
            "momentum_balance_residual_n_s": impulse - MASS_KG * (v[-1] + CLOSING_M_S),
            "peak_force_time_from_onset_s": float(t[np.argmax(f)] - start),
            "force_trace_rmse_n": float(np.sqrt(np.mean((predicted_force - reference_force)[active_window] ** 2))),
            "velocity_trace_rmse_m_s": float(np.sqrt(np.mean((predicted_speed - reference_speed)[active_window] ** 2))),
        }
        report["runs"].append(run)
        write_trace(output / f"oel_{step * 1000:g}ms.csv",
                    ("time_since_contact_s", "force_n", "signed_normal_speed_m_s"), t, f, v)
    fine = report["runs"][-1]
    middle = report["runs"][-2]
    report["fine_minus_reference"] = {
        "peak_force_vs_table_n": fine["peak_force_n"] - PEAK_N,
        "peak_force_vs_trace_n": fine["peak_force_n"] - float(force.max()),
        "peak_force_time_s": fine["peak_force_time_from_onset_s"] - report["reference"]["peak_force_time_from_onset_s"],
        "rebound_speed_m_s": fine["rebound_speed_m_s"] - REBOUND_M_S,
        "force_threshold_duration_s": fine["force_threshold_duration_s"] - (end - begin),
        "trace_impulse_n_s": fine["impulse_n_s"] - reference_impulse,
    }
    report["force_duration_threshold_sensitivity"] = []
    for threshold in (.4, .5, .6):
        measured_start, measured_stop = threshold_crossings(force_t, force, threshold)
        simulated_start, simulated_stop = threshold_crossings(t, f, threshold)
        report["force_duration_threshold_sensitivity"].append({
            "threshold_n": threshold,
            "reference_duration_s": measured_stop - measured_start,
            "oel_duration_s": simulated_stop - simulated_start,
        })
    report["fine_minus_middle_step"] = {key: fine[key] - middle[key] for key in fine if key != "step_s"}
    write_plot(output / "comparison.png", grid, reference_force, predicted_force, reference_speed, predicted_speed)
    write_trace(output / "comparison_traces.csv",
                ("time_since_onset_s", "reference_force_n", "oel_force_n", "reference_signed_speed_m_s",
                 "oel_signed_speed_m_s"),
                grid, reference_force, predicted_force, reference_speed, predicted_speed)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def write_plot(path, time, measured_force, oel_force, measured_speed, oel_speed):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8.8, 6.6), layout="constrained")
    FigureCanvasAgg(figure)
    force, speed = figure.subplots(2, 1, sharex=True)
    figure.suptitle("Tohoku air-bearing impact", fontsize=16, fontweight="bold", x=.065, ha="left")
    for axis, measured, simulated, label in (
        (force, measured_force, oel_force, "Normal force (N)"),
        (speed, 1000 * measured_speed, 1000 * oel_speed, "Normal speed (mm/s)"),
    ):
        axis.plot(time, measured, color="#667889", linewidth=1.25, label="Tohoku, digitized Figure 13(b)")
        axis.plot(time, simulated, color="#006f70", linewidth=2., label="OEL, 0.5 ms step")
        axis.axhline(0, color="#aab5bf", linewidth=.8, zorder=0)
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#e2e8ef", linewidth=.75)
        axis.spines[["top", "right"]].set_visible(False)
    force.legend(loc="upper right", frameon=False, fontsize=9)
    speed.set_xlabel("Time since force first reaches 0.5 N (s)")
    speed.set_xlim(time[0], time[-1])
    figure.savefig(path, dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference_pdf, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
