"""Cross-check OEL joined motion against a Basilisk/MuJoCo weld."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from sim.dynamics.docking import MatedAssembly

SOURCE = "https://avslab.github.io/basilisk/examples/mujoco/scenarioSimpleDocking.html"
MASSES = np.array([13., 21.])
INERTIAS = np.array([np.diag([2., 3., 4.]), np.diag([4., 5., 6.])])
DURATION_S = 2.
SAMPLE_S = .01
NAMES = ("probe", "receiver")
SCENE = """<mujoco model="OEL joined flight comparison">
  <option gravity="0 0 0"><flag contact="disable"/></option>
  <worldbody>
    <body name="probe" pos="0.25 0.1 0.7" quat="0 1 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="13" diaginertia="2 3 4"/>
      <site name="probe_port" pos="0 0 0.4" quat="0 1 0 0"/>
      <site name="probe_com"/>
    </body>
    <body name="receiver" pos="0.1 -0.1 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="21" diaginertia="4 5 6"/>
      <site name="receiver_port" pos="0.15 0.2 0.3"/>
    </body>
  </worldbody>
  <equality>
    <weld name="dock" site1="probe_port" site2="receiver_port" active="false"
          solref="0.015 1" solimp="0.9999 0.9999 0.001"/>
  </equality>
  <actuator>
    <motor name="probe_thruster" site="probe_com" gear="1 0 0 0 0 0"/>
  </actuator>
</mujoco>
"""
METRICS = ("position_m", "velocity_m_s", "attitude_rad", "body_rate_rad_s")
STATE_COLUMNS = ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
                 "q0", "q1", "q2", "q3", "wx_body_rad_s", "wy_body_rad_s", "wz_body_rad_s")
# Absolute limits on maximum vector norms over the stated comparison window.
CASES = {
    "free_rotation": (0., 0., (1e-7, 1e-7, 1e-7, 1e-7)),
    "one_sided_force": (1., 0., (2e-5, 2e-5, 2e-5, 2e-5)),
    "weld_activation": (0., .25, (2e-3, 2e-3, 2e-3, 2e-3)),
}


def initial_pair(case):
    pair = np.array([
        [.25, .1, .7, 0., 0., 0., 0., 1., 0., 0., .22, -.33, -.47],
        [.1, -.1, 0., 0., 0., 0., 1., 0., 0., 0., .22, .33, .47],
    ])
    offsets = pair[:, :3] - np.average(pair[:, :3], axis=0, weights=MASSES)
    pair[:, 3:6] = np.array([.1, .18, -.09]) + np.cross(pair[1, 10:13], offsets)
    if case == "weld_activation":
        pair[:, 3:6] += [[.018, -.014, .011], [-.012, -.006, .019]]
        pair[0, 10:13] += [.015, .02, -.015]
    return pair


def attitudes(rows):
    """Convert scalar-first OEL quaternions to active body-to-inertial rotations."""
    return Rotation.from_quat(np.roll(rows[..., 6:10], -1, axis=-1).reshape(-1, 4))


def errors(a, b):
    differences = (
        np.linalg.norm(a[..., :3] - b[..., :3], axis=-1),
        np.linalg.norm(a[..., 3:6] - b[..., 3:6], axis=-1),
        (attitudes(a).inv() * attitudes(b)).magnitude(),
        np.linalg.norm(a[..., 10:13] - b[..., 10:13], axis=-1),
    )
    return {name: float(np.max(values)) for name, values in zip(METRICS, differences)}


def relative_drift(history):
    receiver = attitudes(history[:, 1])
    offsets = receiver.inv().apply(history[:, 0, :3] - history[:, 1, :3])
    relative = receiver.inv() * attitudes(history[:, 0])
    return {
        "position_m": float(np.linalg.norm(offsets - offsets[0], axis=-1).max()),
        "attitude_rad": float((relative[0].inv() * relative).magnitude().max()),
    }


def run_basilisk(path, pair, step_s, force_n):
    from Basilisk.architecture import messaging
    from Basilisk.simulation import mujoco, svIntegrators
    from Basilisk.utilities import SimulationBaseClass, macros

    simulation = SimulationBaseClass.SimBaseClass()
    process = simulation.CreateNewProcess("dynamics")
    process.addTask(simulation.CreateNewTask("dynamics", macros.sec2nano(step_s)))
    scene = mujoco.MJScene.fromFile(str(path))
    integrator = svIntegrators.svIntegratorRK4(scene)
    scene.setIntegrator(integrator)
    simulation.AddModelToTask("dynamics", scene)

    thruster = messaging.SingleActuatorMsg()
    thruster.write(messaging.SingleActuatorMsgPayload(input=force_n))
    scene.getSingleActuator("probe_thruster").actuatorInMsg.subscribeTo(thruster)
    bodies = [scene.getBody(name) for name in NAMES]
    logs = [body.getCenterOfMass().stateOutMsg.recorder(macros.sec2nano(SAMPLE_S)) for body in bodies]
    for log in logs:
        simulation.AddModelToTask("dynamics", log)
    simulation.InitializeSimulation()
    for body, row in zip(bodies, pair):
        body.setVelocity(row[3:6].tolist())
        body.setAttitudeRate(row[10:13].tolist())
    scene.getEquality("dock").setActive(True)
    simulation.ConfigureStopTime(macros.sec2nano(DURATION_S))
    simulation.ExecuteSimulation()

    times = np.asarray(logs[0].times()) * macros.NANO2SEC
    rows = np.stack([np.column_stack((
        log.r_BN_N, log.v_BN_N, Rotation.from_mrp(log.sigma_BN).as_quat(scalar_first=True),
        log.omega_BN_B,
    )) for log in logs], axis=1)
    if any(not np.array_equal(log.times(), logs[0].times()) for log in logs):
        raise RuntimeError("Basilisk did not return synchronized spacecraft samples.")
    if force_n:
        initial_com_velocity = MASSES @ rows[0, :, 3:6] / MASSES.sum()
        final_com_velocity = MASSES @ rows[-1, :, 3:6] / MASSES.sum()
        if np.linalg.norm(final_com_velocity - initial_com_velocity) < .04:
            raise RuntimeError("The commanded reference force did not accelerate the assembly.")
    if not np.allclose(times, np.arange(round(DURATION_S / SAMPLE_S) + 1) * SAMPLE_S, atol=1e-12, rtol=0):
        raise RuntimeError("Basilisk recorder did not return the expected comparison grid.")
    if max(errors(rows[:1], pair[None, ...]).values()) > 1e-12:
        raise RuntimeError("Basilisk did not start at the shared initial conditions.")
    return times, rows


def run_oel(pair, times, step_s, force_n):
    pair = pair.copy()
    pair[:, :6] /= 1000.
    assembly, state = MatedAssembly.lock(pair, MASSES, INERTIAS)

    def accelerations(_time_s, stage):
        # The reference motor acts at the probe CoM along its body +X.
        # Apply that load through the existing assembly acceleration callback.
        result = np.zeros((2, 3))
        result[0] = attitudes(stage[:1]).apply([force_n, 0., 0.])[0] / MASSES[0] / 1000.
        return result

    substeps = round(SAMPLE_S / step_s)
    rows = [assembly.pair_state(state)]
    for sample_index in range(len(times) - 1):
        for substep in range(substeps):
            start = (sample_index * substeps + substep) * step_s
            state = assembly.step(state, start, step_s, accelerations)
            state[6:10] /= np.linalg.norm(state[6:10])
        rows.append(assembly.pair_state(state))
    history = np.stack(rows)
    history[..., :6] *= 1000.
    return history


def compare(output_dir):
    import Basilisk
    from Basilisk.simulation import _mujoco

    import sim.dynamics.attitude.rigid_body as attitude
    import sim.dynamics.docking as docking
    import sim.dynamics.orbit.integrators as integrators

    output_dir.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    sources = {"runner": __file__, "oel_docking": docking.__file__,
               "oel_rigid_body": attitude.__file__, "oel_integrator": integrators.__file__,
               "basilisk_mujoco_extension": _mujoco.__file__}
    report = {
        "reference_example": SOURCE,
        "basilisk_build": Basilisk.getBuildInfo()["artifact"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "oel_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=repository,
        ).strip() if (repository / ".git").exists() else None,
        "sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in sources.items()},
        "object_order": NAMES,
        "state_columns": STATE_COLUMNS,
        "scene_sha256": hashlib.sha256(SCENE.encode()).hexdigest(),
        "masses_kg": MASSES.tolist(),
        "inertias_kg_m2": INERTIAS.tolist(),
        "duration_s": DURATION_S,
        "sample_interval_s": SAMPLE_S,
        "cases": {},
    }
    histories = {}
    with TemporaryDirectory() as scratch:
        path = Path(scratch) / "scene.xml"
        path.write_text(SCENE)
        for name, (force_n, from_s, limits) in CASES.items():
            pair = initial_pair(name)
            runs = []
            entries = []
            for step_s in (.002, .001):
                time, reference = run_basilisk(path, pair, step_s, force_n)
                actual = run_oel(pair, time, step_s, force_n)
                keep = time >= from_s - 1e-12
                delta = errors(actual[keep], reference[keep])
                entries.append({
                    "step_s": step_s,
                    "maximum_differences": delta,
                    "basilisk_max_relative_pose_drift": relative_drift(reference),
                    "oel_max_relative_pose_drift": relative_drift(actual),
                    "passes": all(delta[key] <= limit for key, limit in zip(METRICS, limits)),
                })
                runs.append((reference, actual))
                history_key = f"{name}_{round(step_s * 1000)}ms"
                histories[f"{history_key}_basilisk"] = reference
                histories[f"{history_key}_oel"] = actual
            histories[f"{name}_time_s"] = time
            coarse, fine = runs
            report["cases"][name] = {
                "compare_from_s": from_s,
                "probe_body_x_force_n": force_n,
                "initial_pair_m_m_s_quaternion_body_rate": pair.tolist(),
                "gates": dict(zip(METRICS, limits)),
                "resolutions": entries,
                "basilisk_halved_step_differences": errors(coarse[0][keep], fine[0][keep]),
                "oel_halved_step_differences": errors(coarse[1][keep], fine[1][keep]),
                "full_window_maximum_differences_at_fine_step": errors(fine[0], fine[1]),
            }
        # Retain a softer weld comparison to distinguish reference constraint
        # compliance from timestep sensitivity without changing the gates.
        path.write_text(SCENE.replace("0.9999 0.9999", "0.999 0.999"))
        pair = initial_pair("free_rotation")
        time, softer = run_basilisk(path, pair, .001, 0.)
        actual = histories["free_rotation_1ms_oel"]
        report["softer_weld_free_rotation"] = {
            "impedance": .999,
            "maximum_differences": errors(actual, softer),
            "basilisk_max_relative_pose_drift": relative_drift(softer),
        }
        histories["free_rotation_1ms_basilisk_softer_weld"] = softer
    report["passes"] = all(
        all(row["passes"] for row in case["resolutions"])
        and all(delta <= case["gates"][key] / 2
                for engine in ("basilisk", "oel")
                for key, delta in case[f"{engine}_halved_step_differences"].items())
        for case in report["cases"].values()
    )
    (output_dir / "scene.xml").write_text(SCENE)
    np.savez_compressed(output_dir / "histories.npz", **histories)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.output_dir)
    for name, case in report["cases"].items():
        print(name, case["resolutions"][-1]["maximum_differences"])
    raise SystemExit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
