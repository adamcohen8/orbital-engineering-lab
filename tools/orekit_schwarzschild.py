"""Regenerate a bounded Schwarzschild comparison against pinned Orekit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.pro_perturbations.schwarzschild import SchwarzschildAcceleration, schwarzschild_acceleration

JAVA = Path(__file__).resolve().parent / "external_reference" / "OrekitSchwarzschild.java"
THRESHOLDS = {"force_m_s2": 1e-22, "position_m": 0.002, "incremental_m": 1e-5}
JARS = {
    "orekit-13.1.7.jar": "43ab5155fd327ca39287caa5440d9ae50473d911eda1b7f0e819197dcfbf63b3",
    "hipparchus-core-4.0.3.jar": "5af54dc3c21b8d85e1264157ee91d93fa67e68c825cfb38834dc2bbb9b0f6583",
    "hipparchus-filtering-4.0.3.jar": "279eece43411f372bd8340355cb89101def0bf9b1b6824231b76181d1ec63a81",
    "hipparchus-fitting-4.0.3.jar": "3712c3b4f43d9c12bf7b7aad187764f4b3c09cc962504899d8951eb310b1e4be",
    "hipparchus-geometry-4.0.3.jar": "19594d5106ec6a46951e859e2924bfcade02a834c06c56bd3d2130a74f0fbb7e",
    "hipparchus-ode-4.0.3.jar": "75f0f0b3fdcd99d13e9c394b7ae809cce5a9c22f1c14b1b90a5109ed951467fe",
    "hipparchus-optim-4.0.3.jar": "4bc0dbb5c6d5fefda5db9d92141c6cbb807174a1d491a85df46bcc57a67b8992",
    "hipparchus-stat-4.0.3.jar": "8b066e3060dc119efcce241fb0720a3b9e59c026d0f5fe9e70abfd5e22d3817b"
}


def compare(directory):
    fixed = np.genfromtxt(Path(directory) / "fixed.csv", delimiter=",", names=True)
    force_errors = []
    for row in fixed:
        state = np.array([row[k] for k in ("x", "y", "z", "vx", "vy", "vz")]) / 1000
        actual = schwarzschild_acceleration(state, 398600.4418) * 1000
        reference = np.array([row[k] for k in ("ax", "ay", "az")])
        force_errors.append(float(np.linalg.norm(actual - reference)))
    groups = {}
    with (Path(directory) / "trajectory.csv").open() as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(row["case"], []).append(row)
    results, states = {}, {}
    for name, rows in groups.items():
        reference = np.array([[float(row[k]) for k in ("x", "y", "z", "vx", "vy", "vz")] for row in rows])
        times = np.array([float(row["time_s"]) for row in rows])
        if not np.array_equal(times, np.arange(0, 21601, 300)):
            raise ValueError("Unexpected reference cadence.")
        prop = OrbitPropagator(
            model="two_body",
            integrator="rkf78",
            plugins=[SchwarzschildAcceleration()] if name.endswith("relativity") else [],
            adaptive_atol=1e-13,
            adaptive_rtol=1e-14,
        )
        x = reference[0] / 1000
        actual = [x.copy() * 1000]
        for t in times[:-1]:
            x = prop.propagate(x, 300, t, np.zeros(3), {}, OrbitContext(398600.4418, 300))
            actual.append(x * 1000)
        actual = np.array(actual)
        states[name] = (actual, reference)
        results[name] = float(np.max(np.linalg.norm(actual[:, :3] - reference[:, :3], axis=1)))
    signals, incremental = {}, {}
    for i in range(3):
        on, off = states[f"{i}_relativity"], states[f"{i}_baseline"]
        delta, refdelta = on[0] - off[0], on[1] - off[1]
        signals[str(i)] = float(np.max(np.linalg.norm(refdelta[:, :3], axis=1)))
        incremental[str(i)] = float(np.max(np.linalg.norm((delta - refdelta)[:, :3], axis=1)))
    if len(fixed) != 6 or len(groups) != 6:
        raise ValueError("Expected six force states and six trajectories.")
    return {
        "force_error_m_s2": max(force_errors),
        "position_errors_m": results,
        "incremental_errors_m": incremental,
        "relativistic_displacement_m": signals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jars", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jdk-bin", type=Path)
    args = parser.parse_args()
    jars = [args.jars.resolve() / name for name in JARS]
    executables = []
    for tool in ("java", "javac"):
        executable = str((args.jdk_bin / tool).resolve()) if args.jdk_bin else shutil.which(tool)
        try:
            version = subprocess.run([executable or tool, "-version"], capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            print(json.dumps({"status": "unavailable", "reason": f"{tool} from JDK 11.0.25 is required"}))
            return 2
        if "11.0.25" not in (version.stdout + version.stderr):
            print(json.dumps({"status": "unavailable", "reason": f"{tool} must be from JDK 11.0.25"}))
            return 2
        executables.append(executable or tool)
    for jar in jars:
        if not jar.is_file() or hashlib.sha256(jar.read_bytes()).hexdigest() != JARS[jar.name]:
            print(json.dumps({"status": "unavailable", "reason": f"Missing or mismatched jar: {jar.name}"}))
            return 2

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    classpath = os.pathsep.join(map(str, jars))
    java, javac = executables
    with tempfile.TemporaryDirectory(prefix="oel-orekit-schwarzschild-") as temporary:
        subprocess.run([javac, "-cp", classpath, "-d", temporary, str(JAVA)], check=True)
        subprocess.run(
            [java, "-cp", os.pathsep.join((temporary, classpath)), "OrekitSchwarzschild",
             str(output / "fixed.csv"), str(output / "trajectory.csv")],
            check=True,
        )
    result = compare(output)
    passed = (
        result["force_error_m_s2"] <= THRESHOLDS["force_m_s2"]
        and max(result["position_errors_m"].values()) <= THRESHOLDS["position_m"]
        and max(result["incremental_errors_m"].values()) <= THRESHOLDS["incremental_m"]
    )
    report = {
        "status": "passed" if passed else "failed",
        "reference": "Orekit 13.1.7, stock Relativity, GCRF, J2000 TT",
        "thresholds": THRESHOLDS,
        "input_jar_sha256": JARS,
        **result,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (output / "comparison.json").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
