"""Offline planetary numerical qualification with explicit SPICE references.

Run from the repository root with a locally built native extension, SciPy and
SpiceyPy. Supply the exact archived pck00008.tpc and pck00005.tpc as reference
inputs. The IAU 1991 derived kernel uses the documented old Mars values, not
the IAU 1994 current-values block. This is numerical agreement, not atmosphere
prediction or mission qualification. No reference providers run in propagation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import oel_rust_orbit as native
import scipy
import spiceypy as spice
from scipy.integrate import solve_ivp
from scipy.special import lpmv

from sim.config.scenario_yaml import load_simulation_yaml
from sim.dynamics.planetary.preparation import prepare_environment
from sim.dynamics.planetary.propagation import PlanetaryForcePlan
from sim.dynamics.planetary.runtime import initial_truth


def reference_harmonics(g, r):
    h = 0.05

    def potential(r):
        d = np.linalg.norm(r)
        phi = r[2] / d
        lon = math.atan2(r[1], r[0])
        c = g.compiled
        total = 0.0
        for n in range(2, c.n_max + 1):
            for m in range(min(n, c.m_max) + 1):
                norm = math.sqrt(
                    (1.0 if m == 0 else 2.0)
                    * (2 * n + 1)
                    * math.factorial(n - m)
                    / math.factorial(n + m)
                )
                total += (
                    (g.radius_km / d) ** n
                    * norm
                    * (-1.0) ** m
                    * lpmv(m, n, phi)
                    * (
                        c.c_nm[n, m] * math.cos(m * lon)
                        + c.s_nm[n, m] * math.sin(m * lon)
                    )
                )
        return g.mu_km3_s2 / d * total

    return np.array(
        [
            (
                -potential(r + 2 * h * e)
                + 8 * potential(r + h * e)
                - 8 * potential(r - h * e)
                + potential(r - 2 * h * e)
            )
            / (12 * h)
            for e in np.eye(3)
        ]
    )


def qualify(pck2000, pckold, output):
    source = json.loads(
        Path("sim/dynamics/planetary/data/mars_iau2000_rotation.json").read_text()
    )["source"]
    assert hashlib.sha256(pck2000.read_bytes()).hexdigest() == source["sha256"]
    source = json.loads(
        Path("sim/dynamics/planetary/data/mars_iau1991_rotation.json").read_text()
    )["source"]
    assert hashlib.sha256(pckold.read_bytes()).hexdigest() == source["sha256"]
    cfg = load_simulation_yaml("configs/mars_reference_drag.yaml")
    env = prepare_environment(cfg)
    plan = PlanetaryForcePlan(env, mass_kg=100.0, cd=2.2, drag_area_m2=1.0)
    truth = initial_truth(cfg.objects["probe"], env)
    x0 = np.r_[truth.position_eci_km, truth.velocity_eci_km_s]
    initial_tt = env.gravity_frame.initial_tt_s
    with tempfile.TemporaryDirectory(prefix="oel-planetary-reference-") as directory:
        derived = Path(directory) / "iau1991.tpc"
        # A separate NAIF body/frame ID prevents replacing IAU2000_MARS.
        derived.write_text("""KPL/PCK
\\begindata
FRAME_OEL_REFERENCE_MARS1991 = 14991
FRAME_14991_NAME = 'OEL_REFERENCE_MARS1991'
FRAME_14991_CLASS = 2
FRAME_14991_CLASS_ID = 14991
FRAME_14991_CENTER = 499
BODY14991_POLE_RA = (317.681 -0.108 0.)
BODY14991_POLE_DEC = (52.886 -0.061 0.)
BODY14991_PM = (176.868 350.8919830 0.)
\\begintext
Derived from the IAU 1991 old-values section of the supplied pck00005.tpc.
""")
        spice.furnsh(str(pck2000))
        spice.furnsh(str(derived))
        frame_errors = []
        force_errors = []

        def transforms(t):
            et = env.gravity_frame.elapsed_tdb_s(t)
            # Convert the independently evaluated SPICE derivative dR/dTDB to
            # dR/dTAI using the documented approximation's analytic chain rule.
            g = np.deg2rad(357.53 + 0.9856003 * (initial_tt + t) / 86400.0)
            rate = 1.0 + np.deg2rad(0.9856003) / 86400.0 * (
                0.001657 * np.cos(g) + 0.000044 * np.cos(2 * g)
            )
            gravity = spice.pxform("J2000", "OEL_REFERENCE_MARS1991", et)
            air = spice.sxform("J2000", "IAU_MARS", et)
            return gravity, air[:3, :3], air[3:, :3] * rate

        def acceleration(t, x):
            r = x[:3]
            v = x[3:]
            radius = np.linalg.norm(r)
            G, A, D = transforms(t)
            rb = A @ r
            a = 3396.19
            b = 3376.2
            e2 = 1.0 - (b / a) ** 2
            xy = np.hypot(rb[0], rb[1])
            phi = np.arctan2(rb[2], xy * (1.0 - e2))
            for _ in range(12):
                N = a / math.sqrt(1.0 - e2 * math.sin(phi) ** 2)
                phi = math.atan2(rb[2] + e2 * N * math.sin(phi), xy)
            N = a / math.sqrt(1.0 - e2 * math.sin(phi) ** 2)
            height = (
                xy * math.cos(phi)
                + rb[2] * math.sin(phi)
                - N * (1.0 - e2 * math.sin(phi) ** 2)
            )
            assert env.atmosphere.heights[0] <= height <= env.atmosphere.heights[-1]
            rho = math.exp(
                np.interp(
                    height, env.atmosphere.heights, np.log(env.atmosphere.densities)
                )
            )
            rel = A.T @ (A @ v + D @ r)
            return (
                -env.gravity.mu_km3_s2 * r / radius**3
                + G.T @ reference_harmonics(env.gravity, G @ r)
                - 0.5 * rho * 2.2 / 100.0 * np.linalg.norm(rel) * 1000.0 * rel
            )

        for t in (0.0, 60.0, 120.0):
            G, A, D = transforms(t)
            g, d = env.gravity_frame.evaluate(t)
            a, ad = env.atmosphere_frame.evaluate(t)
            frame_errors.append(
                {
                    "t_s": t,
                    "gravity_rotation_max_abs": float(np.max(abs(G - g))),
                    "air_rotation_max_abs": float(np.max(abs(A - a))),
                    "air_derivative_max_abs_s": float(np.max(abs(D - ad))),
                }
            )
            force_errors.append(
                float(np.max(abs(plan.stage(t, x0)[0] - acceleration(t, x0))))
            )
        ref = solve_ivp(
            lambda t, x: np.r_[x[3:], acceleration(t, x)],
            (0.0, 120.0),
            x0,
            method="DOP853",
            rtol=2e-12,
            atol=2e-13,
        )
        assert ref.success
        expected = ref.y[:, -1]
        trajectories = []
        for method, dt in (
            ("rkf78", 120.0),
            ("dopri5", 120.0),
            ("rk4", 2.0),
            ("rk4", 1.0),
        ):
            x = x0.copy()
            for t in np.arange(0.0, 120.0, dt):
                x = np.array(
                    plan.native.propagate(x.tolist(), t, dt, method, 1e-12, 1e-12)[0]
                )
            pos = float(np.linalg.norm(x[:3] - expected[:3]) * 1000.0)
            vel = float(np.linalg.norm(x[3:] - expected[3:]) * 1000.0)
            assert pos < 0.003 and vel < 3e-6
            trajectories.append(
                {
                    "method": method,
                    "dt_s": dt,
                    "position_error_m": pos,
                    "velocity_error_m_s": vel,
                    "state": x.tolist(),
                }
            )
        assert max(force_errors) < 3e-11
        assert max(x["gravity_rotation_max_abs"] for x in frame_errors) < 3e-11
        assert max(x["air_rotation_max_abs"] for x in frame_errors) < 3e-11
        assert max(x["air_derivative_max_abs_s"] for x in frame_errors) < 3e-15
        receipt = {
            "environment": env.metadata(),
            "host": platform.platform(),
            "python": sys.version,
            "native_version": native.__version__,
            "native_sha256": hashlib.sha256(
                Path(native.__file__).read_bytes()
            ).hexdigest(),
            "scipy": scipy.__version__,
            "spiceypy": spice.__version__,
            "cspice": spice.tkvrsn("TOOLKIT"),
            "reference_kernels": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (pck2000, pckold)
            },
            "frame_errors": frame_errors,
            "force_max_abs_errors_km_s2": force_errors,
            "reference_solver": "SciPy DOP853 with SPICE rotations, independent associated Legendre potential gradient and independent atmosphere interpolation",
            "arc_s": 120.0,
            "trajectories": trajectories,
            "passed": True,
            "limits": "Numerical agreement for static 20x20 GMM-2B and one bounded fixed MAVEN profile; no weather prediction, calibrated Cd, GMM-3, mission truth or flight qualification.",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        spice.kclear()
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pck2000", type=Path, required=True)
    parser.add_argument("--pck1991-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.pck2000, args.pck1991_source, args.output)
    print(
        json.dumps(
            {"passed": result["passed"], "trajectories": result["trajectories"]},
            indent=2,
        )
    )
