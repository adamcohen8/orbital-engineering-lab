import unittest
from unittest.mock import patch

import numpy as np

from sim.acceleration.kernels.srp import srp_acceleration_kernel
from sim.dynamics.orbit.eclipse import srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM, SUN_RADIUS_KM
from sim.dynamics.orbit.epoch import AU_KM


class OrbitEclipseTests(unittest.TestCase):
    def test_day_side_full_light(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0])}
        r = np.array([6878.137, 0.0, 0.0])  # day side
        f = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
        self.assertGreater(f, 0.999)

    def test_day_side_conical_shadow_fast_path_skips_angular_disk_math(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "conical"}
        r = np.array([6878.137, 0.0, 0.0])
        with patch("numpy.arccos", side_effect=AssertionError("dayside should not need angular disk math")):
            f = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
        self.assertAlmostEqual(f, 1.0)

    def test_night_side_umbra(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0])}
        r = np.array([-6878.137, 0.0, 0.0])  # anti-sun side
        f = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
        self.assertLess(f, 1e-6)

    def test_penumbra_partial(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0])}
        # Around Earth apparent limb from LEO, expect transition zone.
        delta = np.deg2rad(68.0)
        rmag = 6878.137
        r = rmag * np.array([-np.cos(delta), np.sin(delta), 0.0])
        f = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
        self.assertGreater(f, 0.0)
        self.assertLess(f, 1.0)

    def test_penumbra_uses_finite_solar_disc_overlap_and_matches_compiled_kernel(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "conical"}
        rmag = 6878.137
        factors = []
        for angle_deg in (67.8, 68.0, 68.2):
            angle = np.deg2rad(angle_deg)
            r = rmag * np.array([-np.cos(angle), np.sin(angle), 0.0])
            factor = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
            factors.append(factor)
            sun = np.array([AU_KM, 0.0, 0.0])
            rho = sun - r
            rho_norm = float(np.linalg.norm(rho))
            geometry_sun_dir = rho / rho_norm
            distance_scale = (AU_KM / rho_norm) ** 2
            compiled = srp_acceleration_kernel(
                r,
                sun,
                1.0,
                1.0,
                1.0,
                1.0,
                AU_KM,
                EARTH_RADIUS_KM,
                SUN_RADIUS_KM,
                2,
            )
            expected = -(distance_scale / 1.0e3) * factor * geometry_sun_dir
            np.testing.assert_allclose(compiled, expected, rtol=0.0, atol=1.0e-15)

        self.assertAlmostEqual(factors[-1], 0.892834, places=6)
        self.assertGreater(factors[0], 0.0)
        self.assertLess(factors[0], factors[1])
        self.assertLess(factors[1], factors[2])

    def test_shadow_disabled(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "none"}
        r = np.array([-6878.137, 0.0, 0.0])
        f = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
        self.assertAlmostEqual(f, 1.0)

    def test_cylindrical_shadow(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "cylindrical"}
        r1 = np.array([-6878.137, 0.0, 0.0])
        r2 = np.array([0.0, 6878.137, 0.0])
        self.assertAlmostEqual(srp_shadow_factor(r_sc_eci_km=r1, t_s=0.0, env=env), 0.0)
        self.assertAlmostEqual(srp_shadow_factor(r_sc_eci_km=r2, t_s=0.0, env=env), 1.0)

    def test_shadow_uses_ephemeris_mode_callable_when_sun_not_explicit(self):
        def eph_cb(jd_utc: float, env: dict):
            return {
                "sun_pos_eci_km": np.array([-AU_KM, 0.0, 0.0], dtype=float),
                "moon_pos_eci_km": np.array([384400.0, 0.0, 0.0], dtype=float),
            }

        env = {
            "ephemeris_mode": "external",
            "ephemeris_callable": eph_cb,
            "jd_utc_start": 2451545.0,
        }
        r = np.array([6878.137, 0.0, 0.0])
        f = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
        self.assertLess(f, 1e-6)

    def test_srp_geometry_rejects_nonfinite_sun_or_spacecraft_vectors(self):
        with self.assertRaisesRegex(ValueError, "sun_pos_eci_km"):
            srp_shadow_factor(
                r_sc_eci_km=np.array([6878.137, 0.0, 0.0]),
                t_s=0.0,
                env={"sun_pos_eci_km": np.array([AU_KM, np.nan, 0.0])},
            )
        with self.assertRaisesRegex(ValueError, "r_sc_eci_km"):
            srp_shadow_factor(
                r_sc_eci_km=np.array([6878.137, np.inf, 0.0]),
                t_s=0.0,
                env={"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0])},
            )


if __name__ == "__main__":
    unittest.main()
