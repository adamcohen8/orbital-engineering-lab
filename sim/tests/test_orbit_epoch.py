import unittest
from datetime import datetime, timezone

import numpy as np

from sim.dynamics.orbit.epoch import (
    _mean_equator_of_date_to_j2000,
    _sun_position_true_equator_of_date_km,
    _true_equator_of_date_to_j2000,
    datetime_to_julian_date,
    julian_date_to_datetime,
    moon_position_eci_km_enhanced,
    resolve_body_position_eci_km,
    resolve_sun_moon_positions,
    resolve_sun_position_eci_km,
    resolve_time_dependent_env,
    sun_position_eci_km_enhanced,
)


class OrbitEpochTests(unittest.TestCase):
    def test_julian_round_trip(self):
        dt = datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc)
        jd = datetime_to_julian_date(dt)
        dt_back = julian_date_to_datetime(jd)
        self.assertLess(abs((dt_back - dt).total_seconds()), 1e-3)

    def test_time_dependent_env_adds_dynamic_fields(self):
        env = {"jd_utc_start": 2451545.0, "ephemeris_mode": "analytic_simple"}
        out = resolve_time_dependent_env(env, 120.0)
        self.assertIn("jd_utc", out)
        self.assertIn("sun_pos_eci_km", out)
        self.assertIn("moon_pos_eci_km", out)
        self.assertIn("sun_dir_eci", out)
        self.assertAlmostEqual(float(np.linalg.norm(out["sun_dir_eci"])), 1.0, places=8)

    def test_enhanced_ephemeris_vectors_finite(self):
        jd = 2461110.5
        s = sun_position_eci_km_enhanced(jd)
        m = moon_position_eci_km_enhanced(jd)
        self.assertEqual(s.shape, (3,))
        self.assertEqual(m.shape, (3,))
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertTrue(np.all(np.isfinite(m)))
        self.assertGreater(float(np.linalg.norm(s)), 1.0e8)
        self.assertGreater(float(np.linalg.norm(m)), 1.0e5)

    def test_date_equator_to_j2000_rotation_is_identity_at_j2000(self):
        vector = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            _mean_equator_of_date_to_j2000(vector, 2451545.0),
            vector,
            rtol=0.0,
            atol=1.0e-14,
        )
        future = _mean_equator_of_date_to_j2000(vector, 2461110.5)
        assert not np.allclose(future, vector, rtol=0.0, atol=1.0e-8)

    def test_enhanced_sun_removes_true_of_date_nutation_before_precession(self):
        jd = 2460310.5
        date_vector = _sun_position_true_equator_of_date_km(jd)
        actual = sun_position_eci_km_enhanced(jd)
        precession_only = _mean_equator_of_date_to_j2000(date_vector, jd)

        np.testing.assert_allclose(actual, _true_equator_of_date_to_j2000(date_vector, jd))
        self.assertGreater(float(np.linalg.norm(actual - precession_only)), 1000.0)

    def test_external_ephemeris_callable(self):
        def eph_cb(jd_utc: float, env: dict):
            return {
                "sun_pos_eci_km": np.array([1.0, 2.0, 3.0]),
                "moon_pos_eci_km": np.array([4.0, 5.0, 6.0]),
            }

        env = {
            "jd_utc_start": 2451545.0,
            "ephemeris_mode": "external",
            "ephemeris_callable": eph_cb,
        }
        sun, moon = resolve_sun_moon_positions(env, t_s=10.0)
        self.assertTrue(np.allclose(sun, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.allclose(moon, np.array([4.0, 5.0, 6.0])))

    def test_spice_mode_with_callable(self):
        def spice_cb(jd_utc: float, env: dict):
            return {
                "sun_pos_eci_km": np.array([10.0, 20.0, 30.0]),
                "moon_pos_eci_km": np.array([40.0, 50.0, 60.0]),
            }

        env = {
            "jd_utc_start": 2451545.0,
            "ephemeris_mode": "spice",
            "spice_ephemeris_callable": spice_cb,
        }
        sun, moon = resolve_sun_moon_positions(env, t_s=1.0)
        self.assertTrue(np.allclose(sun, np.array([10.0, 20.0, 30.0])))
        self.assertTrue(np.allclose(moon, np.array([40.0, 50.0, 60.0])))

    def test_explicit_ephemeris_history_interpolates_moon(self):
        env = {
            "sun_pos_eci_km": np.array([1.0, 2.0, 3.0]),
            "moon_ephemeris_time_s": [0.0, 10.0],
            "moon_ephemeris_eci_km": [[4.0, 5.0, 6.0], [14.0, 15.0, 16.0]],
        }
        sun, moon = resolve_sun_moon_positions(env, t_s=5.0)
        self.assertTrue(np.allclose(sun, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.allclose(moon, np.array([9.0, 10.0, 11.0])))

    def test_one_body_sampled_ephemeris_is_not_dropped(self):
        sun_samples = {
            "sun_ephemeris_time_s": [0.0, 10.0],
            "sun_ephemeris_eci_km": [[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]],
        }
        sun, moon = resolve_sun_moon_positions(sun_samples, t_s=5.0)
        self.assertTrue(np.allclose(sun, np.array([6.0, 7.0, 8.0])))
        self.assertTrue(np.allclose(moon, np.array([384400.0, 0.0, 0.0])))

        moon_samples = {
            "moon_ephemeris_time_s": [0.0, 10.0],
            "moon_ephemeris_eci_km": [[4.0, 5.0, 6.0], [14.0, 15.0, 16.0]],
        }
        sun, moon = resolve_sun_moon_positions(moon_samples, t_s=5.0)
        self.assertTrue(np.allclose(sun, np.array([149597870.7, 0.0, 0.0])))
        self.assertTrue(np.allclose(moon, np.array([9.0, 10.0, 11.0])))

    def test_explicit_ephemeris_histories_fail_closed_on_invalid_samples(self):
        for times, states in (
            ([0.0, 0.0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            ([1.0, 0.0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            ([0.0, 1.0], [[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0]]),
        ):
            with self.subTest(times=times, states=states):
                with self.assertRaises(ValueError):
                    resolve_sun_moon_positions(
                        {
                            "sun_ephemeris_time_s": times,
                            "sun_ephemeris_eci_km": states,
                        },
                        t_s=0.5,
                    )

    def test_explicit_ephemeris_history_rejects_out_of_range_queries(self):
        env = {
            "sun_ephemeris_time_s": [0.0, 10.0],
            "sun_ephemeris_eci_km": [[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]],
        }
        for t_s in (-1.0, 11.0):
            with self.subTest(t_s=t_s):
                with self.assertRaisesRegex(ValueError, "outside supplied coverage"):
                    resolve_sun_moon_positions(env, t_s=t_s)

    def test_direct_and_callable_ephemeris_vectors_require_exact_finite_shape(self):
        with self.assertRaisesRegex(ValueError, "exactly three finite values"):
            resolve_sun_position_eci_km({"sun_pos_eci_km": [[1.0, 2.0, 3.0]]}, t_s=0.0)

        with self.assertRaisesRegex(ValueError, "exactly three finite values"):
            resolve_body_position_eci_km(
                "planet",
                {
                    "jd_utc": 2451545.0,
                    "ephemeris_body_callable": lambda *_args: [1.0, np.nan, 3.0],
                },
                t_s=0.0,
            )

        with self.assertRaisesRegex(ValueError, "exactly three finite values"):
            resolve_sun_moon_positions(
                {
                    "jd_utc": 2451545.0,
                    "ephemeris_callable": lambda *_args: {
                        "sun_pos_eci_km": [1.0, 2.0, 3.0],
                        "moon_pos_eci_km": [[4.0, 5.0, 6.0]],
                    },
                },
                t_s=0.0,
            )


if __name__ == "__main__":
    unittest.main()
