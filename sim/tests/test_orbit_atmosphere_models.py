import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sim.acceleration.settings import acceleration_context
from sim.aero.core import atmosphere_relative_velocity_eci_km_s
from sim.dynamics.orbit.accelerations import OrbitContext, accel_drag, accel_lift, accel_srp
from sim.dynamics.orbit.atmosphere import (
    _altitude_km_from_eci,
    _local_solar_time_hr,
    _spherical_lat_lon_deg_from_eci,
    density_exponential,
    density_from_model,
    density_harris_priester,
    density_jacchia70,
    density_msis86,
    density_ussa1976,
)
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S, SOLAR_PRESSURE_N_M2
from sim.dynamics.orbit.epoch import (
    AU_KM,
    datetime_to_julian_date,
    sun_position_eci_km_enhanced,
)
from sim.dynamics.orbit.frames import FrameContext, apparent_sidereal_time_hpop_like, transform_state
from sim.dynamics.orbit.harris_priester_backend import (
    _default_coeff_path as harris_priester_default_coeff_path,
)
from sim.dynamics.orbit.harris_priester_backend import (
    harris_priester_density as harris_priester_backend_density,
)
from sim.dynamics.orbit.jacchia70_backend import jacchia70_density as jacchia70_backend_density
from sim.dynamics.orbit.jb2008_backend import (
    jb2006_density as jb2006_backend_density,
)
from sim.dynamics.orbit.jb2008_backend import (
    jb2008_density as jb2008_backend_density,
)
from sim.dynamics.orbit.msis86_backend import msis86_density as msis86_backend_density
from sim.dynamics.orbit.nrlmsise00_backend import nrlmsise00_density as nrlmsise00_backend_density
from sim.dynamics.orbit.propagator import srp_plugin
from sim.utils.geodesy import ecef_to_geodetic_altitude_km, ecef_to_geodetic_deg_km, geodetic_to_ecef_km


class TestOrbitAtmosphereModels(unittest.TestCase):
    @staticmethod
    def _write_minimal_jb2008_tables(sol_path: Path, dtc_path: Path, dt_utc: datetime) -> None:
        jd_floor = int(np.floor(datetime_to_julian_date(dt_utc) - 1.0))
        day_of_year = int(dt_utc.timetuple().tm_yday)
        sol_row = [0.0, 0.0, float(jd_floor), 150.0, 150.0, 140.0, 140.0, 130.0, 130.0, 120.0, 120.0]
        dtc_row = [float(dt_utc.year), float(day_of_year)] + [0.0] * 24
        np.savetxt(sol_path, np.array([sol_row, sol_row], dtype=float), fmt="%.6f")
        np.savetxt(dtc_path, np.array([dtc_row, dtc_row], dtype=float), fmt="%.6f")

    def test_hpop_like_local_solar_time_tracks_subsolar_longitude(self):
        dt_utc = datetime(2020, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        jd = datetime_to_julian_date(dt_utc)
        sun = sun_position_eci_km_enhanced(jd)
        sun_ra = np.arctan2(float(sun[1]), float(sun[0]))
        sidereal = apparent_sidereal_time_hpop_like(jd, None)
        subsolar_lon_deg = np.degrees((sun_ra - sidereal + np.pi) % (2.0 * np.pi) - np.pi)

        noon = _local_solar_time_hr(subsolar_lon_deg, dt_utc, {"drag_frame_model": "hpop_like"})
        midnight = _local_solar_time_hr(subsolar_lon_deg + 180.0, dt_utc, {"drag_frame_model": "hpop_like"})

        self.assertAlmostEqual(noon, 12.0, places=9)
        self.assertTrue(min(midnight, 24.0 - midnight) < 1e-9)

    def test_local_solar_time_honors_eop_hold_policy(self):
        with tempfile.TemporaryDirectory() as td:
            eop_path = Path(td) / "EOP-All.txt"
            eop_path.write_text(
                "\n".join(
                    [
                        "VERSION test",
                        "NUM_OBSERVED_POINTS 2",
                        "2024 01 01 60310.0 0.10 0.20 0.30 0 0 0 0 0 37",
                        "2024 01 02 60311.0 0.11 0.21 0.31 0 0 0 0 0 37",
                    ]
                ),
                encoding="utf-8",
            )
            dt_utc = datetime(2024, 1, 3, tzinfo=timezone.utc)
            env = {
                "density_frame_model": "iau76_80_eop",
                "density_eop_path": str(eop_path),
            }

            with self.assertRaisesRegex(ValueError, "outside EOP coverage"):
                _local_solar_time_hr(0.0, dt_utc, env)

            held = _local_solar_time_hr(0.0, dt_utc, {**env, "eop_extrapolation": "hold"})
            self.assertTrue(np.isfinite(held))

    @staticmethod
    def _write_minimal_jb2006_tables(sol_path: Path, ap_path: Path, dt_utc: datetime) -> None:
        jd_floor = int(np.floor(datetime_to_julian_date(dt_utc)))
        rows = []
        for offset in range(-6, 2):
            rows.append(
                [
                    0.0,
                    0.0,
                    float(jd_floor + offset),
                    150.0 + offset,
                    150.0,
                    140.0 + offset,
                    140.0,
                    130.0 + offset,
                    130.0,
                    120.0,
                    120.0,
                ]
            )
        lag_dt = dt_utc.replace(tzinfo=timezone.utc) - timedelta(hours=6.7)
        day_of_year = int(lag_dt.timetuple().tm_yday)
        ap_row = [float(lag_dt.year), float(day_of_year), 0.0, 0.0] + [4.0] * 8
        np.savetxt(sol_path, np.array(rows, dtype=float), fmt="%.6f")
        np.savetxt(ap_path, np.array([ap_row, ap_row], dtype=float), fmt="%.6f")

    @staticmethod
    def _write_minimal_nrlmsise00_sw_table(sw_path: Path) -> datetime:
        rows = []
        for day in range(7, 11):
            row = np.zeros(33, dtype=float)
            row[0] = 2024.0
            row[1] = 1.0
            row[2] = float(day)
            row[14:22] = 4.0 + day
            row[22] = 4.0 + day
            row[27] = 150.0 + day
            row[29] = 145.0 + day
            rows.append(row)
        np.savetxt(sw_path, np.array(rows, dtype=float), fmt="%.6f")
        return datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

    def test_ussa1976_density_reasonable_at_sea_level(self):
        r = np.array([6378.137, 0.0, 0.0], dtype=float)
        rho = density_ussa1976(r, t_s=0.0)
        self.assertGreater(rho, 1.0)
        self.assertLess(rho, 1.4)

    def test_ussa1976_honors_wgs84_altitude_env(self):
        r = geodetic_to_ecef_km(lat_deg=45.0, lon_deg=0.0, alt_km=50.0)

        rho_wgs84 = density_from_model("ussa1976", r, 0.0, env={"geodetic_model": "wgs84"})
        rho_spherical = density_from_model("ussa1976", r, 0.0, env={})

        self.assertGreater(rho_wgs84, 0.0)
        self.assertGreater(rho_spherical, 0.0)
        self.assertNotAlmostEqual(rho_wgs84, rho_spherical)

    def test_wgs84_altitude_only_path_matches_full_conversion_exactly(self):
        cases = (
            geodetic_to_ecef_km(lat_deg=0.0, lon_deg=0.0, alt_km=0.0),
            geodetic_to_ecef_km(lat_deg=45.0, lon_deg=120.0, alt_km=400.0),
            geodetic_to_ecef_km(lat_deg=-70.0, lon_deg=-35.0, alt_km=1000.0),
            np.array([0.0, 0.0, 6356.752314245179 + 10.0], dtype=float),
        )

        for r_ecef_km in cases:
            with self.subTest(r_ecef_km=r_ecef_km):
                self.assertEqual(ecef_to_geodetic_altitude_km(r_ecef_km), ecef_to_geodetic_deg_km(r_ecef_km)[2])

    def test_density_models_selectable(self):
        r = np.array([6778.137, 0.0, 0.0], dtype=float)
        rho_exp = density_from_model("exponential", r, 0.0, env={})
        rho_ussa = density_from_model("ussa1976", r, 0.0, env={})
        self.assertGreaterEqual(rho_exp, 0.0)
        self.assertGreaterEqual(rho_ussa, 0.0)

    def test_optional_atmosphere_kernels_respect_acceleration_off(self):
        dt_utc = datetime(2022, 3, 31, tzinfo=timezone.utc)
        jb_tables = tempfile.TemporaryDirectory()
        self.addCleanup(jb_tables.cleanup)
        sol_path = Path(jb_tables.name) / "SOLFSMY.txt"
        ap_path = Path(jb_tables.name) / "SOLRESAP.txt"
        self._write_minimal_jb2006_tables(sol_path, ap_path, dt_utc)
        jb_env = {
            "jb2006_sol_path": str(sol_path),
            "jb2006_ap_path": str(ap_path),
        }
        with (
            acceleration_context("off"),
            patch(
                "sim.dynamics.orbit.jacchia70_backend._compiled_jacchia_mid_altitude_log",
                side_effect=AssertionError("Jacchia-70 compiled kernel must stay disabled"),
            ),
            patch(
                "sim.dynamics.orbit.jb2008_backend._compiled_integrate_upper_atmosphere",
                side_effect=AssertionError("JB compiled kernel must stay disabled"),
            ),
            patch(
                "sim.dynamics.orbit.nrlmsise00_backend._compiled_globe7_quiet",
                side_effect=AssertionError("NRLMSISE-00 compiled kernel must stay disabled"),
            ),
            patch(
                "sim.dynamics.orbit.nrlmsise00_backend._compiled_densu",
                side_effect=AssertionError("NRLMSISE-00 compiled density kernel must stay disabled"),
            ),
            patch(
                "sim.dynamics.orbit.nrlmsise00_backend._compiled_quiet_thermosphere_density",
                side_effect=AssertionError("NRLMSISE-00 compiled thermosphere kernel must stay disabled"),
            ),
            patch(
                "sim.dynamics.orbit.atmosphere._compiled_ecef_to_geodetic_deg_km",
                side_effect=AssertionError("compiled WGS-84 kernel must stay disabled"),
            ),
            patch(
                "sim.dynamics.orbit.msis86_backend._compiled_globe5_quiet",
                side_effect=AssertionError("MSIS-86 compiled globe kernel must stay disabled"),
            ),
            patch(
                "sim.dynamics.orbit.msis86_backend._compiled_denss",
                side_effect=AssertionError("MSIS-86 compiled density kernel must stay disabled"),
            ),
        ):
            self.assertGreater(jacchia70_backend_density(400.0, 10.0, 20.0, dt_utc, {}), 0.0)
            self.assertGreater(jb2006_backend_density(400.0, 10.0, 20.0, dt_utc, jb_env), 0.0)
            self.assertGreater(msis86_backend_density(400.0, 10.0, 20.0, dt_utc, {}), 0.0)
            self.assertGreater(nrlmsise00_backend_density(400.0, 10.0, 20.0, dt_utc, {}), 0.0)
            self.assertGreater(
                density_from_model(
                    "nrlmsise00",
                    np.array([6778.137, 0.0, 0.0], dtype=float),
                    0.0,
                    env={"atmo_epoch_utc": dt_utc, "geodetic_model": "wgs84"},
                ),
                0.0,
            )

    def test_density_exponential_remains_positive_above_180_km(self):
        r_200km = np.array([6378.137 + 200.0, 0.0, 0.0], dtype=float)
        rho = density_exponential(r_200km, t_s=0.0)
        self.assertGreater(rho, 0.0)

    def test_density_exponential_accepts_vleo_reference_parameters(self):
        r_220km = np.array([6378.137 + 220.0, 0.0, 0.0], dtype=float)
        env = {
            "exponential_reference_altitude_km": 220.0,
            "exponential_reference_density_kg_m3": 2.5e-10,
            "exponential_scale_height_km": 35.0,
        }
        self.assertAlmostEqual(density_exponential(r_220km, 0.0, env), 2.5e-10)
        r_255km = np.array([6378.137 + 255.0, 0.0, 0.0], dtype=float)
        self.assertAlmostEqual(density_exponential(r_255km, 0.0, env), 2.5e-10 / np.e)

    def test_density_exponential_rejects_nonpositive_scale_height(self):
        r = np.array([6378.137 + 220.0, 0.0, 0.0], dtype=float)
        with self.assertRaisesRegex(ValueError, "scale_height"):
            density_exponential(r, 0.0, {"exponential_scale_height_km": 0.0})

    def test_density_exponential_rejects_nonfinite_position_before_native_dispatch(self):
        r = np.array([6378.137 + 220.0, np.nan, 0.0], dtype=float)
        with self.assertRaisesRegex(ValueError, "r_eci_km"):
            density_exponential(r, 0.0, {"_rust_numeric_backend": "rust"})

    def test_density_exponential_skips_ecef_conversion(self):
        r = np.array([6378.137 + 200.0, 0.0, 0.0], dtype=float)
        with patch("sim.dynamics.orbit.atmosphere.eci_to_ecef_harmonic", side_effect=AssertionError("should not be called")):
            rho = density_exponential(r, t_s=123.0)
            rho_from_model = density_from_model("exponential", r, 123.0, env={"geodetic_model": "wgs84"})
        self.assertGreater(rho, 0.0)
        self.assertAlmostEqual(rho_from_model, rho)

    def test_density_model_callable_hooks(self):
        cases = [
            ("nrlmsise00", "nrlmsise00_density_callable", 1.23e-11),
            ("hpop_msis86", "msis86_density_callable", 9.87e-12),
            ("jb2008", "jb2008_density_callable", 4.56e-12),
            ("jb2006", "jb2006_density_callable", 7.89e-12),
            ("hpop_jacchia70", "jacchia70_density_callable", 2.34e-12),
        ]
        for model, callable_key, expected_density in cases:
            with self.subTest(model=model):
                calls = []

                def _fn(alt_km, lat_deg, lon_deg, dt_utc, env, *, _calls=calls, _density=expected_density):
                    _calls.append((alt_km, lat_deg, lon_deg, dt_utc))
                    return _density

                env = {
                    callable_key: _fn,
                    "atmo_epoch_utc": datetime(2024, 1, 1, tzinfo=timezone.utc),
                }
                r = np.array([7000.0, 0.0, 0.0], dtype=float)
                rho = density_from_model(model, r, t_s=60.0, env=env)
                self.assertAlmostEqual(rho, expected_density)
                self.assertEqual(len(calls), 1)

    def test_density_callable_uses_jd_utc_start_plus_elapsed_time(self):
        calls = []

        def _fn(alt_km, lat_deg, lon_deg, dt_utc, env):
            calls.append(dt_utc)
            return 1.0e-11

        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        rho = density_from_model(
            "nrlmsise00",
            r,
            t_s=3600.0,
            env={"jd_utc_start": datetime_to_julian_date(start), "nrlmsise00_density_callable": _fn},
        )

        self.assertAlmostEqual(rho, 1.0e-11)
        self.assertAlmostEqual(
            calls[0].timestamp(),
            datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp(),
            places=3,
        )

    def test_density_ecef_conversion_falls_back_to_drag_frame_settings(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        with patch(
            "sim.dynamics.orbit.atmosphere.eci_to_ecef_harmonic",
            return_value=np.array([7000.0, 0.0, 0.0], dtype=float),
        ) as rot:
            _spherical_lat_lon_deg_from_eci(
                r,
                12.0,
                env={
                    "drag_frame_model": "hpop_like",
                    "drag_eop_path": "validation/EOP-All.txt",
                    "jd_utc_start": 2460310.5,
                },
            )

        self.assertEqual(rot.call_args.kwargs["frame_model"], "hpop_like")
        self.assertEqual(rot.call_args.kwargs["eop_path"], "validation/EOP-All.txt")

    def test_density_nrlmsise00_builtin_backend_returns_finite_density(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        env = {
            "atmo_epoch_utc": datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            "geodetic_model": "wgs84",
            "f107": 150.0,
            "f107a": 150.0,
            "ap": 4.0,
        }

        rho = density_from_model("nrlmsise00", r, 0.0, env=env)

        self.assertTrue(np.isfinite(rho))
        self.assertGreater(rho, 0.0)

    def test_density_nrlmsise00_reused_workspace_has_no_state_leakage(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        base_env = {
            "atmo_epoch_utc": datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            "geodetic_model": "wgs84",
            "f107": 150.0,
            "f107a": 150.0,
            "ap": 4.0,
        }

        first = density_from_model("nrlmsise00", r, 0.0, env=base_env)
        density_from_model("nrlmsise00", r, 60.0, env={**base_env, "f107": 220.0, "ap": 20.0})
        repeated = density_from_model("nrlmsise00", r, 0.0, env=base_env)

        self.assertEqual(repeated, first)

    def test_density_nrlmsise00_acceleration_is_exact_across_model_branches(self):
        dt_utc = datetime(2022, 3, 31, 12, 34, 56, 789012, tzinfo=timezone.utc)
        cases = (
            (10.0, -80.0, -170.0, 70.0, 70.0, 0.0, [0.0] * 7),
            (72.5, 0.0, 0.0, 150.0, 150.0, 4.0, [4.0] * 7),
            (120.0, 45.0, 120.0, 220.0, 180.0, 40.0, [4.0, 8.0, 12.0, 20.0, 30.0, 40.0, 50.0]),
            (300.0, -89.0, -179.0, 70.0, 70.0, 4.0, [4.0] * 7),
            (320.0, 0.0, 0.0, 150.0, 150.0, 4.0, [4.0] * 7),
            (400.0, 89.0, 179.0, 150.0, 150.0, 4.0, [4.0] * 7),
            (450.0, 45.0, 120.0, 220.0, 180.0, 4.0, [4.0] * 7),
            (450.0001, -45.0, -120.0, 220.0, 180.0, 4.0, [4.0] * 7),
            (1000.0, -80.0, -170.0, 220.0, 180.0, 40.0, [4.0, 8.0, 12.0, 20.0, 30.0, 40.0, 50.0]),
        )
        for alt_km, lat_deg, lon_deg, f107, f107a, ap, ap_a in cases:
            env = {
                "f107": f107,
                "f107a": f107a,
                "ap": ap,
                "nrlmsise00_ap_a": ap_a,
            }
            with acceleration_context("off"):
                expected = nrlmsise00_backend_density(
                    alt_km,
                    lat_deg,
                    lon_deg,
                    dt_utc,
                    env,
                    lst_hr=7.25,
                )
            with acceleration_context("auto"):
                actual = nrlmsise00_backend_density(
                    alt_km,
                    lat_deg,
                    lon_deg,
                    dt_utc,
                    env,
                    lst_hr=7.25,
                )
            with self.subTest(alt_km=alt_km, f107=f107, ap=ap):
                self.assertEqual(actual, expected)

    def test_density_models_are_solar_flux_sensitive_and_local(self):
        cases = [
            ("nrlmsise00", "nrlmsise00", ("f107", "f107a"), "ap"),
            ("msis86", "msis-86", ("f107", "f107a"), "ap"),
            ("jacchia70", "jacchia-70", ("jacchia70_f10", "jacchia70_f10b"), "jacchia70_ap"),
        ]
        for low_model, high_model, flux_keys, ap_key in cases:
            with self.subTest(model=low_model):
                r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
                base_env = {
                    "atmo_epoch_utc": datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
                    "geodetic_model": "wgs84",
                    ap_key: 4.0,
                }
                low_flux = {key: 90.0 for key in flux_keys}
                high_flux = {key: 220.0 for key in flux_keys}
                rho_low = density_from_model(low_model, r, 0.0, env={**base_env, **low_flux})
                rho_high = density_from_model(high_model, r, 0.0, env={**base_env, **high_flux})

                self.assertTrue(np.isfinite(rho_low))
                self.assertTrue(np.isfinite(rho_high))
                self.assertGreater(rho_high, rho_low)

    def test_density_models_read_hpop_style_sw_table(self):
        cases = [
            ("nrlmsise00", "nrlmsise00_sw_path"),
            ("msis86", "msis86_sw_path"),
        ]
        for model, sw_path_key in cases:
            with self.subTest(model=model):
                r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
                with tempfile.TemporaryDirectory() as td:
                    sw_path = Path(td) / "SW-All.txt"
                    dt_utc = self._write_minimal_nrlmsise00_sw_table(sw_path)
                    rho = density_from_model(
                        model,
                        r,
                        0.0,
                        env={
                            "atmo_epoch_utc": dt_utc,
                            sw_path_key: str(sw_path),
                            "geodetic_model": "wgs84",
                        },
                    )

                self.assertTrue(np.isfinite(rho))
                self.assertGreater(rho, 0.0)

    def test_density_msis86_builtin_backend_returns_finite_density(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        env = {
            "atmo_epoch_utc": datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            "geodetic_model": "wgs84",
            "f107": 150.0,
            "f107a": 150.0,
            "ap": 4.0,
        }

        rho = density_msis86(r, 0.0, env=env)

        self.assertTrue(np.isfinite(rho))
        self.assertGreater(rho, 0.0)

    def test_density_msis86_acceleration_is_exact_across_model_branches(self):
        dt_utc = datetime(2022, 3, 31, 12, 34, 56, 789012, tzinfo=timezone.utc)
        cases = (
            (85.0, -80.0, -170.0, 70.0, 70.0, 0.0, [0.0] * 7),
            (120.0, 45.0, 120.0, 220.0, 180.0, 40.0, [4.0, 8.0, 12.0, 20.0, 30.0, 40.0, 50.0]),
            (400.0, 89.0, 179.0, 150.0, 150.0, 4.0, [4.0] * 7),
            (1000.0, -80.0, -170.0, 220.0, 180.0, 4.0, [4.0] * 7),
        )
        for alt_km, lat_deg, lon_deg, f107, f107a, ap, ap_a in cases:
            env = {
                "f107": f107,
                "f107a": f107a,
                "ap": ap,
                "msis86_ap_a": ap_a,
                "msis86_lst_hr": 7.25,
            }
            with acceleration_context("off"):
                expected = msis86_backend_density(alt_km, lat_deg, lon_deg, dt_utc, env)
            with acceleration_context("auto"):
                actual = msis86_backend_density(alt_km, lat_deg, lon_deg, dt_utc, env)
            with self.subTest(alt_km=alt_km, f107=f107, ap=ap):
                self.assertEqual(actual, expected)

    def test_density_msis86_reused_workspace_and_cached_indices_are_path_scoped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sw_a = root / "SW-All-A.txt"
            sw_b = root / "SW-All-B.txt"
            dt_utc = self._write_minimal_nrlmsise00_sw_table(sw_a)
            self._write_minimal_nrlmsise00_sw_table(sw_b)
            alternate_sw = np.loadtxt(sw_b, dtype=float)
            alternate_sw[:, 14:23] += 10.0
            alternate_sw[:, 27] += 40.0
            alternate_sw[:, 29] += 40.0
            np.savetxt(sw_b, alternate_sw, fmt="%.6f")

            first = msis86_backend_density(400.0, 10.0, 20.0, dt_utc, {"msis86_sw_path": str(sw_a)})
            alternate = msis86_backend_density(400.0, 10.0, 20.0, dt_utc, {"msis86_sw_path": str(sw_b)})
            repeated = msis86_backend_density(400.0, 10.0, 20.0, dt_utc, {"msis86_sw_path": str(sw_a)})

        self.assertEqual(repeated, first)
        self.assertNotEqual(alternate, first)

    def test_density_msis86_uses_hpop_angle_compatibility(self):
        r = np.array([5100.0, 3400.0, 2800.0], dtype=float)
        dt_utc = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        env = {
            "atmo_epoch_utc": dt_utc,
            "geodetic_model": "wgs84",
            "f107": 150.0,
            "f107a": 150.0,
            "ap": 4.0,
        }

        alt_km = _altitude_km_from_eci(r, 0.0, env=env)
        lat_deg, lon_deg = _spherical_lat_lon_deg_from_eci(r, 0.0, env=env)
        lst_hr = _local_solar_time_hr(lon_deg, dt_utc, env)
        expected = msis86_backend_density(
            alt_km,
            np.radians(lat_deg),
            np.radians(lon_deg),
            dt_utc,
            {**env, "msis86_lst_hr": lst_hr},
        )

        rho = density_msis86(r, 0.0, env=env)

        self.assertAlmostEqual(rho, expected, delta=max(expected, 1.0e-30) * 1.0e-12)

    def test_density_jb2008_builtin_backend_returns_finite_density(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        dt_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            sol_path = Path(td) / "SOLFSMY.txt"
            dtc_path = Path(td) / "DTCFILE.txt"
            self._write_minimal_jb2008_tables(sol_path, dtc_path, dt_utc)
            rho = density_from_model(
                "jb2008",
                r,
                t_s=60.0,
                env={
                    "atmo_epoch_utc": dt_utc,
                    "jb2008_sol_path": str(sol_path),
                    "jb2008_dtc_path": str(dtc_path),
                    "geodetic_model": "wgs84",
                },
            )
        self.assertTrue(np.isfinite(rho))
        self.assertGreaterEqual(rho, 0.0)

    def test_density_jb2008_cached_indices_are_scoped_to_table_path(self):
        dt_utc = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sol_a = root / "SOLFSMY-A.txt"
            dtc_a = root / "DTCFILE-A.txt"
            sol_b = root / "SOLFSMY-B.txt"
            dtc_b = root / "DTCFILE-B.txt"
            self._write_minimal_jb2008_tables(sol_a, dtc_a, dt_utc)
            self._write_minimal_jb2008_tables(sol_b, dtc_b, dt_utc)
            alternate_sol = np.loadtxt(sol_b, dtype=float)
            alternate_sol[:, 3:] += 40.0
            np.savetxt(sol_b, alternate_sol, fmt="%.6f")

            env_a = {"jb2008_sol_path": str(sol_a), "jb2008_dtc_path": str(dtc_a)}
            env_b = {"jb2008_sol_path": str(sol_b), "jb2008_dtc_path": str(dtc_b)}
            first = jb2008_backend_density(400.0, 10.0, 20.0, dt_utc, env_a)
            alternate = jb2008_backend_density(400.0, 10.0, 20.0, dt_utc, env_b)
            repeated = jb2008_backend_density(400.0, 10.0, 20.0, dt_utc, env_a)

        self.assertEqual(repeated, first)
        self.assertNotEqual(alternate, first)

    def test_density_jb2006_builtin_backend_returns_finite_density(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        dt_utc = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            sol_path = Path(td) / "SOLFSMY.txt"
            ap_path = Path(td) / "SOLRESAP.txt"
            self._write_minimal_jb2006_tables(sol_path, ap_path, dt_utc)
            rho = density_from_model(
                "jb2006",
                r,
                t_s=60.0,
                env={
                    "atmo_epoch_utc": dt_utc,
                    "jb2006_sol_path": str(sol_path),
                    "jb2006_ap_path": str(ap_path),
                    "geodetic_model": "wgs84",
                },
            )
        self.assertTrue(np.isfinite(rho))
        self.assertGreaterEqual(rho, 0.0)

    def test_density_jb2006_cached_indices_are_scoped_to_table_path(self):
        dt_utc = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sol_a = root / "SOLFSMY-A.txt"
            ap_a = root / "SOLRESAP-A.txt"
            sol_b = root / "SOLFSMY-B.txt"
            ap_b = root / "SOLRESAP-B.txt"
            self._write_minimal_jb2006_tables(sol_a, ap_a, dt_utc)
            self._write_minimal_jb2006_tables(sol_b, ap_b, dt_utc)
            alternate_sol = np.loadtxt(sol_b, dtype=float)
            alternate_sol[:, 3:] += 40.0
            np.savetxt(sol_b, alternate_sol, fmt="%.6f")

            env_a = {"jb2006_sol_path": str(sol_a), "jb2006_ap_path": str(ap_a)}
            env_b = {"jb2006_sol_path": str(sol_b), "jb2006_ap_path": str(ap_b)}
            first = jb2006_backend_density(400.0, 10.0, 20.0, dt_utc, env_a)
            alternate = jb2006_backend_density(400.0, 10.0, 20.0, dt_utc, env_b)
            repeated = jb2006_backend_density(400.0, 10.0, 20.0, dt_utc, env_a)

        self.assertEqual(repeated, first)
        self.assertNotEqual(alternate, first)

    def test_density_jacchia70_builtin_backend_returns_finite_density(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        dt_utc = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        env = {
            "atmo_epoch_utc": dt_utc,
            "geodetic_model": "wgs84",
            "jacchia70_f10": 150.0,
            "jacchia70_f10b": 150.0,
            "jacchia70_ap": 4.0,
        }

        rho = density_jacchia70(r, t_s=0.0, env=env)

        self.assertTrue(np.isfinite(rho))
        self.assertGreater(rho, 0.0)

    def test_density_jacchia70_cached_indices_are_scoped_to_table_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sw_a = root / "SW-All-A.txt"
            sw_b = root / "SW-All-B.txt"
            dt_utc = self._write_minimal_nrlmsise00_sw_table(sw_a)
            self._write_minimal_nrlmsise00_sw_table(sw_b)
            alternate_sw = np.loadtxt(sw_b, dtype=float)
            alternate_sw[:, 14:23] += 10.0
            alternate_sw[:, 27] += 40.0
            alternate_sw[:, 29] += 40.0
            np.savetxt(sw_b, alternate_sw, fmt="%.6f")

            first = jacchia70_backend_density(400.0, 10.0, 20.0, dt_utc, {"jacchia70_sw_path": str(sw_a)})
            alternate = jacchia70_backend_density(400.0, 10.0, 20.0, dt_utc, {"jacchia70_sw_path": str(sw_b)})
            repeated = jacchia70_backend_density(400.0, 10.0, 20.0, dt_utc, {"jacchia70_sw_path": str(sw_a)})

        self.assertEqual(repeated, first)
        self.assertNotEqual(alternate, first)

    def test_harris_priester_density_is_local_and_f107_sensitive(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        env_low = {"sun_pos_eci_km": np.array([1.0, 0.0, 0.0]), "harris_priester_f107": 65}
        env_high = {"sun_pos_eci_km": np.array([1.0, 0.0, 0.0]), "harris_priester_f107": 275}

        rho_low = density_harris_priester(r, t_s=0.0, env=env_low)
        rho_high = density_from_model("harris_priester", r, 0.0, env=env_high)

        self.assertTrue(np.isfinite(rho_low))
        self.assertTrue(np.isfinite(rho_high))
        self.assertGreater(rho_low, 0.0)
        self.assertGreater(rho_high, rho_low)

    def test_harris_priester_cached_coefficient_paths_are_isolated(self):
        coefficients = np.loadtxt(
            harris_priester_default_coeff_path(),
            delimiter=",",
            comments="#",
            skiprows=3,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            table_a = root / "hp-a.csv"
            table_b = root / "hp-b.csv"
            np.savetxt(table_a, coefficients, delimiter=",", header="one\ntwo\nthree")
            alternate_coefficients = coefficients.copy()
            alternate_coefficients[:, 2:4] *= 2.0
            np.savetxt(table_b, alternate_coefficients, delimiter=",", header="one\ntwo\nthree")

            r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
            base_env = {"sun_pos_eci_km": np.array([1.0, 0.0, 0.0]), "harris_priester_f107": 175}
            first = harris_priester_backend_density(
                r,
                0.0,
                {**base_env, "harris_priester_coeff_path": str(table_a)},
            )
            alternate = harris_priester_backend_density(
                r,
                0.0,
                {**base_env, "harris_priester_coeff_path": str(table_b)},
            )
            repeated = harris_priester_backend_density(
                r,
                0.0,
                {**base_env, "harris_priester_coeff_path": str(table_a)},
            )

        self.assertEqual(repeated, first)
        self.assertEqual(alternate, 2.0 * first)

    def test_harris_priester_aliases_select_same_model(self):
        r = np.array([6378.137 + 500.0, 0.0, 0.0], dtype=float)
        env = {"sun_pos_eci_km": np.array([1.0, 0.0, 0.0]), "harris_priester_f107": 175}

        rho_named = density_from_model("harris_priester", r, 0.0, env=env)
        rho_alias = density_from_model("hp", r, 0.0, env=env)

        self.assertAlmostEqual(rho_named, rho_alias)

    def test_harris_priester_frame_aliases_apply_the_same_rotation(self):
        r = np.array([6378.137 + 500.0, 0.0, 0.0], dtype=float)
        base_env = {
            "sun_pos_eci_km": np.array([1.0, 0.0, 0.0]),
            "harris_priester_f107": 175,
            "jd_utc_start": 2460310.5,
        }
        with patch(
            "sim.dynamics.orbit.harris_priester_backend.precession_nutation_rotation_hpop_like",
            return_value=np.eye(3),
        ) as rotation:
            legacy = harris_priester_backend_density(r, 60.0, {**base_env, "density_frame_model": "hpop_like"})
            canonical = harris_priester_backend_density(
                r,
                60.0,
                {**base_env, "density_frame_model": "iau76_80_eop"},
            )

        self.assertEqual(rotation.call_count, 2)
        self.assertEqual(canonical, legacy)

    def test_harris_priester_uses_wgs84_geodetic_height_when_requested(self):
        r = np.array([3506.788211789612, 4884.292725366271, 2667.959407741529], dtype=float)
        env = {"sun_pos_eci_km": np.array([1.0, 0.0, 0.0]), "harris_priester_f107": 175}

        rho_spherical = density_harris_priester(r, t_s=0.0, env=env)
        rho_geodetic = density_harris_priester(r, t_s=0.0, env={**env, "geodetic_model": "wgs84"})

        self.assertTrue(np.isfinite(rho_spherical))
        self.assertTrue(np.isfinite(rho_geodetic))
        self.assertGreater(rho_spherical, rho_geodetic)

    def test_drag_uses_rotating_atmosphere_relative_velocity(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        omega = np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float)
        v_atm = np.cross(omega, r)
        a = accel_drag(
            r_eci_km=r,
            v_eci_km_s=v_atm,  # matches corotating atmosphere speed at position
            t_s=0.0,
            mass_kg=100.0,
            area_m2=1.0,
            cd=2.2,
            env={"density_kg_m3": density_exponential(r, 0.0)},
        )
        self.assertTrue(np.linalg.norm(a) < 1e-14)

    def test_drag_hpop_like_frame_matches_shared_relative_velocity_helper(self):
        r = np.array([5100.0, -3200.0, 2600.0], dtype=float)
        v = np.array([3.9, 5.8, -1.2], dtype=float)
        rho = 2.0e-11
        mass_kg = 250.0
        area_m2 = 1.7
        cd = 2.1
        env = {
            "density_kg_m3": rho,
            "drag_area_m2": area_m2,
            "drag_frame_model": "hpop_like",
            "jd_utc_start": 2460310.5,
        }

        a = accel_drag(
            r_eci_km=r,
            v_eci_km_s=v,
            t_s=123.0,
            mass_kg=mass_kg,
            area_m2=area_m2,
            cd=cd,
            env=env,
        )
        v_rel_m_s = (
            atmosphere_relative_velocity_eci_km_s(
                r,
                v,
                t_s=123.0,
                frame_model="hpop_like",
                jd_utc_start=2460310.5,
            )
            * 1000.0
        )
        expected = -0.5 * rho * cd * area_m2 / mass_kg * float(np.linalg.norm(v_rel_m_s)) * v_rel_m_s / 1000.0

        np.testing.assert_allclose(a, expected, rtol=0.0, atol=1e-18)

    def test_eop_relative_velocity_matches_canonical_stationary_ecef_state(self):
        context = FrameContext(
            model="iau76_80_eop",
            jd_utc_start=2460310.5,
            dut1_s=0.15,
            xp_arcsec=0.25,
            yp_arcsec=-0.18,
            ddpsi_rad=2.0e-7,
            ddeps_rad=-3.0e-7,
        )
        position_eci, velocity_eci = transform_state(
            np.array([6378.137, 0.0, 0.0]),
            np.zeros(3),
            "ecef",
            "eci",
            t_s=321.0,
            context=context,
        )
        relative = atmosphere_relative_velocity_eci_km_s(
            position_eci,
            velocity_eci,
            t_s=321.0,
            frame_model=context.model,
            jd_utc_start=context.jd_utc_start,
            dut1_s=context.dut1_s,
            xp_arcsec=context.xp_arcsec,
            yp_arcsec=context.yp_arcsec,
            ddpsi_rad=context.ddpsi_rad,
            ddeps_rad=context.ddeps_rad,
        )
        np.testing.assert_allclose(relative, 0.0, rtol=0.0, atol=2.0e-12)

    def test_lift_projects_attitude_vector_perpendicular_to_relative_wind(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        v = np.array([0.0, 7.6, 0.0], dtype=float)

        a = accel_lift(
            r_eci_km=r,
            v_eci_km_s=v,
            t_s=0.0,
            mass_kg=100.0,
            area_m2=2.0,
            cl=0.8,
            lift_direction_eci=np.array([0.0, 0.0, 1.0]),
            env={"density_kg_m3": 1.0e-9},
        )

        self.assertGreater(a[2], 0.0)
        self.assertAlmostEqual(float(np.dot(a, v)), 0.0, delta=1e-12)

    def test_lift_returns_zero_when_axis_is_along_relative_wind(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        v = np.array([0.0, 7.6, 0.0], dtype=float)

        a = accel_lift(
            r_eci_km=r,
            v_eci_km_s=v,
            t_s=0.0,
            mass_kg=100.0,
            area_m2=2.0,
            cl=0.8,
            lift_direction_eci=np.array([0.0, 1.0, 0.0]),
            env={"density_kg_m3": 1.0e-9},
        )

        np.testing.assert_allclose(a, np.zeros(3), atol=1e-15)

    def test_srp_scales_with_sun_spacecraft_distance(self):
        r = np.array([6878.137, 0.0, 0.0], dtype=float)
        a_1au = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env={"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "none"},
        )
        a_half_au = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env={"sun_pos_eci_km": np.array([0.5 * AU_KM, 0.0, 0.0]), "srp_shadow_model": "none"},
        )
        ratio = float(np.linalg.norm(a_half_au) / max(np.linalg.norm(a_1au), 1e-18))
        self.assertGreater(ratio, 3.95)
        self.assertLess(ratio, 4.05)

    def test_srp_accel_accepts_precomputed_geometry_bundle(self):
        r = np.array([6878.137, 0.0, 0.0], dtype=float)
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "conical"}
        geometry = resolve_srp_geometry(r, 0.0, env)
        shadow = srp_shadow_factor(r, 0.0, env, srp_geometry=geometry)

        a_direct = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env=env,
        )
        a_cached = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env={
                "srp_geometry": geometry,
                "srp_sun_dir_eci": geometry["sun_dir_sc_eci"],
                "srp_distance_scale": geometry["distance_scale"],
                "srp_shadow_factor": shadow,
                "srp_area_m2": 1.0,
                "srp_shadow_model": "conical",
            },
        )
        np.testing.assert_allclose(a_cached, a_direct, rtol=0.0, atol=1e-15)

    def test_srp_accel_accepts_source_pressure_override(self):
        r = np.array([6878.137, 0.0, 0.0], dtype=float)
        base_env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "none"}
        a_default = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env=base_env,
        )
        a_override = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env={**base_env, "srp_pressure_n_m2": 2.0 * SOLAR_PRESSURE_N_M2},
        )

        np.testing.assert_allclose(a_override, 2.0 * a_default, rtol=1e-12, atol=0.0)

    def test_srp_plugin_acceleration_is_exact_across_shadow_branches(self):
        ctx = OrbitContext(mu_km3_s2=398600.4415, mass_kg=123.0, area_m2=4.5, cr=1.37)
        state = np.array([7000.0, 100.0, -50.0, 0.0, 7.5, 0.1], dtype=float)
        cases = (
            ("none", np.array([AU_KM, 2.0e6, -1.0e6], dtype=float)),
            ("cylindrical", np.array([AU_KM, 0.0, 0.0], dtype=float)),
            ("cylinder", np.array([-AU_KM, 0.0, 0.0], dtype=float)),
            ("conical", np.array([AU_KM, 0.0, 0.0], dtype=float)),
            ("conical", np.array([-AU_KM, 0.0, 0.0], dtype=float)),
            ("conical", np.zeros(3, dtype=float)),
            ("unknown-model", np.array([AU_KM, 696000.0, 0.0], dtype=float)),
            (" none ", np.array([-AU_KM, 0.0, 0.0], dtype=float)),
        )
        for shadow_model, sun_position in cases:
            env = {
                "sun_pos_eci_km": sun_position,
                "srp_shadow_model": shadow_model,
                "srp_area_m2": 3.25,
                "solar_irradiance_w_m2": 1358.0,
            }
            with acceleration_context("off"):
                expected = srp_plugin(12.5, state, env, ctx)
            with acceleration_context("auto"):
                actual = srp_plugin(12.5, state, env, ctx)
            with self.subTest(shadow_model=shadow_model, sun_position=sun_position):
                np.testing.assert_array_equal(actual, expected)

    def test_srp_plugin_acceleration_off_does_not_load_compiled_kernel(self):
        ctx = OrbitContext(mu_km3_s2=398600.4415, mass_kg=100.0, area_m2=1.0, cr=1.2)
        state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "conical"}
        with (
            acceleration_context("off"),
            patch(
                "sim.dynamics.orbit.propagator._compiled_srp_acceleration",
                side_effect=AssertionError("compiled SRP kernel must stay disabled"),
            ),
        ):
            acceleration = srp_plugin(0.0, state, env, ctx)
        self.assertTrue(np.all(np.isfinite(acceleration)))


if __name__ == "__main__":
    unittest.main()
