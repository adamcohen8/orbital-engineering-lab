import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

import sim.dynamics.orbit.propagator as orbit_propagator
import sim.dynamics.orbit.spherical_harmonics as spherical_harmonics
from sim import SimulationConfig
from sim.dynamics.orbit.accelerations import OrbitContext, accel_j2
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import (
    OrbitPropagator,
    j2_plugin,
    j3_plugin,
    j4_plugin,
    spherical_harmonics_plugin,
)
from sim.dynamics.orbit.spherical_harmonics import (
    GravityModelDownload,
    SphericalHarmonicTerm,
    accel_spherical_harmonics_terms,
    compile_spherical_harmonic_terms,
    configure_spherical_harmonics_env,
    load_icgem_gfc_terms,
    parse_spherical_harmonic_terms,
)
from sim.runtime_support import _build_orbit_propagator


def scenario_config_from_dict(data: dict):
    return SimulationConfig.from_dict(data).to_scenario_config()


class TestOrbitSphericalHarmonics(unittest.TestCase):
    def test_duplicate_degree_order_terms_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate spherical harmonic coefficient"):
            parse_spherical_harmonic_terms(
                [
                    {"n": 2, "m": 0, "c_nm": -1.0e-3},
                    {"n": 2, "m": 0, "c_nm": -2.0e-3},
                ]
            )

    def test_direct_terms_reject_nonfinite_coefficients(self):
        with self.assertRaisesRegex(ValueError, "coefficients must be finite"):
            SphericalHarmonicTerm(n=2, m=0, c_nm=float("nan"))
        with self.assertRaisesRegex(ValueError, "coefficients must be finite"):
            SphericalHarmonicTerm(n=2, m=0, c_nm=0.0, s_nm=float("inf"))

    @staticmethod
    def _write_minimal_eop(path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "VERSION 1.1",
                    "UPDATED 2026 Apr 03 00:00:00 UTC",
                    "NUM_OBSERVED_POINTS 2",
                    "2024 03 31 60400 0.0 0.0 0.0 0 0 0 0 0 37",
                    "2024 04 01 60401 0.0 0.0 0.0 0 0 0 0 0 37",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def test_parse_terms(self):
        raw = [
            {"n": 3, "m": 3, "c_nm": 1e-6, "s_nm": -2e-6},
            {"n": 4, "m": 2, "c": 5e-7, "s": 3e-7},
        ]
        terms = parse_spherical_harmonic_terms(raw)
        self.assertEqual(len(terms), 2)
        self.assertEqual((terms[0].n, terms[0].m), (3, 3))
        self.assertEqual((terms[1].n, terms[1].m), (4, 2))

    def test_sectoral_and_tesseral_nonzero(self):
        r = np.array([7000.0, 200.0, 300.0], dtype=float)
        terms = [
            SphericalHarmonicTerm(n=3, m=3, c_nm=1e-6, s_nm=2e-6),  # sectoral
            SphericalHarmonicTerm(n=4, m=2, c_nm=-2e-6, s_nm=1e-6),  # tesseral
        ]
        a = accel_spherical_harmonics_terms(r_eci_km=r, t_s=0.0, terms=terms, mu_km3_s2=EARTH_MU_KM3_S2)
        self.assertGreater(float(np.linalg.norm(a)), 0.0)

    def test_plugin_reads_env_m_n_terms(self):
        x = np.array([7000.0, 10.0, 20.0, 0.0, 7.5, 0.0], dtype=float)
        env = {
            "spherical_harmonics_terms": [
                {"n": 3, "m": 3, "c_nm": 1e-6, "s_nm": 0.0},
                {"n": 5, "m": 2, "c_nm": -1e-6, "s_nm": 1e-6},
            ],
            "spherical_harmonics_fd_step_km": 1e-3,
        }

        class _Ctx:
            mu_km3_s2 = EARTH_MU_KM3_S2

        a = spherical_harmonics_plugin(0.0, x, env=env, ctx=_Ctx())
        self.assertGreater(float(np.linalg.norm(a)), 0.0)

    def test_plugin_respects_epoch_for_tesseral_terms(self):
        x = np.array([7000.0, 0.0, 100.0, 0.0, 7.5, 0.0], dtype=float)
        env0 = {
            "spherical_harmonics_terms": [{"n": 2, "m": 2, "c_nm": 1e-6, "s_nm": 0.0}],
            "jd_utc_start": 2451545.0,
        }
        env1 = {
            "spherical_harmonics_terms": [{"n": 2, "m": 2, "c_nm": 1e-6, "s_nm": 0.0}],
            "jd_utc_start": 2451545.25,
        }

        class _Ctx:
            mu_km3_s2 = EARTH_MU_KM3_S2

        a0 = spherical_harmonics_plugin(0.0, x, env=env0, ctx=_Ctx())
        a1 = spherical_harmonics_plugin(0.0, x, env=env1, ctx=_Ctx())
        self.assertFalse(np.allclose(a0, a1))

    def test_spherical_harmonics_honors_eop_hold_policy(self):
        terms = [
            SphericalHarmonicTerm(
                n=2,
                m=2,
                c_nm=1.0e-6,
                s_nm=0.0,
                normalized=True,
            )
        ]
        with tempfile.TemporaryDirectory() as td:
            eop_path = Path(td) / "EOP-All.txt"
            self._write_minimal_eop(eop_path)
            kwargs = {
                "r_eci_km": np.array([7000.0, 100.0, 200.0], dtype=float),
                "t_s": -86400.0,
                "terms": terms,
                "jd_utc_start": 2460400.5,
                "frame_model": "iau76_80_eop",
                "eop_path": str(eop_path),
            }

            with self.assertRaisesRegex(ValueError, "outside EOP coverage"):
                accel_spherical_harmonics_terms(**kwargs)

            held = accel_spherical_harmonics_terms(**kwargs, eop_extrapolation="hold")
            self.assertTrue(np.all(np.isfinite(held)))

    def test_normalized_c20_matches_equivalent_j2_perturbation(self):
        r = np.array([7000.0, 100.0, 200.0], dtype=float)
        terms = [SphericalHarmonicTerm(n=2, m=0, c_nm=-4.841693259705e-04, s_nm=0.0, normalized=True)]
        with tempfile.TemporaryDirectory() as td:
            eop_path = Path(td) / "EOP-All.txt"
            self._write_minimal_eop(eop_path)
            a = accel_spherical_harmonics_terms(
                r_eci_km=r,
                t_s=0.0,
                terms=terms,
                mu_km3_s2=EARTH_MU_KM3_S2,
                re_km=6378.1363,
                jd_utc_start=2460400.5,
                frame_model="simple",
                eop_path=str(eop_path),
            )
        a_j2 = accel_j2(r, EARTH_MU_KM3_S2, j2=0.0010826355254902923, re_km=6378.1363)
        self.assertLess(float(np.linalg.norm(a - a_j2)), 1e-10)

    def test_accelerated_normalized_terms_match_python_reference(self):
        terms = [
            SphericalHarmonicTerm(
                n=degree,
                m=order,
                c_nm=(-1.0 if (degree + order) % 2 else 1.0) * 1.0e-5 / (degree + order + 1.0),
                s_nm=(0.0 if order == 0 else 7.0e-6 / (degree + 2.0 * order + 1.0)),
                normalized=True,
            )
            for degree in range(2, 9)
            for order in range(degree + 1)
        ]
        compiled = compile_spherical_harmonic_terms(terms)
        self.assertIsNotNone(compiled)
        samples = (
            (0.0, np.array([7000.0, 100.0, 200.0], dtype=float)),
            (1234.5, np.array([-4300.0, 5100.0, 1800.0], dtype=float)),
            (5999.0, np.array([250.0, -6900.0, 1200.0], dtype=float)),
            (3210.25, np.array([10.0, 15.0, 7200.0], dtype=float)),
        )

        for t_s, position in samples:
            with self.subTest(t_s=t_s, position=position.tolist()):
                expected = accel_spherical_harmonics_terms(
                    position,
                    t_s,
                    terms,
                    mu_km3_s2=EARTH_MU_KM3_S2,
                    re_km=6378.1363,
                    jd_utc_start=2459669.5,
                    frame_model="simple",
                    compiled=compiled,
                    use_acceleration=False,
                )
                actual = accel_spherical_harmonics_terms(
                    position,
                    t_s,
                    terms,
                    mu_km3_s2=EARTH_MU_KM3_S2,
                    re_km=6378.1363,
                    jd_utc_start=2459669.5,
                    frame_model="simple",
                    compiled=compiled,
                    use_acceleration=True,
                )

                np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=5.0e-18)

    def test_compiled_normalized_terms_are_reentrant_across_threads(self):
        terms = [
            SphericalHarmonicTerm(
                n=degree,
                m=order,
                c_nm=1.0e-6 / (degree + order + 1.0),
                s_nm=0.0 if order == 0 else -8.0e-7 / (degree + 1.0),
                normalized=True,
            )
            for degree in range(2, 18)
            for order in range(degree + 1)
        ]
        compiled = compile_spherical_harmonic_terms(terms)
        self.assertIsNotNone(compiled)
        positions = [
            np.array([7000.0 + index, -1200.0 + 3.0 * index, 800.0 - index])
            for index in range(24)
        ]

        def evaluate(position: np.ndarray) -> np.ndarray:
            return accel_spherical_harmonics_terms(
                position,
                987.0,
                terms,
                mu_km3_s2=EARTH_MU_KM3_S2,
                re_km=6378.1363,
                jd_utc_start=2459669.5,
                frame_model="simple",
                compiled=compiled,
                use_acceleration=False,
            )

        expected = [evaluate(position) for position in positions]
        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = list(pool.map(evaluate, positions))
        for expected_row, actual_row in zip(expected, actual, strict=True):
            np.testing.assert_allclose(actual_row, expected_row, rtol=0.0, atol=1.0e-18)

    def test_accelerated_normalized_terms_match_reference_for_truncated_orders(self):
        position = np.array([-4300.0, 5100.0, 1800.0], dtype=float)
        for max_order in (0, 3, 8):
            terms = [
                SphericalHarmonicTerm(
                    n=degree,
                    m=order,
                    c_nm=(-1.0 if (degree + order) % 2 else 1.0)
                    * 1.0e-5
                    / (degree + order + 1.0),
                    s_nm=(0.0 if order == 0 else 7.0e-6 / (degree + 2.0 * order + 1.0)),
                    normalized=True,
                )
                for degree in range(2, 9)
                for order in range(min(degree, max_order) + 1)
            ]
            compiled = compile_spherical_harmonic_terms(terms)
            self.assertIsNotNone(compiled)
            expected = accel_spherical_harmonics_terms(
                position,
                1234.5,
                terms,
                mu_km3_s2=EARTH_MU_KM3_S2,
                re_km=6378.1363,
                jd_utc_start=2459669.5,
                frame_model="simple",
                compiled=compiled,
                use_acceleration=False,
            )
            actual = accel_spherical_harmonics_terms(
                position,
                1234.5,
                terms,
                mu_km3_s2=EARTH_MU_KM3_S2,
                re_km=6378.1363,
                jd_utc_start=2459669.5,
                frame_model="simple",
                compiled=compiled,
                use_acceleration=True,
            )

            with self.subTest(max_order=max_order):
                np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=5.0e-18)

    def test_orbit_propagator_acceleration_mode_dispatches_normalized_terms(self):
        terms = [
            SphericalHarmonicTerm(
                n=2,
                m=0,
                c_nm=-4.841693259705e-4,
                normalized=True,
            ),
            SphericalHarmonicTerm(
                n=2,
                m=2,
                c_nm=2.43914352398e-6,
                s_nm=-1.40016683654e-6,
                normalized=True,
            ),
        ]
        env = {
            "jd_utc_start": 2459669.5,
            "spherical_harmonics_terms": terms,
            "_parsed_spherical_harmonics_terms": terms,
            "_compiled_spherical_harmonics_terms": compile_spherical_harmonic_terms(terms),
            "spherical_harmonics_reference_radius_km": 6378.1363,
            "spherical_harmonics_frame_model": "simple",
        }
        state = np.array([7000.0, 100.0, 200.0, 0.0, 7.5, 0.1], dtype=float)
        command = np.zeros(3, dtype=float)
        context = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
        for integrator, expected_calls in (("rk4", 4), ("rkf78", 13)):
            with self.subTest(integrator=integrator):
                baseline = OrbitPropagator(
                    integrator=integrator,
                    plugins=[spherical_harmonics_plugin],
                    acceleration_mode="off",
                    numeric_backend="python",
                )
                accelerated = OrbitPropagator(
                    integrator=integrator,
                    plugins=[spherical_harmonics_plugin],
                    acceleration_mode="auto",
                    numeric_backend="python",
                )
                expected = baseline.propagate(state, 1.0, 0.0, command, env, context)

                with (
                    patch.object(accelerated, "_acceleration_enabled", return_value=True),
                    patch.object(
                        orbit_propagator,
                        "_accelerated_spherical_harmonics_plugin",
                        wraps=orbit_propagator._accelerated_spherical_harmonics_plugin,
                    ) as accelerated_evaluator,
                ):
                    actual = accelerated.propagate(state, 1.0, 0.0, command, env, context)

                self.assertEqual(accelerated_evaluator.call_count, expected_calls)
                np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_spherical_harmonics_replaces_explicit_zonals_in_runtime_plugins(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "target": {"enabled": True, "specs": {"mass_kg": 100.0}},
                "chaser": {"enabled": False},
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "dynamics": {
                        "orbit": {
                            "j2": True,
                            "j3": True,
                            "j4": True,
                            "spherical_harmonics": {"enabled": True, "terms": [{"n": 2, "m": 0, "c_nm": -1e-3}]},
                        }
                    },
                },
            }
        )

        propagator = _build_orbit_propagator(cfg)

        self.assertIn(spherical_harmonics_plugin, propagator.plugins)
        self.assertNotIn(j2_plugin, propagator.plugins)
        self.assertNotIn(j3_plugin, propagator.plugins)
        self.assertNotIn(j4_plugin, propagator.plugins)

    def test_inline_terms_infer_degree_and_order(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "target": {"enabled": True, "specs": {"mass_kg": 100.0}},
                "chaser": {"enabled": False},
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "dynamics": {
                        "orbit": {
                            "spherical_harmonics": {
                                "enabled": True,
                                "terms": [
                                    {"n": 2, "m": 0, "c_nm": -4.8e-4, "normalized": True},
                                    {"n": 4, "m": 2, "c_nm": 1.0e-6, "normalized": True},
                                ],
                            }
                        }
                    },
                },
            }
        )

        orbit_cfg = dict(cfg.simulator.dynamics["orbit"])
        sh = dict(orbit_cfg["spherical_harmonics"])
        env = configure_spherical_harmonics_env({}, orbit_cfg)

        self.assertEqual(sh["degree"], 4)
        self.assertEqual(sh["order"], 2)
        self.assertEqual([(term.n, term.m) for term in env["spherical_harmonics_terms"]], [(2, 0), (4, 2)])

    def test_inline_terms_require_an_explicit_finite_coefficient(self):
        base = {
            "simulator": {
                "duration_s": 1.0,
                "dt_s": 1.0,
                "dynamics": {
                    "orbit": {
                        "spherical_harmonics": {
                            "enabled": True,
                            "terms": [{"n": 2, "m": 0}],
                        }
                    }
                },
            }
        }
        with self.assertRaisesRegex(ValueError, "requires at least one C or S coefficient"):
            scenario_config_from_dict(base)

        base["simulator"]["dynamics"]["orbit"]["spherical_harmonics"]["terms"] = [
            {"n": 2, "m": 2, "s_nm": float("nan")}
        ]
        with self.assertRaisesRegex(ValueError, r"terms\[0\]\.s_nm must be a finite number"):
            scenario_config_from_dict(base)

        base["simulator"]["dynamics"]["orbit"]["spherical_harmonics"]["terms"] = [
            {"n": 2, "m": 2, "s_nm": 1.0e-6}
        ]
        cfg = scenario_config_from_dict(base)
        term = cfg.simulator.dynamics["orbit"]["spherical_harmonics"]["terms"][0]
        self.assertEqual(term["s_nm"], 1.0e-6)

    def test_enabled_degree_order_without_coefficients_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "degree and order alone do not define a gravity field"):
            scenario_config_from_dict(
                {
                    "rocket": {"enabled": False},
                    "target": {"enabled": True, "specs": {"mass_kg": 100.0}},
                    "chaser": {"enabled": False},
                    "simulator": {
                        "duration_s": 1.0,
                        "dt_s": 1.0,
                        "dynamics": {
                            "orbit": {
                                "spherical_harmonics": {"enabled": True, "degree": 8, "order": 8}
                            }
                        },
                    },
                }
            )

    def test_nested_zonal_switch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "spherical_harmonics has unsupported field.*j2"):
            scenario_config_from_dict(
                {
                    "rocket": {"enabled": False},
                    "target": {"enabled": True, "specs": {"mass_kg": 100.0}},
                    "chaser": {"enabled": False},
                    "simulator": {
                        "duration_s": 1.0,
                        "dt_s": 1.0,
                        "dynamics": {
                            "orbit": {"spherical_harmonics": {"enabled": True, "j2": True}}
                        },
                    },
                }
            )

    def test_egm96_source_materializes_verified_real_coefficient_mode(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "target": {"enabled": True, "specs": {"mass_kg": 100.0}},
                "chaser": {"enabled": False},
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "dynamics": {
                        "orbit": {
                            "spherical_harmonics": {
                                "enabled": True,
                                "degree": 8,
                                "order": 6,
                                "source": "EGM96",
                                "allow_download": False,
                            }
                        }
                    },
                },
            }
        )

        env = configure_spherical_harmonics_env({}, cfg.simulator.dynamics["orbit"])

        self.assertTrue(env["spherical_harmonics_use_real_coefficients"])
        self.assertEqual(env["spherical_harmonics_model"], "EGM96")
        self.assertEqual(env["spherical_harmonics_max_degree"], 8)
        self.assertEqual(env["spherical_harmonics_max_order"], 6)
        self.assertFalse(env["spherical_harmonics_allow_download"])

    def test_egm96_explicit_file_produces_nonzero_plugin_acceleration(self):
        gfc_txt = "\n".join(
            [
                "modelname EGM96_TEST",
                "norm fully_normalized",
                "gfc 2 0 -4.84165371736e-04 0.0 0.0 0.0",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            coeff_path = Path(td) / "egm96-test.gfc"
            coeff_path.write_text(gfc_txt, encoding="utf-8")
            env = configure_spherical_harmonics_env(
                {},
                {
                    "spherical_harmonics": {
                        "enabled": True,
                        "degree": 2,
                        "order": 0,
                        "source": "egm96",
                        "coeff_path": str(coeff_path),
                        "allow_download": False,
                    }
                },
            )

            class _Ctx:
                mu_km3_s2 = EARTH_MU_KM3_S2

            state = np.array([7000.0, 100.0, 200.0, 0.0, 7.5, 0.0], dtype=float)
            acceleration = spherical_harmonics_plugin(0.0, state, env=env, ctx=_Ctx())

        self.assertGreater(float(np.linalg.norm(acceleration)), 0.0)

    def test_file_backed_field_requires_explicit_degree(self):
        with self.assertRaisesRegex(ValueError, "File-backed.*requires degree >= 2"):
            scenario_config_from_dict(
                {
                    "rocket": {"enabled": False},
                    "target": {"enabled": True, "specs": {"mass_kg": 100.0}},
                    "chaser": {"enabled": False},
                    "simulator": {
                        "duration_s": 1.0,
                        "dt_s": 1.0,
                        "dynamics": {
                            "orbit": {
                                "spherical_harmonics": {
                                    "enabled": True,
                                    "coeff_path": "gravity.gfc",
                                }
                            }
                        },
                    },
                }
            )

    def test_unknown_coefficient_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source must be one of"):
            scenario_config_from_dict(
                {
                    "rocket": {"enabled": False},
                    "target": {"enabled": True, "specs": {"mass_kg": 100.0}},
                    "chaser": {"enabled": False},
                    "simulator": {
                        "duration_s": 1.0,
                        "dt_s": 1.0,
                        "dynamics": {
                            "orbit": {
                                "spherical_harmonics": {
                                    "enabled": True,
                                    "degree": 8,
                                    "source": "unknown_model",
                                }
                            }
                        },
                    },
                }
            )

    def test_icgem_source_materializes_explicit_coefficient_file(self):
        gfc_txt = "\n".join(
            [
                "modelname TEST",
                "earth_gravity_constant 4.0123456789e14",
                "radius 7.012345e6",
                "norm fully_normalized",
                "gfc 2 0 -4.84165371736e-04 0.0 0.0 0.0",
                "gfc 2 2 2.43914352398e-06 -1.40016683654e-06 0.0 0.0",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            coeff_path = Path(td) / "test.gfc"
            coeff_path.write_text(gfc_txt, encoding="utf-8")
            orbit_cfg = {
                "spherical_harmonics": {
                    "enabled": True,
                    "degree": 2,
                    "order": 2,
                    "source": "icgem",
                    "coeff_path": str(coeff_path),
                }
            }
            env = configure_spherical_harmonics_env({}, orbit_cfg)

        self.assertEqual([(term.n, term.m) for term in env["spherical_harmonics_terms"]], [(2, 0), (2, 2)])
        self.assertEqual(Path(env["spherical_harmonics_source"]), coeff_path.resolve())
        self.assertEqual(env["spherical_harmonics_reference_radius_km"], 7012.345)
        self.assertEqual(env["spherical_harmonics_mu_km3_s2"], 401234.56789)

    def test_icgem_explicit_reference_radius_overrides_file_header(self):
        gfc_txt = "\n".join(
            [
                "earth_gravity_constant 3.986004415e14",
                "radius 7.0e6",
                "norm fully_normalized",
                "gfc 2 0 -4.84165371736e-04 0.0 0.0 0.0",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            coeff_path = Path(td) / "test.gfc"
            coeff_path.write_text(gfc_txt, encoding="utf-8")
            env = configure_spherical_harmonics_env(
                {},
                {
                    "spherical_harmonics": {
                        "enabled": True,
                        "degree": 2,
                        "order": 0,
                        "source": "icgem",
                        "coeff_path": str(coeff_path),
                        "reference_radius_km": 6378.0,
                    }
                },
            )

        self.assertEqual(env["spherical_harmonics_reference_radius_km"], 6378.0)
        self.assertEqual(env["spherical_harmonics_mu_km3_s2"], 398600.4415)

    def test_load_icgem_gfc_terms_normalized_flag(self):
        gfc_txt = "\n".join(
            [
                "modelname TEST",
                "norm fully_normalized",
                "gfc 2 0 -4.84165371736e-04 0.0 0.0 0.0",
                "gfc 2 2 2.43914352398e-06 -1.40016683654e-06 0.0 0.0",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.gfc"
            p.write_text(gfc_txt, encoding="utf-8")
            terms = load_icgem_gfc_terms(p, max_degree=8, max_order=8)
        self.assertEqual(len(terms), 2)
        self.assertTrue(all(t.normalized for t in terms))

    def test_configure_env_loads_hpop_terms_and_radius_from_explicit_path(self):
        with tempfile.TemporaryDirectory() as td:
            coeff_path = Path(td) / "GGM03C.txt"
            coeff_path.write_text(
                "2 0 -4.841693259705e-04 0.000000000000e+00 4.68460e-11 0.00000e+00\n",
                encoding="utf-8",
            )
            orbit_cfg = {
                "spherical_harmonics": {
                    "enabled": True,
                    "degree": 2,
                    "order": 0,
                    "source": "hpop_ggm03",
                    "coeff_path": str(coeff_path),
                }
            }
            env = configure_spherical_harmonics_env({}, orbit_cfg)
        self.assertIn("spherical_harmonics_terms", env)
        self.assertEqual([(t.n, t.m) for t in env["spherical_harmonics_terms"]], [(2, 0)])
        term = env["spherical_harmonics_terms"][0]
        self.assertEqual((term.n, term.m), (2, 0))
        self.assertTrue(term.normalized)
        self.assertEqual(Path(env["spherical_harmonics_source"]), coeff_path.resolve())
        self.assertAlmostEqual(float(env["spherical_harmonics_reference_radius_km"]), 6378.1363)

    def test_download_model_file_verifies_sha256_before_cache_write(self):
        payload = b"modelname TEST\n" + (b"gfc 2 0 0 0 0 0\n" * 128)
        digest = hashlib.sha256(payload).hexdigest()
        old_spec = spherical_harmonics._REAL_MODEL_DOWNLOADS.get("TESTMODEL")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.gfc"
            source.write_bytes(payload)
            outpath = root / "cache" / "TESTMODEL.gfc"
            spherical_harmonics._REAL_MODEL_DOWNLOADS["TESTMODEL"] = GravityModelDownload(
                urls=(source.as_uri(),),
                sha256=digest,
                min_size_bytes=1,
            )
            try:
                spherical_harmonics._download_model_file("TESTMODEL", outpath)
            finally:
                if old_spec is None:
                    spherical_harmonics._REAL_MODEL_DOWNLOADS.pop("TESTMODEL", None)
                else:
                    spherical_harmonics._REAL_MODEL_DOWNLOADS["TESTMODEL"] = old_spec

            self.assertEqual(outpath.read_bytes(), payload)

    def test_download_model_file_rejects_hash_mismatch_without_final_cache_file(self):
        payload = b"modelname TEST\n" + (b"gfc 2 0 0 0 0 0\n" * 128)
        old_spec = spherical_harmonics._REAL_MODEL_DOWNLOADS.get("TESTMODEL")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.gfc"
            source.write_bytes(payload)
            outpath = root / "cache" / "TESTMODEL.gfc"
            spherical_harmonics._REAL_MODEL_DOWNLOADS["TESTMODEL"] = GravityModelDownload(
                urls=(source.as_uri(),),
                sha256="0" * 64,
                min_size_bytes=1,
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "integrity check failed"):
                    spherical_harmonics._download_model_file("TESTMODEL", outpath)
            finally:
                if old_spec is None:
                    spherical_harmonics._REAL_MODEL_DOWNLOADS.pop("TESTMODEL", None)
                else:
                    spherical_harmonics._REAL_MODEL_DOWNLOADS["TESTMODEL"] = old_spec

            self.assertFalse(outpath.exists())

    def test_cached_real_terms_verifies_existing_managed_cache(self):
        payload = b"not the expected model"
        old_spec = spherical_harmonics._REAL_MODEL_DOWNLOADS.get("TESTMODEL")
        spherical_harmonics._cached_real_terms.cache_clear()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_file = root / ".orbital_engagement_lab" / "gravity_models" / "TESTMODEL.gfc"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(payload)
            spherical_harmonics._REAL_MODEL_DOWNLOADS["TESTMODEL"] = GravityModelDownload(
                urls=(),
                sha256="0" * 64,
                min_size_bytes=1,
            )
            try:
                with patch("pathlib.Path.home", return_value=root):
                    with self.assertRaisesRegex(RuntimeError, "integrity check failed"):
                        spherical_harmonics._cached_real_terms(
                            model="TESTMODEL",
                            coeff_path=None,
                            max_degree=2,
                            max_order=0,
                            allow_download=False,
                        )
            finally:
                spherical_harmonics._cached_real_terms.cache_clear()
                if old_spec is None:
                    spherical_harmonics._REAL_MODEL_DOWNLOADS.pop("TESTMODEL", None)
                else:
                    spherical_harmonics._REAL_MODEL_DOWNLOADS["TESTMODEL"] = old_spec

    def test_sealed_mode_defensively_disables_gravity_model_downloads(self):
        spherical_harmonics._cached_real_terms.cache_clear()
        with patch.dict(
            "os.environ",
            {"OEL_SEALED_MODE": "1", "OEL_ALLOW_GRAVITY_MODEL_DOWNLOADS": ""},
        ):
            with patch.object(spherical_harmonics, "_cached_real_terms", return_value=()) as load_terms:
                spherical_harmonics.load_real_earth_gravity_terms(
                    max_degree=2,
                    max_order=0,
                    allow_download=True,
                )

        self.assertFalse(load_terms.call_args.kwargs["allow_download"])

    def test_sealed_mode_allows_explicitly_approved_gravity_model_downloads(self):
        spherical_harmonics._cached_real_terms.cache_clear()
        with patch.dict(
            "os.environ",
            {"OEL_SEALED_MODE": "1", "OEL_ALLOW_GRAVITY_MODEL_DOWNLOADS": "1"},
        ):
            with patch.object(spherical_harmonics, "_cached_real_terms", return_value=()) as load_terms:
                spherical_harmonics.load_real_earth_gravity_terms(
                    max_degree=2,
                    max_order=0,
                    allow_download=True,
                )

        self.assertTrue(load_terms.call_args.kwargs["allow_download"])


if __name__ == "__main__":
    unittest.main()
