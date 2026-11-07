import subprocess
import sys
import unittest

import numpy as np

from sim.config import scenario_config_from_dict
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.integrators import integrate_adaptive, rkf78_stage_trace, rkf78_step
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.runtime_support import _build_orbit_propagator


class TestOrbitIntegrators(unittest.TestCase):
    def test_rkf78_production_step_matches_diagnostic_stage_trace_exactly(self):
        x0 = np.linspace(0.1, 4.8, 48)
        weights = np.linspace(0.01, 0.48, 48)

        def deriv(t_s: float, x: np.ndarray) -> np.ndarray:
            return weights * x + 0.001 * t_s

        stages = rkf78_stage_trace(deriv, 123.0, x0, 0.25)
        k1, k6, k7, k8 = (stages[index]["k"] for index in (0, 5, 6, 7))
        k9, k10, k11, k12, k13 = (stages[index]["k"] for index in (8, 9, 10, 11, 12))
        expected_state = x0 + 0.25 * (
            (41.0 / 840.0) * k1
            + (34.0 / 105.0) * k6
            + (9.0 / 35.0) * k7
            + (9.0 / 35.0) * k8
            + (9.0 / 280.0) * k9
            + (9.0 / 280.0) * k10
            + (41.0 / 840.0) * k11
        )
        expected_error = 0.25 * (41.0 / 840.0) * (k1 + k11 - k12 - k13)

        state, error = rkf78_step(deriv, 123.0, x0, 0.25)

        np.testing.assert_array_equal(state, expected_state)
        np.testing.assert_array_equal(error, expected_error)

    def test_orbit_package_constant_import_does_not_eagerly_load_propagation_families(self):
        code = (
            "import sys; "
            "from sim.dynamics.orbit import EARTH_MU_KM3_S2; "
            "assert EARTH_MU_KM3_S2 > 0.0; "
            "blocked = {'sim.dynamics.orbit.atmosphere', 'sim.dynamics.orbit.ogp', "
            "'sim.dynamics.orbit.propagator', 'sim.dynamics.orbit.sdp4'}; "
            "assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)"
        )
        proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_rkf78_adaptive_step_matches_requested_outer_dt(self):
        eval_times: list[float] = []

        def deriv(t_s: float, x: np.ndarray) -> np.ndarray:
            eval_times.append(float(t_s))
            return x

        x0 = np.array([1.0], dtype=float)
        x1 = integrate_adaptive(
            deriv_fn=deriv,
            t_s=0.0,
            x=x0,
            dt_s=1.0,
            atol=1e-12,
            rtol=1e-10,
            method="rkf78",
        )

        self.assertTrue(np.all(x1 > x0))
        self.assertAlmostEqual(float(x1[0]), float(np.e), places=8)
        self.assertTrue(all(0.0 <= t <= 1.0 for t in eval_times))

    def test_adaptive_integrator_reports_accepted_and_rejected_steps(self):
        def deriv(_t_s: float, x: np.ndarray) -> np.ndarray:
            return x

        x0 = np.array([1.0], dtype=float)
        x1, info = integrate_adaptive(
            deriv_fn=deriv,
            t_s=0.0,
            x=x0,
            dt_s=1.0,
            atol=1e-12,
            rtol=1e-10,
            method="rkf78",
            h_init=1.0,
            return_info=True,
        )

        self.assertAlmostEqual(float(x1[0]), float(np.e), places=8)
        self.assertGreaterEqual(info.accepted_steps, 1)
        self.assertGreaterEqual(info.rejected_steps, 1)
        self.assertEqual(info.attempted_steps, info.accepted_steps + info.rejected_steps)
        self.assertIsNotNone(info.suggested_next_step_s)

    def test_adaptive_integrator_rejects_nonfinite_duration(self):
        with self.assertRaisesRegex(ValueError, "dt_s must be finite"):
            integrate_adaptive(
                deriv_fn=lambda _t_s, x: x,
                t_s=0.0,
                x=np.array([1.0], dtype=float),
                dt_s=float("nan"),
            )

    def test_adaptive_integrator_rejects_nonpositive_substep_budget(self):
        with self.assertRaisesRegex(ValueError, "max_substeps must be a positive integer"):
            integrate_adaptive(
                deriv_fn=lambda _t_s, x: x,
                t_s=0.0,
                x=np.array([1.0], dtype=float),
                dt_s=1.0,
                max_substeps=0,
            )

    def test_hpop_integrator_rejects_nonpositive_attempt_budget(self):
        from sim.dynamics.orbit.integrators import integrate_rkf78_hpop

        with self.assertRaisesRegex(ValueError, "max_attempts must be a positive integer"):
            integrate_rkf78_hpop(
                deriv_fn=lambda _t_s, x: x,
                t_s=0.0,
                x=np.array([1.0], dtype=float),
                dt_s=1.0,
                max_attempts=0,
            )

    def test_orbit_propagator_rejects_nonfinite_inputs_before_backend_dispatch(self):
        propagator = OrbitPropagator(model="two_body", numeric_backend="python")
        context = OrbitContext(mu_km3_s2=398600.4418, mass_kg=100.0)
        with self.assertRaisesRegex(ValueError, "state must be finite"):
            propagator.propagate(
                np.array([7000.0, 0.0, 0.0, 0.0, np.nan, 0.0]),
                1.0,
                0.0,
                np.zeros(3),
                {},
                context,
            )
        with self.assertRaisesRegex(ValueError, "dt_s must be finite"):
            propagator.propagate(
                np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]),
                float("nan"),
                0.0,
                np.zeros(3),
                {},
                context,
            )

    def test_python_rk4_preserves_signed_short_coasts_for_tracking(self):
        propagator = OrbitPropagator(model="two_body", numeric_backend="python", integrator="rk4")
        context = OrbitContext(mu_km3_s2=398600.4418, mass_kg=100.0)
        initial = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        forward = propagator.propagate(initial, 0.1, 0.0, np.zeros(3), {}, context)
        restored = propagator.propagate(forward, -0.1, 0.1, np.zeros(3), {}, context)
        np.testing.assert_allclose(restored, initial, rtol=0.0, atol=1e-10)

    def test_adaptive_tolerance_tightening_reduces_error(self):
        def deriv(_t_s: float, x: np.ndarray) -> np.ndarray:
            return x

        x0 = np.array([1.0], dtype=float)
        loose = integrate_adaptive(deriv, 0.0, x0, 2.0, atol=1e-8, rtol=1e-6, method="rkf78")
        tight = integrate_adaptive(deriv, 0.0, x0, 2.0, atol=1e-12, rtol=1e-10, method="rkf78")
        expected = float(np.exp(2.0))

        self.assertLess(abs(float(tight[0]) - expected), abs(float(loose[0]) - expected))

    def test_master_simulator_orbit_propagator_can_select_rkf78(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "rkf78_builder",
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "dynamics": {
                        "orbit": {
                            "integrator": "rkf78",
                            "adaptive_atol": 1e-12,
                            "adaptive_rtol": 1e-10,
                        }
                    },
                },
            }
        )

        prop = _build_orbit_propagator(cfg)
        self.assertEqual(prop.integrator, "rkf78")
        self.assertAlmostEqual(prop.adaptive_atol, 1e-12)
        self.assertAlmostEqual(prop.adaptive_rtol, 1e-10)

    def test_drag_only_orbit_propagator_skips_lift_plugin(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "drag_only_builder",
                "objects": {
                    "sat": {
                        "kind": "satellite",
                        "specs": {
                            "drag_area_m2": 2.0,
                            "cd": 2.2,
                        },
                    }
                },
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "environment": {"atmosphere_model": "harris_priester"},
                    "dynamics": {
                        "orbit": {
                            "drag": True,
                        }
                    },
                },
            }
        )

        prop = _build_orbit_propagator(cfg)
        plugin_names = [plugin.__name__ for plugin in prop.plugins]

        self.assertIn("drag_plugin", plugin_names)
        self.assertNotIn("lift_plugin", plugin_names)

    def test_aero_lift_orbit_propagator_keeps_lift_plugin(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "aero_lift_builder",
                "objects": {
                    "sat": {
                        "kind": "satellite",
                        "specs": {
                            "aero": {
                                "drag_area_m2": 2.0,
                                "cd": 0.2,
                                "cl": 1.2,
                                "lift_area_m2": 20.0,
                                "lift_axis_body": [0.0, 0.0, 1.0],
                            }
                        },
                    }
                },
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "environment": {"atmosphere_model": "harris_priester"},
                    "dynamics": {
                        "orbit": {
                            "drag": True,
                        }
                    },
                },
            }
        )

        prop = _build_orbit_propagator(cfg)
        plugin_names = [plugin.__name__ for plugin in prop.plugins]

        self.assertIn("drag_plugin", plugin_names)
        self.assertIn("lift_plugin", plugin_names)

    def test_orbit_propagator_exposes_adaptive_step_accounting(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "rkf78_builder",
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "dynamics": {
                        "orbit": {
                            "integrator": "rkf78",
                            "adaptive_atol": 1e-12,
                            "adaptive_rtol": 1e-10,
                        }
                    },
                },
            }
        )
        prop = _build_orbit_propagator(cfg)

        prop.propagate(
            np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float),
            1.0,
            0.0,
            np.zeros(3),
            {},
            OrbitContext(mu_km3_s2=398600.4418, mass_kg=100.0),
        )

        self.assertIsNotNone(prop.last_adaptive_step_info)
        self.assertGreater(prop.last_adaptive_step_info.accepted_steps, 0)
        self.assertEqual(prop.adaptive_step_info.accepted_steps, prop.last_adaptive_step_info.accepted_steps)


if __name__ == "__main__":
    unittest.main()
