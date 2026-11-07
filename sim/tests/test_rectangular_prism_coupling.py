import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sim import SimulationConfig
from sim.config.scenario_yaml import scenario_config_from_dict as _parse_scenario_config_dict
from sim.core.models import StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.orbit.environment import SOLAR_PRESSURE_N_M2
from sim.dynamics.spacecraft_geometry import GeometryAreaProfile, RectangularPrismGeometry
from sim.presets import build_sim_object_from_presets
from sim.runtime_support import _create_satellite_runtime


def scenario_config_from_dict(data: dict, **kwargs):
    return _parse_scenario_config_dict(SimulationConfig.from_dict(data).to_dict(), **kwargs)


def _box_mesh(lx_m: float, ly_m: float, lz_m: float) -> tuple[np.ndarray, np.ndarray]:
    hx = 0.5 * lx_m
    hy = 0.5 * ly_m
    hz = 0.5 * lz_m
    faces = [
        (np.array([1.0, 0.0, 0.0]), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
        (np.array([-1.0, 0.0, 0.0]), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
        (np.array([0.0, 1.0, 0.0]), [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)]),
        (np.array([0.0, -1.0, 0.0]), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        (np.array([0.0, 0.0, 1.0]), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        (np.array([0.0, 0.0, -1.0]), [(-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz)]),
    ]
    triangles = []
    normals = []
    for normal, verts in faces:
        quad = [np.array(v, dtype=float) for v in verts]
        triangles.extend([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]])
        normals.extend([normal, normal])
    return np.asarray(triangles, dtype=float), np.asarray(normals, dtype=float)


class TestRectangularPrismCoupling(unittest.TestCase):
    def test_prism_rejects_nonfinite_dimensions(self):
        for bad_dimension in (np.nan, np.inf, -np.inf):
            with self.subTest(bad_dimension=bad_dimension):
                with self.assertRaises(ValueError):
                    RectangularPrismGeometry(bad_dimension, 1.0, 1.0)

    def test_geometry_profile_rejects_nonfinite_mesh_inputs(self):
        triangles = np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=float,
        )
        directions = np.array([[1.0, 0.0, 0.0]], dtype=float)

        bad_triangles = triangles.copy()
        bad_triangles[0, 1, 2] = np.nan
        with self.assertRaises(ValueError):
            GeometryAreaProfile.from_triangles(bad_triangles, directions_body=directions)

        bad_normals = np.array([[np.inf, 0.0, 1.0]], dtype=float)
        with self.assertRaises(ValueError):
            GeometryAreaProfile.from_triangles(
                triangles,
                bad_normals,
                directions_body=directions,
            )

    def test_projected_area_matches_expected_faces(self):
        g = RectangularPrismGeometry(lx_m=2.0, ly_m=3.0, lz_m=4.0)
        self.assertAlmostEqual(g.projected_area_m2(np.array([1.0, 0.0, 0.0])), 12.0, places=12)
        self.assertAlmostEqual(g.projected_area_m2(np.array([0.0, 1.0, 0.0])), 8.0, places=12)
        self.assertAlmostEqual(g.projected_area_m2(np.array([0.0, 0.0, 1.0])), 6.0, places=12)

        u = np.array([1.0, 1.0, 0.0], dtype=float) / np.sqrt(2.0)
        expected = (12.0 + 8.0) / np.sqrt(2.0)
        self.assertAlmostEqual(g.projected_area_m2(u), expected, places=12)

    def test_geometry_area_profile_from_box_mesh_matches_prism_axes(self):
        directions = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        triangles, normals = _box_mesh(2.0, 3.0, 4.0)
        profile = GeometryAreaProfile.from_triangles(triangles, normals, directions_body=directions)
        prism = RectangularPrismGeometry(lx_m=2.0, ly_m=3.0, lz_m=4.0)

        for direction in directions:
            self.assertAlmostEqual(
                profile.projected_area_for_direction_m2(direction),
                prism.projected_area_m2(direction),
                places=12,
            )

    def test_geometry_area_profile_round_trips_json(self):
        triangles, normals = _box_mesh(1.0, 2.0, 3.0)
        profile = GeometryAreaProfile.from_triangles(
            triangles,
            normals,
            directions_body=np.array([[1.0, 0.0, 0.0]], dtype=float),
            metadata={"name": "unit-test-box"},
        )
        with TemporaryDirectory() as td:
            path = Path(td) / "profile.json"
            profile.save(path)
            loaded = GeometryAreaProfile.load(path)
        self.assertEqual(loaded.metadata["name"], "unit-test-box")
        self.assertAlmostEqual(loaded.projected_area_for_direction_m2(np.array([1.0, 0.0, 0.0])), 6.0)

    def test_face_torque_symmetric_axis_flow_is_zero(self):
        g = RectangularPrismGeometry(lx_m=1.2, ly_m=1.0, lz_m=0.8)
        tau = g.face_torque_sum_body_nm(np.array([1.0, 0.0, 0.0]), pressure_n_m2=2.0)
        self.assertTrue(np.linalg.norm(tau) < 1e-12)

    def test_face_torque_uses_center_of_mass_as_moment_origin(self):
        g = RectangularPrismGeometry(lx_m=1.0, ly_m=2.0, lz_m=2.0)
        tau = g.face_torque_sum_body_nm(
            np.array([-1.0, 0.0, 0.0]),
            pressure_n_m2=2.0,
            moment_origin_body_m=np.array([0.0, 0.5, 0.0], dtype=float),
        )

        self.assertTrue(np.allclose(tau, np.array([0.0, 0.0, -4.0])))

    def test_face_force_follows_incoming_flux_direction(self):
        g = RectangularPrismGeometry(lx_m=2.0, ly_m=3.0, lz_m=4.0)
        incoming = np.array([-1.0, 0.0, 0.0], dtype=float)
        f_total = np.sum(g.face_forces_body_n(incoming, pressure_n_m2=2.0), axis=0)
        self.assertTrue(np.allclose(f_total, np.array([-24.0, 0.0, 0.0])))

    def test_prism_srp_torque_matches_incoming_sun_direction(self):
        cfg = DisturbanceTorqueConfig(
            use_gravity_gradient=False,
            use_magnetic=False,
            use_drag=False,
            use_srp=True,
            use_rectangular_prism_faces=True,
            rectangular_prism_dims_m=(1.0, 2.0, 3.0),
            srp_cr=1.0,
        )
        model = DisturbanceTorqueModel(mu_km3_s2=398600.4418, inertia_kg_m2=np.diag([1.0, 2.0, 3.0]), config=cfg)
        truth = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=100.0,
            t_s=0.0,
        )
        sun_dir = np.array([1.0, 1.0, 0.0], dtype=float) / np.sqrt(2.0)
        p_srp = SOLAR_PRESSURE_N_M2
        expected = RectangularPrismGeometry(1.0, 2.0, 3.0).face_torque_sum_body_nm(-sun_dir, p_srp)
        tau = model.total_torque_body_nm(
            truth,
            env={"sun_dir_eci": sun_dir, "srp_shadow_model": "none"},
        )
        self.assertTrue(np.allclose(tau, expected))
        self.assertIs(model._rect_prism_geometry, model._rect_prism_geometry)

    def test_geometry_profile_srp_torque_uses_profile_center_of_pressure(self):
        profile = GeometryAreaProfile(
            directions_body=np.array([[-1.0, 0.0, 0.0]], dtype=float),
            projected_area_m2=np.array([2.0], dtype=float),
            center_of_pressure_body_m=np.array([[0.0, 1.0, 0.0]], dtype=float),
            metadata={"name": "offset plate"},
        )
        cfg = DisturbanceTorqueConfig(
            use_gravity_gradient=False,
            use_magnetic=False,
            use_drag=False,
            use_srp=True,
            srp_cr=1.0,
            geometry_area_profile=profile,
        )
        model = DisturbanceTorqueModel(mu_km3_s2=398600.4418, inertia_kg_m2=np.eye(3), config=cfg)
        truth = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=100.0,
            t_s=0.0,
        )
        tau = model.total_torque_body_nm(
            truth,
            env={"sun_dir_eci": np.array([1.0, 0.0, 0.0]), "srp_shadow_model": "none"},
        )
        self.assertTrue(np.allclose(tau, np.array([0.0, 0.0, 2.0 * SOLAR_PRESSURE_N_M2])))

    def test_geometry_profile_lookup_preserves_area_weighted_center_of_pressure(self):
        profile = GeometryAreaProfile(
            directions_body=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            projected_area_m2=np.array([100.0, 1.0], dtype=float),
            center_of_pressure_body_m=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        )

        lookup = profile.lookup(np.array([1.0, 1.0, 0.0], dtype=float), nearest_neighbors=2)

        self.assertLess(float(lookup.center_of_pressure_body_m[0]), 0.2)

    def test_prism_mode_requires_disturbance_enabled(self):
        with self.assertRaises(ValueError):
            build_sim_object_from_presets(
                object_id="sat_prism_invalid",
                dt_s=1.0,
                enable_disturbances=False,
                use_rectangular_prism_aero_srp=True,
                rectangular_prism_dims_m=(1.0, 1.0, 1.0),
            )

    def test_prism_mode_wires_dynamics_and_disturbance(self):
        sat = build_sim_object_from_presets(
            object_id="sat_prism_valid",
            dt_s=1.0,
            enable_disturbances=True,
            use_rectangular_prism_aero_srp=True,
            rectangular_prism_dims_m=(1.4, 1.1, 0.9),
        )
        self.assertTrue(sat.dynamics.use_rectangular_prism_for_aero_srp)
        self.assertEqual(tuple(float(v) for v in sat.dynamics.rectangular_prism_dims_m), (1.4, 1.1, 0.9))
        self.assertIsNotNone(sat.dynamics.disturbance_model)
        cfg = sat.dynamics.disturbance_model.config
        self.assertTrue(cfg.use_rectangular_prism_faces)
        self.assertEqual(tuple(float(v) for v in cfg.rectangular_prism_dims_m), (1.4, 1.1, 0.9))

    def test_scenario_config_resolves_geometry_profile_path(self):
        profile = GeometryAreaProfile(
            directions_body=np.array([[1.0, 0.0, 0.0]], dtype=float),
            projected_area_m2=np.array([1.0], dtype=float),
            center_of_pressure_body_m=np.zeros((1, 3)),
        )
        with TemporaryDirectory() as td:
            root = Path(td)
            profile_path = root / "sat_profile.json"
            config_path = root / "scenario.yaml"
            profile.save(profile_path)
            cfg = scenario_config_from_dict(
                {
                    "scenario_name": "geometry_profile_path_test",
                    "chaser": {
                        "enabled": True,
                        "kind": "satellite",
                        "specs": {"geometry": {"profile_path": "sat_profile.json"}},
                    },
                    "target": {"enabled": False},
                    "outputs": {"output_dir": str(root / "outputs")},
                },
                source_path=config_path,
            )
            self.assertEqual(cfg.objects["chaser"].specs["geometry"]["profile_path"], str(profile_path.resolve()))

    def test_satellite_runtime_loads_geometry_profile(self):
        profile = GeometryAreaProfile(
            directions_body=np.array([[1.0, 0.0, 0.0]], dtype=float),
            projected_area_m2=np.array([1.0], dtype=float),
            center_of_pressure_body_m=np.zeros((1, 3)),
        )
        with TemporaryDirectory() as td:
            root = Path(td)
            profile_path = root / "sat_profile.json"
            config_path = root / "scenario.yaml"
            profile.save(profile_path)
            cfg = scenario_config_from_dict(
                {
                    "scenario_name": "geometry_profile_runtime_test",
                    "chaser": {
                        "enabled": True,
                        "kind": "satellite",
                        "specs": {"geometry": {"profile_path": "sat_profile.json"}},
                    },
                    "target": {"enabled": False},
                    "simulator": {
                        "dynamics": {
                            "attitude": {
                                "enabled": True,
                                "disturbance_torques": {"srp": True},
                            }
                        }
                    },
                    "outputs": {"output_dir": str(root / "outputs")},
                },
                source_path=config_path,
            )
            runtime = _create_satellite_runtime("chaser", cfg.objects["chaser"], cfg, np.random.default_rng(1))
        self.assertIsNotNone(runtime.dynamics.geometry_area_profile)
        self.assertIs(runtime.dynamics.disturbance_model.config.geometry_area_profile, runtime.dynamics.geometry_area_profile)


if __name__ == "__main__":
    unittest.main()
