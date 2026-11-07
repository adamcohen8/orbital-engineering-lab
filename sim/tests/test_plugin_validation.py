import sys
import tempfile
import unittest
from pathlib import Path

from sim.config import scenario_config_from_dict as _parse_scenario_config_dict
from sim.config import validate_scenario_plugins
from sim.config.plugin_validation import _validate_object_knowledge
from sim.runtime.actuator_factory import _build_satellite_actuator_stack_from_specs

DEEP_SPACE_LINE1 = "1 90003U 24003A   24001.00000000  .00000000  00000+0  00000+0 0    10"
DEEP_SPACE_LINE2 = "2 90003  10.0000  20.0000 0100000  30.0000  40.0000  4.00000000    10"


def scenario_config_from_dict(data: dict):
    root = dict(data)
    objects = dict(root.get("objects", {}) or {})
    for object_id in ("rocket", "chaser", "target"):
        if object_id in root:
            objects.setdefault(object_id, root.pop(object_id))
    if objects:
        root["objects"] = objects
    return _parse_scenario_config_dict(root)


class TestPluginValidation(unittest.TestCase):
    def test_general_sgp4_tle_object_passes_validation(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "propagation_method": "general",
                    "general": {"model": "sgp4"},
                    "specs": {"mass_kg": 420.0},
                    "initial_state": {
                        "tle": {
                            "line1": "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003",
                            "line2": "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004",
                        }
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        self.assertEqual(validate_scenario_plugins(cfg), [])

    def test_general_sgp4_tle_object_accepts_native_teme_output(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "propagation_method": "general",
                    "general": {"model": "sgp4", "output_frame": "teme"},
                    "specs": {"mass_kg": 420.0},
                    "initial_state": {
                        "tle": {
                            "line1": "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003",
                            "line2": "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004",
                        }
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        self.assertEqual(validate_scenario_plugins(cfg), [])

    def test_general_sgp4_tle_object_accepts_vallado_iau80_eci_transform(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "propagation_method": "general",
                    "general": {"model": "sgp4", "output_frame": "eci", "frame_transform": "teme_to_eci_iau80"},
                    "specs": {"mass_kg": 420.0},
                    "initial_state": {
                        "tle": {
                            "line1": "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003",
                            "line2": "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004",
                        }
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        self.assertEqual(validate_scenario_plugins(cfg), [])

    def test_general_sgp4_teme_output_rejects_teme_as_eci_transform(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "propagation_method": "general",
                    "general": {"model": "sgp4", "output_frame": "teme", "frame_transform": "teme_as_eci"},
                    "specs": {"mass_kg": 420.0},
                    "initial_state": {
                        "tle": {
                            "line1": "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003",
                            "line2": "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004",
                        }
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)

        self.assertTrue(any("frame_transform must be 'native'" in err for err in errs))

    def test_general_sgp4_accepts_deep_space_ogp_sdp4_tle(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "propagation_method": "general",
                    "general": {"model": "sgp4"},
                    "specs": {"mass_kg": 420.0},
                    "initial_state": {
                        "tle": {
                            "line1": DEEP_SPACE_LINE1,
                            "line2": DEEP_SPACE_LINE2,
                            "require_checksum": True,
                        }
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)

        self.assertEqual(errs, [])

    def test_general_sgp4_rejects_active_control(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "propagation_method": "general",
                    "general": {"model": "sgp4"},
                    "specs": {"mass_kg": 420.0},
                    "initial_state": {
                        "tle": {
                            "line1": "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003",
                            "line2": "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004",
                        }
                    },
                    "flight_software": {
                        "stack": "fsw.orbit_reference",
                        "hardware_profile": "hardware.ideal_wrench.v1",
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)

        self.assertTrue(any("flight_software must use fsw.passive" in err for err in errs))

    def test_general_sgp4_rejects_attitude_control(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "propagation_method": "general",
                    "general": {"model": "sgp4"},
                    "specs": {"mass_kg": 420.0},
                    "initial_state": {
                        "tle": {
                            "line1": "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003",
                            "line2": "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004",
                        }
                    },
                    "flight_software": {
                        "stack": "fsw.attitude_reference",
                        "hardware_profile": "hardware.ideal_wrench.v1",
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)

        self.assertTrue(any("flight_software must use fsw.passive" in err for err in errs))

    def test_general_sgp4_rejects_non_tle_initializers(self):
        with self.assertRaisesRegex(ValueError, "exactly one orbital-state form"):
            scenario_config_from_dict(
                {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "propagation_method": "general",
                    "general": {"model": "sgp4"},
                    "specs": {"mass_kg": 420.0},
                    "initial_state": {
                        "tle": {
                            "line1": "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003",
                            "line2": "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004",
                        },
                        "relative_ric_rect": [0, 0, 0, 0, 0, 0],
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                }
            )

    def test_tle_rejects_unsupported_sgp4_propagator_field(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {"mass_kg": 420.0},
                    "initial_state": {
                        "tle": {
                            "line1": "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003",
                            "line2": "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004",
                            "propagator": "sgp4",
                        }
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)

        self.assertTrue(any("propagation_method: general" in err and "general.model: sgp4" in err for err in errs))

    def test_knowledge_sensor_block_is_rejected_as_unsupported_modeled_sensor(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {"mass_kg": 420.0},
                    "knowledge": {
                        "sensor": {
                            "type": "optical_camera",
                            "aperture_m": 0.2,
                            "limiting_magnitude": 12.0,
                        }
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)

        self.assertTrue(any("unsupported modeled-sensor configuration block" in err for err in errs))

    def test_knowledge_sensor_error_and_estimation_pass_validation(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {"mass_kg": 420.0},
                    "knowledge": {
                        "refresh_rate_s": 2.0,
                        "targets": ["chaser"],
                        "sensor_error": {
                            "pos_sigma_km": [0.01, 0.01, 0.01],
                            "vel_sigma_km_s": [0.0001, 0.0001, 0.0001],
                        },
                        "estimation": {"type": "ekf"},
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        self.assertEqual(validate_scenario_plugins(cfg), [])

    def test_knowledge_validation_rejects_malformed_targets_and_ekf_diagonals(self):
        errors = _validate_object_knowledge(
            {
                "targets": {"target": True},
                "estimation": {
                    "ekf": {
                        "process_noise_diag": [1.0, 2.0, 3.0],
                        "init_cov_diag": [1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
                    }
                },
            },
            "objects.chaser.knowledge",
        )

        self.assertTrue(any("targets" in error and "list" in error for error in errors))
        self.assertTrue(any("process_noise_diag" in error and "six" in error for error in errors))
        self.assertTrue(any("init_cov_diag" in error and ">= 0" in error for error in errors))

    def test_knowledge_sensor_error_rejects_nonfinite_and_negative_noise(self):
        for field_name, value in (
            ("pos_sigma_km", [-0.01, 0.01, 0.01]),
            ("vel_sigma_km_s", [0.0001, float("nan"), 0.0001]),
            ("range_sigma_km", float("inf")),
        ):
            with self.subTest(field_name=field_name):
                cfg = scenario_config_from_dict(
                    {
                        "rocket": {"enabled": False},
                        "chaser": {"enabled": False},
                        "target": {
                            "enabled": True,
                            "specs": {"mass_kg": 420.0},
                            "knowledge": {
                                "targets": ["chaser"],
                                "sensor_error": {field_name: value},
                            },
                        },
                        "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                    }
                )

                errs = validate_scenario_plugins(cfg)

                self.assertTrue(any(field_name in err for err in errs), errs)

    def test_valid_plugins_pass(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {
                    "enabled": True,
                    "base_guidance": {
                        "module": "sim.rocket.guidance",
                        "class_name": "OpenLoopPitchProgramGuidance",
                        "params": {},
                    },
                    "guidance_modifiers": [
                        {
                            "module": "sim.rocket.guidance",
                            "class_name": "MaxQThrottleLimiterGuidance",
                            "params": {"max_q_pa": 45000.0, "min_throttle": 0.1},
                        }
                    ],
                    "orbit_control": {
                        "module": "sim.control.orbit.zero_controller",
                        "class_name": "ZeroController",
                        "params": {},
                    },
                    "attitude_control": {
                        "module": "sim.control.attitude.zero_torque",
                        "class_name": "ZeroTorqueController",
                        "params": {},
                    },
                },
                "chaser": {
                    "enabled": True,
                    "flight_software": {
                        "stack": "fsw.passive",
                        "hardware_profile": "hardware.passive.v1",
                    },
                },
                "target": {"enabled": False},
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )
        errs = validate_scenario_plugins(cfg)
        self.assertEqual(errs, [])

    def test_satellite_guidance_is_rejected_at_parse_time(self):
        with self.assertRaises(ValueError):
            scenario_config_from_dict(
                {
                    "rocket": {"enabled": False},
                    "chaser": {
                        "enabled": True,
                        "guidance": {
                            "module": "sim.control.orbit.zero_controller",
                            "class_name": "ZeroController",
                            "params": {},
                        },
                    },
                    "target": {"enabled": False},
                    "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                }
            )

    def test_invalid_plugins_fail(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {
                    "enabled": True,
                    "base_guidance": {
                        "module": "sim.control.orbit.zero_controller",
                        "class_name": "ZeroController",
                        "params": {},
                    },
                },
                "chaser": {"enabled": False},
                "target": {"enabled": False},
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )
        errs = validate_scenario_plugins(cfg)
        self.assertTrue(any("rocket.base_guidance" in e for e in errs))

    def test_safe_validation_does_not_import_plugin_modules(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "chaser": {
                    "enabled": True,
                    "flight_software": {
                        "module": "module_that_must_not_be_imported_for_safe_validation",
                        "class_name": "StackShapeOnly",
                        "hardware_profile": "hardware.passive.v1",
                    },
                },
                "target": {"enabled": False},
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        self.assertEqual(validate_scenario_plugins(cfg, import_plugins=False), [])
        self.assertTrue(validate_scenario_plugins(cfg, import_plugins=True))

    def test_named_object_plugins_are_validated(self):
        cfg = scenario_config_from_dict(
            {
                "objects": {
                    "blue_one": {
                        "kind": "satellite",
                        "enabled": True,
                        "flight_software": {
                            "module": "sim.rocket.guidance",
                            "class_name": "OpenLoopPitchProgramGuidance",
                            "hardware_profile": "hardware.passive.v1",
                        },
                    }
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)
        self.assertTrue(any("objects.blue_one.flight_software" in e for e in errs))

    def test_validation_does_not_construct_plugin_classes(self):
        with tempfile.TemporaryDirectory() as td:
            plugin_path = Path(td) / "constructor_side_effect_plugin.py"
            plugin_path.write_text(
                "\n".join(
                    [
                        "class ConstructorRaisesStack:",
                        "    def __init__(self, *args, **kwargs):",
                        "        raise RuntimeError('constructor should not run during validation')",
                        "",
                        "    def boot(self, event): pass",
                        "    def step(self, batch): pass",
                        "    def shutdown(self, event): pass",
                        "    def snapshot(self): pass",
                        "    def restore(self, snapshot): pass",
                    ]
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, td)
            try:
                cfg = scenario_config_from_dict(
                    {
                        "rocket": {"enabled": False},
                        "chaser": {
                            "enabled": True,
                            "flight_software": {
                                "module": "constructor_side_effect_plugin",
                                "class_name": "ConstructorRaisesStack",
                                "hardware_profile": "hardware.passive.v1",
                                "params": {"needs_runtime_context": True},
                            },
                        },
                        "target": {"enabled": False},
                        "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                    }
                )

                self.assertEqual(validate_scenario_plugins(cfg), [])
            finally:
                sys.path.remove(td)
                sys.modules.pop("constructor_side_effect_plugin", None)

    def test_unknown_actuator_preset_fails_strict_validation(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {"mass_kg": 200.0, "actuator_preset": "BASIC_NOT_REAL"},
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)

        self.assertTrue(any("actuator_preset" in err and "BASIC_RCS_6DOF" in err for err in errs))

    def test_malformed_actuator_config_fails_strict_validation(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {
                        "mass_kg": 200.0,
                        "actuators": {
                            "enabled": True,
                            "orbital": {
                                "rcs_cluster": {
                                    "allocation_mode": "force_torque",
                                    "thrusters": [
                                        {
                                            "name": "bad-thruster",
                                            "position_body_m": [0.0, 0.0],
                                            "force_direction_body": [0.0, 0.0, 0.0],
                                            "max_thrust_n": -1.0,
                                        }
                                    ],
                                }
                            },
                        },
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)

        self.assertTrue(any("position_body_m" in err for err in errs))
        self.assertTrue(any("force_direction_body" in err and "nonzero" in err for err in errs))
        self.assertTrue(any("max_thrust_n" in err and ">=" in err for err in errs))

    def test_malformed_rcs_allocation_weight_returns_validation_error(self):
        cfg = scenario_config_from_dict(
            {
                "target": {
                    "enabled": True,
                    "specs": {
                        "mass_kg": 200.0,
                        "actuators": {
                            "enabled": True,
                            "orbital": {
                                "rcs_cluster": {
                                    "force_weight": "bad",
                                    "thrusters": [
                                        {
                                            "position_body_m": [0.0, 0.0, 0.0],
                                            "force_direction_body": [1.0, 0.0, 0.0],
                                            "max_thrust_n": 1.0,
                                        }
                                    ],
                                }
                            },
                        },
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)

        self.assertTrue(any("force_weight" in err and "number" in err for err in errs))

    def test_actuator_preset_with_local_overrides_passes_validation(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {
                        "mass_kg": 200.0,
                        "actuators": {
                            "preset": "BASIC_ELECTRIC_PROPULSION",
                            "orbital": {
                                "electric_propulsion": {
                                    "max_thrust_n": 0.25,
                                }
                            },
                        },
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        self.assertEqual(validate_scenario_plugins(cfg), [])

    def test_scalar_actuator_vector_fields_pass_strict_validation(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {
                        "mass_kg": 200.0,
                        "actuators": {
                            "enabled": True,
                            "attitude": {
                                "magnetorquers": {
                                    "max_dipole_a_m2": 10.0,
                                },
                            },
                        },
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        self.assertEqual(validate_scenario_plugins(cfg), [])

    def test_disabled_attitude_devices_are_not_instantiated(self):
        actuator, _limits, enabled = _build_satellite_actuator_stack_from_specs(
            {
                "actuators": {
                    "enabled": True,
                    "attitude": {
                        "reaction_wheels": {"enabled": False},
                        "magnetorquers": {"enabled": False},
                        "thruster_pulse": {"enabled": False},
                        "control_moment_gyros": {"enabled": False},
                        "wheel_desaturation": {"enabled": False},
                    },
                }
            }
        )

        self.assertTrue(enabled)
        self.assertIsNotNone(actuator)
        attitude = actuator.attitude
        self.assertIsNone(attitude.reaction_wheels)
        self.assertIsNone(attitude.magnetorquers)
        self.assertIsNone(attitude.thruster_pulse)
        self.assertIsNone(attitude.control_moment_gyros)
        self.assertIsNone(attitude.wheel_desaturation)

    def test_redundant_reaction_wheel_configuration_passes_validation(self):
        cfg = scenario_config_from_dict(
            {
                "target": {
                    "enabled": True,
                    "specs": {
                        "mass_kg": 200.0,
                        "actuators": {
                            "enabled": True,
                            "attitude": {
                                "reaction_wheels": {
                                    "max_torque_nm": [0.05, 0.05, 0.05, 0.05],
                                    "max_momentum_nms": [0.2, 0.2, 0.2, 0.2],
                                    "wheel_axes_body": [
                                        [1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [-0.577350269, -0.577350269, -0.577350269],
                                    ],
                                }
                            },
                        },
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        self.assertEqual(validate_scenario_plugins(cfg), [])


if __name__ == "__main__":
    unittest.main()
