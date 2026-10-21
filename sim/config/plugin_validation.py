from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import dataclass
from typing import Any

from sim.actuators.presets import available_actuator_preset_names, resolve_actuator_specs_from_satellite_specs
from sim.config.object_refs import configured_objects, object_parameter_prefix
from sim.config.plugin_specs import instantiate_plugin_spec, iter_nested_plugin_specs, plugin_spec_field
from sim.digital_twin.mass_properties import validate_mass_properties
from sim.dynamics.orbit.tle import ogp_mean_elements_from_mapping, parse_tle_lines
from sim.numeric_backend import normalize_numeric_backend


@dataclass(frozen=True)
class PluginContract:
    methods_all: tuple[str, ...] = ()
    methods_any: tuple[str, ...] = ()
    allow_function: bool = False


_CONTRACTS = {
    "flight_software": PluginContract(
        methods_all=("boot", "step", "shutdown", "snapshot", "restore"),
        allow_function=False,
    ),
    "guidance": PluginContract(methods_all=("command",), allow_function=False),
    "orbit_control": PluginContract(methods_all=("act",), allow_function=False),
    "attitude_control": PluginContract(methods_all=("act",), allow_function=False),
    "mission_strategy": PluginContract(methods_all=(), methods_any=("update", "plan", "decide"), allow_function=True),
    "mission_execution": PluginContract(methods_all=(), methods_any=("update", "execute", "act"), allow_function=True),
    "bridge": PluginContract(
        methods_all=(),
        methods_any=("step", "start", "send_command", "receive_command"),
        allow_function=True,
    ),
    "mission_objective": PluginContract(
        methods_all=(), methods_any=("evaluate", "update", "check", "act"), allow_function=True
    ),
    "force_model": PluginContract(methods_all=("acceleration",), allow_function=True),
    "system_force_model": PluginContract(methods_all=("accelerations",), allow_function=True),
}


def _validate_pointer(pointer: Any, contract: PluginContract, path: str, *, import_plugins: bool = True) -> list[str]:
    errs: list[str] = []
    if pointer is None:
        return errs
    builtin = str(plugin_spec_field(pointer, "builtin", "") or "").strip()
    if builtin:
        try:
            from sim.gnc.catalog import resolve_builtin_target

            resolved_module, resolved_class = resolve_builtin_target(builtin)
        except ValueError as exc:
            return [f"{path}: {exc}"]
    else:
        resolved_module = resolved_class = ""
    module_name = str(plugin_spec_field(pointer, "module", "") or "").strip()
    module_name = module_name or resolved_module
    if not module_name:
        errs.append(f"{path}: missing 'module'.")
        return errs
    class_name = plugin_spec_field(pointer, "class_name", None) or resolved_class or None
    function = plugin_spec_field(pointer, "function", None)
    if class_name and function:
        errs.append(f"{path}: define either 'class_name' or 'function', not both.")
        return errs
    if function and not contract.allow_function:
        errs.append(f"{path}: function pointers are not allowed for this plugin type.")
        return errs
    if not class_name and not function:
        errs.append(f"{path}: must define either 'class_name' or 'function'.")
        return errs
    if not import_plugins:
        return errs

    try:
        mod = importlib.import_module(module_name)
    except Exception as ex:
        errs.append(f"{path}: failed to import module '{module_name}': {ex}")
        return errs

    if class_name:
        if not hasattr(mod, class_name):
            errs.append(f"{path}: class '{class_name}' not found in module '{module_name}'.")
            return errs
        cls = getattr(mod, class_name)
        if not inspect.isclass(cls):
            errs.append(f"{path}: '{class_name}' in module '{module_name}' is not a class.")
            return errs
        for m in contract.methods_all:
            if not _class_has_callable(cls, m):
                errs.append(f"{path}: class '{class_name}' missing required callable method '{m}'.")
        if contract.methods_any:
            if not any(_class_has_callable(cls, m) for m in contract.methods_any):
                errs.append(f"{path}: class '{class_name}' must implement one of {list(contract.methods_any)}.")
        if builtin and not errs:
            try:
                instantiate_plugin_spec(pointer, description=f"built-in GNC {builtin}")
            except RuntimeError as exc:
                errs.append(f"{path}: {exc}")
        return errs

    if function:
        if not hasattr(mod, function):
            errs.append(f"{path}: function '{function}' not found in module '{module_name}'.")
            return errs
        fn = getattr(mod, function)
        if not callable(fn):
            errs.append(f"{path}: '{function}' in module '{module_name}' is not callable.")
        return errs

    errs.append(f"{path}: must define either 'class_name' or 'function'.")
    return errs


def _class_has_callable(cls: type, method_name: str) -> bool:
    for base in cls.__mro__:
        if method_name in base.__dict__:
            attr = base.__dict__[method_name]
            if isinstance(attr, (staticmethod, classmethod)):
                return callable(attr.__func__)
            return callable(attr)
    return False


def validate_scenario_plugins(cfg: Any, *, import_plugins: bool = True) -> list[str]:
    errs: list[str] = []
    orbit_cfg = dict(getattr(getattr(getattr(cfg, "simulator", None), "dynamics", {}), "orbit", {}) or {})
    default_propagation_method = str(orbit_cfg.get("propagation_method", "special") or "special").strip().lower()
    orbit_model = str(orbit_cfg.get("model", "two_body") or "two_body").strip().lower()
    game_cfg = dict(getattr(cfg, "metadata", {}).get("game", {}) or {})
    game_controlled_object_id = str(game_cfg.get("controlled_object_id", "chaser") or "chaser")
    for index, pointer in enumerate(getattr(cfg.simulator, "system_force_models", []) or []):
        pointer_path = f"simulator.system_force_models[{index}]"
        if plugin_spec_field(pointer, "builtin"):
            errs.append(f"{pointer_path}: use an importable module and class_name or function, not builtin.")
            continue
        if plugin_spec_field(pointer, "function") and plugin_spec_field(pointer, "params"):
            errs.append(f"{pointer_path}: function system force models do not accept params; use a class.")
        errs.extend(_validate_pointer(
            pointer, _CONTRACTS["system_force_model"], pointer_path, import_plugins=import_plugins
        ))
        for nested_path, nested_pointer in iter_nested_plugin_specs(pointer, pointer_path):
            errs.extend(_validate_pointer(
                nested_pointer, PluginContract(allow_function=True), nested_path,
                import_plugins=import_plugins,
            ))
    for object_id, agent in configured_objects(cfg).items():
        if not getattr(agent, "enabled", False):
            continue
        path = object_parameter_prefix(str(object_id))
        propagation_method = str(
            getattr(agent, "propagation_method", "") or default_propagation_method or "special"
        ).strip().lower()
        runtime_profile = str(getattr(agent, "runtime_profile", "flight_software") or "flight_software")
        if game_cfg and runtime_profile == "trajectory_only" and str(object_id) == game_controlled_object_id:
            errs.append(
                f"{path}.runtime_profile=trajectory_only cannot be the game controlled_object_id; "
                "pilot and operator input require a flight_software runtime."
            )
        errs.extend(_validate_object_propagation(agent, propagation_method, path))
        force_models = getattr(agent, "force_models", []) or []
        if force_models:
            if str(getattr(agent, "kind", "satellite") or "satellite").strip().lower() != "satellite":
                errs.append(f"{path}.force_models requires a satellite object.")
            if propagation_method != "special" or orbit_model == "cr3bp":
                errs.append(f"{path}.force_models requires ECI ONP special propagation.")
            if getattr(cfg.simulator, "initial_jd_utc", None) is None:
                errs.append(f"{path}.force_models requires simulator.initial_jd_utc.")
        for index, pointer in enumerate(force_models):
            pointer_path = f"{path}.force_models[{index}]"
            if plugin_spec_field(pointer, "builtin"):
                errs.append(f"{pointer_path}: use an importable module and class_name or function, not builtin.")
                continue
            if plugin_spec_field(pointer, "function") and plugin_spec_field(pointer, "params"):
                errs.append(f"{pointer_path}: function force models do not accept params; use a class for parameters.")
            errs.extend(_validate_pointer(
                pointer, _CONTRACTS["force_model"], pointer_path, import_plugins=import_plugins
            ))
        errs.extend(_validate_object_mass_properties(getattr(agent, "specs", {}) or {}, f"{path}.specs"))
        errs.extend(_validate_initial_state(getattr(agent, "initial_state", {}) or {}, f"{path}.initial_state"))
        errs.extend(_validate_object_knowledge(getattr(agent, "knowledge", {}) or {}, f"{path}.knowledge"))
        if str(getattr(agent, "kind", "satellite")).strip().lower() == "rocket":
            errs.extend(
                _validate_pointer(
                    getattr(agent, "guidance", None),
                    _CONTRACTS["guidance"],
                    f"{path}.guidance",
                    import_plugins=import_plugins,
                )
            )
            errs.extend(
                _validate_pointer(
                    getattr(agent, "base_guidance", None),
                    _CONTRACTS["guidance"],
                    f"{path}.base_guidance",
                    import_plugins=import_plugins,
                )
            )
            for i, p in enumerate(getattr(agent, "guidance_modifiers", []) or []):
                errs.extend(
                    _validate_rocket_guidance_modifier(
                        p, f"{path}.guidance_modifiers[{i}]", import_plugins=import_plugins
                    )
                )
        else:
            errs.extend(_validate_satellite_actuator_specs(getattr(agent, "specs", {}) or {}, f"{path}.specs"))
            flight_software = getattr(agent, "flight_software", None)
            if flight_software is not None and getattr(flight_software, "module", None):
                errs.extend(
                    _validate_pointer(
                        flight_software,
                        _CONTRACTS["flight_software"],
                        f"{path}.flight_software",
                        import_plugins=import_plugins,
                    )
                )
        bridge = getattr(agent, "bridge", None)
        if bridge is not None and getattr(bridge, "enabled", False):
            errs.extend(_validate_pointer(bridge, _CONTRACTS["bridge"], f"{path}.bridge", import_plugins=import_plugins))
            if _is_cfs_sil_bridge(bridge):
                errs.extend(_validate_cfs_sil_timing(cfg, f"{path}.bridge"))
        for i, p in enumerate(getattr(agent, "mission_objectives", []) or []):
            errs.extend(
                _validate_pointer(
                    p, _CONTRACTS["mission_objective"], f"{path}.mission_objectives[{i}]", import_plugins=import_plugins
                )
            )
        for pointer_path, pointer in _agent_plugin_pointers(agent, path):
            for nested_path, nested_pointer in iter_nested_plugin_specs(pointer, pointer_path):
                errs.extend(
                    _validate_pointer(
                        nested_pointer,
                        PluginContract(allow_function=True),
                        nested_path,
                        import_plugins=import_plugins,
                    )
                )
    return errs


def _agent_plugin_pointers(agent: Any, base_path: str) -> list[tuple[str, Any]]:
    pointers: list[tuple[str, Any]] = []
    flight_software = getattr(agent, "flight_software", None)
    if flight_software is not None and getattr(flight_software, "module", None):
        pointers.append((f"{base_path}.flight_software", flight_software))
    for field_name in (
        "guidance",
        "base_guidance",
        "orbit_control",
        "attitude_control",
        "mission_strategy",
        "mission_execution",
    ):
        pointer = getattr(agent, field_name, None)
        if pointer is not None:
            pointers.append((f"{base_path}.{field_name}", pointer))
    for field_name in ("guidance_modifiers", "mission_objectives"):
        for index, pointer in enumerate(getattr(agent, field_name, []) or []):
            pointers.append((f"{base_path}.{field_name}[{index}]", pointer))
    for index, pointer in enumerate(getattr(agent, "force_models", []) or []):
        pointers.append((f"{base_path}.force_models[{index}]", pointer))
    bridge = getattr(agent, "bridge", None)
    if bridge is not None and getattr(bridge, "enabled", False):
        pointers.append((f"{base_path}.bridge", bridge))
    return pointers


def _validate_object_propagation(agent: Any, propagation_method: str, path: str) -> list[str]:
    errs: list[str] = []
    method = str(propagation_method or "special").strip().lower()
    if method not in {"special", "general"}:
        return [f"{path}.propagation_method: must be one of: special, general."]
    if method != "general":
        return errs

    kind = str(getattr(agent, "kind", "satellite") or "satellite").strip().lower()
    if kind != "satellite":
        errs.append(f"{path}.propagation_method=general is only supported for satellite objects.")
    general = dict(getattr(agent, "general", {}) or {})
    model = str(general.get("model", "") or "").strip().lower()
    if model != "sgp4":
        errs.append(f"{path}.general.model must be 'sgp4' when propagation_method=general.")
    initial_state = dict(getattr(agent, "initial_state", {}) or {})
    tle_block = initial_state.get("tle")
    mean_element_block = initial_state.get("ogp_mean_elements")
    if isinstance(tle_block, dict):
        try:
            lines = tle_block.get("lines")
            if isinstance(lines, (list, tuple)) and len(lines) >= 2:
                line1 = str(lines[0])
                line2 = str(lines[1])
            else:
                line1 = str(tle_block.get("line1", "") or "")
                line2 = str(tle_block.get("line2", "") or "")
            elements = parse_tle_lines(line1, line2, require_checksum=bool(tle_block.get("require_checksum", True)))
            if float(elements.mean_motion_rev_per_day) <= 0.0:
                errs.append(f"{path}.initial_state.tle: OGP mean motion must be positive.")
            if float(elements.eccentricity) < 0.0 or float(elements.eccentricity) >= 1.0:
                errs.append(f"{path}.initial_state.tle: OGP eccentricity must be in [0, 1).")
        except Exception as ex:
            errs.append(f"{path}.initial_state.tle: invalid TLE for OGP propagation: {ex}")
    elif isinstance(mean_element_block, dict):
        try:
            ogp_mean_elements_from_mapping(mean_element_block)
        except Exception as ex:
            errs.append(f"{path}.initial_state.ogp_mean_elements: invalid OGP mean elements: {ex}")
    else:
        errs.append(
            f"{path}.propagation_method=general with general.model=sgp4 requires "
            "initial_state.tle or initial_state.ogp_mean_elements."
        )
    selected_state_key = "ogp_mean_elements" if isinstance(mean_element_block, dict) else "tle"
    unsupported_initial_forms = [key for key in initial_state if str(key) != selected_state_key]
    if unsupported_initial_forms:
        errs.append(
            f"{path}.propagation_method=general with general.model=sgp4 does not support initial_state field(s): "
            + ", ".join(sorted(unsupported_initial_forms))
            + ". Use exactly one of initial_state.tle or initial_state.ogp_mean_elements."
        )
    flight_software = getattr(agent, "flight_software", None)
    if flight_software is not None:
        stack_id = str(getattr(flight_software, "stack", "") or "")
        custom_module = str(getattr(flight_software, "module", "") or "")
        if custom_module or stack_id not in {"", "fsw.passive"}:
            errs.append(
                f"{path}.flight_software must use fsw.passive, or select runtime_profile=trajectory_only, "
                "for passive general-propagation SGP4 objects."
            )
    if getattr(agent, "mission_strategy", None) is not None or getattr(agent, "mission_execution", None) is not None:
        errs.append(f"{path}.mission_strategy/mission_execution are not supported for passive general-propagation SGP4 objects.")
    if list(getattr(agent, "mission_objectives", []) or []):
        errs.append(f"{path}.mission_objectives are not supported for passive general-propagation SGP4 objects.")
    allowed_general_keys = {"model", "output_frame", "frame_transform", "max_tle_age_days_warning", "numeric_backend"}
    unknown_general_keys = sorted(str(key) for key in general if str(key) not in allowed_general_keys)
    if unknown_general_keys:
        errs.append(f"{path}.general has unsupported field(s): {', '.join(unknown_general_keys)}.")
    try:
        normalize_numeric_backend(general.get("numeric_backend", "rust"), error_message=f"{path}.general.numeric_backend must be 'python' or 'rust'.")
    except ValueError as exc:
        errs.append(str(exc))
    if general.get("max_tle_age_days_warning") is not None:
        try:
            max_age = float(general.get("max_tle_age_days_warning"))
            if not math.isfinite(max_age) or max_age < 0.0:
                raise ValueError
        except (TypeError, ValueError):
            errs.append(f"{path}.general.max_tle_age_days_warning must be a nonnegative finite number.")
    output_frame = str(general.get("output_frame", "teme") or "teme").strip().lower()
    if output_frame not in {"eci", "teme"}:
        errs.append(f"{path}.general.output_frame must be 'eci' or 'teme' for SGP4 v1.")
    default_transform = "native" if output_frame == "teme" else "teme_to_eci_iau80"
    frame_transform = str(general.get("frame_transform", default_transform) or default_transform).strip().lower()
    if output_frame == "eci" and frame_transform not in {"teme_as_eci", "teme_to_eci_iau80"}:
        errs.append(
            f"{path}.general.frame_transform must be 'teme_as_eci' or 'teme_to_eci_iau80' "
            "when output_frame is 'eci'."
        )
    if output_frame == "teme" and frame_transform not in {"native", "none", "identity", "teme"}:
        errs.append(f"{path}.general.frame_transform must be 'native' when output_frame is 'teme'.")
    return errs


def _validate_object_knowledge(knowledge: dict[str, Any], path: str) -> list[str]:
    raw = dict(knowledge or {})
    errs: list[str] = []
    backend_values: list[tuple[str, Any]] = [(f"{path}.numeric_backend", raw.get("numeric_backend", "rust"))]
    estimation = raw.get("estimation")
    if isinstance(estimation, dict):
        backend_values.append((f"{path}.estimation.numeric_backend", estimation.get("numeric_backend", "rust")))
        estimation_ekf = estimation.get("ekf")
        if isinstance(estimation_ekf, dict):
            backend_values.append(
                (f"{path}.estimation.ekf.numeric_backend", estimation_ekf.get("numeric_backend", "rust"))
            )
    top_level_ekf = raw.get("ekf")
    if isinstance(top_level_ekf, dict):
        backend_values.append((f"{path}.ekf.numeric_backend", top_level_ekf.get("numeric_backend", "rust")))
    for backend_path, backend_value in backend_values:
        try:
            normalize_numeric_backend(backend_value, error_message=f"{backend_path} must be 'python' or 'rust'.")
        except ValueError as exc:
            errs.append(str(exc))
    # Validate the opt-in policy before execution, including misspelled fields.
    estimation_map = dict(estimation or {}) if isinstance(estimation, dict) else {}
    ekf_map = estimation_map.get("ekf", raw.get("ekf", {})) or {}
    if isinstance(ekf_map, dict) and ekf_map.get("innovation_recovery") is not None:
        recovery_path = f"{path}.estimation.ekf.innovation_recovery"
        try:
            from sim.estimation.innovation_recovery import InnovationRecoveryConfig, validate_recovery_config

            policy_raw = ekf_map["innovation_recovery"]
            if not isinstance(policy_raw, dict):
                raise ValueError("must be a mapping/object")
            policy = InnovationRecoveryConfig(**policy_raw)
            measurement_model = str(estimation_map.get("measurement_model", "state")).lower()
            if measurement_model in {"full_state", "ric_state"}:
                measurement_model = "relative_state"
            validate_recovery_config(policy, measurement_model)
            if policy.enabled and str(estimation_map.get("type", "ekf")) == "measured_state":
                raise ValueError("requires an EKF estimator")
            if policy.enabled and any(float(v) <= 0.0 for v in ekf_map.get("meas_noise_diag", [1.0] * 6)):
                raise ValueError("requires positive measurement variances")
        except (TypeError, ValueError, OverflowError) as exc:
            errs.append(f"{recovery_path}: {exc}")
    if "sensor" in raw:
        errs.append(
            f"{path}.sensor: unsupported modeled-sensor configuration block. "
            "Use knowledge.sensor_error for measurement error assumptions and knowledge.estimation for estimator "
            "settings; optical/radar camera hardware fields are not modeled through this config path."
        )
    sensor_error = raw.get("sensor_error")
    if sensor_error is None:
        return errs
    if not isinstance(sensor_error, dict):
        errs.append(f"{path}.sensor_error: must be a mapping/object.")
        return errs
    noise = dict(sensor_error)
    allowed_sensor_error = {
        "pos_sigma_km", "vel_sigma_km_s", "quat_sigma", "omega_sigma_rad_s",
        "pos_bias_km", "vel_bias_km_s", "range_sigma_km", "range_rate_sigma_km_s",
        "angle_sigma_rad", "range_bias_km", "range_rate_bias_km_s", "az_bias_rad",
        "el_bias_rad", "seed",
    }
    for key in sorted(set(noise) - allowed_sensor_error):
        errs.append(f"{path}.sensor_error.{key}: unsupported field.")
    nonnegative_vector_fields = ("pos_sigma_km", "vel_sigma_km_s", "quat_sigma", "omega_sigma_rad_s")
    finite_vector_fields = ("pos_bias_km", "vel_bias_km_s")
    for field_name in nonnegative_vector_fields + finite_vector_fields:
        if field_name not in noise:
            continue
        value = noise[field_name]
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        field_path = f"{path}.sensor_error.{field_name}"
        if len(values) not in (1, 3):
            errs.append(f"{field_path}: must be scalar or contain exactly 3 values.")
            continue
        for index, item in enumerate(values):
            parsed = _parse_finite_float(item, f"{field_path}[{index}]", errs)
            if parsed is not None and field_name in nonnegative_vector_fields and parsed < 0.0:
                errs.append(f"{field_path}[{index}]: must be >= 0.")
    for field_name in ("range_sigma_km", "range_rate_sigma_km_s", "angle_sigma_rad"):
        if field_name not in noise:
            continue
        parsed = _parse_finite_float(noise[field_name], f"{path}.sensor_error.{field_name}", errs)
        if parsed is not None and parsed < 0.0:
            errs.append(f"{path}.sensor_error.{field_name}: must be >= 0.")
    for field_name in ("range_bias_km", "range_rate_bias_km_s", "az_bias_rad", "el_bias_rad"):
        if field_name in noise:
            _parse_finite_float(noise[field_name], f"{path}.sensor_error.{field_name}", errs)
    return errs


def _validate_object_mass_properties(specs: dict[str, Any], path: str) -> list[str]:
    raw = dict(specs or {})
    errs: list[str] = []
    if ("dry_mass_kg" in raw) or ("fuel_mass_kg" in raw):
        dry = _parse_finite_float(raw.get("dry_mass_kg", 0.0), f"{path}.dry_mass_kg", errs)
        fuel = _parse_finite_float(raw.get("fuel_mass_kg", 0.0), f"{path}.fuel_mass_kg", errs)
        if dry is not None and fuel is not None:
            if dry < 0.0:
                errs.append(f"{path}.dry_mass_kg: must be >= 0.")
            if fuel < 0.0:
                errs.append(f"{path}.fuel_mass_kg: must be >= 0.")
            if dry + fuel <= 0.0:
                errs.append(f"{path}: dry_mass_kg + fuel_mass_kg must be > 0.")
    elif "mass_kg" in raw:
        mass = _parse_finite_float(raw.get("mass_kg"), f"{path}.mass_kg", errs)
        if mass is not None and mass <= 0.0:
            errs.append(f"{path}.mass_kg: must be > 0.")
    result = validate_mass_properties(raw, path=f"{path}.mass_properties")
    errs.extend(result.errors)
    return errs


def _validate_initial_state(initial_state: dict[str, Any], path: str) -> list[str]:
    errs: list[str] = []
    raw = dict(initial_state or {})
    tle = raw.get("tle")
    if tle is not None:
        if not isinstance(tle, dict):
            errs.append(f"{path}.tle: must be a mapping/object.")
        else:
            allowed_tle_keys = {
                "line1",
                "line2",
                "lines",
                "require_checksum",
                "propagate_to_initial_epoch",
                "initialization_model",
                "initialization_frame_transform",
            }
            unknown_tle_keys = sorted(str(key) for key in tle if str(key) not in allowed_tle_keys)
            if unknown_tle_keys:
                errs.append(
                    f"{path}.tle has unsupported field(s): {', '.join(unknown_tle_keys)}. "
                    "TLE initial states normally initialize OEL numerical propagation; use object-level "
                    "propagation_method: general with general.model: sgp4 for passive OGP propagation."
                )
            init_model = tle.get("initialization_model")
            if init_model is not None:
                init_model_key = str(init_model or "").strip().lower().replace("-", "_")
                if init_model_key not in {
                    "ogp",
                    "general",
                    "general_perturbations",
                    "gp",
                    "sgp4",
                    "sdp4",
                    "ogp_sgp4",
                    "ogp_sdp4",
                    "keplerian",
                    "keplerian_mean_elements",
                    "mean_elements",
                    "two_body",
                    "legacy",
                }:
                    errs.append(
                        f"{path}.tle.initialization_model: must be ogp or keplerian_mean_elements "
                        f"(got {init_model!r})."
                    )
            init_transform = tle.get("initialization_frame_transform")
            if init_transform is not None:
                init_transform_key = str(init_transform or "").strip().lower()
                if init_transform_key not in {"teme_as_eci", "teme_to_eci_iau80"}:
                    errs.append(
                        f"{path}.tle.initialization_frame_transform: must be teme_as_eci or "
                        f"teme_to_eci_iau80 (got {init_transform!r})."
                    )
    rel_block = raw.get("relative_to_target_ric")
    if rel_block is not None:
        if not isinstance(rel_block, dict):
            errs.append(f"{path}.relative_to_target_ric: must be a mapping/object.")
        else:
            rel = dict(rel_block)
            frame = str(rel.get("frame", "rect") or "").strip().lower()
            if frame not in {"rect", "curv"}:
                errs.append(f"{path}.relative_to_target_ric.frame: must be 'rect' or 'curv'.")
            reference_frame = str(rel.get("reference_frame", rel.get("origin", "target")) or "").strip().lower()
            if reference_frame.replace("-", "_") not in {"target", "moon", "moon_ric", "lunar", "lunar_ric"}:
                errs.append(
                    f"{path}.relative_to_target_ric.reference_frame: must be 'target', 'moon', or 'moon_ric'."
                )
            if "state" not in rel:
                errs.append(f"{path}.relative_to_target_ric.state: must be a length-6 finite numeric list.")
            else:
                errs.extend(
                    _validate_numeric_sequence(
                        rel.get("state"),
                        f"{path}.relative_to_target_ric.state",
                        length=6,
                    )
                )
    if "relative_ric_rect" in raw:
        errs.extend(_validate_numeric_sequence(raw.get("relative_ric_rect"), f"{path}.relative_ric_rect", length=6))
    if "relative_ric_curv" in raw:
        errs.extend(_validate_numeric_sequence(raw.get("relative_ric_curv"), f"{path}.relative_ric_curv", length=6))
    rel_cislunar = raw.get("relative_to_target_cislunar")
    if rel_cislunar is not None:
        if not isinstance(rel_cislunar, dict):
            errs.append(f"{path}.relative_to_target_cislunar: must be a mapping/object.")
        elif "state" not in rel_cislunar:
            errs.append(f"{path}.relative_to_target_cislunar.state: must be a length-6 finite numeric list.")
        else:
            errs.extend(
                _validate_numeric_sequence(
                    rel_cislunar.get("state"),
                    f"{path}.relative_to_target_cislunar.state",
                    length=6,
                )
            )
    if "relative_cislunar" in raw:
        errs.extend(_validate_numeric_sequence(raw.get("relative_cislunar"), f"{path}.relative_cislunar", length=6))
    if "position_eci_km" in raw:
        errs.extend(_validate_numeric_sequence(raw.get("position_eci_km"), f"{path}.position_eci_km", length=3))
        try:
            pos = [float(x) for x in list(raw.get("position_eci_km"))]
        except (TypeError, ValueError):
            pos = []
        if len(pos) == 3 and all(math.isfinite(x) for x in pos):
            if math.sqrt(sum(x * x for x in pos)) <= 0.0:
                errs.append(f"{path}.position_eci_km: must be nonzero.")
    if "velocity_eci_km_s" in raw:
        errs.extend(_validate_numeric_sequence(raw.get("velocity_eci_km_s"), f"{path}.velocity_eci_km_s", length=3))
    cr3bp_rotating = raw.get("cr3bp_rotating")
    if cr3bp_rotating is not None:
        if not isinstance(cr3bp_rotating, dict):
            errs.append(f"{path}.cr3bp_rotating: must be a mapping/object.")
        else:
            errs.extend(
                _validate_numeric_sequence(
                    cr3bp_rotating.get("state_km_s", cr3bp_rotating.get("state")),
                    f"{path}.cr3bp_rotating.state_km_s",
                    length=6,
                )
            )
    return errs


def _parse_finite_float(value: Any, path: str, errs: list[str]) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errs.append(f"{path}: must be a finite number.")
        return None
    if not math.isfinite(parsed):
        errs.append(f"{path}: must be a finite number.")
        return None
    return parsed


def _validate_numeric_sequence(value: Any, path: str, *, length: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return [f"{path}: must be a length-{length} finite numeric list."]
    if len(value) != length:
        return [f"{path}: must contain exactly {length} values."]
    errs: list[str] = []
    for i, item in enumerate(value):
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            errs.append(f"{path}[{i}]: must be a finite number.")
            continue
        if not math.isfinite(numeric):
            errs.append(f"{path}[{i}]: must be a finite number.")
    return errs


def _is_cfs_sil_bridge(pointer: Any) -> bool:
    return False


def _validate_cfs_sil_timing(cfg: Any, path: str) -> list[str]:
    return []


def _validate_satellite_actuator_specs(specs: dict[str, Any], path: str) -> list[str]:
    raw = dict(specs or {})
    explicit_block = raw.get("actuators", raw.get("actuator_model"))
    if raw.get("actuator_preset") in (None, "") and explicit_block in (None, ""):
        return []
    if explicit_block is not None and not isinstance(explicit_block, dict):
        return [f"{path}.actuators: must be a mapping/object when provided."]

    try:
        resolved = resolve_actuator_specs_from_satellite_specs(raw)
    except KeyError as exc:
        choices = ", ".join(available_actuator_preset_names())
        return [f"{path}.actuator_preset: {exc.args[0]} Choices: {choices}."]
    if not isinstance(resolved, dict):
        return [f"{path}.actuators: must resolve to a mapping/object."]
    if not bool(resolved.get("enabled", True)):
        return []

    errs: list[str] = []
    allowed = {"enabled", "preset", "orbital", "attitude", "faults"}
    errs.extend(_validate_allowed_keys(resolved, allowed, f"{path}.actuators"))
    orbital = resolved.get("orbital")
    if orbital is not None:
        if not isinstance(orbital, dict):
            errs.append(f"{path}.actuators.orbital: must be a mapping/object.")
        else:
            errs.extend(_validate_orbital_actuator_block(dict(orbital), f"{path}.actuators.orbital"))
    attitude = resolved.get("attitude")
    if attitude is not None:
        if not isinstance(attitude, dict):
            errs.append(f"{path}.actuators.attitude: must be a mapping/object.")
        else:
            errs.extend(_validate_attitude_actuator_block(dict(attitude), f"{path}.actuators.attitude"))
    faults = resolved.get("faults")
    if faults is not None:
        if not isinstance(faults, dict):
            errs.append(f"{path}.actuators.faults: must be a mapping/object.")
        else:
            errs.extend(_validate_fault_block(dict(faults), f"{path}.actuators.faults"))
    return errs


def _validate_allowed_keys(raw: dict[str, Any], allowed: set[str], path: str) -> list[str]:
    return [f"{path}.{key}: unknown actuator configuration key." for key in sorted(set(raw) - allowed)]


def _validate_finite_float(
    value: Any,
    path: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    required: bool = False,
) -> list[str]:
    if value is None:
        return [f"{path}: is required."] if required else []
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return [f"{path}: must be a finite number."]
    if not math.isfinite(parsed):
        return [f"{path}: must be a finite number."]
    if min_value is not None and parsed < min_value:
        return [f"{path}: must be >= {min_value:g}."]
    if max_value is not None and parsed > max_value:
        return [f"{path}: must be <= {max_value:g}."]
    return []


def _validate_vector(
    value: Any,
    path: str,
    *,
    lengths: tuple[int, ...],
    required: bool = False,
    nonzero: bool = False,
    min_value: float | None = None,
) -> list[str]:
    if value is None:
        return [f"{path}: is required."] if required else []
    try:
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            vals = [float(value)]
        else:
            vals = [float(x) for x in list(value)]
    except (TypeError, ValueError):
        return [f"{path}: must be a finite numeric vector."]
    if len(vals) not in lengths:
        expected = " or ".join(str(n) for n in lengths)
        return [f"{path}: must contain {expected} values."]
    if not all(math.isfinite(x) for x in vals):
        return [f"{path}: must contain only finite numbers."]
    if min_value is not None and any(x < min_value for x in vals):
        return [f"{path}: values must be >= {min_value:g}."]
    if nonzero and math.sqrt(sum(x * x for x in vals)) <= 0.0:
        return [f"{path}: must be nonzero."]
    return []


def _finite_vector_values(value: Any, path: str, *, required: bool = False) -> tuple[list[float] | None, list[str]]:
    if value is None:
        return None, ([f"{path}: is required."] if required else [])
    try:
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            vals = [float(value)]
        else:
            vals = [float(x) for x in list(value)]
    except (TypeError, ValueError):
        return None, [f"{path}: must be a non-empty finite numeric vector."]
    if not vals or not all(math.isfinite(x) for x in vals):
        return None, [f"{path}: must be a non-empty finite numeric vector."]
    return vals, []


def _validate_wheel_axes(value: Any, path: str, *, wheel_count: int) -> list[str]:
    if value is None:
        return [] if wheel_count == 3 else [f"{path}: is required for {wheel_count} reaction wheels."]
    try:
        rows = [list(row) for row in list(value)]
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError
        matrix = [[float(x) for x in row] for row in rows]
    except (TypeError, ValueError):
        return [f"{path}: must be a finite matrix with shape (3,N) or (N,3)."]
    row_count = len(matrix)
    column_count = len(matrix[0])
    if not all(math.isfinite(x) for row in matrix for x in row):
        return [f"{path}: must contain only finite numbers."]
    if (row_count, column_count) == (3, wheel_count):
        axes = [[matrix[row][column] for row in range(3)] for column in range(wheel_count)]
    elif (row_count, column_count) == (wheel_count, 3):
        axes = matrix
    else:
        return [f"{path}: must have shape (3,{wheel_count}) or ({wheel_count},3)."]
    norms = [math.sqrt(sum(x * x for x in axis)) for axis in axes]
    if any(norm <= 0.0 for norm in norms):
        return [f"{path}: wheel axes must be nonzero."]
    if any(not math.isclose(norm, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9) for norm in norms):
        return [f"{path}: every wheel axis must be a unit vector."]
    return []


def _validate_orbital_actuator_block(raw: dict[str, Any], path: str) -> list[str]:
    errs: list[str] = []
    allowed = {
        "max_accel_km_s2",
        "max_thrust_n",
        "min_impulse_bit_km_s",
        "max_throttle_rate_km_s2_s",
        "isp_s",
        "thruster_direction_body",
        "thruster_position_body_m",
        "couple_to_attitude",
        "lag_tau_s",
        "rcs_cluster",
        "electric_propulsion",
        "gimbaled_thruster",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    for key in ("max_accel_km_s2", "max_thrust_n", "min_impulse_bit_km_s", "max_throttle_rate_km_s2_s", "lag_tau_s"):
        errs.extend(_validate_finite_float(raw.get(key), f"{path}.{key}", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("isp_s"), f"{path}.isp_s", min_value=0.0))
    errs.extend(_validate_vector(raw.get("thruster_direction_body"), f"{path}.thruster_direction_body", lengths=(3,), nonzero=True))
    errs.extend(_validate_vector(raw.get("thruster_position_body_m"), f"{path}.thruster_position_body_m", lengths=(3,)))
    errs.extend(_validate_rcs_cluster(raw.get("rcs_cluster"), f"{path}.rcs_cluster"))
    errs.extend(_validate_electric_propulsion(raw.get("electric_propulsion"), f"{path}.electric_propulsion"))
    errs.extend(_validate_gimbaled_thruster(raw.get("gimbaled_thruster"), f"{path}.gimbaled_thruster"))
    return errs


def _validate_rcs_cluster(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    enabled, enabled_errs = _validated_enabled(raw, path)
    if enabled_errs or not enabled:
        return enabled_errs
    errs: list[str] = []
    allowed = {
        "numeric_backend",
        "enabled",
        "allocation_mode",
        "pulse_quantum_s",
        "duty_cycle",
        "isp_s",
        "force_weight",
        "torque_weight",
        "thrusters",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    try:
        normalize_numeric_backend(raw.get("numeric_backend", "rust"), error_message=f"{path}.numeric_backend must be python or rust.")
    except ValueError as exc:
        errs.append(str(exc))
    mode = str(raw.get("allocation_mode", "force_torque")).strip()
    if mode not in {"force_torque", "force_only", "torque_only"}:
        errs.append(f"{path}.allocation_mode: must be one of force_torque, force_only, torque_only.")
    errs.extend(_validate_finite_float(raw.get("pulse_quantum_s"), f"{path}.pulse_quantum_s", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("duty_cycle"), f"{path}.duty_cycle", min_value=0.0, max_value=1.0))
    errs.extend(_validate_finite_float(raw.get("isp_s"), f"{path}.isp_s", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("force_weight"), f"{path}.force_weight", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("torque_weight"), f"{path}.torque_weight", min_value=0.0))
    try:
        force_weight = float(raw.get("force_weight", 1.0))
        torque_weight = float(raw.get("torque_weight", 1.0))
    except (TypeError, ValueError):
        pass
    else:
        if mode == "force_torque" and force_weight == 0.0 and torque_weight == 0.0:
            errs.append(f"{path}: force_weight and torque_weight cannot both be zero.")
    thrusters = raw.get("thrusters", [])
    if not isinstance(thrusters, list) or len(thrusters) == 0:
        return [*errs, f"{path}.thrusters: must be a non-empty list."]
    for idx, thruster in enumerate(thrusters):
        t_path = f"{path}.thrusters[{idx}]"
        if not isinstance(thruster, dict):
            errs.append(f"{t_path}: must be a mapping/object.")
            continue
        allowed_thruster = {
            "name",
            "position_body_m",
            "force_direction_body",
            "max_thrust_n",
            "min_impulse_bit_n_s",
            "isp_s",
        }
        errs.extend(_validate_allowed_keys(thruster, allowed_thruster, t_path))
        errs.extend(_validate_vector(thruster.get("position_body_m"), f"{t_path}.position_body_m", lengths=(3,), required=True))
        errs.extend(
            _validate_vector(
                thruster.get("force_direction_body"),
                f"{t_path}.force_direction_body",
                lengths=(3,),
                required=True,
                nonzero=True,
            )
        )
        errs.extend(_validate_finite_float(thruster.get("max_thrust_n"), f"{t_path}.max_thrust_n", min_value=0.0, required=True))
        errs.extend(_validate_finite_float(thruster.get("min_impulse_bit_n_s"), f"{t_path}.min_impulse_bit_n_s", min_value=0.0))
        errs.extend(_validate_finite_float(thruster.get("isp_s"), f"{t_path}.isp_s", min_value=0.0))
    return errs


def _validate_electric_propulsion(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    enabled, enabled_errs = _validated_enabled(raw, path)
    if enabled_errs or not enabled:
        return enabled_errs
    errs: list[str] = []
    allowed = {
        "enabled",
        "max_thrust_n",
        "isp_s",
        "duty_cycle",
        "max_power_w",
        "power_per_newton_w",
        "throttle_time_constant_s",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    errs.extend(_validate_finite_float(raw.get("max_thrust_n"), f"{path}.max_thrust_n", min_value=0.0, required=True))
    errs.extend(_validate_finite_float(raw.get("isp_s"), f"{path}.isp_s", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("duty_cycle"), f"{path}.duty_cycle", min_value=0.0, max_value=1.0))
    errs.extend(_validate_finite_float(raw.get("max_power_w"), f"{path}.max_power_w", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("power_per_newton_w"), f"{path}.power_per_newton_w", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("throttle_time_constant_s"), f"{path}.throttle_time_constant_s", min_value=0.0))
    return errs


def _validate_gimbaled_thruster(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    enabled, enabled_errs = _validated_enabled(raw, path)
    if enabled_errs or not enabled:
        return enabled_errs
    errs: list[str] = []
    allowed = {
        "enabled",
        "neutral_direction_body",
        "position_body_m",
        "max_gimbal_angle_rad",
        "max_gimbal_angle_deg",
        "max_gimbal_rate_rad_s",
        "max_gimbal_rate_deg_s",
        "response_time_constant_s",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    errs.extend(_validate_vector(raw.get("neutral_direction_body"), f"{path}.neutral_direction_body", lengths=(3,), required=True, nonzero=True))
    errs.extend(_validate_vector(raw.get("position_body_m"), f"{path}.position_body_m", lengths=(3,)))
    for key in (
        "max_gimbal_angle_rad",
        "max_gimbal_angle_deg",
        "max_gimbal_rate_rad_s",
        "max_gimbal_rate_deg_s",
        "response_time_constant_s",
    ):
        errs.extend(_validate_finite_float(raw.get(key), f"{path}.{key}", min_value=0.0))
    return errs


def _validate_attitude_actuator_block(raw: dict[str, Any], path: str) -> list[str]:
    errs: list[str] = []
    allowed = {
        "reaction_wheels",
        "magnetorquers",
        "thruster_pulse",
        "control_moment_gyros",
        "wheel_desaturation",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    errs.extend(_validate_reaction_wheels(raw.get("reaction_wheels"), f"{path}.reaction_wheels"))
    errs.extend(_validate_magnetorquers(raw.get("magnetorquers"), f"{path}.magnetorquers"))
    errs.extend(_validate_thruster_pulse(raw.get("thruster_pulse"), f"{path}.thruster_pulse"))
    errs.extend(_validate_cmg(raw.get("control_moment_gyros"), f"{path}.control_moment_gyros"))
    errs.extend(_validate_wheel_desaturation(raw.get("wheel_desaturation"), f"{path}.wheel_desaturation"))
    wheel_desaturation = raw.get("wheel_desaturation")
    reaction_wheels = raw.get("reaction_wheels")
    if (
        isinstance(wheel_desaturation, dict)
        and wheel_desaturation.get("enabled", True) is True
        and not (isinstance(reaction_wheels, dict) and reaction_wheels.get("enabled", True) is True)
    ):
        errs.append(f"{path}.wheel_desaturation: enabled wheel desaturation requires enabled reaction_wheels.")
    return errs


def _validated_enabled(raw: dict[str, Any], path: str) -> tuple[bool, list[str]]:
    value = raw.get("enabled", True)
    if not isinstance(value, bool):
        return False, [f"{path}.enabled: must be a boolean true/false value."]
    return value, []


def _validate_reaction_wheels(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    enabled, enabled_errs = _validated_enabled(raw, path)
    if enabled_errs or not enabled:
        return enabled_errs
    errs: list[str] = []
    allowed = {
        "numeric_backend",
        "enabled",
        "max_torque_nm",
        "max_momentum_nms",
        "wheel_axes_body",
        "wheel_inertia_kg_m2",
        "max_speed_rad_s",
        "torque_time_constant_s",
        "viscous_friction_nms",
        "coulomb_friction_nm",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    try:
        normalize_numeric_backend(raw.get("numeric_backend", "rust"), error_message=f"{path}.numeric_backend must be python or rust.")
    except ValueError as exc:
        errs.append(str(exc))
    max_torque, torque_errs = _finite_vector_values(raw.get("max_torque_nm"), f"{path}.max_torque_nm", required=True)
    errs.extend(torque_errs)
    wheel_count = len(max_torque) if max_torque is not None else 0
    if max_torque is not None and any(value < 0.0 for value in max_torque):
        errs.append(f"{path}.max_torque_nm: values must be >= 0.")

    max_momentum, momentum_errs = _finite_vector_values(
        raw.get("max_momentum_nms"), f"{path}.max_momentum_nms", required=True
    )
    errs.extend(momentum_errs)
    if max_momentum is not None:
        if any(value < 0.0 for value in max_momentum):
            errs.append(f"{path}.max_momentum_nms: values must be >= 0.")
        if wheel_count and len(max_momentum) not in {1, wheel_count}:
            errs.append(f"{path}.max_momentum_nms: must be scalar or length-{wheel_count}.")

    if wheel_count:
        errs.extend(_validate_wheel_axes(raw.get("wheel_axes_body"), f"{path}.wheel_axes_body", wheel_count=wheel_count))
        for name in (
            "wheel_inertia_kg_m2",
            "max_speed_rad_s",
            "viscous_friction_nms",
            "coulomb_friction_nm",
        ):
            values, value_errs = _finite_vector_values(raw.get(name), f"{path}.{name}")
            errs.extend(value_errs)
            if values is not None:
                if len(values) not in {1, wheel_count}:
                    errs.append(f"{path}.{name}: must be scalar or length-{wheel_count}.")
                if any(value < 0.0 for value in values):
                    errs.append(f"{path}.{name}: values must be >= 0.")
    errs.extend(_validate_finite_float(raw.get("torque_time_constant_s"), f"{path}.torque_time_constant_s", min_value=0.0))
    return errs


def _validate_magnetorquers(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    enabled, enabled_errs = _validated_enabled(raw, path)
    if enabled_errs or not enabled:
        return enabled_errs
    errs = _validate_allowed_keys(raw, {"enabled", "max_dipole_a_m2"}, path)
    errs.extend(_validate_vector(raw.get("max_dipole_a_m2"), f"{path}.max_dipole_a_m2", lengths=(1, 3), required=True, min_value=0.0))
    return errs


def _validate_thruster_pulse(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    enabled, enabled_errs = _validated_enabled(raw, path)
    if enabled_errs or not enabled:
        return enabled_errs
    errs = _validate_allowed_keys(raw, {"enabled", "max_torque_nm", "pulse_quantum_s"}, path)
    errs.extend(_validate_vector(raw.get("max_torque_nm"), f"{path}.max_torque_nm", lengths=(3,), required=True, min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("pulse_quantum_s"), f"{path}.pulse_quantum_s", min_value=0.0))
    return errs


def _validate_cmg(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    enabled, enabled_errs = _validated_enabled(raw, path)
    if enabled_errs or not enabled:
        return enabled_errs
    allowed = {"enabled", "max_torque_nm", "momentum_nms", "gimbal_rate_limit_rad_s", "torque_time_constant_s"}
    errs = _validate_allowed_keys(raw, allowed, path)
    errs.extend(_validate_vector(raw.get("max_torque_nm"), f"{path}.max_torque_nm", lengths=(1, 3), required=True, min_value=0.0))
    errs.extend(_validate_vector(raw.get("momentum_nms"), f"{path}.momentum_nms", lengths=(1, 3), required=True, min_value=0.0))
    errs.extend(_validate_vector(raw.get("gimbal_rate_limit_rad_s"), f"{path}.gimbal_rate_limit_rad_s", lengths=(1, 3), min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("torque_time_constant_s"), f"{path}.torque_time_constant_s", min_value=0.0))
    return errs


def _validate_wheel_desaturation(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    enabled, enabled_errs = _validated_enabled(raw, path)
    if enabled_errs or not enabled:
        return enabled_errs
    allowed = {"enabled", "momentum_fraction_threshold", "unload_gain_s_inv", "max_unload_torque_nm"}
    errs = _validate_allowed_keys(raw, allowed, path)
    errs.extend(_validate_finite_float(raw.get("momentum_fraction_threshold"), f"{path}.momentum_fraction_threshold", min_value=0.0, max_value=1.0))
    errs.extend(_validate_finite_float(raw.get("unload_gain_s_inv"), f"{path}.unload_gain_s_inv", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("max_unload_torque_nm"), f"{path}.max_unload_torque_nm", min_value=0.0))
    return errs


def _validate_fault_block(raw: dict[str, Any], path: str) -> list[str]:
    errs = _validate_allowed_keys(
        raw,
        {"stuck_off", "thrust_scale", "torque_scale", "thrust_bias_eci_km_s2", "torque_bias_body_nm"},
        path,
    )
    errs.extend(_validate_finite_float(raw.get("thrust_scale"), f"{path}.thrust_scale", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("torque_scale"), f"{path}.torque_scale", min_value=0.0))
    errs.extend(_validate_vector(raw.get("thrust_bias_eci_km_s2"), f"{path}.thrust_bias_eci_km_s2", lengths=(3,)))
    errs.extend(_validate_vector(raw.get("torque_bias_body_nm"), f"{path}.torque_bias_body_nm", lengths=(3,)))
    return errs


def _validate_rocket_guidance_modifier(pointer: Any, path: str, *, import_plugins: bool = True) -> list[str]:
    errs: list[str] = []
    if pointer is None:
        return errs
    if getattr(pointer, "kind", "python") != "python":
        return [f"{path}: only kind='python' is supported."]
    if not getattr(pointer, "module", None):
        return [f"{path}: 'module' is required for python pointers."]
    class_name = getattr(pointer, "class_name", None)
    if not class_name:
        return [f"{path}: must define 'class_name'."]
    if not import_plugins:
        return errs
    try:
        mod = importlib.import_module(str(pointer.module))
    except Exception as ex:
        return [f"{path}: failed to import module '{pointer.module}': {ex}"]
    if not hasattr(mod, class_name):
        return [f"{path}: class '{class_name}' not found in module '{pointer.module}'."]
    cls = getattr(mod, class_name)
    if not inspect.isclass(cls):
        return [f"{path}: '{class_name}' in module '{pointer.module}' is not a class."]
    if not _class_has_callable(cls, "command"):
        errs.append(f"{path}: class '{class_name}' missing required callable method 'command'.")
    return errs
