from __future__ import annotations

from pathlib import Path
from typing import Any

from sim.config.campaign_file_paths import is_campaign_input_file_path, is_file_location_path
from sim.config.scenario.primitives import (
    _as_dict,
)
from sim.security import ConfigPathPolicy

__all__ = [
    '_validate_config_read_paths',
    '_resolve_geometry_profile_paths',
    '_resolve_geometry_profile_path_in_specs',
]

def _validate_config_read_paths(root: dict[str, Any], path_policy: ConfigPathPolicy | None) -> None:
    if path_policy is None:
        return
    input_base_dir = _config_input_base_dir(path_policy)
    simulator = _as_dict(root.get("simulator"), "simulator")
    planetary = _as_dict(simulator.get("planetary"), "simulator.planetary")
    for section_name in ("gravity", "gravity_frame", "atmosphere", "atmosphere_frame", "ephemeris"):
        section = _as_dict(planetary.get(section_name), f"simulator.planetary.{section_name}")
        for key in ("data_path", "coeff_path", "label_path"):
            if section.get(key) not in (None, ""):
                section[key] = str(path_policy.resolve_input_file(
                    str(section[key]), purpose=f"simulator.planetary.{section_name}.{key}",
                    base_dir=input_base_dir, must_exist=False,
                ))
        if section_name in planetary:
            planetary[section_name] = section
    if "planetary" in simulator:
        simulator["planetary"] = planetary
    if simulator.get("additional_gravity") is not None:
        additional = _as_dict(simulator["additional_gravity"], "simulator.additional_gravity")
        sections = [(additional.get("ephemeris"), "ephemeris")]
        for i, source in enumerate(additional.get("sources", []) or []):
            if isinstance(source, dict):
                sections.extend((source.get(name), f"sources[{i}].{name}") for name in ("gravity", "gravity_frame"))
        for section, name in sections:
            if isinstance(section, dict):
                for key in ("data_path", "coeff_path", "label_path", "eop_path"):
                    if section.get(key) not in (None, ""):
                        section[key] = str(path_policy.resolve_input_file(str(section[key]),
                            purpose=f"simulator.additional_gravity.{name}.{key}",
                            base_dir=input_base_dir, must_exist=False))
    environment = _as_dict(simulator.get("environment"), "simulator.environment")
    atmosphere_env = _as_dict(environment.get("atmosphere_env"), "simulator.environment.atmosphere_env")
    environment_path_fields = (
        "de440_coeff_path", "de440_eop_path", "density_eop_path", "drag_eop_path",
    )
    atmosphere_path_fields = (
        "density_eop_path", "drag_eop_path", "harris_priester_coeff_path", "hp_coeff_path",
        "jacchia70_sw_path", "jb2006_ap_path", "jb2006_sol_path", "jb2008_dtc_path",
        "jb2008_sol_path", "msis86_sw_path", "msis_sw_path", "nrlmsise00_sw_path",
        "spherical_harmonics_eop_path",
    )
    for key in environment_path_fields:
        if environment.get(key) not in (None, ""):
            environment[key] = str(path_policy.resolve_input_file(
                str(environment[key]),
                purpose=f"simulator.environment.{key}",
                base_dir=input_base_dir,
                must_exist=False,
            ))
    for key in atmosphere_path_fields:
        if environment.get(key) not in (None, ""):
            environment[key] = str(path_policy.resolve_input_file(
                str(environment[key]),
                purpose=f"simulator.environment.{key}",
                base_dir=input_base_dir,
                must_exist=False,
            ))
        if atmosphere_env.get(key) not in (None, ""):
            atmosphere_env[key] = str(path_policy.resolve_input_file(
                str(atmosphere_env[key]),
                purpose=f"simulator.environment.atmosphere_env.{key}",
                base_dir=input_base_dir,
                must_exist=False,
            ))
    spice_kernels = environment.get("spice_kernels", []) or []
    if not isinstance(spice_kernels, list):
        raise ValueError("simulator.environment.spice_kernels must be a list of paths.")
    for index, raw_path in enumerate(spice_kernels):
        spice_kernels[index] = str(path_policy.resolve_input_file(
            str(raw_path),
            purpose=f"simulator.environment.spice_kernels[{index}]",
            base_dir=input_base_dir,
            must_exist=False,
        ))
    frames = _as_dict(simulator.get("frames"), "simulator.frames")
    if frames.get("eop_path") not in (None, ""):
        frames["eop_path"] = str(path_policy.resolve_input_file(
            str(frames["eop_path"]),
            purpose="simulator.frames.eop_path",
            base_dir=input_base_dir,
            must_exist=False,
        ))
    dynamics = _as_dict(simulator.get("dynamics"), "simulator.dynamics")
    orbit = _as_dict(dynamics.get("orbit"), "simulator.dynamics.orbit")
    path_fields = (
        ("drag_eop_path", "simulator.dynamics.orbit.drag_eop_path"),
        ("de440_coeff_path", "simulator.dynamics.orbit.de440_coeff_path"),
        ("de440_eop_path", "simulator.dynamics.orbit.de440_eop_path"),
    )
    for key, purpose in path_fields:
        if orbit.get(key) not in (None, ""):
            orbit[key] = str(path_policy.resolve_input_file(
                str(orbit[key]),
                purpose=purpose,
                base_dir=input_base_dir,
                must_exist=False,
            ))
    ocean = _as_dict(orbit.get("ocean_tides"), "simulator.dynamics.orbit.ocean_tides")
    if ocean.get("coeff_path") not in (None, ""):
        ocean["coeff_path"] = str(path_policy.resolve_input_file(
            str(ocean["coeff_path"]), purpose="simulator.dynamics.orbit.ocean_tides.coeff_path",
            base_dir=input_base_dir, must_exist=False,
        ))
    sh = _as_dict(orbit.get("spherical_harmonics"), "simulator.dynamics.orbit.spherical_harmonics")
    sh_path_fields = (
        ("coeff_path", "simulator.dynamics.orbit.spherical_harmonics.coeff_path"),
        ("source_path", "simulator.dynamics.orbit.spherical_harmonics.source_path"),
        ("eop_path", "simulator.dynamics.orbit.spherical_harmonics.eop_path"),
    )
    for key, purpose in sh_path_fields:
        if sh.get(key) not in (None, ""):
            sh[key] = str(path_policy.resolve_input_file(
                str(sh[key]),
                purpose=purpose,
                base_dir=input_base_dir,
                must_exist=False,
            ))
    _resolve_geometry_profile_paths(root, path_policy)
    analysis = _as_dict(root.get("analysis"), "analysis")
    mc = _as_dict(analysis.get("monte_carlo"), "analysis.monte_carlo")
    for bundle_index, bundle in enumerate(mc.get("bundles", []) or []):
        if not isinstance(bundle, dict):
            continue  # The analysis parser reports the shape error.
        for option_index, option in enumerate(bundle.get("options", []) or []):
            if not isinstance(option, dict):
                continue
            values = option.get("values", {})
            if not isinstance(values, dict):
                continue  # The analysis parser reports the shape error.
            for parameter_path, raw_value in values.items():
                if not is_file_location_path(str(parameter_path)):
                    continue
                if not is_campaign_input_file_path(str(parameter_path)):
                    raise ValueError(f"Monte Carlo bundle file path is not an allowlisted input: {parameter_path}")
                if raw_value is None:
                    continue
                values[parameter_path] = str(path_policy.resolve_input_file(
                    str(raw_value),
                    purpose=f"analysis.monte_carlo.bundles[{bundle_index}].options[{option_index}].values[{parameter_path}]",
                    base_dir=input_base_dir,
                    must_exist=True,
                ))
    baseline = _as_dict(analysis.get("baseline"), "analysis.baseline")
    if baseline.get("summary_json") not in (None, ""):
        baseline["summary_json"] = str(path_policy.resolve_input_file(
            str(baseline["summary_json"]),
            purpose="analysis.baseline.summary_json",
            base_dir=input_base_dir,
            must_exist=False,
        ))
    outputs = _as_dict(root.get("outputs"), "outputs")
    monte_carlo = _as_dict(outputs.get("monte_carlo"), "outputs.monte_carlo")
    if monte_carlo.get("baseline_summary_json") not in (None, ""):
        monte_carlo["baseline_summary_json"] = str(path_policy.resolve_input_file(
            str(monte_carlo["baseline_summary_json"]),
            purpose="outputs.monte_carlo.baseline_summary_json",
            base_dir=input_base_dir,
            must_exist=False,
        ))
    if "atmosphere_env" in environment:
        environment["atmosphere_env"] = atmosphere_env
    if "environment" in simulator:
        simulator["environment"] = environment
    if "frames" in simulator:
        simulator["frames"] = frames
    if "spherical_harmonics" in orbit:
        orbit["spherical_harmonics"] = sh
    if "orbit" in dynamics:
        dynamics["orbit"] = orbit
    if "dynamics" in simulator:
        simulator["dynamics"] = dynamics
    root["simulator"] = simulator
    if "baseline" in analysis:
        analysis["baseline"] = baseline
    if "analysis" in root:
        root["analysis"] = analysis
    if "monte_carlo" in outputs:
        outputs["monte_carlo"] = monte_carlo
    if "outputs" in root:
        root["outputs"] = outputs


def _resolve_geometry_profile_paths(root: dict[str, Any], path_policy: ConfigPathPolicy) -> None:
    objects = root.get("objects")
    if isinstance(objects, dict):
        for object_id, section in objects.items():
            if isinstance(section, dict):
                _resolve_geometry_profile_path_in_specs(
                    dict(section.get("specs", {}) or {}),
                    section,
                    path_policy,
                    f"objects.{object_id}.specs",
                )


def _resolve_geometry_profile_path_in_specs(
    specs: dict[str, Any],
    section: dict[str, Any],
    path_policy: ConfigPathPolicy,
    purpose_prefix: str,
) -> None:
    if not specs:
        return
    updates: list[tuple[dict[str, Any], str, str]] = []
    for key in ("geometry_profile_path", "area_profile_path", "attitude_area_profile_path"):
        value = specs.get(key)
        if value not in (None, ""):
            updates.append((specs, key, f"{purpose_prefix}.{key}"))
    geometry = specs.get("geometry")
    if isinstance(geometry, dict):
        for key in ("profile_path", "area_profile_path", "attitude_area_profile_path"):
            value = geometry.get(key)
            if value not in (None, ""):
                updates.append((geometry, key, f"{purpose_prefix}.geometry.{key}"))
    aero = specs.get("aero")
    if isinstance(aero, dict):
        for key in ("geometry_profile_path", "area_profile_path"):
            value = aero.get(key)
            if value not in (None, ""):
                updates.append((aero, key, f"{purpose_prefix}.aero.{key}"))
    if not updates:
        return
    input_base_dir = _config_input_base_dir(path_policy)
    for target, key, purpose in updates:
        resolved = path_policy.resolve_input_file(
            str(target[key]),
            purpose=purpose,
            base_dir=input_base_dir,
            must_exist=True,
        )
        target[key] = str(resolved)
    section["specs"] = specs


def _config_input_base_dir(path_policy: ConfigPathPolicy) -> Path:
    """Preserve repo-root-relative configs while keeping external configs local."""

    config_dir = path_policy.config_dir.resolve()
    workspace_root = path_policy.workspace_root.resolve()
    try:
        config_dir.relative_to(workspace_root)
    except ValueError:
        return config_dir
    return workspace_root
