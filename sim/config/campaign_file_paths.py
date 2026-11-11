"""Explicit file-input allowlist for Monte Carlo bundle alternatives."""

from __future__ import annotations

import re

_ENVIRONMENT_FIELDS = {
    "de440_coeff_path", "de440_eop_path", "density_eop_path", "drag_eop_path",
    "harris_priester_coeff_path", "hp_coeff_path", "jacchia70_sw_path",
    "jb2006_ap_path", "jb2006_sol_path", "jb2008_dtc_path", "jb2008_sol_path",
    "msis86_sw_path", "msis_sw_path", "nrlmsise00_sw_path", "spherical_harmonics_eop_path",
}
_ORBIT_FIELDS = {"drag_eop_path", "de440_coeff_path", "de440_eop_path"}
_OBJECT_FIELDS = {"geometry_profile_path", "area_profile_path", "attitude_area_profile_path"}


def is_file_location_path(path: str) -> bool:
    terminal = path.rsplit(".", 1)[-1].split("[", 1)[0]
    return terminal.endswith(("_path", "_file")) or terminal in {
        "output_dir", "summary_json", "prompt_file", "spice_kernels",
    }


def is_campaign_input_file_path(path: str) -> bool:
    if path == "simulator.frames.eop_path":
        return True
    if re.fullmatch(r"simulator\.environment\.spice_kernels\[\d+\]", path):
        return True
    if path.startswith("simulator.environment."):
        suffix = path.removeprefix("simulator.environment.").removeprefix("atmosphere_env.")
        return suffix in _ENVIRONMENT_FIELDS
    if path.startswith("simulator.dynamics.orbit."):
        suffix = path.removeprefix("simulator.dynamics.orbit.")
        return suffix in _ORBIT_FIELDS or suffix in {
            "ocean_tides.coeff_path", "spherical_harmonics.coeff_path",
            "spherical_harmonics.source_path", "spherical_harmonics.eop_path",
        }
    parts = path.split(".")
    if len(parts) >= 4 and parts[0] == "objects" and parts[2] == "specs":
        suffix = ".".join(parts[3:])
        return suffix in _OBJECT_FIELDS or suffix in {
            "geometry.profile_path", "geometry.area_profile_path", "geometry.attitude_area_profile_path",
            "aero.geometry_profile_path", "aero.area_profile_path",
        }
    return False
