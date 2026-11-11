# ruff: noqa: F401,F403,F405,I001
from .orbit_od_common import *

def selected_orbit_od_parameters(spec: str) -> list[str]:
    aliases = {
        "state": ["state"],
        "initial_state": ["state"],
        "cartesian_state": ["state"],
        "drag": ["drag_scale"],
        "drag_scale": ["drag_scale"],
        "cd": ["cd_scale"],
        "cd_scale": ["cd_scale"],
        "drag_coefficient": ["cd_scale"],
        "drag_coefficient_scale": ["cd_scale"],
        "srp": ["srp_scale"],
        "srp_scale": ["srp_scale"],
    }
    selected: list[str] = []
    for raw in str(spec or "state").replace(";", ",").split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown estimate token '{raw}'. Use state, drag_scale, cd_scale, srp_scale.")
        for value in aliases[key]:
            if value not in selected:
                selected.append(value)
    if not selected:
        raise ValueError("At least one estimated parameter must be selected.")
    return selected


def build_orbit_od_parameter_set(selected: Sequence[str]) -> ParameterSet:
    params: list[EstimatedParameter] = []
    if "state" in selected:
        params.extend(
            [
                EstimatedParameter("dx_m", 0.0, scale=100.0, lower=-100000.0, upper=100000.0, unit="m"),
                EstimatedParameter("dy_m", 0.0, scale=100.0, lower=-100000.0, upper=100000.0, unit="m"),
                EstimatedParameter("dz_m", 0.0, scale=100.0, lower=-100000.0, upper=100000.0, unit="m"),
                EstimatedParameter("dvx_mm_s", 0.0, scale=10.0, lower=-100000.0, upper=100000.0, unit="mm/s"),
                EstimatedParameter("dvy_mm_s", 0.0, scale=10.0, lower=-100000.0, upper=100000.0, unit="mm/s"),
                EstimatedParameter("dvz_mm_s", 0.0, scale=10.0, lower=-100000.0, upper=100000.0, unit="mm/s"),
            ]
        )
    if "drag_scale" in selected:
        params.append(
            EstimatedParameter(
                "drag_scale",
                1.0,
                scale=0.1,
                lower=0.05,
                upper=20.0,
                unit="1",
                description="Multiplier on object drag_area_m2.",
            )
        )
    if "cd_scale" in selected:
        params.append(
            EstimatedParameter(
                "cd_scale",
                1.0,
                scale=0.1,
                lower=0.05,
                upper=20.0,
                unit="1",
                description="Multiplier on object drag coefficient Cd.",
            )
        )
    if "srp_scale" in selected:
        params.append(
            EstimatedParameter(
                "srp_scale",
                1.0,
                scale=0.1,
                lower=0.05,
                upper=20.0,
                unit="1",
                description="Multiplier on object srp_area_m2/solar_area_m2.",
            )
        )
    return ParameterSet(params)

__all__ = [name for name in globals() if not name.startswith("__")]
