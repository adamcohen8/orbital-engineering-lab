from __future__ import annotations

import math
from typing import Any

from sim.config.scenario.models import (
    SimulatorSection,
    _plain_config_data,
)

__all__ = [
    '_plain_config_data',
    '_as_dict',
    '_UnsupportedAliasMap',
    '_reject_unsupported_aliases',
    '_reject_unknown_fields',
    '_ROOT_UNSUPPORTED_ALIASES',
    '_SIMULATOR_UNSUPPORTED_ALIASES',
    '_OUTPUTS_UNSUPPORTED_ALIASES',
    '_OUTPUT_PLOTS_UNSUPPORTED_ALIASES',
    '_OUTPUT_ANIMATIONS_UNSUPPORTED_ALIASES',
    '_parse_bool',
    '_is_bool_like_key',
    '_enforce_strict_booleans',
    '_parse_float',
    '_parse_int',
    '_parse_optional_float',
    '_validate_integer_multiple',
    '_validate_sim_timing',
    '_REENTRY_TERMINATION_LIMIT_FIELDS',
]

def _as_dict(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping/object.")
    return dict(value)


_UnsupportedAliasMap = dict[str, tuple[str, str]]


def _reject_unsupported_aliases(
    d: dict[str, Any],
    section_path: str,
    aliases: _UnsupportedAliasMap,
) -> None:
    for unsupported_key, (canonical_key, guidance) in aliases.items():
        if unsupported_key not in d:
            continue
        unsupported_path = f"{section_path}.{unsupported_key}" if section_path else unsupported_key
        canonical_path = f"{section_path}.{canonical_key}" if section_path else canonical_key
        raise ValueError(f"{unsupported_path} is unsupported. Use {canonical_path}{guidance}.")


def _reject_unknown_fields(d: dict[str, Any], section_path: str, allowed: set[str]) -> None:
    """Reject misspelled or otherwise unconsumed configuration fields.

    Extension-bearing mappings such as ``specs``, plugin ``params``, and
    ``metadata`` deliberately remain open. Public structural sections are
    closed so a successful validation means the requested fields were
    actually consumed.
    """

    unknown = sorted(str(key) for key in d if str(key) not in allowed)
    if not unknown:
        return
    label = section_path or "root"
    raise ValueError(f"{label} has unsupported field(s): {', '.join(unknown)}.")


_ROOT_UNSUPPORTED_ALIASES: _UnsupportedAliasMap = {
    "ground_station": ("ground_stations", " as a list or mapping"),
    "rocket": ("objects.rocket", " under the canonical objects map"),
    "chaser": ("objects.chaser", " under the canonical objects map"),
    "target": ("objects.target", " under the canonical objects map"),
    "monte_carlo": ("analysis", " with analysis.enabled: true and analysis.study_type: monte_carlo"),
}

_SIMULATOR_UNSUPPORTED_ALIASES: _UnsupportedAliasMap = {
    "scenario_type": (
        "objects / simulator.dynamics / analysis",
        "; scenario behavior is inferred from object kind, dynamics, bridges, and analysis settings",
    ),
}

_OUTPUTS_UNSUPPORTED_ALIASES: _UnsupportedAliasMap = {
    "plot": ("plots", " for plot configuration"),
    "animation": ("animations", " for animation configuration"),
}

_OUTPUT_PLOTS_UNSUPPORTED_ALIASES: _UnsupportedAliasMap = {
    "figure_id": ("figure_ids", " as a list"),
}

_OUTPUT_ANIMATIONS_UNSUPPORTED_ALIASES: _UnsupportedAliasMap = {
    "type": ("types", " as a list"),
}




def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean true/false value, not {value!r}.")


def _is_bool_like_key(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized in {
        "enabled",
        "strict",
        "j2",
        "j3",
        "j4",
        "drag",
        "srp",
        "third_body_moon",
        "third_body_sun",
        "parallel_enabled",
        "gravity_gradient",
        "magnetic",
        "magnetic_dipole",
        "aerodynamic_drag",
        "solar_radiation_pressure",
    }:
        return True
    return normalized.startswith(
        (
            "allow_",
            "draw_",
            "fail_on_",
            "include_",
            "use_",
            "save_",
            "display_",
            "print_",
            "require_",
            "show_",
            "strict_",
        )
    ) or normalized.endswith("_enabled") or normalized in {"controller_debug", "dry_run"}


def _enforce_strict_booleans(value: Any, path: str = "root") -> None:
    # Metadata is an application-owned extension surface.  Individual
    # consumers normalize their own values, so scenario-schema heuristics must
    # not reinterpret or reject arbitrary metadata keys such as UI flags.
    if path == "root.metadata" or path.startswith("root.metadata."):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if (
                child_path == "root.simulator.execution.object_parallelism.enabled"
                and isinstance(child, str)
                and child.strip().lower() == "auto"
            ):
                continue
            planetary_force_block = child_path in {
                "root.simulator.planetary.drag", "root.simulator.planetary.srp"
            } and isinstance(child, dict)
            if _is_bool_like_key(str(key)) and not isinstance(child, bool) and not planetary_force_block:
                raise ValueError(f"{child_path} must be a boolean true/false value, not {child!r}.")
            _enforce_strict_booleans(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _enforce_strict_booleans(child, f"{path}[{idx}]")


def _parse_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number, not {value!r}.")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be a finite number.")
    return out


def _parse_int(value: Any, field_name: str) -> int:
    """Parse an integer-valued scalar without silently truncating fractions."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not {value!r}.")
    out = _parse_float(value, field_name)
    if not out.is_integer():
        raise ValueError(f"{field_name} must be an integer, not {value!r}.")
    return int(out)


def _parse_optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _parse_float(value, field_name)


def _validate_integer_multiple(
    *,
    numerator: float,
    denominator: float,
    numerator_name: str,
    denominator_name: str,
) -> None:
    ratio = numerator / denominator
    nearest = round(ratio)
    # Scale the acceptance window with representational error, rather than
    # with the magnitude of the quotient.  A relative tolerance can accept a
    # genuine half-step once sample counts become large.
    tol = min(0.25, max(1.0e-12, 8.0 * math.ulp(max(1.0, abs(ratio)))))
    if abs(ratio - nearest) > tol:
        raise ValueError(
            f"{numerator_name} must be an integer multiple of {denominator_name}; "
            f"got {numerator_name}={numerator:g}, {denominator_name}={denominator:g}."
        )


def _validate_sim_timing(out: SimulatorSection) -> None:
    if out.dt_s <= 0.0:
        raise ValueError("simulator.dt_s must be positive.")
    if out.duration_s <= 0.0:
        raise ValueError("simulator.duration_s must be positive.")
    _validate_integer_multiple(
        numerator=out.duration_s,
        denominator=out.dt_s,
        numerator_name="simulator.duration_s",
        denominator_name="simulator.dt_s",
    )

    dynamics = dict(out.dynamics or {})
    timing_fields = (
        ("simulator.dynamics.orbit.orbit_substep_s", dict(dynamics.get("orbit", {}) or {}).get("orbit_substep_s")),
        (
            "simulator.dynamics.attitude.attitude_substep_s",
            dict(dynamics.get("attitude", {}) or {}).get("attitude_substep_s"),
        ),
    )
    for field_name, raw in timing_fields:
        substep = _parse_optional_float(raw, field_name)
        if substep is None:
            continue
        if substep <= 0.0:
            raise ValueError(f"{field_name} must be positive when provided.")
        if substep > out.dt_s:
            raise ValueError(f"{field_name} must be less than or equal to simulator.dt_s.")
        _validate_integer_multiple(
            numerator=out.dt_s,
            denominator=substep,
            numerator_name="simulator.dt_s",
            denominator_name=field_name,
        )
    orbit = dict(dynamics.get("orbit", {}) or {})
    attitude = dict(dynamics.get("attitude", {}) or {})
    if bool(attitude.get("enabled", True)):
        orbit_substep = float(orbit.get("orbit_substep_s", out.dt_s) or out.dt_s)
        attitude_substep = float(attitude.get("attitude_substep_s", out.dt_s) or out.dt_s)
        if attitude_substep > orbit_substep:
            raise ValueError(
                "simulator.dynamics.attitude.attitude_substep_s must be less than or equal to "
                "simulator.dynamics.orbit.orbit_substep_s."
            )


_REENTRY_TERMINATION_LIMIT_FIELDS = (
    "min_altitude_km",
    "max_dynamic_pressure_pa",
    "max_drag_decel_m_s2",
    "max_g_load",
    "max_heat_rate_w_m2",
    "max_heat_load_j_m2",
)
