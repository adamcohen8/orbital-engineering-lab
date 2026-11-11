# ruff: noqa: E402,F401,F403,F405,I001
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.dynamics.orbit.epoch import julian_date_to_datetime
from sim.dynamics.orbit.tle import parse_tle_lines
from sim.frame_time import FrameTimeError, TimeScale, parse_epoch
from sim.scenarios import ScenarioArtifact

PACKET_VERSION = 1
_JD_TO_S = 86400.0
_EPHEMERIS_CLOCK_TOLERANCE_S = 1.0e-3

from sim.ingestion_contracts import *  # noqa: F403

def _add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--packet", required=True)
    parser.add_argument("--scenario-name", required=True)
    parser.add_argument("--scenario-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-s", required=True, type=float)
    parser.add_argument("--dt-s", required=True, type=float)
    parser.add_argument("--initial-jd-utc", type=float)
    parser.add_argument("--dynamics-model", default="two_body")
    parser.add_argument("--j2", action="store_true")
    parser.add_argument("--review-detail", default="standard")


def _packet(
    *,
    source_type: str,
    objects: Mapping[str, Any],
    warnings: Iterable[str] = (),
    extra_source: Mapping[str, Any] | None = None,
) -> MissionInputPacket:
    warning_list = _unique(str(item) for item in warnings if str(item or "").strip())
    data = {
        "packet_version": PACKET_VERSION,
        "kind": "oel.mission_input_packet",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {"type": str(source_type), **dict(extra_source or {})},
        "objects": deepcopy(dict(objects)),
        "warnings": warning_list,
        "validation": {
            "status": "ready_with_warnings" if warning_list else "ready",
            "notes": [
                "Packet has normalized units and can be converted into OEL scenario YAML.",
                "Run run_simulation.py --validate-only on generated scenario YAML before execution.",
            ],
        },
    }
    return MissionInputPacket(data)


def _scenario_root(
    *,
    scenario_name: str,
    description: str,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    initial_jd_utc: float | None,
    dynamics_model: str,
    j2: bool,
    j3: bool,
    j4: bool,
    drag: bool,
    srp: bool,
    third_body_sun: bool,
    third_body_moon: bool,
    atmosphere_model: str | None,
    spherical_harmonics: Mapping[str, Any] | None,
    review_detail: str,
    packet: MissionInputPacket,
) -> dict[str, Any]:
    name = str(scenario_name or "").strip()
    if not name:
        raise ValueError("scenario_name must be non-empty.")
    duration_value = float(duration_s)
    dt_value = float(dt_s)
    if not math.isfinite(duration_value) or not math.isfinite(dt_value) or duration_value <= 0.0 or dt_value <= 0.0:
        raise ValueError("duration_s and dt_s must be positive.")
    sh = _normalize_spherical_harmonics_block(spherical_harmonics)
    if sh and any(bool(flag) for flag in (j2, j3, j4)):
        raise ValueError("spherical_harmonics cannot be combined with j2, j3, or j4.")
    simulator: dict[str, Any] = {
        "duration_s": duration_value,
        "dt_s": dt_value,
        "dynamics": {
            "orbit": {
                "model": str(dynamics_model or "two_body"),
                "orbit_substep_s": float(dt_s),
                "j2": bool(j2),
                "j3": bool(j3),
                "j4": bool(j4),
                "drag": bool(drag),
                "srp": bool(srp),
                "third_body_sun": bool(third_body_sun),
                "third_body_moon": bool(third_body_moon),
            },
            "attitude": {"enabled": False},
            "rocket": {"enabled": False},
        },
        "environment": {"ephemeris_mode": "analytic_simple", "atmosphere_env": {}},
        "plugin_validation": {"strict": True},
        "termination": {"earth_impact_enabled": False},
    }
    if atmosphere_model:
        model = str(atmosphere_model).strip().lower()
        simulator["dynamics"]["orbit"]["atmosphere_model"] = model
        simulator["environment"]["atmosphere_model"] = model
    if sh:
        simulator["dynamics"]["orbit"]["spherical_harmonics"] = sh
    if initial_jd_utc is not None:
        simulator["initial_jd_utc"] = _finite_float(initial_jd_utc, "initial_jd_utc")
    return {
        "scenario_name": name,
        "scenario_description": description,
        "metadata": {
            "owner": "oel-agent-ingestion",
            "input_packet_validation_status": dict(packet.to_dict().get("validation", {}) or {}).get("status"),
            "input_packet_warnings": packet.warnings,
            "assumptions": [
                "Generated from normalized mission input; no external feeds were queried by this builder.",
                "Default satellite mass is inserted when source data does not include mass; two-body/J2 propagation does not use mass.",
                "Plots and animations are disabled by default for headless rapid analysis.",
            ],
        },
        "objects": {},
        "simulator": simulator,
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": True, "save_csv": False, "save_full_log": False},
            "review": {"enabled": True, "detail": str(review_detail or "standard")},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def _normalize_spherical_harmonics_block(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    raw = dict(value)
    enabled = bool(raw.get("enabled", True))
    if not enabled:
        return None
    out = {"enabled": True}
    for key, item in raw.items():
        if key == "enabled" or item in (None, ""):
            continue
        out[str(key)] = _jsonable(item)
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _scenario_object(
    obj: Mapping[str, Any],
    *,
    forced_role: str | None = None,
    trajectory_only: bool = False,
) -> dict[str, Any]:
    initial_state = dict(obj.get("initial_state", {}) or {})
    if not initial_state:
        raise ValueError(f"object {obj.get('object_id')!r} is missing an initial_state.")
    specs = dict(obj.get("specs", {}) or {})
    specs.setdefault("mass_kg", 100.0)
    result = {
        "kind": "satellite",
        "enabled": True,
        "role": str(forced_role or obj.get("role") or obj.get("object_id") or "agent"),
        "specs": specs,
        "initial_state": initial_state,
    }
    if trajectory_only:
        result["runtime_profile"] = "trajectory_only"
    else:
        result["flight_software"] = {
            "stack": "fsw.passive",
            "hardware_profile": "hardware.passive.v1",
        }
    return result


def _ephemeris_object_from_samples(
    *,
    object_id: str,
    samples: Sequence[Mapping[str, Any]],
    role: str | None,
    source_label: str,
    source_type: str,
    frame_key: str,
    source_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    oid = _clean_id(object_id, "object_id")
    if not samples:
        raise ValueError(f"ephemeris object {oid!r} must contain at least one sample.")
    first = dict(samples[0])
    last = dict(samples[-1])
    initial_state: dict[str, Any] = {
        "position_eci_km": list(first["position_eci_km"]),
        "velocity_eci_km_s": list(first["velocity_eci_km_s"]),
    }
    if first.get("jd_utc") is not None:
        initial_state["epoch_jd_utc"] = float(first["jd_utc"])
    ephemeris: dict[str, Any] = {
        "sample_count": len(samples),
        "first_time_s": first.get("time_s"),
        "last_time_s": last.get("time_s"),
        "first_jd_utc": first.get("jd_utc"),
        "last_jd_utc": last.get("jd_utc"),
    }
    if source_metadata:
        ephemeris["source_metadata"] = deepcopy(dict(source_metadata))
    return {
        "object_id": oid,
        "kind": "satellite",
        "role": str(role or oid),
        "state_type": "structured_ephemeris",
        "frame": _frame_label(frame_key),
        "initial_state": initial_state,
        "normalized_units": {"position": "km", "velocity": "km/s", "time": "s"},
        "ephemeris": ephemeris,
        "provenance": {"source_label": str(source_label or "agent"), "source_type": source_type},
    }


def _normalize_ephemeris_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    position_units: str,
    velocity_units: str,
    time_units: str,
    context: str,
    source_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not samples:
        raise ValueError(f"{context} must contain at least one sample.")
    pos_scale = _position_scale(position_units)
    vel_scale = _velocity_scale(velocity_units)
    time_scale = _time_scale(time_units)
    rows: list[dict[str, Any]] = []
    previous_time: float | None = None
    previous_jd: float | None = None
    clock_anchor_time: float | None = None
    clock_anchor_jd: float | None = None
    tai_elapsed = _uses_tai_elapsed_time(source_metadata)
    for idx, raw in enumerate(samples):
        row = dict(raw or {})
        position = row.get("position_eci_km", row.get("position_eci"))
        velocity = row.get("velocity_eci_km_s", row.get("velocity_eci"))
        if position is None or velocity is None:
            raise ValueError(f"{context} sample {idx} must include position_eci_km and velocity_eci_km_s.")
        time_s = row.get("time_s")
        jd_utc = row.get("jd_utc")
        normalized_time = None if time_s is None else _finite_float(time_s, f"{context} sample {idx} time_s") * time_scale
        normalized_jd = None if jd_utc is None else _finite_float(jd_utc, f"{context} sample {idx} jd_utc")
        if normalized_time is None and normalized_jd is None:
            raise ValueError(f"{context} sample {idx} must include time_s or jd_utc.")
        if normalized_time is not None:
            if previous_time is not None and normalized_time <= previous_time:
                raise ValueError(f"{context} time_s values must be strictly increasing.")
            previous_time = normalized_time
        if normalized_jd is not None:
            if previous_jd is not None and normalized_jd <= previous_jd:
                raise ValueError(f"{context} jd_utc values must be strictly increasing.")
            previous_jd = normalized_jd
        if normalized_time is not None and normalized_jd is not None:
            if clock_anchor_time is None:
                # The first paired sample is the explicit absolute anchor.
                # Compare elapsed differences so a nonzero relative time
                # origin remains valid instead of being mistaken for JD time.
                clock_anchor_time = normalized_time
                clock_anchor_jd = normalized_jd
            else:
                assert clock_anchor_jd is not None
                elapsed_from_jd = _ephemeris_jd_elapsed_seconds(
                    clock_anchor_jd,
                    normalized_jd,
                    tai_elapsed=tai_elapsed,
                    context=context,
                )
                elapsed_from_time = normalized_time - clock_anchor_time
                if abs(elapsed_from_time - elapsed_from_jd) > _EPHEMERIS_CLOCK_TOLERANCE_S:
                    raise ValueError(
                        f"{context} sample {idx} time_s and jd_utc are inconsistent "
                        f"(elapsed time differs by {elapsed_from_time - elapsed_from_jd:.6g} s)."
                    )
        rows.append(
            {
                "time_s": normalized_time,
                "jd_utc": normalized_jd,
                "position_eci_km": _scaled_vector(
                    position,
                    length=3,
                    scale=pos_scale,
                    field_name=f"{context} sample {idx} position",
                ),
                "velocity_eci_km_s": _scaled_vector(
                    velocity,
                    length=3,
                    scale=vel_scale,
                    field_name=f"{context} sample {idx} velocity",
                ),
            }
        )
    return rows


def _uses_tai_elapsed_time(source_metadata: Mapping[str, Any] | None) -> bool:
    """Return whether the source declares linear TAI elapsed seconds.

    UTC Julian dates are civil-clock values, while some supported interchange
    sources (notably the OEM adapter) retain elapsed TAI SI seconds.  The
    distinction matters across a positive UTC leap second; callers must state
    that basis in source metadata before the leap-aware path is used.
    """

    basis = str(dict(source_metadata or {}).get("elapsed_time_basis", "") or "").strip().lower()
    return "tai" in basis and ("elapsed" in basis or "second" in basis)


def _ephemeris_jd_elapsed_seconds(
    first_jd: float,
    current_jd: float,
    *,
    tai_elapsed: bool,
    context: str,
) -> float:
    if not tai_elapsed:
        return (float(current_jd) - float(first_jd)) * _JD_TO_S
    try:
        first_dt = julian_date_to_datetime(float(first_jd))
        current_dt = julian_date_to_datetime(float(current_jd))
        first_epoch = parse_epoch(first_dt.isoformat(timespec="microseconds").replace("+00:00", "Z"), TimeScale.UTC)
        current_epoch = parse_epoch(current_dt.isoformat(timespec="microseconds").replace("+00:00", "Z"), TimeScale.UTC)
    except (FrameTimeError, OverflowError, ValueError) as exc:
        raise ValueError(
            f"{context} TAI elapsed-time validation requires UTC Julian dates within "
            "the supported leap-second table."
        ) from exc
    return float(current_epoch.tai_seconds - first_epoch.tai_seconds)


def _clean_id(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    if not text.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"{field_name} must contain only letters, numbers, underscores, or hyphens.")
    return text


def _scaled_vector(value: Iterable[Any], *, length: int, scale: float, field_name: str) -> list[float]:
    out = [float(item) / float(scale) for item in value]
    if len(out) != int(length):
        raise ValueError(f"{field_name} must contain exactly {length} numeric values.")
    if any(not math.isfinite(item) for item in out):
        raise ValueError(f"{field_name} must contain only finite numeric values.")
    return out


def _finite_float(value: Any, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _position_scale(units: str) -> float:
    key = str(units or "km").strip().lower().replace(" ", "")
    if key in {"km", "kilometer", "kilometers"}:
        return 1.0
    if key in {"m", "meter", "meters"}:
        return 1000.0
    raise ValueError("position units must be 'km' or 'm'.")


def _velocity_scale(units: str) -> float:
    key = str(units or "km/s").strip().lower().replace(" ", "")
    if key in {"km/s", "km_s", "kmps", "kilometer/second", "kilometers/second"}:
        return 1.0
    if key in {"m/s", "m_s", "mps", "meter/second", "meters/second"}:
        return 1000.0
    raise ValueError("velocity units must be 'km/s' or 'm/s'.")


def _time_scale(units: str) -> float:
    key = str(units or "s").strip().lower()
    if key in {"s", "sec", "second", "seconds"}:
        return 1.0
    if key in {"min", "minute", "minutes"}:
        return 60.0
    if key in {"h", "hr", "hour", "hours"}:
        return 3600.0
    raise ValueError("time units must be seconds, minutes, or hours.")


def _inertial_frame_key(frame: str, *, field_name: str) -> str:
    key = str(frame or "eci").strip().lower()
    if key in {"eci"}:
        return "eci"
    if key in {"gcrf", "eme2000", "j2000", "icrf"}:
        return "gcrf"
    raise ValueError(f"{field_name} must be 'eci' or a supported inertial frame alias.")


def _frame_label(frame_key: str) -> str:
    return "ECI" if frame_key == "eci" else "GCRF treated as ECI"


def _ric_frame(frame: str) -> str:
    key = str(frame or "rect").strip().lower()
    if key in {"rect", "ric_rect", "rectangular"}:
        return "rect"
    if key in {"curv", "ric_curv", "curvilinear"}:
        return "curv"
    raise ValueError("relative RIC frame must be 'rect' or 'curv'.")


def _normalize_coes(coes: Mapping[str, Any]) -> dict[str, float]:
    required = ("a_km", "ecc", "inc_deg", "raan_deg", "argp_deg", "true_anomaly_deg")
    out: dict[str, float] = {}
    for key in required:
        if key not in coes:
            raise ValueError(f"coes.{key} is required.")
        out[key] = float(coes[key])
    if out["a_km"] <= 0.0:
        raise ValueError("coes.a_km must be positive.")
    if not (0.0 <= out["ecc"] < 1.0):
        raise ValueError("coes.ecc must be in [0, 1) for this basic ingestion path.")
    if any(not math.isfinite(value) for value in out.values()):
        raise ValueError("coes values must be finite.")
    return out


def _normalize_satellite_card(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    card = dict(raw or {})
    specs = deepcopy(dict(card.get("specs", {}) or {})) if isinstance(card.get("specs"), Mapping) else {}
    capabilities: dict[str, Any] = {}
    warnings: list[str] = [
        "Satellite cards describe simulation assumptions for OEL analysis; they are not verified truth about a real spacecraft."
    ]
    known_sections = {
        "object_id",
        "role",
        "name",
        "description",
        "specs",
        "bus",
        "geometry",
        "propulsion",
        "attitude",
        "sensors",
        "comms",
        "constraints",
        "presets",
    }
    unknown_sections = sorted(str(key) for key in card if str(key) not in known_sections)
    if unknown_sections:
        warnings.append(f"Satellite card contains unrecognized top-level sections: {', '.join(unknown_sections)}.")

    presets = dict(card.get("presets", {}) or {}) if isinstance(card.get("presets"), Mapping) else {}
    _copy_alias(presets, specs, "satellite", "preset_satellite")
    _copy_alias(presets, specs, "thruster", "thruster")
    _copy_alias(presets, specs, "actuator", "actuator_preset")
    _copy_alias(presets, specs, "actuator_preset", "actuator_preset")
    _copy_alias(presets, specs, "attitude_system", "attitude_system")

    bus = dict(card.get("bus", {}) or {}) if isinstance(card.get("bus"), Mapping) else {}
    for key in ("mass_kg", "dry_mass_kg", "fuel_mass_kg"):
        if key in bus:
            specs[key] = _nonnegative_float(bus[key], f"bus.{key}")
    if "bus_size_m" in bus:
        specs["bus_size_m"] = _nonnegative_vector(bus["bus_size_m"], length=3, field_name="bus.bus_size_m")
    if "dimensions_m" in bus and "bus_size_m" not in specs:
        specs["bus_size_m"] = _nonnegative_vector(bus["dimensions_m"], length=3, field_name="bus.dimensions_m")
    if isinstance(bus.get("mass_properties"), Mapping):
        specs["mass_properties"] = deepcopy(dict(bus["mass_properties"]))

    geometry = dict(card.get("geometry", {}) or {}) if isinstance(card.get("geometry"), Mapping) else {}
    for key in ("area_m2", "drag_area_m2", "lift_area_m2", "srp_area_m2", "solar_area_m2"):
        if key in geometry:
            specs[key] = _nonnegative_float(geometry[key], f"geometry.{key}")
    for key in ("cd", "drag_cd", "cr", "srp_cr"):
        if key in geometry:
            specs[key] = _nonnegative_float(geometry[key], f"geometry.{key}")

    propulsion = dict(card.get("propulsion", {}) or {}) if isinstance(card.get("propulsion"), Mapping) else {}
    for key in ("thruster", "actuator_preset", "thruster_mount_name"):
        if propulsion.get(key) not in (None, ""):
            specs[key] = str(propulsion[key])
    for key in ("max_thrust_n", "isp_s", "thruster_isp_s", "min_impulse_bit_n_s"):
        if key in propulsion:
            specs[key] = _nonnegative_float(propulsion[key], f"propulsion.{key}")
    if "thruster_position_body_m" in propulsion:
        specs["thruster_position_body_m"] = _finite_vector(
            propulsion["thruster_position_body_m"], length=3, field_name="propulsion.thruster_position_body_m"
        )
    if "thruster_direction_body" in propulsion:
        specs["thruster_direction_body"] = _finite_vector(
            propulsion["thruster_direction_body"], length=3, field_name="propulsion.thruster_direction_body"
        )
    for key in ("delta_v_budget_m_s", "max_accel_km_s2"):
        if key in propulsion:
            capabilities[key] = _nonnegative_float(propulsion[key], f"propulsion.{key}")

    attitude = dict(card.get("attitude", {}) or {}) if isinstance(card.get("attitude"), Mapping) else {}
    if attitude.get("attitude_system") not in (None, ""):
        specs["attitude_system"] = str(attitude["attitude_system"])
    if attitude:
        capabilities["attitude"] = deepcopy(attitude)
    for section_name in ("sensors", "comms", "constraints"):
        if isinstance(card.get(section_name), Mapping):
            capabilities[section_name] = deepcopy(dict(card[section_name]))

    _validate_known_spec_numbers(specs)
    if "mass_kg" not in specs and not ("dry_mass_kg" in specs or "fuel_mass_kg" in specs):
        warnings.append("Satellite card does not provide mass_kg or dry/fuel mass; generated scenarios will use OEL defaults.")
    if "drag_area_m2" not in specs and "area_m2" not in specs:
        warnings.append("Satellite card does not provide drag_area_m2 or area_m2; drag-enabled studies need this before use.")
    if "srp_area_m2" not in specs and "solar_area_m2" not in specs:
        warnings.append("Satellite card does not provide srp_area_m2 or solar_area_m2; SRP-enabled studies need this before use.")
    if not any(key in specs for key in ("cd", "drag_cd")):
        warnings.append("Satellite card does not provide cd/drag_cd; drag-enabled studies need an explicit coefficient.")
    if not any(key in specs for key in ("cr", "srp_cr")):
        warnings.append("Satellite card does not provide cr/srp_cr; SRP-enabled studies need an explicit coefficient.")
    if "max_thrust_n" not in specs and "thruster" in specs:
        warnings.append("Satellite card names a thruster but does not provide max_thrust_n.")
    if "thruster" not in specs and propulsion:
        warnings.append("Satellite card has a propulsion section but does not name a thruster or preset.")
    if not specs:
        warnings.append("Satellite card does not provide OEL runtime specs; it will only add profile metadata.")
    return specs, capabilities, warnings


def _copy_alias(source: Mapping[str, Any], target: dict[str, Any], source_key: str, target_key: str) -> None:
    value = source.get(source_key)
    if value not in (None, ""):
        target[target_key] = str(value)


def _validate_known_spec_numbers(specs: Mapping[str, Any]) -> None:
    nonnegative_keys = {
        "mass_kg",
        "dry_mass_kg",
        "fuel_mass_kg",
        "area_m2",
        "drag_area_m2",
        "lift_area_m2",
        "srp_area_m2",
        "solar_area_m2",
        "cd",
        "drag_cd",
        "cr",
        "srp_cr",
        "max_thrust_n",
        "isp_s",
        "thruster_isp_s",
        "min_impulse_bit_n_s",
    }
    for key in nonnegative_keys:
        if key in specs:
            _nonnegative_float(specs[key], f"specs.{key}")
    if "bus_size_m" in specs:
        _nonnegative_vector(specs["bus_size_m"], length=3, field_name="specs.bus_size_m")
    if "thruster_position_body_m" in specs:
        _finite_vector(specs["thruster_position_body_m"], length=3, field_name="specs.thruster_position_body_m")
    if "thruster_direction_body" in specs:
        direction = _finite_vector(specs["thruster_direction_body"], length=3, field_name="specs.thruster_direction_body")
        if math.sqrt(sum(float(item) ** 2 for item in direction)) <= 0.0:
            raise ValueError("specs.thruster_direction_body must be nonzero.")


def _nonnegative_float(value: Any, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field_name} must be a finite nonnegative number.")
    return parsed


def _finite_vector(value: Any, *, length: int, field_name: str) -> list[float]:
    out = [float(item) for item in value]
    if len(out) != int(length):
        raise ValueError(f"{field_name} must contain exactly {length} numeric values.")
    if any(not math.isfinite(item) for item in out):
        raise ValueError(f"{field_name} must contain only finite numeric values.")
    return out


def _nonnegative_vector(value: Any, *, length: int, field_name: str) -> list[float]:
    out = _finite_vector(value, length=length, field_name=field_name)
    if any(item < 0.0 for item in out):
        raise ValueError(f"{field_name} must contain only nonnegative values.")
    return out


def _merge_object_entries(base: Mapping[str, Any], update: Mapping[str, Any], *, object_id: str) -> dict[str, Any]:
    out = deepcopy(dict(base or {}))
    incoming = deepcopy(dict(update or {}))
    for key, value in incoming.items():
        if value in (None, {}, []):
            continue
        if key == "object_id":
            out[key] = str(value)
            continue
        if key == "initial_state":
            if out.get("initial_state") and out.get("initial_state") != value:
                raise ValueError(f"conflicting initial_state for duplicate object_id {object_id!r}.")
            out[key] = value
            continue
        if key == "specs":
            out[key] = _merge_no_conflict(dict(out.get("specs", {}) or {}), dict(value or {}), f"{object_id}.specs")
            continue
        if key == "capabilities":
            out[key] = _merge_no_conflict(
                dict(out.get("capabilities", {}) or {}), dict(value or {}), f"{object_id}.capabilities"
            )
            continue
        if key == "satellite_profile":
            out[key] = _merge_no_conflict(
                dict(out.get("satellite_profile", {}) or {}), dict(value or {}), f"{object_id}.satellite_profile"
            )
            continue
        if key == "provenance":
            provenance = dict(out.get("provenance", {}) or {})
            incoming_provenance = dict(value or {})
            sources = [provenance.get("source_type"), incoming_provenance.get("source_type")]
            labels = [provenance.get("source_label"), incoming_provenance.get("source_label")]
            out["provenance"] = {
                "source_type": "merged_object",
                "source_types": [str(item) for item in sources if item],
                "source_labels": [str(item) for item in labels if item],
            }
            continue
        if key in out and out[key] not in (None, "", {}, []) and out[key] != value:
            if key == "role" and str(out[key]) == object_id:
                out[key] = value
                continue
            if key == "state_type" and not value:
                continue
            raise ValueError(f"conflicting {key} for duplicate object_id {object_id!r}.")
        out[key] = value
    return out


def _merge_no_conflict(base: dict[str, Any], update: dict[str, Any], path: str) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if key in out and out[key] != value:
            if isinstance(out[key], Mapping) and isinstance(value, Mapping):
                out[key] = _merge_no_conflict(dict(out[key]), dict(value), f"{path}.{key}")
                continue
            raise ValueError(f"conflicting value for {path}.{key}.")
        out[key] = deepcopy(value)
    return out


def _propagation_readiness_errors(objects: Mapping[str, Any]) -> list[str]:
    if not objects:
        return ["packet contains no objects."]
    errors: list[str] = []
    for oid, obj in objects.items():
        item = dict(obj or {})
        if not item.get("initial_state"):
            errors.append(f"object {oid!r} is missing an initial_state.")
        if str(item.get("state_type", "") or "") == "relative_ric":
            reference_id = str(item.get("reference_id", "") or "")
            if reference_id not in objects:
                errors.append(f"relative RIC object {oid!r} references missing object {reference_id!r}.")
    return errors


def _rpo_readiness_errors(
    objects: Mapping[str, Any], *, target_id: str = "target", chaser_id: str = "chaser"
) -> list[str]:
    errors: list[str] = []
    if target_id not in objects:
        errors.append(f"RPO scenario requires target object {target_id!r} in the mission input packet.")
    if chaser_id not in objects:
        errors.append(f"RPO scenario requires chaser object {chaser_id!r} in the mission input packet.")
    if errors:
        return errors
    chaser = dict(objects[chaser_id] or {})
    if str(chaser.get("state_type", "") or "") != "relative_ric":
        errors.append("RPO scenario chaser input must come from ingest_relative_ric_state.")
    if str(chaser.get("reference_id", "") or "") != target_id:
        errors.append("RPO scenario chaser reference_id must match target_id.")
    if not dict(objects[target_id] or {}).get("initial_state"):
        errors.append(f"RPO target object {target_id!r} is missing an initial_state.")
    return errors


def _print_inspection(summary: Mapping[str, Any]) -> None:
    print(f"packet_version: {summary.get('packet_version')}")
    source = dict(summary.get("source", {}) or {})
    print(f"source_type: {source.get('type')}")
    print(f"object_count: {summary.get('object_count')}")
    print("objects:")
    for obj in list(summary.get("objects", []) or []):
        row = dict(obj or {})
        ref = f" reference={row.get('reference_id')}" if row.get("reference_id") else ""
        print(
            f"- {row.get('object_id')}: role={row.get('role')} state_type={row.get('state_type')}"
            f" frame={row.get('frame')}{ref}"
        )
    warnings = list(summary.get("warnings", []) or [])
    if warnings:
        print("warnings:")
        for item in warnings:
            print(f"- {item}")
    print("buildable_scenarios:")
    for name, status in dict(summary.get("buildable_scenarios", {}) or {}).items():
        row = dict(status or {})
        print(f"- {name}: {'ready' if row.get('ready') else 'not_ready'}")
        for error in list(row.get("errors", []) or []):
            print(f"  - {error}")


def _markdown_cell(value: Any) -> str:
    text = str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")
    return text.strip()


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out

__all__ = [name for name in globals() if not name.startswith("__")]
