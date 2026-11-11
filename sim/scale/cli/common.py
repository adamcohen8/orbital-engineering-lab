from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _initial_state_from_mission_input_packet(
    path: str | Path,
    *,
    object_id: str,
) -> tuple[float, float, float, float, float, float]:
    packet_path = Path(path).expanduser().resolve()
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    objects = dict(payload.get("objects", {}) or {})
    object_payload = dict(objects.get(object_id, {}) or {})
    if not object_payload:
        raise ValueError(f"mission input packet {packet_path} does not contain object_id {object_id!r}.")
    initial_state = dict(object_payload.get("initial_state", {}) or {})
    position = initial_state.get("position_eci_km")
    velocity = initial_state.get("velocity_eci_km_s")
    if position is None or velocity is None:
        raise ValueError(f"mission input packet object {object_id!r} requires position_eci_km and velocity_eci_km_s.")
    state = [*list(position), *list(velocity)]
    if len(state) != 6:
        raise ValueError(f"mission input packet object {object_id!r} must contain a 6-element ECI state.")
    return tuple(float(item) for item in state)


def _first_tle_lines_from_file(path: str | Path) -> tuple[str, str]:
    source = Path(path).expanduser().resolve()
    lines = [line.rstrip("\n") for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.startswith("1 "):
            if idx + 1 >= len(lines) or not lines[idx + 1].startswith("2 "):
                raise ValueError(f"TLE line1 at {source}:{idx + 1} is missing line2.")
            return line, lines[idx + 1]
    raise ValueError(f"No TLE lines found in {source}.")


def _json_mapping_from_path(path: str | Path | None) -> dict[str, Any] | None:
    if path in (None, ""):
        return None
    source = Path(str(path)).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON mapping file must contain an object: {source}")
    return dict(payload)


def _spherical_harmonics_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    term_json = list(getattr(args, "spherical_harmonics_term_json", []) or [])
    raw_items = {
        "degree": getattr(args, "spherical_harmonics_degree", None),
        "order": getattr(args, "spherical_harmonics_order", None),
        "source": getattr(args, "spherical_harmonics_source", ""),
        "coeff_path": getattr(args, "spherical_harmonics_coeff_path", ""),
        "frame_model": getattr(args, "spherical_harmonics_frame_model", ""),
        "eop_path": getattr(args, "spherical_harmonics_eop_path", ""),
        "reference_radius_km": getattr(args, "spherical_harmonics_reference_radius_km", None),
        "fd_step_km": getattr(args, "spherical_harmonics_fd_step_km", None),
    }
    enabled = any(value not in (None, "") for value in raw_items.values()) or bool(term_json)
    if not enabled:
        return None
    degree = raw_items["degree"]
    if degree is None or int(degree) < 2:
        raise ValueError("spherical harmonics requires --spherical-harmonics-degree >= 2.")
    out: dict[str, Any] = {"enabled": True, "degree": int(degree)}
    if raw_items["order"] is not None:
        order = int(raw_items["order"])
        if order < 0 or order > int(degree):
            raise ValueError("spherical harmonics order must satisfy 0 <= order <= degree.")
        out["order"] = order
    else:
        out["order"] = int(degree)
    for key in ("source", "coeff_path", "frame_model", "eop_path", "reference_radius_km", "fd_step_km"):
        value = raw_items[key]
        if value not in (None, ""):
            out[key] = value
    if out.get("coeff_path") and not out.get("source"):
        out["source"] = "hpop_ggm03"
    if bool(getattr(args, "spherical_harmonics_normalized", False)):
        out["normalized"] = True
    if term_json:
        terms = []
        for raw in term_json:
            parsed = json.loads(str(raw))
            if isinstance(parsed, list):
                terms.extend(dict(item) for item in parsed)
            elif isinstance(parsed, dict):
                terms.append(parsed)
            else:
                raise ValueError("--spherical-harmonics-term-json must parse to an object or list of objects.")
        out["terms"] = terms
    return out
