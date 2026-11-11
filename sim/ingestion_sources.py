# ruff: noqa: F401,F403,F405,I001
from sim.ingestion_normalization import *

def ingest_tle(
    *,
    object_id: str,
    line1: str,
    line2: str,
    role: str | None = None,
    source_label: str = "agent",
    require_checksum: bool = False,
    propagate_to_initial_epoch: bool = True,
    reference_jd_utc: float | None = None,
) -> MissionInputPacket:
    oid = _clean_id(object_id, "object_id")
    elements = parse_tle_lines(line1, line2, require_checksum=bool(require_checksum))
    warnings: list[str] = [
        "OEL ingestion uses TLEs to initialize an ECI state and then numerically integrates the configured force model; this is not SGP4 propagation. Use object-level propagation_method: general with general.model: sgp4 for passive SGP4 scenarios."
    ]
    if reference_jd_utc is not None:
        reference_epoch = _finite_float(reference_jd_utc, "reference_jd_utc")
        age_days = abs(reference_epoch - float(elements.epoch_jd_utc))
        if age_days > 14.0:
            warnings.append(f"TLE epoch is {age_days:.1f} days from the requested reference epoch.")
    obj = {
        "object_id": oid,
        "kind": "satellite",
        "role": str(role or oid),
        "state_type": "tle",
        "initial_state": {
            "tle": {
                "line1": str(line1).rstrip("\n"),
                "line2": str(line2).rstrip("\n"),
                "require_checksum": bool(require_checksum),
                "propagate_to_initial_epoch": bool(propagate_to_initial_epoch),
            }
        },
        "normalized_units": {"position": "km", "velocity": "km/s", "time": "UTC Julian date"},
        "derived": {
            "tle_epoch_jd_utc": float(elements.epoch_jd_utc),
            "inclination_deg": float(elements.inclination_deg),
            "eccentricity": float(elements.eccentricity),
            "mean_motion_rev_per_day": float(elements.mean_motion_rev_per_day),
        },
        "provenance": {"source_label": str(source_label or "agent"), "source_type": "tle"},
    }
    return _packet(source_type="tle", objects={oid: obj}, warnings=warnings)


def ingest_state_vector(
    *,
    object_id: str,
    position: Iterable[Any],
    velocity: Iterable[Any],
    role: str | None = None,
    source_label: str = "agent",
    frame: str = "eci",
    epoch_jd_utc: float | None = None,
    position_units: str = "km",
    velocity_units: str = "km/s",
) -> MissionInputPacket:
    oid = _clean_id(object_id, "object_id")
    frame_key = _inertial_frame_key(frame, field_name="state-vector frame")
    pos = _scaled_vector(position, length=3, scale=_position_scale(position_units), field_name="position")
    vel = _scaled_vector(velocity, length=3, scale=_velocity_scale(velocity_units), field_name="velocity")
    initial_state: dict[str, Any] = {"position_eci_km": pos, "velocity_eci_km_s": vel}
    if epoch_jd_utc is not None:
        initial_state["epoch_jd_utc"] = _finite_float(epoch_jd_utc, "epoch_jd_utc")
    obj = {
        "object_id": oid,
        "kind": "satellite",
        "role": str(role or oid),
        "state_type": "state_vector",
        "frame": _frame_label(frame_key),
        "initial_state": initial_state,
        "normalized_units": {"position": "km", "velocity": "km/s"},
        "provenance": {"source_label": str(source_label or "agent"), "source_type": "state_vector"},
    }
    warnings = ["GCRF input is treated as ECI for OEL scenario initialization."] if frame_key == "gcrf" else []
    return _packet(source_type="state_vector", objects={oid: obj}, warnings=warnings)


def ingest_coes(
    *,
    object_id: str,
    coes: Mapping[str, Any],
    role: str | None = None,
    source_label: str = "agent",
    epoch_jd_utc: float | None = None,
) -> MissionInputPacket:
    oid = _clean_id(object_id, "object_id")
    normalized = _normalize_coes(coes)
    initial_state: dict[str, Any] = {"coes": normalized}
    if epoch_jd_utc is not None:
        initial_state["epoch_jd_utc"] = _finite_float(epoch_jd_utc, "epoch_jd_utc")
    obj = {
        "object_id": oid,
        "kind": "satellite",
        "role": str(role or oid),
        "state_type": "coes",
        "frame": "classical_orbital_elements",
        "initial_state": initial_state,
        "normalized_units": {"a": "km", "angles": "deg"},
        "provenance": {"source_label": str(source_label or "agent"), "source_type": "coes"},
    }
    return _packet(source_type="coes", objects={oid: obj})


def ingest_relative_ric_state(
    *,
    object_id: str,
    reference_id: str = "target",
    state: Iterable[Any],
    frame: str = "rect",
    role: str | None = "chaser",
    source_label: str = "agent",
    position_units: str = "km",
    velocity_units: str = "km/s",
) -> MissionInputPacket:
    oid = _clean_id(object_id, "object_id")
    ref = _clean_id(reference_id, "reference_id")
    frame_value = _ric_frame(frame)
    values = list(state)
    if len(values) != 6:
        raise ValueError("relative RIC state must contain exactly six numeric values.")
    pos_scale = _position_scale(position_units)
    vel_scale = _velocity_scale(velocity_units)
    normalized = _scaled_vector(values[:3], length=3, scale=pos_scale, field_name="relative position")
    normalized.extend(_scaled_vector(values[3:], length=3, scale=vel_scale, field_name="relative velocity"))
    initial_state = {
        "relative_to_target_ric": {
            "frame": frame_value,
            "state": normalized,
        },
        "relative_to": ref,
    }
    obj = {
        "object_id": oid,
        "kind": "satellite",
        "role": str(role or oid),
        "state_type": "relative_ric",
        "reference_id": ref,
        "frame": f"RIC {frame_value}",
        "initial_state": initial_state,
        "normalized_units": {"position": "km", "velocity": "km/s"},
        "provenance": {"source_label": str(source_label or "agent"), "source_type": "relative_ric"},
    }
    return _packet(source_type="relative_ric", objects={oid: obj})


def ingest_ephemeris_samples(
    *,
    object_id: str,
    samples: Sequence[Mapping[str, Any]],
    role: str | None = None,
    source_label: str = "agent",
    source_type: str = "structured_ephemeris",
    frame: str = "eci",
    position_units: str = "km",
    velocity_units: str = "km/s",
    time_units: str = "s",
    source_metadata: Mapping[str, Any] | None = None,
) -> MissionInputPacket:
    oid = _clean_id(object_id, "object_id")
    frame_key = _inertial_frame_key(frame, field_name="ephemeris frame")
    normalized = _normalize_ephemeris_samples(
        samples,
        position_units=position_units,
        velocity_units=velocity_units,
        time_units=time_units,
        context=f"ephemeris samples for {oid}",
        source_metadata=source_metadata,
    )
    obj = _ephemeris_object_from_samples(
        object_id=oid,
        samples=normalized,
        role=role,
        source_label=source_label,
        source_type=source_type,
        frame_key=frame_key,
        source_metadata=source_metadata,
    )
    warnings = [
        "Structured ephemeris ingestion uses the first sample as the OEL initial state; it does not replay, interpolate, or fit the full ephemeris history."
    ]
    if frame_key == "gcrf":
        warnings.append("GCRF input is treated as ECI for OEL scenario initialization.")
    return _packet(source_type=source_type, objects={oid: obj}, warnings=warnings)


def ingest_ephemeris_object_set(
    *,
    objects: Mapping[str, Mapping[str, Any]],
    source_label: str = "agent",
    source_type: str = "structured_ephemeris_set",
    frame: str = "eci",
    position_units: str = "km",
    velocity_units: str = "km/s",
    time_units: str = "s",
    source_metadata: Mapping[str, Any] | None = None,
) -> MissionInputPacket:
    out: dict[str, Any] = {}
    for raw_oid, raw_obj in dict(objects or {}).items():
        oid = _clean_id(str(raw_oid), "object_id")
        item = dict(raw_obj or {})
        item_source_metadata = {
            **dict(source_metadata or {}),
            **dict(item.get("source_metadata", {}) or {}),
        }
        samples = _normalize_ephemeris_samples(
            list(item.get("samples", []) or []),
            position_units=str(item.get("position_units") or position_units),
            velocity_units=str(item.get("velocity_units") or velocity_units),
            time_units=str(item.get("time_units") or time_units),
            context=f"ephemeris samples for {oid}",
            source_metadata=item_source_metadata,
        )
        out[oid] = _ephemeris_object_from_samples(
            object_id=oid,
            samples=samples,
            role=str(item.get("role") or oid),
            source_label=source_label,
            source_type=source_type,
            frame_key=_inertial_frame_key(str(item.get("frame") or frame), field_name=f"{oid} frame"),
            source_metadata=item_source_metadata,
        )
    if not out:
        raise ValueError("ephemeris object set must contain at least one object.")
    warnings = [
        "Structured multi-object ephemeris ingestion uses the first sample for each object as the OEL initial state; it does not replay, interpolate, or fit full ephemeris histories."
    ]
    if any(str(obj.get("frame", "")).startswith("GCRF") for obj in out.values()):
        warnings.append("GCRF input is treated as ECI for OEL scenario initialization.")
    return _packet(source_type=source_type, objects=out, warnings=warnings)


def ingest_satellite_card(
    card: Mapping[str, Any],
    *,
    object_id: str | None = None,
    role: str | None = None,
    source_label: str = "agent",
) -> MissionInputPacket:
    raw = dict(card or {})
    oid = _clean_id(str(object_id or raw.get("object_id") or ""), "object_id")
    card_role = str(role or raw.get("role") or oid)
    specs, capabilities, warnings = _normalize_satellite_card(raw)
    obj = {
        "object_id": oid,
        "kind": "satellite",
        "role": card_role,
        "profile_type": "satellite_card",
        "specs": specs,
        "capabilities": capabilities,
        "satellite_profile": {
            "name": str(raw.get("name", "") or ""),
            "description": str(raw.get("description", "") or ""),
            "sections": {
                key: deepcopy(raw.get(key))
                for key in ("bus", "geometry", "propulsion", "attitude", "sensors", "comms", "constraints", "presets")
                if isinstance(raw.get(key), Mapping)
            },
        },
        "provenance": {"source_label": str(source_label or "agent"), "source_type": "satellite_card"},
    }
    return _packet(source_type="satellite_card", objects={oid: obj}, warnings=warnings)


def merge_packets(*packets: MissionInputPacket | Mapping[str, Any]) -> MissionInputPacket:
    if not packets:
        raise ValueError("provide at least one mission input packet to merge.")
    merged_objects: dict[str, Any] = {}
    source_types: list[str] = []
    warnings: list[str] = []
    for item in packets:
        packet = item if isinstance(item, MissionInputPacket) else packet_from_dict(item)
        data = packet.to_dict()
        source = dict(data.get("source", {}) or {})
        source_type = str(source.get("type", "") or "")
        if source_type:
            source_types.append(source_type)
        warnings.extend(str(w) for w in list(data.get("warnings", []) or []))
        for oid, obj in packet.objects.items():
            key = str(oid)
            if key in merged_objects:
                merged_objects[key] = _merge_object_entries(merged_objects[key], dict(obj or {}), object_id=key)
            else:
                merged_objects[key] = deepcopy(obj)
    return _packet(
        source_type="merged",
        objects=merged_objects,
        warnings=_unique(warnings),
        extra_source={"inputs": source_types},
    )


def inspect_packet(packet: MissionInputPacket | Mapping[str, Any]) -> dict[str, Any]:
    packet_obj = packet if isinstance(packet, MissionInputPacket) else packet_from_dict(packet)
    objects = packet_obj.objects
    object_rows: list[dict[str, Any]] = []
    for oid, obj in objects.items():
        item = dict(obj or {})
        object_rows.append(
            {
                "object_id": str(oid),
                "role": str(item.get("role", "") or ""),
                "state_type": str(item.get("state_type", "") or ""),
                "frame": str(item.get("frame", "") or ""),
                "reference_id": str(item.get("reference_id", "") or ""),
                "has_initial_state": bool(item.get("initial_state")),
                "has_satellite_profile": bool(item.get("profile_type") == "satellite_card" or item.get("specs")),
                "profile_type": str(item.get("profile_type", "") or ""),
                "spec_keys": sorted(str(key) for key in dict(item.get("specs", {}) or {}).keys()),
                "source_type": str(dict(item.get("provenance", {}) or {}).get("source_type", "") or ""),
            }
        )
    propagation_errors = _propagation_readiness_errors(objects)
    rpo_errors = _rpo_readiness_errors(objects)
    return {
        "packet_version": int(packet_obj.to_dict().get("packet_version", 0) or 0),
        "source": dict(packet_obj.to_dict().get("source", {}) or {}),
        "object_count": len(objects),
        "objects": object_rows,
        "warnings": packet_obj.warnings,
        "buildable_scenarios": {
            "basic_propagation": {"ready": not propagation_errors, "errors": propagation_errors},
            "basic_rpo": {
                "ready": not rpo_errors,
                "errors": rpo_errors,
                "target_id": "target",
                "chaser_id": "chaser",
            },
        },
        "next_steps": [
            "Build scenario YAML only from a ready scenario type.",
            "Run run_simulation.py --validate-only before execution.",
            "Preserve packet warnings in the analysis summary.",
        ],
    }


def render_ingestion_summary(packet: MissionInputPacket | Mapping[str, Any]) -> str:
    summary = inspect_packet(packet)
    source = dict(summary.get("source", {}) or {})
    lines = [
        "# OEL Ingestion Summary",
        "",
        f"- Packet version: {summary.get('packet_version')}",
        f"- Source type: {source.get('type', '')}",
        f"- Object count: {summary.get('object_count')}",
        "",
        "## Objects",
        "",
        "| Object | Role | State | Frame | Profile | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for obj in list(summary.get("objects", []) or []):
        row = dict(obj or {})
        profile = "yes" if row.get("has_satellite_profile") else "no"
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    row.get("object_id", ""),
                    row.get("role", ""),
                    row.get("state_type", ""),
                    row.get("frame", ""),
                    profile,
                    row.get("source_type", ""),
                )
            )
            + " |"
        )
    lines.extend(["", "## Build Readiness", ""])
    for name, status in dict(summary.get("buildable_scenarios", {}) or {}).items():
        row = dict(status or {})
        state = "ready" if row.get("ready") else "not ready"
        lines.append(f"- {name}: {state}")
        for error in list(row.get("errors", []) or []):
            lines.append(f"  - {error}")
    warnings = list(summary.get("warnings", []) or [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for item in warnings:
            lines.append(f"- {item}")
    next_steps = list(summary.get("next_steps", []) or [])
    if next_steps:
        lines.extend(["", "## Next Steps", ""])
        for item in next_steps:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"
