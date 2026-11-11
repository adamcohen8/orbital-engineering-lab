# ruff: noqa: F401,F403,F405,I001
from sim.observation_normalization import *

def ingest_observations(
    *,
    object_id: str,
    observations: Sequence[Mapping[str, Any]],
    source_label: str = "agent",
    frame: str = "eci",
    position_units: str = "km",
    velocity_units: str = "km/s",
    time_units: str = "s",
    default_position_sigma_km: float | None = None,
    default_velocity_sigma_km_s: float | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> ObservationPacket:
    oid = _clean_id(object_id, "object_id")
    frame_key = _inertial_frame_key(frame)
    rows, has_any_velocity = _normalize_observations(
        observations,
        object_id=oid,
        source_frame=frame_key,
        position_units=position_units,
        velocity_units=velocity_units,
        time_units=time_units,
        default_position_sigma_km=default_position_sigma_km,
        default_velocity_sigma_km_s=default_velocity_sigma_km_s,
    )
    velocity_observation_count = sum(1 for item in rows if "velocity_eci_km_s" in item)
    has_complete_velocity = bool(rows) and velocity_observation_count == len(rows)
    warnings = [
        "Observation packets are estimation inputs, not simulation evidence.",
        "Preliminary OD and Kalman outputs from this packet use simplified estimator models unless a downstream tool states otherwise.",
    ]
    if frame_key == "gcrf":
        warnings.append("GCRF observations are treated as ECI for preliminary estimation.")
    if has_any_velocity and not has_complete_velocity:
        warnings.append(
            "Velocity observations are partial; batch fitting will use position-only measurements unless every row has velocity."
        )
    data = {
        "packet_version": OBSERVATION_PACKET_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "kind": "oel.observation_packet",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "type": "structured_observations",
            "label": str(source_label or "agent"),
            "metadata": deepcopy(dict(source_metadata or {})),
        },
        "object_id": oid,
        "frame": "ECI",
        "measurement_type": "eci_position_velocity" if has_complete_velocity else "eci_position",
        "normalized_units": {"position": "km", "velocity": "km/s", "time": "s"},
        "observation_contract": _observation_contract(rows, source_frame=frame_key),
        "observations": rows,
        "summary": _observation_summary(rows),
        "warnings": warnings,
        "validation": {
            "status": "ready_with_warnings",
            "notes": [
                "Use preliminary OD or Kalman filtering to convert observations into a mission input packet.",
                "Validate any generated scenario YAML before simulation execution.",
            ],
        },
    }
    return ObservationPacket(data)


def inspect_observation_packet(packet: ObservationPacket | Mapping[str, Any]) -> dict[str, Any]:
    packet_obj = packet if isinstance(packet, ObservationPacket) else observation_packet_from_dict(packet)
    data = packet_obj.to_dict()
    observations = packet_obj.observations
    errors = _position_observation_errors(observations)
    return {
        "packet_version": int(data.get("packet_version", 0) or 0),
        "observation_schema_version": int(data.get("observation_schema_version", 0) or 0),
        "observation_contract": deepcopy(dict(data.get("observation_contract", {}) or {})),
        "source": dict(data.get("source", {}) or {}),
        "object_id": str(data.get("object_id", "") or ""),
        "frame": str(data.get("frame", "") or ""),
        "measurement_type": str(data.get("measurement_type", "") or ""),
        "observation_count": len(observations),
        "arc_length_s": float(dict(data.get("summary", {}) or {}).get("arc_length_s", 0.0) or 0.0),
        "warnings": packet_obj.warnings,
        "estimators": {
            "batch_position_fit": {
                "ready": not errors,
                "errors": errors,
                "model": "constant_velocity_least_squares",
            },
            "kalman_position_filter": {
                "ready": not errors,
                "errors": errors,
                "model": "linear_constant_velocity_position_filter",
            },
            "dynamics_orbit_fit": {
                "ready": not errors,
                "errors": errors,
                "model": "oel_dynamics_least_squares",
            },
        },
        "next_steps": [
            "Use fit-orbit for dynamics-based OD, or fit-position / kalman-position for preliminary initial guesses.",
            "Inspect residuals/covariance before trusting the generated state.",
            "Validate generated scenario YAML before execution.",
        ],
    }
