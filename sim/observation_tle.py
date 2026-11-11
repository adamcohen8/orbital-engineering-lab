# ruff: noqa: F401,F403,F405,I001
from sim.observation_normalization import *
from sim.observation_ingestion import ingest_observations

def sample_sgp4_observations_from_tle(
    *,
    object_id: str,
    line1: str,
    line2: str,
    duration_s: float,
    dt_s: float,
    mass_kg: float = 1.0,
    start_jd_utc: float | None = None,
    source_label: str = "sgp4_tle",
    position_sigma_km: float | None = None,
    velocity_sigma_km_s: float | None = None,
    include_velocity: bool = True,
    require_checksum: bool = False,
    output_frame: str = "eci",
    frame_transform: str | None = None,
) -> ObservationPacket:
    """Sample an OEL SGP4 ephemeris as a normal structured observation packet."""

    if str(output_frame or "eci").strip().lower() != "eci":
        raise ValueError(
            "sample_sgp4_observations_from_tle currently writes ECI observation packets; use output_frame='eci'."
        )

    duration = float(duration_s)
    dt = float(dt_s)
    if duration <= 0.0:
        raise ValueError("duration_s must be positive.")
    if dt <= 0.0:
        raise ValueError("dt_s must be positive.")
    if float(mass_kg) <= 0.0:
        raise ValueError("mass_kg must be positive.")

    from sim.dynamics.orbit.sgp4 import SGP4EphemerisProvider

    tle_block = {"line1": str(line1), "line2": str(line2), "require_checksum": bool(require_checksum)}
    provider = SGP4EphemerisProvider.from_tle_block(
        tle_block,
        mass_kg=float(mass_kg),
        start_jd_utc=start_jd_utc,
        duration_s=duration,
        output_frame=output_frame,
        frame_transform=frame_transform,
    )
    metadata = provider.metadata()
    times = [float(t) for t in np.arange(0.0, duration + 1.0e-12, dt, dtype=float) if float(t) <= duration + 1.0e-9]
    if not times or abs(float(times[-1]) - duration) > max(1.0e-9, 1.0e-9 * duration):
        times.append(duration)
    rows: list[dict[str, Any]] = []
    for t_s in times:
        state = provider.canonical_state_at(float(t_s))
        row: dict[str, Any] = {
            "time_s": float(t_s),
            "jd_utc": float(provider.start_jd_utc) + float(t_s) / _JD_TO_S,
            "position_eci_km": np.array(state.position_eci_km, dtype=float).tolist(),
        }
        if position_sigma_km is not None:
            row["position_sigma_km"] = float(position_sigma_km)
        if include_velocity:
            row["velocity_eci_km_s"] = np.array(state.velocity_eci_km_s, dtype=float).tolist()
            if velocity_sigma_km_s is not None:
                row["velocity_sigma_km_s"] = float(velocity_sigma_km_s)
        rows.append(row)

    packet = ingest_observations(
        object_id=object_id,
        observations=rows,
        source_label=source_label,
        frame="eci",
        default_position_sigma_km=position_sigma_km,
        default_velocity_sigma_km_s=velocity_sigma_km_s if include_velocity else None,
        source_metadata={
            "source_type": "sgp4_tle_sample",
            "propagation": asdict(metadata),
            "tle": {
                "line1": str(line1),
                "line2": str(line2),
                "norad_number": provider.elements.norad_number,
                "epoch_text": provider.elements.epoch_text,
                "epoch_jd_utc": float(provider.elements.epoch_jd_utc),
                "require_checksum": bool(require_checksum),
            },
            "sampling": {
                "duration_s": duration,
                "dt_s": dt,
                "include_velocity": bool(include_velocity),
                "sample_count": len(rows),
            },
        },
    )
    data = packet.to_dict()
    data["source"]["type"] = "sgp4_tle_sample"
    warning_list = list(data.get("warnings", []) or [])
    warning_list.append(
        "SGP4-sampled packets are catalog-ephemeris screening inputs; maneuver evidence depends on "
        "TLE quality, arc choice, and the OD force model."
    )
    if str(metadata.frame_transform).lower() == "teme_as_eci":
        warning_list.append("SGP4 TEME states use OEL's v1 teme_as_eci frame treatment.")
    data["warnings"] = _unique(warning_list)
    return ObservationPacket(data)
