"""Source-bound linear ITRF2020-u2024 SLR station coordinates.

SSC positions are at 2020.0, velocities in m/Julian year. Solution validity
intervals select discontinuity segments. This adapter adds no residual-fitted
coordinate correction and does not model seasonal motion, loading or PSD.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np

from sim.dynamics.orbit.epoch import datetime_to_julian_date

REFERENCE_JD = datetime_to_julian_date(datetime(2020, 1, 1, tzinfo=timezone.utc))


def _epoch(token, *, end=False):
    if token == "00:000:00000":
        return np.inf if end else -np.inf
    year, day, second = map(int, token.split(":"))
    year += 2000 if year < 50 else 1900
    if not 1 <= day <= 366 or not 0 <= second < 86400:
        raise ValueError("Invalid SSC solution epoch")
    return datetime_to_julian_date(datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1, seconds=second))


def parse_itrf2020_slr_ssc(content):
    text = content.decode("ascii")
    if "ITRF2020-u2024 STATION POSITIONS AT EPOCH 2020.0 AND VELOCITIES" not in text:
        raise ValueError("Expected IGN ITRF2020-u2024 SLR SSC epoch 2020.0")
    lines = text.splitlines()
    records = {}
    for i, line in enumerate(lines):
        fields = line.split()
        if "SLR" not in fields:
            continue
        at = fields.index("SLR")
        if at < 1:
            continue  # Header, not a station row.
        values = fields[at + 1 :]
        if len(values) not in {7, 10} or i + 1 >= len(lines):
            raise ValueError("Malformed SSC station row")
        velocity = lines[i + 1].split()
        if len(velocity) != 7 or velocity[0] != fields[0]:
            raise ValueError("SSC position and velocity DOMES identifiers disagree")
        record = {
            "domes_id": fields[0],
            "cdp_pad_id": values[0],
            "ecef_m": list(map(float, values[1:4])),
            "velocity_m_year": list(map(float, velocity[1:4])),
            "position_sigma_m": [None if "*" in v else float(v) for v in values[4:7]],
            "velocity_sigma_m_year": [None if "*" in v else float(v) for v in velocity[4:7]],
            "solution": int(values[7]) if len(values) == 10 else 1,
            "start_jd": _epoch(values[8]) if len(values) == 10 else -np.inf,
            "end_jd": _epoch(values[9], end=True) if len(values) == 10 else np.inf,
        }
        numeric = [
            *record["ecef_m"],
            *record["velocity_m_year"],
            *record["position_sigma_m"],
            *record["velocity_sigma_m_year"],
        ]
        if not all(v is None or np.isfinite(v) for v in numeric) or any(v is not None and v < 0 for v in numeric[6:]):
            raise ValueError("Invalid SSC station coordinates or formal errors")
        records.setdefault(values[0], []).append(record)
    if not records:
        raise ValueError("Empty SSC station product")
    return records


def apply_itrf2020_station_positions(rows, station_catalog, content, *, apply_srp_tie=False):
    """Return copied rows with exact-epoch linear positions and provenance.

    A missing match is an error. Do not silently drop observations, select a
    neighboring station, add an SRP eccentricity twice, or use a closed segment.
    Formal errors are retained as metadata, never used to rescale range noise.
    """
    product = parse_itrf2020_slr_ssc(content)
    digest = hashlib.sha256(content).hexdigest()
    result = copy.deepcopy(rows)
    for row in result:
        station = station_catalog[row["station_id"]]
        pad, domes = str(station["cdp_pad_id"]), station["domes_id"]
        epoch = float(row["time_jd_utc"])
        matches = [r for r in product.get(pad, []) if r["domes_id"] == domes and r["start_jd"] <= epoch < r["end_jd"]]
        if len(matches) != 1:
            raise ValueError(f"No unique current ITRF station solution for {row['station_id']}")
        record = matches[0]
        position = np.asarray(record["ecef_m"]) + np.asarray(record["velocity_m_year"]) * (
            (epoch - REFERENCE_JD) / 365.25
        )
        # CRD rows in one session may share the same metadata dictionary.
        # Detach each row before applying its epoch and instrument tie.
        metadata = copy.deepcopy(row["station_metadata"])
        row["station_metadata"] = metadata
        metadata.update(
            ecef_km=(position / 1000).tolist(),
            coordinate_frame="ITRF2020-u2024",
            coordinate_epoch=epoch,
            source_url=station["itrf_source_url"],
            source_sha256=digest,
            reference_epoch_jd_utc=REFERENCE_JD,
            reference_ecef_m=record["ecef_m"],
            published_velocity_m_year=record["velocity_m_year"],
            domes_id=domes,
            cdp_pad_id=pad,
            solution=record["solution"],
            quality="linear_ITRF_position_at_observation_epoch_no_PSD_seasonal_loading",
            formal_position_sigma_m=record["position_sigma_m"],
            formal_velocity_sigma_m_year=record["velocity_sigma_m_year"],
        )
        # The linear velocity was already applied to ecef_km at this row's
        # epoch. Prevent downstream adapters from applying it a second time.
        metadata.pop("velocity_ecef_km_year", None)
        if apply_srp_tie:
            from sim.utils.geodesy import ecef_to_enu_rotation, ecef_to_geodetic_deg_km

            if metadata.get("srp_eccentricity_applied") or row.get("corrections", {}).get(
                "station_eccentricity", {}
            ).get("status") in {"applied", "model"}:
                raise ValueError("Station eccentricity would be applied twice")
            enu = np.asarray(station.get("srp_eccentricity_enu_m"), dtype=float)
            if enu.shape != (3,) or not np.all(np.isfinite(enu)):
                raise ValueError("Station requires a resolved monument-to-instrument tie")
            lat, lon, _height = ecef_to_geodetic_deg_km(position / 1000)
            offset = ecef_to_enu_rotation(lat, lon).T @ enu / 1000
            metadata["ecef_km"] = (position / 1000 + offset).tolist()
            metadata.update(
                srp_eccentricity_applied=True,
                srp_eccentricity_enu_m=enu.tolist(),
                srp_eccentricity_policy=station["srp_eccentricity_policy"],
                srp_eccentricity_source_sha256=station["station_identity_source_sha256"],
            )
            if "corrections" in row:
                row["corrections"]["station_eccentricity"] = {
                    "status": "applied",
                    "source_sha256": station["station_identity_source_sha256"],
                    "disposition": "already_in_station_ecef_coordinate",
                }
    return result


def validate_crd_station_identities(packet, station_catalog):
    """Bind the CRD site code and physical pad before using a geodetic solution."""
    for session in packet["sessions"]:
        station = session["station"]
        entry = station_catalog[station["code"]]
        if str(entry["cdp_pad_id"]) != str(station["pad_id"]):
            raise ValueError("CRD station pad disagrees with the governed station identity")
        for crd_key, catalog_key in (
            ("system_number", "srp_system_number"),
            ("occupation_number", "srp_occupation_number"),
        ):
            if entry.get(catalog_key) is not None and int(station[crd_key]) != int(entry[catalog_key]):
                raise ValueError("CRD system/occupation disagrees with the published instrument tie")
