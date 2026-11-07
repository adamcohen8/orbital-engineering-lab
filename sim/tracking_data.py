from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sim.frame_time import TimeScale, epoch_julian_date, parse_epoch
from sim.interchange.ccsds_tdm import TdmMessage, serialize_tdm_kvn

TRACKING_ADAPTER_SCHEMA_VERSION = 1
NORMALIZED_TRACKING_DATASET_SCHEMA = "oel.normalized_tracking_dataset.v1"
SUPPORTED_FORMATS = {"oel_ground_tracking", "ccsds_tdm_compatible_json"}
_TDM_COMPONENT_MAP = {
    "ANGLE_1": "azimuth_deg",
    "ANGLE_2": "elevation_deg",
    "RANGE": "range_km",
}
_OEL_COMPONENT_MAP = {
    **_TDM_COMPONENT_MAP,
    "RANGE_RATE": "range_rate_km_s",
    "AZIMUTH": "azimuth_deg",
    "ELEVATION": "elevation_deg",
}


def adapt_ground_tracking_packet(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a versioned tracking packet into ground-OD measurement rows.

    ``ccsds_tdm_compatible_json`` is an intentionally bounded adapter profile,
    not a complete CCSDS 503.0 implementation. It maps RANGE and AZEL
    ANGLE_1/ANGLE_2 records while preserving the source keywords.
    """

    packet = dict(payload or {})
    schema_version = int(packet.get("schema_version", 0) or 0)
    if schema_version != TRACKING_ADAPTER_SCHEMA_VERSION:
        raise ValueError(f"tracking packet schema_version must be {TRACKING_ADAPTER_SCHEMA_VERSION}.")
    format_name = str(packet.get("format", "") or "").strip().lower()
    if format_name not in SUPPORTED_FORMATS:
        raise ValueError(f"tracking packet format must be one of {sorted(SUPPORTED_FORMATS)}.")
    metadata = dict(packet.get("metadata", {}) or {})
    time_system = str(metadata.get("time_system", "UTC") or "UTC").strip().upper()
    if time_system != "UTC":
        raise ValueError("tracking adapter currently requires metadata.time_system='UTC'.")
    angle_type = str(metadata.get("angle_type", "AZEL") or "AZEL").strip().upper()
    if angle_type != "AZEL":
        raise ValueError("tracking adapter currently supports only metadata.angle_type='AZEL'.")
    angle_units = str(metadata.get("angle_units", "deg") or "deg").strip().lower()
    range_units = str(metadata.get("range_units", "km") or "km").strip().lower()
    range_rate_units = str(metadata.get("range_rate_units", "km/s") or "km/s").strip().lower()
    if angle_units not in {"deg", "degree", "degrees"}:
        raise ValueError("tracking adapter requires degree angular units.")
    if range_units not in {"km", "kilometer", "kilometers"}:
        raise ValueError("tracking adapter requires kilometer range units.")
    if range_rate_units not in {"km/s", "km_s"}:
        raise ValueError("tracking adapter requires km/s range-rate units.")

    stations = _normalize_station_catalog(packet.get("stations", {}))
    if packet.get("observations") is not None:
        rows = _adapt_observations(
            list(packet.get("observations", []) or []),
            stations=stations,
            format_name=format_name,
        )
        source_route = "observation_rows"
    else:
        rows = _adapt_observable_records(
            list(packet.get("observable_records", []) or []),
            stations=stations,
            format_name=format_name,
        )
        source_route = "observable_records_grouped_by_station_and_epoch"
    if not rows:
        raise ValueError("tracking packet did not contain any supported observations.")
    return {
        "schema_version": TRACKING_ADAPTER_SCHEMA_VERSION,
        "adapter": "oel_ground_tracking_adapter",
        "format": format_name,
        "measurement_rows": rows,
        "station_ids": sorted({str(row["station_id"]) for row in rows}),
        "measurement_count": len(rows),
        "provenance": {
            "source_route": source_route,
            "time_system": time_system,
            "angle_type": angle_type,
            "angle_units": "deg",
            "range_units": "km",
            "range_rate_units": "km/s",
            "tdm_profile": (
                "bounded_ccsds_503_0_b_2_compatible_keyword_mapping"
                if format_name == "ccsds_tdm_compatible_json"
                else "not_applicable"
            ),
            "unsupported_tdm_observables": [
                "DOPPLER_INSTANTANEOUS",
                "DOPPLER_INTEGRATED",
                "RECEIVE_FREQ",
                "TRANSMIT_FREQ",
            ],
            "light_time_contract": "not_applied",
        },
        "non_claims": [
            "The CCSDS route is a bounded JSON keyword mapping, not a complete KVN/XML TDM parser.",
            "Doppler is not converted to range rate without an explicit signal, count-interval, and sign contract.",
            "No light-time, media, transponder, or calibration correction is inferred by the adapter.",
        ],
    }


def normalize_tdm_tracking_dataset(
    message: TdmMessage,
    *,
    stations: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    measurement_semantics: str,
    angle_sigma_deg: float,
    range_sigma_km: float,
    expected_object_id: str | None = None,
) -> dict[str, Any]:
    """Normalize a bounded parsed TDM into native ground-OD measurement rows.

    The caller must explicitly attest that the supplied observables are reduced
    geometric values compatible with OEL's current instantaneous measurement
    model. Raw radiometric range is not silently treated as geometric range.
    """

    if measurement_semantics != "reduced_geometric":
        raise ValueError(
            "measurement_semantics must be 'reduced_geometric'; raw radiometric/light-time observables are unsupported."
        )
    sigma_by_component = {
        "azimuth_deg": float(angle_sigma_deg),
        "elevation_deg": float(angle_sigma_deg),
        "range_km": float(range_sigma_km),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in sigma_by_component.values()):
        raise ValueError("TDM angular and range sigmas must be positive and finite.")
    station_catalog = _normalize_station_catalog(stations)
    object_ids = sorted({segment.metadata.object_id for segment in message.segments})
    if len(object_ids) != 1:
        raise ValueError("The bounded TDM OD workflow requires exactly one PARTICIPANT_2 object identifier.")
    object_id = object_ids[0]
    if expected_object_id is not None and object_id != str(expected_object_id):
        raise ValueError(
            f"TDM PARTICIPANT_2 {object_id!r} does not match expected object_id {expected_object_id!r}."
        )

    component_by_keyword = {
        "ANGLE_1": "azimuth_deg",
        "ANGLE_2": "elevation_deg",
        "RANGE": "range_km",
    }
    rows: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(message.segments):
        station_id = segment.metadata.station_id
        station = _station(station_catalog, station_id)
        grouped: dict[float, dict[str, Any]] = {}
        for source_index, observation in enumerate(segment.observations):
            epoch_tai_seconds = (
                float(observation.epoch_tai_seconds)
                if observation.epoch_tai_seconds is not None
                else _epoch_tai_seconds({"epoch_utc": observation.epoch_utc})
            )
            epoch_jd_utc = _epoch_jd_utc({"epoch_utc": observation.epoch_utc})
            _validate_epoch_consistency(
                {
                    "epoch_utc": observation.epoch_utc,
                    "time_tai_seconds": observation.epoch_tai_seconds,
                },
                epoch_jd_utc=epoch_jd_utc,
                epoch_tai_seconds=epoch_tai_seconds,
            )
            group = grouped.setdefault(
                epoch_tai_seconds,
                {
                    "station_id": station_id,
                    "station_metadata": dict(station),
                    "time_jd_utc": epoch_jd_utc,
                    "time_tai_seconds": epoch_tai_seconds,
                    "epoch_utc": observation.epoch_utc,
                    "components": [],
                    "vector": [],
                    "sigma": [],
                    "source_keywords": [],
                    "source_indices": [],
                    "source_segment_index": segment_index,
                },
            )
            component = component_by_keyword[observation.keyword]
            if component in group["components"]:
                raise ValueError(
                    f"Duplicate {observation.keyword} for station {station_id!r} at {observation.epoch_utc}."
                )
            group["components"].append(component)
            group["vector"].append(float(observation.value))
            group["sigma"].append(sigma_by_component[component])
            group["source_keywords"].append(observation.keyword)
            group["source_indices"].append(source_index)
        for _epoch, row in sorted(grouped.items()):
            row["measurement_id"] = f"tdm:{segment_index:04d}:{len(rows):08d}"
            row["measurement_type"] = "+".join(row["components"])
            row["arc_id"] = str(segment.metadata.values.get("TRACK_ID", f"tdm-segment-{segment_index}"))
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("TDM OD requires at least two normalized measurement epochs.")
    rows.sort(
        key=lambda row: (
            float(row["time_tai_seconds"]),
            str(row["station_id"]),
            str(row["measurement_id"]),
        )
    )
    canonical_tdm = serialize_tdm_kvn(message).encode("utf-8")
    canonical_sha256 = hashlib.sha256(canonical_tdm).hexdigest()
    normalized_identity = {
        "schema_version": NORMALIZED_TRACKING_DATASET_SCHEMA,
        "source_tdm_sha256": canonical_sha256,
        "canonical_tdm_sha256": canonical_sha256,
        "object_id": object_id,
        "measurement_semantics": measurement_semantics,
        "station_catalog": station_catalog,
        "measurement_rows": rows,
    }
    dataset_sha256 = hashlib.sha256(
        json.dumps(normalized_identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {
        **normalized_identity,
        "dataset_sha256": dataset_sha256,
        "station_ids": sorted({str(row["station_id"]) for row in rows}),
        "measurement_epoch_count": len(rows),
        "observable_record_count": sum(len(row["components"]) for row in rows),
        "first_epoch_jd_utc": float(rows[0]["time_jd_utc"]),
        "last_epoch_jd_utc": float(rows[-1]["time_jd_utc"]),
        "first_epoch_tai_seconds": float(rows[0]["time_tai_seconds"]),
        "last_epoch_tai_seconds": float(rows[-1]["time_tai_seconds"]),
        "provenance": {
            "source_format": "ccsds_tdm_2_0_kvn",
            "source_profile": "oel.ccsds-tdm-kvn.v0.1",
            "time_system": "UTC",
            "angle_type": "AZEL",
            "angle_units": "deg",
            "range_units": "km",
            "measurement_semantics": "analyst_attested_reduced_geometric",
            "light_time_contract": "not_applied_input_must_already_be_reduced",
            "raw_source_tdm_sha256": message.source_sha256,
            "epoch_identity": "utc_parsed_to_tai_seconds_before_scalar_julian_date_conversion",
        },
        "non_claims": [
            "Normalization does not apply light-time, media, transponder, clock, station, or sensor calibration corrections.",
            "The reduced_geometric declaration is an analyst input assertion, not a calibration result produced by OEL.",
            "Doppler, frequency, phase, ambiguous range, and multi-way range are outside this dataset profile.",
        ],
    }


def _normalize_station_catalog(value: Any) -> dict[str, dict[str, float | str]]:
    if isinstance(value, Mapping):
        raw_items = [(str(key), dict(item or {})) for key, item in value.items()]
    else:
        raw_items = [(str(dict(item or {}).get("id", "")), dict(item or {})) for item in list(value or [])]
    stations: dict[str, dict[str, float | str]] = {}
    for station_id, raw in raw_items:
        station_key = station_id.strip()
        if not station_key:
            raise ValueError("tracking station IDs must be non-empty.")
        if station_key in stations:
            raise ValueError(f"tracking station {station_key!r} is declared more than once.")
        if not {"lat_deg", "lon_deg"}.issubset(raw):
            raise ValueError(f"tracking station {station_key!r} requires lat_deg and lon_deg.")
        coordinates = np.array([raw["lat_deg"], raw["lon_deg"], raw.get("alt_km", 0.0)], dtype=float)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"tracking station {station_key!r} coordinates must be finite.")
        if not -90.0 <= coordinates[0] <= 90.0:
            raise ValueError(f"tracking station {station_key!r} latitude must lie within [-90, 90] degrees.")
        if not -180.0 <= coordinates[1] <= 180.0:
            raise ValueError(f"tracking station {station_key!r} longitude must lie within [-180, 180] degrees.")
        stations[station_key] = {
            "id": station_key,
            "lat_deg": float(coordinates[0]),
            "lon_deg": float(coordinates[1]),
            "alt_km": float(coordinates[2]),
        }
    return stations


def _adapt_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    stations: Mapping[str, Mapping[str, Any]],
    format_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw_item in enumerate(observations):
        item = dict(raw_item or {})
        station_id = str(item.get("station_id", "") or "").strip()
        station = _station(stations, station_id)
        measurements = dict(item.get("measurements", {}) or {})
        if not measurements:
            raise ValueError(f"tracking observation {index} requires measurements.")
        components = [str(name) for name in measurements]
        allowed = {"azimuth_deg", "elevation_deg", "range_km"}
        if format_name == "oel_ground_tracking":
            allowed.add("range_rate_km_s")
        unknown = sorted(set(components) - allowed)
        if unknown:
            raise ValueError(f"tracking observation {index} has unsupported measurements: {unknown}.")
        row = _base_row(item, station_id=station_id, station=station, default_id=f"tracking:{index}")
        row.update(
            {
                "components": components,
                "vector": [
                    _measurement_value(measurements[name], component=name, context=f"tracking observation {index}")
                    for name in components
                ],
            }
        )
        _copy_uncertainty(item, row)
        rows.append(row)
    return rows


def _adapt_observable_records(
    records: Sequence[Mapping[str, Any]],
    *,
    stations: Mapping[str, Mapping[str, Any]],
    format_name: str,
) -> list[dict[str, Any]]:
    component_map = _TDM_COMPONENT_MAP if format_name == "ccsds_tdm_compatible_json" else _OEL_COMPONENT_MAP
    grouped: dict[tuple[str, str, float], dict[str, Any]] = {}
    for index, raw_record in enumerate(records):
        record = dict(raw_record or {})
        keyword = str(record.get("keyword", "") or "").strip().upper()
        if keyword.startswith("DOPPLER") or keyword in {"RECEIVE_FREQ", "TRANSMIT_FREQ"}:
            raise ValueError(
                f"tracking observable {keyword!r} requires an explicit signal/count/sign contract and is not adapted."
            )
        if keyword not in component_map:
            raise ValueError(f"unsupported tracking observable keyword {keyword!r}.")
        station_id = str(record.get("station_id", "") or "").strip()
        station = _station(stations, station_id)
        epoch_jd_utc = _epoch_jd_utc(record)
        epoch_tai_seconds = _epoch_tai_seconds(record) if _has_exact_epoch(record) else None
        _validate_epoch_consistency(record, epoch_jd_utc=epoch_jd_utc, epoch_tai_seconds=epoch_tai_seconds)
        # Julian dates are useful for interoperability but do not retain enough
        # binary64 precision to identify sub-millisecond records.  Group by the
        # exact parsed epoch whenever the source supplies UTC or TAI identity;
        # retain the scalar JD only as the explicitly approximate fallback.
        key = (
            station_id,
            "tai_seconds" if epoch_tai_seconds is not None else "julian_date",
            epoch_tai_seconds if epoch_tai_seconds is not None else epoch_jd_utc,
        )
        group = grouped.setdefault(
            key,
            {
                "station_id": station_id,
                "station_metadata": dict(station),
                "time_jd_utc": epoch_jd_utc,
                **({"time_tai_seconds": epoch_tai_seconds} if epoch_tai_seconds is not None else {}),
                "components": [],
                "vector": [],
                "source_keywords": [],
                "source_indices": [],
            },
        )
        component = component_map[keyword]
        if component in group["components"]:
            epoch_label = (
                f"TAI seconds {epoch_tai_seconds}"
                if epoch_tai_seconds is not None
                else f"JD {epoch_jd_utc}"
            )
            raise ValueError(f"duplicate {keyword} record for station {station_id!r} at {epoch_label}.")
        group["components"].append(component)
        group["vector"].append(
            _measurement_value(record.get("value"), component=component, context=f"tracking observable {keyword!r}")
        )
        group["source_keywords"].append(keyword)
        group["source_indices"].append(index)
    rows = []

    def _group_sort_key(group: Mapping[str, Any]) -> tuple[float, int, float, str]:
        exact_epoch = group.get("time_tai_seconds")
        return (
            float(group["time_jd_utc"]),
            0 if exact_epoch is not None else 1,
            float(exact_epoch) if exact_epoch is not None else 0.0,
            str(group["station_id"]),
        )

    for row_index, group in enumerate(sorted(grouped.values(), key=_group_sort_key)):
        group["measurement_id"] = f"tracking-group:{row_index}"
        group["measurement_type"] = "+".join(group["components"])
        rows.append(group)
    return rows


def _base_row(
    item: Mapping[str, Any],
    *,
    station_id: str,
    station: Mapping[str, Any],
    default_id: str,
) -> dict[str, Any]:
    row = {
        "measurement_id": str(item.get("measurement_id", default_id) or default_id),
        "station_id": station_id,
        "station_metadata": dict(station),
        "time_jd_utc": _epoch_jd_utc(item),
        "measurement_type": str(item.get("measurement_type", "tracking_adapter") or "tracking_adapter"),
        "arc_id": str(item.get("arc_id", "tracking_adapter") or "tracking_adapter"),
    }
    if _has_exact_epoch(item):
        row["time_tai_seconds"] = _epoch_tai_seconds(item)
    _validate_epoch_consistency(
        item,
        epoch_jd_utc=float(row["time_jd_utc"]),
        epoch_tai_seconds=row.get("time_tai_seconds"),
    )
    return row


def _copy_uncertainty(source: Mapping[str, Any], target: dict[str, Any]) -> None:
    for key in ("sigma", "covariance", "uncertainty"):
        if source.get(key) is not None:
            target[key] = source[key]


def _station(stations: Mapping[str, Mapping[str, Any]], station_id: str) -> Mapping[str, Any]:
    if not station_id:
        raise ValueError("tracking observations require station_id.")
    if station_id not in stations:
        raise ValueError(f"tracking observation references unknown station {station_id!r}.")
    return stations[station_id]


def _epoch_jd_utc(value: Mapping[str, Any]) -> float:
    if value.get("time_jd_utc") is not None:
        parsed = float(value["time_jd_utc"])
    elif value.get("epoch_utc") is not None:
        epoch = parse_epoch(str(value["epoch_utc"]).strip(), TimeScale.UTC)
        parsed = float(epoch_julian_date(epoch, TimeScale.UTC))
    else:
        raise ValueError("tracking observation requires time_jd_utc or epoch_utc.")
    if not np.isfinite(parsed):
        raise ValueError("tracking observation epoch must be finite.")
    return parsed


def _epoch_tai_seconds(value: Mapping[str, Any]) -> float:
    if value.get("time_tai_seconds") is not None:
        parsed = float(value["time_tai_seconds"])
    elif value.get("epoch_utc") is not None:
        parsed = float(parse_epoch(str(value["epoch_utc"]).strip(), TimeScale.UTC).tai_seconds)
    else:
        raise ValueError("tracking observation requires time_tai_seconds or epoch_utc.")
    if not np.isfinite(parsed):
        raise ValueError("tracking observation epoch must be finite.")
    return parsed


def _has_exact_epoch(value: Mapping[str, Any]) -> bool:
    return value.get("time_tai_seconds") is not None or value.get("epoch_utc") is not None


def _validate_epoch_consistency(
    value: Mapping[str, Any],
    *,
    epoch_jd_utc: float,
    epoch_tai_seconds: float | None,
) -> None:
    """Reject contradictory exact and scalar epoch representations.

    Julian dates are retained for interoperability, but a UTC string or TAI
    scalar is the exact grouping identity.  Accepting contradictory fields
    would silently pair one time for grouping with another time for the OD
    frame conversion.
    """

    epoch_utc = value.get("epoch_utc")
    if epoch_utc is None:
        return
    parsed = parse_epoch(str(epoch_utc).strip(), TimeScale.UTC)
    parsed_jd_utc = float(epoch_julian_date(parsed, TimeScale.UTC))
    tolerance_s = 1.0e-3
    if value.get("time_jd_utc") is not None and abs((float(epoch_jd_utc) - parsed_jd_utc) * 86400.0) > tolerance_s:
        raise ValueError("tracking observation time_jd_utc is inconsistent with epoch_utc.")
    if value.get("time_tai_seconds") is not None and epoch_tai_seconds is not None:
        if abs(float(epoch_tai_seconds) - float(parsed.tai_seconds)) > tolerance_s:
            raise ValueError("tracking observation time_tai_seconds is inconsistent with epoch_utc.")


def _measurement_value(value: Any, *, component: str, context: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} {component} must be a finite numeric value.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} {component} must be a finite numeric value.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{context} {component} must be a finite numeric value.")
    if component == "azimuth_deg" and not -180.0 <= parsed < 360.0:
        raise ValueError(f"{context} azimuth_deg must lie within [-180, 360) degrees.")
    if component == "elevation_deg" and not -90.0 <= parsed <= 90.0:
        raise ValueError(f"{context} elevation_deg must lie within [-90, 90] degrees.")
    if component == "range_km" and parsed < 0.0:
        raise ValueError(f"{context} range_km must be nonnegative.")
    return parsed
