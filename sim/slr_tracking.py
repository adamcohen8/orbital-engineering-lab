from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.epoch import datetime_to_julian_date
from sim.dynamics.orbit.frames import eci_to_ecef_rotation_hpop_like
from sim.utils.io import read_regular_file_nofollow

C_LIGHT_KM_S = 299_792.458
ILRS_CRD_ADAPTER_SCHEMA_VERSION = 1
SLR_TWO_WAY_CONTRACT = "ilrs_crd_v2_two_way_normal_point_v1"
MAX_CRD_BYTES = 64 * 1024 * 1024


def _norm3(values: Sequence[float]) -> float:
    """Keep scalar SLR geometry arithmetic identical to the native path."""

    x, y, z = (float(value) for value in values)
    return math.sqrt(x * x + y * y + z * z)


def parse_ilrs_crd_v2_normal_points(path: str | Path) -> dict[str, Any]:
    """Parse governed CRD v2 normal-point records without changing their correction semantics."""

    source = Path(path).expanduser().absolute()
    content = read_regular_file_nofollow(source, min_bytes=1, max_bytes=MAX_CRD_BYTES)
    lines = content.decode("utf-8", errors="strict").splitlines()
    file_header: dict[str, Any] | None = None
    station: dict[str, Any] | None = None
    target: dict[str, Any] | None = None
    session: dict[str, Any] | None = None
    sessions: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        record = fields[0].upper()
        if record == "H1":
            if len(fields) < 3 or fields[1].upper() != "CRD" or int(fields[2]) != 2:
                raise ValueError("ILRS adapter requires CRD version 2 normal-point data.")
            file_header = {
                "format": "CRD",
                "version": int(fields[2]),
                "production_fields": fields[3:],
                "raw": line,
            }
        elif record == "H2":
            if len(fields) < 6:
                raise ValueError(f"Malformed H2 station header at line {line_number}.")
            station = {
                "code": fields[1].upper(),
                "pad_id": fields[2],
                "system_number": fields[3],
                "occupation_number": fields[4],
                "timescale": fields[5],
                "network": fields[6] if len(fields) > 6 else None,
                "raw": line,
            }
        elif record == "H3":
            if len(fields) < 8:
                raise ValueError(f"Malformed H3 target header at line {line_number}.")
            target = {
                "name": fields[1],
                "ilrs_id": fields[2],
                "sic": fields[3],
                "norad_id": fields[4],
                "spacecraft_timescale": fields[5],
                "target_class": fields[6],
                "target_location": fields[7],
                "raw": line,
            }
        elif record == "H4":
            if file_header is None or station is None or target is None:
                raise ValueError(f"H4 session at line {line_number} lacks H1/H2/H3 context.")
            if len(fields) < 22:
                raise ValueError(f"Malformed H4 session header at line {line_number}.")
            start = _crd_datetime(fields[2:8])
            end = _crd_datetime(fields[8:14])
            session = {
                "station": dict(station),
                "target": dict(target),
                "data_type": int(fields[1]),
                "start_utc": start.isoformat().replace("+00:00", "Z"),
                "end_utc": end.isoformat().replace("+00:00", "Z"),
                "start_jd_utc": datetime_to_julian_date(start),
                "end_jd_utc": datetime_to_julian_date(end),
                "data_release": int(fields[14]),
                "correction_flags": {
                    "tropospheric_refraction_applied": bool(int(fields[15])),
                    "center_of_mass_applied": bool(int(fields[16])),
                    "receive_amplitude_applied": bool(int(fields[17])),
                    "station_system_delay_applied": bool(int(fields[18])),
                    "spacecraft_system_delay_applied": bool(int(fields[19])),
                },
                "range_type": int(fields[20]),
                "data_quality_alert": int(fields[21]),
                "normal_points": [],
                "meteorology": [],
                "range_supplements": [],
                "calibration_records": [],
                "session_statistics": [],
                "configuration_records": [],
                "raw_header": line,
            }
            sessions.append(session)
        elif record.startswith("C") and session is not None:
            session["configuration_records"].append({"record_type": record, "raw": line})
        elif record in {"40", "41", "42"} and session is not None:
            session["calibration_records"].append({"record_type": record, "raw": line})
        elif record == "20" and session is not None:
            session["meteorology"].append(_parse_meteorology(fields, line_number=line_number))
        elif record == "12" and session is not None:
            session["range_supplements"].append(_parse_range_supplement(fields, line_number=line_number))
        elif record == "11" and session is not None:
            session["normal_points"].append(_parse_normal_point(fields, session=session, line_number=line_number))
        elif record == "50" and session is not None:
            session["session_statistics"].append({"raw": line})
        elif record == "H8":
            session = None

    if file_header is None:
        raise ValueError(f"No CRD v2 H1 header found in {source}.")
    normal_point_count = sum(len(item["normal_points"]) for item in sessions)
    if normal_point_count == 0:
        raise ValueError(f"No CRD normal-point records found in {source}.")
    return {
        "schema_version": ILRS_CRD_ADAPTER_SCHEMA_VERSION,
        "format": "ilrs_crd_v2",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "file_header": file_header,
        "sessions": sessions,
        "session_count": len(sessions),
        "normal_point_count": normal_point_count,
        "measurement_contract": SLR_TWO_WAY_CONTRACT,
    }


def adapt_ilrs_crd_normal_points(
    packet: Mapping[str, Any],
    *,
    station_catalog: Mapping[str, Mapping[str, Any]],
    station_ids: Sequence[str] = (),
    sigma_floor_ps: float = 50.0,
    minimum_normal_points: int = 2,
) -> dict[str, Any]:
    """Attach governed station coordinates and expose two-way TOF observation rows.

    Forecast batches may explicitly accept a singleton file. Estimators and
    qualification protocols still enforce their own combined sample minimum.
    The default two-point contract and its validation error are unchanged.
    """

    if (
        isinstance(minimum_normal_points, (bool, np.bool_))
        or not isinstance(minimum_normal_points, (int, np.integer))
        or minimum_normal_points not in (1, 2)
    ):
        raise ValueError("minimum_normal_points must be 1 or 2.")
    if int(packet.get("schema_version", 0)) != ILRS_CRD_ADAPTER_SCHEMA_VERSION:
        raise ValueError(f"CRD packet schema_version must be {ILRS_CRD_ADAPTER_SCHEMA_VERSION}.")
    selected = {str(item).upper() for item in station_ids if str(item)}
    if isinstance(sigma_floor_ps, (bool, np.bool_)):
        raise ValueError("sigma_floor_ps must be finite and positive.")
    floor_ps = float(sigma_floor_ps)
    if not np.isfinite(floor_ps) or floor_ps <= 0.0:
        raise ValueError("sigma_floor_ps must be finite and positive.")
    rows: list[dict[str, Any]] = []
    omitted_sessions: list[dict[str, Any]] = []
    for session_index, raw_session in enumerate(packet.get("sessions", [])):
        session = dict(raw_session)
        station = dict(session["station"])
        code = str(station["code"]).upper()
        if selected and code not in selected:
            continue
        if code not in station_catalog:
            omitted_sessions.append(
                {"session_index": session_index, "station_id": code, "reason": "missing_station_metadata"}
            )
            continue
        station_meta = _normalize_station_metadata(code, station_catalog[code])
        if int(session["data_type"]) != 1:
            omitted_sessions.append(
                {"session_index": session_index, "station_id": code, "reason": "not_normal_point_data"}
            )
            continue
        if int(session["range_type"]) != 2:
            omitted_sessions.append({"session_index": session_index, "station_id": code, "reason": "not_two_way_range"})
            continue
        flags = dict(session["correction_flags"])
        for point_index, raw_point in enumerate(session["normal_points"]):
            point = dict(raw_point)
            raw_epoch_event = point.get("epoch_event")
            if isinstance(raw_epoch_event, (bool, np.bool_)):
                raise ValueError("CRD epoch_event must be a finite integer in {0, 1, 2}.")
            try:
                epoch_event = float(raw_epoch_event)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("CRD epoch_event must be a finite integer in {0, 1, 2}.") from exc
            if not np.isfinite(epoch_event) or not epoch_event.is_integer():
                raise ValueError("CRD epoch_event must be a finite integer in {0, 1, 2}.")
            if int(epoch_event) not in {0, 1, 2}:
                omitted_sessions.append(
                    {"session_index": session_index, "station_id": code, "reason": "unsupported_epoch_event"}
                )
                continue
            rms_ps = point.get("bin_rms_ps")
            if rms_ps is not None:
                if isinstance(rms_ps, (bool, np.bool_)):
                    raise ValueError("CRD bin_rms_ps must be finite and nonnegative when provided.")
                try:
                    rms_ps = float(rms_ps)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("CRD bin_rms_ps must be finite and nonnegative when provided.") from exc
                if not np.isfinite(rms_ps) or rms_ps < 0.0:
                    raise ValueError("CRD bin_rms_ps must be finite and nonnegative when provided.")
            sigma_ps = max(floor_ps, rms_ps) if rms_ps is not None else floor_ps
            try:
                tof_s = float(point["time_of_flight_s"])
                time_jd_utc = float(point["time_jd_utc"])
                normal_point_window_s = float(point["normal_point_window_s"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("CRD normal-point time fields must be numeric.") from exc
            if not np.isfinite(tof_s) or tof_s <= 0.0:
                raise ValueError("CRD time_of_flight_s must be finite and positive.")
            if not np.isfinite(time_jd_utc):
                raise ValueError("CRD time_jd_utc must be finite.")
            if not np.isfinite(normal_point_window_s) or normal_point_window_s <= 0.0:
                raise ValueError("CRD normal_point_window_s must be finite and positive.")
            rows.append(
                {
                    "measurement_id": f"crd:{session_index:04d}:{point_index:04d}",
                    "source_format": "ilrs_crd_v2",
                    "source_sha256": str(packet["source_sha256"]),
                    "measurement_contract": SLR_TWO_WAY_CONTRACT,
                    "station_id": code,
                    "station_metadata": station_meta,
                    "target_metadata": dict(session["target"]),
                    "time_jd_utc": time_jd_utc,
                    "epoch_event": int(epoch_event),
                    "observed_time_of_flight_s": tof_s,
                    "observed_two_way_range_km": 0.5 * C_LIGHT_KM_S * tof_s,
                    "sigma_time_of_flight_s": sigma_ps * 1.0e-12,
                    "sigma_range_km": 0.5 * C_LIGHT_KM_S * sigma_ps * 1.0e-12,
                    "normal_point_window_s": normal_point_window_s,
                    "raw_range_count": int(point["raw_range_count"]),
                    "bin_rms_ps": rms_ps,
                    "bin_skew": point.get("bin_skew"),
                    "bin_kurtosis": point.get("bin_kurtosis"),
                    "return_rate_percent": point.get("return_rate_percent"),
                    "signal_to_noise_ratio": point.get("signal_to_noise_ratio"),
                    "data_quality_alert": int(session["data_quality_alert"]),
                    "correction_flags": flags,
                    "calibration_records": list(session["calibration_records"]),
                    "range_supplements": list(session["range_supplements"]),
                }
            )
    if len(rows) < minimum_normal_points:
        count = "two" if minimum_normal_points == 2 else "one"
        raise ValueError(f"CRD adaptation requires at least {count} supported normal points with governed station metadata.")
    rows.sort(key=lambda item: (float(item["time_jd_utc"]), str(item["station_id"]), str(item["measurement_id"])))
    epoch0 = float(rows[0]["time_jd_utc"])
    for row in rows:
        row["time_s"] = (float(row["time_jd_utc"]) - epoch0) * 86400.0
    return {
        "schema_version": ILRS_CRD_ADAPTER_SCHEMA_VERSION,
        "adapter": "oel_ilrs_crd_v2_normal_point_adapter",
        "measurement_contract": SLR_TWO_WAY_CONTRACT,
        "measurement_rows": rows,
        "measurement_count": len(rows),
        "station_ids": sorted({str(row["station_id"]) for row in rows}),
        "omitted_sessions": omitted_sessions,
        "provenance": {
            "source_path": str(packet["source_path"]),
            "source_sha256": str(packet["source_sha256"]),
            "time_system": "UTC",
            "observed_quantity": "two_way_time_of_flight_s",
            "range_conversion": "c * time_of_flight / 2",
            "speed_of_light_km_s": C_LIGHT_KM_S,
            "sigma_floor_ps": floor_ps,
            "correction_policy": "preserve_h4_flags_and_expose_unmodeled_path_terms_as_explicit_bias",
        },
        "non_claims": [
            "Prediction products are not observations or perfect truth.",
            "Approximate station coordinates do not establish millimeter station-orbit accuracy.",
            "An estimated aggregate station range bias is not a replacement for troposphere, target-signature, or center-of-mass calibration.",
        ],
    }


def earth_shapiro_range_delay_km(emitter_km, receiver_km, *, earth_mu_km3_s2):
    """Earth monopole photon-path delay for one leg, expressed as range.

    IERS relativistic signal propagation: 2 GM/c² log((r1+r2+R)/(r1+r2-R)).
    Endpoints must be geocentric in one common inertial frame. This is separate
    from the relativistic acceleration of the spacecraft.
    """
    mu = float(earth_mu_km3_s2)
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError("Earth Shapiro GM must be finite and nonnegative.")
    a, b = np.asarray(emitter_km, dtype=float), np.asarray(receiver_km, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not np.all(np.isfinite([a, b])):
        raise ValueError("Earth Shapiro endpoints must be finite three-vectors.")
    r1, r2, distance = _norm3(a), _norm3(b), _norm3(a-b)
    denominator = r1 + r2 - distance
    if min(r1, r2, denominator) <= 0.0:
        raise ValueError("Earth Shapiro path has singular geocentric geometry.")
    return float(2.0 * mu / C_LIGHT_KM_S**2 * math.log1p(2.0 * distance / denominator))


def predict_slr_two_way_time_of_flight_s(
    *,
    epoch_jd_utc: float,
    epoch_event: int,
    station_ecef_km: Sequence[float],
    state_at_jd_utc: Callable[[float], Sequence[float]],
    frame_model: str = "simple",
    eop_path: str | Path | None = None,
    aggregate_one_way_range_correction_km: float = 0.0,
    state_at_offset_s: Callable[[float], Sequence[float]] | None = None,
    epoch_offset_s: float = 0.0,
    earth_shapiro_mu_km3_s2: float = 0.0,
    max_iterations: int = 12,
    tolerance_s: float = 1.0e-13,
) -> dict[str, Any]:
    """Predict passive-reflector two-way TOF for CRD epoch events 0, 1, or 2."""

    event_value = float(epoch_event)
    if not np.isfinite(event_value) or event_value not in {0.0, 1.0, 2.0}:
        raise ValueError("SLR two-way prediction supports CRD epoch events 0, 1, and 2 only.")
    event = int(event_value)
    if not np.isfinite(float(epoch_jd_utc)):
        raise ValueError("SLR epoch_jd_utc must be finite.")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations <= 0:
        raise ValueError("SLR max_iterations must be a positive integer.")
    if not np.isfinite(float(tolerance_s)) or float(tolerance_s) <= 0.0:
        raise ValueError("SLR tolerance_s must be finite and positive.")
    if not np.isfinite(float(aggregate_one_way_range_correction_km)):
        raise ValueError("SLR aggregate_one_way_range_correction_km must be finite.")
    station_ecef = np.asarray(station_ecef_km, dtype=float).reshape(3)
    if not np.all(np.isfinite(station_ecef)):
        raise ValueError("station_ecef_km must contain three finite values.")
    frame = str(frame_model).strip().lower()
    if frame not in {"simple", "hpop_like"}:
        raise ValueError("frame_model must be simple or hpop_like.")
    if frame == "hpop_like" and eop_path is None:
        raise ValueError("hpop_like SLR prediction requires eop_path.")
    reference_jd = np.floor(float(epoch_jd_utc) - 0.5) + 0.5
    relative_time = state_at_offset_s is not None
    if not np.isfinite(epoch_offset_s) or (epoch_offset_s != 0 and not relative_time):
        raise ValueError("epoch_offset_s requires a finite relative-time SLR contract")
    shapiro_mu = float(earth_shapiro_mu_km3_s2)
    if not np.isfinite(shapiro_mu) or shapiro_mu < 0.0 or (shapiro_mu > 0 and not relative_time):
        raise ValueError("Earth Shapiro correction requires nonnegative finite GM and relative light time.")
    event_epoch = 0.0 if relative_time else float(epoch_jd_utc)
    second = 1.0 if relative_time else 1.0 / 86400.0
    station_cache: dict[float, np.ndarray] = {}
    satellite_cache: dict[float, np.ndarray] = {}

    def station_eci(jd_utc: float) -> np.ndarray:
        key = float(jd_utc)
        cached = station_cache.get(key)
        if cached is not None:
            return cached
        rot = eci_to_ecef_rotation_hpop_like(
            key + epoch_offset_s if relative_time else (key - reference_jd) * 86400.0,
            jd_utc_start=float(epoch_jd_utc) if relative_time else reference_jd,
            eop_path=str(eop_path) if frame == "hpop_like" else None,
        )
        value = np.asarray([
            float(rot[0, i]) * float(station_ecef[0])
            + float(rot[1, i]) * float(station_ecef[1])
            + float(rot[2, i]) * float(station_ecef[2])
            for i in range(3)
        ])
        station_cache[key] = value
        return value

    def sat_position(jd_utc: float) -> np.ndarray:
        key = float(jd_utc)
        cached = satellite_cache.get(key)
        if cached is not None:
            return cached
        state = np.asarray(state_at_offset_s(key) if relative_time else state_at_jd_utc(key), dtype=float).reshape(-1)
        if state.size < 3 or not np.all(np.isfinite(state[:3])):
            raise ValueError("state_at_jd_utc returned an invalid state.")
        value = state[:3]
        satellite_cache[key] = value
        return value

    def light_time(sat_epoch, station_epoch):
        sat, site = sat_position(sat_epoch), station_eci(station_epoch)
        geometric = _norm3(sat - site)
        if shapiro_mu:
            geometric += earth_shapiro_range_delay_km(sat, site, earth_mu_km3_s2=shapiro_mu)
        return geometric / C_LIGHT_KM_S

    if event == 2:
        tx_jd = event_epoch
        tau_out = 0.04
        out_converged = False
        for _ in range(max_iterations):
            bounce_jd = tx_jd + tau_out * second
            updated = light_time(bounce_jd, tx_jd)
            if abs(updated - tau_out) <= tolerance_s:
                tau_out = float(updated)
                out_converged = True
                break
            tau_out = float(updated)
        bounce_jd = tx_jd + tau_out * second
        tau_in = tau_out
        in_converged = False
        for _ in range(max_iterations):
            rx_jd = bounce_jd + tau_in * second
            updated = light_time(bounce_jd, rx_jd)
            if abs(updated - tau_in) <= tolerance_s:
                tau_in = float(updated)
                in_converged = True
                break
            tau_in = float(updated)
        rx_jd = bounce_jd + tau_in * second
    elif event == 0:
        rx_jd = event_epoch
        tau_in = 0.04
        in_converged = False
        for _ in range(max_iterations):
            bounce_jd = rx_jd - tau_in * second
            updated = light_time(bounce_jd, rx_jd)
            if abs(updated - tau_in) <= tolerance_s:
                tau_in = float(updated)
                in_converged = True
                break
            tau_in = float(updated)
        bounce_jd = rx_jd - tau_in * second
        tau_out = tau_in
        out_converged = False
        for _ in range(max_iterations):
            tx_jd = bounce_jd - tau_out * second
            updated = light_time(bounce_jd, tx_jd)
            if abs(updated - tau_out) <= tolerance_s:
                tau_out = float(updated)
                out_converged = True
                break
            tau_out = float(updated)
        tx_jd = bounce_jd - tau_out * second
    else:
        bounce_jd = event_epoch
        tau_out = 0.04
        out_converged = False
        for _ in range(max_iterations):
            tx_jd = bounce_jd - tau_out * second
            updated = light_time(bounce_jd, tx_jd)
            if abs(updated - tau_out) <= tolerance_s:
                tau_out = float(updated)
                out_converged = True
                break
            tau_out = float(updated)
        tx_jd = bounce_jd - tau_out * second
        tau_in = tau_out
        in_converged = False
        for _ in range(max_iterations):
            rx_jd = bounce_jd + tau_in * second
            updated = light_time(bounce_jd, rx_jd)
            if abs(updated - tau_in) <= tolerance_s:
                tau_in = float(updated)
                in_converged = True
                break
            tau_in = float(updated)
        rx_jd = bounce_jd + tau_in * second

    out_error = abs(tau_out - light_time(bounce_jd, tx_jd))
    in_error = abs(tau_in - light_time(bounce_jd, rx_jd))
    if not (out_converged and in_converged) or max(out_error, in_error) > tolerance_s:
        raise ValueError("SLR light-time iteration did not converge")
    offsets = {"transmit_offset_s": tx_jd, "bounce_offset_s": bounce_jd, "receive_offset_s": rx_jd,
               "epoch_offset_s": float(epoch_offset_s)} if relative_time else {}
    shapiro = {}
    shapiro_tof_s = 0.0
    if shapiro_mu:
        delays = [earth_shapiro_range_delay_km(sat_position(bounce_jd), station_eci(t),
                  earth_mu_km3_s2=shapiro_mu) for t in (tx_jd, rx_jd)]
        shapiro_tof_s = sum(delays) / C_LIGHT_KM_S
        shapiro = {"earth_shapiro_mu_km3_s2": shapiro_mu,
                   "earth_shapiro_time_of_flight_s": shapiro_tof_s,
                   "earth_shapiro_one_way_equivalent_km": 0.5 * sum(delays)}
    if relative_time:
        tx_jd, bounce_jd, rx_jd = [float(epoch_jd_utc) + (v + epoch_offset_s) / 86400.0 for v in (tx_jd, bounce_jd, rx_jd)]
    geometric_tof_s = float(tau_out + tau_in) - shapiro_tof_s
    correction_km = float(aggregate_one_way_range_correction_km)
    predicted_tof_s = float(tau_out + tau_in) + 2.0 * correction_km / C_LIGHT_KM_S
    return {
        **offsets,
        **shapiro,
        "measurement_contract": SLR_TWO_WAY_CONTRACT,
        "epoch_event": event,
        "transmit_jd_utc": float(tx_jd),
        "bounce_jd_utc": float(bounce_jd),
        "receive_jd_utc": float(rx_jd),
        "outbound_light_time_s": float(tau_out),
        "inbound_light_time_s": float(tau_in),
        "geometric_time_of_flight_s": geometric_tof_s,
        "aggregate_one_way_range_correction_km": correction_km,
        "predicted_time_of_flight_s": float(predicted_tof_s),
        "predicted_two_way_range_km": 0.5 * C_LIGHT_KM_S * float(predicted_tof_s),
        "frame_model": frame,
    }


def _parse_normal_point(fields: list[str], *, session: Mapping[str, Any], line_number: int) -> dict[str, Any]:
    if len(fields) < 14:
        raise ValueError(f"Malformed CRD 11 normal-point record at line {line_number}.")
    seconds_of_day = float(fields[1])
    epoch = _epoch_from_session(seconds_of_day, session)
    return {
        "seconds_of_day": seconds_of_day,
        "time_jd_utc": datetime_to_julian_date(epoch),
        "time_of_flight_s": float(fields[2]),
        "system_configuration_id": fields[3],
        "epoch_event": int(fields[4]),
        "normal_point_window_s": float(fields[5]),
        "raw_range_count": int(fields[6]),
        "bin_rms_ps": _optional_float(fields[7]),
        "bin_skew": _optional_float(fields[8]),
        "bin_kurtosis": _optional_float(fields[9]),
        "bin_peak_mean_ps": _optional_float(fields[10]),
        "return_rate_percent": _optional_float(fields[11]),
        "detector_channel": int(fields[12]),
        "signal_to_noise_ratio": _optional_float(fields[13]),
    }


def _parse_meteorology(fields: list[str], *, line_number: int) -> dict[str, Any]:
    if len(fields) < 6:
        raise ValueError(f"Malformed CRD 20 meteorology record at line {line_number}.")
    return {
        "seconds_of_day": float(fields[1]),
        "pressure_mbar": float(fields[2]),
        "temperature_k": float(fields[3]),
        "relative_humidity_percent": float(fields[4]),
        "origin": int(fields[5]),
    }


def _parse_range_supplement(fields: list[str], *, line_number: int) -> dict[str, Any]:
    if len(fields) < 8:
        raise ValueError(f"Malformed CRD 12 range-supplement record at line {line_number}.")
    return {
        "seconds_of_day": float(fields[1]),
        "system_configuration_id": fields[2],
        "tropospheric_refraction_one_way_ps": _optional_float(fields[3]),
        "target_center_of_mass_one_way_m": _optional_float(fields[4]),
        "neutral_density_filter": _optional_float(fields[5]),
        "time_bias_applied_s": _optional_float(fields[6]),
        "range_rate_s_per_s": _optional_float(fields[7]),
    }


def _normalize_station_metadata(code: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    xyz = np.asarray(data.get("ecef_m"), dtype=float).reshape(3)
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"station {code!r} requires finite ecef_m coordinates.")
    source_url = str(data.get("source_url", "")).strip()
    if not source_url:
        raise ValueError(f"station {code!r} requires source_url provenance.")
    return {
        "id": code,
        "ecef_km": [float(value) / 1000.0 for value in xyz],
        "coordinate_frame": str(data.get("coordinate_frame", "ITRF_approximate")),
        "coordinate_epoch": data.get("coordinate_epoch"),
        "source_url": source_url,
        "source_path": data.get("source_path"),
        "quality": str(data.get("quality", "official_site_approximate_position")),
    }


def _epoch_from_session(seconds_of_day: float, session: Mapping[str, Any]) -> datetime:
    start = datetime.fromisoformat(str(session["start_utc"]).replace("Z", "+00:00"))
    midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
    epoch = midnight + timedelta(seconds=float(seconds_of_day))
    if epoch + timedelta(seconds=1.0) < start:
        epoch += timedelta(days=1)
    return epoch


def _crd_datetime(fields: Sequence[str]) -> datetime:
    if len(fields) != 6:
        raise ValueError("CRD date/time requires six fields.")
    year, month, day, hour, minute = (int(value) for value in fields[:5])
    second = float(fields[5])
    base = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return base + timedelta(seconds=second)


def _optional_float(value: str) -> float | None:
    if str(value).strip().lower() in {"na", "nan", "none"}:
        return None
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
