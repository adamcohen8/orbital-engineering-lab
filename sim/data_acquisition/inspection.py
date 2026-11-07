"""Acquisition inspection delegates orbital interpretation to existing owners."""
from __future__ import annotations

import io
import math
from collections import Counter
from datetime import datetime, timedelta, timezone

from .contracts import INSPECTOR_VERSION, AcquisitionError, digest, stamp, strict_json


def provider_epoch(value):
    # Space-Track explicitly documents UTC even when its field omits a suffix.
    if not isinstance(value, str):
        raise ValueError("Invalid provider epoch")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _coverage(values, *, daily=False):
    if not values:
        return None
    values = sorted(values)
    stop = values[-1] + timedelta(days=1) if daily else values[-1]
    return {"start": stamp(values[0]), "stop": stamp(stop), "stop_inclusive": not daily}


def gravity_metadata(raw):
    headers, masks, temporal = {}, {}, set()
    supplied_context = {}
    header_keys = {"product_type", "modelname", "earth_gravity_constant", "radius", "max_degree", "norm", "tide_system"}
    for binary in io.BytesIO(raw):
        parts = binary.decode("utf-8").split()
        if not parts:
            continue
        key = parts[0].lower()
        if key in header_keys and len(parts) > 1:
            if key in headers:
                raise ValueError("Duplicate gravity header")
            headers[key] = parts[1]
        elif key in {"gfct", "trnd", "dot", "asin", "acos"}:
            temporal.add(key)
        elif key == "gfc":
            if len(parts) < 5:
                raise ValueError("Incomplete gravity coefficient")
            n, m = int(parts[1]), int(parts[2])
            c, s = (float(v.replace("D", "E").replace("d", "e")) for v in parts[3:5])
            if not 0 <= m <= n <= 2400 or not math.isfinite(c) or not math.isfinite(s):
                raise ValueError("Invalid gravity coefficient")
            bit = 1 << m
            if masks.get(n, 0) & bit:
                raise ValueError("Duplicate gravity coefficient")
            masks[n] = masks.get(n, 0) | bit
    if "norm" not in headers:
        from sim.dynamics.orbit.gravity_assets import REAL_MODEL_DOWNLOADS

        # Only this exact legacy product receives its documented convention.
        # Unfamiliar user files still require explicit normalization metadata.
        if digest(raw) == REAL_MODEL_DOWNLOADS["EGM96"].sha256:
            headers["norm"] = "fully_normalized"
            supplied_context["norm"] = "fully_normalized (digest-pinned ICGEM EGM96 profile)"
    required = header_keys - {"product_type"}
    if not required <= headers.keys():
        raise ValueError("Gravity needs explicit model, GM, radius, max_degree, normalization, and tide_system headers")
    if headers.get("product_type", "gravity_field") != "gravity_field" or not masks:
        raise ValueError("Expected spherical harmonic gravity coefficients, not a geoid grid")
    mu, radius = (float(headers[k].replace("D", "E").replace("d", "e")) for k in ("earth_gravity_constant", "radius"))
    maximum = int(headers["max_degree"])
    if not 1e14 < mu < 1e15 or not 6e6 < radius < 7e6 or not 2 <= maximum <= 2400 or max(masks) > maximum:
        raise ValueError("Only bounded Earth gravity models with consistent degree and constants are supported")
    if headers["norm"] not in {"fully_normalized", "unnormalized"} or headers["tide_system"] not in {"tide_free", "zero_tide", "mean_tide"}:
        raise ValueError("Unsupported normalization or tide convention")
    complete = {}
    for n, mask in masks.items():
        m = 0
        while mask & (1 << m):
            m += 1
        complete[str(n)] = m - 1
    reasons = ["Time-dependent gravity terms require a separate supported consumer"] if temporal else []
    return {"model": headers["modelname"][:80], "center": "EARTH", "normalization": headers["norm"],
            "tide_system": headers["tide_system"], "gm_m3_s2": mu, "radius_m": radius,
            "max_degree": maximum, "max_order": max(mask.bit_length() - 1 for mask in masks.values()),
            "contiguous_order_by_degree": complete, "temporal_terms": sorted(temporal),
            "source_profile_context": supplied_context,
            "compatible": not reasons, "reasons": reasons}


def inspect_bytes(raw: bytes, *, kind: str, format: str, provider: str = "local"):
    result = {"inspector_version": INSPECTOR_VERSION, "kind": kind, "format": format,
              "parsed": False, "compatible": False, "reasons": [], "coverage": None}
    try:
        if kind == "gravity" and format == "icgem":
            result.update(gravity_metadata(raw), parsed=True)
        elif kind == "catalog" and format in {"gp-json", "tle", "gp-csv"}:
            from sim.data_io.catalog import from_gp_fields, parse_catalog
            if format == "gp-json":
                rows = strict_json(raw)
                if not isinstance(rows, list) or len(rows) > 50_000 or any(not isinstance(r, dict) for r in rows):
                    raise ValueError("Expected a bounded catalog history")
                records = [from_gp_fields(r, provider_profile="spacetrack-gp" if provider == "spacetrack" else "strict") for r in rows]
            else:
                records = parse_catalog(raw.decode("utf-8"), format=format, provider_profile="strict")
            from sim.dynamics.orbit.epoch import julian_date_to_datetime
            times = [julian_date_to_datetime(r.elements.epoch_jd_utc) for r in records]
            result.update(parsed=True, compatible=bool(records), record_count=len(records),
                          object_ids=sorted({str(r.object_id) for r in records}), coverage=_coverage(times))
            if not records:
                result["reasons"] = ["Provider returned no orbital records"]
        elif kind == "catalog-metadata" and format == "satcat-json":
            rows = strict_json(raw)
            if not isinstance(rows, list) or any(not isinstance(r, dict) or not str(r.get("NORAD_CAT_ID", "")).isdigit() for r in rows):
                raise ValueError("Expected SATCAT rows with catalog identifiers")
            result.update(parsed=True, record_count=len(rows), object_ids=sorted({str(r["NORAD_CAT_ID"]) for r in rows}),
                          reasons=["Catalog metadata does not supply an orbital state"])
        elif kind == "eop" and format in {"celestrak-eop-csv", "finals2000a", "c04_csv"}:
            from sim.dynamics.orbit.eop import EopSeries, _parse_c04_csv, _parse_finals2000a, parse_celestrak_eop
            text = raw.decode("utf-8")
            if format == "celestrak-eop-csv":
                series = parse_celestrak_eop(text, source_label="acquired-eop", source_sha256=digest(raw))
            else:
                records = tuple(_parse_c04_csv(text) if format == "c04_csv" else _parse_finals2000a(text.splitlines()))
                series = EopSeries(records, format, "acquired-eop", digest(raw))
            times = [datetime.fromtimestamp((r.mjd_utc - 40587) * 86400, timezone.utc) for r in series.records]
            gaps = [stamp(a) for a, b in zip(times, times[1:]) if (b - a).total_seconds() > 86400.01]
            result.update(parsed=True, compatible=not gaps, record_count=len(times), coverage=_coverage(times),
                          gaps_after=gaps, quality_counts=dict(Counter(r.quality for r in series.records)),
                          runtime_frames_compatible=format == "celestrak-eop-csv")
        elif kind == "space-weather" and format == "celestrak-weather-csv":
            from sim.dynamics.orbit.space_weather_data import parse_celestrak_weather
            rows = parse_celestrak_weather(raw.decode("utf-8"))
            times = [datetime.combine(r["date"], datetime.min.time(), timezone.utc) for r in rows]
            result.update(parsed=True, compatible=True, record_count=len(rows), coverage=_coverage(times, daily=True),
                          quality_counts=dict(Counter(r["F10.7_DATA_TYPE"] for r in rows)),
                          required_history_days=3, supported_consumers=["nrlmsise00", "msis86"],
                          flux_policy="Existing MSIS consumer: previous observed F10.7, current adjusted centered-81-day F10.7")
        elif kind == "opaque" and format == "opaque":
            result["reasons"] = ["Stored as opaque bytes; no OEL compatibility has been established"]
        else:
            result["reasons"] = ["This acquisition inspection profile is not supported"]
    except (ValueError, KeyError, TypeError, UnicodeError, OverflowError) as exc:
        # Never echo provider contents/HTML or arbitrary exceptions into agent output.
        result["reasons"] = ["Source failed its declared format/convention checks"]
        result["failure_type"] = type(exc).__name__
    return result


def require_compatible(inspection):
    if not inspection["parsed"] or not inspection["compatible"]:
        raise AcquisitionError("INCOMPATIBLE_DATA", "; ".join(inspection["reasons"]) or "Dataset is incompatible with this consumer.")
