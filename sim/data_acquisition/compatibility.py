"""Explicit study requirements; no downloaded artifact authorizes execution."""
from __future__ import annotations

from datetime import timedelta

from .contracts import AcquisitionError, epoch
from .inspection import inspect_bytes, require_compatible

REQUIREMENTS = {"consumer", "start", "stop", "allow_predictions", "degree", "order", "tide_system", "normalization"}


def check(raw, receipt, requirements):
    if not isinstance(requirements, dict) or set(requirements) - REQUIREMENTS:
        raise AcquisitionError("INVALID_REQUIREMENTS", "Use documented study requirement fields only.")
    value = inspect_bytes(raw, kind=receipt["kind"], format=receipt["format"], provider=receipt["provider"])
    require_compatible(value)
    consumer = requirements.get("consumer")
    kind = receipt["kind"]
    if "allow_predictions" in requirements and type(requirements["allow_predictions"]) is not bool:
        raise AcquisitionError("INVALID_REQUIREMENTS", "allow_predictions must be a boolean.")
    if kind == "gravity":
        if consumer != "earth-gravity":
            raise AcquisitionError("INVALID_REQUIREMENTS", "Gravity requires consumer=earth-gravity.")
        degree, order = requirements.get("degree"), requirements.get("order")
        if type(degree) is not int or type(order) is not int or not 2 <= degree <= value["max_degree"] or not 0 <= order <= degree:
            raise AcquisitionError("INCOMPATIBLE_DATA", "Specify an available degree >= 2 and order in 0..degree.")
        if any(value["contiguous_order_by_degree"].get(str(n), -1) < min(n, order) for n in range(2, degree + 1)):
            raise AcquisitionError("INCOMPATIBLE_DATA", "Source is missing coefficients required by this degree/order selection.")
        if requirements.get("tide_system") != value["tide_system"] or requirements.get("normalization") != value["normalization"]:
            raise AcquisitionError("INCOMPATIBLE_DATA", "Select the source tide convention and normalization explicitly.")
    elif kind in {"eop", "space-weather"}:
        start, stop = epoch(requirements.get("start")), epoch(requirements.get("stop"))
        if stop <= start:
            raise AcquisitionError("INVALID_REQUIREMENTS", "Requirements need start before stop.")
        # Simulation outputs include the final sample, so stop must itself be covered.
        coverage = value["coverage"]
        if start < epoch(coverage["start"]) or stop > epoch(coverage["stop"]) or (kind == "space-weather" and stop == epoch(coverage["stop"])):
            raise AcquisitionError("COVERAGE_MISSING", "Dataset does not cover every requested simulation sample, including stop.")
        if kind == "space-weather":
            if consumer not in {"nrlmsise00", "msis86"}:
                raise AcquisitionError("INCOMPATIBLE_DATA", "This weather profile supports only the existing NRLMSISE-00 and MSIS-86 consumers.")
            from sim.dynamics.orbit.space_weather_data import check_msis_coverage, parse_celestrak_weather
            try:
                check_msis_coverage(parse_celestrak_weather(raw.decode()), start.date(), stop.date(),
                                    allow_predictions=requirements.get("allow_predictions", False))
            except ValueError as exc:
                raise AcquisitionError("COVERAGE_MISSING", str(exc)) from exc
        else:
            if consumer not in {"runtime-frames", "frame-time"} or (consumer == "runtime-frames" and not value.get("runtime_frames_compatible")):
                raise AcquisitionError("INCOMPATIBLE_DATA", "This EOP format requires a compatible frame consumer; runtime frames require explicit DAT.")
            from sim.dynamics.orbit.eop import _parse_c04_csv, _parse_finals2000a, parse_celestrak_eop
            if receipt["format"] == "celestrak-eop-csv":
                rows = parse_celestrak_eop(raw.decode(), source_label="acquired-eop", source_sha256=receipt["sha256"]).records
            else:
                rows = tuple(_parse_c04_csv(raw.decode()) if receipt["format"] == "c04_csv" else _parse_finals2000a(raw.decode().splitlines()))
            # Include interpolation neighbors; no unannounced use of predicted EOP.
            lo = (start.timestamp() / 86400 + 40587).__floor__()
            hi = (stop.timestamp() / 86400 + 40587).__ceil__()
            if not requirements.get("allow_predictions", False) and any(r.quality == "predicted" and lo <= r.mjd_utc <= hi for r in rows):
                raise AcquisitionError("PREDICTED_DATA", "Predicted EOP requires allow_predictions=true.")
    elif kind == "catalog":
        if consumer != "ogp":
            raise AcquisitionError("INCOMPATIBLE_DATA", "Catalog mean elements require explicit OGP selection; they are not osculating ONP elements.")
        raise AcquisitionError("SELECTION_REQUIRED", "Select exact catalog records with data select before binding a propagation study.")
    else:
        raise AcquisitionError("INCOMPATIBLE_DATA", "No executable study consumer is available for this product.")
    return {"compatible": True, "requirements": requirements, "inspection": value}


def consumer_bytes(raw, receipt, requirements):
    check(raw, receipt, requirements)
    if receipt["kind"] == "eop" and requirements["consumer"] == "runtime-frames":
        from sim.dynamics.orbit.eop import encode_eop_all, parse_celestrak_eop
        return encode_eop_all(parse_celestrak_eop(raw.decode(), source_label="acquired-eop", source_sha256=receipt["sha256"])), "eop-all"
    if receipt["kind"] == "space-weather":
        from sim.dynamics.orbit.space_weather_data import encode_msis_table, parse_celestrak_weather
        start, stop = epoch(requirements["start"]).date() - timedelta(days=3), epoch(requirements["stop"]).date()
        rows = [r for r in parse_celestrak_weather(raw.decode()) if start <= r["date"] <= stop]
        return encode_msis_table(rows), "msis-sw"
    return raw, receipt["format"]
