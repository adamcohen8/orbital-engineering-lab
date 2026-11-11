"""Opt-in CRD elapsed seconds and sub-Julian-date station epoch offsets."""
from __future__ import annotations

import copy
from datetime import datetime, timezone

import numpy as np

from sim.dynamics.orbit.epoch import datetime_to_julian_date
from sim.slr_tracking import _epoch_from_session


def bind_precise_slr_epochs(rows, packets, *, reference_epoch=None):
    """Return copied rows and a reusable two-part UTC reference epoch.

    Match the original source hash, session/point identity, station and scalar
    JD before using its unrounded seconds-of-day. Keep scalar JDs for resource
    coverage/metadata, but do not subtract them to schedule propagation.
    ``epoch_offset_s`` is the small correction to each scalar station epoch;
    it requires the relative-light-time prediction contract. This does not
    make the force environment's scalar absolute epoch leap-second aware.
    """
    if not rows:
        raise ValueError("Precise SLR timing requires observations")
    points = {}
    for packet in packets:
        for si, session in enumerate(packet["sessions"]):
            for pi, point in enumerate(session["normal_points"]):
                seconds = float(point["seconds_of_day"])
                if not np.isfinite(seconds) or not 0 <= seconds < 86400:
                    raise ValueError("Precise SLR timing requires ordinary UTC seconds-of-day; leap seconds are unsupported")
                day = _epoch_from_session(seconds, session).date()
                midnight = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
                value = (datetime_to_julian_date(midnight), seconds,
                         str(session["station"]["code"]).upper(), float(point["time_jd_utc"]))
                key = (str(packet["source_sha256"]), f"crd:{si:04d}:{pi:04d}")
                if key in points:
                    raise ValueError("Duplicate precise SLR source/point identity")
                points[key] = value
    result = copy.deepcopy(rows)
    bound = []
    for row in result:
        key = (str(row["source_sha256"]), str(row["measurement_id"]))
        if key not in points:
            raise ValueError("Precise SLR timing source/point is missing")
        day, seconds, station, jd = points[key]
        if station != row["station_id"] or jd != float(row["time_jd_utc"]):
            raise ValueError("Precise SLR timing observation identity differs from the source")
        bound.append((day, seconds))
    reference = dict(reference_epoch) if reference_epoch is not None else dict(
        utc_midnight_jd=bound[0][0], seconds_of_day=bound[0][1])
    if set(reference) != {"utc_midnight_jd", "seconds_of_day"}:
        raise ValueError("Precise SLR reference requires a UTC midnight JD and seconds-of-day")
    midnight, seconds0 = float(reference["utc_midnight_jd"]), float(reference["seconds_of_day"])
    if (not np.isfinite(midnight) or midnight % 1 != .5
            or not np.isfinite(seconds0) or not 0 <= seconds0 < 86400):
        raise ValueError("Invalid precise SLR reference epoch")
    for row, (day, seconds) in zip(result, bound, strict=True):
        row["time_s"] = (day-midnight) * 86400. + (seconds-seconds0)
        offset = seconds - (float(row["time_jd_utc"])-day) * 86400.
        if not np.isfinite(offset) or abs(offset) > .001:
            raise ValueError("Precise SLR station epoch differs by more than Julian-date rounding")
        row["epoch_offset_s"] = offset
    return result, reference
