"""Source-bound ILRS center-of-mass corrections, separate from fitted biases."""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

from sim.dynamics.orbit.epoch import julian_date_to_datetime


class MissingCenterOfMassError(LookupError):
    """No table row covers this valid instrument request."""


class CenterOfMassTable:
    """One explicitly identified satellite table; no average/default fallback.

    Rows use the ILRS model's inclusive dates and wavelength bands. Ambiguous
    hardware transitions raise rather than selecting an arbitrary last row.
    A compatible ILRS data-handling model remains the caller's responsibility.
    """

    def __init__(self, path, *, satellite_ilrs_id, expected_sha256, source_url):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != expected_sha256 or len(expected_sha256) != 64:
            raise ValueError("Center-of-mass table SHA-256 mismatch.")
        if not str(satellite_ilrs_id).isdigit() or not str(source_url).startswith("https://"):
            raise ValueError("Center-of-mass table requires satellite identity and source URL.")
        self.identity = dict(satellite_ilrs_id=str(satellite_ilrs_id), sha256=self.sha256, source_url=str(source_url))
        self.entries = []
        for number, line in enumerate(raw.decode("utf8").splitlines(), start=1):
            if not line.strip() or line.startswith("*"):
                continue
            fields = line.split()
            if len(fields) != 9:
                raise ValueError(f"Invalid center-of-mass table row {number}.")
            pad = int(fields[0])
            start = date(*[int(fields[i]) for i in (3, 2, 1)])
            end = date(*[int(fields[i]) for i in (6, 5, 4)])
            wave, value = float(fields[7]), float(fields[8])
            if start > end or not math.isfinite(wave) or wave <= 0 or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid center-of-mass table values at row {number}.")
            self.entries.append((pad, start, end, wave, value, number))
        if not self.entries:
            raise ValueError("Empty center-of-mass table.")

    def correction(self, *, satellite_ilrs_id, pad_id, observation_date, wavelength_nm):
        if str(satellite_ilrs_id) != self.identity["satellite_ilrs_id"]:
            raise ValueError("Center-of-mass table satellite mismatch.")
        wave = float(wavelength_nm)
        if not math.isfinite(wave) or wave <= 0:
            raise ValueError("Center-of-mass wavelength must be finite and positive.")
        selected_wave = 694.0 if 600.0 <= wave <= 700.0 else 1064.0 if wave >= 1000.0 else wave
        matches = [
            entry
            for entry in self.entries
            if entry[0] == int(pad_id)
            and entry[1] <= observation_date <= entry[2]
            and abs(entry[3] - selected_wave) <= 50.0
        ]
        if not matches:
            raise MissingCenterOfMassError(
                "No authoritative center-of-mass correction for satellite, pad, date and wavelength."
            )
        if len(matches) != 1:
            raise ValueError("Ambiguous center-of-mass hardware/date/wavelength interval.")
        entry = matches[0]
        return dict(
            **self.identity,
            table_line=entry[5],
            pad_id=int(pad_id),
            wavelength_nm=wave,
            table_wavelength_nm=entry[3],
            start_date=entry[1].isoformat(),
            end_date=entry[2].isoformat(),
            correction_mm=entry[4],
            prediction_offset_km=-entry[4] / 1.0e6,
            status="modeled",
        )


def attach_reflector_wavelengths(rows, packets):
    """Bind CoM wavelength from CRD C0 even when media is already corrected."""
    by_source = {}
    for packet in packets:
        key = str(packet["source_sha256"])
        if key in by_source:
            raise ValueError("Duplicate reflector source identity.")
        by_source[key] = packet
    result = deepcopy(list(rows))
    for row in result:
        if row["correction_flags"].get("center_of_mass_applied", False):
            continue
        packet = by_source[str(row["source_sha256"])]
        _, si, pi = str(row["measurement_id"]).split(":")
        session = packet["sessions"][int(si)]
        point = session["normal_points"][int(pi)]
        if (
            session["station"]["code"].upper() != row["station_id"]
            or float(point["time_jd_utc"]) != float(row["time_jd_utc"])
            or str(session["target"]["ilrs_id"]) != str(row["target_metadata"]["ilrs_id"])
        ):
            raise ValueError("Reflector CRD observation identity mismatch.")
        wavelengths = {
            float(record["raw"].split()[2])
            for record in session["configuration_records"]
            if record["record_type"] == "C0" and record["raw"].split()[3] == point["system_configuration_id"]
        }
        if len(wavelengths) != 1:
            raise ValueError("Reflector correction requires one matching CRD C0 wavelength.")
        wave = wavelengths.pop()
        if not math.isfinite(wave) or wave <= 0.0:
            raise ValueError("Invalid reflector CRD wavelength.")
        if "slr_reflector_wavelength_nm" in row and float(row["slr_reflector_wavelength_nm"]) != wave:
            raise ValueError("Conflicting reflector wavelength.")
        row["slr_reflector_wavelength_nm"] = wave
    return result


def apply_center_of_mass(rows, *, table, unresolved_station_ids=()):
    """Copy rows and attach a physical prediction offset; leave observations raw.

    Missing coverage fails by default. An explicit development-only exception
    records an unresolved station; it confers no station calibration or freeze
    eligibility. A fitted residual bias is still a distinct estimated parameter.
    """
    allowed = set(map(str, unresolved_station_ids))
    result = deepcopy(list(rows))
    for row in result:
        if "slr_center_of_mass" in row:
            raise ValueError("SLR center-of-mass model is already attached.")
        if str(row["target_metadata"]["ilrs_id"]) != table.identity["satellite_ilrs_id"]:
            raise ValueError("Center-of-mass table satellite mismatch.")
        if row["correction_flags"].get("center_of_mass_applied", False):
            row["slr_center_of_mass"] = dict(**table.identity, status="already_applied", prediction_offset_km=0.0)
            continue
        try:
            day = (
                julian_date_to_datetime(float(row["time_jd_utc"]))
                + timedelta(seconds=float(row.get("epoch_offset_s", 0.0)))
            ).date()
            value = table.correction(
                satellite_ilrs_id=row["target_metadata"]["ilrs_id"],
                pad_id=row["station_metadata"]["cdp_pad_id"],
                observation_date=day,
                wavelength_nm=row.get("slr_reflector_wavelength_nm", row.get("slr_media", {}).get("wavelength_nm")),
            )
        except MissingCenterOfMassError:
            if str(row["station_id"]) not in allowed:
                raise
            value = dict(**table.identity, status="unresolved_development_only", prediction_offset_km=0.0)
        row["slr_center_of_mass"] = value
    return result


def center_of_mass_prediction_offset(row):
    """Validate physical correction status before scalar or native prediction."""
    value = row.get("slr_center_of_mass")
    if value is None:
        return 0.0
    offset = float(value["prediction_offset_km"])
    status = value["status"]
    if (
        not math.isfinite(offset)
        or status not in {"modeled", "already_applied", "unresolved_development_only"}
        or (status == "modeled" and (offset >= 0.0 or row["correction_flags"].get("center_of_mass_applied", False)))
        or (status != "modeled" and offset != 0.0)
    ):
        raise ValueError("Invalid or double-applied center-of-mass prediction offset.")
    return offset
