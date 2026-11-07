"""Governed IERS Earth-orientation ingestion for canonical frame transforms."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sim.dynamics.orbit.frame_time import EarthOrientation, Epoch, TimeScale, epoch_julian_date

EOP_CONTRACT = "oel.iers-eop.v1"
_MJD_OFFSET = 2400000.5


class EopError(ValueError):
    """Raised when an EOP source cannot satisfy the governed public contract."""


@dataclass(frozen=True)
class EopRecord:
    mjd_utc: float
    xp_arcsec: float
    yp_arcsec: float
    dut1_s: float
    dx_mas: float = 0.0
    dy_mas: float = 0.0
    lod_ms: float | None = None
    quality: str = "observed"
    ddpsi_mas: float = 0.0
    ddeps_mas: float = 0.0
    dat_s: float | None = None


@dataclass(frozen=True)
class EopSeries:
    records: tuple[EopRecord, ...]
    source_format: str
    source_label: str
    source_sha256: str

    def __post_init__(self) -> None:
        if len(self.records) < 2:
            raise EopError("An EOP series requires at least two records for bounded interpolation.")
        for line_number, record in enumerate(self.records, start=1):
            _validate_record(record, line_number)
        mjd = [record.mjd_utc for record in self.records]
        if any(later <= earlier for earlier, later in zip(mjd, mjd[1:])):
            raise EopError("EOP records must have unique, strictly increasing MJD epochs.")
        if not self.source_label.strip() or not _is_sha256(self.source_sha256):
            raise EopError("EOP source provenance requires a label and lowercase SHA-256 digest.")

    @property
    def start_mjd_utc(self) -> float:
        return self.records[0].mjd_utc

    @property
    def stop_mjd_utc(self) -> float:
        return self.records[-1].mjd_utc

    def sample(self, epoch: Epoch) -> EarthOrientation:
        mjd = epoch_julian_date(epoch, TimeScale.UTC) - _MJD_OFFSET
        if mjd < self.start_mjd_utc or mjd > self.stop_mjd_utc:
            raise EopError(
                f"EOP epoch MJD {mjd:.9f} is outside source coverage "
                f"[{self.start_mjd_utc:.9f}, {self.stop_mjd_utc:.9f}]."
            )
        grid = np.asarray([record.mjd_utc for record in self.records], dtype=float)

        def interpolate(field: str) -> float:
            values = np.asarray([float(getattr(record, field)) for record in self.records], dtype=float)
            return float(np.interp(mjd, grid, values))

        mas_to_rad = math.pi / (180.0 * 3600.0 * 1000.0)
        return EarthOrientation(
            dut1_s=_interpolate_dut1(mjd, grid, self.records),
            xp_arcsec=interpolate("xp_arcsec"),
            yp_arcsec=interpolate("yp_arcsec"),
            dx_mas=interpolate("dx_mas"),
            dy_mas=interpolate("dy_mas"),
            ddpsi_rad=interpolate("ddpsi_mas") * mas_to_rad,
            ddeps_rad=interpolate("ddeps_mas") * mas_to_rad,
            source=f"{self.source_label} ({self.source_format}, linearly interpolated)",
            source_sha256=self.source_sha256,
        )

    def receipt(self, *, as_of: datetime | None = None, max_observed_age_days: float = 45.0) -> dict[str, Any]:
        maximum_age = float(max_observed_age_days)
        if not math.isfinite(maximum_age) or maximum_age < 0.0:
            raise EopError("max_observed_age_days must be finite and non-negative.")
        observed = [record for record in self.records if record.quality != "predicted"]
        last_observed = observed[-1] if observed else None
        payload: dict[str, Any] = {
            "schema": "oel.iers-eop-receipt.v1",
            "contract": EOP_CONTRACT,
            "source_format": self.source_format,
            "source_label": self.source_label,
            "source_sha256": self.source_sha256,
            "record_count": len(self.records),
            "coverage_mjd_utc": [self.start_mjd_utc, self.stop_mjd_utc],
            "last_observed_mjd_utc": None if last_observed is None else last_observed.mjd_utc,
            "interpolation": "linear; no extrapolation",
            "prediction_present": any(record.quality == "predicted" for record in self.records),
        }
        if as_of is not None:
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
            as_of_mjd = _datetime_to_mjd(as_of)
            eligible_observed = [record for record in observed if record.mjd_utc <= as_of_mjd]
            last_observed = eligible_observed[-1] if eligible_observed else None
            observed_age = math.inf if last_observed is None else as_of_mjd - last_observed.mjd_utc
            right_index = min(int(np.searchsorted(
                np.asarray([record.mjd_utc for record in self.records], dtype=float),
                as_of_mjd,
                side="left",
            )), len(self.records) - 1)
            left_index = max(0, right_index - (self.records[right_index].mjd_utc > as_of_mjd))
            prediction_used = any(
                self.records[index].quality == "predicted"
                for index in {left_index, right_index}
            )
            if as_of_mjd < self.start_mjd_utc:
                status = "not-yet-valid"
            elif as_of_mjd > self.stop_mjd_utc:
                status = "expired"
            elif last_observed is None:
                status = "prediction-only"
            elif observed_age > maximum_age:
                status = "stale"
            elif prediction_used:
                status = "prediction-only"
            else:
                status = "current"
            payload["last_observed_mjd_utc"] = None if last_observed is None else last_observed.mjd_utc
            payload["freshness"] = {
                "as_of_utc": as_of.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "as_of_mjd_utc": as_of_mjd,
                "max_observed_age_days": maximum_age,
                "observed_age_days": observed_age,
                "status": status,
            }
        return payload


def load_iers_eop(path: str | Path, *, source_format: str = "auto") -> EopSeries:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    selected = str(source_format or "auto").strip().lower()
    if selected == "auto":
        selected = "c04_csv" if "," in next((line for line in text.splitlines() if line.strip()), "") else "finals2000a"
    if selected == "finals2000a":
        records = tuple(_parse_finals2000a(text.splitlines()))
    elif selected == "celestrak-csv":
        return parse_celestrak_eop(text, source_label=source.name, source_sha256=digest)
    elif selected == "c04_csv":
        records = tuple(_parse_c04_csv(text))
    else:
        raise EopError("source_format must be auto, finals2000a, or c04_csv.")
    return EopSeries(records, selected, source.name, digest)


def parse_celestrak_eop(text: str, *, source_label: str, source_sha256: str) -> EopSeries:
    """CelesTrak CSV uses arcseconds and seconds; C04 uses different units."""
    reader = csv.DictReader(io.StringIO(text))
    fields = {"DATE", "MJD", "X", "Y", "UT1-UTC", "LOD", "DPSI", "DEPS", "DX", "DY", "DAT", "DATA_TYPE"}
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != fields:
        raise EopError("Expected the documented CelesTrak EOP CSV columns.")
    records = []
    for line, row in enumerate(reader, 2):
        try:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("row length")
            day = datetime.strptime(row["DATE"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            mjd = float(row["MJD"])
            if abs(mjd - _datetime_to_mjd(day)) > 1e-8 or row["DATA_TYPE"] not in {"O", "P"}:
                raise ValueError("date/quality")
            records.append(EopRecord(mjd, float(row["X"]), float(row["Y"]), float(row["UT1-UTC"]),
                dx_mas=1000 * float(row["DX"]), dy_mas=1000 * float(row["DY"]),
                lod_ms=1000 * float(row["LOD"]), quality="observed" if row["DATA_TYPE"] == "O" else "predicted",
                ddpsi_mas=1000 * float(row["DPSI"]), ddeps_mas=1000 * float(row["DEPS"]), dat_s=float(row["DAT"])))
        except (ValueError, TypeError) as exc:
            raise EopError(f"Invalid CelesTrak EOP row {line}.") from exc
    return EopSeries(tuple(records), "celestrak-csv", source_label, source_sha256)


def encode_eop_all(series: EopSeries) -> bytes:
    """Serialize explicit DAT/equinox fields for OEL's existing frame owner."""
    lines = ["# Derived from " + series.source_sha256, "NUM_OBSERVED_POINTS " + str(len(series.records))]
    for row in series.records:
        if row.dat_s is None:
            raise EopError("EOP-All export requires source-provided TAI-UTC; no leap-second guessing.")
        day = datetime.fromtimestamp((row.mjd_utc - 40587) * 86400, timezone.utc)
        values = [day.year, day.month, day.day, row.mjd_utc, row.xp_arcsec, row.yp_arcsec, row.dut1_s,
                  (row.lod_ms or 0) / 1000, row.ddpsi_mas / 1000, row.ddeps_mas / 1000,
                  row.dx_mas / 1000, row.dy_mas / 1000, row.dat_s]
        lines.append(" ".join(format(v, ".15g") for v in values))
    return ("\n".join(lines) + "\n").encode()


def audit_eop_series(
    series: EopSeries,
    *,
    as_of: datetime,
    max_observed_age_days: float = 45.0,
) -> dict[str, Any]:
    receipt = series.receipt(as_of=as_of, max_observed_age_days=max_observed_age_days)
    status = receipt["freshness"]["status"]
    return {
        "schema": "oel.iers-eop-audit.v1",
        "status": "pass" if status == "current" else "warning" if status == "prediction-only" else "fail",
        "eop": receipt,
        "non_claims": [
            "The audit checks source identity, coverage, and freshness; it does not certify EOP accuracy.",
            "Prediction use remains explicit and does not become observed data through interpolation.",
        ],
    }


def _parse_finals2000a(lines: Iterable[str]) -> Iterable[EopRecord]:
    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or len(line) < 68:
            continue
        try:
            mjd = float(line[7:15])
        except ValueError:
            continue
        bulletin_b = len(line) >= 185 and all(line[start:stop].strip() for start, stop in ((134, 144), (144, 154), (154, 165)))
        try:
            if bulletin_b:
                xp, yp, dut1 = float(line[134:144]), float(line[144:154]), float(line[154:165])
                ddpsi = _optional_float(line[165:175], 0.0)
                ddeps = _optional_float(line[175:185], 0.0)
                quality = "observed-final"
            else:
                xp, yp, dut1 = float(line[18:27]), float(line[37:46]), float(line[58:68])
                ddpsi = 0.0
                ddeps = 0.0
                flags = {line[index : index + 1] for index in (16, 57, 95)}
                quality = "predicted" if "P" in flags else "observed-rapid"
            dx = _optional_float(line[97:106], 0.0)
            dy = _optional_float(line[116:125], 0.0)
            lod = _optional_float(line[79:86], math.nan)
        except ValueError as exc:
            raise EopError(f"Invalid finals2000A numeric field on line {line_number}.") from exc
        record = EopRecord(
            mjd_utc=mjd,
            xp_arcsec=xp,
            yp_arcsec=yp,
            dut1_s=dut1,
            dx_mas=dx,
            dy_mas=dy,
            lod_ms=None if math.isnan(lod) else lod,
            quality=quality,
            ddpsi_mas=ddpsi,
            ddeps_mas=ddeps,
        )
        _validate_record(record, line_number)
        yield record


def _parse_c04_csv(text: str) -> Iterable[EopRecord]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise EopError("IERS C04 CSV has no header.")
    for line_number, row in enumerate(reader, start=2):
        normalized = {_column_key(key): str(value or "").strip() for key, value in row.items() if key is not None}
        try:
            record = EopRecord(
                mjd_utc=float(_pick(normalized, "mjd", "mjdutc")),
                xp_arcsec=float(_pick(normalized, "x", "xp", "xpole", "xarcsec")),
                yp_arcsec=float(_pick(normalized, "y", "yp", "ypole", "yarcsec")),
                dut1_s=float(_pick(normalized, "ut1utc", "dut1", "ut1minusutc")),
                dx_mas=float(_pick(normalized, "dx", "dxmas", default="0")),
                dy_mas=float(_pick(normalized, "dy", "dymas", default="0")),
                lod_ms=float(_pick(normalized, "lod", "lodms")) if _pick(normalized, "lod", "lodms", default="") else None,
                quality=_normalize_quality(_pick(normalized, "datatype", "quality", default="observed-final")),
            )
        except ValueError as exc:
            raise EopError(f"Invalid IERS C04 CSV numeric field on line {line_number}.") from exc
        _validate_record(record, line_number)
        yield record


def _validate_record(record: EopRecord, line_number: int) -> None:
    values = (
        record.mjd_utc,
        record.xp_arcsec,
        record.yp_arcsec,
        record.dut1_s,
        record.dx_mas,
        record.dy_mas,
        record.ddpsi_mas,
        record.ddeps_mas,
    )
    if not all(math.isfinite(value) for value in values):
        raise EopError(f"EOP line {line_number} contains non-finite values.")
    if abs(record.xp_arcsec) >= 10 or abs(record.yp_arcsec) >= 10 or abs(record.dut1_s) >= 2:
        raise EopError(f"EOP line {line_number} is outside bounded polar-motion/DUT1 limits.")
    if record.lod_ms is not None and not math.isfinite(float(record.lod_ms)):
        raise EopError(f"EOP line {line_number} contains a non-finite LOD value.")
    if record.dat_s is not None and (not math.isfinite(record.dat_s) or not 0 <= record.dat_s <= 100 or record.dat_s != int(record.dat_s)):
        raise EopError(f"EOP line {line_number} has invalid TAI-UTC.")
    if record.quality not in {"observed", "observed-final", "observed-rapid", "predicted"}:
        raise EopError(f"EOP line {line_number} has unsupported quality {record.quality!r}.")


def _interpolate_dut1(mjd: float, grid: np.ndarray, records: tuple[EopRecord, ...]) -> float:
    """Interpolate continuous UT1-TAI while preserving UTC leap discontinuities."""

    continuous: list[float] = []
    wraps: list[int] = []
    wrap = 0
    previous = float(records[0].dut1_s)
    continuous.append(previous)
    wraps.append(wrap)
    for record in records[1:]:
        current = float(record.dut1_s)
        delta = current - previous
        nearest_integer = int(round(delta))
        if abs(nearest_integer) == 1 and abs(delta - nearest_integer) <= 0.1:
            wrap -= nearest_integer
        continuous.append(current + wrap)
        wraps.append(wrap)
        previous = current
    interpolated = float(np.interp(mjd, grid, np.asarray(continuous, dtype=float)))
    record_index = int(np.searchsorted(grid, mjd, side="right") - 1)
    return interpolated - wraps[max(0, min(record_index, len(wraps) - 1))]


def _normalize_quality(value: str) -> str:
    key = str(value or "").strip().lower()
    aliases = {
        "observed": "observed",
        "observed-final": "observed-final",
        "final": "observed-final",
        "observed-rapid": "observed-rapid",
        "rapid": "observed-rapid",
        "predicted": "predicted",
        "prediction": "predicted",
        "p": "predicted",
    }
    if key not in aliases:
        raise EopError(f"Unsupported EOP quality {value!r}.")
    return aliases[key]


def _column_key(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _pick(row: dict[str, str], *keys: str, default: str | None = None) -> str:
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    if default is not None:
        return default
    raise EopError(f"IERS C04 CSV is missing one of the required columns: {keys}.")


def _optional_float(value: str, default: float) -> float:
    return default if not value.strip() else float(value)


def _datetime_to_mjd(value: datetime) -> float:
    unix_days = value.astimezone(timezone.utc).timestamp() / 86400.0
    return 40587.0 + unix_days


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "EOP_CONTRACT",
    "EopError",
    "EopRecord",
    "EopSeries",
    "audit_eop_series",
    "load_iers_eop",
]
