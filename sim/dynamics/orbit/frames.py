from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.epoch import gmst_angle_rad_from_jd

_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)
_MJD0 = 2400000.5
_J2000 = 2451545.0
_DAYSEC = 86400.0
_JULIAN_CENTURY_DAYS = 36525.0
_DEFAULT_TT_MINUS_UTC_S = 69.184

FRAME_MODEL_SIMPLE_GMST = "simple_gmst"
FRAME_MODEL_IAU76_80_EOP = "iau76_80_eop"

_FRAME_MODEL_ALIASES = {
    "": FRAME_MODEL_SIMPLE_GMST,
    "simple": FRAME_MODEL_SIMPLE_GMST,
    "simple_gmst": FRAME_MODEL_SIMPLE_GMST,
    "simple_earth_rotation": FRAME_MODEL_SIMPLE_GMST,
    "gmst": FRAME_MODEL_SIMPLE_GMST,
    "inertial_z": FRAME_MODEL_SIMPLE_GMST,
    "hpop_like": FRAME_MODEL_IAU76_80_EOP,
    "hpop": FRAME_MODEL_IAU76_80_EOP,
    "iau76_80_eop": FRAME_MODEL_IAU76_80_EOP,
    "iau76_fk5_iau80_eop": FRAME_MODEL_IAU76_80_EOP,
}


@dataclass(frozen=True)
class FrameContext:
    """Scenario-level frame and time-scale provenance for Earth-fixed transforms."""

    model: str = FRAME_MODEL_SIMPLE_GMST
    jd_utc_start: float | None = None
    eop_path: str | None = None
    eop_extrapolation: str = "error"
    time_scale_model: str = "utc_only"
    tt_minus_utc_s: float = _DEFAULT_TT_MINUS_UTC_S
    dut1_s: float | None = None
    xp_arcsec: float | None = None
    yp_arcsec: float | None = None
    dat_s: float | None = None
    ddpsi_rad: float = 0.0
    ddeps_rad: float = 0.0
    source: str = "scenario"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", normalize_frame_model(self.model))
        if self.eop_path in ("",):
            object.__setattr__(self, "eop_path", None)
        policy = str(self.eop_extrapolation or "error").strip().lower()
        if policy not in {"error", "hold"}:
            raise ValueError("eop_extrapolation must be 'error' or 'hold'.")
        object.__setattr__(self, "eop_extrapolation", policy)

    @property
    def legacy_frame_model(self) -> str:
        return "hpop_like" if self.model == FRAME_MODEL_IAU76_80_EOP else "simple"

    @property
    def eop_rotation_available(self) -> bool:
        return bool(
            self.model == FRAME_MODEL_IAU76_80_EOP
            and self.jd_utc_start is not None
            and (self.eop_path or self.has_manual_eop)
        )

    @property
    def has_manual_eop(self) -> bool:
        return bool(
            self.dut1_s is not None
            or self.xp_arcsec is not None
            or self.yp_arcsec is not None
            or self.dat_s is not None
            or float(self.ddpsi_rad) != 0.0
            or float(self.ddeps_rad) != 0.0
        )

    def at(self, t_s: float) -> FrameContext:
        if self.jd_utc_start is None or self.model != FRAME_MODEL_IAU76_80_EOP or not self.eop_path:
            return self
        _validate_eop_elapsed_interval(self.jd_utc_start, t_s, self.eop_path)
        jd_utc = float(self.jd_utc_start) + float(t_s) / _DAYSEC
        xp_arcsec, yp_arcsec, dut1_s, dat_s = _interp_eop(
            jd_utc - _MJD0,
            self.eop_path,
            extrapolation=self.eop_extrapolation,
        )
        return FrameContext(
            model=self.model,
            jd_utc_start=self.jd_utc_start,
            eop_path=self.eop_path,
            eop_extrapolation=self.eop_extrapolation,
            time_scale_model="eop_utc_ut1_tt",
            tt_minus_utc_s=float(dat_s) + 32.184,
            dut1_s=float(dut1_s),
            xp_arcsec=float(xp_arcsec),
            yp_arcsec=float(yp_arcsec),
            dat_s=float(dat_s),
            ddpsi_rad=self.ddpsi_rad,
            ddeps_rad=self.ddeps_rad,
            source=self.source,
        )

    def metadata(self, *, sample_t_s: float = 0.0) -> dict[str, Any]:
        sampled = self.at(sample_t_s)
        data = asdict(sampled)
        data["legacy_frame_model"] = sampled.legacy_frame_model
        data["sample_t_s"] = float(sample_t_s)
        data["polar_motion_applied"] = sampled.eop_rotation_available
        if sampled.eop_rotation_available and sampled.eop_path:
            data["eop_table_sha256"] = _eop_table_sha256(sampled.eop_path)
        data["nutation_corrections_applied"] = bool(
            sampled.eop_rotation_available and (float(sampled.ddpsi_rad) != 0.0 or float(sampled.ddeps_rad) != 0.0)
        )
        if sampled.eop_rotation_available and sampled.eop_path:
            dp, de = _interp_eop_nutation(
                float(sampled.jd_utc_start) + float(sample_t_s) / _DAYSEC - _MJD0,
                sampled.eop_path, extrapolation=sampled.eop_extrapolation,
            )
            data["effective_ddpsi_rad"] = sampled.ddpsi_rad + dp
            data["effective_ddeps_rad"] = sampled.ddeps_rad + de
            data["nutation_corrections_applied"] = bool(
                data["effective_ddpsi_rad"] != 0.0 or data["effective_ddeps_rad"] != 0.0
            )
        return data


def _eop_file_signature(path_value: str) -> tuple[int, int, int, int]:
    """Return metadata that changes when an EOP file is replaced or rewritten."""
    # stat follows symlinks and preserves inode/mtime invalidation without
    # repeating realpath resolution at every integrator stage.
    stat = os.stat(os.path.expanduser(path_value))
    return (int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))


@lru_cache(maxsize=16)
def _eop_table_sha256_cached(path_value: str, signature: tuple[int, int, int, int]) -> str:
    digest = hashlib.sha256()
    for array in _load_eop_records_cached(path_value, signature):
        normalized = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(normalized.dtype.str.encode("ascii"))
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _eop_table_sha256(path_value: str) -> str:
    resolved = str(Path(path_value).expanduser().resolve())
    return _eop_table_sha256_cached(resolved, _eop_file_signature(resolved))


def normalize_frame_model(model: Any) -> str:
    key = str(model or "").strip().lower().replace("-", "_")
    if key in _FRAME_MODEL_ALIASES:
        return _FRAME_MODEL_ALIASES[key]
    choices = ", ".join(sorted(set(_FRAME_MODEL_ALIASES) - {""}))
    raise ValueError(f"Unsupported frame model {model!r}; expected one of: {choices}.")


def _resolve_frame_path(raw_path: Any) -> str | None:
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    return str(path.resolve())


def frame_context_from_mapping(
    frames: dict[str, Any] | None,
    *,
    jd_utc_start: float | None = None,
    source: str = "scenario",
) -> FrameContext:
    data = dict(frames or {})
    if jd_utc_start is None and data.get("jd_utc_start") is not None:
        jd_utc_start = float(data["jd_utc_start"])
    model = normalize_frame_model(data.get("model", data.get("frame_model", FRAME_MODEL_SIMPLE_GMST)))
    eop_path = _resolve_frame_path(data.get("eop_path"))
    tt_minus_raw = data.get("tt_minus_utc_s", _DEFAULT_TT_MINUS_UTC_S)
    tt_minus_utc_s = _DEFAULT_TT_MINUS_UTC_S if tt_minus_raw is None else float(tt_minus_raw)
    if data.get("dat_s") is not None:
        tt_minus_utc_s = float(data["dat_s"]) + 32.184
    return FrameContext(
        model=model,
        jd_utc_start=jd_utc_start,
        eop_path=eop_path,
        eop_extrapolation=str(data.get("eop_extrapolation", "error") or "error"),
        time_scale_model=str(
            data.get(
                "time_scale_model",
                "eop_utc_ut1_tt"
                if model == FRAME_MODEL_IAU76_80_EOP
                and (
                    eop_path
                    or any(
                        data.get(key) is not None
                        for key in ("dut1_s", "xp_arcsec", "yp_arcsec", "dat_s")
                    )
                    or float(data.get("ddpsi_rad", 0.0) or 0.0) != 0.0
                    or float(data.get("ddeps_rad", 0.0) or 0.0) != 0.0
                )
                else "utc_only",
            )
        ),
        tt_minus_utc_s=tt_minus_utc_s,
        dut1_s=None if data.get("dut1_s") is None else float(data["dut1_s"]),
        xp_arcsec=None if data.get("xp_arcsec") is None else float(data["xp_arcsec"]),
        yp_arcsec=None if data.get("yp_arcsec") is None else float(data["yp_arcsec"]),
        dat_s=None if data.get("dat_s") is None else float(data["dat_s"]),
        ddpsi_rad=float(data.get("ddpsi_rad", 0.0) or 0.0),
        ddeps_rad=float(data.get("ddeps_rad", 0.0) or 0.0),
        source=source,
    )


def frame_context_from_environment(env: dict[str, Any] | None) -> FrameContext:
    data = dict(env or {})
    model = data.get("frame_model", data.get("spherical_harmonics_frame_model", data.get("drag_frame_model", "simple")))
    eop_path = data.get("eop_path", data.get("spherical_harmonics_eop_path", data.get("drag_eop_path")))
    frames = {
        "model": model,
        "eop_path": eop_path,
        "eop_extrapolation": data.get("eop_extrapolation", "error"),
        "tt_minus_utc_s": data.get("tt_minus_utc_s", _DEFAULT_TT_MINUS_UTC_S),
        "dut1_s": data.get("dut1_s"),
        "xp_arcsec": data.get("xp_arcsec"),
        "yp_arcsec": data.get("yp_arcsec"),
        "dat_s": data.get("dat_s"),
        "ddpsi_rad": data.get("ddpsi_rad", 0.0),
        "ddeps_rad": data.get("ddeps_rad", 0.0),
    }
    return frame_context_from_mapping(
        frames,
        jd_utc_start=None if data.get("jd_utc_start") is None else float(data["jd_utc_start"]),
        source="environment",
    )


@lru_cache(maxsize=8192)
def _eci_to_ecef_rotation_components(
    t_s: float,
    jd_utc_start: float | None,
) -> tuple[np.floating, np.floating]:
    if jd_utc_start is None:
        theta = EARTH_ROT_RATE_RAD_S * t_s
    else:
        theta = gmst_angle_rad_from_jd(float(jd_utc_start) + float(t_s) / 86400.0)
    return np.cos(theta), np.sin(theta)


def eci_to_ecef_rotation(t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    c, s = _eci_to_ecef_rotation_components(
        float(t_s),
        None if jd_utc_start is None else float(jd_utc_start),
    )
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def eci_to_ecef(r_eci_km: np.ndarray, t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start) @ r_eci_km


def ecef_to_eci(r_ecef_km: np.ndarray, t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start).T @ r_ecef_km


def _rx(angle_rad: float) -> np.ndarray:
    s = math.sin(float(angle_rad))
    c = math.cos(float(angle_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=float)


def _ry(angle_rad: float) -> np.ndarray:
    s = math.sin(float(angle_rad))
    c = math.cos(float(angle_rad))
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=float)


def _rz(angle_rad: float) -> np.ndarray:
    s = math.sin(float(angle_rad))
    c = math.cos(float(angle_rad))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


@lru_cache(maxsize=4)
def _load_eop_records_cached(
    eop_path: str,
    signature: tuple[int, int, int, int],
) -> tuple[np.ndarray, ...]:
    mjd = []
    xp_arcsec = []
    yp_arcsec = []
    dut1_s = []
    dat_s = []
    ddpsi_rad = []
    ddeps_rad = []
    with Path(eop_path).expanduser().resolve().open("r", encoding="utf-8", errors="ignore") as f:
        in_data = False
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("NUM_OBSERVED_POINTS") or line.startswith("NUM_PREDICTED_POINTS"):
                in_data = True
                continue
            if not in_data or line.startswith("#") or line.startswith("VERSION") or line.startswith("UPDATED"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                # EOP-All equinox corrections are in arcseconds (columns 8, 9
                # after LOD), not the later celestial-pole dX/dY columns.
                values = [float(parts[i]) for i in (3, 4, 5, 6, 12, 8, 9)]
            except ValueError:
                continue
            mjd.append(values[0])
            xp_arcsec.append(values[1])
            yp_arcsec.append(values[2])
            dut1_s.append(values[3])
            dat_s.append(values[4])
            ddpsi_rad.append(values[5] * _ARCSEC_TO_RAD)
            ddeps_rad.append(values[6] * _ARCSEC_TO_RAD)
    if not mjd:
        raise ValueError(f"No EOP records parsed from {eop_path}")
    arrays = (
        np.array(mjd, dtype=float),
        np.array(xp_arcsec, dtype=float),
        np.array(yp_arcsec, dtype=float),
        np.array(dut1_s, dtype=float),
        np.array(dat_s, dtype=float),
        np.array(ddpsi_rad, dtype=float),
        np.array(ddeps_rad, dtype=float),
    )
    # Cached EOP records are shared by every object in a campaign.  Prevent a
    # caller from accidentally mutating the process-wide cache while another
    # object is interpolating the same table.
    for array in arrays:
        array.setflags(write=False)
    return arrays


@lru_cache(maxsize=4)
def _load_eop_table_cached(eop_path: str, signature: tuple[int, int, int, int]):
    """Keep the existing five-column interpolation contract."""
    return _load_eop_records_cached(eop_path, signature)[:5]


def _interp_eop_nutation(mjd_utc: float, eop_path: str, *, extrapolation: str = "error"):
    path = _eop_cache_path(eop_path)
    return _interp_eop_nutation_cached(mjd_utc, path, _eop_file_signature(path), extrapolation=extrapolation)


@lru_cache(maxsize=8192)
def _interp_eop_nutation_cached(mjd_utc, eop_path, signature, *, extrapolation="error"):
    # Share coverage and leap-second guards with the established EOP path.
    _interp_eop_cached(mjd_utc, eop_path, signature, extrapolation=extrapolation)
    records = _load_eop_records_cached(eop_path, signature)
    return tuple(float(np.interp(mjd_utc, records[0], column)) for column in records[5:])


def _load_eop_table(eop_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resolved = _eop_cache_path(eop_path)
    return _load_eop_table_cached(resolved, _eop_file_signature(resolved))


def _load_eop_stage_table(eop_path: str) -> tuple[np.ndarray, ...]:
    resolved = _eop_cache_path(eop_path)
    return _load_eop_records_cached(resolved, _eop_file_signature(resolved))


@lru_cache(maxsize=16)
def _eop_dat_change_intervals_cached(
    eop_path: str,
    signature: tuple[int, int, int, int],
) -> tuple[tuple[float, float], ...]:
    """Return EOP intervals over which DAT changes between records."""

    mjd, _xp_arcsec, _yp_arcsec, _dut1_s, dat_s = _load_eop_table_cached(eop_path, signature)
    if mjd.size < 2:
        return ()
    changes = np.flatnonzero(
        np.isfinite(dat_s[:-1])
        & np.isfinite(dat_s[1:])
        & (np.abs(np.diff(dat_s)) > 1.0e-9)
    )
    return tuple((float(mjd[index]), float(mjd[index + 1])) for index in changes)


def _eop_dat_change_intervals(eop_path: str) -> tuple[tuple[float, float], ...]:
    resolved = _eop_cache_path(eop_path)
    return _eop_dat_change_intervals_cached(resolved, _eop_file_signature(resolved))


def _validate_eop_elapsed_interval(jd_utc_start: float, t_s: float, eop_path: str) -> None:
    """Reject EOP-backed elapsed-time requests that cross a DAT step."""

    elapsed = float(t_s)
    if elapsed == 0.0:
        return
    start_mjd = float(jd_utc_start) - _MJD0
    end_mjd = start_mjd + elapsed / _DAYSEC
    lower, upper = sorted((start_mjd, end_mjd))
    for _left, right in _eop_dat_change_intervals(eop_path):
        # Include the right endpoint: 86400 SI seconds from the start of a
        # positive leap-second day still lands one second before the next
        # UTC midnight, so mapping it to the DAT transition is unsafe.
        if lower < right <= upper:
            raise ValueError(
                "EOP-backed UTC elapsed time crosses a DAT leap-second boundary "
                f"near MJD {right:.9f}; the current fixed-86400-second UTC "
                "conversion is not leap-second aware."
            )


@lru_cache(maxsize=8192)
def _interp_eop_cached(
    mjd_utc: float,
    eop_path: str,
    signature: tuple[int, int, int, int],
    *,
    extrapolation: str = "error",
) -> tuple[float, float, float, float]:
    mjd, xp_arcsec, yp_arcsec, dut1_s, dat_s = _load_eop_table_cached(eop_path, signature)
    x = float(mjd_utc)
    policy = str(extrapolation or "error").strip().lower()
    if policy not in {"error", "hold"}:
        raise ValueError("EOP extrapolation policy must be 'error' or 'hold'.")
    if (x < float(mjd[0]) or x > float(mjd[-1])) and policy == "error":
        raise ValueError(
            f"Requested MJD {x:.9f} is outside EOP coverage "
            f"[{float(mjd[0]):.9f}, {float(mjd[-1]):.9f}] for {eop_path}. "
            "Set eop_extrapolation='hold' only when endpoint holding is intentional."
        )
    for left, right in _eop_dat_change_intervals_cached(eop_path, signature):
        if left < x < right:
            raise ValueError(
                "EOP DAT changes within the requested interpolation bracket "
                f"[{left:.9f}, {right:.9f}]; linear DAT interpolation is "
                "unsafe across a leap-second boundary."
            )
    xp = float(np.interp(x, mjd, xp_arcsec))
    yp = float(np.interp(x, mjd, yp_arcsec))
    dut1 = float(np.interp(x, mjd, dut1_s))
    dat = float(np.interp(x, mjd, dat_s))
    return xp, yp, dut1, dat


def _interp_eop(
    mjd_utc: float,
    eop_path: str,
    *,
    extrapolation: str = "error",
) -> tuple[float, float, float, float]:
    resolved = _eop_cache_path(eop_path)
    return _interp_eop_cached(
        float(mjd_utc),
        resolved,
        _eop_file_signature(resolved),
        extrapolation=extrapolation,
    )


def _precession_iau1976_matrix(jd_tt: float) -> np.ndarray:
    t = (float(jd_tt) - _J2000) / _JULIAN_CENTURY_DAYS
    zeta = (2306.2181 * t + 0.30188 * (t**2) + 0.017998 * (t**3)) * _ARCSEC_TO_RAD
    theta = (2004.3109 * t - 0.42665 * (t**2) - 0.041833 * (t**3)) * _ARCSEC_TO_RAD
    z = (2306.2181 * t + 1.09468 * (t**2) + 0.018203 * (t**3)) * _ARCSEC_TO_RAD
    return _rz(-z) @ _ry(theta) @ _rz(-zeta)


@lru_cache(maxsize=1)
def _load_nut80_table() -> tuple[np.ndarray, np.ndarray]:
    path = Path(__file__).with_name("data") / "nut80.dat"
    raw = np.loadtxt(path, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 9:
        raise ValueError(f"Invalid IAU-80 nutation table at {path}")
    coeffs = raw[:, :5].astype(int)
    terms = raw[:, 5:9] * (0.0001 * _ARCSEC_TO_RAD)
    return coeffs, terms


def _fundamental_args_iau1980(ttt: float) -> tuple[float, float, float, float, float]:
    t = float(ttt)
    args_deg = (
        (((0.064 * t + 31.310) * t + 1717915922.6330) * t) / 3600.0 + 134.96298139,
        (((-0.012 * t - 0.577) * t + 129596581.2240) * t) / 3600.0 + 357.52772333,
        (((0.011 * t - 13.257) * t + 1739527263.1370) * t) / 3600.0 + 93.27191028,
        (((0.019 * t - 6.891) * t + 1602961601.3280) * t) / 3600.0 + 297.85036306,
        (((0.008 * t + 7.455) * t - 6962890.5390) * t) / 3600.0 + 125.04452222,
    )
    return tuple(math.radians(arg % 360.0) for arg in args_deg)


def _precession_iau1976_vallado_matrix(jd_tt: float) -> np.ndarray:
    ttt = (float(jd_tt) - _J2000) / _JULIAN_CENTURY_DAYS
    ttt2 = ttt * ttt
    ttt3 = ttt2 * ttt
    zeta = (2306.2181 * ttt + 0.30188 * ttt2 + 0.017998 * ttt3) * _ARCSEC_TO_RAD
    theta = (2004.3109 * ttt - 0.42665 * ttt2 - 0.041833 * ttt3) * _ARCSEC_TO_RAD
    z = (2306.2181 * ttt + 1.09468 * ttt2 + 0.018203 * ttt3) * _ARCSEC_TO_RAD
    coszeta = math.cos(zeta)
    sinzeta = math.sin(zeta)
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    cosz = math.cos(z)
    sinz = math.sin(z)
    return np.array(
        [
            [
                coszeta * costheta * cosz - sinzeta * sinz,
                coszeta * costheta * sinz + sinzeta * cosz,
                coszeta * sintheta,
            ],
            [
                -sinzeta * costheta * cosz - coszeta * sinz,
                -sinzeta * costheta * sinz + coszeta * cosz,
                -sinzeta * sintheta,
            ],
            [-sintheta * cosz, -sintheta * sinz, costheta],
        ],
        dtype=float,
    )


def _nutation_iau1980_vallado_matrix(
    jd_tt: float,
    *,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> tuple[float, float, float, float, np.ndarray]:
    ttt = (float(jd_tt) - _J2000) / _JULIAN_CENTURY_DAYS
    ttt2 = ttt * ttt
    ttt3 = ttt2 * ttt
    meaneps = ((-46.8150 * ttt - 0.00059 * ttt2 + 0.001813 * ttt3 + 84381.448) / 3600.0) % 360.0
    meaneps_rad = math.radians(meaneps)
    args = np.array(_fundamental_args_iau1980(ttt), dtype=float)
    coeffs, terms = _load_nut80_table()
    phase = coeffs @ args
    deltapsi = float(np.sum((terms[:, 0] + terms[:, 1] * ttt) * np.sin(phase)))
    deltaeps = float(np.sum((terms[:, 2] + terms[:, 3] * ttt) * np.cos(phase)))
    deltapsi = float(math.fmod(deltapsi + float(ddpsi_rad), 2.0 * math.pi))
    deltaeps = float(math.fmod(deltaeps + float(ddeps_rad), 2.0 * math.pi))
    trueeps = meaneps_rad + deltaeps
    omega = float(args[4])

    cospsi = math.cos(deltapsi)
    sinpsi = math.sin(deltapsi)
    coseps = math.cos(meaneps_rad)
    sineps = math.sin(meaneps_rad)
    costrueeps = math.cos(trueeps)
    sintrueeps = math.sin(trueeps)
    nut = np.array(
        [
            [cospsi, costrueeps * sinpsi, sintrueeps * sinpsi],
            [
                -coseps * sinpsi,
                costrueeps * coseps * cospsi + sintrueeps * sineps,
                sintrueeps * coseps * cospsi - sineps * costrueeps,
            ],
            [
                -sineps * sinpsi,
                costrueeps * sineps * cospsi - sintrueeps * coseps,
                sintrueeps * sineps * cospsi + costrueeps * coseps,
            ],
        ],
        dtype=float,
    )
    return deltapsi, trueeps, meaneps_rad, omega, nut


@lru_cache(maxsize=8192)
def teme_to_eci_matrix_vallado_iau80(
    jd_utc: float,
    *,
    tt_minus_utc_s: float = _DEFAULT_TT_MINUS_UTC_S,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> np.ndarray:
    """Return Vallado IAU-76/FK5 + IAU-80 TEME-to-ECI rotation.

    The implementation mirrors the MATLAB SGP4 package's ``teme2eci.m`` path.
    The default TT-UTC value matches 2024-era TLE validation cases; callers with
    EOP data can pass a different value and nutation corrections explicitly.

    Cached arrays are backed by immutable ``bytes`` rather than owning mutable
    NumPy storage. A caller therefore cannot make the shared value writeable and
    poison later cache hits.
    """
    jd_tt = float(jd_utc) + float(tt_minus_utc_s) / _DAYSEC
    prec = _precession_iau1976_vallado_matrix(jd_tt)
    deltapsi, _trueeps, meaneps, _omega, nut = _nutation_iau1980_vallado_matrix(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    eqeg = math.fmod(deltapsi * math.cos(meaneps), 2.0 * math.pi)
    eqe = np.array(
        [
            [math.cos(eqeg), math.sin(eqeg), 0.0],
            [-math.sin(eqeg), math.cos(eqeg), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rotation = prec @ nut @ eqe.T
    return np.frombuffer(rotation.tobytes(order="C"), dtype=np.float64).reshape((3, 3))


def teme_to_eci_vallado_iau80(
    position_teme_km: np.ndarray,
    velocity_teme_km_s: np.ndarray,
    *,
    jd_utc: float,
    tt_minus_utc_s: float = _DEFAULT_TT_MINUS_UTC_S,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rot = teme_to_eci_matrix_vallado_iau80(
        jd_utc,
        tt_minus_utc_s=tt_minus_utc_s,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    return rot @ np.array(position_teme_km, dtype=float), rot @ np.array(velocity_teme_km_s, dtype=float)


def _precession_nutation_matrix_iau76_80(
    jd_tt: float,
    *,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    dpsi, true_eps, mean_eps, _omega, _nut_vallado = _nutation_iau1980_vallado_matrix(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    precession = _precession_iau1976_matrix(jd_tt)
    nutation = _rx(-true_eps) @ _rz(-dpsi) @ _rx(mean_eps)
    # This explicit Vallado rotation is the transpose of the matrix returned
    # by ``_nutation_iau1980_vallado_matrix`` (which is documented as TOD ->
    # MOD).  It therefore maps MOD -> TOD; compose it with J2000 -> MOD
    # precession to obtain the J2000 -> TOD celestial factor.
    return nutation @ precession, dpsi, true_eps


@lru_cache(maxsize=8192)
def _precession_nutation_matrix_approx(
    jd_tt: float,
    *,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Compatibility alias for the full IAU-76/FK5 + IAU-80 reduction.

    The returned matrix is treated as immutable by the internal frame consumers.
    Caching lets density/local-solar-time and body-fixed gravity share the exact
    same epoch reduction when both are active.
    """

    return _precession_nutation_matrix_iau76_80(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )


def _gmst_iau1982_split(jd_utc_start: float, elapsed_s: float, dut1_s: float) -> float:
    """IAU 1982 GMST with UTC epoch and elapsed/UT1 seconds kept separate.

    IERS 1996, p. 21: evaluate GMST-UT1 in centuries and add only the
    fractional UT1 day. Adding DUT1 or a light-time interval to a large JD
    first needlessly quantizes Earth rotation at the centimeter scale.
    """
    offset_days = (float(elapsed_s) + float(dut1_s)) / _DAYSEC
    centuries = ((float(jd_utc_start) - _J2000) + offset_days) / _JULIAN_CENTURY_DAYS
    fractional_seconds = _DAYSEC * (math.fmod(float(jd_utc_start), 1.0) + math.fmod(offset_days, 1.0))
    gmst_seconds = (24110.54841 - _DAYSEC / 2.0
                    + (8640184.812866 + (0.093104 - 6.2e-6 * centuries) * centuries) * centuries
                    + fractional_seconds)
    return float((gmst_seconds * (2.0 * math.pi / _DAYSEC)) % (2.0 * math.pi))


def _equation_of_equinoxes_iau1994(jd_tt: float, dpsi_rad: float) -> float:
    """IAU 1980 nutation with the IAU 1994 resolution C7 complementary terms.

    The longitude term uses mean obliquity. The two complementary terms
    become active at 1997-02-27 00:00 UTC (IERS 1996/Orekit convention).
    """
    centuries = (float(jd_tt) - _J2000) / _JULIAN_CENTURY_DAYS
    mean_eps = (84381.448 + (-46.8150 + (-0.00059 + 0.001813 * centuries) * centuries) * centuries) * _ARCSEC_TO_RAD
    complementary = 0.0
    if float(jd_tt) >= 2450506.5 + 62.184 / _DAYSEC:
        omega = _fundamental_args_iau1980(centuries)[4]
        complementary = (0.00264 * math.sin(omega) + 0.000063 * math.sin(2.0 * omega)) * _ARCSEC_TO_RAD
    return float(dpsi_rad) * math.cos(mean_eps) + complementary


def apparent_sidereal_time_hpop_like(
    jd_utc: float,
    eop_path: str | None = None,
    *,
    dut1_s: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
) -> float:
    has_manual_eop = any(value is not None for value in (dut1_s, dat_s, tt_minus_utc_s)) or (
        float(ddpsi_rad) != 0.0 or float(ddeps_rad) != 0.0
    )
    if not eop_path and not has_manual_eop:
        return gmst_angle_rad_from_jd(float(jd_utc))
    if eop_path:
        mjd_utc = float(jd_utc) - _MJD0
        _xp_arcsec, _yp_arcsec, dut1_s, dat_s = _interp_eop(
            mjd_utc,
            eop_path,
            extrapolation=eop_extrapolation,
        )
        dp, de = _interp_eop_nutation(mjd_utc, eop_path, extrapolation=eop_extrapolation)
        ddpsi_rad += dp
        ddeps_rad += de
    else:
        dut1_s = 0.0 if dut1_s is None else float(dut1_s)
        if dat_s is None:
            dat_s = (float(tt_minus_utc_s) if tt_minus_utc_s is not None else _DEFAULT_TT_MINUS_UTC_S) - 32.184
        else:
            dat_s = float(dat_s)
    jd_tt = float(jd_utc) + (float(dat_s) + 32.184) / _DAYSEC
    _rbpn, dpsi, _true_obliquity = _precession_nutation_matrix_approx(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    return float((_gmst_iau1982_split(float(jd_utc), 0.0, float(dut1_s))
                  + _equation_of_equinoxes_iau1994(jd_tt, dpsi)) % (2.0 * math.pi))


def precession_nutation_rotation_hpop_like(
    t_s: float,
    jd_utc_start: float | None = None,
    eop_path: str | None = None,
    *,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
) -> np.ndarray:
    if jd_utc_start is None or not eop_path:
        return np.eye(3, dtype=float)
    _validate_eop_elapsed_interval(jd_utc_start, t_s, eop_path)
    jd_utc = float(jd_utc_start) + float(t_s) / _DAYSEC
    mjd_utc = jd_utc - _MJD0
    _xp_arcsec, _yp_arcsec, _dut1_s, dat_s = _interp_eop(
        mjd_utc,
        eop_path,
        extrapolation=eop_extrapolation,
    )
    dp, de = _interp_eop_nutation(mjd_utc, eop_path, extrapolation=eop_extrapolation)
    ddpsi_rad += dp
    ddeps_rad += de
    jd_tt = jd_utc + (dat_s + 32.184) / _DAYSEC
    rbpn, _dpsi, _true_obliquity = _precession_nutation_matrix_approx(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    return rbpn


def _polar_motion_matrix(xp_rad: float, yp_rad: float, sp_rad: float) -> np.ndarray:
    return _rz(sp_rad) @ _ry(-xp_rad) @ _rx(-yp_rad)


def _eci_to_ecef_rotation_hpop_like_uncached(
    t_s: float,
    jd_utc_start: float | None = None,
    eop_path: str | None = None,
    *,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
    eop_file_signature: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    has_manual_eop = any(value is not None for value in (dut1_s, xp_arcsec, yp_arcsec, dat_s)) or (
        float(ddpsi_rad) != 0.0 or float(ddeps_rad) != 0.0
    )
    if jd_utc_start is None and (eop_path or has_manual_eop):
        raise ValueError("IAU76/80 EOP frame transforms require jd_utc_start when EOP data is configured.")
    if jd_utc_start is not None and eop_path:
        _validate_eop_elapsed_interval(jd_utc_start, t_s, eop_path)
    if jd_utc_start is None or (not eop_path and not has_manual_eop):
        return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start)

    jd_utc = float(jd_utc_start) + float(t_s) / 86400.0
    if eop_path:
        mjd_utc = jd_utc - _MJD0
        xp_arcsec, yp_arcsec, dut1_s, dat_s = _interp_eop(
            mjd_utc,
            eop_path,
            extrapolation=eop_extrapolation,
        )
        dp, de = _interp_eop_nutation(mjd_utc, eop_path, extrapolation=eop_extrapolation)
        ddpsi_rad += dp
        ddeps_rad += de
    else:
        xp_arcsec = 0.0 if xp_arcsec is None else float(xp_arcsec)
        yp_arcsec = 0.0 if yp_arcsec is None else float(yp_arcsec)
        dut1_s = 0.0 if dut1_s is None else float(dut1_s)
        if dat_s is None:
            dat_s = (float(tt_minus_utc_s) if tt_minus_utc_s is not None else _DEFAULT_TT_MINUS_UTC_S) - 32.184
        else:
            dat_s = float(dat_s)
    jd_tt = jd_utc + (float(dat_s) + 32.184) / _DAYSEC
    rbpn, dpsi, _true_obliquity = _precession_nutation_matrix_approx(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    gast = float((_gmst_iau1982_split(float(jd_utc_start), float(t_s), float(dut1_s))
                  + _equation_of_equinoxes_iau1994(jd_tt, dpsi)) % (2.0 * math.pi))
    sp = -47.0e-6 * ((jd_tt - _J2000) / _JULIAN_CENTURY_DAYS) * _ARCSEC_TO_RAD
    return _polar_motion_matrix(float(xp_arcsec) * _ARCSEC_TO_RAD, float(yp_arcsec) * _ARCSEC_TO_RAD, sp) @ _rz(gast) @ rbpn


@lru_cache(maxsize=8192)
def _cached_eci_to_ecef_rotation_hpop_like(
    t_s: float,
    jd_utc_start: float | None,
    eop_path: str | None,
    dut1_s: float | None,
    xp_arcsec: float | None,
    yp_arcsec: float | None,
    dat_s: float | None,
    tt_minus_utc_s: float | None,
    ddpsi_rad: float,
    ddeps_rad: float,
    eop_extrapolation: str,
    eop_file_signature: tuple[int, int, int, int] | None = None,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    rotation = _eci_to_ecef_rotation_hpop_like_uncached(
        t_s,
        jd_utc_start=jd_utc_start,
        eop_path=eop_path,
        dut1_s=dut1_s,
        xp_arcsec=xp_arcsec,
        yp_arcsec=yp_arcsec,
        dat_s=dat_s,
        tt_minus_utc_s=tt_minus_utc_s,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
        eop_extrapolation=eop_extrapolation,
        eop_file_signature=eop_file_signature,
    )
    flattened = rotation.ravel()
    return (
        float(flattened[0]),
        float(flattened[1]),
        float(flattened[2]),
        float(flattened[3]),
        float(flattened[4]),
        float(flattened[5]),
        float(flattened[6]),
        float(flattened[7]),
        float(flattened[8]),
    )


def eci_to_ecef_rotation_hpop_like(
    t_s: float,
    jd_utc_start: float | None = None,
    eop_path: str | None = None,
    *,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
) -> np.ndarray:
    """Return a fresh ECI-to-ECEF rotation, reusing exact frame inputs."""

    values = _cached_eci_to_ecef_rotation_hpop_like(
        float(t_s),
        None if jd_utc_start is None else float(jd_utc_start),
        None if eop_path is None else str(eop_path),
        None if dut1_s is None else float(dut1_s),
        None if xp_arcsec is None else float(xp_arcsec),
        None if yp_arcsec is None else float(yp_arcsec),
        None if dat_s is None else float(dat_s),
        None if tt_minus_utc_s is None else float(tt_minus_utc_s),
        float(ddpsi_rad),
        float(ddeps_rad),
        str(eop_extrapolation),
        eop_file_signature=None
        if eop_path is None
        else _eop_file_signature(str(eop_path)),
    )
    return np.array(values, dtype=float).reshape(3, 3)


def eci_to_ecef_rotation_context(t_s: float, context: FrameContext) -> np.ndarray:
    ctx = context.at(t_s)
    if ctx.model == FRAME_MODEL_IAU76_80_EOP:
        if ctx.eop_path and ctx.jd_utc_start is None:
            raise ValueError("IAU76/80 EOP frame rotation requires simulator.initial_jd_utc when eop_path is set.")
        if ctx.has_manual_eop and ctx.jd_utc_start is None:
            raise ValueError("IAU76/80 manual EOP frame rotation requires simulator.initial_jd_utc.")
        return eci_to_ecef_rotation_hpop_like(
            t_s,
            jd_utc_start=ctx.jd_utc_start,
            eop_path=ctx.eop_path,
            dut1_s=ctx.dut1_s,
            xp_arcsec=ctx.xp_arcsec,
            yp_arcsec=ctx.yp_arcsec,
            dat_s=ctx.dat_s,
            tt_minus_utc_s=ctx.tt_minus_utc_s,
            ddpsi_rad=ctx.ddpsi_rad,
            ddeps_rad=ctx.ddeps_rad,
            eop_extrapolation=ctx.eop_extrapolation,
        )
    return eci_to_ecef_rotation(t_s, jd_utc_start=ctx.jd_utc_start)


def rotation_between(
    source_frame: str,
    target_frame: str,
    *,
    t_s: float,
    context: FrameContext,
) -> np.ndarray:
    source = str(source_frame or "").strip().lower()
    target = str(target_frame or "").strip().lower()
    if source == target:
        return np.eye(3, dtype=float)
    if source == "eci" and target in {"ecef", "itrf"}:
        return eci_to_ecef_rotation_context(t_s, context)
    if source in {"ecef", "itrf"} and target == "eci":
        return eci_to_ecef_rotation_context(t_s, context).T
    raise ValueError(f"Unsupported frame rotation {source_frame!r} -> {target_frame!r}.")


def transform_position(
    position_km: np.ndarray,
    source_frame: str,
    target_frame: str,
    *,
    t_s: float,
    context: FrameContext,
) -> np.ndarray:
    return rotation_between(source_frame, target_frame, t_s=t_s, context=context) @ np.array(position_km, dtype=float)


def transform_state(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    source_frame: str,
    target_frame: str,
    *,
    t_s: float,
    context: FrameContext,
) -> tuple[np.ndarray, np.ndarray]:
    source = str(source_frame or "").strip().lower()
    target = str(target_frame or "").strip().lower()
    pos = np.array(position_km, dtype=float).reshape(3)
    vel = np.array(velocity_km_s, dtype=float).reshape(3)
    if source == target:
        return pos.copy(), vel.copy()
    if source == "eci" and target in {"ecef", "itrf"}:
        rot = eci_to_ecef_rotation_context(t_s, context)
        rot_dot = _eci_to_ecef_rotation_derivative_context(t_s, context)
        return rot @ pos, rot @ vel + rot_dot @ pos
    if source in {"ecef", "itrf"} and target == "eci":
        rot = eci_to_ecef_rotation_context(t_s, context)
        rot_dot = _eci_to_ecef_rotation_derivative_context(t_s, context)
        pos_eci = rot.T @ pos
        vel_eci = rot.T @ (vel - rot_dot @ pos_eci)
        return pos_eci, vel_eci
    raise ValueError(f"Unsupported frame state transform {source_frame!r} -> {target_frame!r}.")


def _eci_to_ecef_rotation_derivative_context(t_s: float, context: FrameContext) -> np.ndarray:
    if context.model == FRAME_MODEL_SIMPLE_GMST:
        return _simple_rotation_derivative(t_s, context)
    return _rotation_derivative_from_evaluator(
        float(t_s), lambda sample_t: eci_to_ecef_rotation_context(sample_t, context)
    )


def _simple_rotation_derivative(t_s: float, context: FrameContext) -> np.ndarray:
    # Differentiate the selected rotation law directly. Finite differences of
    # large Julian dates inject epoch quantization into station velocities.
    rate = EARTH_ROT_RATE_RAD_S
    if context.jd_utc_start is not None:
        centuries = (float(context.jd_utc_start) + float(t_s) / _DAYSEC - _J2000) / _JULIAN_CENTURY_DAYS
        rate = np.deg2rad(
            360.98564736629
            + (2.0 * 0.000387933 * centuries - 3.0 * centuries**2 / 38710000.0)
            / _JULIAN_CENTURY_DAYS
        ) / _DAYSEC
    rotation = eci_to_ecef_rotation(float(t_s), context.jd_utc_start)
    return rate * np.array([rotation[1], -rotation[0], np.zeros(3)])


def _rotation_derivative_from_evaluator(t_s: float, rotation) -> np.ndarray:
    # Julian dates near the present epoch have a floating-point spacing of
    # roughly tens of microseconds.  A 0.01 s two-point difference therefore
    # amplified epoch quantization into mm/s-to-m/s station-velocity errors.
    # A symmetric five-point stencil over 30 s suppresses that roundoff while
    # retaining fourth-order accuracy for Earth rotation and the much slower
    # precession/nutation/EOP terms.
    step_s = 30.0
    t = float(t_s)
    try:
        return (
            -rotation(t + 2.0 * step_s)
            + 8.0 * rotation(t + step_s)
            - 8.0 * rotation(t - step_s)
            + rotation(t - 2.0 * step_s)
        ) / (12.0 * step_s)
    except ValueError as centered_error:
        # A valid state at the first/last EOP record must remain transformable.
        # Use a fourth-order one-sided stencil when the centered stencil would
        # leave the configured EOP coverage.
        try:
            return (
                -25.0 * rotation(t)
                + 48.0 * rotation(t + step_s)
                - 36.0 * rotation(t + 2.0 * step_s)
                + 16.0 * rotation(t + 3.0 * step_s)
                - 3.0 * rotation(t + 4.0 * step_s)
            ) / (12.0 * step_s)
        except ValueError:
            try:
                return (
                    25.0 * rotation(t)
                    - 48.0 * rotation(t - step_s)
                    + 36.0 * rotation(t - 2.0 * step_s)
                    - 16.0 * rotation(t - 3.0 * step_s)
                    + 3.0 * rotation(t - 4.0 * step_s)
                ) / (12.0 * step_s)
            except ValueError as backward_error:
                raise centered_error from backward_error


def _eop_cache_path(eop_path: str) -> str:
    # A lexical absolute path is sufficient for a signature-bound cache.
    # stat and the reader follow symlinks, so retargeting still invalidates it.
    # Resolving every parent directory at every force stage is unnecessary.
    expanded = os.path.expanduser(eop_path)
    # Preserve '..' after a symlink: lexical normalization would change which
    # file the OS opens. Relative paths still bind to the current directory.
    return expanded if os.path.isabs(expanded) else os.path.join(os.getcwd(), expanded)

def _validate_eop_elapsed_intervals(jd_utc_start: float, t_s: float, intervals: tuple) -> None:
    """Apply the elapsed-time policy to an already signature-bound table."""

    elapsed = float(t_s)
    if elapsed == 0.0:
        return
    start_mjd = float(jd_utc_start) - _MJD0
    end_mjd = start_mjd + elapsed / _DAYSEC
    lower, upper = sorted((start_mjd, end_mjd))
    for _left, right in intervals:
        # Include the right endpoint: 86400 SI seconds from the start of a
        # positive leap-second day still lands one second before the next
        # UTC midnight, so mapping it to the DAT transition is unsafe.
        if lower < right <= upper:
            raise ValueError(
                "EOP-backed UTC elapsed time crosses a DAT leap-second boundary "
                f"near MJD {right:.9f}; the current fixed-86400-second UTC "
                "conversion is not leap-second aware."
            )


class PreparedFrameEvaluator:
    """Exact frame arithmetic over one validated resource snapshot per stage.

    Call ``refresh`` at every integration stage before reading rotations. A
    rewritten, replaced, or missing EOP file invalidates all cached rows. The
    same interpolation, leap-second and derivative-stencil policies as the
    ordinary frame API apply; only time-dependent values are retained.
    """

    def __init__(self, context: FrameContext, native_context=None):
        self.context = context
        self.native_rotation = getattr(native_context, "rotation_iau76_80", None)
        active_eop_path = context.eop_path if context.eop_rotation_available else None
        self.path = None if not active_eop_path else _eop_cache_path(active_eop_path)
        self._relative_eop_path = bool(
            active_eop_path and not os.path.isabs(os.path.expanduser(active_eop_path))
        )
        self.signature = None
        self.intervals = ()
        self.rotations = {}
        self.derivatives = {}
        self.refresh()

    def refresh(self) -> None:
        if self.path is None:
            return
        # The direct frame API binds relative paths to the current directory
        # on each call. Absolute scenario paths retain their prepared route.
        path = _eop_cache_path(self.context.eop_path) if self._relative_eop_path else self.path
        signature = _eop_file_signature(path)
        if path != self.path or signature != self.signature:
            self.intervals = _eop_dat_change_intervals_cached(path, signature)
            self.path = path
            self.signature = signature
            self.rotations.clear()
            self.derivatives.clear()

    @staticmethod
    def _remember(cache: dict, key: float, value: np.ndarray) -> np.ndarray:
        cache[key] = value
        if len(cache) > 512:
            cache.pop(next(iter(cache)))
        return value

    def rotation(self, t_s: float) -> np.ndarray:
        t = float(t_s)
        cached = self.rotations.get(t)
        if cached is not None:
            return cached
        ctx = self.context
        if ctx.model != FRAME_MODEL_IAU76_80_EOP or not ctx.eop_rotation_available:
            # Keep the reference API's missing-epoch checks and its simple
            # fallback when no EOP inputs are configured.
            rotation = eci_to_ecef_rotation_context(t, ctx)
        elif not callable(self.native_rotation):
            rotation = eci_to_ecef_rotation_context(t, ctx)
        else:
            jd_start = float(ctx.jd_utc_start)
            if self.path is not None:
                _validate_eop_elapsed_intervals(jd_start, t, self.intervals)
                xp, yp, dut1, dat = _interp_eop_cached(
                    jd_start + t / _DAYSEC - _MJD0,
                    self.path, self.signature, extrapolation=ctx.eop_extrapolation,
                )
                dp, de = _interp_eop_nutation_cached(
                    jd_start + t / _DAYSEC - _MJD0,
                    self.path, self.signature, extrapolation=ctx.eop_extrapolation,
                )
            else:
                xp = 0.0 if ctx.xp_arcsec is None else float(ctx.xp_arcsec)
                yp = 0.0 if ctx.yp_arcsec is None else float(ctx.yp_arcsec)
                dut1 = 0.0 if ctx.dut1_s is None else float(ctx.dut1_s)
                dat = ctx.tt_minus_utc_s - 32.184 if ctx.dat_s is None else float(ctx.dat_s)
                dp, de = 0.0, 0.0
            rotation = np.asarray(
                self.native_rotation(t, jd_start, xp, yp, dut1, dat, ctx.ddpsi_rad + dp, ctx.ddeps_rad + de),
                dtype=float,
            ).reshape(3, 3)
        return self._remember(self.rotations, t, rotation)

    def derivative(self, t_s: float) -> np.ndarray:
        t = float(t_s)
        cached = self.derivatives.get(t)
        if cached is not None:
            return cached
        if self.context.model == FRAME_MODEL_SIMPLE_GMST:
            return self._remember(self.derivatives, t, _simple_rotation_derivative(t, self.context))
        return self._remember(
            self.derivatives, t, _rotation_derivative_from_evaluator(t, self.rotation)
        )


def eci_to_ecef_rotation_derivative_context(
    t_s: float,
    context: FrameContext,
) -> np.ndarray:
    """Return d(ECEF-from-ECI)/dt for the selected frame context."""

    return _eci_to_ecef_rotation_derivative_context(float(t_s), context)


def eci_to_ecef_harmonic(
    r_eci_km: np.ndarray,
    t_s: float,
    jd_utc_start: float | None = None,
    frame_model: str = "simple",
    eop_path: str | None = None,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
) -> np.ndarray:
    r_eci = (
        r_eci_km
        if isinstance(r_eci_km, np.ndarray) and r_eci_km.dtype == np.float64
        else np.asarray(r_eci_km, dtype=float)
    )
    model = normalize_frame_model(frame_model)
    if model == FRAME_MODEL_IAU76_80_EOP:
        rot = eci_to_ecef_rotation_hpop_like(
            t_s,
            jd_utc_start=jd_utc_start,
            eop_path=eop_path,
            dut1_s=dut1_s,
            xp_arcsec=xp_arcsec,
            yp_arcsec=yp_arcsec,
            dat_s=dat_s,
            tt_minus_utc_s=tt_minus_utc_s,
            ddpsi_rad=ddpsi_rad,
            ddeps_rad=ddeps_rad,
            eop_extrapolation=eop_extrapolation,
        )
        return rot @ r_eci
    return eci_to_ecef(r_eci, t_s, jd_utc_start=jd_utc_start)


def ecef_to_eci_harmonic(
    r_ecef_km: np.ndarray,
    t_s: float,
    jd_utc_start: float | None = None,
    frame_model: str = "simple",
    eop_path: str | None = None,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
) -> np.ndarray:
    model = normalize_frame_model(frame_model)
    if model == FRAME_MODEL_IAU76_80_EOP:
        rot = eci_to_ecef_rotation_hpop_like(
            t_s,
            jd_utc_start=jd_utc_start,
            eop_path=eop_path,
            dut1_s=dut1_s,
            xp_arcsec=xp_arcsec,
            yp_arcsec=yp_arcsec,
            dat_s=dat_s,
            tt_minus_utc_s=tt_minus_utc_s,
            ddpsi_rad=ddpsi_rad,
            ddeps_rad=ddeps_rad,
            eop_extrapolation=eop_extrapolation,
        )
        return rot.T @ np.array(r_ecef_km, dtype=float)
    return ecef_to_eci(np.array(r_ecef_km, dtype=float), t_s, jd_utc_start=jd_utc_start)
