from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.acceleration.settings import acceleration_enabled_from_mode
from sim.dynamics.orbit.frames import _interp_eop

_EMRAT = 81.3005682214972154
_DE440_POSITION_CACHE_KEY = "_oel_de440_position_cache"
_DE440_LIGHT_ROW_CACHE_KEY = "_oel_de440_light_row_cache"


@lru_cache(maxsize=16)
def _resolve_absolute_path(path_text: str) -> Path:
    return Path(path_text).resolve()


@lru_cache(maxsize=16)
def _resolve_relative_path(path_text: str, cwd: str) -> Path:
    return (Path(cwd) / path_text).resolve()


def _resolve_path(path_value: str | Path) -> Path:
    """Resolve a configured ephemeris resource against the current process path."""

    expanded = os.path.expanduser(os.fspath(path_value))
    if os.path.isabs(expanded):
        return _resolve_absolute_path(expanded)
    return _resolve_relative_path(expanded, os.getcwd())


def _resource_signature(path_value: str | Path) -> tuple:
    """Return a file identity used to invalidate loaded DE440 resources."""
    path = _resolve_path(path_value)
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None)
    return (
        str(path),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


@lru_cache(maxsize=1)
def default_de440_coeff_path() -> Path:
    return (Path(__file__).resolve().parents[3] / "validation" / "data" / "DE440Coeff.mat").resolve()


@lru_cache(maxsize=1)
def default_hpop_eop_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "validation"
        / "High Precision Orbit Propagator_4-2"
        / "High Precision Orbit Propagator_4.2.2"
        / "EOP-All.txt"
    ).resolve()


@lru_cache(maxsize=2)
def _load_de440_coeff_matrix_cached(path_str: str, signature: tuple) -> np.ndarray:
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "DE440 HPOP ephemeris mode requires scipy.io.loadmat to read DE440Coeff.mat. "
            "Install a compatible SciPy in the active environment."
        ) from exc

    path = _resolve_path(path_str)
    mat = loadmat(path)
    coeff = mat.get("DE440Coeff")
    if coeff is None:
        raise ValueError(f"DE440Coeff.mat did not contain 'DE440Coeff': {path}")
    arr = np.asarray(coeff, dtype=float)
    if arr.ndim != 2:
        arr = np.atleast_2d(arr)
    arr.setflags(write=False)
    return arr


def _load_de440_coeff_matrix(path_str: str) -> np.ndarray:
    """Load a full DE440 matrix keyed by the current resource identity."""
    path = _resolve_path(path_str)
    return _load_de440_coeff_matrix_cached(str(path), _resource_signature(path))


@lru_cache(maxsize=2)
def _load_de440_light_cached(path_str: str, signature: tuple) -> dict[str, object]:
    path = _resolve_path(path_str)
    with np.load(path, allow_pickle=False) as data:
        fmt = str(np.asarray(data["format_version"]).item())
        if fmt != "oel_de440_light_v1":
            raise ValueError(f"Unsupported OEL DE440 light format '{fmt}': {path}")
        bodies = tuple(str(v) for v in np.asarray(data["bodies"]).tolist())
        out: dict[str, object] = {
            "path": str(path),
            "format_version": fmt,
            "source": str(np.asarray(data["source"]).item()) if "source" in data else "",
            "row_start_jd_tdb": np.asarray(data["row_start_jd_tdb"], dtype=float),
            "row_end_jd_tdb": np.asarray(data["row_end_jd_tdb"], dtype=float),
            "bodies": bodies,
            "body_set": frozenset(bodies),
        }
        body_specs: dict[str, tuple[int, int, float]] = {}
        body_records: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for body in bodies:
            key = str(body)
            coeff_count = int(np.asarray(data[f"{key}_coeffs"]).item())
            segments = int(np.asarray(data[f"{key}_segments"]).item())
            span_days = float(np.asarray(data[f"{key}_span_days"]).item())
            out[f"{key}_coeffs"] = coeff_count
            out[f"{key}_segments"] = segments
            out[f"{key}_span_days"] = span_days
            body_specs[key] = (coeff_count, segments, span_days)
            if f"{key}_start_jd_tdb" in data and f"{key}_end_jd_tdb" in data:
                out[f"{key}_start_jd_tdb"] = np.asarray(data[f"{key}_start_jd_tdb"], dtype=float)
                out[f"{key}_end_jd_tdb"] = np.asarray(data[f"{key}_end_jd_tdb"], dtype=float)
            out[f"{key}_x"] = np.asarray(data[f"{key}_x"], dtype=float)
            out[f"{key}_y"] = np.asarray(data[f"{key}_y"], dtype=float)
            out[f"{key}_z"] = np.asarray(data[f"{key}_z"], dtype=float)
            body_records[key] = (
                np.asarray(out.get(f"{key}_start_jd_tdb", out["row_start_jd_tdb"]), dtype=float),
                np.asarray(out.get(f"{key}_end_jd_tdb", out["row_end_jd_tdb"]), dtype=float),
                out[f"{key}_x"],  # type: ignore[arg-type]
                out[f"{key}_y"],  # type: ignore[arg-type]
                out[f"{key}_z"],  # type: ignore[arg-type]
            )
        out["body_specs"] = body_specs
        out["body_records"] = body_records
        for value in out.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return out


def _load_de440_light(path_str: str) -> dict[str, object]:
    """Load a lightbank keyed by current file identity, not path alone."""
    path = _resolve_path(path_str)
    return _load_de440_light_cached(str(path), _resource_signature(path))


@lru_cache(maxsize=1)
def _compiled_de440_light_core():
    from sim.acceleration.kernels.de440 import de440_light_core_kernel

    return de440_light_core_kernel


@lru_cache(maxsize=1)
def _compiled_de440_sun_moon_from_utc():
    from sim.acceleration.kernels.de440 import de440_sun_moon_from_utc_kernel

    return de440_sun_moon_from_utc_kernel


def _eval_light_core_accelerated(
    light: dict[str, object], jd_tdb: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_specs = light["body_specs"]
    body_records = light["body_records"]
    if not isinstance(body_specs, dict) or not isinstance(body_records, dict):
        raise TypeError("Invalid OEL DE440 light body metadata.")
    earthmoon_spec = body_specs["earthmoon"]
    moon_spec = body_specs["moon"]
    sun_spec = body_specs["sun"]
    earthmoon_record = body_records["earthmoon"]
    moon_record = body_records["moon"]
    sun_record = body_records["sun"]
    result = _compiled_de440_light_core()(
        float(jd_tdb),
        *earthmoon_spec,
        *earthmoon_record,
        *moon_spec,
        *moon_record,
        *sun_spec,
        *sun_record,
    )
    r_earthmoon, r_moon, r_sun, earthmoon_row, moon_row, sun_row = result
    for body, row_index in (
        ("earthmoon", earthmoon_row),
        ("moon", moon_row),
        ("sun", sun_row),
    ):
        if int(row_index) < 0:
            raise ValueError(f"JD_TDB {jd_tdb} is outside OEL DE440 light coverage for body '{body}'.")
    return r_earthmoon, r_moon, r_sun


def _light_core_arguments(light: dict[str, object]) -> tuple[object, ...]:
    body_specs = light["body_specs"]
    body_records = light["body_records"]
    if not isinstance(body_specs, dict) or not isinstance(body_records, dict):
        raise TypeError("Invalid OEL DE440 light body metadata.")
    return (
        *body_specs["earthmoon"],
        *body_records["earthmoon"],
        *body_specs["moon"],
        *body_records["moon"],
        *body_specs["sun"],
        *body_records["sun"],
    )


def _cheb3d(t: float, n: int, ta: float, tb: float, cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> np.ndarray:
    if t < ta or t > tb:
        raise ValueError("Time out of range in _cheb3d")
    tau = (2.0 * t - ta - tb) / (tb - ta)
    f1x = f1y = f1z = 0.0
    f2x = f2y = f2z = 0.0
    for i in range(int(n) - 1, 0, -1):
        old_f1x, old_f1y, old_f1z = f1x, f1y, f1z
        f1x = 2.0 * tau * f1x - f2x + float(cx[i])
        f1y = 2.0 * tau * f1y - f2y + float(cy[i])
        f1z = 2.0 * tau * f1z - f2z + float(cz[i])
        f2x, f2y, f2z = old_f1x, old_f1y, old_f1z
    return np.array(
        [
            tau * f1x - f2x + float(cx[0]),
            tau * f1y - f2y + float(cy[0]),
            tau * f1z - f2z + float(cz[0]),
        ],
        dtype=float,
    )


def _subinterval_state(jd_tdb: float, t1: float, span_days: float, segments: int) -> tuple[int, float, float]:
    dt = float(jd_tdb - t1)
    seg_len = float(span_days) / float(segments)
    idx = int(np.floor(dt / seg_len))
    idx = max(0, min(int(segments) - 1, idx))
    jd0 = float(t1 + seg_len * idx)
    return idx, jd0, jd0 + seg_len


def _extract_axis(row: np.ndarray, starts_1b: tuple[int, ...], coeff_count: int) -> np.ndarray:
    out = []
    for s1 in starts_1b:
        s0 = int(s1) - 1
        out.append(np.asarray(row[s0 : s0 + coeff_count], dtype=float))
    return np.concatenate(out)


_BODY_SPECS: dict[str, dict[str, object]] = {
    "mercury": {"starts": (3, 45, 87, 129), "coeffs": 14, "segments": 4, "span_days": 32.0},
    "venus": {"starts": (171, 201), "coeffs": 10, "segments": 2, "span_days": 32.0},
    "earthmoon": {"starts": (231, 270), "coeffs": 13, "segments": 2, "span_days": 32.0},
    "mars": {"starts": (309,), "coeffs": 11, "segments": 1, "span_days": 32.0},
    "jupiter": {"starts": (342,), "coeffs": 8, "segments": 1, "span_days": 32.0},
    "saturn": {"starts": (366,), "coeffs": 7, "segments": 1, "span_days": 32.0},
    "uranus": {"starts": (387,), "coeffs": 6, "segments": 1, "span_days": 32.0},
    "neptune": {"starts": (405,), "coeffs": 6, "segments": 1, "span_days": 32.0},
    "pluto": {"starts": (423,), "coeffs": 6, "segments": 1, "span_days": 32.0},
    "moon": {"starts": (441, 480, 519, 558, 597, 636, 675, 714), "coeffs": 13, "segments": 8, "span_days": 32.0},
    "sun": {"starts": (753, 786), "coeffs": 11, "segments": 2, "span_days": 32.0},
}


def _eval_body(row: np.ndarray, jd_tdb: float, body_name: str) -> np.ndarray:
    spec = _BODY_SPECS[body_name]
    starts = tuple(int(v) for v in spec["starts"])  # type: ignore[index]
    coeff_count = int(spec["coeffs"])  # type: ignore[index]
    segments = int(spec["segments"])  # type: ignore[index]
    span_days = float(spec["span_days"])  # type: ignore[index]
    t1 = float(row[0])
    idx, ta, tb = _subinterval_state(jd_tdb, t1, span_days, segments)

    x_all = _extract_axis(row, starts, coeff_count)
    y_all = _extract_axis(row, tuple(s + coeff_count for s in starts), coeff_count)
    z_all = _extract_axis(row, tuple(s + 2 * coeff_count for s in starts), coeff_count)

    i0 = idx * coeff_count
    i1 = i0 + coeff_count
    return 1e3 * _cheb3d(jd_tdb, coeff_count, ta, tb, x_all[i0:i1], y_all[i0:i1], z_all[i0:i1])


def _find_light_row(light: dict[str, object], jd_tdb: float) -> int:
    starts = np.asarray(light["row_start_jd_tdb"], dtype=float)
    ends = np.asarray(light["row_end_jd_tdb"], dtype=float)
    jd = float(jd_tdb)
    mask = (starts <= jd) & (jd <= ends)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"JD_TDB {jd_tdb} is outside OEL DE440 light coverage.")
    return int(idx[0])


def _find_light_body_row(light: dict[str, object], jd_tdb: float, body: str) -> int:
    start_key = f"{body}_start_jd_tdb"
    end_key = f"{body}_end_jd_tdb"
    if start_key in light and end_key in light:
        starts = np.asarray(light[start_key], dtype=float)
        ends = np.asarray(light[end_key], dtype=float)
        jd = float(jd_tdb)
        mask = (starts <= jd) & (jd <= ends)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError(f"JD_TDB {jd_tdb} is outside OEL DE440 light coverage for body '{body}'.")
        return int(idx[0])
    return _find_light_row(light, jd_tdb)


def _eval_light_body(light: dict[str, object], jd_tdb: float, body_name: str) -> np.ndarray:
    body = str(body_name).strip().lower()
    body_set = light.get("body_set", light["bodies"])
    if body not in body_set:
        raise RuntimeError(f"Body '{body_name}' is not available in OEL DE440 light file.")
    row_index = _find_light_body_row(light, jd_tdb, body)
    body_specs = light.get("body_specs")
    if isinstance(body_specs, dict) and body in body_specs:
        coeff_count, segments, span_days = body_specs[body]
    else:
        coeff_count = int(light[f"{body}_coeffs"])
        segments = int(light[f"{body}_segments"])
        span_days = float(light[f"{body}_span_days"])
    start_key = f"{body}_start_jd_tdb"
    if start_key in light:
        t1 = float(np.asarray(light[start_key], dtype=float)[row_index])
    else:
        t1 = float(np.asarray(light["row_start_jd_tdb"], dtype=float)[row_index])
    idx, ta, tb = _subinterval_state(jd_tdb, t1, span_days, segments)
    i0 = idx * coeff_count
    i1 = i0 + coeff_count
    x_all = np.asarray(light[f"{body}_x"], dtype=float)[row_index, :]
    y_all = np.asarray(light[f"{body}_y"], dtype=float)[row_index, :]
    z_all = np.asarray(light[f"{body}_z"], dtype=float)[row_index, :]
    return 1e3 * _cheb3d(jd_tdb, coeff_count, ta, tb, x_all[i0:i1], y_all[i0:i1], z_all[i0:i1])


def _find_coeff_row(pc: np.ndarray, jd_tdb: float) -> np.ndarray:
    mask = (pc[:, 0] <= jd_tdb) & (jd_tdb <= pc[:, 1])
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"JD_TDB {jd_tdb} is outside DE440Coeff coverage.")
    return np.asarray(pc[int(idx[0]), :], dtype=float).reshape(-1)


def mjd_tt_to_mjd_tdb(mjd_tt: float) -> float:
    t_tt = (float(mjd_tt) - 51544.5) / 36525.0
    return (
        float(mjd_tt)
        + (
            0.001657 * np.sin(628.3076 * t_tt + 6.2401)
            + 0.000022 * np.sin(575.3385 * t_tt + 4.2970)
            + 0.000014 * np.sin(1256.6152 * t_tt + 6.1969)
            + 0.000005 * np.sin(606.9777 * t_tt + 4.0212)
            + 0.000005 * np.sin(52.9691 * t_tt + 0.4444)
            + 0.000002 * np.sin(21.3299 * t_tt + 5.5431)
            + 0.000010 * np.sin(628.3076 * t_tt + 4.2490)
        )
        / 86400.0
    )


def jd_utc_to_jd_tdb(jd_utc: float, eop_path: str | None = None, tai_utc_s: float | None = None) -> float:
    mjd_utc = float(jd_utc) - 2400000.5
    if tai_utc_s is None:
        if eop_path:
            _xp, _yp, _dut1, dat_s = _interp_eop(mjd_utc, eop_path)
            tai_utc_s = float(dat_s)
        else:
            tai_utc_s = 37.0
    tt_utc_s = 32.184 + float(tai_utc_s)
    mjd_tt = mjd_utc + tt_utc_s / 86400.0
    mjd_tdb = mjd_tt_to_mjd_tdb(mjd_tt)
    return float(mjd_tdb + 2400000.5)


def hpop_de440_positions_m(jd_tdb: float, coeff_path: str | Path | None = None) -> dict[str, np.ndarray]:
    path = default_de440_coeff_path() if coeff_path is None else _resolve_path(coeff_path)
    if path.suffix.lower() == ".npz":
        light = _load_de440_light(str(path))
        bodies = light["bodies"]
        body_set = light.get("body_set", bodies)
        if not {"earthmoon", "moon", "sun"}.issubset(body_set):
            raise RuntimeError("OEL DE440 light file must include earthmoon, moon, and sun for geocentric ephemerides.")

        if acceleration_enabled_from_mode():
            r_earthmoon, r_moon, r_sun = _eval_light_core_accelerated(light, jd_tdb)
        else:
            r_earthmoon = _eval_light_body(light, jd_tdb, "earthmoon")
            r_moon = _eval_light_body(light, jd_tdb, "moon")
            r_sun = _eval_light_body(light, jd_tdb, "sun")
        emrat1 = 1.0 / (1.0 + _EMRAT)
        r_earth = r_earthmoon - emrat1 * r_moon
        out = {
            "earth": r_earth,
            "sun_ssb": r_sun.copy(),
            "sun": -r_earth + r_sun,
            "moon": r_moon,
        }
        for body in bodies:
            if body in {"earthmoon", "moon", "sun"}:
                continue
            out[body] = -r_earth + _eval_light_body(light, jd_tdb, body)
        return out

    pc = _load_de440_coeff_matrix(str(path))
    row = _find_coeff_row(pc, float(jd_tdb))

    r_earthmoon = _eval_body(row, jd_tdb, "earthmoon")
    r_moon = _eval_body(row, jd_tdb, "moon")
    r_sun = _eval_body(row, jd_tdb, "sun")
    r_mercury = _eval_body(row, jd_tdb, "mercury")
    r_venus = _eval_body(row, jd_tdb, "venus")
    r_mars = _eval_body(row, jd_tdb, "mars")
    r_jupiter = _eval_body(row, jd_tdb, "jupiter")
    r_saturn = _eval_body(row, jd_tdb, "saturn")
    r_uranus = _eval_body(row, jd_tdb, "uranus")
    r_neptune = _eval_body(row, jd_tdb, "neptune")
    r_pluto = _eval_body(row, jd_tdb, "pluto")

    emrat1 = 1.0 / (1.0 + _EMRAT)
    r_earth = r_earthmoon - emrat1 * r_moon
    return {
        "earth": r_earth,
        "sun_ssb": r_sun.copy(),
        "sun": -r_earth + r_sun,
        "moon": r_moon,
        "mercury": -r_earth + r_mercury,
        "venus": -r_earth + r_venus,
        "mars": -r_earth + r_mars,
        "jupiter": -r_earth + r_jupiter,
        "saturn": -r_earth + r_saturn,
        "uranus": -r_earth + r_uranus,
        "neptune": -r_earth + r_neptune,
        "pluto": -r_earth + r_pluto,
    }


def hpop_de440_positions_km(jd_utc: float, env: dict) -> dict[str, np.ndarray]:
    coeff_path_raw = env.get("de440_coeff_path")
    coeff_path = default_de440_coeff_path() if coeff_path_raw is None else _resolve_path(str(coeff_path_raw))
    eop_path_raw = env.get("de440_eop_path") or env.get("spherical_harmonics_eop_path") or env.get("drag_eop_path")
    eop_path = None if eop_path_raw is None else str(_resolve_path(str(eop_path_raw)))
    tai_utc_raw = env.get("de440_tai_utc_s")
    cache_key = (
        float(jd_utc),
        str(coeff_path),
        _resource_signature(coeff_path),
        eop_path,
        None if eop_path is None else _resource_signature(eop_path),
        None if tai_utc_raw is None else float(tai_utc_raw),
    )
    cached = env.get(_DE440_POSITION_CACHE_KEY)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == cache_key and isinstance(cached[1], dict):
        return {str(name): np.array(value, dtype=float, copy=True) for name, value in cached[1].items()}
    jd_tdb = jd_utc_to_jd_tdb(
        float(jd_utc), eop_path=eop_path, tai_utc_s=None if tai_utc_raw is None else float(tai_utc_raw)
    )
    pos_m = hpop_de440_positions_m(jd_tdb, coeff_path=coeff_path)
    positions = {k: v / 1e3 for k, v in pos_m.items()}
    env[_DE440_POSITION_CACHE_KEY] = (
        cache_key,
        {str(name): np.array(value, dtype=float, copy=True) for name, value in positions.items()},
    )
    return positions


def hpop_de440_sun_moon_positions_km(jd_utc: float, env: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the mandatory geocentric Sun/Moon pair through the accelerated light path.

    The complete dictionary-producing API remains authoritative for callers that
    need Earth, barycentric Sun, or optional planetary bodies.
    """

    coeff_path_raw = env.get("de440_coeff_path")
    coeff_path = default_de440_coeff_path() if coeff_path_raw is None else _resolve_path(str(coeff_path_raw))
    native_requested = env.get("_rust_numeric_backend") == "rust"
    if coeff_path.suffix.lower() != ".npz" or (not acceleration_enabled_from_mode() and not native_requested):
        positions = hpop_de440_positions_km(jd_utc, env)
        return np.array(positions["sun"], dtype=float), np.array(positions["moon"], dtype=float)

    eop_path_raw = env.get("de440_eop_path") or env.get("spherical_harmonics_eop_path") or env.get("drag_eop_path")
    eop_path = None if eop_path_raw is None else str(_resolve_path(str(eop_path_raw)))
    tai_utc_raw = env.get("de440_tai_utc_s")
    light = _load_de440_light(str(coeff_path))
    body_set = light.get("body_set", light["bodies"])
    if not {"earthmoon", "moon", "sun"}.issubset(body_set):
        raise RuntimeError("OEL DE440 light file must include earthmoon, moon, and sun for geocentric ephemerides.")
    if tai_utc_raw is None:
        if eop_path:
            _xp, _yp, _dut1, tai_utc_s = _interp_eop(float(jd_utc) - 2400000.5, eop_path)
        else:
            tai_utc_s = 37.0
    else:
        tai_utc_s = float(tai_utc_raw)
    native_context = env.get("_rust_environment_context")
    if native_context is not None:
        from sim.rust_environment_backend import try_context_de440_sun_moon_km

        jd_tdb = jd_utc_to_jd_tdb(float(jd_utc), eop_path=eop_path, tai_utc_s=float(tai_utc_s))
        # Keep Python's exact coverage/row policy at the resource boundary;
        # the native context only evaluates already selected coefficients.
        for body in ("earthmoon", "moon", "sun"):
            _find_light_body_row(light, jd_tdb, body)
        pair = try_context_de440_sun_moon_km(
            native_context,
            jd_tdb,
            earth_moon_mass_ratio=_EMRAT,
        )
        if pair is not None:
            return pair
    if env.get("_rust_numeric_backend") == "rust":
        from sim.rust_environment_backend import try_de440_chebyshev_batch

        jd_tdb = jd_utc_to_jd_tdb(float(jd_utc), eop_path=eop_path, tai_utc_s=float(tai_utc_s))

        def native_body(body: str) -> np.ndarray | None:
            starts = np.asarray(light.get(f"{body}_start_jd_tdb", light["row_start_jd_tdb"]), dtype=float)
            ends = np.asarray(light.get(f"{body}_end_jd_tdb", light["row_end_jd_tdb"]), dtype=float)
            spec = light["body_specs"][body]
            records = light["body_records"][body]
            # Keep resource selection and coverage policy in Python, but pass
            # only the covering coefficient row across the native boundary.
            # Sending the complete multi-decade lightbank for every RK stage
            # dominated the Rust force-plan path and repeatedly serialized
            # tens of thousands of coefficients.
            row_index = _find_light_body_row(light, jd_tdb, body)
            x_rows = np.asarray(records[2], dtype=float)[row_index : row_index + 1, :]
            y_rows = np.asarray(records[3], dtype=float)[row_index : row_index + 1, :]
            z_rows = np.asarray(records[4], dtype=float)[row_index : row_index + 1, :]
            return try_de440_chebyshev_batch(
                [jd_tdb],
                row_starts_jd_tdb=starts[row_index : row_index + 1],
                row_ends_jd_tdb=ends[row_index : row_index + 1],
                coeff_count=int(spec[0]),
                segments=int(spec[1]),
                span_days=float(spec[2]),
                x_rows=x_rows.reshape(-1),
                y_rows=y_rows.reshape(-1),
                z_rows=z_rows.reshape(-1),
            )

        earthmoon = native_body("earthmoon")
        moon = native_body("moon")
        sun_ssb = native_body("sun")
        if earthmoon is not None and moon is not None and sun_ssb is not None:
            earth = earthmoon[0] - moon[0] / (1.0 + _EMRAT)
            return -earth + sun_ssb[0], moon[0]
        if native_requested and not acceleration_enabled_from_mode():
            earthmoon = _eval_light_body(light, jd_tdb, "earthmoon")
            moon = _eval_light_body(light, jd_tdb, "moon")
            sun_ssb = _eval_light_body(light, jd_tdb, "sun")
            earth = earthmoon - moon / (1.0 + _EMRAT)
            return -earth + sun_ssb, moon
    sun, moon, earthmoon_row, moon_row, sun_row = _compiled_de440_sun_moon_from_utc()(
        float(jd_utc),
        float(tai_utc_s),
        float(_EMRAT),
        *_light_core_arguments(light),
    )
    for body, row_index in (
        ("earthmoon", earthmoon_row),
        ("moon", moon_row),
        ("sun", sun_row),
    ):
        if int(row_index) < 0:
            jd_tdb = jd_utc_to_jd_tdb(float(jd_utc), eop_path=eop_path, tai_utc_s=float(tai_utc_s))
            raise ValueError(f"JD_TDB {jd_tdb} is outside OEL DE440 light coverage for body '{body}'.")
    return sun, moon
