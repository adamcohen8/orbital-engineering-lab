"""IERS rotational station deformation with an explicit pole convention.

IERS chapter 7, equations 21 and 25--28 (2018 update). The original 2010
conventional mean pole remains an explicit compatibility choice. This is
station motion; the corresponding gravity pole tide is a separate force.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from sim.pro_perturbations.solid_earth_tides import _poly


def station_pole_displacement_m(station_m, *, xp_arcsec, yp_arcsec, jd_tt, pole_model):
    p = np.asarray(station_m, dtype=float)
    if (p.shape != (3,) or not np.all(np.isfinite(p)) or np.linalg.norm(p) == 0
            or not np.all(np.isfinite([xp_arcsec, yp_arcsec, jd_tt]))):
        raise ValueError("Invalid station pole-tide geometry or epoch.")
    years = (jd_tt - 2451545.) / 365.25
    if pole_model == "iers2018_secular":
        mean_x, mean_y = (55. + 1.677 * years) / 1000., (320.5 + 3.460 * years) / 1000.
    elif pole_model == "iers2010_mean":
        if jd_tt <= 2455197.5:
            mean_x = _poly((55.974, 1.8243, .18413, .007024), years) / 1000.
            mean_y = _poly((346.346, 1.7896, -.10729, -.000908), years) / 1000.
        else:
            mean_x, mean_y = (23.513 + 7.6141 * years) / 1000., (358.891 - .6287 * years) / 1000.
    else:
        raise ValueError("Explicit supported station pole model required.")
    m1, m2 = xp_arcsec - mean_x, mean_y - yp_arcsec
    radius, rho = np.linalg.norm(p), np.hypot(p[0], p[1])
    s, c = p[2] / radius, rho / radius
    sl, cl = (p[1] / rho, p[0] / rho) if rho else (0., 1.)
    q = m1 * cl + m2 * sl
    # colatitude theta: cos(2theta)=s²-c², sin(2theta)=2sc.
    south = -.009 * (s*s-c*c) * q
    east = .009 * s * (m1 * sl-m2 * cl)
    up = -.033 * 2*s*c * q
    return south * np.array([s*cl, s*sl, -c]) + east * np.array([-sl, cl, 0.]) + up * p/radius


def apply_station_pole_tides(rows, *, eop_path, pole_model):
    from hashlib import sha256
    from pathlib import Path

    from sim.dynamics.orbit.frames import _interp_eop

    result = deepcopy(list(rows))
    source_hash = sha256(Path(eop_path).read_bytes()).hexdigest()
    for row in result:
        if "slr_station_pole_tide" in row:
            raise ValueError("Station pole tide is already applied.")
        jd = float(row["time_jd_utc"]) + float(row.get("epoch_offset_s", 0.)) / 86400.
        xp, yp, _, dat = _interp_eop(jd - 2400000.5, str(eop_path))
        point = np.array(row["station_metadata"]["ecef_km"]) * 1000.
        delta = station_pole_displacement_m(point, xp_arcsec=xp, yp_arcsec=yp,
                                           jd_tt=jd + (dat + 32.184)/86400., pole_model=pole_model)
        row["station_metadata"]["ecef_km"] = ((point + delta) / 1000.).tolist()
        row["slr_station_pole_tide"] = dict(pole_model=pole_model, displacement_ecef_m=delta.tolist(),
            earth_orientation_sha256=source_hash, evaluation="observation_epoch_held_over_light_time")
    return result
