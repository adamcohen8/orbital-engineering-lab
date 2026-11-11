"""IERS 2010 solid-Earth station tides, including frequency dependence.

Port of Orekit 13.1.7 TidalDisplacement (Apache-2.0); see NOTICE.txt.
Input positions and output displacements use meters in a common Earth frame.
This owner does not model ocean/atmospheric loading or station pole tides.
"""

from __future__ import annotations

import hashlib
import math
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.pro_perturbations.solid_earth_tides import _ARCSEC, _DELAUNAY, _poly

_TABLES = Path(__file__).with_name("data")


def _frequency_terms(name):
    return _parse_frequency_terms((_TABLES / name).read_text())


@lru_cache(maxsize=4)
def _parse_frequency_terms(text):
    terms = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        indices = [i for i, value in enumerate(fields) if re.fullmatch(r"\d{2,3},\d{3}", value)]
        if indices:
            fields = fields[indices[0] + 1 :]
            if len(fields) != 15:
                raise ValueError("Invalid IERS station displacement frequency table.")
            terms.append(
                (int(fields[0]), np.array(fields[6:11], dtype=int), np.array(fields[11:15], dtype=float) * 1.0e-3)
            )
    return tuple(terms)


def station_tide_arguments(jd_utc, *, dut1_s, tai_utc_s, epoch_offset_s=0.0):
    """IERS 2010 gamma and Delaunay arguments with a split UTC epoch."""
    values = [jd_utc, dut1_s, tai_utc_s, epoch_offset_s]
    if not np.all(np.isfinite(values)):
        raise ValueError("Station tide times must be finite.")
    days = float(jd_utc) - 2451545.0
    tt = days + (epoch_offset_s + tai_utc_s + 32.184) / 86400.0
    ut1 = days + (epoch_offset_s + dut1_s) / 86400.0
    t = tt / 36525.0
    gamma = (
        2 * math.pi * (0.7790572732640 + 1.00273781191135448 * ut1)
        + _poly((0.014506, 4612.156534, 1.3915817, -0.00000044, -0.000029956, -0.0000000368), t) * _ARCSEC
        + math.pi
    )
    return gamma, np.array([_poly(row, t) * _ARCSEC for row in _DELAUNAY])


def solid_earth_station_displacement_m(
    station_m,
    sun_m,
    moon_m,
    *,
    gamma,
    delaunay,
    tide_system,
    earth_radius_m=6378136.6,
    sun_earth_system_mass_ratio=328900.56,
    earth_moon_mass_ratio=81.300596,
):
    """IERS degree 2/3 elastic and anelastic displacement at one epoch.

    ``tide_free`` includes permanent deformation; ``mean_tide`` removes it,
    because mean-tide station coordinates already contain that deformation.
    No implicit tide system is accepted.
    """
    if tide_system not in {"tide_free", "mean_tide"}:
        raise ValueError("Station tide system must be tide_free or mean_tide.")
    p, sun, moon = [np.asarray(v, dtype=float) for v in (station_m, sun_m, moon_m)]
    args = np.asarray(delaunay, dtype=float)
    constants = [gamma, earth_radius_m, sun_earth_system_mass_ratio, earth_moon_mass_ratio]
    if (
        any(v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) <= 0 for v in (p, sun, moon))
        or args.shape != (5,)
        or not np.all(np.isfinite(args))
        or not np.all(np.isfinite(constants))
        or min(constants[1:]) <= 0
    ):
        raise ValueError("Invalid station tide geometry or constants.")
    r, rho = np.linalg.norm(p), np.hypot(p[0], p[1])
    s, c = p[2] / r, rho / r
    sl, cl = (p[1] / rho, p[0] / rho) if rho else (0.0, 1.0)
    s2, c2 = 2 * s * c, c * c - s * s
    radial, north, east = p / r, np.array([-cl * s, -sl * s, c]), np.array([-sl, cl, 0.0])
    f = (p[2] ** 2 - 0.5 * rho**2) / r**2
    h2, l2 = 0.6078 - 0.0006 * f, 0.0847 + 0.0002 * f
    displacement = np.zeros(3)
    dr = dn = de = 0.0
    ratios = [sun_earth_system_mass_ratio * (1 + 1 / earth_moon_mass_ratio), 1 / earth_moon_mass_ratio]
    for body, ratio in zip((sun, moon), ratios, strict=True):
        b = np.linalg.norm(body)
        brho = np.hypot(body[0], body[1])
        bs, bc = body[2] / b, brho / b
        bsl, bcl = (body[1] / brho, body[0] / brho) if brho else (0.0, 1.0)
        sd, cd = sl * bcl - cl * bsl, cl * bcl + sl * bsl
        sd2, cd2 = 2 * sd * cd, cd * cd - sd * sd
        dot = float(body @ p) / (b * r)
        q2 = earth_radius_m**4 * ratio / b**3
        q3 = q2 * earth_radius_m / b
        R2, R3 = q2 * 3 * l2 * dot, q3 * 0.015 * (7.5 * dot**2 - 1.5)
        r2 = q2 * 0.5 * h2 * (3 * dot**2 - 1) - R2 * dot
        r3 = q3 * 0.292 * dot * (2.5 * dot**2 - 1.5) - R3 * dot
        displacement += (r2 + r3) * radial + (R2 + R3) * body / b
        bs2, bc_sq, p21, p22 = 2 * bs * bc, bc * bc, 3 * bs * bc, 3 * bc * bc
        dr += -0.75 * (-0.0025) * s2 * q2 * bs2 * sd - 0.75 * (-0.0022) * c * c * q2 * bc_sq * sd2
        dn += (
            -1.5 * (-0.0007) * c2 * q2 * bs2 * sd
            + 0.75 * (-0.0007) * s2 * q2 * bc_sq * sd2
            - 0.0012 * s * s * q2 * p21 * cd
            - 0.25 * 0.0024 * s2 * q2 * p22 * cd2
        )
        de += (
            -1.5 * (-0.0007) * s * q2 * bs2 * cd
            - 1.5 * (-0.0007) * c * q2 * bc_sq * cd2
            + 0.0012 * s * c2 * q2 * p21 * sd
            - 0.25 * 0.0024 * s2 * s * q2 * p22 * sd2
        )
    for multiplier, arguments, amplitudes in _frequency_terms("iers2010_tab7.3a.txt"):
        phase = multiplier * gamma - float(arguments @ args)
        sn, cs = math.sin(phase), math.cos(phase)
        rip, rop, tip, top = amplitudes
        dr += s2 * ((rip * sn + rop * cs) * cl + (-rop * sn + rip * cs) * sl)
        dn += c2 * ((tip * sn + top * cs) * cl + (-top * sn + tip * cs) * sl)
        de += s * ((-top * sn + tip * cs) * cl + (-tip * sn - top * cs) * sl)
    for multiplier, arguments, amplitudes in _frequency_terms("iers2010_tab7.3b.txt"):
        phase = multiplier * gamma - float(arguments @ args)
        sn, cs = math.sin(phase), math.cos(phase)
        rip, rop, tip, top = amplitudes
        dr += (1.5 * s * s - 0.5) * (rop * sn + rip * cs)
        dn += s2 * (top * sn + tip * cs)
    displacement += dr * radial + dn * north + de * east
    if tide_system == "mean_tide":
        factor = math.sqrt(1.25 / math.pi) * (-0.31460)
        displacement -= factor * h2 * f * radial + factor * 1.5 * l2 * s2 * north
    return displacement


def apply_solid_earth_station_tides(rows, *, environment, eop_path, tide_system):
    """Return copied rows with source-bound, observation-epoch station tides.

    The displacement is held fixed over the sub-second light-time legs. Orbit
    propagation and frame rotation retain their exact leg epochs. The input
    catalog must declare its tide system; loading is not inferred from it.
    """
    from sim.dynamics.orbit.epoch import resolve_sun_moon_positions
    from sim.dynamics.orbit.frames import _interp_eop, eci_to_ecef_rotation_hpop_like

    if tide_system not in {"tide_free", "mean_tide"}:
        raise ValueError("Station tide system must be tide_free or mean_tide.")
    hashes = {"earth_orientation": hashlib.sha256(Path(eop_path).read_bytes()).hexdigest()}
    for key in ("de440_coeff_path",):
        if key in environment:
            p = Path(environment[key]).resolve()
            hashes[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(_TABLES.glob("iers2010_tab7.3*.txt")):
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    result = deepcopy(list(rows))
    for row in result:
        if "slr_station_displacement" in row:
            raise ValueError("SLR station displacement is already applied.")
        jd = float(row["time_jd_utc"])
        offset = float(row.get("epoch_offset_s", 0.0))
        env = {**environment, "jd_utc_start": jd}
        sun, moon = resolve_sun_moon_positions(env, offset)
        rotation = eci_to_ecef_rotation_hpop_like(offset, jd_utc_start=jd, eop_path=str(eop_path))
        _, _, dut1, dat = _interp_eop(jd - 2400000.5 + offset / 86400.0, str(eop_path))
        gamma, args = station_tide_arguments(jd, dut1_s=dut1, tai_utc_s=dat, epoch_offset_s=offset)
        station = np.array(row["station_metadata"]["ecef_km"]) * 1000.0
        delta = solid_earth_station_displacement_m(
            station,
            rotation @ sun * 1000.0,
            rotation @ moon * 1000.0,
            gamma=gamma,
            delaunay=args,
            tide_system=tide_system,
        )
        row["station_metadata"]["ecef_km"] = ((station + delta) / 1000.0).tolist()
        row["slr_station_displacement"] = dict(
            model="iers2010_solid_earth.v1",
            tide_system=tide_system,
            displacement_ecef_m=delta.tolist(),
            reference_ecef_km=(station / 1000.0).tolist(),
            evaluation="observation_epoch_held_over_light_time",
            source_sha256=hashes,
        )
    return result
