"""Source-bound harmonic station loading from the public HARPOS contract.

Implements the IMLS-provided phase/frequency/acceleration and Cartesian site
coordinates directly. These are prescribed geophysical corrections, not
residual fits or replacements for orbit dynamics.
Reference: https://massloading.net/harpos_format.txt (2005.03.28).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


class HarmonicStationLoading:
    def __init__(self, path, *, expected_sha256, source_url, frame):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != expected_sha256:
            raise ValueError("Station loading source SHA256 mismatch.")
        text = raw.decode("ascii")
        lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
        if (
            not lines
            or " ".join(lines[0].split()) != "HARPOS Format version of 2005.03.28"
            or " ".join(lines[-1].split()) != " ".join(lines[0].split())
        ):
            raise ValueError("Station loading requires complete HARPOS2005 header and trailer.")
        if frame not in {"center_of_mass", "center_of_figure"} or not str(source_url).startswith("https://"):
            raise ValueError("Station loading requires explicit terrestrial origin and source URL.")
        self.identity = dict(
            sha256=self.sha256, source_url=source_url, origin=frame, model="harpos2005_harmonic_station_loading.v1"
        )
        self.harmonics = {}
        self.sites = {}
        self.coefficients = {}
        self.radius_m = None

        def number(value):
            return float(value.replace("D", "E").replace("d", "e"))

        for line in lines[1:-1]:
            if line[0] == "H":
                name = line[3:11].strip()
                if name in self.harmonics:
                    raise ValueError("Duplicate station-loading harmonic.")
                self.harmonics[name] = np.array([number(line[a:b]) for a, b in [(13, 26), (28, 47), (49, 59)]])
            elif line[0] == "A":
                if self.radius_m is not None:
                    raise ValueError("Duplicate station-loading applicability radius.")
                self.radius_m = number(line[3:17])
            elif line[0] == "S":
                site = line[3:11].strip()
                if site in self.sites:
                    raise ValueError("Duplicate station-loading site.")
                self.sites[site] = np.array([number(line[a:b]) for a, b in [(13, 26), (27, 40), (41, 54)]])
            elif line[0] == "D":
                harmonic, site = line[3:11].strip(), line[13:21].strip()
                key = (site, harmonic)
                if key in self.coefficients or harmonic not in self.harmonics or site not in self.sites:
                    raise ValueError("Invalid or duplicate station-loading coefficient identity.")
                self.coefficients[key] = np.array(
                    [number(line[a:b]) for a, b in [(24, 32), (33, 41), (42, 50), (53, 61), (62, 70), (71, 79)]]
                )
            else:
                raise ValueError("Unsupported station-loading HARPOS record.")
        if (
            not self.sites
            or not self.harmonics
            or self.radius_m is None
            or not math.isfinite(self.radius_m)
            or self.radius_m <= 0
        ):
            raise ValueError("Incomplete station-loading source.")
        arrays = [*self.harmonics.values(), *self.sites.values(), *self.coefficients.values()]
        if any(not np.all(np.isfinite(v)) for v in arrays):
            raise ValueError("Nonfinite station-loading source.")
        self.names = tuple(self.sites)
        self.coordinates = np.array([self.sites[name] for name in self.names])

    def displacement(self, station_ecef_m, *, tt_seconds_since_j2000):
        point = np.asarray(station_ecef_m, dtype=float)
        t = float(tt_seconds_since_j2000)
        if point.shape != (3,) or not np.all(np.isfinite(point)) or not math.isfinite(t) or np.linalg.norm(point) == 0:
            raise ValueError("Invalid station-loading geometry or epoch.")
        distance = np.linalg.norm(self.coordinates - point, axis=1)
        index = int(np.argmin(distance))
        site = self.names[index]
        if distance[index] > self.radius_m:
            raise ValueError("No loading site covers the requested station.")
        if sum(np.isclose(distance, distance[index], rtol=0.0, atol=1.0e-7)) > 1:
            raise ValueError("Ambiguous nearest loading site.")
        p = self.sites[site]
        r = np.linalg.norm(p)
        rho = np.hypot(p[0], p[1])
        s, c = p[2] / r, rho / r
        sl, cl = (p[1] / rho, p[0] / rho) if rho else (0.0, 1.0)
        basis = np.array([p / r, [-sl, cl, 0.0], [-cl * s, -sl * s, c]])
        components = np.zeros(3)
        for name, arguments in self.harmonics.items():
            if (site, name) not in self.coefficients:
                raise ValueError("Incomplete loading coefficients for selected site.")
            a = self.coefficients[(site, name)]
            phase = math.remainder(arguments[0] + arguments[1] * t + 0.5 * arguments[2] * t * t, 2 * math.pi)
            components += a[:3] * math.cos(phase) + a[3:] * math.sin(phase)
        return components @ basis, dict(
            **self.identity,
            source_site=site,
            site_distance_m=float(distance[index]),
            applicability_radius_m=self.radius_m,
        )


class GriddedStationLoading(HarmonicStationLoading):
    """One source-pinned IMLS FES2014b grid patch, materialized for one site.

    The provider NetCDF explicitly defines geodetic Up/East/North. Convert
    those interpolated coefficients into the existing HARPOS radial basis;
    the maintained harmonic/time evaluator remains the sole evaluation owner.
    No h5py, network access, extrapolation, or tracking-residual fitting occurs
    here. The small JSON patch and its byte-range provenance are retained by
    the acquisition workflow. The one-metre radius bounds reuse at the site.
    """

    def __init__(self, path, *, expected_sha256):
        from sim.utils.geodesy import ecef_to_geodetic_deg_km, enu_to_ecef_rotation

        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != expected_sha256:
            raise ValueError("Station loading grid patch SHA256 mismatch.")
        patch = json.loads(raw)
        if (
            patch.get("schema") != "oel.imls_grid_patch.v1"
            or patch.get("coefficient_frame") != "geodetic_up_east_north"
            or patch.get("reference_ellipsoid") != "WGS84"
        ):
            raise ValueError("Unsupported loading grid patch convention.")
        source = patch["source"]
        if (
            not str(source["grid_url"]).startswith("https://")
            or not isinstance(source["grid_etag"], str)
            or not source["grid_etag"]
            or source["grid_size"] <= 0
            or source.get("origin") != "center_of_mass"
        ):
            raise ValueError("Loading grid patch needs retained source identity.")
        lat, lon = np.asarray(patch["latitude_deg"], dtype=float), np.asarray(patch["longitude_deg"], dtype=float)
        point = np.asarray(patch["station_ecef_m"], dtype=float)
        coefficients = np.asarray(patch["coefficients_m"], dtype=float)
        names = patch["harmonic_names"]
        if (
            lat.shape != (2,)
            or lon.shape != (2,)
            or point.shape != (3,)
            or coefficients.shape != (44, 2, 3, 2, 2)
            or len(names) != 44
            or len(set(names)) != 44
            or set(names) != set(patch["harmonics"])
            or not np.all(np.isfinite(np.r_[lat, lon, point, coefficients.ravel()]))
            or not -90 <= lat[0] < lat[1] <= 90
            or not 0 <= lon[0] < lon[1] <= 360
            or np.linalg.norm(point) == 0
            or np.any(np.abs(coefficients) > 0.64)
        ):
            raise ValueError("Invalid station loading grid patch.")
        phi, lam, _ = ecef_to_geodetic_deg_km(point / 1000.0)
        lam %= 360.0
        if not lat[0] <= phi <= lat[1] or not lon[0] <= lam <= lon[1]:
            raise ValueError("Loading grid patch does not contain its station.")
        u, v = (phi - lat[0]) / (lat[1] - lat[0]), (lam - lon[0]) / (lon[1] - lon[0])
        weights = np.array([[(1 - u) * (1 - v), (1 - u) * v], [u * (1 - v), u * v]])
        uen = np.sum(coefficients * weights, axis=(-2, -1))
        geodetic_basis = enu_to_ecef_rotation(phi, lam)[:, [2, 0, 1]].T
        radius, rho = np.linalg.norm(point), np.hypot(point[0], point[1])
        s, c = point[2] / radius, rho / radius
        sl, cl = (point[1] / rho, point[0] / rho) if rho else (0.0, 1.0)
        radial_basis = np.array([point / radius, [-sl, cl, 0.0], [-cl * s, -sl * s, c]])
        radial = (uen @ geodetic_basis) @ radial_basis.T
        self.harmonics = {name: np.asarray(patch["harmonics"][name], dtype=float) for name in names}
        if any(a.shape != (3,) or not np.all(np.isfinite(a)) for a in self.harmonics.values()):
            raise ValueError("Invalid loading grid harmonic definitions.")
        site = str(patch["station_id"])
        if not site:
            raise ValueError("Loading grid patch needs a station identity.")
        self.sites, self.names, self.coordinates = {site: point}, (site,), point[None, :]
        self.radius_m = 1.0
        self.coefficients = {(site, name): radial[i].reshape(6) for i, name in enumerate(names)}
        self.identity = dict(
            sha256=self.sha256,
            source_url=source["grid_url"],
            source_etag=source["grid_etag"],
            origin="center_of_mass",
            model="imls_fes2014b_local_grid_patch.v1",
            interpolation="bilinear_geodetic_UEN_coefficients",
            source_coefficient_frame="geodetic_up_east_north",
            evaluation_coefficient_frame="geocentric_radial_east_north",
        )


def apply_harmonic_station_loading(rows, *, model, tai_utc_s):
    """Return copied rows with observation-epoch radial/east/north loading."""
    result = copy.deepcopy(list(rows))
    for row in result:
        if "slr_station_loading" in row:
            raise ValueError("Station loading is already applied.")
        point = np.array(row["station_metadata"]["ecef_km"]) * 1000.0
        t = (
            (float(row["time_jd_utc"]) - 2451545.0) * 86400.0
            + float(row.get("epoch_offset_s", 0.0))
            + float(tai_utc_s)
            + 32.184
        )
        delta, identity = model.displacement(point, tt_seconds_since_j2000=t)
        row["station_metadata"]["ecef_km"] = ((point + delta) / 1000.0).tolist()
        row["slr_station_loading"] = dict(
            **identity, displacement_ecef_m=delta.tolist(), evaluation="observation_epoch_held_over_light_time"
        )
    return result
