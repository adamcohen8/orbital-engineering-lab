"""Focused regression checks for the 2026-10-09 orbital-force audit."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.epoch import datetime_to_julian_date
from sim.dynamics.orbit.jb2008_backend import jb2006_density, jb2008_density


def _write_jb2006_tables(root: Path, epoch: datetime) -> dict[str, str]:
    jd_floor = int(np.floor(datetime_to_julian_date(epoch)))
    sol_rows = []
    for offset in range(-6, 2):
        sol_rows.append(
            [
                0.0,
                0.0,
                float(jd_floor + offset),
                150.0 + offset,
                150.0,
                140.0 + offset,
                140.0,
                130.0 + offset,
                130.0,
                120.0 + offset,
                120.0,
            ]
        )
    ap_epoch = epoch - timedelta(hours=6.7)
    ap_row = [float(ap_epoch.year), float(ap_epoch.timetuple().tm_yday), 0.0, 0.0] + [4.0] * 8
    sol_path = root / "SOLFSMY-2006.txt"
    ap_path = root / "SOLRESAP-2006.txt"
    np.savetxt(sol_path, np.asarray(sol_rows), fmt="%.6f")
    np.savetxt(ap_path, np.asarray([ap_row, ap_row]), fmt="%.6f")
    return {"jb2006_sol_path": str(sol_path), "jb2006_ap_path": str(ap_path)}


def _write_jb2008_tables(root: Path, epoch: datetime) -> dict[str, str]:
    jd_floor = int(np.floor(datetime_to_julian_date(epoch) - 1.0))
    sol_row = [0.0, 0.0, float(jd_floor), 150.0, 150.0, 140.0, 140.0, 130.0, 130.0, 120.0, 120.0]
    dtc_row = [float(epoch.year), float(epoch.timetuple().tm_yday)] + [0.0] * 24
    sol_path = root / "SOLFSMY-2008.txt"
    dtc_path = root / "DTCFILE-2008.txt"
    np.savetxt(sol_path, np.asarray([sol_row, sol_row]), fmt="%.6f")
    np.savetxt(dtc_path, np.asarray([dtc_row, dtc_row]), fmt="%.6f")
    return {"jb2008_sol_path": str(sol_path), "jb2008_dtc_path": str(dtc_path)}


def test_jb2006_and_jb2008_reject_below_domain_and_accept_90_km(tmp_path: Path) -> None:
    """The 90 km quadrature boundary is finite; lower inputs fail clearly."""

    epoch = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    cases = (
        (jb2006_density, _write_jb2006_tables(tmp_path, epoch), "jb2006"),
        (jb2008_density, _write_jb2008_tables(tmp_path, epoch), "jb2008"),
    )
    for density_model, env, name in cases:
        rho_90 = density_model(90.0, 10.0, 20.0, epoch, env)
        assert np.isfinite(rho_90), name
        for altitude_km in (0.0, 80.0, 89.999):
            try:
                density_model(altitude_km, 10.0, 20.0, epoch, env)
            except ValueError as exc:
                assert "at or above 90 km" in str(exc), name
            else:
                raise AssertionError(f"{name} accepted unsupported altitude {altitude_km} km")

        # Domain validation precedes weather-table lookup, so an unsupported
        # state remains diagnosable even when no model tables are configured.
        try:
            density_model(80.0, 10.0, 20.0, epoch, {})
        except ValueError as exc:
            assert "at or above 90 km" in str(exc), name
        else:
            raise AssertionError(f"{name} did not reject unsupported altitude without tables")
