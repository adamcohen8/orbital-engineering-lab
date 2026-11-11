"""Normalize retained ILRS sources for the continuous precision-SLR workflow."""

from __future__ import annotations

import copy
from pathlib import Path

from sim.pro_slr_od.media import attach_slr_media_inputs
from sim.slr_tracking import adapt_ilrs_crd_normal_points, parse_ilrs_crd_v2_normal_points
from sim.utils.io import sha256_regular_file_nofollow


def prepare_slr_rows(
    crd_path, station_catalog, *, epoch_jd_utc=None, correction_policy=None, sigma_floor_ps=6671.281904
):
    """Keep original correction flags; missing effects stay explicitly unmodeled.

    ``correction_policy`` supplies source-bound target/instrument/eccentricity
    corrections. It cannot overwrite a CRD-applied troposphere disposition.
    """
    path = Path(crd_path).absolute()
    packet = parse_ilrs_crd_v2_normal_points(path)
    adapted = adapt_ilrs_crd_normal_points(packet, station_catalog=station_catalog, sigma_floor_ps=sigma_floor_ps)
    if adapted.get("omitted_sessions"):
        raise ValueError("SLR source has unsupported observations; review exclusions explicitly")
    rows = attach_slr_media_inputs(adapted["measurement_rows"], packet)
    if not rows:
        raise ValueError("SLR source contains no supported normal points")
    epoch = float(epoch_jd_utc if epoch_jd_utc is not None else rows[0]["time_jd_utc"])
    for row in rows:
        row["kind"] = "slr"
        row["time_s"] = (row["time_jd_utc"] - epoch) * 86400.0
        row["measurement_id"] = packet["source_sha256"] + ":" + row["measurement_id"]
        row["corrections"] = {
            name: {"status": "unmodeled"}
            for name in ("center_of_mass", "target_signature", "station_eccentricity", "instrument")
        }
        policy = copy.deepcopy((correction_policy or {}).get(row["station_id"], {}))
        if set(policy) - set(row["corrections"]):
            raise ValueError("SLR policy may only supply target/instrument/eccentricity corrections")
        row["corrections"].update(policy)
        if row["correction_flags"].get("center_of_mass_applied"):
            if policy.get("center_of_mass", {}).get("status") == "model":
                raise ValueError("cannot model a CRD-applied target center-of-mass correction twice")
            row["corrections"]["center_of_mass"] = {"status": "applied", "source_sha256": packet["source_sha256"]}
        row["corrections"]["troposphere"] = {
            "status": "applied" if row["correction_flags"].get("tropospheric_refraction_applied") else "model",
            "source_sha256": packet["source_sha256"],
        }
        original_station = station_catalog[row["station_id"]]
        for key in ("velocity_ecef_km_year", "reference_epoch_jd_utc"):
            if key in original_station:
                row["station_metadata"][key] = copy.deepcopy(original_station[key])
    digest, size = sha256_regular_file_nofollow(path, max_bytes=64 * 1024 * 1024)
    return {
        "epoch_jd_utc": epoch,
        "rows": rows,
        "source": {"path": str(path), "sha256": digest, "bytes": size, "role": "ILRS_CRD_observations"},
        "unmodeled_effects_require_review": True,
    }
