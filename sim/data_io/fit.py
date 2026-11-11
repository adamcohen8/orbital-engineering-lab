"""Local review-arc fitting. No dependencies on the simulator execution loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.dynamics.orbit.tle import ogp_mean_elements_from_mapping
from sim.estimation.ogp_od import fit_ogp_mean_elements_to_observations
from sim.frame_time import epoch_julian_date

from .catalog import CatalogRecord, encode_catalog, from_elements, parse_catalog
from .common import json_bytes, publish
from .contracts import DataIOError
from .review import read_review
from .time import at
from .trajectory import transform_trajectory


def fit_review_catalog(
    completed_run,
    *,
    template: CatalogRecord,
    output_dir,
    object_id=None,
    fit_until_s: float,
    max_position_rms_m: float,
    max_velocity_rms_m_s: float,
    formats=("omm-kvn", "tle"),
    eop_series=None,
    max_nfev=120,
    estimate_bstar=False,
):
    """Fit saved PV through the existing OGP estimator, then check encoded output.

    TLE quantization is evaluated on fit and untouched holdout samples. Source
    truth, estimator residuals, and encoded-product residuals remain distinct.
    """
    if any(not np.isfinite(v) or v <= 0 for v in (max_position_rms_m, max_velocity_rms_m_s)):
        raise DataIOError("Explicit positive finite position/velocity RMS acceptance limits are required.")
    if not formats or any(f not in {"omm-kvn", "omm-xml", "gp-json", "gp-csv", "tle"} for f in formats):
        raise DataIOError("Select supported fitted catalog output formats.")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or not 1 <= max_nfev <= 500:
        raise DataIOError("max_nfev must be an integer in [1,500].")
    selection = read_review(completed_run, object_id=object_id)
    trajectory = transform_trajectory(selection.trajectory, "TEME", eop_series=eop_series)
    if len(trajectory.times_s) > 10_000:
        raise DataIOError("Review fit is limited to 10,000 samples; select a bounded arc.")
    fit = trajectory.times_s <= float(fit_until_s)
    if np.count_nonzero(fit) < 6 or np.count_nonzero(~fit) < 3:
        raise DataIOError("Fitting requires at least six fit and three untouched holdout samples.")
    times = np.asarray([epoch_julian_date(at(trajectory.epoch, time), "UTC") for time in trajectory.times_s])
    report = fit_ogp_mean_elements_to_observations(
        times_jd_utc=times,
        positions_teme_km=trajectory.states[:, :3],
        velocities_teme_km_s=trajectory.states[:, 3:],
        template_elements=template.elements,
        fit_until_jd_utc=epoch_julian_date(at(trajectory.epoch, fit_until_s), "UTC"),
        estimate_bstar=estimate_bstar,
        max_nfev=int(max_nfev),
        propagation_backend="scalar_reference",
    )
    fitted = ogp_mean_elements_from_mapping(report["estimated_elements"])
    record = from_elements(
        fitted,
        object_name=trajectory.object_name or trajectory.object_id,
        object_id=template.fields.get("OBJECT_ID", ""),
    )
    from sim.dynamics.orbit.ogp import ogp_propagate_teme_batch_reference

    encoded_metrics, prepared = {}, {}
    gates = report["quality_gates"]
    accepted = all(gates[key] for key in ("solver_success", "data_full_rank", "regime_stable"))
    for format in formats:
        data = encode_catalog([record], format=format)
        encoded_elements = parse_catalog(data.decode(), format=format)[0].elements
        batch = ogp_propagate_teme_batch_reference([encoded_elements], (times - encoded_elements.epoch_jd_utc) * 1440)
        if not np.all(batch.success):
            raise DataIOError("Encoded mean-element product failed OGP propagation.")
        errors_r = (batch.position_teme_km[0] - trajectory.states[:, :3]) * 1000
        errors_v = (batch.velocity_teme_km_s[0] - trajectory.states[:, 3:]) * 1000
        metrics = {}
        for name, mask in (("fit", fit), ("holdout", ~fit)):
            pr = float(np.sqrt(np.mean(np.sum(errors_r[mask] ** 2, axis=1))))
            vr = float(np.sqrt(np.mean(np.sum(errors_v[mask] ** 2, axis=1))))
            metrics[name] = {"position_rms_m": pr, "velocity_rms_m_s": vr, "samples": int(np.count_nonzero(mask))}
            accepted = accepted and pr <= max_position_rms_m and vr <= max_velocity_rms_m_s
        encoded_metrics[format] = metrics
        prepared[format] = data
    evidence = {
        "schema": "oel.review-mean-element-fit.v1",
        "source": selection.receipt,
        "template_fields": template.fields,
        "fit_until_s": fit_until_s,
        "limits": {"position_rms_m": max_position_rms_m, "velocity_rms_m_s": max_velocity_rms_m_s},
        "estimator": report,
        "encoded_product_metrics": encoded_metrics,
        "status": "accepted" if accepted else "rejected",
        "simulation_executed": False,
        "non_claims": [
            "A fitted catalog model approximates the selected simulated arc; it is not an operational catalog publication.",
            "Residuals to simulator truth do not establish measured orbit accuracy or covariance calibration.",
        ],
    }
    root = Path(output_dir).expanduser().absolute()
    from sim.review import ReviewWorkspace

    with ReviewWorkspace.open(completed_run) as workspace:
        if workspace.evidence_identity()["sha256"] != selection.receipt["source_review_store"]["sha256"]:
            raise DataIOError("Source review store changed during mean-element fitting.")
    root.mkdir(parents=True, exist_ok=False)
    evidence_path = root / "fit-evidence.json"
    publish(
        evidence_path, json_bytes(evidence), receipt={"operation": "fit-review-catalog", "source": selection.receipt}
    )
    outputs = []
    if accepted:
        suffixes = {"tle": ".tle", "omm-kvn": ".omm", "omm-xml": ".omm.xml", "gp-json": ".gp.json", "gp-csv": ".gp.csv"}
        for format, data in prepared.items():
            outputs.append(
                publish(
                    root / ("fitted" + suffixes[format]),
                    data,
                    receipt={
                        "operation": "fit-review-catalog",
                        "format": format,
                        "source": selection.receipt,
                        "fit_evidence": str(evidence_path),
                        "encoded_metrics": encoded_metrics[format],
                    },
                )
            )
    return {"status": evidence["status"], "evidence_path": str(evidence_path), "outputs": outputs}
