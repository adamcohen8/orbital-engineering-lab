"""Formal precision ONP uncertainty, using maintained native propagation.

No noise scale or missing-force uncertainty is inferred from forecast data.
Joint projections retain common orbit errors across observation epochs.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from sim.estimation.covariance import validate_covariance
from sim.estimation.orbit_od_covariance import propagate_joint_covariance, transport_process_covariance
from sim.pro_tracking_od.precision import SCHEMA, PrecisionOrbitPredictor


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def forecast_precision_covariance(solution, *, model, times_s, parameter_steps=None,
                                  process_noise_joint_covariance=None,
                                  require_independent=True):
    """Differentiate the complete fitted state/radiation vector without refitting.

    Optional process covariance has shape (6*N, 6*N), observation-major, EME2000
    km/km-s. It must already be mapped to the requested epochs and must retain
    its cross-time blocks. The caller must document its development provenance.
    Omitting it explicitly means a deterministic force model; it does not claim
    that unknown physical forces vanish. This function never fits a noise scale.
    """
    if solution.get("schema") != SCHEMA or model.identity() != solution.get("model"):
        raise ValueError("Precision solution/model identity mismatch")
    if solution.get("process_policy_sha256") and require_independent:
        raise ValueError("Conditioned process solution requires its joint precision_slr_process_candidate forecast")
    evidence = solution.get("covariance_evidence", {})
    if (not solution.get("solver", {}).get("success")
            or evidence.get("status") != "formal_local_gaussian"
            or evidence.get("scale_policy") != "absolute"
            or solution.get("rejected_observations")
            or solution.get("covariance") is None):
        raise ValueError("Precision forecast requires a successful unselected absolute Gaussian covariance")
    times = np.asarray(times_s, dtype=float)
    predictor = PrecisionOrbitPredictor(model, times)
    if require_independent and np.any(times <= solution["training_end_s"]):
        raise ValueError("Independent precision epochs must follow every fitted observation")
    radiation = (solution["ecom2_coefficients_km_s2"] if model.radiation_model == "ecom2"
                 else [solution["srp_cr_area_mass_m2_kg"]])
    values = np.r_[solution["initial_state_eme2000_km_km_s"], radiation]
    names = ["initial_" + k for k in ["x", "y", "z", "vx", "vy", "vz"]]
    names += (["ecom2_" + k for k in ["D0", "Y0", "B0", "D2c", "D2s", "B1c", "B1s"]]
              if model.radiation_model == "ecom2" else ["srp_cr_area_mass"])
    expected = "delta_state_and_" + model.radiation_model
    if getattr(model, "earth_radiation", False):
        values = np.r_[values, solution["earth_radiation_cr_area_mass_m2_kg"]]
        names.append("earth_radiation_cr_area_mass")
        expected += "_and_earth_radiation"
    if solution.get("covariance_coordinates") != expected:
        raise ValueError("Precision covariance coordinate contract mismatch")
    # Small sub-millimeter fit uncertainties require differences comfortably
    # above long-arc RK4 floating-point cancellation. These larger differences
    # are convergence-checked against fixed-parameter nonlinear predictions.
    steps = (np.r_[[1e-2]*3, [1e-5]*3, [1e-12]*7 if model.radiation_model == "ecom2" else [1e-5],
                   [1e-4] if getattr(model, "earth_radiation", False) else []]
             if parameter_steps is None else parameter_steps)

    def evaluate(parameters, unused_considers):
        earth = dict(earth_radiation_cr_area_mass_m2_kg=parameters[-1]) if getattr(model, "earth_radiation", False) else {}
        return (predictor.predict(parameters[:6], 0., ecom_coefficients_km_s2=parameters[6:13], **earth)
                if model.radiation_model == "ecom2" else predictor.predict(parameters[:6], parameters[6], **earth))

    q = None
    if process_noise_joint_covariance is not None:
        _matrix_budget(6 * len(times))
        q = validate_covariance(process_noise_joint_covariance, 6*len(times), name="joint process covariance")
    result = propagate_joint_covariance(
        evaluate, parameter_values=values, parameter_covariance=solution["covariance"],
        parameter_steps=steps, times_s=times, parameter_names=names,
        process_noise_covariances=None if q is None else np.array([
            q[6*i:6*i+6, 6*i:6*i+6] for i in range(len(times))]),
    )
    result.update(schema="oel.precision_onp_covariance.v1", frame="EME2000",
                  model=model.identity(), solution_sha256=_digest(solution),
                  parameter_values=values.tolist(), parameter_steps=np.asarray(steps).tolist(),
                  process_noise_joint_covariance=None if q is None else q.tolist(),
                  process_policy="explicit_joint_mapped_covariance" if q is not None else "deterministic_no_process_noise",
                  unknown_force_uncertainty_included=q is not None,
                  qualification_scope="formal_linearized_not_empirically_calibrated")
    result["content_sha256"] = _digest(result)
    return result


def _matrix_budget(dimension):
    if dimension < 1 or dimension > 4096:
        raise ValueError("Joint covariance requires 1..4096 coordinates; partition distinct arc units explicitly")


def precision_joint_process_covariance(times_s, state_transitions, *, start_time_s, local_covariance):
    """Joint mapped uncertainty for independent declared discrete increments.

    The grid must include the exact final assimilation epoch. The caller chooses
    and convergence-checks the increment grid; no process draws occur in fitting.
    State transitions come from the maintained ONP sensitivities, not Keplerian
    approximations. Pre-forecast process covariance is explicitly zero here.
    """
    times = np.asarray(times_s, dtype=float)
    _matrix_budget(6 * len(times))
    transitions = np.asarray(state_transitions, dtype=float)
    marginal = transport_process_covariance(times, transitions, start_time_s=start_time_s,
                                             local_covariance=local_covariance)
    n = len(times)
    joint = np.zeros((n, 6, n, 6))
    for i in range(n):
        joint[i, :, i, :] = marginal[i]
        if times[i] < start_time_s:
            continue
        for j in range(i + 1, n):
            phi = np.linalg.solve(transitions[i].T, transitions[j].T).T
            block = marginal[i] @ phi.T
            joint[i, :, j, :] = block
            joint[j, :, i, :] = block.T
    return joint.reshape(6*n, 6*n)


def project_precision_covariance(forecast, *, measurement_jacobians, observation_covariance, epoch_indices=None):
    """Project shared parameter/process errors into a joint measurement likelihood.

    Jacobians are (epochs, measurement-components, 6), differentiated through
    the maintained measurement owner. Observation covariance is the full joint
    matrix in those native measurement units and includes station/session terms.
    Never stack marginal state covariances as if successive orbit errors were
    independent. Full-state empirical qualification remains a separate test.
    """
    payload = {k: v for k, v in forecast.items() if k != "content_sha256"}
    if forecast.get("schema") != "oel.precision_onp_covariance.v1" or forecast.get("content_sha256") != _digest(payload):
        raise ValueError("Precision covariance forecast identity drift")
    h = np.asarray(measurement_jacobians, dtype=float)
    records = forecast["records"]
    if h.ndim != 3 or h.shape[2] != 6 or not np.all(np.isfinite(h)):
        raise ValueError("Measurement Jacobians require finite (epochs, components, 6) coordinates")
    n, components, _ = h.shape
    indices = np.arange(len(records)) if epoch_indices is None else np.asarray(epoch_indices)
    if (indices.shape != (n,) or np.any(~np.isfinite(indices)) or np.any(indices != np.floor(indices))
            or np.any(indices < 0) or np.any(indices >= len(records))):
        raise ValueError("One valid forecast epoch index is required per measurement row")
    indices = indices.astype(int)
    _matrix_budget(n * components)
    r = validate_covariance(observation_covariance, n*components, name="joint observation covariance")
    g = np.asarray([records[i]["sensitivity_native"] for i in indices])
    f = np.einsum("nij,njk->nik", h, g).reshape(n*components, -1)
    p = validate_covariance(forecast["parameter_covariance"], f.shape[1])
    state_term = f @ p @ f.T
    process_term = np.zeros_like(r)
    q = forecast.get("process_noise_joint_covariance")
    if q is not None:
        q = validate_covariance(q, 6*len(records), name="joint process covariance")
        coordinates = (6*indices[:,None]+np.arange(6)).ravel()
        q = q[np.ix_(coordinates,coordinates)]
        process_term = np.einsum("nia,namb,mjb->nimj", h, q.reshape(n, 6, n, 6), h).reshape(n*components, n*components)
    total = state_term + process_term + r
    total = (total + total.T) * .5
    # The measurement-space likelihood requires PD, even when state errors are
    # low rank. There is no numerical jitter or hidden covariance inflation.
    try:
        factor = np.linalg.cholesky(total)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Projected measurement covariance must be positive definite") from exc
    return dict(schema="oel.precision_onp_measurement_covariance.v1", covariance=total.tolist(),
                cholesky=factor.tolist(), parameter_component=state_term.tolist(),
                process_component=process_term.tolist(), observation_component=r.tolist(),
                row_order="observation_major", forecast_sha256=forecast["content_sha256"],
                measurement_jacobian_sha256=hashlib.sha256(h.astype("<f8").tobytes()).hexdigest(),
                empirically_calibrated=False)
