"""Batch companion to continuous tracking OD, using the same measurements/ONP."""

from __future__ import annotations

import numpy as np

from sim.estimation.batch_least_squares import solve_batch_least_squares
from sim.estimation.parameters import EstimatedParameter, ParameterSet
from sim.pro_tracking_od.continuous import validate_problem, write


def fit_tracking_batch(problem, *, output_path=None, max_nfev=40):
    _rows, groups, dynamics, measurement, _duplicates = validate_problem(problem)
    if dynamics.events or any(p.model != "constant" for p in dynamics.parameters):
        raise ValueError(
            "batch companion estimates constant parameters only; stochastic histories require sequential OD"
        )
    if np.any(dynamics.spectral_density):
        raise ValueError("batch companion does not substitute deterministic fitting for stochastic orbit dynamics")
    active = np.flatnonzero(dynamics.update_mask(0.0))
    initial = np.asarray(problem["initial_state"], dtype=float)
    covariance = np.asarray(problem["initial_covariance"])
    parameters = ParameterSet(
        EstimatedParameter(
            dynamics.state_names[i],
            initial[i],
            scale=max(np.sqrt(covariance[i, i]), dynamics.steps[i]),
            unit=dynamics.state_units[i],
            lower=1e-6 if i >= 6 and dynamics.parameters[i - 6].kind in {"cd_scale", "srp_scale"} else -np.inf,
        )
        for i in active
    )
    fit = [g for g in groups if g["time_s"] <= problem["fit_through_s"]]

    def residual(values):
        x = initial.copy()
        x[active] = values
        states = {}
        t = 0.0
        for row in fit:
            x = dynamics.propagate_mean(x, t, row["time_s"] - t)
            t = row["time_s"]
            states[t] = x.copy()
        return measurement.batch_residual(fit, states.__getitem__)

    result = solve_batch_least_squares(
        parameters,
        residual,
        max_nfev=max_nfev,
        covariance_scale_policy="absolute",
        jacobian_step_native=dynamics.steps[active],
    )
    report = {
        "schema": "oel.tracking_od_batch.v1",
        "success": bool(result.success),
        "parameter_names": parameters.names,
        "values": result.x_native.tolist(),
        "covariance": None if result.covariance_native is None else result.covariance_native.tolist(),
        "data_covariance": None if result.data_covariance_native is None else result.data_covariance_native.tolist(),
        "covariance_evidence": result.covariance_evidence,
        "residual": result.residual.tolist(),
        "physics": dynamics.identity(),
        "empirically_calibrated": False,
    }
    if output_path is not None:
        write(output_path, report)
    return report
