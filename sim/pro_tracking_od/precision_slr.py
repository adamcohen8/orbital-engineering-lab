"""Precision ONP laser-range covariance through the maintained SLR observable.

Rows must already carry trusted station coordinates, weather and precise CRD
epochs. These functions do not acquire observations or estimate noise/offsets.
"""
from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.pro_slr_od.media import two_way_optical_media_correction
from sim.pro_tracking_od.precision import PrecisionOrbitPredictor
from sim.pro_tracking_od.precision_covariance import forecast_precision_covariance, project_precision_covariance
from sim.slr_tracking import predict_slr_two_way_time_of_flight_s


def precision_slr_range_km(row, state, *, predictor, context):
    """Two-way normal-point range (one-way equivalent), including media delay."""
    model = predictor.model
    time = (row["time_jd_utc"]-model.epoch_jd_utc)*86400.

    def local(offset):
        return predictor.propagator.propagate(state, float(offset)+row["epoch_offset_s"],time,
            np.zeros(3),predictor.environment,context)

    def at_jd(jd):
        return local((jd-row["time_jd_utc"])*86400.-row["epoch_offset_s"])

    geometry=predict_slr_two_way_time_of_flight_s(epoch_jd_utc=row["time_jd_utc"],epoch_event=row["epoch_event"],
        station_ecef_km=row["station_metadata"]["ecef_km"],state_at_jd_utc=at_jd,state_at_offset_s=local,
        epoch_offset_s=row["epoch_offset_s"],frame_model="hpop_like",eop_path=model.eop_path)
    media=two_way_optical_media_correction(row,geometry,at_jd,frame_model="hpop_like",eop_path=model.eop_path,state_at_offset_s=local)
    return float(geometry["predicted_two_way_range_km"]+media["one_way_equivalent_km"])


def precision_slr_covariance(solution, *, model, rows, observation_covariance, process_noise_joint_covariance=None,
                              state_steps=None):
    """Formal range likelihood retaining fitted orbit and session correlations.

    Native state sensitivities cover the full arc. Six-state measurement
    Jacobians include local light time and weather-dependent geometry. Radiation
    coefficients are fixed during each subsecond local measurement derivative;
    their direct subsecond force effect is negligible only within the declared
    precision error budget and must be checked for the intended application.
    Their full-arc effects and all fitted correlations are included.
    """
    if not rows:
        raise ValueError("Precision SLR covariance requires observations")
    row_times=np.array([(r["time_jd_utc"]-model.epoch_jd_utc)*86400. for r in rows])
    times, indices=np.unique(row_times,return_inverse=True)
    forecast=forecast_precision_covariance(solution,model=model,times_s=times,
        process_noise_joint_covariance=process_noise_joint_covariance)
    predictor=PrecisionOrbitPredictor(model,times)
    earth_radiation = (dict(earth_radiation_cr_area_mass_m2_kg=solution["earth_radiation_cr_area_mass_m2_kg"])
                       if model.earth_radiation else {})
    states=predictor.predict(solution["initial_state_eme2000_km_km_s"],solution["srp_cr_area_mass_m2_kg"],
        ecom_coefficients_km_s2=solution.get("ecom2_coefficients_km_s2"), **earth_radiation)
    context=OrbitContext(mu_km3_s2=model.mu_km3_s2,mass_kg=1.,area_m2=solution["srp_cr_area_mass_m2_kg"],cr=1.)
    steps=np.array([1e-4]*3+[1e-7]*3 if state_steps is None else state_steps,dtype=float)
    if steps.shape!=(6,) or np.any(~np.isfinite(steps)) or np.any(steps<=0):
        raise ValueError("Six positive finite SLR state sensitivity steps are required")
    predicted=[]
    jacobian=np.empty((len(rows),1,6))
    for i,(row,index) in enumerate(zip(rows,indices,strict=True)):
        state=states[index]
        predicted.append(precision_slr_range_km(row,state,predictor=predictor,context=context))
        for j,step in enumerate(steps):
            plus,minus=state.copy(),state.copy()
            plus[j]+=step
            minus[j]-=step
            jacobian[i,0,j]=(precision_slr_range_km(row,plus,predictor=predictor,context=context)
                            -precision_slr_range_km(row,minus,predictor=predictor,context=context))/(2*step)
    projected=project_precision_covariance(forecast,measurement_jacobians=jacobian,
        observation_covariance=observation_covariance,epoch_indices=indices)
    projected.update(predicted_range_km=predicted,
        residual_km=(np.array(predicted)-np.array([r["observed_two_way_range_km"] for r in rows])).tolist(),
        measurement_ids=[r["measurement_id"] for r in rows],state_steps=steps.tolist(),
        forecast=forecast,measurement_units="km",measurement_model="maintained_relative_light_time_and_optical_media",
        direct_radiation_light_time_sensitivity="fixed_coefficient_local_measurement_linearization")
    return projected
