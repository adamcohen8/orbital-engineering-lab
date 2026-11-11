"""Opt-in raw-SLR precision models; legacy frozen predictions stay unchanged.

Nominal yaw is deliberately bounded away from the Galileo maneuver region.
This owner composes maintained ONP, light-time, media and station-tide models.
It does not establish a new inertial-frame realization or infer noise scales.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.epoch import resolve_sun_moon_positions
from sim.estimation.batch_least_squares import solve_batch_least_squares
from sim.estimation.parameters import EstimatedParameter, ParameterSet
from sim.estimation.weighting import prepare_covariance_whitener, whiten_residual_with_factor
from sim.pro_slr_od.media import two_way_optical_media_correction
from sim.pro_slr_od.station_displacement import apply_solid_earth_station_tides
from sim.pro_tracking_od.precision import SCHEMA, PrecisionOrbitPredictor
from sim.pro_tracking_od.precision_covariance import forecast_precision_covariance, project_precision_covariance
from sim.slr_tracking import predict_slr_two_way_time_of_flight_s
from sim.utils.io import read_regular_file_nofollow


@dataclass(frozen=True)
class NominalYawReflector:
    """IGS/ANTEX body-coordinate reflector vector relative to center of mass."""

    offset_body_m: tuple[float, float, float]
    source_sha256: str
    spacecraft: str
    minimum_abs_beta_deg: float = 4.1

    def __post_init__(self):
        v = np.asarray(self.offset_body_m, dtype=float)
        if (v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) > 10
                or not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.source_sha256)
                or not isinstance(self.spacecraft, str) or not self.spacecraft
                or isinstance(self.minimum_abs_beta_deg, bool) or not 0 < self.minimum_abs_beta_deg < 90):
            raise ValueError("Invalid source-bound nominal-yaw reflector")
        object.__setattr__(self, "offset_body_m", tuple(float(x) for x in v))

    def identity(self):
        return dict(model="igs_antex_nominal_yaw_reflector.v1", offset_body_m=list(self.offset_body_m),
            spacecraft=self.spacecraft, source_sha256=self.source_sha256,
            minimum_abs_beta_deg=self.minimum_abs_beta_deg, maneuver_region="unsupported_fail_closed")


def _sinex_date(value):
    if value == "0000:000:00000":
        return None
    year, doy, second = map(int, value.split(":"))
    if year < 1980 or not 1 <= doy <= 366 or not 0 <= second < 86400:
        raise ValueError("Invalid satellite metadata interval")
    date = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy-1, seconds=second)
    if date.year != year:
        raise ValueError("Invalid satellite metadata day")
    return date


def reflector_from_igs_metadata(path, *, spacecraft, start_utc, end_utc):
    """Resolve one valid Galileo LRA/CoM pair from authoritative IGS bytes.

    No missing coordinates or changes of CoM across the study are averaged.
    The IGS file explicitly uses IGS body axes and mechanical origins for Galileo.
    """
    raw = read_regular_file_nofollow(Path(path), min_bytes=1, max_bytes=2*1024*1024)
    if not spacecraft.startswith("E") or not start_utc.tzinfo or not end_utc.tzinfo or end_utc < start_utc:
        raise ValueError("Galileo geometry requires an explicit UTC study interval")
    block, centers, reflectors = "", [], []
    for line in raw.decode("ascii").splitlines():
        if line.startswith("+"):
            block = line[1:].strip()
        elif line.startswith("-"):
            block = ""
        f = line.split()
        if not f or f[0] != spacecraft:
            continue
        if block == "SATELLITE/COM":
            lo, hi = _sinex_date(f[1]), _sinex_date(f[2])
            if lo is not None and lo <= start_utc and (hi is None or end_utc < hi):
                centers.append(np.array(f[3:6], dtype=float))
        elif block == "SATELLITE/ECCENTRICITY" and f[2] == "L":
            reflectors.append(np.array(f[3:6], dtype=float))
    if len(centers) != 1 or len(reflectors) != 1:
        raise ValueError("One matching LRA and one CoM interval covering the complete study are required")
    return NominalYawReflector(tuple(reflectors[0]-centers[0]), hashlib.sha256(raw).hexdigest(), spacecraft)


def antenna_thrust_from_igs_metadata(path, *, spacecraft, start_utc, end_utc):
    """Bind one mass/transmit-power interval to nominal outward photon recoil.

    Metadata block means are not calibrated random-variable uncertainties.
    Neither EIRP nor an assumed error bar is substituted for total transmit power.
    """
    raw = read_regular_file_nofollow(Path(path), min_bytes=1, max_bytes=2*1024*1024)
    if (not isinstance(spacecraft, str) or not spacecraft.startswith("E")
            or not start_utc.tzinfo or not end_utc.tzinfo or end_utc < start_utc):
        raise ValueError("Galileo recoil requires an explicit UTC interval")
    block, records = "", {"SATELLITE/MASS": [], "SATELLITE/TX_POWER": []}
    for line in raw.decode("ascii").splitlines():
        if line.startswith("+"):
            block = line[1:].strip()
        elif line.startswith("-"):
            block = ""
        f = line.split()
        if block not in records or not f or f[0] != spacecraft:
            continue
        if len(f) < 4:
            raise ValueError("Malformed satellite mass or transmit-power row")
        lo, hi = _sinex_date(f[1]), _sinex_date(f[2])
        if lo is not None and lo <= start_utc and (hi is None or end_utc < hi):
            value = float(f[3])
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Mass and total transmit power must be finite and positive")
            records[block].append(value)
    if any(len(v) != 1 for v in records.values()):
        raise ValueError("One mass and one transmit-power interval must cover the complete study")
    mass, power = records["SATELLITE/MASS"][0], records["SATELLITE/TX_POWER"][0]
    return dict(mass_kg=mass, transmit_power_w=power, source_sha256=hashlib.sha256(raw).hexdigest(),
                acceleration_km_s2=power/(299792458.*mass)/1000.,
                direction="outward_radial_opposite_nadir_transmission",
                uncertainty="not_included_conditional_on_metadata_block_means")


def nominal_yaw_reflector_offset_km(state, sun_eci_km, reflector):
    """Nominal GNSS yaw: +Z nadir, +X toward the projected Sun (IGS axes)."""
    state, sun = np.asarray(state, dtype=float), np.asarray(sun_eci_km, dtype=float)
    if (state.shape != (6,) or sun.shape != (3,) or not np.all(np.isfinite(state))
            or not np.all(np.isfinite(sun)) or np.linalg.norm(state[:3]) <= 6378.137):
        raise ValueError("Invalid nominal-yaw geometry")
    norms = [np.linalg.norm(state[:3]), np.linalg.norm(state[3:]), np.linalg.norm(sun)]
    if not np.all(np.isfinite(norms)) or min(norms) <= 0:
        raise ValueError("Degenerate nominal-yaw geometry")
    normal = np.cross(state[:3], state[3:])
    if not np.isfinite(np.linalg.norm(normal)) or np.linalg.norm(normal) == 0:
        raise ValueError("Degenerate nominal-yaw geometry")
    beta = math.asin(float(np.clip(normal@sun/(np.linalg.norm(normal)*np.linalg.norm(sun)), -1, 1)))
    if abs(math.degrees(beta)) < reflector.minimum_abs_beta_deg:
        raise ValueError("Nominal-yaw reflector does not support the maneuver beta region")
    z = -state[:3]/np.linalg.norm(state[:3])
    y = np.cross(z, sun-state[:3])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.column_stack((x, y, z)) @ np.asarray(reflector.offset_body_m)/1000.


class PrecisionSlrObservationModel:
    """Prepared raw-SLR observables; fixed metadata, no observation selection."""

    def __init__(self, model, rows, reflector, *, station_tide_system, fixed_station_range_biases_km=None):
        if not rows or model.radiation_model != "ecom2":
            raise ValueError("Corrected precision SLR requires raw observations and ECOM2")
        for row in rows:
            if (row["correction_flags"]["center_of_mass_applied"]
                    or not 0 < row["observed_time_of_flight_s"] < 1):
                raise ValueError("Raw uncorrected reflector ranges with subsecond light time are required")
        self.model, self.reflector = model, reflector
        supplied = {} if fixed_station_range_biases_km is None else fixed_station_range_biases_km
        if not isinstance(supplied, Mapping):
            raise ValueError("Fixed residual station biases require an explicit station-to-km mapping")
        self.fixed_station_range_biases_km = {}
        for station, value in supplied.items():
            if (not isinstance(station, str) or len(station) != 4 or not station.isalnum()
                    or station != station.upper() or isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or abs(value) > .01):
                raise ValueError("Fixed residual station biases require four-character IDs and finite values within10m")
            self.fixed_station_range_biases_km[station] = float(value)
        self.times, self.indices = np.unique([(r["time_jd_utc"]-model.epoch_jd_utc)*86400.
                                             for r in rows], return_inverse=True)
        self.predictor = PrecisionOrbitPredictor(model, self.times)
        self.rows = apply_solid_earth_station_tides(rows, environment=self.predictor.environment,
            eop_path=model.eop_path, tide_system=station_tide_system)
        self.context = OrbitContext(mu_km3_s2=model.mu_km3_s2, mass_kg=1., area_m2=0., cr=1.)
        self.identity = dict(model="precision_nominal_yaw_lra_station_tides_shapiro.v1",
            reflector=reflector.identity(), station_tide_system=station_tide_system,
            station_tide_evaluation="observation_epoch_held_over_subsecond_light_time",
            station_displacement_resources=self.rows[0]["slr_station_displacement"]["source_sha256"],
            earth_shapiro_mu_km3_s2=model.mu_km3_s2,
            frame_realization="existing_native_iau76_80_precision_contract_not_new_GCRF_interchange",
            empirically_calibrated=False)
        if self.fixed_station_range_biases_km:
            self.identity.update(fixed_residual_station_range_biases_km=dict(sorted(self.fixed_station_range_biases_km.items())),
                station_bias_uncertainty="not_included_conditional_on_declared_constants")

    def range_km(self, row, state):
        model, predictor = self.model, self.predictor
        time = (row["time_jd_utc"]-model.epoch_jd_utc)*86400.
        environment = {**predictor.environment, "jd_utc_start": row["time_jd_utc"]}

        def local(offset):
            dt = float(offset)+row["epoch_offset_s"]
            com = predictor.propagator.propagate(state, dt, time, np.zeros(3),
                predictor.environment, self.context)
            sun, _ = resolve_sun_moon_positions(environment, dt)
            point = com.copy()
            point[:3] += nominal_yaw_reflector_offset_km(com, sun, self.reflector)
            return point

        def at_jd(jd):
            return local((jd-row["time_jd_utc"])*86400.-row["epoch_offset_s"])

        geometry = predict_slr_two_way_time_of_flight_s(epoch_jd_utc=row["time_jd_utc"],
            epoch_event=row["epoch_event"], station_ecef_km=row["station_metadata"]["ecef_km"],
            state_at_jd_utc=at_jd, state_at_offset_s=local, epoch_offset_s=row["epoch_offset_s"],
            frame_model="hpop_like", eop_path=model.eop_path, earth_shapiro_mu_km3_s2=model.mu_km3_s2)
        media = two_way_optical_media_correction(row, geometry, at_jd, frame_model="hpop_like",
            eop_path=model.eop_path, state_at_offset_s=local)
        return float(geometry["predicted_two_way_range_km"]+media["one_way_equivalent_km"]
                     +self.fixed_station_range_biases_km.get(row["station_id"], 0.))

    def predict(self, initial_state, coefficients, *, earth_radiation_cr_area_mass_m2_kg=None):
        states = self.predictor.predict(initial_state, 0., ecom_coefficients_km_s2=coefficients,
            earth_radiation_cr_area_mass_m2_kg=earth_radiation_cr_area_mass_m2_kg)
        return np.array([self.range_km(row, states[index])
                         for row, index in zip(self.rows, self.indices, strict=True)])


def fit_precision_slr(*, model, rows, reflector, station_tide_system, initial_state,
                      initial_ecom2_coefficients_km_s2, observation_covariance, max_nfev=100,
                      fixed_station_range_biases_km=None, initial_earth_radiation_cr_area_mass_m2_kg=.026):
    """Thirteen-parameter raw-range GLS, without fitted biases or variance scaling.

    Optional declared station offsets are fixed constants. Their uncertainty is
    explicitly excluded from this conditional fit; they are not inferred here.
    """
    if len(rows) < 21+int(model.earth_radiation):
        raise ValueError("Raw SLR fitting needs at least 13 parameters plus eight residual degrees of freedom")
    observation = PrecisionSlrObservationModel(model, rows, reflector, station_tide_system=station_tide_system,
        fixed_station_range_biases_km=fixed_station_range_biases_km)
    seed, coefficients = np.asarray(initial_state, dtype=float), np.asarray(initial_ecom2_coefficients_km_s2, dtype=float)
    if seed.shape != (6,) or coefficients.shape != (7,):
        raise ValueError("Six initial coordinates and seven ECOM2 coefficients are required")
    factor = prepare_covariance_whitener(observation_covariance, dimension=len(rows), field_name="raw SLR covariance")
    parameters = ParameterSet([
        *[EstimatedParameter("delta_"+k, 0., scale=1., lower=-100., upper=100., unit="km") for k in "xyz"],
        *[EstimatedParameter("delta_v"+k, 0., scale=.001, lower=-.1, upper=.1, unit="km/s") for k in "xyz"],
        *[EstimatedParameter("ecom2_"+k, v, scale=1e-11, lower=-1e-8, upper=1e-8, unit="km/s2")
          for k, v in zip(["D0", "Y0", "B0", "D2c", "D2s", "B1c", "B1s"], coefficients, strict=True)],
        *([EstimatedParameter("earth_radiation_cr_area_mass", initial_earth_radiation_cr_area_mass_m2_kg,
            scale=.01, lower=0., upper=1., unit="m2/kg")] if model.earth_radiation else []),
    ])
    if model.earth_radiation and (isinstance(initial_earth_radiation_cr_area_mass_m2_kg, bool)
            or not np.isfinite(initial_earth_radiation_cr_area_mass_m2_kg)
            or not 0 < initial_earth_radiation_cr_area_mass_m2_kg < 1):
        raise ValueError("Earth radiation seed must be interior in (0, 1) m2/kg")
    observed = np.array([r["observed_two_way_range_km"] for r in rows])

    def residual(values):
        return whiten_residual_with_factor(prediction(values)-observed, factor)

    def prediction(values):
        earth = dict(earth_radiation_cr_area_mass_m2_kg=values[-1]) if model.earth_radiation else {}
        return observation.predict(seed+values[:6], values[6:13], **earth)

    steps = [1e-2]*3+[1e-5]*3+[1e-12]*7+([1e-4] if model.earth_radiation else [])
    fit = solve_batch_least_squares(parameters, residual, max_nfev=max_nfev,
        xtol=1e-11, ftol=1e-11, gtol=1e-11, jacobian_step_native=steps,
        robust_loss="linear", sigma_clip_threshold=None, covariance_scale_policy="absolute")
    raw = prediction(fit.x_native)-observed
    result = dict(schema=SCHEMA, model=model.identity(), input_family="raw_slr",
        measurement_model=observation.identity, initial_state_eme2000_km_km_s=(seed+fit.x_native[:6]).tolist(),
        ecom2_coefficients_km_s2=fit.x_native[6:13].tolist(), srp_cr_area_mass_m2_kg=0.,
        covariance=None if fit.covariance_native is None else fit.covariance_native.tolist(),
        covariance_evidence=fit.covariance_evidence, covariance_coordinates="delta_state_and_ecom2",
        solver=dict(success=bool(fit.success), message=fit.message, nfev=fit.nfev,
                    initial_cost=fit.initial_cost, final_cost=fit.cost),
        diagnostics=fit.diagnostics, parameters=parameters.metadata(fit.x_native),
        training_times_s=observation.times.tolist(), training_count=len(rows),
        training_end_s=max((r["time_jd_utc"]-model.epoch_jd_utc)*86400.+r["epoch_offset_s"]
            +(r["observed_time_of_flight_s"] if r["epoch_event"]==2 else 0.) for r in rows),
        measurement_ids=[r["measurement_id"] for r in rows], range_rms_m=float(np.sqrt(np.mean(raw*raw)))*1000.,
        residual_km=raw.tolist(), jacobian_steps_native=steps,
        observation_covariance_sha256=hashlib.sha256(np.asarray(observation_covariance, dtype="<f8").tobytes()).hexdigest(),
        rejected_observations=[], empirically_calibrated=False)
    if model.earth_radiation:
        result["earth_radiation_cr_area_mass_m2_kg"] = float(fit.x_native[-1])
        result["covariance_coordinates"] += "_and_earth_radiation"
    return result


def corrected_slr_covariance(solution, *, model, rows, reflector, station_tide_system,
                             observation_covariance, process_noise_joint_covariance=None,
                             fixed_station_range_biases_km=None):
    """Joint corrected range covariance with the complete fitted 13-coordinate P."""
    observation = PrecisionSlrObservationModel(model, rows, reflector, station_tide_system=station_tide_system,
        fixed_station_range_biases_km=fixed_station_range_biases_km)
    if solution.get("measurement_model") != observation.identity:
        raise ValueError("Corrected SLR solution/measurement model mismatch")
    forecast = forecast_precision_covariance(solution, model=model, times_s=observation.times,
        process_noise_joint_covariance=process_noise_joint_covariance)
    observation.predictor.predict(solution["initial_state_eme2000_km_km_s"], 0.,
        ecom_coefficients_km_s2=solution["ecom2_coefficients_km_s2"],
        **(dict(earth_radiation_cr_area_mass_m2_kg=solution["earth_radiation_cr_area_mass_m2_kg"])
           if model.earth_radiation else {}))
    states = np.array([r["state_eci_km_km_s"] for r in forecast["records"]])
    h = np.empty((len(rows), 1, 6))
    predicted = []
    for i, (row, index) in enumerate(zip(observation.rows, observation.indices, strict=True)):
        state = states[index]
        predicted.append(observation.range_km(row, state))
        for j, step in enumerate([1e-4]*3+[1e-7]*3):
            plus, minus = state.copy(), state.copy()
            plus[j] += step
            minus[j] -= step
            h[i, 0, j] = (observation.range_km(row, plus)-observation.range_km(row, minus))/(2*step)
    result = project_precision_covariance(forecast, measurement_jacobians=h,
        observation_covariance=observation_covariance, epoch_indices=observation.indices)
    result.update(measurement_model=observation.identity, predicted_range_km=predicted,
        residual_km=(np.array(predicted)-np.array([r["observed_two_way_range_km"] for r in rows])).tolist(),
        measurement_ids=[r["measurement_id"] for r in rows], forecast=forecast,
        direct_radiation_light_time_sensitivity="fixed_coefficient_local_linearization_requires_error_budget")
    return result
