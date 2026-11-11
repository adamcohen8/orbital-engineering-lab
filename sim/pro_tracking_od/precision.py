"""Separate deterministic precision ONP batch OD for Earth-centered Cartesian data.

Uses maintained numerical forces and batch estimation; never alters OGP/SDP4.
The effective SRP coefficient is Cr*A/m, not separate recovery of mass or area.
Formal fit covariance does not establish empirical calibration.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.frames import FrameContext
from sim.dynamics.orbit.propagator import (
    OrbitPropagator,
    spherical_harmonics_plugin,
    srp_plugin,
    third_body_moon_plugin,
    third_body_sun_plugin,
)
from sim.dynamics.orbit.spherical_harmonics import load_hpop_ggm03_terms
from sim.estimation.batch_least_squares import solve_batch_least_squares
from sim.estimation.parameters import EstimatedParameter, ParameterSet
from sim.estimation.weighting import prepare_covariance_whitener, whiten_residual_with_factor
from sim.pro_perturbations.earth_radiation import EarthRadiationPressure
from sim.pro_perturbations.ecom import ECOM2Radiation
from sim.pro_perturbations.empirical import EmpiricalRICAcceleration
from sim.pro_perturbations.ocean_tides import OceanTides
from sim.pro_perturbations.schwarzschild import SchwarzschildAcceleration
from sim.pro_perturbations.solid_earth_tides import SolidEarthTides

SCHEMA = "oel.precision_onp_od.v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PrecisionOrbitModel:
    epoch_jd_utc: float
    gravity_path: str
    eop_path: str
    ephemeris_path: str | None = None
    gravity_degree: int = 20
    gravity_order: int = 20
    step_s: float = 15.0
    mu_km3_s2: float = 398600.4418
    gravity_radius_km: float = 6378.1363
    relativity: bool = False
    solid_earth_tides: bool = False
    radiation_model: str = "cannonball"
    earth_radiation: bool = False
    earth_radiation_quadrature_order: int = 32
    ocean_tides_path: str | None = None
    antenna_thrust_km_s2: float = 0.
    antenna_thrust_source_sha256: str | None = None

    def __post_init__(self):
        if (type(self.relativity) is not bool or type(self.solid_earth_tides) is not bool
                or type(self.earth_radiation) is not bool
                or self.radiation_model not in {"cannonball", "ecom2"}):
            raise ValueError("Precision model requires boolean force switches and cannonball or ecom2 radiation")
        if (type(self.earth_radiation_quadrature_order) is not int
                or not 8 <= self.earth_radiation_quadrature_order <= 128):
            raise ValueError("Precision Earth radiation quadrature requires an integer in [8, 128]")
        if (isinstance(self.antenna_thrust_km_s2, (bool, np.bool_))
                or not np.isfinite(self.antenna_thrust_km_s2) or not 0 <= self.antenna_thrust_km_s2 <= 1e-9):
            raise ValueError("Antenna recoil requires finite outward radial acceleration in [0, 1e-9] km/s2")
        if self.antenna_thrust_km_s2:
            digest = self.antenna_thrust_source_sha256
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError("Antenna recoil requires a source SHA256")
        elif self.antenna_thrust_source_sha256 is not None:
            raise ValueError("Disabled antenna recoil cannot declare a source")
        for name in ("epoch_jd_utc", "step_s", "mu_km3_s2", "gravity_radius_km"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Precision model {name} must be finite and positive")
        if not 2400000 < self.epoch_jd_utc < 2600000 or not .1 <= self.step_s <= 60:
            raise ValueError("Precision model requires an absolute modern epoch and step in [0.1, 60] s")
        if (any(isinstance(v, bool) or not isinstance(v, int) for v in (self.gravity_degree, self.gravity_order))
                or not 2 <= self.gravity_degree <= 20 or not 0 <= self.gravity_order <= self.gravity_degree):
            raise ValueError("Precision gravity requires integer degree 2..20 and order 0..degree")
        for path in (self.gravity_path, self.eop_path, *([self.ephemeris_path] if self.ephemeris_path else []),
                     *([self.ocean_tides_path] if self.ocean_tides_path else [])):
            if not Path(path).is_file() or Path(path).is_symlink():
                raise ValueError("Precision force resources must be regular trusted files")

        if self.ephemeris_path:
            with np.load(self.ephemeris_path, allow_pickle=False) as asset:
                if (str(asset["coordinate_contract"].item()) != "oel.de440.geocentric_moon.v1"
                        or any(int(asset[f"{body}_reference_center_id"].item()) != center
                               for body, center in [("earthmoon", 0), ("moon", 399), ("sun", 0)])):
                    raise ValueError("Precision DE440 requires explicit geocentric Moon provenance")

    def identity(self):
        settings = asdict(self)
        # Preserve identities of already-frozen models with the option absent.
        if not self.earth_radiation:
            settings.pop("earth_radiation")
            settings.pop("earth_radiation_quadrature_order")
        if self.ocean_tides_path is None:
            settings.pop("ocean_tides_path")
        else:
            settings.update(ocean_tides_sha256=_sha(self.ocean_tides_path), ocean_tides_degree=6,
                            ocean_pole_tide=True)
        if not self.antenna_thrust_km_s2:
            settings.pop("antenna_thrust_km_s2")
            settings.pop("antenna_thrust_source_sha256")
        else:
            settings.update(antenna_thrust_uncertainty="not_included_conditional_on_metadata",
                            antenna_direction="outward_radial_opposite_nadir_transmission")
        return dict(
            **settings, gravity_sha256=_sha(self.gravity_path), eop_sha256=_sha(self.eop_path),
            method="precision_onp_cartesian_srp", frame="EME2000", integrator="native_rk4",
            ephemeris="de440" if self.ephemeris_path else "analytic_enhanced",
            ephemeris_sha256=_sha(self.ephemeris_path) if self.ephemeris_path else None, sun=True, moon=True, srp=self.radiation_model + "_conical",
            drag=False, empirical_acceleration=False, empirical_radiation=self.radiation_model == "ecom2",
        )


class PrecisionOrbitPredictor:
    """Exact requested epochs on one forward native RK4 schedule; no interpolation."""

    def __init__(self, model: PrecisionOrbitModel, times_s):
        self.model = model
        self.identity = model.identity()
        self.times = np.asarray(times_s, dtype=float)
        if (self.times.ndim != 1 or not len(self.times) or not np.all(np.isfinite(self.times))
                or self.times[0] < 0 or np.any(np.diff(self.times) <= 0) or self.times[-1] > 32 * 86400):
            raise ValueError("Precision prediction requires distinct increasing forward epochs within 32 days")
        if np.sum(np.ceil(np.diff(np.r_[0., self.times]) / model.step_s)) > 1_000_000:
            raise ValueError("Precision schedule exceeds one million integration steps")
        widths, starts, ends = [], [], []
        previous = 0.0
        for time in self.times:
            count = int(np.ceil((time - previous) / model.step_s))
            for index in range(count):
                start = previous + index * model.step_s
                width = min(model.step_s, time - start)
                if width > 0:
                    starts.append(start)
                    widths.append(width)
            ends.append(len(widths))
            previous = float(time)
        self.widths, self.starts, self.ends = np.asarray(widths), np.asarray(starts), ends
        self.propagator = OrbitPropagator(integrator="rk4", numeric_backend="rust", plugins=[
            spherical_harmonics_plugin, third_body_sun_plugin, third_body_moon_plugin,
        ])
        if model.relativity:
            self.propagator.plugins.append(SchwarzschildAcceleration())
        if model.solid_earth_tides:
            self.propagator.plugins.append(SolidEarthTides(FrameContext(model="iau76_80_eop",
                jd_utc_start=model.epoch_jd_utc, eop_path=model.eop_path), "tide_free"))
        if model.ocean_tides_path:
            self.propagator.plugins.append(OceanTides(FrameContext(model="iau76_80_eop",
                jd_utc_start=model.epoch_jd_utc, eop_path=model.eop_path), model.ocean_tides_path))
        if model.antenna_thrust_km_s2:
            # The maintained RIC owner evaluates the instantaneous radial axis
            # at every stage; the source-bound recoil is prescribed, not fitted.
            self.propagator.plugins.append(EmpiricalRICAcceleration((model.antenna_thrust_km_s2, 0., 0.)))
        self.earth_radiation_index = len(self.propagator.plugins)
        if model.earth_radiation:
            self.propagator.plugins.append(EarthRadiationPressure(
                FrameContext(model="iau76_80_eop", jd_utc_start=model.epoch_jd_utc, eop_path=model.eop_path),
                quadrature_order=model.earth_radiation_quadrature_order, area_m2=0.))
        self.propagator.plugins.append(srp_plugin if model.radiation_model == "cannonball" else ECOM2Radiation())
        self.environment = dict(
            jd_utc_start=model.epoch_jd_utc, ephemeris_mode="de440" if model.ephemeris_path else "analytic_enhanced",
            de440_coeff_path=model.ephemeris_path, de440_eop_path=model.eop_path,
            spherical_harmonics_terms=[asdict(t) for t in load_hpop_ggm03_terms(
                model.gravity_path, max_degree=model.gravity_degree, max_order=model.gravity_order)],
            spherical_harmonics_reference_radius_km=model.gravity_radius_km,
            spherical_harmonics_frame_model="hpop_like", spherical_harmonics_eop_path=model.eop_path,
            eop_extrapolation="error", srp_shadow_model="conical",
            _precision_immutable_sampled=True,
        )

    def predict(self, initial_state, srp_area_mass_m2_kg, *, ecom_coefficients_km_s2=None,
                earth_radiation_cr_area_mass_m2_kg=None):
        state = np.asarray(initial_state, dtype=float)
        coefficient = float(srp_area_mass_m2_kg)
        if state.shape != (6,) or not np.all(np.isfinite(state)) or np.linalg.norm(state[:3]) <= 6378.137:
            raise ValueError("Precision initial state requires six finite EME2000 values above Earth")
        if not np.isfinite(coefficient) or not 0 <= coefficient <= 1:
            raise ValueError("Precision effective SRP Cr*A/m must be in [0, 1] m2/kg")
        if self.model.earth_radiation:
            value = earth_radiation_cr_area_mass_m2_kg
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Enabled Earth radiation requires explicit effective Cr*A/m in [0, 1] m2/kg")
            self.propagator.plugins[self.earth_radiation_index] = replace(
                self.propagator.plugins[self.earth_radiation_index], area_m2=float(value))
        elif earth_radiation_cr_area_mass_m2_kg is not None:
            raise ValueError("Disabled Earth radiation cannot accept a response coefficient")
        if self.model.radiation_model == "ecom2":
            if coefficient != 0 or ecom_coefficients_km_s2 is None:
                raise ValueError("ECOM2 replaces cannonball and requires seven explicit coefficients")
            self.propagator.plugins[-1] = ECOM2Radiation(ecom_coefficients_km_s2)
        elif ecom_coefficients_km_s2 is not None:
            raise ValueError("Cannonball prediction cannot accept ECOM2 coefficients")
        if self.model.identity() != self.identity:
            raise ValueError("Precision force resource drift")
        if not len(self.widths):
            return state[None, :].copy()
        history = self.propagator.try_propagate_sampled_steps(
            state, self.widths, [e for e in self.ends if e > 0], 0.0, np.zeros(3), self.environment,
            OrbitContext(mu_km3_s2=self.model.mu_km3_s2, mass_kg=1., area_m2=coefficient, cr=1.),
            step_times=self.starts,
        )
        if history is None:
            raise ValueError("Precision OD requires the supported native ONP sampled force path")
        positive = iter(history[1:])
        result = np.asarray([state if e == 0 else next(positive) for e in self.ends])
        if result.shape != (len(self.times), 6) or not np.all(np.isfinite(result)):
            raise ValueError("Precision ONP returned invalid states")
        return result


def fit_precision_orbit(*, model, times_s, positions_eme2000_km, initial_state,
                        velocities_eme2000_km_s=None, initial_srp_area_mass_m2_kg=0.026,
                        position_sigma_km=0.005, velocity_sigma_km_s=5e-6, max_nfev=150,
                        fit_observation_covariance=None, initial_ecom2_coefficients_km_s2=None,
                        jacobian_steps=None, initial_earth_radiation_cr_area_mass_m2_kg=.026):
    """Fit a physical initial state and the selected radiation coefficients on supplied data only."""
    predictor = PrecisionOrbitPredictor(model, times_s)
    positions = np.asarray(positions_eme2000_km, dtype=float)
    velocities = None if velocities_eme2000_km_s is None else np.asarray(velocities_eme2000_km_s, dtype=float)
    seed = np.asarray(initial_state, dtype=float)
    if (len(predictor.times) < 8 or positions.shape != (len(predictor.times), 3)
            or not np.all(np.isfinite(positions)) or seed.shape != (6,) or not np.all(np.isfinite(seed))
            or (velocities is not None and (velocities.shape != positions.shape or not np.all(np.isfinite(velocities))))):
        raise ValueError("Precision fitting requires eight or more finite Cartesian observations and a six-state seed")
    for sigma in (position_sigma_km, velocity_sigma_km_s):
        if isinstance(sigma, bool) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Precision working sigmas must be finite and positive")
    if (not np.isfinite(initial_srp_area_mass_m2_kg) or not 0 < initial_srp_area_mass_m2_kg < 1
            or isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1):
        raise ValueError("Precision SRP seed must be interior and max_nfev a positive integer")
    # Explicit full covariance uses observation-major native km/km-s rows.
    # Preserve the existing objective and arithmetic when this option is absent.
    dimension = len(positions) * (3 if velocities is None else 6)
    whitener = None if fit_observation_covariance is None else prepare_covariance_whitener(
        fit_observation_covariance, dimension=dimension, field_name="precision fit observation covariance")
    ecom_seed = [-4.56e-9*initial_srp_area_mass_m2_kg, 0., 0., 0., 0., 0., 0.]
    if initial_ecom2_coefficients_km_s2 is not None:
        supplied = np.asarray(initial_ecom2_coefficients_km_s2,dtype=float)
        if (model.radiation_model != "ecom2" or supplied.shape != (7,)
                or not np.all(np.isfinite(supplied)) or np.any(abs(supplied) >= 1e-8)):
            raise ValueError("ECOM2 seed requires seven finite interior coefficients and the ECOM2 model")
        ecom_seed = supplied.tolist()
    radiation_parameters = ([EstimatedParameter("ecom2_" + name, value, scale=1e-11,
        lower=-1e-8, upper=1e-8, unit="km/s2") for name, value in zip(
            ["D0", "Y0", "B0", "D2c", "D2s", "B1c", "B1s"],
            ecom_seed, strict=True)]
        if model.radiation_model == "ecom2" else [EstimatedParameter("srp_cr_area_mass",
            initial_srp_area_mass_m2_kg, scale=.01, lower=1e-8, upper=1., unit="m2/kg")])
    parameters = ParameterSet([
        *[EstimatedParameter("delta_" + k, 0., scale=1., lower=-100., upper=100., unit="km") for k in "xyz"],
        *[EstimatedParameter("delta_v" + k, 0., scale=.001, lower=-.1, upper=.1, unit="km/s") for k in "xyz"],
        *radiation_parameters,
        *([EstimatedParameter("earth_radiation_cr_area_mass", initial_earth_radiation_cr_area_mass_m2_kg,
            scale=.01, lower=0., upper=1., unit="m2/kg")] if getattr(model, "earth_radiation", False) else []),
    ])
    if getattr(model, "earth_radiation", False) and (isinstance(initial_earth_radiation_cr_area_mass_m2_kg, bool)
            or not np.isfinite(initial_earth_radiation_cr_area_mass_m2_kg)
            or not 0 < initial_earth_radiation_cr_area_mass_m2_kg < 1):
        raise ValueError("Earth radiation seed must be interior in (0, 1) m2/kg")

    def prediction(values):
        earth = dict(earth_radiation_cr_area_mass_m2_kg=values[-1]) if getattr(model, "earth_radiation", False) else {}
        return (predictor.predict(seed + values[:6], 0., ecom_coefficients_km_s2=values[6:13], **earth)
                if model.radiation_model == "ecom2" else predictor.predict(seed + values[:6], values[6], **earth))

    def residual(values):
        predicted = prediction(values)
        if whitener is not None:
            raw = (predicted[:, :3] - positions if velocities is None else
                   np.column_stack((predicted[:, :3] - positions, predicted[:, 3:] - velocities)))
            return whiten_residual_with_factor(raw.ravel(), whitener)
        errors = [(predicted[:, :3] - positions).ravel() / position_sigma_km]
        if velocities is not None:
            errors.append((predicted[:, 3:] - velocities).ravel() / velocity_sigma_km_s)
        return np.concatenate(errors)

    fit = solve_batch_least_squares(
        parameters, residual, max_nfev=max_nfev, xtol=1e-11, ftol=1e-11, gtol=1e-11,
        jacobian_step_native=([1e-4] * 3 + [1e-7] * 3 + ([1e-14]*7 if model.radiation_model == "ecom2" else [1e-5])
                              + ([1e-4] if getattr(model, "earth_radiation", False) else [])
                              if jacobian_steps is None else jacobian_steps),
        robust_loss="linear", sigma_clip_threshold=None, covariance_scale_policy="absolute",
    )
    predicted = prediction(fit.x_native)
    rms = float(np.sqrt(np.mean(np.sum((predicted[:, :3] - positions)**2, axis=1))) * 1000)
    report = dict(
        schema=SCHEMA, model=predictor.identity, initial_state_eme2000_km_km_s=(seed + fit.x_native[:6]).tolist(),
        srp_cr_area_mass_m2_kg=0. if model.radiation_model == "ecom2" else float(fit.x_native[6]),
        ecom2_coefficients_km_s2=fit.x_native[6:13].tolist() if model.radiation_model == "ecom2" else None,
        training_times_s=predictor.times.tolist(),
        training_end_s=float(predictor.times[-1]), training_count=len(predictor.times),
        observation_sha256=hashlib.sha256(np.column_stack((predictor.times, positions,
            *([velocities] if velocities is not None else []))).astype("<f8").tobytes()).hexdigest(),
        solver=dict(success=bool(fit.success), message=fit.message, nfev=fit.nfev,
                    initial_cost=fit.initial_cost, final_cost=fit.cost),
        position_rms_m=rms, parameters=parameters.metadata(fit.x_native), diagnostics=fit.diagnostics,
        covariance=None if fit.covariance_native is None else fit.covariance_native.tolist(),
        covariance_evidence=fit.covariance_evidence, covariance_coordinates="delta_state_and_" + model.radiation_model,
        working_position_sigma_km=position_sigma_km, working_velocity_sigma_km_s=velocity_sigma_km_s,
        velocities_used=velocities is not None, rejected_observations=[], empirically_calibrated=False,
    )
    if fit_observation_covariance is not None:
        report["fit_observation_covariance"] = dict(
            row_order="observation_major_position_xyz_then_velocity_xyz" if velocities is not None else "observation_major_position_xyz",
            dimension=dimension, units=["km"]*3 + (["km/s"]*3 if velocities is not None else []),
            sha256=hashlib.sha256(np.asarray(fit_observation_covariance, dtype="<f8").tobytes()).hexdigest(),
            uncertainty_status="caller_declared_formal_assumption_not_empirical_qualification",
        )
    if jacobian_steps is not None:
        report["jacobian_steps_native"] = np.asarray(jacobian_steps,dtype=float).tolist()
    if getattr(model, "earth_radiation", False):
        report["earth_radiation_cr_area_mass_m2_kg"] = float(fit.x_native[-1])
        report["covariance_coordinates"] += "_and_earth_radiation"
    return report


def predict_precision_orbit(solution, *, model, times_s, require_independent=True):
    """Predict exact frozen parameters. This function never calls an optimizer."""
    if solution.get("schema") != SCHEMA or model.identity() != solution["model"]:
        raise ValueError("Precision solution/model identity mismatch")
    if solution.get("process_policy_sha256") and require_independent:
        raise ValueError("Conditioned process solution requires precision_slr_process_candidate.conditioned_states")
    times = np.asarray(times_s, dtype=float)
    if require_independent and np.any(times <= solution["training_end_s"]):
        raise ValueError("Independent precision epochs must follow every fitted observation")
    return PrecisionOrbitPredictor(model, times).predict(
        solution["initial_state_eme2000_km_km_s"], solution["srp_cr_area_mass_m2_kg"],
        ecom_coefficients_km_s2=solution.get("ecom2_coefficients_km_s2"),
        **(dict(earth_radiation_cr_area_mass_m2_kg=solution["earth_radiation_cr_area_mass_m2_kg"])
           if getattr(model, "earth_radiation", False) else {}),
    )
