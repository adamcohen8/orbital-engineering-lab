"""Private adapter for exact-schedule native SLR prediction and sensitivities.

Unsupported models retain the reference implementation. Resource loading,
frame/time policy, CRD semantics and fitted-parameter policy stay Python-owned.
"""
from __future__ import annotations

import math

import numpy as np

from sim.dynamics.orbit.frames import eci_to_ecef_rotation_hpop_like
from sim.pro_slr_od.media import two_way_optical_media_correction
from sim.pro_slr_od.reflector import center_of_mass_prediction_offset
from sim.rust_orbit_backend import _extension
from sim.utils.geodesy import ecef_to_enu_rotation, ecef_to_geodetic_deg_km


def _pack(values):
    return np.ascontiguousarray(values, dtype='<f8').tobytes()


class SlrObservationContext:
    """One immutable observation layout; force/resources refreshed per call."""
    def __init__(self, rows, *, initial_time_s, propagator, context, config,
                 environment, frame_model, eop_path, range_correction_model, relative_light_time=False, earth_shapiro_mu_km3_s2=0.):
        self.rows = rows
        self.propagator, self.context = propagator, context
        self.environment = dict(environment)
        self.frame_model, self.eop_path = frame_model, eop_path
        self.media = range_correction_model is two_way_optical_media_correction
        self.epoch_offsets = {}
        prepared = []
        for row in rows:
            offset = float(row.get('epoch_offset_s', 0.))
            if not math.isfinite(offset) or (offset != 0. and not relative_light_time):
                raise ValueError('Precise SLR epoch offsets require relative light time and finite seconds.')
            jd = float(row['time_jd_utc'])
            if jd in self.epoch_offsets and self.epoch_offsets[jd] != offset:
                raise ValueError('Distinct precise SLR epochs collide in the native scalar-JD frame key.')
            self.epoch_offsets[jd] = offset
            station = np.asarray(row['station_metadata']['ecef_km'], dtype=float)
            lat, lon, height = ecef_to_geodetic_deg_km(station)
            enu = ecef_to_enu_rotation(lat, lon)
            applied = bool(row.get('correction_flags', {}).get('tropospheric_refraction_applied', False))
            conditions = row['slr_media'] if self.media and not applied else {}
            prepared.append([float(row['time_s']), float(row['time_jd_utc']), float(row['epoch_event']),
                *station, math.radians(lat), height * 1000., *enu.ravel(), float(self.media and not applied),
                *[float(conditions.get(key, 0.)) for key in ('pressure_mbar', 'temperature_k',
                    'relative_humidity_percent', 'wavelength_nm')]])
        self.shapiro_mu = float(earth_shapiro_mu_km3_s2)
        self.output_width = 10 if self.shapiro_mu else 9
        self.relative_light_time = bool(relative_light_time)
        args = (_pack(prepared), float(initial_time_s), float(config.integration_substep_s))
        self.native = (_extension().SLRObservationContext(*args, True, self.shapiro_mu) if self.shapiro_mu else
                       _extension().SLRObservationContext(*args, True) if self.relative_light_time
                       else _extension().SLRObservationContext(*args))

    def _plan(self, state):
        from sim.dynamics.orbit.rust_force_plan import force_codes, immutable_builtin_plan_signature, make_plan
        *_, stages, force = make_plan(self.propagator, np.asarray(state),
            float(self.rows[0]['time_s']), self.environment, self.context)
        # Share only immutable built-in time metadata. Drag always evaluates
        # density and relative velocity at every current trial state.
        shared = (2 not in force_codes(self.propagator, self.environment)
                  and immutable_builtin_plan_signature(self.propagator, self.environment, self.context) is not None)
        return force, stages, shared

    def _frame(self, jd, midnight):
        return eci_to_ecef_rotation_hpop_like((jd - midnight) * 86400., jd_utc_start=midnight,
            eop_path=self.eop_path if self.frame_model == 'hpop_like' else None).ravel().tolist()

    def _relative_frame(self, epoch_jd, offset_s):
        return eci_to_ecef_rotation_hpop_like(float(offset_s) + self.epoch_offsets[float(epoch_jd)], jd_utc_start=float(epoch_jd),
            eop_path=self.eop_path if self.frame_model == 'hpop_like' else None).ravel().tolist()

    def _biases(self, bias_maps):
        return [[float(biases.get(str(row['station_id']), 0.)) + center_of_mass_prediction_offset(row) for row in self.rows] for biases in bias_maps]

    def evaluate_batch(self, states, bias_maps):
        values = np.asarray(states, dtype=float)
        if values.ndim != 2 or values.shape[1] != 6 or len(bias_maps) != len(values):
            raise ValueError('SLR native trials require (trials, 6) states and one bias map per trial.')
        force, stages, shared = self._plan(values[0])
        raw = self.native.evaluate_bytes(_pack(values), _pack(self._biases(bias_maps)), force, stages, self._relative_frame if self.relative_light_time else self._frame, shared)
        self.propagator.last_numeric_path = 'rust_slr_observation_batch'
        return np.frombuffer(raw, dtype='<f8').copy().reshape(len(values), len(self.rows), self.output_width)

    def predict(self, state, biases, *, correction_audit=None):
        result = self.evaluate_batch(np.asarray(state)[None, :], [biases])[0]
        self._audit(result, correction_audit)
        return result[:, 0]

    def sensitivities(self, state, biases, *, steps=(1.e-3, 1.e-3, 1.e-3, 1.e-6, 1.e-6, 1.e-6)):
        force, stages, shared = self._plan(state)
        raw, derivative = self.native.central_sensitivities_bytes(_pack(state),
            _pack(self._biases([biases])[0]), _pack(steps), force, stages, self._relative_frame if self.relative_light_time else self._frame, shared)
        result = np.frombuffer(raw, dtype='<f8').copy().reshape(len(self.rows), self.output_width)
        self.propagator.last_numeric_path = 'rust_slr_observation_central_sensitivities'
        return result[:, 0], np.frombuffer(derivative, dtype='<f8').copy().reshape(len(self.rows), 6)

    def _audit(self, result, audit):
        if audit is None or not (self.media or self.shapiro_mu or any("slr_center_of_mass" in r for r in self.rows)):
            return
        for row, values in zip(self.rows, result, strict=True):
            entry = {'measurement_id': str(row['measurement_id']),
                     'one_way_equivalent_km': float(values[6] / 1000.), 'troposphere_m': float(values[6])}
            if self.media and not bool(row.get('correction_flags', {}).get('tropospheric_refraction_applied', False)):
                entry.update(transmit_elevation_rad=float(values[7]), receive_elevation_rad=float(values[8]))
            if self.shapiro_mu:
                entry["earth_shapiro_one_way_equivalent_km"] = float(values[9])
            if "slr_center_of_mass" in row:
                entry["center_of_mass_prediction_offset_km"] = center_of_mass_prediction_offset(row)
            audit.append(entry)


def try_slr_context(rows, *, initial_time_s, propagator, context, config, environment,
                    frame_model, eop_path, range_correction_model, prediction_backend='auto', relative_light_time=False, earth_shapiro_mu_km3_s2=0.):
    if prediction_backend not in {'auto', 'python', 'rust'}:
        raise ValueError('prediction_backend must be auto, python, or rust.')
    if not isinstance(relative_light_time, bool):
        raise ValueError('relative_light_time must be boolean.')
    mu = float(earth_shapiro_mu_km3_s2)
    if not math.isfinite(mu) or mu < 0. or (mu > 0. and not relative_light_time):
        raise ValueError('Earth Shapiro correction requires nonnegative finite GM and relative light time.')
    if prediction_backend == 'python':
        return None
    offsets = {}
    for row in rows:
        jd, offset = float(row['time_jd_utc']), float(row.get('epoch_offset_s', 0.))
        if jd in offsets and offsets[jd] != offset:
            # The native callback key contains a scalar JD. The scalar
            # measurement owner can retain distinct sub-JD row identities.
            if prediction_backend == 'rust':
                raise ValueError('Distinct precise SLR epochs collide in the native scalar-JD frame key.')
            return None
        offsets[jd] = offset
    from sim.dynamics.orbit.propagator import OrbitPropagator
    from sim.dynamics.orbit.rust_force_plan import force_codes, native_force_plan_codes
    supported = (type(propagator) is OrbitPropagator and propagator.numeric_backend == 'rust'
        and propagator.integrator == 'rk4' and propagator.model == 'two_body'
        and propagator.state_frame == 'eci' and environment is not None and 0 < len(rows) <= 1000
        and frame_model in {'simple', 'hpop_like'}
        and (frame_model != 'hpop_like' or eop_path is not None)
        and (range_correction_model is None or range_correction_model is two_way_optical_media_correction)
        and native_force_plan_codes(propagator, environment) is not None)
    if supported and 1 in force_codes(propagator, environment):
        # Mirror the sampled-history owner's harmonic admission before making
        # a native context. Valid unnormalized/high-degree models retain their
        # established scalar callback route under automatic selection.
        from sim.dynamics.orbit.spherical_harmonics import parse_spherical_harmonic_terms
        from sim.rust_orbit_backend import native_harmonic_degree_limit
        harmonic = environment.get('_compiled_spherical_harmonics_terms')
        limit = native_harmonic_degree_limit()
        if harmonic is None:
            terms = environment.get('_parsed_spherical_harmonics_terms')
            if terms is None:
                terms = parse_spherical_harmonic_terms(environment.get('spherical_harmonics_terms'))
            supported = bool(terms) and all(term.normalized and max(term.n, term.m) <= limit for term in terms)
        else:
            supported = harmonic.all_normalized and max(harmonic.n_max, harmonic.m_max) <= limit
    if mu and getattr(_extension(), 'SLR_EARTH_SHAPIRO_VERSION', 0) != 1:
        supported = False
    if relative_light_time and not getattr(_extension(), 'SLR_RELATIVE_LIGHT_TIME_SUPPORTED', False):
        supported = False
    if supported and hasattr(_extension(), 'SLRObservationContext'):
        return SlrObservationContext(rows, initial_time_s=initial_time_s, propagator=propagator,
            context=context, config=config, environment=environment, frame_model=frame_model,
            eop_path=eop_path, range_correction_model=range_correction_model, relative_light_time=relative_light_time, earth_shapiro_mu_km3_s2=mu)
    if prediction_backend == 'rust':
        raise ValueError('SLR native batching requires a supported RK4 force plan and SLRObservationContext symbol.')
    return None
