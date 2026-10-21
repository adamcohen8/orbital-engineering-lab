//! OEL numerical core: ONP ECI and CR3BP rotating barycentric states.
//! States use km and km/s; times use seconds.

#[cfg(feature = "python")]
mod actuation;
pub mod adaptive;
pub mod attitude;
pub mod attitude_ekf;
pub mod communications;
pub mod conjunction;
pub mod conjunction_probability;
pub mod collisions;
pub mod control;
pub mod covariance;
pub mod covariance_orbit;
pub mod coverage;
pub mod cr3bp;
pub mod environment;
pub mod estimation;
pub mod force_context;
#[cfg(feature = "python")]
mod stage_context;
#[cfg(feature = "python")]
mod analytic_bodies;
pub mod force_plan;
pub mod planetary;
pub mod additional_gravity;
#[cfg(feature = "python")]
mod additional_gravity_binding;
#[cfg(feature = "python")]
mod planetary_binding;

pub mod harmonics;
pub mod iod;
pub mod links;
pub mod measurements;
pub mod optical_collection;
pub mod mechanical;
pub mod mission_schedule;
#[path = "vendor/brahe_nrlmsise00/data.rs"]
mod nrlmsise00_coeff;
#[path = "vendor/brahe_nrlmsise00/model.rs"]
#[allow(dead_code)] // Upstream core retains its public constructors for source parity.
mod nrlmsise00_model;
mod nrlmsise00_native;
mod nrlmsise00_stage;
#[cfg(feature = "python")]
mod numeric_bytes;
#[cfg(feature = "python")]
pub mod output;
pub mod ogp;
pub mod passive;
pub mod perturbations;
pub mod precision_forces;
pub mod precision_plan;
pub mod solid_tides;
pub mod ocean_tides;
pub mod relative;
pub mod resources;
pub mod spacecraft;
pub mod targeting;
#[cfg(feature = "python")]
mod tasking_search;
pub mod target_coast;
pub mod target_rollouts;
pub mod tracking;
pub mod vehicle;
pub mod zonal;

const EARTH_RADIUS_KM: f64 = 6378.137;
const EARTH_J2: f64 = 1.08262668e-3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub position_km: [f64; 3],
    pub velocity_km_s: [f64; 3],
}

impl State {
    pub fn from_array(values: [f64; 6]) -> Result<Self, &'static str> {
        if !values.iter().all(|value| value.is_finite()) {
            return Err("state must contain only finite values");
        }
        let position_km = [values[0], values[1], values[2]];
        if position_km.iter().all(|value| *value == 0.0) {
            return Err("position must be nonzero");
        }
        Ok(Self {
            position_km,
            velocity_km_s: [values[3], values[4], values[5]],
        })
    }

    pub fn as_array(self) -> [f64; 6] {
        [
            self.position_km[0],
            self.position_km[1],
            self.position_km[2],
            self.velocity_km_s[0],
            self.velocity_km_s[1],
            self.velocity_km_s[2],
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OrbitParameters {
    pub mu_km3_s2: f64,
    pub include_j2: bool,
    pub command_accel_eci_km_s2: [f64; 3],
}

impl OrbitParameters {
    fn validate(self) -> Result<(), &'static str> {
        if !self.mu_km3_s2.is_finite() || self.mu_km3_s2 <= 0.0 {
            return Err("mu_km3_s2 must be finite and positive");
        }
        if !self
            .command_accel_eci_km_s2
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("command acceleration must contain only finite values");
        }
        Ok(())
    }
}

fn derivative(state: [f64; 6], parameters: OrbitParameters) -> [f64; 6] {
    let x = state[0];
    let y = state[1];
    let z = state[2];
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let scale = -parameters.mu_km3_s2 / (r * r2);
    let mut acceleration = [
        scale * x + parameters.command_accel_eci_km_s2[0],
        scale * y + parameters.command_accel_eci_km_s2[1],
        scale * z + parameters.command_accel_eci_km_s2[2],
    ];
    if parameters.include_j2 {
        let f =
            1.5 * EARTH_J2 * parameters.mu_km3_s2 * (EARTH_RADIUS_KM * EARTH_RADIUS_KM) / r.powi(5);
        let g = 5.0 * (z * z) / r2;
        acceleration[0] += f * x * (g - 1.0);
        acceleration[1] += f * y * (g - 1.0);
        acceleration[2] += f * z * (g - 3.0);
    }
    [
        state[3],
        state[4],
        state[5],
        acceleration[0],
        acceleration[1],
        acceleration[2],
    ]
}

fn stage(state: [f64; 6], derivative: [f64; 6], scale: f64) -> [f64; 6] {
    std::array::from_fn(|index| state[index] + scale * derivative[index])
}

pub fn rk4_step(
    state: State,
    dt_s: f64,
    parameters: OrbitParameters,
) -> Result<State, &'static str> {
    parameters.validate()?;
    if !dt_s.is_finite() {
        return Err("dt_s must be finite");
    }
    rk4_step_validated(state, dt_s, parameters)
}

/// Internal step for a loop that has validated its immutable inputs once.
/// Every output state still receives the public finite/nonzero checks.
pub(crate) fn rk4_step_validated(
    state: State,
    dt_s: f64,
    parameters: OrbitParameters,
) -> Result<State, &'static str> {
    let x0 = state.as_array();
    let k1 = derivative(x0, parameters);
    let k2 = derivative(stage(x0, k1, 0.5 * dt_s), parameters);
    let k3 = derivative(stage(x0, k2, 0.5 * dt_s), parameters);
    let k4 = derivative(stage(x0, k3, dt_s), parameters);
    let next = std::array::from_fn(|index| {
        let mut weighted = k1[index] + 2.0 * k2[index];
        weighted += 2.0 * k3[index];
        weighted += k4[index];
        x0[index] + (dt_s / 6.0) * weighted
    });
    State::from_array(next)
}

pub fn propagate_history(
    initial: State,
    dt_s: f64,
    steps: usize,
    parameters: OrbitParameters,
) -> Result<Vec<f64>, &'static str> {
    parameters.validate()?;
    if !dt_s.is_finite() || dt_s == 0.0 {
        return Err("dt_s must be finite and nonzero");
    }
    let capacity = steps
        .checked_add(1)
        .and_then(|count| count.checked_mul(6))
        .ok_or("history size overflow")?;
    let mut history = Vec::new();
    history
        .try_reserve_exact(capacity)
        .map_err(|_| "unable to allocate history")?;
    history.extend(initial.as_array());
    let mut state = initial;
    for _ in 0..steps {
        state = rk4_step_validated(state, dt_s, parameters)?;
        history.extend(state.as_array());
    }
    Ok(history)
}

pub fn rk4_zonal_step(
    state: State,
    dt_s: f64,
    mu_km3_s2: f64,
    command: [f64; 3],
    force_codes: &[u8],
) -> Result<State, &'static str> {
    validate_zonal_inputs(dt_s, mu_km3_s2, command, force_codes)?;
    let x0 = state.as_array();
    let deriv = |x| zonal::derivative(x, mu_km3_s2, command, force_codes);
    let k1 = deriv(x0);
    let k2 = deriv(stage(x0, k1, 0.5 * dt_s));
    let k3 = deriv(stage(x0, k2, 0.5 * dt_s));
    let k4 = deriv(stage(x0, k3, dt_s));
    let next = std::array::from_fn(|index| {
        x0[index] + (dt_s / 6.0) * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
    });
    State::from_array(next)
}

fn validate_zonal_inputs(
    dt_s: f64,
    mu_km3_s2: f64,
    command: [f64; 3],
    force_codes: &[u8],
) -> Result<(), &'static str> {
    if !dt_s.is_finite() {
        return Err("dt_s must be finite");
    }
    if !mu_km3_s2.is_finite() || mu_km3_s2 <= 0.0 {
        return Err("mu_km3_s2 must be finite and positive");
    }
    if !command.iter().all(|value| value.is_finite()) {
        return Err("command acceleration must contain only finite values");
    }
    zonal::validate_codes(force_codes)
}

pub fn rkf78_zonal_step(
    state: State,
    t_s: f64,
    dt_s: f64,
    mu_km3_s2: f64,
    command: [f64; 3],
    force_codes: &[u8],
    atol: f64,
    rtol: f64,
    h_init: Option<f64>,
) -> Result<(State, adaptive::StepInfo), String> {
    validate_zonal_inputs(dt_s, mu_km3_s2, command, force_codes).map_err(str::to_owned)?;
    let (next, info) =
        adaptive::integrate_rkf78(state.as_array(), t_s, dt_s, atol, rtol, h_init, |_, x| {
            Ok(zonal::derivative(x, mu_km3_s2, command, force_codes))
        })?;
    Ok((State::from_array(next).map_err(str::to_owned)?, info))
}

#[cfg(feature = "python")]
mod python {
    use super::force_context::plan_from_inputs as force_plan;
    use super::numeric_bytes::{decode_f64_bytes, output_bytes};
    use super::ogp;
    use super::{adaptive, force_plan::StageInputs};
    use super::{
        propagate_history, rk4_step, rk4_zonal_step, rkf78_zonal_step, OrbitParameters, State,
    };
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    fn fixed<const N: usize>(values: Vec<f64>, name: &str) -> PyResult<[f64; N]> {
        values
            .try_into()
            .map_err(|_| PyValueError::new_err(format!("{name} must have {N} values")))
    }

    #[pyfunction]
    fn ogp_propagate_batch_bytes(
        py: Python<'_>,
        element_bytes: &[u8],
        time_bytes: &[u8],
        samples_per_object: usize,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Vec<(usize, String)>)> {
        let elements = decode_f64_bytes(element_bytes)?;
        let times = decode_f64_bytes(time_bytes)?;
        let result = py
            .detach(|| ogp::propagate_batch(&elements, &times, samples_per_object))
            .map_err(PyValueError::new_err)?;
        Ok((
            output_bytes(py, &result.positions)?,
            output_bytes(py, &result.velocities)?,
            result.errors,
        ))
    }

    #[pyfunction]
    fn ogp_propagate_batch_parallel_bytes(
        py: Python<'_>,
        element_bytes: &[u8],
        time_bytes: &[u8],
        samples_per_object: usize,
        workers: usize,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Vec<(usize, String)>, usize)> {
        let elements = decode_f64_bytes(element_bytes)?;
        let times = decode_f64_bytes(time_bytes)?;
        let result = py
            .detach(|| {
                ogp::propagate_batch_with_workers(&elements, &times, samples_per_object, workers)
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            output_bytes(py, &result.positions)?,
            output_bytes(py, &result.velocities)?,
            result.errors,
            result.workers_used,
        ))
    }

    #[pyclass(name = "OGPObservationBatch")]
    struct OgpObservationBatch {
        batch: ogp::ObservationBatch,
    }

    #[pymethods]
    impl OgpObservationBatch {
        #[new]
        fn new(
            times_jd_utc: Vec<f64>,
            observations: Vec<f64>,
            factors: Vec<f64>,
        ) -> PyResult<Self> {
            let batch = ogp::ObservationBatch::new(times_jd_utc, observations, factors)
                .map_err(PyValueError::new_err)?;
            Ok(Self { batch })
        }

        #[staticmethod]
        fn from_bytes(times_jd_utc: &[u8], observations: &[u8], factors: &[u8]) -> PyResult<Self> {
            Self::new(
                decode_f64_bytes(times_jd_utc)?.into_owned(),
                decode_f64_bytes(observations)?.into_owned(),
                decode_f64_bytes(factors)?.into_owned(),
            )
        }

        fn residual_bytes(&self, py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyBytes>> {
            let elements = ogp::Elements::from_array(fixed(values, "OGP elements")?)
                .map_err(PyValueError::new_err)?;
            let result = py
                .detach(|| self.batch.residuals(elements))
                .map_err(PyValueError::new_err)?;
            output_bytes(py, &result)
        }
    }

    #[pyclass(name = "OGPContext")]
    struct OgpContext {
        propagator: ogp::Propagator,
    }

    #[pymethods]
    impl OgpContext {
        #[new]
        fn new(values: Vec<f64>) -> PyResult<Self> {
            let elements = ogp::Elements::from_array(fixed(values, "OGP elements")?)
                .map_err(PyValueError::new_err)?;
            let propagator = ogp::Propagator::new(elements).map_err(PyValueError::new_err)?;
            Ok(Self { propagator })
        }

        fn propagate(&mut self, tsince_min: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
            let (position, velocity) = self
                .propagator
                .propagate(tsince_min)
                .map_err(PyValueError::new_err)?;
            Ok((position.to_vec(), velocity.to_vec()))
        }

        fn propagate_bytes(&mut self, py: Python<'_>, tsince_min: f64) -> PyResult<Py<PyBytes>> {
            let (position, velocity) = self
                .propagator
                .propagate(tsince_min)
                .map_err(PyValueError::new_err)?;
            let values = [
                position[0],
                position[1],
                position[2],
                velocity[0],
                velocity[1],
                velocity[2],
            ];
            output_bytes(py, &values)
        }

        fn propagate_many_bytes(
            &mut self,
            py: Python<'_>,
            time_bytes: &[u8],
        ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Vec<(usize, String)>)> {
            let times = decode_f64_bytes(time_bytes)?;
            let (positions, velocities, errors) = py.detach(|| {
                let mut positions = Vec::with_capacity(times.len() * 3);
                let mut velocities = Vec::with_capacity(times.len() * 3);
                let mut errors = Vec::new();
                for (index, time) in times.iter().copied().enumerate() {
                    match self.propagator.propagate(time) {
                        Ok((position, velocity)) => {
                            positions.extend_from_slice(&position);
                            velocities.extend_from_slice(&velocity);
                        }
                        Err(error) => {
                            positions.extend_from_slice(&[0.0; 3]);
                            velocities.extend_from_slice(&[0.0; 3]);
                            errors.push((index, error));
                        }
                    }
                }
                (positions, velocities, errors)
            });
            Ok((
                output_bytes(py, &positions)?,
                output_bytes(py, &velocities)?,
                errors,
            ))
        }

        fn propagate_many(&mut self, times_min: Vec<f64>) -> (Vec<f64>, Vec<f64>, Vec<String>) {
            let mut positions = Vec::with_capacity(times_min.len() * 3);
            let mut velocities = Vec::with_capacity(times_min.len() * 3);
            let mut errors = Vec::with_capacity(times_min.len());
            for time in times_min {
                match self.propagator.propagate(time) {
                    Ok((position, velocity)) => {
                        positions.extend_from_slice(&position);
                        velocities.extend_from_slice(&velocity);
                        errors.push(String::new());
                    }
                    Err(error) => {
                        positions.extend_from_slice(&[0.0; 3]);
                        velocities.extend_from_slice(&[0.0; 3]);
                        errors.push(error);
                    }
                }
            }
            (positions, velocities, errors)
        }
    }

    fn parameters(
        mu_km3_s2: f64,
        include_j2: bool,
        command: Vec<f64>,
    ) -> PyResult<OrbitParameters> {
        Ok(OrbitParameters {
            mu_km3_s2,
            include_j2,
            command_accel_eci_km_s2: fixed(command, "command acceleration")?,
        })
    }

    #[pyfunction]
    fn force_plan_accel_eci(
        state: Vec<f64>,
        codes: Vec<u8>,
        scalars: Vec<f64>,
        shadow_model: u8,
        harmonic_dims: (usize, usize),
        tables: Vec<Vec<f64>>,
        stage_inputs: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let state = State::from_array(fixed(state, "state")?).map_err(PyValueError::new_err)?;
        let plan = force_plan(codes, scalars, shadow_model, harmonic_dims, tables)?;
        let stage = StageInputs::from_flat(fixed(stage_inputs, "stage inputs")?)
            .map_err(PyValueError::new_err)?;
        Ok(plan
            .acceleration(state.as_array(), &stage, [0.0; 3])
            .map_err(PyValueError::new_err)?
            .to_vec())
    }

    #[allow(clippy::too_many_arguments)]
    fn adaptive_force_plan_eci_impl(
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        codes: Vec<u8>,
        scalars: Vec<f64>,
        shadow_model: u8,
        harmonic_dims: (usize, usize),
        tables: Vec<Vec<f64>>,
        command: Vec<f64>,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
        stage_callback: Py<PyAny>,
        method: &str,
    ) -> PyResult<(Vec<f64>, AdaptiveInfoTuple)> {
        let state = State::from_array(fixed(state, "state")?).map_err(PyValueError::new_err)?;
        let plan = force_plan(codes, scalars, shadow_model, harmonic_dims, tables)?;
        let command = fixed(command, "command acceleration")?;
        if !command.iter().all(|v| v.is_finite()) {
            return Err(PyValueError::new_err("command acceleration must be finite"));
        }
        let mut callback_error: Option<PyErr> = None;
        let derivative = |stage_time: f64, stage_state: [f64; 6]| -> Result<[f64; 6], String> {
            let values = match stage_callback
                .call1(py, (stage_time, stage_state.to_vec()))
                .and_then(|v| v.extract::<Vec<f64>>(py))
            {
                Ok(v) => v,
                Err(err) => {
                    callback_error = Some(err);
                    return Err("stage input callback failed".to_owned());
                }
            };
            let stage = StageInputs::from_flat(
                values
                    .try_into()
                    .map_err(|_| "stage inputs must have 19 values".to_owned())?,
            )
            .map_err(str::to_owned)?;
            plan.derivative(stage_state, &stage, command)
        };
        let result = match method {
            "rkf78" => adaptive::integrate_rkf78(
                state.as_array(),
                t_s,
                dt_s,
                atol,
                rtol,
                h_init,
                derivative,
            ),
            "dopri5" => adaptive::integrate_dopri5(
                state.as_array(),
                t_s,
                dt_s,
                atol,
                rtol,
                h_init,
                derivative,
            ),
            _ => return Err(PyValueError::new_err("invalid force-plan integrator")),
        };
        if let Some(err) = callback_error {
            return Err(err);
        }
        let (next, info) = result.map_err(PyValueError::new_err)?;
        Ok((
            next.to_vec(),
            (
                info.accepted_steps,
                info.rejected_steps,
                info.attempted_steps,
                info.min_step_s,
                info.max_step_s,
                info.final_step_s,
                info.suggested_next_step_s,
                info.max_error_ratio,
            ),
        ))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn rkf78_force_plan_eci(
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        codes: Vec<u8>,
        scalars: Vec<f64>,
        shadow_model: u8,
        harmonic_dims: (usize, usize),
        tables: Vec<Vec<f64>>,
        command: Vec<f64>,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
        stage_callback: Py<PyAny>,
    ) -> PyResult<(Vec<f64>, AdaptiveInfoTuple)> {
        adaptive_force_plan_eci_impl(
            py,
            state,
            t_s,
            dt_s,
            codes,
            scalars,
            shadow_model,
            harmonic_dims,
            tables,
            command,
            atol,
            rtol,
            h_init,
            stage_callback,
            "rkf78",
        )
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn dopri5_force_plan_eci(
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        codes: Vec<u8>,
        scalars: Vec<f64>,
        shadow_model: u8,
        harmonic_dims: (usize, usize),
        tables: Vec<Vec<f64>>,
        command: Vec<f64>,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
        stage_callback: Py<PyAny>,
    ) -> PyResult<(Vec<f64>, AdaptiveInfoTuple)> {
        adaptive_force_plan_eci_impl(
            py,
            state,
            t_s,
            dt_s,
            codes,
            scalars,
            shadow_model,
            harmonic_dims,
            tables,
            command,
            atol,
            rtol,
            h_init,
            stage_callback,
            "dopri5",
        )
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn rk4_force_plan_eci(
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        codes: Vec<u8>,
        scalars: Vec<f64>,
        shadow_model: u8,
        harmonic_dims: (usize, usize),
        tables: Vec<Vec<f64>>,
        command: Vec<f64>,
        stage_callback: Py<PyAny>,
    ) -> PyResult<Vec<f64>> {
        let state = State::from_array(fixed(state, "state")?).map_err(PyValueError::new_err)?;
        let plan = force_plan(codes, scalars, shadow_model, harmonic_dims, tables)?;
        let command = fixed(command, "command acceleration")?;
        if !t_s.is_finite() || !dt_s.is_finite() || dt_s < 0.0 {
            return Err(PyValueError::new_err(
                "t_s and non-negative dt_s must be finite",
            ));
        }
        let derivative = |time, stage_state: [f64; 6]| -> PyResult<[f64; 6]> {
            let values = stage_callback
                .call1(py, (time, stage_state.to_vec()))?
                .extract::<Vec<f64>>(py)?;
            let stage = StageInputs::from_flat(fixed(values, "stage inputs")?)
                .map_err(PyValueError::new_err)?;
            plan.derivative(stage_state, &stage, command).map_err(PyValueError::new_err)
        };
        let x = state.as_array();
        let k1 = derivative(t_s, x)?;
        let k2 = derivative(t_s + 0.5 * dt_s, super::stage(x, k1, 0.5 * dt_s))?;
        let k3 = derivative(t_s + 0.5 * dt_s, super::stage(x, k2, 0.5 * dt_s))?;
        let k4 = derivative(t_s + dt_s, super::stage(x, k3, dt_s))?;
        let next = std::array::from_fn(|axis| {
            x[axis] + (dt_s / 6.0) * (k1[axis] + 2.0 * k2[axis] + 2.0 * k3[axis] + k4[axis])
        });
        Ok(State::from_array(next)
            .map_err(PyValueError::new_err)?
            .as_array()
            .to_vec())
    }

    #[pyfunction]
    fn rk4_step_eci(
        state: Vec<f64>,
        dt_s: f64,
        mu_km3_s2: f64,
        include_j2: bool,
        command: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let state = State::from_array(fixed(state, "state")?).map_err(PyValueError::new_err)?;
        let params = parameters(mu_km3_s2, include_j2, command)?;
        let next = rk4_step(state, dt_s, params).map_err(PyValueError::new_err)?;
        Ok(next.as_array().to_vec())
    }

    #[pyfunction]
    fn propagate_history_eci(
        state: Vec<f64>,
        dt_s: f64,
        steps: usize,
        mu_km3_s2: f64,
        include_j2: bool,
        command: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let state = State::from_array(fixed(state, "state")?).map_err(PyValueError::new_err)?;
        let params = parameters(mu_km3_s2, include_j2, command)?;
        propagate_history(state, dt_s, steps, params).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn propagate_history_eci_bytes(
        py: Python<'_>,
        state: Vec<f64>,
        dt_s: f64,
        steps: usize,
        mu_km3_s2: f64,
        include_j2: bool,
        command: Vec<f64>,
    ) -> PyResult<Py<PyBytes>> {
        let state = State::from_array(fixed(state, "state")?).map_err(PyValueError::new_err)?;
        let params = parameters(mu_km3_s2, include_j2, command)?;
        let history = py
            .detach(|| propagate_history(state, dt_s, steps, params))
            .map_err(PyValueError::new_err)?;
        output_bytes(py, &history)
    }

    #[pyfunction]
    fn rk4_zonal_step_eci(
        state: Vec<f64>,
        dt_s: f64,
        mu_km3_s2: f64,
        force_codes: Vec<u8>,
        command: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let state = State::from_array(fixed(state, "state")?).map_err(PyValueError::new_err)?;
        let next = rk4_zonal_step(
            state,
            dt_s,
            mu_km3_s2,
            fixed(command, "command acceleration")?,
            &force_codes,
        )
        .map_err(PyValueError::new_err)?;
        Ok(next.as_array().to_vec())
    }

    type AdaptiveInfoTuple = (
        usize,
        usize,
        usize,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        f64,
    );

    fn callback_acceleration(
        py: Python<'_>,
        callback: &Py<PyAny>,
        t_s: f64,
        state: [f64; 6],
    ) -> PyResult<[f64; 3]> {
        let values = callback
            .call1(py, (t_s, state.to_vec()))?
            .extract::<Vec<f64>>(py)?;
        let result = fixed(values, "callback acceleration")?;
        if !result.iter().all(|value| value.is_finite()) {
            return Err(PyValueError::new_err(
                "callback acceleration must be finite",
            ));
        }
        Ok(result)
    }

    #[pyfunction]
    fn rk4_callback_eci(
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        acceleration_callback: Py<PyAny>,
    ) -> PyResult<Vec<f64>> {
        let state = State::from_array(fixed(state, "state")?).map_err(PyValueError::new_err)?;
        if !t_s.is_finite() || !dt_s.is_finite() || dt_s < 0.0 {
            return Err(PyValueError::new_err(
                "t_s and non-negative dt_s must be finite",
            ));
        }
        let x = state.as_array();
        let derivative = |time, stage_state| -> PyResult<[f64; 6]> {
            let acceleration =
                callback_acceleration(py, &acceleration_callback, time, stage_state)?;
            Ok([
                stage_state[3],
                stage_state[4],
                stage_state[5],
                acceleration[0],
                acceleration[1],
                acceleration[2],
            ])
        };
        let k1 = derivative(t_s, x)?;
        let k2 = derivative(t_s + 0.5 * dt_s, super::stage(x, k1, 0.5 * dt_s))?;
        let k3 = derivative(t_s + 0.5 * dt_s, super::stage(x, k2, 0.5 * dt_s))?;
        let k4 = derivative(t_s + dt_s, super::stage(x, k3, dt_s))?;
        let next = std::array::from_fn(|axis| {
            x[axis] + (dt_s / 6.0) * (k1[axis] + 2.0 * k2[axis] + 2.0 * k3[axis] + k4[axis])
        });
        Ok(State::from_array(next)
            .map_err(PyValueError::new_err)?
            .as_array()
            .to_vec())
    }

    #[pyfunction]
    fn rk4_pair_callback_eci(
        py: Python<'_>,
        states: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        derivative_callback: Py<PyAny>,
    ) -> PyResult<Vec<f64>> {
        let x = fixed::<12>(states, "pair states")?;
        if !x.iter().all(|value| value.is_finite())
            || !t_s.is_finite()
            || !dt_s.is_finite()
            || dt_s < 0.0
        {
            return Err(PyValueError::new_err(
                "pair states, time, and non-negative step must be finite",
            ));
        }
        let derivative = |time, stage_state: [f64; 12]| -> PyResult<[f64; 12]> {
            let values = derivative_callback
                .call1(py, (time, stage_state.to_vec()))?
                .extract::<Vec<f64>>(py)?;
            let result = fixed::<12>(values, "pair derivative")?;
            if !result.iter().all(|value| value.is_finite()) {
                return Err(PyValueError::new_err("pair derivative must be finite"));
            }
            Ok(result)
        };
        let stage = |values: [f64; 12], slope: [f64; 12], scale: f64| -> [f64; 12] {
            std::array::from_fn(|axis| values[axis] + scale * slope[axis])
        };
        let k1 = derivative(t_s, x)?;
        let k2 = derivative(t_s + 0.5 * dt_s, stage(x, k1, 0.5 * dt_s))?;
        let k3 = derivative(t_s + 0.5 * dt_s, stage(x, k2, 0.5 * dt_s))?;
        let k4 = derivative(t_s + dt_s, stage(x, k3, dt_s))?;
        let next: [f64; 12] = std::array::from_fn(|axis| {
            x[axis] + (dt_s / 6.0) * (k1[axis] + 2.0 * k2[axis] + 2.0 * k3[axis] + k4[axis])
        });
        if !next.iter().all(|value| value.is_finite()) {
            return Err(PyValueError::new_err(
                "pair propagation produced a non-finite state",
            ));
        }
        Ok(next.to_vec())
    }

    #[pyfunction]
    fn rk4_coupled_callback(
        py: Python<'_>,
        state: Vec<f64>,
        quaternion: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        stage_callback: Py<PyAny>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if state.len() < 10
            || state.iter().any(|value| !value.is_finite())
            || quaternion.len() != 4
            || quaternion.iter().any(|value| !value.is_finite())
            || !t_s.is_finite()
            || !dt_s.is_finite()
            || dt_s < 0.0
        {
            return Err(PyValueError::new_err(
                "coupled state, quaternion, and time must be finite with valid dimensions",
            ));
        }
        let stage = |values: &[f64], slope: &[f64], scale: f64| -> Vec<f64> {
            values
                .iter()
                .zip(slope)
                .map(|(value, rate)| value + scale * rate)
                .collect()
        };
        let derivative =
            |index: usize, time: f64, values: &[f64]| -> PyResult<(Vec<f64>, Vec<f64>)> {
                let (rate, q_rate): (Vec<f64>, Vec<f64>) = stage_callback
                    .call1(py, (index, time, values.to_vec()))?
                    .extract(py)?;
                if rate.len() != state.len()
                    || q_rate.len() != 4
                    || rate.iter().chain(&q_rate).any(|value| !value.is_finite())
                {
                    return Err(PyValueError::new_err(
                        "coupled stage callback returned invalid derivatives",
                    ));
                }
                Ok((rate, q_rate))
            };
        let (k1, q1) = derivative(0, t_s, &state)?;
        let (k2, q2) = derivative(1, t_s + 0.5 * dt_s, &stage(&state, &k1, 0.5 * dt_s))?;
        let (k3, q3) = derivative(2, t_s + 0.5 * dt_s, &stage(&state, &k2, 0.5 * dt_s))?;
        let (k4, q4) = derivative(3, t_s + dt_s, &stage(&state, &k3, dt_s))?;
        let next: Vec<f64> = (0..state.len())
            .map(|axis| {
                state[axis] + (dt_s / 6.0) * (k1[axis] + 2.0 * k2[axis] + 2.0 * k3[axis] + k4[axis])
            })
            .collect();
        let next_q: Vec<f64> = (0..4)
            .map(|axis| {
                quaternion[axis]
                    + (dt_s / 6.0) * (q1[axis] + 2.0 * q2[axis] + 2.0 * q3[axis] + q4[axis])
            })
            .collect();
        if next.iter().chain(&next_q).any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "coupled propagation produced a non-finite state",
            ));
        }
        Ok((next, next_q))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn adaptive_callback_eci(
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
        method: &str,
        acceleration_callback: Py<PyAny>,
    ) -> PyResult<(Vec<f64>, AdaptiveInfoTuple)> {
        let state = State::from_array(fixed(state, "state")?).map_err(PyValueError::new_err)?;
        let mut callback_error: Option<PyErr> = None;
        let derivative = |time, stage_state: [f64; 6]| -> Result<[f64; 6], String> {
            match callback_acceleration(py, &acceleration_callback, time, stage_state) {
                Ok(acceleration) => Ok([
                    stage_state[3],
                    stage_state[4],
                    stage_state[5],
                    acceleration[0],
                    acceleration[1],
                    acceleration[2],
                ]),
                Err(err) => {
                    callback_error = Some(err);
                    Err("acceleration callback failed".to_owned())
                }
            }
        };
        let result = match method {
            "rkf78" => adaptive::integrate_rkf78(
                state.as_array(),
                t_s,
                dt_s,
                atol,
                rtol,
                h_init,
                derivative,
            ),
            "dopri5" => adaptive::integrate_dopri5(
                state.as_array(),
                t_s,
                dt_s,
                atol,
                rtol,
                h_init,
                derivative,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "adaptive method must be rkf78 or dopri5",
                ))
            }
        };
        if let Some(err) = callback_error {
            return Err(err);
        }
        let (next, info) = result.map_err(PyValueError::new_err)?;
        Ok((
            State::from_array(next)
                .map_err(PyValueError::new_err)?
                .as_array()
                .to_vec(),
            (
                info.accepted_steps,
                info.rejected_steps,
                info.attempted_steps,
                info.min_step_s,
                info.max_step_s,
                info.final_step_s,
                info.suggested_next_step_s,
                info.max_error_ratio,
            ),
        ))
    }

    #[pyfunction]
    fn rkf78_zonal_step_eci(
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        mu_km3_s2: f64,
        force_codes: Vec<u8>,
        command: Vec<f64>,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
    ) -> PyResult<(Vec<f64>, AdaptiveInfoTuple)> {
        let state = State::from_array(fixed(state, "state")?).map_err(PyValueError::new_err)?;
        let (next, info) = rkf78_zonal_step(
            state,
            t_s,
            dt_s,
            mu_km3_s2,
            fixed(command, "command acceleration")?,
            &force_codes,
            atol,
            rtol,
            h_init,
        )
        .map_err(PyValueError::new_err)?;
        Ok((
            next.as_array().to_vec(),
            (
                info.accepted_steps,
                info.rejected_steps,
                info.attempted_steps,
                info.min_step_s,
                info.max_step_s,
                info.final_step_s,
                info.suggested_next_step_s,
                info.max_error_ratio,
            ),
        ))
    }

    #[pymodule]
    fn oel_rust_orbit(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("ONP_PRECISION_FORCE_PLAN_VERSION", 1)?;
        module.add("ONP_EMPIRICAL_RIC_VERSION", 1)?;
        module.add("ONP_EMPIRICAL_RIC_HARMONIC_VERSION", 1)?;
        module.add("ONP_ECOM2_VERSION", u32::from(cfg!(feature = "pro")))?;
        module.add("ONP_IMMUTABLE_STAGE_TABLE_MAX_EPOCHS", 262144)?;
        module.add("ONP_IAU76_80_SIDEREAL_VERSION", 2)?;
        module.add("__version__", env!("CARGO_PKG_VERSION"))?;
        // Registered callback-free object kernels release the Python GIL.
        // This marker does not qualify arbitrary complete object steps.
        module.add("NATIVE_OBJECT_KERNELS_GIL_RELEASED", true)?;
        module.add(
            "__source_sha256__",
            option_env!("OEL_NATIVE_SOURCE_SHA256").unwrap_or("unattested"),
        )?;
        super::cr3bp::register(module)?;
        super::attitude::register(module)?;
        super::attitude_ekf::register(module)?;
        super::actuation::register(module)?;
        super::mechanical::register(module)?;
        super::mission_schedule::register(module)?;
        super::nrlmsise00_native::register(module)?;
        super::collisions::register(module)?;
        super::control::register(module)?;
        super::covariance::register(module)?;
        super::coverage::register(module)?;
        super::tasking_search::register(module)?;
        super::environment::register(module)?;
        super::force_context::register(module)?;
        super::stage_context::register(module)?;
        super::passive::register(module)?;
        super::estimation::register(module)?;
        super::links::register(module)?;
        super::measurements::register(module)?;
        super::optical_collection::register(module)?;
        super::perturbations::register(module)?;
        super::solid_tides::register_py(module)?;
        super::ocean_tides::register_py(module)?;
        super::relative::register(module)?;
        super::resources::register(module)?;
        super::spacecraft::register(module)?;
        super::communications::register(module)?;
        super::conjunction::register(module)?;
        super::conjunction_probability::register(module)?;
        super::covariance_orbit::register(module)?;
        super::iod::register(module)?;
        super::targeting::register(module)?;
        super::target_coast::register(module)?;
        super::target_rollouts::register(module)?;
        super::tracking::register(module)?;
        super::vehicle::register(module)?;
        super::output::register(module)?;
        module.add_class::<OgpContext>()?;
        module.add_class::<super::planetary_binding::NativePlanetaryPlan>()?;
        module.add("PlanetaryGravityDomainError", module.py().get_type::<super::planetary_binding::PlanetaryGravityDomainError>())?;
        module.add_class::<super::additional_gravity_binding::NativeAdditionalGravity>()?;
        module.add_function(wrap_pyfunction!(super::planetary_binding::planetary_tdb_seconds, module)?)?;
        module.add_function(wrap_pyfunction!(super::planetary_binding::planetary_rotation, module)?)?;
        module.add_function(wrap_pyfunction!(super::planetary_binding::planetary_atmosphere_sample, module)?)?;
        module.add_class::<OgpObservationBatch>()?;
        module.add("MAX_HARMONIC_DEGREE", super::harmonics::MAX_DEGREE)?;
        module.add_function(wrap_pyfunction!(ogp_propagate_batch_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(ogp_propagate_batch_parallel_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(rk4_step_eci, module)?)?;
        module.add_function(wrap_pyfunction!(propagate_history_eci, module)?)?;
        module.add_function(wrap_pyfunction!(propagate_history_eci_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(rk4_zonal_step_eci, module)?)?;
        module.add_function(wrap_pyfunction!(rkf78_zonal_step_eci, module)?)?;
        module.add_function(wrap_pyfunction!(force_plan_accel_eci, module)?)?;
        module.add_function(wrap_pyfunction!(rkf78_force_plan_eci, module)?)?;
        module.add_function(wrap_pyfunction!(dopri5_force_plan_eci, module)?)?;
        module.add_function(wrap_pyfunction!(rk4_force_plan_eci, module)?)?;
        module.add_function(wrap_pyfunction!(rk4_callback_eci, module)?)?;
        module.add_function(wrap_pyfunction!(adaptive_callback_eci, module)?)?;
        module.add_function(wrap_pyfunction!(rk4_pair_callback_eci, module)?)?;
        module.add_function(wrap_pyfunction!(rk4_coupled_callback, module)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{propagate_history, rk4_step, OrbitParameters, State};

    fn test_state() -> State {
        State::from_array([7000.0, 0.0, 0.0, 0.0, 7.5, 1.0]).unwrap()
    }

    fn parameters(include_j2: bool) -> OrbitParameters {
        OrbitParameters {
            mu_km3_s2: 398600.4415,
            include_j2,
            command_accel_eci_km_s2: [0.0; 3],
        }
    }

    #[test]
    fn two_body_energy_is_stable_over_one_orbit() {
        let history = propagate_history(test_state(), 10.0, 600, parameters(false)).unwrap();
        let initial = &history[0..6];
        let final_state = &history[history.len() - 6..];
        let energy = |state: &[f64]| {
            let r = (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
            let v2 = state[3].powi(2) + state[4].powi(2) + state[5].powi(2);
            0.5 * v2 - 398600.4415 / r
        };
        assert!((energy(final_state) - energy(initial)).abs() < 1e-8);
    }

    #[test]
    fn j2_changes_the_trajectory() {
        let plain = rk4_step(test_state(), 10.0, parameters(false)).unwrap();
        let zonal = rk4_step(test_state(), 10.0, parameters(true)).unwrap();
        assert!((plain.velocity_km_s[0] - zonal.velocity_km_s[0]).abs() > 1e-7);
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert!(State::from_array([0.0; 6]).is_err());
        assert!(rk4_step(test_state(), f64::NAN, parameters(false)).is_err());
        assert!(propagate_history(test_state(), 0.0, 1, parameters(false)).is_err());
    }
}
