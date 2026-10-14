//! Python preparation boundary for the separate native planetary plan.
use crate::planetary::{
    Atmosphere, AtmosphereInput, EphemerisInput, Plan, Rotation, RotationInput,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pyo3::create_exception!(oel_rust_orbit, PlanetaryGravityDomainError, PyValueError);

pub(crate) fn plan_error(error: String) -> PyErr {
    if error == crate::planetary::GRAVITY_DOMAIN_ERROR {
        PlanetaryGravityDomainError::new_err(error)
    } else {
        PyValueError::new_err(error)
    }
}

#[pyfunction]
pub fn planetary_tdb_seconds(tt_start_s: f64, t_s: f64) -> PyResult<f64> {
    if !tt_start_s.is_finite() || !t_s.is_finite() {
        return Err(PyValueError::new_err(
            "planetary elapsed time must be finite",
        ));
    }
    Ok(crate::planetary::tdb(tt_start_s + t_s).0)
}

fn state(values: Vec<f64>) -> PyResult<[f64; 6]> {
    values
        .try_into()
        .map_err(|_| PyValueError::new_err("planetary state requires six values"))
}

#[pyclass(name = "PlanetaryPlan")]
pub struct NativePlanetaryPlan {
    pub(crate) plan: Plan,
}

#[pymethods]
impl NativePlanetaryPlan {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        center: i64,
        scalars: Vec<f64>,
        harmonic_dims: (usize, usize),
        coefficients: (Vec<f64>, Vec<f64>),
        gravity_rotation: RotationInput,
        air_rotation: RotationInput,
        atmosphere: AtmosphereInput,
        ephemeris: EphemerisInput,
        third_bodies: (Vec<i64>, Vec<f64>),
    ) -> PyResult<Self> {
        Ok(Self {
            plan: Plan::new(
                center,
                scalars,
                harmonic_dims,
                coefficients,
                gravity_rotation,
                air_rotation,
                atmosphere,
                ephemeris,
                third_bodies,
            )
            .map_err(PyValueError::new_err)?,
        })
    }
    fn stage(
        &self,
        py: Python<'_>,
        t_s: f64,
        values: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> {
        let x = state(values)?;
        let result = py.detach(|| self.plan.stage(t_s, x)).map_err(plan_error)?;
        Ok((
            result.acceleration.to_vec(),
            result.diagnostics.to_vec(),
            result.components.iter().map(|v| v.to_vec()).collect(),
        ))
    }
    fn propagate(
        &self,
        py: Python<'_>,
        values: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        method: &str,
        atol: f64,
        rtol: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let x = state(values)?;
        let (result, info) = py
            .detach(|| self.plan.propagate(x, t_s, dt_s, method, atol, rtol))
            .map_err(plan_error)?;
        Ok((
            result.to_vec(),
            vec![
                info.accepted_steps as f64,
                info.rejected_steps as f64,
                info.attempted_steps as f64,
                info.min_step_s.unwrap_or(0.),
                info.max_step_s.unwrap_or(0.),
                info.final_step_s.unwrap_or(0.),
                info.suggested_next_step_s.unwrap_or(0.),
                info.max_error_ratio,
            ],
        ))
    }
}

#[pyfunction]
pub fn planetary_rotation(
    definition: RotationInput,
    tt_start_s: f64,
    t_s: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (r, d) = Rotation::new(definition)
        .and_then(|v| v.evaluate(tt_start_s + t_s))
        .map_err(PyValueError::new_err)?;
    Ok((r.to_vec(), d.to_vec()))
}

#[pyfunction]
pub fn planetary_atmosphere_sample(
    definition: AtmosphereInput,
    position: Vec<f64>,
) -> PyResult<(f64, f64, Vec<f64>, bool)> {
    let r: [f64; 3] = position.try_into().map_err(|_| {
        PyValueError::new_err("planetary atmosphere position requires three values")
    })?;
    let (h, rho, wind, inside) = Atmosphere::new(definition)
        .and_then(|v| v.sample(r))
        .map_err(PyValueError::new_err)?;
    Ok((h, rho, wind.to_vec(), inside))
}
