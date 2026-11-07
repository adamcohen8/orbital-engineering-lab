//! Shared public OGP bindings; private OD and campaign bindings live elsewhere.
use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
use crate::ogp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn fixed<const N: usize>(values: Vec<f64>, name: &str) -> PyResult<[f64; N]> {
    values.try_into().map_err(|_| PyValueError::new_err(format!("{name} must have {N} values")))
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

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<OgpContext>()?;
    module.add_function(wrap_pyfunction!(ogp_propagate_batch_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(ogp_propagate_batch_parallel_bytes, module)?)?;
    Ok(())
}
