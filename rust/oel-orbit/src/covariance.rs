//! Small dense covariance and finite-difference kernels.
//!
//! Force evaluation stays on the OEL Python side so configured ONP plugins,
//! atmosphere data, frame policy, and event semantics remain authoritative.
//! These routines own only the repeated fixed-size arithmetic around those
//! evaluations.

fn validate_finite(values: &[f64], name: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn mat6_mul(left: &[f64], right: &[f64]) -> [f64; 36] {
    let mut out = [0.0; 36];
    for row in 0..6 {
        for inner in 0..6 {
            let value = left[row * 6 + inner];
            for column in 0..6 {
                out[row * 6 + column] += value * right[inner * 6 + column];
            }
        }
    }
    out
}

fn mat6_transpose(value: &[f64]) -> [f64; 36] {
    let mut out = [0.0; 36];
    for row in 0..6 {
        for column in 0..6 {
            out[column * 6 + row] = value[row * 6 + column];
        }
    }
    out
}

pub fn finite_difference_stm(values: &[f64], steps: &[f64]) -> Result<[f64; 36], String> {
    if values.len() != 72 {
        return Err("finite-difference samples must contain 72 values".into());
    }
    if steps.len() != 6 {
        return Err("finite-difference steps must contain 6 values".into());
    }
    validate_finite(values, "finite-difference samples")?;
    validate_finite(steps, "finite-difference steps")?;
    if steps.iter().any(|step| *step <= 0.0) {
        return Err("finite-difference steps must be positive".into());
    }
    let mut phi = [0.0; 36];
    // Each axis occupies one plus/minus six-state pair.  This layout avoids
    // nested Python allocations while preserving the reference central
    // difference operation order.
    for axis in 0..6 {
        let base = axis * 12;
        for row in 0..6 {
            phi[row * 6 + axis] =
                (values[base + row] - values[base + 6 + row]) / (2.0 * steps[axis]);
        }
    }
    if phi.iter().any(|value| !value.is_finite()) {
        return Err("finite-difference STM produced non-finite values".into());
    }
    Ok(phi)
}

pub fn finite_difference_stm_batch(values: &[f64], steps: &[f64]) -> Result<Vec<f64>, String> {
    if values.len() % 72 != 0 {
        return Err("finite-difference batch must contain 72 values per sample".into());
    }
    if steps.len() != values.len() / 12 {
        return Err("finite-difference batch must contain 6 steps per sample".into());
    }
    let mut output = Vec::with_capacity(values.len() / 2);
    for sample in 0..values.len() / 72 {
        let phi = finite_difference_stm(
            &values[sample * 72..(sample + 1) * 72],
            &steps[sample * 6..(sample + 1) * 6],
        )?;
        output.extend_from_slice(&phi);
    }
    Ok(output)
}

pub fn propagate_history(
    initial_covariance: &[f64],
    transition_history: &[f64],
    process_noise_history: &[f64],
) -> Result<Vec<f64>, String> {
    if initial_covariance.len() != 36 {
        return Err("initial covariance must contain 36 values".into());
    }
    if transition_history.len() % 36 != 0 {
        return Err("transition history must contain 36 values per interval".into());
    }
    let intervals = transition_history.len() / 36;
    if process_noise_history.len() != 36 && process_noise_history.len() != intervals * 36 {
        return Err(
            "process noise must contain 36 values or one 36-value matrix per interval".into(),
        );
    }
    validate_finite(initial_covariance, "initial covariance")?;
    validate_finite(transition_history, "transition history")?;
    validate_finite(process_noise_history, "process noise history")?;
    let mut output = Vec::with_capacity((intervals + 1) * 36);
    let mut covariance = [0.0; 36];
    covariance.copy_from_slice(initial_covariance);
    output.extend_from_slice(&covariance);
    for interval in 0..intervals {
        let phi = &transition_history[interval * 36..(interval + 1) * 36];
        let phi_t = mat6_transpose(phi);
        let left = mat6_mul(phi, &covariance);
        let propagated = mat6_mul(&left, &phi_t);
        let q = if process_noise_history.len() == 36 {
            &process_noise_history[..]
        } else {
            &process_noise_history[interval * 36..(interval + 1) * 36]
        };
        for index in 0..36 {
            covariance[index] = propagated[index] + q[index];
        }
        // Symmetrize after adding Q.  The previous line intentionally keeps
        // the matrix multiplication order deterministic; this pass prevents
        // roundoff asymmetry from accumulating in long histories.
        for row in 0..6 {
            for column in row + 1..6 {
                let value = 0.5 * (covariance[row * 6 + column] + covariance[column * 6 + row]);
                covariance[row * 6 + column] = value;
                covariance[column * 6 + row] = value;
            }
        }
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err("covariance propagation produced non-finite values".into());
        }
        output.extend_from_slice(&covariance);
    }
    Ok(output)
}

/// Evaluate the small-hard-body encounter-plane probability approximation.
///
/// OEL's covariance workflow deliberately reports this as a screening
/// estimate.  The inputs are the two-dimensional encounter-plane mean and
/// covariance in km and km^2.  The covariance regularization mirrors the
/// Python reference implementation so that degenerate but finite planes keep
/// the same bounded behavior.
pub fn small_object_collision_probability(
    mean_plane: &[f64],
    covariance_plane: &[f64],
    hard_body_radius: f64,
) -> Result<Option<f64>, String> {
    if mean_plane.len() != 2 {
        return Err("encounter-plane mean must contain 2 values".into());
    }
    if covariance_plane.len() != 4 {
        return Err("encounter-plane covariance must contain 4 values".into());
    }
    validate_finite(mean_plane, "encounter-plane mean")?;
    validate_finite(covariance_plane, "encounter-plane covariance")?;
    if !hard_body_radius.is_finite() || hard_body_radius < 0.0 {
        return Err("hard-body radius must be finite and non-negative".into());
    }

    // The Python path first symmetrizes and then floors the smaller
    // eigenvalue at max(lambda_max * 1e-12, 1e-18).  For a symmetric 2x2
    // matrix, the closed-form eigenvalues avoid pulling a linear algebra
    // dependency into this fixed-size kernel.
    let mut a = covariance_plane[0];
    let mut b = 0.5 * (covariance_plane[1] + covariance_plane[2]);
    let mut d = covariance_plane[3];
    let discriminant = ((a - d) * (a - d) + 4.0 * b * b).sqrt();
    let trace = a + d;
    let lambda_min = 0.5 * (trace - discriminant);
    let lambda_max = 0.5 * (trace + discriminant);
    let floor = (lambda_max * 1.0e-12).max(1.0e-18);
    if lambda_min < floor {
        let adjustment = floor - lambda_min;
        a += adjustment;
        d += adjustment;
    }
    // Keep the same symmetry and regularization assumptions as the Python
    // helper, including returning None for a non-positive determinant.
    b = 0.5 * (covariance_plane[1] + covariance_plane[2]);
    let determinant = a * d - b * b;
    if determinant <= 0.0 || !determinant.is_finite() {
        return Ok(None);
    }
    let inv00 = d / determinant;
    let inv01 = -b / determinant;
    let inv11 = a / determinant;
    let exponent = -0.5
        * (mean_plane[0] * (inv00 * mean_plane[0] + inv01 * mean_plane[1])
            + mean_plane[1] * (inv01 * mean_plane[0] + inv11 * mean_plane[1]));
    if exponent < -745.0 {
        return Ok(Some(0.0));
    }
    let density_at_origin = exponent.exp() / (2.0 * std::f64::consts::PI * determinant.sqrt());
    let probability = std::f64::consts::PI * hard_body_radius.powi(2) * density_at_origin;
    if !probability.is_finite() {
        return Ok(None);
    }
    Ok(Some(probability.clamp(0.0, 1.0)))
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    fn covariance_finite_difference_stm(values: Vec<f64>, steps: Vec<f64>) -> PyResult<Vec<f64>> {
        finite_difference_stm(&values, &steps)
            .map(|value| value.to_vec())
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn covariance_finite_difference_stm_batch(
        values: Vec<f64>,
        steps: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        finite_difference_stm_batch(&values, &steps).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn covariance_propagate_history(
        initial_covariance: Vec<f64>,
        transition_history: Vec<f64>,
        process_noise_history: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        propagate_history(
            &initial_covariance,
            &transition_history,
            &process_noise_history,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn covariance_small_object_collision_probability(
        mean_plane: Vec<f64>,
        covariance_plane: Vec<f64>,
        hard_body_radius: f64,
    ) -> PyResult<Option<f64>> {
        small_object_collision_probability(&mean_plane, &covariance_plane, hard_body_radius)
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn covariance_propagate_history_bytes(
        py: Python<'_>,
        initial_covariance: &[u8],
        transition_history: &[u8],
        process_noise_history: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let values = propagate_history(
            &decode_f64_bytes(initial_covariance)?,
            &decode_f64_bytes(transition_history)?,
            &decode_f64_bytes(process_noise_history)?,
        )
        .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }
    #[pyfunction]
    fn covariance_finite_difference_stm_batch_bytes(
        py: Python<'_>,
        values: &[u8],
        steps: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let values =
            finite_difference_stm_batch(&decode_f64_bytes(values)?, &decode_f64_bytes(steps)?)
                .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(covariance_finite_difference_stm, module)?)?;
        module.add_function(wrap_pyfunction!(
            covariance_finite_difference_stm_batch,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(covariance_propagate_history, module)?)?;
        module.add_function(wrap_pyfunction!(
            covariance_small_object_collision_probability,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(
            covariance_propagate_history_bytes,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(
            covariance_finite_difference_stm_batch_bytes,
            module
        )?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    python::register(module)
}

#[cfg(test)]
mod tests {
    use super::{finite_difference_stm, propagate_history, small_object_collision_probability};

    #[test]
    fn finite_difference_recovers_diagonal_map() {
        let mut values = [0.0; 72];
        for axis in 0..6 {
            for row in 0..6 {
                values[axis * 12 + row] = if row == axis { 2.0 } else { 0.0 };
                values[axis * 12 + 6 + row] = if row == axis { 0.0 } else { 0.0 };
            }
        }
        let phi = finite_difference_stm(&values, &[1.0; 6]).unwrap();
        for row in 0..6 {
            for col in 0..6 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert_eq!(phi[row * 6 + col], expected);
            }
        }
    }

    #[test]
    fn covariance_history_preserves_zero_covariance() {
        let mut phi = [0.0; 36];
        for axis in 0..6 {
            phi[axis * 6 + axis] = 1.0;
        }
        let out = propagate_history(&[0.0; 36], &phi, &[0.0; 36]).unwrap();
        assert_eq!(out.len(), 72);
        assert!(out.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn encounter_probability_is_bounded_and_regularized() {
        let probability =
            small_object_collision_probability(&[0.01, -0.02], &[0.0, 0.0, 0.0, 0.0], 0.005)
                .unwrap()
                .unwrap();
        assert!(probability.is_finite());
        assert!((0.0..=1.0).contains(&probability));
    }
}
