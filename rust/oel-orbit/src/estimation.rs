//! Small, allocation-conscious estimation kernels.
//!
//! The Python estimators remain the owners of configuration, solver policy,
//! priors, clipping, and report semantics.  This module contains the repeated
//! fixed-size operations used by EKF/UKF and batch residual paths.  Matrices
//! are row-major and states use the same units supplied by the caller.

fn ensure_len(values: &[f64], expected: usize, name: &str) -> Result<(), String> {
    if values.len() != expected {
        return Err(format!("{name} must contain {expected} values"));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn ensure_square(values: &[f64], dimension: usize, name: &str) -> Result<(), String> {
    let expected = dimension
        .checked_mul(dimension)
        .ok_or_else(|| format!("{name} dimension overflow"))?;
    ensure_len(values, expected, name)
}

fn matmul(
    left: &[f64],
    left_rows: usize,
    left_cols: usize,
    right: &[f64],
    right_cols: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; left_rows * right_cols];
    for row in 0..left_rows {
        for inner in 0..left_cols {
            let value = left[row * left_cols + inner];
            if value == 0.0 {
                continue;
            }
            for column in 0..right_cols {
                output[row * right_cols + column] += value * right[inner * right_cols + column];
            }
        }
    }
    output
}

fn transpose(values: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut output = vec![0.0; rows * cols];
    for row in 0..rows {
        for column in 0..cols {
            output[column * rows + row] = values[row * cols + column];
        }
    }
    output
}

fn add_in_place(left: &mut [f64], right: &[f64]) {
    for (target, value) in left.iter_mut().zip(right) {
        *target += *value;
    }
}

/// Cholesky factorization of a positive-definite matrix.
pub fn cholesky_lower(matrix: &[f64], dimension: usize) -> Result<Vec<f64>, String> {
    ensure_square(matrix, dimension, "matrix")?;
    let mut lower = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = matrix[row * dimension + column];
            for index in 0..column {
                value -= lower[row * dimension + index] * lower[column * dimension + index];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err("matrix must be positive definite".to_owned());
                }
                lower[row * dimension + column] = value.sqrt();
            } else {
                let diagonal = lower[column * dimension + column];
                if diagonal <= 0.0 || !diagonal.is_finite() {
                    return Err("matrix must be positive definite".to_owned());
                }
                lower[row * dimension + column] = value / diagonal;
            }
        }
    }
    Ok(lower)
}

fn solve_lower(
    lower: &[f64],
    dimension: usize,
    right: &[f64],
    columns: usize,
) -> Result<Vec<f64>, String> {
    ensure_square(lower, dimension, "lower factor")?;
    ensure_len(right, dimension * columns, "right-hand side")?;
    let mut result = vec![0.0; dimension * columns];
    for row in 0..dimension {
        let diagonal = lower[row * dimension + row];
        if diagonal <= 0.0 || !diagonal.is_finite() {
            return Err("lower factor diagonal must be finite and positive".to_owned());
        }
        for column in 0..columns {
            let mut value = right[row * columns + column];
            for index in 0..row {
                value -= lower[row * dimension + index] * result[index * columns + column];
            }
            result[row * columns + column] = value / diagonal;
        }
    }
    Ok(result)
}

fn solve_upper_from_lower(
    lower: &[f64],
    dimension: usize,
    right: &[f64],
    columns: usize,
) -> Result<Vec<f64>, String> {
    ensure_square(lower, dimension, "lower factor")?;
    ensure_len(right, dimension * columns, "right-hand side")?;
    let mut result = vec![0.0; dimension * columns];
    for row in (0..dimension).rev() {
        let diagonal = lower[row * dimension + row];
        if diagonal <= 0.0 || !diagonal.is_finite() {
            return Err("lower factor diagonal must be finite and positive".to_owned());
        }
        for column in 0..columns {
            let mut value = right[row * columns + column];
            for index in row + 1..dimension {
                value -= lower[index * dimension + row] * result[index * columns + column];
            }
            result[row * columns + column] = value / diagonal;
        }
    }
    Ok(result)
}

fn symmetrize(matrix: &mut [f64], dimension: usize) {
    for row in 0..dimension {
        for column in row + 1..dimension {
            let value = 0.5 * (matrix[row * dimension + column] + matrix[column * dimension + row]);
            matrix[row * dimension + column] = value;
            matrix[column * dimension + row] = value;
        }
    }
}

/// EKF measurement update using a Joseph covariance update.
pub fn ekf_update(
    state: &[f64],
    covariance: &[f64],
    measurement: &[f64],
    measurement_matrix: &[f64],
    measurement_covariance: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64), String> {
    let n = state.len();
    let m = measurement.len();
    if n == 0 || m == 0 {
        return Err("state and measurement dimensions must be positive".to_owned());
    }
    ensure_len(state, n, "state")?;
    ensure_square(covariance, n, "covariance")?;
    ensure_len(measurement, m, "measurement")?;
    ensure_len(measurement_matrix, m * n, "measurement_matrix")?;
    ensure_square(measurement_covariance, m, "measurement_covariance")?;

    let predicted_measurement = matmul(measurement_matrix, m, n, state, 1);
    let innovation: Vec<f64> = measurement
        .iter()
        .zip(predicted_measurement)
        .map(|(observed, predicted)| observed - predicted)
        .collect();
    ekf_update_innovation(
        state,
        covariance,
        &innovation,
        measurement_matrix,
        measurement_covariance,
    )
}

/// Joseph EKF update with an innovation prepared by a nonlinear measurement
/// owner, including its angle wrapping and measurement-specific gates.
pub fn ekf_update_innovation(
    state: &[f64],
    covariance: &[f64],
    innovation: &[f64],
    measurement_matrix: &[f64],
    measurement_covariance: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64), String> {
    let n = state.len();
    let m = innovation.len();
    if n == 0 || m == 0 {
        return Err("state and innovation dimensions must be positive".into());
    }
    ensure_len(state, n, "state")?;
    ensure_square(covariance, n, "covariance")?;
    ensure_len(innovation, m, "innovation")?;
    ensure_len(measurement_matrix, m * n, "measurement_matrix")?;
    ensure_square(measurement_covariance, m, "measurement_covariance")?;
    let hp = matmul(measurement_matrix, m, n, covariance, n);
    let h_t = transpose(measurement_matrix, m, n);
    let mut innovation_covariance = matmul(hp.as_slice(), m, n, &h_t, m);
    add_in_place(&mut innovation_covariance, measurement_covariance);
    symmetrize(&mut innovation_covariance, m);

    // P H^T is n x m.  Solve S K^T = H P for K.
    let p_h_t = matmul(covariance, n, n, &h_t, m);
    let lower = cholesky_lower(&innovation_covariance, m)?;
    let forward_gain = solve_lower(&lower, m, &transpose(&p_h_t, n, m), n)?;
    let gain_transposed = solve_upper_from_lower(&lower, m, &forward_gain, n)?;
    let gain = transpose(&gain_transposed, m, n);
    let forward_innovation = solve_lower(&lower, m, &innovation, 1)?;
    let solved_innovation = solve_upper_from_lower(&lower, m, &forward_innovation, 1)?;
    let nis = innovation
        .iter()
        .zip(&solved_innovation)
        .map(|(left, right)| left * right)
        .sum::<f64>();

    let correction = matmul(&gain, n, m, &innovation, 1);
    let updated_state: Vec<f64> = state
        .iter()
        .zip(correction)
        .map(|(left, right)| left + right)
        .collect();

    let mut identity_minus_gain_h = vec![0.0; n * n];
    for index in 0..n {
        identity_minus_gain_h[index * n + index] = 1.0;
    }
    let gain_h = matmul(&gain, n, m, measurement_matrix, n);
    for (target, value) in identity_minus_gain_h.iter_mut().zip(gain_h) {
        *target -= value;
    }
    let left = matmul(&identity_minus_gain_h, n, n, covariance, n);
    let left_right = matmul(&left, n, n, &transpose(&identity_minus_gain_h, n, n), n);
    let gain_r = matmul(&gain, n, m, measurement_covariance, m);
    let gain_r_gain_t = matmul(&gain_r, n, m, &transpose(&gain, n, m), n);
    let mut updated_covariance = left_right;
    add_in_place(&mut updated_covariance, &gain_r_gain_t);
    symmetrize(&mut updated_covariance, n);
    if !updated_state.iter().all(|value| value.is_finite())
        || !updated_covariance.iter().all(|value| value.is_finite())
        || !nis.is_finite()
    {
        return Err("EKF update produced non-finite values".to_owned());
    }
    Ok((
        updated_state,
        updated_covariance,
        innovation.to_vec(),
        innovation_covariance,
        nis,
    ))
}

/// Generate standard scaled unscented sigma points and weights.
pub fn ukf_sigma_points(
    state: &[f64],
    covariance: &[f64],
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let n = state.len();
    if n == 0 {
        return Err("state dimension must be positive".to_owned());
    }
    ensure_len(state, n, "state")?;
    ensure_square(covariance, n, "covariance")?;
    if !alpha.is_finite() || alpha <= 0.0 || !beta.is_finite() || !kappa.is_finite() {
        return Err("alpha, beta, and kappa must be finite with positive alpha".to_owned());
    }
    let lambda = alpha * alpha * (n as f64 + kappa) - n as f64;
    let scale = n as f64 + lambda;
    if !scale.is_finite() || scale <= 0.0 {
        return Err("alpha and kappa produce invalid sigma-point scaling".to_owned());
    }
    let mut scaled_covariance = covariance.to_vec();
    for value in &mut scaled_covariance {
        *value *= scale;
    }
    // Match the Python reference's small diagonal regularization before
    // factoring a semidefinite covariance.
    for index in 0..n {
        scaled_covariance[index * n + index] += 1.0e-12;
    }
    let lower = cholesky_lower(&scaled_covariance, n)?;
    let count = 2 * n + 1;
    let mut sigma = vec![0.0; count * n];
    sigma[..n].copy_from_slice(state);
    for index in 0..n {
        for axis in 0..n {
            sigma[(1 + 2 * index) * n + axis] = state[axis] + lower[axis * n + index];
            sigma[(2 + 2 * index) * n + axis] = state[axis] - lower[axis * n + index];
        }
    }
    let mut wm = vec![1.0 / (2.0 * scale); count];
    let mut wc = wm.clone();
    wm[0] = lambda / scale;
    wc[0] = wm[0] + (1.0 - alpha * alpha + beta);
    Ok((sigma, wm, wc))
}

/// Recombine propagated sigma points and add process covariance.
pub fn ukf_recombine(
    sigma_points: &[f64],
    mean_weights: &[f64],
    covariance_weights: &[f64],
    process_covariance: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if mean_weights.is_empty() || mean_weights.len() != covariance_weights.len() {
        return Err("UKF weight vectors must be non-empty and have matching lengths".to_owned());
    }
    let count = mean_weights.len();
    if sigma_points.len() % count != 0 {
        return Err("sigma_points length must be a multiple of the weight count".to_owned());
    }
    let n = sigma_points.len() / count;
    ensure_len(sigma_points, sigma_points.len(), "sigma_points")?;
    ensure_len(mean_weights, count, "mean_weights")?;
    ensure_len(covariance_weights, count, "covariance_weights")?;
    ensure_square(process_covariance, n, "process_covariance")?;
    let mut mean = vec![0.0; n];
    for sample in 0..count {
        for axis in 0..n {
            mean[axis] += mean_weights[sample] * sigma_points[sample * n + axis];
        }
    }
    let mut covariance = process_covariance.to_vec();
    for sample in 0..count {
        for row in 0..n {
            let delta_row = sigma_points[sample * n + row] - mean[row];
            for column in 0..n {
                covariance[row * n + column] += covariance_weights[sample]
                    * delta_row
                    * (sigma_points[sample * n + column] - mean[column]);
            }
        }
    }
    symmetrize(&mut covariance, n);
    if !mean.iter().all(|value| value.is_finite())
        || !covariance.iter().all(|value| value.is_finite())
    {
        return Err("UKF recombination produced non-finite values".to_owned());
    }
    Ok((mean, covariance))
}

/// Two-body RK4 prediction and the estimator's six forward differences.
/// The numerical owner is the shared RK4 engine; epsilon and summation order
/// match the existing Python estimator, with no variational approximation.
/// At zero duration, successful validation yields the exact identity transition.
pub fn two_body_predict_and_jacobian(
    state: &[f64],
    dt_s: f64,
    mu: f64,
    epsilon: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    ensure_len(state, 6, "state")?;
    if !dt_s.is_finite()
        || dt_s < 0.0
        || !mu.is_finite()
        || mu < 0.0
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err("prediction requires finite nonnegative dt and mu and positive epsilon".into());
    }
    let parameters = crate::OrbitParameters {
        mu_km3_s2: mu,
        include_j2: false,
        command_accel_eci_km_s2: [0.0; 3],
    };
    let initial = crate::State::from_array(state.try_into().unwrap()).map_err(str::to_owned)?;
    let predicted = crate::rk4_step_validated(initial, dt_s, parameters)
        .map_err(str::to_owned)?
        .as_array();
    let mut perturbed = [0.0; 36];
    for axis in 0..6 {
        let mut values: [f64; 6] = state.try_into().unwrap();
        values[axis] += epsilon;
        let initial = crate::State::from_array(values).map_err(str::to_owned)?;
        let values = crate::rk4_step_validated(initial, dt_s, parameters)
            .map_err(str::to_owned)?
            .as_array();
        perturbed[axis * 6..axis * 6 + 6].copy_from_slice(&values);
    }
    let mut jacobian = jacobian_from_perturbed(&predicted, &perturbed, epsilon)?;
    if dt_s == 0.0 {
        // Retain nominal/perturbed evaluation and validation order above.
        // A zero-time flow has an exact identity derivative, independent of
        // how state + epsilon rounds in the forward-difference calculation.
        jacobian.fill(0.0);
        for axis in 0..6 {
            jacobian[axis * 6 + axis] = 1.0;
        }
    }
    Ok((predicted.to_vec(), jacobian))
}

/// Assemble a forward-difference Jacobian from already evaluated perturbations.
pub fn jacobian_from_perturbed(
    base: &[f64],
    perturbed: &[f64],
    epsilon: f64,
) -> Result<Vec<f64>, String> {
    let n = base.len();
    ensure_len(base, n, "base evaluation")?;
    if n == 0 || perturbed.len() != n * n {
        return Err("perturbed evaluations must contain n*n values".to_owned());
    }
    ensure_len(perturbed, n * n, "perturbed evaluations")?;
    if !epsilon.is_finite() || epsilon == 0.0 {
        return Err("epsilon must be finite and nonzero".to_owned());
    }
    let mut jacobian = vec![0.0; n * n];
    for column in 0..n {
        for row in 0..n {
            jacobian[row * n + column] = (perturbed[column * n + row] - base[row]) / epsilon;
        }
    }
    Ok(jacobian)
}

/// Whiten variable-count residual blocks with precomputed lower factors.
pub fn whiten_residual_blocks(
    residuals: &[f64],
    factors: &[f64],
    block_dimensions: &[usize],
) -> Result<Vec<f64>, String> {
    let expected_residuals = block_dimensions.iter().try_fold(0usize, |sum, dimension| {
        sum.checked_add(*dimension)
            .ok_or_else(|| "residual dimension overflow".to_owned())
    })?;
    let expected_factors = block_dimensions.iter().try_fold(0usize, |sum, dimension| {
        let square = dimension
            .checked_mul(*dimension)
            .ok_or_else(|| "factor dimension overflow".to_owned())?;
        sum.checked_add(square)
            .ok_or_else(|| "factor dimension overflow".to_owned())
    })?;
    ensure_len(residuals, expected_residuals, "residuals")?;
    ensure_len(factors, expected_factors, "factors")?;
    let mut output = vec![0.0; expected_residuals];
    let mut residual_offset = 0usize;
    let mut factor_offset = 0usize;
    for dimension in block_dimensions {
        if *dimension == 0 {
            return Err("residual block dimensions must be positive".to_owned());
        }
        let factor = &factors[factor_offset..factor_offset + dimension * dimension];
        for row in 0..*dimension {
            let diagonal = factor[row * dimension + row];
            if diagonal <= 0.0 || !diagonal.is_finite() {
                return Err("residual factors must have positive finite diagonals".to_owned());
            }
            let mut value = residuals[residual_offset + row];
            for column in 0..row {
                let lower = factor[row * dimension + column];
                if !lower.is_finite() {
                    return Err("residual factors must be finite".to_owned());
                }
                value -= lower * output[residual_offset + column];
            }
            output[residual_offset + row] = value / diagonal;
        }
        residual_offset += *dimension;
        factor_offset += dimension * dimension;
    }
    Ok(output)
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::prelude::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    fn estimation_ekf_update(
        py: Python<'_>,
        state: Vec<f64>,
        covariance: Vec<f64>,
        measurement: Vec<f64>,
        measurement_matrix: Vec<f64>,
        measurement_covariance: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
        py.detach(|| {
            ekf_update(
                &state,
                &covariance,
                &measurement,
                &measurement_matrix,
                &measurement_covariance,
            )
            .map_err(PyValueError::new_err)
        })
    }

    #[pyfunction]
    fn estimation_ukf_sigma_points(
        py: Python<'_>,
        state: Vec<f64>,
        covariance: Vec<f64>,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        py.detach(|| {
            ukf_sigma_points(&state, &covariance, alpha, beta, kappa).map_err(PyValueError::new_err)
        })
    }

    #[pyfunction]
    fn estimation_ukf_recombine(
        py: Python<'_>,
        sigma_points: Vec<f64>,
        mean_weights: Vec<f64>,
        covariance_weights: Vec<f64>,
        process_covariance: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        py.detach(|| {
            ukf_recombine(
                &sigma_points,
                &mean_weights,
                &covariance_weights,
                &process_covariance,
            )
            .map_err(PyValueError::new_err)
        })
    }

    #[pyfunction]
    fn estimation_jacobian_from_perturbed(
        py: Python<'_>,
        base: Vec<f64>,
        perturbed: Vec<f64>,
        epsilon: f64,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            jacobian_from_perturbed(&base, &perturbed, epsilon).map_err(PyValueError::new_err)
        })
    }

    #[pyfunction]
    fn estimation_jacobian_from_perturbed_bytes(
        py: Python<'_>,
        base: &[u8],
        perturbed: &[u8],
        epsilon: f64,
    ) -> PyResult<Py<PyBytes>> {
        let base = decode_f64_bytes(base)?;
        let perturbed = decode_f64_bytes(perturbed)?;
        let jacobian = py
            .detach(|| jacobian_from_perturbed(&base, &perturbed, epsilon))
            .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &jacobian)?)
    }

    #[pyfunction]
    fn estimation_whiten_residual_blocks(
        py: Python<'_>,
        residuals: Vec<f64>,
        factors: Vec<f64>,
        block_dimensions: Vec<usize>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            whiten_residual_blocks(&residuals, &factors, &block_dimensions)
                .map_err(PyValueError::new_err)
        })
    }

    #[pyfunction]
    fn estimation_whiten_residual_blocks_bytes(
        py: Python<'_>,
        residuals: &[u8],
        factors: &[u8],
        block_dimensions: Vec<usize>,
    ) -> PyResult<Py<PyBytes>> {
        let residuals = decode_f64_bytes(residuals)?;
        let factors = decode_f64_bytes(factors)?;
        let values = py
            .detach(|| whiten_residual_blocks(&residuals, &factors, &block_dimensions))
            .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }

    #[pyfunction]
    fn estimation_ekf_update_bytes(
        py: Python<'_>,
        state: &[u8],
        covariance: &[u8],
        measurement: &[u8],
        measurement_matrix: &[u8],
        measurement_covariance: &[u8],
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, f64)> {
        let (state, covariance, innovation, innovation_covariance, nis) = ekf_update(
            &decode_f64_bytes(state)?,
            &decode_f64_bytes(covariance)?,
            &decode_f64_bytes(measurement)?,
            &decode_f64_bytes(measurement_matrix)?,
            &decode_f64_bytes(measurement_covariance)?,
        )
        .map_err(PyValueError::new_err)?;
        Ok((
            output_bytes(py, &state)?,
            output_bytes(py, &covariance)?,
            output_bytes(py, &innovation)?,
            output_bytes(py, &innovation_covariance)?,
            nis,
        ))
    }

    #[pyfunction]
    fn estimation_ekf_update_innovation_bytes(
        py: Python<'_>,
        state: &[u8],
        covariance: &[u8],
        innovation: &[u8],
        measurement_matrix: &[u8],
        measurement_covariance: &[u8],
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, f64)> {
        let (state, covariance, innovation, innovation_covariance, nis) = ekf_update_innovation(
            &decode_f64_bytes(state)?,
            &decode_f64_bytes(covariance)?,
            &decode_f64_bytes(innovation)?,
            &decode_f64_bytes(measurement_matrix)?,
            &decode_f64_bytes(measurement_covariance)?,
        )
        .map_err(PyValueError::new_err)?;
        Ok((
            output_bytes(py, &state)?,
            output_bytes(py, &covariance)?,
            output_bytes(py, &innovation)?,
            output_bytes(py, &innovation_covariance)?,
            nis,
        ))
    }
    #[pyfunction]
    fn estimation_ukf_sigma_points_bytes(
        py: Python<'_>,
        state: &[u8],
        covariance: &[u8],
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Py<PyBytes>)> {
        let (sigma, wm, wc) = ukf_sigma_points(
            &decode_f64_bytes(state)?,
            &decode_f64_bytes(covariance)?,
            alpha,
            beta,
            kappa,
        )
        .map_err(PyValueError::new_err)?;
        Ok((
            output_bytes(py, &sigma)?,
            output_bytes(py, &wm)?,
            output_bytes(py, &wc)?,
        ))
    }
    #[pyfunction]
    fn estimation_ukf_recombine_bytes(
        py: Python<'_>,
        sigma_points: &[u8],
        mean_weights: &[u8],
        covariance_weights: &[u8],
        process_covariance: &[u8],
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let (mean, covariance) = ukf_recombine(
            &decode_f64_bytes(sigma_points)?,
            &decode_f64_bytes(mean_weights)?,
            &decode_f64_bytes(covariance_weights)?,
            &decode_f64_bytes(process_covariance)?,
        )
        .map_err(PyValueError::new_err)?;
        Ok((output_bytes(py, &mean)?, output_bytes(py, &covariance)?))
    }
    #[pyfunction]
    fn estimation_two_body_predict_and_jacobian_bytes(
        py: Python<'_>,
        state: &[u8],
        dt_s: f64,
        mu: f64,
        epsilon: f64,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let (state, phi) =
            two_body_predict_and_jacobian(&decode_f64_bytes(state)?, dt_s, mu, epsilon)
                .map_err(PyValueError::new_err)?;
        Ok((output_bytes(py, &state)?, output_bytes(py, &phi)?))
    }
    module.add_function(wrap_pyfunction!(
        estimation_two_body_predict_and_jacobian_bytes,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(estimation_ekf_update_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(
        estimation_ekf_update_innovation_bytes,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(estimation_ukf_sigma_points_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(estimation_ukf_recombine_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(estimation_ekf_update, module)?)?;
    module.add_function(wrap_pyfunction!(estimation_ukf_sigma_points, module)?)?;
    module.add_function(wrap_pyfunction!(estimation_ukf_recombine, module)?)?;
    module.add_function(wrap_pyfunction!(
        estimation_jacobian_from_perturbed,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        estimation_jacobian_from_perturbed_bytes,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(estimation_whiten_residual_blocks, module)?)?;
    module.add_function(wrap_pyfunction!(
        estimation_whiten_residual_blocks_bytes,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ekf_update, jacobian_from_perturbed, two_body_predict_and_jacobian, ukf_recombine,
        ukf_sigma_points, whiten_residual_blocks,
    };

    #[test]
    fn zero_duration_prediction_has_exact_identity_transition() {
        let states = [
            [7001.0, 2.0, -0.5, 0.01, 7.49, -0.015],
            [42164.0, 0.02, -0.01, 0.001, 3.074666284, -0.002],
        ];
        for state in states {
            for dt in [0.0, -0.0] {
                for mu in [0.0, 398600.4415] {
                    for epsilon in [5.0e-7, 1.0e-6, 2.0e-6] {
                        let (predicted, transition) =
                            two_body_predict_and_jacobian(&state, dt, mu, epsilon).unwrap();
                        assert_eq!(predicted.as_slice(), &state);
                        for row in 0..6 {
                            for column in 0..6 {
                                let expected = if row == column { 1.0 } else { 0.0 };
                                assert_eq!(transition[row * 6 + column], expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_duration_prediction_retains_validation_order() {
        let state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let scalar_error = "prediction requires finite nonnegative dt and mu and positive epsilon";
        assert_eq!(
            two_body_predict_and_jacobian(&[f64::NAN; 5], f64::NAN, -1.0, 0.0).unwrap_err(),
            "state must contain 6 values"
        );
        assert_eq!(
            two_body_predict_and_jacobian(&[f64::NAN; 6], f64::NAN, -1.0, 0.0).unwrap_err(),
            "state must contain only finite values"
        );
        assert_eq!(
            two_body_predict_and_jacobian(&[0.0; 6], 0.0, -1.0, 0.0).unwrap_err(),
            scalar_error
        );
        for dt in [0.0, -0.0, 1.0] {
            for (mu, epsilon) in [
                (-1.0, 1.0e-6),
                (f64::NAN, 1.0e-6),
                (f64::INFINITY, 1.0e-6),
                (398600.4415, 0.0),
                (398600.4415, -1.0e-6),
                (398600.4415, f64::NAN),
                (398600.4415, f64::INFINITY),
            ] {
                assert_eq!(
                    two_body_predict_and_jacobian(&state, dt, mu, epsilon).unwrap_err(),
                    scalar_error
                );
            }
            assert_eq!(
                two_body_predict_and_jacobian(&[0.0; 6], dt, 398600.4415, 1.0e-6).unwrap_err(),
                "position must be nonzero"
            );
        }
        for dt in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                two_body_predict_and_jacobian(&state, dt, 398600.4415, 1.0e-6).unwrap_err(),
                scalar_error
            );
        }
    }

    #[test]
    fn zero_duration_prediction_retains_perturbed_state_validation() {
        // Zero duration must not skip the existing perturbation-domain errors.
        for dt in [0.0, -0.0] {
            assert_eq!(
                two_body_predict_and_jacobian(&[-1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0], dt, 0.0, 1.0e-6)
                    .unwrap_err(),
                "position must be nonzero"
            );
            assert_eq!(
                two_body_predict_and_jacobian(
                    &[f64::MAX, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dt,
                    0.0,
                    f64::MAX
                )
                .unwrap_err(),
                "state must contain only finite values"
            );
        }
    }

    #[test]
    fn correlated_ekf_update_matches_expected_identity_case() {
        let state = vec![1.0, -2.0];
        let covariance = vec![4.0, 1.0, 1.0, 3.0];
        let measurement = vec![1.5, -1.0];
        let measurement_covariance = vec![1.0, 0.2, 0.2, 2.0];
        let (updated, posterior, innovation, _, nis) = ekf_update(
            &state,
            &covariance,
            &measurement,
            &[1.0, 0.0, 0.0, 1.0],
            &measurement_covariance,
        )
        .unwrap();
        assert_eq!(innovation, vec![0.5, 1.0]);
        assert!(nis > 0.0);
        assert!(updated.iter().all(|value| value.is_finite()));
        assert!(posterior[0] < covariance[0]);
        assert!((posterior[1] - posterior[2]).abs() < 1.0e-12);
    }

    #[test]
    fn rectangular_ekf_update_accepts_non_square_measurements() {
        let state = [1.0, -2.0, 0.4, 3.0, -0.7, 1.2];
        let covariance = [
            2.0, 0.2, 0.1, 0.0, 0.1, 0.0, 0.2, 1.5, 0.0, 0.1, 0.0, 0.05, 0.1, 0.0, 1.2, 0.0, 0.02,
            0.0, 0.0, 0.1, 0.0, 0.9, 0.0, 0.03, 0.1, 0.0, 0.02, 0.0, 1.1, 0.04, 0.0, 0.05, 0.0,
            0.03, 0.04, 0.8,
        ];
        let measurement_matrix = [
            1.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.1, 0.2, -0.1, 0.0, 0.0, 0.0,
            1.0,
        ];
        let measurement = [0.9, 1.2, 1.5];
        let measurement_covariance = [0.4, 0.05, 0.02, 0.05, 0.6, 0.04, 0.02, 0.04, 0.5];
        let (updated, posterior, innovation, innovation_covariance, nis) = ekf_update(
            &state,
            &covariance,
            &measurement,
            &measurement_matrix,
            &measurement_covariance,
        )
        .unwrap();
        assert_eq!(innovation.len(), 3);
        assert_eq!(innovation_covariance.len(), 9);
        assert_eq!(updated.len(), 6);
        assert_eq!(posterior.len(), 36);
        assert!(nis.is_finite());
        assert!(updated.iter().all(|value| value.is_finite()));
        assert!(posterior.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn ukf_sigma_points_recombine_constant_map() {
        let state = vec![2.0, -1.0];
        let covariance = vec![1.0, 0.2, 0.2, 2.0];
        let (sigma, wm, wc) = ukf_sigma_points(&state, &covariance, 0.4, 2.0, 0.0).unwrap();
        let (mean, result) = ukf_recombine(&sigma, &wm, &wc, &[0.0; 4]).unwrap();
        assert!((mean[0] - state[0]).abs() < 1.0e-10);
        assert!((mean[1] - state[1]).abs() < 1.0e-10);
        assert!((result[0] - covariance[0]).abs() < 1.0e-9);
        assert!((result[1] - covariance[1]).abs() < 1.0e-9);
    }

    #[test]
    fn batch_helpers_preserve_block_order() {
        let values =
            whiten_residual_blocks(&[2.0, 3.0, 4.0], &[2.0, 0.0, 1.0, 1.0, 3.0], &[2, 1]).unwrap();
        assert!((values[0] - 1.0).abs() < 1.0e-12);
        assert!((values[1] - 2.0).abs() < 1.0e-12);
        assert!((values[2] - 4.0 / 3.0).abs() < 1.0e-12);
        let jac = jacobian_from_perturbed(&[1.0, 2.0], &[1.1, 2.0, 1.0, 2.2], 0.1).unwrap();
        let expected = [1.0, 0.0, 0.0, 2.0];
        assert_eq!(jac.len(), expected.len());
        for (actual, expected) in jac.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }
}
