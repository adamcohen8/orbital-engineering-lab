//! Measurement geometry and residual batching kernels.
//!
//! These functions deliberately accept already transformed geometry.  Frame
//! models, EOP policy, light-time iteration, station catalogs, and provenance
//! remain Python-owned contracts.  The native boundary therefore cannot
//! silently change an observation's frame or correction policy.

// Signatures follow the macOS SDK vForce.h double precision API. Small
// collections and non-macOS platforms retain scalar libm calls.
#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vvatan2(output: *mut f64, y: *const f64, x: *const f64, count: *const i32);
    fn vvasin(output: *mut f64, input: *const f64, count: *const i32);
}

fn angles_batch(y: &[f64], x: &[f64], sine_elevation: &[f64]) -> (Vec<f64>, Vec<f64>) {
    #[cfg(target_os = "macos")]
    if y.len() >= 16 && y.len() <= i32::MAX as usize {
        let count = y.len() as i32;
        let mut azimuth = vec![0.0; y.len()];
        let mut elevation = vec![0.0; y.len()];
        // SAFETY: all input/output arrays contain count f64 values, output
        // allocations are distinct, and vForce completes synchronously.
        unsafe {
            vvatan2(azimuth.as_mut_ptr(), y.as_ptr(), x.as_ptr(), &count);
            vvasin(elevation.as_mut_ptr(), sine_elevation.as_ptr(), &count);
        }
        return (azimuth, elevation);
    }
    (
        y.iter().zip(x).map(|(y, x)| y.atan2(*x)).collect(),
        sine_elevation.iter().map(|value| value.asin()).collect(),
    )
}

fn ensure_len(values: &[f64], expected: usize, name: &str) -> Result<(), String> {
    if values.len() != expected {
        return Err(format!("{name} must contain {expected} values"));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

/// Whiten fixed-width residual rows using row-major lower Cholesky factors.
pub fn whiten_rows(
    residuals: &[f64],
    factors: &[f64],
    rows: usize,
    dimension: usize,
) -> Result<Vec<f64>, String> {
    let residual_count = rows
        .checked_mul(dimension)
        .ok_or_else(|| "residual dimensions overflow".to_owned())?;
    let factor_count = rows
        .checked_mul(dimension)
        .and_then(|value| value.checked_mul(dimension))
        .ok_or_else(|| "factor dimensions overflow".to_owned())?;
    ensure_len(residuals, residual_count, "residuals")?;
    ensure_len(factors, factor_count, "factors")?;
    let mut output = vec![0.0; residual_count];
    for row in 0..rows {
        let residual_offset = row * dimension;
        let factor_offset = row * dimension * dimension;
        for i in 0..dimension {
            let diagonal = factors[factor_offset + i * dimension + i];
            if diagonal <= 0.0 || !diagonal.is_finite() {
                return Err("Cholesky factors must have positive finite diagonals".to_owned());
            }
            let mut value = residuals[residual_offset + i];
            for j in 0..i {
                value -= factors[factor_offset + i * dimension + j] * output[residual_offset + j];
            }
            output[residual_offset + i] = value / diagonal;
        }
    }
    Ok(output)
}

/// Whiten a batch whose rows may have different dimensions.
pub fn whiten_variable_rows(
    residuals: &[f64],
    factors: &[f64],
    dimensions: &[usize],
) -> Result<Vec<f64>, String> {
    let expected_residuals = dimensions.iter().try_fold(0usize, |sum, dimension| {
        sum.checked_add(*dimension)
            .ok_or_else(|| "residual dimensions overflow".to_owned())
    })?;
    let expected_factors = dimensions.iter().try_fold(0usize, |sum, dimension| {
        let square = dimension
            .checked_mul(*dimension)
            .ok_or_else(|| "factor dimensions overflow".to_owned())?;
        sum.checked_add(square)
            .ok_or_else(|| "factor dimensions overflow".to_owned())
    })?;
    ensure_len(residuals, expected_residuals, "residuals")?;
    ensure_len(factors, expected_factors, "factors")?;
    let mut output = vec![0.0; expected_residuals];
    let mut residual_offset = 0usize;
    let mut factor_offset = 0usize;
    for dimension in dimensions {
        if *dimension == 0 {
            return Err("measurement rows must have positive dimensions".to_owned());
        }
        for i in 0..*dimension {
            let diagonal = factors[factor_offset + i * dimension + i];
            if diagonal <= 0.0 || !diagonal.is_finite() {
                return Err("Cholesky factors must have positive finite diagonals".to_owned());
            }
            let mut value = residuals[residual_offset + i];
            for j in 0..i {
                value -= factors[factor_offset + i * dimension + j] * output[residual_offset + j];
            }
            output[residual_offset + i] = value / diagonal;
        }
        residual_offset += *dimension;
        factor_offset += dimension * dimension;
    }
    Ok(output)
}

/// Predict azimuth, elevation, range, and range-rate from prepared geometry.
///
/// `states_eci` is row-major `(N, 6)`, `target_ecef` and the station vectors
/// are `(N, 3)`, and `enu_rotations` is `(N, 3, 3)` mapping ECEF deltas to ENU.
pub fn ground_station_predictions(
    states_eci: &[f64],
    target_ecef: &[f64],
    station_ecef: &[f64],
    station_eci: &[f64],
    station_velocity_eci: &[f64],
    enu_rotations: &[f64],
) -> Result<Vec<f64>, String> {
    if states_eci.len() % 6 != 0 {
        return Err("states_eci must contain rows of six values".to_owned());
    }
    let rows = states_eci.len() / 6;
    ensure_len(states_eci, rows * 6, "states_eci")?;
    ensure_len(target_ecef, rows * 3, "target_ecef")?;
    ensure_len(station_ecef, rows * 3, "station_ecef")?;
    ensure_len(station_eci, rows * 3, "station_eci")?;
    ensure_len(station_velocity_eci, rows * 3, "station_velocity_eci")?;
    ensure_len(enu_rotations, rows * 9, "enu_rotations")?;
    let mut output = vec![0.0; rows * 4];
    let vector_angles = cfg!(target_os = "macos") && rows >= 16 && rows <= i32::MAX as usize;
    let mut azimuth_y = if vector_angles {
        vec![0.0; rows]
    } else {
        Vec::new()
    };
    let mut azimuth_x = if vector_angles {
        vec![1.0; rows]
    } else {
        Vec::new()
    };
    let mut sine_elevation = if vector_angles {
        vec![1.0; rows]
    } else {
        Vec::new()
    };
    for row in 0..rows {
        let state = &states_eci[row * 6..row * 6 + 6];
        let target = &target_ecef[row * 3..row * 3 + 3];
        let station = &station_ecef[row * 3..row * 3 + 3];
        let station_eci_row = &station_eci[row * 3..row * 3 + 3];
        let station_velocity = &station_velocity_eci[row * 3..row * 3 + 3];
        let rotation = &enu_rotations[row * 9..row * 9 + 9];
        let delta_ecef = [
            target[0] - station[0],
            target[1] - station[1],
            target[2] - station[2],
        ];
        let enu = [
            rotation[0] * delta_ecef[0] + rotation[1] * delta_ecef[1] + rotation[2] * delta_ecef[2],
            rotation[3] * delta_ecef[0] + rotation[4] * delta_ecef[1] + rotation[5] * delta_ecef[2],
            rotation[6] * delta_ecef[0] + rotation[7] * delta_ecef[1] + rotation[8] * delta_ecef[2],
        ];
        let range = (delta_ecef[0] * delta_ecef[0]
            + delta_ecef[1] * delta_ecef[1]
            + delta_ecef[2] * delta_ecef[2])
            .sqrt();
        let (azimuth, elevation, range_rate) = if range <= 0.0 {
            (0.0, std::f64::consts::FRAC_PI_2, 0.0)
        } else {
            let sine = (enu[2] / range).clamp(-1.0, 1.0);
            let (azimuth, elevation) = if vector_angles {
                azimuth_y[row] = enu[0];
                azimuth_x[row] = enu[1];
                sine_elevation[row] = sine;
                (0.0, 0.0)
            } else {
                (
                    enu[0].atan2(enu[1]).rem_euclid(2.0 * std::f64::consts::PI),
                    sine.asin(),
                )
            };
            let delta_eci = [
                state[0] - station_eci_row[0],
                state[1] - station_eci_row[1],
                state[2] - station_eci_row[2],
            ];
            let distance_eci = (delta_eci[0] * delta_eci[0]
                + delta_eci[1] * delta_eci[1]
                + delta_eci[2] * delta_eci[2])
                .sqrt()
                .max(1.0e-12);
            let line_of_sight = [
                delta_eci[0] / distance_eci,
                delta_eci[1] / distance_eci,
                delta_eci[2] / distance_eci,
            ];
            let relative_velocity = [
                state[3] - station_velocity[0],
                state[4] - station_velocity[1],
                state[5] - station_velocity[2],
            ];
            let range_rate = relative_velocity[0] * line_of_sight[0]
                + relative_velocity[1] * line_of_sight[1]
                + relative_velocity[2] * line_of_sight[2];
            (azimuth, elevation, range_rate)
        };
        output[row * 4] = azimuth.to_degrees();
        output[row * 4 + 1] = elevation.to_degrees();
        output[row * 4 + 2] = range;
        output[row * 4 + 3] = range_rate;
    }
    if vector_angles {
        let (azimuth, elevation) = angles_batch(&azimuth_y, &azimuth_x, &sine_elevation);
        for row in 0..rows {
            output[row * 4] = azimuth[row]
                .rem_euclid(2.0 * std::f64::consts::PI)
                .to_degrees();
            output[row * 4 + 1] = elevation[row].to_degrees();
        }
    }
    Ok(output)
}

/// Predict right ascension and declination in degrees from relative ECI rows.
pub fn optical_radec_predictions(relative_positions: &[f64]) -> Result<Vec<f64>, String> {
    if relative_positions.len() % 3 != 0 {
        return Err("relative_positions must contain rows of three values".to_owned());
    }
    let rows = relative_positions.len() / 3;
    ensure_len(relative_positions, rows * 3, "relative_positions")?;
    let mut output = vec![0.0; rows * 2];
    let vector_angles = cfg!(target_os = "macos") && rows >= 16 && rows <= i32::MAX as usize;
    let mut azimuth_y = if vector_angles {
        vec![0.0; rows]
    } else {
        Vec::new()
    };
    let mut azimuth_x = if vector_angles {
        vec![0.0; rows]
    } else {
        Vec::new()
    };
    let mut sine_elevation = if vector_angles {
        vec![0.0; rows]
    } else {
        Vec::new()
    };
    for row in 0..rows {
        let x = relative_positions[row * 3];
        let y = relative_positions[row * 3 + 1];
        let z = relative_positions[row * 3 + 2];
        let radius = (x * x + y * y + z * z).sqrt();
        if radius <= 0.0 {
            return Err("optical relative position must be nonzero".to_owned());
        }
        let sine = (z / radius).clamp(-1.0, 1.0);
        if vector_angles {
            azimuth_y[row] = y;
            azimuth_x[row] = x;
            sine_elevation[row] = sine;
        } else {
            output[row * 2] = y.atan2(x).to_degrees().rem_euclid(360.0);
            output[row * 2 + 1] = sine.asin().to_degrees();
        }
    }
    if vector_angles {
        let (azimuth, elevation) = angles_batch(&azimuth_y, &azimuth_x, &sine_elevation);
        for row in 0..rows {
            output[row * 2] = azimuth[row].to_degrees().rem_euclid(360.0);
            output[row * 2 + 1] = elevation[row].to_degrees();
        }
    }
    Ok(output)
}

/// Predict geometric range for radar/SLR rows from relative positions.
pub fn range_predictions(relative_positions: &[f64]) -> Result<Vec<f64>, String> {
    if relative_positions.len() % 3 != 0 {
        return Err("relative_positions must contain rows of three values".to_owned());
    }
    let rows = relative_positions.len() / 3;
    ensure_len(relative_positions, rows * 3, "relative_positions")?;
    Ok(relative_positions
        .chunks_exact(3)
        .map(|row| (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt())
        .collect())
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::prelude::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    fn measurement_whiten_rows(
        py: Python<'_>,
        residuals: Vec<f64>,
        factors: Vec<f64>,
        rows: usize,
        dimension: usize,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            whiten_rows(&residuals, &factors, rows, dimension).map_err(PyValueError::new_err)
        })
    }

    #[pyfunction]
    fn measurement_whiten_rows_bytes(
        py: Python<'_>,
        residuals: &[u8],
        factors: &[u8],
        rows: usize,
        dimension: usize,
    ) -> PyResult<Py<PyBytes>> {
        let residuals = decode_f64_bytes(residuals)?;
        let factors = decode_f64_bytes(factors)?;
        let values = py
            .detach(|| whiten_rows(&residuals, &factors, rows, dimension))
            .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }

    #[pyfunction]
    fn measurement_whiten_variable_rows(
        py: Python<'_>,
        residuals: Vec<f64>,
        factors: Vec<f64>,
        dimensions: Vec<usize>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            whiten_variable_rows(&residuals, &factors, &dimensions).map_err(PyValueError::new_err)
        })
    }

    #[pyfunction]
    fn measurement_whiten_variable_rows_bytes(
        py: Python<'_>,
        residuals: &[u8],
        factors: &[u8],
        dimensions: Vec<usize>,
    ) -> PyResult<Py<PyBytes>> {
        let residuals = decode_f64_bytes(residuals)?;
        let factors = decode_f64_bytes(factors)?;
        let values = py
            .detach(|| whiten_variable_rows(&residuals, &factors, &dimensions))
            .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }

    #[pyfunction]
    fn measurement_ground_station_batch(
        py: Python<'_>,
        states_eci: Vec<f64>,
        target_ecef: Vec<f64>,
        station_ecef: Vec<f64>,
        station_eci: Vec<f64>,
        station_velocity_eci: Vec<f64>,
        enu_rotations: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            ground_station_predictions(
                &states_eci,
                &target_ecef,
                &station_ecef,
                &station_eci,
                &station_velocity_eci,
                &enu_rotations,
            )
            .map_err(PyValueError::new_err)
        })
    }

    #[pyfunction]
    fn measurement_ground_station_batch_bytes(
        py: Python<'_>,
        states_eci: &[u8],
        target_ecef: &[u8],
        station_ecef: &[u8],
        station_eci: &[u8],
        station_velocity_eci: &[u8],
        enu_rotations: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let states_eci = decode_f64_bytes(states_eci)?;
        let target_ecef = decode_f64_bytes(target_ecef)?;
        let station_ecef = decode_f64_bytes(station_ecef)?;
        let station_eci = decode_f64_bytes(station_eci)?;
        let station_velocity_eci = decode_f64_bytes(station_velocity_eci)?;
        let enu_rotations = decode_f64_bytes(enu_rotations)?;
        let values = py
            .detach(|| {
                ground_station_predictions(
                    &states_eci,
                    &target_ecef,
                    &station_ecef,
                    &station_eci,
                    &station_velocity_eci,
                    &enu_rotations,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }

    #[pyfunction]
    fn measurement_optical_radec_batch(
        py: Python<'_>,
        relative_positions: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| optical_radec_predictions(&relative_positions).map_err(PyValueError::new_err))
    }

    #[pyfunction]
    fn measurement_optical_radec_batch_bytes(
        py: Python<'_>,
        relative_positions: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let relative_positions = decode_f64_bytes(relative_positions)?;
        let values = py
            .detach(|| optical_radec_predictions(&relative_positions))
            .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }

    #[pyfunction]
    fn measurement_range_batch(py: Python<'_>, relative_positions: Vec<f64>) -> PyResult<Vec<f64>> {
        py.detach(|| range_predictions(&relative_positions).map_err(PyValueError::new_err))
    }

    #[pyfunction]
    fn measurement_range_batch_bytes(
        py: Python<'_>,
        relative_positions: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let relative_positions = decode_f64_bytes(relative_positions)?;
        let values = py
            .detach(|| range_predictions(&relative_positions))
            .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }

    module.add_function(wrap_pyfunction!(measurement_whiten_rows, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_whiten_rows_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_whiten_variable_rows, module)?)?;
    module.add_function(wrap_pyfunction!(
        measurement_whiten_variable_rows_bytes,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(measurement_ground_station_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        measurement_ground_station_batch_bytes,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(measurement_optical_radec_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        measurement_optical_radec_batch_bytes,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(measurement_range_batch, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_range_batch_bytes, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ground_station_predictions, optical_radec_predictions, range_predictions, whiten_rows,
    };

    #[test]
    fn correlated_rows_are_whitened_forward() {
        let result = whiten_rows(&[2.0, 3.0], &[2.0, 0.0, 1.0, 1.0], 1, 2).unwrap();
        assert!((result[0] - 1.0).abs() < 1.0e-12);
        assert!((result[1] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn simple_measurement_geometry_matches_contract() {
        let radec = optical_radec_predictions(&[1.0, 1.0, 1.0]).unwrap();
        assert!((radec[0] - 45.0).abs() < 1.0e-12);
        assert!((radec[1] - 35.264389682754654).abs() < 1.0e-12);
        let range = range_predictions(&[3.0, 4.0, 12.0]).unwrap();
        assert!((range[0] - 13.0).abs() < 1.0e-12);
    }

    #[test]
    fn ground_station_identity_geometry_is_deterministic() {
        let result = ground_station_predictions(
            &[2.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            &[2.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        assert!((result[0] - 63.43494882292201).abs() < 1.0e-12);
        assert!((result[1] - 24.094842552110705).abs() < 1.0e-12);
        assert!((result[2] - 2.449489742783178).abs() < 1.0e-12);
        assert!((result[3] - 0.4082482904638631).abs() < 1.0e-12);
    }
}
