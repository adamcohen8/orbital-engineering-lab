//! Relative onboard tracking measurement kernels.
//!
//! The Python knowledge layer owns detection, sensor pose, noise generation,
//! cadence, and estimator policy.  This module only evaluates the prepared
//! target/observer ECI states.  Keeping that boundary explicit means that a
//! Rust selected backend cannot change the observation frame or consume the
//! Python RNG in a different order.

use std::f64::consts::PI;

const STATE_DIMENSION: usize = 6;
const MAX_MEASUREMENT_DIMENSION: usize = 4;

/// Fixed model codes used at the Python boundary.
///
/// The codes are intentionally stable and avoid repeatedly parsing model
/// strings inside a batched call.  They mirror the five relative measurement
/// models in `sim.knowledge.object_tracking`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementModel {
    Range = 0,
    RangeRate = 1,
    Angles = 2,
    AnglesRange = 3,
    AnglesRangeRate = 4,
}

impl MeasurementModel {
    fn from_code(code: i32) -> Result<Self, String> {
        match code {
            0 => Ok(Self::Range),
            1 => Ok(Self::RangeRate),
            2 => Ok(Self::Angles),
            3 => Ok(Self::AnglesRange),
            4 => Ok(Self::AnglesRangeRate),
            _ => Err(format!("unsupported relative tracking model code {code}")),
        }
    }

    pub fn dimension(self) -> usize {
        match self {
            Self::Range => 1,
            Self::RangeRate => 2,
            Self::Angles => 2,
            Self::AnglesRange => 3,
            Self::AnglesRangeRate => 4,
        }
    }

    fn wraps_angles(self) -> bool {
        matches!(
            self,
            Self::Angles | Self::AnglesRange | Self::AnglesRangeRate
        )
    }
}

fn validate_state(state: &[f64; STATE_DIMENSION], name: &str) -> Result<(), String> {
    if state.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn validate_steps(steps: &[f64]) -> Result<[f64; STATE_DIMENSION], String> {
    if steps.len() != STATE_DIMENSION {
        return Err("finite-difference steps must contain 6 values".to_owned());
    }
    if steps
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("finite-difference steps must be finite and positive".to_owned());
    }
    let mut output = [0.0; STATE_DIMENSION];
    output.copy_from_slice(steps);
    Ok(output)
}

fn wrap_angle(value: f64) -> f64 {
    (value + PI).rem_euclid(2.0 * PI) - PI
}

fn measurement(
    target: &[f64; STATE_DIMENSION],
    observer: &[f64; STATE_DIMENSION],
    model: MeasurementModel,
) -> [f64; MAX_MEASUREMENT_DIMENSION] {
    let relative_position = [
        target[0] - observer[0],
        target[1] - observer[1],
        target[2] - observer[2],
    ];
    let relative_velocity = [
        target[3] - observer[3],
        target[4] - observer[4],
        target[5] - observer[5],
    ];
    let range = (relative_position[0] * relative_position[0]
        + relative_position[1] * relative_position[1]
        + relative_position[2] * relative_position[2])
        .sqrt();
    let (line_of_sight, range_rate) = if range <= 0.0 {
        ([0.0; 3], 0.0)
    } else {
        let line_of_sight = [
            relative_position[0] / range,
            relative_position[1] / range,
            relative_position[2] / range,
        ];
        let range_rate = relative_velocity[0] * line_of_sight[0]
            + relative_velocity[1] * line_of_sight[1]
            + relative_velocity[2] * line_of_sight[2];
        (line_of_sight, range_rate)
    };
    let (azimuth, elevation) = if model.wraps_angles() && range > 0.0 {
        (
            line_of_sight[1].atan2(line_of_sight[0]),
            line_of_sight[2].clamp(-1.0, 1.0).asin(),
        )
    } else {
        (0.0, 0.0)
    };
    match model {
        MeasurementModel::Range => [range, 0.0, 0.0, 0.0],
        MeasurementModel::RangeRate => [range, range_rate, 0.0, 0.0],
        MeasurementModel::Angles => [azimuth, elevation, 0.0, 0.0],
        MeasurementModel::AnglesRange => [azimuth, elevation, range, 0.0],
        MeasurementModel::AnglesRangeRate => [azimuth, elevation, range, range_rate],
    }
}

/// Evaluate one relative measurement vector.
pub fn tracking_measurement(
    target: [f64; STATE_DIMENSION],
    observer: [f64; STATE_DIMENSION],
    model_code: i32,
) -> Result<Vec<f64>, String> {
    validate_state(&target, "target_state")?;
    validate_state(&observer, "observer_state")?;
    let model = MeasurementModel::from_code(model_code)?;
    Ok(measurement(&target, &observer, model)[..model.dimension()].to_vec())
}

/// Evaluate one measurement and its forward finite-difference Jacobian.
///
/// The base vector and all six perturbed vectors are evaluated in this one
/// native call.  The operation order, perturbation sizes, and angle wrapping
/// mirror `_relative_measurement_jacobian` in the Python reference path.
pub fn tracking_measurement_and_jacobian(
    target: [f64; STATE_DIMENSION],
    observer: [f64; STATE_DIMENSION],
    model_code: i32,
    steps: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    validate_state(&target, "target_state")?;
    validate_state(&observer, "observer_state")?;
    let model = MeasurementModel::from_code(model_code)?;
    let steps = validate_steps(steps)?;
    let base = measurement(&target, &observer, model);
    let dimension = model.dimension();
    let mut jacobian = vec![0.0; dimension * STATE_DIMENSION];
    for axis in 0..STATE_DIMENSION {
        let mut perturbed = target;
        perturbed[axis] += steps[axis];
        let plus = measurement(&perturbed, &observer, model);
        for row in 0..dimension {
            let mut delta = plus[row] - base[row];
            if model.wraps_angles() && row < 2 {
                delta = wrap_angle(delta);
            }
            jacobian[row * STATE_DIMENSION + axis] = delta / steps[axis];
        }
    }
    if jacobian.iter().any(|value| !value.is_finite()) {
        return Err("relative tracking Jacobian produced non-finite values".to_owned());
    }
    Ok((base[..dimension].to_vec(), jacobian))
}

/// Evaluate a batch of target/observer states using one model code.
pub fn tracking_measurement_batch(
    target_states: &[f64],
    observer_states: &[f64],
    model_code: i32,
) -> Result<Vec<f64>, String> {
    if target_states.len() != observer_states.len() || target_states.len() % STATE_DIMENSION != 0 {
        return Err(
            "target_states and observer_states must have equal lengths divisible by 6".to_owned(),
        );
    }
    if target_states.iter().any(|value| !value.is_finite())
        || observer_states.iter().any(|value| !value.is_finite())
    {
        return Err("tracking batch states must contain only finite values".to_owned());
    }
    let model = MeasurementModel::from_code(model_code)?;
    let mut output =
        Vec::with_capacity((target_states.len() / STATE_DIMENSION) * model.dimension());
    for offset in (0..target_states.len()).step_by(STATE_DIMENSION) {
        let target: [f64; STATE_DIMENSION] = target_states[offset..offset + STATE_DIMENSION]
            .try_into()
            .expect("validated six-state batch row");
        let observer: [f64; STATE_DIMENSION] = observer_states[offset..offset + STATE_DIMENSION]
            .try_into()
            .expect("validated six-state batch row");
        output.extend_from_slice(&measurement(&target, &observer, model)[..model.dimension()]);
    }
    Ok(output)
}

/// Evaluate mixed-model batches with a fixed-width record layout.
///
/// Each output row contains 29 values: `[dimension, measurement[4],
/// jacobian[24]]`.  Only the first `dimension` measurement entries and the
/// first `dimension * 6` Jacobian entries are populated.  `steps` may contain
/// one six-value vector shared by every row or six values per row.
pub fn tracking_measurement_and_jacobian_batch(
    target_states: &[f64],
    observer_states: &[f64],
    model_codes: &[i32],
    steps: &[f64],
) -> Result<Vec<f64>, String> {
    if target_states.len() != observer_states.len() || target_states.len() % STATE_DIMENSION != 0 {
        return Err(
            "target_states and observer_states must have equal lengths divisible by 6".to_owned(),
        );
    }
    let rows = target_states.len() / STATE_DIMENSION;
    if model_codes.len() != rows {
        return Err("model_codes must contain one code per tracking row".to_owned());
    }
    if steps.len() != STATE_DIMENSION && steps.len() != rows * STATE_DIMENSION {
        return Err("steps must contain 6 values or 6 values per tracking row".to_owned());
    }
    if target_states.iter().any(|value| !value.is_finite())
        || observer_states.iter().any(|value| !value.is_finite())
    {
        return Err("tracking batch states must contain only finite values".to_owned());
    }
    // Validate all step rows before allocating output so malformed input
    // cannot produce a partially valid result.
    if steps.len() == STATE_DIMENSION {
        validate_steps(steps)?;
    } else {
        for row in 0..rows {
            validate_steps(&steps[row * STATE_DIMENSION..(row + 1) * STATE_DIMENSION])?;
        }
    }
    let mut output = Vec::with_capacity(
        rows * (1 + MAX_MEASUREMENT_DIMENSION + MAX_MEASUREMENT_DIMENSION * STATE_DIMENSION),
    );
    for row in 0..rows {
        let target: [f64; STATE_DIMENSION] = target_states
            [row * STATE_DIMENSION..(row + 1) * STATE_DIMENSION]
            .try_into()
            .expect("validated six-state batch row");
        let observer: [f64; STATE_DIMENSION] = observer_states
            [row * STATE_DIMENSION..(row + 1) * STATE_DIMENSION]
            .try_into()
            .expect("validated six-state batch row");
        let step_row = if steps.len() == STATE_DIMENSION {
            steps
        } else {
            &steps[row * STATE_DIMENSION..(row + 1) * STATE_DIMENSION]
        };
        let model = MeasurementModel::from_code(model_codes[row])?;
        let (measurement, jacobian) =
            tracking_measurement_and_jacobian(target, observer, model_codes[row], step_row)?;
        let dimension = model.dimension();
        output.push(dimension as f64);
        output.extend_from_slice(&measurement);
        output.resize(output.len() + MAX_MEASUREMENT_DIMENSION - dimension, 0.0);
        output.extend_from_slice(&jacobian);
        output.resize(
            output.len() + MAX_MEASUREMENT_DIMENSION * STATE_DIMENSION
                - dimension * STATE_DIMENSION,
            0.0,
        );
    }
    Ok(output)
}

#[cfg(feature = "python")]
mod python {
    use super::STATE_DIMENSION;
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    fn fixed_state(values: Vec<f64>, name: &str) -> PyResult<[f64; STATE_DIMENSION]> {
        values
            .try_into()
            .map_err(|_| PyValueError::new_err(format!("{name} must contain exactly 6 values")))
    }

    #[pyfunction]
    fn tracking_measurement(
        target_state: Vec<f64>,
        observer_state: Vec<f64>,
        model_code: i32,
    ) -> PyResult<Vec<f64>> {
        super::tracking_measurement(
            fixed_state(target_state, "target_state")?,
            fixed_state(observer_state, "observer_state")?,
            model_code,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn tracking_measurement_and_jacobian(
        target_state: Vec<f64>,
        observer_state: Vec<f64>,
        model_code: i32,
        steps: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        super::tracking_measurement_and_jacobian(
            fixed_state(target_state, "target_state")?,
            fixed_state(observer_state, "observer_state")?,
            model_code,
            &steps,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn tracking_measurement_batch(
        target_states: Vec<f64>,
        observer_states: Vec<f64>,
        model_code: i32,
    ) -> PyResult<Vec<f64>> {
        super::tracking_measurement_batch(&target_states, &observer_states, model_code)
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn tracking_measurement_and_jacobian_batch(
        target_states: Vec<f64>,
        observer_states: Vec<f64>,
        model_codes: Vec<i32>,
        steps: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        super::tracking_measurement_and_jacobian_batch(
            &target_states,
            &observer_states,
            &model_codes,
            &steps,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn tracking_measurement_and_jacobian_bytes(
        py: Python<'_>,
        target_state: Vec<f64>,
        observer_state: Vec<f64>,
        model_code: i32,
        steps: Vec<f64>,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let (measurement, jacobian) =
            tracking_measurement_and_jacobian(target_state, observer_state, model_code, steps)?;
        Ok((
            output_bytes(py, &measurement)?,
            output_bytes(py, &jacobian)?,
        ))
    }
    #[pyfunction]
    fn tracking_measurement_batch_bytes(
        py: Python<'_>,
        target_states: &[u8],
        observer_states: &[u8],
        model_code: i32,
    ) -> PyResult<Py<PyBytes>> {
        let values = super::tracking_measurement_batch(
            &decode_f64_bytes(target_states)?,
            &decode_f64_bytes(observer_states)?,
            model_code,
        )
        .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }
    #[pyfunction]
    fn tracking_measurement_and_jacobian_batch_bytes(
        py: Python<'_>,
        target_states: &[u8],
        observer_states: &[u8],
        model_codes: Vec<i32>,
        steps: Vec<f64>,
    ) -> PyResult<Py<PyBytes>> {
        let values = super::tracking_measurement_and_jacobian_batch(
            &decode_f64_bytes(target_states)?,
            &decode_f64_bytes(observer_states)?,
            &model_codes,
            &steps,
        )
        .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(tracking_measurement, module)?)?;
        module.add_function(wrap_pyfunction!(tracking_measurement_and_jacobian, module)?)?;
        module.add_function(wrap_pyfunction!(tracking_measurement_batch, module)?)?;
        module.add_function(wrap_pyfunction!(
            tracking_measurement_and_jacobian_batch,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(
            tracking_measurement_and_jacobian_bytes,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(tracking_measurement_batch_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(
            tracking_measurement_and_jacobian_batch_bytes,
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
    use super::{
        tracking_measurement, tracking_measurement_and_jacobian,
        tracking_measurement_and_jacobian_batch,
    };

    fn target() -> [f64; 6] {
        [7000.0, 2.0, -3.0, 0.1, 7.4, -0.02]
    }

    fn observer() -> [f64; 6] {
        [6999.0, -1.0, -2.0, 0.0, 7.3, 0.01]
    }

    #[test]
    fn zero_range_has_finite_zero_los_outputs() {
        let state = target();
        let values = tracking_measurement(state, state, 4).unwrap();
        assert_eq!(values, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn forward_jacobian_matches_direct_perturbation() {
        let target = target();
        let observer = observer();
        let steps = [1.0e-3, 1.0e-3, 1.0e-3, 1.0e-6, 1.0e-6, 1.0e-6];
        let (base, jacobian) =
            tracking_measurement_and_jacobian(target, observer, 4, &steps).unwrap();
        assert_eq!(base.len(), 4);
        assert_eq!(jacobian.len(), 24);
        let mut perturbed = target;
        perturbed[0] += steps[0];
        let plus = tracking_measurement(perturbed, observer, 4).unwrap();
        let wrapped_delta = super::wrap_angle(plus[0] - base[0]);
        assert!((jacobian[0] - wrapped_delta / steps[0]).abs() < 1.0e-15);
    }

    #[test]
    fn mixed_batch_uses_fixed_parseable_records() {
        let t = target();
        let o = observer();
        let values = tracking_measurement_and_jacobian_batch(
            &[t.to_vec(), t.to_vec()].concat(),
            &[o.to_vec(), o.to_vec()].concat(),
            &[0, 4],
            &[1.0e-3, 1.0e-3, 1.0e-3, 1.0e-6, 1.0e-6, 1.0e-6],
        )
        .unwrap();
        assert_eq!(values.len(), 2 * 29);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[29], 4.0);
    }
}
