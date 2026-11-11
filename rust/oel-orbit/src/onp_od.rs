//! Persistent ONP observation evaluation for opt-in state OD fits.
//!
//! The context owns observation epochs, reference rows, and already prepared
//! lower Cholesky factors.  Each candidate state is propagated to the exact
//! requested epochs and whitened in one native call.  Resource loading,
//! custom Python force callbacks, priors, robust loss, clipping, and report
//! policy remain Python-owned.  The initial fused envelope is deliberately
//! small: RK4 two-body or Earth-J2 Cartesian state fits.

use crate::{rk4_step_validated, OrbitParameters, State};

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub states: Vec<f64>,
    pub residuals: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct ObservationContext {
    times_s: Vec<f64>,
    reference_positions_km: Vec<f64>,
    reference_velocities_km_s: Vec<f64>,
    factors: Vec<f64>,
    dimensions: Vec<usize>,
    mu_km3_s2: f64,
    include_j2: bool,
    step_widths_s: Vec<f64>,
    sample_end_indices: Vec<usize>,
}

impl ObservationContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        times_s: Vec<f64>,
        reference_positions_km: Vec<f64>,
        reference_velocities_km_s: Vec<f64>,
        factors: Vec<f64>,
        dimensions: Vec<usize>,
        mu_km3_s2: f64,
        include_j2: bool,
        step_widths_s: Vec<f64>,
        sample_end_indices: Vec<usize>,
    ) -> Result<Self, String> {
        if times_s.is_empty() {
            return Err("ONP observation context requires at least one epoch".to_owned());
        }
        if !times_s
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        {
            return Err("ONP observation epochs must be finite and non-negative".to_owned());
        }
        if times_s.windows(2).any(|window| window[1] < window[0]) {
            return Err("ONP observation epochs must be non-decreasing".to_owned());
        }
        let rows = times_s.len();
        if reference_positions_km.len() != rows * 3
            || !reference_positions_km.iter().all(|value| value.is_finite())
        {
            return Err("reference positions must contain finite rows of three values".to_owned());
        }
        if !reference_velocities_km_s.is_empty()
            && (reference_velocities_km_s.len() != rows * 3
                || !reference_velocities_km_s
                    .iter()
                    .all(|value| value.is_finite()))
        {
            return Err(
                "reference velocities must be empty or contain finite rows of three values"
                    .to_owned(),
            );
        }
        if dimensions.len() != rows
            || dimensions
                .iter()
                .any(|dimension| !matches!(dimension, 3 | 6))
        {
            return Err(
                "ONP observation dimensions must contain one 3 or 6 entry per epoch".to_owned(),
            );
        }
        if !reference_velocities_km_s.is_empty()
            && dimensions.iter().any(|dimension| *dimension != 6)
        {
            return Err("reference velocities require six-component observation rows".to_owned());
        }
        if reference_velocities_km_s.is_empty()
            && dimensions.iter().any(|dimension| *dimension == 6)
        {
            return Err("six-component observation rows require reference velocities".to_owned());
        }
        let expected_factors = dimensions.iter().try_fold(0usize, |sum, dimension| {
            let block_size = dimension
                .checked_mul(*dimension)
                .ok_or_else(|| "observation factor dimensions overflow".to_owned())?;
            sum.checked_add(block_size)
                .ok_or_else(|| "observation factor dimensions overflow".to_owned())
        })?;
        if factors.len() != expected_factors || !factors.iter().all(|value| value.is_finite()) {
            return Err("observation factors must contain finite row-major matrices".to_owned());
        }
        let mut factor_offset = 0usize;
        for dimension in &dimensions {
            for row in 0..*dimension {
                let diagonal = factors[factor_offset + row * *dimension + row];
                if diagonal <= 0.0 {
                    return Err("observation factors must have positive diagonals".to_owned());
                }
            }
            factor_offset += *dimension * *dimension;
        }
        if !mu_km3_s2.is_finite() || mu_km3_s2 <= 0.0 {
            return Err("mu_km3_s2 must be finite and positive".to_owned());
        }
        if sample_end_indices.len() != rows {
            return Err("sample_end_indices must contain one entry per epoch".to_owned());
        }
        if !step_widths_s
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err("step widths must be finite and positive".to_owned());
        }
        if sample_end_indices
            .windows(2)
            .any(|window| window[1] < window[0])
            || sample_end_indices
                .iter()
                .any(|end| *end > step_widths_s.len())
        {
            return Err(
                "sample_end_indices must be non-decreasing and within the step schedule".to_owned(),
            );
        }
        if sample_end_indices.last().copied().unwrap_or_default() != step_widths_s.len() {
            return Err("step schedule must be consumed by the final observation epoch".to_owned());
        }
        let mut schedule_time = 0.0;
        let mut previous_end = 0usize;
        for (time_s, end) in times_s.iter().zip(sample_end_indices.iter()) {
            for width in &step_widths_s[previous_end..*end] {
                schedule_time += *width;
            }
            if (schedule_time - *time_s).abs() > 1.0e-9_f64.max(1.0e-12 * time_s.abs().max(1.0)) {
                return Err("step schedule does not terminate at each observation epoch".to_owned());
            }
            previous_end = *end;
        }
        Ok(Self {
            times_s,
            reference_positions_km,
            reference_velocities_km_s,
            factors,
            dimensions,
            mu_km3_s2,
            include_j2,
            step_widths_s,
            sample_end_indices,
        })
    }

    pub fn rows(&self) -> usize {
        self.times_s.len()
    }

    pub fn evaluate(&self, initial_state: [f64; 6]) -> Result<Evaluation, String> {
        let initial = State::from_array(initial_state).map_err(str::to_owned)?;
        let parameters = OrbitParameters {
            mu_km3_s2: self.mu_km3_s2,
            include_j2: self.include_j2,
            command_accel_eci_km_s2: [0.0; 3],
        };
        let mut state = initial;
        let mut states = Vec::with_capacity(self.rows() * 6);
        let residual_count = self.dimensions.iter().sum();
        let mut residuals = Vec::with_capacity(residual_count);
        let mut factor_offset = 0usize;
        let mut step_index = 0usize;
        for (index, _time_s) in self.times_s.iter().enumerate() {
            let sample_end = self.sample_end_indices[index];
            while step_index < sample_end {
                state = rk4_step_validated(state, self.step_widths_s[step_index], parameters)
                    .map_err(str::to_owned)?;
                step_index += 1;
            }
            let values = state.as_array();
            states.extend(values);
            let dimension = self.dimensions[index];
            let mut residual = [0.0; 6];
            residual[0] = values[0] - self.reference_positions_km[index * 3];
            residual[1] = values[1] - self.reference_positions_km[index * 3 + 1];
            residual[2] = values[2] - self.reference_positions_km[index * 3 + 2];
            if dimension == 6 {
                residual[3] = values[3] - self.reference_velocities_km_s[index * 3];
                residual[4] = values[4] - self.reference_velocities_km_s[index * 3 + 1];
                residual[5] = values[5] - self.reference_velocities_km_s[index * 3 + 2];
            }
            let factor = &self.factors[factor_offset..factor_offset + dimension * dimension];
            let mut whitened = [0.0; 6];
            for row in 0..dimension {
                let diagonal = factor[row * dimension + row];
                let mut value = residual[row];
                for column in 0..row {
                    value -= factor[row * dimension + column] * whitened[column];
                }
                whitened[row] = value / diagonal;
            }
            residuals.extend_from_slice(&whitened[..dimension]);
            factor_offset += dimension * dimension;
        }
        if !states.iter().all(|value| value.is_finite())
            || !residuals.iter().all(|value| value.is_finite())
        {
            return Err("ONP observation evaluation produced non-finite values".to_owned());
        }
        Ok(Evaluation { states, residuals })
    }
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::prelude::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    fn fixed_state(values: Vec<f64>) -> PyResult<[f64; 6]> {
        values
            .try_into()
            .map_err(|_| PyValueError::new_err("initial state must contain six values"))
    }

    #[pyclass(name = "ONPObservationContext")]
    struct OnpObservationContext {
        context: ObservationContext,
    }

    #[pymethods]
    impl OnpObservationContext {
        #[new]
        #[allow(clippy::too_many_arguments)]
        fn new(
            times_s: Vec<f64>,
            reference_positions_km: Vec<f64>,
            reference_velocities_km_s: Vec<f64>,
            factors: Vec<f64>,
            dimensions: Vec<usize>,
            mu_km3_s2: f64,
            include_j2: bool,
            step_widths_s: Vec<f64>,
            sample_end_indices: Vec<usize>,
        ) -> PyResult<Self> {
            let context = ObservationContext::new(
                times_s,
                reference_positions_km,
                reference_velocities_km_s,
                factors,
                dimensions,
                mu_km3_s2,
                include_j2,
                step_widths_s,
                sample_end_indices,
            )
            .map_err(PyValueError::new_err)?;
            Ok(Self { context })
        }

        #[getter]
        fn rows(&self) -> usize {
            self.context.rows()
        }

        fn evaluate(
            &self,
            py: Python<'_>,
            initial_state: Vec<f64>,
        ) -> PyResult<(Vec<f64>, Vec<f64>)> {
            let initial_state = fixed_state(initial_state)?;
            let result = py
                .detach(|| self.context.evaluate(initial_state))
                .map_err(PyValueError::new_err)?;
            Ok((result.states, result.residuals))
        }

        fn evaluate_bytes(
            &self,
            py: Python<'_>,
            initial_state: &[u8],
        ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
            let initial_state = fixed_state(decode_f64_bytes(initial_state)?.into_owned())?;
            let result = py
                .detach(|| self.context.evaluate(initial_state))
                .map_err(PyValueError::new_err)?;
            Ok((
                output_bytes(py, &result.states)?,
                output_bytes(py, &result.residuals)?,
            ))
        }
    }

    module.add_class::<OnpObservationContext>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ObservationContext;

    fn context() -> ObservationContext {
        ObservationContext::new(
            vec![0.0, 1.0, 1.0, 2.0],
            vec![
                7000.0, 0.0, 0.0, 7000.0, 7.5, 0.0, 7000.0, 7.5, 0.0, 7000.0, 7.5, 0.0,
            ],
            vec![],
            vec![1.0; 4 * 9],
            vec![3, 3, 3, 3],
            398600.4418,
            false,
            vec![1.0, 1.0],
            vec![0, 1, 1, 2],
        )
        .unwrap()
    }

    #[test]
    fn exact_epoch_order_and_duplicate_epochs_are_preserved() {
        let context = context();
        let result = context.evaluate([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]).unwrap();
        assert_eq!(result.states.len(), 24);
        assert_eq!(&result.states[6..12], &result.states[12..18]);
        assert_eq!(result.residuals.len(), 12);
    }

    #[test]
    fn correlated_position_factor_is_applied_forward() {
        let context = ObservationContext::new(
            vec![0.0],
            vec![7002.0, 3.0, 0.0],
            vec![],
            vec![2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![3],
            398600.4418,
            false,
            vec![],
            vec![0],
        )
        .unwrap();
        let result = context.evaluate([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]).unwrap();
        assert!((result.residuals[0] + 1.0).abs() < 1.0e-12);
        assert!((result.residuals[1] + 2.0).abs() < 1.0e-12);
    }
}
