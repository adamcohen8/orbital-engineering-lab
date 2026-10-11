//! Independent finite-difference orbit trials for covariance STM construction.

use crate::{rk4_step_validated, OrbitParameters, State};

/// Independent caller-owned RK4 trajectories. Order and substep widths are
/// retained; no filter, event, or adaptive-step policy is evaluated here.
pub fn propagate_trials(
    states: &[f64],
    widths: &[f64],
    mu: f64,
    codes: &[u8],
    command: [f64; 3],
) -> Result<Vec<f64>, &'static str> {
    if states.len() % 6 != 0
        || states.len() / 6 > 1_000_000
        || widths.len() > 10_000_000
        || (states.len() / 6).saturating_mul(widths.len()) > 10_000_000
        || widths.iter().any(|h| !h.is_finite() || *h < 0.0)
        || !mu.is_finite()
        || mu <= 0.0
        || command.iter().any(|v| !v.is_finite())
    {
        return Err("invalid bounded orbit trial batch");
    }
    crate::zonal::validate_codes(codes)?;
    let mut output = Vec::with_capacity(states.len());
    for row in states.chunks_exact(6) {
        let mut state = State::from_array(row.try_into().unwrap())?;
        for &width in widths {
            state = if codes.is_empty() || codes == [2] {
                rk4_step_validated(
                    state,
                    width,
                    OrbitParameters {
                        mu_km3_s2: mu,
                        include_j2: !codes.is_empty(),
                        command_accel_eci_km_s2: command,
                    },
                )?
            } else {
                crate::rk4_zonal_step(state, width, mu, command, codes)?
            };
        }
        output.extend(state.as_array());
    }
    Ok(output)
}

pub fn finite_difference_trials(
    nominal: [f64; 6],
    differences: [f64; 6],
    substeps_s: &[f64],
    mu: f64,
    include_j2: bool,
    command: [f64; 3],
) -> Result<Vec<f64>, &'static str> {
    if !mu.is_finite()
        || mu <= 0.0
        || !differences
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        || !substeps_s
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        || !command.iter().all(|value| value.is_finite())
    {
        return Err("finite-difference orbit trial parameters must be finite and positive");
    }
    let parameters = OrbitParameters {
        mu_km3_s2: mu,
        include_j2,
        command_accel_eci_km_s2: command,
    };
    let mut samples = vec![0.0; 72];
    for axis in 0..6 {
        for sign_index in 0..2 {
            let mut perturbed = nominal;
            perturbed[axis] = if sign_index == 0 {
                nominal[axis] + differences[axis]
            } else {
                nominal[axis] - differences[axis]
            };
            let mut state = State::from_array(perturbed)?;
            for &step in substeps_s {
                state = rk4_step_validated(state, step, parameters)?;
            }
            let offset = axis * 12 + sign_index * 6;
            samples[offset..offset + 6].copy_from_slice(&state.as_array());
        }
    }
    Ok(samples)
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    fn orbit_trial_batch_bytes(
        py: Python<'_>,
        states: &[u8],
        widths: &[u8],
        mu: f64,
        codes: Vec<u8>,
        command: [f64; 3],
    ) -> PyResult<Py<PyBytes>> {
        let states = decode_f64_bytes(states)?;
        let widths = decode_f64_bytes(widths)?;
        let output = py
            .detach(|| propagate_trials(&states, &widths, mu, &codes, command))
            .map_err(PyValueError::new_err)?;
        output_bytes(py, &output)
    }

    #[pyfunction]
    fn orbit_zonal_history_bytes(
        py: Python<'_>,
        state: &[u8],
        widths: &[u8],
        mu: f64,
        codes: Vec<u8>,
    ) -> PyResult<Py<PyBytes>> {
        let state = decode_f64_bytes(state)?;
        let widths = decode_f64_bytes(widths)?;
        if state.len() != 6 || widths.len() > 10_000_000 {
            return Err(PyValueError::new_err("invalid bounded zonal history"));
        }
        let output = py
            .detach(|| {
                let mut current = propagate_trials(&state, &[], mu, &codes, [0.0; 3])?;
                let mut output = Vec::with_capacity(6 * (widths.len() + 1));
                output.extend(&current);
                for &width in widths.iter() {
                    current = propagate_trials(&current, &[width], mu, &codes, [0.0; 3])?;
                    output.extend(&current);
                }
                Ok::<_, &'static str>(output)
            })
            .map_err(PyValueError::new_err)?;
        output_bytes(py, &output)
    }

    #[pyfunction]
    fn covariance_orbit_trials_bytes(
        py: Python<'_>,
        nominal: &[u8],
        differences: &[u8],
        substeps: &[u8],
        mu: f64,
        include_j2: bool,
        command: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let nominal = decode_f64_bytes(nominal)?;
        let differences = decode_f64_bytes(differences)?;
        let substeps = decode_f64_bytes(substeps)?;
        let command = decode_f64_bytes(command)?;
        let state: [f64; 6] = nominal
            .as_ref()
            .try_into()
            .map_err(|_| PyValueError::new_err("nominal state must have six elements"))?;
        let steps: [f64; 6] = differences
            .as_ref()
            .try_into()
            .map_err(|_| PyValueError::new_err("difference vector must have six elements"))?;
        let acceleration: [f64; 3] = command
            .as_ref()
            .try_into()
            .map_err(|_| PyValueError::new_err("command must have three elements"))?;
        let output = py
            .detach(|| {
                finite_difference_trials(
                    state,
                    steps,
                    substeps.as_ref(),
                    mu,
                    include_j2,
                    acceleration,
                )
            })
            .map_err(PyValueError::new_err)?;
        output_bytes(py, &output)
    }

    module.add_function(wrap_pyfunction!(covariance_orbit_trials_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(orbit_trial_batch_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(orbit_zonal_history_bytes, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::finite_difference_trials;

    #[test]
    fn returns_six_plus_minus_pairs() {
        let output = finite_difference_trials(
            [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            [0.001, 0.001, 0.001, 1.0e-6, 1.0e-6, 1.0e-6],
            &[1.0],
            398600.4418,
            true,
            [0.0; 3],
        )
        .unwrap();
        assert_eq!(output.len(), 72);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
