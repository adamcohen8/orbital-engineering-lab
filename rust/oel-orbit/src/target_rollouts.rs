//! Batched fixed-duration ONP trials for the public single-shooting Jacobian.

use crate::{rk4_step_validated, OrbitParameters, State};

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(a: [f64; 3]) -> Result<[f64; 3], String> {
    let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err("invalid RIC basis".into());
    }
    Ok([a[0] / norm, a[1] / norm, a[2] / norm])
}

pub fn rollouts(
    initial: [f64; 6],
    trials: &[Vec<[f64; 5]>],
    step_s: f64,
    mu: f64,
) -> Result<Vec<f64>, String> {
    zonal_rollouts(initial, trials, step_s, mu, &[])
}

pub fn zonal_rollouts(
    initial: [f64; 6],
    trials: &[Vec<[f64; 5]>],
    step_s: f64,
    mu: f64,
    codes: &[u8],
) -> Result<Vec<f64>, String> {
    crate::zonal::validate_codes(codes).map_err(str::to_owned)?;
    if !step_s.is_finite() || step_s <= 0.0 || !mu.is_finite() || mu <= 0.0 {
        return Err("invalid targeting propagation settings".into());
    }
    let parameters = OrbitParameters {
        mu_km3_s2: mu,
        include_j2: codes == [2],
        command_accel_eci_km_s2: [0.0; 3],
    };
    let mut output = Vec::with_capacity(trials.len() * 8);
    for (trial_index, actions) in trials.iter().enumerate() {
        let mut state = State::from_array(initial).map_err(str::to_owned)?;
        let mut elapsed = 0.0;
        let mut steps = 0_usize;
        for action in actions {
            if action.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "targeting trial {trial_index} has nonfinite action"
                ));
            }
            match action[0] {
                0.0 => {
                    let mut remaining = action[1];
                    if remaining <= 0.0 {
                        return Err(format!("targeting trial {trial_index} has invalid coast"));
                    }
                    while remaining > 0.0 {
                        let width = step_s.min(remaining);
                        state = if codes.is_empty() || codes == [2] {
                            rk4_step_validated(state, width, parameters)
                        } else {
                            crate::rk4_zonal_step(state, width, mu, [0.0; 3], codes)
                        }
                        .map_err(str::to_owned)?;
                        remaining -= width;
                        steps += 1;
                        if steps > 10_000_000 {
                            return Err("targeting trial step limit exceeded".into());
                        }
                    }
                    elapsed += action[1];
                }
                1.0 | 2.0 => {
                    let mut delta = [action[1] / 1000.0, action[2] / 1000.0, action[3] / 1000.0];
                    if action[0] == 2.0 {
                        let radial = unit(state.position_km)?;
                        let cross_track = unit(cross(state.position_km, state.velocity_km_s))?;
                        let in_track = cross(cross_track, radial);
                        delta = std::array::from_fn(|index| {
                            radial[index] * delta[0]
                                + in_track[index] * delta[1]
                                + cross_track[index] * delta[2]
                        });
                    }
                    let mut next = state.as_array();
                    for axis in 0..3 {
                        next[axis + 3] += delta[axis];
                    }
                    state = State::from_array(next).map_err(str::to_owned)?;
                }
                _ => return Err(format!("targeting trial {trial_index} has unknown action")),
            }
        }
        output.extend(state.as_array());
        output.push(elapsed);
        output.push(steps as f64);
    }
    Ok(output)
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

    #[pyfunction]
    fn targeting_rollouts_bytes(
        py: Python<'_>,
        state: &[u8],
        actions: Vec<Vec<Vec<f64>>>,
        step_s: f64,
        mu: f64,
    ) -> PyResult<Py<PyBytes>> {
        targeting_zonal_rollouts_bytes(py, state, actions, step_s, mu, vec![])
    }

    #[pyfunction]
    fn targeting_zonal_rollouts_bytes(
        py: Python<'_>,
        state: &[u8],
        actions: Vec<Vec<Vec<f64>>>,
        step_s: f64,
        mu: f64,
        codes: Vec<u8>,
    ) -> PyResult<Py<PyBytes>> {
        let state = decode_f64_bytes(state)?;
        let initial: [f64; 6] = state
            .as_ref()
            .try_into()
            .map_err(|_| PyValueError::new_err("state must have six values"))?;
        let trials: Vec<Vec<[f64; 5]>> = actions
            .iter()
            .map(|trial| {
                trial
                    .iter()
                    .map(|action| {
                        action
                            .as_slice()
                            .try_into()
                            .map_err(|_| PyValueError::new_err("action must have five values"))
                    })
                    .collect()
            })
            .collect::<PyResult<_>>()?;
        let result = py
            .detach(|| zonal_rollouts(initial, &trials, step_s, mu, &codes))
            .map_err(PyValueError::new_err)?;
        output_bytes(py, &result)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(targeting_rollouts_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(targeting_zonal_rollouts_bytes, module)?)
    }
}

#[cfg(feature = "python")]
pub use python::register;
