//! Passive RK4 intervals using the same step sequence as the Python owner.

use crate::{rk4_step, OrbitParameters, State};

pub fn sampled_history(
    initial: State,
    widths: &[f64],
    sample_ends: &[usize],
    parameters: OrbitParameters,
) -> Result<Vec<f64>, &'static str> {
    parameters.validate()?;
    if widths.iter().any(|h| !h.is_finite() || *h <= 0.0) {
        return Err("passive step widths must be finite and positive");
    }
    if sample_ends.first().is_some_and(|index| *index == 0)
        || sample_ends.windows(2).any(|pair| pair[0] >= pair[1])
        || sample_ends
            .last()
            .is_some_and(|index| *index > widths.len())
    {
        return Err("sample ends must be strictly increasing within the step sequence");
    }
    let capacity = sample_ends
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_mul(6))
        .ok_or("history size overflow")?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(capacity)
        .map_err(|_| "unable to allocate passive history")?;
    rows.extend(initial.as_array());
    let mut current = initial;
    let mut sample = 0;
    for (index, &h) in widths.iter().enumerate() {
        current = rk4_step(current, h, parameters)?;
        if sample < sample_ends.len() && sample_ends[sample] == index + 1 {
            rows.extend(current.as_array());
            sample += 1;
        }
    }
    Ok(rows)
}

/// Visit a caller-owned sequence of adaptive zonal intervals, carrying
/// the suggested next step exactly as the ordinary ONP propagator does.
pub fn sampled_adaptive_history(
    initial: State,
    times: &[f64],
    widths: &[f64],
    sample_ends: &[usize],
    mu: f64,
    force_codes: &[u8],
    atol: f64,
    rtol: f64,
    h_init: Option<f64>,
    method: u8,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if times.len() != widths.len()
        || sample_ends.first().is_some_and(|value| *value == 0)
        || sample_ends.windows(2).any(|pair| pair[0] >= pair[1])
        || sample_ends
            .last()
            .is_some_and(|value| *value != widths.len())
        || method > 1
    {
        return Err("invalid adaptive sampled step sequence".into());
    }
    let mut states = Vec::with_capacity(6 * (sample_ends.len() + 1));
    let mut infos = Vec::with_capacity(8 * widths.len());
    states.extend(initial.as_array());
    let mut state = initial;
    let mut next_h = h_init;
    let mut sample_index = 0;
    for (index, (&time, &width)) in times.iter().zip(widths).enumerate() {
        let (next, info) = if method == 0 {
            crate::rkf78_zonal_step(
                state,
                time,
                width,
                mu,
                [0.0; 3],
                force_codes,
                atol,
                rtol,
                next_h,
            )?
        } else {
            crate::zonal::validate_codes(force_codes).map_err(str::to_owned)?;
            let (values, info) = crate::adaptive::integrate_dopri5(
                state.as_array(),
                time,
                width,
                atol,
                rtol,
                next_h,
                |_, values| Ok(crate::zonal::derivative(values, mu, [0.0; 3], force_codes)),
            )?;
            (State::from_array(values).map_err(str::to_owned)?, info)
        };
        state = next;
        next_h = info.suggested_next_step_s;
        infos.extend([
            info.accepted_steps as f64,
            info.rejected_steps as f64,
            info.attempted_steps as f64,
            info.min_step_s.unwrap_or(f64::NAN),
            info.max_step_s.unwrap_or(f64::NAN),
            info.final_step_s.unwrap_or(f64::NAN),
            info.suggested_next_step_s.unwrap_or(f64::NAN),
            info.max_error_ratio,
        ]);
        if sample_index < sample_ends.len() && sample_ends[sample_index] == index + 1 {
            states.extend(state.as_array());
            sample_index += 1;
        }
    }
    Ok((states, infos))
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

    #[pyfunction]
    fn propagate_passive_sampled_eci_bytes<'py>(
        py: Python<'py>,
        state: Vec<f64>,
        step_widths: &[u8],
        sample_ends: Vec<usize>,
        mu_km3_s2: f64,
        include_j2: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let initial = State::from_array(
            state
                .try_into()
                .map_err(|_| PyValueError::new_err("state must have six values"))?,
        )
        .map_err(PyValueError::new_err)?;
        if step_widths.len() % 8 != 0 {
            return Err(PyValueError::new_err(
                "step width bytes must contain float64 values",
            ));
        }
        let widths: Vec<f64> = step_widths
            .chunks_exact(8)
            .map(|value| f64::from_le_bytes(value.try_into().unwrap()))
            .collect();
        let parameters = OrbitParameters {
            mu_km3_s2,
            include_j2,
            command_accel_eci_km_s2: [0.0; 3],
        };
        // All inputs are Rust-owned before detaching. Only the returned Python
        // bytes allocation needs the interpreter; propagation and packing do not.
        let bytes = py
            .detach(|| {
                let rows = sampled_history(initial, &widths, &sample_ends, parameters)?;
                Ok::<Vec<u8>, &'static str>(
                    rows.iter().flat_map(|value| value.to_le_bytes()).collect(),
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn propagate_adaptive_sampled_eci_bytes(
        py: Python<'_>,
        state: Vec<f64>,
        step_times: &[u8],
        step_widths: &[u8],
        sample_ends: Vec<usize>,
        mu: f64,
        force_codes: Vec<u8>,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
        method: u8,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let initial = State::from_array(
            state
                .try_into()
                .map_err(|_| PyValueError::new_err("state must have six values"))?,
        )
        .map_err(PyValueError::new_err)?;
        let times = decode_f64_bytes(step_times)?;
        let widths = decode_f64_bytes(step_widths)?;
        let (states, infos) = py
            .detach(|| {
                sampled_adaptive_history(
                    initial,
                    &times,
                    &widths,
                    &sample_ends,
                    mu,
                    &force_codes,
                    atol,
                    rtol,
                    h_init,
                    method,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((output_bytes(py, &states)?, output_bytes(py, &infos)?))
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("PASSIVE_HISTORY_GIL_RELEASED", true)?;
        module.add_function(wrap_pyfunction!(
            propagate_adaptive_sampled_eci_bytes,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(
            propagate_passive_sampled_eci_bytes,
            module
        )?)
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonuniform_intervals_use_identical_rk4_steps() {
        let initial = State::from_array([7000.0, 0.0, 0.0, 0.0, 7.5, 1.0]).unwrap();
        let parameters = OrbitParameters {
            mu_km3_s2: 398600.4418,
            include_j2: true,
            command_accel_eci_km_s2: [0.0; 3],
        };
        let widths = [0.3, 0.3, 0.1, 0.3, 0.05];
        let rows = sampled_history(initial, &widths, &[3, 5], parameters).unwrap();
        let mut state = initial;
        for (index, width) in widths.into_iter().enumerate() {
            state = rk4_step(state, width, parameters).unwrap();
            if index == 2 {
                assert_eq!(&rows[6..12], &state.as_array());
            }
        }
        assert_eq!(&rows[12..18], &state.as_array());
        assert!(sampled_history(initial, &[-1.0], &[1], parameters).is_err());
        assert!(sampled_history(initial, &widths, &[2, 2], parameters).is_err());
    }
}
