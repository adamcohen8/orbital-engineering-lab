//! Batched two-body observation-epoch propagation for angles-only IOD.

use crate::{rk4_step_validated, OrbitParameters, State};

pub fn states_at_offsets(
    initial: [f64; 6],
    offsets_s: &[f64],
    mu_km3_s2: f64,
    max_step_s: f64,
) -> Result<Vec<f64>, &'static str> {
    if !max_step_s.is_finite() || max_step_s <= 0.0 {
        return Err("max_step_s must be finite and positive");
    }
    let initial = State::from_array(initial)?;
    let parameters = OrbitParameters {
        mu_km3_s2,
        include_j2: false,
        command_accel_eci_km_s2: [0.0; 3],
    };
    if !mu_km3_s2.is_finite() || mu_km3_s2 <= 0.0 {
        return Err("mu_km3_s2 must be finite and positive");
    }
    let mut result = Vec::with_capacity(offsets_s.len() * 6);
    for &duration in offsets_s {
        if !duration.is_finite() {
            return Err("observation offsets must be finite");
        }
        if duration == 0.0 {
            result.extend_from_slice(&initial.as_array());
            continue;
        }
        let steps = ((duration.abs() / max_step_s).ceil() as usize).max(1);
        let dt = duration / steps as f64;
        let mut state = initial;
        for _ in 0..steps {
            state = rk4_step_validated(state, dt, parameters)?;
            let position = state.position_km;
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            if !radius.is_finite() || radius <= crate::EARTH_RADIUS_KM * 0.5 {
                return Err("IOD hypothesis propagation failed or entered a nonphysical radius.");
            }
        }
        result.extend_from_slice(&state.as_array());
    }
    Ok(result)
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    fn iod_two_body_states_bytes(
        py: Python<'_>,
        initial: &[u8],
        offsets: &[u8],
        mu: f64,
        max_step: f64,
    ) -> PyResult<Py<PyBytes>> {
        let state = decode_f64_bytes(initial)?;
        if state.len() != 6 {
            return Err(PyValueError::new_err(
                "initial state must have six elements",
            ));
        }
        let offsets = decode_f64_bytes(offsets)?;
        let values: [f64; 6] = state.as_ref().try_into().unwrap();
        let output = py
            .detach(|| states_at_offsets(values, offsets.as_ref(), mu, max_step))
            .map_err(PyValueError::new_err)?;
        output_bytes(py, &output)
    }

    module.add_function(wrap_pyfunction!(iod_two_body_states_bytes, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::states_at_offsets;

    #[test]
    fn preserves_order_and_zero_epoch() {
        let state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let rows = states_at_offsets(state, &[60.0, 0.0, -30.0], 398600.4418, 30.0).unwrap();
        assert_eq!(&rows[6..12], &state);
        assert_eq!(rows.len(), 18);
    }
}
