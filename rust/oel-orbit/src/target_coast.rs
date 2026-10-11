//! Bounded two-body coasts used by the optional trajectory targeter.

use crate::{rk4_step_validated, OrbitParameters, State};

fn coast(mut state: State, step_s: f64, steps: usize, mu: f64) -> Result<State, &'static str> {
    if !step_s.is_finite() || !mu.is_finite() || mu <= 0.0 {
        return Err("two-body coast step and mu must be finite and valid");
    }
    let parameters = OrbitParameters {
        mu_km3_s2: mu,
        include_j2: false,
        command_accel_eci_km_s2: [0.0; 3],
    };
    for _ in 0..steps {
        state = rk4_step_validated(state, step_s, parameters)?;
    }
    Ok(state)
}

fn radial_dot(state: State) -> f64 {
    let p = state.position_km;
    let v = state.velocity_km_s;
    p[0] * v[0] + p[1] * v[1] + p[2] * v[2]
}

pub fn first_apsis(
    state: [f64; 6],
    kind: u8,
    step_s: f64,
    max_coast_s: f64,
    mu: f64,
) -> Result<(f64, [f64; 6]), &'static str> {
    if kind > 1
        || !step_s.is_finite()
        || step_s <= 0.0
        || !max_coast_s.is_finite()
        || max_coast_s <= 0.0
    {
        return Err("invalid first-apsis search parameters");
    }
    let sign = if kind == 0 { 1.0 } else { -1.0 };
    let mut left = State::from_array(state)?;
    let mut time_left = 0.0;
    let mut radial_left = sign * radial_dot(left);
    while time_left < max_coast_s {
        let dt = step_s.min(max_coast_s - time_left);
        let right = coast(left, dt, 1, mu)?;
        let radial_right = sign * radial_dot(right);
        if radial_left > 0.0 && radial_right <= 0.0 {
            let (mut lo, mut hi) = (0.0, dt);
            for _ in 0..36 {
                let mid = 0.5 * (lo + hi);
                let midpoint = coast(left, mid, 1, mu)?;
                if sign * radial_dot(midpoint) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let event_dt = 0.5 * (lo + hi);
            return Ok((
                time_left + event_dt,
                coast(left, event_dt, 1, mu)?.as_array(),
            ));
        }
        left = right;
        time_left += dt;
        radial_left = radial_right;
    }
    Err("No first apsis event occurred within max_coast_s")
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    fn targeting_first_apsis_bytes(
        py: Python<'_>,
        state: &[u8],
        kind: u8,
        step_s: f64,
        max_coast_s: f64,
        mu: f64,
    ) -> PyResult<(f64, Py<PyBytes>)> {
        let state = decode_f64_bytes(state)?;
        let state: [f64; 6] = state
            .as_ref()
            .try_into()
            .map_err(|_| PyValueError::new_err("state must have six elements"))?;
        let (time, event) = py
            .detach(|| first_apsis(state, kind, step_s, max_coast_s, mu))
            .map_err(PyValueError::new_err)?;
        Ok((time, output_bytes(py, &event)?))
    }

    #[pyfunction]
    fn targeting_two_body_coast_bytes(
        py: Python<'_>,
        state: &[u8],
        step_s: f64,
        steps: usize,
        mu: f64,
    ) -> PyResult<Py<PyBytes>> {
        let values = decode_f64_bytes(state)?;
        let state: [f64; 6] = values
            .as_ref()
            .try_into()
            .map_err(|_| PyValueError::new_err("state must have six elements"))?;
        let result = py
            .detach(|| coast(State::from_array(state)?, step_s, steps, mu))
            .map_err(PyValueError::new_err)?;
        output_bytes(py, &result.as_array())
    }

    module.add_function(wrap_pyfunction!(targeting_first_apsis_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(targeting_two_body_coast_bytes, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{coast, first_apsis};
    use crate::State;

    #[test]
    fn coast_and_apogee_are_finite() {
        let state = [7000.0, 0.0, 0.0, 0.0, 8.0, 0.0];
        let next = coast(State::from_array(state).unwrap(), 30.0, 4, 398600.4418).unwrap();
        assert!(next.position_km.iter().all(|value| value.is_finite()));
        let (time, event) = first_apsis(state, 0, 30.0, 20_000.0, 398600.4418).unwrap();
        assert!(time > 0.0);
        assert!(event.iter().all(|value| value.is_finite()));
    }
}
