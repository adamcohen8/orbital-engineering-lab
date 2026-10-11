//! Prepared immutable history interpolation and ordered Hermite batches.
//! Python owns stationary roots, encounter selection, Pc and evidence policy.

pub fn hermite(left: &[f64; 6], right: &[f64; 6], dt: f64, u: f64) -> [f64; 6] {
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    let dh00 = 6.0 * u2 - 6.0 * u;
    let dh10 = 3.0 * u2 - 4.0 * u + 1.0;
    let dh01 = -6.0 * u2 + 6.0 * u;
    let dh11 = 3.0 * u2 - 2.0 * u;
    std::array::from_fn(|i| {
        if i < 3 {
            h00 * left[i] + (h10 * dt) * left[i + 3] + h01 * right[i] + (h11 * dt) * right[i + 3]
        } else {
            (dh00 * left[i - 3] + dh01 * right[i - 3]) / dt + dh10 * left[i] + dh11 * right[i]
        }
    })
}

pub struct History {
    times: Vec<f64>,
    states: Vec<[f64; 6]>,
    incoming: Vec<[f64; 3]>,
}

impl History {
    pub fn new(
        times: Vec<f64>,
        states: Vec<[f64; 6]>,
        incoming: Vec<[f64; 3]>,
    ) -> Result<Self, &'static str> {
        if times.len() < 2
            || times.len() > 1_000_000
            || states.len() != times.len()
            || incoming.len() != times.len()
            || times
                .iter()
                .chain(states.iter().flatten())
                .chain(incoming.iter().flatten())
                .any(|v| !v.is_finite())
            || times.windows(2).any(|r| r[1] <= r[0])
        {
            return Err("invalid bounded conjunction history");
        }
        Ok(Self {
            times,
            states,
            incoming,
        })
    }

    pub fn interpolate(&self, query: f64, left_side: bool) -> Result<[f64; 6], &'static str> {
        if !query.is_finite() || query < self.times[0] || query > *self.times.last().unwrap() {
            return Err("Interpolation time lies outside the supplied history.");
        }
        let upper = self.times.partition_point(|t| *t <= query);
        let exact = upper > 0 && self.times[upper - 1] == query;
        if exact && (!left_side || upper == 1) {
            return Ok(self.states[upper - 1]);
        }
        let index = if exact { upper - 2 } else { upper - 1 }.min(self.times.len() - 2);
        let left = self.states[index];
        let mut right = self.states[index + 1];
        right[3..].copy_from_slice(&self.incoming[index + 1]);
        let dt = self.times[index + 1] - self.times[index];
        Ok(hermite(&left, &right, dt, (query - self.times[index]) / dt))
    }
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
    #[pyclass(name = "ConjunctionHistoryContext")]
    struct Context {
        history: History,
    }
    #[pymethods]
    impl Context {
        #[new]
        fn new(times: &[u8], states: &[u8], incoming: &[u8]) -> PyResult<Self> {
            let times = decode_f64_bytes(times)?;
            let states = decode_f64_bytes(states)?;
            let incoming = decode_f64_bytes(incoming)?;
            if times.len() > 1_000_000
                || states.len() != times.len() * 6
                || incoming.len() != times.len() * 3
            {
                return Err(PyValueError::new_err("invalid bounded conjunction history"));
            }
            let history = History::new(
                times.into_owned(),
                states
                    .chunks_exact(6)
                    .map(|r| r.try_into().unwrap())
                    .collect(),
                incoming
                    .chunks_exact(3)
                    .map(|r| r.try_into().unwrap())
                    .collect(),
            )
            .map_err(PyValueError::new_err)?;
            Ok(Self { history })
        }
        fn interpolate(
            &self,
            py: Python<'_>,
            queries: &[u8],
            left_side: bool,
        ) -> PyResult<Py<PyBytes>> {
            let queries = decode_f64_bytes(queries)?;
            if queries.len() > 1_000_000 {
                return Err(PyValueError::new_err(
                    "conjunction query batch exceeds bound",
                ));
            }
            let out = py
                .detach(|| {
                    let mut out = Vec::with_capacity(queries.len() * 6);
                    for query in queries.iter() {
                        out.extend(self.history.interpolate(*query, left_side)?);
                    }
                    Ok::<_, &'static str>(out)
                })
                .map_err(PyValueError::new_err)?;
            output_bytes(py, &out)
        }
    }
    #[pyfunction]
    fn conjunction_hermite_batch(py: Python<'_>, cases: &[u8]) -> PyResult<Py<PyBytes>> {
        let values = decode_f64_bytes(cases)?;
        if values.len() % 14 != 0 || values.len() > 14_000_000 {
            return Err(PyValueError::new_err(
                "invalid bounded conjunction Hermite batch",
            ));
        }
        let out = py
            .detach(|| {
                let mut out = Vec::with_capacity(values.len() / 14 * 6);
                for row in values.chunks_exact(14) {
                    if row.iter().any(|v| !v.is_finite())
                        || row[12] <= 0.0
                        || !(0.0..=1.0).contains(&row[13])
                    {
                        return Err("invalid conjunction Hermite inputs");
                    }
                    out.extend(hermite(
                        row[..6].try_into().unwrap(),
                        row[6..12].try_into().unwrap(),
                        row[12],
                        row[13],
                    ));
                }
                Ok::<_, &'static str>(out)
            })
            .map_err(PyValueError::new_err)?;
        output_bytes(py, &out)
    }
    module.add_class::<Context>()?;
    module.add_function(wrap_pyfunction!(conjunction_hermite_batch, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incoming_velocity_is_used_on_the_left_of_an_impulse() {
        let history = History::new(
            vec![0.0, 1.0, 2.0],
            vec![
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            ],
            vec![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        )
        .unwrap();
        assert_eq!(history.interpolate(1.0, true).unwrap()[3], 1.0);
        assert_eq!(history.interpolate(1.0, false).unwrap()[3], 3.0);
        assert_eq!(history.interpolate(0.5, false).unwrap()[0], 0.5);
        assert!(history.interpolate(-1.0, false).is_err());
        assert!(history.interpolate(f64::NAN, false).is_err());
    }
}
