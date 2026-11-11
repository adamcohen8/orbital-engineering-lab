//! Native numerical kernels used by the private Scale workflows.
//!
//! The Scale planner, store, provenance, and candidate policy remain Python
//! owned.  This module only handles flat floating point arrays that are
//! already selected by those workflows.  It deliberately does not know about
//! product IDs, database rows, or campaign identity.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestApproach {
    pub sample_index: usize,
    pub distance_km: f64,
    pub jd_utc: f64,
    pub relative_speed_km_s: f64,
}

/// Find the first minimum sampled relative distance for two synchronized
/// position/velocity histories.
///
/// Each history is packed as `[px, py, pz, vx, vy, vz]` per sample.  Rows with
/// non-finite values are ignored so callers can preserve the Scale convention
/// that propagation error rows do not participate in screening.  Ties retain
/// the first row, matching the scalar Scale pair recomputation path.
pub fn closest_sampled_approach(
    left_states: &[f64],
    right_states: &[f64],
    sample_jd_utc: &[f64],
) -> Result<Option<ClosestApproach>, String> {
    if left_states.len() != right_states.len()
        || left_states.len() != sample_jd_utc.len().saturating_mul(6)
        || left_states.len() % 6 != 0
    {
        return Err("Scale pair histories have inconsistent shapes".to_owned());
    }
    let mut best: Option<ClosestApproach> = None;
    for sample_index in 0..sample_jd_utc.len() {
        let left = &left_states[sample_index * 6..sample_index * 6 + 6];
        let right = &right_states[sample_index * 6..sample_index * 6 + 6];
        if !sample_jd_utc[sample_index].is_finite()
            || !left.iter().chain(right).all(|value| value.is_finite())
        {
            continue;
        }
        let dx = right[0] - left[0];
        let dy = right[1] - left[1];
        let dz = right[2] - left[2];
        let distance_km = (dx * dx + dy * dy + dz * dz).sqrt();
        if !distance_km.is_finite()
            || best.is_some_and(|current| distance_km >= current.distance_km)
        {
            continue;
        }
        // The speed is needed only for an improving candidate, but remains a
        // validity condition before that candidate can replace the best row.
        let dvx = right[3] - left[3];
        let dvy = right[4] - left[4];
        let dvz = right[5] - left[5];
        let relative_speed_km_s = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();
        if relative_speed_km_s.is_finite() {
            best = Some(ClosestApproach {
                sample_index,
                distance_km,
                jd_utc: sample_jd_utc[sample_index],
                relative_speed_km_s,
            });
        }
    }
    Ok(best)
}

/// Compute synchronized relative distance and speed rows in one native loop.
///
/// The returned vector is packed `[distance_km, relative_speed_km_s]` per
/// sample.  Invalid rows are represented by two NaNs and can be filtered by
/// the Python owner without changing row order.
pub fn relative_metrics(left_states: &[f64], right_states: &[f64]) -> Result<Vec<f64>, String> {
    if left_states.len() != right_states.len() || left_states.len() % 6 != 0 {
        return Err("Scale pair histories have inconsistent shapes".to_owned());
    }
    let samples = left_states.len() / 6;
    let mut output = Vec::with_capacity(samples * 2);
    for sample_index in 0..samples {
        let left = &left_states[sample_index * 6..sample_index * 6 + 6];
        let right = &right_states[sample_index * 6..sample_index * 6 + 6];
        if !left.iter().chain(right).all(|value| value.is_finite()) {
            output.extend_from_slice(&[f64::NAN, f64::NAN]);
            continue;
        }
        let dx = right[0] - left[0];
        let dy = right[1] - left[1];
        let dz = right[2] - left[2];
        let dvx = right[3] - left[3];
        let dvy = right[4] - left[4];
        let dvz = right[5] - left[5];
        output.push((dx * dx + dy * dy + dz * dz).sqrt());
        output.push((dvx * dvx + dvy * dvy + dvz * dvz).sqrt());
    }
    Ok(output)
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;
    use std::borrow::Cow;

    fn decode_f64_bytes(values: &[u8]) -> PyResult<Cow<'_, [f64]>> {
        if values.len() % 8 != 0 {
            return Err(PyValueError::new_err(
                "numeric bytes must contain complete float64 values",
            ));
        }
        // Immutable Python bytes remain alive for this call. Every f64 bit
        // pattern is valid; borrow only aligned, complete native-endian rows.
        #[cfg(target_endian = "little")]
        {
            let (prefix, floats, suffix) = unsafe { values.align_to::<f64>() };
            if prefix.is_empty() && suffix.is_empty() {
                return Ok(Cow::Borrowed(floats));
            }
        }
        Ok(Cow::Owned(
            values
                .chunks_exact(8)
                .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
                .collect(),
        ))
    }

    fn encode_f64_bytes(values: &[f64]) -> Cow<'_, [u8]> {
        #[cfg(target_endian = "little")]
        {
            // f64 storage is contiguous and contains exactly size_of_val bytes.
            // PyBytes copies this borrowed view before the vector is dropped.
            return Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values))
            });
        }
        #[cfg(target_endian = "big")]
        Cow::Owned(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
    }

    #[pyfunction]
    pub fn scale_closest_sampled_approach(
        left_states: Vec<f64>,
        right_states: Vec<f64>,
        sample_jd_utc: Vec<f64>,
    ) -> PyResult<Option<(usize, f64, f64, f64)>> {
        closest_sampled_approach(&left_states, &right_states, &sample_jd_utc)
            .map(|result| {
                result.map(|value| {
                    (
                        value.sample_index,
                        value.distance_km,
                        value.jd_utc,
                        value.relative_speed_km_s,
                    )
                })
            })
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    pub fn scale_relative_metrics(
        left_states: Vec<f64>,
        right_states: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        relative_metrics(&left_states, &right_states).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    pub fn scale_closest_sampled_approach_bytes(
        py: Python<'_>,
        left_states: &[u8],
        right_states: &[u8],
        sample_jd_utc: &[u8],
    ) -> PyResult<Option<(usize, f64, f64, f64)>> {
        let left_states = decode_f64_bytes(left_states)?;
        let right_states = decode_f64_bytes(right_states)?;
        let sample_jd_utc = decode_f64_bytes(sample_jd_utc)?;
        let result = py
            .detach(|| closest_sampled_approach(&left_states, &right_states, &sample_jd_utc))
            .map_err(PyValueError::new_err)?;
        Ok(result.map(|value| {
            (
                value.sample_index,
                value.distance_km,
                value.jd_utc,
                value.relative_speed_km_s,
            )
        }))
    }

    #[pyfunction]
    pub fn scale_relative_metrics_bytes(
        py: Python<'_>,
        left_states: &[u8],
        right_states: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let left_states = decode_f64_bytes(left_states)?;
        let right_states = decode_f64_bytes(right_states)?;
        let values = py
            .detach(|| relative_metrics(&left_states, &right_states))
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &encode_f64_bytes(&values)).unbind())
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(scale_closest_sampled_approach, module)?)?;
        module.add_function(wrap_pyfunction!(scale_relative_metrics, module)?)?;
        module.add_function(wrap_pyfunction!(
            scale_closest_sampled_approach_bytes,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(scale_relative_metrics_bytes, module)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::{closest_sampled_approach, relative_metrics};

    #[test]
    fn closest_approach_skips_invalid_rows_and_keeps_first_tie() {
        let left = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            f64::NAN,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let right = [
            2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ];
        let result = closest_sampled_approach(&left, &right, &[1.0, 2.0, 3.0]).unwrap();
        let result = result.unwrap();
        assert_eq!(result.sample_index, 0);
        assert_eq!(result.distance_km, 2.0);
        assert_eq!(result.relative_speed_km_s, 3.0);
    }

    #[test]
    fn relative_metrics_preserves_invalid_row_positions() {
        let left = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            f64::NAN,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let right = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = relative_metrics(&left, &right).unwrap();
        assert_eq!(&result[..2], &[5.0, 0.0]);
        assert!(result[2].is_nan() && result[3].is_nan());
    }
}
