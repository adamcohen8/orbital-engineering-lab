//! Small deterministic numerical kernels for campaign and optimization loops.
//!
//! Python retains trial construction, objective callbacks, scheduling,
//! persistence, and evidence identity.  These functions only operate on
//! caller-owned flat arrays and return deterministic state updates.

#[derive(Clone, Debug, PartialEq)]
pub struct PsoUpdate {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub personal_best_positions: Vec<f64>,
    pub personal_best_costs: Vec<f64>,
    pub global_best_index: usize,
}

/// Propose one bounded PSO step without evaluating objectives.
///
/// This split boundary is useful when the objective is a Python callback:
/// Rust performs the repeated particle arithmetic, Python evaluates the new
/// positions, and callers can then either update bests in Python or use
/// [`pso_update`] with the returned costs.
pub fn pso_propose(
    positions: &[f64],
    velocities: &[f64],
    personal_best_positions: &[f64],
    global_best_position: &[f64],
    random_cognitive: &[f64],
    random_social: &[f64],
    lower: &[f64],
    upper: &[f64],
    inertia: f64,
    cognitive: f64,
    social: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if positions.is_empty() || lower.is_empty() || positions.len() % lower.len() != 0 {
        return Err("PSO arrays have inconsistent particle dimensions".to_owned());
    }
    let dimension = lower.len();
    let particles = positions.len() / dimension;
    if upper.len() != dimension
        || global_best_position.len() != dimension
        || velocities.len() != positions.len()
        || personal_best_positions.len() != positions.len()
        || random_cognitive.len() != positions.len()
        || random_social.len() != positions.len()
    {
        return Err("PSO arrays have inconsistent shapes".to_owned());
    }
    if !inertia.is_finite() || !cognitive.is_finite() || !social.is_finite() {
        return Err("PSO coefficients must be finite".to_owned());
    }
    for axis in 0..dimension {
        if !lower[axis].is_finite() || !upper[axis].is_finite() || upper[axis] <= lower[axis] {
            return Err("PSO bounds must be finite with upper greater than lower".to_owned());
        }
    }
    // Finite checking has no side effects. A full boolean reduction lets the
    // compiler check contiguous lanes together instead of a chained iterator
    // branch at every scalar. The same malformed inputs still fail here.
    let all_finite = |values: &[f64]| {
        values
            .iter()
            .fold(true, |valid, value| valid & value.is_finite())
    };
    if !all_finite(positions)
        || !all_finite(velocities)
        || !all_finite(personal_best_positions)
        || !all_finite(global_best_position)
        || !all_finite(random_cognitive)
        || !all_finite(random_social)
    {
        return Err("PSO arrays must contain only finite values".to_owned());
    }
    let mut next_positions = vec![0.0; positions.len()];
    let mut next_velocities = vec![0.0; velocities.len()];
    for particle in 0..particles {
        for axis in 0..dimension {
            let index = particle * dimension + axis;
            let velocity = inertia * velocities[index]
                + cognitive
                    * random_cognitive[index]
                    * (personal_best_positions[index] - positions[index])
                + social * random_social[index] * (global_best_position[axis] - positions[index]);
            next_velocities[index] = velocity;
            next_positions[index] = (positions[index] + velocity).clamp(lower[axis], upper[axis]);
        }
    }
    Ok((next_positions, next_velocities))
}

pub fn pso_update(
    positions: &[f64],
    velocities: &[f64],
    personal_best_positions: &[f64],
    personal_best_costs: &[f64],
    global_best_position: &[f64],
    random_cognitive: &[f64],
    random_social: &[f64],
    lower: &[f64],
    upper: &[f64],
    inertia: f64,
    cognitive: f64,
    social: f64,
    costs: &[f64],
) -> Result<PsoUpdate, String> {
    let dimension = lower.len();
    if personal_best_costs.len() != positions.len() / dimension.max(1)
        || costs.len() != positions.len() / dimension.max(1)
        || personal_best_costs
            .iter()
            .chain(costs)
            .any(|value| !value.is_finite())
    {
        return Err("PSO arrays must contain only finite values".to_owned());
    }
    let (next_positions, next_velocities) = pso_propose(
        positions,
        velocities,
        personal_best_positions,
        global_best_position,
        random_cognitive,
        random_social,
        lower,
        upper,
        inertia,
        cognitive,
        social,
    )?;
    let particles = personal_best_costs.len();
    let mut next_personal_best_positions = personal_best_positions.to_vec();
    let mut next_personal_best_costs = personal_best_costs.to_vec();
    for particle in 0..particles {
        if costs[particle] < next_personal_best_costs[particle] {
            let start = particle * dimension;
            next_personal_best_positions[start..start + dimension]
                .copy_from_slice(&next_positions[start..start + dimension]);
            next_personal_best_costs[particle] = costs[particle];
        }
    }
    let mut global_best_index = 0;
    for particle in 1..particles {
        if next_personal_best_costs[particle] < next_personal_best_costs[global_best_index] {
            global_best_index = particle;
        }
    }
    Ok(PsoUpdate {
        positions: next_positions,
        velocities: next_velocities,
        personal_best_positions: next_personal_best_positions,
        personal_best_costs: next_personal_best_costs,
        global_best_index,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricStats {
    pub count: usize,
    pub missing_count: usize,
    pub min: Option<f64>,
    pub p05: Option<f64>,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub p95: Option<f64>,
    pub max: Option<f64>,
    pub std: Option<f64>,
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }
}

/// Stable byte radix sorting of finite floats. Equal signed zeros share a
/// key, preserving the same order as the prior stable numeric comparison.
fn sort_finite(values: &mut Vec<f64>) {
    if values.len() < 128 {
        values.sort_by(|left, right| left.partial_cmp(right).unwrap());
        return;
    }
    fn key(value: f64) -> u64 {
        let bits = if value == 0.0 { 0 } else { value.to_bits() };
        if bits >> 63 != 0 {
            !bits
        } else {
            bits ^ (1_u64 << 63)
        }
    }
    let mut scratch = vec![0.0; values.len()];
    for shift in (0..64).step_by(8) {
        let mut counts = [0_usize; 256];
        for &value in values.iter() {
            counts[((key(value) >> shift) & 255) as usize] += 1;
        }
        if counts.iter().any(|&count| count == values.len()) {
            continue;
        }
        let mut offset = 0;
        for count in &mut counts {
            let next = offset + *count;
            *count = offset;
            offset = next;
        }
        for &value in values.iter() {
            let bucket = ((key(value) >> shift) & 255) as usize;
            scratch[counts[bucket]] = value;
            counts[bucket] += 1;
        }
        std::mem::swap(values, &mut scratch);
    }
}

/// Match NumPy's linear percentile and sample standard deviation conventions
/// used by the Scale Monte Carlo summary writer.
pub fn aggregate_metric(values: &[f64], attempted_count: usize) -> Result<MetricStats, String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("campaign metric aggregation expects finite values".to_owned());
    }
    let mut sorted = values.to_vec();
    sort_finite(&mut sorted);
    let count = sorted.len();
    let missing_count = attempted_count.saturating_sub(count);
    if count == 0 {
        return Ok(MetricStats {
            count,
            missing_count,
            min: None,
            p05: None,
            mean: None,
            median: None,
            p95: None,
            max: None,
            std: None,
        });
    }
    let sum: f64 = sorted.iter().sum();
    let mean = sum / count as f64;
    let std = if count > 1 {
        let squared: f64 = sorted.iter().map(|value| (value - mean).powi(2)).sum();
        Some((squared / (count - 1) as f64).sqrt())
    } else {
        Some(0.0)
    };
    Ok(MetricStats {
        count,
        missing_count,
        min: Some(sorted[0]),
        p05: Some(percentile(&sorted, 0.05)),
        mean: Some(mean),
        median: Some(percentile(&sorted, 0.5)),
        p95: Some(percentile(&sorted, 0.95)),
        max: Some(sorted[count - 1]),
        std,
    })
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyByteArray, PyBytes};
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
    #[allow(clippy::too_many_arguments)]
    pub fn campaign_pso_propose(
        positions: Vec<f64>,
        velocities: Vec<f64>,
        personal_best_positions: Vec<f64>,
        global_best_position: Vec<f64>,
        random_cognitive: Vec<f64>,
        random_social: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        inertia: f64,
        cognitive: f64,
        social: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        pso_propose(
            &positions,
            &velocities,
            &personal_best_positions,
            &global_best_position,
            &random_cognitive,
            &random_social,
            &lower,
            &upper,
            inertia,
            cognitive,
            social,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    pub fn campaign_pso_propose_bytes(
        py: Python<'_>,
        positions: &[u8],
        velocities: &[u8],
        personal_best_positions: &[u8],
        global_best_position: &[u8],
        random_cognitive: &[u8],
        random_social: &[u8],
        lower: &[u8],
        upper: &[u8],
        inertia: f64,
        cognitive: f64,
        social: f64,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let positions = decode_f64_bytes(positions)?;
        let velocities = decode_f64_bytes(velocities)?;
        let personal_best_positions = decode_f64_bytes(personal_best_positions)?;
        let global_best_position = decode_f64_bytes(global_best_position)?;
        let random_cognitive = decode_f64_bytes(random_cognitive)?;
        let random_social = decode_f64_bytes(random_social)?;
        let lower = decode_f64_bytes(lower)?;
        let upper = decode_f64_bytes(upper)?;
        let compute = || {
            pso_propose(
                &positions,
                &velocities,
                &personal_best_positions,
                &global_best_position,
                &random_cognitive,
                &random_social,
                &lower,
                &upper,
                inertia,
                cognitive,
                social,
            )
        };
        // Small control updates are shorter than a useful Python thread
        // timeslice. Hold the GIL as the original list API does; release it
        // for larger particle batches so long work remains cooperative.
        let result = if positions.len() <= 16_384 {
            compute()
        } else {
            py.detach(compute)
        };
        let (next_positions, next_velocities) = result.map_err(PyValueError::new_err)?;
        Ok((
            PyBytes::new(py, &encode_f64_bytes(&next_positions)).unbind(),
            PyBytes::new(py, &encode_f64_bytes(&next_velocities)).unbind(),
        ))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    pub fn campaign_pso_propose_buffers(
        py: Python<'_>,
        positions: &[u8],
        velocities: &[u8],
        personal_best_positions: &[u8],
        global_best_position: &[u8],
        random_cognitive: &[u8],
        random_social: &[u8],
        lower: &[u8],
        upper: &[u8],
        inertia: f64,
        cognitive: f64,
        social: f64,
    ) -> PyResult<(Py<PyByteArray>, Py<PyByteArray>)> {
        let positions = decode_f64_bytes(positions)?;
        let velocities = decode_f64_bytes(velocities)?;
        let personal_best_positions = decode_f64_bytes(personal_best_positions)?;
        let global_best_position = decode_f64_bytes(global_best_position)?;
        let random_cognitive = decode_f64_bytes(random_cognitive)?;
        let random_social = decode_f64_bytes(random_social)?;
        let lower = decode_f64_bytes(lower)?;
        let upper = decode_f64_bytes(upper)?;
        let compute = || {
            pso_propose(
                &positions,
                &velocities,
                &personal_best_positions,
                &global_best_position,
                &random_cognitive,
                &random_social,
                &lower,
                &upper,
                inertia,
                cognitive,
                social,
            )
        };
        // Small control updates are shorter than a useful Python thread
        // timeslice. Hold the GIL as the original list API does; release it
        // for larger particle batches so long work remains cooperative.
        let result = if positions.len() <= 16_384 {
            compute()
        } else {
            py.detach(compute)
        };
        let (next_positions, next_velocities) = result.map_err(PyValueError::new_err)?;
        Ok((
            PyByteArray::new(py, &encode_f64_bytes(&next_positions)).unbind(),
            PyByteArray::new(py, &encode_f64_bytes(&next_velocities)).unbind(),
        ))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    pub fn campaign_pso_update(
        positions: Vec<f64>,
        velocities: Vec<f64>,
        personal_best_positions: Vec<f64>,
        personal_best_costs: Vec<f64>,
        global_best_position: Vec<f64>,
        random_cognitive: Vec<f64>,
        random_social: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        inertia: f64,
        cognitive: f64,
        social: f64,
        costs: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize)> {
        let result = pso_update(
            &positions,
            &velocities,
            &personal_best_positions,
            &personal_best_costs,
            &global_best_position,
            &random_cognitive,
            &random_social,
            &lower,
            &upper,
            inertia,
            cognitive,
            social,
            &costs,
        )
        .map_err(PyValueError::new_err)?;
        Ok((
            result.positions,
            result.velocities,
            result.personal_best_positions,
            result.personal_best_costs,
            result.global_best_index,
        ))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    pub fn campaign_pso_update_bytes(
        py: Python<'_>,
        positions: &[u8],
        velocities: &[u8],
        personal_best_positions: &[u8],
        personal_best_costs: &[u8],
        global_best_position: &[u8],
        random_cognitive: &[u8],
        random_social: &[u8],
        lower: &[u8],
        upper: &[u8],
        inertia: f64,
        cognitive: f64,
        social: f64,
        costs: &[u8],
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, usize)> {
        let positions = decode_f64_bytes(positions)?;
        let velocities = decode_f64_bytes(velocities)?;
        let personal_best_positions = decode_f64_bytes(personal_best_positions)?;
        let personal_best_costs = decode_f64_bytes(personal_best_costs)?;
        let global_best_position = decode_f64_bytes(global_best_position)?;
        let random_cognitive = decode_f64_bytes(random_cognitive)?;
        let random_social = decode_f64_bytes(random_social)?;
        let lower = decode_f64_bytes(lower)?;
        let upper = decode_f64_bytes(upper)?;
        let costs = decode_f64_bytes(costs)?;
        let result = py
            .detach(|| {
                pso_update(
                    &positions,
                    &velocities,
                    &personal_best_positions,
                    &personal_best_costs,
                    &global_best_position,
                    &random_cognitive,
                    &random_social,
                    &lower,
                    &upper,
                    inertia,
                    cognitive,
                    social,
                    &costs,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            PyBytes::new(py, &encode_f64_bytes(&result.positions)).unbind(),
            PyBytes::new(py, &encode_f64_bytes(&result.velocities)).unbind(),
            PyBytes::new(py, &encode_f64_bytes(&result.personal_best_positions)).unbind(),
            PyBytes::new(py, &encode_f64_bytes(&result.personal_best_costs)).unbind(),
            result.global_best_index,
        ))
    }

    #[pyfunction]
    pub fn campaign_aggregate_metric(
        values: Vec<f64>,
        attempted_count: usize,
    ) -> PyResult<(
        usize,
        usize,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
    )> {
        let result = aggregate_metric(&values, attempted_count).map_err(PyValueError::new_err)?;
        Ok((
            result.count,
            result.missing_count,
            result.min,
            result.p05,
            result.mean,
            result.median,
            result.p95,
            result.max,
            result.std,
        ))
    }

    #[pyfunction]
    pub fn campaign_aggregate_metric_bytes(
        values: &[u8],
        attempted_count: usize,
    ) -> PyResult<(
        usize,
        usize,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
    )> {
        let values = decode_f64_bytes(values)?;
        let result = aggregate_metric(&values, attempted_count).map_err(PyValueError::new_err)?;
        Ok((
            result.count,
            result.missing_count,
            result.min,
            result.p05,
            result.mean,
            result.median,
            result.p95,
            result.max,
            result.std,
        ))
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(campaign_pso_propose, module)?)?;
        module.add_function(wrap_pyfunction!(campaign_pso_update, module)?)?;
        module.add_function(wrap_pyfunction!(campaign_aggregate_metric, module)?)?;
        module.add_function(wrap_pyfunction!(campaign_pso_propose_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(campaign_pso_propose_buffers, module)?)?;
        module.add_function(wrap_pyfunction!(campaign_pso_update_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(campaign_aggregate_metric_bytes, module)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::{aggregate_metric, pso_update};

    #[test]
    fn radix_sort_matches_stable_numeric_order_bit_for_bit() {
        let mut values: Vec<f64> = (0..2000)
            .map(|index| {
                let bits = (index as u64).wrapping_mul(0x9e3779b97f4a7c15);
                let bits = if bits & (0x7ff_u64 << 52) == 0x7ff_u64 << 52 {
                    bits & !(1_u64 << 52)
                } else {
                    bits
                };
                f64::from_bits(bits)
            })
            .collect();
        values.extend([-0.0, 0.0, -0.0, 0.0, f64::MAX, -f64::MAX, 1.0, -1.0]);
        let mut expected = values.clone();
        expected.sort_by(|left, right| left.partial_cmp(right).unwrap());
        super::sort_finite(&mut values);
        assert_eq!(
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn pso_update_matches_bounded_scalar_step() {
        let result = pso_update(
            &[0.0, 0.5, 1.0, 0.5],
            &[0.2, -0.1, -0.2, 0.1],
            &[0.1, 0.5, 0.8, 0.5],
            &[3.0, 1.0],
            &[0.5, 0.5],
            &[1.0, 0.0, 0.5, 1.0],
            &[0.0, 1.0, 1.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            0.5,
            1.0,
            1.0,
            &[0.1, 2.0],
        )
        .unwrap();
        assert_eq!(result.positions.len(), 4);
        assert_eq!(result.personal_best_costs, vec![0.1, 1.0]);
        assert_eq!(result.global_best_index, 0);
        assert!(result
            .positions
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn metric_stats_match_linear_percentiles() {
        let result = aggregate_metric(&[1.0, 2.0, 3.0, 4.0], 5).unwrap();
        assert_eq!(result.count, 4);
        assert_eq!(result.missing_count, 1);
        assert!((result.p05.unwrap() - 1.15).abs() < 1e-14);
        assert_eq!(result.median, Some(2.5));
        assert!((result.p95.unwrap() - 3.85).abs() < 1e-14);
        assert!((result.std.unwrap() - 1.2909944487358056).abs() < 1e-15);
    }
}
