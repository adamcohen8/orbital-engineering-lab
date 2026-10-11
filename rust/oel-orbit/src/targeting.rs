//! Native numerical kernels for Lambert transfers and bounded target studies.
//!
//! Targeting policy and trajectory evaluation stay in Python.  The routines
//! here cover the repeated universal-variable and residual arithmetic used by
//! candidate sweeps and differential correction.

#[derive(Clone, Copy, Debug)]
pub struct LambertResult {
    pub v1: [f64; 3],
    pub v2: [f64; 3],
    pub residual_s: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn stumpff_pair(z: f64) -> (f64, f64) {
    if z > 1.0e-8 {
        let root = z.sqrt();
        let (sine, cosine) = root.sin_cos();
        ((1.0 - cosine) / z, (root - sine) / root.powi(3))
    } else if z < -1.0e-8 {
        let root = (-z).sqrt();
        (
            (root.cosh() - 1.0) / (-z),
            (root.sinh() - root) / root.powi(3),
        )
    } else {
        (
            0.5 - z / 24.0 + z * z / 720.0 - z * z * z / 40320.0,
            1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z * z * z / 362880.0,
        )
    }
}

fn norm(values: [f64; 3]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

/// Solve a zero-revolution universal-variable Lambert transfer.
pub fn solve_lambert(
    r1: [f64; 3],
    r2: [f64; 3],
    time_of_flight_s: f64,
    mu_km3_s2: f64,
    short_way: bool,
    max_iterations: usize,
    tolerance_s: f64,
) -> Result<LambertResult, String> {
    if r1.iter().chain(r2.iter()).any(|value| !value.is_finite())
        || !time_of_flight_s.is_finite()
        || !mu_km3_s2.is_finite()
        || !tolerance_s.is_finite()
    {
        return Err("Lambert inputs must be finite".into());
    }
    if time_of_flight_s <= 0.0 || mu_km3_s2 <= 0.0 || max_iterations == 0 || tolerance_s <= 0.0 {
        return Err("Lambert time, mu, iterations, and tolerance must be positive".into());
    }
    let r1_norm = norm(r1);
    let r2_norm = norm(r2);
    if r1_norm <= 0.0 || r2_norm <= 0.0 {
        return Err("Lambert endpoint position norms must be positive".into());
    }
    let cos_theta = (dot(r1, r2) / (r1_norm * r2_norm)).clamp(-1.0, 1.0);
    let mut sin_theta = norm(cross(r1, r2)) / (r1_norm * r2_norm);
    if !short_way {
        sin_theta = -sin_theta;
    }
    let one_minus_cos = 1.0 - cos_theta;
    if one_minus_cos <= 1.0e-14 || sin_theta.abs() <= 1.0e-14 {
        return Err("Lambert transfer angle is singular for the current solver".into());
    }
    let a = sin_theta * (r1_norm * r2_norm / one_minus_cos).sqrt();
    if a.abs() <= 1.0e-14 {
        return Err("Lambert geometry produced a singular transfer parameter".into());
    }
    let mu_root = mu_km3_s2.sqrt();
    let norm_sum = r1_norm + r2_norm;
    let time_for_z = |z: f64| -> (f64, f64, f64, f64) {
        let (c, s) = stumpff_pair(z);
        if c <= 0.0 || !c.is_finite() || !s.is_finite() {
            return (f64::NAN, f64::NAN, c, s);
        }
        let y = norm_sum + a * ((z * s - 1.0) / c.sqrt());
        if y < 0.0 || !y.is_finite() {
            return (f64::NAN, y, c, s);
        }
        let x = (y / c).sqrt();
        let dt = (x.powi(3) * s + a * y.sqrt()) / mu_root;
        (dt, y, c, s)
    };

    let lower = -4.0 * std::f64::consts::PI.powi(2);
    // Keep the zero-revolution bracket below the first positive Stumpff pole.
    // Halving the angular gap reaches long times without crossing into a later
    // revolution; the negative lower bound remains available for hyperbolic arcs.
    let mut upper = 0.0;
    let mut dt_upper = time_for_z(upper).0;
    let mut pole_gap_rad = std::f64::consts::PI;
    let mut expansion = 0;
    while (!dt_upper.is_finite() || dt_upper < time_of_flight_s) && expansion < 64 {
        let candidate_upper = (2.0 * std::f64::consts::PI - pole_gap_rad).powi(2);
        if candidate_upper <= upper {
            break;
        }
        let candidate_dt = time_for_z(candidate_upper).0;
        if !candidate_dt.is_finite() {
            break;
        }
        upper = candidate_upper;
        dt_upper = candidate_dt;
        pole_gap_rad *= 0.5;
        expansion += 1;
    }
    if !dt_upper.is_finite() || dt_upper < time_of_flight_s {
        return Err("Lambert solver could not bracket the requested time of flight".into());
    }

    let mut lo = lower;
    let mut dt = f64::NAN;
    let mut y = f64::NAN;
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    for index in 1..=max_iterations {
        iterations = index;
        let z = 0.5 * (lo + upper);
        let (candidate_dt, candidate_y, _, _) = time_for_z(z);
        dt = candidate_dt;
        y = candidate_y;
        if !dt.is_finite() || y < 0.0 {
            lo = z;
            continue;
        }
        residual = dt - time_of_flight_s;
        if residual.abs() <= tolerance_s {
            break;
        }
        if residual <= 0.0 {
            lo = z;
        } else {
            upper = z;
        }
    }
    if !dt.is_finite() || !y.is_finite() || y < 0.0 {
        return Err("Lambert solver did not converge to a finite transfer".into());
    }
    let f = 1.0 - y / r1_norm;
    let g = a * (y / mu_km3_s2).sqrt();
    let gdot = 1.0 - y / r2_norm;
    if g.abs() <= 1.0e-14 || !g.is_finite() {
        return Err("Lambert solver produced a singular Lagrange g coefficient".into());
    }
    let v1 = [
        (r2[0] - f * r1[0]) / g,
        (r2[1] - f * r1[1]) / g,
        (r2[2] - f * r1[2]) / g,
    ];
    let v2 = [
        (gdot * r2[0] - r1[0]) / g,
        (gdot * r2[1] - r1[1]) / g,
        (gdot * r2[2] - r1[2]) / g,
    ];
    if v1.iter().chain(v2.iter()).any(|value| !value.is_finite()) {
        return Err("Lambert solver produced non-finite endpoint velocities".into());
    }
    Ok(LambertResult {
        v1,
        v2,
        residual_s: residual,
        iterations,
        converged: residual.abs() <= tolerance_s,
    })
}

pub fn scaled_residual_norms(
    observed: &[f64],
    targets: &[f64],
    tolerances: &[f64],
    rows: usize,
    columns: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if rows == 0
        || columns == 0
        || observed.len() != rows * columns
        || targets.len() != columns
        || tolerances.len() != columns
    {
        return Err("target residual dimensions are inconsistent".into());
    }
    if observed
        .iter()
        .chain(targets)
        .chain(tolerances)
        .any(|value| !value.is_finite())
        || tolerances.iter().any(|value| *value <= 0.0)
    {
        return Err("target residual values must be finite with positive tolerances".into());
    }
    let mut norms = Vec::with_capacity(rows);
    let mut maxima = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut sum = 0.0;
        let mut maximum: f64 = 0.0;
        for column in 0..columns {
            let residual =
                (observed[row * columns + column] - targets[column]) / tolerances[column];
            sum += residual * residual;
            maximum = maximum.max(residual.abs());
        }
        norms.push(sum.sqrt());
        maxima.push(maximum);
    }
    Ok((norms, maxima))
}

pub fn finite_difference_jacobian(
    left: &[f64],
    right: &[f64],
    variable_steps: &[f64],
    goal_tolerances: &[f64],
    variables: usize,
    goals: usize,
) -> Result<Vec<f64>, String> {
    if variables == 0
        || goals == 0
        || left.len() != variables * goals
        || right.len() != variables * goals
        || variable_steps.len() != variables
        || goal_tolerances.len() != goals
    {
        return Err("target Jacobian dimensions are inconsistent".into());
    }
    if left
        .iter()
        .chain(right)
        .chain(variable_steps)
        .chain(goal_tolerances)
        .any(|value| !value.is_finite())
        || variable_steps.iter().any(|value| *value <= 0.0)
        || goal_tolerances.iter().any(|value| *value <= 0.0)
    {
        return Err(
            "target Jacobian values must be finite with positive steps and tolerances".into(),
        );
    }
    let mut result = vec![0.0; variables * goals];
    for variable in 0..variables {
        for goal in 0..goals {
            result[goal * variables + variable] = (right[variable * goals + goal]
                - left[variable * goals + goal])
                / variable_steps[variable]
                / goal_tolerances[goal];
        }
    }
    Ok(result)
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    #[pyo3(signature = (r1, r2, time_of_flight_s, mu_km3_s2, short_way=true, max_iterations=100, tolerance_s=1e-7))]
    fn targeting_lambert(
        r1: Vec<f64>,
        r2: Vec<f64>,
        time_of_flight_s: f64,
        mu_km3_s2: f64,
        short_way: bool,
        max_iterations: usize,
        tolerance_s: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>, f64, usize, bool)> {
        let r1: [f64; 3] = r1
            .try_into()
            .map_err(|_| PyValueError::new_err("r1 must have three values"))?;
        let r2: [f64; 3] = r2
            .try_into()
            .map_err(|_| PyValueError::new_err("r2 must have three values"))?;
        let result = solve_lambert(
            r1,
            r2,
            time_of_flight_s,
            mu_km3_s2,
            short_way,
            max_iterations,
            tolerance_s,
        )
        .map_err(PyValueError::new_err)?;
        Ok((
            result.v1.to_vec(),
            result.v2.to_vec(),
            result.residual_s,
            result.iterations,
            result.converged,
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (r1_flat, r2_flat, times, mu_km3_s2, short_way=true, max_iterations=100, tolerance_s=1e-7))]
    fn targeting_lambert_batch(
        r1_flat: Vec<f64>,
        r2_flat: Vec<f64>,
        times: Vec<f64>,
        mu_km3_s2: f64,
        short_way: bool,
        max_iterations: usize,
        tolerance_s: f64,
    ) -> PyResult<(Vec<f64>, Vec<(usize, String)>)> {
        if r1_flat.len() != r2_flat.len() || r1_flat.len() != times.len() * 3 {
            return Err(PyValueError::new_err(
                "Lambert batch arrays have inconsistent shapes",
            ));
        }
        let mut outputs = vec![f64::NAN; times.len() * 8];
        let mut errors = Vec::new();
        for index in 0..times.len() {
            let r1 = [
                r1_flat[index * 3],
                r1_flat[index * 3 + 1],
                r1_flat[index * 3 + 2],
            ];
            let r2 = [
                r2_flat[index * 3],
                r2_flat[index * 3 + 1],
                r2_flat[index * 3 + 2],
            ];
            match solve_lambert(
                r1,
                r2,
                times[index],
                mu_km3_s2,
                short_way,
                max_iterations,
                tolerance_s,
            ) {
                Ok(result) => {
                    outputs[index * 8..index * 8 + 3].copy_from_slice(&result.v1);
                    outputs[index * 8 + 3..index * 8 + 6].copy_from_slice(&result.v2);
                    outputs[index * 8 + 6] = result.residual_s;
                    outputs[index * 8 + 7] =
                        result.iterations as f64 + if result.converged { 0.0 } else { 0.5 };
                }
                Err(error) => errors.push((index, error)),
            }
        }
        Ok((outputs, errors))
    }

    #[pyfunction]
    fn targeting_scaled_residual_norms(
        observed: Vec<f64>,
        targets: Vec<f64>,
        tolerances: Vec<f64>,
        rows: usize,
        columns: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        scaled_residual_norms(&observed, &targets, &tolerances, rows, columns)
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn targeting_finite_difference_jacobian(
        left: Vec<f64>,
        right: Vec<f64>,
        variable_steps: Vec<f64>,
        goal_tolerances: Vec<f64>,
        variables: usize,
        goals: usize,
    ) -> PyResult<Vec<f64>> {
        finite_difference_jacobian(
            &left,
            &right,
            &variable_steps,
            &goal_tolerances,
            variables,
            goals,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn targeting_scaled_residual_norms_packed<'py>(
        py: Python<'py>,
        data: &[u8],
        rows: usize,
        columns: usize,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let size = rows
            .checked_mul(columns)
            .ok_or_else(|| PyValueError::new_err("target dimensions overflow"))?;
        let expected = columns
            .checked_mul(2)
            .and_then(|n| size.checked_add(n))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| PyValueError::new_err("target dimensions overflow"))?;
        if data.len() != expected {
            return Err(PyValueError::new_err(
                "target residual dimensions are inconsistent",
            ));
        }
        let values = crate::numeric_bytes::decode_f64_bytes(data)?;
        let (norms, maxima) = scaled_residual_norms(
            &values[..size],
            &values[size..size + columns],
            &values[size + columns..],
            rows,
            columns,
        )
        .map_err(PyValueError::new_err)?;
        Ok((
            crate::numeric_bytes::output_bytes(py, &norms)?.into_bound(py),
            crate::numeric_bytes::output_bytes(py, &maxima)?.into_bound(py),
        ))
    }

    #[pyfunction]
    fn targeting_lambert_packed<'py>(
        py: Python<'py>,
        data: &[u8],
        time_of_flight_s: f64,
        mu_km3_s2: f64,
        short_way: bool,
        max_iterations: usize,
        tolerance_s: f64,
    ) -> PyResult<(Bound<'py, PyBytes>, f64, usize, bool)> {
        if data.len() != 6 * 8 {
            return Err(PyValueError::new_err(
                "Lambert endpoints must have three values each",
            ));
        }
        let values: [f64; 6] = std::array::from_fn(|i| {
            f64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap())
        });
        let result = solve_lambert(
            values[..3].try_into().unwrap(),
            values[3..].try_into().unwrap(),
            time_of_flight_s,
            mu_km3_s2,
            short_way,
            max_iterations,
            tolerance_s,
        )
        .map_err(PyValueError::new_err)?;
        let output = [
            result.v1[0],
            result.v1[1],
            result.v1[2],
            result.v2[0],
            result.v2[1],
            result.v2[2],
        ];
        Ok((
            crate::numeric_bytes::output_bytes(py, &output)?.into_bound(py),
            result.residual_s,
            result.iterations,
            result.converged,
        ))
    }

    #[pyfunction]
    fn targeting_lambert_batch_packed<'py>(
        py: Python<'py>,
        data: &[u8],
        mu_km3_s2: f64,
        short_way: bool,
        max_iterations: usize,
        tolerance_s: f64,
    ) -> PyResult<(Bound<'py, PyBytes>, Vec<(usize, String)>)> {
        if data.len() % (7 * 8) != 0 {
            return Err(PyValueError::new_err(
                "Lambert batch arrays have inconsistent shapes",
            ));
        }
        let count = data.len() / (7 * 8);
        let values = crate::numeric_bytes::decode_f64_bytes(data)?;
        let mut outputs = vec![f64::NAN; count * 8];
        let mut errors = Vec::new();
        for i in 0..count {
            let r1 = values[i * 3..i * 3 + 3].try_into().unwrap();
            let r2 = values[count * 3 + i * 3..count * 3 + i * 3 + 3]
                .try_into()
                .unwrap();
            let output = &mut outputs[i * 8..i * 8 + 8];
            match solve_lambert(
                r1,
                r2,
                values[count * 6 + i],
                mu_km3_s2,
                short_way,
                max_iterations,
                tolerance_s,
            ) {
                Ok(result) => {
                    output[..3].copy_from_slice(&result.v1);
                    output[3..6].copy_from_slice(&result.v2);
                    output[6] = result.residual_s;
                    output[7] = result.iterations as f64 + if result.converged { 0.0 } else { 0.5 };
                }
                Err(error) => errors.push((i, error)),
            }
        }
        Ok((
            crate::numeric_bytes::output_bytes(py, &outputs)?.into_bound(py),
            errors,
        ))
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(
            targeting_scaled_residual_norms_packed,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(targeting_lambert_packed, module)?)?;
        module.add_function(wrap_pyfunction!(targeting_lambert_batch_packed, module)?)?;
        module.add_function(wrap_pyfunction!(targeting_lambert, module)?)?;
        module.add_function(wrap_pyfunction!(targeting_lambert_batch, module)?)?;
        module.add_function(wrap_pyfunction!(targeting_scaled_residual_norms, module)?)?;
        module.add_function(wrap_pyfunction!(
            targeting_finite_difference_jacobian,
            module
        )?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod lambert_tests {
    use super::solve_lambert;

    const MU: f64 = 398600.4418;
    const TAU: f64 = 2.0 * std::f64::consts::PI;

    fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
        left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
    }

    fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    }

    fn norm(value: [f64; 3]) -> f64 {
        dot(value, value).sqrt()
    }

    fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
        [value[0] * factor, value[1] * factor, value[2] * factor]
    }

    fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
        [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
    }

    fn first_arrival_and_period(r1: [f64; 3], v1: [f64; 3], r2: [f64; 3]) -> (f64, f64, f64, f64) {
        let r1_norm = norm(r1);
        let h = cross(r1, v1);
        let h_norm = norm(h);
        let h_hat = scale(h, 1.0 / h_norm);
        let eccentricity_vector = sub(scale(cross(v1, h), 1.0 / MU), scale(r1, 1.0 / r1_norm));
        let eccentricity = norm(eccentricity_vector);
        let e_hat = scale(eccentricity_vector, 1.0 / eccentricity);
        let q_hat = cross(h_hat, e_hat);
        let true_anomaly = |position: [f64; 3]| {
            let r_hat = scale(position, 1.0 / norm(position));
            dot(r_hat, q_hat).atan2(dot(r_hat, e_hat)).rem_euclid(TAU)
        };
        let eccentric_anomaly = |true_anomaly: f64| {
            let value = 2.0
                * ((1.0 - eccentricity).sqrt() * (0.5 * true_anomaly).sin())
                    .atan2((1.0 + eccentricity).sqrt() * (0.5 * true_anomaly).cos());
            value.rem_euclid(TAU)
        };
        let mean_anomaly = |true_anomaly: f64| {
            let eccentric_anomaly = eccentric_anomaly(true_anomaly);
            eccentric_anomaly - eccentricity * eccentric_anomaly.sin()
        };
        let specific_energy = 0.5 * dot(v1, v1) - MU / r1_norm;
        let semimajor_axis = -MU / (2.0 * specific_energy);
        let mean_motion = (MU / semimajor_axis.powi(3)).sqrt();
        let first_arrival = (mean_anomaly(true_anomaly(r2)) - mean_anomaly(true_anomaly(r1)))
            .rem_euclid(TAU)
            / mean_motion;
        let period = TAU / mean_motion;
        let semilatus_rectum = h_norm.powi(2) / MU;
        let target_radius_from_conic =
            semilatus_rectum / (1.0 + eccentricity * true_anomaly(r2).cos());
        (
            first_arrival,
            period,
            eccentricity,
            target_radius_from_conic,
        )
    }

    #[test]
    fn zero_revolution_lambert_stays_on_first_stumpff_domain() {
        let r1 = [7000.0, 0.0, 0.0];
        let r2 = [0.0, 8000.0, 0.0];
        let time_of_flight_s = 100_000.0;
        let result = solve_lambert(r1, r2, time_of_flight_s, MU, true, 100, 1.0e-7).unwrap();

        assert!(result.converged, "residual={}", result.residual_s);
        assert!(result.residual_s.abs() < 1.0e-5);
        let (first_arrival, period, eccentricity, target_radius) =
            first_arrival_and_period(r1, result.v1, r2);
        assert!(eccentricity < 1.0, "eccentricity={eccentricity}");
        assert!((target_radius - norm(r2)).abs() < 1.0e-6);
        assert!((first_arrival - time_of_flight_s).abs() < 1.0e-3);
        assert_eq!(((time_of_flight_s - first_arrival) / period).round(), 0.0);
    }

    #[test]
    fn hyperbolic_short_time_and_long_way_branches_remain_available() {
        let r1 = [7000.0, 0.0, 0.0];
        let quarter_turn = [0.0, 7000.0, 0.0];
        let hyperbolic = solve_lambert(r1, quarter_turn, 60.0, MU, true, 100, 1.0e-7).unwrap();
        assert!(hyperbolic.converged);
        assert!(0.5 * dot(hyperbolic.v1, hyperbolic.v1) - MU / norm(r1) > 0.0);

        let angle = 120.0_f64.to_radians();
        let r2 = [7000.0 * angle.cos(), 7000.0 * angle.sin(), 0.0];
        let short_way = solve_lambert(r1, r2, 4000.0, MU, true, 100, 1.0e-7).unwrap();
        let long_way = solve_lambert(r1, r2, 4000.0, MU, false, 100, 1.0e-7).unwrap();
        assert!(short_way.converged && long_way.converged);
        assert!(cross(r1, short_way.v1)[2] > 0.0);
        assert!(cross(r1, long_way.v1)[2] < 0.0);
    }
}
