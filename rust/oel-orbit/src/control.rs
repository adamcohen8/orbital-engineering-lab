//! Native numerical kernels for OEL guidance and actuator allocation.
//!
//! This module deliberately owns only dense numerical loops.  Controller policy,
//! command sequencing, saturation diagnostics, and Python plug-ins remain in
//! the Python façade.  Matrices are row-major and all values use the units of
//! the caller.

fn finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn mat_vec(matrix: &[f64], rows: usize, cols: usize, vector: &[f64]) -> Vec<f64> {
    (0..rows)
        .map(|row| {
            matrix[row * cols..row * cols + cols]
                .iter()
                .zip(vector)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect()
}

/// Project a sequence of controls onto independent Euclidean norm balls.
pub fn project_sequence(
    controls: &[f64],
    horizon: usize,
    control_dim: usize,
    max_norm: f64,
) -> Result<Vec<f64>, String> {
    if horizon == 0 || control_dim == 0 || controls.len() != horizon * control_dim {
        return Err("control sequence dimensions are inconsistent".into());
    }
    if !max_norm.is_finite() || max_norm < 0.0 || !finite(controls) {
        return Err("control projection requires finite controls and a non-negative limit".into());
    }
    let mut result = controls.to_vec();
    for step in 0..horizon {
        let start = step * control_dim;
        let norm = result[start..start + control_dim]
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if max_norm == 0.0 {
            result[start..start + control_dim].fill(0.0);
        } else if norm > max_norm {
            let scale = max_norm / norm;
            for value in &mut result[start..start + control_dim] {
                *value *= scale;
            }
        }
    }
    Ok(result)
}

/// Apply a dense matrix to a vector without allocating an intermediate
/// matrix object at the Python boundary.
pub fn mat_vec_clip(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    vector: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Result<Vec<f64>, String> {
    if rows == 0
        || columns == 0
        || matrix.len() != rows * columns
        || vector.len() != columns
        || lower.len() != rows
        || upper.len() != rows
        || !finite(matrix)
        || !finite(vector)
        || !finite(lower)
        || !finite(upper)
    {
        return Err("matrix-vector dimensions or values are inconsistent".into());
    }
    if (0..rows).any(|row| lower[row] > upper[row]) {
        return Err("matrix-vector lower bound exceeds upper bound".into());
    }
    Ok(mat_vec(matrix, rows, columns, vector)
        .into_iter()
        .enumerate()
        .map(|(row, value)| value.clamp(lower[row], upper[row]))
        .collect())
}

/// Apply a dense matrix-vector map without clipping.
pub fn mat_vec_native(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    vector: &[f64],
) -> Result<Vec<f64>, String> {
    if rows == 0
        || columns == 0
        || matrix.len() != rows * columns
        || vector.len() != columns
        || !finite(matrix)
        || !finite(vector)
    {
        return Err("matrix-vector dimensions or values are inconsistent".into());
    }
    Ok(mat_vec(matrix, rows, columns, vector))
}

/// Roll out a linear discrete model and evaluate the quadratic MPC objective.
/// Returns `(states, cost, analytic_gradient)` where states includes `x_0`.
fn mpc_rollout(
    x0: &[f64],
    target: &[f64],
    a: &[f64],
    b: &[f64],
    q: &[f64],
    terminal: &[f64],
    r: &[f64],
    rd: &[f64],
    previous_control: &[f64],
    controls: &[f64],
    state_dim: usize,
    control_dim: usize,
    compute_gradient: bool,
) -> Result<(Vec<f64>, f64, Vec<f64>), String> {
    if state_dim == 0
        || control_dim == 0
        || x0.len() != state_dim
        || target.len() != state_dim
        || a.len() != state_dim * state_dim
        || b.len() != state_dim * control_dim
        || q.len() != state_dim
        || terminal.len() != state_dim
        || r.len() != control_dim
        || rd.len() != control_dim
        || previous_control.len() != control_dim
        || controls.is_empty()
        || controls.len() % control_dim != 0
    {
        return Err("MPC dimensions are inconsistent".into());
    }
    let horizon = controls.len() / control_dim;
    if ![
        x0,
        target,
        a,
        b,
        q,
        terminal,
        r,
        rd,
        previous_control,
        controls,
    ]
    .iter()
    .all(|values| finite(values))
        || q.iter().any(|value| *value < 0.0)
        || terminal.iter().any(|value| *value < 0.0)
        || r.iter().any(|value| *value < 0.0)
        || rd.iter().any(|value| *value < 0.0)
    {
        return Err("MPC inputs must be finite with non-negative weights".into());
    }
    let mut states = vec![0.0; (horizon + 1) * state_dim];
    states[..state_dim].copy_from_slice(x0);
    let mut cost = 0.0;
    for step in 0..horizon {
        let u = &controls[step * control_dim..(step + 1) * control_dim];
        let (known, remaining) = states.split_at_mut((step + 1) * state_dim);
        let x = &known[step * state_dim..];
        let next = &mut remaining[..state_dim];
        for index in 0..state_dim {
            let ax = a[index * state_dim..(index + 1) * state_dim]
                .iter()
                .zip(x)
                .map(|(l, r)| l * r)
                .sum::<f64>();
            let bu = b[index * control_dim..(index + 1) * control_dim]
                .iter()
                .zip(u)
                .map(|(l, r)| l * r)
                .sum::<f64>();
            next[index] = ax + bu;
        }
        cost += next
            .iter()
            .zip(target)
            .zip(q)
            .map(|((value, goal), weight)| {
                let error = value - goal;
                weight * error * error
            })
            .sum::<f64>();
        let previous = if step == 0 {
            previous_control
        } else {
            &controls[(step - 1) * control_dim..step * control_dim]
        };
        cost += u
            .iter()
            .zip(r)
            .map(|(value, weight)| weight * value * value)
            .sum::<f64>();
        cost += u
            .iter()
            .zip(previous)
            .zip(rd)
            .map(|((value, old), weight)| weight * (value - old).powi(2))
            .sum::<f64>();
    }
    let final_state = &states[horizon * state_dim..(horizon + 1) * state_dim];
    cost += final_state
        .iter()
        .zip(target)
        .zip(terminal)
        .map(|((value, goal), weight)| weight * (value - goal).powi(2))
        .sum::<f64>();

    if !compute_gradient {
        return Ok((states, cost, Vec::new()));
    }
    let mut adjoint = vec![0.0; (horizon + 1) * state_dim];
    for state_index in 1..=horizon {
        let state = &states[state_index * state_dim..(state_index + 1) * state_dim];
        for index in 0..state_dim {
            adjoint[state_index * state_dim + index] =
                2.0 * q[index] * (state[index] - target[index]);
        }
    }
    for index in 0..state_dim {
        adjoint[horizon * state_dim + index] +=
            2.0 * terminal[index] * (final_state[index] - target[index]);
    }
    let mut gradient = vec![0.0; controls.len()];
    for step in (0..horizon).rev() {
        let u = &controls[step * control_dim..(step + 1) * control_dim];
        let old = if step == 0 {
            previous_control
        } else {
            &controls[(step - 1) * control_dim..step * control_dim]
        };
        let (preceding, following) = adjoint.split_at_mut((step + 1) * state_dim);
        let next_adjoint = &following[..state_dim];
        let current_adjoint = &mut preceding[step * state_dim..];
        for index in 0..state_dim {
            current_adjoint[index] = (0..state_dim)
                .map(|row| a[row * state_dim + index] * next_adjoint[row])
                .sum::<f64>();
            if step > 0 {
                current_adjoint[index] +=
                    2.0 * q[index] * (states[step * state_dim + index] - target[index]);
            }
        }
        for index in 0..control_dim {
            gradient[step * control_dim + index] = (0..state_dim)
                .map(|row| b[row * control_dim + index] * next_adjoint[row])
                .sum::<f64>()
                + 2.0 * r[index] * u[index]
                + 2.0 * rd[index] * (u[index] - old[index]);
            if step + 1 < horizon {
                gradient[step * control_dim + index] -=
                    2.0 * rd[index] * (controls[(step + 1) * control_dim + index] - u[index]);
            }
        }
    }
    Ok((states, cost, gradient))
}

/// Preserve the full rollout/gradient API; cost-only callers share exactly
/// the same forward arithmetic without constructing the unused adjoint.
#[allow(clippy::too_many_arguments)]
pub fn mpc_rollout_cost_gradient(
    x0: &[f64],
    target: &[f64],
    a: &[f64],
    b: &[f64],
    q: &[f64],
    terminal: &[f64],
    r: &[f64],
    rd: &[f64],
    previous_control: &[f64],
    controls: &[f64],
    state_dim: usize,
    control_dim: usize,
) -> Result<(Vec<f64>, f64, Vec<f64>), String> {
    mpc_rollout(
        x0,
        target,
        a,
        b,
        q,
        terminal,
        r,
        rd,
        previous_control,
        controls,
        state_dim,
        control_dim,
        true,
    )
}

/// Evaluate several complete two-body relative-MPC trials behind one boundary.
/// The caller retains candidate generation, ordering, deadlines and acceptance.
pub fn relative_mpc_cost_batch(
    prepared: &[f64],
    controls: &[f64],
    candidates: usize,
    horizon: usize,
    dt_s: f64,
    mu_km3_s2: f64,
    max_accel_km_s2: f64,
) -> Result<Vec<f64>, String> {
    use crate::{rk4_step_validated, OrbitParameters, State};

    if prepared.len() != 45 || candidates == 0 || horizon == 0
        || controls.len() != candidates.checked_mul(horizon).and_then(|n| n.checked_mul(3))
            .ok_or("relative MPC dimensions overflow")?
        || !finite(prepared) || !finite(controls)
        || !dt_s.is_finite() || dt_s <= 0.0
        || !mu_km3_s2.is_finite() || mu_km3_s2 <= 0.0
        || !max_accel_km_s2.is_finite() || max_accel_km_s2 < 0.0
    {
        return Err("relative MPC dimensions or values are invalid".into());
    }
    let initial_chaser: [f64; 6] = prepared[0..6].try_into().expect("checked length");
    let initial_target: [f64; 6] = prepared[6..12].try_into().expect("checked length");
    let previous: [f64; 3] = prepared[12..15].try_into().expect("checked length");
    let target: [f64; 6] = prepared[15..21].try_into().expect("checked length");
    let signs: [f64; 6] = prepared[21..27].try_into().expect("checked length");
    let q: [f64; 6] = prepared[27..33].try_into().expect("checked length");
    let terminal: [f64; 6] = prepared[33..39].try_into().expect("checked length");
    let r: [f64; 3] = prepared[39..42].try_into().expect("checked length");
    let rd: [f64; 3] = prepared[42..45].try_into().expect("checked length");
    let initial_chaser = State::from_array(initial_chaser).map_err(str::to_owned)?;
    let initial_target = State::from_array(initial_target).map_err(str::to_owned)?;
    let target_params = OrbitParameters {
        mu_km3_s2, include_j2: false, command_accel_eci_km_s2: [0.0; 3],
    };
    let mut costs = Vec::with_capacity(candidates);
    for candidate in 0..candidates {
        let start = candidate * horizon * 3;
        let projected = project_sequence(&controls[start..start + horizon * 3], horizon, 3, max_accel_km_s2)?;
        let mut chaser = initial_chaser;
        let mut chief = initial_target;
        let mut old = previous;
        let mut cost = 0.0;
        let mut error = [0.0; 6];
        for step in 0..horizon {
            let u: [f64; 3] = projected[step * 3..step * 3 + 3].try_into().expect("checked length");
            chief = rk4_step_validated(chief, dt_s, target_params).map_err(str::to_owned)?;
            chaser = rk4_step_validated(chaser, dt_s, OrbitParameters {
                mu_km3_s2, include_j2: false, command_accel_eci_km_s2: u,
            }).map_err(str::to_owned)?;
            let relative = crate::relative::eci_relative_to_ric(chaser.as_array(), chief.as_array())?;
            for axis in 0..6 {
                error[axis] = signs[axis] * (relative[axis] - target[axis]);
            }
            cost += (0..6).map(|axis| q[axis] * error[axis] * error[axis]).sum::<f64>();
            cost += (0..3).map(|axis| r[axis] * u[axis] * u[axis]).sum::<f64>();
            cost += (0..3).map(|axis| {
                let difference = u[axis] - old[axis];
                rd[axis] * difference * difference
            }).sum::<f64>();
            old = u;
        }
        cost += (0..6).map(|axis| terminal[axis] * error[axis] * error[axis]).sum::<f64>();
        costs.push(cost);
    }
    Ok(costs)
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    #[pyo3(signature = (controls, horizon, control_dim, max_norm))]
    fn control_project_sequence(
        controls: Vec<f64>,
        horizon: usize,
        control_dim: usize,
        max_norm: f64,
    ) -> PyResult<Vec<f64>> {
        project_sequence(&controls, horizon, control_dim, max_norm).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[pyo3(signature = (matrix, rows, columns, vector, lower, upper))]
    fn control_mat_vec_clip(
        matrix: Vec<f64>,
        rows: usize,
        columns: usize,
        vector: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        mat_vec_clip(&matrix, rows, columns, &vector, &lower, &upper).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[pyo3(signature = (matrix, rows, columns, vector))]
    fn control_mat_vec(
        matrix: Vec<f64>,
        rows: usize,
        columns: usize,
        vector: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        mat_vec_native(&matrix, rows, columns, &vector).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[pyo3(signature = (x0, target, a, b, q, terminal, r, rd, previous_control, controls, state_dim, control_dim))]
    fn control_mpc_rollout_cost_gradient(
        x0: Vec<f64>,
        target: Vec<f64>,
        a: Vec<f64>,
        b: Vec<f64>,
        q: Vec<f64>,
        terminal: Vec<f64>,
        r: Vec<f64>,
        rd: Vec<f64>,
        previous_control: Vec<f64>,
        controls: Vec<f64>,
        state_dim: usize,
        control_dim: usize,
    ) -> PyResult<(Vec<f64>, f64, Vec<f64>)> {
        mpc_rollout_cost_gradient(
            &x0,
            &target,
            &a,
            &b,
            &q,
            &terminal,
            &r,
            &rd,
            &previous_control,
            &controls,
            state_dim,
            control_dim,
        )
        .map_err(PyValueError::new_err)
    }

    fn decode(values: &[u8]) -> PyResult<std::borrow::Cow<'_, [f64]>> {
        crate::numeric_bytes::decode_f64_bytes(values)
    }
    fn encode<'py>(py: Python<'py>, values: &[f64]) -> PyResult<Bound<'py, PyBytes>> {
        Ok(crate::numeric_bytes::output_bytes(py, values)?.into_bound(py))
    }

    /// Immutable, Python-prepared thruster geometry. Allocation policy stays in NumPy.
    #[pyclass(frozen)]
    struct RCSGeometryContext {
        force: Vec<f64>,
        torque: Vec<f64>,
        columns: usize,
    }

    #[pymethods]
    impl RCSGeometryContext {
        #[new]
        fn new(force: &[u8], torque: &[u8], columns: usize) -> PyResult<Self> {
            let force = decode(force)?.into_owned();
            let torque = decode(torque)?.into_owned();
            if columns == 0
                || columns > 4096
                || force.len() != 3 * columns
                || torque.len() != 3 * columns
                || !finite(&force)
                || !finite(&torque)
            {
                return Err(PyValueError::new_err(
                    "invalid RCS geometry dimensions or values",
                ));
            }
            Ok(Self {
                force,
                torque,
                columns,
            })
        }

        fn achieved<'py>(
            &self,
            py: Python<'py>,
            forces: &[u8],
            body_to_eci: &[u8],
        ) -> PyResult<Bound<'py, PyBytes>> {
            let forces = decode(forces)?;
            let rotation = decode(body_to_eci)?;
            let force = mat_vec_native(&self.force, 3, self.columns, &forces)
                .map_err(PyValueError::new_err)?;
            let torque = mat_vec_native(&self.torque, 3, self.columns, &forces)
                .map_err(PyValueError::new_err)?;
            let eci = mat_vec_native(&rotation, 3, 3, &force).map_err(PyValueError::new_err)?;
            encode(py, &[force, torque, eci].concat())
        }
    }

    #[pyfunction]
    fn control_mat_vec_packed<'py>(
        py: Python<'py>,
        data: &[u8],
        rows: usize,
        columns: usize,
        clip: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let v = decode(data)?;
        let size = rows
            .checked_mul(columns)
            .ok_or_else(|| PyValueError::new_err("matrix dimensions overflow"))?;
        let expected = size
            .checked_add(columns)
            .and_then(|n| {
                if clip {
                    rows.checked_mul(2).and_then(|bounds| n.checked_add(bounds))
                } else {
                    Some(n)
                }
            })
            .ok_or_else(|| PyValueError::new_err("matrix dimensions overflow"))?;
        if v.len() != expected {
            return Err(PyValueError::new_err(
                "matrix-vector dimensions or values are inconsistent",
            ));
        }
        let output = if clip {
            mat_vec_clip(
                &v[..size],
                rows,
                columns,
                &v[size..size + columns],
                &v[size + columns..size + columns + rows],
                &v[size + columns + rows..],
            )
        } else {
            mat_vec_native(&v[..size], rows, columns, &v[size..])
        }
        .map_err(PyValueError::new_err)?;
        encode(py, &output)
    }

    #[pyfunction]
    fn control_mpc_packed<'py>(
        py: Python<'py>,
        data: &[u8],
        state_dim: usize,
        control_dim: usize,
        compute_gradient: bool,
    ) -> PyResult<(Bound<'py, PyBytes>, f64, Bound<'py, PyBytes>)> {
        let v = decode(data)?;
        let sizes = [
            state_dim,
            state_dim,
            state_dim
                .checked_mul(state_dim)
                .ok_or_else(|| PyValueError::new_err("MPC dimensions overflow"))?,
            state_dim
                .checked_mul(control_dim)
                .ok_or_else(|| PyValueError::new_err("MPC dimensions overflow"))?,
            state_dim,
            state_dim,
            control_dim,
            control_dim,
            control_dim,
        ];
        let required = sizes
            .iter()
            .try_fold(0_usize, |sum, &size| sum.checked_add(size))
            .ok_or_else(|| PyValueError::new_err("MPC dimensions overflow"))?;
        if v.len() <= required {
            return Err(PyValueError::new_err("MPC dimensions are inconsistent"));
        }
        let mut cursor = 0;
        let parts: Vec<&[f64]> = sizes
            .iter()
            .map(|&n| {
                let part = &v[cursor..cursor + n];
                cursor += n;
                part
            })
            .collect();
        let (states, cost, gradient) = mpc_rollout(
            parts[0],
            parts[1],
            parts[2],
            parts[3],
            parts[4],
            parts[5],
            parts[6],
            parts[7],
            parts[8],
            &v[cursor..],
            state_dim,
            control_dim,
            compute_gradient,
        )
        .map_err(PyValueError::new_err)?;
        // The cost-only caller does not need state-history serialization.
        Ok((
            encode(py, if compute_gradient { &states } else { &[] })?,
            cost,
            encode(py, &gradient)?,
        ))
    }

    #[pyfunction]
    fn control_relative_mpc_cost_batch_packed<'py>(
        py: Python<'py>,
        data: &[u8],
        candidates: usize,
        horizon: usize,
        dt_s: f64,
        mu_km3_s2: f64,
        max_accel_km_s2: f64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let values = decode(data)?.into_owned();
        let controls = candidates.checked_mul(horizon).and_then(|n| n.checked_mul(3))
            .ok_or_else(|| PyValueError::new_err("relative MPC dimensions overflow"))?;
        let expected = 45_usize.checked_add(controls)
            .ok_or_else(|| PyValueError::new_err("relative MPC dimensions overflow"))?;
        if values.len() != expected {
            return Err(PyValueError::new_err("relative MPC packed input has invalid length"));
        }
        let costs = py.detach(|| relative_mpc_cost_batch(
            &values[..45], &values[45..], candidates, horizon, dt_s, mu_km3_s2, max_accel_km_s2,
        )).map_err(PyValueError::new_err)?;
        encode(py, &costs)
    }

    #[pyfunction]
    fn control_reaction_wheel_step_packed<'py>(
        py: Python<'py>,
        data: &[u8],
        wheels: usize,
        dt_s: f64,
        torque_tau_s: f64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let expected = wheels
            .checked_mul(12 * 8)
            .ok_or_else(|| PyValueError::new_err("wheel dimensions overflow"))?;
        if wheels == 0 || data.len() != expected {
            return Err(PyValueError::new_err("wheel dimensions are inconsistent"));
        }
        let v = decode(data)?;
        let g = &v[..3 * wheels];
        let mut cursor = 3 * wheels;
        let mut take = || {
            let result = &v[cursor..cursor + wheels];
            cursor += wheels;
            result
        };
        let maximum_torque = take();
        let maximum_momentum = take();
        let maximum_speed = take();
        let viscous = take();
        let coulomb = take();
        let inertia = take();
        let old_motor = take();
        let omega = take();
        let command = take();
        // Limits are prepared by the Python owner; reject malformed direct
        // callers before using clamp (which requires ordered, non-NaN bounds).
        if !dt_s.is_finite()
            || !torque_tau_s.is_finite()
            || maximum_torque
                .iter()
                .chain(maximum_momentum)
                .chain(maximum_speed)
                .any(|x| x.is_nan() || *x < 0.0)
            || inertia.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(PyValueError::new_err(
                "wheel limits and timestep are invalid",
            ));
        }
        let alpha = if dt_s <= 0.0 || torque_tau_s <= 0.0 {
            1.0
        } else {
            (dt_s / torque_tau_s).clamp(0.0, 1.0)
        };
        let mut motor = vec![0.0; wheels];
        let mut next_omega = vec![0.0; wheels];
        let mut momentum = vec![0.0; wheels];
        let mut realized = vec![0.0; wheels];
        for i in 0..wheels {
            let cmd = command[i].clamp(-maximum_torque[i], maximum_torque[i]);
            let lagged = if dt_s <= 0.0 || torque_tau_s <= 0.0 {
                cmd
            } else {
                old_motor[i] + alpha * (cmd - old_motor[i])
            };
            motor[i] = lagged.clamp(-maximum_torque[i], maximum_torque[i]);
            let sign = if omega[i] > 0.0 {
                1.0
            } else if omega[i] < 0.0 {
                -1.0
            } else if omega[i].is_nan() {
                f64::NAN
            } else {
                0.0
            };
            let friction = viscous[i] * omega[i] + coulomb[i] * sign;
            let mut net = motor[i] - friction;
            let old_momentum = inertia[i] * omega[i];
            if (old_momentum >= maximum_momentum[i] - 1e-12 && net > 0.0)
                || (old_momentum <= -maximum_momentum[i] + 1e-12 && net < 0.0)
            {
                net = 0.0;
                motor[i] = friction;
            }
            let proposed = omega[i] + dt_s * (net / inertia[i]);
            if (proposed >= maximum_speed[i] && net > 0.0)
                || (proposed <= -maximum_speed[i] && net < 0.0)
            {
                net = 0.0;
                motor[i] = friction;
            }
            next_omega[i] =
                (omega[i] + dt_s * (net / inertia[i])).clamp(-maximum_speed[i], maximum_speed[i]);
            momentum[i] =
                (inertia[i] * next_omega[i]).clamp(-maximum_momentum[i], maximum_momentum[i]);
            next_omega[i] = momentum[i] / inertia[i];
            realized[i] = if dt_s > 0.0 {
                (momentum[i] - old_momentum) / dt_s
            } else {
                net
            };
        }
        let body_momentum = mat_vec(g, 3, wheels, &momentum);
        let body_torque: Vec<f64> = mat_vec(g, 3, wheels, &realized)
            .iter()
            .map(|x| -*x)
            .collect();
        let output: Vec<f64> = motor
            .iter()
            .chain(&next_omega)
            .chain(&momentum)
            .chain(&realized)
            .chain(&body_momentum)
            .chain(&body_torque)
            .copied()
            .collect();
        encode(py, &output)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<RCSGeometryContext>()?;
        module.add_function(wrap_pyfunction!(
            control_reaction_wheel_step_packed,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(control_mat_vec_packed, module)?)?;
        module.add_function(wrap_pyfunction!(control_mpc_packed, module)?)?;
        module.add_function(wrap_pyfunction!(control_relative_mpc_cost_batch_packed, module)?)?;
        module.add_function(wrap_pyfunction!(control_project_sequence, module)?)?;
        module.add_function(wrap_pyfunction!(control_mat_vec_clip, module)?)?;
        module.add_function(wrap_pyfunction!(control_mat_vec, module)?)?;
        module.add_function(wrap_pyfunction!(control_mpc_rollout_cost_gradient, module)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;
