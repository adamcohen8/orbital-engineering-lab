//! OEL CR3BP in physical rotating barycentric km, km/s, and seconds.
//! Equations and the primary-distance floor match sim.dynamics.orbit.cr3bp.

use crate::adaptive::{integrate_dopri5, integrate_rkf78, StepInfo};

#[derive(Clone, Copy, Debug)]
pub struct System {
    pub distance_km: f64,
    pub mu: f64,
    pub mean_motion_rad_s: f64,
}

impl System {
    fn validate(self) -> Result<(), String> {
        if !self.distance_km.is_finite()
            || self.distance_km <= 0.0
            || !self.mean_motion_rad_s.is_finite()
            || self.mean_motion_rad_s <= 0.0
            || !self.mu.is_finite()
            || !(0.0..=1.0).contains(&self.mu)
        {
            return Err(
                "CR3BP system requires positive finite distance and mean motion, and mu in [0, 1]"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Geometry {
    x: f64,
    y: f64,
    z: f64,
    x1: f64,
    x2: f64,
    r1: f64,
    r2: f64,
    r1_3: f64,
    r2_3: f64,
}

fn geometry(state: [f64; 6], system: System) -> Geometry {
    let d = system.distance_km;
    let [x, y, z] = [state[0] / d, state[1] / d, state[2] / d];
    let x1 = x + system.mu;
    let x2 = x - (1.0 - system.mu);
    let r1 = (x1 * x1 + y * y + z * z).sqrt().max(1e-12);
    let r2 = (x2 * x2 + y * y + z * z).sqrt().max(1e-12);
    Geometry {
        x,
        y,
        z,
        x1,
        x2,
        r1,
        r2,
        r1_3: r1.powi(3),
        r2_3: r2.powi(3),
    }
}

pub fn derivative(state: [f64; 6], command: [f64; 3], system: System) -> [f64; 6] {
    derivative_with_geometry(state, command, system, geometry(state, system))
}

fn derivative_with_geometry(
    state: [f64; 6],
    command: [f64; 3],
    system: System,
    g: Geometry,
) -> [f64; 6] {
    let d = system.distance_km;
    let n = system.mean_motion_rad_s;
    let vx = state[3] / (d * n);
    let vy = state[4] / (d * n);
    let mu2 = system.mu;
    let mu1 = 1.0 - mu2;
    let Geometry {
        x,
        y,
        z,
        x1,
        x2,
        r1_3,
        r2_3,
        ..
    } = g;
    let ox = x - mu1 * x1 / r1_3 - mu2 * x2 / r2_3;
    let oy = y - mu1 * y / r1_3 - mu2 * y / r2_3;
    let oz = -mu1 * z / r1_3 - mu2 * z / r2_3;
    let scale = d * n.powi(2);
    [
        state[3],
        state[4],
        state[5],
        (2.0 * vy + ox) * scale + command[0],
        (-2.0 * vx + oy) * scale + command[1],
        oz * scale + command[2],
    ]
}

pub fn jacobian(state: [f64; 6], system: System) -> [[f64; 6]; 6] {
    jacobian_with_geometry(system, geometry(state, system))
}

fn jacobian_with_geometry(system: System, g: Geometry) -> [[f64; 6]; 6] {
    let n = system.mean_motion_rad_s;
    let mu2 = system.mu;
    let mu1 = 1.0 - mu2;
    let Geometry {
        y,
        z,
        x1,
        x2,
        r1,
        r2,
        r1_3,
        r2_3,
        ..
    } = g;
    let (r1_5, r2_5) = (r1.powi(5), r2.powi(5));
    let xx =
        1.0 - mu1 * (1.0 / r1_3 - 3.0 * x1 * x1 / r1_5) - mu2 * (1.0 / r2_3 - 3.0 * x2 * x2 / r2_5);
    let yy =
        1.0 - mu1 * (1.0 / r1_3 - 3.0 * y * y / r1_5) - mu2 * (1.0 / r2_3 - 3.0 * y * y / r2_5);
    let zz = -mu1 * (1.0 / r1_3 - 3.0 * z * z / r1_5) - mu2 * (1.0 / r2_3 - 3.0 * z * z / r2_5);
    let xy = 3.0 * mu1 * x1 * y / r1_5 + 3.0 * mu2 * x2 * y / r2_5;
    let xz = 3.0 * mu1 * x1 * z / r1_5 + 3.0 * mu2 * x2 * z / r2_5;
    let yz = 3.0 * mu1 * y * z / r1_5 + 3.0 * mu2 * y * z / r2_5;
    let hessian = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
    let mut a = [[0.0; 6]; 6];
    for row in 0..3 {
        a[row][row + 3] = 1.0;
        for col in 0..3 {
            a[row + 3][col] = n.powi(2) * hessian[row][col];
        }
    }
    a[3][4] = 2.0 * n;
    a[4][3] = -2.0 * n;
    a
}

fn stm_derivative(state: [f64; 42], system: System) -> [f64; 42] {
    let reference = std::array::from_fn(|axis| state[axis]);
    let g = geometry(reference, system);
    let rate = derivative_with_geometry(reference, [0.0; 3], system, g);
    let a = jacobian_with_geometry(system, g);
    let mut out = [0.0; 42];
    out[..6].copy_from_slice(&rate);
    // The first three Jacobian rows are [0, I]. Preserve the original
    // zero-initialized summation, including the signed-zero convention.
    for row in 0..3 {
        for col in 0..6 {
            out[6 + row * 6 + col] = 0.0 + state[6 + (row + 3) * 6 + col];
        }
    }
    for row in 3..6 {
        for col in 0..6 {
            let mut value = 0.0;
            for k in 0..3 {
                value += a[row][k] * state[6 + k * 6 + col];
            }
            if row == 3 {
                value += a[row][4] * state[6 + 4 * 6 + col];
            } else if row == 4 {
                value += a[row][3] * state[6 + 3 * 6 + col];
            }
            out[6 + row * 6 + col] = value;
        }
    }
    out
}

#[derive(Clone, Copy)]
pub struct Integration {
    pub t_s: f64,
    pub dt_s: f64,
    pub atol: f64,
    pub rtol: f64,
    pub h_init: Option<f64>,
}

fn integrate<const N: usize, F>(
    initial: [f64; N],
    opts: Integration,
    method: &str,
    mut deriv: F,
) -> Result<([f64; N], Option<StepInfo>), String>
where
    F: FnMut(f64, [f64; N]) -> Result<[f64; N], String>,
{
    if initial.iter().any(|v| !v.is_finite()) {
        return Err("CR3BP state and STM must contain only finite values".into());
    }
    if !opts.t_s.is_finite() || !opts.dt_s.is_finite() {
        return Err("CR3BP times must be finite".into());
    }
    let result = match method {
        "rk4" => {
            let h = opts.dt_s;
            let stage =
                |k: [f64; N], scale: f64| std::array::from_fn(|i| initial[i] + scale * k[i]);
            let k1 = deriv(opts.t_s, initial)?;
            let k2 = deriv(opts.t_s + 0.5 * h, stage(k1, 0.5 * h))?;
            let k3 = deriv(opts.t_s + 0.5 * h, stage(k2, 0.5 * h))?;
            let k4 = deriv(opts.t_s + h, stage(k3, h))?;
            (
                std::array::from_fn(|i| {
                    initial[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
                }),
                None,
            )
        }
        "rkf78" | "adaptive" | "dopri5" => {
            if !opts.atol.is_finite()
                || opts.atol <= 0.0
                || !opts.rtol.is_finite()
                || opts.rtol <= 0.0
            {
                return Err("CR3BP adaptive tolerances must be positive and finite.".into());
            }
            let (state, info) = if method == "dopri5" {
                integrate_dopri5(
                    initial,
                    opts.t_s,
                    opts.dt_s,
                    opts.atol,
                    opts.rtol,
                    opts.h_init,
                    deriv,
                )?
            } else {
                integrate_rkf78(
                    initial,
                    opts.t_s,
                    opts.dt_s,
                    opts.atol,
                    opts.rtol,
                    opts.h_init,
                    deriv,
                )?
            };
            (state, Some(info))
        }
        _ => return Err(format!("Unsupported CR3BP integrator '{method}'.")),
    };
    if result.0.iter().any(|v| !v.is_finite()) {
        return Err("CR3BP propagation produced non-finite values".into());
    }
    Ok(result)
}

pub fn propagate(
    state: [f64; 6],
    command: [f64; 3],
    system: System,
    opts: Integration,
    method: &str,
) -> Result<([f64; 6], Option<StepInfo>), String> {
    system.validate()?;
    if command.iter().any(|v| !v.is_finite()) {
        return Err("CR3BP command acceleration must be finite".into());
    }
    integrate(state, opts, method, |_, x| {
        let out = derivative(x, command, system);
        if out.iter().any(|v| !v.is_finite()) {
            return Err("CR3BP derivative produced non-finite values".into());
        }
        Ok(out)
    })
}

pub fn propagate_stm(
    state: [f64; 42],
    system: System,
    opts: Integration,
    method: &str,
) -> Result<([f64; 42], Option<StepInfo>), String> {
    system.validate()?;
    integrate(state, opts, method, |_, x| {
        let out = stm_derivative(x, system);
        if out.iter().any(|v| !v.is_finite()) {
            return Err("CR3BP derivative produced non-finite values".into());
        }
        Ok(out)
    })
}

/// Save passive RK4 rows at caller-owned step boundaries. This is exactly the
/// scalar propagation sequence; event and command lifecycle remains Python's.
pub fn sampled_history(
    initial: [f64; 6],
    widths: &[f64],
    sample_ends: &[usize],
    system: System,
) -> Result<Vec<f64>, String> {
    system.validate()?;
    if initial.iter().any(|value| !value.is_finite()) {
        return Err("CR3BP state must contain only finite values".into());
    }
    if widths
        .iter()
        .any(|width| !width.is_finite() || *width <= 0.0)
    {
        return Err("CR3BP step widths must be finite and positive".into());
    }
    if sample_ends.first().is_some_and(|index| *index == 0)
        || sample_ends.windows(2).any(|pair| pair[0] >= pair[1])
        || sample_ends
            .last()
            .is_some_and(|index| *index > widths.len())
    {
        return Err("sample ends must be strictly increasing within the step sequence".into());
    }
    let capacity = sample_ends
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_mul(6))
        .ok_or("CR3BP history size overflow")?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(capacity)
        .map_err(|_| "unable to allocate CR3BP history")?;
    rows.extend_from_slice(&initial);
    let mut current = initial;
    let mut sample = 0;
    let mut time = 0.0;
    for (index, &width) in widths.iter().enumerate() {
        let opts = Integration {
            t_s: time,
            dt_s: width,
            atol: 1e-9,
            rtol: 1e-7,
            h_init: None,
        };
        current = propagate(current, [0.0; 3], system, opts, "rk4")?.0;
        time += width;
        if sample < sample_ends.len() && sample_ends[sample] == index + 1 {
            rows.extend_from_slice(&current);
            sample += 1;
        }
    }
    Ok(rows)
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::{
        exceptions::{PyRuntimeError, PyValueError},
        prelude::*,
    };
    type Info = (
        usize,
        usize,
        usize,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        f64,
    );
    fn info(value: Option<StepInfo>) -> Option<Info> {
        value.map(|i| {
            (
                i.accepted_steps,
                i.rejected_steps,
                i.attempted_steps,
                i.min_step_s,
                i.max_step_s,
                i.final_step_s,
                i.suggested_next_step_s,
                i.max_error_ratio,
            )
        })
    }
    fn error(message: String) -> PyErr {
        if message.starts_with("Adaptive integrator") {
            PyRuntimeError::new_err(message)
        } else {
            PyValueError::new_err(message)
        }
    }
    fn fixed<const N: usize>(values: Vec<f64>, name: &str) -> PyResult<[f64; N]> {
        let values: [f64; N] = values
            .try_into()
            .map_err(|_| PyValueError::new_err(format!("{name} must have {N} values")))?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "{name} must contain only finite values"
            )));
        }
        Ok(values)
    }
    #[pyfunction]
    #[pyo3(signature = (state, t_s, dt_s, distance_km, mu, mean_motion_rad_s, command, method="rk4", atol=1e-9, rtol=1e-7, h_init=None))]
    fn cr3bp_propagate(
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        distance_km: f64,
        mu: f64,
        mean_motion_rad_s: f64,
        command: Vec<f64>,
        method: &str,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
    ) -> PyResult<(Vec<f64>, Option<Info>)> {
        let x = fixed(state, "CR3BP state")?;
        let command = fixed(command, "CR3BP command")?;
        let system = System {
            distance_km,
            mu,
            mean_motion_rad_s,
        };
        let opts = Integration {
            t_s,
            dt_s,
            atol,
            rtol,
            h_init,
        };
        let (out, details) = py
            .detach(|| propagate(x, command, system, opts, method))
            .map_err(error)?;
        Ok((out.to_vec(), info(details)))
    }
    #[pyfunction]
    #[pyo3(signature = (state, t_s, dt_s, distance_km, mu, mean_motion_rad_s, method="rk4", atol=1e-9, rtol=1e-7, h_init=None))]
    fn cr3bp_propagate_stm(
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        distance_km: f64,
        mu: f64,
        mean_motion_rad_s: f64,
        method: &str,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
    ) -> PyResult<(Vec<f64>, Option<Info>)> {
        let x = fixed(state, "CR3BP augmented state")?;
        let system = System {
            distance_km,
            mu,
            mean_motion_rad_s,
        };
        let opts = Integration {
            t_s,
            dt_s,
            atol,
            rtol,
            h_init,
        };
        let (out, details) = py
            .detach(|| propagate_stm(x, system, opts, method))
            .map_err(error)?;
        Ok((out.to_vec(), info(details)))
    }
    #[pyfunction]
    #[pyo3(signature = (state, t_s, dt_s, distance_km, mu, mean_motion_rad_s, method="rk4", atol=1e-9, rtol=1e-7, h_init=None, diagnostics=true))]
    fn cr3bp_propagate_stm_buffer(
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        distance_km: f64,
        mu: f64,
        mean_motion_rad_s: f64,
        method: &str,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
        diagnostics: bool,
    ) -> PyResult<(Py<pyo3::types::PyByteArray>, Option<Info>)> {
        let x = fixed(state, "CR3BP augmented state")?;
        let system = System {
            distance_km,
            mu,
            mean_motion_rad_s,
        };
        let opts = Integration {
            t_s,
            dt_s,
            atol,
            rtol,
            h_init,
        };
        let (out, details) = py
            .detach(|| propagate_stm(x, system, opts, method))
            .map_err(error)?;
        #[cfg(target_endian = "little")]
        let bytes = unsafe {
            // The owned output array is copied into fresh Python storage.
            std::slice::from_raw_parts(out.as_ptr().cast(), std::mem::size_of_val(&out))
        };
        #[cfg(target_endian = "big")]
        let packed: Vec<u8> = out.iter().flat_map(|value| value.to_le_bytes()).collect();
        #[cfg(target_endian = "big")]
        let bytes = packed.as_slice();
        let metadata = if diagnostics { info(details) } else { None };
        Ok((pyo3::types::PyByteArray::new(py, bytes).unbind(), metadata))
    }
    #[pyfunction]
    fn cr3bp_derivative(
        state: Vec<f64>,
        distance_km: f64,
        mu: f64,
        mean_motion_rad_s: f64,
        command: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let system = System {
            distance_km,
            mu,
            mean_motion_rad_s,
        };
        system.validate().map_err(error)?;
        Ok(derivative(
            fixed(state, "CR3BP state")?,
            fixed(command, "CR3BP command")?,
            system,
        )
        .to_vec())
    }
    #[pyfunction]
    fn cr3bp_jacobian(
        state: Vec<f64>,
        distance_km: f64,
        mu: f64,
        mean_motion_rad_s: f64,
    ) -> PyResult<Vec<f64>> {
        let system = System {
            distance_km,
            mu,
            mean_motion_rad_s,
        };
        system.validate().map_err(error)?;
        Ok(jacobian(fixed(state, "CR3BP state")?, system)
            .into_iter()
            .flatten()
            .collect())
    }
    #[pyfunction]
    fn cr3bp_sampled_history_bytes<'py>(
        py: Python<'py>,
        state: Vec<f64>,
        step_widths: &[u8],
        sample_ends: Vec<usize>,
        distance_km: f64,
        mu: f64,
        mean_motion_rad_s: f64,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let initial = fixed(state, "CR3BP state")?;
        if step_widths.len() % 8 != 0 {
            return Err(PyValueError::new_err(
                "step width bytes must contain float64 values",
            ));
        }
        let widths: Vec<f64> = step_widths
            .chunks_exact(8)
            .map(|value| f64::from_le_bytes(value.try_into().unwrap()))
            .collect();
        let system = System {
            distance_km,
            mu,
            mean_motion_rad_s,
        };
        let rows = py
            .detach(|| sampled_history(initial, &widths, &sample_ends, system))
            .map_err(error)?;
        #[cfg(target_endian = "little")]
        let bytes = unsafe {
            // Byte view of the owned contiguous row vector; copied into PyBytes.
            std::slice::from_raw_parts(rows.as_ptr().cast(), std::mem::size_of_val(rows.as_slice()))
        };
        #[cfg(target_endian = "big")]
        let packed: Vec<u8> = rows.iter().flat_map(|value| value.to_le_bytes()).collect();
        #[cfg(target_endian = "big")]
        let bytes = packed.as_slice();
        Ok(pyo3::types::PyBytes::new(py, bytes))
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(cr3bp_sampled_history_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(cr3bp_propagate, module)?)?;
        module.add_function(wrap_pyfunction!(cr3bp_propagate_stm, module)?)?;
        module.add_function(wrap_pyfunction!(cr3bp_propagate_stm_buffer, module)?)?;
        module.add_function(wrap_pyfunction!(cr3bp_derivative, module)?)?;
        module.add_function(wrap_pyfunction!(cr3bp_jacobian, module)?)?;
        Ok(())
    }
}
#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;
    fn system() -> System {
        System {
            distance_km: 384400.0,
            mu: 0.012150585609624,
            mean_motion_rad_s: 2.6616995e-6,
        }
    }
    fn opts(dt_s: f64) -> Integration {
        Integration {
            t_s: 0.0,
            dt_s,
            atol: 1e-12,
            rtol: 1e-11,
            h_init: None,
        }
    }
    #[test]
    fn sampled_history_matches_scalar_nonuniform_steps() {
        let initial = [320000.0, 10.0, 30000.0, 0.0, 0.2, 0.0];
        let widths = [0.3, 0.3, 0.1, 30.0, 0.05];
        let rows = sampled_history(initial, &widths, &[3, 5], system()).unwrap();
        let mut state = initial;
        for (index, width) in widths.into_iter().enumerate() {
            state = propagate(state, [0.0; 3], system(), opts(width), "rk4")
                .unwrap()
                .0;
            if index == 2 {
                assert_eq!(&rows[6..12], &state);
            }
        }
        assert_eq!(&rows[12..18], &state);
        assert!(sampled_history(initial, &[-1.0], &[1], system()).is_err());
        assert!(sampled_history(initial, &widths, &[2, 2], system()).is_err());
    }

    #[test]
    fn sparse_stm_rate_matches_dense_arithmetic() {
        let state = std::array::from_fn(|index| {
            if index < 6 {
                [320000.0, 10.0, 30000.0, 0.0, 0.2, 0.0][index]
            } else if index % 5 == 0 {
                -0.0
            } else {
                (index as f64 - 27.0) * 0.125
            }
        });
        // Keep these runtime inputs out of constant folding: LLVM/libm's
        // compile-time transcendental path can differ by a few ulps.
        let state = std::hint::black_box(state);
        let sys = std::hint::black_box(system());
        let reference = std::array::from_fn(|axis| state[axis]);
        let rate = derivative(reference, [0.0; 3], sys);
        let a = jacobian(reference, sys);
        let mut expected = [0.0; 42];
        expected[..6].copy_from_slice(&rate);
        for row in 0..6 {
            for col in 0..6 {
                for k in 0..6 {
                    expected[6 + row * 6 + col] += a[row][k] * state[6 + k * 6 + col];
                }
            }
        }
        let actual = stm_derivative(state, sys);
        assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
    }

    #[test]
    fn origin_is_valid_rotating_barycentric_state() {
        assert!(propagate([0.0; 6], [0.0; 3], system(), opts(0.01), "rk4")
            .unwrap()
            .0
            .iter()
            .all(|v| v.is_finite()));
    }
    #[test]
    fn stm_zero_duration_preserves_nonidentity_matrix() {
        let x = std::array::from_fn(|i| {
            if i < 6 {
                [320000.0, 0.0, 30000.0, 0.0, 0.2, 0.0][i]
            } else {
                (i - 6) as f64
            }
        });
        for method in ["rk4", "rkf78", "dopri5"] {
            let (out, details) = propagate_stm(x, system(), opts(0.0), method).unwrap();
            assert_eq!(out, x);
            assert_eq!(details.is_none(), method == "rk4");
        }
    }
    #[test]
    fn bad_system_and_state_fail_closed() {
        let mut bad = system();
        bad.distance_km = 0.0;
        assert!(propagate([0.0; 6], [0.0; 3], bad, opts(1.0), "rk4").is_err());
        assert!(propagate([f64::NAN; 6], [0.0; 3], system(), opts(1.0), "rk4").is_err());
    }
}
