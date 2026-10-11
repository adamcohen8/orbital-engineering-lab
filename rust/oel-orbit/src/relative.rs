//! Relative-orbit kernels used by the opt-in Rust numerical backend.
//!
//! The public OEL state convention is rectangular RIC, deputy relative to
//! chief, in km and km/s.  The Rust entry points mirror the reference Python
//! implementations in `sim.estimation.relative_th_ekf` and
//! `sim.dynamics.orbit.relative_linear`; they do not replace the Python force
//! model or frame contract.  In particular, the nonlinear ONP route remains
//! authoritative for configured force plugins.

use std::f64::consts::PI;

pub type Mat6 = [f64; 36];

fn validate_finite(values: &[f64], name: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn mat6_identity() -> Mat6 {
    let mut out = [0.0; 36];
    for index in 0..6 {
        out[index * 6 + index] = 1.0;
    }
    out
}

fn mat6_add_scaled(target: &mut Mat6, source: &Mat6, scale: f64) {
    for index in 0..36 {
        target[index] += scale * source[index];
    }
}

fn mat6_scale(source: &Mat6, scale: f64) -> Mat6 {
    std::array::from_fn(|index| source[index] * scale)
}

fn mat6_mul(left: &Mat6, right: &Mat6) -> Mat6 {
    let mut out = [0.0; 36];
    for row in 0..6 {
        for inner in 0..6 {
            let value = left[row * 6 + inner];
            if value == 0.0 {
                continue;
            }
            for column in 0..6 {
                out[row * 6 + column] += value * right[inner * 6 + column];
            }
        }
    }
    out
}

fn mat6_vec(left: &Mat6, right: &[f64; 6]) -> [f64; 6] {
    std::array::from_fn(|row| {
        (0..6)
            .map(|column| left[row * 6 + column] * right[column])
            .sum()
    })
}

fn mat6_norm1(value: &Mat6) -> f64 {
    (0..6)
        .map(|column| (0..6).map(|row| value[row * 6 + column].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Exponentiate a 6x6 matrix with scaling and a converged Taylor series.
///
/// The relative SS model is a constant-coefficient linear system.  This
/// implementation avoids introducing a second linear-algebra dependency into
/// the small extension while retaining the accuracy needed by the Python
/// reference's `scipy.linalg.expm` path.  Scaling keeps the series argument
/// below 1/2 and repeated squaring restores the requested interval.
fn mat6_expm(value: &Mat6) -> Result<Mat6, String> {
    if value.iter().any(|entry| !entry.is_finite()) {
        return Err("relative system matrix must contain only finite values".into());
    }
    let norm = mat6_norm1(value);
    if !norm.is_finite() {
        return Err("relative system matrix norm is non-finite".into());
    }
    // A useful OEL propagation interval never needs more than a few dozen
    // squarings. Reject pathological finite inputs before a float-to-integer
    // conversion could request an unbounded loop or overflow the scale.
    const MAX_SCALE_POWER: f64 = 60.0;
    let scale_power_f = if norm <= 0.5 { 0.0 } else { norm.log2().ceil() };
    if !scale_power_f.is_finite() || scale_power_f > MAX_SCALE_POWER {
        return Err("relative system matrix norm is too large for matrix exponential".into());
    }
    let scale_power = scale_power_f.max(0.0) as usize;
    let scale = 2.0_f64.powi(scale_power as i32);
    let scaled = mat6_scale(value, 1.0 / scale);
    let mut result = mat6_identity();
    let mut term = mat6_identity();
    for order in 1..=48 {
        term = mat6_mul(&term, &scaled);
        let factor = 1.0 / order as f64;
        term = mat6_scale(&term, factor);
        mat6_add_scaled(&mut result, &term, 1.0);
        if order >= 18 && term.iter().fold(0.0_f64, |max, v| max.max(v.abs())) < 1.0e-17 {
            break;
        }
    }
    for _ in 0..scale_power {
        result = mat6_mul(&result, &result);
    }
    if result.iter().any(|entry| !entry.is_finite()) {
        return Err("relative matrix exponential produced non-finite values".into());
    }
    Ok(result)
}

fn validate_common(mean_motion: f64, dt_s: f64) -> Result<(), String> {
    if !mean_motion.is_finite() || mean_motion <= 0.0 {
        return Err("mean_motion_rad_s must be finite and positive".into());
    }
    if !dt_s.is_finite() || dt_s < 0.0 {
        return Err("dt_s must be finite and nonnegative".into());
    }
    Ok(())
}

pub fn hcw_stm(mean_motion: f64, dt_s: f64) -> Result<Mat6, String> {
    validate_common(mean_motion, dt_s)?;
    let n = mean_motion;
    let nt = n * dt_s;
    let cosine = nt.cos();
    let sine = nt.sin();
    Ok([
        4.0 - 3.0 * cosine,
        0.0,
        0.0,
        sine / n,
        2.0 * (1.0 - cosine) / n,
        0.0,
        6.0 * (sine - nt),
        1.0,
        0.0,
        -2.0 * (1.0 - cosine) / n,
        (4.0 * sine - 3.0 * nt) / n,
        0.0,
        0.0,
        0.0,
        cosine,
        0.0,
        0.0,
        sine / n,
        3.0 * n * sine,
        0.0,
        0.0,
        cosine,
        2.0 * sine,
        0.0,
        -6.0 * n * (1.0 - cosine),
        0.0,
        0.0,
        -2.0 * sine,
        4.0 * cosine - 3.0,
        0.0,
        0.0,
        0.0,
        -n * sine,
        0.0,
        0.0,
        cosine,
    ])
}

fn ss_parameters(
    mean_motion: f64,
    reference_radius_km: f64,
    reference_inclination_rad: f64,
    j2: f64,
    earth_radius_km: f64,
) -> Result<(f64, f64), String> {
    validate_common(mean_motion, 0.0)?;
    if !reference_radius_km.is_finite()
        || reference_radius_km <= earth_radius_km
        || !reference_inclination_rad.is_finite()
        || !(0.0..=PI).contains(&reference_inclination_rad)
        || !j2.is_finite()
        || j2 < 0.0
        || !earth_radius_km.is_finite()
        || earth_radius_km <= 0.0
    {
        return Err("invalid SS-J2 reference orbit parameters".into());
    }
    let ratio_sq = (earth_radius_km / reference_radius_km).powi(2);
    let s = 3.0 * j2 * ratio_sq * (1.0 + 3.0 * (2.0 * reference_inclination_rad).cos()) / 8.0;
    if 1.0 + s <= 0.0 {
        return Err("SS-J2 parameters produce a non-real mean-motion correction".into());
    }
    let c = (1.0 + s).sqrt();
    let q = mean_motion * c
        + 1.5 * j2 * mean_motion * ratio_sq * reference_inclination_rad.cos().powi(2);
    Ok((c, q))
}

pub fn ss_j2_stm(
    mean_motion: f64,
    reference_radius_km: f64,
    reference_inclination_rad: f64,
    j2: f64,
    earth_radius_km: f64,
    dt_s: f64,
) -> Result<Mat6, String> {
    validate_common(mean_motion, dt_s)?;
    let (c, q) = ss_parameters(
        mean_motion,
        reference_radius_km,
        reference_inclination_rad,
        j2,
        earth_radius_km,
    )?;
    let n = mean_motion;
    let mut system = [0.0; 36];
    system[0 * 6 + 3] = 1.0;
    system[1 * 6 + 4] = 1.0;
    system[2 * 6 + 5] = 1.0;
    system[3 * 6] = (5.0 * c * c - 2.0) * n * n;
    system[3 * 6 + 4] = 2.0 * n * c;
    system[4 * 6 + 3] = -2.0 * n * c;
    system[5 * 6 + 2] = -q * q;
    mat6_expm(&mat6_scale(&system, dt_s))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn th_system_matrix(chief_r: [f64; 3], chief_v: [f64; 3], mu: f64) -> Mat6 {
    let radius = norm(chief_r);
    if !radius.is_finite() || radius <= 1.0e-9 || !mu.is_finite() || mu <= 0.0 {
        return [0.0; 36];
    }
    let h_norm = norm(cross(chief_r, chief_v));
    let theta_dot = h_norm / (radius * radius).max(1.0e-12);
    let radial_rate = dot(chief_r, chief_v) / radius;
    let theta_ddot = -2.0 * theta_dot * radial_rate / radius;
    let gravity = mu / radius.powi(3);
    let mut a = [0.0; 36];
    for axis in 0..3 {
        a[axis * 6 + axis + 3] = 1.0;
    }
    a[3 * 6] = 2.0 * gravity + theta_dot * theta_dot;
    a[3 * 6 + 1] = theta_ddot;
    a[3 * 6 + 4] = 2.0 * theta_dot;
    a[4 * 6] = -theta_ddot;
    a[4 * 6 + 1] = -gravity + theta_dot * theta_dot;
    a[4 * 6 + 3] = -2.0 * theta_dot;
    a[5 * 6 + 2] = -gravity;
    a
}

fn two_body_derivative(state: [f64; 6], mu: f64) -> [f64; 6] {
    let r = [state[0], state[1], state[2]];
    let radius = norm(r);
    if !radius.is_finite() || radius <= 1.0e-9 || !mu.is_finite() || mu <= 0.0 {
        return [0.0; 6];
    }
    let scale = -mu / radius.powi(3);
    [
        state[3],
        state[4],
        state[5],
        scale * r[0],
        scale * r[1],
        scale * r[2],
    ]
}

fn th_combined_derivative(state: [f64; 12], mu: f64) -> [f64; 12] {
    let chief_r = [state[0], state[1], state[2]];
    let chief_v = [state[3], state[4], state[5]];
    let rho = [state[6], state[7], state[8]];
    let rho_dot = [state[9], state[10], state[11]];
    let radius = norm(chief_r);
    if !radius.is_finite() || radius <= 1.0e-9 || !mu.is_finite() || mu <= 0.0 {
        return [0.0; 12];
    }
    let h_norm = norm(cross(chief_r, chief_v));
    let theta_dot = h_norm / (radius * radius).max(1.0e-12);
    let radial_rate = dot(chief_r, chief_v) / radius;
    let theta_ddot = -2.0 * theta_dot * radial_rate / radius;
    let gravity_gradient = [
        2.0 * mu * rho[0] / radius.powi(3),
        -mu * rho[1] / radius.powi(3),
        -mu * rho[2] / radius.powi(3),
    ];
    let coriolis = [-theta_dot * rho_dot[1], theta_dot * rho_dot[0], 0.0];
    let euler = [-theta_ddot * rho[1], theta_ddot * rho[0], 0.0];
    let omega_rho = [-theta_dot * rho[1], theta_dot * rho[0], 0.0];
    let centrifugal = [-theta_dot * omega_rho[1], theta_dot * omega_rho[0], 0.0];
    let chief_acc = two_body_derivative(
        [
            chief_r[0], chief_r[1], chief_r[2], chief_v[0], chief_v[1], chief_v[2],
        ],
        mu,
    );
    [
        chief_v[0],
        chief_v[1],
        chief_v[2],
        chief_acc[3],
        chief_acc[4],
        chief_acc[5],
        rho_dot[0],
        rho_dot[1],
        rho_dot[2],
        gravity_gradient[0] - 2.0 * coriolis[0] - euler[0] - centrifugal[0],
        gravity_gradient[1] - 2.0 * coriolis[1] - euler[1] - centrifugal[1],
        gravity_gradient[2] - 2.0 * coriolis[2] - euler[2] - centrifugal[2],
    ]
}

fn rk4_step<const N: usize, F>(state: [f64; N], step_s: f64, mut derivative: F) -> [f64; N]
where
    F: FnMut([f64; N]) -> [f64; N],
{
    let k1 = derivative(state);
    let k2 = derivative(std::array::from_fn(|i| state[i] + 0.5 * step_s * k1[i]));
    let k3 = derivative(std::array::from_fn(|i| state[i] + 0.5 * step_s * k2[i]));
    let k4 = derivative(std::array::from_fn(|i| state[i] + step_s * k3[i]));
    std::array::from_fn(|i| state[i] + step_s / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

fn integrate<const N: usize, F>(
    state: [f64; N],
    dt_s: f64,
    max_step_s: f64,
    mut derivative: F,
) -> [f64; N]
where
    F: FnMut([f64; N]) -> [f64; N],
{
    let duration = dt_s.abs();
    if duration <= 0.0 {
        return state;
    }
    let direction = if dt_s >= 0.0 { 1.0 } else { -1.0 };
    let step_limit = max_step_s.max(1.0e-9);
    let mut elapsed = 0.0;
    let mut out = state;
    while elapsed < duration - 1.0e-12 {
        let step = (duration - elapsed).min(step_limit) * direction;
        out = rk4_step(out, step, &mut derivative);
        elapsed += step.abs();
    }
    out
}

/// Propagate the two-body chief reference with the relative models' existing
/// signed RK4 schedule, including the caller's partial final step.
pub fn chief_propagate(
    chief_state: [f64; 6],
    dt_s: f64,
    mu: f64,
    max_step_s: f64,
) -> Result<[f64; 6], String> {
    validate_finite(&chief_state, "chief_state_eci_km_s")?;
    if !dt_s.is_finite()
        || !mu.is_finite()
        || mu <= 0.0
        || !max_step_s.is_finite()
        || max_step_s <= 0.0
    {
        return Err(
            "chief propagation requires finite dt, positive max_step_s, and positive mu".into(),
        );
    }
    if (dt_s.abs() / max_step_s.max(1.0e-9)).ceil() > 10_000_000.0 {
        return Err("chief propagation exceeds the bounded RK4 step budget".into());
    }
    Ok(integrate(chief_state, dt_s, max_step_s, |state| {
        let r = [state[0], state[1], state[2]];
        let radius = norm(r);
        if !radius.is_finite() || radius <= 1.0e-9 {
            return [0.0; 6];
        }
        // Preserve the Python chief reference's multiply-then-divide order.
        // Its radius is a Python float, whose power uses floating-point pow.
        let radius_cubed = radius.powf(3.0);
        let negative_mu = -mu;
        [
            state[3],
            state[4],
            state[5],
            (negative_mu * r[0]) / radius_cubed,
            (negative_mu * r[1]) / radius_cubed,
            (negative_mu * r[2]) / radius_cubed,
        ]
    }))
}

pub fn th_propagate(
    relative_state: [f64; 6],
    dt_s: f64,
    chief_state: [f64; 6],
    mu: f64,
    max_step_s: f64,
) -> Result<[f64; 6], String> {
    validate_finite(&relative_state, "relative_state_ric")?;
    validate_finite(&chief_state, "chief_state_eci_km_s")?;
    if !dt_s.is_finite()
        || !max_step_s.is_finite()
        || max_step_s <= 0.0
        || !mu.is_finite()
        || mu <= 0.0
    {
        return Err(
            "TH propagation requires finite dt, positive max_step_s, and positive mu".into(),
        );
    }
    let initial = std::array::from_fn(|index| {
        if index < 6 {
            chief_state[index]
        } else {
            relative_state[index - 6]
        }
    });
    let output = integrate(initial, dt_s, max_step_s, |value| {
        th_combined_derivative(value, mu)
    });
    if output.iter().any(|value| !value.is_finite()) {
        return Err("TH propagation produced non-finite values".into());
    }
    Ok(std::array::from_fn(|index| output[index + 6]))
}

pub fn th_variational(
    relative_state: [f64; 6],
    dt_s: f64,
    chief_state: [f64; 6],
    mu: f64,
    max_step_s: f64,
) -> Result<([f64; 6], Mat6), String> {
    validate_finite(&relative_state, "relative_state_ric")?;
    validate_finite(&chief_state, "chief_state_eci_km_s")?;
    if !dt_s.is_finite()
        || !max_step_s.is_finite()
        || max_step_s <= 0.0
        || !mu.is_finite()
        || mu <= 0.0
    {
        return Err(
            "TH variational propagation requires finite dt, positive max_step_s, and positive mu"
                .into(),
        );
    }
    let mut initial = [0.0; 48];
    initial[..6].copy_from_slice(&chief_state);
    initial[6..12].copy_from_slice(&relative_state);
    initial[12..48].copy_from_slice(&mat6_identity());
    let output = integrate(initial, dt_s, max_step_s, |value| {
        let base = std::array::from_fn(|index| value[index]);
        let base_rate = th_combined_derivative(base, mu);
        let chief_r = [value[0], value[1], value[2]];
        let chief_v = [value[3], value[4], value[5]];
        let phi: Mat6 = std::array::from_fn(|index| value[12 + index]);
        let a = th_system_matrix(chief_r, chief_v, mu);
        // TH has ten nonzero entries. Preserve their accumulation order
        // while avoiding the dense matrix scan at every RK stage.
        let mut phi_rate = [0.0; 36];
        if a[3] != 0.0 {
            for column in 0..6 {
                phi_rate[column] = 0.0 + phi[18 + column];
                phi_rate[6 + column] = 0.0 + phi[24 + column];
                phi_rate[12 + column] = 0.0 + phi[30 + column];
                phi_rate[18 + column] = (0.0 + a[18] * phi[column])
                    + a[19] * phi[6 + column]
                    + a[22] * phi[24 + column];
                phi_rate[24 + column] = (0.0 + a[24] * phi[column])
                    + a[25] * phi[6 + column]
                    + a[27] * phi[18 + column];
                phi_rate[30 + column] = 0.0 + a[32] * phi[12 + column];
            }
        }
        let mut out = [0.0; 48];
        out[..12].copy_from_slice(&base_rate);
        out[12..48].copy_from_slice(&phi_rate);
        out
    });
    if output.iter().any(|value| !value.is_finite()) {
        return Err("TH variational propagation produced non-finite values".into());
    }
    Ok((
        std::array::from_fn(|index| output[6 + index]),
        std::array::from_fn(|index| output[12 + index]),
    ))
}

pub fn th_finite_difference(
    relative_state: [f64; 6],
    dt_s: f64,
    chief_state: [f64; 6],
    mu: f64,
    max_step_s: f64,
) -> Result<Mat6, String> {
    let base = th_propagate(relative_state, dt_s, chief_state, mu, max_step_s)?;
    let steps = [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-8, 1.0e-8, 1.0e-8];
    let mut phi = [0.0; 36];
    for axis in 0..6 {
        let mut shifted = relative_state;
        shifted[axis] += steps[axis];
        let plus = th_propagate(shifted, dt_s, chief_state, mu, max_step_s)?;
        for row in 0..6 {
            phi[row * 6 + axis] = (plus[row] - base[row]) / steps[axis];
        }
    }
    Ok(phi)
}

fn chief_elements(chief: [f64; 6], mu: f64) -> Result<(f64, f64, f64), String> {
    let r = [chief[0], chief[1], chief[2]];
    let v = [chief[3], chief[4], chief[5]];
    let radius = norm(r);
    if !radius.is_finite() || radius <= 1.0e-12 || !mu.is_finite() || mu <= 0.0 {
        return Err("YA chief requires finite positive radius and mu".into());
    }
    let h_vec = cross(r, v);
    let h = norm(h_vec);
    if !h.is_finite() || h <= 1.0e-12 {
        return Err("YA chief requires nonzero angular momentum".into());
    }
    let rv = dot(r, v);
    let e_vec = [
        ((dot(v, v) - mu / radius) * r[0] - rv * v[0]) / mu,
        ((dot(v, v) - mu / radius) * r[1] - rv * v[1]) / mu,
        ((dot(v, v) - mu / radius) * r[2] - rv * v[2]) / mu,
    ];
    let eccentricity = norm(e_vec);
    // Match OEL's circular-orbit threshold and argument-of-latitude convention.
    // RK4 roundoff can otherwise turn a circular final state into an unstable
    // eccentricity-vector anomaly and corrupt the YA STM.
    let true_anomaly = if eccentricity > 1.0e-10 {
        let cosine = (dot(e_vec, r) / (eccentricity * radius)).clamp(-1.0, 1.0);
        let sine = dot(cross(e_vec, r), h_vec) / (eccentricity * radius * h);
        sine.atan2(cosine)
    } else {
        let node = [-h_vec[1], h_vec[0], 0.0];
        let node_norm = norm(node);
        if node_norm > 1.0e-12 {
            let angle = (dot(node, r) / (node_norm * radius))
                .clamp(-1.0, 1.0)
                .acos();
            if r[2] < 0.0 {
                2.0 * std::f64::consts::PI - angle
            } else {
                angle
            }
        } else {
            r[1].atan2(r[0])
        }
    };
    Ok((eccentricity, true_anomaly, h))
}

fn ya_phi(eccentricity: f64, true_anomaly: f64, integral_i: f64) -> Result<Mat6, String> {
    let e = eccentricity;
    let f = true_anomaly;
    let sin_f = f.sin();
    let cos_f = f.cos();
    let k = 1.0 + e * cos_f;
    if k <= 1.0e-12 {
        return Err("YA STM is singular for this chief true anomaly/eccentricity".into());
    }
    let s = k * sin_f;
    let c = k * cos_f;
    let s_prime = k * cos_f - e * sin_f * sin_f;
    let c_prime = -k * sin_f - e * sin_f * cos_f;
    let i_val = integral_i;
    let mut phi = [0.0; 36];
    phi[0] = s;
    phi[1] = c;
    phi[2] = 2.0 - 3.0 * e * s * i_val;
    phi[6] = s_prime;
    phi[7] = c_prime;
    phi[8] = -3.0 * e * (s_prime * i_val + s / k.powi(2));
    phi[12] = c * (1.0 + 1.0 / k);
    phi[13] = -s * (1.0 + 1.0 / k);
    phi[14] = -3.0 * k.powi(2) * i_val;
    phi[15] = 1.0;
    phi[18] = -2.0 * s;
    phi[19] = e - 2.0 * c;
    phi[20] = -3.0 * (1.0 - 2.0 * e * s * i_val);
    phi[28] = cos_f;
    phi[29] = sin_f;
    phi[34] = -sin_f;
    phi[35] = cos_f;
    Ok(phi)
}

fn ya_inverse(eccentricity: f64, true_anomaly: f64) -> Result<Mat6, String> {
    let e = eccentricity;
    let f = true_anomaly;
    let sin_f = f.sin();
    let cos_f = f.cos();
    let eta_sq = 1.0 - e * e;
    if eta_sq <= 1.0e-12 {
        return Err("YA STM requires elliptical chief eccentricity below one".into());
    }
    let k = 1.0 + e * cos_f;
    if k <= 1.0e-12 {
        return Err("YA STM is singular for this initial true anomaly/eccentricity".into());
    }
    let s = k * sin_f;
    let c = k * cos_f;
    let mut inv = [0.0; 36];
    inv[0] = -3.0 * s * (k + e * e) / k.powi(2);
    inv[1] = c - 2.0 * e;
    inv[3] = -s * (k + 1.0) / k;
    inv[6] = -3.0 * (e + c / k);
    inv[7] = -s;
    inv[9] = -(c * (k + 1.0) / k + e);
    inv[12] = 3.0 * k - eta_sq;
    inv[13] = e * s;
    inv[15] = k * k;
    inv[18] = -3.0 * e * s * (k + 1.0) / k.powi(2);
    inv[19] = -2.0 + e * c;
    inv[20] = eta_sq;
    inv[21] = -e * s * (k + 1.0) / k;
    inv[28] = eta_sq * cos_f;
    inv[29] = -eta_sq * sin_f;
    inv[34] = eta_sq * sin_f;
    inv[35] = eta_sq * cos_f;
    Ok(mat6_scale(&inv, 1.0 / eta_sq))
}

fn ric_to_ya_normalized(chief: [f64; 6]) -> Result<Mat6, String> {
    let r = [chief[0], chief[1], chief[2]];
    let v = [chief[3], chief[4], chief[5]];
    let radius = norm(r);
    let h = norm(cross(r, v));
    if radius <= 1.0e-12 || h <= 1.0e-12 {
        return Err(
            "YA normalized conversion requires nonzero chief radius and angular momentum".into(),
        );
    }
    let radial_rate = dot(r, v) / radius;
    let true_anomaly_rate = h / radius.powi(2);
    let scale = 1.0 / (radius * true_anomaly_rate);
    let mut mat = [0.0; 36];
    for &(coord, row) in &[(0usize, 0usize), (1, 2), (2, 4)] {
        let vel = coord + 3;
        mat[row * 6 + coord] = 1.0 / radius;
        mat[(row + 1) * 6 + coord] = -radial_rate / (radius * radius * true_anomaly_rate);
        mat[(row + 1) * 6 + vel] = scale;
    }
    Ok(mat)
}

fn ya_normalized_to_ric(chief: [f64; 6]) -> Result<Mat6, String> {
    let r = [chief[0], chief[1], chief[2]];
    let v = [chief[3], chief[4], chief[5]];
    let radius = norm(r);
    let h = norm(cross(r, v));
    if radius <= 1.0e-12 || h <= 1.0e-12 {
        return Err(
            "YA dimensional conversion requires nonzero chief radius and angular momentum".into(),
        );
    }
    let radial_rate = dot(r, v) / radius;
    let true_anomaly_rate = h / radius.powi(2);
    let mut mat = [0.0; 36];
    for &(coord, row) in &[(0usize, 0usize), (1, 2), (2, 4)] {
        let vel = coord + 3;
        mat[coord * 6 + row] = radius;
        mat[vel * 6 + row] = radial_rate;
        mat[vel * 6 + row + 1] = radius * true_anomaly_rate;
    }
    Ok(mat)
}

pub fn ya_stm(
    dt_s: f64,
    chief_start: [f64; 6],
    chief_end: Option<[f64; 6]>,
    mu: f64,
    max_step_s: f64,
) -> Result<Mat6, String> {
    if !dt_s.is_finite()
        || !mu.is_finite()
        || mu <= 0.0
        || !max_step_s.is_finite()
        || max_step_s <= 0.0
    {
        return Err("YA STM requires finite dt, positive mu, and positive max_step_s".into());
    }
    validate_finite(&chief_start, "chief_start_eci_km_s")?;
    let end = match chief_end {
        Some(value) => {
            validate_finite(&value, "chief_end_eci_km_s")?;
            value
        }
        None => integrate(chief_start, dt_s, max_step_s, |state| {
            two_body_derivative(state, mu)
        }),
    };
    let (e0, f0, h0) = chief_elements(chief_start, mu)?;
    let (e1, f1, _) = chief_elements(end, mu)?;
    if (e1 - e0).abs() > 1.0e-7 {
        return Err("YA STM requires a two-body chief with constant eccentricity".into());
    }
    let integral_i = mu.powi(2) * dt_s / h0.powi(3);
    let normalized = mat6_mul(&ya_phi(e0, f1, integral_i)?, &ya_inverse(e0, f0)?);
    let to_normalized = ric_to_ya_normalized(chief_start)?;
    let from_normalized = ya_normalized_to_ric(end)?;
    Ok(mat6_mul(
        &mat6_mul(&from_normalized, &normalized),
        &to_normalized,
    ))
}

pub fn ya_propagate(
    relative_state: [f64; 6],
    dt_s: f64,
    chief_start: [f64; 6],
    mu: f64,
    max_step_s: f64,
) -> Result<([f64; 6], Mat6), String> {
    let chief_end = integrate(chief_start, dt_s, max_step_s, |state| {
        two_body_derivative(state, mu)
    });
    let phi = ya_stm(dt_s, chief_start, Some(chief_end), mu, max_step_s)?;
    Ok((mat6_vec(&phi, &relative_state), phi))
}

fn ric_dcm_ir(chief: [f64; 6]) -> Result<Mat6, String> {
    let r = [chief[0], chief[1], chief[2]];
    let v = [chief[3], chief[4], chief[5]];
    let radius = norm(r);
    if !radius.is_finite() || radius <= 1.0e-12 {
        return Err("RIC frame is undefined for a zero or non-finite position vector".into());
    }
    let r_hat = [r[0] / radius, r[1] / radius, r[2] / radius];
    let h = cross(r, v);
    let h_norm = norm(h);
    if !h_norm.is_finite() || h_norm <= 1.0e-12 {
        return Err("RIC frame is undefined for zero angular momentum".into());
    }
    let c_hat = [h[0] / h_norm, h[1] / h_norm, h[2] / h_norm];
    let i_raw = cross(c_hat, r_hat);
    let i_norm = norm(i_raw);
    if !i_norm.is_finite() || i_norm <= 1.0e-12 {
        return Err("RIC frame is undefined for a degenerate basis".into());
    }
    let i_hat = [i_raw[0] / i_norm, i_raw[1] / i_norm, i_raw[2] / i_norm];
    let mut dcm = [0.0; 36];
    for row in 0..3 {
        dcm[row * 6] = r_hat[row];
        dcm[row * 6 + 1] = i_hat[row];
        dcm[row * 6 + 2] = c_hat[row];
    }
    Ok(dcm)
}

fn frame_transform(deputy: [f64; 6], chief: [f64; 6], to_eci: bool) -> Result<[f64; 6], String> {
    let dcm = ric_dcm_ir(chief)?;
    let r = [chief[0], chief[1], chief[2]];
    let v = [chief[3], chief[4], chief[5]];
    let h = cross(r, v);
    let radius = norm(r);
    let omega = [
        h[0] / radius.powi(2),
        h[1] / radius.powi(2),
        h[2] / radius.powi(2),
    ];
    let cross_omega = |x: [f64; 3]| cross(omega, x);
    if to_eci {
        let rel = deputy;
        let dr_ric = [rel[0], rel[1], rel[2]];
        let dv_ric = [rel[3], rel[4], rel[5]];
        let dr_eci = [
            dcm[0] * dr_ric[0] + dcm[1] * dr_ric[1] + dcm[2] * dr_ric[2],
            dcm[6] * dr_ric[0] + dcm[7] * dr_ric[1] + dcm[8] * dr_ric[2],
            dcm[12] * dr_ric[0] + dcm[13] * dr_ric[1] + dcm[14] * dr_ric[2],
        ];
        let omega_cross = cross_omega(dr_eci);
        let dv_rot = [
            dcm[0] * dv_ric[0] + dcm[1] * dv_ric[1] + dcm[2] * dv_ric[2],
            dcm[6] * dv_ric[0] + dcm[7] * dv_ric[1] + dcm[8] * dv_ric[2],
            dcm[12] * dv_ric[0] + dcm[13] * dv_ric[1] + dcm[14] * dv_ric[2],
        ];
        return Ok([
            r[0] + dr_eci[0],
            r[1] + dr_eci[1],
            r[2] + dr_eci[2],
            v[0] + dv_rot[0] + omega_cross[0],
            v[1] + dv_rot[1] + omega_cross[1],
            v[2] + dv_rot[2] + omega_cross[2],
        ]);
    }
    let dr_eci = [deputy[0] - r[0], deputy[1] - r[1], deputy[2] - r[2]];
    let dv_eci = [deputy[3] - v[0], deputy[4] - v[1], deputy[5] - v[2]];
    let omega_cross = cross_omega(dr_eci);
    let dr_ric = [
        dcm[0] * dr_eci[0] + dcm[6] * dr_eci[1] + dcm[12] * dr_eci[2],
        dcm[1] * dr_eci[0] + dcm[7] * dr_eci[1] + dcm[13] * dr_eci[2],
        dcm[2] * dr_eci[0] + dcm[8] * dr_eci[1] + dcm[14] * dr_eci[2],
    ];
    let dv_adj = [
        dv_eci[0] - omega_cross[0],
        dv_eci[1] - omega_cross[1],
        dv_eci[2] - omega_cross[2],
    ];
    let dv_ric = [
        dcm[0] * dv_adj[0] + dcm[6] * dv_adj[1] + dcm[12] * dv_adj[2],
        dcm[1] * dv_adj[0] + dcm[7] * dv_adj[1] + dcm[13] * dv_adj[2],
        dcm[2] * dv_adj[0] + dcm[8] * dv_adj[1] + dcm[14] * dv_adj[2],
    ];
    Ok([
        dr_ric[0], dr_ric[1], dr_ric[2], dv_ric[0], dv_ric[1], dv_ric[2],
    ])
}

pub fn eci_relative_to_ric(deputy: [f64; 6], chief: [f64; 6]) -> Result<[f64; 6], String> {
    frame_transform(deputy, chief, false)
}

pub fn ric_relative_to_eci(relative: [f64; 6], chief: [f64; 6]) -> Result<[f64; 6], String> {
    frame_transform(relative, chief, true)
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    fn fixed<const N: usize>(values: Vec<f64>, name: &str) -> PyResult<[f64; N]> {
        values
            .try_into()
            .map_err(|_| PyValueError::new_err(format!("{name} must have {N} values")))
    }
    fn mat(values: Mat6) -> Vec<f64> {
        values.to_vec()
    }

    #[pyfunction]
    #[pyo3(signature = (mean_motion_rad_s, dt_s))]
    fn relative_hcw_stm(mean_motion_rad_s: f64, dt_s: f64) -> PyResult<Vec<f64>> {
        hcw_stm(mean_motion_rad_s, dt_s)
            .map(mat)
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[pyo3(signature = (mean_motion_rad_s, reference_radius_km, reference_inclination_rad, j2, earth_radius_km, dt_s))]
    fn relative_ss_j2_stm(
        mean_motion_rad_s: f64,
        reference_radius_km: f64,
        reference_inclination_rad: f64,
        j2: f64,
        earth_radius_km: f64,
        dt_s: f64,
    ) -> PyResult<Vec<f64>> {
        ss_j2_stm(
            mean_motion_rad_s,
            reference_radius_km,
            reference_inclination_rad,
            j2,
            earth_radius_km,
            dt_s,
        )
        .map(mat)
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[pyo3(signature = (relative_state, dt_s, chief_state, mu=398600.4415, max_step_s=10.0))]
    fn relative_th_propagate(
        relative_state: Vec<f64>,
        dt_s: f64,
        chief_state: Vec<f64>,
        mu: f64,
        max_step_s: f64,
    ) -> PyResult<Vec<f64>> {
        th_propagate(
            fixed(relative_state, "relative_state_ric")?,
            dt_s,
            fixed(chief_state, "chief_state_eci_km_s")?,
            mu,
            max_step_s,
        )
        .map(|value| value.to_vec())
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[pyo3(signature = (relative_state, dt_s, chief_state, mu=398600.4415, max_step_s=10.0))]
    fn relative_th_finite_difference(
        relative_state: Vec<f64>,
        dt_s: f64,
        chief_state: Vec<f64>,
        mu: f64,
        max_step_s: f64,
    ) -> PyResult<Vec<f64>> {
        th_finite_difference(
            fixed(relative_state, "relative_state_ric")?,
            dt_s,
            fixed(chief_state, "chief_state_eci_km_s")?,
            mu,
            max_step_s,
        )
        .map(mat)
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[pyo3(signature = (relative_state, dt_s, chief_state, mu=398600.4415, max_step_s=10.0))]
    fn relative_th_variational(
        relative_state: Vec<f64>,
        dt_s: f64,
        chief_state: Vec<f64>,
        mu: f64,
        max_step_s: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        th_variational(
            fixed(relative_state, "relative_state_ric")?,
            dt_s,
            fixed(chief_state, "chief_state_eci_km_s")?,
            mu,
            max_step_s,
        )
        .map(|(state, phi)| (state.to_vec(), phi.to_vec()))
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[pyo3(signature = (dt_s, chief_start, chief_end=None, mu=398600.4415, max_step_s=10.0))]
    fn relative_ya_stm(
        dt_s: f64,
        chief_start: Vec<f64>,
        chief_end: Option<Vec<f64>>,
        mu: f64,
        max_step_s: f64,
    ) -> PyResult<Vec<f64>> {
        let start = fixed(chief_start, "chief_start_eci_km_s")?;
        let end = chief_end
            .map(|value| fixed(value, "chief_end_eci_km_s"))
            .transpose()?;
        ya_stm(dt_s, start, end, mu, max_step_s)
            .map(mat)
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[pyo3(signature = (relative_state, dt_s, chief_start, mu=398600.4415, max_step_s=10.0))]
    fn relative_ya_propagate(
        relative_state: Vec<f64>,
        dt_s: f64,
        chief_start: Vec<f64>,
        mu: f64,
        max_step_s: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        ya_propagate(
            fixed(relative_state, "relative_state_ric")?,
            dt_s,
            fixed(chief_start, "chief_start_eci_km_s")?,
            mu,
            max_step_s,
        )
        .map(|(state, phi)| (state.to_vec(), phi.to_vec()))
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn relative_eci_to_ric_batch(
        deputy_states: Vec<f64>,
        chief_states: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        if deputy_states.len() != chief_states.len() || deputy_states.len() % 6 != 0 {
            return Err(PyValueError::new_err(
                "deputy_states and chief_states must have equal lengths divisible by 6",
            ));
        }
        validate_finite(&deputy_states, "deputy_states").map_err(PyValueError::new_err)?;
        validate_finite(&chief_states, "chief_states").map_err(PyValueError::new_err)?;
        let mut out = Vec::with_capacity(deputy_states.len());
        for offset in (0..deputy_states.len()).step_by(6) {
            let deputy = deputy_states[offset..offset + 6].try_into().unwrap();
            let chief = chief_states[offset..offset + 6].try_into().unwrap();
            out.extend_from_slice(
                &eci_relative_to_ric(deputy, chief).map_err(PyValueError::new_err)?,
            );
        }
        Ok(out)
    }

    #[pyfunction]
    fn relative_ric_to_eci_batch(
        relative_states: Vec<f64>,
        chief_states: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        if relative_states.len() != chief_states.len() || relative_states.len() % 6 != 0 {
            return Err(PyValueError::new_err(
                "relative_states and chief_states must have equal lengths divisible by 6",
            ));
        }
        validate_finite(&relative_states, "relative_states").map_err(PyValueError::new_err)?;
        validate_finite(&chief_states, "chief_states").map_err(PyValueError::new_err)?;
        let mut out = Vec::with_capacity(relative_states.len());
        for offset in (0..relative_states.len()).step_by(6) {
            let relative = relative_states[offset..offset + 6].try_into().unwrap();
            let chief = chief_states[offset..offset + 6].try_into().unwrap();
            out.extend_from_slice(
                &ric_relative_to_eci(relative, chief).map_err(PyValueError::new_err)?,
            );
        }
        Ok(out)
    }

    #[pyfunction]
    fn relative_hcw_stm_bytes(
        py: Python<'_>,
        mean_motion_rad_s: f64,
        dt_s: f64,
    ) -> PyResult<Py<PyBytes>> {
        let values = hcw_stm(mean_motion_rad_s, dt_s).map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }

    #[pyfunction]
    fn relative_ss_j2_stm_bytes(
        py: Python<'_>,
        mean_motion_rad_s: f64,
        reference_radius_km: f64,
        reference_inclination_rad: f64,
        j2: f64,
        earth_radius_km: f64,
        dt_s: f64,
    ) -> PyResult<Py<PyBytes>> {
        let values = ss_j2_stm(
            mean_motion_rad_s,
            reference_radius_km,
            reference_inclination_rad,
            j2,
            earth_radius_km,
            dt_s,
        )
        .map_err(PyValueError::new_err)?;
        Ok(output_bytes(py, &values)?)
    }

    #[pyfunction]
    fn relative_chief_propagate_bytes(
        py: Python<'_>,
        chief_state: Vec<f64>,
        dt_s: f64,
        mu: f64,
        max_step_s: f64,
    ) -> PyResult<Py<PyBytes>> {
        let chief = fixed(chief_state, "chief_state_eci_km_s")?;
        let state = py
            .detach(|| chief_propagate(chief, dt_s, mu, max_step_s))
            .map_err(PyValueError::new_err)?;
        output_bytes(py, &state)
    }

    #[pyfunction]
    fn relative_th_variational_bytes(
        py: Python<'_>,
        relative_state: Vec<f64>,
        dt_s: f64,
        chief_state: Vec<f64>,
        mu: f64,
        max_step_s: f64,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let (state, phi) = th_variational(
            fixed(relative_state, "relative_state_ric")?,
            dt_s,
            fixed(chief_state, "chief_state_eci_km_s")?,
            mu,
            max_step_s,
        )
        .map_err(PyValueError::new_err)?;
        Ok((output_bytes(py, &state)?, output_bytes(py, &phi)?))
    }

    #[pyfunction]
    fn relative_ya_propagate_bytes(
        py: Python<'_>,
        relative_state: Vec<f64>,
        dt_s: f64,
        chief_start: Vec<f64>,
        mu: f64,
        max_step_s: f64,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let (state, phi) = ya_propagate(
            fixed(relative_state, "relative_state_ric")?,
            dt_s,
            fixed(chief_start, "chief_start_eci_km_s")?,
            mu,
            max_step_s,
        )
        .map_err(PyValueError::new_err)?;
        Ok((output_bytes(py, &state)?, output_bytes(py, &phi)?))
    }

    #[pyfunction]
    fn relative_eci_to_ric_batch_bytes(
        py: Python<'_>,
        deputy_states: &[u8],
        chief_states: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let values = relative_eci_to_ric_batch(
            decode_f64_bytes(deputy_states)?.into_owned(),
            decode_f64_bytes(chief_states)?.into_owned(),
        )?;
        Ok(output_bytes(py, &values)?)
    }

    #[pyfunction]
    fn relative_ric_to_eci_batch_bytes(
        py: Python<'_>,
        relative_states: &[u8],
        chief_states: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let values = relative_ric_to_eci_batch(
            decode_f64_bytes(relative_states)?.into_owned(),
            decode_f64_bytes(chief_states)?.into_owned(),
        )?;
        Ok(output_bytes(py, &values)?)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(relative_hcw_stm, module)?)?;
        module.add_function(wrap_pyfunction!(relative_ss_j2_stm, module)?)?;
        module.add_function(wrap_pyfunction!(relative_th_propagate, module)?)?;
        module.add_function(wrap_pyfunction!(relative_th_finite_difference, module)?)?;
        module.add_function(wrap_pyfunction!(relative_th_variational, module)?)?;
        module.add_function(wrap_pyfunction!(relative_ya_stm, module)?)?;
        module.add_function(wrap_pyfunction!(relative_ya_propagate, module)?)?;
        module.add_function(wrap_pyfunction!(relative_eci_to_ric_batch, module)?)?;
        module.add_function(wrap_pyfunction!(relative_ric_to_eci_batch, module)?)?;
        module.add_function(wrap_pyfunction!(relative_hcw_stm_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(relative_ss_j2_stm_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(relative_chief_propagate_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(relative_th_variational_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(relative_ya_propagate_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(relative_eci_to_ric_batch_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(relative_ric_to_eci_batch_bytes, module)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    python::register(module)
}

#[cfg(test)]
mod tests {
    use super::{chief_propagate, hcw_stm, mat6_expm, ss_j2_stm, th_propagate, th_variational};

    #[test]
    fn chief_zero_duration_preserves_signed_zero_and_bounded_inputs() {
        let state = [7000.0, -0.0, 0.0, 0.0, 7.5, -0.0];
        let result = chief_propagate(state, -0.0, 398600.4418, 10.0).unwrap();
        for (actual, expected) in result.iter().zip(state.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert!(chief_propagate(state, f64::NAN, 398600.4418, 10.0).is_err());
        assert!(chief_propagate(state, 1.0, 398600.4418, 0.0).is_err());
        assert!(chief_propagate(state, 100_000_001.0, 398600.4418, 10.0).is_err());
    }

    #[test]
    fn hcw_identity_at_zero() {
        let phi = hcw_stm(0.001, 0.0).unwrap();
        for row in 0..6 {
            for col in 0..6 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert_eq!(phi[row * 6 + col], expected);
            }
        }
    }

    #[test]
    fn ss_identity_at_zero() {
        let phi = ss_j2_stm(0.001, 7000.0, 0.5, 1.08262668e-3, 6378.137, 0.0).unwrap();
        for row in 0..6 {
            for col in 0..6 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((phi[row * 6 + col] - expected).abs() < 1.0e-14);
            }
        }
    }

    #[test]
    fn th_variational_is_identity_at_zero() {
        let chief = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let (_state, phi) =
            th_variational([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0, chief, 398600.4415, 1.0).unwrap();
        for row in 0..6 {
            for col in 0..6 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert_eq!(phi[row * 6 + col], expected);
            }
        }
    }

    #[test]
    fn th_circular_state_is_finite() {
        let chief = [7000.0, 0.0, 0.0, 0.0, 7.546, 0.0];
        let value = th_propagate(
            [0.1, -0.2, 0.03, 0.0, 0.0, 0.0],
            30.0,
            chief,
            398600.4415,
            1.0,
        )
        .unwrap();
        assert!(value.iter().all(|entry| entry.is_finite()));
    }

    #[test]
    fn matrix_exponential_rejects_extreme_finite_norm() {
        let mut matrix = [0.0; 36];
        matrix[0] = f64::MAX;
        let error = mat6_expm(&matrix).unwrap_err();
        assert!(error.contains("too large"));
    }
}
