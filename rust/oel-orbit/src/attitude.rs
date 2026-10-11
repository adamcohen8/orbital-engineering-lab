//! Native rigid-body attitude and built-in disturbance kernels.
//!
//! The public Python attitude model uses a scalar-first body-to-inertial
//! quaternion, SI angular rates and torques, and ECI position in kilometres.
//! This module intentionally mirrors the numerical kernels in
//! `sim.acceleration.kernels.attitude`: the Python implementation remains the
//! reference path and this module is an explicitly selected accelerator.

const MAX_ABS_RATE_RAD_S: f64 = 1.0e6;
const MAX_ABS_TORQUE_NM: f64 = 1.0e12;
const EARTH_MAGNETIC_DIPOLE_T_M3: f64 = -7.94e15;

pub const STAT_NON_FINITE_INPUT: usize = 0;
pub const STAT_RATE_CLAMP: usize = 1;
pub const STAT_TORQUE_CLAMP: usize = 2;
pub const STAT_NON_FINITE_CORIOLIS: usize = 3;
pub const STAT_SINGULAR_INERTIA: usize = 4;
pub const STAT_NON_FINITE_OUTPUT: usize = 5;

pub const FACET_MODE_NONE: u8 = 0;
pub const FACET_MODE_SCALAR: u8 = 1;
pub const FACET_MODE_FACETS: u8 = 2;

fn identity_quaternion() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

fn finite4(value: &[f64; 4]) -> bool {
    value.iter().all(|item| item.is_finite())
}

fn finite3(value: &[f64; 3]) -> bool {
    value.iter().all(|item| item.is_finite())
}

fn norm3(value: [f64; 3]) -> f64 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn mat_vec3(matrix: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * value[0] + matrix[0][1] * value[1] + matrix[0][2] * value[2],
        matrix[1][0] * value[0] + matrix[1][1] * value[1] + matrix[1][2] * value[2],
        matrix[2][0] * value[0] + matrix[2][1] * value[1] + matrix[2][2] * value[2],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn nan_to_num_clamped(value: f64, limit: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else if value == f64::INFINITY {
        limit
    } else if value == f64::NEG_INFINITY {
        -limit
    } else {
        value
    }
}

/// Normalize a scalar-first quaternion using the same identity fallback as
/// the Python and Numba kernels.
pub fn normalize_quaternion(value: [f64; 4]) -> [f64; 4] {
    if !finite4(&value) {
        return identity_quaternion();
    }
    let norm_sq = value.iter().map(|item| item * item).sum::<f64>();
    if norm_sq <= 0.0 || !norm_sq.is_finite() {
        return identity_quaternion();
    }
    let norm = norm_sq.sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return identity_quaternion();
    }
    std::array::from_fn(|index| value[index] / norm)
}

pub fn quaternion_multiply(first: [f64; 4], second: [f64; 4]) -> [f64; 4] {
    let a = normalize_quaternion(first);
    let b = normalize_quaternion(second);
    normalize_quaternion([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])
}

pub fn quaternion_delta_from_body_rate(omega: [f64; 3], dt_s: f64) -> [f64; 4] {
    if !finite3(&omega) || !dt_s.is_finite() {
        return identity_quaternion();
    }
    let norm = norm3(omega);
    if norm <= 1.0e-15 || dt_s == 0.0 {
        return identity_quaternion();
    }
    let mut half_theta = 0.5 * norm * dt_s;
    if !half_theta.is_finite() {
        return identity_quaternion();
    }
    half_theta %= 2.0 * std::f64::consts::PI;
    let sine = half_theta.sin();
    let cosine = half_theta.cos();
    normalize_quaternion([
        cosine,
        omega[0] / norm * sine,
        omega[1] / norm * sine,
        omega[2] / norm * sine,
    ])
}

pub fn quaternion_to_dcm_bn(value: [f64; 4]) -> [[f64; 3]; 3] {
    let q = normalize_quaternion(value);
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];
    [
        [
            1.0 - 2.0 * (q2 * q2 + q3 * q3),
            2.0 * (q1 * q2 + q0 * q3),
            2.0 * (q1 * q3 - q0 * q2),
        ],
        [
            2.0 * (q1 * q2 - q0 * q3),
            1.0 - 2.0 * (q1 * q1 + q3 * q3),
            2.0 * (q2 * q3 + q0 * q1),
        ],
        [
            2.0 * (q1 * q3 + q0 * q2),
            2.0 * (q2 * q3 - q0 * q1),
            1.0 - 2.0 * (q1 * q1 + q2 * q2),
        ],
    ]
}

fn determinant3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn solve3(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    let determinant = determinant3(matrix);
    if !determinant.is_finite() || determinant == 0.0 {
        return None;
    }
    let mut out = [0.0; 3];
    for column in 0..3 {
        let mut replaced = matrix;
        for row in 0..3 {
            replaced[row][column] = rhs[row];
        }
        out[column] = determinant3(replaced) / determinant;
    }
    Some(out)
}

fn rigid_body_derivatives_impl(
    quat: [f64; 4],
    omega: [f64; 3],
    inertia: [[f64; 3]; 3],
    torque: [f64; 3],
    quaternion_derivative: bool,
) -> ([f64; 4], [f64; 3], [i64; 6]) {
    let mut stats = [0_i64; 6];
    let quaternion_bad = !finite4(&quat) || quat.iter().map(|item| item * item).sum::<f64>() <= 0.0;
    if quaternion_bad {
        stats[STAT_NON_FINITE_INPUT] += 1;
    }
    let q = if quaternion_derivative {
        normalize_quaternion(quat)
    } else {
        [0.0; 4]
    };
    let mut w = omega;
    let mut tau = torque;
    let mut input_bad = false;
    for axis in 0..3 {
        if !w[axis].is_finite() {
            input_bad = true;
            stats[STAT_RATE_CLAMP] += 1;
        }
        if !tau[axis].is_finite() {
            input_bad = true;
            stats[STAT_TORQUE_CLAMP] += 1;
        }
        w[axis] = nan_to_num_clamped(w[axis], MAX_ABS_RATE_RAD_S);
        tau[axis] = nan_to_num_clamped(tau[axis], MAX_ABS_TORQUE_NM);
    }
    for row in 0..3 {
        for column in 0..3 {
            if !inertia[row][column].is_finite() {
                input_bad = true;
            }
        }
    }
    if input_bad {
        stats[STAT_NON_FINITE_INPUT] += 1;
    }
    for axis in 0..3 {
        let clipped_w = w[axis].clamp(-MAX_ABS_RATE_RAD_S, MAX_ABS_RATE_RAD_S);
        let clipped_tau = tau[axis].clamp(-MAX_ABS_TORQUE_NM, MAX_ABS_TORQUE_NM);
        if clipped_w != w[axis] {
            stats[STAT_RATE_CLAMP] += 1;
        }
        if clipped_tau != tau[axis] {
            stats[STAT_TORQUE_CLAMP] += 1;
        }
        w[axis] = clipped_w;
        tau[axis] = clipped_tau;
    }

    let iw = mat_vec3(inertia, w);
    let coriolis = cross3(w, iw);
    if !finite3(&coriolis) {
        stats[STAT_NON_FINITE_CORIOLIS] += 1;
    }
    let rhs = std::array::from_fn(|axis| {
        tau[axis] - nan_to_num_clamped(coriolis[axis], MAX_ABS_TORQUE_NM)
    });
    let mut omega_dot = solve3(inertia, rhs).unwrap_or_else(|| {
        stats[STAT_SINGULAR_INERTIA] += 1;
        [0.0; 3]
    });
    if !finite3(&omega_dot) {
        stats[STAT_NON_FINITE_OUTPUT] += 1;
    }
    for axis in 0..3 {
        omega_dot[axis] = nan_to_num_clamped(omega_dot[axis], MAX_ABS_RATE_RAD_S);
    }
    let q_dot = [
        0.5 * (-w[0] * q[1] - w[1] * q[2] - w[2] * q[3]),
        0.5 * (w[0] * q[0] + w[2] * q[2] - w[1] * q[3]),
        0.5 * (w[1] * q[0] - w[2] * q[1] + w[0] * q[3]),
        0.5 * (w[2] * q[0] + w[1] * q[1] - w[0] * q[2]),
    ];
    (q_dot, omega_dot, stats)
}

pub fn rigid_body_derivatives(
    quat: [f64; 4],
    omega: [f64; 3],
    inertia: [[f64; 3]; 3],
    torque: [f64; 3],
) -> ([f64; 4], [f64; 3], [i64; 6]) {
    rigid_body_derivatives_impl(quat, omega, inertia, torque, true)
}

fn propagate_attitude_rates(
    quat: [f64; 4],
    omega: [f64; 3],
    inertia: [[f64; 3]; 3],
    torque: [f64; 3],
    dt_s: f64,
) -> ([f64; 3], [f64; 3], [i64; 6]) {
    let (_, omega_dot, mut stats) =
        rigid_body_derivatives_impl(quat, omega, inertia, torque, false);
    let dt = dt_s.max(0.0);
    let mut omega_next = std::array::from_fn(|axis| omega[axis] + dt * omega_dot[axis]);
    if !finite3(&omega_next) {
        stats[STAT_NON_FINITE_OUTPUT] += 1;
    }
    for axis in 0..3 {
        omega_next[axis] = nan_to_num_clamped(omega_next[axis], MAX_ABS_RATE_RAD_S)
            .clamp(-MAX_ABS_RATE_RAD_S, MAX_ABS_RATE_RAD_S);
    }
    let mut omega_mid = std::array::from_fn(|axis| omega[axis] + 0.5 * dt * omega_dot[axis]);
    if !finite3(&omega_mid) {
        stats[STAT_NON_FINITE_OUTPUT] += 1;
    }
    for axis in 0..3 {
        omega_mid[axis] = nan_to_num_clamped(omega_mid[axis], MAX_ABS_RATE_RAD_S)
            .clamp(-MAX_ABS_RATE_RAD_S, MAX_ABS_RATE_RAD_S);
    }
    (omega_next, omega_mid, stats)
}

/// Observe discarded command predictions without materializing a quaternion.
/// Quaternion construction always normalizes to a finite unit quaternion or
/// identity, so only the shared rate step can contribute output guardrails.
pub fn propagate_attitude_rate_guardrails(
    quat: [f64; 4],
    omega: [f64; 3],
    inertia: [[f64; 3]; 3],
    torque: [f64; 3],
    dt_s: f64,
) -> ([f64; 3], [i64; 6]) {
    let (next, _, stats) = propagate_attitude_rates(quat, omega, inertia, torque, dt_s);
    (next, stats)
}

pub fn propagate_attitude_exponential_map(
    quat: [f64; 4],
    omega: [f64; 3],
    inertia: [[f64; 3]; 3],
    torque: [f64; 3],
    dt_s: f64,
) -> ([f64; 4], [f64; 3], [i64; 6]) {
    let (omega_next, omega_mid, mut stats) =
        propagate_attitude_rates(quat, omega, inertia, torque, dt_s);
    let dt = dt_s.max(0.0);
    let q_next = quaternion_multiply(
        normalize_quaternion(quat),
        quaternion_delta_from_body_rate(omega_mid, dt),
    );
    if !finite4(&q_next) {
        stats[STAT_NON_FINITE_OUTPUT] += 1;
    }
    (q_next, omega_next, stats)
}

fn gravity_gradient_torque(
    position_eci_km: [f64; 3],
    c_bn: [[f64; 3]; 3],
    inertia: [[f64; 3]; 3],
    mu_km3_s2: f64,
) -> [f64; 3] {
    let r_i_m = position_eci_km.map(|item| item * 1.0e3);
    let r_norm_m = norm3(r_i_m);
    if r_norm_m == 0.0 {
        return [0.0; 3];
    }
    let r_hat_b = mat_vec3(c_bn, r_i_m.map(|item| item / r_norm_m));
    let inertia_r_hat = mat_vec3(inertia, r_hat_b);
    let scale = 3.0 * (mu_km3_s2 * 1.0e9) / r_norm_m.powi(3);
    let cross = cross3(r_hat_b, inertia_r_hat);
    cross.map(|item| item * scale)
}

fn magnetic_torque(
    position_eci_km: [f64; 3],
    c_bn: [[f64; 3]; 3],
    magnetic_dipole_body: [f64; 3],
    magnetic_field_eci: [f64; 3],
    magnetic_field_provided: bool,
) -> [f64; 3] {
    let b_eci = if magnetic_field_provided {
        magnetic_field_eci
    } else {
        let r_i_m = position_eci_km.map(|item| item * 1.0e3);
        let r_norm_m = norm3(r_i_m);
        if r_norm_m == 0.0 {
            return [0.0; 3];
        }
        let r_hat = r_i_m.map(|item| item / r_norm_m);
        let dipole_dot_r = EARTH_MAGNETIC_DIPOLE_T_M3 * r_hat[2];
        let denominator = r_norm_m.powi(3);
        [
            3.0 * r_hat[0] * dipole_dot_r / denominator,
            3.0 * r_hat[1] * dipole_dot_r / denominator,
            (3.0 * r_hat[2] * dipole_dot_r - EARTH_MAGNETIC_DIPOLE_T_M3) / denominator,
        ]
    };
    cross3(magnetic_dipole_body, mat_vec3(c_bn, b_eci))
}

fn drag_torque(
    c_bn: [[f64; 3]; 3],
    density_kg_m3: f64,
    drag_v_rel_eci_m_s: [f64; 3],
    drag_v_rel_norm_m_s: f64,
    mode: u8,
    drag_area_m2: f64,
    drag_cd: f64,
    drag_cp_offset_body: [f64; 3],
    facet_normals_body: &[[f64; 3]],
    facet_areas_m2: &[f64],
    facet_cd: &[f64],
    facet_cp_offsets_body: &[[f64; 3]],
) -> [f64; 3] {
    if drag_v_rel_norm_m_s == 0.0 || density_kg_m3 <= 0.0 || mode == FACET_MODE_NONE {
        return [0.0; 3];
    }
    let v_rel_body = mat_vec3(c_bn, drag_v_rel_eci_m_s);
    if mode == FACET_MODE_SCALAR {
        let force_mag = 0.5 * density_kg_m3 * drag_v_rel_norm_m_s.powi(2) * drag_cd * drag_area_m2;
        let force_body = v_rel_body.map(|item| -force_mag * item / drag_v_rel_norm_m_s);
        return cross3(drag_cp_offset_body, force_body);
    }
    let mut torque = [0.0; 3];
    let v_hat_body = v_rel_body.map(|item| item / drag_v_rel_norm_m_s);
    for index in 0..facet_areas_m2.len() {
        let normal = facet_normals_body[index];
        let normal_norm = norm3(normal);
        if normal_norm <= 0.0 {
            continue;
        }
        let projected_cosine = dot3(normal.map(|item| item / normal_norm), v_hat_body);
        let projected_area = facet_areas_m2[index] * projected_cosine.max(0.0);
        if projected_area <= 0.0 {
            continue;
        }
        let scale =
            -0.5 * density_kg_m3 * projected_area * facet_cd[index] * drag_v_rel_norm_m_s.powi(2);
        let force_body = v_hat_body.map(|item| scale * item);
        let component = cross3(facet_cp_offsets_body[index], force_body);
        for axis in 0..3 {
            torque[axis] += component[axis];
        }
    }
    torque
}

fn srp_torque(
    c_bn: [[f64; 3]; 3],
    sun_dir_eci_unit: [f64; 3],
    pressure_scaled_n_m2: f64,
    mode: u8,
    srp_area_m2: f64,
    srp_cp_offset_body: [f64; 3],
    facet_normals_body: &[[f64; 3]],
    facet_areas_m2: &[f64],
    facet_cp_offsets_body: &[[f64; 3]],
) -> [f64; 3] {
    if pressure_scaled_n_m2 <= 0.0 || mode == FACET_MODE_NONE {
        return [0.0; 3];
    }
    let sun_dir_body = mat_vec3(c_bn, sun_dir_eci_unit);
    if mode == FACET_MODE_SCALAR {
        let force_body = sun_dir_body.map(|item| -pressure_scaled_n_m2 * srp_area_m2 * item);
        return cross3(srp_cp_offset_body, force_body);
    }
    let mut torque = [0.0; 3];
    for index in 0..facet_areas_m2.len() {
        let normal = facet_normals_body[index];
        let normal_norm = norm3(normal);
        if normal_norm <= 0.0 {
            continue;
        }
        let illumination = dot3(normal.map(|item| item / normal_norm), sun_dir_body).max(0.0);
        if illumination <= 0.0 {
            continue;
        }
        let scale = -pressure_scaled_n_m2 * facet_areas_m2[index] * illumination;
        let force_body = sun_dir_body.map(|item| scale * item);
        let component = cross3(facet_cp_offsets_body[index], force_body);
        for axis in 0..3 {
            torque[axis] += component[axis];
        }
    }
    torque
}

#[allow(clippy::too_many_arguments)]
pub fn builtin_disturbance_torque(
    quat: [f64; 4],
    position_eci_km: [f64; 3],
    inertia: [[f64; 3]; 3],
    mu_km3_s2: f64,
    enabled: [i64; 4],
    magnetic_dipole_body: [f64; 3],
    magnetic_field_eci: [f64; 3],
    magnetic_field_provided: bool,
    density_kg_m3: f64,
    drag_v_rel_eci_m_s: [f64; 3],
    drag_v_rel_norm_m_s: f64,
    drag_mode: u8,
    drag_area_m2: f64,
    drag_cd: f64,
    drag_cp_offset_body: [f64; 3],
    drag_facet_normals: &[[f64; 3]],
    drag_facet_areas: &[f64],
    drag_facet_cd: &[f64],
    drag_facet_cp_offsets: &[[f64; 3]],
    sun_dir_eci_unit: [f64; 3],
    srp_pressure_scaled_n_m2: f64,
    srp_mode: u8,
    srp_area_m2: f64,
    srp_cp_offset_body: [f64; 3],
    srp_facet_normals: &[[f64; 3]],
    srp_facet_areas: &[f64],
    srp_facet_cp_offsets: &[[f64; 3]],
) -> [f64; 3] {
    let c_bn = quaternion_to_dcm_bn(quat);
    let mut torque = [0.0; 3];
    if enabled[0] != 0 {
        let component = gravity_gradient_torque(position_eci_km, c_bn, inertia, mu_km3_s2);
        for axis in 0..3 {
            torque[axis] += component[axis];
        }
    }
    if enabled[1] != 0 {
        let component = magnetic_torque(
            position_eci_km,
            c_bn,
            magnetic_dipole_body,
            magnetic_field_eci,
            magnetic_field_provided,
        );
        for axis in 0..3 {
            torque[axis] += component[axis];
        }
    }
    if enabled[2] != 0 {
        let component = drag_torque(
            c_bn,
            density_kg_m3,
            drag_v_rel_eci_m_s,
            drag_v_rel_norm_m_s,
            drag_mode,
            drag_area_m2,
            drag_cd,
            drag_cp_offset_body,
            drag_facet_normals,
            drag_facet_areas,
            drag_facet_cd,
            drag_facet_cp_offsets,
        );
        for axis in 0..3 {
            torque[axis] += component[axis];
        }
    }
    if enabled[3] != 0 {
        let component = srp_torque(
            c_bn,
            sun_dir_eci_unit,
            srp_pressure_scaled_n_m2,
            srp_mode,
            srp_area_m2,
            srp_cp_offset_body,
            srp_facet_normals,
            srp_facet_areas,
            srp_facet_cp_offsets,
        );
        for axis in 0..3 {
            torque[axis] += component[axis];
        }
    }
    torque
}

#[allow(clippy::too_many_arguments)]
pub fn propagate_builtin_disturbances(
    quat: [f64; 4],
    omega: [f64; 3],
    inertia: [[f64; 3]; 3],
    command_torque: [f64; 3],
    substeps_s: &[f64],
    position_eci_km: [f64; 3],
    mu_km3_s2: f64,
    enabled: [i64; 4],
    magnetic_dipole_body: [f64; 3],
    magnetic_field_eci: [f64; 3],
    magnetic_field_provided: bool,
    density_kg_m3: f64,
    drag_v_rel_eci_m_s: [f64; 3],
    drag_v_rel_norm_m_s: f64,
    drag_mode: u8,
    drag_area_m2: f64,
    drag_cd: f64,
    drag_cp_offset_body: [f64; 3],
    drag_facet_normals: &[[f64; 3]],
    drag_facet_areas: &[f64],
    drag_facet_cd: &[f64],
    drag_facet_cp_offsets: &[[f64; 3]],
    sun_dir_eci_unit: [f64; 3],
    srp_pressure_scaled_n_m2: f64,
    srp_mode: u8,
    srp_area_m2: f64,
    srp_cp_offset_body: [f64; 3],
    srp_facet_normals: &[[f64; 3]],
    srp_facet_areas: &[f64],
    srp_facet_cp_offsets: &[[f64; 3]],
) -> ([f64; 4], [f64; 3], [i64; 6]) {
    if substeps_s.is_empty() {
        return (quat, omega, [0; 6]);
    }
    // These inputs remain fixed throughout this one propagation call. Keep
    // their original operation grouping, and prepare only values independent
    // of the changing quaternion. The standalone/stagewise torque API remains
    // authoritative for callers whose environment changes at each RK stage.
    let gravity_gradient = if enabled[0] != 0 {
        let r_i_m = position_eci_km.map(|item| item * 1.0e3);
        let r_norm_m = norm3(r_i_m);
        if r_norm_m == 0.0 {
            None
        } else {
            Some((
                r_i_m.map(|item| item / r_norm_m),
                3.0 * (mu_km3_s2 * 1.0e9) / r_norm_m.powi(3),
            ))
        }
    } else {
        None
    };
    let magnetic_field = if enabled[1] == 0 {
        None
    } else if magnetic_field_provided {
        Some(magnetic_field_eci)
    } else {
        let r_i_m = position_eci_km.map(|item| item * 1.0e3);
        let r_norm_m = norm3(r_i_m);
        if r_norm_m == 0.0 {
            None
        } else {
            let r_hat = r_i_m.map(|item| item / r_norm_m);
            let dipole_dot_r = EARTH_MAGNETIC_DIPOLE_T_M3 * r_hat[2];
            let denominator = r_norm_m.powi(3);
            Some([
                3.0 * r_hat[0] * dipole_dot_r / denominator,
                3.0 * r_hat[1] * dipole_dot_r / denominator,
                (3.0 * r_hat[2] * dipole_dot_r - EARTH_MAGNETIC_DIPOLE_T_M3) / denominator,
            ])
        }
    };
    let drag_active = enabled[2] != 0
        && !(drag_v_rel_norm_m_s == 0.0 || density_kg_m3 <= 0.0 || drag_mode == FACET_MODE_NONE);
    let drag_speed_squared = drag_v_rel_norm_m_s.powi(2);
    let drag_force_magnitude = if drag_active && drag_mode == FACET_MODE_SCALAR {
        Some(0.5 * density_kg_m3 * drag_speed_squared * drag_cd * drag_area_m2)
    } else {
        None
    };
    let drag_unit_normals: Vec<Option<[f64; 3]>> = if drag_active && drag_mode != FACET_MODE_SCALAR
    {
        (0..drag_facet_areas.len())
            .map(|index| {
                let normal = drag_facet_normals[index];
                let normal_norm = norm3(normal);
                if normal_norm <= 0.0 {
                    None
                } else {
                    Some(normal.map(|item| item / normal_norm))
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let srp_active =
        enabled[3] != 0 && !(srp_pressure_scaled_n_m2 <= 0.0 || srp_mode == FACET_MODE_NONE);
    let srp_scalar_scale = -srp_pressure_scaled_n_m2 * srp_area_m2;
    let srp_unit_normals: Vec<Option<[f64; 3]>> = if srp_active && srp_mode != FACET_MODE_SCALAR {
        (0..srp_facet_areas.len())
            .map(|index| {
                let normal = srp_facet_normals[index];
                let normal_norm = norm3(normal);
                if normal_norm <= 0.0 {
                    None
                } else {
                    Some(normal.map(|item| item / normal_norm))
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut q = quat;
    let mut w = omega;
    let mut aggregate_stats = [0_i64; 6];
    for dt_s in substeps_s {
        let c_bn = quaternion_to_dcm_bn(q);
        let mut disturbance_torque = [0.0; 3];
        if let Some((r_hat, scale)) = gravity_gradient {
            let r_hat_b = mat_vec3(c_bn, r_hat);
            let inertia_r_hat = mat_vec3(inertia, r_hat_b);
            let component = cross3(r_hat_b, inertia_r_hat).map(|item| item * scale);
            for axis in 0..3 {
                disturbance_torque[axis] += component[axis];
            }
        }
        if let Some(field) = magnetic_field {
            let component = cross3(magnetic_dipole_body, mat_vec3(c_bn, field));
            for axis in 0..3 {
                disturbance_torque[axis] += component[axis];
            }
        }
        if drag_active {
            let v_rel_body = mat_vec3(c_bn, drag_v_rel_eci_m_s);
            let component = if let Some(force_mag) = drag_force_magnitude {
                let force_body = v_rel_body.map(|item| -force_mag * item / drag_v_rel_norm_m_s);
                cross3(drag_cp_offset_body, force_body)
            } else {
                let mut torque = [0.0; 3];
                let v_hat_body = v_rel_body.map(|item| item / drag_v_rel_norm_m_s);
                for index in 0..drag_facet_areas.len() {
                    let Some(normal) = drag_unit_normals[index] else {
                        continue;
                    };
                    let projected_cosine = dot3(normal, v_hat_body);
                    let projected_area = drag_facet_areas[index] * projected_cosine.max(0.0);
                    if projected_area <= 0.0 {
                        continue;
                    }
                    let scale = -0.5
                        * density_kg_m3
                        * projected_area
                        * drag_facet_cd[index]
                        * drag_speed_squared;
                    let force_body = v_hat_body.map(|item| scale * item);
                    let component = cross3(drag_facet_cp_offsets[index], force_body);
                    for axis in 0..3 {
                        torque[axis] += component[axis];
                    }
                }
                torque
            };
            for axis in 0..3 {
                disturbance_torque[axis] += component[axis];
            }
        }
        if srp_active {
            let sun_dir_body = mat_vec3(c_bn, sun_dir_eci_unit);
            let component = if srp_mode == FACET_MODE_SCALAR {
                let force_body = sun_dir_body.map(|item| srp_scalar_scale * item);
                cross3(srp_cp_offset_body, force_body)
            } else {
                let mut torque = [0.0; 3];
                for index in 0..srp_facet_areas.len() {
                    let Some(normal) = srp_unit_normals[index] else {
                        continue;
                    };
                    let illumination = dot3(normal, sun_dir_body).max(0.0);
                    if illumination <= 0.0 {
                        continue;
                    }
                    let scale = -srp_pressure_scaled_n_m2 * srp_facet_areas[index] * illumination;
                    let force_body = sun_dir_body.map(|item| scale * item);
                    let component = cross3(srp_facet_cp_offsets[index], force_body);
                    for axis in 0..3 {
                        torque[axis] += component[axis];
                    }
                }
                torque
            };
            for axis in 0..3 {
                disturbance_torque[axis] += component[axis];
            }
        }
        let total_torque =
            std::array::from_fn(|axis| command_torque[axis] + disturbance_torque[axis]);
        let (next_q, next_w, stats) =
            propagate_attitude_exponential_map(q, w, inertia, total_torque, *dt_s);
        q = next_q;
        w = next_w;
        for index in 0..6 {
            aggregate_stats[index] += stats[index];
        }
    }
    (q, w, aggregate_stats)
}

#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, prelude::*};

#[cfg(feature = "python")]
#[pyclass(frozen, module = "oel_rust_orbit")]
pub(crate) struct BuiltinDisturbanceContext {
    fixed: [f64; 54],
    drag_normals: Vec<[f64; 3]>,
    drag_areas: Vec<f64>,
    drag_coefficients: Vec<f64>,
    drag_offsets: Vec<[f64; 3]>,
    srp_normals: Vec<[f64; 3]>,
    srp_areas: Vec<f64>,
    srp_offsets: Vec<[f64; 3]>,
}

#[cfg(feature = "python")]
#[pymethods]
impl BuiltinDisturbanceContext {
    #[new]
    fn new(data: &[u8], drag_count: usize, srp_count: usize) -> PyResult<Self> {
        let expected = drag_count
            .checked_mul(8)
            .and_then(|n| srp_count.checked_mul(7).and_then(|m| n.checked_add(m)))
            .and_then(|n| n.checked_add(54))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| PyValueError::new_err("disturbance buffer dimensions overflow"))?;
        if data.len() != expected {
            return Err(PyValueError::new_err(
                "disturbance buffer dimensions are inconsistent",
            ));
        }
        let values = crate::numeric_bytes::decode_f64_bytes(data)?;
        if !(0.0..=255.0).contains(&values[39]) || !(0.0..=255.0).contains(&values[49]) {
            return Err(PyValueError::new_err("facet modes must fit u8"));
        }
        let mut cursor = 54;
        let drag_normals: Vec<[f64; 3]> = values[cursor..cursor + drag_count * 3]
            .chunks_exact(3)
            .map(|v| v.try_into().unwrap())
            .collect();
        cursor += drag_count * 3;
        let drag_areas = values[cursor..cursor + drag_count].to_vec();
        cursor += drag_count;
        let drag_coefficients = values[cursor..cursor + drag_count].to_vec();
        cursor += drag_count;
        let drag_offsets: Vec<[f64; 3]> = values[cursor..cursor + drag_count * 3]
            .chunks_exact(3)
            .map(|v| v.try_into().unwrap())
            .collect();
        cursor += drag_count * 3;
        let srp_normals: Vec<[f64; 3]> = values[cursor..cursor + srp_count * 3]
            .chunks_exact(3)
            .map(|v| v.try_into().unwrap())
            .collect();
        cursor += srp_count * 3;
        let srp_areas = values[cursor..cursor + srp_count].to_vec();
        cursor += srp_count;
        let srp_offsets: Vec<[f64; 3]> = values[cursor..]
            .chunks_exact(3)
            .map(|v| v.try_into().unwrap())
            .collect();
        Ok(Self {
            fixed: values[..54].try_into().unwrap(),
            drag_normals,
            drag_areas,
            drag_coefficients,
            drag_offsets,
            srp_normals,
            srp_areas,
            srp_offsets,
        })
    }

    fn step<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        substep_bytes: &[u8],
    ) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, [i64; 6])> {
        if data.len() != 26 * 8 {
            return Err(PyValueError::new_err(
                "dynamic disturbance buffer must have 26 values",
            ));
        }
        let dynamic = crate::numeric_bytes::decode_f64_bytes(data)?;
        let mut values = self.fixed;
        values[..7].copy_from_slice(&dynamic[..7]);
        values[16..22].copy_from_slice(&dynamic[7..13]);
        values[30..39].copy_from_slice(&dynamic[13..22]);
        values[45..49].copy_from_slice(&dynamic[22..26]);
        self.run(py, &values, substep_bytes)
    }
}

#[cfg(feature = "python")]
impl BuiltinDisturbanceContext {
    pub(crate) fn torque_for_stage(
        &self,
        quat: [f64; 4],
        position: [f64; 3],
        dynamic: [f64; 13],
    ) -> [f64; 3] {
        let v = &self.fixed;
        let inertia = [
            v[7..10].try_into().unwrap(),
            v[10..13].try_into().unwrap(),
            v[13..16].try_into().unwrap(),
        ];
        builtin_disturbance_torque(
            quat,
            position,
            inertia,
            v[22],
            std::array::from_fn(|i| v[23 + i] as i64),
            v[27..30].try_into().unwrap(),
            dynamic[..3].try_into().unwrap(),
            dynamic[3] != 0.0,
            dynamic[4],
            dynamic[5..8].try_into().unwrap(),
            dynamic[8],
            v[39] as u8,
            v[40],
            v[41],
            v[42..45].try_into().unwrap(),
            &self.drag_normals,
            &self.drag_areas,
            &self.drag_coefficients,
            &self.drag_offsets,
            dynamic[9..12].try_into().unwrap(),
            dynamic[12],
            v[49] as u8,
            v[50],
            v[51..54].try_into().unwrap(),
            &self.srp_normals,
            &self.srp_areas,
            &self.srp_offsets,
        )
    }

    fn run<'py>(
        &self,
        py: Python<'py>,
        values: &[f64; 54],
        substep_bytes: &[u8],
    ) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, [i64; 6])> {
        let substeps = crate::numeric_bytes::decode_f64_bytes(substep_bytes)?;
        let quat = &values[0..4];
        let omega = &values[4..7];
        let inertia = &values[7..16];
        let command_torque = &values[16..19];
        let position = &values[19..22];
        let mu = values[22];
        let enabled = &values[23..27];
        let dipole = &values[27..30];
        let field = &values[30..33];
        let provided = values[33];
        let density = values[34];
        let drag_velocity = &values[35..38];
        let drag_norm = values[38];
        let drag_mode = values[39];
        let drag_area = values[40];
        let drag_cd = values[41];
        let drag_cp = &values[42..45];
        let sun = &values[45..48];
        let pressure = values[48];
        let srp_mode = values[49];
        let srp_area = values[50];
        let srp_cp = &values[51..54];
        let inertia = [
            inertia[..3].try_into().unwrap(),
            inertia[3..6].try_into().unwrap(),
            inertia[6..].try_into().unwrap(),
        ];
        let (q_next, w_next, counts) = propagate_builtin_disturbances(
            quat.try_into().unwrap(),
            omega.try_into().unwrap(),
            inertia,
            command_torque.try_into().unwrap(),
            &substeps,
            position.try_into().unwrap(),
            mu,
            std::array::from_fn(|i| enabled[i] as i64),
            dipole.try_into().unwrap(),
            field.try_into().unwrap(),
            provided != 0.0,
            density,
            drag_velocity.try_into().unwrap(),
            drag_norm,
            drag_mode as u8,
            drag_area,
            drag_cd,
            drag_cp.try_into().unwrap(),
            &self.drag_normals,
            &self.drag_areas,
            &self.drag_coefficients,
            &self.drag_offsets,
            sun.try_into().unwrap(),
            pressure,
            srp_mode as u8,
            srp_area,
            srp_cp.try_into().unwrap(),
            &self.srp_normals,
            &self.srp_areas,
            &self.srp_offsets,
        );
        let output = [
            q_next[0], q_next[1], q_next[2], q_next[3], w_next[0], w_next[1], w_next[2],
        ];
        Ok((
            crate::numeric_bytes::output_bytes(py, &output)?.into_bound(py),
            counts,
        ))
    }
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::prelude::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    fn fixed<const N: usize>(values: Vec<f64>, name: &str) -> PyResult<[f64; N]> {
        values
            .try_into()
            .map_err(|_| PyValueError::new_err(format!("{name} must have {N} values")))
    }

    fn matrix3(values: Vec<f64>, name: &str) -> PyResult<[[f64; 3]; 3]> {
        let values: [f64; 9] = values
            .try_into()
            .map_err(|_| PyValueError::new_err(format!("{name} must have 9 values")))?;
        Ok([
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
            [values[6], values[7], values[8]],
        ])
    }

    fn facets<const N: usize>(values: Vec<f64>, name: &str) -> PyResult<Vec<[f64; N]>> {
        if values.len() % N != 0 {
            return Err(PyValueError::new_err(format!(
                "{name} must be a multiple of {N}"
            )));
        }
        Ok(values
            .chunks_exact(N)
            .map(|chunk| std::array::from_fn(|index| chunk[index]))
            .collect())
    }

    /// One packed boundary for the tiny fixed-size rigid body kernel.
    #[pyfunction]
    fn attitude_step_packed<'py>(
        py: Python<'py>,
        data: &[u8],
        dt_s: Option<f64>,
    ) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, [i64; 6])> {
        if data.len() != 19 * 8 {
            return Err(PyValueError::new_err(
                "attitude input must contain 19 f64 values",
            ));
        }
        if dt_s.is_some_and(|dt| !dt.is_finite()) {
            return Err(PyValueError::new_err("dt_s must be finite"));
        }
        let values: [f64; 19] = std::array::from_fn(|i| {
            f64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap())
        });
        let q = values[..4].try_into().unwrap();
        let w = values[4..7].try_into().unwrap();
        let inertia = [
            values[7..10].try_into().unwrap(),
            values[10..13].try_into().unwrap(),
            values[13..16].try_into().unwrap(),
        ];
        let torque = values[16..19].try_into().unwrap();
        let (q_next, w_next, counts) = match dt_s {
            Some(dt) => propagate_attitude_exponential_map(q, w, inertia, torque, dt),
            None => rigid_body_derivatives(q, w, inertia, torque),
        };
        let mut bytes = [0_u8; 7 * 8];
        for (i, value) in q_next.iter().chain(w_next.iter()).enumerate() {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&value.to_le_bytes());
        }
        Ok((pyo3::types::PyBytes::new(py, &bytes), counts))
    }

    #[pyfunction]
    fn attitude_rigid_body_derivatives(
        quat: Vec<f64>,
        omega: Vec<f64>,
        inertia: Vec<f64>,
        torque: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i64>)> {
        let (q_dot, omega_dot, stats) = rigid_body_derivatives(
            fixed(quat, "quaternion")?,
            fixed(omega, "body rate")?,
            matrix3(inertia, "inertia")?,
            fixed(torque, "torque")?,
        );
        Ok((q_dot.to_vec(), omega_dot.to_vec(), stats.to_vec()))
    }

    #[pyfunction]
    fn attitude_propagate_exponential_map(
        quat: Vec<f64>,
        omega: Vec<f64>,
        inertia: Vec<f64>,
        torque: Vec<f64>,
        dt_s: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i64>)> {
        if !dt_s.is_finite() {
            return Err(PyValueError::new_err("dt_s must be finite"));
        }
        let (q_next, omega_next, stats) = propagate_attitude_exponential_map(
            fixed(quat, "quaternion")?,
            fixed(omega, "body rate")?,
            matrix3(inertia, "inertia")?,
            fixed(torque, "torque")?,
            dt_s,
        );
        Ok((q_next.to_vec(), omega_next.to_vec(), stats.to_vec()))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyfunction]
    fn attitude_builtin_disturbance_torque(
        quat: Vec<f64>,
        position_eci_km: Vec<f64>,
        inertia: Vec<f64>,
        mu_km3_s2: f64,
        enabled: Vec<i64>,
        magnetic_dipole_body: Vec<f64>,
        magnetic_field_eci: Vec<f64>,
        magnetic_field_provided: bool,
        density_kg_m3: f64,
        drag_v_rel_eci_m_s: Vec<f64>,
        drag_v_rel_norm_m_s: f64,
        drag_mode: u8,
        drag_area_m2: f64,
        drag_cd: f64,
        drag_cp_offset_body: Vec<f64>,
        drag_facet_normals: Vec<f64>,
        drag_facet_areas: Vec<f64>,
        drag_facet_cd: Vec<f64>,
        drag_facet_cp_offsets: Vec<f64>,
        sun_dir_eci_unit: Vec<f64>,
        srp_pressure_scaled_n_m2: f64,
        srp_mode: u8,
        srp_area_m2: f64,
        srp_cp_offset_body: Vec<f64>,
        srp_facet_normals: Vec<f64>,
        srp_facet_areas: Vec<f64>,
        srp_facet_cp_offsets: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let enabled = fixed::<4>(
            enabled.into_iter().map(|value| value as f64).collect(),
            "enabled",
        )?;
        let drag_normals = facets::<3>(drag_facet_normals, "drag facet normals")?;
        let drag_offsets = facets::<3>(drag_facet_cp_offsets, "drag facet offsets")?;
        let srp_normals = facets::<3>(srp_facet_normals, "SRP facet normals")?;
        let srp_offsets = facets::<3>(srp_facet_cp_offsets, "SRP facet offsets")?;
        if drag_normals.len() != drag_facet_areas.len()
            || drag_normals.len() != drag_facet_cd.len()
            || drag_normals.len() != drag_offsets.len()
            || srp_normals.len() != srp_facet_areas.len()
            || srp_normals.len() != srp_offsets.len()
        {
            return Err(PyValueError::new_err(
                "facet arrays must have matching lengths",
            ));
        }
        let torque = builtin_disturbance_torque(
            fixed(quat, "quaternion")?,
            fixed(position_eci_km, "position")?,
            matrix3(inertia, "inertia")?,
            mu_km3_s2,
            enabled.map(|value| value as i64),
            fixed(magnetic_dipole_body, "magnetic dipole")?,
            fixed(magnetic_field_eci, "magnetic field")?,
            magnetic_field_provided,
            density_kg_m3,
            fixed(drag_v_rel_eci_m_s, "drag velocity")?,
            drag_v_rel_norm_m_s,
            drag_mode,
            drag_area_m2,
            drag_cd,
            fixed(drag_cp_offset_body, "drag center of pressure")?,
            &drag_normals,
            &drag_facet_areas,
            &drag_facet_cd,
            &drag_offsets,
            fixed(sun_dir_eci_unit, "sun direction")?,
            srp_pressure_scaled_n_m2,
            srp_mode,
            srp_area_m2,
            fixed(srp_cp_offset_body, "SRP center of pressure")?,
            &srp_normals,
            &srp_facet_areas,
            &srp_offsets,
        );
        Ok(torque.to_vec())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyfunction]
    fn attitude_propagate_builtin_disturbances(
        quat: Vec<f64>,
        omega: Vec<f64>,
        inertia: Vec<f64>,
        command_torque: Vec<f64>,
        substeps_s: Vec<f64>,
        position_eci_km: Vec<f64>,
        mu_km3_s2: f64,
        enabled: Vec<i64>,
        magnetic_dipole_body: Vec<f64>,
        magnetic_field_eci: Vec<f64>,
        magnetic_field_provided: bool,
        density_kg_m3: f64,
        drag_v_rel_eci_m_s: Vec<f64>,
        drag_v_rel_norm_m_s: f64,
        drag_mode: u8,
        drag_area_m2: f64,
        drag_cd: f64,
        drag_cp_offset_body: Vec<f64>,
        drag_facet_normals: Vec<f64>,
        drag_facet_areas: Vec<f64>,
        drag_facet_cd: Vec<f64>,
        drag_facet_cp_offsets: Vec<f64>,
        sun_dir_eci_unit: Vec<f64>,
        srp_pressure_scaled_n_m2: f64,
        srp_mode: u8,
        srp_area_m2: f64,
        srp_cp_offset_body: Vec<f64>,
        srp_facet_normals: Vec<f64>,
        srp_facet_areas: Vec<f64>,
        srp_facet_cp_offsets: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i64>)> {
        let enabled = fixed::<4>(
            enabled.into_iter().map(|value| value as f64).collect(),
            "enabled",
        )?
        .map(|value| value as i64);
        let drag_normals = facets::<3>(drag_facet_normals, "drag facet normals")?;
        let drag_offsets = facets::<3>(drag_facet_cp_offsets, "drag facet offsets")?;
        let srp_normals = facets::<3>(srp_facet_normals, "SRP facet normals")?;
        let srp_offsets = facets::<3>(srp_facet_cp_offsets, "SRP facet offsets")?;
        if drag_normals.len() != drag_facet_areas.len()
            || drag_normals.len() != drag_facet_cd.len()
            || drag_normals.len() != drag_offsets.len()
            || srp_normals.len() != srp_facet_areas.len()
            || srp_normals.len() != srp_offsets.len()
        {
            return Err(PyValueError::new_err(
                "facet arrays must have matching lengths",
            ));
        }
        let (q_next, omega_next, stats) = propagate_builtin_disturbances(
            fixed(quat, "quaternion")?,
            fixed(omega, "body rate")?,
            matrix3(inertia, "inertia")?,
            fixed(command_torque, "command torque")?,
            &substeps_s,
            fixed(position_eci_km, "position")?,
            mu_km3_s2,
            enabled,
            fixed(magnetic_dipole_body, "magnetic dipole")?,
            fixed(magnetic_field_eci, "magnetic field")?,
            magnetic_field_provided,
            density_kg_m3,
            fixed(drag_v_rel_eci_m_s, "drag velocity")?,
            drag_v_rel_norm_m_s,
            drag_mode,
            drag_area_m2,
            drag_cd,
            fixed(drag_cp_offset_body, "drag center of pressure")?,
            &drag_normals,
            &drag_facet_areas,
            &drag_facet_cd,
            &drag_offsets,
            fixed(sun_dir_eci_unit, "sun direction")?,
            srp_pressure_scaled_n_m2,
            srp_mode,
            srp_area_m2,
            fixed(srp_cp_offset_body, "SRP center of pressure")?,
            &srp_normals,
            &srp_facet_areas,
            &srp_offsets,
        );
        Ok((q_next.to_vec(), omega_next.to_vec(), stats.to_vec()))
    }

    #[pyfunction]
    fn attitude_propagate_disturbances_packed<'py>(
        py: Python<'py>,
        data: &[u8],
        substep_bytes: &[u8],
        drag_count: usize,
        srp_count: usize,
    ) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, [i64; 6])> {
        let context = BuiltinDisturbanceContext::new(data, drag_count, srp_count)?;
        context.run(py, &context.fixed, substep_bytes)
    }

    module.add_class::<BuiltinDisturbanceContext>()?;
    module.add_function(wrap_pyfunction!(
        attitude_propagate_disturbances_packed,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(attitude_step_packed, module)?)?;
    module.add_function(wrap_pyfunction!(attitude_rigid_body_derivatives, module)?)?;
    module.add_function(wrap_pyfunction!(
        attitude_propagate_exponential_map,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        attitude_builtin_disturbance_torque,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        attitude_propagate_builtin_disturbances,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inertia() -> [[f64; 3]; 3] {
        [[120.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 80.0]]
    }

    #[test]
    fn prepared_disturbance_call_matches_original_stage_arithmetic_exactly() {
        // Each iteration changes environment, coefficients, facet normals and
        // guardrail inputs. The reference independently calls the unchanged
        // torque API at every substep, followed by the unchanged expmap API.
        for case in 0..8 {
            let position = if case == 3 {
                [0.0; 3]
            } else {
                [6800.0 + case as f64, -120.0, 250.0]
            };
            let initial_q = if case == 7 {
                [f64::NAN, 0.0, 0.0, 0.0]
            } else {
                [0.9, 0.2, -0.1, 0.3]
            };
            let initial_w = [0.01, -0.02, 0.03];
            let mut tensor = inertia();
            if case == 6 {
                tensor = [[0.0; 3]; 3];
            }
            let command = if case == 7 {
                [f64::NAN, 0.002, -0.001]
            } else {
                [0.001, 0.002, -0.001]
            };
            let enabled = if case == 4 { [0, 1, 0, 1] } else { [1; 4] };
            let dipole = [0.04 + case as f64 * 0.01, -0.01, 0.02];
            let field = [1.5e-5, -2.0e-5 - case as f64 * 1.0e-6, 3.0e-5];
            let field_provided = case % 2 == 0;
            let density = if case == 5 {
                0.0
            } else {
                1.2e-12 + case as f64 * 1.0e-13
            };
            let wind = [1200.0, -300.0 + case as f64, 80.0];
            let wind_norm = if case == 5 { 0.0 } else { norm3(wind) };
            let mode = if case % 2 == 0 { FACET_MODE_SCALAR } else { 2 };
            let drag_area = 1.5 + case as f64 * 0.1;
            let cd = 2.2;
            let drag_offset = [0.05, 0.02, -0.01];
            let normals = [[1.0 + case as f64, 0.2, 0.0], [0.0; 3], [-0.2, 2.0, 0.3]];
            let areas = [1.0, 0.2, 0.7];
            let coefficients = [2.2, 2.1, 2.3];
            let offsets = [[0.05, 0.02, -0.01], [0.0; 3], [-0.01, 0.04, 0.02]];
            let sun = [0.8, 0.2, 0.56 + case as f64 * 0.01];
            let pressure = if case == 5 {
                0.0
            } else {
                4.56e-6 * 0.82 * 0.97 * 1.3
            };
            let srp_area = 1.0 + case as f64 * 0.1;
            let srp_offset = [-0.02, 0.03, 0.01];
            let mut substeps = [0.01; 200];
            if case == 7 {
                substeps[1] = 0.0;
                substeps[2] = -0.01;
                substeps[3] = f64::NAN;
            }
            let actual = propagate_builtin_disturbances(
                initial_q,
                initial_w,
                tensor,
                command,
                &substeps,
                position,
                398600.4418,
                enabled,
                dipole,
                field,
                field_provided,
                density,
                wind,
                wind_norm,
                mode,
                drag_area,
                cd,
                drag_offset,
                &normals,
                &areas,
                &coefficients,
                &offsets,
                sun,
                pressure,
                mode,
                srp_area,
                srp_offset,
                &normals,
                &areas,
                &offsets,
            );
            let (mut q, mut w, mut counts) = (initial_q, initial_w, [0; 6]);
            for dt in substeps {
                let disturbance = builtin_disturbance_torque(
                    q,
                    position,
                    tensor,
                    398600.4418,
                    enabled,
                    dipole,
                    field,
                    field_provided,
                    density,
                    wind,
                    wind_norm,
                    mode,
                    drag_area,
                    cd,
                    drag_offset,
                    &normals,
                    &areas,
                    &coefficients,
                    &offsets,
                    sun,
                    pressure,
                    mode,
                    srp_area,
                    srp_offset,
                    &normals,
                    &areas,
                    &offsets,
                );
                let total = std::array::from_fn(|axis| command[axis] + disturbance[axis]);
                let (next_q, next_w, stats) =
                    propagate_attitude_exponential_map(q, w, tensor, total, dt);
                q = next_q;
                w = next_w;
                for index in 0..6 {
                    counts[index] += stats[index];
                }
            }
            assert_eq!(actual, (q, w, counts), "changing input case {case}");
        }
    }

    #[test]
    fn constant_rate_preserves_unit_quaternion() {
        let (q, w, stats) = propagate_attitude_exponential_map(
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            inertia(),
            [0.0; 3],
            10.0,
        );
        assert!((q.iter().map(|item| item * item).sum::<f64>() - 1.0).abs() < 1.0e-14);
        assert_eq!(w, [0.0, 0.0, 0.1]);
        assert_eq!(stats, [0; 6]);
    }

    #[test]
    fn singular_inertia_is_sanitized() {
        let (_, _, stats) = propagate_attitude_exponential_map(
            [1.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [[0.0; 3]; 3],
            [1.0, 0.0, 0.0],
            1.0,
        );
        assert_eq!(stats[STAT_SINGULAR_INERTIA], 1);
    }

    #[test]
    fn scalar_disturbance_order_is_deterministic() {
        let torque = builtin_disturbance_torque(
            [1.0, 0.0, 0.0, 0.0],
            [7000.0, 0.0, 0.0],
            inertia(),
            398600.4418,
            [1, 1, 1, 1],
            [0.05, 0.0, 0.0],
            [2.0e-5, -1.0e-5, 3.0e-5],
            true,
            1.0e-12,
            [1000.0, 0.0, 0.0],
            1000.0,
            FACET_MODE_SCALAR,
            1.5,
            2.2,
            [0.05, 0.02, -0.01],
            &[],
            &[],
            &[],
            &[],
            [1.0, 0.0, 0.0],
            4.56e-6,
            FACET_MODE_SCALAR,
            1.0,
            [-0.02, 0.03, 0.01],
            &[],
            &[],
            &[],
        );
        assert!(finite3(&torque));
    }
}
