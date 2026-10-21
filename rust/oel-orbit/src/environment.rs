//! Small, data-free ONP environment kernels.
//!
//! The Python layer remains responsible for loading EOP/DE440 resources and
//! applying their provenance and coverage policy.  This module only evaluates
//! numerical kernels over already prepared values.  That boundary lets the
//! native path preserve Python callbacks and fail-closed resource handling.

#[cfg(feature = "python")]
use pyo3::prelude::*;

const DAY_S: f64 = 86_400.0;
const J2000: f64 = 2_451_545.0;
const WGS84_A_KM: f64 = 6_378.137;
const WGS84_F: f64 = 1.0 / 298.257_223_563;
const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);
const WGS84_B_KM: f64 = WGS84_A_KM * (1.0 - WGS84_F);
pub(crate) const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3600.0);
const JULIAN_CENTURY_DAYS: f64 = 36_525.0;

fn validate_finite(values: &[f64], label: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} must contain only finite values"));
    }
    Ok(())
}

fn gmst_angle_rad(jd_utc: f64) -> f64 {
    let t = (jd_utc - J2000) / 36_525.0;
    let theta_deg = 280.460_618_37 + 360.985_647_366_29 * (jd_utc - J2000) + 0.000_387_933 * t * t
        - t * t * t / 38_710_000.0;
    // Match Python's ``mod(theta_deg, 360)`` before converting to radians.
    // Reducing after conversion loses a few ulps for multi-million-degree
    // absolute epochs and makes frame parity needlessly worse.
    theta_deg.rem_euclid(360.0).to_radians()
}

fn rotation_z(angle: f64) -> [f64; 9] {
    let (s, c) = angle.sin_cos();
    [c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0]
}

pub(crate) fn mat_vec(matrix: &[f64; 9], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    ]
}

pub(crate) fn mat_mul(left: &[f64; 9], right: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for row in 0..3 {
        for inner in 0..3 {
            let value = left[row * 3 + inner];
            for column in 0..3 {
                out[row * 3 + column] += value * right[inner * 3 + column];
            }
        }
    }
    out
}

fn frame_rx(angle: f64) -> [f64; 9] {
    let (sine, cosine) = angle.sin_cos();
    [1.0, 0.0, 0.0, 0.0, cosine, sine, 0.0, -sine, cosine]
}

pub(crate) fn frame_ry(angle: f64) -> [f64; 9] {
    let (sine, cosine) = angle.sin_cos();
    [cosine, 0.0, -sine, 0.0, 1.0, 0.0, sine, 0.0, cosine]
}

pub(crate) fn frame_rz(angle: f64) -> [f64; 9] {
    rotation_z(angle)
}

fn signed_fmod(value: f64, modulus: f64) -> f64 {
    value % modulus
}

fn validate_nutation_table(coefficients: &[f64], terms: &[f64]) -> Result<usize, String> {
    if coefficients.len() % 5 != 0 || terms.len() % 4 != 0 {
        return Err("IAU-80 nutation tables must contain 5 and 4 values per row".to_owned());
    }
    let rows = coefficients.len() / 5;
    if rows == 0 || rows != terms.len() / 4 {
        return Err("IAU-80 nutation table row counts must match and be nonzero".to_owned());
    }
    validate_finite(coefficients, "nutation coefficients")?;
    validate_finite(terms, "nutation terms")?;
    Ok(rows)
}

fn iau80_nutation_angles(
    jd_tt: f64,
    ddpsi_rad: f64,
    ddeps_rad: f64,
    nutation_coefficients: &[f64],
    nutation_terms: &[f64],
    rows: usize,
) -> Result<(f64, f64, f64, f64), String> {
    if !jd_tt.is_finite() || !ddpsi_rad.is_finite() || !ddeps_rad.is_finite() {
        return Err("IAU-80 epoch and corrections must be finite".to_owned());
    }
    let centuries_tt = (jd_tt - J2000) / JULIAN_CENTURY_DAYS;
    let centuries_tt2 = centuries_tt * centuries_tt;
    let centuries_tt3 = centuries_tt2 * centuries_tt;
    let fundamental_degrees = [
        ((0.064 * centuries_tt + 31.310) * centuries_tt + 1_717_915_922.6330) * centuries_tt
            / 3600.0
            + 134.96298139,
        ((-0.012 * centuries_tt - 0.577) * centuries_tt + 129_596_581.2240) * centuries_tt / 3600.0
            + 357.52772333,
        ((0.011 * centuries_tt - 13.257) * centuries_tt + 1_739_527_263.1370) * centuries_tt
            / 3600.0
            + 93.27191028,
        ((0.019 * centuries_tt - 6.891) * centuries_tt + 1_602_961_601.3280) * centuries_tt
            / 3600.0
            + 297.85036306,
        ((0.008 * centuries_tt + 7.455) * centuries_tt - 6_962_890.5390) * centuries_tt / 3600.0
            + 125.04452222,
    ];
    let mut fundamental_arguments = [0.0; 5];
    for index in 0..5 {
        fundamental_arguments[index] = fundamental_degrees[index].rem_euclid(360.0).to_radians();
    }
    let mut delta_psi = 0.0;
    let mut delta_epsilon = 0.0;
    for row in 0..rows {
        let mut phase = 0.0;
        for column in 0..5 {
            phase += nutation_coefficients[row * 5 + column] * fundamental_arguments[column];
        }
        let (phase_sin, phase_cos) = phase.sin_cos();
        delta_psi +=
            (nutation_terms[row * 4] + nutation_terms[row * 4 + 1] * centuries_tt) * phase_sin;
        delta_epsilon +=
            (nutation_terms[row * 4 + 2] + nutation_terms[row * 4 + 3] * centuries_tt) * phase_cos;
    }
    delta_psi = signed_fmod(delta_psi + ddpsi_rad, 2.0 * std::f64::consts::PI);
    delta_epsilon = signed_fmod(delta_epsilon + ddeps_rad, 2.0 * std::f64::consts::PI);
    let mean_epsilon = ((-46.8150 * centuries_tt - 0.00059 * centuries_tt2
        + 0.001813 * centuries_tt3
        + 84381.448)
        / 3600.0)
        .rem_euclid(360.0)
        .to_radians();
    let true_epsilon = mean_epsilon + delta_epsilon;
    Ok((centuries_tt, delta_psi, true_epsilon, mean_epsilon))
}

/// Full OEL IAU-76/FK5 + IAU-80 EOP rotation.  EOP interpolation and
/// coverage/leap-second policy remain in Python; this function consumes the
/// already validated scalar EOP values and the immutable nutation table.
pub fn iau76_80_rotation(
    t_s: f64,
    jd_utc_start: f64,
    xp_arcsec: f64,
    yp_arcsec: f64,
    dut1_s: f64,
    dat_s: f64,
    ddpsi_rad: f64,
    ddeps_rad: f64,
    nutation_coefficients: &[f64],
    nutation_terms: &[f64],
) -> Result<[f64; 9], String> {
    let rows = validate_nutation_table(nutation_coefficients, nutation_terms)?;
    iau76_80_rotation_with_rows(
        t_s,
        jd_utc_start,
        xp_arcsec,
        yp_arcsec,
        dut1_s,
        dat_s,
        ddpsi_rad,
        ddeps_rad,
        nutation_coefficients,
        nutation_terms,
        rows,
    )
}

fn iau76_80_rotation_with_rows(
    t_s: f64,
    jd_utc_start: f64,
    xp_arcsec: f64,
    yp_arcsec: f64,
    dut1_s: f64,
    dat_s: f64,
    ddpsi_rad: f64,
    ddeps_rad: f64,
    nutation_coefficients: &[f64],
    nutation_terms: &[f64],
    nutation_rows: usize,
) -> Result<[f64; 9], String> {
    if [
        t_s,
        jd_utc_start,
        xp_arcsec,
        yp_arcsec,
        dut1_s,
        dat_s,
        ddpsi_rad,
        ddeps_rad,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err("IAU-76/80 frame inputs must be finite".to_owned());
    }
    let jd_utc = jd_utc_start + t_s / DAY_S;
    let jd_tt = jd_utc + (dat_s + 32.184) / DAY_S;
    let (centuries_tt, delta_psi, true_epsilon, mean_epsilon) = iau80_nutation_angles(
        jd_tt,
        ddpsi_rad,
        ddeps_rad,
        nutation_coefficients,
        nutation_terms,
        nutation_rows,
    )?;
    let centuries_tt2 = centuries_tt * centuries_tt;
    let centuries_tt3 = centuries_tt2 * centuries_tt;
    let zeta = (2306.2181 * centuries_tt + 0.30188 * centuries_tt2 + 0.017998 * centuries_tt3)
        * ARCSEC_TO_RAD;
    let theta = (2004.3109 * centuries_tt - 0.42665 * centuries_tt2 - 0.041833 * centuries_tt3)
        * ARCSEC_TO_RAD;
    let z = (2306.2181 * centuries_tt + 1.09468 * centuries_tt2 + 0.018203 * centuries_tt3)
        * ARCSEC_TO_RAD;
    let precession = mat_mul(&mat_mul(&frame_rz(-z), &frame_ry(theta)), &frame_rz(-zeta));
    let nutation = mat_mul(
        &mat_mul(&frame_rx(-true_epsilon), &frame_rz(-delta_psi)),
        &frame_rx(mean_epsilon),
    );
    let rbpn = mat_mul(&nutation, &precession);
    // IERS 1996, p. 21 (IAU 1982 GMST-UT1). Preserve sub-JD elapsed
    // seconds and DUT1 instead of rounding them into one large epoch.
    let offset_days = (t_s + dut1_s) / DAY_S;
    let centuries_ut1 = ((jd_utc_start - J2000) + offset_days) / JULIAN_CENTURY_DAYS;
    let fractional_seconds = DAY_S * (jd_utc_start % 1.0 + offset_days % 1.0);
    let gmst_seconds = 24110.54841 - DAY_S / 2.0
        + (8640184.812866 + (0.093104 - 6.2e-6 * centuries_ut1) * centuries_ut1)
            * centuries_ut1
        + fractional_seconds;
    let gmst = (gmst_seconds * (2.0 * std::f64::consts::PI / DAY_S))
        .rem_euclid(2.0 * std::f64::consts::PI);
    // IAU 1994 resolution C7, effective 1997-02-27 00:00 UTC (TAI-UTC=30s). The
    // equation of equinoxes uses mean, not true, obliquity.
    let complementary = if jd_tt >= 2450506.5 + 62.184 / DAY_S {
        let omega = (((0.008 * centuries_tt + 7.455) * centuries_tt - 6_962_890.5390)
            * centuries_tt / 3600.0 + 125.04452222)
            .rem_euclid(360.0).to_radians();
        (0.00264 * omega.sin() + 0.000063 * (2.0 * omega).sin()) * ARCSEC_TO_RAD
    } else {
        0.0
    };
    let gast = (gmst + delta_psi * mean_epsilon.cos() + complementary)
        .rem_euclid(2.0 * std::f64::consts::PI);
    let sp = -47.0e-6 * ((jd_tt - J2000) / JULIAN_CENTURY_DAYS) * ARCSEC_TO_RAD;
    let polar_motion = mat_mul(
        &mat_mul(&frame_rz(sp), &frame_ry(-xp_arcsec * ARCSEC_TO_RAD)),
        &frame_rx(-yp_arcsec * ARCSEC_TO_RAD),
    );
    Ok(mat_mul(&mat_mul(&polar_motion, &frame_rz(gast)), &rbpn))
}

/// Simple OEL ECI->ECEF rotation, matching `gmst_angle_rad_from_jd`.
pub fn simple_rotation(
    t_s: f64,
    jd_utc_start: Option<f64>,
    earth_rotation_rad_s: f64,
) -> Result<[f64; 9], String> {
    if !t_s.is_finite() || !earth_rotation_rad_s.is_finite() {
        return Err("rotation time and Earth rate must be finite".to_owned());
    }
    if let Some(jd) = jd_utc_start {
        if !jd.is_finite() {
            return Err("jd_utc_start must be finite".to_owned());
        }
        Ok(rotation_z(gmst_angle_rad(jd + t_s / DAY_S)))
    } else {
        Ok(rotation_z(earth_rotation_rad_s * t_s))
    }
}

/// Evaluate a batch of simple ECI->ECEF rotations.
pub fn rotation_batch(
    times_s: &[f64],
    jd_utc_start: Option<f64>,
    earth_rotation_rad_s: f64,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(times_s.len() * 9);
    for &time in times_s {
        out.extend_from_slice(&simple_rotation(time, jd_utc_start, earth_rotation_rad_s)?);
    }
    Ok(out)
}

/// Simple-frame relative velocity with the reference rotation/subtraction order.
/// Positions and velocities are consumed freshly for every RK stage.
pub fn simple_atmosphere_relative_velocity(
    state: [f64; 6],
    t_s: f64,
    jd_utc_start: Option<f64>,
    atmosphere_rotation_rad_s: f64,
) -> Result<[f64; 3], String> {
    validate_finite(&state, "state")?;
    if !atmosphere_rotation_rad_s.is_finite() {
        return Err("atmosphere rotation rate must be finite".to_owned());
    }
    // The reference simple frame rotates at the canonical Earth rate even
    // when an explicitly configured atmosphere rate differs from that rate.
    let rotation = simple_rotation(t_s, jd_utc_start, 7.292_115_9e-5)?;
    let fixed_position = mat_vec(&rotation, [state[0], state[1], state[2]]);
    let mut fixed_velocity = mat_vec(&rotation, [state[3], state[4], state[5]]);
    fixed_velocity[0] -= -atmosphere_rotation_rad_s * fixed_position[1];
    fixed_velocity[1] -= atmosphere_rotation_rad_s * fixed_position[0];
    Ok(std::array::from_fn(|axis| {
        rotation[axis] * fixed_velocity[0]
            + rotation[3 + axis] * fixed_velocity[1]
            + rotation[6 + axis] * fixed_velocity[2]
    }))
}

/// Transform synchronized ECI position/velocity rows to ECEF.
///
/// When `subtract_atmosphere_rotation` is true, velocity is the velocity
/// relative to a stationary atmosphere.  This is the exact state transform
/// used by OEL's simple-frame atmosphere path.
pub fn eci_to_ecef_state_batch(
    times_s: &[f64],
    positions_eci_km: &[f64],
    velocities_eci_km_s: &[f64],
    jd_utc_start: Option<f64>,
    earth_rotation_rad_s: f64,
    subtract_atmosphere_rotation: bool,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if positions_eci_km.len() != times_s.len() * 3 || velocities_eci_km_s.len() != times_s.len() * 3
    {
        return Err("ECI state batch dimensions do not match times".to_owned());
    }
    validate_finite(times_s, "times")?;
    validate_finite(positions_eci_km, "positions")?;
    validate_finite(velocities_eci_km_s, "velocities")?;
    let mut positions = Vec::with_capacity(positions_eci_km.len());
    let mut velocities = Vec::with_capacity(velocities_eci_km_s.len());
    for index in 0..times_s.len() {
        let rotation = simple_rotation(times_s[index], jd_utc_start, earth_rotation_rad_s)?;
        let r = [
            positions_eci_km[3 * index],
            positions_eci_km[3 * index + 1],
            positions_eci_km[3 * index + 2],
        ];
        let v = [
            velocities_eci_km_s[3 * index],
            velocities_eci_km_s[3 * index + 1],
            velocities_eci_km_s[3 * index + 2],
        ];
        let r_ecef = mat_vec(&rotation, r);
        let mut v_ecef = mat_vec(&rotation, v);
        if subtract_atmosphere_rotation {
            v_ecef[0] -= -earth_rotation_rad_s * r_ecef[1];
            v_ecef[1] -= earth_rotation_rad_s * r_ecef[0];
        }
        positions.extend_from_slice(&r_ecef);
        velocities.extend_from_slice(&v_ecef);
    }
    Ok((positions, velocities))
}

/// Convert ECEF positions to WGS-84 geodetic latitude/longitude/altitude.
pub fn ecef_to_geodetic_deg_km(position: [f64; 3]) -> Result<[f64; 3], String> {
    validate_finite(&position, "ECEF position")?;
    let [x, y, z] = position;
    let p = x.hypot(y);
    let lon = y.atan2(x);
    if p <= 1.0e-12 {
        let lat = if z >= 0.0 {
            std::f64::consts::FRAC_PI_2
        } else {
            -std::f64::consts::FRAC_PI_2
        };
        return Ok([lat.to_degrees(), lon.to_degrees(), z.abs() - WGS84_B_KM]);
    }
    let mut lat = z.atan2(p * (1.0 - WGS84_E2));
    for _ in 0..8 {
        let sin_lat = lat.sin();
        let prime_vertical = WGS84_A_KM / (1.0 - WGS84_E2 * sin_lat * sin_lat).max(1.0e-15).sqrt();
        let altitude = p / lat.cos().abs().max(1.0e-15) - prime_vertical;
        let next = z.atan2(
            p * (1.0 - WGS84_E2 * prime_vertical / (prime_vertical + altitude).max(1.0e-15)),
        );
        if (next - lat).abs() <= 1.0e-13 {
            lat = next;
            break;
        }
        lat = next;
    }
    let sin_lat = lat.sin();
    let prime_vertical = WGS84_A_KM / (1.0 - WGS84_E2 * sin_lat * sin_lat).max(1.0e-15).sqrt();
    let altitude = p / lat.cos().abs().max(1.0e-15) - prime_vertical;
    Ok([lat.to_degrees(), lon.to_degrees(), altitude])
}

pub fn ecef_to_geodetic_batch(positions_ecef_km: &[f64]) -> Result<Vec<f64>, String> {
    if positions_ecef_km.len() % 3 != 0 {
        return Err("ECEF position batch must contain groups of three values".to_owned());
    }
    let mut out = Vec::with_capacity(positions_ecef_km.len());
    for chunk in positions_ecef_km.chunks_exact(3) {
        out.extend_from_slice(&ecef_to_geodetic_deg_km([chunk[0], chunk[1], chunk[2]])?);
    }
    Ok(out)
}

/// Exponential atmosphere density kernel.  Model policy and custom models
/// remain in Python; this is the repeated scalar arithmetic only.
pub fn exponential_density(
    altitude_km: f64,
    reference_density_kg_m3: f64,
    reference_altitude_km: f64,
    scale_height_km: f64,
    ceiling_km: f64,
) -> Result<f64, String> {
    if ![
        altitude_km,
        reference_density_kg_m3,
        reference_altitude_km,
        scale_height_km,
        ceiling_km,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        return Err("exponential atmosphere inputs must be finite".to_owned());
    }
    if reference_density_kg_m3 < 0.0 || scale_height_km <= 0.0 || ceiling_km < 0.0 {
        return Err("exponential atmosphere parameters are outside their valid range".to_owned());
    }
    if altitude_km > ceiling_km {
        return Ok(0.0);
    }
    Ok(reference_density_kg_m3 * (-(altitude_km - reference_altitude_km) / scale_height_km).exp())
}

/// Radial-altitude exponential density, with position evaluated for this state.
pub fn radial_exponential_density(
    position_km: [f64; 3],
    radius_km: f64,
    reference_density_kg_m3: f64,
    reference_altitude_km: f64,
    scale_height_km: f64,
    ceiling_km: f64,
) -> Result<f64, String> {
    if !position_km.iter().all(|value| value.is_finite())
        || !radius_km.is_finite()
        || radius_km <= 0.0
    {
        return Err(
            "radial density position and radius must be finite with positive radius".to_owned(),
        );
    }
    let altitude = ((position_km[0] * position_km[0]
        + position_km[1] * position_km[1]
        + position_km[2] * position_km[2])
        .sqrt()
        - radius_km)
        .max(0.0);
    exponential_density(
        altitude,
        reference_density_kg_m3,
        reference_altitude_km,
        scale_height_km,
        ceiling_km,
    )
}

/// The checked-in JB2006/JB2008 lower-atmosphere quadrature.  This retains
/// every subdivision and four-node evaluation of the Python reference.
pub fn jb_lower_atmosphere(zht: f64, tc: [f64; 4]) -> Result<[f64; 9], String> {
    if !zht.is_finite() || zht < 90.0 || tc.iter().any(|v| !v.is_finite()) {
        return Err(
            "JB lower quadrature needs finite inputs and altitude at least 90 km".to_owned(),
        );
    }
    const C: [f64; 7] = [
        28.15204, -8.5586e-2, 1.2840e-4, -1.0056e-5, -1.0210e-5, 1.5044e-6, 9.9826e-8,
    ];
    const WT: [f64; 5] = [
        0.311111111111111,
        1.422222222222222,
        0.533333333333333,
        1.422222222222222,
        0.311111111111111,
    ];
    let molecular_weight = |z: f64| {
        let dz = z - 100.0;
        let mut value = C[6];
        for &coefficient in C[..6].iter().rev() {
            value = dz * value + coefficient;
        }
        value
    };
    let gravity = |z: f64| 9.80665 / (1.0 + z / 6356.766).powi(2);
    let temperature = |z: f64| {
        let dz = z - 125.0;
        if dz > 0.0 {
            tc[0] + tc[2] * (tc[3] * dz * (1.0 + 4.5e-6 * dz.powf(2.5))).atan()
        } else {
            (((-9.8204695e-6 * dz - 7.3039742e-4) * dz * dz + 1.0) * dz * tc[1]) + tc[0]
        }
    };
    let z1 = 90.0;
    let z2 = zht.min(105.0);
    let al = (z2 / z1).ln();
    let n = (al / 0.010).floor() as usize + 1;
    let zr = (al / n as f64).exp();
    let ambar1 = molecular_weight(z1);
    let tloc1 = temperature(z1);
    let mut zend = z1;
    let mut z = z1;
    let mut sum2 = 0.0;
    let mut ain = ambar1 * gravity(z1) / tloc1;
    let mut ambar2 = ambar1;
    let mut tloc2 = tloc1;
    let mut gravl = gravity(z1);
    for _ in 0..n {
        z = zend;
        zend = zr * z;
        let dz = 0.25 * (zend - z);
        let mut sum1 = WT[0] * ain;
        for weight in &WT[1..] {
            z += dz;
            let zm100 = z - 100.0;
            ambar2 = zm100 * C[6] + C[5];
            ambar2 = zm100 * ambar2 + C[4];
            ambar2 = zm100 * ambar2 + C[3];
            ambar2 = zm100 * ambar2 + C[2];
            ambar2 = zm100 * ambar2 + C[1];
            ambar2 = zm100 * ambar2 + C[0];
            tloc2 = temperature(z);
            gravl = gravity(z);
            ain = ambar2 * gravl / tloc2;
            sum1 += weight * ain;
        }
        sum2 += dz * sum1;
    }
    Ok([zend, z, tloc1, tloc2, ambar1, ambar2, gravl, ain, sum2])
}

/// Shared JB2006/JB2008 four-node upper-atmosphere quadrature.
pub fn jb_upper_atmosphere(
    mut zend: f64, z_current: f64, z_target: f64, r_step: f64,
    mut ain: f64, tc: [f64; 4],
) -> Result<[f64; 6], String> {
    if ![zend, z_current, z_target, r_step, ain].iter().chain(tc.iter()).all(|v| v.is_finite())
        || zend <= 0.0 || z_current <= 0.0 || z_target <= 0.0 || r_step <= 0.0
        || z_target < z_current {
        return Err("JB upper quadrature needs finite increasing positive altitude inputs".into());
    }
    const WT: [f64; 5] = [0.311111111111111, 1.422222222222222,
                          0.533333333333333, 1.422222222222222, 0.311111111111111];
    let al = (z_target / z_current).ln();
    let n = (al / r_step).floor() as usize + 1;
    if n > 1_000_000 { return Err("JB upper quadrature subdivision limit exceeded".into()); }
    let zr = (al / n as f64).exp();
    let mut total = 0.0;
    let mut z = zend;
    let mut tloc = tc[0];
    let mut gravl = 0.0;
    for _ in 0..n {
        z = zend;
        zend = zr * z;
        let dz = 0.25 * (zend - z);
        let mut sum1 = WT[0] * ain;
        for weight in &WT[1..] {
            z += dz;
            let delta = z - 125.0;
            tloc = if delta > 0.0 {
                tc[0] + tc[2] * (tc[3] * delta * (1.0 + 4.5e-6 * delta.powf(2.5))).atan()
            } else {
                (((-9.8204695e-6 * delta - 7.3039742e-4) * delta * delta + 1.0)
                    * delta * tc[1]) + tc[0]
            };
            gravl = 9.80665 / (1.0 + z / 6356.766).powi(2);
            ain = gravl / tloc;
            sum1 += weight * ain;
        }
        total += dz * sum1;
    }
    Ok([zend, z, tloc, gravl, ain, total])
}

fn chebyshev_value(t: f64, coeffs: &[f64]) -> Result<f64, String> {
    if coeffs.is_empty() || coeffs.iter().any(|value| !value.is_finite()) {
        return Err("Chebyshev coefficients must be finite and nonempty".to_owned());
    }
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    for &coefficient in coeffs.iter().skip(1).rev() {
        let old_f1 = f1;
        f1 = 2.0 * t * f1 - f2 + coefficient;
        f2 = old_f1;
    }
    Ok(t * f1 - f2 + coeffs[0])
}

/// Evaluate one DE440 Chebyshev subinterval.  Coefficients are supplied by
/// Python after the resource has been loaded and validated.
pub fn de440_chebyshev_position(
    jd_tdb: f64,
    ta: f64,
    tb: f64,
    x_coefficients: &[f64],
    y_coefficients: &[f64],
    z_coefficients: &[f64],
) -> Result<[f64; 3], String> {
    if ![jd_tdb, ta, tb].iter().all(|value| value.is_finite()) || tb <= ta {
        return Err("DE440 Chebyshev interval must be finite and increasing".to_owned());
    }
    if jd_tdb < ta
        || jd_tdb > tb
        || x_coefficients.len() != y_coefficients.len()
        || x_coefficients.len() != z_coefficients.len()
    {
        return Err("DE440 Chebyshev input shape or coverage is invalid".to_owned());
    }
    let tau = (2.0 * jd_tdb - ta - tb) / (tb - ta);
    Ok([
        1.0e3 * chebyshev_value(tau, x_coefficients)?,
        1.0e3 * chebyshev_value(tau, y_coefficients)?,
        1.0e3 * chebyshev_value(tau, z_coefficients)?,
    ])
}

/// Evaluate a flattened DE440 light-body table over many epochs.
pub fn de440_chebyshev_batch(
    times_jd_tdb: &[f64],
    row_starts_jd_tdb: &[f64],
    row_ends_jd_tdb: &[f64],
    coeff_count: usize,
    segments: usize,
    span_days: f64,
    x_rows: &[f64],
    y_rows: &[f64],
    z_rows: &[f64],
) -> Result<Vec<f64>, String> {
    if row_starts_jd_tdb.len() != row_ends_jd_tdb.len()
        || coeff_count == 0
        || segments == 0
        || !span_days.is_finite()
        || span_days <= 0.0
    {
        return Err("DE440 table metadata is invalid".to_owned());
    }
    let row_width = coeff_count
        .checked_mul(segments)
        .ok_or("DE440 table dimensions overflow")?;
    let expected = row_width
        .checked_mul(row_starts_jd_tdb.len())
        .ok_or("DE440 table dimensions overflow")?;
    if x_rows.len() != expected || y_rows.len() != expected || z_rows.len() != expected {
        return Err("DE440 table coefficient shape is invalid".to_owned());
    }
    let mut out = Vec::with_capacity(times_jd_tdb.len() * 3);
    for &jd in times_jd_tdb {
        if !jd.is_finite() {
            return Err("DE440 query epochs must be finite".to_owned());
        }
        let row = row_starts_jd_tdb
            .iter()
            .zip(row_ends_jd_tdb)
            .position(|(&start, &end)| start <= jd && jd <= end)
            .ok_or_else(|| format!("JD_TDB {jd} is outside DE440 light coverage"))?;
        let t1 = row_starts_jd_tdb[row];
        let segment_length = span_days / segments as f64;
        let mut segment = ((jd - t1) / segment_length).floor() as isize;
        segment = segment.clamp(0, segments as isize - 1);
        let ta = t1 + segment_length * segment as f64;
        let tb = ta + segment_length;
        let offset = row * row_width + segment as usize * coeff_count;
        let x = de440_chebyshev_position(
            jd,
            ta,
            tb,
            &x_rows[offset..offset + coeff_count],
            &y_rows[offset..offset + coeff_count],
            &z_rows[offset..offset + coeff_count],
        )?;
        out.extend_from_slice(&x);
    }
    Ok(out)
}

#[cfg(feature = "python")]
#[cfg(feature = "python")]
#[derive(Clone)]
struct De440BodyTable {
    row_starts_jd_tdb: Vec<f64>,
    row_ends_jd_tdb: Vec<f64>,
    coeff_count: usize,
    segments: usize,
    span_days: f64,
    x_rows: Vec<f64>,
    y_rows: Vec<f64>,
    z_rows: Vec<f64>,
}

#[cfg(feature = "python")]
#[pyclass(name = "ONPEnvironmentContext")]
pub struct NativeEnvironmentContext {
    nutation_coefficients: Vec<f64>,
    nutation_terms: Vec<f64>,
    nutation_rows: usize,
    de440_bodies: Vec<Option<De440BodyTable>>,
}

#[cfg(feature = "python")]
impl NativeEnvironmentContext {
    pub(crate) fn analytic_bodies(
        &self,
        jd: f64,
        enhanced: bool,
    ) -> PyResult<([f64; 3], [f64; 3])> {
        self.analytic_positions(jd, enhanced, false)
    }

    fn analytic_positions(
        &self,
        jd: f64,
        enhanced: bool,
        sun_only: bool,
    ) -> PyResult<([f64; 3], [f64; 3])> {
        let nutation = if enhanced {
            let (_, psi, trueeps, meaneps) = iau80_nutation_angles(
                jd + 69.184 / 86400.,
                0.,
                0.,
                &self.nutation_coefficients,
                &self.nutation_terms,
                self.nutation_rows,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let (sp, cp) = psi.sin_cos();
            let (se, ce) = meaneps.sin_cos();
            let (st, ct) = trueeps.sin_cos();
            Some([
                cp,
                ct * sp,
                st * sp,
                -ce * sp,
                ct * ce * cp + st * se,
                st * ce * cp - se * ct,
                -se * sp,
                ct * se * cp - st * ce,
                st * se * cp + ct * ce,
            ])
        } else {
            None
        };
        Ok(crate::analytic_bodies::bodies(
            jd, enhanced, nutation, sun_only,
        ))
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl NativeEnvironmentContext {
    #[new]
    fn new(nutation_coefficients: Vec<f64>, nutation_terms: Vec<f64>) -> PyResult<Self> {
        let nutation_rows = validate_nutation_table(&nutation_coefficients, &nutation_terms)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            nutation_coefficients,
            nutation_terms,
            nutation_rows,
            de440_bodies: vec![None, None, None],
        })
    }

    /// Same analytic ephemeris as the force owner, with caller-selected NUT80 resources.
    fn analytic_sun_km(&self, jd_utc: f64, enhanced: bool) -> PyResult<Vec<f64>> {
        if !jd_utc.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Sun epoch must be finite.",
            ));
        }
        Ok(self.analytic_positions(jd_utc, enhanced, true)?.0.to_vec())
    }

    pub(crate) fn rotation_iau76_80(
        &self,
        t_s: f64,
        jd_utc_start: f64,
        xp_arcsec: f64,
        yp_arcsec: f64,
        dut1_s: f64,
        dat_s: f64,
        ddpsi_rad: f64,
        ddeps_rad: f64,
    ) -> PyResult<Vec<f64>> {
        iau76_80_rotation_with_rows(
            t_s,
            jd_utc_start,
            xp_arcsec,
            yp_arcsec,
            dut1_s,
            dat_s,
            ddpsi_rad,
            ddeps_rad,
            &self.nutation_coefficients,
            &self.nutation_terms,
            self.nutation_rows,
        )
        .map(|rotation| rotation.to_vec())
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_de440_body(
        &mut self,
        body_index: usize,
        row_starts_jd_tdb: Vec<f64>,
        row_ends_jd_tdb: Vec<f64>,
        coeff_count: usize,
        segments: usize,
        span_days: f64,
        x_rows: Vec<f64>,
        y_rows: Vec<f64>,
        z_rows: Vec<f64>,
    ) -> PyResult<()> {
        if body_index >= 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "DE440 body index must be 0 (earthmoon), 1 (moon), or 2 (sun)",
            ));
        }
        let row_width = coeff_count.checked_mul(segments).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("DE440 table dimensions overflow")
        })?;
        let expected = row_width
            .checked_mul(row_starts_jd_tdb.len())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("DE440 table dimensions overflow")
            })?;
        if row_starts_jd_tdb.len() != row_ends_jd_tdb.len()
            || coeff_count == 0
            || segments == 0
            || !span_days.is_finite()
            || span_days <= 0.0
            || x_rows.len() != expected
            || y_rows.len() != expected
            || z_rows.len() != expected
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "DE440 body table metadata or coefficient shape is invalid",
            ));
        }
        validate_finite(&row_starts_jd_tdb, "DE440 row starts")
            .and_then(|_| validate_finite(&row_ends_jd_tdb, "DE440 row ends"))
            .and_then(|_| validate_finite(&x_rows, "DE440 x coefficients"))
            .and_then(|_| validate_finite(&y_rows, "DE440 y coefficients"))
            .and_then(|_| validate_finite(&z_rows, "DE440 z coefficients"))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        self.de440_bodies[body_index] = Some(De440BodyTable {
            row_starts_jd_tdb,
            row_ends_jd_tdb,
            coeff_count,
            segments,
            span_days,
            x_rows,
            y_rows,
            z_rows,
        });
        Ok(())
    }

    pub(crate) fn de440_sun_moon_km(
        &self,
        jd_tdb: f64,
        earth_moon_mass_ratio: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if !jd_tdb.is_finite() || !earth_moon_mass_ratio.is_finite() || earth_moon_mass_ratio <= 0.0
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "DE440 epoch and Earth-Moon mass ratio must be finite and valid",
            ));
        }
        let positions_m = (0..3)
            .map(|index| {
                let table = self.de440_bodies[index]
                    .as_ref()
                    .ok_or_else(|| format!("DE440 body table {index} is not configured"))?;
                let values = de440_chebyshev_batch(
                    &[jd_tdb],
                    &table.row_starts_jd_tdb,
                    &table.row_ends_jd_tdb,
                    table.coeff_count,
                    table.segments,
                    table.span_days,
                    &table.x_rows,
                    &table.y_rows,
                    &table.z_rows,
                )?;
                Ok::<[f64; 3], String>([values[0], values[1], values[2]])
            })
            .collect::<Result<Vec<[f64; 3]>, String>>()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let earth: [f64; 3] = std::array::from_fn(|axis| {
            positions_m[0][axis] - positions_m[1][axis] / (1.0 + earth_moon_mass_ratio)
        });
        let sun: [f64; 3] =
            std::array::from_fn(|axis| (positions_m[2][axis] - earth[axis]) / 1.0e3);
        let moon: [f64; 3] = std::array::from_fn(|axis| positions_m[1][axis] / 1.0e3);
        Ok((sun.to_vec(), moon.to_vec()))
    }
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::prelude::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    fn environment_rotation_batch(
        times_s: Vec<f64>,
        jd_utc_start: Option<f64>,
        earth_rotation_rad_s: f64,
    ) -> PyResult<Vec<f64>> {
        rotation_batch(&times_s, jd_utc_start, earth_rotation_rad_s).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn environment_state_batch(
        times_s: Vec<f64>,
        positions_eci_km: Vec<f64>,
        velocities_eci_km_s: Vec<f64>,
        jd_utc_start: Option<f64>,
        earth_rotation_rad_s: f64,
        subtract_atmosphere_rotation: bool,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        eci_to_ecef_state_batch(
            &times_s,
            &positions_eci_km,
            &velocities_eci_km_s,
            jd_utc_start,
            earth_rotation_rad_s,
            subtract_atmosphere_rotation,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn environment_geodetic_batch(positions_ecef_km: Vec<f64>) -> PyResult<Vec<f64>> {
        ecef_to_geodetic_batch(&positions_ecef_km).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn environment_exponential_density(
        altitude_km: f64,
        reference_density_kg_m3: f64,
        reference_altitude_km: f64,
        scale_height_km: f64,
        ceiling_km: f64,
    ) -> PyResult<f64> {
        exponential_density(
            altitude_km,
            reference_density_kg_m3,
            reference_altitude_km,
            scale_height_km,
            ceiling_km,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn environment_radial_exponential_density(
        position_km: [f64; 3],
        radius_km: f64,
        reference_density_kg_m3: f64,
        reference_altitude_km: f64,
        scale_height_km: f64,
        ceiling_km: f64,
    ) -> PyResult<f64> {
        radial_exponential_density(
            position_km,
            radius_km,
            reference_density_kg_m3,
            reference_altitude_km,
            scale_height_km,
            ceiling_km,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn environment_simple_relative_velocity(
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        time_s: f64,
        jd_utc_start: Option<f64>,
        atmosphere_rotation_rad_s: f64,
    ) -> PyResult<Vec<f64>> {
        simple_atmosphere_relative_velocity(
            [
                position_km[0],
                position_km[1],
                position_km[2],
                velocity_km_s[0],
                velocity_km_s[1],
                velocity_km_s[2],
            ],
            time_s,
            jd_utc_start,
            atmosphere_rotation_rad_s,
        )
        .map(|values| values.to_vec())
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn environment_radial_exponential_density_batch<'py>(
        py: Python<'py>,
        positions_km: &[u8],
        radius_km: f64,
        reference_density_kg_m3: f64,
        reference_altitude_km: f64,
        scale_height_km: f64,
        ceiling_km: f64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if positions_km.len() % 24 != 0 || positions_km.len() / 24 > 10_000_000 {
            return Err(PyValueError::new_err(
                "radial density batch must contain at most 10000000 float64 3-vectors",
            ));
        }
        // Validate parameters even for an empty batch.
        exponential_density(
            0.0,
            reference_density_kg_m3,
            reference_altitude_km,
            scale_height_km,
            ceiling_km,
        )
        .map_err(PyValueError::new_err)?;
        if !radius_km.is_finite() || radius_km <= 0.0 {
            return Err(PyValueError::new_err(
                "radial density radius must be finite and positive",
            ));
        }
        let mut out = Vec::with_capacity(positions_km.len() / 3);
        for row in positions_km.chunks_exact(24) {
            let position = std::array::from_fn(|axis| {
                f64::from_le_bytes(row[axis * 8..axis * 8 + 8].try_into().unwrap())
            });
            let rho = radial_exponential_density(
                position,
                radius_km,
                reference_density_kg_m3,
                reference_altitude_km,
                scale_height_km,
                ceiling_km,
            )
            .map_err(PyValueError::new_err)?;
            out.extend_from_slice(&rho.to_le_bytes());
        }
        Ok(PyBytes::new(py, &out))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn environment_simple_batch<'py>(
        py: Python<'py>,
        times_s: &[u8],
        states_eci: &[u8],
        jd_utc_start: Option<f64>,
        earth_rotation_rad_s: f64,
        subtract_atmosphere_rotation: bool,
        radius_km: f64,
        reference_density_kg_m3: f64,
        reference_altitude_km: f64,
        scale_height_km: f64,
        ceiling_km: f64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let count = times_s.len() / 8;
        if times_s.len() % 8 != 0
            || states_eci.len() % 48 != 0
            || count > 10_000_000
            || states_eci.len() / 48 != count
        {
            return Err(PyValueError::new_err("simple environment batch must contain matching float64 times and 6-value states, at most 10000000 rows"));
        }
        // Apply the same scalar policy to empty batches as nonempty batches.
        simple_rotation(0.0, jd_utc_start, earth_rotation_rad_s).map_err(PyValueError::new_err)?;
        radial_exponential_density(
            [0.0; 3],
            radius_km,
            reference_density_kg_m3,
            reference_altitude_km,
            scale_height_km,
            ceiling_km,
        )
        .map_err(PyValueError::new_err)?;
        let mut out = Vec::with_capacity(count * 19 * 8);
        for index in 0..count {
            let time = f64::from_le_bytes(times_s[index * 8..index * 8 + 8].try_into().unwrap());
            let row = &states_eci[index * 48..index * 48 + 48];
            let state: [f64; 6] = std::array::from_fn(|axis| {
                f64::from_le_bytes(row[axis * 8..axis * 8 + 8].try_into().unwrap())
            });
            validate_finite(&state, "states").map_err(PyValueError::new_err)?;
            let position = [state[0], state[1], state[2]];
            let rotation = simple_rotation(time, jd_utc_start, earth_rotation_rad_s)
                .map_err(PyValueError::new_err)?;
            let fixed_position = mat_vec(&rotation, position);
            let mut fixed_velocity = mat_vec(&rotation, [state[3], state[4], state[5]]);
            if subtract_atmosphere_rotation {
                fixed_velocity[0] -= -earth_rotation_rad_s * fixed_position[1];
                fixed_velocity[1] -= earth_rotation_rad_s * fixed_position[0];
            }
            let geodetic =
                ecef_to_geodetic_deg_km(fixed_position).map_err(PyValueError::new_err)?;
            let density = radial_exponential_density(
                position,
                radius_km,
                reference_density_kg_m3,
                reference_altitude_km,
                scale_height_km,
                ceiling_km,
            )
            .map_err(PyValueError::new_err)?;
            for value in rotation
                .into_iter()
                .chain(fixed_position)
                .chain(fixed_velocity)
                .chain(geodetic)
                .chain([density])
            {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(PyBytes::new(py, &out))
    }

    #[pyfunction]
    fn environment_jb_lower_atmosphere(
        altitude_km: f64,
        temperature_coefficients: [f64; 4],
    ) -> PyResult<Vec<f64>> {
        jb_lower_atmosphere(altitude_km, temperature_coefficients)
            .map(|values| values.to_vec())
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn environment_jb_upper_atmosphere(
        zend: f64, z_current: f64, z_target: f64, r_step: f64,
        ain: f64, temperature_coefficients: [f64; 4],
    ) -> PyResult<Vec<f64>> {
        jb_upper_atmosphere(zend, z_current, z_target, r_step, ain, temperature_coefficients)
            .map(|values| values.to_vec())
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn environment_de440_chebyshev_batch(
        times_jd_tdb: Vec<f64>,
        row_starts_jd_tdb: Vec<f64>,
        row_ends_jd_tdb: Vec<f64>,
        coeff_count: usize,
        segments: usize,
        span_days: f64,
        x_rows: Vec<f64>,
        y_rows: Vec<f64>,
        z_rows: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        de440_chebyshev_batch(
            &times_jd_tdb,
            &row_starts_jd_tdb,
            &row_ends_jd_tdb,
            coeff_count,
            segments,
            span_days,
            &x_rows,
            &y_rows,
            &z_rows,
        )
        .map_err(PyValueError::new_err)
    }

    module.add_function(wrap_pyfunction!(environment_rotation_batch, module)?)?;
    module.add_function(wrap_pyfunction!(environment_state_batch, module)?)?;
    module.add_function(wrap_pyfunction!(environment_geodetic_batch, module)?)?;
    module.add_function(wrap_pyfunction!(environment_exponential_density, module)?)?;
    module.add_function(wrap_pyfunction!(
        environment_radial_exponential_density,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        environment_simple_relative_velocity,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        environment_radial_exponential_density_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(environment_simple_batch, module)?)?;
    module.add_function(wrap_pyfunction!(environment_jb_lower_atmosphere, module)?)?;
    module.add_function(wrap_pyfunction!(environment_jb_upper_atmosphere, module)?)?;
    module.add_function(wrap_pyfunction!(environment_de440_chebyshev_batch, module)?)?;
    module.add_class::<NativeEnvironmentContext>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{de440_chebyshev_batch, ecef_to_geodetic_deg_km, rotation_batch};

    #[test]
    fn simple_rotation_starts_at_identity_without_absolute_epoch() {
        let values = rotation_batch(&[0.0], None, 7.292_115_9e-5).unwrap();
        assert!((values[0] - 1.0).abs() < 1.0e-15);
        assert!((values[4] - 1.0).abs() < 1.0e-15);
        assert!(values[1].abs() < 1.0e-15);
        assert!(values[3].abs() < 1.0e-15);
    }

    #[test]
    fn geodetic_equator_matches_wgs84_radius() {
        let result = ecef_to_geodetic_deg_km([super::WGS84_A_KM, 0.0, 0.0]).unwrap();
        assert!(result[0].abs() < 1.0e-12);
        assert!(result[1].abs() < 1.0e-12);
        assert!(result[2].abs() < 1.0e-12);
    }

    #[test]
    fn chebyshev_batch_preserves_linear_coefficients() {
        let result = de440_chebyshev_batch(
            &[0.0, 1.0],
            &[-1.0],
            &[1.0],
            2,
            1,
            2.0,
            &[2.0, 3.0],
            &[4.0, 5.0],
            &[6.0, 7.0],
        )
        .unwrap();
        assert!((result[0] - 2.0e3).abs() < 1.0e-9);
        assert!((result[3] - 5.0e3).abs() < 1.0e-9);
    }
}
