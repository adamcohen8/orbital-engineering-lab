//! Numerical kernels for the optional launch-vehicle and re-entry backends.
//!
//! The surrounding OEL Python code owns atmosphere selection, frames, guidance,
//! event handling, stage objects, and output semantics.  This module only
//! evaluates the small, repeatedly executed scalar/vector kernels.  Units are
//! explicit in the function names and comments: aerodynamic quantities use SI
//! units, while no orbital state is handled here.

const G0_M_S2: f64 = 9.80665;

fn finite(values: &[f64], name: &str) -> Result<(), String> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(format!("{name} must contain only finite values"))
    }
}

fn vec3(values: &[f64], name: &str) -> Result<[f64; 3], String> {
    if values.len() != 3 {
        return Err(format!("{name} must have three values"));
    }
    finite(values, name)?;
    Ok([values[0], values[1], values[2]])
}

fn norm3(value: [f64; 3]) -> f64 {
    value[0].hypot(value[1]).hypot(value[2])
}

fn clamp(value: f64, low: f64, high: f64) -> f64 {
    value.max(low).min(high)
}

/// Compute the canonical rocket aerodynamic state.
///
/// The returned row is `[rho, pressure, temperature, sound_speed, q, speed,
/// mach, alpha, beta]`, matching the fields of `RocketAeroState` in Python.
pub fn aero_state(
    rho_kg_m3: f64,
    pressure_pa: f64,
    temperature_k: f64,
    sound_speed_m_s: f64,
    v_rel_body_m_s: &[f64],
    alpha_limit_deg: f64,
    beta_limit_deg: f64,
) -> Result<[f64; 9], String> {
    finite(
        &[
            rho_kg_m3,
            pressure_pa,
            temperature_k,
            sound_speed_m_s,
            alpha_limit_deg,
            beta_limit_deg,
        ],
        "aerodynamic inputs",
    )?;
    let velocity = vec3(v_rel_body_m_s, "v_rel_body_m_s")?;
    let speed = norm3(velocity);
    let q = 0.5 * rho_kg_m3 * speed * speed;
    let (mut alpha, mut beta, mut mach) = (0.0, 0.0, 0.0);
    if speed > 1.0e-12 {
        alpha = velocity[2].atan2(velocity[0].max(1.0e-12));
        beta = (velocity[1] / speed).clamp(-1.0, 1.0).asin();
        mach = speed / sound_speed_m_s.max(1.0e-6);
    }
    let alpha_limit = alpha_limit_deg.to_radians();
    let beta_limit = beta_limit_deg.to_radians();
    alpha = clamp(alpha, -alpha_limit, alpha_limit);
    beta = clamp(beta, -beta_limit, beta_limit);
    Ok([
        rho_kg_m3.max(0.0),
        pressure_pa.max(0.0),
        temperature_k.max(1.0),
        sound_speed_m_s.max(1.0e-3),
        q.max(0.0),
        speed.max(0.0),
        mach.max(0.0),
        alpha,
        beta,
    ])
}

/// Evaluate the rocket coefficient and load model.
///
/// The returned row is `[force(3), moment(3), force_coeff(3),
/// moment_coeff(3), cd]`, all in the same SI units as the Python model.
pub fn aero_loads(
    v_rel_body_m_s: &[f64],
    atmosphere: &[f64],
    enabled: bool,
    reference_area_m2: f64,
    reference_length_m: f64,
    cp_offset_body_m: &[f64],
    cd_base: f64,
    cd_alpha2: f64,
    cd_supersonic: f64,
    transonic_peak_cd: f64,
    transonic_mach: f64,
    transonic_width: f64,
    cl_alpha_per_rad: f64,
    cy_beta_per_rad: f64,
    cm_alpha_per_rad: f64,
    cn_beta_per_rad: f64,
    cl_roll_per_rad: f64,
) -> Result<[f64; 13], String> {
    if atmosphere.len() != 9 {
        return Err("atmosphere must contain nine values".to_owned());
    }
    finite(atmosphere, "atmosphere")?;
    finite(
        &[
            reference_area_m2,
            reference_length_m,
            cd_base,
            cd_alpha2,
            cd_supersonic,
            transonic_peak_cd,
            transonic_mach,
            transonic_width,
            cl_alpha_per_rad,
            cy_beta_per_rad,
            cm_alpha_per_rad,
            cn_beta_per_rad,
            cl_roll_per_rad,
        ],
        "aerodynamic coefficients",
    )?;
    let velocity = vec3(v_rel_body_m_s, "v_rel_body_m_s")?;
    let cp_offset = vec3(cp_offset_body_m, "cp_offset_body_m")?;
    let q = atmosphere[4];
    let mach = atmosphere[6];
    let alpha = atmosphere[7];
    let beta = atmosphere[8];
    if !enabled || q <= 0.0 {
        return Ok([0.0; 13]);
    }
    let m = mach.max(0.0);
    let cd_m = if m <= 1.0 {
        cd_base
    } else {
        cd_base + (cd_supersonic - cd_base) * (1.0 - (-(m - 1.0)).exp())
    };
    let width = transonic_width.max(1.0e-3);
    let bump = transonic_peak_cd * (-0.5 * ((m - transonic_mach) / width).powi(2)).exp();
    let cd = (cd_m + bump + cd_alpha2 * alpha * alpha).max(0.0);
    let speed = norm3(velocity);
    let flow_hat = if speed <= 1.0e-12 {
        [0.0; 3]
    } else {
        [
            velocity[0] / speed,
            velocity[1] / speed,
            velocity[2] / speed,
        ]
    };
    let cl = cl_alpha_per_rad * alpha;
    let cy = -cy_beta_per_rad * beta;
    let coeff_force = [
        -cd * flow_hat[0],
        -cd * flow_hat[1] + cy,
        -cd * flow_hat[2] - cl,
    ];
    let force = [
        q * reference_area_m2 * coeff_force[0],
        q * reference_area_m2 * coeff_force[1],
        q * reference_area_m2 * coeff_force[2],
    ];
    let coeff_moment = [
        cl_roll_per_rad * beta,
        cm_alpha_per_rad * alpha,
        cn_beta_per_rad * beta,
    ];
    let moment_coeff = q * reference_area_m2 * reference_length_m;
    let moment_base = [
        moment_coeff * coeff_moment[0],
        moment_coeff * coeff_moment[1],
        moment_coeff * coeff_moment[2],
    ];
    let offset_moment = [
        cp_offset[1] * force[2] - cp_offset[2] * force[1],
        cp_offset[2] * force[0] - cp_offset[0] * force[2],
        cp_offset[0] * force[1] - cp_offset[1] * force[0],
    ];
    let moment = [
        moment_base[0] + offset_moment[0],
        moment_base[1] + offset_moment[1],
        moment_base[2] + offset_moment[2],
    ];
    let mut result = [0.0; 13];
    result[0..3].copy_from_slice(&force);
    result[3..6].copy_from_slice(&moment);
    result[6..9].copy_from_slice(&coeff_force);
    result[9..12].copy_from_slice(&coeff_moment);
    result[12] = cd;
    Ok(result)
}

/// Interpolate stage thrust and specific impulse between sea-level and vacuum.
pub fn stage_engine_perf(
    pressure_pa: f64,
    sea_level_thrust_n: f64,
    vacuum_thrust_n: f64,
    sea_level_isp_s: f64,
    vacuum_isp_s: f64,
) -> Result<[f64; 2], String> {
    finite(
        &[
            pressure_pa,
            sea_level_thrust_n,
            vacuum_thrust_n,
            sea_level_isp_s,
            vacuum_isp_s,
        ],
        "stage performance inputs",
    )?;
    let p = clamp(pressure_pa, 0.0, 101325.0);
    let sea_weight = p / 101325.0;
    let vacuum_weight = 1.0 - sea_weight;
    let thrust = (sea_weight * sea_level_thrust_n + vacuum_weight * vacuum_thrust_n).max(0.0);
    let isp = (sea_weight * sea_level_isp_s + vacuum_weight * vacuum_isp_s).max(1.0e-9);
    Ok([thrust, isp])
}

/// Advance the propellant and mass portion of one rocket engine interval.
///
/// The row is `[thrust_n, isp_s, dm_prop_kg, burn_fraction, mass_for_thrust_kg,
/// remaining_prop_kg, remaining_mass_kg]`.  Dry-stage separation is deliberately
/// left to Python's event/state owner.
pub fn propellant_step(
    propellant_left_kg: f64,
    throttle: f64,
    pressure_pa: f64,
    dt_s: f64,
    mass_start_kg: f64,
    sea_level_thrust_n: f64,
    vacuum_thrust_n: f64,
    sea_level_isp_s: f64,
    vacuum_isp_s: f64,
) -> Result<[f64; 7], String> {
    finite(
        &[
            propellant_left_kg,
            throttle,
            pressure_pa,
            dt_s,
            mass_start_kg,
            sea_level_thrust_n,
            vacuum_thrust_n,
            sea_level_isp_s,
            vacuum_isp_s,
        ],
        "propellant-step inputs",
    )?;
    if propellant_left_kg < 0.0 || dt_s < 0.0 {
        return Err("propellant_left_kg and dt_s must be nonnegative".to_owned());
    }
    let perf = stage_engine_perf(
        pressure_pa,
        sea_level_thrust_n,
        vacuum_thrust_n,
        sea_level_isp_s,
        vacuum_isp_s,
    )?;
    let throttle = clamp(throttle, 0.0, 1.0);
    let full_thrust = throttle * perf[0];
    let mdot = full_thrust / (perf[1] * G0_M_S2).max(1.0e-9);
    let full_dm = mdot * dt_s;
    let dm = propellant_left_kg.min(full_dm);
    let burn_fraction = if full_dm <= 0.0 { 0.0 } else { dm / full_dm };
    let thrust = full_thrust * burn_fraction;
    let mass_for_thrust = (mass_start_kg - 0.5 * dm).max(1.0e-9);
    Ok([
        thrust,
        perf[1],
        dm,
        burn_fraction,
        mass_for_thrust,
        propellant_left_kg - dm,
        (mass_start_kg - dm).max(0.0),
    ])
}

/// Evaluate the scalar re-entry loads and trapezoidal heat accumulation.
///
/// The row is `[dynamic_pressure_pa, drag_decel_m_s2, lift_accel_m_s2,
/// lift_to_drag, g_load, heat_rate_w_m2, heat_load_j_m2]`.
pub fn reentry_metrics(
    density_kg_m3: f64,
    speed_m_s: f64,
    mass_kg: f64,
    drag_area_m2: f64,
    cd: f64,
    lift_area_m2: f64,
    cl: f64,
    nose_radius_m: f64,
    coefficient: f64,
    dt_s: f64,
    previous_heat_load_j_m2: f64,
    previous_heat_rate_w_m2: f64,
) -> Result<[f64; 7], String> {
    finite(
        &[
            density_kg_m3,
            speed_m_s,
            mass_kg,
            drag_area_m2,
            cd,
            lift_area_m2,
            cl,
            nose_radius_m,
            coefficient,
            dt_s,
            previous_heat_load_j_m2,
        ],
        "re-entry inputs",
    )?;
    if !previous_heat_rate_w_m2.is_nan() && !previous_heat_rate_w_m2.is_finite() {
        return Err("previous_heat_rate_w_m2 must be finite or NaN".to_owned());
    }
    let rho = density_kg_m3.max(0.0);
    let speed = speed_m_s.max(0.0);
    let q = 0.5 * rho * speed * speed;
    let mass = mass_kg.max(1.0e-12);
    let drag = q * cd.max(0.0) * drag_area_m2.max(0.0) / mass;
    let lift = q * lift_area_m2.max(0.0) * cl.abs() / mass;
    let lift_to_drag = if drag <= 0.0 { f64::NAN } else { lift / drag };
    let heat_rate = coefficient * (rho / nose_radius_m.max(1.0e-9)).sqrt() * speed.powi(3);
    let prior_rate = if previous_heat_rate_w_m2.is_nan() {
        heat_rate
    } else {
        previous_heat_rate_w_m2.max(0.0)
    };
    let heat_load =
        previous_heat_load_j_m2.max(0.0) + dt_s.max(0.0) * 0.5 * (prior_rate + heat_rate.max(0.0));
    Ok([
        q,
        drag,
        lift,
        lift_to_drag,
        (drag.hypot(lift) / G0_M_S2),
        heat_rate,
        heat_load,
    ])
}

#[cfg(feature = "python")]
mod python {
    use super::{
        aero_loads as compute_aero_loads, aero_state as compute_aero_state,
        propellant_step as compute_propellant_step, reentry_metrics as compute_reentry_metrics,
        stage_engine_perf as compute_stage_engine_perf,
    };
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyfunction]
    fn rocket_aero_state(
        rho_kg_m3: f64,
        pressure_pa: f64,
        temperature_k: f64,
        sound_speed_m_s: f64,
        v_rel_body_m_s: Vec<f64>,
        alpha_limit_deg: f64,
        beta_limit_deg: f64,
    ) -> PyResult<Vec<f64>> {
        compute_aero_state(
            rho_kg_m3,
            pressure_pa,
            temperature_k,
            sound_speed_m_s,
            &v_rel_body_m_s,
            alpha_limit_deg,
            beta_limit_deg,
        )
        .map(|values| values.to_vec())
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn rocket_aero_loads(
        v_rel_body_m_s: Vec<f64>,
        atmosphere: Vec<f64>,
        enabled: bool,
        reference_area_m2: f64,
        reference_length_m: f64,
        cp_offset_body_m: Vec<f64>,
        cd_base: f64,
        cd_alpha2: f64,
        cd_supersonic: f64,
        transonic_peak_cd: f64,
        transonic_mach: f64,
        transonic_width: f64,
        cl_alpha_per_rad: f64,
        cy_beta_per_rad: f64,
        cm_alpha_per_rad: f64,
        cn_beta_per_rad: f64,
        cl_roll_per_rad: f64,
    ) -> PyResult<Vec<f64>> {
        compute_aero_loads(
            &v_rel_body_m_s,
            &atmosphere,
            enabled,
            reference_area_m2,
            reference_length_m,
            &cp_offset_body_m,
            cd_base,
            cd_alpha2,
            cd_supersonic,
            transonic_peak_cd,
            transonic_mach,
            transonic_width,
            cl_alpha_per_rad,
            cy_beta_per_rad,
            cm_alpha_per_rad,
            cn_beta_per_rad,
            cl_roll_per_rad,
        )
        .map(|values| values.to_vec())
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn rocket_stage_engine_perf(
        pressure_pa: f64,
        sea_level_thrust_n: f64,
        vacuum_thrust_n: f64,
        sea_level_isp_s: f64,
        vacuum_isp_s: f64,
    ) -> PyResult<Vec<f64>> {
        compute_stage_engine_perf(
            pressure_pa,
            sea_level_thrust_n,
            vacuum_thrust_n,
            sea_level_isp_s,
            vacuum_isp_s,
        )
        .map(|values| values.to_vec())
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn rocket_propellant_step(
        propellant_left_kg: f64,
        throttle: f64,
        pressure_pa: f64,
        dt_s: f64,
        mass_start_kg: f64,
        sea_level_thrust_n: f64,
        vacuum_thrust_n: f64,
        sea_level_isp_s: f64,
        vacuum_isp_s: f64,
    ) -> PyResult<Vec<f64>> {
        compute_propellant_step(
            propellant_left_kg,
            throttle,
            pressure_pa,
            dt_s,
            mass_start_kg,
            sea_level_thrust_n,
            vacuum_thrust_n,
            sea_level_isp_s,
            vacuum_isp_s,
        )
        .map(|values| values.to_vec())
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn reentry_metrics(
        density_kg_m3: f64,
        speed_m_s: f64,
        mass_kg: f64,
        drag_area_m2: f64,
        cd: f64,
        lift_area_m2: f64,
        cl: f64,
        nose_radius_m: f64,
        coefficient: f64,
        dt_s: f64,
        previous_heat_load_j_m2: f64,
        previous_heat_rate_w_m2: f64,
    ) -> PyResult<Vec<f64>> {
        compute_reentry_metrics(
            density_kg_m3,
            speed_m_s,
            mass_kg,
            drag_area_m2,
            cd,
            lift_area_m2,
            cl,
            nose_radius_m,
            coefficient,
            dt_s,
            previous_heat_load_j_m2,
            previous_heat_rate_w_m2,
        )
        .map(|values| values.to_vec())
        .map_err(PyValueError::new_err)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(rocket_aero_state, module)?)?;
        module.add_function(wrap_pyfunction!(rocket_aero_loads, module)?)?;
        module.add_function(wrap_pyfunction!(rocket_stage_engine_perf, module)?)?;
        module.add_function(wrap_pyfunction!(rocket_propellant_step, module)?)?;
        module.add_function(wrap_pyfunction!(reentry_metrics, module)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{aero_loads, aero_state, propellant_step, reentry_metrics, stage_engine_perf};

    #[test]
    fn aero_zero_flow_is_finite_and_zero_load() {
        let state = aero_state(1.0, 101325.0, 288.15, 340.0, &[0.0; 3], 20.0, 20.0).unwrap();
        assert_eq!(state[4], 0.0);
        let loads = aero_loads(
            &[0.0; 3], &state, true, 10.0, 30.0, &[0.0; 3], 0.2, 0.1, 0.28, 0.22, 1.0, 0.22, 0.15,
            0.15, -0.02, -0.02, -0.01,
        )
        .unwrap();
        assert!(loads.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn stage_and_propellant_are_monotonic() {
        let perf = stage_engine_perf(0.0, 100.0, 120.0, 250.0, 300.0).unwrap();
        assert_eq!(perf, [120.0, 300.0]);
        let step = propellant_step(10.0, 1.0, 0.0, 1.0, 100.0, 100.0, 120.0, 250.0, 300.0).unwrap();
        assert!(step[2] > 0.0 && step[5] < 10.0 && step[6] < 100.0);
    }

    #[test]
    fn reentry_heat_load_accumulates() {
        let first = reentry_metrics(
            1.0,
            1000.0,
            100.0,
            2.0,
            2.0,
            2.0,
            0.2,
            0.5,
            1.83e-4,
            1.0,
            0.0,
            f64::NAN,
        )
        .unwrap();
        let second = reentry_metrics(
            1.0, 1000.0, 100.0, 2.0, 2.0, 2.0, 0.2, 0.5, 1.83e-4, 1.0, first[6], first[5],
        )
        .unwrap();
        assert!(second[6] > first[6]);
    }
}

#[cfg(feature = "python")]
pub use python::register;
