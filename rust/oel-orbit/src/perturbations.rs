//! Opt-in ONP perturbation kernels.
//!
//! Resource loading, licensing, epoch/EOP policy, and custom Python force
//! plugins stay in Python.  These routines receive prepared numeric inputs and
//! return the same km/s² quantities used by OEL's reference implementations.

pub(crate) use crate::precision_forces::solid_harmonics_values;
use crate::precision_forces::{dot, finite_vector, norm};
pub use crate::precision_forces::{
    earth_radiation_components, schwarzschild_acceleration, tidal_acceleration_fixed,
};
#[cfg(feature = "python")]
use crate::precision_forces::{earth_radiation_components_prevalidated, AU_KM};

/// Additive J2 acceleration in ECI km/s².
pub fn j2_acceleration(
    position_km: [f64; 3],
    mu_km3_s2: f64,
    j2: f64,
    reference_radius_km: f64,
) -> Result<[f64; 3], String> {
    finite_vector(position_km, "position")?;
    if ![mu_km3_s2, j2, reference_radius_km]
        .iter()
        .all(|value| value.is_finite())
        || mu_km3_s2 <= 0.0
        || reference_radius_km <= 0.0
    {
        return Err("J2 parameters must be finite with positive mu and radius".to_owned());
    }
    let r2 = dot(position_km, position_km);
    let r = r2.sqrt();
    if r == 0.0 {
        return Ok([0.0; 3]);
    }
    let z2 = position_km[2] * position_km[2];
    let scale = 1.5 * j2 * mu_km3_s2 * reference_radius_km.powi(2) / r.powi(5);
    let factor = 5.0 * z2 / r2;
    Ok([
        scale * position_km[0] * (factor - 1.0),
        scale * position_km[1] * (factor - 1.0),
        scale * position_km[2] * (factor - 3.0),
    ])
}

/// Additive J3 acceleration in ECI km/s².
pub fn j3_acceleration(
    position_km: [f64; 3],
    mu_km3_s2: f64,
    j3: f64,
    reference_radius_km: f64,
) -> Result<[f64; 3], String> {
    finite_vector(position_km, "position")?;
    if ![mu_km3_s2, j3, reference_radius_km]
        .iter()
        .all(|value| value.is_finite())
        || mu_km3_s2 <= 0.0
        || reference_radius_km <= 0.0
    {
        return Err("J3 parameters must be finite with positive mu and radius".to_owned());
    }
    let r2 = dot(position_km, position_km);
    let r = r2.sqrt();
    if r == 0.0 {
        return Ok([0.0; 3]);
    }
    let s = position_km[2] / r;
    let s2 = s * s;
    let s4 = s2 * s2;
    let xy_scale = mu_km3_s2 * j3 * reference_radius_km.powi(3) / r.powi(6);
    let xy_factor = 2.5 * s * (7.0 * s2 - 3.0);
    let z_scale = mu_km3_s2 * j3 * reference_radius_km.powi(3) / r.powi(5);
    let z_factor = 0.5 * (35.0 * s4 - 30.0 * s2 + 3.0);
    Ok([
        xy_scale * position_km[0] * xy_factor,
        xy_scale * position_km[1] * xy_factor,
        z_scale * z_factor,
    ])
}

/// Additive J4 acceleration in ECI km/s².
pub fn j4_acceleration(
    position_km: [f64; 3],
    mu_km3_s2: f64,
    j4: f64,
    reference_radius_km: f64,
) -> Result<[f64; 3], String> {
    finite_vector(position_km, "position")?;
    if ![mu_km3_s2, j4, reference_radius_km]
        .iter()
        .all(|value| value.is_finite())
        || mu_km3_s2 <= 0.0
        || reference_radius_km <= 0.0
    {
        return Err("J4 parameters must be finite with positive mu and radius".to_owned());
    }
    let r2 = dot(position_km, position_km);
    let r = r2.sqrt();
    if r == 0.0 {
        return Ok([0.0; 3]);
    }
    let s = position_km[2] / r;
    let s2 = s * s;
    let s4 = s2 * s2;
    let xy_scale = mu_km3_s2 * j4 * reference_radius_km.powi(4) / r.powi(7);
    let xy_factor = 0.625 * (63.0 * s4 - 42.0 * s2 + 3.0);
    let z_scale = mu_km3_s2 * j4 * reference_radius_km.powi(4) / r.powi(6);
    let z_factor = 0.625 * s * (63.0 * s4 - 70.0 * s2 + 15.0);
    Ok([
        xy_scale * position_km[0] * xy_factor,
        xy_scale * position_km[1] * xy_factor,
        z_scale * z_factor,
    ])
}

/// Third-body differential acceleration in ECI km/s².
pub fn third_body_acceleration(
    position_km: [f64; 3],
    body_position_km: [f64; 3],
    body_mu_km3_s2: f64,
) -> Result<[f64; 3], String> {
    finite_vector(position_km, "spacecraft position")?;
    finite_vector(body_position_km, "body position")?;
    if !body_mu_km3_s2.is_finite() || body_mu_km3_s2 <= 0.0 {
        return Err("third-body mu must be finite and positive".to_owned());
    }
    let delta = [
        body_position_km[0] - position_km[0],
        body_position_km[1] - position_km[1],
        body_position_km[2] - position_km[2],
    ];
    let delta_norm2 = dot(delta, delta);
    let body_norm2 = dot(body_position_km, body_position_km);
    if delta_norm2 == 0.0 || body_norm2 == 0.0 {
        return Ok([0.0; 3]);
    }
    let delta_norm3 = delta_norm2.sqrt().powi(3);
    let body_norm3 = body_norm2.sqrt().powi(3);
    Ok(std::array::from_fn(|axis| {
        body_mu_km3_s2 * (delta[axis] / delta_norm3 - body_position_km[axis] / body_norm3)
    }))
}

fn illumination(alpha: f64, beta: f64, gamma: f64) -> f64 {
    if beta <= 0.0 || gamma >= alpha + beta {
        return 1.0;
    }
    if alpha > beta && gamma <= alpha - beta {
        return 0.0;
    }
    if beta > alpha && gamma <= beta - alpha {
        return (1.0 - alpha * alpha / (beta * beta)).max(0.0);
    }
    if gamma <= 0.0 {
        return if alpha >= beta {
            0.0
        } else {
            (1.0 - alpha * alpha / (beta * beta)).max(0.0)
        };
    }
    let e = ((gamma * gamma + alpha * alpha - beta * beta) / (2.0 * gamma * alpha))
        .clamp(-1.0, 1.0)
        .acos();
    let s = ((gamma * gamma + beta * beta - alpha * alpha) / (2.0 * gamma * beta))
        .clamp(-1.0, 1.0)
        .acos();
    let product = ((-gamma + alpha + beta)
        * (gamma + alpha - beta)
        * (gamma - alpha + beta)
        * (gamma + alpha + beta))
        .max(0.0);
    (1.0 - (alpha * alpha * e + beta * beta * s - 0.5 * product.sqrt())
        / (std::f64::consts::PI * beta * beta))
        .clamp(0.0, 1.0)
}

/// Direct solar-radiation-pressure acceleration in ECI km/s².
pub fn srp_acceleration(
    position_km: [f64; 3],
    sun_position_km: [f64; 3],
    mass_kg: f64,
    area_m2: f64,
    reflectivity: f64,
    pressure_pa: f64,
    au_km: f64,
    earth_radius_km: f64,
    sun_radius_km: f64,
    shadow_model: u8,
) -> Result<[f64; 3], String> {
    finite_vector(position_km, "spacecraft position")?;
    finite_vector(sun_position_km, "Sun position")?;
    if ![
        mass_kg,
        area_m2,
        reflectivity,
        pressure_pa,
        au_km,
        earth_radius_km,
        sun_radius_km,
    ]
    .iter()
    .all(|value| value.is_finite())
        || mass_kg <= 0.0
        || area_m2 < 0.0
        || reflectivity < 0.0
        || pressure_pa < 0.0
        || au_km <= 0.0
        || earth_radius_km <= 0.0
        || sun_radius_km <= 0.0
        || shadow_model > 2
    {
        return Err("SRP parameters are outside their valid range".to_owned());
    }
    if area_m2 == 0.0 {
        return Ok([0.0; 3]);
    }
    let r_norm = norm(position_km);
    let sun_norm = norm(sun_position_km);
    let rho = [
        sun_position_km[0] - position_km[0],
        sun_position_km[1] - position_km[1],
        sun_position_km[2] - position_km[2],
    ];
    let rho_norm = norm(rho);
    if r_norm == 0.0 || sun_norm == 0.0 || rho_norm == 0.0 {
        return Ok([0.0; 3]);
    }
    let mut shadow = 1.0;
    if shadow_model != 0 {
        if r_norm <= earth_radius_km {
            shadow = 0.0;
        } else {
            let sunward = dot(position_km, sun_position_km) / sun_norm >= 0.0;
            if !sunward {
                if shadow_model == 1 {
                    let along = dot(position_km, sun_position_km) / sun_norm;
                    if along < 0.0
                        && (dot(position_km, position_km) - along * along).max(0.0)
                            < earth_radius_km * earth_radius_km
                    {
                        shadow = 0.0;
                    }
                } else {
                    let alpha = (earth_radius_km / r_norm).clamp(-1.0, 1.0).asin();
                    let beta = (sun_radius_km / rho_norm).clamp(-1.0, 1.0).asin();
                    let separation = (-dot(position_km, rho) / (r_norm * rho_norm))
                        .clamp(-1.0, 1.0)
                        .acos();
                    shadow = illumination(alpha, beta, separation);
                }
            }
        }
    }
    let scale =
        -pressure_pa * (au_km / rho_norm).powi(2) * reflectivity * area_m2 / mass_kg / 1.0e3
            * shadow;
    Ok(std::array::from_fn(|axis| scale * rho[axis] / rho_norm))
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::prelude::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    fn row<const N: usize>(bytes: &[u8]) -> [f64; N] {
        std::array::from_fn(|axis| {
            f64::from_le_bytes(bytes[axis * 8..axis * 8 + 8].try_into().unwrap())
        })
    }

    fn batch_count(bytes: &[u8], width: usize, label: &str) -> PyResult<usize> {
        if bytes.len() % (width * 8) != 0 || bytes.len() / (width * 8) > 10_000_000 {
            return Err(PyValueError::new_err(format!(
                "{label} must contain at most 10000000 float64 rows of width {width}"
            )));
        }
        Ok(bytes.len() / (width * 8))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn perturbation_force_bundle_batch<'py>(
        py: Python<'py>,
        states: &[u8],
        body_positions: &[u8],
        sun_positions: &[u8],
        scalars: [f64; 12],
        shadow_model: u8,
        codes: Vec<u8>,
        c_nm: Vec<f64>,
        s_nm: Vec<f64>,
        degree: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let count = batch_count(states, 6, "force bundle states")?;
        if batch_count(body_positions, 3, "force bundle body positions")? != count
            || batch_count(sun_positions, 3, "force bundle Sun positions")? != count
        {
            return Err(PyValueError::new_err(
                "force bundle batch dimensions must match",
            ));
        }
        let [mu, j2, j3, j4, radius, body_mu, mass, area, cr, pressure, au, sun_radius] = scalars;
        if degree
            .checked_add(1)
            .and_then(|side| side.checked_mul(side))
            != Some(c_nm.len())
            || s_nm.len() != c_nm.len()
        {
            return Err(PyValueError::new_err(
                "tidal coefficient dimensions are invalid",
            ));
        }
        // Empty batches must still reject invalid parameter sets.
        let probe = [radius + 1.0, 0.0, 0.0];
        for &code in &codes {
            match code {
                2 => j2_acceleration(probe, mu, j2, radius),
                3 => j3_acceleration(probe, mu, j3, radius),
                4 => j4_acceleration(probe, mu, j4, radius),
                _ => return Err(PyValueError::new_err("zonal code must be 2, 3, or 4")),
            }
            .map_err(PyValueError::new_err)?;
        }
        third_body_acceleration(probe, [au, 0.0, 0.0], body_mu).map_err(PyValueError::new_err)?;
        srp_acceleration(
            probe,
            [au, 0.0, 0.0],
            mass,
            area,
            cr,
            pressure,
            au,
            radius,
            sun_radius,
            shadow_model,
        )
        .map_err(PyValueError::new_err)?;
        schwarzschild_acceleration([probe[0], 0.0, 0.0, 0.0, 0.0, 0.0], mu)
            .map_err(PyValueError::new_err)?;
        tidal_acceleration_fixed(probe, &c_nm, &s_nm, degree, mu, radius)
            .map_err(PyValueError::new_err)?;
        let mut out = Vec::with_capacity(count * 24);
        for index in 0..count {
            let state = row::<6>(&states[index * 48..index * 48 + 48]);
            let position = [state[0], state[1], state[2]];
            let body = row::<3>(&body_positions[index * 24..index * 24 + 24]);
            let sun = row::<3>(&sun_positions[index * 24..index * 24 + 24]);
            let mut acceleration = [0.0; 3];
            for &code in &codes {
                let part = match code {
                    2 => j2_acceleration(position, mu, j2, radius),
                    3 => j3_acceleration(position, mu, j3, radius),
                    4 => j4_acceleration(position, mu, j4, radius),
                    _ => unreachable!(),
                }
                .map_err(PyValueError::new_err)?;
                for axis in 0..3 {
                    acceleration[axis] += part[axis];
                }
            }
            for part in [
                third_body_acceleration(position, body, body_mu),
                srp_acceleration(
                    position,
                    sun,
                    mass,
                    area,
                    cr,
                    pressure,
                    au,
                    radius,
                    sun_radius,
                    shadow_model,
                ),
                schwarzschild_acceleration(state, mu),
                tidal_acceleration_fixed(position, &c_nm, &s_nm, degree, mu, radius),
            ] {
                let part = part.map_err(PyValueError::new_err)?;
                for axis in 0..3 {
                    acceleration[axis] += part[axis];
                }
            }
            for value in acceleration {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(PyBytes::new(py, &out))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn perturbation_earth_radiation_batch<'py>(
        py: Python<'py>,
        positions: &[u8],
        sun_positions: &[u8],
        elapsed_times: &[u8],
        radius_km: f64,
        nodes: Vec<f64>,
        weights: Vec<f64>,
        cos_azimuth: Vec<f64>,
        sin_azimuth: Vec<f64>,
        include_albedo: bool,
        include_infrared: bool,
        uniform_albedo: Option<f64>,
        uniform_emissivity: Option<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let count = batch_count(positions, 3, "Earth radiation positions")?;
        if batch_count(sun_positions, 3, "Earth radiation Sun positions")? != count
            || batch_count(elapsed_times, 1, "Earth radiation times")? != count
        {
            return Err(PyValueError::new_err(
                "Earth radiation batch dimensions must match",
            ));
        }
        if count == 0 {
            earth_radiation_components(
                [radius_km + 1.0, 0.0, 0.0],
                [AU_KM, 0.0, 0.0],
                0.0,
                radius_km,
                &nodes,
                &weights,
                &cos_azimuth,
                &sin_azimuth,
                include_albedo,
                include_infrared,
                uniform_albedo,
                uniform_emissivity,
            )
            .map_err(PyValueError::new_err)?;
        }
        let mut out = Vec::with_capacity(count * 48);
        for index in 0..count {
            let evaluate = if index == 0 {
                earth_radiation_components
            } else {
                earth_radiation_components_prevalidated
            };
            let (albedo, infrared) = evaluate(
                row::<3>(&positions[index * 24..index * 24 + 24]),
                row::<3>(&sun_positions[index * 24..index * 24 + 24]),
                row::<1>(&elapsed_times[index * 8..index * 8 + 8])[0],
                radius_km,
                &nodes,
                &weights,
                &cos_azimuth,
                &sin_azimuth,
                include_albedo,
                include_infrared,
                uniform_albedo,
                uniform_emissivity,
            )
            .map_err(PyValueError::new_err)?;
            for value in albedo.into_iter().chain(infrared) {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(PyBytes::new(py, &out))
    }

    #[pyfunction]
    fn perturbation_zonal_acceleration(
        position: Vec<f64>,
        mu: f64,
        j2: f64,
        j3: f64,
        j4: f64,
        radius_km: f64,
        codes: Vec<u8>,
    ) -> PyResult<Vec<f64>> {
        let position: [f64; 3] = position
            .try_into()
            .map_err(|_| PyValueError::new_err("position must have three values"))?;
        let mut result = [0.0; 3];
        for code in codes {
            let part = match code {
                2 => j2_acceleration(position, mu, j2, radius_km),
                3 => j3_acceleration(position, mu, j3, radius_km),
                4 => j4_acceleration(position, mu, j4, radius_km),
                _ => return Err(PyValueError::new_err("zonal code must be 2, 3, or 4")),
            }
            .map_err(PyValueError::new_err)?;
            for axis in 0..3 {
                result[axis] += part[axis];
            }
        }
        Ok(result.to_vec())
    }

    #[pyfunction]
    fn perturbation_third_body(
        position: Vec<f64>,
        body_position: Vec<f64>,
        mu: f64,
    ) -> PyResult<Vec<f64>> {
        let position: [f64; 3] = position
            .try_into()
            .map_err(|_| PyValueError::new_err("position must have three values"))?;
        let body_position: [f64; 3] = body_position
            .try_into()
            .map_err(|_| PyValueError::new_err("body_position must have three values"))?;
        Ok(third_body_acceleration(position, body_position, mu)
            .map_err(PyValueError::new_err)?
            .to_vec())
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn perturbation_srp(
        position: Vec<f64>,
        sun_position: Vec<f64>,
        mass_kg: f64,
        area_m2: f64,
        reflectivity: f64,
        pressure_pa: f64,
        au_km: f64,
        earth_radius_km: f64,
        sun_radius_km: f64,
        shadow_model: u8,
    ) -> PyResult<Vec<f64>> {
        let position: [f64; 3] = position
            .try_into()
            .map_err(|_| PyValueError::new_err("position must have three values"))?;
        let sun_position: [f64; 3] = sun_position
            .try_into()
            .map_err(|_| PyValueError::new_err("sun_position must have three values"))?;
        Ok(srp_acceleration(
            position,
            sun_position,
            mass_kg,
            area_m2,
            reflectivity,
            pressure_pa,
            au_km,
            earth_radius_km,
            sun_radius_km,
            shadow_model,
        )
        .map_err(PyValueError::new_err)?
        .to_vec())
    }

    #[pyfunction]
    fn perturbation_schwarzschild(state: Vec<f64>, mu: f64) -> PyResult<Vec<f64>> {
        let state: [f64; 6] = state
            .try_into()
            .map_err(|_| PyValueError::new_err("state must have six values"))?;
        Ok(schwarzschild_acceleration(state, mu)
            .map_err(PyValueError::new_err)?
            .to_vec())
    }

    // The adapter supplies a shape-checked six-state. Fixed arrays remove the
    // temporary Rust Vec allocations without changing the arithmetic kernel.
    #[pyfunction]
    fn perturbation_schwarzschild_prepared(state: [f64; 6], mu: f64) -> PyResult<[f64; 3]> {
        if state.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "state must have shape (6,) and contain finite values",
            ));
        }
        schwarzschild_acceleration(state, mu).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn perturbation_tidal_acceleration(
        position: Vec<f64>,
        c_nm: Vec<f64>,
        s_nm: Vec<f64>,
        degree: usize,
        mu: f64,
        radius_km: f64,
    ) -> PyResult<Vec<f64>> {
        let position: [f64; 3] = position
            .try_into()
            .map_err(|_| PyValueError::new_err("position must have three values"))?;
        Ok(
            tidal_acceleration_fixed(position, &c_nm, &s_nm, degree, mu, radius_km)
                .map_err(PyValueError::new_err)?
                .to_vec(),
        )
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn perturbation_earth_radiation(
        position: Vec<f64>,
        sun_position: Vec<f64>,
        elapsed_s: f64,
        radius_km: f64,
        nodes: Vec<f64>,
        weights: Vec<f64>,
        cos_azimuth: Vec<f64>,
        sin_azimuth: Vec<f64>,
        include_albedo: bool,
        include_infrared: bool,
        uniform_albedo: Option<f64>,
        uniform_emissivity: Option<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let position: [f64; 3] = position
            .try_into()
            .map_err(|_| PyValueError::new_err("position must have three values"))?;
        let sun_position: [f64; 3] = sun_position
            .try_into()
            .map_err(|_| PyValueError::new_err("sun_position must have three values"))?;
        let (albedo, infrared) = earth_radiation_components(
            position,
            sun_position,
            elapsed_s,
            radius_km,
            &nodes,
            &weights,
            &cos_azimuth,
            &sin_azimuth,
            include_albedo,
            include_infrared,
            uniform_albedo,
            uniform_emissivity,
        )
        .map_err(PyValueError::new_err)?;
        Ok((albedo.to_vec(), infrared.to_vec()))
    }

    module.add_function(wrap_pyfunction!(perturbation_zonal_acceleration, module)?)?;
    module.add_function(wrap_pyfunction!(perturbation_third_body, module)?)?;
    module.add_function(wrap_pyfunction!(perturbation_srp, module)?)?;
    module.add_function(wrap_pyfunction!(perturbation_schwarzschild, module)?)?;
    module.add_function(wrap_pyfunction!(
        perturbation_schwarzschild_prepared,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(perturbation_tidal_acceleration, module)?)?;
    module.add_function(wrap_pyfunction!(perturbation_earth_radiation, module)?)?;
    module.add_function(wrap_pyfunction!(perturbation_force_bundle_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        perturbation_earth_radiation_batch,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::srp_acceleration;

    #[test]
    fn srp_zero_area_returns_zero_acceleration() {
        let acceleration = srp_acceleration(
            [7_000.0, 0.0, 0.0],
            [149_597_870.7, 0.0, 0.0],
            10.0,
            0.0,
            1.2,
            4.5606e-6,
            149_597_870.7,
            6_378.137,
            695_700.0,
            0,
        )
        .expect("zero area is a valid no-SRP case");

        assert_eq!(acceleration, [0.0; 3]);
    }

    #[test]
    fn srp_negative_area_is_rejected() {
        let result = srp_acceleration(
            [7_000.0, 0.0, 0.0],
            [149_597_870.7, 0.0, 0.0],
            10.0,
            -1.0,
            1.2,
            4.5606e-6,
            149_597_870.7,
            6_378.137,
            695_700.0,
            0,
        );

        let error = result.expect_err("negative area must remain invalid");
        assert!(error.contains("SRP parameters are outside their valid range"));
    }
}
