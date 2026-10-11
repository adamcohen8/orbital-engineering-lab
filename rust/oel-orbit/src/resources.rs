//! Spacecraft thermal, electrical and Earth-radiation numerical kernels.
//!
//! Python retains resource-spec validation, environment resolution, geometry
//! profile lookup and output dictionaries.  The native boundary receives
//! validated scalar settings and evaluates the repeated arithmetic with a
//! deterministic order matching `sim.spacecraft_resources.model`.

const EARTH_RADIUS_KM: f64 = 6378.137;
const AU_KM: f64 = 149_597_870.7;
const SIGMA: f64 = 5.670_374_419e-8;

use std::sync::{Mutex, OnceLock};

fn norm3(value: [f64; 3]) -> f64 {
    value[0]
        .mul_add(value[0], value[1].mul_add(value[1], value[2] * value[2]))
        .sqrt()
}

fn validate_vec(value: &[f64], name: &str) -> Result<(), String> {
    if !value.iter().all(|item| item.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

/// Gauss-Legendre nodes and weights on [-1, 1].
///
/// The implementation follows the standard Newton iteration used by common
/// scientific libraries.  Orders are deliberately bounded by the Python
/// configuration contract (4..64).
fn legendre_gauss_uncached(order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; order];
    let mut weights = vec![0.0; order];
    let half = (order + 1) / 2;
    for index in 0..half {
        let mut z = (std::f64::consts::PI * (index as f64 + 0.75) / (order as f64 + 0.5)).cos();
        for _ in 0..64 {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 1..=order {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 - 1.0) * z * p2 - (j as f64 - 1.0) * p3) / j as f64;
            }
            let pp = order as f64 * (z * p1 - p2) / (z * z - 1.0);
            let next = z - p1 / pp;
            if (next - z).abs() <= 3.0e-15 {
                z = next;
                break;
            }
            z = next;
        }
        let mut p1 = 1.0;
        let mut p2 = 0.0;
        for j in 1..=order {
            let p3 = p2;
            p2 = p1;
            p1 = ((2.0 * j as f64 - 1.0) * z * p2 - (j as f64 - 1.0) * p3) / j as f64;
        }
        let pp = order as f64 * (z * p1 - p2) / (z * z - 1.0);
        let weight = 2.0 / ((1.0 - z * z) * pp * pp);
        nodes[index] = -z;
        nodes[order - 1 - index] = z;
        weights[index] = weight;
        weights[order - 1 - index] = weight;
    }
    (nodes, weights)
}

fn legendre_gauss(order: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if !(4..=64).contains(&order) {
        return Err("earth quadrature order must be in 4..64".to_owned());
    }
    static CACHE: OnceLock<Mutex<Vec<Option<(Vec<f64>, Vec<f64>)>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(vec![None; 65]));
    if let Ok(guard) = cache.lock() {
        if let Some((nodes, weights)) = &guard[order] {
            return Ok((nodes.clone(), weights.clone()));
        }
    }
    let computed = legendre_gauss_uncached(order);
    if let Ok(mut guard) = cache.lock() {
        guard[order] = Some(computed.clone());
    }
    Ok(computed)
}

/// Deterministic apparent-disk quadrature for a uniform Lambertian Earth.
///
/// Returns flattened ray directions (N*3), solid-angle weights (N), and
/// surface illumination factors (N), matching the Python reference ordering.
pub fn earth_radiation(
    position: [f64; 3],
    sun_position: [f64; 3],
    order: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    validate_vec(&position, "position")?;
    validate_vec(&sun_position, "sun_position")?;
    let distance = norm3(position);
    if !distance.is_finite() || distance <= EARTH_RADIUS_KM {
        return Err("spacecraft resources require position above the Earth surface".to_owned());
    }
    let axis = [
        -position[0] / distance,
        -position[1] / distance,
        -position[2] / distance,
    ];
    let helper = if axis[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let mut a = [
        axis[1] * helper[2] - axis[2] * helper[1],
        axis[2] * helper[0] - axis[0] * helper[2],
        axis[0] * helper[1] - axis[1] * helper[0],
    ];
    let a_norm = norm3(a);
    a = [a[0] / a_norm, a[1] / a_norm, a[2] / a_norm];
    let b = [
        axis[1] * a[2] - axis[2] * a[1],
        axis[2] * a[0] - axis[0] * a[2],
        axis[0] * a[1] - axis[1] * a[0],
    ];
    let (nodes, weights_1d) = legendre_gauss(order)?;
    let phi_count = 2 * order;
    let mu_min = (1.0 - (EARTH_RADIUS_KM / distance).powi(2)).sqrt();
    let mut rays = Vec::with_capacity(order * phi_count * 3);
    let mut weights = Vec::with_capacity(order * phi_count);
    let mut illumination = Vec::with_capacity(order * phi_count);
    for index in 0..order {
        let mu = mu_min + (nodes[index] + 1.0) * (1.0 - mu_min) / 2.0;
        let radial = (1.0 - mu * mu).max(0.0).sqrt();
        let row_weight =
            weights_1d[index] * (1.0 - mu_min) / 2.0 * std::f64::consts::PI / order as f64;
        for azimuth in 0..phi_count {
            let phi = (azimuth as f64 + 0.5) * std::f64::consts::PI / order as f64;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let ray = [
                mu * axis[0] + radial * (cos_phi * a[0] + sin_phi * b[0]),
                mu * axis[1] + radial * (cos_phi * a[1] + sin_phi * b[1]),
                mu * axis[2] + radial * (cos_phi * a[2] + sin_phi * b[2]),
            ];
            let dot = ray[0] * position[0] + ray[1] * position[1] + ray[2] * position[2];
            let discriminant =
                (dot * dot - (distance * distance - EARTH_RADIUS_KM * EARTH_RADIUS_KM)).max(0.0);
            let travel = -dot - discriminant.sqrt();
            let point = [
                position[0] + ray[0] * travel,
                position[1] + ray[1] * travel,
                position[2] + ray[2] * travel,
            ];
            let normal = [
                point[0] / EARTH_RADIUS_KM,
                point[1] / EARTH_RADIUS_KM,
                point[2] / EARTH_RADIUS_KM,
            ];
            let to_sun = [
                sun_position[0] - point[0],
                sun_position[1] - point[1],
                sun_position[2] - point[2],
            ];
            let sun_distance = norm3(to_sun);
            if !sun_distance.is_finite() || sun_distance <= 0.0 {
                return Err("sun position must not coincide with a quadrature point".to_owned());
            }
            let surface_cosine =
                (normal[0] * to_sun[0] + normal[1] * to_sun[1] + normal[2] * to_sun[2])
                    / sun_distance;
            rays.extend_from_slice(&ray);
            weights.push(row_weight);
            illumination.push(surface_cosine.max(0.0) * (AU_KM / sun_distance).powi(2));
        }
    }
    Ok((rays, weights, illumination))
}

/// Return the resource rates and updated state as 22 values.
///
/// Layout: generation, demand, served, unmet, curtailed, charge, discharge,
/// battery energy, SOC, battery loss, conversion loss, electrical heat,
/// power residual, direct solar heat, albedo heat, Earth IR heat, internal
/// heat, incoming heat, temperature, radiated heat, stored heat, thermal
/// residual.  `thermal` and `power` are fixed-length settings vectors and may
/// be empty when that subsystem is disabled.  `ray_data` contains four values
/// per ray: projected area, quadrature weight, illumination, and an unused
/// reserved slot for ABI evolution.  The power settings vector is
/// `[baseline_load, capacity, charge_efficiency, discharge_efficiency,
/// conversion_efficiency, max_charge, max_discharge, load_heat_fraction,
/// solar_irradiance]`.
#[allow(clippy::too_many_arguments)]
pub fn advance(
    old_temperature: f64,
    old_battery_energy_wh: f64,
    thermal_enabled: bool,
    power_enabled: bool,
    thermal: &[f64],
    power: &[f64],
    panels: &[f64],
    sun_body: [f64; 3],
    distance_scale: f64,
    shadow: f64,
    dt_s: f64,
    direct_area_m2: f64,
    ray_data: &[f64],
) -> Result<Vec<f64>, String> {
    if !old_temperature.is_finite()
        || !old_battery_energy_wh.is_finite()
        || !distance_scale.is_finite()
        || !shadow.is_finite()
        || !dt_s.is_finite()
        || dt_s <= 0.0
        || !direct_area_m2.is_finite()
        || direct_area_m2 < 0.0
    {
        return Err(
            "resource state and interval inputs must be finite with positive dt_s".to_owned(),
        );
    }
    validate_vec(&sun_body, "sun_body")?;
    if thermal_enabled && thermal.len() != 9 {
        return Err("thermal settings must contain nine values".to_owned());
    }
    if power_enabled && power.len() != 9 {
        return Err("power settings must contain nine values".to_owned());
    }
    if panels.len() % 5 != 0 {
        return Err("panel settings must contain five values per panel".to_owned());
    }
    if ray_data.len() % 4 != 0 {
        return Err("ray data must contain four values per ray".to_owned());
    }
    validate_vec(thermal, "thermal settings")?;
    validate_vec(power, "power settings")?;
    validate_vec(panels, "panel settings")?;
    validate_vec(ray_data, "ray data")?;
    let solar_irradiance = if thermal_enabled {
        thermal[7]
    } else if power_enabled {
        power[8]
    } else {
        0.0
    };
    if !solar_irradiance.is_finite() || solar_irradiance < 0.0 {
        return Err("solar irradiance must be finite and nonnegative".to_owned());
    }
    let solar = solar_irradiance * distance_scale * shadow;
    let mut generation = 0.0;
    let mut demand = 0.0;
    let mut served = 0.0;
    let mut unmet = 0.0;
    let mut curtailed = 0.0;
    let mut charge = 0.0;
    let mut discharge = 0.0;
    let mut battery_energy = old_battery_energy_wh;
    let mut soc = 0.0;
    let mut battery_loss = 0.0;
    let mut conversion_loss = 0.0;
    let mut electrical_heat = 0.0;
    let mut power_residual = 0.0;
    if power_enabled {
        let baseline_load = power[0];
        let capacity = power[1];
        let charge_efficiency = power[2];
        let discharge_efficiency = power[3];
        let conversion_efficiency = power[4];
        let max_charge = power[5];
        let max_discharge = power[6];
        let load_heat_fraction = power[7];
        let mut panel_factor = 0.0;
        for panel in panels.chunks_exact(5) {
            let normal = [panel[1], panel[2], panel[3]];
            panel_factor += panel[0]
                * panel[4]
                * (normal[0] * sun_body[0] + normal[1] * sun_body[1] + normal[2] * sun_body[2])
                    .max(0.0);
        }
        generation = solar * panel_factor * conversion_efficiency;
        demand = baseline_load;
        charge = (generation - demand)
            .max(0.0)
            .min(max_charge)
            .min((capacity - old_battery_energy_wh).max(0.0) * 3600.0 / (dt_s * charge_efficiency));
        discharge = (demand - generation)
            .max(0.0)
            .min(max_discharge)
            .min(old_battery_energy_wh.max(0.0) * 3600.0 * discharge_efficiency / dt_s);
        served = demand.min(generation + discharge);
        curtailed = (generation - served - charge).max(0.0);
        let battery_rate = charge * charge_efficiency - discharge / discharge_efficiency;
        battery_energy = old_battery_energy_wh + dt_s * battery_rate / 3600.0;
        soc = battery_energy / capacity;
        battery_loss =
            charge * (1.0 - charge_efficiency) + discharge * (1.0 / discharge_efficiency - 1.0);
        conversion_loss = (generation - curtailed) * (1.0 / conversion_efficiency - 1.0);
        electrical_heat = served * load_heat_fraction + battery_loss + conversion_loss;
        unmet = demand - served;
        power_residual = generation + discharge - served - charge - curtailed;
    }

    let mut direct_heat = 0.0;
    let mut albedo_heat = 0.0;
    let mut earth_ir_heat = 0.0;
    let mut internal_heat = 0.0;
    let mut incoming = 0.0;
    let mut temperature = old_temperature;
    let mut radiated = 0.0;
    let mut stored = 0.0;
    let mut thermal_residual = 0.0;
    if thermal_enabled {
        let absorptivity = thermal[0];
        let emissivity = thermal[1];
        let radiating_area = thermal[3];
        let thermal_internal = thermal[4];
        let earth_albedo = thermal[5];
        let earth_ir_w_m2 = thermal[6];
        let capacity = thermal[8];
        direct_heat = solar * direct_area_m2 * absorptivity;
        let mut weighted_area_surface = 0.0;
        let mut weighted_area_total = 0.0;
        for row in ray_data.chunks_exact(4) {
            let weighted_area = row[0] * row[1] / std::f64::consts::PI;
            weighted_area_surface += weighted_area * row[2];
            weighted_area_total += weighted_area;
        }
        albedo_heat = absorptivity * earth_albedo * solar_irradiance * weighted_area_surface;
        earth_ir_heat = emissivity * earth_ir_w_m2 * weighted_area_total;
        internal_heat = thermal_internal + electrical_heat;
        incoming = direct_heat + albedo_heat + earth_ir_heat + internal_heat;
        let radiation_coefficient = emissivity * SIGMA * radiating_area;
        let rhs = old_temperature + dt_s * incoming / capacity;
        let mut lo = 0.0;
        let mut hi = rhs;
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if mid + dt_s * radiation_coefficient * mid.powi(4) / capacity > rhs {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        temperature = 0.5 * (lo + hi);
        radiated = radiation_coefficient * temperature.powi(4);
        stored = capacity * (temperature - old_temperature) / dt_s;
        thermal_residual = incoming - radiated - stored;
    }
    Ok(vec![
        generation,
        demand,
        served,
        unmet,
        curtailed,
        charge,
        discharge,
        battery_energy,
        soc,
        battery_loss,
        conversion_loss,
        electrical_heat,
        power_residual,
        direct_heat,
        albedo_heat,
        earth_ir_heat,
        internal_heat,
        incoming,
        temperature,
        radiated,
        stored,
        thermal_residual,
    ])
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;

    use pyo3::types::PyBytes;

    fn encode_f64_bytes(py: Python<'_>, values: &[f64]) -> PyResult<Py<PyBytes>> {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: initialized f64 slices have no padding, all bytes
            // are valid u8, and PyBytes::new synchronously copies the data.
            // The local little-endian representation is the boundary ABI.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            Ok(PyBytes::new(py, bytes).unbind())
        }
        #[cfg(target_endian = "big")]
        {
            Ok(PyBytes::new_with(py, values.len() * 8, |output| {
                for (bytes, value) in output.chunks_exact_mut(8).zip(values) {
                    bytes.copy_from_slice(&value.to_le_bytes());
                }
                Ok(())
            })?
            .unbind())
        }
    }

    #[pyfunction]
    fn resource_earth_radiation_bytes(
        py: Python<'_>,
        position: Vec<f64>,
        sun_position: Vec<f64>,
        order: usize,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Py<PyBytes>)> {
        if position.len() != 3 || sun_position.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "position and sun_position must have three values",
            ));
        }
        let (rays, weights, illumination) = earth_radiation(
            [position[0], position[1], position[2]],
            [sun_position[0], sun_position[1], sun_position[2]],
            order,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok((
            encode_f64_bytes(py, &rays)?,
            encode_f64_bytes(py, &weights)?,
            encode_f64_bytes(py, &illumination)?,
        ))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn resource_advance_isotropic(
        old_temperature: f64,
        old_battery_energy_wh: f64,
        thermal_enabled: bool,
        power_enabled: bool,
        thermal: Vec<f64>,
        power: Vec<f64>,
        panels: Vec<f64>,
        sun_body: Vec<f64>,
        distance_scale: f64,
        shadow: f64,
        dt_s: f64,
        direct_area_m2: f64,
        position: Vec<f64>,
        sun_position: Vec<f64>,
        order: usize,
    ) -> PyResult<Vec<f64>> {
        if sun_body.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sun_body must have three values",
            ));
        }
        if position.len() != 3 || sun_position.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "position and sun_position must have three values",
            ));
        }
        // A constant projected area does not require a body-frame transform.
        // Retain exactly the same ray weights, order, and advance arithmetic.
        let mut ray_data = Vec::new();
        if thermal_enabled {
            let (_, weights, illumination) = earth_radiation(
                [position[0], position[1], position[2]],
                [sun_position[0], sun_position[1], sun_position[2]],
                order,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            ray_data.reserve(weights.len() * 4);
            for (weight, surface_sun) in weights.iter().zip(&illumination) {
                ray_data.extend_from_slice(&[direct_area_m2, *weight, *surface_sun, 0.0]);
            }
        }
        advance(
            old_temperature,
            old_battery_energy_wh,
            thermal_enabled,
            power_enabled,
            &thermal,
            &power,
            &panels,
            [sun_body[0], sun_body[1], sun_body[2]],
            distance_scale,
            shadow,
            dt_s,
            direct_area_m2,
            &ray_data,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[pyfunction]
    fn resource_earth_radiation(
        position: Vec<f64>,
        sun_position: Vec<f64>,
        order: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if position.len() != 3 || sun_position.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "position and sun_position must have three values",
            ));
        }
        earth_radiation(
            [position[0], position[1], position[2]],
            [sun_position[0], sun_position[1], sun_position[2]],
            order,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn resource_advance(
        old_temperature: f64,
        old_battery_energy_wh: f64,
        thermal_enabled: bool,
        power_enabled: bool,
        thermal: Vec<f64>,
        power: Vec<f64>,
        panels: Vec<f64>,
        sun_body: Vec<f64>,
        distance_scale: f64,
        shadow: f64,
        dt_s: f64,
        direct_area_m2: f64,
        ray_data: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        if sun_body.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sun_body must have three values",
            ));
        }
        advance(
            old_temperature,
            old_battery_energy_wh,
            thermal_enabled,
            power_enabled,
            &thermal,
            &power,
            &panels,
            [sun_body[0], sun_body[1], sun_body[2]],
            distance_scale,
            shadow,
            dt_s,
            direct_area_m2,
            &ray_data,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    module.add_function(wrap_pyfunction!(resource_earth_radiation, module)?)?;
    module.add_function(wrap_pyfunction!(resource_earth_radiation_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(resource_advance_isotropic, module)?)?;
    module.add_function(wrap_pyfunction!(resource_advance, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{advance, earth_radiation};

    #[test]
    fn quadrature_solid_angle_matches_apparent_disk() {
        let (_, weights, _) =
            earth_radiation([7000.0, 0.0, 0.0], [149_597_870.7, 0.0, 0.0], 16).unwrap();
        let actual: f64 = weights.iter().sum();
        let expected =
            2.0 * std::f64::consts::PI * (1.0 - (1.0 - (6378.137_f64 / 7000.0).powi(2)).sqrt());
        assert!((actual - expected).abs() < 1e-13);
    }

    #[test]
    fn power_and_thermal_step_closes_balances() {
        let values = advance(
            290.0,
            50.0,
            true,
            true,
            &[0.3, 0.8, 1.0, 4.0, 0.0, 0.3, 237.0, 1361.0, 10_000.0],
            &[100.0, 100.0, 0.95, 0.95, 0.95, 200.0, 200.0, 1.0, 1361.0],
            &[2.0, 1.0, 0.0, 0.0, 0.3],
            [1.0, 0.0, 0.0],
            1.0,
            1.0,
            1.0,
            1.0,
            &[],
        )
        .unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values[12].abs() < 1e-12);
        // The implicit radiation solve is a finite-precision bisection; the
        // residual is at the sub-nanowatt level after 64 iterations.
        assert!(values[21].abs() < 1e-9);
    }
}
