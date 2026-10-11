//! Ordered built-in zonal perturbations. Codes 2, 3, and 4 mean J2, J3, J4.

const EARTH_RADIUS_KM: f64 = 6378.137;
const EARTH_J2: f64 = 1.08262668e-3;
const EARTH_J3: f64 = -2.53215306e-6;
const EARTH_J4: f64 = -1.61098761e-6;

pub fn validate_codes(codes: &[u8]) -> Result<(), &'static str> {
    if codes.iter().any(|code| !matches!(code, 2 | 3 | 4)) {
        return Err("Rust zonal force plan accepts only J2, J3, and J4 codes");
    }
    Ok(())
}

pub fn derivative(
    state: [f64; 6],
    mu_km3_s2: f64,
    command_accel_eci_km_s2: [f64; 3],
    codes: &[u8],
) -> [f64; 6] {
    let x = state[0];
    let y = state[1];
    let z = state[2];
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let scale = -mu_km3_s2 / (r * r2);
    let mut a = [
        scale * x + command_accel_eci_km_s2[0],
        scale * y + command_accel_eci_km_s2[1],
        scale * z + command_accel_eci_km_s2[2],
    ];
    for code in codes {
        let component = component(*code, [x, y, z], mu_km3_s2);
        a[0] += component[0];
        a[1] += component[1];
        a[2] += component[2];
    }
    [state[3], state[4], state[5], a[0], a[1], a[2]]
}

pub fn component(code: u8, position_km: [f64; 3], mu_km3_s2: f64) -> [f64; 3] {
    let [x, y, z] = position_km;
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    match code {
        2 => {
            let f = 1.5 * EARTH_J2 * mu_km3_s2 * (EARTH_RADIUS_KM * EARTH_RADIUS_KM) / r.powi(5);
            let g = 5.0 * (z * z) / r2;
            [f * x * (g - 1.0), f * y * (g - 1.0), f * z * (g - 3.0)]
        }
        3 => {
            let s = z / r;
            let s2 = s * s;
            let s4 = s2 * s2;
            let axy_scale = mu_km3_s2 * EARTH_J3 * EARTH_RADIUS_KM.powi(3) / r.powi(6);
            let axy_factor = 2.5 * s * (7.0 * s2 - 3.0);
            let az_scale = mu_km3_s2 * EARTH_J3 * EARTH_RADIUS_KM.powi(3) / r.powi(5);
            let az_factor = 0.5 * (35.0 * s4 - 30.0 * s2 + 3.0);
            [
                axy_scale * x * axy_factor,
                axy_scale * y * axy_factor,
                az_scale * az_factor,
            ]
        }
        4 => {
            let s = z / r;
            let s2 = s * s;
            let s4 = s2 * s2;
            let axy_scale = mu_km3_s2 * EARTH_J4 * EARTH_RADIUS_KM.powi(4) / r.powi(7);
            let axy_factor = 0.625 * (63.0 * s4 - 42.0 * s2 + 3.0);
            let az_scale = mu_km3_s2 * EARTH_J4 * EARTH_RADIUS_KM.powi(4) / r.powi(6);
            let az_factor = 0.625 * s * (63.0 * s4 - 70.0 * s2 + 15.0);
            [
                axy_scale * x * axy_factor,
                axy_scale * y * axy_factor,
                az_scale * az_factor,
            ]
        }
        _ => unreachable!("force plan validated before integration"),
    }
}
