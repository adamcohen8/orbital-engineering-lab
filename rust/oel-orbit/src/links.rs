//! Directed-link and atmospheric numerical kernels.
//!
//! Python owns terminal/profile validation, frame provenance, reason labels,
//! event refinement callbacks, and artifact assembly.  This module contains
//! the repeated vector arithmetic used by those workflows.

const PI: f64 = std::f64::consts::PI;
const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;
const BOLTZMANN_J_K: f64 = 1.380_649e-23;
const WGS84_A_KM: f64 = 6378.137;
const WGS84_B_KM: f64 = 6356.752314245179;

fn finite_slice(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn row3(values: &[f64], index: usize) -> [f64; 3] {
    [
        values[3 * index],
        values[3 * index + 1],
        values[3 * index + 2],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm(value: [f64; 3]) -> f64 {
    dot(value, value).sqrt()
}

fn validate_vector_batch(values: &[f64], name: &str) -> Result<usize, String> {
    if values.len() % 3 != 0 || !finite_slice(values) {
        return Err(format!("{name} must be finite with shape (samples, 3)"));
    }
    Ok(values.len() / 3)
}

#[derive(Clone, Debug, PartialEq)]
pub struct FreeSpaceResult {
    pub tx_power_dbw: Vec<f64>,
    pub tx_gain_dbi: Vec<f64>,
    pub rx_gain_dbi: Vec<f64>,
    pub free_space_path_loss_db: Vec<f64>,
    pub eirp_dbw: Vec<f64>,
    pub received_power_dbw: Vec<f64>,
    pub noise_density_dbw_hz: Vec<f64>,
    pub cn0_db_hz: Vec<f64>,
    pub eb_n0_db: Vec<f64>,
    pub margin_db: Vec<f64>,
    pub margin_pass: Vec<u8>,
}

/// Evaluate the OEL free-space link equation over one already-broadcast batch.
pub fn free_space_link(
    range_km: &[f64],
    tx_gain_dbi: &[f64],
    rx_gain_dbi: &[f64],
    carrier_frequency_hz: f64,
    tx_power_w: f64,
    data_rate_bps: f64,
    system_noise_temperature_k: f64,
    required_eb_n0_db: f64,
    tx_line_loss_db: f64,
    rx_line_loss_db: f64,
    misc_loss_db: f64,
) -> Result<FreeSpaceResult, String> {
    if range_km.is_empty() || !finite_slice(range_km) || range_km.iter().any(|value| *value <= 0.0)
    {
        return Err("range_km must contain positive finite values".into());
    }
    let count = range_km.len();
    if tx_gain_dbi.len() != count || rx_gain_dbi.len() != count {
        return Err("terminal gains must match range_km".into());
    }
    if !finite_slice(tx_gain_dbi) || !finite_slice(rx_gain_dbi) {
        return Err("terminal gains must be finite".into());
    }
    for (value, name) in [
        (carrier_frequency_hz, "carrier_frequency_hz"),
        (tx_power_w, "tx_power_w"),
        (data_rate_bps, "data_rate_bps"),
        (system_noise_temperature_k, "system_noise_temperature_k"),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be positive and finite"));
        }
    }
    for (value, name) in [
        (tx_line_loss_db, "tx_line_loss_db"),
        (rx_line_loss_db, "rx_line_loss_db"),
        (misc_loss_db, "misc_loss_db"),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} must be nonnegative and finite"));
        }
    }
    if !required_eb_n0_db.is_finite() {
        return Err("required_eb_n0_db must be finite".into());
    }
    let tx_power = 10.0 * tx_power_w.log10();
    let noise = 10.0 * (BOLTZMANN_J_K * system_noise_temperature_k).log10();
    let rate = 10.0 * data_rate_bps.log10();
    let mut result = FreeSpaceResult {
        tx_power_dbw: vec![tx_power; count],
        tx_gain_dbi: tx_gain_dbi.to_vec(),
        rx_gain_dbi: rx_gain_dbi.to_vec(),
        free_space_path_loss_db: Vec::with_capacity(count),
        eirp_dbw: Vec::with_capacity(count),
        received_power_dbw: Vec::with_capacity(count),
        noise_density_dbw_hz: vec![noise; count],
        cn0_db_hz: Vec::with_capacity(count),
        eb_n0_db: Vec::with_capacity(count),
        margin_db: Vec::with_capacity(count),
        margin_pass: Vec::with_capacity(count),
    };
    for index in 0..count {
        let path_loss = 20.0
            * (4.0 * PI * range_km[index] * 1000.0 * carrier_frequency_hz / SPEED_OF_LIGHT_M_S)
                .log10();
        let eirp = tx_power + tx_gain_dbi[index] - tx_line_loss_db;
        let received = eirp + rx_gain_dbi[index] - path_loss - rx_line_loss_db - misc_loss_db;
        let cn0 = received - noise;
        let eb_n0 = cn0 - rate;
        let margin = eb_n0 - required_eb_n0_db;
        result.free_space_path_loss_db.push(path_loss);
        result.eirp_dbw.push(eirp);
        result.received_power_dbw.push(received);
        result.cn0_db_hz.push(cn0);
        result.eb_n0_db.push(eb_n0);
        result.margin_db.push(margin);
        result.margin_pass.push((margin >= -1.0e-10) as u8);
    }
    Ok(result)
}

/// Determine whether each open endpoint segment intersects the WGS84 ellipsoid.
pub fn earth_occulted(
    tx_position_ecef_km: &[f64],
    rx_position_ecef_km: &[f64],
) -> Result<Vec<u8>, String> {
    let count = validate_vector_batch(tx_position_ecef_km, "tx_position_ecef_km")?;
    if validate_vector_batch(rx_position_ecef_km, "rx_position_ecef_km")? != count {
        return Err("endpoint position batches must have equal lengths".into());
    }
    let axes = [WGS84_A_KM, WGS84_A_KM, WGS84_B_KM];
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let tx = row3(tx_position_ecef_km, index);
        let rx = row3(rx_position_ecef_km, index);
        let tx_scaled = [tx[0] / axes[0], tx[1] / axes[1], tx[2] / axes[2]];
        let delta = [
            (rx[0] - tx[0]) / axes[0],
            (rx[1] - tx[1]) / axes[1],
            (rx[2] - tx[2]) / axes[2],
        ];
        let quadratic = dot(delta, delta);
        if quadratic <= 0.0 || !quadratic.is_finite() {
            return Err("endpoint positions must be distinct".into());
        }
        let linear = 2.0 * dot(tx_scaled, delta);
        let constant = dot(tx_scaled, tx_scaled) - 1.0;
        let discriminant = linear * linear - 4.0 * quadratic * constant;
        let tangent_or_crossing = discriminant >= -1.0e-15;
        let root = discriminant.max(0.0).sqrt();
        let first = (-linear - root) / (2.0 * quadratic);
        let second = (-linear + root) / (2.0 * quadratic);
        let epsilon = 1.0e-12;
        let open_segment_hit = (first > epsilon && first < 1.0 - epsilon)
            || (second > epsilon && second < 1.0 - epsilon);
        result.push((tangent_or_crossing && open_segment_hit) as u8);
    }
    Ok(result)
}

#[derive(Clone, Debug, PartialEq)]
pub struct KinematicsResult {
    pub range_km: Vec<f64>,
    pub range_rate_km_s: Vec<f64>,
}

pub fn endpoint_kinematics(
    tx_position_eci_km: &[f64],
    rx_position_eci_km: &[f64],
    tx_velocity_eci_km_s: &[f64],
    rx_velocity_eci_km_s: &[f64],
) -> Result<KinematicsResult, String> {
    let count = validate_vector_batch(tx_position_eci_km, "tx_position_eci_km")?;
    for (values, name) in [
        (rx_position_eci_km, "rx_position_eci_km"),
        (tx_velocity_eci_km_s, "tx_velocity_eci_km_s"),
        (rx_velocity_eci_km_s, "rx_velocity_eci_km_s"),
    ] {
        if validate_vector_batch(values, name)? != count {
            return Err("endpoint kinematics batches must have equal lengths".into());
        }
    }
    let mut ranges = Vec::with_capacity(count);
    let mut rates = Vec::with_capacity(count);
    for index in 0..count {
        let tx = row3(tx_position_eci_km, index);
        let rx = row3(rx_position_eci_km, index);
        let delta = [rx[0] - tx[0], rx[1] - tx[1], rx[2] - tx[2]];
        let range = norm(delta);
        if !range.is_finite() || range <= 0.0 {
            return Err("endpoint range must be positive and finite".into());
        }
        let tx_v = row3(tx_velocity_eci_km_s, index);
        let rx_v = row3(rx_velocity_eci_km_s, index);
        let relative_v = [rx_v[0] - tx_v[0], rx_v[1] - tx_v[1], rx_v[2] - tx_v[2]];
        ranges.push(range);
        rates.push(dot(relative_v, delta) / range);
    }
    Ok(KinematicsResult {
        range_km: ranges,
        range_rate_km_s: rates,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatternResult {
    pub off_axis_rad: Vec<f64>,
    pub pass: Vec<u8>,
}

fn quaternion_dcm_bn(q: [f64; 4]) -> [[f64; 3]; 3] {
    let norm2 = q.iter().map(|value| value * value).sum::<f64>();
    let scale = if norm2 > 0.0 && norm2.is_finite() {
        norm2.sqrt().recip()
    } else {
        1.0
    };
    let (q0, q1, q2, q3) = (q[0] * scale, q[1] * scale, q[2] * scale, q[3] * scale);
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

fn mat_vec(matrix: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    [
        dot(matrix[0], value),
        dot(matrix[1], value),
        dot(matrix[2], value),
    ]
}

/// Evaluate an axisymmetric terminal pattern for a batch of peer directions.
pub fn terminal_pattern(
    peer_direction_eci: &[f64],
    dcm_parent_from_eci: &[f64],
    quat_parent_from_terminal: [f64; 4],
    half_angle_rad: f64,
) -> Result<PatternResult, String> {
    let count = validate_vector_batch(peer_direction_eci, "peer_direction_eci")?;
    if dcm_parent_from_eci.len() != count * 9 || !finite_slice(dcm_parent_from_eci) {
        return Err("dcm_parent_from_eci must be finite with shape (samples, 3, 3)".into());
    }
    if !finite_slice(&quat_parent_from_terminal) {
        return Err("terminal mounting quaternion must be finite".into());
    }
    if !half_angle_rad.is_finite() || !(0.0 < half_angle_rad && half_angle_rad <= PI) {
        return Err("half_angle_rad must be within (0, pi]".into());
    }
    let terminal_from_parent = quaternion_dcm_bn(quat_parent_from_terminal);
    // quaternion_to_dcm_bn returns body-from-parent for q_parent_from_terminal;
    // the terminal-from-parent transform is its transpose.
    let terminal_from_parent_t = [
        [
            terminal_from_parent[0][0],
            terminal_from_parent[1][0],
            terminal_from_parent[2][0],
        ],
        [
            terminal_from_parent[0][1],
            terminal_from_parent[1][1],
            terminal_from_parent[2][1],
        ],
        [
            terminal_from_parent[0][2],
            terminal_from_parent[1][2],
            terminal_from_parent[2][2],
        ],
    ];
    let mut angles = Vec::with_capacity(count);
    let mut pass = Vec::with_capacity(count);
    for index in 0..count {
        let peer = row3(peer_direction_eci, index);
        let peer_norm = norm(peer);
        if !peer_norm.is_finite() || peer_norm <= 0.0 {
            return Err("peer directions must be positive and finite".into());
        }
        let base = index * 9;
        let parent_from_eci = [
            [
                dcm_parent_from_eci[base],
                dcm_parent_from_eci[base + 1],
                dcm_parent_from_eci[base + 2],
            ],
            [
                dcm_parent_from_eci[base + 3],
                dcm_parent_from_eci[base + 4],
                dcm_parent_from_eci[base + 5],
            ],
            [
                dcm_parent_from_eci[base + 6],
                dcm_parent_from_eci[base + 7],
                dcm_parent_from_eci[base + 8],
            ],
        ];
        let parent_direction = mat_vec(parent_from_eci, peer);
        let terminal_direction = mat_vec(terminal_from_parent_t, parent_direction);
        let cosine = (terminal_direction[2] / peer_norm).clamp(-1.0, 1.0);
        let angle = cosine.acos();
        angles.push(angle);
        pass.push((angle <= half_angle_rad + 1.0e-12) as u8);
    }
    Ok(PatternResult {
        off_axis_rad: angles,
        pass,
    })
}

// ITU-R P.838-3 coefficient constants.  These are the exact public OEL
// coefficients and intentionally remain in the numeric kernel so a long RF
// sweep does not cross the Python boundary for every sample.
const KH: [[f64; 3]; 4] = [
    [-5.33980, -0.10008, 1.13098],
    [-0.35351, 1.26970, 0.45400],
    [-0.23789, 0.86036, 0.15354],
    [-0.94158, 0.64552, 0.16817],
];
const KV: [[f64; 3]; 4] = [
    [-3.80595, 0.56934, 0.81061],
    [-3.44965, -0.22911, 0.51059],
    [-0.39902, 0.73042, 0.11899],
    [0.50167, 1.07319, 0.27195],
];
const AH: [[f64; 3]; 5] = [
    [-0.14318, 1.82442, -0.55187],
    [0.29591, 0.77564, 0.19822],
    [0.32177, 0.63773, 0.13164],
    [-5.37610, -0.96230, 1.47828],
    [16.1721, -3.29980, 3.43990],
];
const AV: [[f64; 3]; 5] = [
    [-0.07771, 2.33840, -0.76284],
    [0.56727, 0.95545, 0.54039],
    [-0.20238, 1.14520, 0.26809],
    [-48.2991, 0.791669, 0.116226],
    [48.5833, 0.791459, 0.116479],
];

fn p838_coeff(
    frequency_ghz: f64,
    rows: &[[f64; 3]],
    slope: f64,
    intercept: f64,
    logarithmic: bool,
) -> f64 {
    let log_frequency = frequency_ghz.log10();
    let summed = rows.iter().fold(0.0, |total, row| {
        total + row[0] * (-((log_frequency - row[1]) / row[2]).powi(2)).exp()
    });
    let result = summed + slope * log_frequency + intercept;
    if logarithmic {
        10.0_f64.powf(result)
    } else {
        result
    }
}

pub fn p838_rain_specific(
    frequency_ghz: &[f64],
    rain_rate_mm_h: &[f64],
    elevation_deg: &[f64],
    polarization_tilt_deg: &[f64],
) -> Result<Vec<f64>, String> {
    let count = frequency_ghz.len();
    if count == 0
        || rain_rate_mm_h.len() != count
        || elevation_deg.len() != count
        || polarization_tilt_deg.len() != count
    {
        return Err("P.838 inputs must be non-empty and aligned".into());
    }
    if !finite_slice(frequency_ghz)
        || !finite_slice(rain_rate_mm_h)
        || !finite_slice(elevation_deg)
        || !finite_slice(polarization_tilt_deg)
        || frequency_ghz
            .iter()
            .any(|value| !(1.0..=1000.0).contains(value))
        || rain_rate_mm_h.iter().any(|value| *value < 0.0)
        || elevation_deg
            .iter()
            .any(|value| !(0.0..=90.0).contains(value))
    {
        return Err("P.838 inputs are outside the supported finite envelope".into());
    }
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let f = frequency_ghz[index];
        let elevation = elevation_deg[index].to_radians();
        let tilt = polarization_tilt_deg[index].to_radians();
        let kh = p838_coeff(f, &KH, -0.18961, 0.71147, true);
        let kv = p838_coeff(f, &KV, -0.16398, 0.63297, true);
        let ah = p838_coeff(f, &AH, 0.67849, -1.95537, false);
        let av = p838_coeff(f, &AV, -0.053739, 0.83433, false);
        let geometry = elevation.cos().powi(2) * (2.0 * tilt).cos();
        let k = 0.5 * (kh + kv + (kh - kv) * geometry);
        let alpha = 0.5 * (kh * ah + kv * av + (kh * ah - kv * av) * geometry) / k;
        output.push(k * rain_rate_mm_h[index].powf(alpha));
    }
    Ok(output)
}

pub fn p840_cloud_attenuation(
    frequency_ghz: &[f64],
    elevation_deg: &[f64],
    integrated_cloud_liquid_water_kg_m2: &[f64],
) -> Result<Vec<f64>, String> {
    let count = frequency_ghz.len();
    if count == 0
        || elevation_deg.len() != count
        || integrated_cloud_liquid_water_kg_m2.len() != count
    {
        return Err("P.840 inputs must be non-empty and aligned".into());
    }
    if !finite_slice(frequency_ghz)
        || !finite_slice(elevation_deg)
        || !finite_slice(integrated_cloud_liquid_water_kg_m2)
        || frequency_ghz
            .iter()
            .any(|value| !(1.0..=200.0).contains(value))
        || elevation_deg
            .iter()
            .any(|value| !(5.0..=90.0).contains(value))
        || integrated_cloud_liquid_water_kg_m2
            .iter()
            .any(|value| *value < 0.0)
    {
        return Err("P.840 inputs are outside the supported finite envelope".into());
    }
    let temperature = 273.75;
    let theta: f64 = 300.0 / temperature - 1.0;
    let epsilon_0 = 77.66 + 103.3 * theta;
    let epsilon_1 = 0.0671 * epsilon_0;
    let epsilon_2 = 3.52;
    let f_p = 20.20 - 146.0 * theta + 316.0 * theta.powi(2);
    let f_s = 39.8 * f_p;
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let frequency = frequency_ghz[index];
        let epsilon_imag = frequency * (epsilon_0 - epsilon_1)
            / (f_p * (1.0 + (frequency / f_p).powi(2)))
            + frequency * (epsilon_1 - epsilon_2) / (f_s * (1.0 + (frequency / f_s).powi(2)));
        let epsilon_real = (epsilon_0 - epsilon_1) / (1.0 + (frequency / f_p).powi(2))
            + (epsilon_1 - epsilon_2) / (1.0 + (frequency / f_s).powi(2))
            + epsilon_2;
        let eta = (2.0 + epsilon_real) / epsilon_imag;
        let base = 0.819 * frequency / (epsilon_imag * (1.0 + eta.powi(2)));
        let correction = 0.1522 * (-((frequency + 23.9589).powi(2)) / 3.2991e3).exp()
            + 11.51 * (-((frequency - 219.2096).powi(2)) / 2.7595e6).exp()
            - 10.4912;
        output.push(
            base * correction * integrated_cloud_liquid_water_kg_m2[index]
                / elevation_deg[index].to_radians().sin(),
        );
    }
    Ok(output)
}

pub fn p618_total_attenuation(
    gaseous_attenuation_db: &[f64],
    cloud_attenuation_db: &[f64],
    rain_attenuation_db: &[f64],
    scintillation_fade_depth_db: &[f64],
) -> Result<Vec<f64>, String> {
    let count = gaseous_attenuation_db.len();
    if count == 0
        || cloud_attenuation_db.len() != count
        || rain_attenuation_db.len() != count
        || scintillation_fade_depth_db.len() != count
        || !finite_slice(gaseous_attenuation_db)
        || !finite_slice(cloud_attenuation_db)
        || !finite_slice(rain_attenuation_db)
        || !finite_slice(scintillation_fade_depth_db)
        || gaseous_attenuation_db
            .iter()
            .chain(cloud_attenuation_db)
            .chain(rain_attenuation_db)
            .chain(scintillation_fade_depth_db)
            .any(|value| *value < 0.0)
    {
        return Err("P.618 attenuation components must be aligned, finite, and nonnegative".into());
    }
    Ok((0..count)
        .map(|index| {
            gaseous_attenuation_db[index]
                + ((rain_attenuation_db[index] + cloud_attenuation_db[index]).powi(2)
                    + scintillation_fade_depth_db[index].powi(2))
                .sqrt()
        })
        .collect())
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

    fn fixed4(values: Vec<f64>, name: &str) -> PyResult<[f64; 4]> {
        values
            .try_into()
            .map_err(|_| PyValueError::new_err(format!("{name} must have four values")))
    }

    fn decode_f64_bytes<'a>(values: &'a [u8], name: &str) -> PyResult<std::borrow::Cow<'a, [f64]>> {
        if values.len() % 8 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} bytes must contain complete float64 values"
            )));
        }
        #[cfg(target_endian = "little")]
        if values.as_ptr().align_offset(std::mem::align_of::<f64>()) == 0 {
            // SAFETY: &[u8] extraction accepts immutable PyBytes only (PyO3
            // conversions/std/slice.rs). The caller keeps the object alive
            // through the synchronous native call, including GIL detachment.
            // Alignment/size are checked and every f64 bit pattern is valid.
            let view = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<f64>(), values.len() / 8)
            };
            return Ok(std::borrow::Cow::Borrowed(view));
        }
        Ok(std::borrow::Cow::Owned(
            values
                .chunks_exact(8)
                .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
                .collect(),
        ))
    }

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
    fn link_free_space_ledger(
        py: Python<'_>,
        range_km: Vec<f64>,
        tx_gain_dbi: Vec<f64>,
        rx_gain_dbi: Vec<f64>,
        carrier_frequency_hz: f64,
        tx_power_w: f64,
        data_rate_bps: f64,
        system_noise_temperature_k: f64,
        required_eb_n0_db: f64,
        tx_line_loss_db: f64,
        rx_line_loss_db: f64,
        misc_loss_db: f64,
    ) -> PyResult<(
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<bool>,
    )> {
        let result = py
            .detach(|| {
                free_space_link(
                    &range_km,
                    &tx_gain_dbi,
                    &rx_gain_dbi,
                    carrier_frequency_hz,
                    tx_power_w,
                    data_rate_bps,
                    system_noise_temperature_k,
                    required_eb_n0_db,
                    tx_line_loss_db,
                    rx_line_loss_db,
                    misc_loss_db,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            result.tx_power_dbw,
            result.tx_gain_dbi,
            result.rx_gain_dbi,
            result.free_space_path_loss_db,
            result.eirp_dbw,
            result.received_power_dbw,
            result.noise_density_dbw_hz,
            result.cn0_db_hz,
            result.eb_n0_db,
            result.margin_db,
            result
                .margin_pass
                .into_iter()
                .map(|value| value != 0)
                .collect(),
        ))
    }

    #[pyfunction]
    fn link_free_space_ledger_bytes(
        py: Python<'_>,
        range_km: &[u8],
        tx_gain_dbi: &[u8],
        rx_gain_dbi: &[u8],
        carrier_frequency_hz: f64,
        tx_power_w: f64,
        data_rate_bps: f64,
        system_noise_temperature_k: f64,
        required_eb_n0_db: f64,
        tx_line_loss_db: f64,
        rx_line_loss_db: f64,
        misc_loss_db: f64,
    ) -> PyResult<(
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
    )> {
        let range_km = decode_f64_bytes(range_km, "range_km")?;
        let tx_gain_dbi = decode_f64_bytes(tx_gain_dbi, "tx_gain_dbi")?;
        let rx_gain_dbi = decode_f64_bytes(rx_gain_dbi, "rx_gain_dbi")?;
        let result = py
            .detach(|| {
                free_space_link(
                    &range_km,
                    &tx_gain_dbi,
                    &rx_gain_dbi,
                    carrier_frequency_hz,
                    tx_power_w,
                    data_rate_bps,
                    system_noise_temperature_k,
                    required_eb_n0_db,
                    tx_line_loss_db,
                    rx_line_loss_db,
                    misc_loss_db,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            encode_f64_bytes(py, &result.tx_power_dbw)?,
            encode_f64_bytes(py, &result.tx_gain_dbi)?,
            encode_f64_bytes(py, &result.rx_gain_dbi)?,
            encode_f64_bytes(py, &result.free_space_path_loss_db)?,
            encode_f64_bytes(py, &result.eirp_dbw)?,
            encode_f64_bytes(py, &result.received_power_dbw)?,
            encode_f64_bytes(py, &result.noise_density_dbw_hz)?,
            encode_f64_bytes(py, &result.cn0_db_hz)?,
            encode_f64_bytes(py, &result.eb_n0_db)?,
            encode_f64_bytes(py, &result.margin_db)?,
            PyBytes::new(py, &result.margin_pass).unbind(),
        ))
    }

    #[pyfunction]
    fn link_earth_occulted_bytes(
        py: Python<'_>,
        tx_position_ecef_km: &[u8],
        rx_position_ecef_km: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let tx = decode_f64_bytes(tx_position_ecef_km, "tx_position_ecef_km")?;
        let rx = decode_f64_bytes(rx_position_ecef_km, "rx_position_ecef_km")?;
        let flags = py
            .detach(|| earth_occulted(&tx, &rx))
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &flags).unbind())
    }

    #[pyfunction]
    fn link_endpoint_kinematics_bytes(
        py: Python<'_>,
        tx_position_eci_km: &[u8],
        rx_position_eci_km: &[u8],
        tx_velocity_eci_km_s: &[u8],
        rx_velocity_eci_km_s: &[u8],
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let tx = decode_f64_bytes(tx_position_eci_km, "tx_position_eci_km")?;
        let rx = decode_f64_bytes(rx_position_eci_km, "rx_position_eci_km")?;
        let tv = decode_f64_bytes(tx_velocity_eci_km_s, "tx_velocity_eci_km_s")?;
        let rv = decode_f64_bytes(rx_velocity_eci_km_s, "rx_velocity_eci_km_s")?;
        let result = py
            .detach(|| endpoint_kinematics(&tx, &rx, &tv, &rv))
            .map_err(PyValueError::new_err)?;
        Ok((
            encode_f64_bytes(py, &result.range_km)?,
            encode_f64_bytes(py, &result.range_rate_km_s)?,
        ))
    }

    #[pyfunction]
    fn link_earth_occulted(
        py: Python<'_>,
        tx_position_ecef_km: Vec<f64>,
        rx_position_ecef_km: Vec<f64>,
    ) -> PyResult<Vec<bool>> {
        Ok(py
            .detach(|| earth_occulted(&tx_position_ecef_km, &rx_position_ecef_km))
            .map_err(PyValueError::new_err)?
            .into_iter()
            .map(|value| value != 0)
            .collect())
    }

    #[pyfunction]
    fn link_endpoint_kinematics(
        py: Python<'_>,
        tx_position_eci_km: Vec<f64>,
        rx_position_eci_km: Vec<f64>,
        tx_velocity_eci_km_s: Vec<f64>,
        rx_velocity_eci_km_s: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let result = py
            .detach(|| {
                endpoint_kinematics(
                    &tx_position_eci_km,
                    &rx_position_eci_km,
                    &tx_velocity_eci_km_s,
                    &rx_velocity_eci_km_s,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((result.range_km, result.range_rate_km_s))
    }

    #[pyfunction]
    fn link_terminal_pattern(
        py: Python<'_>,
        peer_direction_eci: Vec<f64>,
        dcm_parent_from_eci: Vec<f64>,
        quat_parent_from_terminal: Vec<f64>,
        half_angle_rad: f64,
    ) -> PyResult<(Vec<f64>, Vec<bool>)> {
        let mounting = fixed4(quat_parent_from_terminal, "quat_parent_from_terminal")?;
        let result = py
            .detach(|| {
                terminal_pattern(
                    &peer_direction_eci,
                    &dcm_parent_from_eci,
                    mounting,
                    half_angle_rad,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            result.off_axis_rad,
            result.pass.into_iter().map(|value| value != 0).collect(),
        ))
    }

    #[pyfunction]
    fn link_p838_rain_specific(
        py: Python<'_>,
        frequency_ghz: Vec<f64>,
        rain_rate_mm_h: Vec<f64>,
        elevation_deg: Vec<f64>,
        polarization_tilt_deg: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            p838_rain_specific(
                &frequency_ghz,
                &rain_rate_mm_h,
                &elevation_deg,
                &polarization_tilt_deg,
            )
        })
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn link_p840_cloud_attenuation(
        py: Python<'_>,
        frequency_ghz: Vec<f64>,
        elevation_deg: Vec<f64>,
        integrated_cloud_liquid_water_kg_m2: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            p840_cloud_attenuation(
                &frequency_ghz,
                &elevation_deg,
                &integrated_cloud_liquid_water_kg_m2,
            )
        })
        .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn link_p618_total_attenuation(
        py: Python<'_>,
        gaseous_attenuation_db: Vec<f64>,
        cloud_attenuation_db: Vec<f64>,
        rain_attenuation_db: Vec<f64>,
        scintillation_fade_depth_db: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            p618_total_attenuation(
                &gaseous_attenuation_db,
                &cloud_attenuation_db,
                &rain_attenuation_db,
                &scintillation_fade_depth_db,
            )
        })
        .map_err(PyValueError::new_err)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(link_free_space_ledger, module)?)?;
        module.add_function(wrap_pyfunction!(link_free_space_ledger_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(link_earth_occulted, module)?)?;
        module.add_function(wrap_pyfunction!(link_earth_occulted_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(link_endpoint_kinematics_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(link_endpoint_kinematics, module)?)?;
        module.add_function(wrap_pyfunction!(link_terminal_pattern, module)?)?;
        module.add_function(wrap_pyfunction!(link_p838_rain_specific, module)?)?;
        module.add_function(wrap_pyfunction!(link_p840_cloud_attenuation, module)?)?;
        module.add_function(wrap_pyfunction!(link_p618_total_attenuation, module)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_space_doubles_range_by_twenty_db() {
        let first = free_space_link(
            &[1000.0, 2000.0],
            &[6.0, 6.0],
            &[12.0, 12.0],
            2.2e9,
            10.0,
            1.0e6,
            500.0,
            9.6,
            1.0,
            2.0,
            3.0,
        )
        .unwrap();
        assert!(
            (first.free_space_path_loss_db[1]
                - first.free_space_path_loss_db[0]
                - 20.0 * 2.0_f64.log10())
            .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn tangent_segment_is_occulted() {
        let tangent =
            earth_occulted(&[WGS84_A_KM, -1000.0, 0.0], &[WGS84_A_KM, 1000.0, 0.0]).unwrap();
        assert_eq!(tangent, vec![1]);
        let crossing = earth_occulted(&[7000.0, 0.0, 0.0], &[-7000.0, 0.0, 0.0]).unwrap();
        assert_eq!(crossing, vec![1]);
    }
}
