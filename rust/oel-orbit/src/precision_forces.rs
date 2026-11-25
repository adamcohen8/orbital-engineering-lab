//! Public Earth radiation, Schwarzschild and shared tidal harmonic kernels.
//!
//! This is the authoritative numeric owner used by OEL and its standalone
//! public precision-force crate. Caller-owned epochs, EOP, frames, ephemerides
//! and quadrature are unchanged. Positions and velocities use km and km/s; accelerations use km/s²
//! and radiation components use N/m².

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const DAY_S: f64 = 86_400.0;
const SPEED_OF_LIGHT_KM_S: f64 = 299_792.458;
pub(crate) const AU_KM: f64 = 149_597_870.7;
const SOLAR_PRESSURE_PA: f64 = 4.5606e-6;

pub(crate) fn finite_vector(vector: [f64; 3], label: &str) -> Result<(), String> {
    if vector.iter().any(|value| !value.is_finite()) {
        Err(format!("{label} must contain only finite values"))
    } else {
        Ok(())
    }
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

/// Central-body first post-Newtonian Schwarzschild correction in km/s².
pub fn schwarzschild_acceleration(state: [f64; 6], mu_km3_s2: f64) -> Result<[f64; 3], String> {
    if state.iter().any(|value| !value.is_finite()) || !mu_km3_s2.is_finite() || mu_km3_s2 <= 0.0 {
        return Err("Schwarzschild requires a finite state and positive finite mu".to_owned());
    }
    let position = [state[0], state[1], state[2]];
    let velocity = [state[3], state[4], state[5]];
    let radius = norm(position);
    if radius <= 0.0 {
        return Err("Schwarzschild requires a positive geocentric radius".to_owned());
    }
    let v2 = dot(velocity, velocity);
    let rv = dot(position, velocity);
    let scale = mu_km3_s2 / (SPEED_OF_LIGHT_KM_S * SPEED_OF_LIGHT_KM_S * radius.powi(3));
    Ok(std::array::from_fn(|axis| {
        scale * ((4.0 * mu_km3_s2 / radius - v2) * position[axis] + 4.0 * rv * velocity[axis])
    }))
}

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn scale(self, value: f64) -> Self {
        Self::new(self.re * value, self.im * value)
    }
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

fn factorial(value: usize) -> f64 {
    (2..=value).fold(1.0, |acc, next| acc * next as f64)
}

fn build_harmonic_normalizations(degree: usize) -> Arc<[f64]> {
    let mut values = vec![0.0; (degree + 1) * (degree + 1)];
    for n in 0..=degree {
        for m in 0..=n {
            values[n * (degree + 1) + m] =
                (((if m == 0 { 1 } else { 2 }) as f64) * (2 * n + 1) as f64 * factorial(n - m)
                    / factorial(n + m))
                .sqrt();
        }
    }
    values.into()
}

fn harmonic_normalizations(degree: usize) -> Arc<[f64]> {
    // Tide contexts support degrees through 20. Each immutable normalization
    // table is initialized once, with no cache lock or eviction on force calls.
    static TIDE_CACHE: [OnceLock<Arc<[f64]>>; 21] = [const { OnceLock::new() }; 21];
    if let Some(slot) = TIDE_CACHE.get(degree) {
        return Arc::clone(slot.get_or_init(|| build_harmonic_normalizations(degree)));
    }
    // Preserve the existing bounded arbitrary-degree path for public callers.
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<[f64]>>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(values) = cache.get(&degree) {
        return Arc::clone(values);
    }
    let values = build_harmonic_normalizations(degree);
    if cache.len() >= 8 {
        cache.clear();
    }
    cache.insert(degree, Arc::clone(&values));
    values
}

/// Body solid harmonics without gradient work; same recurrence and normalization.
pub(crate) fn solid_harmonics_values(position: [f64; 3], degree: usize) -> Vec<(f64, f64)> {
    let count = (degree + 1) * (degree + 1);
    let mut h = vec![Complex::default(); count];
    let index = |n: usize, m: usize| n * (degree + 1) + m;
    let [x, y, z] = position;
    let r2 = dot(position, position);
    h[0] = Complex::new(1.0, 0.0);
    for m in 0..=degree {
        if m > 0 {
            h[index(m, m)] = Complex::new(x, y)
                .mul(h[index(m - 1, m - 1)])
                .scale((2 * m - 1) as f64);
        }
        if m < degree {
            h[index(m + 1, m)] = h[index(m, m)].scale((2 * m + 1) as f64 * z);
        }
        for n in (m + 2)..=degree {
            let first = h[index(n - 1, m)].scale((2 * n - 1) as f64 * z);
            let second = h[index(n - 2, m)].scale((n + m - 1) as f64 * r2);
            h[index(n, m)] = first.sub(second).scale(1.0 / (n - m) as f64);
        }
    }
    let normalizations = harmonic_normalizations(degree);
    h.iter()
        .zip(normalizations.iter())
        .map(|(value, scale)| {
            let normalized = value.scale(*scale);
            (normalized.re, normalized.im)
        })
        .collect()
}

/// Evaluate normalized solid harmonics and their Cartesian gradients.
fn solid_harmonics_into(
    position: [f64; 3],
    degree: usize,
    h: &mut [Complex],
    g: &mut [[Complex; 3]],
) {
    let index = |n: usize, m: usize| n * (degree + 1) + m;
    let [x, y, z] = position;
    let r2 = dot(position, position);
    h[0] = Complex::new(1.0, 0.0);
    for m in 0..=degree {
        if m > 0 {
            let current = index(m, m);
            let prior = h[index(m - 1, m - 1)];
            let prior_g = g[index(m - 1, m - 1)];
            let factor = (2 * m - 1) as f64;
            let xy = Complex::new(x, y);
            h[current] = xy.mul(prior).scale(factor);
            g[current] = [
                xy.mul(prior_g[0])
                    .add(Complex::new(1.0, 0.0).mul(prior))
                    .scale(factor),
                xy.mul(prior_g[1])
                    .add(Complex::new(0.0, 1.0).mul(prior))
                    .scale(factor),
                xy.mul(prior_g[2]).scale(factor),
            ];
        }
        if m < degree {
            let current = index(m + 1, m);
            let prior = index(m, m);
            let factor = (2 * m + 1) as f64;
            h[current] = h[prior].scale(factor * z);
            g[current] = [
                g[prior][0].scale(factor * z),
                g[prior][1].scale(factor * z),
                g[prior][2].add(h[prior]).scale(factor),
            ];
        }
        for n in (m + 2)..=degree {
            let current = index(n, m);
            let prior = index(n - 1, m);
            let prior_two = index(n - 2, m);
            let denominator = (n - m) as f64;
            let first = h[prior].scale((2 * n - 1) as f64 * z);
            let second = h[prior_two].scale((n + m - 1) as f64 * r2);
            h[current] = first.sub(second).scale(1.0 / denominator);
            let mut gradient = [Complex::default(); 3];
            for axis in 0..3 {
                let z_term = g[prior][axis]
                    .scale((2 * n - 1) as f64 * z)
                    .add(h[prior].scale((2 * n - 1) as f64 * if axis == 2 { 1.0 } else { 0.0 }));
                let r_term = g[prior_two][axis]
                    .scale((n + m - 1) as f64 * r2)
                    .add(h[prior_two].scale((n + m - 1) as f64 * 2.0 * position[axis]));
                gradient[axis] = z_term.sub(r_term).scale(1.0 / denominator);
            }
            g[current] = gradient;
        }
    }
    let normalizations = harmonic_normalizations(degree);
    for n in 0..=degree {
        for m in 0..=n {
            let current = index(n, m);
            let scale = normalizations[current];
            h[current] = h[current].scale(scale);
            for axis in 0..3 {
                g[current][axis] = g[current][axis].scale(scale);
            }
        }
    }
}

/// Evaluate additive normalized tidal gravity from prepared C/S coefficients.
///
/// `c_nm` and `s_nm` are row-major square arrays with side `degree + 1`.
pub fn tidal_acceleration_fixed(
    position_km: [f64; 3],
    c_nm: &[f64],
    s_nm: &[f64],
    degree: usize,
    mu_km3_s2: f64,
    reference_radius_km: f64,
) -> Result<[f64; 3], String> {
    finite_vector(position_km, "spacecraft position")?;
    let coefficient_count = degree
        .checked_add(1)
        .and_then(|side| side.checked_mul(side));
    if coefficient_count != Some(c_nm.len()) || s_nm.len() != c_nm.len() {
        return Err("tidal coefficient dimensions are invalid".to_owned());
    }
    if !c_nm.iter().chain(s_nm).all(|value| value.is_finite())
        || ![mu_km3_s2, reference_radius_km]
            .iter()
            .all(|value| value.is_finite())
        || mu_km3_s2 <= 0.0
        || reference_radius_km <= 0.0
    {
        return Err("tidal parameters must be finite with positive mu and radius".to_owned());
    }
    let radius = norm(position_km);
    if radius <= 0.0 {
        return Err("tidal acceleration requires a positive radius".to_owned());
    }
    let unit_position = std::array::from_fn(|axis| position_km[axis] / radius);
    // Common solid/ocean degrees avoid allocating scratch on every force call.
    // Larger and arbitrary degrees retain dynamically sized scratch.
    if degree <= 6 {
        let mut h = [Complex::default(); 49];
        let mut g = [[Complex::default(); 3]; 49];
        solid_harmonics_into(unit_position, degree, &mut h, &mut g);
        return Ok(tidal_gradient_sum(
            position_km,
            radius,
            c_nm,
            s_nm,
            degree,
            mu_km3_s2,
            reference_radius_km,
            &h,
            &g,
        ));
    }
    let mut h = vec![Complex::default(); c_nm.len()];
    let mut g = vec![[Complex::default(); 3]; c_nm.len()];
    solid_harmonics_into(unit_position, degree, &mut h, &mut g);
    Ok(tidal_gradient_sum(
        position_km,
        radius,
        c_nm,
        s_nm,
        degree,
        mu_km3_s2,
        reference_radius_km,
        &h,
        &g,
    ))
}

#[allow(clippy::too_many_arguments)]
fn tidal_gradient_sum(
    position_km: [f64; 3],
    radius: f64,
    c_nm: &[f64],
    s_nm: &[f64],
    degree: usize,
    mu_km3_s2: f64,
    reference_radius_km: f64,
    h: &[Complex],
    g: &[[Complex; 3]],
) -> [f64; 3] {
    let index = |n: usize, m: usize| n * (degree + 1) + m;
    let mut acceleration = [0.0; 3];
    for n in 2..=degree {
        let scale = mu_km3_s2 / radius.powi(2) * (reference_radius_km / radius).powi(n as i32);
        for m in 0..=n {
            let idx = index(n, m);
            for axis in 0..3 {
                let gradient =
                    g[idx][axis].sub(h[idx].scale((2 * n + 1) as f64 * position_km[axis] / radius));
                acceleration[axis] += scale * (c_nm[idx] * gradient.re + s_nm[idx] * gradient.im);
            }
        }
    }
    acceleration
}

/// Earth-disk albedo and infrared pressure in the supplied Earth-fixed frame.
///
/// The quadrature nodes and weights are generated by Python's established
/// Gauss-Legendre implementation and passed in, so this kernel does not own
/// resource/file I/O or quadrature policy.
#[allow(clippy::too_many_arguments)]
pub fn earth_radiation_components(
    position_km: [f64; 3],
    sun_position_km: [f64; 3],
    elapsed_s: f64,
    radius_km: f64,
    nodes: &[f64],
    weights_gauss: &[f64],
    cos_azimuth: &[f64],
    sin_azimuth: &[f64],
    include_albedo: bool,
    include_infrared: bool,
    uniform_albedo: Option<f64>,
    uniform_emissivity: Option<f64>,
) -> Result<([f64; 3], [f64; 3]), String> {
    finite_vector(position_km, "spacecraft position")?;
    finite_vector(sun_position_km, "Sun position")?;
    if !elapsed_s.is_finite()
        || !radius_km.is_finite()
        || radius_km <= 0.0
        || nodes.is_empty()
        || nodes.len() != weights_gauss.len()
        || cos_azimuth.len() != 4 * nodes.len()
        || sin_azimuth.len() != cos_azimuth.len()
    {
        return Err("Earth radiation geometry or quadrature dimensions are invalid".to_owned());
    }
    if nodes
        .iter()
        .chain(weights_gauss)
        .chain(cos_azimuth)
        .chain(sin_azimuth)
        .any(|value| !value.is_finite())
    {
        return Err("Earth radiation quadrature values must be finite".to_owned());
    }
    if let (Some(albedo), Some(emissivity)) = (uniform_albedo, uniform_emissivity) {
        if !albedo.is_finite()
            || !emissivity.is_finite()
            || !(0.0..=1.0).contains(&albedo)
            || !(0.0..=1.0).contains(&emissivity)
        {
            return Err("uniform albedo and emissivity must lie in [0, 1]".to_owned());
        }
    } else if uniform_albedo.is_some() || uniform_emissivity.is_some() {
        return Err("uniform albedo and emissivity must be supplied together".to_owned());
    }
    earth_radiation_components_prevalidated(
        position_km,
        sun_position_km,
        elapsed_s,
        radius_km,
        nodes,
        weights_gauss,
        cos_azimuth,
        sin_azimuth,
        include_albedo,
        include_infrared,
        uniform_albedo,
        uniform_emissivity,
    )
}

// Batch rows share quadrature/radius/coefficient validation. The first row goes
// through the full public contract, retaining validation error precedence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn earth_radiation_components_prevalidated(
    position_km: [f64; 3],
    sun_position_km: [f64; 3],
    elapsed_s: f64,
    radius_km: f64,
    nodes: &[f64],
    weights_gauss: &[f64],
    cos_azimuth: &[f64],
    sin_azimuth: &[f64],
    include_albedo: bool,
    include_infrared: bool,
    uniform_albedo: Option<f64>,
    uniform_emissivity: Option<f64>,
) -> Result<([f64; 3], [f64; 3]), String> {
    finite_vector(position_km, "spacecraft position")?;
    finite_vector(sun_position_km, "Sun position")?;
    if !elapsed_s.is_finite() {
        return Err("Earth radiation geometry or quadrature dimensions are invalid".to_owned());
    }
    if !include_albedo && !include_infrared {
        return Ok(([0.0; 3], [0.0; 3]));
    }
    let integrate = match (include_albedo, include_infrared, uniform_albedo.is_some()) {
        (true, true, false) => earth_radiation_integrate::<true, true, false>,
        (true, false, false) => earth_radiation_integrate::<true, false, false>,
        (false, true, false) => earth_radiation_integrate::<false, true, false>,
        (true, true, true) => earth_radiation_integrate::<true, true, true>,
        (true, false, true) => earth_radiation_integrate::<true, false, true>,
        (false, true, true) => earth_radiation_integrate::<false, true, true>,
        (false, false, _) => unreachable!("disabled components returned above"),
    };
    integrate(
        position_km,
        sun_position_km,
        elapsed_s,
        radius_km,
        nodes,
        weights_gauss,
        cos_azimuth,
        sin_azimuth,
        uniform_albedo,
        uniform_emissivity,
    )
}

// Four independent rays use ordinary IEEE operations, without fused arithmetic
// or horizontal sums. AArch64's mandatory NEON handles two lanes per register;
// other targets retain the exact portable lane calculation.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct RadiationRays {
    #[cfg(target_arch = "aarch64")]
    value: [std::arch::aarch64::float64x2_t; 2],
    #[cfg(not(target_arch = "aarch64"))]
    value: [f64; 4],
}

#[cfg(target_arch = "aarch64")]
impl RadiationRays {
    #[inline(always)]
    fn new(value: [f64; 4]) -> Self {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on AArch64; four input lanes are in bounds.
        unsafe {
            Self {
                value: [
                    std::arch::aarch64::vld1q_f64(value.as_ptr()),
                    std::arch::aarch64::vld1q_f64(value.as_ptr().add(2)),
                ],
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self { value }
        }
    }
    #[inline(always)]
    fn splat(value: f64) -> Self {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on AArch64.
        unsafe {
            Self {
                value: [std::arch::aarch64::vdupq_n_f64(value); 2],
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self { value: [value; 4] }
        }
    }
    #[inline(always)]
    fn lanes(self) -> [f64; 4] {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: The local array has space for four output lanes.
        unsafe {
            let mut lanes = [0.0; 4];
            std::arch::aarch64::vst1q_f64(lanes.as_mut_ptr(), self.value[0]);
            std::arch::aarch64::vst1q_f64(lanes.as_mut_ptr().add(2), self.value[1]);
            lanes
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.value
        }
    }
    #[inline(always)]
    fn max_zero(self) -> Self {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: Mandatory NEON. FMAXNM matches f64::max's NaN handling.
        unsafe {
            Self {
                value: self.value.map(|value| {
                    std::arch::aarch64::vmaxnmq_f64(value, std::arch::aarch64::vdupq_n_f64(0.0))
                }),
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self {
                value: self.value.map(|value| value.max(0.0)),
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
macro_rules! radiation_ray_operation {
    ($trait:ident, $method:ident, $intrinsic:ident, $operator:tt) => {
        impl std::ops::$trait for RadiationRays {
            type Output = Self;
            #[inline(always)]
            fn $method(self, other: Self) -> Self {
                #[cfg(target_arch = "aarch64")]
                // SAFETY: These are ordinary lane-wise mandatory NEON operations.
                unsafe { Self { value: [std::arch::aarch64::$intrinsic(self.value[0], other.value[0]),
                    std::arch::aarch64::$intrinsic(self.value[1], other.value[1])] } }
                #[cfg(not(target_arch = "aarch64"))]
                { Self::new(std::array::from_fn(|lane| self.value[lane] $operator other.value[lane])) }
            }
        }
    };
}
#[cfg(target_arch = "aarch64")]
radiation_ray_operation!(Add, add, vaddq_f64, +);
#[cfg(target_arch = "aarch64")]
radiation_ray_operation!(Sub, sub, vsubq_f64, -);
#[cfg(target_arch = "aarch64")]
radiation_ray_operation!(Mul, mul, vmulq_f64, *);
#[cfg(target_arch = "aarch64")]
radiation_ray_operation!(Div, div, vdivq_f64, /);

// Specialize component/coefficient choices before the ray loop. Contributions
// from independent rays are added in the original radial-node/azimuth order.
#[allow(clippy::too_many_arguments)]
fn earth_radiation_integrate<const ALBEDO: bool, const INFRARED: bool, const UNIFORM: bool>(
    position_km: [f64; 3],
    sun_position_km: [f64; 3],
    elapsed_s: f64,
    radius_km: f64,
    nodes: &[f64],
    weights_gauss: &[f64],
    cos_azimuth: &[f64],
    sin_azimuth: &[f64],
    uniform_albedo: Option<f64>,
    uniform_emissivity: Option<f64>,
) -> Result<([f64; 3], [f64; 3]), String> {
    let distance = norm(position_km);
    let sun_distance = norm(sun_position_km);
    if distance <= radius_km || sun_distance <= radius_km {
        return Err("Earth radiation requires spacecraft and Sun outside Earth".to_owned());
    }
    let radial = [
        position_km[0] / distance,
        position_km[1] / distance,
        position_km[2] / distance,
    ];
    let mut east = [-radial[1], radial[0], 0.0];
    let east_norm = norm(east);
    if east_norm < 1.0e-12 {
        east = [0.0, 1.0, 0.0];
    } else {
        east = east.map(|value| value / east_norm);
    }
    let north = [
        radial[1] * east[2] - radial[2] * east[1],
        radial[2] * east[0] - radial[0] * east[2],
        radial[0] * east[1] - radial[1] * east[0],
    ];
    let ratio = radius_km / distance;
    let cos_limb = (1.0 - ratio * ratio).max(0.0).sqrt();
    let width = ratio * ratio / (1.0 + cos_limb);
    let pressure = SOLAR_PRESSURE_PA * (AU_KM / sun_distance).powi(2);
    let sun_hat = sun_position_km.map(|value| value / sun_distance);
    #[cfg(target_arch = "aarch64")]
    let mut pressure_totals = [[0.0; 2]; 3];
    let order = nodes.len();
    let seasonal = (2.0 * std::f64::consts::PI * elapsed_s / (365.25 * DAY_S)).cos();
    #[cfg(target_arch = "aarch64")]
    {
        let tangents: Vec<[RadiationRays; 3]> = (0..cos_azimuth.len())
            .step_by(4)
            .map(|start| {
                std::array::from_fn(|axis| {
                    RadiationRays::new(std::array::from_fn(|lane| {
                        cos_azimuth[start + lane] * east[axis]
                            + sin_azimuth[start + lane] * north[axis]
                    }))
                })
            })
            .collect();
        let position_pair = position_km.map(RadiationRays::splat);
        let radius_pair = RadiationRays::splat(radius_km);
        let sun_pair = sun_hat.map(RadiationRays::splat);
        let pressure_pair = RadiationRays::splat(pressure);
        let seasonal_albedo = RadiationRays::splat(0.10 * seasonal);
        let seasonal_infrared = RadiationRays::splat(0.07 * seasonal);
        for (i, &node) in nodes.iter().enumerate() {
            let cosine = 1.0 - width * (1.0 - node) / 2.0;
            let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
            let root = (radius_km * radius_km - distance * distance * (1.0 - cosine * cosine))
                .max(0.0)
                .sqrt();
            let ray_length =
                (distance * distance - radius_km * radius_km) / (distance * cosine + root);
            let quadrature_weight = weights_gauss[i]
                * (width / 2.0)
                * (2.0 * std::f64::consts::PI / (4.0 * order as f64));
            let radial_pair = radial.map(|value| RadiationRays::splat(cosine * value));
            let sine_pair = RadiationRays::splat(sine);
            let ray_pair = RadiationRays::splat(ray_length);
            let weight_pair = RadiationRays::splat(quadrature_weight);
            for tangent in &tangents {
                let direction: [RadiationRays; 3] =
                    std::array::from_fn(|axis| radial_pair[axis] + sine_pair * tangent[axis]);
                let normal: [RadiationRays; 3] = std::array::from_fn(|axis| {
                    (position_pair[axis] - ray_pair * direction[axis]) / radius_pair
                });
                let latitude = normal[2];
                let p2 = RadiationRays::splat(0.5)
                    * (RadiationRays::splat(3.0) * latitude * latitude - RadiationRays::splat(1.0));
                let albedo = if UNIFORM {
                    RadiationRays::splat(uniform_albedo.unwrap())
                } else {
                    RadiationRays::splat(0.34)
                        + seasonal_albedo * latitude
                        + RadiationRays::splat(0.29) * p2
                };
                let emissivity = if UNIFORM {
                    RadiationRays::splat(uniform_emissivity.unwrap())
                } else {
                    RadiationRays::splat(0.68)
                        - seasonal_infrared * latitude
                        - RadiationRays::splat(0.18) * p2
                };
                let incidence = if ALBEDO {
                    (normal[0] * sun_pair[0] + normal[1] * sun_pair[1] + normal[2] * sun_pair[2])
                        .max_zero()
                } else {
                    RadiationRays::splat(0.0)
                };
                let albedo_rays: [[f64; 4]; 3] = if ALBEDO {
                    std::array::from_fn(|axis| {
                        (weight_pair * direction[axis] * albedo * incidence * pressure_pair
                            / RadiationRays::splat(std::f64::consts::PI))
                        .lanes()
                    })
                } else {
                    [[0.0; 4]; 3]
                };
                let infrared_rays: [[f64; 4]; 3] = if INFRARED {
                    std::array::from_fn(|axis| {
                        (weight_pair * direction[axis] * emissivity * pressure_pair
                            / RadiationRays::splat(4.0 * std::f64::consts::PI))
                        .lanes()
                    })
                } else {
                    [[0.0; 4]; 3]
                };
                for lane in 0..4 {
                    for axis in 0..3 {
                        if ALBEDO {
                            pressure_totals[axis][0] += albedo_rays[axis][lane];
                        }
                        if INFRARED {
                            pressure_totals[axis][1] += infrared_rays[axis][lane];
                        }
                    }
                }
            }
        }
        Ok((
            pressure_totals.map(|pair| pair[0]),
            pressure_totals.map(|pair| pair[1]),
        ))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut albedo_pressure = [0.0; 3];
        let mut infrared_pressure = [0.0; 3];
        let tangents: Vec<[f64; 3]> = cos_azimuth
            .iter()
            .zip(sin_azimuth)
            .map(|(&cosine, &sine)| {
                [
                    cosine * east[0] + sine * north[0],
                    cosine * east[1] + sine * north[1],
                    cosine * east[2] + sine * north[2],
                ]
            })
            .collect();
        for (i, &node) in nodes.iter().enumerate() {
            let cosine = 1.0 - width * (1.0 - node) / 2.0;
            let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
            let root = (radius_km * radius_km - distance * distance * (1.0 - cosine * cosine))
                .max(0.0)
                .sqrt();
            let ray_length =
                (distance * distance - radius_km * radius_km) / (distance * cosine + root);
            let quadrature_weight = weights_gauss[i]
                * (width / 2.0)
                * (2.0 * std::f64::consts::PI / (4.0 * order as f64));
            for tangent in &tangents {
                let direction = [
                    cosine * radial[0] + sine * tangent[0],
                    cosine * radial[1] + sine * tangent[1],
                    cosine * radial[2] + sine * tangent[2],
                ];
                let normal = [
                    (position_km[0] - ray_length * direction[0]) / radius_km,
                    (position_km[1] - ray_length * direction[1]) / radius_km,
                    (position_km[2] - ray_length * direction[2]) / radius_km,
                ];
                let (albedo, emissivity) = if UNIFORM {
                    (uniform_albedo.unwrap(), uniform_emissivity.unwrap())
                } else {
                    let latitude = normal[2];
                    let p2 = 0.5 * (3.0 * latitude * latitude - 1.0);
                    (
                        0.34 + 0.10 * seasonal * latitude + 0.29 * p2,
                        0.68 - 0.07 * seasonal * latitude - 0.18 * p2,
                    )
                };
                if ALBEDO {
                    let incidence = dot(normal, sun_hat).max(0.0);
                    for axis in 0..3 {
                        albedo_pressure[axis] +=
                            quadrature_weight * direction[axis] * albedo * incidence * pressure
                                / std::f64::consts::PI;
                    }
                }
                if INFRARED {
                    for axis in 0..3 {
                        infrared_pressure[axis] +=
                            quadrature_weight * direction[axis] * emissivity * pressure
                                / (4.0 * std::f64::consts::PI);
                    }
                }
            }
        }
        Ok((albedo_pressure, infrared_pressure))
    }
}

#[cfg(test)]
#[path = "earth_radiation_reference_tests.rs"]
mod earth_radiation_reference_tests;
