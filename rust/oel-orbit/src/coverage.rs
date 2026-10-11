//! Vectorized coverage geometry and sampled coverage aggregation.
//!
//! The public Python coverage products keep their orchestration, frame
//! preparation, provenance, and artifact contracts in Python.  This module
//! owns the hot, pure-numeric kernels only.  Distances are kilometres and
//! times are seconds.  Boolean arrays crossing the Python boundary are
//! represented as `u8` internally so the same code can be used by the binary
//! adapters and by the list based compatibility bindings.

const PI: f64 = std::f64::consts::PI;
const WGS84_A_KM: f64 = 6378.137;
const WGS84_B_KM: f64 = 6356.752314245179;

fn finite3(value: &[f64; 3]) -> bool {
    value.iter().all(|item| item.is_finite())
}

fn norm3(value: &[f64; 3]) -> f64 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn dot3(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn row3(values: &[f64], index: usize) -> [f64; 3] {
    [
        values[3 * index],
        values[3 * index + 1],
        values[3 * index + 2],
    ]
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn surface_geometry_pair(
    observer: [f64; 3],
    targets: &[f64],
    normals: &[f64],
    boresight: [f64; 3],
) -> ([f64; 2], [f64; 2], [f64; 2], [f64; 2]) {
    use std::arch::aarch64::*;
    // SAFETY: caller provides exactly two contiguous 3-vectors in each
    // validated slice. NEON is enabled by the compile-time target gate.
    // Each store below writes exactly two initialized f64 destination slots.
    let target = vld3q_f64(targets.as_ptr());
    let normal = vld3q_f64(normals.as_ptr());
    let ox = vdupq_n_f64(observer[0]);
    let oy = vdupq_n_f64(observer[1]);
    let oz = vdupq_n_f64(observer[2]);
    let dx = vsubq_f64(target.0, ox);
    let dy = vsubq_f64(target.1, oy);
    let dz = vsubq_f64(target.2, oz);
    // Explicit multiply and left-associated add operations match dot3;
    // no fused multiply-add or reduction reassociation is used.
    let ranges = vsqrtq_f64(vaddq_f64(
        vaddq_f64(vmulq_f64(dx, dx), vmulq_f64(dy, dy)),
        vmulq_f64(dz, dz),
    ));
    let norm_values = vsqrtq_f64(vaddq_f64(
        vaddq_f64(vmulq_f64(normal.0, normal.0), vmulq_f64(normal.1, normal.1)),
        vmulq_f64(normal.2, normal.2),
    ));
    let horizons = vaddq_f64(
        vaddq_f64(
            vmulq_f64(normal.0, vsubq_f64(ox, target.0)),
            vmulq_f64(normal.1, vsubq_f64(oy, target.1)),
        ),
        vmulq_f64(normal.2, vsubq_f64(oz, target.2)),
    );
    let cosines = vdivq_f64(
        vaddq_f64(
            vaddq_f64(
                vmulq_f64(dx, vdupq_n_f64(boresight[0])),
                vmulq_f64(dy, vdupq_n_f64(boresight[1])),
            ),
            vmulq_f64(dz, vdupq_n_f64(boresight[2])),
        ),
        ranges,
    );
    let mut range = [0.0; 2];
    let mut cosine = [0.0; 2];
    let mut horizon = [0.0; 2];
    let mut normal_norm = [0.0; 2];
    vst1q_f64(range.as_mut_ptr(), ranges);
    vst1q_f64(cosine.as_mut_ptr(), cosines);
    vst1q_f64(horizon.as_mut_ptr(), horizons);
    vst1q_f64(normal_norm.as_mut_ptr(), norm_values);
    (range, cosine, horizon, normal_norm)
}

#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceTargetResult {
    pub range_km: Vec<f64>,
    pub cosine_off_axis: Vec<f64>,
    pub horizon_clearance_km: Vec<f64>,
    pub visible: Vec<u8>,
    pub inside_pattern: Vec<u8>,
    pub inside_range: Vec<u8>,
    pub available: Vec<u8>,
}

/// Evaluate the hard-cone observer to WGS84 surface-target geometry.
pub fn surface_targets_ecef(
    observer_ecef_km: [f64; 3],
    target_ecef_km: &[f64],
    target_outward_normal_ecef: &[f64],
    boresight_ecef: [f64; 3],
    half_angle_rad: f64,
    max_range_km: Option<f64>,
    angular_tolerance_rad: f64,
    range_tolerance_km: f64,
) -> Result<SurfaceTargetResult, String> {
    if !finite3(&observer_ecef_km) || !finite3(&boresight_ecef) {
        return Err("observer and boresight vectors must be finite".into());
    }
    if target_ecef_km.len() % 3 != 0 || target_outward_normal_ecef.len() != target_ecef_km.len() {
        return Err("target and normal arrays must have matching (n, 3) shape".into());
    }
    if !target_ecef_km.iter().all(|value| value.is_finite())
        || !target_outward_normal_ecef
            .iter()
            .all(|value| value.is_finite())
    {
        return Err("target and normal arrays must be finite".into());
    }
    let boresight_norm = norm3(&boresight_ecef);
    if (boresight_norm - 1.0).abs() > 1.0e-10 {
        return Err("boresight_ecef must be a unit vector within 1e-10".into());
    }
    if !half_angle_rad.is_finite() || !(0.0 < half_angle_rad && half_angle_rad < 0.5 * PI) {
        return Err("half_angle_rad must be finite and strictly within (0, pi/2)".into());
    }
    if !angular_tolerance_rad.is_finite() || angular_tolerance_rad < 0.0 {
        return Err("angular_tolerance_rad must be finite and nonnegative".into());
    }
    if !range_tolerance_km.is_finite() || range_tolerance_km < 0.0 {
        return Err("range_tolerance_km must be finite and nonnegative".into());
    }
    if half_angle_rad + angular_tolerance_rad >= PI {
        return Err("half_angle_rad plus angular_tolerance_rad must remain below pi".into());
    }
    if let Some(maximum) = max_range_km {
        if !maximum.is_finite() || maximum <= 0.0 {
            return Err("max_range_km must be positive and finite when provided".into());
        }
    }
    let count = target_ecef_km.len() / 3;
    let mut result = SurfaceTargetResult {
        range_km: Vec::with_capacity(count),
        cosine_off_axis: Vec::with_capacity(count),
        horizon_clearance_km: Vec::with_capacity(count),
        visible: Vec::with_capacity(count),
        inside_pattern: Vec::with_capacity(count),
        inside_range: Vec::with_capacity(count),
        available: Vec::with_capacity(count),
    };
    let pattern_cosine = (half_angle_rad + angular_tolerance_rad).cos();
    #[allow(unused_mut)]
    let mut next_index = 0;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    while next_index + 1 < count {
        // SAFETY: next_index+1<count and both shape-validated arrays have
        // count*3 entries, so these slices each contain exactly two rows.
        let (ranges, cosines, horizons, normal_norms) = unsafe {
            surface_geometry_pair(
                observer_ecef_km,
                &target_ecef_km[next_index * 3..(next_index + 2) * 3],
                &target_outward_normal_ecef[next_index * 3..(next_index + 2) * 3],
                boresight_ecef,
            )
        };
        for lane in 0..2 {
            // Retain scalar row ordering and validation error precedence.
            if (normal_norms[lane] - 1.0).abs() > 1.0e-10 {
                return Err("target outward normals must be unit vectors within 1e-10".into());
            }
            let range = ranges[lane];
            if !range.is_finite() || range <= 0.0 {
                return Err("observer/target range must be positive and finite".into());
            }
            let visible = horizons[lane] > range_tolerance_km;
            let inside_pattern = cosines[lane] >= pattern_cosine;
            let inside_range = max_range_km
                .map(|maximum| range <= maximum + range_tolerance_km)
                .unwrap_or(true);
            result.range_km.push(range);
            result.cosine_off_axis.push(cosines[lane]);
            result.horizon_clearance_km.push(horizons[lane]);
            result.visible.push(visible as u8);
            result.inside_pattern.push(inside_pattern as u8);
            result.inside_range.push(inside_range as u8);
            result
                .available
                .push((visible && inside_pattern && inside_range) as u8);
        }
        next_index += 2;
    }
    for index in next_index..count {
        let target = row3(target_ecef_km, index);
        let normal = row3(target_outward_normal_ecef, index);
        if (norm3(&normal) - 1.0).abs() > 1.0e-10 {
            return Err("target outward normals must be unit vectors within 1e-10".into());
        }
        let delta = [
            target[0] - observer_ecef_km[0],
            target[1] - observer_ecef_km[1],
            target[2] - observer_ecef_km[2],
        ];
        let range = norm3(&delta);
        if !range.is_finite() || range <= 0.0 {
            return Err("observer/target range must be positive and finite".into());
        }
        let horizon = dot3(
            &normal,
            &[
                observer_ecef_km[0] - target[0],
                observer_ecef_km[1] - target[1],
                observer_ecef_km[2] - target[2],
            ],
        );
        let cosine = dot3(&delta, &boresight_ecef) / range;
        let visible = horizon > range_tolerance_km;
        let inside_pattern = cosine >= pattern_cosine;
        let inside_range = max_range_km
            .map(|maximum| range <= maximum + range_tolerance_km)
            .unwrap_or(true);
        result.range_km.push(range);
        result.cosine_off_axis.push(cosine);
        result.horizon_clearance_km.push(horizon);
        result.visible.push(visible as u8);
        result.inside_pattern.push(inside_pattern as u8);
        result.inside_range.push(inside_range as u8);
        result
            .available
            .push((visible && inside_pattern && inside_range) as u8);
    }
    Ok(result)
}

/// Rich coverage cell gates. The output code is the first failed gate, or zero.
#[allow(clippy::too_many_arguments)]
pub fn rich_cell_reasons(
    observer: [f64; 3],
    targets: &[f64],
    normals: &[f64],
    rotation: [f64; 9],
    sun: Option<[f64; 3]>,
    pattern_kind: u8,
    x_angle: f64,
    y_angle: f64,
    max_range: Option<f64>,
    max_off_nadir: Option<f64>,
    max_incidence: Option<f64>,
    min_sun: Option<f64>,
    max_sun: Option<f64>,
    angular_tol: f64,
    range_tol: f64,
) -> Result<Vec<u8>, String> {
    if targets.len() % 3 != 0 || targets.len() != normals.len() || pattern_kind > 2 {
        return Err("invalid rich coverage target shape or pattern".into());
    }
    if !finite3(&observer)
        || !rotation.iter().all(|v| v.is_finite())
        || !targets.iter().chain(normals).all(|v| v.is_finite())
        || sun.is_some_and(|v| !finite3(&v))
    {
        return Err("rich coverage inputs must be finite".into());
    }
    if (min_sun.is_some() || max_sun.is_some()) && sun.is_none() {
        return Err("rich coverage illumination requires Sun position".into());
    }
    let mut observer_normal = [
        observer[0] / WGS84_A_KM.powi(2),
        observer[1] / WGS84_A_KM.powi(2),
        observer[2] / WGS84_B_KM.powi(2),
    ];
    let observer_normal_norm = norm3(&observer_normal);
    if observer_normal_norm <= 0.0 {
        return Err("invalid observer normal".into());
    }
    for v in &mut observer_normal {
        *v /= observer_normal_norm;
    }
    let mut reasons = Vec::with_capacity(targets.len() / 3);
    for index in 0..targets.len() / 3 {
        let target = row3(targets, index);
        let normal = row3(normals, index);
        if (norm3(&normal) - 1.0).abs() > 1.0e-10 {
            return Err("target normals must be unit vectors".into());
        }
        let delta = [
            target[0] - observer[0],
            target[1] - observer[1],
            target[2] - observer[2],
        ];
        let range = norm3(&delta);
        if range <= 0.0 || !range.is_finite() {
            return Err("invalid observer-to-target range".into());
        }
        let dir = [delta[0] / range, delta[1] / range, delta[2] / range];
        let sensor = [
            rotation[0] * dir[0] + rotation[1] * dir[1] + rotation[2] * dir[2],
            rotation[3] * dir[0] + rotation[4] * dir[1] + rotation[5] * dir[2],
            rotation[6] * dir[0] + rotation[7] * dir[1] + rotation[8] * dir[2],
        ];
        let horizontal = sensor[0].atan2(sensor[2]);
        let vertical = sensor[1].atan2(sensor[2]);
        let off_axis = sensor[2].clamp(-1.0, 1.0).acos();
        let pattern = if pattern_kind == 0 {
            off_axis <= x_angle + angular_tol
        } else {
            sensor[2] > 0.0
                && horizontal.abs() <= x_angle + angular_tol
                && vertical.abs() <= y_angle + angular_tol
        };
        let horizon = normal[0] * (observer[0] - target[0])
            + normal[1] * (observer[1] - target[1])
            + normal[2] * (observer[2] - target[2]);
        let target_off_nadir = (-(dir[0] * observer_normal[0]
            + dir[1] * observer_normal[1]
            + dir[2] * observer_normal[2]))
            .clamp(-1.0, 1.0)
            .acos();
        let incidence = (-(normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2]))
            .clamp(-1.0, 1.0)
            .acos();
        let illumination = if let Some(sun_pos) = sun {
            let sun_delta = [
                sun_pos[0] - target[0],
                sun_pos[1] - target[1],
                sun_pos[2] - target[2],
            ];
            let sun_range = norm3(&sun_delta);
            if sun_range <= 0.0 || !sun_range.is_finite() {
                return Err("invalid target-to-Sun range".into());
            }
            let elevation =
                ((normal[0] * sun_delta[0] + normal[1] * sun_delta[1] + normal[2] * sun_delta[2])
                    / sun_range)
                    .clamp(-1.0, 1.0)
                    .asin();
            min_sun.is_none_or(|v| elevation >= v - angular_tol)
                && max_sun.is_none_or(|v| elevation <= v + angular_tol)
        } else {
            true
        };
        let code = if horizon <= range_tol {
            1
        } else if !pattern {
            2
        } else if max_range.is_some_and(|v| range > v + range_tol) {
            3
        } else if max_off_nadir.is_some_and(|v| target_off_nadir > v + angular_tol) {
            4
        } else if max_incidence.is_some_and(|v| incidence > v + angular_tol) {
            5
        } else if !illumination {
            6
        } else {
            0
        };
        reasons.push(code);
    }
    Ok(reasons)
}

#[derive(Clone, Debug, PartialEq)]
pub struct RayIntersectionResult {
    pub hit: Vec<u8>,
    pub distance_km: Vec<f64>,
    pub point_ecef_km: Vec<f64>,
}

/// Intersect a batch of unit-direction rays with the WGS84 ellipsoid.
///
/// The implementation follows `sim.analysis.sensor_footprint_geometry`:
/// nearest positive root wins, tangencies within the discriminant tolerance
/// count as hits, and misses return NaN distance/point values.  Directions
/// are normalized here so callers retain the Python helper's accepted input
/// contract rather than having to pre-normalize boundary rays.
pub fn intersect_rays_wgs84(
    observer_ecef_km: [f64; 3],
    direction_ecef: &[f64],
    discriminant_tolerance: f64,
    distance_tolerance_km: f64,
) -> Result<RayIntersectionResult, String> {
    if !finite3(&observer_ecef_km) {
        return Err("observer_ecef_km must contain three finite values".into());
    }
    if direction_ecef.len() % 3 != 0 || !direction_ecef.iter().all(|value| value.is_finite()) {
        return Err("direction_ecef must be finite with shape (items, 3)".into());
    }
    if !discriminant_tolerance.is_finite() || discriminant_tolerance < 0.0 {
        return Err("discriminant_tolerance must be finite and nonnegative".into());
    }
    if !distance_tolerance_km.is_finite() || distance_tolerance_km < 0.0 {
        return Err("distance_tolerance_km must be finite and nonnegative".into());
    }
    let axes = [WGS84_A_KM, WGS84_A_KM, WGS84_B_KM];
    let observer_scaled = [
        observer_ecef_km[0] / axes[0],
        observer_ecef_km[1] / axes[1],
        observer_ecef_km[2] / axes[2],
    ];
    if dot3(&observer_scaled, &observer_scaled) <= 1.0 {
        return Err("Ray observer must be outside the WGS84 ellipsoid".into());
    }
    let count = direction_ecef.len() / 3;
    let mut hit = Vec::with_capacity(count);
    let mut distance_km = Vec::with_capacity(count);
    let mut point_ecef_km = Vec::with_capacity(direction_ecef.len());
    for index in 0..count {
        let direction = row3(direction_ecef, index);
        let direction_norm = norm3(&direction);
        if !direction_norm.is_finite() || direction_norm <= 0.0 {
            return Err("direction_ecef vectors must have positive finite norm".into());
        }
        let unit = [
            direction[0] / direction_norm,
            direction[1] / direction_norm,
            direction[2] / direction_norm,
        ];
        let scaled = [unit[0] / axes[0], unit[1] / axes[1], unit[2] / axes[2]];
        let quadratic = dot3(&scaled, &scaled);
        let linear = dot3(&scaled, &observer_scaled);
        let constant = dot3(&observer_scaled, &observer_scaled) - 1.0;
        let discriminant = linear * linear - quadratic * constant;
        let candidate = discriminant >= -discriminant_tolerance;
        let root = discriminant.max(0.0).sqrt();
        let first = (-linear - root) / quadratic;
        let second = (-linear + root) / quadratic;
        let first_valid = first > distance_tolerance_km;
        let second_valid = second > distance_tolerance_km;
        let distance = if first_valid {
            first
        } else if second_valid {
            second
        } else {
            f64::NAN
        };
        let is_hit = candidate && (first_valid || second_valid);
        hit.push(is_hit as u8);
        distance_km.push(if is_hit { distance } else { f64::NAN });
        if is_hit {
            point_ecef_km.extend_from_slice(&[
                observer_ecef_km[0] + distance * unit[0],
                observer_ecef_km[1] + distance * unit[1],
                observer_ecef_km[2] + distance * unit[2],
            ]);
        } else {
            point_ecef_km.extend_from_slice(&[f64::NAN, f64::NAN, f64::NAN]);
        }
    }
    Ok(RayIntersectionResult {
        hit,
        distance_km,
        point_ecef_km,
    })
}

/// Evaluate range and conical access gates for an arbitrary observer/target
/// batch.  Ground-site Earth rotation and cadence state remain Python-owned.
pub fn access_batch(
    observer_eci_km: &[f64],
    target_eci_km: &[f64],
    boresight_eci: Option<&[f64]>,
    max_range_km: Option<f64>,
    fov_half_angle_rad: Option<f64>,
) -> Result<(Vec<f64>, Vec<u8>, Vec<String>), String> {
    let count = observer_eci_km.len() / 3;
    if observer_eci_km.len() % 3 != 0
        || target_eci_km.len() != observer_eci_km.len()
        || !observer_eci_km.iter().all(|value| value.is_finite())
        || !target_eci_km.iter().all(|value| value.is_finite())
    {
        return Err(
            "observer and target batches must be finite arrays with shape (samples, 3)".into(),
        );
    }
    if let Some(values) = boresight_eci {
        if values.len() != observer_eci_km.len() || !values.iter().all(|value| value.is_finite()) {
            return Err("boresight batch must match observer shape and be finite".into());
        }
    }
    if let Some(maximum) = max_range_km {
        if !maximum.is_finite() || maximum <= 0.0 {
            return Err("max_range_km must be positive and finite when provided".into());
        }
    }
    if let Some(angle) = fov_half_angle_rad {
        if !angle.is_finite() || !(0.0 <= angle && angle <= PI) {
            return Err("fov_half_angle_rad must be finite and within [0, pi]".into());
        }
    }
    let mut ranges = Vec::with_capacity(count);
    let mut available = Vec::with_capacity(count);
    let mut reasons = Vec::with_capacity(count);
    for index in 0..count {
        let observer = row3(observer_eci_km, index);
        let target = row3(target_eci_km, index);
        let delta = [
            target[0] - observer[0],
            target[1] - observer[1],
            target[2] - observer[2],
        ];
        let range = norm3(&delta);
        if !range.is_finite() {
            return Err("observer/target range must be finite".into());
        }
        ranges.push(range);
        if let Some(maximum) = max_range_km {
            if range > maximum {
                available.push(0);
                reasons.push("range".into());
                continue;
            }
        }
        if let Some(angle) = fov_half_angle_rad {
            if range > 0.0 {
                let boresight = if let Some(values) = boresight_eci {
                    let supplied = row3(values, index);
                    let supplied_norm = norm3(&supplied);
                    if supplied_norm <= 0.0 || !supplied_norm.is_finite() {
                        available.push(0);
                        reasons.push("boresight".into());
                        continue;
                    }
                    [
                        supplied[0] / supplied_norm,
                        supplied[1] / supplied_norm,
                        supplied[2] / supplied_norm,
                    ]
                } else {
                    // Keep the Python reference's small-observer behavior as
                    // well as its exact-zero fallback.
                    let denominator = norm3(&observer).max(1.0e-12);
                    [
                        observer[0] / denominator,
                        observer[1] / denominator,
                        observer[2] / denominator,
                    ]
                };
                let cosine = (dot3(&boresight, &delta) / range).clamp(-1.0, 1.0);
                if cosine.acos() > angle {
                    available.push(0);
                    reasons.push("solid_angle".into());
                    continue;
                }
            }
        }
        available.push(1);
        reasons.push("ok".into());
    }
    Ok((ranges, available, reasons))
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoverageMetrics {
    pub dwell_s: Vec<f64>,
    pub interval_count: Vec<i64>,
    pub observed_acquisition_count: Vec<i64>,
    pub max_complete_revisit_gap_s: Vec<f64>,
    pub prefix_boundary_gap_s: Vec<f64>,
    pub suffix_boundary_gap_s: Vec<f64>,
    pub start_censored: Vec<u8>,
    pub end_censored: Vec<u8>,
    pub interval_cell_index: Vec<i64>,
    pub interval_offset: Vec<i64>,
    pub start_sample_index: Vec<i64>,
    pub end_sample_index_exclusive: Vec<i64>,
}

/// Convert a row-major sampled coverage mask to OEL's sparse interval metrics.
pub fn summarize_sampled_mask(
    times_s: &[f64],
    covered_by_sample: &[u8],
    sample_count: usize,
    cell_count: usize,
    cell_indices: Option<&[i64]>,
) -> Result<CoverageMetrics, String> {
    if sample_count < 2 || times_s.len() != sample_count {
        return Err("times_s must contain at least two samples".into());
    }
    if !times_s.iter().all(|value| value.is_finite())
        || times_s.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err("times_s must be finite and strictly increasing".into());
    }
    if covered_by_sample.len() != sample_count.saturating_mul(cell_count)
        || covered_by_sample.iter().any(|value| *value > 1)
    {
        return Err("covered_by_sample must be a row-major boolean mask".into());
    }
    let cells: Vec<i64> = match cell_indices {
        None => (0..cell_count).map(|value| value as i64).collect(),
        Some(values) => {
            if values.len() != cell_count || values.windows(2).any(|pair| pair[1] <= pair[0]) {
                return Err(
                    "cell_indices must be strictly increasing and match the cell dimension".into(),
                );
            }
            values.to_vec()
        }
    };
    let duration = times_s[sample_count - 1] - times_s[0];
    let mut metrics = CoverageMetrics {
        dwell_s: vec![0.0; cell_count],
        interval_count: vec![0; cell_count],
        observed_acquisition_count: vec![0; cell_count],
        max_complete_revisit_gap_s: vec![f64::NAN; cell_count],
        prefix_boundary_gap_s: vec![f64::NAN; cell_count],
        suffix_boundary_gap_s: vec![f64::NAN; cell_count],
        start_censored: vec![0; cell_count],
        end_censored: vec![0; cell_count],
        interval_cell_index: Vec::new(),
        interval_offset: vec![0],
        start_sample_index: Vec::new(),
        end_sample_index_exclusive: Vec::new(),
    };
    for cell in 0..cell_count {
        let initial = covered_by_sample[cell] != 0;
        metrics.start_censored[cell] = initial as u8;
        metrics.end_censored[cell] =
            (covered_by_sample[(sample_count - 1) * cell_count + cell] != 0) as u8;
        let mut active_start: Option<usize> = None;
        let mut previous_end: Option<usize> = None;
        let mut first_start: Option<usize> = None;
        let mut last_end: Option<usize> = None;
        let mut maximum_gap: Option<f64> = None;
        let interval_begin = metrics.start_sample_index.len();
        // Dwell and transition detection share one pass. Each cell's sums
        // retain ascending sample order, including irregular epoch spacing.
        for sample in 0..=sample_count {
            let covered =
                sample < sample_count && covered_by_sample[sample * cell_count + cell] != 0;
            if covered {
                if sample + 1 < sample_count {
                    metrics.dwell_s[cell] += times_s[sample + 1] - times_s[sample];
                }
                if active_start.is_none() {
                    active_start = Some(sample);
                    if first_start.is_none() {
                        first_start = Some(sample);
                    }
                    if sample > 0 {
                        metrics.observed_acquisition_count[cell] += 1;
                    }
                    if let Some(end) = previous_end {
                        let gap = times_s[sample] - times_s[end];
                        maximum_gap = Some(maximum_gap.map_or(gap, |value| value.max(gap)));
                    }
                }
            } else if let Some(start) = active_start.take() {
                metrics.start_sample_index.push(start as i64);
                metrics.end_sample_index_exclusive.push(sample as i64);
                metrics.interval_count[cell] += 1;
                previous_end = Some(sample);
                last_end = Some(sample);
            }
        }
        if metrics.start_sample_index.len() > interval_begin {
            metrics.interval_cell_index.push(cells[cell]);
            metrics
                .interval_offset
                .push(metrics.start_sample_index.len() as i64);
        }
        if let Some(value) = maximum_gap {
            metrics.max_complete_revisit_gap_s[cell] = value;
        }
        if let Some(start) = first_start {
            if start > 0 {
                metrics.prefix_boundary_gap_s[cell] = times_s[start] - times_s[0];
            }
        } else {
            metrics.prefix_boundary_gap_s[cell] = duration;
        }
        if let Some(end) = last_end {
            if end < sample_count {
                metrics.suffix_boundary_gap_s[cell] = times_s[sample_count - 1] - times_s[end];
            }
        } else {
            metrics.suffix_boundary_gap_s[cell] = duration;
        }
    }
    Ok(metrics)
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiplicityResult {
    pub multiplicity: Vec<u16>,
    pub qualified: Vec<u8>,
    pub active_asset_count: Vec<i64>,
    pub maximum_by_sample: Vec<u16>,
    pub mean_per_cell: Vec<f64>,
    pub maximum_per_cell: Vec<u16>,
    pub histogram: Vec<i64>,
}

/// Aggregate member masks for constellation/service multiplicity studies.
pub fn aggregate_multiplicity(
    member_masks: &[u8],
    member_count: usize,
    sample_count: usize,
    cell_count: usize,
    required_multiplicity: usize,
) -> Result<MultiplicityResult, String> {
    if member_count == 0 || required_multiplicity == 0 || required_multiplicity > member_count {
        return Err("required_multiplicity must be within the member count".into());
    }
    let plane = sample_count
        .checked_mul(cell_count)
        .ok_or_else(|| "coverage mask size overflow".to_string())?;
    if member_masks.len() != member_count.saturating_mul(plane)
        || member_masks.iter().any(|value| *value > 1)
    {
        return Err("member_masks must be a row-major boolean mask per member".into());
    }
    let mut multiplicity = vec![0u16; plane];
    let mut active_asset_count = vec![0i64; sample_count];
    for member in 0..member_count {
        let offset = member * plane;
        for sample in 0..sample_count {
            let row = offset + sample * cell_count;
            if member_masks[row..row + cell_count]
                .iter()
                .any(|value| *value != 0)
            {
                active_asset_count[sample] += 1;
            }
            for cell in 0..cell_count {
                multiplicity[sample * cell_count + cell] += member_masks[row + cell] as u16;
            }
        }
    }
    let qualified: Vec<u8> = multiplicity
        .iter()
        .map(|value| (*value as usize >= required_multiplicity) as u8)
        .collect();
    let mut maximum_by_sample = vec![0u16; sample_count];
    for sample in 0..sample_count {
        maximum_by_sample[sample] = *multiplicity[sample * cell_count..(sample + 1) * cell_count]
            .iter()
            .max()
            .unwrap_or(&0);
    }
    let mut mean_per_cell = vec![0.0; cell_count];
    let mut maximum_per_cell = vec![0u16; cell_count];
    for cell in 0..cell_count {
        let mut total = 0.0;
        for sample in 0..sample_count {
            let value = multiplicity[sample * cell_count + cell];
            total += value as f64;
            maximum_per_cell[cell] = maximum_per_cell[cell].max(value);
        }
        mean_per_cell[cell] = total / sample_count as f64;
    }
    let mut histogram = vec![0i64; member_count + 1];
    for value in &multiplicity {
        histogram[*value as usize] += 1;
    }
    Ok(MultiplicityResult {
        multiplicity,
        qualified,
        active_asset_count,
        maximum_by_sample,
        mean_per_cell,
        maximum_per_cell,
        histogram,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledTransition {
    pub transition_kind: String,
    pub time_s: f64,
    pub bracket_start_s: f64,
    pub bracket_end_s: f64,
    pub disposition: String,
    pub iterations: usize,
    pub reason_before: String,
    pub reason_after: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledInterval {
    pub interval_index: usize,
    pub start_s: f64,
    pub end_s: f64,
    pub duration_s: f64,
    pub start_censored: bool,
    pub end_censored: bool,
    pub acquisition_disposition: String,
    pub loss_disposition: String,
    pub acquisition_reason: String,
    pub loss_reason: String,
}

/// Build conservative sample-bounded transitions and intervals.
pub fn sampled_windows(
    times_s: &[f64],
    available: &[u8],
    reasons: &[String],
) -> Result<(Vec<SampledTransition>, Vec<SampledInterval>), String> {
    if times_s.len() < 2
        || available.len() != times_s.len()
        || reasons.len() != times_s.len()
        || !times_s.iter().all(|value| value.is_finite())
        || times_s.windows(2).any(|pair| pair[1] <= pair[0])
        || available.iter().any(|value| *value > 1)
        || reasons.iter().any(|value| value.is_empty())
    {
        return Err("times, availability, and reasons must be finite, ordered, and aligned".into());
    }
    let mut transitions = Vec::new();
    for index in 1..times_s.len() {
        if available[index] != available[index - 1] {
            transitions.push(SampledTransition {
                transition_kind: if available[index] != 0 {
                    "acquisition".into()
                } else {
                    "loss".into()
                },
                time_s: times_s[index],
                bracket_start_s: times_s[index - 1],
                bracket_end_s: times_s[index],
                disposition: "sample_bounded".into(),
                iterations: 0,
                reason_before: reasons[index - 1].clone(),
                reason_after: reasons[index].clone(),
            });
        }
    }
    let mut intervals = Vec::new();
    let mut active_start: Option<f64> = if available[0] != 0 {
        Some(times_s[0])
    } else {
        None
    };
    let start_censored = available[0] != 0;
    let mut acquisition_disposition = if start_censored {
        "study_start_censored".into()
    } else {
        "not_applicable".into()
    };
    let mut acquisition_reason = if start_censored {
        reasons[0].clone()
    } else {
        "not_applicable".into()
    };
    let mut transition_cursor = 0usize;
    for index in 1..times_s.len() {
        if available[index] == available[index - 1] {
            continue;
        }
        let transition = &transitions[transition_cursor];
        transition_cursor += 1;
        if available[index] != 0 {
            active_start = Some(transition.time_s);
            acquisition_disposition = transition.disposition.clone();
            acquisition_reason = transition.reason_before.clone();
            continue;
        }
        let start = active_start
            .ok_or_else(|| "loss transition encountered without active interval".to_string())?;
        intervals.push(SampledInterval {
            interval_index: intervals.len(),
            start_s: start,
            end_s: transition.time_s,
            duration_s: (transition.time_s - start).max(0.0),
            start_censored: if intervals.is_empty() {
                start_censored
            } else {
                false
            },
            end_censored: false,
            acquisition_disposition: acquisition_disposition.clone(),
            loss_disposition: transition.disposition.clone(),
            acquisition_reason: acquisition_reason.clone(),
            loss_reason: transition.reason_after.clone(),
        });
        active_start = None;
    }
    if let Some(start) = active_start {
        intervals.push(SampledInterval {
            interval_index: intervals.len(),
            start_s: start,
            end_s: times_s[times_s.len() - 1],
            duration_s: (times_s[times_s.len() - 1] - start).max(0.0),
            start_censored: if intervals.is_empty() {
                start_censored
            } else {
                false
            },
            end_censored: true,
            acquisition_disposition,
            loss_disposition: "study_end_censored".into(),
            acquisition_reason,
            loss_reason: "available".into(),
        });
    }
    Ok((transitions, intervals))
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

    fn fixed3(values: Vec<f64>, name: &str) -> PyResult<[f64; 3]> {
        values
            .try_into()
            .map_err(|_| PyValueError::new_err(format!("{name} must have three values")))
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

    fn decode_i64_bytes<'a>(values: &'a [u8], name: &str) -> PyResult<std::borrow::Cow<'a, [i64]>> {
        if values.len() % 8 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} bytes must contain complete int64 values"
            )));
        }
        #[cfg(target_endian = "little")]
        if values.as_ptr().align_offset(std::mem::align_of::<i64>()) == 0 {
            // SAFETY: &[u8] extraction accepts immutable PyBytes only (PyO3
            // conversions/std/slice.rs). The caller keeps the object alive
            // through the synchronous native call, including GIL detachment.
            // Alignment/size are checked and every i64 bit pattern is valid.
            let view = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<i64>(), values.len() / 8)
            };
            return Ok(std::borrow::Cow::Borrowed(view));
        }
        Ok(std::borrow::Cow::Owned(
            values
                .chunks_exact(8)
                .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
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

    fn encode_i64_bytes(py: Python<'_>, values: &[i64]) -> PyResult<Py<PyBytes>> {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: initialized i64 slices have no padding, all bytes
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
    #[pyo3(signature = (observer_ecef_km, target_ecef_km, target_outward_normal_ecef, boresight_ecef, half_angle_rad, max_range_km=None, angular_tolerance_rad=1e-12, range_tolerance_km=1e-9))]
    fn coverage_surface_targets_ecef(
        py: Python<'_>,
        observer_ecef_km: Vec<f64>,
        target_ecef_km: Vec<f64>,
        target_outward_normal_ecef: Vec<f64>,
        boresight_ecef: Vec<f64>,
        half_angle_rad: f64,
        max_range_km: Option<f64>,
        angular_tolerance_rad: f64,
        range_tolerance_km: f64,
    ) -> PyResult<(
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<bool>,
        Vec<bool>,
        Vec<bool>,
        Vec<bool>,
    )> {
        let observer = fixed3(observer_ecef_km, "observer_ecef_km")?;
        let boresight = fixed3(boresight_ecef, "boresight_ecef")?;
        let result = py
            .detach(|| {
                surface_targets_ecef(
                    observer,
                    &target_ecef_km,
                    &target_outward_normal_ecef,
                    boresight,
                    half_angle_rad,
                    max_range_km,
                    angular_tolerance_rad,
                    range_tolerance_km,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            result.range_km,
            result.cosine_off_axis,
            result.horizon_clearance_km,
            result.visible.into_iter().map(|value| value != 0).collect(),
            result
                .inside_pattern
                .into_iter()
                .map(|value| value != 0)
                .collect(),
            result
                .inside_range
                .into_iter()
                .map(|value| value != 0)
                .collect(),
            result
                .available
                .into_iter()
                .map(|value| value != 0)
                .collect(),
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (observer_ecef_km, target_ecef_km, target_outward_normal_ecef, boresight_ecef, half_angle_rad, max_range_km=None, angular_tolerance_rad=1e-12, range_tolerance_km=1e-9))]
    fn coverage_surface_targets_ecef_bytes(
        py: Python<'_>,
        observer_ecef_km: &[u8],
        target_ecef_km: &[u8],
        target_outward_normal_ecef: &[u8],
        boresight_ecef: &[u8],
        half_angle_rad: f64,
        max_range_km: Option<f64>,
        angular_tolerance_rad: f64,
        range_tolerance_km: f64,
    ) -> PyResult<(
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
    )> {
        let observer = fixed3(
            decode_f64_bytes(observer_ecef_km, "observer_ecef_km")?.into_owned(),
            "observer_ecef_km",
        )?;
        let target_ecef_km = decode_f64_bytes(target_ecef_km, "target_ecef_km")?;
        let target_outward_normal_ecef =
            decode_f64_bytes(target_outward_normal_ecef, "target_outward_normal_ecef")?;
        let boresight = fixed3(
            decode_f64_bytes(boresight_ecef, "boresight_ecef")?.into_owned(),
            "boresight_ecef",
        )?;
        let result = py
            .detach(|| {
                surface_targets_ecef(
                    observer,
                    &target_ecef_km,
                    &target_outward_normal_ecef,
                    boresight,
                    half_angle_rad,
                    max_range_km,
                    angular_tolerance_rad,
                    range_tolerance_km,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            encode_f64_bytes(py, &result.range_km)?,
            encode_f64_bytes(py, &result.cosine_off_axis)?,
            encode_f64_bytes(py, &result.horizon_clearance_km)?,
            PyBytes::new(py, &result.visible).unbind(),
            PyBytes::new(py, &result.inside_pattern).unbind(),
            PyBytes::new(py, &result.inside_range).unbind(),
            PyBytes::new(py, &result.available).unbind(),
        ))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (observers, targets, normals, boresights, half_angle_rad, max_range_km=None, angular_tolerance_rad=1e-12, range_tolerance_km=1e-9))]
    fn coverage_surface_targets_tile_bytes(
        py: Python<'_>, observers: &[u8], targets: &[u8], normals: &[u8], boresights: &[u8],
        half_angle_rad: f64, max_range_km: Option<f64>, angular_tolerance_rad: f64,
        range_tolerance_km: f64,
    ) -> PyResult<Py<PyBytes>> {
        let observers = decode_f64_bytes(observers, "observers")?;
        let targets = decode_f64_bytes(targets, "targets")?;
        let normals = decode_f64_bytes(normals, "normals")?;
        let boresights = decode_f64_bytes(boresights, "boresights")?;
        if observers.len() % 3 != 0 || observers.len() != boresights.len()
            || targets.len() % 3 != 0 || targets.len() != normals.len() {
            return Err(PyValueError::new_err("coverage tile arrays have inconsistent shapes"));
        }
        let result = py.detach(|| {
            let mut result = Vec::with_capacity(observers.len() / 3 * targets.len() / 3);
            for sample in 0..observers.len() / 3 {
                result.extend(surface_targets_ecef(
                    row3(&observers, sample), &targets, &normals, row3(&boresights, sample),
                    half_angle_rad, max_range_km, angular_tolerance_rad, range_tolerance_km,
                )?.available);
            }
            Ok::<_, String>(result)
        }).map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &result).unbind())
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn coverage_rich_cell_reasons_tile_bytes(
        py: Python<'_>, observers: &[u8], targets: &[u8], normals: &[u8], rotations: &[u8],
        suns: Option<&[u8]>, pattern_kind: u8, x_angle: f64, y_angle: f64,
        max_range: Option<f64>, max_off_nadir: Option<f64>, max_incidence: Option<f64>,
        min_sun: Option<f64>, max_sun: Option<f64>, angular_tol: f64, range_tol: f64,
    ) -> PyResult<Py<PyBytes>> {
        let observers = decode_f64_bytes(observers, "observers")?;
        let targets = decode_f64_bytes(targets, "targets")?;
        let normals = decode_f64_bytes(normals, "normals")?;
        let rotations = decode_f64_bytes(rotations, "rotations")?;
        let suns = suns.map(|value| decode_f64_bytes(value, "suns")).transpose()?;
        let samples = observers.len() / 3;
        if observers.len() % 3 != 0 || rotations.len() != samples * 9
            || suns.as_ref().is_some_and(|value| value.len() != observers.len())
            || targets.len() % 3 != 0 || targets.len() != normals.len() {
            return Err(PyValueError::new_err("rich coverage tile arrays have inconsistent shapes"));
        }
        let result = py.detach(|| {
            let mut result = Vec::with_capacity(samples * targets.len() / 3);
            for sample in 0..samples {
                let rotation: [f64; 9] = rotations[sample * 9..sample * 9 + 9].try_into().unwrap();
                let sun = suns.as_ref().map(|values| row3(values, sample));
                result.extend(rich_cell_reasons(
                    row3(&observers, sample), &targets, &normals, rotation, sun, pattern_kind,
                    x_angle, y_angle, max_range, max_off_nadir, max_incidence, min_sun, max_sun,
                    angular_tol, range_tol,
                )?);
            }
            Ok::<_, String>(result)
        }).map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &result).unbind())
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn coverage_rich_cell_reasons_bytes(
        py: Python<'_>,
        observer: &[u8],
        targets: &[u8],
        normals: &[u8],
        rotation: &[u8],
        sun: Option<&[u8]>,
        pattern_kind: u8,
        x_angle: f64,
        y_angle: f64,
        max_range: Option<f64>,
        max_off_nadir: Option<f64>,
        max_incidence: Option<f64>,
        min_sun: Option<f64>,
        max_sun: Option<f64>,
        angular_tol: f64,
        range_tol: f64,
    ) -> PyResult<Py<PyBytes>> {
        let observer = fixed3(
            decode_f64_bytes(observer, "observer")?.into_owned(),
            "observer",
        )?;
        let targets = decode_f64_bytes(targets, "targets")?;
        let normals = decode_f64_bytes(normals, "normals")?;
        let rotation: [f64; 9] = decode_f64_bytes(rotation, "rotation")?
            .as_ref()
            .try_into()
            .map_err(|_| PyValueError::new_err("rotation must have nine values"))?;
        let sun = sun
            .map(|data| fixed3(decode_f64_bytes(data, "sun")?.into_owned(), "sun"))
            .transpose()?;
        let result = py
            .detach(|| {
                rich_cell_reasons(
                    observer,
                    &targets,
                    &normals,
                    rotation,
                    sun,
                    pattern_kind,
                    x_angle,
                    y_angle,
                    max_range,
                    max_off_nadir,
                    max_incidence,
                    min_sun,
                    max_sun,
                    angular_tol,
                    range_tol,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &result).unbind())
    }

    #[pyfunction]
    #[pyo3(signature = (observer_ecef_km, direction_ecef, discriminant_tolerance=1e-12, distance_tolerance_km=1e-9))]
    fn coverage_intersect_rays_wgs84(
        py: Python<'_>,
        observer_ecef_km: Vec<f64>,
        direction_ecef: Vec<f64>,
        discriminant_tolerance: f64,
        distance_tolerance_km: f64,
    ) -> PyResult<(Vec<bool>, Vec<f64>, Vec<f64>)> {
        let observer = fixed3(observer_ecef_km, "observer_ecef_km")?;
        let result = py
            .detach(|| {
                intersect_rays_wgs84(
                    observer,
                    &direction_ecef,
                    discriminant_tolerance,
                    distance_tolerance_km,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            result.hit.into_iter().map(|value| value != 0).collect(),
            result.distance_km,
            result.point_ecef_km,
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (observer_ecef_km, direction_ecef, discriminant_tolerance=1e-12, distance_tolerance_km=1e-9))]
    fn coverage_intersect_rays_wgs84_bytes(
        py: Python<'_>,
        observer_ecef_km: &[u8],
        direction_ecef: &[u8],
        discriminant_tolerance: f64,
        distance_tolerance_km: f64,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Py<PyBytes>)> {
        let observer = fixed3(
            decode_f64_bytes(observer_ecef_km, "observer_ecef_km")?.into_owned(),
            "observer_ecef_km",
        )?;
        let direction_ecef = decode_f64_bytes(direction_ecef, "direction_ecef")?;
        let result = py
            .detach(|| {
                intersect_rays_wgs84(
                    observer,
                    &direction_ecef,
                    discriminant_tolerance,
                    distance_tolerance_km,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            PyBytes::new(py, &result.hit).unbind(),
            encode_f64_bytes(py, &result.distance_km)?,
            encode_f64_bytes(py, &result.point_ecef_km)?,
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (observer_eci_km, target_eci_km, boresight_eci=None, max_range_km=None, fov_half_angle_rad=None))]
    fn coverage_access_batch(
        py: Python<'_>,
        observer_eci_km: Vec<f64>,
        target_eci_km: Vec<f64>,
        boresight_eci: Option<Vec<f64>>,
        max_range_km: Option<f64>,
        fov_half_angle_rad: Option<f64>,
    ) -> PyResult<(Vec<f64>, Vec<bool>, Vec<String>)> {
        let result = py
            .detach(|| {
                access_batch(
                    &observer_eci_km,
                    &target_eci_km,
                    boresight_eci.as_deref(),
                    max_range_km,
                    fov_half_angle_rad,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            result.0,
            result.1.into_iter().map(|value| value != 0).collect(),
            result.2,
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (times_s, covered_by_sample, sample_count, cell_count, cell_indices=None))]
    fn coverage_summarize_mask(
        py: Python<'_>,
        times_s: Vec<f64>,
        covered_by_sample: Vec<u8>,
        sample_count: usize,
        cell_count: usize,
        cell_indices: Option<Vec<i64>>,
    ) -> PyResult<(
        Vec<f64>,
        Vec<i64>,
        Vec<i64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<bool>,
        Vec<bool>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
    )> {
        let result = py
            .detach(|| {
                summarize_sampled_mask(
                    &times_s,
                    &covered_by_sample,
                    sample_count,
                    cell_count,
                    cell_indices.as_deref(),
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            result.dwell_s,
            result.interval_count,
            result.observed_acquisition_count,
            result.max_complete_revisit_gap_s,
            result.prefix_boundary_gap_s,
            result.suffix_boundary_gap_s,
            result
                .start_censored
                .into_iter()
                .map(|value| value != 0)
                .collect(),
            result
                .end_censored
                .into_iter()
                .map(|value| value != 0)
                .collect(),
            result.interval_cell_index,
            result.interval_offset,
            result.start_sample_index,
            result.end_sample_index_exclusive,
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (times_s, covered_by_sample, sample_count, cell_count, cell_indices=None))]
    fn coverage_summarize_mask_bytes(
        py: Python<'_>,
        times_s: &[u8],
        covered_by_sample: &[u8],
        sample_count: usize,
        cell_count: usize,
        cell_indices: Option<&[u8]>,
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
        Py<PyBytes>,
    )> {
        let times_s = decode_f64_bytes(times_s, "times_s")?;
        let cell_indices = cell_indices
            .map(|values| decode_i64_bytes(values, "cell_indices"))
            .transpose()?;
        let result = py
            .detach(|| {
                summarize_sampled_mask(
                    &times_s,
                    &covered_by_sample,
                    sample_count,
                    cell_count,
                    cell_indices.as_deref(),
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            encode_f64_bytes(py, &result.dwell_s)?,
            encode_i64_bytes(py, &result.interval_count)?,
            encode_i64_bytes(py, &result.observed_acquisition_count)?,
            encode_f64_bytes(py, &result.max_complete_revisit_gap_s)?,
            encode_f64_bytes(py, &result.prefix_boundary_gap_s)?,
            encode_f64_bytes(py, &result.suffix_boundary_gap_s)?,
            PyBytes::new(py, &result.start_censored).unbind(),
            PyBytes::new(py, &result.end_censored).unbind(),
            encode_i64_bytes(py, &result.interval_cell_index)?,
            encode_i64_bytes(py, &result.interval_offset)?,
            encode_i64_bytes(py, &result.start_sample_index)?,
            encode_i64_bytes(py, &result.end_sample_index_exclusive)?,
        ))
    }

    fn encode_u16_bytes(py: Python<'_>, values: &[u16]) -> PyResult<Py<PyBytes>> {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: initialized u16 has no padding and bytes are copied
            // synchronously. Little-endian representation matches the ABI.
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
            Ok(PyBytes::new_with(py, values.len() * 2, |output| {
                for (bytes, value) in output.chunks_exact_mut(2).zip(values) {
                    bytes.copy_from_slice(&value.to_le_bytes());
                }
                Ok(())
            })?
            .unbind())
        }
    }

    #[pyfunction]
    fn coverage_aggregate_multiplicity_bytes(
        py: Python<'_>,
        member_masks: &[u8],
        member_count: usize,
        sample_count: usize,
        cell_count: usize,
        required_multiplicity: usize,
    ) -> PyResult<(
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
        Py<PyBytes>,
    )> {
        let result = py
            .detach(|| {
                aggregate_multiplicity(
                    member_masks,
                    member_count,
                    sample_count,
                    cell_count,
                    required_multiplicity,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            encode_u16_bytes(py, &result.multiplicity)?,
            PyBytes::new(py, &result.qualified).unbind(),
            encode_i64_bytes(py, &result.active_asset_count)?,
            encode_u16_bytes(py, &result.maximum_by_sample)?,
            encode_f64_bytes(py, &result.mean_per_cell)?,
            encode_u16_bytes(py, &result.maximum_per_cell)?,
            encode_i64_bytes(py, &result.histogram)?,
        ))
    }

    #[pyfunction]
    fn coverage_aggregate_multiplicity(
        py: Python<'_>,
        member_masks: Vec<u8>,
        member_count: usize,
        sample_count: usize,
        cell_count: usize,
        required_multiplicity: usize,
    ) -> PyResult<(
        Vec<u16>,
        Vec<bool>,
        Vec<i64>,
        Vec<u16>,
        Vec<f64>,
        Vec<u16>,
        Vec<i64>,
    )> {
        let result = py
            .detach(|| {
                aggregate_multiplicity(
                    &member_masks,
                    member_count,
                    sample_count,
                    cell_count,
                    required_multiplicity,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            result.multiplicity,
            result
                .qualified
                .into_iter()
                .map(|value| value != 0)
                .collect(),
            result.active_asset_count,
            result.maximum_by_sample,
            result.mean_per_cell,
            result.maximum_per_cell,
            result.histogram,
        ))
    }

    #[pyfunction]
    fn coverage_sampled_windows(
        py: Python<'_>,
        times_s: Vec<f64>,
        available: Vec<u8>,
        reasons: Vec<String>,
    ) -> PyResult<(
        Vec<(String, f64, f64, f64, String, usize, String, String)>,
        Vec<(
            usize,
            f64,
            f64,
            f64,
            bool,
            bool,
            String,
            String,
            String,
            String,
        )>,
    )> {
        let (transitions, intervals) = py
            .detach(|| sampled_windows(&times_s, &available, &reasons))
            .map_err(PyValueError::new_err)?;
        Ok((
            transitions
                .into_iter()
                .map(|value| {
                    (
                        value.transition_kind,
                        value.time_s,
                        value.bracket_start_s,
                        value.bracket_end_s,
                        value.disposition,
                        value.iterations,
                        value.reason_before,
                        value.reason_after,
                    )
                })
                .collect(),
            intervals
                .into_iter()
                .map(|value| {
                    (
                        value.interval_index,
                        value.start_s,
                        value.end_s,
                        value.duration_s,
                        value.start_censored,
                        value.end_censored,
                        value.acquisition_disposition,
                        value.loss_disposition,
                        value.acquisition_reason,
                        value.loss_reason,
                    )
                })
                .collect(),
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (times_s, available, reasons, evaluator_at_time=None, time_tolerance_s=None, max_iterations=None))]
    fn coverage_refine_transitions(
        py: Python<'_>,
        times_s: Vec<f64>,
        available: Vec<u8>,
        reasons: Vec<String>,
        evaluator_at_time: Option<Py<PyAny>>,
        time_tolerance_s: Option<f64>,
        max_iterations: Option<usize>,
    ) -> PyResult<Vec<(String, f64, f64, f64, String, usize, String, String)>> {
        if times_s.len() < 2
            || available.len() != times_s.len()
            || reasons.len() != times_s.len()
            || !times_s.iter().all(|value| value.is_finite())
            || times_s.windows(2).any(|pair| pair[1] <= pair[0])
            || available.iter().any(|value| *value > 1)
            || reasons.iter().any(|value| value.trim().is_empty())
        {
            return Err(PyValueError::new_err(
                "times, availability, and reasons must be finite, ordered, and aligned",
            ));
        }
        if evaluator_at_time.is_none() {
            if time_tolerance_s.is_some() || max_iterations.is_some() {
                return Err(PyValueError::new_err(
                    "Refinement tolerances require evaluator_at_time.",
                ));
            }
        } else if time_tolerance_s.is_none()
            || !time_tolerance_s.unwrap().is_finite()
            || time_tolerance_s.unwrap() <= 0.0
            || max_iterations.is_none()
            || max_iterations.unwrap() == 0
        {
            return Err(PyValueError::new_err(
                "time_tolerance_s and max_iterations must be positive for refinement.",
            ));
        }
        let mut transitions = Vec::new();
        for index in 0..times_s.len() - 1 {
            if available[index] == available[index + 1] {
                continue;
            }
            let kind = if available[index + 1] != 0 {
                "acquisition"
            } else {
                "loss"
            };
            if evaluator_at_time.is_none() {
                transitions.push((
                    kind.to_string(),
                    times_s[index + 1],
                    times_s[index],
                    times_s[index + 1],
                    "sample_bounded".to_string(),
                    0,
                    reasons[index].clone(),
                    reasons[index + 1].clone(),
                ));
                continue;
            }
            let original_left_time = times_s[index];
            let original_right_time = times_s[index + 1];
            let original_left_reason = reasons[index].clone();
            let original_right_reason = reasons[index + 1].clone();
            let left_available = available[index] != 0;
            let right_available = available[index + 1] != 0;
            let mut left_time = original_left_time;
            let mut right_time = original_right_time;
            let mut left_reason = original_left_reason.clone();
            let mut right_reason = original_right_reason.clone();
            let mut iterations = 0usize;
            let mut provider_unavailable = false;
            let tolerance = time_tolerance_s.unwrap();
            let limit = max_iterations.unwrap();
            while right_time - left_time > tolerance && iterations < limit {
                let midpoint = 0.5 * (left_time + right_time);
                let callback_result = evaluator_at_time
                    .as_ref()
                    .expect("evaluator is present")
                    .call1(py, (midpoint,))
                    .and_then(|value| value.extract::<(bool, String)>(py));
                let (midpoint_available, midpoint_reason) = match callback_result {
                    Ok((value, reason)) if !reason.trim().is_empty() => (value, reason),
                    _ => {
                        left_time = original_left_time;
                        right_time = original_right_time;
                        left_reason = original_left_reason.clone();
                        right_reason = original_right_reason.clone();
                        iterations = 0;
                        provider_unavailable = true;
                        break;
                    }
                };
                if midpoint_available == left_available {
                    left_time = midpoint;
                    left_reason = midpoint_reason;
                } else if midpoint_available == right_available {
                    right_time = midpoint;
                    right_reason = midpoint_reason;
                } else {
                    left_time = original_left_time;
                    right_time = original_right_time;
                    left_reason = original_left_reason.clone();
                    right_reason = original_right_reason.clone();
                    iterations = 0;
                    provider_unavailable = true;
                    break;
                }
                iterations += 1;
            }
            let converged = !provider_unavailable && right_time - left_time <= tolerance;
            transitions.push((
                kind.to_string(),
                0.5 * (left_time + right_time),
                left_time,
                right_time,
                if converged {
                    "provider_refined".to_string()
                } else if provider_unavailable {
                    "sample_bounded".to_string()
                } else {
                    "iteration_limited".to_string()
                },
                iterations,
                left_reason,
                right_reason,
            ));
        }
        Ok(transitions)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(coverage_surface_targets_ecef, module)?)?;
        module.add_function(wrap_pyfunction!(coverage_surface_targets_tile_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(coverage_rich_cell_reasons_tile_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(coverage_rich_cell_reasons_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(
            coverage_surface_targets_ecef_bytes,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(coverage_intersect_rays_wgs84, module)?)?;
        module.add_function(wrap_pyfunction!(
            coverage_intersect_rays_wgs84_bytes,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(coverage_access_batch, module)?)?;
        module.add_function(wrap_pyfunction!(coverage_summarize_mask, module)?)?;
        module.add_function(wrap_pyfunction!(coverage_summarize_mask_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(coverage_aggregate_multiplicity, module)?)?;
        module.add_function(wrap_pyfunction!(
            coverage_aggregate_multiplicity_bytes,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(coverage_sampled_windows, module)?)?;
        module.add_function(wrap_pyfunction!(coverage_refine_transitions, module)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    const WGS84_A_KM: f64 = 6378.137;

    #[test]
    fn access_default_boresight_matches_python_norm_floor() {
        let observers = [5.0e-13, 0.0, 0.0, 1.0e-12, 0.0, 0.0, 2.0e-12, 0.0, 0.0];
        let targets = [1.0e-12, 0.0, 0.0, 2.0e-12, 0.0, 0.0, 3.0e-12, 0.0, 0.0];
        let (_, available, reasons) = access_batch(
            &observers,
            &targets,
            None,
            None,
            Some(30.0_f64.to_radians()),
        )
        .unwrap();

        assert_eq!(available, vec![0, 1, 1]);
        assert_eq!(reasons, vec!["solid_angle", "ok", "ok"]);
    }

    #[test]
    fn surface_geometry_matches_tangent_and_range_contracts() {
        let visible = surface_targets_ecef(
            [WGS84_A_KM + 500.0, 0.0, 0.0],
            &[WGS84_A_KM, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            0.2,
            None,
            1.0e-12,
            1.0e-9,
        )
        .unwrap();
        assert_eq!(visible.available, vec![1]);
        let tangent = surface_targets_ecef(
            [WGS84_A_KM, 500.0, 0.0],
            &[WGS84_A_KM, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            0.2,
            None,
            1.0e-12,
            1.0e-9,
        )
        .unwrap();
        assert_eq!(tangent.visible, vec![0]);
        let limited = surface_targets_ecef(
            [WGS84_A_KM + 500.0, 0.0, 0.0],
            &[WGS84_A_KM, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            0.2,
            Some(100.0),
            1.0e-12,
            1.0e-9,
        )
        .unwrap();
        assert_eq!(limited.inside_range, vec![0]);
    }

    #[test]
    fn ray_intersection_keeps_nearest_hit_and_tangent_boundaries() {
        let observer = [WGS84_A_KM + 500.0, 0.0, 0.0];
        let tangent = [
            -((observer[0] * observer[0] - WGS84_A_KM * WGS84_A_KM).sqrt() / observer[0]),
            WGS84_A_KM / observer[0],
            0.0,
        ];
        let result = intersect_rays_wgs84(
            observer,
            &[
                -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, tangent[0], tangent[1], tangent[2],
            ],
            1.0e-12,
            1.0e-9,
        )
        .unwrap();
        assert_eq!(result.hit, vec![1, 0, 1]);
        assert!(result.distance_km[0].is_finite());
        assert!(result.distance_km[1].is_nan());
        assert!(result.distance_km[2].is_finite());
    }

    #[test]
    fn sampled_metrics_keep_sparse_order_and_censoring() {
        let times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0];
        let result = summarize_sampled_mask(&times, &mask, 6, 3, Some(&[10, 20, 30])).unwrap();
        assert_eq!(result.interval_cell_index, vec![10, 20]);
        assert_eq!(result.interval_offset, vec![0, 2, 4]);
        assert_eq!(result.start_sample_index, vec![1, 4, 0, 4]);
        assert_eq!(result.end_sample_index_exclusive, vec![3, 5, 2, 6]);
        assert_eq!(result.dwell_s, vec![3.0, 3.0, 0.0]);
        assert_eq!(result.interval_count, vec![2, 2, 0]);
        assert_eq!(result.observed_acquisition_count, vec![2, 1, 0]);
        assert_eq!(result.start_censored, vec![0, 1, 0]);
        assert_eq!(result.end_censored, vec![0, 1, 0]);
    }
}
