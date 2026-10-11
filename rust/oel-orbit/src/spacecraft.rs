//! Deterministic spacecraft surface geometry kernels.
//!
//! The public Python model owns mesh loading, units, profile metadata and
//! qualification.  This module only evaluates numerical facet and raster
//! loops.  Distances are metres and directions are body-frame unit vectors.

fn norm3(value: [f64; 3]) -> f64 {
    value[0]
        .mul_add(value[0], value[1].mul_add(value[1], value[2] * value[2]))
        .sqrt()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn normalized(value: [f64; 3], name: &str) -> Result<[f64; 3], String> {
    if !value.iter().all(|item| item.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    let length = norm3(value);
    if !length.is_finite() || length <= 0.0 {
        return Err(format!("{name} must be nonzero"));
    }
    Ok([value[0] / length, value[1] / length, value[2] / length])
}

fn triangle_geometry(
    triangles: &[f64],
    normals: Option<&[f64]>,
    index: usize,
) -> Result<Option<([f64; 3], [f64; 3], f64)>, String> {
    let base = index * 9;
    let a = [triangles[base], triangles[base + 1], triangles[base + 2]];
    let b = [
        triangles[base + 3],
        triangles[base + 4],
        triangles[base + 5],
    ];
    let c = [
        triangles[base + 6],
        triangles[base + 7],
        triangles[base + 8],
    ];
    if !a
        .iter()
        .chain(b.iter())
        .chain(c.iter())
        .all(|item| item.is_finite())
    {
        return Err("mesh triangles must contain only finite values".to_owned());
    }
    let area_vector = {
        let edge_a = sub(b, a);
        let edge_b = sub(c, a);
        let cross_product = cross(edge_a, edge_b);
        [
            0.5 * cross_product[0],
            0.5 * cross_product[1],
            0.5 * cross_product[2],
        ]
    };
    let area = norm3(area_vector);
    // Python's reference filters degenerate triangles before it evaluates a
    // profile.  Silently skip those rows while retaining input order.
    if !area.is_finite() {
        return Err("mesh triangle area must be finite".to_owned());
    }
    if area <= 0.0 {
        return Ok(None);
    }
    let normal = if let Some(raw_normals) = normals {
        let raw = [
            raw_normals[index * 3],
            raw_normals[index * 3 + 1],
            raw_normals[index * 3 + 2],
        ];
        if !raw.iter().all(|item| item.is_finite()) {
            return Err("mesh normals must contain only finite values".to_owned());
        }
        let length = norm3(raw);
        if length > 0.0 {
            [raw[0] / length, raw[1] / length, raw[2] / length]
        } else {
            [
                area_vector[0] / area,
                area_vector[1] / area,
                area_vector[2] / area,
            ]
        }
    } else {
        [
            area_vector[0] / area,
            area_vector[1] / area,
            area_vector[2] / area,
        ]
    };
    Ok(Some((
        normal,
        [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ],
        area,
    )))
}

/// Evaluate illuminated facet projected area and projected-area centroid.
///
/// The output contains four values per direction: area, cp_x, cp_y, cp_z.
/// Incoming flux follows the Python convention `max(0, -normal dot dir)`.
pub fn facet_projected_geometry(
    triangles: &[f64],
    normals: Option<&[f64]>,
    directions: &[f64],
) -> Result<Vec<f64>, String> {
    if triangles.is_empty() || triangles.len() % 9 != 0 {
        return Err("triangles must contain one or more Nx3x3 rows".to_owned());
    }
    if directions.is_empty() || directions.len() % 3 != 0 {
        return Err("directions must contain one or more 3-vectors".to_owned());
    }
    if let Some(values) = normals {
        if values.len() != triangles.len() / 3 {
            return Err("normals must contain one 3-vector per triangle".to_owned());
        }
    }
    let triangle_count = triangles.len() / 9;
    let direction_count = directions.len() / 3;
    let mut geometry = Vec::with_capacity(triangle_count);
    for index in 0..triangle_count {
        if let Some(row) = triangle_geometry(triangles, normals, index)? {
            geometry.push(row);
        }
    }
    if geometry.is_empty() {
        return Err("mesh does not contain any nondegenerate triangles".to_owned());
    }
    let mut result = Vec::with_capacity(direction_count * 4);
    for index in 0..direction_count {
        let direction = normalized(
            [
                directions[index * 3],
                directions[index * 3 + 1],
                directions[index * 3 + 2],
            ],
            "profile directions",
        )?;
        let mut area_sum = 0.0;
        let mut weighted_cp = [0.0; 3];
        for (normal, centroid, area) in &geometry {
            let illumination =
                (-(normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2]))
                    .max(0.0);
            let weight = *area * illumination;
            area_sum += weight;
            weighted_cp[0] += centroid[0] * weight;
            weighted_cp[1] += centroid[1] * weight;
            weighted_cp[2] += centroid[2] * weight;
        }
        if area_sum > 0.0 {
            weighted_cp[0] /= area_sum;
            weighted_cp[1] /= area_sum;
            weighted_cp[2] /= area_sum;
        } else {
            weighted_cp = [0.0; 3];
        }
        result.extend_from_slice(&[area_sum, weighted_cp[0], weighted_cp[1], weighted_cp[2]]);
    }
    Ok(result)
}

fn silhouette_one(
    triangles: &[f64],
    direction: [f64; 3],
    resolution: usize,
) -> Result<[f64; 4], String> {
    if !(16..=1024).contains(&resolution) {
        return Err("raster resolution must be 16..1024".to_owned());
    }
    let u = normalized(direction, "direction")?;
    let mut helper_index = 0usize;
    let mut helper_abs = u[0].abs();
    for axis in 1..3 {
        if u[axis].abs() < helper_abs {
            helper_index = axis;
            helper_abs = u[axis].abs();
        }
    }
    let helper = {
        let mut value = [0.0; 3];
        value[helper_index] = 1.0;
        value
    };
    let mut x = cross(u, helper);
    let x_norm = norm3(x);
    if x_norm <= 0.0 || !x_norm.is_finite() {
        return Err("direction basis is degenerate".to_owned());
    }
    x = [x[0] / x_norm, x[1] / x_norm, x[2] / x_norm];
    let y = cross(u, x);

    let triangle_count = triangles.len() / 9;
    let mut projected = Vec::with_capacity(triangle_count * 6);
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for index in 0..triangle_count {
        let base = index * 9;
        let mut row = [0.0; 6];
        for vertex in 0..3 {
            let point = [
                triangles[base + vertex * 3],
                triangles[base + vertex * 3 + 1],
                triangles[base + vertex * 3 + 2],
            ];
            if !point.iter().all(|item| item.is_finite()) {
                return Err("mesh triangles must contain only finite values".to_owned());
            }
            row[vertex * 2] = point[0] * x[0] + point[1] * x[1] + point[2] * x[2];
            row[vertex * 2 + 1] = point[0] * y[0] + point[1] * y[1] + point[2] * y[2];
            lo[0] = lo[0].min(row[vertex * 2]);
            lo[1] = lo[1].min(row[vertex * 2 + 1]);
            hi[0] = hi[0].max(row[vertex * 2]);
            hi[1] = hi[1].max(row[vertex * 2 + 1]);
        }
        projected.extend_from_slice(&row);
    }
    let width = [hi[0] - lo[0], hi[1] - lo[1]];
    if !width.iter().all(|value| value.is_finite()) {
        return Err("projected mesh coordinates must be finite".to_owned());
    }
    if width[0] <= 1e-14 || width[1] <= 1e-14 {
        return Ok([0.0; 4]);
    }
    let step = [width[0] / resolution as f64, width[1] / resolution as f64];
    let mut occupied = vec![false; resolution * resolution];
    for row in projected.chunks_exact(6) {
        let mut tri_lo = [f64::INFINITY; 2];
        let mut tri_hi = [f64::NEG_INFINITY; 2];
        for vertex in 0..3 {
            tri_lo[0] = tri_lo[0].min(row[vertex * 2]);
            tri_lo[1] = tri_lo[1].min(row[vertex * 2 + 1]);
            tri_hi[0] = tri_hi[0].max(row[vertex * 2]);
            tri_hi[1] = tri_hi[1].max(row[vertex * 2 + 1]);
        }
        let start_x = ((tri_lo[0] - lo[0]) / step[0]).floor().max(0.0) as usize;
        let start_y = ((tri_lo[1] - lo[1]) / step[1]).floor().max(0.0) as usize;
        let end_x = ((tri_hi[0] - lo[0]) / step[0])
            .ceil()
            .min(resolution as f64) as usize;
        let end_y = ((tri_hi[1] - lo[1]) / step[1])
            .ceil()
            .min(resolution as f64) as usize;
        if end_x <= start_x || end_y <= start_y {
            continue;
        }
        let a = [row[0], row[1]];
        let b = [row[2], row[3]];
        let c = [row[4], row[5]];
        let denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
        if denominator.abs() <= 1e-20 {
            continue;
        }
        for i in start_x..end_x {
            let xx = lo[0] + (i as f64 + 0.5) * step[0];
            for j in start_y..end_y {
                let yy = lo[1] + (j as f64 + 0.5) * step[1];
                let aa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator;
                let bb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator;
                if aa >= -1e-12 && bb >= -1e-12 && aa + bb <= 1.0 + 1e-12 {
                    occupied[i * resolution + j] = true;
                }
            }
        }
    }
    let mut count = 0usize;
    let mut mean = [0.0; 2];
    for i in 0..resolution {
        for j in 0..resolution {
            if occupied[i * resolution + j] {
                count += 1;
                mean[0] += i as f64;
                mean[1] += j as f64;
            }
        }
    }
    if count == 0 {
        return Ok([0.0; 4]);
    }
    mean[0] = lo[0] + (mean[0] / count as f64 + 0.5) * step[0];
    mean[1] = lo[1] + (mean[1] / count as f64 + 0.5) * step[1];
    let cp = [
        mean[0] * x[0] + mean[1] * y[0],
        mean[0] * x[1] + mean[1] * y[1],
        mean[0] * x[2] + mean[1] * y[2],
    ];
    Ok([count as f64 * step[0] * step[1], cp[0], cp[1], cp[2]])
}

/// Evaluate the occlusion-aware raster silhouette for every direction.
pub fn silhouette_batch(
    triangles: &[f64],
    directions: &[f64],
    resolution: usize,
) -> Result<Vec<f64>, String> {
    if triangles.is_empty() || triangles.len() % 9 != 0 {
        return Err("triangles must contain one or more Nx3x3 rows".to_owned());
    }
    if directions.is_empty() || directions.len() % 3 != 0 {
        return Err("directions must contain one or more 3-vectors".to_owned());
    }
    let mut result = Vec::with_capacity(directions.len() / 3 * 4);
    for direction in directions.chunks_exact(3) {
        result.extend_from_slice(&silhouette_one(
            triangles,
            [direction[0], direction[1], direction[2]],
            resolution,
        )?);
    }
    Ok(result)
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;

    use pyo3::types::PyBytes;

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
    fn spacecraft_facet_projected_batch_bytes(
        py: Python<'_>,
        triangles: &[u8],
        normals: &[u8],
        directions: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let triangles = decode_f64_bytes(triangles, "triangles")?;
        let normals = decode_f64_bytes(normals, "normals")?;
        let directions = decode_f64_bytes(directions, "directions")?;
        let values = py
            .detach(|| {
                facet_projected_geometry(
                    &triangles,
                    if normals.is_empty() {
                        None
                    } else {
                        Some(&normals)
                    },
                    &directions,
                )
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        encode_f64_bytes(py, &values)
    }

    #[pyfunction]
    fn spacecraft_silhouette_batch_bytes(
        py: Python<'_>,
        triangles: &[u8],
        directions: &[u8],
        resolution: usize,
    ) -> PyResult<Py<PyBytes>> {
        let triangles = decode_f64_bytes(triangles, "triangles")?;
        let directions = decode_f64_bytes(directions, "directions")?;
        let values = py
            .detach(|| silhouette_batch(&triangles, &directions, resolution))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        encode_f64_bytes(py, &values)
    }

    #[pyfunction]
    fn spacecraft_facet_projected_batch(
        triangles: Vec<f64>,
        normals: Vec<f64>,
        directions: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        facet_projected_geometry(
            &triangles,
            if normals.is_empty() {
                None
            } else {
                Some(&normals)
            },
            &directions,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[pyfunction]
    fn spacecraft_silhouette_batch(
        triangles: Vec<f64>,
        directions: Vec<f64>,
        resolution: usize,
    ) -> PyResult<Vec<f64>> {
        silhouette_batch(&triangles, &directions, resolution)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    module.add_function(wrap_pyfunction!(spacecraft_facet_projected_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        spacecraft_facet_projected_batch_bytes,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(spacecraft_silhouette_batch_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(spacecraft_silhouette_batch, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{facet_projected_geometry, silhouette_batch};

    fn square() -> Vec<f64> {
        vec![
            -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0,
            1.0, 0.0,
        ]
    }

    #[test]
    fn facet_projection_has_expected_area_and_centroid() {
        let values =
            facet_projected_geometry(&square(), None, &[0.0, 0.0, -1.0, 1.0, 0.0, 0.0]).unwrap();
        assert!((values[0] - 4.0).abs() < 1e-12);
        assert!(values[1].abs() < 1e-12 && values[2].abs() < 1e-12 && values[3].abs() < 1e-12);
        assert!(values[4].abs() < 1e-12);
    }

    #[test]
    fn silhouette_is_deterministic_for_overlapping_triangles() {
        let first = silhouette_batch(&square(), &[0.0, 0.0, 1.0], 32).unwrap();
        let second = silhouette_batch(&square(), &[0.0, 0.0, 1.0], 32).unwrap();
        assert_eq!(first, second);
        assert!(first[0] > 0.0);
        assert!(first[1].abs() < 1e-12 && first[2].abs() < 1e-12 && first[3].abs() < 1e-12);
    }
}
