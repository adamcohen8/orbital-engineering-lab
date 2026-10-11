//! Chunked geometric gates for global communications coverage.

pub fn geometry_tile(
    sources: &[f64],
    terminal_rotations: &[f64],
    centers: &[f64],
    normals: &[f64],
    minimum_elevation_rad: f64,
    max_range_km: Option<f64>,
    source_half_angle_rad: Option<f64>,
    earth_half_angle_rad: Option<f64>,
    direct_cosine: bool,
) -> Result<(Vec<f64>, Vec<u8>), &'static str> {
    if sources.len() % 3 != 0
        || terminal_rotations.len() != sources.len() * 3
        || centers.len() % 3 != 0
        || centers.len() != normals.len()
    {
        return Err("communications tile arrays have inconsistent shapes");
    }
    if !sources
        .iter()
        .chain(terminal_rotations)
        .chain(centers)
        .chain(normals)
        .all(|x| x.is_finite())
    {
        return Err("communications tile arrays must be finite");
    }
    let samples = sources.len() / 3;
    let cells = centers.len() / 3;
    let mut ranges = Vec::with_capacity(samples * cells);
    let mut gates = Vec::with_capacity(samples * cells);
    let source_cosine = source_half_angle_rad.map(|angle| (angle + 1.0e-12).cos());
    let earth_cosine = earth_half_angle_rad.map(|angle| (angle + 1.0e-12).cos());
    for sample in 0..samples {
        let source = &sources[3 * sample..3 * sample + 3];
        let z_axis = &terminal_rotations[9 * sample + 6..9 * sample + 9];
        for cell in 0..cells {
            let center = &centers[3 * cell..3 * cell + 3];
            let normal = &normals[3 * cell..3 * cell + 3];
            let delta = [
                center[0] - source[0],
                center[1] - source[1],
                center[2] - source[2],
            ];
            let range = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            if !range.is_finite() || range <= 0.0 {
                return Err("communications source-to-cell range must be positive");
            }
            let direction = [delta[0] / range, delta[1] / range, delta[2] / range];
            let horizon = normal[0] * (source[0] - center[0])
                + normal[1] * (source[1] - center[1])
                + normal[2] * (source[2] - center[2]);
            let earth_cos =
                (normal[0] * -direction[0] + normal[1] * -direction[1] + normal[2] * -direction[2])
                    .clamp(-1.0, 1.0);
            let source_cos =
                (z_axis[0] * direction[0] + z_axis[1] * direction[1] + z_axis[2] * direction[2])
                    .clamp(-1.0, 1.0);
            let source_pass = match source_half_angle_rad {
                None => true,
                Some(_) if direct_cosine => source_cos >= source_cosine.unwrap(),
                Some(angle) => source_cos.acos() <= angle + 1.0e-12,
            };
            let earth_pass = match earth_half_angle_rad {
                None => true,
                Some(_) if direct_cosine => earth_cos >= earth_cosine.unwrap(),
                Some(angle) => earth_cos.acos() <= angle + 1.0e-12,
            };
            let mut flags = 0_u8;
            if horizon > 1.0e-9 {
                flags |= 1;
            }
            if earth_cos.asin() >= minimum_elevation_rad - 1.0e-12 {
                flags |= 2;
            }
            if max_range_km.is_none_or(|maximum| range <= maximum + 1.0e-9) {
                flags |= 4;
            }
            if source_pass {
                flags |= 8;
            }
            if earth_pass {
                flags |= 16;
            }
            ranges.push(range);
            gates.push(flags);
        }
    }
    Ok((ranges, gates))
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn coverage_communications_geometry_tile_bytes(
        py: Python<'_>,
        sources: &[u8],
        rotations: &[u8],
        centers: &[u8],
        normals: &[u8],
        min_elevation: f64,
        max_range: Option<f64>,
        source_angle: Option<f64>,
        earth_angle: Option<f64>,
        direct_cosine: bool,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let sources = decode_f64_bytes(sources)?;
        let rotations = decode_f64_bytes(rotations)?;
        let centers = decode_f64_bytes(centers)?;
        let normals = decode_f64_bytes(normals)?;
        let (ranges, flags) = py
            .detach(|| {
                geometry_tile(
                    sources.as_ref(),
                    rotations.as_ref(),
                    centers.as_ref(),
                    normals.as_ref(),
                    min_elevation,
                    max_range,
                    source_angle,
                    earth_angle,
                    direct_cosine,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            output_bytes(py, &ranges)?,
            PyBytes::new(py, &flags).unbind(),
        ))
    }

    module.add_function(wrap_pyfunction!(
        coverage_communications_geometry_tile_bytes,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::geometry_tile;

    #[test]
    fn returns_one_range_and_gate_row() {
        let (ranges, flags) = geometry_tile(
            &[6878.137, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            &[6378.137, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            0.0,
            None,
            Some(0.5),
            None,
            false,
        )
        .unwrap();
        assert_eq!(ranges, vec![500.0]);
        assert_eq!(flags, vec![31]);
    }
}
