//! Local nadir and target-tracking arithmetic from analysis/optical_collection.py.
type Vector = [f64; 3];
fn dot(a: Vector, b: Vector) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: Vector) -> f64 {
    dot(a, a).sqrt()
}
fn cross(a: Vector, b: Vector) -> Vector {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(a: Vector) -> Vector {
    let n = norm(a);
    a.map(|v| v / n)
}
fn prepare(state: &[f64], target: &[f64], mode: &str) -> Result<([Vector; 3], Vector), String> {
    if state.len() != 6 || target.len() != 3 || state.iter().chain(target).any(|v| !v.is_finite()) {
        return Err("Pointing requires a finite six-state and three-target position.".into());
    }
    if mode != "nadir_fixed" && mode != "target_track_gimbal" {
        return Err("Unsupported pointing mode.".into());
    }
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    if !norm(r).is_finite() || norm(r) <= 0. {
        return Err("Local nadir requires nonzero finite radius.".into());
    }
    let z = unit(r).map(|v| -v);
    let vz = dot(v, z);
    let along = std::array::from_fn(|i| v[i] - vz * z[i]);
    if !norm(along).is_finite() || norm(along) <= 1e-12 {
        return Err("Local nadir requires transverse velocity.".into());
    }
    let x = unit(cross(unit(along), z));
    let y = cross(z, x);
    let los = std::array::from_fn(|i| target[i] - r[i]);
    if mode == "target_track_gimbal" && (!norm(los).is_finite() || norm(los) <= 0.) {
        return Err("Target tracking requires a distinct finite target position.".into());
    }
    Ok(([x, y, z], los))
}
pub fn gimbal(state: &[f64], target: &[f64], mode: &str) -> Result<Vector, String> {
    let (axes, los) = prepare(state, target, mode)?;
    if mode == "nadir_fixed" {
        return Ok([0., 0., 1.]);
    }
    let los = unit(los);
    Ok(axes.map(|axis| dot(axis, los)))
}
pub fn pointing(
    state: &[f64],
    target: &[f64],
    mode: &str,
) -> Result<([Vector; 3], Vector, f64), String> {
    let (axes, los) = prepare(state, target, mode)?;
    if mode == "nadir_fixed" {
        return Ok((axes, [0., 0., 1.], 0.));
    }
    let los = unit(los);
    let g = axes.map(|axis| dot(axis, los));
    let angle = g[2].clamp(-1., 1.).acos();
    let xd = dot(axes[0], los);
    let mut x = std::array::from_fn(|i| axes[0][i] - xd * los[i]);
    if norm(x) <= 1e-12 {
        let yd = dot(axes[1], los);
        x = std::array::from_fn(|i| axes[1][i] - yd * los[i]);
    }
    let x = unit(x);
    Ok(([x, cross(los, x), los], g, angle))
}
#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;
    #[pyfunction]
    fn optical_pointing(
        state: Vec<f64>,
        target: Vec<f64>,
        pointing_mode: &str,
    ) -> PyResult<(Vec<f64>, Vec<f64>, f64)> {
        let (axes, g, angle) = pointing(&state, &target, pointing_mode)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok((axes.into_iter().flatten().collect(), g.to_vec(), angle))
    }
    #[pyfunction]
    fn optical_gimbal_vector(
        state: Vec<f64>,
        target: Vec<f64>,
        pointing_mode: &str,
    ) -> PyResult<Vec<f64>> {
        Ok(gimbal(&state, &target, pointing_mode)
            .map_err(pyo3::exceptions::PyValueError::new_err)?
            .to_vec())
    }
    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(optical_pointing, module)?)?;
        module.add_function(wrap_pyfunction!(optical_gimbal_vector, module)?)?;
        Ok(())
    }
}
#[cfg(feature = "python")]
pub use python::register;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nadir_and_degenerate_inputs() {
        let s = [7000., 0., 0., 0., 7.5, 0.];
        let (axes, g, angle) = pointing(&s, &[6378., 0., 0.], "target_track_gimbal").unwrap();
        assert_eq!(g, [0., 0., 1.]);
        assert_eq!(angle, 0.);
        assert_eq!(axes[2], [-1., 0., 0.]);
        assert!(pointing(&s, &s[..3], "target_track_gimbal").is_err());
        assert!(pointing(&[0.; 6], &[0.; 3], "nadir_fixed").is_err());
        assert!(gimbal(&s, &[0.; 3], "unknown").is_err());
    }
}
