//! Small numerical kernels for the opt-in spherical collision stepper.
//!
//! Python retains object validation, propagation, recursive curvature
//! refinement, event records, and lifecycle policy.  These helpers only cover
//! geometry and the elastic impulse so that a collision step does not spend its
//! hot path in repeated Python vector arithmetic.

fn vec3(values: &[f64], name: &str) -> Result<[f64; 3], String> {
    if values.len() != 3 {
        return Err(format!("{name} must have three values"));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok([values[0], values[1], values[2]])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    a[0].hypot(a[1]).hypot(a[2])
}

/// Return the closest point fraction and distance for a relative chord.
///
/// The result is `[fraction, miss_distance, chord_norm_squared]`; the
/// fraction is clamped to `[0, 1]`.  This is the same chord test used by the
/// Python collision stepper and does not make a continuous-time physics claim.
pub fn chord_geometry(start_rel: &[f64], end_rel: &[f64]) -> Result<[f64; 3], String> {
    let start = vec3(start_rel, "start_rel")?;
    let end = vec3(end_rel, "end_rel")?;
    let chord = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
    let chord_norm2 = dot(chord, chord);
    let fraction = if chord_norm2 > 0.0 {
        (-dot(start, chord) / chord_norm2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = [
        start[0] + fraction * chord[0],
        start[1] + fraction * chord[1],
        start[2] + fraction * chord[2],
    ];
    Ok([fraction, norm(closest), chord_norm2])
}

/// Solve a straight-line relative-motion sphere crossing in normalized time.
///
/// Returns the earliest root in `[0, 1]`, or `None` when no crossing exists.
/// This helper is useful for a straight-line caller; the existing OEL stepper
/// continues to use propagated endpoints and its curvature refinement.
pub fn linear_contact_fraction(
    start_rel: &[f64],
    end_rel: &[f64],
    radius_km: f64,
) -> Result<Option<f64>, String> {
    let start = vec3(start_rel, "start_rel")?;
    let end = vec3(end_rel, "end_rel")?;
    if !radius_km.is_finite() || radius_km <= 0.0 {
        return Err("radius_km must be finite and positive".to_owned());
    }
    let delta = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
    let a = dot(delta, delta);
    let b = 2.0 * dot(start, delta);
    let c = dot(start, start) - radius_km * radius_km;
    if c <= 0.0 {
        return Ok(Some(0.0));
    }
    if a <= 0.0 {
        return Ok(None);
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return Ok(None);
    }
    let root = discriminant.sqrt();
    let first = (-b - root) / (2.0 * a);
    let second = (-b + root) / (2.0 * a);
    if (0.0..=1.0).contains(&first) {
        Ok(Some(first))
    } else if (0.0..=1.0).contains(&second) {
        Ok(Some(second))
    } else {
        Ok(None)
    }
}

/// Apply an elastic spherical impulse and return post-impact velocities,
/// contact normal, and closing speed.
///
/// The output is `[post_a(3), post_b(3), normal(3), closing_speed_km_s]`.
pub fn elastic_impact(
    position_a: &[f64],
    position_b: &[f64],
    velocity_a: &[f64],
    velocity_b: &[f64],
    mass_a_kg: f64,
    mass_b_kg: f64,
    restitution: f64,
) -> Result<[f64; 10], String> {
    let position_a = vec3(position_a, "position_a")?;
    let position_b = vec3(position_b, "position_b")?;
    let velocity_a = vec3(velocity_a, "velocity_a")?;
    let velocity_b = vec3(velocity_b, "velocity_b")?;
    if !mass_a_kg.is_finite() || !mass_b_kg.is_finite() || mass_a_kg <= 0.0 || mass_b_kg <= 0.0 {
        return Err("collision masses must be finite and positive".to_owned());
    }
    if !restitution.is_finite() || restitution < 0.0 {
        return Err("restitution must be finite and nonnegative".to_owned());
    }
    let relative_position = [
        position_b[0] - position_a[0],
        position_b[1] - position_a[1],
        position_b[2] - position_a[2],
    ];
    let distance = norm(relative_position);
    if distance <= 0.0 {
        return Err("collision has no contact normal at coincident centers".to_owned());
    }
    let normal = [
        relative_position[0] / distance,
        relative_position[1] / distance,
        relative_position[2] / distance,
    ];
    let relative_velocity = [
        velocity_b[0] - velocity_a[0],
        velocity_b[1] - velocity_a[1],
        velocity_b[2] - velocity_a[2],
    ];
    let closing = -dot(relative_velocity, normal);
    if closing <= 0.0 {
        return Err("collision impulse requires approaching objects".to_owned());
    }
    let impulse = (1.0 + restitution) * closing / (1.0 / mass_a_kg + 1.0 / mass_b_kg);
    let post_a = [
        velocity_a[0] - impulse * normal[0] / mass_a_kg,
        velocity_a[1] - impulse * normal[1] / mass_a_kg,
        velocity_a[2] - impulse * normal[2] / mass_a_kg,
    ];
    let post_b = [
        velocity_b[0] + impulse * normal[0] / mass_b_kg,
        velocity_b[1] + impulse * normal[1] / mass_b_kg,
        velocity_b[2] + impulse * normal[2] / mass_b_kg,
    ];
    Ok([
        post_a[0], post_a[1], post_a[2], post_b[0], post_b[1], post_b[2], normal[0], normal[1],
        normal[2], closing,
    ])
}

#[cfg(feature = "python")]
mod python {
    use super::{
        chord_geometry as compute_chord_geometry, elastic_impact as compute_elastic_impact,
        linear_contact_fraction as compute_linear_contact_fraction,
    };
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyfunction]
    fn collision_chord_geometry(start_rel: Vec<f64>, end_rel: Vec<f64>) -> PyResult<Vec<f64>> {
        compute_chord_geometry(&start_rel, &end_rel)
            .map(|values| values.to_vec())
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn collision_linear_contact_fraction(
        start_rel: Vec<f64>,
        end_rel: Vec<f64>,
        radius_km: f64,
    ) -> PyResult<Option<f64>> {
        compute_linear_contact_fraction(&start_rel, &end_rel, radius_km)
            .map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn collision_elastic_impact(
        position_a: Vec<f64>,
        position_b: Vec<f64>,
        velocity_a: Vec<f64>,
        velocity_b: Vec<f64>,
        mass_a_kg: f64,
        mass_b_kg: f64,
        restitution: f64,
    ) -> PyResult<Vec<f64>> {
        compute_elastic_impact(
            &position_a,
            &position_b,
            &velocity_a,
            &velocity_b,
            mass_a_kg,
            mass_b_kg,
            restitution,
        )
        .map(|values| values.to_vec())
        .map_err(PyValueError::new_err)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(collision_chord_geometry, module)?)?;
        module.add_function(wrap_pyfunction!(collision_linear_contact_fraction, module)?)?;
        module.add_function(wrap_pyfunction!(collision_elastic_impact, module)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{chord_geometry, elastic_impact, linear_contact_fraction};

    #[test]
    fn chord_and_linear_crossing_match() {
        let start = [0.01, 0.0, 0.0];
        let end = [-0.01, 0.0, 0.0];
        let chord = chord_geometry(&start, &end).unwrap();
        assert_eq!(chord[0], 0.5);
        assert_eq!(chord[1], 0.0);
        assert!(
            (linear_contact_fraction(&start, &end, 0.004)
                .unwrap()
                .unwrap()
                - 0.3)
                .abs()
                < 1.0e-15
        );
    }

    #[test]
    fn elastic_impact_conserves_momentum() {
        let result = elastic_impact(
            &[0.0, 0.0, 0.0],
            &[0.004, 0.0, 0.0],
            &[0.01, 0.0, 0.0],
            &[-0.01, 0.0, 0.0],
            100.0,
            200.0,
            1.0,
        )
        .unwrap();
        assert!(
            (100.0 * 0.01 + 200.0 * -0.01 - (100.0 * result[0] + 200.0 * result[3])).abs()
                < 1.0e-14
        );
        assert!((result[9] - 0.02).abs() < 1.0e-14);
    }
}

#[cfg(feature = "python")]
pub use python::register;
