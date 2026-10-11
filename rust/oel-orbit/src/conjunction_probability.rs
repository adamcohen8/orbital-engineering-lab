//! Stateless conditional-Gaussian callback. SciPy owns quadrature and policy.

fn normal_cdf(value: f64) -> f64 {
    0.5 * libm::erfc(-value / std::f64::consts::SQRT_2)
}

pub fn integrand(values: &[f64]) -> f64 {
    if values.len() != 7
        || values.iter().any(|x| !x.is_finite())
        || values[2] <= 0.0
        || values[3] <= 0.0
        || values[6] <= 0.0
    {
        return f64::NAN;
    }
    let [z, mean_x, sigma_x, radius, mean_y, slope, sigma_y] =
        <[f64; 7]>::try_from(values).unwrap();
    let x = mean_x + sigma_x * z;
    let half_height_squared = radius * radius - x * x;
    if half_height_squared <= 0.0 {
        return 0.0;
    }
    let half_height = half_height_squared.sqrt();
    let conditional_mean = mean_y + slope * z;
    let lower = (-half_height - conditional_mean) / sigma_y;
    let upper = (half_height - conditional_mean) / sigma_y;
    let interval = if lower >= upper {
        0.0
    } else if lower > 0.0 {
        normal_cdf(-lower) - normal_cdf(-upper)
    } else {
        normal_cdf(upper) - normal_cdf(lower)
    };
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt() * interval
}

/// SciPy supplies a pointer to `count` doubles for the duration of this call.
/// No Python objects, retained pointers, global state or allocation are used.
unsafe extern "C" fn callback(count: i32, values: *mut f64) -> f64 {
    if count != 7 || values.is_null() {
        return f64::NAN;
    }
    integrand(std::slice::from_raw_parts(values, count as usize))
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    #[pyfunction]
    fn conjunction_probability_integrand_capsule(py: Python<'_>) -> PyResult<Py<PyAny>> {
        // A capsule contains the function address itself, with SciPy's exact ABI
        // name. Its function is static for the lifetime of this extension.
        unsafe {
            let ptr = pyo3::ffi::PyCapsule_New(
                callback as *const () as *mut std::ffi::c_void,
                c"double (int, double *)".as_ptr(),
                None,
            );
            if ptr.is_null() {
                return Err(PyErr::fetch(py));
            }
            Ok(Bound::from_owned_ptr(py, ptr).unbind())
        }
    }
    module.add_function(wrap_pyfunction!(
        conjunction_probability_integrand_capsule,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positive_tail_and_invalid_callback_fail_closed() {
        let centered = integrand(&[0., 0., 0.1, 0.02, 0., 0., 0.1]);
        assert!((centered - 0.06324009845518144).abs() < 1e-15);
        assert!(integrand(&[0.; 7]).is_nan());
        assert!(unsafe { callback(6, std::ptr::null_mut()) }.is_nan());
        let tail = integrand(&[0., 0., 1., 1., -10., 0., 1.]);
        assert!(tail > 0.0 && tail < 1e-17);
    }
}
