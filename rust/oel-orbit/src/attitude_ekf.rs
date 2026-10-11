//! Eight ordered Euler attitude-EKF trials. Measurement and covariance policy
//! remain with the Python estimator; this is not an exponential-map integrator.

#[derive(Clone)]
pub struct Context {
    inertia: [[f64; 3]; 3],
    lu: [[f64; 3]; 3],
    pivots: [usize; 3],
}

impl Context {
    pub fn new(inertia: [[f64; 3]; 3]) -> Result<Self, &'static str> {
        if inertia.iter().flatten().any(|v| !v.is_finite()) {
            return Err("inertia_kg_m2 must contain finite values.");
        }
        let mut lu = inertia;
        let mut pivots = [0, 1, 2];
        for k in 0..3 {
            let mut pivot = k;
            for i in k + 1..3 {
                if lu[i][k].abs() > lu[pivot][k].abs() {
                    pivot = i;
                }
            }
            if lu[pivot][k] == 0.0 {
                return Err("Singular matrix");
            }
            pivots[k] = pivot;
            lu.swap(k, pivot);
            for i in k + 1..3 {
                lu[i][k] /= lu[k][k];
                for j in k + 1..3 {
                    lu[i][j] -= lu[i][k] * lu[k][j];
                }
            }
        }
        Ok(Self {
            inertia,
            lu,
            pivots,
        })
    }

    fn solve(&self, mut rhs: [f64; 3]) -> [f64; 3] {
        for k in 0..3 {
            rhs.swap(k, self.pivots[k]);
        }
        for i in 0..3 {
            for j in 0..i {
                rhs[i] -= self.lu[i][j] * rhs[j];
            }
        }
        for i in (0..3).rev() {
            for j in i + 1..3 {
                rhs[i] -= self.lu[i][j] * rhs[j];
            }
            rhs[i] /= self.lu[i][i];
        }
        rhs
    }

    pub fn propagate(&self, state: [f64; 7], dt: f64) -> [f64; 7] {
        let q = crate::attitude::normalize_quaternion(state[..4].try_into().unwrap());
        let w: [f64; 3] = state[4..].try_into().unwrap();
        let q_dot = [
            0.5 * (-w[0] * q[1] - w[1] * q[2] - w[2] * q[3]),
            0.5 * (w[0] * q[0] + w[2] * q[2] - w[1] * q[3]),
            0.5 * (w[1] * q[0] - w[2] * q[1] + w[0] * q[3]),
            0.5 * (w[2] * q[0] + w[1] * q[1] - w[0] * q[2]),
        ];
        let next =
            crate::attitude::normalize_quaternion(std::array::from_fn(|i| q[i] + dt * q_dot[i]));
        let iw: [f64; 3] = std::array::from_fn(|i| {
            self.inertia[i][0] * w[0] + self.inertia[i][1] * w[1] + self.inertia[i][2] * w[2]
        });
        let rate = self.solve([
            -(w[1] * iw[2] - w[2] * iw[1]),
            -(w[2] * iw[0] - w[0] * iw[2]),
            -(w[0] * iw[1] - w[1] * iw[0]),
        ]);
        [
            next[0],
            next[1],
            next[2],
            next[3],
            w[0] + dt * rate[0],
            w[1] + dt * rate[1],
            w[2] + dt * rate[2],
        ]
    }

    pub fn predict(
        &self,
        state: [f64; 7],
        dt: f64,
        base: Option<[f64; 7]>,
    ) -> ([f64; 7], [[f64; 7]; 7]) {
        let nominal = base.unwrap_or_else(|| self.propagate(state, dt));
        let mut jacobian = [[0.0; 7]; 7];
        for axis in 0..7 {
            let mut trial = state;
            trial[axis] += 1.0e-6;
            let value = self.propagate(trial, dt);
            for row in 0..7 {
                jacobian[row][axis] = (value[row] - nominal[row]) / 1.0e-6;
            }
        }
        (nominal, jacobian)
    }
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
    #[pyclass(name = "AttitudeEKFContext")]
    struct PythonContext {
        core: Context,
    }
    fn state(data: &[u8]) -> PyResult<[f64; 7]> {
        decode_f64_bytes(data)?
            .as_ref()
            .try_into()
            .map_err(|_| PyValueError::new_err("attitude EKF state requires seven values"))
    }
    #[pymethods]
    impl PythonContext {
        #[new]
        fn new(data: &[u8]) -> PyResult<Self> {
            let values = decode_f64_bytes(data)?;
            if values.len() != 9 {
                return Err(PyValueError::new_err(
                    "attitude EKF inertia requires nine values",
                ));
            }
            let inertia = std::array::from_fn(|i| std::array::from_fn(|j| values[3 * i + j]));
            Ok(Self {
                core: Context::new(inertia).map_err(PyValueError::new_err)?,
            })
        }
        fn propagate(&self, py: Python<'_>, data: &[u8], dt: f64) -> PyResult<Py<PyBytes>> {
            let state = state(data)?;
            if !dt.is_finite() {
                return Err(PyValueError::new_err("attitude EKF dt must be finite"));
            }
            output_bytes(py, &self.core.propagate(state, dt))
        }
        #[pyo3(signature = (data, dt, base=None))]
        fn predict(
            &self,
            py: Python<'_>,
            data: &[u8],
            dt: f64,
            base: Option<&[u8]>,
        ) -> PyResult<Py<PyBytes>> {
            let initial = state(data)?;
            let base = base.map(state).transpose()?;
            if !dt.is_finite() {
                return Err(PyValueError::new_err("attitude EKF dt must be finite"));
            }
            let (nominal, jacobian) = self.core.predict(initial, dt, base);
            let mut out = nominal.to_vec();
            out.extend(jacobian.into_iter().flatten());
            output_bytes(py, &out)
        }
    }
    module.add_class::<PythonContext>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_euler_identity_and_rejects_singular_inertia() {
        assert!(Context::new([[0.0; 3]; 3]).is_err());
        let core = Context::new([[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 8.0]]).unwrap();
        let state = [1.0, 0.0, 0.0, 0.0, 0.01, -0.02, 0.03];
        let (next, jac) = core.predict(state, 0.0, None);
        assert_eq!(state, next);
        assert!(jac[0][0].abs() < 1.0e-9);
        assert!((jac[4][4] - 1.0).abs() < 1.0e-9);
    }
}
