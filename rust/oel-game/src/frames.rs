//! Fused scalar geometry with the same NumPy dot/matmul reduction owners.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

type V3 = [f64; 3];
type V6 = [f64; 6];
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn divide(a: V3, n: f64) -> V3 {
    a.map(|x| x / n)
}
fn sub(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] - b[i])
}
fn pos(a: V6) -> V3 {
    [a[0], a[1], a[2]]
}
fn vel(a: V6) -> V3 {
    [a[3], a[4], a[5]]
}

#[pyclass]
pub struct FrameConverter {
    frombuffer: Py<PyAny>,
    dot: Py<PyAny>,
}
impl FrameConverter {
    fn array<'py>(&self, py: Python<'py>, values: &[f64]) -> PyResult<Bound<'py, PyAny>> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.frombuffer.bind(py).call1((PyBytes::new(py, &bytes),))
    }
    fn norm2(&self, py: Python<'_>, values: &[f64]) -> PyResult<f64> {
        let a = self.array(py, values)?;
        self.dot.bind(py).call1((&a, &a))?.extract()
    }
    fn basis<'py>(&self, py: Python<'py>, chief: V6) -> PyResult<(Bound<'py, PyAny>, V3)> {
        let r = pos(chief);
        let v = vel(chief);
        let r2 = self.norm2(py, &r)?;
        let rn = r2.sqrt();
        if !rn.is_finite() || rn <= 1e-12 {
            return Err(PyValueError::new_err(
                "RIC frame is undefined for a zero or non-finite position vector.",
            ));
        }
        let rh = divide(r, rn);
        let h = cross(r, v);
        let hn = self.norm2(py, &h)?.sqrt();
        if !hn.is_finite() || hn <= 1e-12 {
            return Err(PyValueError::new_err(
                "RIC frame is undefined for zero angular momentum.",
            ));
        }
        let ch = divide(h, hn);
        let ih = cross(ch, rh);
        let inn = self.norm2(py, &ih)?.sqrt();
        if !inn.is_finite() || inn <= 1e-12 {
            return Err(PyValueError::new_err(
                "RIC frame is undefined for a degenerate basis.",
            ));
        }
        let ih = divide(ih, inn);
        let rows = [
            rh[0], ih[0], ch[0], rh[1], ih[1], ch[1], rh[2], ih[2], ch[2],
        ];
        let matrix = self
            .array(py, &rows)?
            .call_method1("reshape", ((3, 3),))?
            .getattr("T")?;
        let omega = if r2 <= 1e-12 { [0.; 3] } else { divide(h, r2) };
        Ok((matrix, omega))
    }
    fn project(&self, py: Python<'_>, matrix: &Bound<'_, PyAny>, vector: V3) -> PyResult<V3> {
        matrix
            .call_method1("__matmul__", (self.array(py, &vector)?,))?
            .extract()
    }
    fn relative(
        &self,
        py: Python<'_>,
        chief: V6,
        deputy: V6,
        matrix: &Bound<'_, PyAny>,
        omega: V3,
    ) -> PyResult<V6> {
        let dr = sub(pos(deputy), pos(chief));
        let dv = sub(sub(vel(deputy), vel(chief)), cross(omega, dr));
        let p = self.project(py, matrix, dr)?;
        let v = self.project(py, matrix, dv)?;
        Ok([p[0], p[1], p[2], v[0], v[1], v[2]])
    }
    fn quaternion(&self, py: Python<'_>, q: Vec<f64>) -> PyResult<[[f64; 3]; 3]> {
        let q = if q.len() != 4 || q.iter().any(|x| !x.is_finite()) {
            [1., 0., 0., 0.]
        } else {
            let n2 = self.norm2(py, &q)?;
            if n2 <= 0. || !n2.is_finite() {
                [1., 0., 0., 0.]
            } else if n2 == 1. {
                [q[0], q[1], q[2], q[3]]
            } else {
                let n = n2.sqrt();
                if n <= 0. || !n.is_finite() {
                    [1., 0., 0., 0.]
                } else {
                    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
                }
            }
        };
        let [q0, q1, q2, q3] = q;
        let values = self.array(py, &q)?;
        let sq1: f64 = values
            .get_item(1)?
            .call_method1("__pow__", (2,))?
            .extract()?;
        let sq2: f64 = values
            .get_item(2)?
            .call_method1("__pow__", (2,))?
            .extract()?;
        let sq3: f64 = values
            .get_item(3)?
            .call_method1("__pow__", (2,))?
            .extract()?;
        Ok([
            [
                1. - 2. * (sq2 + sq3),
                2. * (q1 * q2 + q0 * q3),
                2. * (q1 * q3 - q0 * q2),
            ],
            [
                2. * (q1 * q2 - q0 * q3),
                1. - 2. * (sq1 + sq3),
                2. * (q2 * q3 + q0 * q1),
            ],
            [
                2. * (q1 * q3 + q0 * q2),
                2. * (q2 * q3 - q0 * q1),
                1. - 2. * (sq1 + sq2),
            ],
        ])
    }
}
#[pymethods]
impl FrameConverter {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        let numpy = py.import("numpy")?;
        Ok(Self {
            frombuffer: numpy.getattr("frombuffer")?.unbind(),
            dot: numpy.getattr("dot")?.unbind(),
        })
    }
    fn sample(
        &self,
        py: Python<'_>,
        target: V6,
        chaser: V6,
        reference: Option<V6>,
        thrust: V3,
    ) -> PyResult<(V6, V6, V3)> {
        let (matrix, omega) = self.basis(py, target)?;
        let relative = self.relative(py, target, chaser, &matrix, omega)?;
        let reference_relative = if let Some(reference) = reference {
            let (m, w) = self.basis(py, reference)?;
            self.relative(py, reference, target, &m, w)?
        } else {
            [f64::NAN; 6]
        };
        Ok((
            relative,
            reference_relative,
            self.project(py, &matrix, thrust)?,
        ))
    }
    fn relative_state(&self, py: Python<'_>, deputy: V6, chief: V6) -> PyResult<V6> {
        let (matrix, omega) = self.basis(py, chief)?;
        self.relative(py, chief, deputy, &matrix, omega)
    }
    fn quaternion_matrices(
        &self,
        py: Python<'_>,
        quaternions: Vec<Vec<f64>>,
    ) -> PyResult<Vec<[[f64; 3]; 3]>> {
        quaternions
            .into_iter()
            .map(|q| self.quaternion(py, q))
            .collect()
    }
}
