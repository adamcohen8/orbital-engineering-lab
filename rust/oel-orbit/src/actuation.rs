//! Numeric control/allocation and device intervals; Python owns commands and policy.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

fn norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |s, v| s + v * v).sqrt()
}
fn cross(a: &[f64], b: &[f64]) -> Vec<f64> {
    vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn clip(x: f64, limit: f64) -> f64 {
    if x < -limit {
        -limit
    } else if x > limit {
        limit
    } else {
        x
    }
}
fn limited(x: &[f64], limit: f64) -> (Vec<f64>, bool) {
    let n = norm(x);
    if n <= limit || n == 0.0 {
        (x.to_vec(), false)
    } else {
        (x.iter().map(|v| v * (limit / n)).collect(), true)
    }
}
fn shape(x: &[f64], n: usize) -> PyResult<()> {
    if x.len() != n {
        Err(PyValueError::new_err(
            "native actuation vector length mismatch",
        ))
    } else {
        Ok(())
    }
}

#[pyclass]
pub struct AttitudeActuationContext {
    kind: u8,
    kp: [f64; 3],
    kd: [f64; 3],
    maximum: f64,
    detumble: f64,
    limits: Vec<f64>,
    momentum: Vec<f64>,
    inverse: Py<PyAny>,
    axes: Py<PyAny>,
    dot: Py<PyAny>,
    matvec: Py<PyAny>,
}
impl AttitudeActuationContext {
    fn dot(&self, py: Python<'_>, x: &[f64], y: &[f64]) -> PyResult<f64> {
        self.dot.bind(py).call1((x.to_vec(), y.to_vec()))?.extract()
    }
    fn normalize(&self, py: Python<'_>, mut q: [f64; 4]) -> PyResult<[f64; 4]> {
        let n2 = self.dot(py, &q, &q)?;
        if !q.iter().all(|x| x.is_finite()) || !n2.is_finite() || n2 <= 0.0 {
            return Ok([1., 0., 0., 0.]);
        }
        let n = n2.sqrt();
        for x in &mut q {
            *x /= n;
        }
        Ok(q)
    }
    fn allocate_inner(
        &self,
        py: Python<'_>,
        torque: [f64; 3],
        magnetic: Option<[f64; 3]>,
    ) -> PyResult<(Vec<f64>, Vec<f64>, u8)> {
        let t = torque.to_vec();
        let mut channels = t.clone();
        let mut achieved = t.clone();
        match self.kind {
            1 => {
                channels = self
                    .matvec
                    .bind(py)
                    .call1((self.inverse.bind(py), &t))?
                    .extract::<Vec<f64>>()?
                    .iter()
                    .map(|v| -v)
                    .collect();
            }
            2 => {
                let b = match magnetic {
                    Some(v) => v,
                    None => return Ok((vec![], vec![0.; 3], 3)),
                };
                let n = self.dot(py, &b, &b)?;
                if n <= 1e-24 {
                    return Ok((vec![], vec![0.; 3], 3));
                }
                channels = cross(&b, &t).iter().map(|v| v / n).collect();
            }
            3 => {
                channels = t.iter().zip(&self.momentum).map(|(v, m)| v / m).collect();
            }
            _ => return Ok((channels, vec![0.; 3], 0)),
        }
        let saturated = channels
            .iter()
            .zip(&self.limits)
            .any(|(v, l)| v.abs() > l + 1e-15);
        for (v, l) in channels.iter_mut().zip(&self.limits) {
            *v = clip(*v, *l);
        }
        match self.kind {
            1 => {
                achieved = self
                    .matvec
                    .bind(py)
                    .call1((self.axes.bind(py), &channels))?
                    .extract::<Vec<f64>>()?
                    .iter()
                    .map(|v| -v)
                    .collect()
            }
            2 => achieved = cross(&channels, &magnetic.unwrap()),
            3 => {
                achieved = channels
                    .iter()
                    .zip(&self.momentum)
                    .map(|(v, m)| v * m)
                    .collect()
            }
            _ => {}
        }
        let residual: Vec<f64> = t.iter().zip(&achieved).map(|(x, y)| x - y).collect();
        let status = if saturated {
            1
        } else if self.dot(py, &residual, &residual)?.sqrt() > 1e-12 {
            2
        } else {
            0
        };
        Ok((channels, residual, status))
    }
}
#[pymethods]
impl AttitudeActuationContext {
    #[new]
    fn new(
        kind: u8,
        kp: [f64; 3],
        kd: [f64; 3],
        maximum: f64,
        detumble: f64,
        limits: Vec<f64>,
        momentum: Vec<f64>,
        inverse: Py<PyAny>,
        axes: Py<PyAny>,
        dot: Py<PyAny>,
        matvec: Py<PyAny>,
    ) -> PyResult<Self> {
        if kind > 3 || limits.is_empty() || !limits.iter().all(|x| x.is_finite() && *x > 0.) {
            return Err(PyValueError::new_err("invalid native allocation context"));
        }
        Ok(Self {
            kind,
            kp,
            kd,
            maximum,
            detumble,
            limits,
            momentum,
            inverse,
            axes,
            dot,
            matvec,
        })
    }
    fn allocate(
        &self,
        py: Python<'_>,
        torque: [f64; 3],
        magnetic: Option<[f64; 3]>,
    ) -> PyResult<(Vec<f64>, Vec<f64>, u8)> {
        self.allocate_inner(py, torque, magnetic)
    }
    fn control_allocate(
        &self,
        py: Python<'_>,
        q: [f64; 4],
        desired: [f64; 4],
        omega: [f64; 3],
        magnetic: Option<[f64; 3]>,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, u8)> {
        let mut torque = [0.; 3];
        if self.dot(py, &omega, &omega)?.sqrt() > self.detumble {
            for i in 0..3 {
                torque[i] = -self.kd[i] * omega[i];
            }
        } else {
            let a = self.normalize(py, [desired[0], -desired[1], -desired[2], -desired[3]])?;
            let b = self.normalize(py, q)?;
            let mut e = self.normalize(
                py,
                [
                    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
                ],
            )?;
            if e[0] < 0. {
                for v in &mut e {
                    *v = -*v;
                }
            }
            for i in 0..3 {
                torque[i] = -self.kp[i] * e[i + 1] - self.kd[i] * omega[i];
            }
        }
        let n = self.dot(py, &torque, &torque)?.sqrt();
        if n > self.maximum && self.maximum > 0. {
            for v in &mut torque {
                *v *= self.maximum / n;
            }
        }
        let (channels, residual, status) = self.allocate_inner(py, torque, magnetic)?;
        Ok((torque.to_vec(), channels, residual, status))
    }
}

#[pyclass]
pub struct ActuatorIntervalContext {
    kind: u8,
    params: Vec<f64>,
    axes: Py<PyAny>,
    matvec: Py<PyAny>,
}
#[pymethods]
impl ActuatorIntervalContext {
    #[new]
    fn new(kind: u8, params: Vec<f64>, axes: Py<PyAny>, matvec: Py<PyAny>) -> PyResult<Self> {
        if kind > 4 {
            return Err(PyValueError::new_err("invalid actuator context kind"));
        }
        Ok(Self {
            kind,
            params,
            axes,
            matvec,
        })
    }
    fn advance(
        &self,
        py: Python<'_>,
        request: Vec<f64>,
        previous: Vec<f64>,
        dt: f64,
    ) -> PyResult<(
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        bool,
        f64,
        Vec<f64>,
    )> {
        if dt < 0. {
            return Err(PyValueError::new_err(
                "actuator interval must be nonnegative",
            ));
        }
        let p = &self.params;
        let z = vec![0.; 3];
        match self.kind {
            0 => {
                shape(&request, 6)?;
                shape(&previous, 6)?;
                shape(p, 4)?;
                let (f, fs) = limited(&request[..3], p[0]);
                let (t, ts) = limited(&request[3..], p[1]);
                let alpha = if p[2] <= 0. {
                    1.
                } else {
                    1. - (-dt / p[2]).exp()
                };
                let target: Vec<f64> = f.into_iter().chain(t).collect();
                let state: Vec<f64> = previous
                    .iter()
                    .zip(&target)
                    .map(|(old, new)| old + alpha * (new - old))
                    .collect();
                let flow = if p[3] <= 0. {
                    0.
                } else {
                    norm(&state[..3]) / (p[3] * 9.80665)
                };
                Ok((
                    request[..3].to_vec(),
                    request[3..].to_vec(),
                    state[..3].to_vec(),
                    state[3..].to_vec(),
                    state,
                    fs || ts,
                    flow,
                    vec![],
                ))
            }
            1 => {
                let n = request.len();
                shape(&previous, n)?;
                shape(p, 2 * n)?;
                let clipped: Vec<f64> = request
                    .iter()
                    .zip(&p[..n])
                    .map(|(x, l)| clip(*x, *l))
                    .collect();
                let state = if dt > 0. {
                    previous
                        .iter()
                        .zip(&clipped)
                        .zip(&p[n..])
                        .map(|((old, t), l)| clip(old + t * dt, *l))
                        .collect::<Vec<f64>>()
                } else {
                    previous.clone()
                };
                let realized = if dt > 0. {
                    state
                        .iter()
                        .zip(&previous)
                        .map(|(new, old)| (new - old) / dt)
                        .collect::<Vec<f64>>()
                } else {
                    clipped
                };
                let requested: Vec<f64> = self
                    .matvec
                    .bind(py)
                    .call1((self.axes.bind(py), &request))?
                    .extract()?;
                let body: Vec<f64> = self
                    .matvec
                    .bind(py)
                    .call1((self.axes.bind(py), &realized))?
                    .extract()?;
                let saturated = request
                    .iter()
                    .zip(&realized)
                    .any(|(a, b)| (a - b).abs() > 1e-15);
                Ok((
                    z.clone(),
                    requested.iter().map(|v| -v).collect(),
                    z,
                    body.iter().map(|v| -v).collect(),
                    state,
                    saturated,
                    0.,
                    vec![],
                ))
            }
            2 => {
                shape(&request, 3)?;
                shape(p, 6)?;
                let realized: Vec<f64> = request.iter().zip(p).map(|(v, l)| clip(*v, *l)).collect();
                let saturated = request != realized;
                Ok((
                    z.clone(),
                    cross(&request, &p[3..]),
                    z,
                    cross(&realized, &p[3..]),
                    vec![],
                    saturated,
                    0.,
                    realized,
                ))
            }
            3 => {
                shape(&request, 3)?;
                shape(&previous, 3)?;
                shape(p, 6)?;
                let realized: Vec<f64> = request
                    .iter()
                    .zip(&p[3..])
                    .map(|(v, l)| clip(*v, *l))
                    .collect();
                let state = previous
                    .iter()
                    .zip(&realized)
                    .map(|(old, v)| old + v * dt)
                    .collect();
                let requested = request.iter().zip(p).map(|(v, m)| v * m).collect();
                let body = realized.iter().zip(p).map(|(v, m)| v * m).collect();
                Ok((
                    z.clone(),
                    requested,
                    z,
                    body,
                    state,
                    request != realized,
                    0.,
                    vec![],
                ))
            }
            _ => {
                shape(&request, 3)?;
                shape(p, 2)?;
                let throttle = request[0];
                let yaw = request[1];
                let pitch = request[2];
                let force = vec![
                    pitch.cos() * yaw.cos() * throttle * p[0],
                    pitch.cos() * yaw.sin() * throttle * p[0],
                    -pitch.sin() * throttle * p[0],
                ];
                let flow = if p[1] <= 0. {
                    0.
                } else {
                    norm(&force) / (p[1] * 9.80665)
                };
                Ok((
                    force.clone(),
                    z.clone(),
                    force,
                    z,
                    vec![],
                    false,
                    flow,
                    request,
                ))
            }
        }
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<AttitudeActuationContext>()?;
    module.add_class::<ActuatorIntervalContext>()?;
    Ok(())
}
