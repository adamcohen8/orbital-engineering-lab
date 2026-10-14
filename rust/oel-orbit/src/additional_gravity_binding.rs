//! Optional composition boundary. Ordinary Earth and planetary plans do not visit it.
use crate::{
    adaptive::{self, StepInfo},
    additional_gravity::{AdditionalGravity, SourceInput},
    force_context::NativeForcePlanContext,
    planetary::EphemerisInput,
    planetary_binding::NativePlanetaryPlan,
};
use pyo3::{exceptions::PyValueError, prelude::*};

fn fixed<const N: usize>(v: Vec<f64>) -> PyResult<[f64; N]> {
    let a: [f64; N] = v
        .try_into()
        .map_err(|_| PyValueError::new_err("incorrect vector length"))?;
    if !a.iter().all(|x| x.is_finite()) {
        return Err(PyValueError::new_err("inputs must be finite"));
    }
    Ok(a)
}
fn info(v: StepInfo) -> Vec<f64> {
    vec![
        v.accepted_steps as f64,
        v.rejected_steps as f64,
        v.attempted_steps as f64,
        v.min_step_s.unwrap_or(0.),
        v.max_step_s.unwrap_or(0.),
        v.final_step_s.unwrap_or(0.),
        v.suggested_next_step_s.unwrap_or(0.),
        v.max_error_ratio,
    ]
}
fn integrate<F>(
    x: [f64; 6],
    t: f64,
    dt: f64,
    method: &str,
    atol: f64,
    rtol: f64,
    h: Option<f64>,
    mut derivative: F,
) -> PyResult<(Vec<f64>, Vec<f64>)>
where
    F: FnMut(f64, [f64; 6]) -> PyResult<[f64; 6]>,
{
    if !t.is_finite()
        || !dt.is_finite()
        || dt < 0.
        || !atol.is_finite()
        || !rtol.is_finite()
        || atol < 0.
        || rtol < 0.
        || h.is_some_and(|v| !v.is_finite() || v <= 0.)
        || !matches!(method, "rk4" | "rkf78" | "adaptive" | "dopri5")
    {
        return Err(PyValueError::new_err(
            "invalid additional gravity integration inputs",
        ));
    }
    if dt == 0. {
        derivative(t, x)?;
        return Ok((x.to_vec(), info(StepInfo::default())));
    }
    let mut original_error = None;
    let mut d = |t, x| {
        derivative(t, x).map_err(|e| {
            original_error = Some(e);
            "additional gravity stage failed".to_string()
        })
    };
    let result = match method {
        "rkf78" | "adaptive" => adaptive::integrate_rkf78(x, t, dt, atol, rtol, h, &mut d),
        "dopri5" => adaptive::integrate_dopri5(x, t, dt, atol, rtol, h, &mut d),
        "rk4" => (|| {
            let k1 = d(t, x)?;
            let k2 = d(t + dt / 2., crate::stage(x, k1, dt / 2.))?;
            let k3 = d(t + dt / 2., crate::stage(x, k2, dt / 2.))?;
            let k4 = d(t + dt, crate::stage(x, k3, dt))?;
            let y =
                std::array::from_fn(|i| x[i] + dt / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]));
            if !y.iter().all(|v| v.is_finite()) {
                return Err("nonfinite integrated state".into());
            }
            Ok((y, StepInfo::default()))
        })(),
        _ => Err("unsupported additional gravity integrator".into()),
    };
    if let Some(error) = original_error {
        return Err(error);
    }
    let (y, i) = result.map_err(PyValueError::new_err)?;
    Ok((y.to_vec(), info(i)))
}

#[pyclass(name = "AdditionalGravityPlan")]
pub struct NativeAdditionalGravity {
    plan: AdditionalGravity,
    frames: Vec<Option<Py<PyAny>>>,
}
impl NativeAdditionalGravity {
    fn components(&self, py: Python<'_>, t: f64, x: [f64; 6]) -> PyResult<Vec<[f64; 3]>> {
        let rotations = self
            .frames
            .iter()
            .map(|f| {
                f.as_ref()
                    .map(|callback| {
                        crate::stage_context::prepare(py, callback, t, x).map(|s| s.rotation)
                    })
                    .transpose()
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.plan
            .components(t, [x[0], x[1], x[2]], &rotations)
            .map_err(PyValueError::new_err)
    }
    fn earth(
        &self,
        py: Python<'_>,
        base: &NativeForcePlanContext,
        callback: &Py<PyAny>,
        t: f64,
        x: [f64; 6],
        command: [f64; 3],
    ) -> PyResult<[f64; 3]> {
        let stage = crate::stage_context::prepare(py, callback, t, x)?;
        let mut a = base
            .plan
            .acceleration(x, &stage, command)
            .map_err(PyValueError::new_err)?;
        for part in self.components(py, t, x)? {
            for i in 0..3 {
                a[i] += part[i];
            }
        }
        Ok(a)
    }
}
#[pymethods]
impl NativeAdditionalGravity {
    #[new]
    fn new(
        py: Python<'_>,
        center: i64,
        tt: f64,
        duration: f64,
        ephemeris: EphemerisInput,
        sources: Vec<SourceInput>,
        frames: Vec<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        if sources.len() != frames.len()
            || frames.iter().flatten().any(|f| {
                !f.bind(py)
                    .is_instance_of::<crate::stage_context::NativeStageContext>()
            })
        {
            return Err(PyValueError::new_err(
                "additional frame contexts must be native and match sources",
            ));
        }
        Ok(Self {
            plan: AdditionalGravity::new(center, tt, duration, ephemeris, sources)
                .map_err(PyValueError::new_err)?,
            frames,
        })
    }
    fn stage(&self, py: Python<'_>, t: f64, x: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        Ok(self
            .components(py, t, fixed(x)?)?
            .iter()
            .map(|v| v.to_vec())
            .collect())
    }
    fn acceleration_earth(
        &self,
        py: Python<'_>,
        base: &NativeForcePlanContext,
        callback: Py<PyAny>,
        x: Vec<f64>,
        t: f64,
        command: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        Ok(self
            .earth(py, base, &callback, t, fixed(x)?, fixed(command)?)?
            .to_vec())
    }
    #[allow(clippy::too_many_arguments)]
    fn propagate_earth(
        &self,
        py: Python<'_>,
        base: &NativeForcePlanContext,
        callback: Py<PyAny>,
        x: Vec<f64>,
        t: f64,
        dt: f64,
        command: Vec<f64>,
        method: &str,
        atol: f64,
        rtol: f64,
        h: Option<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let command = fixed(command)?;
        integrate(fixed(x)?, t, dt, method, atol, rtol, h, |t, x| {
            let a = self.earth(py, base, &callback, t, x, command)?;
            Ok([x[3], x[4], x[5], a[0], a[1], a[2]])
        })
    }
    fn stage_planetary(
        &self,
        py: Python<'_>,
        base: &NativePlanetaryPlan,
        t: f64,
        x: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> {
        let x = fixed(x)?;
        let mut s = base
            .plan
            .stage(t, x)
            .map_err(crate::planetary_binding::plan_error)?;
        for part in self.components(py, t, x)? {
            for i in 0..3 {
                s.acceleration[i] += part[i];
            }
            s.components.push(part);
        }
        Ok((
            s.acceleration.to_vec(),
            s.diagnostics.to_vec(),
            s.components.iter().map(|v| v.to_vec()).collect(),
        ))
    }
    #[allow(clippy::too_many_arguments)]
    fn propagate_planetary(
        &self,
        py: Python<'_>,
        base: &NativePlanetaryPlan,
        x: Vec<f64>,
        t: f64,
        dt: f64,
        method: &str,
        atol: f64,
        rtol: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let method = if method == "adaptive" {
            "dopri5"
        } else {
            method
        };
        integrate(fixed(x)?, t, dt, method, atol, rtol, None, |t, x| {
            let mut a = base
                .plan
                .stage(t, x)
                .map_err(crate::planetary_binding::plan_error)?
                .acceleration;
            for part in self.components(py, t, x)? {
                for i in 0..3 {
                    a[i] += part[i];
                }
            }
            Ok([x[3], x[4], x[5], a[0], a[1], a[2]])
        })
    }
}
