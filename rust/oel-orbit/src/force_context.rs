//! Persistent Python-facing ONP force-plan context.
//!
//! Force coefficients are parsed once and retained in native memory.  Stage
//! environments use a native built-in context or the reference Python callback.
//! Resource and custom-plugin policy remains authoritative in the adapter.

#[cfg(feature = "python")]
use crate::{
    force_plan::{ForcePlan, StageInputs},
    harmonics::HarmonicTerms,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
fn fixed<const N: usize>(values: Vec<f64>, label: &str) -> PyResult<[f64; N]> {
    values.try_into().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!("{label} must have {N} values"))
    })
}

#[cfg(feature = "python")]
pub(crate) fn fixed_from_py<const N: usize>(
    values: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<[f64; N]> {
    fixed_from_py_with_error(values, || format!("{label} must have {N} values"))
}

#[cfg(feature = "python")]
pub(crate) fn fixed_from_py_with_error<const N: usize, F: FnOnce() -> String>(
    values: &Bound<'_, PyAny>,
    length_error: F,
) -> PyResult<[f64; N]> {
    // Exact built-in containers need no generic Python indexing protocol.
    // Subclasses/custom sequences retain their original __getitem__ path.
    if values.is_exact_instance_of::<pyo3::types::PyList>() {
        use pyo3::types::PyListMethods;
        let values = values.cast::<pyo3::types::PyList>()?;
        if values.len() == N {
            let mut out = [0.0; N];
            for (index, value) in out.iter_mut().enumerate() {
                let item = values.get_item(index)?;
                // Coercion hooks may mutate the container. Retain the original
                // sequence iterator whenever an item could run Python code.
                if !item.is_exact_instance_of::<pyo3::types::PyFloat>()
                    && !item.is_exact_instance_of::<pyo3::types::PyInt>()
                    && !item.is_exact_instance_of::<pyo3::types::PyBool>()
                {
                    return fixed_vec_with_error(values.as_any().extract()?, length_error);
                }
                *value = item.extract()?;
            }
            return Ok(out);
        }
    }
    if values.is_exact_instance_of::<pyo3::types::PyTuple>() {
        use pyo3::types::PyTupleMethods;
        let values = values.cast::<pyo3::types::PyTuple>()?;
        if values.len() == N {
            let mut out = [0.0; N];
            for (index, value) in out.iter_mut().enumerate() {
                *value = values.get_item(index)?.extract()?;
            }
            return Ok(out);
        }
    }
    fixed_vec_with_error(values.extract()?, length_error)
}

#[cfg(feature = "python")]
fn fixed_vec_with_error<const N: usize, F: FnOnce() -> String>(
    values: Vec<f64>,
    error: F,
) -> PyResult<[f64; N]> {
    values
        .try_into()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(error()))
}

#[cfg(feature = "python")]
pub(crate) fn plan_from_inputs(
    codes: Vec<u8>,
    scalars: Vec<f64>,
    shadow_model: u8,
    harmonic_dims: (usize, usize),
    tables: Vec<Vec<f64>>,
) -> PyResult<ForcePlan> {
    plan_with_precision(
        codes,
        scalars,
        shadow_model,
        harmonic_dims,
        tables,
        Vec::new(),
    )
}

#[cfg(feature = "python")]
fn plan_with_precision(
    codes: Vec<u8>,
    scalars: Vec<f64>,
    shadow_model: u8,
    harmonic_dims: (usize, usize),
    tables: Vec<Vec<f64>>,
    precision_specs: Vec<crate::precision_plan::Specification>,
) -> PyResult<ForcePlan> {
    let harmonics = if tables.is_empty() {
        None
    } else {
        if tables.len() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "harmonic tables must have seven arrays",
            ));
        }
        let mut values = tables.into_iter();
        Some(HarmonicTerms {
            n_max: harmonic_dims.0,
            m_max: harmonic_dims.1,
            c_nm: values.next().unwrap(),
            s_nm: values.next().unwrap(),
            diag: values.next().unwrap(),
            subdiag: values.next().unwrap(),
            recur_a: values.next().unwrap(),
            recur_b: values.next().unwrap(),
            recur_c: values.next().unwrap(),
        })
    };
    let plan = ForcePlan {
        codes,
        scalars: fixed(scalars, "force-plan scalars")?,
        shadow_model,
        harmonics,
        precision: precision_specs
            .into_iter()
            .map(crate::precision_plan::PrecisionForce::new)
            .collect::<Result<_, _>>()
            .map_err(pyo3::exceptions::PyValueError::new_err)?,
    };
    plan.validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(plan)
}

#[cfg(feature = "python")]
fn info_tuple(
    info: crate::adaptive::StepInfo,
) -> (
    usize,
    usize,
    usize,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    f64,
) {
    (
        info.accepted_steps,
        info.rejected_steps,
        info.attempted_steps,
        info.min_step_s,
        info.max_step_s,
        info.final_step_s,
        info.suggested_next_step_s,
        info.max_error_ratio,
    )
}

#[cfg(feature = "python")]
#[pyclass(name = "ONPForceContext")]
pub struct NativeForcePlanContext {
    pub(crate) plan: ForcePlan,
}

#[cfg(feature = "python")]
#[pymethods]
impl NativeForcePlanContext {
    #[new]
    #[pyo3(signature = (codes, scalars, shadow_model, harmonic_dims, tables, precision_specs=None))]
    fn new(
        codes: Vec<u8>,
        scalars: Vec<f64>,
        shadow_model: u8,
        harmonic_dims: (usize, usize),
        tables: Vec<Vec<f64>>,
        precision_specs: Option<Vec<crate::precision_plan::Specification>>,
    ) -> PyResult<Self> {
        Ok(Self {
            plan: plan_with_precision(
                codes,
                scalars,
                shadow_model,
                harmonic_dims,
                tables,
                precision_specs.unwrap_or_default(),
            )?,
        })
    }

    fn supports_simple_drag_stages(&self) -> bool {
        true
    }

    fn acceleration(
        &self,
        py: Python<'_>,
        state: Vec<f64>,
        stage_inputs: Vec<f64>,
        command: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let state = crate::State::from_array(fixed(state, "state")?)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let stage = StageInputs::from_values(stage_inputs)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let command: [f64; 3] = fixed(command, "command acceleration")?;
        if !command.iter().all(|value| value.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "command acceleration must be finite",
            ));
        }
        let acceleration = py.detach(|| self.plan.acceleration(state.as_array(), &stage, command));
        Ok(acceleration.map_err(pyo3::exceptions::PyValueError::new_err)?.to_vec())
    }

    #[allow(clippy::too_many_arguments)]
    fn adaptive(
        &self,
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        command: Vec<f64>,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
        method: String,
        stage_callback: Py<PyAny>,
    ) -> PyResult<(
        Vec<f64>,
        (
            usize,
            usize,
            usize,
            Option<f64>,
            Option<f64>,
            Option<f64>,
            Option<f64>,
            f64,
        ),
    )> {
        let state = crate::State::from_array(fixed(state, "state")?)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let command = fixed(command, "command acceleration")?;
        if !command.iter().all(|value| value.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "command acceleration must be finite",
            ));
        }
        let mut callback_error: Option<PyErr> = None;
        let derivative = |stage_time: f64, stage_state: [f64; 6]| -> Result<[f64; 6], String> {
            let stage =
                match crate::stage_context::prepare(py, &stage_callback, stage_time, stage_state) {
                    Ok(stage) => stage,
                    Err(error) => {
                        callback_error = Some(error);
                        return Err("stage input callback failed".to_owned());
                    }
                };
            self.plan.derivative(stage_state, &stage, command)
        };
        let result = match method.as_str() {
            "rkf78" => crate::adaptive::integrate_rkf78(
                state.as_array(),
                t_s,
                dt_s,
                atol,
                rtol,
                h_init,
                derivative,
            ),
            "dopri5" => crate::adaptive::integrate_dopri5(
                state.as_array(),
                t_s,
                dt_s,
                atol,
                rtol,
                h_init,
                derivative,
            ),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "invalid force-plan integrator",
                ))
            }
        };
        if let Some(error) = callback_error {
            return Err(error);
        }
        let (next, info) = result.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok((next.to_vec(), info_tuple(info)))
    }

    /// Sample an explicit adaptive schedule using the same adaptive owner as
    /// scalar stepping. Diagnostics remain per interval for lifecycle replay.
    #[allow(clippy::too_many_arguments)]
    fn sampled_adaptive_history(
        &self,
        py: Python<'_>,
        state: Vec<f64>,
        step_times: &[u8],
        step_widths: &[u8],
        sample_ends: Vec<usize>,
        command: Vec<f64>,
        atol: f64,
        rtol: f64,
        h_init: Option<f64>,
        method: String,
        stage_callback: Py<PyAny>,
    ) -> PyResult<(Py<pyo3::types::PyBytes>, Py<pyo3::types::PyBytes>)> {
        use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
        use pyo3::exceptions::PyValueError;
        let times = decode_f64_bytes(step_times)?;
        let widths = decode_f64_bytes(step_widths)?;
        if times.len() != widths.len()
            || widths.len() > 10_000_000
            || times.iter().zip(widths.iter()).any(|(&t, &h)| {
                !t.is_finite() || !h.is_finite() || h <= 0.0 || !(t + h).is_finite()
            })
            || sample_ends.first().is_some_and(|n| *n == 0)
            || sample_ends.windows(2).any(|p| p[0] >= p[1])
            || sample_ends.last().is_some_and(|n| *n != widths.len())
            || !matches!(method.as_str(), "rkf78" | "dopri5")
        {
            return Err(PyValueError::new_err(
                "invalid adaptive sampled step sequence",
            ));
        }
        let mut current = crate::State::from_array(fixed(state, "state")?)
            .map_err(PyValueError::new_err)?
            .as_array()
            .to_vec();
        let command = fixed::<3>(command, "command acceleration")?.to_vec();
        if command.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("command acceleration must be finite"));
        }
        let mut states = Vec::with_capacity(6 * (sample_ends.len() + 1));
        let mut infos = Vec::with_capacity(8 * widths.len());
        states.extend(&current);
        let mut next_h = h_init;
        let mut sample = 0;
        for (index, (&time, &width)) in times.iter().zip(widths.iter()).enumerate() {
            let (next, info) = self.adaptive(
                py,
                current,
                time,
                width,
                command.clone(),
                atol,
                rtol,
                next_h,
                method.clone(),
                stage_callback.clone_ref(py),
            )?;
            current = next;
            next_h = info.6;
            infos.extend([
                info.0 as f64,
                info.1 as f64,
                info.2 as f64,
                info.3.unwrap_or(f64::NAN),
                info.4.unwrap_or(f64::NAN),
                info.5.unwrap_or(f64::NAN),
                info.6.unwrap_or(f64::NAN),
                info.7,
            ]);
            if sample < sample_ends.len() && sample_ends[sample] == index + 1 {
                states.extend(&current);
                sample += 1;
            }
        }
        Ok((output_bytes(py, &states)?, output_bytes(py, &infos)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn rk4(
        &self,
        py: Python<'_>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        command: Vec<f64>,
        stage_callback: Py<PyAny>,
    ) -> PyResult<Vec<f64>> {
        self.rk4_state(
            py,
            fixed(state, "state")?,
            t_s,
            dt_s,
            fixed(command, "command acceleration")?,
            &stage_callback,
        )
        .map(|state| state.to_vec())
    }

    /// Advance a fixed-step segment without Python per-step dispatch. A zero
    /// sample stride retains only the final state; otherwise retain the initial
    /// state, requested samples and the final state. Each RK4 stage uses its
    /// current state and time through native preparation or the callback.
    #[allow(clippy::too_many_arguments)]
    fn fixed_history<'py>(
        &self,
        py: Python<'py>,
        state: Vec<f64>,
        t_s: f64,
        dt_s: f64,
        steps: usize,
        sample_stride: usize,
        command: Vec<f64>,
        stage_callback: Py<PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        if steps > 10_000_000
            || !dt_s.is_finite()
            || dt_s <= 0.0
            || !t_s.is_finite()
            || !(t_s + dt_s * steps as f64).is_finite()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid fixed-step segment",
            ));
        }
        let mut current = crate::State::from_array(fixed(state, "state")?)
            .map_err(pyo3::exceptions::PyValueError::new_err)?
            .as_array();
        let command = fixed(command, "command acceleration")?;
        let mut bytes = Vec::new();
        let mut record = |values: &[f64]| {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        };
        if sample_stride > 0 {
            record(&current);
        }
        for index in 0..steps {
            current = self.rk4_state(
                py,
                current,
                t_s + index as f64 * dt_s,
                dt_s,
                command,
                &stage_callback,
            )?;
            if sample_stride > 0 && (index + 1) % sample_stride == 0 {
                record(&current);
            }
        }
        if sample_stride == 0 || steps % sample_stride != 0 {
            record(&current);
        }
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Visit an explicit RK4 schedule and retain its exact sample boundaries.
    /// Per-step epochs are supplied separately from widths so decimal nested
    /// simulation clocks retain the Python owner's addition/reset semantics.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(state, step_times, step_widths, sample_ends, command, stage_callback, simple_drag_mode=false, simple_drag_jd_utc_start=None))]
    fn sampled_history<'py>(
        &self,
        py: Python<'py>,
        state: Vec<f64>,
        step_times: &[u8],
        step_widths: &[u8],
        sample_ends: Vec<usize>,
        command: Vec<f64>,
        stage_callback: Py<PyAny>,
        simple_drag_mode: bool,
        simple_drag_jd_utc_start: Option<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        use pyo3::exceptions::PyValueError;
        if simple_drag_mode
            && self
                .plan
                .codes
                .iter()
                .any(|code| !matches!(*code, 2 | 6 | 7 | 8))
        {
            return Err(PyValueError::new_err(
                "simple drag stages support only drag and zonal forces",
            ));
        }
        if step_widths.len() % 8 != 0
            || step_times.len() != step_widths.len()
            || step_widths.len() / 8 > 10_000_000
        {
            return Err(PyValueError::new_err(
                "step times and widths must contain matching bounded float64 values",
            ));
        }
        let times: Vec<f64> = step_times
            .chunks_exact(8)
            .map(|value| f64::from_le_bytes(value.try_into().unwrap()))
            .collect();
        let widths: Vec<f64> = step_widths
            .chunks_exact(8)
            .map(|value| f64::from_le_bytes(value.try_into().unwrap()))
            .collect();
        if times
            .iter()
            .zip(&widths)
            .any(|(&t, &h)| !t.is_finite() || !h.is_finite() || h <= 0.0 || !(t + h).is_finite())
        {
            return Err(PyValueError::new_err(
                "sampled step epochs must be finite and widths positive",
            ));
        }
        if sample_ends.first().is_some_and(|index| *index == 0)
            || sample_ends.windows(2).any(|pair| pair[0] >= pair[1])
            || sample_ends
                .last()
                .is_some_and(|index| *index > widths.len())
        {
            return Err(PyValueError::new_err(
                "sample ends must be strictly increasing within the step sequence",
            ));
        }
        let mut current = crate::State::from_array(fixed(state, "state")?)
            .map_err(PyValueError::new_err)?
            .as_array();
        let command = fixed(command, "command acceleration")?;
        if command.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("command acceleration must be finite"));
        }
        let capacity = sample_ends
            .len()
            .checked_add(1)
            .and_then(|n| n.checked_mul(48))
            .ok_or_else(|| PyValueError::new_err("sampled history size overflow"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| PyValueError::new_err("unable to allocate sampled history"))?;
        for value in current {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut sample = 0;
        for (index, (&time, &width)) in times.iter().zip(&widths).enumerate() {
            current = if simple_drag_mode {
                self.rk4_density_state(
                    py,
                    current,
                    time,
                    width,
                    command,
                    &stage_callback,
                    simple_drag_jd_utc_start,
                )?
            } else {
                self.rk4_state(py, current, time, width, command, &stage_callback)?
            };
            if sample < sample_ends.len() && sample_ends[sample] == index + 1 {
                for value in current {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                sample += 1;
            }
        }
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }
}

#[cfg(feature = "python")]
impl NativeForcePlanContext {
    pub(crate) fn plan(&self) -> &ForcePlan {
        &self.plan
    }

    pub(crate) fn rk4_state(
        &self,
        py: Python<'_>,
        state: [f64; 6],
        t_s: f64,
        dt_s: f64,
        command: [f64; 3],
        stage_callback: &Py<PyAny>,
    ) -> PyResult<[f64; 6]> {
        if !dt_s.is_finite() || dt_s < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t_s and non-negative dt_s must be finite",
            ));
        }
        self.rk4_signed_state(py, state, t_s, dt_s, command, stage_callback)
    }

    /// The SLR light-time owner may query either side of a reference epoch.
    /// Ordinary forward-history admission remains in `rk4_state`.
    pub(crate) fn rk4_signed_state(
        &self, py: Python<'_>, state: [f64; 6], t_s: f64, dt_s: f64,
        command: [f64; 3], stage_callback: &Py<PyAny>,
    ) -> PyResult<[f64; 6]> {
        self.rk4_state_with_stage(state, t_s, dt_s, command, |time, stage_state| {
            crate::stage_context::prepare(py, stage_callback, time, stage_state)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn rk4_density_state(
        &self,
        py: Python<'_>,
        state: [f64; 6],
        t_s: f64,
        dt_s: f64,
        command: [f64; 3],
        density_callback: &Py<PyAny>,
        jd_utc_start: Option<f64>,
    ) -> PyResult<[f64; 6]> {
        self.rk4_state_with_stage(state, t_s, dt_s, command, |time, stage_state| {
            let density: f64 = density_callback
                .call1(py, (time, stage_state))?
                .extract(py)?;
            let relative_velocity = crate::environment::simple_atmosphere_relative_velocity(
                stage_state,
                time,
                jd_utc_start,
                self.plan.scalars[5],
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let mut values = [0.0; 19];
            values[0] = 1.0;
            values[4] = 1.0;
            values[8] = 1.0;
            values[9] = density;
            values[10..13].copy_from_slice(&relative_velocity);
            StageInputs::from_flat(values).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    pub(crate) fn rk4_state_with_stage<F>(
        &self,
        state: [f64; 6],
        t_s: f64,
        dt_s: f64,
        command: [f64; 3],
        mut prepare_stage: F,
    ) -> PyResult<[f64; 6]>
    where
        F: FnMut(f64, [f64; 6]) -> PyResult<StageInputs>,
    {
        let state =
            crate::State::from_array(state).map_err(pyo3::exceptions::PyValueError::new_err)?;
        if !t_s.is_finite() || !dt_s.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t_s and non-negative dt_s must be finite",
            ));
        }
        if !command.iter().all(|value| value.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "command acceleration must be finite",
            ));
        }
        let mut derivative = |time: f64, stage_state: [f64; 6]| -> PyResult<[f64; 6]> {
            let stage = prepare_stage(time, stage_state)?;
            self.plan
                .derivative(stage_state, &stage, command)
                .map_err(pyo3::exceptions::PyValueError::new_err)
        };
        let x = state.as_array();
        let k1 = derivative(t_s, x)?;
        let k2 = derivative(t_s + 0.5 * dt_s, crate::stage(x, k1, 0.5 * dt_s))?;
        let k3 = derivative(t_s + 0.5 * dt_s, crate::stage(x, k2, 0.5 * dt_s))?;
        let k4 = derivative(t_s + dt_s, crate::stage(x, k3, dt_s))?;
        let next = std::array::from_fn(|axis| {
            x[axis] + (dt_s / 6.0) * (k1[axis] + 2.0 * k2[axis] + 2.0 * k3[axis] + k4[axis])
        });
        Ok(crate::State::from_array(next)
            .map_err(pyo3::exceptions::PyValueError::new_err)?
            .as_array())
    }
}

#[cfg(feature = "python")]
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeForcePlanContext>()?;
    module.add_function(wrap_pyfunction!(coast_forecast_batch, module)?)?;
    Ok(())
}

/// Object-major, deterministic coasts. Each row owns its force plan and exact
/// configured substep widths. Errors are returned per completed row so callers
/// retain the same earlier cache entries as ordinary serial stepping.
#[cfg(feature = "python")]
#[pyfunction]
fn coast_forecast_batch(
    py: Python<'_>,
    contexts: Vec<Py<NativeForcePlanContext>>,
    stages: Vec<Py<PyAny>>,
    states: Vec<Vec<f64>>,
    starts: Vec<f64>,
    widths: Vec<Vec<f64>>,
) -> PyResult<(Vec<Vec<f64>>, Option<Py<PyAny>>)> {
    let n = contexts.len();
    if stages.len() != n || states.len() != n || starts.len() != n || widths.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "forecast batch dimensions must match",
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let result = (|| -> PyResult<[f64; 6]> {
            let context = contexts[i].borrow(py);
            let mut state = fixed(states[i].clone(), "forecast state")?;
            let mut time = starts[i];
            let plan = &context.plan;
            let zonal = plan.codes.iter().all(|code| matches!(code, 6 | 7 | 8));
            for &width in &widths[i] {
                state = if zonal {
                    let input = crate::State::from_array(state)
                        .map_err(pyo3::exceptions::PyValueError::new_err)?;
                    if plan.codes.is_empty() || plan.codes == [6] {
                        crate::rk4_step(
                            input,
                            width,
                            crate::OrbitParameters {
                                mu_km3_s2: plan.scalars[0],
                                include_j2: !plan.codes.is_empty(),
                                command_accel_eci_km_s2: [0.; 3],
                            },
                        )
                    } else {
                        let codes: Vec<u8> = plan.codes.iter().map(|code| code - 4).collect();
                        crate::rk4_zonal_step(input, width, plan.scalars[0], [0.; 3], &codes)
                    }
                    .map_err(pyo3::exceptions::PyValueError::new_err)?
                    .as_array()
                } else {
                    context.rk4_state(py, state, time, width, [0.; 3], &stages[i])?
                };
                time += width;
            }
            Ok(state)
        })();
        match result {
            Ok(state) => out.push(state.to_vec()),
            Err(error) => return Ok((out, Some(error.into_value(py).into_any()))),
        }
    }
    Ok((out, None))
}
