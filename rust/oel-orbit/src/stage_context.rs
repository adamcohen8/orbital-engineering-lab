//! Built-in ONP stage assembly. Only time-dependent frames/ephemerides are
//! cached; density and atmosphere-relative velocity use every current stage.
use crate::{
    environment::{self, NativeEnvironmentContext},
    force_plan::StageInputs,
};
use pyo3::prelude::*;
use pyo3::{PyTraverseError, PyVisit};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

type FrameSpec = (u8, Option<f64>, Vec<f64>, Vec<Vec<f64>>, String);
const IDENTITY: [f64; 9] = [1., 0., 0., 0., 1., 0., 0., 0., 1.];

struct Cache<T> {
    rows: HashMap<u64, T>,
    order: VecDeque<u64>,
}
impl<T: Clone> Cache<T> {
    fn new() -> Self {
        Self {
            rows: HashMap::new(),
            order: VecDeque::new(),
        }
    }
    fn get(&self, t: f64) -> Option<T> {
        self.rows.get(&t.to_bits()).cloned()
    }
    fn put(&mut self, t: f64, value: T, limit: usize) {
        let key = t.to_bits();
        if !self.rows.contains_key(&key) {
            self.order.push_back(key);
        }
        self.rows.insert(key, value);
        while self.order.len() > limit {
            self.rows.remove(&self.order.pop_front().unwrap());
        }
    }
}

struct Frame {
    path: String,
    mode: u8,
    jd: Option<f64>,
    values: [f64; 7],
    eop: Vec<Vec<f64>>,
    dat_intervals: Vec<(f64, f64)>,
    rotations: Mutex<Cache<[f64; 9]>>,
    derivatives: Mutex<Cache<[f64; 9]>>,
}
fn error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}
impl Frame {
    fn new((mode, jd, values, eop, path): FrameSpec) -> PyResult<Self> {
        let values: [f64; 7] = values
            .try_into()
            .map_err(|_| error("frame scalars must have seven values"))?;
        if mode > 2 || jd.is_some_and(|x| !x.is_finite()) || values.iter().any(|x| !x.is_finite()) {
            return Err(error("invalid native stage frame"));
        }
        if !eop.is_empty()
            && (eop.len() != 5
                || eop[0].is_empty()
                || eop
                    .iter()
                    .any(|row| row.len() != eop[0].len() || row.iter().any(|x| !x.is_finite()))
                || eop[0].windows(2).any(|w| w[1] <= w[0]))
        {
            return Err(error("invalid native EOP table"));
        }
        let dat_intervals = if eop.is_empty() {
            Vec::new()
        } else {
            (1..eop[0].len())
                .filter(|&i| (eop[4][i] - eop[4][i - 1]).abs() > 1e-9)
                .map(|i| (eop[0][i - 1], eop[0][i]))
                .collect()
        };
        Ok(Self {
            path,
            mode,
            jd,
            values,
            eop,
            dat_intervals,
            rotations: Mutex::new(Cache::new()),
            derivatives: Mutex::new(Cache::new()),
        })
    }
    fn elapsed_guard(&self, t: f64) -> PyResult<()> {
        if self.dat_intervals.is_empty() || self.jd.is_none() || t == 0.0 {
            return Ok(());
        }
        let start = self.jd.unwrap() - 2400000.5;
        let end = start + t / 86400.;
        let (lo, hi) = (start.min(end), start.max(end));
        for &(_, right) in &self.dat_intervals {
            if lo < right && right <= hi {
                return Err(error(format!(
                    "EOP-backed UTC elapsed time crosses a DAT leap-second boundary near MJD {:.9}; the current fixed-86400-second UTC conversion is not leap-second aware.",
                    right
                )));
            }
        }
        Ok(())
    }
    fn interp(&self, jd: f64) -> PyResult<[f64; 4]> {
        if self.eop.is_empty() {
            return Ok([
                self.values[0],
                self.values[1],
                self.values[2],
                self.values[3],
            ]);
        }
        let x = jd - 2400000.5;
        let tt = &self.eop[0];
        let last = tt.len() - 1;
        if self.mode != 2 && (x < tt[0] || x > tt[last]) {
            return Err(error(format!(
                "Requested MJD {x:.9} is outside EOP coverage [{:.9}, {:.9}] for {}. Set eop_extrapolation='hold' only when endpoint holding is intentional.",
                tt[0], tt[last], self.path
            )));
        }
        if x <= tt[0] {
            return Ok(std::array::from_fn(|a| self.eop[a + 1][0]));
        }
        if x >= tt[last] {
            return Ok(std::array::from_fn(|a| self.eop[a + 1][last]));
        }
        let right = tt.partition_point(|value| *value <= x);
        let left = right - 1;
        if x == tt[left] {
            return Ok(std::array::from_fn(|a| self.eop[a + 1][left]));
        }
        if (self.eop[4][right] - self.eop[4][left]).abs() > 1e-9 {
            return Err(error(format!("EOP DAT changes within the requested interpolation bracket [{:.9}, {:.9}]; linear DAT interpolation is unsafe across a leap-second boundary.", tt[left], tt[right])));
        }
        Ok(std::array::from_fn(|a| {
            let yy = &self.eop[a + 1];
            let slope = (yy[right] - yy[left]) / (tt[right] - tt[left]);
            slope * (x - tt[left]) + yy[left]
        }))
    }
    fn rotation(
        &self,
        py: Python<'_>,
        native: Option<&Py<NativeEnvironmentContext>>,
        t: f64,
    ) -> PyResult<[f64; 9]> {
        if let Some(row) = self.rotations.lock().unwrap().get(t) {
            return Ok(row);
        }
        let row = if self.mode == 0 {
            environment::simple_rotation(t, self.jd, 7.2921159e-5).map_err(error)?
        } else {
            let jd = self
                .jd
                .ok_or_else(|| error("IAU76/80 frame requires an absolute epoch"))?;
            self.elapsed_guard(t)?;
            let [xp, yp, dut, dat] = self.interp(jd + t / 86400.)?;
            native
                .ok_or_else(|| error("native environment context is required"))?
                .borrow(py)
                .rotation_iau76_80(t, jd, xp, yp, dut, dat, self.values[4], self.values[5])?
                .try_into()
                .map_err(|_| error("invalid rotation shape"))?
        };
        self.rotations.lock().unwrap().put(t, row, 512);
        Ok(row)
    }
    fn derivative(
        &self,
        py: Python<'_>,
        native: Option<&Py<NativeEnvironmentContext>>,
        t: f64,
    ) -> PyResult<[f64; 9]> {
        if let Some(row) = self.derivatives.lock().unwrap().get(t) {
            return Ok(row);
        }
        let stencil = |offsets: &[f64], weights: &[f64]| -> PyResult<[f64; 9]> {
            let mut rows = Vec::new();
            for offset in offsets {
                rows.push(self.rotation(py, native, t + offset)?);
            }
            Ok(std::array::from_fn(|a| {
                let mut sum = weights[0] * rows[0][a];
                for i in 1..rows.len() {
                    sum += weights[i] * rows[i][a];
                }
                sum / 360.
            }))
        };
        let row = match stencil(&[60., 30., -30., -60.], &[-1., 8., -8., 1.]) {
            Ok(row) => row,
            Err(first) => match stencil(&[0., 30., 60., 90., 120.], &[-25., 48., -36., 16., -3.]) {
                Ok(row) => row,
                Err(_) => stencil(&[0., -30., -60., -90., -120.], &[25., -48., 36., -16., 3.])
                    .map_err(|_| first)?,
            },
        };
        self.derivatives.lock().unwrap().put(t, row, 512);
        Ok(row)
    }
}

// Match Python resource signatures (including inode and nanosecond ctime).
#[derive(PartialEq)]
struct Resource {
    path: String,
    signature: Option<(u64, u64, i128, i128)>,
}
fn signature(path: &str) -> Option<(u64, u64, i128, i128)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let m = std::fs::metadata(path).ok()?;
        Some((
            m.ino(),
            m.len(),
            m.mtime() as i128 * 1_000_000_000 + m.mtime_nsec() as i128,
            m.ctime() as i128 * 1_000_000_000 + m.ctime_nsec() as i128,
        ))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}
fn mv(m: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[3 * i] * v[0] + m[3 * i + 1] * v[1] + m[3 * i + 2] * v[2])
}
fn mtv(m: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[i] * v[0] + m[i + 3] * v[1] + m[i + 6] * v[2])
}

#[pyclass(name = "ONPStageContext")]
pub(crate) struct NativeStageContext {
    native: Option<Py<NativeEnvironmentContext>>,
    frames: Vec<Frame>,
    codes: Vec<u8>,
    density_mode: u8,
    density_values: [f64; 6],
    geodetic: bool,
    density_callback: Option<Py<PyAny>>,
    density_body_inputs: bool,
    density_frame_inputs: bool,
    nrlmsise00_inputs: Option<(Vec<f64>, Option<f64>, Option<Py<PyAny>>)>,
    body_callback: Option<Py<PyAny>>,
    body_mode: u8,
    jd: Option<f64>,
    fixed_jd: Option<f64>,
    dat: Option<f64>,
    resources: Vec<Resource>,
    fallback: Option<Py<PyAny>>,
    stale: AtomicBool,
    bodies: Mutex<Cache<([f64; 3], [f64; 3])>>,
}
#[pymethods]
impl NativeStageContext {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(native, frames, codes, density_mode, density_values, geodetic, density_callback, body_callback, body_mode, jd, fixed_jd, dat, resource_paths, fallback, density_body_inputs=false, density_frame_inputs=false, nrlmsise00_inputs=None))]
    fn new(
        native: Option<Py<NativeEnvironmentContext>>,
        frames: Vec<FrameSpec>,
        codes: Vec<u8>,
        density_mode: u8,
        density_values: Vec<f64>,
        geodetic: bool,
        density_callback: Option<Py<PyAny>>,
        body_callback: Option<Py<PyAny>>,
        body_mode: u8,
        jd: Option<f64>,
        fixed_jd: Option<f64>,
        dat: Option<f64>,
        resource_paths: Vec<String>,
        fallback: Py<PyAny>,
        density_body_inputs: bool,
        density_frame_inputs: bool,
        nrlmsise00_inputs: Option<(Vec<f64>, Option<f64>, Option<Py<PyAny>>)>,
    ) -> PyResult<Self> {
        if frames.len() != 5
            || density_mode > 2
            || body_mode > 4
            || (matches!(body_mode, 1..=3)
                && (native.is_none() || (jd.is_none() && fixed_jd.is_none())))
            || frames
                .iter()
                .any(|frame| frame.0 != 0 && (native.is_none() || frame.1.is_none()))
        {
            return Err(error("invalid stage context specification"));
        }
        if density_body_inputs && !codes.iter().any(|code| matches!(code, 3 | 4 | 5)) {
            return Err(error(
                "density body inputs require SRP or third-body forces",
            ));
        }
        if density_frame_inputs && !density_body_inputs {
            return Err(error("density frame inputs require prepared body inputs"));
        }
        let density_values: [f64; 6] = density_values
            .try_into()
            .map_err(|_| error("density scalars must have six values"))?;
        if density_values.iter().any(|x| !x.is_finite())
            || [jd, fixed_jd, dat]
                .into_iter()
                .flatten()
                .any(|x| !x.is_finite())
        {
            return Err(error("stage scalars must be finite"));
        }
        if let Some((drivers, lst, solar)) = &nrlmsise00_inputs {
            if density_mode != 2
                || drivers.len() != 10
                || drivers.iter().any(|x| !x.is_finite())
                || lst.is_some_and(|x| !x.is_finite())
                || solar.is_none()
                || (jd.is_none() && fixed_jd.is_none())
                || body_mode != 1
            {
                return Err(error("invalid explicit NRLMSISE stage inputs"));
            }
        }
        Ok(Self {
            native,
            frames: frames
                .into_iter()
                .map(Frame::new)
                .collect::<PyResult<_>>()?,
            codes,
            density_mode,
            density_values,
            geodetic,
            density_callback,
            density_body_inputs,
            density_frame_inputs,
            nrlmsise00_inputs,
            body_callback,
            body_mode,
            jd,
            fixed_jd,
            dat,
            resources: resource_paths
                .into_iter()
                .map(|path| Resource {
                    signature: signature(&path),
                    path,
                })
                .collect(),
            fallback: Some(fallback),
            stale: AtomicBool::new(false),
            bodies: Mutex::new(Cache::new()),
        })
    }
    fn __call__(&self, py: Python<'_>, t: f64, state: Vec<f64>) -> PyResult<Vec<f64>> {
        let state: [f64; 6] = state
            .try_into()
            .map_err(|_| error("stage state must have six values"))?;
        Ok(self.flat(py, t, state)?.to_vec())
    }
    #[staticmethod]
    fn supports_density_body_inputs() -> bool {
        true
    }
    #[staticmethod]
    fn supports_density_frame_inputs() -> bool {
        true
    }
    #[staticmethod]
    fn supports_nrlmsise00_inputs() -> bool {
        true
    }
    #[getter]
    fn native_nrlmsise00_inputs(&self) -> bool {
        self.nrlmsise00_inputs.is_some() && !self.using_fallback()
    }
    #[staticmethod]
    fn _nrlmsise00_packet(
        jd: f64,
        coordinates: [f64; 3],
        lst: f64,
        drivers: [f64; 10],
    ) -> PyResult<Vec<f64>> {
        crate::nrlmsise00_stage::packet(jd, coordinates, lst, &drivers)
            .map(|x| x.to_vec())
            .ok_or_else(|| error("unsupported NRLMSISE UTC calendar"))
    }
    fn using_fallback(&self) -> bool {
        self.stale.load(Ordering::Relaxed)
    }
    #[getter]
    fn native_preparation(&self) -> bool {
        true
    }
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        // Retain the reference adapter's private density-only compatibility
        // seam for older callers; new histories use this full native context.
        Ok(self
            .fallback
            .as_ref()
            .ok_or_else(|| error("cleared stage context"))?
            .bind(py)
            .getattr(name)?
            .unbind())
    }
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.native)?;
        visit.call(&self.fallback)?;
        visit.call(&self.density_callback)?;
        if let Some((_, _, solar)) = &self.nrlmsise00_inputs {
            visit.call(solar)?;
        }
        visit.call(&self.body_callback)
    }
    fn __clear__(&mut self) {
        self.native = None;
        self.fallback = None;
        self.density_callback = None;
        self.nrlmsise00_inputs = None;
        self.body_callback = None;
        self.stale.store(true, Ordering::Relaxed);
    }
}
impl NativeStageContext {
    fn flat(&self, py: Python<'_>, t: f64, state: [f64; 6]) -> PyResult<[f64; 19]> {
        if self.stale.load(Ordering::Relaxed)
            || self
                .resources
                .iter()
                .any(|r| r.signature.is_none() || signature(&r.path) != r.signature)
        {
            self.stale.store(true, Ordering::Relaxed);
            let row = self
                .fallback
                .as_ref()
                .ok_or_else(|| error("cleared stage context"))?
                .call1(py, (t, state))?;
            return crate::force_context::fixed_from_py(row.bind(py), "stage inputs");
        }
        if !t.is_finite() || state.iter().any(|x| !x.is_finite()) {
            return Err(error("stage state and time must be finite"));
        }
        let mut row = [0.; 19];
        row[..9].copy_from_slice(&IDENTITY);
        if self.codes.iter().any(|x| matches!(x, 3 | 4 | 5)) {
            let cached = self.bodies.lock().unwrap().get(t);
            let bodies = if let Some(bodies) = cached {
                bodies
            } else {
                let bodies = if self.body_mode == 1 {
                    if self.fixed_jd.is_none() {
                        self.frames[4].elapsed_guard(t)?;
                    }
                    let jd = self
                        .fixed_jd
                        .unwrap_or_else(|| self.jd.unwrap() + t / 86400.);
                    let dat = match self.dat {
                        Some(value) => value,
                        None => self.frames[3].interp(jd)?[3],
                    };
                    let mjd = jd - 2400000.5 + (32.184 + dat) / 86400.;
                    let tt = (mjd - 51544.5) / 36525.;
                    let correction = 0.001657 * (628.3076 * tt + 6.2401).sin()
                        + 0.000022 * (575.3385 * tt + 4.2970).sin()
                        + 0.000014 * (1256.6152 * tt + 6.1969).sin()
                        + 0.000005 * (606.9777 * tt + 4.0212).sin()
                        + 0.000005 * (52.9691 * tt + 0.4444).sin()
                        + 0.000002 * (21.3299 * tt + 5.5431).sin()
                        + 0.000010 * (628.3076 * tt + 4.2490).sin();
                    let tdb = mjd + correction / 86400. + 2400000.5;
                    let (sun, moon) = self
                        .native
                        .as_ref()
                        .ok_or_else(|| error("DE440 context required"))?
                        .borrow(py)
                        .de440_sun_moon_km(tdb, 81.3005682214972154)?;
                    (sun.try_into().unwrap(), moon.try_into().unwrap())
                } else if self.body_mode == 2 || self.body_mode == 3 {
                    if self.fixed_jd.is_none() {
                        self.frames[4].elapsed_guard(t)?;
                    }
                    let jd = self
                        .fixed_jd
                        .unwrap_or_else(|| self.jd.unwrap() + t / 86400.);
                    self.native
                        .as_ref()
                        .ok_or_else(|| error("analytic environment context required"))?
                        .borrow(py)
                        .analytic_bodies(jd, self.body_mode == 3)?
                } else if self.body_mode == 4 {
                    ([149597870.7, 0., 0.], [384400., 0., 0.])
                } else {
                    self.body_callback
                        .as_ref()
                        .ok_or_else(|| error("body callback required"))?
                        .call1(py, (t,))?
                        .extract(py)?
                };
                self.bodies.lock().unwrap().put(t, bodies, 8);
                bodies
            };
            row[13..16].copy_from_slice(&bodies.0);
            row[16..19].copy_from_slice(&bodies.1);
        }
        if self.codes.contains(&1) {
            row[..9].copy_from_slice(&self.frames[0].rotation(py, self.native.as_ref(), t)?);
        }
        let position = [state[0], state[1], state[2]];
        if self.codes.contains(&2) {
            row[9] = match self.density_mode {
                0 => self.density_values[0],
                1 => environment::radial_exponential_density(
                    position,
                    6378.137,
                    self.density_values[1],
                    self.density_values[2],
                    self.density_values[3],
                    self.density_values[4],
                )
                .map_err(error)?,
                _ => {
                    let rot = self.frames[2].rotation(py, self.native.as_ref(), t)?;
                    let fixed = mv(&rot, position);
                    let coords = if self.geodetic {
                        let [lat, lon, alt] =
                            environment::ecef_to_geodetic_deg_km(fixed).map_err(error)?;
                        [alt.max(0.), lat, lon]
                    } else {
                        let radius =
                            (fixed[0] * fixed[0] + fixed[1] * fixed[1] + fixed[2] * fixed[2])
                                .sqrt();
                        [
                            (radius - 6378.137).max(0.),
                            if radius > 0. {
                                (fixed[2] / radius).clamp(-1., 1.).asin().to_degrees()
                            } else {
                                0.
                            },
                            fixed[1].atan2(fixed[0]).to_degrees(),
                        ]
                    };
                    let callback = self
                        .density_callback
                        .as_ref()
                        .ok_or_else(|| error("density callback required"))?;
                    let prepared_density = if let Some((drivers, _lst, solar)) =
                        &self.nrlmsise00_inputs
                    {
                        // The compatibility callback observes replaced owners and
                        // mutable Ap histories before any snapshot is consumed.
                        let lst: Option<f64> = solar
                            .as_ref()
                            .unwrap()
                            .call1(py, (t, coords[2], [row[13], row[14], row[15]], rot))?
                            .extract(py)?;
                        match lst {
                            Some(lst) => {
                                let jd = self
                                    .fixed_jd
                                    .unwrap_or_else(|| self.jd.unwrap() + t / 86400.);
                                let drivers: &[f64; 10] = drivers.as_slice().try_into().unwrap();
                                match crate::nrlmsise00_stage::packet(jd, coords, lst, drivers) {
                                    Some(values) => Some(
                                        crate::nrlmsise00_native::density(values).map_err(error)?,
                                    ),
                                    None => None,
                                }
                            }
                            None => None,
                        }
                    } else {
                        None
                    };
                    let density = if let Some(value) = prepared_density {
                        value
                    } else if self.density_body_inputs {
                        // Reuse this exact stage's resource-bound ephemerides.
                        // Density and state-derived coordinates are never cached.
                        let bodies = ([row[13], row[14], row[15]], [row[16], row[17], row[18]]);
                        if self.density_frame_inputs {
                            // Share the exact guarded density-frame matrix;
                            // retain the Python solar projection arithmetic.
                            callback
                                .call1(py, (t, state, coords, bodies, rot))?
                                .extract::<f64>(py)?
                        } else {
                            callback
                                .call1(py, (t, state, coords, bodies))?
                                .extract::<f64>(py)?
                        }
                    } else {
                        callback.call1(py, (t, state, coords))?.extract::<f64>(py)?
                    };
                    density
                }
            };
            let relative = if self.frames[1].mode == 0 {
                environment::simple_atmosphere_relative_velocity(
                    state,
                    t,
                    self.frames[1].jd,
                    self.density_values[5],
                )
                .map_err(error)?
            } else {
                let rot = self.frames[1].rotation(py, self.native.as_ref(), t)?;
                let dr = self.frames[1].derivative(py, self.native.as_ref(), t)?;
                let vel = mv(&rot, [state[3], state[4], state[5]]);
                let rate = mv(&dr, position);
                mtv(&rot, std::array::from_fn(|a| vel[a] + rate[a]))
            };
            row[10..13].copy_from_slice(&relative);
        }
        Ok(row)
    }
    pub(crate) fn prepare(&self, py: Python<'_>, t: f64, state: [f64; 6]) -> PyResult<StageInputs> {
        StageInputs::from_flat(self.flat(py, t, state)?).map_err(error)
    }
}

pub(crate) fn prepare(
    py: Python<'_>,
    callback: &Py<PyAny>,
    t: f64,
    state: [f64; 6],
) -> PyResult<StageInputs> {
    if callback.bind(py).is_instance_of::<NativeStageContext>() {
        let native = callback
            .bind(py)
            .extract::<PyRef<'_, NativeStageContext>>()?;
        native.prepare(py, t, state)
    } else {
        let values = callback.call1(py, (t, state))?;
        if values.bind(py).len()? == 19 {
            StageInputs::from_flat(crate::force_context::fixed_from_py(
                values.bind(py),
                "stage inputs",
            )?)
            .map_err(error)
        } else {
            StageInputs::from_values(values.extract::<Vec<f64>>(py)?).map_err(error)
        }
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeStageContext>()
}
