//! SLR observation batches. The existing force plan owns all dynamics;
//! Python owns resources, frames, CRD normalization, fitting policy and gates.
use crate::force_context::NativeForcePlanContext;
use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use std::collections::HashMap;

const C: f64 = 299792.458;
const DAY: f64 = 86400.;
const ROW: usize = 22;
const OUTPUT: usize = 10;
fn error(s: impl Into<String>) -> PyErr {
    PyValueError::new_err(s.into())
}
fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn rotate(m: &[f64; 9], v: [f64; 3], transpose: bool) -> [f64; 3] {
    std::array::from_fn(|i| {
        (0..3)
            .map(|j| m[if transpose { j * 3 + i } else { i * 3 + j }] * v[j])
            .sum()
    })
}

#[derive(Clone)]
struct Observation {
    time: f64,
    jd: f64,
    event: u8,
    station: [f64; 3],
    enu: [f64; 9],
    media: Option<Media>,
}
#[derive(Clone)]
struct Media {
    zenith: f64,
    a: f64,
    b: f64,
    c: f64,
}
impl Media {
    fn new(lat: f64, height: f64, p: f64, t: f64, rh: f64, wavelength: f64) -> PyResult<Self> {
        if !(-std::f64::consts::FRAC_PI_2..=std::f64::consts::FRAC_PI_2).contains(&lat)
            || !(-500. ..=6000.).contains(&height)
            || !(200. ..=1100.).contains(&p)
            || !(200. ..=330.).contains(&t)
            || !(0. ..=100.).contains(&rh)
            || !(355. ..=1064.).contains(&wavelength)
        {
            return Err(error(
                "Optical troposphere input is outside the supported physical envelope.",
            ));
        }
        let s = (1000. / wavelength).powi(2);
        let hydro_disp = 0.01
            * 0.99995995
            * (19990.975 * (238.0185 + s) / (238.0185 - s).powi(2)
                + 579.55174 * (57.362 + s) / (57.362 - s).powi(2));
        let wet_disp = 0.003101
            * (295.235 + 3. * 2.6422 * s - 5. * 0.032380 * s.powi(2) + 7. * 0.004028 * s.powi(3));
        let tc = t - 273.15;
        let vapor = (rh / 100.)
            * (1.00062 + 3.14e-6 * p + 5.6e-7 * tc.powi(2))
            * (t * (t * 1.2378847e-5 - 1.9121316e-2) + 33.93711047 - 6343.1645 / t).exp();
        let site = 1. - 0.00266 * (2. * lat).cos() - 2.8e-7 * height;
        let zenith = 0.002416579 * p * hydro_disp / site
            + 1.0e-6 * (5.316 * wet_disp - 3.759 * hydro_disp) * vapor / site;
        let coeffs = [
            [12100.8e-7, 1729.5e-9, 319.1e-7, -1847.8e-11],
            [30496.5e-7, 234.4e-8, -103.5e-6, -185.6e-10],
            [6877.7e-5, 197.2e-7, -345.8e-5, 106.0e-9],
        ];
        let x = [1., tc, lat.cos(), height];
        let [a, b, c] = coeffs.map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum());
        Ok(Self { zenith, a, b, c })
    }
    fn delay(&self, elevation: f64) -> PyResult<f64> {
        if !(3_f64.to_radians()..=std::f64::consts::FRAC_PI_2 + 1.0e-12).contains(&elevation) {
            return Err(error(
                "Optical troposphere requires geometric elevation in [3, 90] degrees.",
            ));
        }
        let sine = elevation.sin();
        Ok(self.zenith * (1. + self.a / (1. + self.b / (1. + self.c)))
            / (sine + self.a / (sine + self.b / (sine + self.c))))
    }
}

type Frames = HashMap<(u64, u64), [f64; 9]>;
fn rotation(
    py: Python<'_>,
    callback: &Py<PyAny>,
    cache: &mut Frames,
    jd: f64,
    midnight: f64,
) -> PyResult<[f64; 9]> {
    let key = (jd.to_bits(), midnight.to_bits());
    if let Some(m) = cache.get(&key) {
        return Ok(*m);
    }
    let m: Vec<f64> = callback.call1(py, (jd, midnight))?.extract(py)?;
    let m: [f64; 9] = m
        .try_into()
        .map_err(|_| error("SLR frame must contain nine values"))?;
    if m.iter().any(|x| !x.is_finite()) {
        return Err(error("SLR frame must be finite"));
    }
    cache.insert(key, m);
    Ok(m)
}

struct Geometry<'a, 'py> {
    py: Python<'py>,
    obs: &'a Observation,
    relative: bool,
    shapiro_mu: f64,
    state: [f64; 6],
    force: &'a NativeForcePlanContext,
    stages: &'a Py<PyAny>,
    frame: &'a Py<PyAny>,
    frames: &'a mut Frames,
    satellites: HashMap<u64, [f64; 3]>,
    stations: HashMap<u64, [f64; 3]>,
}
impl Geometry<'_, '_> {
    fn satellite(&mut self, jd: f64) -> PyResult<[f64; 3]> {
        if let Some(x) = self.satellites.get(&jd.to_bits()) {
            return Ok(*x);
        }
        let next = self.force.rk4_signed_state(
            self.py,
            self.state,
            self.obs.time,
            if self.relative {
                jd
            } else {
                (jd - self.obs.jd) * DAY
            },
            [0.; 3],
            self.stages,
        )?;
        let value = [next[0], next[1], next[2]];
        self.satellites.insert(jd.to_bits(), value);
        Ok(value)
    }
    fn station(&mut self, jd: f64) -> PyResult<[f64; 3]> {
        if let Some(x) = self.stations.get(&jd.to_bits()) {
            return Ok(*x);
        }
        let midnight = (self.obs.jd - 0.5).floor() + 0.5;
        let rot = if self.relative {
            rotation(self.py, self.frame, self.frames, self.obs.jd, jd)?
        } else {
            rotation(self.py, self.frame, self.frames, jd, midnight)?
        };
        let value = rotate(&rot, self.obs.station, true);
        self.stations.insert(jd.to_bits(), value);
        Ok(value)
    }
    fn distance(&mut self, sat_jd: f64, station_jd: f64) -> PyResult<f64> {
        let sat = self.satellite(sat_jd)?;
        let station = self.station(station_jd)?;
        let mut range = norm(sub(sat, station));
        if self.shapiro_mu > 0. {
            range += self.shapiro_delay(sat, station)?;
        }
        Ok(range / C)
    }
    fn shapiro_delay(&self, sat: [f64; 3], station: [f64; 3]) -> PyResult<f64> {
        let distance = norm(sub(sat, station));
        let denominator = norm(sat) + norm(station) - distance;
        if denominator <= 0. || norm(sat) <= 0. || norm(station) <= 0. {
            return Err(error("Earth Shapiro path has singular geocentric geometry"));
        }
        Ok(2. * self.shapiro_mu / (C * C) * (2. * distance / denominator).ln_1p())
    }
    fn predict(&mut self, bias: f64) -> PyResult<[f64; OUTPUT]> {
        let (tx, bounce, rx, out, inn);
        let epoch = if self.relative { 0. } else { self.obs.jd };
        let unit = if self.relative { 1. } else { DAY };
        let mut converged;
        if self.obs.event == 2 {
            tx = epoch;
            let mut tau = 0.04;
            converged = false;
            for _ in 0..12 {
                let next = self.distance(tx + tau / unit, tx)?;
                let done = (next - tau).abs() <= 1.0e-13;
                tau = next;
                if done {
                    converged = true;
                    break;
                }
            }
            if self.relative && !converged {
                return Err(error("SLR relative light-time iteration did not converge"));
            }
            out = tau;
            bounce = tx + out / unit;
            converged = false;
            for _ in 0..12 {
                let next = self.distance(bounce, bounce + tau / unit)?;
                let done = (next - tau).abs() <= 1.0e-13;
                tau = next;
                if done {
                    converged = true;
                    break;
                }
            }
            if self.relative && !converged {
                return Err(error("SLR relative light-time iteration did not converge"));
            }
            inn = tau;
            rx = bounce + inn / unit;
        } else if self.obs.event == 0 {
            rx = epoch;
            let mut tau = 0.04;
            converged = false;
            for _ in 0..12 {
                let next = self.distance(rx - tau / unit, rx)?;
                let done = (next - tau).abs() <= 1.0e-13;
                tau = next;
                if done {
                    converged = true;
                    break;
                }
            }
            if self.relative && !converged {
                return Err(error("SLR relative light-time iteration did not converge"));
            }
            inn = tau;
            bounce = rx - inn / unit;
            converged = false;
            for _ in 0..12 {
                let next = self.distance(bounce, bounce - tau / unit)?;
                let done = (next - tau).abs() <= 1.0e-13;
                tau = next;
                if done {
                    converged = true;
                    break;
                }
            }
            if self.relative && !converged {
                return Err(error("SLR relative light-time iteration did not converge"));
            }
            out = tau;
            tx = bounce - out / unit;
        } else {
            bounce = epoch;
            let mut tau = 0.04;
            converged = false;
            for _ in 0..12 {
                let next = self.distance(bounce, bounce - tau / unit)?;
                let done = (next - tau).abs() <= 1.0e-13;
                tau = next;
                if done {
                    converged = true;
                    break;
                }
            }
            if self.relative && !converged {
                return Err(error("SLR relative light-time iteration did not converge"));
            }
            out = tau;
            tx = bounce - out / unit;
            converged = false;
            for _ in 0..12 {
                let next = self.distance(bounce, bounce + tau / unit)?;
                let done = (next - tau).abs() <= 1.0e-13;
                tau = next;
                if done {
                    converged = true;
                    break;
                }
            }
            if self.relative && !converged {
                return Err(error("SLR relative light-time iteration did not converge"));
            }
            inn = tau;
            rx = bounce + inn / unit;
        }
        let mut delay = 0.;
        let mut elevations = [0.; 2];
        if let Some(media) = self.obs.media.clone() {
            let sat = self.satellite(bounce)?;
            let mut delays = [0.; 2];
            for (i, jd) in [tx, rx].into_iter().enumerate() {
                let rot = if self.relative {
                    rotation(self.py, self.frame, self.frames, self.obs.jd, jd)?
                } else {
                    rotation(
                        self.py,
                        self.frame,
                        self.frames,
                        jd,
                        (jd - 0.5).floor() + 0.5,
                    )?
                };
                let relative = rotate(
                    &self.obs.enu,
                    sub(rotate(&rot, sat, false), self.obs.station),
                    false,
                );
                elevations[i] = relative[2].atan2(relative[0].hypot(relative[1]));
                delays[i] = media.delay(elevations[i])?;
            }
            delay = 0.5 * (delays[0] + delays[1]);
        }
        let shapiro = if self.shapiro_mu > 0. {
            let sat = self.satellite(bounce)?;
            let tx_station = self.station(tx)?;
            let rx_station = self.station(rx)?;
            0.5 * (self.shapiro_delay(sat, tx_station)? + self.shapiro_delay(sat, rx_station)?)
        } else {
            0.
        };
        let range = 0.5 * C * (out + inn + 2. * bias / C) + delay / 1000.;
        Ok([
            range,
            if self.relative {
                self.obs.jd + tx / DAY
            } else {
                tx
            },
            if self.relative {
                self.obs.jd + bounce / DAY
            } else {
                bounce
            },
            if self.relative {
                self.obs.jd + rx / DAY
            } else {
                rx
            },
            out,
            inn,
            delay,
            elevations[0],
            elevations[1],
            shapiro,
        ])
    }
}

#[pyclass(name = "SLRObservationContext")]
pub struct SlrContext {
    rows: Vec<Observation>,
    starts: Vec<Vec<(f64, f64)>>,
    relative: bool,
    shapiro_mu: f64,
}
impl SlrContext {
    fn width(&self) -> usize {
        if self.shapiro_mu > 0. {
            OUTPUT
        } else {
            OUTPUT - 1
        }
    }
    fn evaluate(
        &self,
        py: Python<'_>,
        states: &[f64],
        biases: &[f64],
        force: &NativeForcePlanContext,
        stages: &Py<PyAny>,
        frame: &Py<PyAny>,
        time_only_stages: bool,
    ) -> PyResult<Vec<f64>> {
        if time_only_stages && force.plan().codes.contains(&2) {
            return Err(error(
                "State-dependent drag stages cannot be shared across SLR trials",
            ));
        }
        let count = states.len() / 6;
        let n = self.rows.len();
        if count == 0
            || count > 257
            || states.len() != count * 6
            || biases.len() != count * n
            || states.iter().chain(biases).any(|x| !x.is_finite())
        {
            return Err(error(
                "SLR trials require bounded finite states and one bias per observation",
            ));
        }
        // Station-bias trials share exactly the same Cartesian trajectory.
        // Deduplicate by bits, never by tolerance, and apply each bias using
        // the original TOF arithmetic after the geometric prediction.
        let mut index = HashMap::new();
        let mut current: Vec<[f64; 6]> = Vec::new();
        let mut trial_indices = Vec::new();
        for row in states.chunks_exact(6) {
            let key: [u64; 6] = std::array::from_fn(|i| row[i].to_bits());
            let j = *index.entry(key).or_insert_with(|| {
                current.push(row.try_into().unwrap());
                current.len() - 1
            });
            trial_indices.push(j);
        }
        for state in &current {
            crate::State::from_array(*state).map_err(error)?;
        }
        let width = self.width();
        let mut out = vec![0.; count * n * width];
        let mut frames = Frames::new();
        // Same schedule for all trials. Interleaving shares exact time-only
        // force metadata; every state-dependent stage is still evaluated.
        for (i, obs) in self.rows.iter().enumerate() {
            for &(t, h) in &self.starts[i] {
                let mut shared = HashMap::<u64, crate::force_plan::StageInputs>::new();
                for x in &mut current {
                    *x = if time_only_stages {
                        force.rk4_state_with_stage(*x, t, h, [0.; 3], |time, state| {
                            if let Some(stage) = shared.get(&time.to_bits()) {
                                return Ok(stage.clone());
                            }
                            let stage = crate::stage_context::prepare(py, stages, time, state)?;
                            shared.insert(time.to_bits(), stage.clone());
                            Ok(stage)
                        })?
                    } else {
                        force.rk4_state(py, *x, t, h, [0.; 3], stages)?
                    };
                }
            }
            let mut geometries = Vec::with_capacity(current.len());
            for &state in &current {
                let mut g = Geometry {
                    py,
                    obs,
                    relative: self.relative,
                    shapiro_mu: self.shapiro_mu,
                    state,
                    force,
                    stages,
                    frame,
                    frames: &mut frames,
                    satellites: HashMap::new(),
                    stations: HashMap::new(),
                };
                geometries.push(g.predict(0.)?);
            }
            for (j, &unique) in trial_indices.iter().enumerate() {
                let mut row = geometries[unique];
                row[0] = 0.5 * C * (row[4] + row[5] + 2. * biases[j * n + i] / C) + row[6] / 1000.;
                out[(j * n + i) * width..(j * n + i + 1) * width].copy_from_slice(&row[..width]);
            }
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(error("SLR batch produced non-finite predictions"));
        }
        Ok(out)
    }
}
#[pymethods]
impl SlrContext {
    #[new]
    #[pyo3(signature=(rows, initial_time_s, maximum_step_s, relative_light_time=false, earth_shapiro_mu_km3_s2=0.0))]
    fn new(
        rows: &[u8],
        initial_time_s: f64,
        maximum_step_s: f64,
        relative_light_time: bool,
        earth_shapiro_mu_km3_s2: f64,
    ) -> PyResult<Self> {
        if !earth_shapiro_mu_km3_s2.is_finite()
            || earth_shapiro_mu_km3_s2 < 0.
            || (earth_shapiro_mu_km3_s2 > 0. && !relative_light_time)
        {
            return Err(error(
                "Earth Shapiro correction requires nonnegative finite GM and relative light time",
            ));
        }
        let flat = decode_f64_bytes(rows)?;
        if flat.is_empty()
            || flat.len() % ROW != 0
            || flat.len() / ROW > 1000
            || flat.iter().any(|x| !x.is_finite())
            || !initial_time_s.is_finite()
            || !maximum_step_s.is_finite()
            || maximum_step_s <= 0.
        {
            return Err(error("Invalid bounded SLR observation context"));
        }
        let mut result = Vec::new();
        let mut starts = Vec::new();
        let mut previous = initial_time_s;
        let mut total_steps = 0usize;
        for r in flat.chunks_exact(ROW) {
            if r[0] < previous
                || (!result.is_empty() && r[0] == previous)
                || ![0., 1., 2.].contains(&r[2])
                || ![0., 1.].contains(&r[17])
            {
                return Err(error(
                    "SLR epochs must increase and event/media codes must be supported",
                ));
            }
            if norm(r[3..6].try_into().unwrap()) <= 0. {
                return Err(error("SLR station position must be nonzero"));
            }
            let media = if r[17] == 1. {
                Some(Media::new(r[6], r[7], r[18], r[19], r[20], r[21])?)
            } else {
                None
            };
            result.push(Observation {
                time: r[0],
                jd: r[1],
                event: r[2] as u8,
                station: r[3..6].try_into().unwrap(),
                enu: r[8..17].try_into().unwrap(),
                media,
            });
            let gap = r[0] - previous;
            if (gap / maximum_step_s).ceil() > 10_000_000. {
                return Err(error("SLR integration schedule exceeds step bound"));
            }
            let mut steps = Vec::new();
            let mut elapsed = 0.;
            while elapsed < gap - 1.0e-12 {
                let h = maximum_step_s.min(gap - elapsed);
                steps.push((previous + elapsed, h));
                elapsed += h;
            }
            total_steps += steps.len();
            if total_steps > 10_000_000 {
                return Err(error("SLR integration schedule exceeds step bound"));
            }
            starts.push(steps);
            previous = r[0];
        }
        Ok(Self {
            rows: result,
            starts,
            relative: relative_light_time,
            shapiro_mu: earth_shapiro_mu_km3_s2,
        })
    }
    #[pyo3(signature=(states, biases, force, stages, frame, time_only_stages=false))]
    fn evaluate_bytes<'py>(
        &self,
        py: Python<'py>,
        states: &[u8],
        biases: &[u8],
        force: PyRef<'py, NativeForcePlanContext>,
        stages: Py<PyAny>,
        frame: Py<PyAny>,
        time_only_stages: bool,
    ) -> PyResult<Py<PyBytes>> {
        let out = self.evaluate(
            py,
            &decode_f64_bytes(states)?,
            &decode_f64_bytes(biases)?,
            &force,
            &stages,
            &frame,
            time_only_stages,
        )?;
        output_bytes(py, &out)
    }
    #[pyo3(signature=(state, biases, steps, force, stages, frame, time_only_stages=false))]
    fn central_sensitivities_bytes<'py>(
        &self,
        py: Python<'py>,
        state: &[u8],
        biases: &[u8],
        steps: &[u8],
        force: PyRef<'py, NativeForcePlanContext>,
        stages: Py<PyAny>,
        frame: Py<PyAny>,
        time_only_stages: bool,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let state = decode_f64_bytes(state)?;
        let biases = decode_f64_bytes(biases)?;
        let steps = decode_f64_bytes(steps)?;
        if state.len() != 6
            || steps.len() != 6
            || steps.iter().any(|x| !x.is_finite() || *x <= 0.)
            || biases.len() != self.rows.len()
        {
            return Err(error(
                "SLR central sensitivities require six positive steps and one state",
            ));
        }
        let mut trials = state.to_vec();
        for (i, h) in steps.iter().enumerate() {
            for sign in [1., -1.] {
                let mut x = state.to_vec();
                x[i] += sign * h;
                trials.extend(x);
            }
        }
        let out = self.evaluate(
            py,
            &trials,
            &biases.repeat(13),
            &force,
            &stages,
            &frame,
            time_only_stages,
        )?;
        let n = self.rows.len();
        let width = self.width();
        let mut jac = vec![0.; n * 6];
        for i in 0..6 {
            for r in 0..n {
                jac[r * 6 + i] = (out[((1 + 2 * i) * n + r) * width]
                    - out[((2 + 2 * i) * n + r) * width])
                    / (2. * steps[i]);
            }
        }
        Ok((
            output_bytes(py, &out[..n * width])?,
            output_bytes(py, &jac)?,
        ))
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SlrContext>()?;
    module.add("SLR_EARTH_SHAPIRO_VERSION", 1)?;
    module.add("SLR_RELATIVE_LIGHT_TIME_SUPPORTED", true)
}
