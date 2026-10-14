//! Prepared body-centered ONP. No Earth force ABI or Python stage callbacks.
use crate::adaptive::{self, StepInfo};

pub type RotationInput = (Vec<u8>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>, bool);
pub type AtmosphereInput = (u8, u8, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, bool);
pub type EphemerisInput = (Vec<f64>, Vec<i64>, Vec<Vec<f64>>);
type Matrix = [f64; 9];
pub(crate) const GRAVITY_DOMAIN_ERROR: &str =
    "planetary harmonics require an exterior position above the gravity reference sphere";

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn mv(m: Matrix, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| m[3 * i + j] * v[j]).sum())
}
fn mtv(m: Matrix, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| m[3 * j + i] * v[j]).sum())
}
fn mm(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|i| (0..3).map(|j| a[(i / 3) * 3 + j] * b[j * 3 + i % 3]).sum())
}
fn finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}
pub fn tdb(tt: f64) -> (f64, f64) {
    let g = (357.53 + 0.9856003 * tt / 86400.).to_radians();
    let rate = 0.9856003_f64.to_radians() / 86400.;
    (
        tt + 0.001657 * g.sin() + 0.000022 * (2. * g).sin(),
        1. + rate * (0.001657 * g.cos() + 0.000044 * (2. * g).cos()),
    )
}

#[derive(Clone)]
pub struct Rotation {
    axes: Vec<u8>,
    polynomials: Vec<f64>,
    terms: Vec<Vec<f64>>,
    bounds: [f64; 2],
    pole: bool,
}
impl Rotation {
    pub fn new(x: RotationInput) -> Result<Self, String> {
        let (axes, polynomials, terms, bounds, pole) = x;
        if axes.is_empty()
            || axes.len() > 8
            || axes.iter().any(|v| *v != 0 && *v != 2)
            || polynomials.len() != 3 * axes.len()
            || !finite(&polynomials)
            || bounds.len() != 2
            || !finite(&bounds)
            || bounds[0] >= bounds[1]
            || (pole && axes.len() != 3)
        {
            return Err("invalid planetary rotation dimensions/coverage".into());
        }
        for term in &terms {
            if term.len() != 5
                || !finite(term)
                || term[0] < 0.
                || term[0].fract() != 0.
                || term[0] >= axes.len() as f64
                || (term[4] != 0. && term[4] != 1.)
            {
                return Err("invalid planetary periodic rotation term".into());
            }
        }
        Ok(Self {
            axes,
            polynomials,
            terms,
            bounds: [bounds[0], bounds[1]],
            pole,
        })
    }
    pub fn evaluate(&self, tt: f64) -> Result<(Matrix, Matrix), String> {
        let (time, time_rate) = tdb(tt);
        if !time.is_finite() || time < self.bounds[0] || time > self.bounds[1] {
            return Err("planetary rotation epoch outside coverage".into());
        }
        let days = time / 86400.;
        let mut angles = Vec::new();
        let mut rates = Vec::new();
        for p in self.polynomials.chunks_exact(3) {
            angles.push(p[0] + p[1] * days + p[2] * days * days);
            rates.push((p[1] + 2. * p[2] * days) / 86400. * time_rate);
        }
        for term in &self.terms {
            let phase = (term[2] + term[3] * days).to_radians();
            let i = term[0] as usize;
            let (trig, derivative) = if term[4] == 0. {
                (phase.sin(), phase.cos())
            } else {
                (phase.cos(), -phase.sin())
            };
            angles[i] += term[1] * trig;
            rates[i] += term[1] * derivative * term[3].to_radians() / 86400. * time_rate;
        }
        if self.pole {
            angles = vec![angles[2], 90. - angles[1], 90. + angles[0]];
            rates = vec![rates[2], -rates[1], rates[0]];
        }
        let mut r = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let mut d = [0.; 9];
        for ((axis, angle), rate) in self.axes.iter().zip(angles).zip(rates) {
            let (s, c) = angle.rem_euclid(360.).to_radians().sin_cos();
            let (a, ap) = if *axis == 0 {
                (
                    [1., 0., 0., 0., c, s, 0., -s, c],
                    [0., 0., 0., 0., -s, c, 0., -c, -s],
                )
            } else {
                (
                    [c, s, 0., -s, c, 0., 0., 0., 1.],
                    [-s, c, 0., -c, -s, 0., 0., 0., 0.],
                )
            };
            let da = mm(d, a);
            let ra = mm(r, ap);
            d = std::array::from_fn(|i| da[i] + ra[i] * rate.to_radians());
            r = mm(r, a);
        }
        Ok((r, d))
    }
}

#[derive(Clone)]
pub struct Atmosphere {
    kind: u8,
    datum: u8,
    p: Vec<f64>,
    heights: Vec<f64>,
    logrho: Vec<f64>,
    wind: Vec<f64>,
    vacuum: bool,
}
impl Atmosphere {
    pub fn new(x: AtmosphereInput) -> Result<Self, String> {
        let (kind, datum, p, heights, logrho, wind, vacuum) = x;
        if kind > 2
            || datum > 2
            || p.len() != 10
            || !finite(&p)
            || p[0] <= 0.
            || p[1] <= 0.
            || p[2] >= p[3]
            || (datum == 2 && p[1] > p[0])
        {
            return Err("invalid planetary atmosphere parameters/datum".into());
        }
        if kind == 1 && (p[4] <= 0. || p[6] <= 0.) {
            return Err("exponential density/scale height must be positive".into());
        }
        if kind == 2
            && (heights.len() < 2
                || logrho.len() != heights.len()
                || wind.len() != 3 * heights.len()
                || !finite(&heights)
                || !finite(&logrho)
                || !finite(&wind)
                || heights.windows(2).any(|v| v[0] >= v[1])
                || heights[0] != p[2]
                || heights[heights.len() - 1] != p[3])
        {
            return Err("invalid planetary atmosphere table".into());
        }
        Ok(Self {
            kind,
            datum,
            p,
            heights,
            logrho,
            wind,
            vacuum,
        })
    }
    pub fn sample(&self, r: [f64; 3]) -> Result<(f64, f64, [f64; 3], bool), String> {
        let a = self.p[0];
        let b = self.p[1];
        let distance = norm(r);
        if distance <= 0. || !distance.is_finite() {
            return Err("singular atmosphere position".into());
        }
        let h = match self.datum {
            0 => distance - a,
            1 => {
                distance
                    - 1. / ((r[0] * r[0] + r[1] * r[1]) / distance.powi(2) / a.powi(2)
                        + r[2].powi(2) / distance.powi(2) / b.powi(2))
                    .sqrt()
            }
            _ => {
                let xy = r[0].hypot(r[1]);
                let e2 = 1. - (b / a).powi(2);
                if xy < 1e-12 {
                    r[2].abs() - b
                } else {
                    let mut phi = r[2].atan2(xy * (1. - e2));
                    for _ in 0..12 {
                        let n = a / (1. - e2 * phi.sin().powi(2)).sqrt();
                        phi = (r[2] + e2 * n * phi.sin()).atan2(xy);
                    }
                    let n = a / (1. - e2 * phi.sin().powi(2)).sqrt();
                    xy * phi.cos() + r[2] * phi.sin() - n * (1. - e2 * phi.sin().powi(2))
                }
            }
        };
        let inside = h >= self.p[2] - 1e-9 && h <= self.p[3] + 1e-9;
        if !inside {
            if self.vacuum && h > self.p[3] {
                return Ok((h, 0., [0.; 3], false));
            }
            return Err(format!(
                "planetary atmosphere altitude {h} outside [{}, {}]",
                self.p[2], self.p[3]
            ));
        }
        let interpolation_h = h.clamp(self.p[2], self.p[3]);
        let (density, wind) = match self.kind {
            0 => (0., [0.; 3]),
            1 => (
                self.p[4] * (-(h - self.p[5]) / self.p[6]).exp(),
                [self.p[7], self.p[8], self.p[9]],
            ),
            _ => {
                let upper = self
                    .heights
                    .partition_point(|v| *v <= interpolation_h)
                    .clamp(1, self.heights.len() - 1);
                let lower = upper - 1;
                let u = (interpolation_h - self.heights[lower])
                    / (self.heights[upper] - self.heights[lower]);
                (
                    (self.logrho[lower] + u * (self.logrho[upper] - self.logrho[lower])).exp(),
                    std::array::from_fn(|i| {
                        self.wind[3 * lower + i]
                            + u * (self.wind[3 * upper + i] - self.wind[3 * lower + i])
                    }),
                )
            }
        };
        if !density.is_finite() || density < 0. {
            return Err("nonfinite planetary atmosphere density".into());
        }
        Ok((h, density, wind, inside))
    }
}

pub(crate) struct Ephemeris {
    times: Vec<f64>,
    targets: Vec<i64>,
    states: Vec<Vec<f64>>,
}
impl Ephemeris {
    pub(crate) fn new(x: EphemerisInput) -> Result<Self, String> {
        let (times, targets, states) = x;
        if targets.is_empty() && times.is_empty() && states.is_empty() {
            return Ok(Self {
                times,
                targets,
                states,
            });
        }
        if times.len() < 2
            || !finite(&times)
            || times.windows(2).any(|v| v[0] >= v[1])
            || states.len() != targets.len()
            || states
                .iter()
                .any(|v| v.len() != 6 * times.len() || !finite(v))
            || targets
                .iter()
                .enumerate()
                .any(|(i, v)| targets[..i].contains(v))
        {
            return Err("invalid planetary ephemeris table".into());
        }
        Ok(Self {
            times,
            targets,
            states,
        })
    }
    pub(crate) fn position(&self, target: i64, time: f64) -> Result<[f64; 3], String> {
        let index = self
            .targets
            .iter()
            .position(|v| *v == target)
            .ok_or("planetary ephemeris missing target")?;
        if time < self.times[0] || time > self.times[self.times.len() - 1] {
            return Err("planetary ephemeris epoch outside coverage".into());
        }
        let upper = self
            .times
            .partition_point(|v| *v <= time)
            .clamp(1, self.times.len() - 1);
        let lower = upper - 1;
        let dt = self.times[upper] - self.times[lower];
        let u = (time - self.times[lower]) / dt;
        let values = &self.states[index];
        Ok(std::array::from_fn(|i| {
            (2. * u.powi(3) - 3. * u.powi(2) + 1.) * values[6 * lower + i]
                + (u.powi(3) - 2. * u.powi(2) + u) * dt * values[6 * lower + 3 + i]
                + (-2. * u.powi(3) + 3. * u.powi(2)) * values[6 * upper + i]
                + (u.powi(3) - u.powi(2)) * dt * values[6 * upper + 3 + i]
        }))
    }
}

// Value plus Cartesian derivatives, with separate cosine/sine channels.
type Solid = [[f64; 2]; 4];
fn multiply_xy(h: Solid, q: [f64; 3], scale: f64) -> Solid {
    let mut out = [[0.; 2]; 4];
    for i in 0..4 {
        out[i] = [
            scale * (q[0] * h[i][0] - q[1] * h[i][1]),
            scale * (q[0] * h[i][1] + q[1] * h[i][0]),
        ];
    }
    out[1][0] += scale * h[0][0];
    out[1][1] += scale * h[0][1];
    out[2][0] -= scale * h[0][1];
    out[2][1] += scale * h[0][0];
    out
}
pub(crate) struct Harmonics {
    n: usize,
    m: usize,
    c: Vec<f64>,
    s: Vec<f64>,
}
impl Harmonics {
    pub(crate) fn new(n: usize, m: usize, mut c: Vec<f64>, mut s: Vec<f64>) -> Result<Self, String> {
        if n > 20
            || m > n
            || c.len() != (n + 1) * (m + 1)
            || s.len() != c.len()
            || !finite(&c)
            || !finite(&s)
        {
            return Err(
                "planetary harmonics require finite normalized degree/order 0..20 arrays".into(),
            );
        }
        for degree in 0..=n {
            for order in 0..=degree.min(m) {
                let mut ratio = 1.;
                for k in (degree - order + 1)..=(degree + order) {
                    ratio /= k as f64;
                }
                let normalization =
                    ((if order == 0 { 1. } else { 2. }) * (2 * degree + 1) as f64 * ratio).sqrt();
                let i = degree * (m + 1) + order;
                c[i] *= normalization;
                s[i] *= normalization;
            }
        }
        Ok(Self { n, m, c, s })
    }
    pub(crate) fn acceleration(&self, r: [f64; 3], rotation: Matrix, mu: f64, radius: f64) -> [f64; 3] {
        if self.n < 2 {
            return [0.; 3];
        }
        let q = mv(rotation, r).map(|v| v / radius);
        let q2 = q.iter().map(|v| v * v).sum::<f64>();
        let mut h = vec![[[0.; 2]; 4]; (self.n + 1) * (self.m + 1)];
        let stride = self.m + 1;
        h[0][0][0] = 1.;
        for n in 1..=self.n {
            if n <= self.m {
                h[n * stride + n] = multiply_xy(h[(n - 1) * stride + n - 1], q, (2 * n - 1) as f64);
            }
            let m = n - 1;
            if m <= self.m {
                let prior = h[(n - 1) * stride + m];
                let scale = (2 * m + 1) as f64;
                let mut out = prior;
                for i in 0..4 {
                    for j in 0..2 {
                        out[i][j] *= scale * q[2];
                    }
                }
                for j in 0..2 {
                    out[3][j] += scale * prior[0][j];
                }
                h[n * stride + m] = out;
            }
            if n >= 2 {
                for m in 0..=self.m.min(n - 2) {
                    let prev = h[(n - 1) * stride + m];
                    let older = h[(n - 2) * stride + m];
                    let mut out = [[0.; 2]; 4];
                    let b = (2 * n - 1) as f64;
                    let c = (n + m - 1) as f64;
                    let denominator = (n - m) as f64;
                    for i in 0..4 {
                        for j in 0..2 {
                            out[i][j] =
                                (b * q[2] * prev[i][j] - c * q2 * older[i][j]) / denominator;
                        }
                    }
                    for axis in 0..3 {
                        for j in 0..2 {
                            out[axis + 1][j] += ((if axis == 2 { b * prev[0][j] } else { 0. })
                                - c * 2. * q[axis] * older[0][j])
                                / denominator;
                        }
                    }
                    h[n * stride + m] = out;
                }
            }
        }
        let mut gradient = [0.; 3];
        for n in 2..=self.n {
            let radial = q2.powf(-(n as f64 + 0.5));
            for m in 0..=self.m.min(n) {
                let i = n * stride + m;
                let value = self.c[i] * h[i][0][0] + self.s[i] * h[i][0][1];
                for axis in 0..3 {
                    let derivative = self.c[i] * h[i][axis + 1][0] + self.s[i] * h[i][axis + 1][1];
                    gradient[axis] +=
                        radial * (derivative - (2 * n + 1) as f64 / q2 * q[axis] * value);
                }
            }
        }
        mtv(rotation, gradient).map(|v| v * mu / radius.powi(2))
    }
}

pub struct Stage {
    pub acceleration: [f64; 3],
    pub diagnostics: [f64; 4],
    pub components: Vec<[f64; 3]>,
}
pub struct Plan {
    s: [f64; 12],
    gravity: Rotation,
    air: Rotation,
    atmosphere: Atmosphere,
    harmonics: Harmonics,
    ephemeris: Ephemeris,
    targets: Vec<i64>,
    gms: Vec<f64>,
    center: i64,
}
impl Plan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        center: i64,
        scalars: Vec<f64>,
        dims: (usize, usize),
        coefficients: (Vec<f64>, Vec<f64>),
        gravity: RotationInput,
        air: RotationInput,
        atmosphere: AtmosphereInput,
        ephemeris: EphemerisInput,
        bodies: (Vec<i64>, Vec<f64>),
    ) -> Result<Self, String> {
        if scalars.len() != 12 || !finite(&scalars) {
            return Err("planetary plan requires 12 finite scalars".into());
        }
        let s: [f64; 12] = scalars.try_into().unwrap();
        // mu, harmonic radius, mass, CdA, CrA, shadow radius, initial TT seconds,
        // duration, is_sun, drag_enabled, srp_enabled, cylindrical_shadow.
        if s[0] <= 0.
            || s[1] <= 0.
            || s[2] <= 0.
            || s[3] < 0.
            || s[4] < 0.
            || s[5] <= 0.
            || s[7] < 0.
            || s[8..].iter().any(|v| *v != 0. && *v != 1.)
            || (center == 10) != (s[8] == 1.)
            || (s[8] == 1. && (s[9] == 1. || s[11] == 1. || dims.0 != 0))
        {
            return Err("invalid planetary force-plan scalars".into());
        }
        let (targets, gms) = bodies;
        if targets.len() != gms.len()
            || !finite(&gms)
            || gms.iter().any(|v| *v <= 0.)
            || targets
                .iter()
                .enumerate()
                .any(|(i, v)| *v == center || targets[..i].contains(v))
        {
            return Err("invalid external planetary perturbers".into());
        }
        let result = Self {
            s,
            center,
            targets,
            gms,
            gravity: Rotation::new(gravity)?,
            air: Rotation::new(air)?,
            atmosphere: Atmosphere::new(atmosphere)?,
            harmonics: Harmonics::new(dims.0, dims.1, coefficients.0, coefficients.1)?,
            ephemeris: Ephemeris::new(ephemeris)?,
        };
        if result.s[9] == 1. && result.atmosphere.kind == 0 {
            return Err("planetary drag requires atmosphere".into());
        }
        for elapsed in [0., s[7]] {
            result.gravity.evaluate(s[6] + elapsed)?;
            result.air.evaluate(s[6] + elapsed)?;
            let time = tdb(s[6] + elapsed).0;
            for target in &result.targets {
                result.ephemeris.position(*target, time)?;
            }
            if s[10] == 1. && center != 10 {
                result.ephemeris.position(10, time)?;
            }
        }
        Ok(result)
    }
    pub fn stage(&self, t: f64, x: [f64; 6]) -> Result<Stage, String> {
        if !t.is_finite() || t < 0. || t > self.s[7] + 1e-8 || !finite(&x) {
            return Err("planetary stage outside prepared time/state domain".into());
        }
        let r = [x[0], x[1], x[2]];
        let v = [x[3], x[4], x[5]];
        let distance = norm(r);
        if distance <= 0. {
            return Err("singular planetary radius".into());
        }
        if self.harmonics.n > 0 && distance <= self.s[1] {
            return Err(GRAVITY_DOMAIN_ERROR.into());
        }
        let (g, _) = self.gravity.evaluate(self.s[6] + t)?;
        let (a, ad) = self.air.evaluate(self.s[6] + t)?;
        let (height, rho, wind, inside) = self.atmosphere.sample(mv(a, r))?;
        let av = mv(a, v);
        let adr = mv(ad, r);
        let relative = mtv(a, std::array::from_fn(|i| av[i] + adr[i] - wind[i] / 1000.));
        let speed = norm(relative) * 1000.;
        let mut components = vec![
            r.map(|v| -self.s[0] * v / distance.powi(3)),
            self.harmonics.acceleration(r, g, self.s[0], self.s[1]),
            [0.; 3],
            [0.; 3],
        ];
        if self.s[9] == 1. {
            components[2] = relative.map(|v| -0.5 * rho * self.s[3] / self.s[2] * speed * v);
        }
        let time = tdb(self.s[6] + t).0;
        if self.s[10] == 1. {
            let sun = if self.center == 10 {
                [0.; 3]
            } else {
                self.ephemeris.position(10, time)?
            };
            let delta = std::array::from_fn(|i| r[i] - sun[i]);
            let d = norm(delta);
            if d <= 0. {
                return Err("singular spacecraft-Sun distance".into());
            }
            let mut shadow = 1.;
            if self.s[11] == 1. {
                let sun_distance = norm(sun);
                if sun_distance <= 0. {
                    return Err("singular shadow geometry".into());
                }
                let u = sun.map(|v| v / sun_distance);
                let projection = (0..3).map(|i| r[i] * u[i]).sum::<f64>();
                let transverse = std::array::from_fn(|i| r[i] - projection * u[i]);
                if projection < 0. && norm(transverse) < self.s[5] {
                    shadow = 0.;
                }
            }
            components[3] = delta.map(|v| {
                shadow * (1361. / 299792458.) * (149597870.7 / d).powi(2) * self.s[4] / self.s[2]
                    * v
                    / d
                    / 1000.
            });
        }
        for (target, mu) in self.targets.iter().zip(&self.gms) {
            let pos = self.ephemeris.position(*target, time)?;
            let delta = std::array::from_fn(|i| pos[i] - r[i]);
            let d = norm(delta);
            let origin = norm(pos);
            if d <= 0. || origin <= 0. {
                return Err("singular planetary third-body geometry".into());
            }
            components.push(std::array::from_fn(|i| {
                mu * (delta[i] / d.powi(3) - pos[i] / origin.powi(3))
            }));
        }
        let acceleration = std::array::from_fn(|i| components.iter().map(|v| v[i]).sum());
        if !finite(&acceleration) {
            return Err("nonfinite planetary acceleration".into());
        }
        Ok(Stage {
            acceleration,
            diagnostics: [height, rho, speed, if inside { 1. } else { 0. }],
            components,
        })
    }
    pub fn propagate(
        &self,
        x: [f64; 6],
        t: f64,
        dt: f64,
        method: &str,
        atol: f64,
        rtol: f64,
    ) -> Result<([f64; 6], StepInfo), String> {
        if !t.is_finite()
            || !dt.is_finite()
            || dt < 0.
            || t < 0.
            || t + dt > self.s[7] + 1e-8
            || !finite(&x)
            || !atol.is_finite()
            || !rtol.is_finite()
            || atol < 0.
            || rtol < 0.
        {
            return Err("invalid planetary integration inputs".into());
        }
        let derivative = |time: f64, state: [f64; 6]| -> Result<[f64; 6], String> {
            let a = self.stage(time, state)?.acceleration;
            Ok([state[3], state[4], state[5], a[0], a[1], a[2]])
        };
        match method {
            "rkf78" => adaptive::integrate_rkf78(x, t, dt, atol, rtol, None, derivative),
            "dopri5" | "adaptive" => {
                adaptive::integrate_dopri5(x, t, dt, atol, rtol, None, derivative)
            }
            "rk4" => {
                let k1 = derivative(t, x)?;
                let k2 = derivative(t + dt / 2., crate::stage(x, k1, dt / 2.))?;
                let k3 = derivative(t + dt / 2., crate::stage(x, k2, dt / 2.))?;
                let k4 = derivative(t + dt, crate::stage(x, k3, dt))?;
                let next = std::array::from_fn(|i| {
                    x[i] + dt / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i])
                });
                if !finite(&next) {
                    return Err("nonfinite planetary integrated state".into());
                }
                Ok((next, StepInfo::default()))
            }
            _ => Err("unsupported planetary integrator".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity() -> RotationInput {
        (vec![2], vec![0., 0., 0.], vec![], vec![-1e12, 1e12], false)
    }
    fn atmosphere(kind: u8) -> AtmosphereInput {
        (
            kind,
            0,
            vec![3396., 3396., 0., 2000., 1e-8, 600., 20., 0., 0., 0.],
            vec![],
            vec![],
            vec![],
            false,
        )
    }
    fn plan(drag: bool) -> Plan {
        Plan::new(
            499,
            vec![
                42828.,
                3396.,
                100.,
                2.2,
                0.,
                3396.,
                0.,
                600.,
                0.,
                f64::from(drag),
                0.,
                0.,
            ],
            (0, 0),
            (vec![1.], vec![0.]),
            identity(),
            identity(),
            atmosphere(if drag { 1 } else { 0 }),
            (vec![], vec![], vec![]),
            (vec![], vec![]),
        )
        .unwrap()
    }
    #[test]
    fn drag_has_kilometre_units_and_opposes_flow() {
        let stage = plan(true).stage(0., [3996., 0., 0., 0., 3., 0.]).unwrap();
        assert!((stage.components[2][1] + 0.5 * 1e-8 * 2.2 / 100. * 3000. * 3.).abs() < 1e-20);
        assert_eq!(stage.diagnostics[2], 3000.);
    }
    #[test]
    fn table_bounds_and_log_interpolation() {
        let air = Atmosphere::new((
            2,
            0,
            vec![3396., 3396., 100., 200., 0., 0., 1., 0., 0., 0.],
            vec![100., 200.],
            vec![1e-8_f64.ln(), 1e-10_f64.ln()],
            vec![0.; 6],
            false,
        ))
        .unwrap();
        assert!((air.sample([3546., 0., 0.]).unwrap().1 - 1e-9).abs() < 1e-23);
        assert!(air.sample([3600., 0., 0.]).is_err());
    }
    #[test]
    fn polar_tesseral_gradient_is_finite_and_nonzero() {
        let mut c = vec![0.; 9];
        let mut s = vec![0.; 9];
        c[7] = 1e-5;
        s[7] = -2e-5;
        let h = Harmonics::new(2, 2, c, s).unwrap();
        let a = h.acceleration(
            [0., 0., 4000.],
            [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            42828.,
            3397.,
        );
        assert!(finite(&a));
        assert!(a[0].abs() > 1e-8);
        assert!(a[1].abs() > 1e-8);
    }
    #[test]
    fn no_forces_are_silently_dropped() {
        assert!(plan(false)
            .propagate([4000., 0., 0., 0., 3., 0.], 0., 601., "rk4", 1e-10, 1e-10)
            .is_err());
        assert!(plan(false)
            .propagate([4000., 0., 0., 0., 3., 0.], 0., 10., "bogus", 1e-10, 1e-10)
            .is_err());
    }
    #[test]
    fn hermite_reproduces_cubic_position_and_velocity() {
        let e = Ephemeris::new((
            vec![0., 2.],
            vec![10],
            vec![vec![0., 0., 0., 0., 0., 0., 8., 0., 0., 12., 0., 0.]],
        ))
        .unwrap();
        assert_eq!(e.position(10, 1.).unwrap(), [1., 0., 0.]);
        assert!(e.position(10, 3.).is_err());
    }
}
