//! Ordered built-in ONP forces. Time-dependent environment values are supplied by Python.

use crate::{
    harmonics::HarmonicTerms,
    precision_plan::{PrecisionForce, PrecisionStage, STAGE_WIDTH},
    zonal,
};

pub const HARMONICS: u8 = 1;
pub const DRAG: u8 = 2;
pub const SRP: u8 = 3;
pub const SUN: u8 = 4;
pub const MOON: u8 = 5;
pub const J2: u8 = 6;
pub const J3: u8 = 7;
pub const J4: u8 = 8;

pub struct ForcePlan {
    pub codes: Vec<u8>,
    pub scalars: [f64; 15],
    pub shadow_model: u8,
    pub harmonics: Option<HarmonicTerms>,
    pub precision: Vec<PrecisionForce>,
}

#[derive(Clone)]
pub struct StageInputs {
    pub precision: Vec<PrecisionStage>,
    pub rotation: [f64; 9],
    pub density_kg_m3: f64,
    pub relative_velocity_eci_km_s: [f64; 3],
    pub sun_eci_km: [f64; 3],
    pub moon_eci_km: [f64; 3],
}

impl StageInputs {
    pub fn from_values(values: Vec<f64>) -> Result<Self, &'static str> {
        if values.len() < 19 || (values.len() - 19) % STAGE_WIDTH != 0 {
            return Err("stage inputs require 19 values plus 21 per precision force");
        }
        let mut stage = Self::from_flat(values[..19].try_into().unwrap())?;
        stage.precision = values[19..]
            .chunks_exact(STAGE_WIDTH)
            .map(PrecisionStage::from_flat)
            .collect::<Result<_, _>>()?;
        Ok(stage)
    }
    pub fn from_flat(values: [f64; 19]) -> Result<Self, &'static str> {
        if !values.iter().all(|value| value.is_finite()) {
            return Err("stage inputs must contain only finite values");
        }
        Ok(Self {
            precision: Vec::new(),
            rotation: values[0..9].try_into().unwrap(),
            density_kg_m3: values[9],
            relative_velocity_eci_km_s: values[10..13].try_into().unwrap(),
            sun_eci_km: values[13..16].try_into().unwrap(),
            moon_eci_km: values[16..19].try_into().unwrap(),
        })
    }
}

impl ForcePlan {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.scalars.iter().all(|value| value.is_finite()) || self.scalars[0] <= 0.0 {
            return Err("force-plan scalars must be finite and central mu positive");
        }
        if self.shadow_model > 3 {
            return Err("shadow model must be 0, 1, 2, or 3");
        }
        for &code in &self.codes {
            if !(HARMONICS..=crate::precision_plan::ECOM2).contains(&code) {
                return Err("unsupported Rust force code");
            }
            if code == HARMONICS && self.harmonics.is_none() {
                return Err("harmonic force requires normalized harmonic terms");
            }
        }
        if self
            .codes
            .iter()
            .copied()
            .filter(|code| *code > J4)
            .ne(self.precision.iter().map(PrecisionForce::code))
        {
            return Err("precision specifications must match ordered force codes");
        }
        if let Some(terms) = &self.harmonics {
            terms.validate()?;
        }
        Ok(())
    }

    pub fn acceleration(
        &self,
        state: [f64; 6],
        stage: &StageInputs,
        command: [f64; 3],
    ) -> Result<[f64; 3], String> {
        self.acceleration_with_mass(state, stage, command, self.scalars[2])
    }

    pub fn acceleration_with_mass(
        &self,
        state: [f64; 6],
        stage: &StageInputs,
        command: [f64; 3],
        mass_kg: f64,
    ) -> Result<[f64; 3], String> {
        if stage.precision.len() != self.precision.len() {
            return Err("precision stage count does not match force plan".into());
        }
        let mut precision_index = 0;
        let r = [state[0], state[1], state[2]];
        let norm2 = dot(r, r);
        let scale = if norm2 == 0.0 {
            0.0
        } else {
            -self.scalars[0] / (norm2.sqrt() * norm2)
        };
        let mut total = [
            scale * r[0] + command[0],
            scale * r[1] + command[1],
            scale * r[2] + command[2],
        ];
        for &code in &self.codes {
            let part = match code {
                HARMONICS => self.harmonics.as_ref().unwrap().acceleration_eci(
                    r,
                    stage.rotation,
                    self.scalars[14],
                    self.scalars[1],
                ),
                DRAG => drag(stage, mass_kg, self.scalars[3], self.scalars[4]),
                SRP => srp(
                    r,
                    stage.sun_eci_km,
                    &self.scalars,
                    self.shadow_model,
                    mass_kg,
                    stage.rotation,
                ),
                SUN => third_body(r, stage.sun_eci_km, self.scalars[12]),
                MOON => third_body(r, stage.moon_eci_km, self.scalars[13]),
                J2..=J4 => zonal::component(code - 4, r, self.scalars[14]),
                crate::precision_plan::RADIATION..=crate::precision_plan::ECOM2 => {
                    let index = precision_index;
                    precision_index += 1;
                    self.precision[index].acceleration(
                        state,
                        &stage.precision[index],
                        self.scalars[0],
                        mass_kg,
                        self.scalars[7],
                    )?
                }
                _ => unreachable!(),
            };
            for axis in 0..3 {
                total[axis] += part[axis];
            }
        }
        Ok(total)
    }

    pub fn derivative(
        &self,
        state: [f64; 6],
        stage: &StageInputs,
        command: [f64; 3],
    ) -> Result<[f64; 6], String> {
        let a = self.acceleration(state, stage, command)?;
        Ok([state[3], state[4], state[5], a[0], a[1], a[2]])
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn third_body(r: [f64; 3], body: [f64; 3], mu: f64) -> [f64; 3] {
    let delta = [body[0] - r[0], body[1] - r[1], body[2] - r[2]];
    let d2 = dot(delta, delta);
    let b2 = dot(body, body);
    if d2 == 0.0 || b2 == 0.0 {
        return [0.0; 3];
    }
    let d3 = d2.sqrt().powi(3);
    let b3 = b2.sqrt().powi(3);
    std::array::from_fn(|i| mu * (delta[i] / d3 - body[i] / b3))
}

fn drag(stage: &StageInputs, mass: f64, cd: f64, area: f64) -> [f64; 3] {
    if stage.density_kg_m3 <= 0.0 || mass <= 0.0 || area <= 0.0 {
        return [0.0; 3];
    }
    let v = stage.relative_velocity_eci_km_s;
    let speed = dot(v, v).sqrt() * 1.0e3;
    let scale = -0.5 * stage.density_kg_m3 * cd * area / mass * speed;
    std::array::from_fn(|i| scale * v[i])
}

pub(crate) fn illumination(alpha: f64, beta: f64, gamma: f64) -> f64 {
    if beta <= 0.0 || gamma >= alpha + beta {
        return 1.0;
    }
    if alpha > beta && gamma <= alpha - beta {
        return 0.0;
    }
    if beta > alpha && gamma <= beta - alpha {
        return (1.0 - alpha * alpha / (beta * beta)).max(0.0);
    }
    if gamma <= 0.0 {
        return if alpha >= beta {
            0.0
        } else {
            (1.0 - alpha * alpha / (beta * beta)).max(0.0)
        };
    }
    let e = ((gamma * gamma + alpha * alpha - beta * beta) / (2.0 * gamma * alpha))
        .clamp(-1.0, 1.0)
        .acos();
    let s = ((gamma * gamma + beta * beta - alpha * alpha) / (2.0 * gamma * beta))
        .clamp(-1.0, 1.0)
        .acos();
    let product = ((-gamma + alpha + beta)
        * (gamma + alpha - beta)
        * (gamma - alpha + beta)
        * (gamma + alpha + beta))
        .max(0.0);
    (1.0 - (alpha * alpha * e + beta * beta * s - 0.5 * product.sqrt())
        / (std::f64::consts::PI * beta * beta))
        .clamp(0.0, 1.0)
}

fn ellipsoid_limb_angle(r: [f64; 3], sun: [f64; 3], radius: f64) -> f64 {
    let distance = dot(r, r).sqrt();
    let u = r.map(|x| x / distance);
    let mut v = std::array::from_fn(|i| sun[i] - dot(sun, u) * u[i]);
    if dot(v, v).sqrt() <= 1e-12 * dot(sun, sun).sqrt().max(1.) {
        let axis = (0..3).min_by(|&a, &b| u[a].abs().total_cmp(&u[b].abs())).unwrap();
        v = std::array::from_fn(|i| if i == axis { 1. - u[axis] * u[i] } else { -u[axis] * u[i] });
    }
    let norm = dot(v, v).sqrt();
    v = v.map(|x| x / norm);
    let q = [1., 1., (1.0_f64 - 1. / 298.257223563).powi(-2)].map(|x| x / radius.powi(2));
    let a: f64 = (0..3).map(|i| u[i] * q[i] * u[i]).sum();
    let b: f64 = (0..3).map(|i| u[i] * q[i] * v[i]).sum();
    let c: f64 = (0..3).map(|i| v[i] * q[i] * v[i]).sum();
    if a * distance.powi(2) <= 1. { return std::f64::consts::FRAC_PI_2; }
    let y = ((1. - 1. / (a * distance.powi(2))) / (c - b * b / a)).sqrt();
    let x = (1. / distance - b * y) / a;
    y.atan2(distance - x)
}

fn srp(r: [f64; 3], sun: [f64; 3], s: &[f64; 15], shadow_model: u8, mass_kg: f64, rotation: [f64; 9]) -> [f64; 3] {
    let r_norm = dot(r, r).sqrt();
    let sun_norm = dot(sun, sun).sqrt();
    let rho = [sun[0] - r[0], sun[1] - r[1], sun[2] - r[2]];
    let rho_norm = dot(rho, rho).sqrt();
    if rho_norm == 0.0 || mass_kg <= 0.0 || s[6] <= 0.0 {
        return [0.0; 3];
    }
    let mut shadow = 1.0;
    if shadow_model != 0 {
        if r_norm <= s[10] {
            shadow = 0.0;
        } else if sun_norm > 0.0 {
            let sunward = dot(r, sun) / sun_norm >= 0.0;
            if !sunward {
                if shadow_model == 1 {
                    let along = dot(r, sun) / sun_norm;
                    if along < 0.0 && (dot(r, r) - along * along).max(0.0) < s[10] * s[10] {
                        shadow = 0.0;
                    }
                } else {
                    let alpha = if shadow_model == 3 {
                        let rotate = |v: [f64; 3]| std::array::from_fn(|i| rotation[i*3]*v[0]+rotation[i*3+1]*v[1]+rotation[i*3+2]*v[2]);
                        ellipsoid_limb_angle(rotate(r), rotate(sun), s[10])
                    } else { (s[10] / r_norm).clamp(-1.0, 1.0).asin() };
                    let beta = (s[11] / rho_norm).clamp(-1.0, 1.0).asin();
                    let separation = (-dot(r, rho) / (r_norm * rho_norm)).clamp(-1.0, 1.0).acos();
                    shadow = illumination(alpha, beta, separation);
                }
            }
        }
    }
    let scale = -s[8] * (s[9] / rho_norm).powi(2) * s[7] * s[6] / mass_kg / 1.0e3 * shadow;
    std::array::from_fn(|i| scale * rho[i] / rho_norm)
}
