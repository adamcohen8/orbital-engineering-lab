//! Prepared precision forces for the ordered ONP plan. Physics stays with its
//! existing kernel owners; this adapter supplies immutable tables and frames.

use crate::{ocean_tides::OceanTidesContext, precision_forces, solid_tides};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type TideCoefficients = Arc<([f64; 25], [f64; 25])>;
type TideCache = HashMap<Vec<u64>, TideCoefficients>;
static TIDE_CACHE: OnceLock<Mutex<TideCache>> = OnceLock::new();

// Repeated batch-OD trajectories revisit identical epochs. Cache only the
// authoritative, state-independent coefficient calculation, with every numeric
// input and convention in the key. Spacecraft gradients remain uncached.
fn solid_coefficients(stage: &PrecisionStage, mu: f64, zero: bool, pole: bool,
                      sun_mu: f64, moon_mu: f64) -> Result<TideCoefficients, String> {
    let mut key: Vec<u64> = stage.sun.iter().chain(stage.moon.iter()).copied()
        .chain([mu, stage.radius, stage.jd_tt, stage.jd_ut1, stage.pole[0], stage.pole[1], sun_mu, moon_mu])
        .map(f64::to_bits).collect();
    key.extend([zero as u64, pole as u64]);
    let cache = TIDE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(value) = cache.lock().unwrap().get(&key) { return Ok(Arc::clone(value)); }
    let value = Arc::new(solid_tides::coefficients(stage.sun, stage.moon, mu, stage.radius,
        stage.jd_tt, stage.jd_ut1, if zero { "zero_tide" } else { "tide_free" },
        if pole { Some((stage.pole[0], stage.pole[1])) } else { None }, sun_mu, moon_mu)?);
    let mut cache = cache.lock().unwrap();
    if cache.len() >= 131072 { cache.clear(); }
    cache.insert(key, Arc::clone(&value));
    Ok(value)
}

pub const RADIATION: u8 = 9;
pub const SCHWARZSCHILD: u8 = 10;
pub const SOLID: u8 = 11;
pub const OCEAN: u8 = 12;
pub const EMPIRICAL_RIC: u8 = 13;
pub const ECOM2: u8 = 14;
pub const STAGE_WIDTH: usize = 21;
pub type Specification = (u8, Vec<f64>, Vec<Vec<f64>>);

pub enum PrecisionForce {
    Radiation {
        area: f64,
        albedo: bool,
        infrared: bool,
        quadrature: Vec<Vec<f64>>,
    },
    Schwarzschild,
    #[cfg(feature = "pro")]
    Ecom2 { coefficients: [f64; 7] },
    EmpiricalRic {
        coefficients: [f64; 3],
        cosine: [f64; 3],
        sine: [f64; 3],
        omega: f64,
        reference: f64,
    },
    Solid {
        zero_tide: bool,
        pole: bool,
        sun_mu: f64,
        moon_mu: f64,
    },
    Ocean {
        context: OceanTidesContext,
        pole: bool,
    },
}

#[derive(Clone)]
pub struct PrecisionStage {
    pub rotation: [f64; 9],
    pub sun: [f64; 3],
    pub moon: [f64; 3],
    pub jd_tt: f64,
    pub jd_ut1: f64,
    pub pole: [f64; 2],
    pub elapsed: f64,
    pub radius: f64,
}

impl PrecisionStage {
    pub fn from_flat(row: &[f64]) -> Result<Self, &'static str> {
        if row.len() != STAGE_WIDTH || row.iter().any(|v| !v.is_finite()) {
            return Err("precision stage inputs must have 21 finite values");
        }
        Ok(Self {
            rotation: row[..9].try_into().unwrap(),
            sun: row[9..12].try_into().unwrap(),
            moon: row[12..15].try_into().unwrap(),
            jd_tt: row[15],
            jd_ut1: row[16],
            pole: row[17..19].try_into().unwrap(),
            elapsed: row[19],
            radius: row[20],
        })
    }
}

fn mv(rotation: &[f64; 9], vector: [f64; 3], transpose: bool) -> [f64; 3] {
    std::array::from_fn(|i| {
        let index = |j| if transpose { j * 3 + i } else { i * 3 + j };
        rotation[index(0)] * vector[0]
            + rotation[index(1)] * vector[1]
            + rotation[index(2)] * vector[2]
    })
}

impl PrecisionForce {
    pub fn new((code, values, tables): Specification) -> Result<Self, String> {
        if values
            .iter()
            .chain(tables.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err("precision force specifications must be finite".into());
        }
        match code {
            RADIATION
                if values.len() == 3
                    && tables.len() == 4
                    && values[0] >= 0.0
                    && (8..=128).contains(&tables[0].len())
                    && tables[1].len() == tables[0].len()
                    && tables[2].len() == 4 * tables[0].len()
                    && tables[3].len() == tables[2].len()
                    && matches!(values[1], 0.0 | 1.0)
                    && matches!(values[2], 0.0 | 1.0)
                    && (values[1] == 1.0 || values[2] == 1.0) =>
            {
                Ok(Self::Radiation {
                    area: values[0],
                    albedo: values[1] == 1.0,
                    infrared: values[2] == 1.0,
                    quadrature: tables,
                })
            }
            SCHWARZSCHILD if values.is_empty() && tables.is_empty() => Ok(Self::Schwarzschild),
            #[cfg(feature = "pro")]
            ECOM2 if values.len() == 7 && tables.is_empty() => Ok(Self::Ecom2 { coefficients: values.try_into().unwrap() }),
            SOLID
                if values.len() == 4
                    && tables.is_empty()
                    && matches!(values[0], 0.0 | 1.0)
                    && matches!(values[1], 0.0 | 1.0)
                    && values[2] > 0.0
                    && values[3] > 0.0 =>
            {
                Ok(Self::Solid {
                    zero_tide: values[0] == 1.0,
                    pole: values[1] == 1.0,
                    sun_mu: values[2],
                    moon_mu: values[3],
                })
            }
            OCEAN
                if values.len() == 2
                    && tables.len() == 2
                    && values[0].fract() == 0.0
                    && (2.0..=20.0).contains(&values[0])
                    && matches!(values[1], 0.0 | 1.0) =>
            {
                let mut tables = tables.into_iter();
                Ok(Self::Ocean {
                    context: OceanTidesContext::new(
                        tables.next().unwrap(),
                        tables.next().unwrap(),
                        values[0] as usize,
                    )?,
                    pole: values[1] == 1.0,
                })
            }
            EMPIRICAL_RIC if values.len() == 3 && tables.is_empty() => Ok(Self::EmpiricalRic {
                coefficients: values.try_into().unwrap(),
                cosine: [0.; 3],
                sine: [0.; 3],
                omega: 0.,
                reference: 0.,
            }),
            EMPIRICAL_RIC if values.len() == 11 && tables.is_empty() && values[9] > 0. => {
                Ok(Self::EmpiricalRic {
                    coefficients: values[..3].try_into().unwrap(),
                    cosine: values[3..6].try_into().unwrap(),
                    sine: values[6..9].try_into().unwrap(),
                    omega: values[9],
                    reference: values[10],
                })
            }
            _ => Err("invalid precision force specification".into()),
        }
    }

    pub fn code(&self) -> u8 {
        match self {
            Self::Radiation { .. } => RADIATION,
            Self::Schwarzschild => SCHWARZSCHILD,
            #[cfg(feature = "pro")]
            Self::Ecom2 { .. } => ECOM2,
            Self::EmpiricalRic { .. } => EMPIRICAL_RIC,
            Self::Solid { .. } => SOLID,
            Self::Ocean { .. } => OCEAN,
        }
    }

    pub fn acceleration(
        &self,
        state: [f64; 6],
        stage: &PrecisionStage,
        mu: f64,
        mass: f64,
        reflectivity: f64,
    ) -> Result<[f64; 3], String> {
        #[cfg(feature = "pro")]
        if let Self::Ecom2 { coefficients } = self {
            return crate::ecom::acceleration(state, stage.sun, *coefficients);
        }
        if let Self::EmpiricalRic {
            coefficients,
            cosine,
            sine,
            omega,
            reference,
        } = self
        {
            let (sn, cs) = (omega * (stage.elapsed - reference)).sin_cos();
            let a: [f64; 3] =
                std::array::from_fn(|i| coefficients[i] + cosine[i] * cs + sine[i] * sn);
            let r: [f64; 3] = state[..3].try_into().unwrap();
            let v: [f64; 3] = state[3..].try_into().unwrap();
            let cross = |a: [f64; 3], b: [f64; 3]| {
                [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ]
            };
            let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let h = cross(r, v);
            let rn = norm(r);
            let hn = norm(h);
            if rn <= 0. || hn <= 0. {
                return Err(
                    "Empirical RIC frame requires nonzero radius and angular momentum".into(),
                );
            }
            let radial = r.map(|x| x / rn);
            let normal = h.map(|x| x / hn);
            let transverse = cross(normal, radial);
            return Ok(std::array::from_fn(|i| {
                a[0] * radial[i] + a[1] * transverse[i] + a[2] * normal[i]
            }));
        }
        if matches!(self, Self::Schwarzschild) {
            return precision_forces::schwarzschild_acceleration(state, mu);
        }
        if let Self::Radiation {
            area,
            albedo,
            infrared,
            quadrature: q,
        } = self
        {
            if !mass.is_finite() || mass <= 0.0 || !reflectivity.is_finite() || reflectivity < 0.0 {
                return Err(
                    "Earth radiation requires positive mass and nonnegative finite area/Cr.".into(),
                );
            }
            if *area == 0.0 || reflectivity == 0.0 {
                return Ok([0.0; 3]);
            }
            let position = mv(&stage.rotation, state[..3].try_into().unwrap(), false);
            let (a, ir) = precision_forces::earth_radiation_components(
                position,
                stage.sun,
                stage.elapsed,
                6378.137,
                &q[0],
                &q[1],
                &q[2],
                &q[3],
                *albedo,
                *infrared,
                None,
                None,
            )?;
            let scale = area * reflectivity / mass / 1000.0;
            let a = mv(&stage.rotation, a, true);
            let ir = mv(&stage.rotation, ir, true);
            return Ok(std::array::from_fn(|i| a[i] * scale + ir[i] * scale));
        }
        let position = mv(&stage.rotation, state[..3].try_into().unwrap(), false);
        let fixed = match self {
            Self::Solid {
                zero_tide,
                pole,
                sun_mu,
                moon_mu,
            } => {
                let coefficients = solid_coefficients(stage, mu, *zero_tide, *pole, *sun_mu, *moon_mu)?;
                crate::perturbations::tidal_acceleration_fixed(position, &coefficients.0, &coefficients.1, 4, mu, stage.radius)?
            },
            Self::Ocean { context, pole } => context.acceleration(
                position,
                stage.jd_tt,
                stage.jd_ut1,
                if *pole {
                    Some((stage.pole[0], stage.pole[1]))
                } else {
                    None
                },
                mu,
                stage.radius,
            )?,
            _ => unreachable!(),
        };
        Ok(mv(&stage.rotation, fixed, true))
    }
}

#[cfg(all(test, not(feature = "pro")))]
mod public_boundary_tests {
    use super::{PrecisionForce, ECOM2};

    #[test]
    fn public_plan_rejects_private_ecom_specification() {
        assert!(PrecisionForce::new((ECOM2, vec![0.0; 7], vec![])).is_err());
    }
}
