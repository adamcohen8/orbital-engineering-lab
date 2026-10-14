//! Opt-in gravity of extended external bodies; existing Earth plans are unchanged.
use crate::planetary::{tdb, Ephemeris, EphemerisInput, Harmonics, Rotation, RotationInput};

pub type SourceInput = (
    i64,
    f64,
    f64,
    (usize, usize),
    (Vec<f64>, Vec<f64>),
    RotationInput,
);

struct Source {
    target: i64,
    mu: f64,
    radius: f64,
    harmonics: Harmonics,
    rotation: Rotation,
}

pub struct AdditionalGravity {
    tt: f64,
    duration: f64,
    ephemeris: Ephemeris,
    sources: Vec<Source>,
}

fn norm(r: [f64; 3]) -> f64 {
    r.iter().map(|x| x * x).sum::<f64>().sqrt()
}

impl AdditionalGravity {
    pub fn new(
        center: i64,
        tt: f64,
        duration: f64,
        ephemeris: EphemerisInput,
        inputs: Vec<SourceInput>,
    ) -> Result<Self, String> {
        if !tt.is_finite()
            || !duration.is_finite()
            || duration <= 0.
            || inputs.is_empty()
            || inputs.len() > 16
        {
            return Err(
                "additional gravity requires a finite epoch, positive duration and 1..16 sources"
                    .into(),
            );
        }
        let mut sources = Vec::new();
        for (target, mu, radius, (n, m), (c, s), rotation) in inputs {
            if target == center
                || target == 10
                || sources.iter().any(|v: &Source| v.target == target)
                || !mu.is_finite()
                || mu <= 0.
                || !radius.is_finite()
                || radius <= 0.
                || n < 2
            {
                return Err("invalid, duplicate or self additional gravity source".into());
            }
            sources.push(Source {
                target,
                mu,
                radius,
                harmonics: Harmonics::new(n, m, c, s)?,
                rotation: Rotation::new(rotation)?,
            });
        }
        let result = Self {
            tt,
            duration,
            ephemeris: Ephemeris::new(ephemeris)?,
            sources,
        };
        for source in &result.sources {
            source.rotation.evaluate(tt)?;
            source.rotation.evaluate(tt + duration)?;
            result.ephemeris.position(source.target, tdb(tt).0)?;
            result
                .ephemeris
                .position(source.target, tdb(tt + duration).0)?;
        }
        Ok(result)
    }

    pub fn components(
        &self,
        t: f64,
        r: [f64; 3],
        rotations: &[Option<[f64; 9]>],
    ) -> Result<Vec<[f64; 3]>, String> {
        if !t.is_finite()
            || t < -1e-9
            || t > self.duration + 1e-9
            || !r.iter().all(|x| x.is_finite())
            || rotations.len() != self.sources.len()
        {
            return Err(
                "additional gravity stage outside declared domain or invalid inputs".into(),
            );
        }
        self.sources.iter().zip(rotations).map(|(source, override_rotation)| {
            let body = self.ephemeris.position(source.target, tdb(self.tt + t).0)?;
            let relative = std::array::from_fn(|i| r[i] - body[i]);
            let origin = body.map(|v| -v);
            let d = norm(relative);
            let b = norm(origin);
            if d <= source.radius || b <= source.radius {
                return Err("additional harmonic gravity requires spacecraft and origin outside the source reference sphere".into());
            }
            let rotation = match override_rotation { Some(r) => *r, None => source.rotation.evaluate(self.tt + t)?.0 };
            let satellite_harmonic = source.harmonics.acceleration(relative, rotation, source.mu, source.radius);
            let origin_harmonic = source.harmonics.acceleration(origin, rotation, source.mu, source.radius);
            let part = std::array::from_fn(|i| -source.mu * (relative[i] / d.powi(3) - origin[i] / b.powi(3)) + satellite_harmonic[i] - origin_harmonic[i]);
            if !part.iter().all(|x| x.is_finite()) { return Err("nonfinite additional gravity acceleration".into()); }
            Ok(part)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan() -> AdditionalGravity {
        let rotation = (vec![2], vec![0., 0., 0.], vec![], vec![-1e12, 1e12], false);
        AdditionalGravity::new(
            399,
            0.,
            100.,
            (
                vec![-100., 200.],
                vec![301],
                vec![vec![
                    384400., 0., 0., 0., 0., 0., 384400., 0., 0., 0., 0., 0.,
                ]],
            ),
            vec![(
                301,
                4902.8,
                1737.4,
                (2, 2),
                (vec![0.; 9], vec![0.; 9]),
                rotation,
            )],
        )
        .unwrap()
    }
    #[test]
    fn point_mass_limit_and_origin_cancellation() {
        let p = plan();
        assert_eq!(p.components(0., [0.; 3], &[None]).unwrap()[0], [0.; 3]);
        let got = p.components(0., [7000., 0., 0.], &[None]).unwrap()[0];
        let want = 4902.8 * (1. / (384400_f64 - 7000.).powi(2) - 1. / 384400_f64.powi(2));
        assert!((got[0] - want).abs() < 1e-22);
        assert_eq!(got[1], 0.);
        assert!(p.components(0., [384400., 0., 0.], &[None]).is_err());
        assert!(p.components(101., [7000., 0., 0.], &[None]).is_err());
    }
}
