//! P.676 numerical arithmetic. Python owns edition tables, applicability,
//! broadcast validation, atmospheric ledgers and RF closure decisions.

pub fn specific(
    f: f64,
    pressure: f64,
    temperature: f64,
    density: f64,
    oxygen: &[[f64; 7]],
    water: &[[f64; 7]],
) -> Result<[f64; 2], &'static str> {
    if [f, pressure, temperature, density]
        .iter()
        .any(|v| !v.is_finite())
        || f <= 0.0
        || f > 1000.0
        || pressure <= 0.0
        || temperature <= 0.0
        || density < 0.0
    {
        return Err("invalid P.676 specific attenuation inputs");
    }
    let th = 300.0 / temperature;
    let e = density * temperature / 216.7;
    let p = pressure - e;
    if p <= 0.0 {
        return Err("vapour partial pressure must remain below total pressure");
    }
    let mut refractivity_o = 0.0;
    for &[f0, a1, a2, a3, a4, a5, a6] in oxygen {
        let strength = a1 * 1.0e-7 * p * th.powf(3.0) * (a2 * (1.0 - th)).exp();
        let width = a3 * 1.0e-4 * (p * th.powf(0.8 - a4) + 1.1 * e * th);
        let width = (width * width + 2.25e-6).sqrt();
        let delta = (a5 + a6 * th) * 1.0e-4 * (p + e) * th.powf(0.8);
        let shape = (f / f0)
            * ((width - delta * (f0 - f)) / ((f0 - f).powi(2) + width * width)
                + (width - delta * (f0 + f)) / ((f0 + f).powi(2) + width * width));
        refractivity_o += strength * shape;
    }
    let debye = 5.6e-4 * (p + e) * th.powf(0.8);
    let continuum = f
        * p
        * th.powf(2.0)
        * (6.14e-5 / (debye * (1.0 + (f / debye).powi(2)))
            + 1.4e-12 * p * th.powf(1.5) / (1.0 + 1.9e-5 * f.powf(1.5)));
    let mut refractivity_w = 0.0;
    for &[f0, b1, b2, b3, b4, b5, b6] in water {
        let strength = b1 * 1.0e-1 * e * th.powf(3.5) * (b2 * (1.0 - th)).exp();
        let base = b3 * 1.0e-4 * (p * th.powf(b4) + b5 * e * th.powf(b6));
        let width = 0.535 * base + (0.217 * base * base + 2.1316e-12 * f0 * f0 / th).sqrt();
        let shape = (f / f0)
            * (width / ((f0 - f).powi(2) + width * width)
                + width / ((f0 + f).powi(2) + width * width));
        refractivity_w += strength * shape;
    }
    Ok([
        0.1820 * f * (refractivity_o + continuum),
        0.1820 * f * refractivity_w,
    ])
}

pub fn slant(row: &[f64; 11]) -> Result<[f64; 4], &'static str> {
    let [f, elevation, p, t, density, oxygen, water, c0, c1, c2, c3] = *row;
    if row.iter().any(|v| !v.is_finite())
        || !(1.0..=350.0).contains(&f)
        || !(5.0..=90.0).contains(&elevation)
    {
        return Err("invalid P.676 slant attenuation inputs");
    }
    let ho = c0 + c1 * t + c2 * p + c3 * density;
    let mut hw = 5.6585e-5 * f + 1.8348;
    for (line, numerator, denominator) in [
        (22.235080, 2.6846, 2.7649),
        (183.310087, 5.8905, 4.9219),
        (325.152888, 2.9810, 3.0748),
    ] {
        hw += numerator / ((f - line).powi(2) + denominator);
    }
    let sine = elevation.to_radians().sin();
    Ok([oxygen * ho / sine, water * hw / sine, ho, hw])
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use crate::numeric_bytes::{decode_f64_bytes, output_bytes};
    use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

    #[pyclass(name = "P676Context")]
    struct Context {
        oxygen: Vec<[f64; 7]>,
        water: Vec<[f64; 7]>,
    }

    fn table(bytes: &[u8]) -> PyResult<Vec<[f64; 7]>> {
        let values = decode_f64_bytes(bytes)?;
        if values.is_empty()
            || values.len() % 7 != 0
            || values.len() > 7000
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(PyValueError::new_err("invalid P.676 line table"));
        }
        Ok(values
            .chunks_exact(7)
            .map(|r| r.try_into().unwrap())
            .collect())
    }

    #[pymethods]
    impl Context {
        #[new]
        fn new(oxygen: &[u8], water: &[u8]) -> PyResult<Self> {
            Ok(Self {
                oxygen: table(oxygen)?,
                water: table(water)?,
            })
        }

        fn specific_batch(&self, py: Python<'_>, cases: &[u8]) -> PyResult<Py<PyBytes>> {
            let cases = decode_f64_bytes(cases)?;
            if cases.len() % 4 != 0 || cases.len() > 4_000_000 {
                return Err(PyValueError::new_err("invalid bounded P.676 case batch"));
            }
            let out = py
                .detach(|| {
                    let mut out = Vec::with_capacity(cases.len() / 2);
                    for r in cases.chunks_exact(4) {
                        out.extend(specific(r[0], r[1], r[2], r[3], &self.oxygen, &self.water)?);
                    }
                    Ok::<_, &'static str>(out)
                })
                .map_err(PyValueError::new_err)?;
            output_bytes(py, &out)
        }

        fn slant_batch(&self, py: Python<'_>, cases: &[u8]) -> PyResult<Py<PyBytes>> {
            let cases = decode_f64_bytes(cases)?;
            if cases.len() % 11 != 0 || cases.len() > 11_000_000 {
                return Err(PyValueError::new_err("invalid bounded P.676 slant batch"));
            }
            let out = py
                .detach(|| {
                    let mut out = Vec::with_capacity(cases.len() / 11 * 4);
                    for r in cases.chunks_exact(11) {
                        out.extend(slant(r.try_into().unwrap())?);
                    }
                    Ok::<_, &'static str>(out)
                })
                .map_err(PyValueError::new_err)?;
            output_bytes(py, &out)
        }

        fn slant_prepared(
            &self,
            py: Python<'_>,
            profile: &[u8],
            elevations: &[u8],
        ) -> PyResult<Py<PyBytes>> {
            let profile = decode_f64_bytes(profile)?;
            let elevations = decode_f64_bytes(elevations)?;
            if profile.len() != 10 || elevations.len() > 1_000_000 {
                return Err(PyValueError::new_err(
                    "invalid bounded prepared P.676 slant batch",
                ));
            }
            let out = py
                .detach(|| {
                    let row = [
                        profile[0], 90.0, profile[1], profile[2], profile[3], profile[4],
                        profile[5], profile[6], profile[7], profile[8], profile[9],
                    ];
                    let heights = slant(&row)?;
                    let mut out = Vec::with_capacity(elevations.len() * 4);
                    for &elevation in elevations.iter() {
                        if !elevation.is_finite() || !(5.0..=90.0).contains(&elevation) {
                            return Err("invalid P.676 slant attenuation inputs");
                        }
                        let sine = elevation.to_radians().sin();
                        out.extend([
                            profile[4] * heights[2] / sine,
                            profile[5] * heights[3] / sine,
                            heights[2],
                            heights[3],
                        ]);
                    }
                    Ok::<_, &'static str>(out)
                })
                .map_err(PyValueError::new_err)?;
            output_bytes(py, &out)
        }
    }
    module.add_class::<Context>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_conditions_and_retains_zero_water() {
        assert!(specific(30.0, 10.0, 300.0, 100.0, &[], &[]).is_err());
        assert!(specific(1001.0, 1013.0, 300.0, 0.0, &[], &[]).is_err());
        let out = specific(30.0, 1013.0, 300.0, 0.0, &[], &[]).unwrap();
        assert!(out[0] > 0.0);
        assert_eq!(out[1], 0.0);
        assert!(slant(&[30.0, 4.0, 1013.0, 300.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).is_err());
    }
}
