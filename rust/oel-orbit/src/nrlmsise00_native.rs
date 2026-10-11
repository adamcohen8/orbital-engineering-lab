//! OEL input contract for the vendored NRLMSISE-00 numerical core.

use crate::nrlmsise00_model::{gtd7d, NrlmsiseFlags, NrlmsiseInput, NrlmsiseOutput};

/// Packed order: day, second, altitude, latitude, longitude, local solar time,
/// F10.7A, F10.7, daily Ap, then the seven historical Ap entries.
pub(crate) fn density(values: [f64; 16]) -> Result<f64, String> {
    if values.iter().any(|value| !value.is_finite())
        || values[0].fract() != 0.0
        || !(1.0..=366.0).contains(&values[0])
        || !(0.0..=86400.0).contains(&values[1])
        || values[2] < 0.0
        || !(-90.0..=90.0).contains(&values[3])
    {
        return Err("native NRLMSISE-00 requires finite nonnegative-altitude inputs".into());
    }
    let mut input = NrlmsiseInput::default();
    input.doy = values[0] as i32;
    input.sec = values[1];
    input.alt = values[2];
    input.g_lat = values[3];
    input.g_lon = values[4];
    input.lst = values[5];
    input.f107a = values[6];
    input.f107 = values[7];
    input.ap = values[8];
    input.ap_array.copy_from_slice(&values[9..16]);
    let mut flags = NrlmsiseFlags::default();
    flags.switches = [1; 24];
    flags.switches[0] = 0; // Reference uses g/cm^3 internally.
    flags.switches[9] = -1; // Historical Ap array.
    let mut output = NrlmsiseOutput::default();
    gtd7d(&mut input, &mut flags, &mut output);
    let result = (output.d[5] * 1000.0).max(0.0);
    if !result.is_finite() {
        return Err("native NRLMSISE-00 returned non-finite density".into());
    }
    Ok(result)
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::{exceptions::PyValueError, prelude::*};

    #[pyfunction]
    fn environment_nrlmsise00_density(values: Vec<f64>) -> PyResult<f64> {
        let input: [f64; 16] = values
            .try_into()
            .map_err(|_| PyValueError::new_err("NRLMSISE-00 requires 16 prepared values"))?;
        density(input).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    fn environment_nrlmsise00_low_altitude_density(values: Vec<f64>) -> PyResult<f64> {
        environment_nrlmsise00_density(values)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(environment_nrlmsise00_density, module)?)?;
        module.add_function(wrap_pyfunction!(environment_nrlmsise00_low_altitude_density, module)?)
    }
}

#[cfg(feature = "python")]
pub(crate) use python::register;

#[cfg(test)]
mod tests {
    use super::density;

    #[test]
    fn high_altitude_quiet_and_disturbed_match_reference_values() {
        let mut input = [
            91.0, 43200.0, 400.0, 20.0, 30.0, 14.0, 150.0, 150.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
            4.0, 4.0,
        ];
        let quiet = density(input).unwrap();
        assert!((quiet / 7.068356430115259e-12 - 1.0).abs() < 5.0e-14);
        input[8..].fill(40.0);
        let disturbed = density(input).unwrap();
        assert!((disturbed / 8.902212997965044e-12 - 1.0).abs() < 5.0e-14);
        input[2] = -1.0;
        assert!(density(input).is_err());
    }

    #[test]
    fn low_altitude_disturbed_matches_reference_value() {
        let mut input = [
            264.0, 11700.0, 95.0, 20.0, 30.0, 14.0, 150.0, 150.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ];
        let disturbed = density(input).unwrap();
        assert!((disturbed / 1.2695327861286798e-6 - 1.0).abs() < 5.0e-14);
        input[2] = 0.0;
        assert!(density(input).unwrap().is_finite());
    }

    fn canonical_species(
        alt: f64,
        fluxa: f64,
        flux: f64,
        ap: f64,
        mixing: i32,
        species: usize,
    ) -> f64 {
        use crate::nrlmsise00_model::{gtd7d, NrlmsiseFlags, NrlmsiseInput, NrlmsiseOutput};
        let mut input = NrlmsiseInput::default();
        input.doy = 90;
        input.sec = 0.0;
        input.alt = alt;
        input.g_lat = 0.12311480767211523;
        input.g_lon = 171.9327590811191;
        input.lst = 11.39003115622072;
        input.f107a = fluxa;
        input.f107 = flux;
        input.ap = ap;
        input.ap_array.fill(ap);
        let mut flags = NrlmsiseFlags::default();
        flags.switches = [1; 24];
        flags.switches[0] = 0;
        flags.switches[9] = -1;
        flags.switches[15] = mixing;
        let mut output = NrlmsiseOutput::default();
        gtd7d(&mut input, &mut flags, &mut output);
        output.d[species]
    }
    #[test]
    fn canonical_species_cutoffs_include_equality() {
        for (fluxa, flux, ap) in [(85.0, 90.0, 4.0), (150.0, 175.0, 40.0)] {
            for (species, cutoff) in [(0, 200.0), (1, 300.0)] {
                for alt in [cutoff - 0.000001, cutoff] {
                    assert_ne!(
                        canonical_species(alt, fluxa, flux, ap, 1, species),
                        canonical_species(alt, fluxa, flux, ap, 0, species)
                    );
                }
                assert_eq!(
                    canonical_species(cutoff + 0.000001, fluxa, flux, ap, 1, species),
                    canonical_species(cutoff + 0.000001, fluxa, flux, ap, 0, species)
                );
            }
        }
    }
}
