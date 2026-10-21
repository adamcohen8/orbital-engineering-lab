//! Optional OGP-SGP4/SDP4 propagation in native TEME coordinates.
//!
//! OEL keeps TLE parsing, frame transforms, and scenario evidence in Python.
//! The pinned MIT-licensed `sgp4` crate evaluates the general-perturbations
//! equations in Rust using OEL's WGS-72 and improved-mode conventions.

use sgp4::{Constants, MinutesSinceEpoch, Orbit, PreparedPropagation, ResonanceState, WGS72};

/// Numeric OGP input independent of TLE text, so fitted mean elements use the
/// same backend without manufacturing a TLE.
#[derive(Clone, Copy, Debug)]
pub struct Elements {
    pub epoch_jd_utc: f64,
    pub mean_motion_rev_day: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub raan_deg: f64,
    pub argp_deg: f64,
    pub mean_anomaly_deg: f64,
    pub bstar: f64,
}

impl Elements {
    pub fn from_array(values: [f64; 8]) -> Result<Self, String> {
        if !values.iter().all(|value| value.is_finite()) {
            return Err("OGP elements must be finite".into());
        }
        let result = Self {
            epoch_jd_utc: values[0],
            mean_motion_rev_day: values[1],
            eccentricity: values[2],
            inclination_deg: values[3],
            raan_deg: values[4],
            argp_deg: values[5],
            mean_anomaly_deg: values[6],
            bstar: values[7],
        };
        if result.epoch_jd_utc <= 0.0 || result.mean_motion_rev_day <= 0.0 {
            return Err("OGP epoch and mean motion must be positive".into());
        }
        if !(0.0..1.0).contains(&result.eccentricity)
            || !(0.0..=180.0).contains(&result.inclination_deg)
        {
            return Err("OGP eccentricity or inclination is out of range".into());
        }
        Ok(result)
    }
}

pub struct Propagator {
    constants: PreparedPropagation,
    resonance: Option<ResonanceState>,
    last_t: Option<f64>,
}

impl Propagator {
    pub fn new(elements: Elements) -> Result<Self, String> {
        let orbit = Orbit::from_kozai_elements(
            &WGS72,
            elements.inclination_deg.to_radians(),
            elements.raan_deg.to_radians(),
            elements.eccentricity,
            elements.argp_deg.to_radians(),
            elements.mean_anomaly_deg.to_radians(),
            elements.mean_motion_rev_day * std::f64::consts::PI / 720.0,
        )
        .map_err(|error| error.to_string())?;
        let constants = Constants::new(
            WGS72,
            sgp4::iau_epoch_to_sidereal_time,
            (elements.epoch_jd_utc - 2_451_545.0) / 365.25,
            elements.bstar,
            orbit,
        )
        .map_err(|error| error.to_string())?
        .into_prepared();
        let resonance = constants.initial_state();
        Ok(Self {
            constants,
            resonance,
            last_t: None,
        })
    }

    pub fn propagate(&mut self, tsince_min: f64) -> Result<([f64; 3], [f64; 3]), String> {
        if !tsince_min.is_finite() {
            return Err("OGP time offset must be finite".into());
        }
        let mut resonance = if self.last_t.is_some_and(|last| {
            tsince_min.signum() != last.signum() || tsince_min.abs() < last.abs()
        }) {
            self.constants.initial_state()
        } else {
            self.resonance
        };
        // Deep-space propagation mutates resonance state before it can return
        // an error. Keep that mutation local until the full query succeeds so
        // a failed sample cannot poison a later query on this context.
        let prediction = self
            .constants
            .propagate_from_state(MinutesSinceEpoch(tsince_min), resonance.as_mut(), false)
            .map_err(|error| error.to_string())?;
        self.resonance = resonance;
        self.last_t = Some(tsince_min);
        Ok((prediction.position, prediction.velocity))
    }
}

pub struct BatchPrediction {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub errors: Vec<(usize, String)>,
    pub workers_used: usize,
}

/// Below this object/sample work count, thread startup costs dominate.
pub const MIN_PARALLEL_BATCH_SAMPLES: usize = 4096;

pub fn propagate_batch(
    numeric_elements: &[f64],
    times_min: &[f64],
    samples_per_object: usize,
) -> Result<BatchPrediction, String> {
    propagate_batch_with_workers(numeric_elements, times_min, samples_per_object, 1)
}

/// Each worker owns complete object rows, including mutable resonance state.
/// Python supplies an execution-local upper bound; no global pool is changed.
pub fn propagate_batch_with_workers(
    numeric_elements: &[f64],
    times_min: &[f64],
    samples_per_object: usize,
    requested_workers: usize,
) -> Result<BatchPrediction, String> {
    if requested_workers == 0 {
        return Err("OGP batch workers must be positive".into());
    }
    if numeric_elements.is_empty() || numeric_elements.len() % 8 != 0 || samples_per_object == 0 {
        return Err("OGP batch requires element rows and time samples".into());
    }
    let objects = numeric_elements.len() / 8;
    if Some(times_min.len()) != objects.checked_mul(samples_per_object) {
        return Err("OGP batch time grid has an inconsistent shape".into());
    }
    let output_len = times_min
        .len()
        .checked_mul(3)
        .ok_or("OGP batch output size overflow")?;
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let workers = if times_min.len() < MIN_PARALLEL_BATCH_SAMPLES {
        1
    } else {
        requested_workers.min(objects).min(available)
    };
    let mut result = BatchPrediction {
        positions: vec![0.0; output_len],
        velocities: vec![0.0; output_len],
        errors: Vec::new(),
        workers_used: workers,
    };
    if workers == 1 {
        result.errors = propagate_object_rows(
            numeric_elements,
            times_min,
            samples_per_object,
            0,
            &mut result.positions,
            &mut result.velocities,
        );
    } else {
        result.errors = std::thread::scope(|scope| -> Result<_, String> {
            let mut positions = result.positions.as_mut_slice();
            let mut velocities = result.velocities.as_mut_slice();
            let mut first_object = 0;
            let mut handles = Vec::with_capacity(workers);
            for worker in 0..workers {
                let count = objects / workers + usize::from(worker < objects % workers);
                let first_sample = first_object * samples_per_object;
                let sample_count = count * samples_per_object;
                let elements = &numeric_elements[first_object * 8..(first_object + count) * 8];
                let times = &times_min[first_sample..first_sample + sample_count];
                let (worker_positions, rest) = positions.split_at_mut(sample_count * 3);
                positions = rest;
                let (worker_velocities, rest) = velocities.split_at_mut(sample_count * 3);
                velocities = rest;
                let handle = std::thread::Builder::new()
                    .name(format!("oel-ogp-{worker}"))
                    .spawn_scoped(scope, move || {
                        propagate_object_rows(
                            elements,
                            times,
                            samples_per_object,
                            first_sample,
                            worker_positions,
                            worker_velocities,
                        )
                    })
                    .map_err(|error| format!("OGP worker creation failed: {error}"))?;
                handles.push(handle);
                first_object += count;
            }
            let mut errors = Vec::new();
            // Join in object order, independent of completion order.
            for handle in handles {
                errors.extend(
                    handle
                        .join()
                        .map_err(|_| "OGP worker panicked".to_owned())?,
                );
            }
            Ok(errors)
        })?;
    }
    Ok(result)
}

fn propagate_object_rows(
    numeric_elements: &[f64],
    times_min: &[f64],
    samples_per_object: usize,
    sample_offset: usize,
    positions: &mut [f64],
    velocities: &mut [f64],
) -> Vec<(usize, String)> {
    let objects = numeric_elements.len() / 8;
    let mut errors = Vec::new();
    for object in 0..objects {
        let values: [f64; 8] = numeric_elements[object * 8..object * 8 + 8]
            .try_into()
            .unwrap();
        let mut propagator = match Elements::from_array(values).and_then(Propagator::new) {
            Ok(value) => value,
            Err(error) => {
                for sample in 0..samples_per_object {
                    errors.push((
                        sample_offset + object * samples_per_object + sample,
                        error.clone(),
                    ));
                }
                continue;
            }
        };
        for sample in 0..samples_per_object {
            let index = object * samples_per_object + sample;
            match propagator.propagate(times_min[index]) {
                Ok((position, velocity)) => {
                    positions[index * 3..index * 3 + 3].copy_from_slice(&position);
                    velocities[index * 3..index * 3 + 3].copy_from_slice(&velocity);
                }
                Err(error) => errors.push((sample_offset + index, error)),
            }
        }
    }
    errors
}

/// Immutable observations and validated lower Cholesky factors, prepared once
/// per fit. Only candidate elements cross the Python boundary per evaluation.
pub struct ObservationBatch {
    times_jd_utc: Vec<f64>,
    observations: Vec<f64>,
    factors: Vec<f64>,
}

impl ObservationBatch {
    pub fn new(
        times_jd_utc: Vec<f64>,
        observations: Vec<f64>,
        factors: Vec<f64>,
    ) -> Result<Self, String> {
        let count = times_jd_utc.len();
        if count == 0 || observations.len() != 6 * count || factors.len() != 36 * count {
            return Err("OGP observation batch arrays have inconsistent shapes".into());
        }
        if !times_jd_utc
            .iter()
            .chain(&observations)
            .chain(&factors)
            .all(|value| value.is_finite())
        {
            return Err("OGP observation batch values must be finite".into());
        }
        for sample in 0..count {
            let factor = &factors[sample * 36..sample * 36 + 36];
            for row in 0..6 {
                if factor[row * 6 + row] <= 0.0 {
                    return Err("OGP whitening factor diagonal must be positive".into());
                }
                for column in row + 1..6 {
                    if factor[row * 6 + column] != 0.0 {
                        return Err("OGP whitening factors must be lower triangular".into());
                    }
                }
            }
        }
        Ok(Self {
            times_jd_utc,
            observations,
            factors,
        })
    }

    pub fn residuals(&self, elements: Elements) -> Result<Vec<f64>, String> {
        let mut propagator = Propagator::new(elements)?;
        let mut result = Vec::with_capacity(self.observations.len());
        for (sample, jd_utc) in self.times_jd_utc.iter().enumerate() {
            let (position, velocity) =
                propagator.propagate((*jd_utc - elements.epoch_jd_utc) * 1440.0)?;
            let observed = &self.observations[sample * 6..sample * 6 + 6];
            let predicted = [
                position[0],
                position[1],
                position[2],
                velocity[0],
                velocity[1],
                velocity[2],
            ];
            let factor = &self.factors[sample * 36..sample * 36 + 36];
            let mut whitened = [0.0; 6];
            for row in 0..6 {
                let mut residual = predicted[row] - observed[row];
                for column in 0..row {
                    residual -= factor[row * 6 + column] * whitened[column];
                }
                whitened[row] = residual / factor[row * 6 + row];
                if !whitened[row].is_finite() {
                    return Err("OGP residual block must be finite".into());
                }
            }
            result.extend_from_slice(&whitened);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{propagate_batch, propagate_batch_with_workers, Elements, Propagator};

    fn high_eccentric_resonant_elements() -> Elements {
        Elements::from_array([
            2_453_911.83215444,
            2.00491383,
            0.99,
            90.0,
            279.0717,
            0.0,
            0.0,
            0.00011873,
        ])
        .unwrap()
    }

    #[test]
    fn parallel_batch_preserves_bits_resonance_queries_and_error_order() {
        let mut elements = Vec::new();
        let mut times = Vec::new();
        let queries = [2880.0, 60.0, -1440.0, -720.0, 0.0, 720.0, f64::NAN];
        for object in 0..16 {
            let mut row = [
                2_460_310.5,
                if object % 2 == 0 { 15.5 } else { 1.0027 },
                0.0001,
                0.05,
                80.0,
                10.0,
                20.0,
                0.0,
            ];
            if object == 3 || object == 11 {
                row[2] = 1.1;
            }
            elements.extend_from_slice(&row);
            times.extend((0..513).map(|sample| queries[sample % queries.len()]));
        }
        let serial = propagate_batch(&elements, &times, 513).unwrap();
        for workers in [1, 2, 4, usize::MAX] {
            let parallel = propagate_batch_with_workers(&elements, &times, 513, workers).unwrap();
            assert_eq!(parallel.positions, serial.positions);
            assert_eq!(parallel.velocities, serial.velocities);
            assert_eq!(parallel.errors, serial.errors);
            assert!(parallel.errors.windows(2).all(|pair| pair[0].0 < pair[1].0));
            assert!(parallel.workers_used <= workers.min(16));
            if workers > 1 && std::thread::available_parallelism().unwrap().get() > 1 {
                assert!(parallel.workers_used > 1);
            }
        }
    }

    #[test]
    fn parallel_batch_keeps_small_or_single_object_work_serial_and_checks_inputs() {
        let row = [2_460_310.5, 15.5, 0.0001, 51.0, 80.0, 10.0, 20.0, 0.0];
        let small = propagate_batch_with_workers(&row.repeat(8), &[0.0; 8], 1, 4).unwrap();
        assert_eq!(small.workers_used, 1);
        let long = propagate_batch_with_workers(&row, &vec![0.0; 4096], 4096, 4).unwrap();
        assert_eq!(long.workers_used, 1);
        assert!(propagate_batch_with_workers(&row, &[0.0], 1, 0).is_err());
        assert!(propagate_batch_with_workers(&row, &[0.0], 2, 4).is_err());
        assert!(propagate_batch_with_workers(&[], &[], 0, 4).is_err());
    }

    #[test]
    fn prepared_inclination_preserves_original_constants_propagation() {
        for mean_motion in [15.3, 1.0027, 2.0] {
            let orbit = sgp4::Orbit::from_kozai_elements(
                &sgp4::WGS72,
                0.4,
                1.2,
                0.001,
                0.5,
                0.7,
                mean_motion * std::f64::consts::PI / 720.0,
            )
            .unwrap();
            let constants = sgp4::Constants::new(
                sgp4::WGS72,
                sgp4::iau_epoch_to_sidereal_time,
                (2_460_310.5 - 2_451_545.0) / 365.25,
                0.0001,
                orbit,
            )
            .unwrap();
            let prepared = constants.clone().into_prepared();
            for compatibility_mode in [false, true] {
                for time in [-2880.0, -720.0, 0.0, 60.0, 720.0, 2880.0] {
                    let actual = prepared.propagate_from_state(
                        sgp4::MinutesSinceEpoch(time),
                        prepared.initial_state().as_mut(),
                        compatibility_mode,
                    );
                    let expected = constants.propagate_from_state(
                        sgp4::MinutesSinceEpoch(time),
                        constants.initial_state().as_mut(),
                        compatibility_mode,
                    );
                    match (actual, expected) {
                        (Ok(actual), Ok(expected)) => {
                            for axis in 0..3 {
                                assert_eq!(
                                    actual.position[axis].to_bits(),
                                    expected.position[axis].to_bits()
                                );
                                assert_eq!(
                                    actual.velocity[axis].to_bits(),
                                    expected.velocity[axis].to_bits()
                                );
                            }
                        }
                        (Err(actual), Err(expected)) => {
                            assert_eq!(actual.to_string(), expected.to_string())
                        }
                        _ => panic!("prepared and original propagation must agree"),
                    }
                }
            }
        }
    }

    #[test]
    fn resonance_restarts_for_queries_toward_and_across_epoch() {
        let elements =
            Elements::from_array([2_460_310.5, 1.0027, 0.0001, 0.05, 80.0, 10.0, 20.0, 0.0])
                .unwrap();
        let mut reused = Propagator::new(elements).unwrap();
        for time in [-2880.0, -1440.0, -720.0, 0.0, 720.0, 0.0] {
            let actual = reused.propagate(time).unwrap();
            let expected = Propagator::new(elements).unwrap().propagate(time).unwrap();
            for axis in 0..3 {
                assert!((actual.0[axis] - expected.0[axis]).abs() < 1.0e-9);
                assert!((actual.1[axis] - expected.1[axis]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn failed_resonant_query_does_not_poison_an_earlier_valid_query() {
        let elements = high_eccentric_resonant_elements();
        let mut reused = Propagator::new(elements).unwrap();

        reused.propagate(0.0).unwrap();
        let error = reused.propagate(600_000.0).unwrap_err();
        assert!(error.contains("semi-latus rectum is negative"), "{error}");

        let actual = reused.propagate(500_000.0).unwrap();
        let expected = Propagator::new(elements)
            .unwrap()
            .propagate(500_000.0)
            .unwrap();
        for axis in 0..3 {
            assert!((actual.0[axis] - expected.0[axis]).abs() < 1.0e-9);
            assert!((actual.1[axis] - expected.1[axis]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn failed_resonant_batch_sample_remains_sparse_and_continues() {
        let elements = high_eccentric_resonant_elements();
        let values = [
            elements.epoch_jd_utc,
            elements.mean_motion_rev_day,
            elements.eccentricity,
            elements.inclination_deg,
            elements.raan_deg,
            elements.argp_deg,
            elements.mean_anomaly_deg,
            elements.bstar,
        ];
        let batch = propagate_batch(&values, &[0.0, 600_000.0, 500_000.0], 3).unwrap();

        assert_eq!(batch.errors.len(), 1);
        assert_eq!(batch.errors[0].0, 1);
        assert!(batch.errors[0].1.contains("semi-latus rectum is negative"));
        assert_eq!(&batch.positions[3..6], &[0.0; 3]);
        assert_eq!(&batch.velocities[3..6], &[0.0; 3]);

        let expected = Propagator::new(elements)
            .unwrap()
            .propagate(500_000.0)
            .unwrap();
        for axis in 0..3 {
            assert!((batch.positions[6 + axis] - expected.0[axis]).abs() < 1.0e-9);
            assert!((batch.velocities[6 + axis] - expected.1[axis]).abs() < 1.0e-12);
        }
    }
}
