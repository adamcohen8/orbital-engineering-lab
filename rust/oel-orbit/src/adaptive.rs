//! OEL RKF78 tableau and adaptive acceptance semantics, without fast math.

#[derive(Clone, Debug, Default)]
pub struct StepInfo {
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub attempted_steps: usize,
    pub min_step_s: Option<f64>,
    pub max_step_s: Option<f64>,
    pub final_step_s: Option<f64>,
    pub suggested_next_step_s: Option<f64>,
    pub max_error_ratio: f64,
}

const C: [f64; 13] = [
    0.0,
    2.0 / 27.0,
    1.0 / 9.0,
    1.0 / 6.0,
    5.0 / 12.0,
    0.5,
    5.0 / 6.0,
    1.0 / 6.0,
    2.0 / 3.0,
    1.0 / 3.0,
    1.0,
    0.0,
    1.0,
];

const A: [&[(usize, f64)]; 13] = [
    &[],
    &[(0, 2.0 / 27.0)],
    &[(0, 1.0 / 36.0), (1, 1.0 / 12.0)],
    &[(0, 1.0 / 24.0), (2, 1.0 / 8.0)],
    &[(0, 5.0 / 12.0), (2, -25.0 / 16.0), (3, 25.0 / 16.0)],
    &[(0, 1.0 / 20.0), (3, 0.25), (4, 0.2)],
    &[
        (0, -25.0 / 108.0),
        (3, 125.0 / 108.0),
        (4, -65.0 / 27.0),
        (5, 125.0 / 54.0),
    ],
    &[
        (0, 31.0 / 300.0),
        (4, 61.0 / 225.0),
        (5, -2.0 / 9.0),
        (6, 13.0 / 900.0),
    ],
    &[
        (0, 2.0),
        (3, -53.0 / 6.0),
        (4, 704.0 / 45.0),
        (5, -107.0 / 9.0),
        (6, 67.0 / 90.0),
        (7, 3.0),
    ],
    &[
        (0, -91.0 / 108.0),
        (3, 23.0 / 108.0),
        (4, -976.0 / 135.0),
        (5, 311.0 / 54.0),
        (6, -19.0 / 60.0),
        (7, 17.0 / 6.0),
        (8, -1.0 / 12.0),
    ],
    &[
        (0, 2383.0 / 4100.0),
        (3, -341.0 / 164.0),
        (4, 4496.0 / 1025.0),
        (5, -301.0 / 82.0),
        (6, 2133.0 / 4100.0),
        (7, 45.0 / 82.0),
        (8, 45.0 / 164.0),
        (9, 18.0 / 41.0),
    ],
    &[
        (0, 3.0 / 205.0),
        (5, -6.0 / 41.0),
        (6, -3.0 / 205.0),
        (7, -3.0 / 41.0),
        (8, 3.0 / 41.0),
        (9, 6.0 / 41.0),
    ],
    &[
        (0, -1777.0 / 4100.0),
        (3, -341.0 / 164.0),
        (4, 4496.0 / 1025.0),
        (5, -289.0 / 82.0),
        (6, 2193.0 / 4100.0),
        (7, 51.0 / 82.0),
        (8, 33.0 / 164.0),
        (9, 12.0 / 41.0),
        (11, 1.0),
    ],
];

const B: &[(usize, f64)] = &[
    (0, 41.0 / 840.0),
    (5, 34.0 / 105.0),
    (6, 9.0 / 35.0),
    (7, 9.0 / 35.0),
    (8, 9.0 / 280.0),
    (9, 9.0 / 280.0),
    (10, 41.0 / 840.0),
];

fn combine<const N: usize>(
    x: [f64; N],
    h: f64,
    ks: &[[f64; N]; 13],
    terms: &[(usize, f64)],
) -> [f64; N] {
    let mut weighted = [0.0; N];
    for &(stage, coefficient) in terms {
        for (value, &rate) in weighted.iter_mut().zip(&ks[stage]) {
            *value += coefficient * rate;
        }
    }
    std::array::from_fn(|axis| x[axis] + h * weighted[axis])
}

fn rkf78_step<F, const N: usize>(
    x: [f64; N],
    t: f64,
    h: f64,
    derivative: &mut F,
) -> Result<([f64; N], [f64; N]), String>
where
    F: FnMut(f64, [f64; N]) -> Result<[f64; N], String>,
{
    let mut ks = [[0.0; N]; 13];
    for stage in 0..13 {
        let stage_x = combine(x, h, &ks, A[stage]);
        ks[stage] = derivative(t + h * C[stage], stage_x)?;
    }
    let next = combine(x, h, &ks, B);
    let error = std::array::from_fn(|axis| {
        h * (41.0 / 840.0) * (ks[0][axis] + ks[10][axis] - ks[11][axis] - ks[12][axis])
    });
    Ok((next, error))
}

fn dopri5_step<F, const N: usize>(
    x: [f64; N],
    t: f64,
    h: f64,
    derivative: &mut F,
) -> Result<([f64; N], [f64; N]), String>
where
    F: FnMut(f64, [f64; N]) -> Result<[f64; N], String>,
{
    let stage = |terms: &[(f64, &[f64; N])]| -> [f64; N] {
        let mut weighted = [0.0; N];
        for &(coefficient, values) in terms {
            for (value, &rate) in weighted.iter_mut().zip(values) {
                *value += coefficient * rate;
            }
        }
        std::array::from_fn(|axis| x[axis] + h * weighted[axis])
    };
    let k1 = derivative(t, x)?;
    let k2 = derivative(t + h / 5.0, stage(&[(1.0 / 5.0, &k1)]))?;
    let k3 = derivative(
        t + h * 3.0 / 10.0,
        stage(&[(3.0 / 40.0, &k1), (9.0 / 40.0, &k2)]),
    )?;
    let k4 = derivative(
        t + h * 4.0 / 5.0,
        stage(&[(44.0 / 45.0, &k1), (-56.0 / 15.0, &k2), (32.0 / 9.0, &k3)]),
    )?;
    let k5 = derivative(
        t + h * 8.0 / 9.0,
        stage(&[
            (19372.0 / 6561.0, &k1),
            (-25360.0 / 2187.0, &k2),
            (64448.0 / 6561.0, &k3),
            (-212.0 / 729.0, &k4),
        ]),
    )?;
    let k6 = derivative(
        t + h,
        stage(&[
            (9017.0 / 3168.0, &k1),
            (-355.0 / 33.0, &k2),
            (46732.0 / 5247.0, &k3),
            (49.0 / 176.0, &k4),
            (-5103.0 / 18656.0, &k5),
        ]),
    )?;
    let fifth = stage(&[
        (35.0 / 384.0, &k1),
        (500.0 / 1113.0, &k3),
        (125.0 / 192.0, &k4),
        (-2187.0 / 6784.0, &k5),
        (11.0 / 84.0, &k6),
    ]);
    let k7 = derivative(t + h, fifth)?;
    let fourth = stage(&[
        (5179.0 / 57600.0, &k1),
        (7571.0 / 16695.0, &k3),
        (393.0 / 640.0, &k4),
        (-92097.0 / 339200.0, &k5),
        (187.0 / 2100.0, &k6),
        (1.0 / 40.0, &k7),
    ]);
    Ok((
        fifth,
        std::array::from_fn(|axis| fifth[axis] - fourth[axis]),
    ))
}

#[derive(Clone, Copy)]
enum Method {
    Rkf78,
    Dopri5,
}

pub fn integrate_rkf78<F, const N: usize>(
    initial: [f64; N],
    t_s: f64,
    dt_s: f64,
    atol: f64,
    rtol: f64,
    h_init: Option<f64>,
    derivative: F,
) -> Result<([f64; N], StepInfo), String>
where
    F: FnMut(f64, [f64; N]) -> Result<[f64; N], String>,
{
    integrate_adaptive(
        initial,
        t_s,
        dt_s,
        atol,
        rtol,
        h_init,
        Method::Rkf78,
        derivative,
    )
}

pub fn integrate_dopri5<F, const N: usize>(
    initial: [f64; N],
    t_s: f64,
    dt_s: f64,
    atol: f64,
    rtol: f64,
    h_init: Option<f64>,
    derivative: F,
) -> Result<([f64; N], StepInfo), String>
where
    F: FnMut(f64, [f64; N]) -> Result<[f64; N], String>,
{
    integrate_adaptive(
        initial,
        t_s,
        dt_s,
        atol,
        rtol,
        h_init,
        Method::Dopri5,
        derivative,
    )
}

#[allow(clippy::too_many_arguments)]
fn integrate_adaptive<F, const N: usize>(
    initial: [f64; N],
    t_s: f64,
    dt_s: f64,
    atol: f64,
    rtol: f64,
    h_init: Option<f64>,
    method: Method,
    mut derivative: F,
) -> Result<([f64; N], StepInfo), String>
where
    F: FnMut(f64, [f64; N]) -> Result<[f64; N], String>,
{
    if !t_s.is_finite() || !dt_s.is_finite() || dt_s < 0.0 {
        return Err("t_s must be finite and dt_s must be finite and non-negative".into());
    }
    if !atol.is_finite() || atol < 0.0 || !rtol.is_finite() || rtol < 0.0 {
        return Err("adaptive tolerances must be finite and non-negative".into());
    }
    if initial.iter().any(|value| !value.is_finite()) {
        return Err("Adaptive integrator initial state must be finite".into());
    }
    if dt_s == 0.0 {
        return Ok((initial, StepInfo::default()));
    }
    let mut t = t_s;
    let mut xk = initial;
    let mut remain = dt_s;
    let mut h = dt_s.min(h_init.filter(|value| *value > 0.0).unwrap_or(1.0));
    let min_h = 1e-12_f64.max(1e-12 * dt_s.abs().max(1.0));
    let mut info = StepInfo::default();
    let mut steps = 0;

    while remain > 0.0 && steps < 4096 {
        h = h.min(remain);
        let (next, err) = match method {
            Method::Rkf78 => rkf78_step(xk, t, h, &mut derivative)?,
            Method::Dopri5 => dopri5_step(xk, t, h, &mut derivative)?,
        };
        if next
            .iter()
            .chain(err.iter())
            .any(|value| !value.is_finite())
        {
            return Err(format!(
                "Adaptive integrator produced a non-finite state or error at t={t:.9}s."
            ));
        }
        info.attempted_steps += 1;
        let mut ratio = 0.0_f64;
        for axis in 0..N {
            let scale = (atol + rtol * xk[axis].abs().max(next[axis].abs())).max(1e-14);
            ratio = ratio.max(err[axis].abs() / scale);
        }
        info.max_error_ratio = info.max_error_ratio.max(ratio);
        let exponent = match method {
            Method::Rkf78 => -1.0 / 8.0,
            Method::Dopri5 => -1.0 / 5.0,
        };
        if ratio <= 1.0 {
            t += h;
            xk = next;
            remain -= h;
            info.accepted_steps += 1;
            info.final_step_s = Some(h);
            info.min_step_s = Some(info.min_step_s.map_or(h, |old| old.min(h)));
            info.max_step_s = Some(info.max_step_s.map_or(h, |old| old.max(h)));
            if ratio < 1e-10 {
                h *= 2.0;
            } else {
                h *= (0.9 * ratio.powf(exponent)).clamp(0.5, 2.0);
            }
        } else {
            info.rejected_steps += 1;
            h *= (0.9 * ratio.powf(exponent)).max(0.1);
            if h < min_h {
                return Err(format!("Adaptive integrator step size underflow at t={t:.9}s while trying to cover dt={dt_s:.9}s."));
            }
        }
        steps += 1;
    }
    if remain > min_h.max(1e-9 * dt_s.abs().max(1.0)) {
        return Err(format!(
            "Adaptive integrator exhausted 4096 internal substeps with {remain:.9e}s remaining."
        ));
    }
    info.suggested_next_step_s = Some(h);
    Ok((xk, info))
}

#[cfg(test)]
mod weighted_order_tests {
    use super::*;
    fn combine_axis_reference<const N: usize>(
        x: [f64; N],
        h: f64,
        ks: &[[f64; N]; 13],
        terms: &[(usize, f64)],
    ) -> [f64; N] {
        std::array::from_fn(|axis| {
            let mut weighted = 0.0;
            for &(stage, coefficient) in terms {
                weighted += coefficient * ks[stage][axis];
            }
            x[axis] + h * weighted
        })
    }

    fn dopri5_step_axis_reference<F, const N: usize>(
        x: [f64; N],
        t: f64,
        h: f64,
        derivative: &mut F,
    ) -> Result<([f64; N], [f64; N]), String>
    where
        F: FnMut(f64, [f64; N]) -> Result<[f64; N], String>,
    {
        let stage = |terms: &[(f64, [f64; N])]| -> [f64; N] {
            std::array::from_fn(|axis| {
                let mut weighted = 0.0;
                for (coefficient, values) in terms {
                    weighted += coefficient * values[axis];
                }
                x[axis] + h * weighted
            })
        };
        let k1 = derivative(t, x)?;
        let k2 = derivative(t + h / 5.0, stage(&[(1.0 / 5.0, k1)]))?;
        let k3 = derivative(
            t + h * 3.0 / 10.0,
            stage(&[(3.0 / 40.0, k1), (9.0 / 40.0, k2)]),
        )?;
        let k4 = derivative(
            t + h * 4.0 / 5.0,
            stage(&[(44.0 / 45.0, k1), (-56.0 / 15.0, k2), (32.0 / 9.0, k3)]),
        )?;
        let k5 = derivative(
            t + h * 8.0 / 9.0,
            stage(&[
                (19372.0 / 6561.0, k1),
                (-25360.0 / 2187.0, k2),
                (64448.0 / 6561.0, k3),
                (-212.0 / 729.0, k4),
            ]),
        )?;
        let k6 = derivative(
            t + h,
            stage(&[
                (9017.0 / 3168.0, k1),
                (-355.0 / 33.0, k2),
                (46732.0 / 5247.0, k3),
                (49.0 / 176.0, k4),
                (-5103.0 / 18656.0, k5),
            ]),
        )?;
        let fifth = stage(&[
            (35.0 / 384.0, k1),
            (500.0 / 1113.0, k3),
            (125.0 / 192.0, k4),
            (-2187.0 / 6784.0, k5),
            (11.0 / 84.0, k6),
        ]);
        let k7 = derivative(t + h, fifth)?;
        let fourth = stage(&[
            (5179.0 / 57600.0, k1),
            (7571.0 / 16695.0, k3),
            (393.0 / 640.0, k4),
            (-92097.0 / 339200.0, k5),
            (187.0 / 2100.0, k6),
            (1.0 / 40.0, k7),
        ]);
        Ok((
            fifth,
            std::array::from_fn(|axis| fifth[axis] - fourth[axis]),
        ))
    }

    fn rkf78_step_axis_reference<F, const N: usize>(
        x: [f64; N],
        t: f64,
        h: f64,
        derivative: &mut F,
    ) -> Result<([f64; N], [f64; N]), String>
    where
        F: FnMut(f64, [f64; N]) -> Result<[f64; N], String>,
    {
        let mut ks = [[0.0; N]; 13];
        for stage in 0..13 {
            let stage_x = combine_axis_reference(x, h, &ks, A[stage]);
            ks[stage] = derivative(t + h * C[stage], stage_x)?;
        }
        let next = combine_axis_reference(x, h, &ks, B);
        let error = std::array::from_fn(|axis| {
            h * (41.0 / 840.0) * (ks[0][axis] + ks[10][axis] - ks[11][axis] - ks[12][axis])
        });
        Ok((next, error))
    }

    fn check<const N: usize>() {
        let initial = std::array::from_fn(|axis| {
            if axis % 5 == 0 {
                -0.0
            } else {
                (axis as f64 - 12.0) * 173.125
            }
        });
        let derivative = |t: f64, x: [f64; N]| -> Result<[f64; N], String> {
            Ok(std::array::from_fn(|axis| {
                x[axis] * 0.007 + t * 1e-5 + axis as f64 * 0.031
            }))
        };
        for h in [0.0, 0.003, 30.0, -1.0] {
            for dopri in [false, true] {
                let mut actual_calls = Vec::new();
                let mut actual_derivative = |t, x| {
                    actual_calls.push(t);
                    derivative(t, x)
                };
                let mut expected_calls = Vec::new();
                let mut expected_derivative = |t, x| {
                    expected_calls.push(t);
                    derivative(t, x)
                };
                let actual = if dopri {
                    dopri5_step(initial, 123.0, h, &mut actual_derivative)
                } else {
                    rkf78_step(initial, 123.0, h, &mut actual_derivative)
                }
                .unwrap();
                let expected = if dopri {
                    dopri5_step_axis_reference(initial, 123.0, h, &mut expected_derivative)
                } else {
                    rkf78_step_axis_reference(initial, 123.0, h, &mut expected_derivative)
                }
                .unwrap();
                assert_eq!(actual.0.map(f64::to_bits), expected.0.map(f64::to_bits));
                assert_eq!(actual.1.map(f64::to_bits), expected.1.map(f64::to_bits));
                assert_eq!(actual_calls, expected_calls);
            }
        }
    }

    #[test]
    fn state_stage_combinations_preserve_bits() {
        check::<6>();
    }

    #[test]
    fn stm_stage_combinations_preserve_bits() {
        check::<42>();
    }

    #[test]
    fn rkf78_rejects_non_finite_derivative_output() {
        let result = integrate_rkf78([1.0], 0.0, 1.0, 1e-9, 1e-9, None, |_, _| Ok([f64::NAN]));

        let error = result.expect_err("RKF78 must reject a non-finite derivative");
        assert!(error.contains("non-finite state or error"), "{error}");
    }

    #[test]
    fn dopri5_rejects_non_finite_derivative_output() {
        let result = integrate_dopri5([1.0], 0.0, 1.0, 1e-9, 1e-9, None, |_, _| {
            Ok([f64::INFINITY])
        });

        let error = result.expect_err("DOPRI5 must reject a non-finite derivative");
        assert!(error.contains("non-finite state or error"), "{error}");
    }

    #[test]
    fn rkf78_rejects_non_finite_initial_state_for_zero_duration() {
        let result = integrate_rkf78([f64::NAN], 0.0, 0.0, 1e-9, 1e-9, None, |_, _| Ok([0.0]));

        let error =
            result.expect_err("RKF78 must validate the initial state before zero-step return");
        assert!(error.contains("initial state must be finite"), "{error}");
    }

    #[test]
    fn dopri5_rejects_non_finite_initial_state_for_zero_duration() {
        let result = integrate_dopri5([f64::INFINITY], 0.0, 0.0, 1e-9, 1e-9, None, |_, _| {
            Ok([0.0])
        });

        let error =
            result.expect_err("DOPRI5 must validate the initial state before zero-step return");
        assert!(error.contains("initial state must be finite"), "{error}");
    }
}
