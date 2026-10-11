//! Trainer numerical kernels. Python retains typed ingress, mission/fault policy,
//! command sequencing and snapshots. FSW kernels receive onboard values; separate
//! display/scoring kernels receive game truth through their existing owners.
#[cfg(feature = "python")]
mod boundary;
#[cfg(feature = "python")]
mod frames;
#[cfg(feature = "python")]
mod packets;
#[cfg(any(feature = "python", test))]
mod preview;
#[cfg(feature = "python")]
mod scoring;

type V3 = [f64; 3];
fn norm(v: V3) -> f64 {
    dot(v, v).sqrt()
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn scale(v: V3, s: f64) -> V3 {
    v.map(|x| x * s)
}
fn unit(v: V3) -> Result<V3, String> {
    let n = norm(v);
    if n <= 0.0 || !n.is_finite() {
        Err("vector must be finite and nonzero".into())
    } else {
        Ok(scale(v, 1.0 / n))
    }
}
fn finite(v: &[f64]) -> Result<(), String> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err("kernel inputs must be finite".into())
    }
}
fn nonnegative(v: &[f64]) -> Result<(), String> {
    finite(v)?;
    if v.iter().all(|x| *x >= 0.0) {
        Ok(())
    } else {
        Err("kernel gains and limits must be nonnegative".into())
    }
}
fn clip(v: V3, limit: f64) -> (V3, f64) {
    let n = norm(v);
    let s = if limit <= 0.0 {
        0.0
    } else if n > limit {
        limit / n
    } else {
        1.0
    };
    (scale(v, s), s)
}
fn normalize(q: [f64; 4]) -> [f64; 4] {
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n <= 0.0 || !n.is_finite() {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        q.map(|x| x / n)
    }
}
fn multiply(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [a0, a1, a2, a3] = normalize(a);
    let [b0, b1, b2, b3] = normalize(b);
    [
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ]
}
fn dcm(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = normalize(q);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}
fn basis(state: [f64; 6]) -> Result<[V3; 3], String> {
    finite(&state)?;
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let radial = unit(r)?;
    let c = unit(cross(r, v))?;
    Ok([radial, unit(cross(c, radial))?, c])
}
pub fn ric_to_eci(vector: V3, state: [f64; 6]) -> Result<V3, String> {
    finite(&vector)?;
    let b = basis(state)?;
    Ok(std::array::from_fn(|i| {
        b[0][i] * vector[0] + b[1][i] * vector[1] + b[2][i] * vector[2]
    }))
}
#[allow(clippy::too_many_arguments)] // Flat, versioned PyO3 kernel boundary.
pub fn pilot_force(
    mode: u8,
    axes: V3,
    throttle: f64,
    max_accel: f64,
    mass: f64,
    reference: [f64; 6],
    q: [f64; 4],
    impulse: bool,
) -> Result<V3, String> {
    finite(&axes)?;
    finite(&q)?;
    nonnegative(&[throttle, max_accel, mass])?;
    let v = if impulse {
        axes
    } else {
        scale(clip(axes, 1.0).0, max_accel * throttle)
    };
    let a = match mode {
        0 => ric_to_eci(v, reference)?,
        1 => v,
        2 => scale(dcm(q)[0], max_accel * throttle),
        _ => return Err("unknown pilot mode".into()),
    };
    Ok(scale(a, mass))
}
pub fn ideal_wrench(force: V3, q: [f64; 4], maximum: f64) -> Result<(V3, V3, f64), String> {
    finite(&force)?;
    finite(&q)?;
    nonnegative(&[maximum])?;
    let n = norm(force);
    let s = if n > 0.0 { (maximum / n).min(1.0) } else { 1.0 };
    let achieved = scale(force, s);
    let c = dcm(q);
    Ok((
        [
            dot(c[0], achieved),
            dot(c[1], achieved),
            dot(c[2], achieved),
        ],
        achieved,
        s,
    ))
}
pub fn advance_attitude(q: [f64; 4], rate: V3, dt: f64) -> Result<[f64; 4], String> {
    finite(&q)?;
    finite(&rate)?;
    finite(&[dt])?;
    let n = norm(rate);
    if n <= 1e-15 || dt == 0.0 {
        return Ok(normalize(q));
    }
    let half = (0.5 * n * dt).rem_euclid(2.0 * std::f64::consts::PI);
    let v = scale(rate, half.sin() / n);
    Ok(normalize(multiply(q, [half.cos(), v[0], v[1], v[2]])))
}
pub fn attitude_torque(
    q: [f64; 4],
    desired: [f64; 4],
    omega: V3,
    kp: V3,
    kd: V3,
    maximum: f64,
    detumble: f64,
) -> Result<V3, String> {
    finite(&q)?;
    finite(&desired)?;
    finite(&omega)?;
    nonnegative(&kp)?;
    nonnegative(&kd)?;
    nonnegative(&[maximum, detumble])?;
    let mut error = normalize(multiply(
        [desired[0], -desired[1], -desired[2], -desired[3]],
        q,
    ));
    if error[0] < 0.0 {
        error = error.map(|x| -x)
    }
    let torque = std::array::from_fn(|i| {
        -kd[i] * omega[i]
            - if norm(omega) > detumble {
                0.0
            } else {
                kp[i] * error[i + 1]
            }
    });
    Ok(clip(torque, maximum).0)
}
pub fn pd_feedback(
    state: [f64; 6],
    target: [f64; 6],
    kp: f64,
    kd: f64,
    n: f64,
) -> Result<V3, String> {
    finite(&state)?;
    finite(&target)?;
    nonnegative(&[kp, kd, n])?;
    let mut a = std::array::from_fn(|i| {
        -(kp * (state[i] - target[i]) + kd * (state[i + 3] - target[i + 3]))
    });
    if n > 0.0 {
        a[0] += -3.0 * n * n * state[0] - 2.0 * n * state[4];
        a[1] += 2.0 * n * state[3];
        a[2] += n * n * state[2];
    }
    Ok(a)
}
pub fn retreat(
    state: [f64; 6],
    kd: f64,
    speed: f64,
    coast: f64,
) -> Result<(V3, f64, f64, bool), String> {
    finite(&state)?;
    nonnegative(&[kd, speed, coast])?;
    let pos = [state[0], state[1], state[2]];
    let r = norm(pos);
    let outward = if r > 1e-12 {
        scale(pos, 1.0 / r)
    } else {
        [0.0, -1.0, 0.0]
    };
    let v = dot([state[3], state[4], state[5]], outward);
    let coasting = r >= coast && v >= 0.9 * speed;
    let a = if coasting {
        [0.0; 3]
    } else {
        scale(outward, kd * (speed - v))
    };
    Ok((a, (coast - r).max(0.0), (speed - v).max(0.0), coasting))
}
/// Evaluate all candidate trajectories in one native crossing using matrices from
/// the authoritative HCW model. Matrices may differ at pulse/horizon boundaries.
pub fn predict_batch(
    state: [f64; 6],
    actions: Vec<V3>,
    matrices: Vec<Vec<f64>>,
    times: Vec<f64>,
    active: Vec<bool>,
    radius: f64,
) -> Result<Vec<(f64, f64, Option<f64>)>, String> {
    finite(&state)?;
    nonnegative(&[radius])?;
    if matrices.len() != times.len()
        || times.len() != active.len()
        || times.len() > 10000
        || actions.len().saturating_mul(times.len()) > 2000000
    {
        return Err("invalid or excessive prediction schedule".into());
    }
    let mut prior = 0.0;
    for (m, t) in matrices.iter().zip(&times) {
        if m.len() != 54 {
            return Err("prediction matrix must contain 54 values".into());
        }
        finite(m)?;
        finite(&[*t])?;
        if *t <= prior {
            return Err("prediction times must increase".into());
        }
        prior = *t;
    }
    let mut result = Vec::with_capacity(actions.len());
    for action in actions {
        finite(&action)?;
        let mut current = state;
        let mut closest = norm([state[0], state[1], state[2]]);
        let mut when = 0.0;
        let mut capture = if closest <= radius { Some(0.0) } else { None };
        for ((m, t), on) in matrices.iter().zip(&times).zip(&active) {
            let mut next = [0.0; 6];
            for i in 0..6 {
                for j in 0..6 {
                    next[i] += m[i * 6 + j] * current[j];
                }
                if *on {
                    for j in 0..3 {
                        next[i] += m[36 + i * 3 + j] * action[j];
                    }
                }
            }
            current = next;
            finite(&current)?;
            let r = norm([current[0], current[1], current[2]]);
            if r < closest {
                closest = r;
                when = *t;
            }
            if capture.is_none() && r <= radius {
                capture = Some(*t)
            }
        }
        result.push((closest, when, capture));
    }
    Ok(result)
}
#[allow(clippy::too_many_arguments)] // Flat, versioned PyO3 kernel boundary.
pub fn transfer_control(
    state: [f64; 6],
    target: V3,
    planned: V3,
    gain: Vec<f64>,
    drift: Vec<f64>,
    remaining: f64,
    terminal: bool,
    final_brake: bool,
    failed: bool,
    brake_start: f64,
    tau: f64,
    maximum: f64,
    deadband: f64,
) -> Result<(V3, V3, V3, V3, f64, u8), String> {
    finite(&state)?;
    finite(&target)?;
    finite(&planned)?;
    finite(&gain)?;
    finite(&drift)?;
    finite(&[remaining])?;
    nonnegative(&[brake_start, tau, deadband])?;
    // The outer SI translation controller owns the authoritative limit;
    // its subordinate transfer law deliberately uses +infinity.
    if maximum.is_nan() || maximum < 0.0 {
        return Err("invalid transfer acceleration limit".into());
    }
    if gain.len() != 18 || drift.len() != 18 || tau <= 0.0 {
        return Err("invalid transfer matrix or time constant".into());
    }
    let mut target_velocity = target;
    let mut error = std::array::from_fn(|i| target_velocity[i] - state[i + 3]);
    let mut pre = [0.0; 3];
    let phase;
    if terminal || failed {
        for i in 0..3 {
            let mut feedback = 0.0;
            let mut natural = 0.0;
            for j in 0..6 {
                feedback += gain[i * 6 + j] * state[j];
                natural += drift[i * 6 + j] * state[j]
            }
            pre[i] = -feedback - natural;
        }
        phase = 0;
    } else if final_brake {
        let ramp = if brake_start > 0.0 {
            (remaining / brake_start).clamp(0.0, 1.0)
        } else {
            0.0
        };
        target_velocity = scale(planned, ramp);
        error = std::array::from_fn(|i| target_velocity[i] - state[i + 3]);
        pre = scale(error, 1.0 / tau);
        phase = 1;
    } else if norm(error) * 1000.0 > deadband {
        pre = scale(error, 1.0 / tau);
        phase = 2;
    } else {
        phase = 3;
    }
    let (accel, s) = if phase == 3 {
        ([0.0; 3], 1.0)
    } else {
        clip(pre, maximum)
    };
    Ok((accel, pre, target_velocity, error, s, phase))
}
pub fn effector_positions(axes: Vec<f64>, limits: Vec<V3>) -> Result<Vec<f64>, String> {
    if axes.len() != limits.len() {
        return Err("effector dimensions differ".into());
    }
    finite(&axes)?;
    let mut out = Vec::new();
    for (axis, [minimum, maximum, neutral]) in axes.into_iter().zip(limits) {
        finite(&[minimum, maximum, neutral])?;
        if minimum > neutral || neutral > maximum {
            return Err("invalid effector limits".into());
        }
        let a = axis.clamp(-1.0, 1.0);
        out.push(if a >= 0.0 {
            neutral + a * (maximum - neutral)
        } else {
            neutral + (-a) * (minimum - neutral)
        });
    }
    Ok(out)
}

pub fn relative_observation(
    los: V3,
    transform: Vec<f64>,
    range: f64,
    rate: f64,
    angular: Option<V3>,
) -> Result<[f64; 6], String> {
    finite(&los)?;
    finite(&transform)?;
    finite(&[range, rate])?;
    if transform.len() != 9 || range < 0.0 {
        return Err("invalid relative observation".into());
    }
    let apply = |v: V3| -> V3 {
        std::array::from_fn(|i| {
            transform[i * 3] * v[0] + transform[i * 3 + 1] * v[1] + transform[i * 3 + 2] * v[2]
        })
    };
    let direction = unit(apply(los))?;
    let position = scale(direction, range);
    let mut velocity = scale(direction, rate);
    if let Some(w) = angular {
        finite(&w)?;
        let tangent = scale(cross(apply(w), direction), range);
        for i in 0..3 {
            velocity[i] += tangent[i]
        }
    }
    Ok([
        position[0],
        position[1],
        position[2],
        velocity[0],
        velocity[1],
        velocity[2],
    ])
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    fn error(e: String) -> PyErr {
        PyValueError::new_err(e)
    }
    #[pyfunction(name = "ric_to_eci")]
    fn py_ric(vector: V3, state: [f64; 6]) -> PyResult<V3> {
        ric_to_eci(vector, state).map_err(error)
    }
    #[pyfunction(name = "pilot_force")]
    #[allow(clippy::too_many_arguments)]
    fn py_pilot(
        mode: u8,
        axes: V3,
        throttle: f64,
        max_accel: f64,
        mass: f64,
        reference: [f64; 6],
        q: [f64; 4],
        impulse: bool,
    ) -> PyResult<V3> {
        pilot_force(mode, axes, throttle, max_accel, mass, reference, q, impulse).map_err(error)
    }
    #[pyfunction(name = "ideal_wrench")]
    fn py_wrench(force: V3, q: [f64; 4], maximum: f64) -> PyResult<(V3, V3, f64)> {
        ideal_wrench(force, q, maximum).map_err(error)
    }
    #[pyfunction(name = "advance_attitude")]
    fn py_advance(q: [f64; 4], rate: V3, dt: f64) -> PyResult<[f64; 4]> {
        advance_attitude(q, rate, dt).map_err(error)
    }
    #[pyfunction(name = "attitude_torque")]
    #[allow(clippy::too_many_arguments)]
    fn py_torque(
        q: [f64; 4],
        desired: [f64; 4],
        omega: V3,
        kp: V3,
        kd: V3,
        maximum: f64,
        detumble: f64,
    ) -> PyResult<V3> {
        attitude_torque(q, desired, omega, kp, kd, maximum, detumble).map_err(error)
    }
    #[pyfunction(name = "pd_feedback")]
    fn py_pd(state: [f64; 6], target: [f64; 6], kp: f64, kd: f64, n: f64) -> PyResult<V3> {
        pd_feedback(state, target, kp, kd, n).map_err(error)
    }
    #[pyfunction(name = "retreat")]
    fn py_retreat(
        state: [f64; 6],
        kd: f64,
        speed: f64,
        coast: f64,
    ) -> PyResult<(V3, f64, f64, bool)> {
        retreat(state, kd, speed, coast).map_err(error)
    }
    #[pyfunction(name = "predict_batch")]
    fn py_predict(
        py: Python<'_>,
        state: [f64; 6],
        actions: Vec<V3>,
        matrices: Vec<Vec<f64>>,
        times: Vec<f64>,
        active: Vec<bool>,
        radius: f64,
    ) -> PyResult<Vec<(f64, f64, Option<f64>)>> {
        py.detach(move || predict_batch(state, actions, matrices, times, active, radius))
            .map_err(error)
    }
    #[pyfunction(name = "transfer_control")]
    #[allow(clippy::too_many_arguments)]
    fn py_transfer(
        state: [f64; 6],
        target: V3,
        planned: V3,
        gain: Vec<f64>,
        drift: Vec<f64>,
        remaining: f64,
        terminal: bool,
        final_brake: bool,
        failed: bool,
        brake_start: f64,
        tau: f64,
        maximum: f64,
        deadband: f64,
    ) -> PyResult<(V3, V3, V3, V3, f64, u8)> {
        transfer_control(
            state,
            target,
            planned,
            gain,
            drift,
            remaining,
            terminal,
            final_brake,
            failed,
            brake_start,
            tau,
            maximum,
            deadband,
        )
        .map_err(error)
    }
    #[pyfunction(name = "effector_positions")]
    fn py_effectors(axes: Vec<f64>, limits: Vec<V3>) -> PyResult<Vec<f64>> {
        effector_positions(axes, limits).map_err(error)
    }
    #[pyfunction(name = "relative_observation")]
    #[pyo3(signature=(los, transform, range, rate, angular=None))]
    fn py_observation(
        los: V3,
        transform: Vec<f64>,
        range: f64,
        rate: f64,
        angular: Option<V3>,
    ) -> PyResult<[f64; 6]> {
        relative_observation(los, transform, range, rate, angular).map_err(error)
    }
    #[pyfunction(name = "preview_two_body")]
    fn py_preview_two_body(state: [f64; 6], duration: f64, mu: f64) -> PyResult<[f64; 6]> {
        preview::two_body(state, duration, mu).map_err(error)
    }
    #[pyfunction(name = "preview_th_history")]
    fn py_preview_th(
        py: Python<'_>,
        relative: [f64; 6],
        chief: [f64; 6],
        times: Vec<f64>,
        step: f64,
        mu: f64,
    ) -> PyResult<Vec<[f64; 6]>> {
        py.detach(move || preview::th_history(relative, chief, times, step, mu))
            .map_err(error)
    }
    #[pyfunction(name = "score_sample")]
    fn py_score_sample(
        relative: [f64; 6],
        thrust: V3,
        target_thrust: V3,
        goal_points: &[u8],
    ) -> PyResult<[f64; 5]> {
        scoring::sample(relative, thrust, target_thrust, goal_points).map_err(error)
    }
    #[pymodule]
    fn oel_rust_game(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        m.add(
            "__source_sha256__",
            option_env!("OEL_NATIVE_SOURCE_SHA256").unwrap_or("unattested"),
        )?;
        m.add("BACKEND_ABI", 1)?;
        m.add_class::<crate::packets::TrustedPacketEncoder>()?;
        m.add_class::<crate::boundary::BoundaryValidator>()?;
        m.add_class::<crate::frames::FrameConverter>()?;
        m.add_function(wrap_pyfunction!(py_preview_two_body, m)?)?;
        m.add_function(wrap_pyfunction!(py_preview_th, m)?)?;
        m.add_function(wrap_pyfunction!(py_score_sample, m)?)?;
        m.add_function(wrap_pyfunction!(py_observation, m)?)?;
        m.add_function(wrap_pyfunction!(py_ric, m)?)?;
        m.add_function(wrap_pyfunction!(py_pilot, m)?)?;
        m.add_function(wrap_pyfunction!(py_wrench, m)?)?;
        m.add_function(wrap_pyfunction!(py_advance, m)?)?;
        m.add_function(wrap_pyfunction!(py_torque, m)?)?;
        m.add_function(wrap_pyfunction!(py_pd, m)?)?;
        m.add_function(wrap_pyfunction!(py_retreat, m)?)?;
        m.add_function(wrap_pyfunction!(py_predict, m)?)?;
        m.add_function(wrap_pyfunction!(py_transfer, m)?)?;
        m.add_function(wrap_pyfunction!(py_effectors, m)?)?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frames_and_saturation() {
        assert_eq!(
            ric_to_eci([1.0, 2.0, 3.0], [2.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            [1.0, 2.0, 3.0]
        );
        let (body, achieved, s) = ideal_wrench([3.0, 4.0, 0.0], [1.0, 0.0, 0.0, 0.0], 2.5).unwrap();
        assert_eq!(body, achieved);
        assert_eq!(s, 0.5);
        assert!(ric_to_eci([0.0; 3], [0.0; 6]).is_err());
    }
    #[test]
    fn attitude_shortest_arc() {
        let q = advance_attitude([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0).unwrap();
        assert!((q[0] - 0.5f64.cos()).abs() < 1e-14);
        let a = attitude_torque(
            q,
            [1.0, 0.0, 0.0, 0.0],
            [0.0; 3],
            [1.0; 3],
            [1.0; 3],
            2.0,
            0.5,
        )
        .unwrap();
        let b = attitude_torque(
            q.map(|x| -x),
            [1.0, 0.0, 0.0, 0.0],
            [0.0; 3],
            [1.0; 3],
            [1.0; 3],
            2.0,
            0.5,
        )
        .unwrap();
        assert_eq!(a, b);
    }
    #[test]
    fn retreat_and_invalid() {
        let (a, _, _, coast) =
            retreat([2000.0, 0.0, 0.0, 0.2, 0.0, 0.0], 0.004, 0.2, 1000.0).unwrap();
        assert!(coast);
        assert_eq!(a, [0.0; 3]);
        assert!(pd_feedback([f64::NAN; 6], [0.0; 6], 1.0, 1.0, 0.0).is_err());
    }
    #[test]
    fn prediction_dimensions_and_capture() {
        let mut m = vec![0.0; 54];
        for i in 0..6 {
            m[i * 6 + i] = 1.0
        }
        let r = predict_batch(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![[0.0; 3]],
            vec![m],
            vec![1.0],
            vec![false],
            2.0,
        )
        .unwrap();
        assert_eq!(r, vec![(1.0, 0.0, Some(0.0))]);
        assert!(predict_batch(
            [0.0; 6],
            vec![[0.0; 3]],
            vec![vec![]],
            vec![1.0],
            vec![true],
            1.0
        )
        .is_err());
    }
    #[test]
    fn transfer_phases() {
        let r = transfer_control(
            [0.0; 6],
            [0.001, 0.0, 0.0],
            [0.0; 3],
            vec![0.0; 18],
            vec![0.0; 18],
            100.0,
            false,
            false,
            false,
            0.0,
            10.0,
            1e-5,
            0.01,
        )
        .unwrap();
        assert_eq!(r.5, 2);
        assert!((r.0[0] - 1e-5).abs() < 1e-18);
        assert_eq!(
            effector_positions(vec![-0.5, 0.5], vec![[0.0, 10.0, 2.0]; 2]).unwrap(),
            vec![1.0, 6.0]
        );
    }
}
