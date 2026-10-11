//! Fused point-mass orbit and rigid attitude stepping for the maintained
//! mechanical model. Gravity-gradient torque uses the same translational
//! midpoint and attitude substep ordering as OrbitalAttitudeDynamics.

use crate::{attitude, OrbitParameters, State};

const MIN_MECHANICAL_SUBSTEP_S: f64 = 1e-9;
const MAX_MECHANICAL_SUBSTEPS: f64 = 1_000_000.0;

pub struct MechanicalContext {
    pub mu: f64,
    pub inertia: [[f64; 3]; 3],
    pub gravity_gradient: bool,
    pub magnetic_enabled: bool,
    pub magnetic_dipole: [f64; 3],
    pub magnetic_field: [f64; 3],
    pub magnetic_field_provided: bool,
}

fn effective_substep(dt: f64, requested: Option<f64>) -> f64 {
    requested.map_or(dt, |value| value.max(MIN_MECHANICAL_SUBSTEP_S).min(dt))
}

fn exceeds_step_budget(dt: f64, requested: Option<f64>) -> bool {
    dt / effective_substep(dt, requested) > MAX_MECHANICAL_SUBSTEPS
}

fn substeps(dt: f64, requested: Option<f64>) -> impl Iterator<Item = f64> {
    let h = effective_substep(dt, requested);
    let n = if h >= dt {
        1
    } else {
        (dt / h).floor() as usize
    };
    let remainder = if h >= dt { 0.0 } else { dt - n as f64 * h };
    let count = n + usize::from(remainder > 1e-12);
    (0..count).map(move |i| if i < n { h } else { remainder })
}

impl MechanicalContext {
    pub fn step(
        &self,
        state: [f64; 13],
        dt: f64,
        orbit_substep: Option<f64>,
        attitude_substep: Option<f64>,
        command: [f64; 3],
        torque: [f64; 3],
    ) -> Result<([f64; 13], [i64; 6]), &'static str> {
        if !dt.is_finite()
            || dt <= 0.0
            || !self.mu.is_finite()
            || self.mu <= 0.0
            || [orbit_substep, attitude_substep]
                .into_iter()
                .flatten()
                .any(|h| !h.is_finite() || h <= 0.0)
            || exceeds_step_budget(dt, orbit_substep)
            || exceeds_step_budget(dt, attitude_substep)
        {
            return Err("invalid mechanical timestep or central mu");
        }
        let initial: [f64; 6] = state[..6].try_into().unwrap();
        let mut orbit = State::from_array(initial)?;
        let mut q: [f64; 4] = state[6..10].try_into().unwrap();
        let mut w: [f64; 3] = state[10..13].try_into().unwrap();
        let mut counts = [0; 6];
        let mut count = |stats: [i64; 6]| {
            for i in 0..6 {
                counts[i] += stats[i];
            }
        };
        // Preserve guardrail observations made by command-only attitude
        // predictions, including those discarded by the mechanical owner.
        count(attitude::propagate_attitude_rate_guardrails(q, w, self.inertia, torque, 0.5 * dt).1);
        let (mut q_command, mut w_command) = (q, w);
        let parameters = OrbitParameters {
            mu_km3_s2: self.mu,
            include_j2: false,
            command_accel_eci_km_s2: command,
        };
        for h in substeps(dt, orbit_substep) {
            count(
                attitude::propagate_attitude_rate_guardrails(
                    q_command,
                    w_command,
                    self.inertia,
                    torque,
                    0.5 * h,
                )
                .1,
            );
            orbit = crate::rk4_step(orbit, h, parameters)?;
            let (next_w, stats) = attitude::propagate_attitude_rate_guardrails(
                q_command,
                w_command,
                self.inertia,
                torque,
                h,
            );
            // The discarded prediction produces a finite normalized quaternion.
            // Its value cannot affect this point-mass context or later counts.
            q_command = [1.0, 0.0, 0.0, 0.0];
            w_command = next_w;
            count(stats);
        }
        let next_orbit = orbit.as_array();
        let midpoint = std::array::from_fn(|i| 0.5 * (initial[i] + next_orbit[i]));
        for h in substeps(dt, attitude_substep) {
            let disturbance = attitude::builtin_disturbance_torque(
                q,
                midpoint,
                self.inertia,
                self.mu,
                [
                    i64::from(self.gravity_gradient),
                    i64::from(self.magnetic_enabled),
                    0,
                    0,
                ],
                self.magnetic_dipole,
                self.magnetic_field,
                self.magnetic_field_provided,
                0.0,
                [0.0; 3],
                0.0,
                0,
                0.0,
                0.0,
                [0.0; 3],
                &[],
                &[],
                &[],
                &[],
                [0.0; 3],
                0.0,
                0,
                0.0,
                [0.0; 3],
                &[],
                &[],
                &[],
            );
            let total = std::array::from_fn(|i| torque[i] + disturbance[i]);
            let (next_q, next_w, stats) =
                attitude::propagate_attitude_exponential_map(q, w, self.inertia, total, h);
            q = next_q;
            w = next_w;
            count(stats);
        }
        let mut result = [0.0; 13];
        result[..6].copy_from_slice(&next_orbit);
        result[6..10].copy_from_slice(&q);
        result[10..13].copy_from_slice(&w);
        Ok((result, counts))
    }
}

#[cfg(test)]
mod tests {
    use super::{effective_substep, exceeds_step_budget, substeps, MechanicalContext};

    fn context() -> MechanicalContext {
        MechanicalContext {
            mu: 398600.4415,
            inertia: [[120.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 80.0]],
            gravity_gradient: false,
            magnetic_enabled: false,
            magnetic_dipole: [0.0; 3],
            magnetic_field: [0.0; 3],
            magnetic_field_provided: false,
        }
    }

    fn state() -> [f64; 13] {
        [
            7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    }

    #[test]
    fn positive_subnanosecond_substeps_are_clamped_and_accepted() {
        let context = context();
        let (actual, _) = context
            .step(
                state(),
                2e-9,
                Some(0.5e-9),
                Some(0.5e-9),
                [0.0; 3],
                [0.0; 3],
            )
            .unwrap();
        let (clamped, _) = context
            .step(state(), 2e-9, Some(1e-9), Some(1e-9), [0.0; 3], [0.0; 3])
            .unwrap();
        assert_eq!(actual, clamped);
    }

    #[test]
    fn subnanosecond_outer_interval_is_not_overshot() {
        let dt = 0.5e-9;
        let steps: Vec<_> = substeps(dt, Some(0.25e-9)).collect();
        assert_eq!(steps, [dt]);
        assert_eq!(steps.iter().sum::<f64>(), dt);

        let context = context();
        let (requested, _) = context
            .step(
                state(),
                dt,
                Some(0.25e-9),
                Some(0.25e-9),
                [0.0; 3],
                [0.0; 3],
            )
            .unwrap();
        let (outer_interval, _) = context
            .step(state(), dt, None, None, [0.0; 3], [0.0; 3])
            .unwrap();
        assert_eq!(requested, outer_interval);
    }

    #[test]
    fn million_step_guard_uses_effective_substep() {
        let requested = Some(0.5e-9);
        assert_eq!(effective_substep(1e-3, requested), 1e-9);
        assert!(!exceeds_step_budget(1e-3, requested));
        assert!(exceeds_step_budget(1.000_001e-3, requested));

        let dt = 0.5e-9;
        assert_eq!(effective_substep(dt, Some(0.25e-9)), dt);
        assert!(!exceeds_step_budget(dt, Some(0.25e-9)));
    }
}

fn coupled_quaternion_rate(q: [f64; 4], w: [f64; 3]) -> [f64; 4] {
    let matrix = [
        [0.0, -w[0], -w[1], -w[2]],
        [w[0], 0.0, w[2], -w[1]],
        [w[1], -w[2], 0.0, w[0]],
        [w[2], w[1], -w[0], 0.0],
    ];
    matrix.map(|row| 0.5 * row.iter().zip(q).map(|(a, b)| a * b).sum::<f64>())
}

#[allow(clippy::too_many_arguments)]
impl MechanicalContext {
    pub fn coupled_step(
        &self,
        state: [f64; 14],
        start_time_s: f64,
        dt_s: f64,
        maximum_step_s: f64,
        force_eci: [f64; 3],
        force_body: [f64; 3],
        mass_flow: f64,
        minimum_mass: f64,
        command_torque: [f64; 3],
        enabled: [i64; 4],
        magnetic_dipole: [f64; 3],
        magnetic_field: [f64; 3],
        magnetic_field_provided: bool,
    ) -> Result<([f64; 14], [i64; 6]), &'static str> {
        if !state
            .iter()
            .chain(force_eci.iter())
            .chain(force_body.iter())
            .chain(command_torque.iter())
            .all(|v| v.is_finite())
            || !start_time_s.is_finite()
            || !dt_s.is_finite()
            || dt_s <= 0.0
            || !maximum_step_s.is_finite()
            || maximum_step_s <= 0.0
            || !mass_flow.is_finite()
            || mass_flow < 0.0
            || !minimum_mass.is_finite()
            || minimum_mass < 0.0
            || !self.mu.is_finite()
            || self.mu <= 0.0
            || state[13] <= 0.0
            || dt_s / maximum_step_s > 1_000_000.0
            || enabled[2] != 0
            || enabled[3] != 0
        {
            return Err(
                "coupled state, command, mass and time must be finite with valid dimensions",
            );
        }
        let input_q: [f64; 4] = state[6..10].try_into().unwrap();
        let input_norm = input_q.iter().map(|x| x * x).sum::<f64>().sqrt();
        if (input_norm - 1.0).abs() > 1e-10 {
            return Err("attitude_quat_bn must be normalized within 1e-10");
        }
        let derivative = |_time: f64,
                          stage: [f64; 10],
                          raw_q: [f64; 4]|
         -> Result<([f64; 10], [f64; 4], [i64; 6]), &'static str> {
            if !stage.iter().all(|v| v.is_finite()) || stage[9] <= 0.0 {
                return Err("coupled stage mass and state must be finite and positive");
            }
            // CoupledSatelliteState normalizes every stage independently.
            let stage_q = attitude::normalize_quaternion(raw_q);
            let c = attitude::quaternion_to_dcm_bn(stage_q);
            let available = stage[9] > minimum_mass + 1e-12;
            let force_on = available || mass_flow == 0.0;
            let physical: [f64; 3] = std::array::from_fn(|axis| {
                let rotated = (0..3)
                    .map(|j| c[j][axis] * if force_on { force_body[j] } else { 0.0 })
                    .sum::<f64>();
                (if force_on { force_eci[axis] } else { 0.0 }) + rotated
            });
            let radius2 = stage[..3].iter().map(|x| x * x).sum::<f64>();
            let scale = if radius2 == 0.0 {
                0.0
            } else {
                -self.mu / (radius2.sqrt() * radius2)
            };
            let disturbance = attitude::builtin_disturbance_torque(
                stage_q,
                stage[..3].try_into().unwrap(),
                self.inertia,
                self.mu,
                enabled,
                magnetic_dipole,
                magnetic_field,
                magnetic_field_provided,
                0.0,
                [0.0; 3],
                0.0,
                0,
                0.0,
                0.0,
                [0.0; 3],
                &[],
                &[],
                &[],
                &[],
                [0.0; 3],
                0.0,
                0,
                0.0,
                [0.0; 3],
                &[],
                &[],
                &[],
            );
            let total = std::array::from_fn(|i| command_torque[i] + disturbance[i]);
            let w = stage[6..9].try_into().unwrap();
            let (_, w_dot, stats) =
                attitude::rigid_body_derivatives(stage_q, w, self.inertia, total);
            let mut dz = [0.0; 10];
            dz[..3].copy_from_slice(&stage[3..6]);
            for i in 0..3 {
                dz[3 + i] = scale * stage[i] + physical[i] / stage[9] / 1e3;
            }
            dz[6..9].copy_from_slice(&w_dot);
            dz[9] = if available { -mass_flow } else { -0.0 };
            let q_dot = coupled_quaternion_rate(stage_q, w);
            if !dz.iter().chain(q_dot.iter()).all(|v| v.is_finite()) {
                return Err("coupled stage callback returned invalid derivatives");
            }
            Ok((dz, q_dot, stats))
        };
        coupled_integrate(
            state,
            start_time_s,
            dt_s,
            maximum_step_s,
            minimum_mass,
            derivative,
            |message| message,
        )
    }
}

fn coupled_integrate<E>(
    state: [f64; 14],
    start_time_s: f64,
    dt_s: f64,
    maximum_step_s: f64,
    minimum_mass: f64,
    mut derivative: impl FnMut(f64, [f64; 10], [f64; 4]) -> Result<([f64; 10], [f64; 4], [i64; 6]), E>,
    error: impl Fn(&'static str) -> E,
) -> Result<([f64; 14], [i64; 6]), E> {
    if !(start_time_s + dt_s).is_finite() {
        return Err(error("coupled end time must be finite"));
    }
    let mut q = attitude::normalize_quaternion(state[6..10].try_into().unwrap());
    let mut z = [0.0; 10];
    z[..6].copy_from_slice(&state[..6]);
    z[6..9].copy_from_slice(&state[10..13]);
    z[9] = state[13];
    let mut t = start_time_s;
    let end = start_time_s + dt_s;
    let mut counts = [0_i64; 6];
    while t < end - 1e-14 {
        let h = maximum_step_s.min(end - t);
        let stage = |slope: [f64; 10], width: f64| std::array::from_fn(|i| z[i] + width * slope[i]);
        let stage_q = |previous: [f64; 4], rate: [f64; 3], width: f64| {
            let dq = coupled_quaternion_rate(previous, rate);
            attitude::normalize_quaternion(std::array::from_fn(|i| q[i] + width * dq[i]))
        };
        let (k1, dq1, c1) = derivative(t, z, q)?;
        let q2 = stage_q(q, z[6..9].try_into().unwrap(), 0.5 * h);
        let z2 = stage(k1, 0.5 * h);
        let (k2, dq2, c2) = derivative(t + 0.5 * h, z2, q2)?;
        let q3 = stage_q(q2, z2[6..9].try_into().unwrap(), 0.5 * h);
        let z3 = stage(k2, 0.5 * h);
        let (k3, dq3, c3) = derivative(t + 0.5 * h, z3, q3)?;
        let q4 = stage_q(q3, z3[6..9].try_into().unwrap(), h);
        let z4 = stage(k3, h);
        let (k4, dq4, c4) = derivative(t + h, z4, q4)?;
        z = std::array::from_fn(|i| z[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
        let raw = std::array::from_fn(|i| {
            q[i] + (h / 6.0) * (dq1[i] + 2.0 * dq2[i] + 2.0 * dq3[i] + dq4[i])
        });
        q = attitude::normalize_quaternion(attitude::normalize_quaternion(raw));
        if !z.iter().chain(q.iter()).all(|v| v.is_finite()) || z[9] <= 0.0 {
            return Err(error(
                "coupled propagation produced a non-finite or non-positive state",
            ));
        }
        for i in 0..6 {
            counts[i] += c1[i] + c2[i] + c3[i] + c4[i];
        }
        t += h;
    }
    let mut result = [0.0; 14];
    result[..6].copy_from_slice(&z[..6]);
    result[6..10].copy_from_slice(&q);
    result[10..13].copy_from_slice(&z[6..9]);
    result[13] = z[9].max(minimum_mass);
    Ok((result, counts))
}

#[cfg(feature = "python")]
pub fn register(module: &pyo3::prelude::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
    fn fixed<const N: usize>(v: Vec<f64>) -> PyResult<[f64; N]> {
        v.try_into()
            .map_err(|_| PyValueError::new_err(format!("expected {N} values")))
    }
    #[pyclass(name = "MechanicalContext")]
    struct NativeMechanicalContext {
        context: MechanicalContext,
    }
    #[pymethods]
    impl NativeMechanicalContext {
        #[new]
        fn new(mu: f64, inertia: Vec<f64>, gravity_gradient: bool) -> PyResult<Self> {
            let a: [f64; 9] = fixed(inertia)?;
            Ok(Self {
                context: MechanicalContext {
                    mu,
                    inertia: [
                        a[..3].try_into().unwrap(),
                        a[3..6].try_into().unwrap(),
                        a[6..].try_into().unwrap(),
                    ],
                    gravity_gradient,
                    magnetic_enabled: false,
                    magnetic_dipole: [0.0; 3],
                    magnetic_field: [0.0; 3],
                    magnetic_field_provided: false,
                },
            })
        }
        #[staticmethod]
        fn with_magnetic(
            mu: f64,
            inertia: Vec<f64>,
            gravity_gradient: bool,
            dipole: Vec<f64>,
            field: Vec<f64>,
            field_provided: bool,
        ) -> PyResult<Self> {
            let dipole: [f64; 3] = fixed(dipole)?;
            let field: [f64; 3] = fixed(field)?;
            if !dipole.iter().chain(&field).all(|v| v.is_finite()) {
                return Err(PyValueError::new_err(
                    "magnetic dipole and field must be finite",
                ));
            }
            let mut result = Self::new(mu, inertia, gravity_gradient)?;
            result.context.magnetic_enabled = true;
            result.context.magnetic_dipole = dipole;
            result.context.magnetic_field = field;
            result.context.magnetic_field_provided = field_provided;
            Ok(result)
        }
        #[allow(clippy::too_many_arguments)]
        fn step(
            &self,
            state: Vec<f64>,
            dt: f64,
            orbit_substep: Option<f64>,
            attitude_substep: Option<f64>,
            command: Vec<f64>,
            torque: Vec<f64>,
        ) -> PyResult<(Vec<f64>, Vec<i64>)> {
            let (next, counts) = self
                .context
                .step(
                    fixed(state)?,
                    dt,
                    orbit_substep,
                    attitude_substep,
                    fixed(command)?,
                    fixed(torque)?,
                )
                .map_err(PyValueError::new_err)?;
            Ok((next.to_vec(), counts.to_vec()))
        }
        #[allow(clippy::too_many_arguments)]
        fn coupled_step(
            &self,
            state: Vec<f64>,
            start_time_s: f64,
            dt_s: f64,
            maximum_step_s: f64,
            force_eci: Vec<f64>,
            force_body: Vec<f64>,
            mass_flow: f64,
            minimum_mass: f64,
            command_torque: Vec<f64>,
            enabled: Vec<i64>,
            magnetic_dipole: Vec<f64>,
            magnetic_field: Vec<f64>,
            magnetic_field_provided: bool,
        ) -> PyResult<(Vec<f64>, Vec<i64>)> {
            let enabled: [i64; 4] = enabled
                .try_into()
                .map_err(|_| PyValueError::new_err("expected 4 disturbance flags"))?;
            let (next, counts) = self
                .context
                .coupled_step(
                    fixed(state)?,
                    start_time_s,
                    dt_s,
                    maximum_step_s,
                    fixed(force_eci)?,
                    fixed(force_body)?,
                    mass_flow,
                    minimum_mass,
                    fixed(command_torque)?,
                    enabled,
                    fixed(magnetic_dipole)?,
                    fixed(magnetic_field)?,
                    magnetic_field_provided,
                )
                .map_err(PyValueError::new_err)?;
            Ok((next.to_vec(), counts.to_vec()))
        }

        #[allow(clippy::too_many_arguments)]
        fn coupled_force_step(
            &self,
            state: Vec<f64>,
            start_time_s: f64,
            dt_s: f64,
            maximum_step_s: f64,
            force_eci: Vec<f64>,
            force_body: Vec<f64>,
            mass_flow: f64,
            minimum_mass: f64,
            command_torque: Vec<f64>,
            force_context: PyRef<'_, crate::force_context::NativeForcePlanContext>,
            disturbance_context: PyRef<'_, crate::attitude::BuiltinDisturbanceContext>,
            stage_callback: &Bound<'_, PyAny>,
        ) -> PyResult<(Vec<f64>, Vec<i64>)> {
            let state: [f64; 14] = fixed(state)?;
            let force_eci: [f64; 3] = fixed(force_eci)?;
            let force_body: [f64; 3] = fixed(force_body)?;
            let command_torque: [f64; 3] = fixed(command_torque)?;
            if !state
                .iter()
                .chain(&force_eci)
                .chain(&force_body)
                .chain(&command_torque)
                .all(|v| v.is_finite())
                || !start_time_s.is_finite()
                || !dt_s.is_finite()
                || dt_s <= 0.0
                || !(start_time_s + dt_s).is_finite()
                || !maximum_step_s.is_finite()
                || maximum_step_s <= 0.0
                || dt_s / maximum_step_s > 1_000_000.0
                || state[13] <= 0.0
                || !mass_flow.is_finite()
                || mass_flow < 0.0
                || !minimum_mass.is_finite()
                || minimum_mass < 0.0
            {
                return Err(PyValueError::new_err(
                    "coupled state, command, mass and time must be finite with valid dimensions",
                ));
            }
            let norm = state[6..10].iter().map(|v| v * v).sum::<f64>().sqrt();
            if (norm - 1.0).abs() > 1e-10 {
                return Err(PyValueError::new_err(
                    "attitude_quat_bn must be normalized within 1e-10",
                ));
            }
            let derivative = |time: f64,
                              stage: [f64; 10],
                              raw_q: [f64; 4]|
             -> PyResult<([f64; 10], [f64; 4], [i64; 6])> {
                if !stage.iter().all(|v| v.is_finite()) || stage[9] <= 0.0 {
                    return Err(PyValueError::new_err(
                        "coupled stage mass and state must be finite and positive",
                    ));
                }
                let q = attitude::normalize_quaternion(raw_q);
                let c = attitude::quaternion_to_dcm_bn(q);
                let available = stage[9] > minimum_mass + 1e-12;
                let force_on = available || mass_flow == 0.0;
                let command = std::array::from_fn(|axis| {
                    let rotated: f64 = (0..3)
                        .map(|j| c[j][axis] * if force_on { force_body[j] } else { 0.0 })
                        .sum();
                    ((if force_on { force_eci[axis] } else { 0.0 }) + rotated) / stage[9] / 1e3
                });
                let state6: [f64; 6] = stage[..6].try_into().unwrap();
                let inputs: Vec<f64> = stage_callback.call1((time, state6.to_vec()))?.extract()?;
                if inputs.len() != 32 || !inputs.iter().all(|v| v.is_finite()) {
                    return Err(PyValueError::new_err(
                        "coupled stage inputs must have 32 finite values",
                    ));
                }
                let environment =
                    crate::force_plan::StageInputs::from_flat(inputs[..19].try_into().unwrap())
                        .map_err(PyValueError::new_err)?;
                let acceleration = force_context.plan().acceleration_with_mass(
                    state6,
                    &environment,
                    command,
                    stage[9],
                ).map_err(PyValueError::new_err)?;
                let disturbance = disturbance_context.torque_for_stage(
                    q,
                    stage[..3].try_into().unwrap(),
                    inputs[19..].try_into().unwrap(),
                );
                let total = std::array::from_fn(|i| command_torque[i] + disturbance[i]);
                let w = stage[6..9].try_into().unwrap();
                let (_, w_dot, counts) =
                    attitude::rigid_body_derivatives(q, w, self.context.inertia, total);
                let mut dz = [0.0; 10];
                dz[..3].copy_from_slice(&stage[3..6]);
                dz[3..6].copy_from_slice(&acceleration);
                dz[6..9].copy_from_slice(&w_dot);
                dz[9] = if available { -mass_flow } else { -0.0 };
                let dq = coupled_quaternion_rate(q, w);
                if !dz.iter().chain(&dq).all(|v| v.is_finite()) {
                    return Err(PyValueError::new_err(
                        "coupled stage callback returned invalid derivatives",
                    ));
                }
                Ok((dz, dq, counts))
            };
            let (next, counts) = coupled_integrate(
                state,
                start_time_s,
                dt_s,
                maximum_step_s,
                minimum_mass,
                derivative,
                PyValueError::new_err,
            )?;
            Ok((next.to_vec(), counts.to_vec()))
        }

        #[allow(clippy::too_many_arguments)]
        fn history<'py>(
            &self,
            py: Python<'py>,
            state: Vec<f64>,
            widths: &[u8],
            orbit_substep: Option<f64>,
            attitude_substep: Option<f64>,
            command: Vec<f64>,
            torque: Vec<f64>,
        ) -> PyResult<(Bound<'py, PyBytes>, Vec<i64>)> {
            if widths.len() % 8 != 0 || widths.len() / 8 > 1_000_000 {
                return Err(PyValueError::new_err("invalid history widths"));
            }
            let mut current: [f64; 13] = fixed(state)?;
            let command = fixed(command)?;
            let torque = fixed(torque)?;
            State::from_array(current[..6].try_into().unwrap()).map_err(PyValueError::new_err)?;
            let mut counts = [0_i64; 6];
            let mut bytes = Vec::with_capacity((widths.len() / 8 + 1) * 13 * 8);
            let mut record = |value: [f64; 13]| {
                for item in value {
                    bytes.extend_from_slice(&item.to_le_bytes());
                }
            };
            record(current);
            for chunk in widths.chunks_exact(8) {
                let h = f64::from_le_bytes(chunk.try_into().unwrap());
                let (next, stats) = self
                    .context
                    .step(current, h, orbit_substep, attitude_substep, command, torque)
                    .map_err(PyValueError::new_err)?;
                current = next;
                for i in 0..6 {
                    counts[i] += stats[i];
                }
                let mut sample = current;
                sample[6..10].copy_from_slice(&attitude::normalize_quaternion(
                    current[6..10].try_into().unwrap(),
                ));
                record(sample);
            }
            Ok((PyBytes::new(py, &bytes), counts.to_vec()))
        }
    }
    module.add_class::<NativeMechanicalContext>()?;
    Ok(())
}
