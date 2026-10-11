//! Trainer display forecasts. Keep the Python forecast's RK4 and interval schedule.
use crate::{cross, dot, norm};

fn derivative<const N: usize>(x: [f64; N], mu: f64) -> [f64; N] {
    let r = [x[0], x[1], x[2]];
    let v = [x[3], x[4], x[5]];
    let radius = norm(r);
    let mut dx = [0.0; N];
    if radius <= 1.0e-9 || !radius.is_finite() {
        return dx;
    }
    let gravity = mu / radius.powf(3.0);
    for i in 0..3 {
        dx[i] = v[i];
        dx[i + 3] = -mu * r[i] / radius.powf(3.0);
    }
    if N == 12 {
        let omega = norm(cross(r, v)) / (radius * radius).max(1.0e-12);
        let omega_dot = -2.0 * omega * (dot(r, v) / radius) / radius;
        let rho = [x[6], x[7], x[8]];
        let velocity = [x[9], x[10], x[11]];
        let coriolis = cross([0.0, 0.0, omega], velocity);
        let euler = cross([0.0, 0.0, omega_dot], rho);
        let centrifugal = cross([0.0, 0.0, omega], cross([0.0, 0.0, omega], rho));
        let gradient = [2.0 * rho[0], -rho[1], -rho[2]];
        for i in 0..3 {
            dx[i + 6] = velocity[i];
            dx[i + 9] = gravity * gradient[i] - 2.0 * coriolis[i] - euler[i] - centrifugal[i];
        }
    }
    dx
}

fn rk4<const N: usize>(x: [f64; N], h: f64, mu: f64) -> [f64; N] {
    let k1 = derivative(x, mu);
    let k2: [f64; N] = derivative(std::array::from_fn(|i| x[i] + 0.5 * h * k1[i]), mu);
    let k3: [f64; N] = derivative(std::array::from_fn(|i| x[i] + 0.5 * h * k2[i]), mu);
    let k4: [f64; N] = derivative(std::array::from_fn(|i| x[i] + h * k3[i]), mu);
    std::array::from_fn(|i| x[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

fn integrate<const N: usize>(
    mut x: [f64; N],
    duration: f64,
    step: f64,
    mu: f64,
) -> Result<[f64; N], String> {
    if !duration.is_finite() || !step.is_finite() || step <= 0.0 || !mu.is_finite() {
        return Err("forecast times, step and mu must be finite; step must be positive".into());
    }
    if (duration / step).ceil() > 2_000_000.0 {
        return Err("forecast exceeds native work bound".into());
    }
    let mut t = 0.0;
    while t < duration {
        let h = step.min(duration - t);
        x = rk4(x, h, mu);
        t += h;
    }
    Ok(x)
}

pub fn two_body(x: [f64; 6], duration: f64, mu: f64) -> Result<[f64; 6], String> {
    integrate(x, duration.max(0.0), (duration / 4.0).clamp(1.0, 10.0), mu)
}

pub fn th_history(
    relative: [f64; 6],
    chief: [f64; 6],
    times: Vec<f64>,
    step: f64,
    mu: f64,
) -> Result<Vec<[f64; 6]>, String> {
    if times.len() > 100_000 || times.iter().any(|t| !t.is_finite()) {
        return Err("forecast grid must be finite and bounded".into());
    }
    let mut x = std::array::from_fn(|i| if i < 6 { chief[i] } else { relative[i - 6] });
    let mut current = 0.0;
    let mut out = Vec::with_capacity(times.len());
    if !step.is_finite() || step <= 0.0 || !mu.is_finite() {
        return Err("forecast step and mu must be finite; step must be positive".into());
    }
    let mut steps = 0;
    // Caller sorts the grid and restores its original order, just like Python.
    for time in times {
        let target = time.max(current);
        while current < target {
            if steps >= 2_000_000 {
                return Err("forecast exceeds native work bound".into());
            }
            let h = step.min(target - current);
            x = rk4::<12>(x, h, mu);
            current += h;
            steps += 1;
        }
        out.push(std::array::from_fn(|i| x[i + 6]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn circular_chief_matches_analytic_orbit() {
        let mu: f64 = 398600.4418;
        let radius: f64 = 7000.0;
        let n = (mu / radius.powi(3)).sqrt();
        let actual = two_body([radius, 0.0, 0.0, 0.0, n * radius, 0.0], 600.0, mu).unwrap();
        assert!((actual[0] - radius * (n * 600.0).cos()).abs() < 1.0e-5);
        assert!((actual[1] - radius * (n * 600.0).sin()).abs() < 1.0e-5);
        assert!(two_body([0.0; 6], f64::INFINITY, mu).is_err());
        assert!(th_history([0.0; 6], [0.0; 6], vec![1.0], 0.0, mu).is_err());
    }
}
