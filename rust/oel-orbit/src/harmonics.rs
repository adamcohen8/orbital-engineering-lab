//! Normalized spherical-harmonic perturbation kernel, matching OEL's degree-major recurrence.

pub const MAX_DEGREE: usize = 70;

#[derive(Clone, Debug)]
pub struct HarmonicTerms {
    pub n_max: usize,
    pub m_max: usize,
    pub c_nm: Vec<f64>,
    pub s_nm: Vec<f64>,
    pub diag: Vec<f64>,
    pub subdiag: Vec<f64>,
    pub recur_a: Vec<f64>,
    pub recur_b: Vec<f64>,
    pub recur_c: Vec<f64>,
}

impl HarmonicTerms {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.n_max > MAX_DEGREE || self.m_max > self.n_max {
            return Err("harmonic degree/order outside the supported 0..70 range");
        }
        let matrix_len = (self.n_max + 1) * (self.m_max + 1);
        if self.c_nm.len() != matrix_len
            || self.s_nm.len() != matrix_len
            || self.recur_a.len() != matrix_len
            || self.recur_b.len() != matrix_len
            || self.recur_c.len() != matrix_len
            || self.diag.len() != self.n_max + 1
            || self.subdiag.len() != self.n_max + 1
        {
            return Err("harmonic coefficient/recurrent table shape mismatch");
        }
        if ![
            &self.c_nm,
            &self.s_nm,
            &self.diag,
            &self.subdiag,
            &self.recur_a,
            &self.recur_b,
            &self.recur_c,
        ]
        .iter()
        .all(|values| values.iter().all(|value| value.is_finite()))
        {
            return Err("harmonic tables must contain only finite values");
        }
        Ok(())
    }

    fn index(&self, degree: usize, order: usize) -> usize {
        degree * (self.m_max + 1) + order
    }

    pub fn acceleration_eci(
        &self,
        position_eci_km: [f64; 3],
        rotation: [f64; 9],
        mu_km3_s2: f64,
        reference_radius_km: f64,
    ) -> [f64; 3] {
        let r = mat_vec(rotation, position_eci_km);
        let [x, y, z] = r;
        let distance_squared = x * x + y * y + z * z;
        if distance_squared <= 0.0 {
            return [0.0; 3];
        }
        let distance = distance_squared.sqrt();
        let latitude_gc = (z / distance).asin();
        let sin_lat = latitude_gc.sin();
        let cos_lat = latitude_gc.cos();
        // The degree-major recurrence needs only the preceding two rows.
        // Keep three bounded rows on the stack instead of allocating full
        // degree/order matrices at every force evaluation. Arithmetic and
        // degree/order accumulation remain identical to the full-table path.
        let mut p = [[0.0; MAX_DEGREE + 1]; 3];
        let mut dp = [[0.0; MAX_DEGREE + 1]; 3];
        p[0][0] = 1.0;
        let longitude = y.atan2(x);
        let mut cos_ml = [0.0; MAX_DEGREE + 1];
        let mut sin_ml = [0.0; MAX_DEGREE + 1];
        for order in 0..=self.m_max {
            cos_ml[order] = ((order as f64) * longitude).cos();
            sin_ml[order] = ((order as f64) * longitude).sin();
        }
        let mut radial_derivative = 0.0;
        let mut latitude_derivative = 0.0;
        let mut longitude_derivative = 0.0;
        for degree in 0..=self.n_max {
            let row = degree % 3;
            let prior = (degree + 2) % 3;
            let prior_two = (degree + 1) % 3;
            if degree == 1 && self.m_max >= 1 {
                let sqrt_three = 3.0_f64.sqrt();
                p[row][1] = sqrt_three * cos_lat;
                dp[row][1] = -sqrt_three * sin_lat;
            } else if degree >= 2 && degree <= self.m_max {
                let scale = self.diag[degree];
                let previous = p[prior][degree - 1];
                p[row][degree] = scale * cos_lat * previous;
                dp[row][degree] = scale * (cos_lat * dp[prior][degree - 1] - sin_lat * previous);
            }
            if degree >= 1 && degree - 1 <= self.m_max {
                let order = degree - 1;
                let scale = self.subdiag[degree];
                p[row][order] = scale * sin_lat * p[prior][order];
                dp[row][order] = scale * (cos_lat * p[prior][order] + sin_lat * dp[prior][order]);
            }
            if degree >= 2 {
                for order in 0..=self.m_max.min(degree - 2) {
                    let idx = self.index(degree, order);
                    let a = self.recur_a[idx];
                    let b = self.recur_b[idx];
                    let c = self.recur_c[idx];
                    let b_value = b * sin_lat * p[prior][order];
                    let c_value = c * p[prior_two][order];
                    p[row][order] = a * (b_value - c_value);
                    let db_value = b * sin_lat * dp[prior][order];
                    let dc_value = b * cos_lat * p[prior][order];
                    let dd_value = c * dp[prior_two][order];
                    dp[row][order] = a * (db_value + dc_value - dd_value);
                }
            }
            let radius_power = (reference_radius_km / distance).powi(degree as i32);
            let radial_scale =
                (-mu_km3_s2 / distance_squared) * radius_power * (degree as f64 + 1.0);
            let angular_scale = (mu_km3_s2 / distance) * radius_power;
            let mut radial_sum = 0.0;
            let mut latitude_sum = 0.0;
            let mut longitude_sum = 0.0;
            for order in 0..=self.m_max.min(degree) {
                let idx = self.index(degree, order);
                let amplitude = self.c_nm[idx] * cos_ml[order] + self.s_nm[idx] * sin_ml[order];
                radial_sum += p[row][order] * amplitude;
                latitude_sum += dp[row][order] * amplitude;
                longitude_sum += (order as f64)
                    * p[row][order]
                    * (self.s_nm[idx] * cos_ml[order] - self.c_nm[idx] * sin_ml[order]);
            }
            radial_derivative += radial_sum * radial_scale;
            latitude_derivative += latitude_sum * angular_scale;
            longitude_derivative += longitude_sum * angular_scale;
        }
        let xy_squared = (x * x + y * y).max(f64::EPSILON);
        let xy = xy_squared.sqrt();
        let common =
            radial_derivative / distance - z * latitude_derivative / (distance_squared * xy);
        let a_body = [
            common * x - longitude_derivative * y / xy_squared,
            common * y + longitude_derivative * x / xy_squared,
            radial_derivative * z / distance + xy * latitude_derivative / distance_squared,
        ];
        let mut a = mat_t_vec(rotation, a_body);
        let r2_eci = position_eci_km[0] * position_eci_km[0]
            + position_eci_km[1] * position_eci_km[1]
            + position_eci_km[2] * position_eci_km[2];
        if r2_eci <= 0.0 {
            return [0.0; 3];
        }
        let two_body_scale = -mu_km3_s2 / (r2_eci.sqrt() * r2_eci);
        for axis in 0..3 {
            a[axis] -= two_body_scale * position_eci_km[axis];
        }
        a
    }
}

fn mat_vec(m: [f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

fn mat_t_vec(m: [f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}
