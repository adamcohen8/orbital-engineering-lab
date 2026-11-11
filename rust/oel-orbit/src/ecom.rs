//! Seven-coefficient ECOM2; km/s² at one AU, D toward Sun, Y=D×R, B=D×Y.
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(a: [f64; 3]) -> Result<[f64; 3], String> {
    let norm = dot(a, a).sqrt();
    if norm <= 1e-12 {
        return Err("ECOM2 Sun/orbit frame is singular".into());
    }
    Ok(a.map(|x| x / norm))
}
pub fn acceleration(state: [f64; 6], sun: [f64; 3], c: [f64; 7]) -> Result<[f64; 3], String> {
    if state
        .iter()
        .chain(sun.iter())
        .chain(c.iter())
        .any(|x| !x.is_finite())
    {
        return Err("ECOM2 requires finite inputs".into());
    }
    let r: [f64; 3] = state[..3].try_into().unwrap();
    let v: [f64; 3] = state[3..].try_into().unwrap();
    let delta = std::array::from_fn(|i| sun[i] - r[i]);
    let distance = dot(delta, delta).sqrt();
    let rn = dot(r, r).sqrt();
    let d = unit(delta)?;
    let radial = unit(r)?;
    let n = unit(cross(r, v))?;
    let y = unit(cross(d, radial))?;
    let b = cross(d, y);
    let projected = unit(std::array::from_fn(|i| d[i] - dot(d, n) * n[i]))?;
    let cs = dot(radial, projected);
    let sn = dot(radial, cross(n, projected));
    let mut shadow = 1.;
    if rn <= 6378.137 {
        shadow = 0.;
    } else if dot(r, sun) < 0. {
        let alpha = (6378.137 / rn).clamp(-1., 1.).asin();
        let beta = (695700. / distance).clamp(-1., 1.).asin();
        let sep = (-dot(r, delta) / (rn * distance)).clamp(-1., 1.).acos();
        shadow = crate::force_plan::illumination(alpha, beta, sep);
    }
    let scale = (149597870.7 / distance).powi(2) * shadow;
    Ok(std::array::from_fn(|i| {
        scale
            * ((c[0] + c[3] * (2. * cs * cs - 1.) + c[4] * 2. * sn * cs) * d[i]
                + c[1] * y[i]
                + (c[2] + c[5] * cs + c[6] * sn) * b[i])
    }))
}
