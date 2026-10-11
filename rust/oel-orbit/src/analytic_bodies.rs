//! Low-cost analytic ephemerides, matching orbit/epoch.py in J2000.
use crate::environment::{frame_ry, frame_rz, mat_mul, mat_vec, ARCSEC_TO_RAD};
fn wrap(x: f64) -> f64 {
    x.rem_euclid(360.)
}
fn transpose(m: [f64; 9]) -> [f64; 9] {
    std::array::from_fn(|i| m[(i % 3) * 3 + i / 3])
}
fn prec(jd: f64) -> [f64; 9] {
    let t = (jd - 2451545.) / 36525.;
    let t2 = t * t;
    let t3 = t2 * t;
    let zeta = (2306.2181 * t + 0.30188 * t2 + 0.017998 * t3) * ARCSEC_TO_RAD;
    let theta = (2004.3109 * t - 0.42665 * t2 - 0.041833 * t3) * ARCSEC_TO_RAD;
    let z = (2306.2181 * t + 1.09468 * t2 + 0.018203 * t3) * ARCSEC_TO_RAD;
    mat_mul(&mat_mul(&frame_rz(-z), &frame_ry(theta)), &frame_rz(-zeta))
}
pub(crate) fn bodies(
    jd: f64,
    enhanced: bool,
    nutation: Option<[f64; 9]>,
    sun_only: bool,
) -> ([f64; 3], [f64; 3]) {
    let t = (jd - 2451545.) / 36525.;
    let n = jd - 2451545.;
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    if !enhanced {
        let l = wrap(280.460 + 0.9856474 * n);
        let g = wrap(357.528 + 0.9856003 * n).to_radians();
        let lam = (l + 1.915 * g.sin() + 0.020 * (2. * g).sin()).to_radians();
        let eps = (23.439 - 0.0000004 * n).to_radians();
        let r = (1.00014 - 0.01671 * g.cos() - 0.00014 * (2. * g).cos()) * 149597870.7;
        let sun = [
            r * lam.cos(),
            r * eps.cos() * lam.sin(),
            r * eps.sin() * lam.sin(),
        ];
        if sun_only {
            return (mat_vec(&transpose(prec(jd)), sun), [0.; 3]);
        }
        let l0 = wrap(218.316 + 13.176396 * n).to_radians();
        let m = wrap(134.963 + 13.064993 * n).to_radians();
        let f = wrap(93.272 + 13.229350 * n).to_radians();
        let lon = l0 + 6.289_f64.to_radians() * m.sin();
        let lat = 5.128_f64.to_radians() * f.sin();
        let r = 385001. - 20905. * m.cos();
        let x = r * lat.cos() * lon.cos();
        let y = r * lat.cos() * lon.sin();
        let z = r * lat.sin();
        return (
            mat_vec(&transpose(prec(jd)), sun),
            mat_vec(
                &transpose(prec(jd)),
                [
                    x,
                    y * eps.cos() - z * eps.sin(),
                    y * eps.sin() + z * eps.cos(),
                ],
            ),
        );
    }
    let l0 = wrap(280.46646 + 36000.76983 * t + 0.0003032 * t2);
    let mdeg = wrap(357.52911 + 35999.05029 * t - 0.0001537 * t2);
    let m = mdeg.to_radians();
    let e = 0.016708634 - 0.000042037 * t - 0.0000001267 * t2;
    let c = (1.914602 - 0.004817 * t - 0.000014 * t2) * m.sin()
        + (0.019993 - 0.000101 * t) * (2. * m).sin()
        + 0.000289 * (3. * m).sin();
    let long = wrap(l0 + c);
    let anomaly = wrap(mdeg + c).to_radians();
    let r = (1.000001018 * (1. - e * e)) / (1. + e * anomaly.cos()) * 149597870.7;
    let omega = wrap(125.04 - 1934.136 * t).to_radians();
    let lam = (long - 0.00569 - 0.00478 * omega.sin()).to_radians();
    let eps0 =
        23. + 26. / 60. + 21.448 / 3600. - (46.8150 * t + 0.00059 * t2 - 0.001813 * t3) / 3600.;
    let eps = (eps0 + 0.00256 * omega.cos()).to_radians();
    let tod = [
        r * lam.cos(),
        r * eps.cos() * lam.sin(),
        r * eps.sin() * lam.sin(),
    ];
    let sun = mat_vec(
        &mat_mul(&transpose(prec(jd + 69.184 / 86400.)), &nutation.unwrap()),
        tod,
    );
    if sun_only {
        return (sun, [0.; 3]);
    }
    let lp =
        wrap(218.3164477 + 481267.88123421 * t - 0.0015786 * t2 + t3 / 538841. - t4 / 65194000.);
    let d =
        wrap(297.8501921 + 445267.1114034 * t - 0.0018819 * t2 + t3 / 545868. - t4 / 113065000.)
            .to_radians();
    let m = wrap(357.5291092 + 35999.0502909 * t - 0.0001536 * t2 + t3 / 24490000.).to_radians();
    let mp = wrap(134.9633964 + 477198.8675055 * t + 0.0087414 * t2 + t3 / 69699. - t4 / 14712000.)
        .to_radians();
    let f =
        wrap(93.2720950 + 483202.0175233 * t - 0.0036539 * t2 - t3 / 3526000. + t4 / 863310000.)
            .to_radians();
    let lon = wrap(
        lp + 6.289 * mp.sin()
            + 1.274 * (2. * d - mp).sin()
            + 0.658 * (2. * d).sin()
            + 0.214 * (2. * mp).sin()
            + 0.11 * d.sin(),
    )
    .to_radians();
    let lat = (5.128 * f.sin()
        + 0.280 * (mp + f).sin()
        + 0.277 * (mp - f).sin()
        + 0.173 * (2. * d - f).sin()
        + 0.055 * (2. * d + f - mp).sin()
        + 0.046 * (2. * d - f - mp).sin()
        + 0.033 * (2. * d + f).sin()
        + 0.017 * (2. * mp + f).sin())
    .to_radians();
    let r = 385000.56
        - 20905. * mp.cos()
        - 3699. * (2. * d - mp).cos()
        - 2956. * (2. * d).cos()
        - 570. * (2. * mp).cos()
        + 246. * (2. * mp - 2. * d).cos()
        - 205. * (m - 2. * d).cos()
        - 171. * (mp + 2. * d).cos();
    let eps = (23.439291 - 0.0130042 * t).to_radians();
    let x = r * lat.cos() * lon.cos();
    let y = r * lat.cos() * lon.sin();
    let z = r * lat.sin();
    (
        sun,
        mat_vec(
            &transpose(prec(jd)),
            [
                x,
                y * eps.cos() - z * eps.sin(),
                y * eps.sin() + z * eps.cos(),
            ],
        ),
    )
}
