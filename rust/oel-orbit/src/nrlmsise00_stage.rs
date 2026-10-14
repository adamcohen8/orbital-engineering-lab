//! Calendar packet assembly matching OEL's Python Julian-date owner.
//! Solar projection remains with its NumPy owner; density uses the unchanged core.
pub(crate) fn packet(
    jd_utc: f64,
    coordinates: [f64; 3],
    lst: f64,
    drivers: &[f64; 10],
) -> Option<[f64; 16]> {
    if !jd_utc.is_finite() || !(1721425.5..5373484.5).contains(&jd_utc) {
        return None;
    }
    let jd = jd_utc + 0.5;
    let z = jd.floor() as i64;
    let f = jd - z as f64;
    let a = if z < 2299161 {
        z
    } else {
        let alpha = ((z as f64 - 1867216.25) / 36524.25) as i64;
        z + 1 + alpha - alpha / 4
    };
    let b = a + 1524;
    let c = ((b as f64 - 122.1) / 365.25) as i64;
    let d = (365.25 * c as f64) as i64;
    let e = ((b - d) as f64 / 30.6001) as i64;
    let day = (b - d - (30.6001 * e as f64) as i64) as f64 + f;
    let month = if e < 14 { e - 1 } else { e - 13 };
    let year = if month > 2 { c - 4716 } else { c - 4715 };
    let mut day_int = day.floor() as i64;
    let mut sec_total = (day - day_int as f64) * 86400.;
    let mut hour = (sec_total / 3600.).floor() as i64;
    sec_total -= hour as f64 * 3600.;
    let mut minute = (sec_total / 60.).floor() as i64;
    sec_total -= minute as f64 * 60.;
    let mut second = sec_total.floor() as i64;
    let mut usec = ((sec_total - second as f64) * 1e6).round_ties_even() as i64;
    if usec >= 1_000_000 {
        usec -= 1_000_000;
        second += 1;
    }
    if second >= 60 {
        second -= 60;
        minute += 1;
    }
    if minute >= 60 {
        minute -= 60;
        hour += 1;
    }
    if hour >= 24 {
        hour -= 24;
        day_int += 1;
    }
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return None;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = [
        31_i64,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=days[(month - 1) as usize]).contains(&day_int) {
        return None;
    }
    let doy = days[..(month - 1) as usize].iter().sum::<i64>() + day_int;
    let sec = hour as f64 * 3600. + minute as f64 * 60. + second as f64 + usec as f64 * 1e-6;
    let mut values = [0.; 16];
    values[..6].copy_from_slice(&[
        doy as f64,
        sec,
        coordinates[0].max(0.),
        coordinates[1],
        coordinates[2],
        lst,
    ]);
    values[6..].copy_from_slice(drivers);
    Some(values)
}
#[cfg(test)]
mod tests {
    use super::packet;
    #[test]
    fn utc_days_and_cutoffs() {
        let drivers = [150., 175., 4., 4., 4., 4., 4., 4., 4., 4.];
        for (jd, doy) in [(2451544.5, 1.), (2451603.5, 60.), (2459669.5, 90.)] {
            for alt in [199.999999, 200., 200.000001, 299.999999, 300., 300.000001] {
                let row = packet(jd, [alt, 20., 30.], 14., &drivers).unwrap();
                assert_eq!(row[0], doy);
                assert_eq!(row[1], 0.);
                assert_eq!(row[2], alt);
                assert_eq!(&row[6..], &drivers);
            }
        }
        assert!(packet(f64::NAN, [400., 0., 0.], 12., &drivers).is_none());
    }
}
