//! Scalar score geometry; decisions, thresholds and event policy stay in Python.
use crate::{norm, V3};

pub fn sample(
    relative: [f64; 6],
    thrust: V3,
    target_thrust: V3,
    goal_points: &[u8],
) -> Result<[f64; 5], String> {
    if !goal_points.len().is_multiple_of(24) || goal_points.len() > 100_000 * 24 {
        return Err("score geometry exceeds native work bound".into());
    }
    let position = [relative[0], relative[1], relative[2]];
    let mut distance = f64::INFINITY;
    for row in goal_points.as_chunks::<24>().0 {
        let point: V3 =
            std::array::from_fn(|i| f64::from_le_bytes(row[i * 8..i * 8 + 8].try_into().unwrap()));
        let candidate = norm(std::array::from_fn(|i| position[i] - point[i]));
        if candidate.is_nan() {
            distance = f64::NAN;
            break;
        }
        distance = distance.min(candidate);
    }
    Ok([
        norm(position),
        norm([relative[3], relative[4], relative[5]]),
        norm(thrust),
        norm(target_thrust),
        distance,
    ])
}
