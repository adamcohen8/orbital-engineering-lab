//! Scalar score geometry; decisions, thresholds and event policy stay in Python.
use crate::{norm, V3};

pub fn sample(
    relative: [f64; 6],
    thrust: V3,
    target_thrust: V3,
    goal_points: &[u8],
) -> Result<[f64; 5], String> {
    if goal_points.len() % 24 != 0 || goal_points.len() > 100_000 * 24 {
        return Err("score geometry exceeds native work bound".into());
    }
    let position = [relative[0], relative[1], relative[2]];
    let mut distance = f64::INFINITY;
    for row in goal_points.chunks_exact(24) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_goal_points_preserve_scalar_score_geometry() {
        let points: Vec<u8> = [10_f64, 0., 0., 3., 4., 0.]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        assert_eq!(
            sample(
                [0., 0., 0., 0., 0., 2.],
                [3., 4., 0.],
                [0., 0., 1.],
                &points
            )
            .unwrap(),
            [0., 2., 5., 1., 5.],
        );
    }

    #[test]
    fn empty_and_malformed_geometry_keep_the_boundary_contract() {
        let score = |points: &[u8]| sample([0.; 6], [0.; 3], [0.; 3], points);
        assert_eq!(score(&[]).unwrap()[4], f64::INFINITY);
        assert!(score(&[0; 23]).is_err());
        assert!(score(&vec![0; 100_001 * 24]).is_err());
        assert!(score(&vec![0; 100_000 * 24]).is_ok());
        let nan: Vec<u8> = [f64::NAN, 0., 0.]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        assert!(score(&nan).unwrap()[4].is_nan());
    }
}
