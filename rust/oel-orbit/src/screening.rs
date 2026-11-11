//! Numeric sampled-pair discovery for Scale's spatial-hash screening stage.
//! Python retains store, promotion, and evidence ownership.

use std::collections::{hash_map::Entry, HashMap};

#[derive(Clone, Copy)]
struct Row {
    object: usize,
    sample_index: i64,
    jd_utc: f64,
    position: [f64; 3],
    velocity: [f64; 3],
}

#[derive(Clone, Copy)]
struct Best {
    comparison: f64,
    sample_index: i64,
    distance: f64,
    left_row: usize,
    right_row: usize,
}

pub struct PairHit {
    pub left: usize,
    pub right: usize,
    pub distance_km: f64,
    pub jd_utc: f64,
    pub relative_speed_km_s: f64,
}

/// Each state row is `jd, px, py, pz, vx, vy, vz`. Objects are indexed in
/// lexicographic product-ID order before crossing the Python boundary.
pub fn discover_pairs(
    object_indices: &[usize],
    sample_indices: &[i64],
    states: &[f64],
    threshold_km: f64,
    squared_comparison: bool,
) -> Result<Vec<PairHit>, String> {
    let count = object_indices.len();
    if !threshold_km.is_finite() || threshold_km <= 0.0 {
        return Err("screening threshold must be positive and finite".into());
    }
    if sample_indices.len() != count || states.len() != 7 * count {
        return Err("screening input arrays have inconsistent lengths".into());
    }
    let mut rows = Vec::with_capacity(count);
    let mut by_sample: HashMap<i64, Vec<usize>> = HashMap::new();
    let mut sample_order = Vec::new();
    for index in 0..count {
        let values = &states[index * 7..index * 7 + 7];
        let position = [values[1], values[2], values[3]];
        if !position.iter().all(|value| value.is_finite()) {
            return Err("screening positions must be finite".into());
        }
        rows.push(Row {
            object: object_indices[index],
            sample_index: sample_indices[index],
            jd_utc: values[0],
            position,
            velocity: [values[4], values[5], values[6]],
        });
        match by_sample.entry(sample_indices[index]) {
            Entry::Vacant(entry) => {
                sample_order.push(sample_indices[index]);
                entry.insert(vec![index]);
            }
            Entry::Occupied(mut entry) => entry.get_mut().push(index),
        }
    }

    let threshold_comparison = if squared_comparison {
        threshold_km * threshold_km
    } else {
        threshold_km
    };
    let mut best_by_pair: HashMap<(usize, usize), Best> = HashMap::new();
    for sample_index in sample_order {
        let indices = &by_sample[&sample_index];
        let mut buckets: HashMap<[i64; 3], Vec<usize>> = HashMap::with_capacity(indices.len());
        let mut cell_order = Vec::new();
        for &index in indices {
            let mut cell = [0_i64; 3];
            for (axis, coordinate) in rows[index].position.iter().enumerate() {
                let scaled = (*coordinate / threshold_km).floor();
                if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled >= i64::MAX as f64 {
                    return Err("screening cell coordinate exceeds the native range".into());
                }
                cell[axis] = scaled as i64;
            }
            match buckets.entry(cell) {
                Entry::Vacant(entry) => {
                    cell_order.push(cell);
                    entry.insert(vec![index]);
                }
                Entry::Occupied(mut entry) => entry.get_mut().push(index),
            }
        }
        for cell in cell_order {
            let cell_rows = &buckets[&cell];
            // Only the lexicographically current/later half of the 27
            // neighboring cells can contribute. Retain their original order.
            for dx in 0_i64..=1 {
                for dy in (if dx == 0 { 0_i64 } else { -1_i64 })..=1 {
                    for dz in (if dx == 0 && dy == 0 { 0_i64 } else { -1_i64 })..=1 {
                        let Some(nx) = cell[0].checked_add(dx) else {
                            continue;
                        };
                        let Some(ny) = cell[1].checked_add(dy) else {
                            continue;
                        };
                        let Some(nz) = cell[2].checked_add(dz) else {
                            continue;
                        };
                        let neighbor = [nx, ny, nz];
                        let Some(neighbor_rows) = buckets.get(&neighbor) else {
                            continue;
                        };
                        for &left_index in cell_rows {
                            for &right_index in neighbor_rows {
                                let left = rows[left_index];
                                let right = rows[right_index];
                                if left.object == right.object
                                    || (neighbor == cell && left.object >= right.object)
                                {
                                    continue;
                                }
                                let delta = [
                                    right.position[0] - left.position[0],
                                    right.position[1] - left.position[1],
                                    right.position[2] - left.position[2],
                                ];
                                let squared_distance =
                                    delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
                                let comparison = if squared_comparison {
                                    squared_distance
                                } else {
                                    squared_distance.sqrt()
                                };
                                if comparison > threshold_comparison {
                                    continue;
                                }
                                let pair =
                                    (left.object.min(right.object), left.object.max(right.object));
                                let (ordered_left, ordered_right) = if left.object == pair.0 {
                                    (left_index, right_index)
                                } else {
                                    (right_index, left_index)
                                };
                                let candidate = Best {
                                    comparison,
                                    sample_index: rows[ordered_left].sample_index,
                                    distance: comparison,
                                    left_row: ordered_left,
                                    right_row: ordered_right,
                                };
                                let replace = best_by_pair.get(&pair).is_none_or(|previous| {
                                    (candidate.comparison, candidate.sample_index)
                                        < (previous.comparison, previous.sample_index)
                                });
                                if replace {
                                    best_by_pair.insert(pair, candidate);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut pairs: Vec<_> = best_by_pair.into_iter().collect();
    pairs.sort_by_key(|(pair, _)| *pair);
    Ok(pairs
        .into_iter()
        .map(|((left, right), best)| {
            let left_row = rows[best.left_row];
            let right_row = rows[best.right_row];
            let delta = [
                right_row.velocity[0] - left_row.velocity[0],
                right_row.velocity[1] - left_row.velocity[1],
                right_row.velocity[2] - left_row.velocity[2],
            ];
            PairHit {
                left,
                right,
                distance_km: if squared_comparison {
                    best.distance.sqrt()
                } else {
                    best.distance
                },
                jd_utc: left_row.jd_utc,
                relative_speed_km_s: (delta[0] * delta[0]
                    + delta[1] * delta[1]
                    + delta[2] * delta[2])
                    .sqrt(),
            }
        })
        .collect())
}
