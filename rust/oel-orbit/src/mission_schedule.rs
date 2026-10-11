//! Bounded exact subset search for validated public mission opportunities.

const EPS: f64 = 1.0e-12;

#[derive(Clone, Copy)]
pub struct Asset {
    pub storage_capacity: f64,
    pub initial_storage: f64,
    pub energy_budget: f64,
    pub maximum_duty: f64,
}

#[derive(Clone, Copy)]
pub struct Opportunity {
    pub asset: usize,
    pub kind: u8, // observation=0, downlink=1, other=2
    pub start: f64,
    pub end: f64,
    pub objective: f64,
    pub energy: f64,
    pub data: f64,
    pub downlink: f64,
}

fn lexicographically_before(left: usize, right: usize, count: usize) -> bool {
    (0..count)
        .find_map(|index| {
            let bit = 1_usize << index;
            if left & bit == right & bit {
                None
            } else {
                Some(left & bit != 0)
            }
        })
        .unwrap_or(false)
}

// The bounded nonnegative objective values use the same compensated partials
// and final half-even correction as Python math.fsum.
fn objective_fsum(values: impl Iterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for mut value in values {
        let mut used = 0;
        for index in 0..partials.len() {
            let mut part = partials[index];
            if value.abs() < part.abs() {
                std::mem::swap(&mut value, &mut part);
            }
            let high = value + part;
            let low = part - (high - value);
            if low != 0.0 {
                partials[used] = low;
                used += 1;
            }
            value = high;
        }
        partials.truncate(used);
        partials.push(value);
    }
    let Some(mut high) = partials.pop() else {
        return 0.0;
    };
    let mut low = 0.0;
    while let Some(part) = partials.pop() {
        let previous = high;
        high += part;
        low = part - (high - previous);
        if low != 0.0 {
            break;
        }
    }
    if let Some(&next) = partials.last() {
        if (low < 0.0 && next < 0.0) || (low > 0.0 && next > 0.0) {
            let correction = 2.0 * low;
            let candidate = high + correction;
            if correction == candidate - high {
                high = candidate;
            }
        }
    }
    high
}

#[allow(clippy::too_many_arguments)]
pub fn search(
    assets: &[Asset],
    opportunities: &[Opportunity],
    asset_orders: &[Vec<usize>],
    station_conflicts: &[usize],
    transition_conflicts: &[usize],
    minimum_observations: usize,
    require_delivery: bool,
    horizon: f64,
) -> Result<(Option<usize>, usize), String> {
    let count = opportunities.len();
    if count > 18
        || assets.is_empty()
        || asset_orders.len() != assets.len()
        || station_conflicts.len() != count
        || transition_conflicts.len() != count
        || !horizon.is_finite()
        || horizon <= 0.0
    {
        return Err("invalid bounded mission schedule dimensions".into());
    }
    let mut best_mask = None;
    let mut best_objective = 0.0;
    let mut feasible_count = 0;
    let mut time_order: Vec<usize> = (0..count).collect();
    time_order.sort_by(|&left, &right| {
        opportunities[left]
            .start
            .partial_cmp(&opportunities[right].start)
            .unwrap()
            .then_with(|| {
                opportunities[left]
                    .end
                    .partial_cmp(&opportunities[right].end)
                    .unwrap()
            })
            .then_with(|| left.cmp(&right))
    });
    for mask in 0..(1_usize << count) {
        let observations = opportunities
            .iter()
            .enumerate()
            .filter(|(index, opportunity)| mask & (1 << index) != 0 && opportunity.kind == 0)
            .count();
        if observations < minimum_observations {
            continue;
        }
        if (0..count)
            .any(|index| mask & (1 << index) != 0 && (mask & station_conflicts[index]) != 0)
        {
            continue;
        }
        let mut valid = true;
        for (asset_index, asset) in assets.iter().enumerate() {
            let mut storage = asset.initial_storage;
            let mut initial_remaining = asset.initial_storage;
            let mut energy = 0.0;
            let mut observation_duration = 0.0;
            let mut generated: Vec<(usize, f64)> = Vec::new();
            let mut previous = None;
            for &index in &asset_orders[asset_index] {
                if index >= count || opportunities[index].asset != asset_index {
                    return Err("invalid asset order".into());
                }
                if mask & (1 << index) == 0 {
                    continue;
                }
                if let Some(prior) = previous {
                    if transition_conflicts[prior] & (1 << index) != 0 {
                        valid = false;
                        break;
                    }
                }
                previous = Some(index);
                let item = opportunities[index];
                energy += item.energy;
                if energy > asset.energy_budget + EPS {
                    valid = false;
                    break;
                }
                if item.kind == 0 {
                    observation_duration += item.end - item.start;
                    if observation_duration / horizon > asset.maximum_duty + EPS {
                        valid = false;
                        break;
                    }
                    storage += item.data;
                    generated.push((index, item.data));
                    if storage > asset.storage_capacity + EPS {
                        valid = false;
                        break;
                    }
                } else if item.kind == 1 {
                    let transferred = item.downlink.min(storage);
                    storage -= transferred;
                    let mut remaining = transferred - initial_remaining.min(transferred);
                    initial_remaining -= initial_remaining.min(transferred);
                    for (_, amount) in &mut generated {
                        let consumed = amount.min(remaining);
                        *amount -= consumed;
                        remaining -= consumed;
                        if remaining <= EPS {
                            break;
                        }
                    }
                }
            }
            if valid && require_delivery && generated.iter().any(|(_, amount)| *amount > EPS) {
                valid = false;
            }
            if !valid {
                break;
            }
        }
        if !valid {
            continue;
        }
        feasible_count += 1;
        let objective = objective_fsum(time_order.iter().filter_map(|&index| {
            (mask & (1 << index) != 0).then_some(opportunities[index].objective)
        }));
        let selected_count = mask.count_ones();
        let better = best_mask.is_none_or(|prior: usize| {
            objective > best_objective
                || (objective == best_objective
                    && (selected_count < prior.count_ones()
                        || (selected_count == prior.count_ones()
                            && lexicographically_before(mask, prior, count))))
        });
        if better {
            best_mask = Some(mask);
            best_objective = objective;
        }
    }
    Ok((best_mask, feasible_count))
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::{exceptions::PyValueError, prelude::*};

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn mission_schedule_search(
        asset_rows: Vec<Vec<f64>>,
        opportunity_rows: Vec<Vec<f64>>,
        asset_orders: Vec<Vec<usize>>,
        station_conflicts: Vec<usize>,
        transition_conflicts: Vec<usize>,
        minimum_observations: usize,
        require_delivery: bool,
        horizon: f64,
    ) -> PyResult<(Option<usize>, usize)> {
        let assets: Vec<Asset> = asset_rows
            .iter()
            .map(|row| {
                if row.len() != 4 || row.iter().any(|value| !value.is_finite()) {
                    return Err(PyValueError::new_err("invalid asset row"));
                }
                Ok(Asset {
                    storage_capacity: row[0],
                    initial_storage: row[1],
                    energy_budget: row[2],
                    maximum_duty: row[3],
                })
            })
            .collect::<PyResult<_>>()?;
        let opportunities: Vec<Opportunity> = opportunity_rows
            .iter()
            .map(|row| {
                if row.len() != 8
                    || row.iter().any(|value| !value.is_finite())
                    || row[0] < 0.0
                    || row[0].fract() != 0.0
                    || row[1].fract() != 0.0
                    || !(0.0..=2.0).contains(&row[1])
                    || row[0] as usize >= assets.len()
                {
                    return Err(PyValueError::new_err("invalid opportunity row"));
                }
                Ok(Opportunity {
                    asset: row[0] as usize,
                    kind: row[1] as u8,
                    start: row[2],
                    end: row[3],
                    objective: row[4],
                    energy: row[5],
                    data: row[6],
                    downlink: row[7],
                })
            })
            .collect::<PyResult<_>>()?;
        search(
            &assets,
            &opportunities,
            &asset_orders,
            &station_conflicts,
            &transition_conflicts,
            minimum_observations,
            require_delivery,
            horizon,
        )
        .map_err(PyValueError::new_err)
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(mission_schedule_search, module)?)
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::objective_fsum;

    #[test]
    fn compensated_objective_preserves_small_terms() {
        assert_eq!(objective_fsum([1.0e16, 1.0, 1.0].into_iter()), 1.0e16 + 2.0);
        assert_eq!(objective_fsum([0.1, 0.2, 0.3].into_iter()), 0.6);
    }
}
