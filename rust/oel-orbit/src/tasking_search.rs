//! Exact bounded coverage-tasking search. Python owns validation and products.

use pyo3::{exceptions::PyValueError, prelude::*};

struct Search<'a> {
    ids: &'a [String],
    rows: &'a [f64], // start, end, value, storage, energy, observation duration
    suffix: &'a [f64],
    transitions: &'a [f64],
    capacity: f64,
    budget: f64,
    duty: f64,
    horizon: f64,
    best_value: f64,
    best: Vec<usize>,
    evaluated: u64,
}

impl Search<'_> {
    fn better(&self, value: f64, chosen: &[usize]) -> bool {
        if value > self.best_value {
            return true;
        }
        if value != self.best_value {
            return false;
        }
        let mut new_ids: Vec<&str> = chosen.iter().map(|&i| self.ids[i].as_str()).collect();
        let mut best_ids: Vec<&str> = self.best.iter().map(|&i| self.ids[i].as_str()).collect();
        new_ids.sort_unstable();
        best_ids.sort_unstable();
        new_ids < best_ids
    }

    fn visit(
        &mut self,
        index: usize,
        chosen: &mut Vec<usize>,
        value: f64,
        storage: f64,
        energy: f64,
        observation: f64,
    ) {
        let n = self.ids.len();
        if value + self.suffix[index] < self.best_value {
            return;
        }
        if index == n {
            self.evaluated += 1;
            if self.better(value, chosen) {
                self.best_value = value;
                self.best.clone_from(chosen);
            }
            return;
        }
        self.visit(index + 1, chosen, value, storage, energy, observation);
        let row = &self.rows[index * 6..index * 6 + 6];
        if let Some(&previous) = chosen.last() {
            let previous_end = self.rows[previous * 6 + 1];
            if row[0] < previous_end - 1.0e-12 {
                return;
            }
            if row[0] - previous_end + 1.0e-12 < self.transitions[previous * n + index] {
                return;
            }
        }
        let next_storage = storage + row[3];
        if next_storage < 0.0 || next_storage > self.capacity {
            return;
        }
        let next_energy = energy + row[4];
        if next_energy > self.budget {
            return;
        }
        let next_observation = observation + row[5];
        if next_observation / self.horizon > self.duty {
            return;
        }
        chosen.push(index);
        self.visit(
            index + 1,
            chosen,
            value + row[2],
            next_storage,
            next_energy,
            next_observation,
        );
        chosen.pop();
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn coverage_tasking_exact_search(
    py: Python<'_>,
    ids: Vec<String>,
    rows: Vec<f64>,
    suffix: Vec<f64>,
    transitions: Vec<f64>,
    initial_storage: f64,
    capacity: f64,
    budget: f64,
    duty: f64,
    horizon: f64,
) -> PyResult<(Vec<usize>, f64, u64)> {
    let n = ids.len();
    if n == 0
        || n > 24
        || rows.len() != n * 6
        || suffix.len() != n + 1
        || transitions.len() != n * n
        || rows
            .iter()
            .chain(&suffix)
            .chain(&transitions)
            .any(|v| !v.is_finite())
        || [initial_storage, capacity, budget, duty, horizon]
            .iter()
            .any(|v| !v.is_finite())
        || horizon <= 0.0
    {
        return Err(PyValueError::new_err(
            "invalid coverage tasking search inputs",
        ));
    }
    let mut search = Search {
        ids: &ids,
        rows: &rows,
        suffix: &suffix,
        transitions: &transitions,
        capacity,
        budget,
        duty,
        horizon,
        best_value: -1.0,
        best: Vec::new(),
        evaluated: 0,
    };
    py.detach(|| search.visit(0, &mut Vec::new(), 0.0, initial_storage, 0.0, 0.0));
    Ok((search.best, search.best_value, search.evaluated))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(coverage_tasking_exact_search, module)?)
}
