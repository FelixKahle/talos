// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use pyo3::prelude::*;
use talos_ls::{LocalSearchMonitor, SearchProgress};

/// The monitor that calls an optional Python callback at every new best solution.
///
/// The callback receives `(objective: int, berths: list[int], start_times: list[int])`.
pub struct PyCallbackMonitor {
    callback: Option<Py<PyAny>>,
}

impl PyCallbackMonitor {
    pub fn new(callback: Option<Py<PyAny>>) -> Self {
        Self { callback }
    }
}

impl LocalSearchMonitor<i64> for PyCallbackMonitor {
    fn on_best_solution_updated(&mut self, progress: SearchProgress<'_, i64>) {
        let Some(callback) = &self.callback else {
            return;
        };
        let best = progress.best;
        let objective = best.objective_value();
        let berths: Vec<usize> = best
            .assigned_berths()
            .iter()
            .map(|berth| berth.index())
            .collect();
        let start_times: Vec<i64> = best.start_times().iter().copied().collect();
        Python::attach(|py| {
            if let Err(error) = callback.call1(py, (objective, berths, start_times)) {
                eprintln!("Python callback error: {}", error);
            }
        });
    }
}
