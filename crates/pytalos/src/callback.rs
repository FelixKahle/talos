// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use pyo3::prelude::*;
use std::time::{Duration, Instant};
use talos_ls::{LocalSearchMonitor, SearchProgress};
use talos_model::Model;

/// The monitor that calls an optional Python callback at every new best solution, and times the search.
///
/// The callback receives `(objective: int, berths: list[int], start_times: list[int])`.
pub struct PyCallbackMonitor {
    callback: Option<Py<PyAny>>,
    /// When the search started, once it has.
    started: Option<Instant>,
    /// The time from the start of the search to its end, zero until it has ended.
    elapsed: Duration,
}

impl PyCallbackMonitor {
    pub fn new(callback: Option<Py<PyAny>>) -> Self {
        Self {
            callback,
            started: None,
            elapsed: Duration::ZERO,
        }
    }

    /// The time the search took, from its start to its end, without the loading of the schedule before it.
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

impl LocalSearchMonitor<i64> for PyCallbackMonitor {
    fn on_start(&mut self, _model: &Model<i64>, _progress: SearchProgress<'_, i64>) {
        self.started = Some(Instant::now());
    }

    fn on_end(&mut self, _progress: SearchProgress<'_, i64>) {
        self.elapsed = self
            .started
            .map_or(Duration::ZERO, |started| started.elapsed());
    }

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
