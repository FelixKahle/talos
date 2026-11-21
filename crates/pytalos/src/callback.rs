// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use pyo3::prelude::*;
use talos_model::solution::SolutionView;

/// Wraps an optional Python callback into a Rust closure.
///
/// The callback receives `(objective: int, berths: list[int], start_times: list[int])`.
#[allow(clippy::type_complexity)]
pub fn make_callback(py_cb: Option<Py<PyAny>>) -> Box<dyn FnMut(SolutionView<'_, i64>) + Send> {
    match py_cb {
        Some(cb) => Box::new(move |sol: SolutionView<'_, i64>| {
            let obj = sol.objective_value();
            let berths: Vec<usize> = sol.berths().iter().map(|bi| bi.index()).collect();
            let start_times: Vec<i64> = sol.start_times().to_vec();

            Python::attach(|py| {
                if let Err(e) = cb.call1(py, (obj, berths, start_times)) {
                    eprintln!("Python callback error: {}", e);
                }
            });
        }),
        None => Box::new(|_| {}),
    }
}
