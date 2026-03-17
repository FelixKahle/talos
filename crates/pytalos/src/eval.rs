// Copyright (c) 2026 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use pyo3::prelude::*;
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
    solution::SolutionView,
};

/// Default evaluator using weighted turnaround time.
pub fn wtt_evaluator(
    model: &Model<i64>,
    vessel: VesselIndex,
    berth: BerthIndex,
    start: i64,
) -> Option<i64> {
    unsafe {
        talos_ls::eval::calculate_weighted_turnaround_time_unchecked(model, vessel, berth, start)
    }
}

/// Wraps an optional Python callback into a Rust closure.
///
/// The callback receives `(objective: int, berths: list[int], start_times: list[int])`.
#[allow(clippy::type_complexity)]
pub fn make_callback(py_cb: Option<Py<PyAny>>) -> Box<dyn FnMut(SolutionView<'_, i64>) + Send> {
    match py_cb {
        Some(cb) => Box::new(move |sol: SolutionView<'_, i64>| {
            let obj = sol.objective_value();
            let berths: Vec<usize> = sol.berths().iter().map(|bi| bi.get()).collect();
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
