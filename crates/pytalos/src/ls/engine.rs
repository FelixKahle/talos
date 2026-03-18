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

use itertools::Itertools;
use pyo3::prelude::*;

/// Neighbourhood operator.
#[pyclass(name = "Operator", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PyOperator {
    IntraSwap = 0,
    InterSwap = 1,
    IntraShift = 2,
    InterShift = 3,
}

/// Configuration for the local search engine.
#[pyclass(name = "LocalSearchConfig", from_py_object)]
#[derive(Clone)]
pub struct PyLocalSearchConfig {
    #[pyo3(get)]
    pub operators: Vec<PyOperator>,
    #[pyo3(get)]
    pub max_iterations: Option<u64>,
    #[pyo3(get)]
    pub max_solutions: Option<u64>,
    #[pyo3(get)]
    pub max_cycles: Option<u64>,
    #[pyo3(get)]
    pub max_non_improving_iterations: Option<u64>,
    #[pyo3(get)]
    pub max_non_improving_cycles: Option<u64>,
    #[pyo3(get)]
    pub max_non_improving_time_secs: Option<f64>,
    #[pyo3(get)]
    pub time_limit_secs: Option<f64>,
}

#[pymethods]
impl PyLocalSearchConfig {
    #[new]
    #[pyo3(signature = (
        operators,
        max_iterations = None,
        max_solutions = None,
        max_cycles = None,
        max_non_improving_iterations = None,
        max_non_improving_cycles = None,
        max_non_improving_time_secs = None,
        time_limit_secs = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        operators: Vec<PyOperator>,
        max_iterations: Option<u64>,
        max_solutions: Option<u64>,
        max_cycles: Option<u64>,
        max_non_improving_iterations: Option<u64>,
        max_non_improving_cycles: Option<u64>,
        max_non_improving_time_secs: Option<f64>,
        time_limit_secs: Option<f64>,
    ) -> Self {
        Self {
            operators: operators.into_iter().unique_by(|op| *op as u8).collect(),
            max_iterations,
            max_solutions,
            max_cycles,
            max_non_improving_iterations,
            max_non_improving_cycles,
            max_non_improving_time_secs,
            time_limit_secs,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LocalSearchConfig(operators={:?}, time_limit={:?}s)",
            self.operators, self.time_limit_secs
        )
    }
}
