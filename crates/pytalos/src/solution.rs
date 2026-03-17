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

use pyo3::{exceptions::PyValueError, prelude::*};
use talos_model::{index::BerthIndex, solution::Solution};

/// A DBAP solution: berth assignments and start times for every vessel.
#[pyclass(name = "Solution", skip_from_py_object)]
#[derive(Clone)]
pub struct PySolution {
    inner: Solution<i64>,
}

#[pymethods]
impl PySolution {
    #[new]
    #[pyo3(signature = (berths, start_times, objective))]
    pub fn new(berths: Vec<usize>, start_times: Vec<i64>, objective: i64) -> PyResult<Self> {
        if berths.len() != start_times.len() {
            return Err(PyValueError::new_err(format!(
                "berths length {} != start_times length {}",
                berths.len(),
                start_times.len()
            )));
        }
        let berth_indices: Vec<BerthIndex> = berths.iter().map(|&b| BerthIndex::new(b)).collect();
        Ok(Self {
            inner: Solution::new(berth_indices, start_times, objective),
        })
    }

    #[getter]
    pub fn objective(&self) -> i64 {
        self.inner.objective_value()
    }

    #[getter]
    pub fn berths(&self) -> Vec<usize> {
        self.inner.berths().iter().map(|b| b.get()).collect()
    }

    #[getter]
    pub fn start_times(&self) -> Vec<i64> {
        self.inner.start_times().to_vec()
    }

    #[getter]
    pub fn num_vessels(&self) -> usize {
        self.inner.num_vessels()
    }

    fn __repr__(&self) -> String {
        format!(
            "Solution(num_vessels={}, objective={})",
            self.inner.num_vessels(),
            self.inner.objective_value()
        )
    }
}

impl PySolution {
    /// Returns a reference to the inner solution (crate-internal).
    #[inline]
    pub(crate) fn inner(&self) -> &Solution<i64> {
        &self.inner
    }

    /// Constructs a PySolution from an owned Solution (crate-internal).
    #[inline]
    pub(crate) fn from_inner(solution: Solution<i64>) -> Self {
        Self { inner: solution }
    }
}
