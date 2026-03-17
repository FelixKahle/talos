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
use talos_core::math::interval::ClosedOpenInterval;
use talos_model::model::{Model, ProcessingTime};

/// The DBAP problem model.
#[pyclass(name = "Model", skip_from_py_object)]
#[derive(Clone)]
pub struct PyModel {
    inner: Model<i64>,
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (num_vessels, num_berths, arrivals, deadlines, weights, processing_times, time_windows))]
    pub fn new(
        num_vessels: usize,
        num_berths: usize,
        arrivals: Vec<i64>,
        deadlines: Vec<i64>,
        weights: Vec<i64>,
        processing_times: Vec<Option<i64>>,
        time_windows: Vec<Vec<(i64, i64)>>,
    ) -> PyResult<Self> {
        if arrivals.len() != num_vessels {
            return Err(PyValueError::new_err(format!(
                "arrivals length {} != num_vessels {num_vessels}",
                arrivals.len()
            )));
        }
        if deadlines.len() != num_vessels {
            return Err(PyValueError::new_err(format!(
                "deadlines length {} != num_vessels {num_vessels}",
                deadlines.len()
            )));
        }
        if weights.len() != num_vessels {
            return Err(PyValueError::new_err(format!(
                "weights length {} != num_vessels {num_vessels}",
                weights.len()
            )));
        }
        if processing_times.len() != num_vessels * num_berths {
            return Err(PyValueError::new_err(format!(
                "processing_times length {} != {}",
                processing_times.len(),
                num_vessels * num_berths
            )));
        }
        if time_windows.len() != num_berths {
            return Err(PyValueError::new_err(format!(
                "time_windows length {} != num_berths {num_berths}",
                time_windows.len()
            )));
        }

        let pts: Vec<ProcessingTime<i64>> = processing_times
            .into_iter()
            .map(|opt| match opt {
                Some(v) => ProcessingTime::some(v),
                None => ProcessingTime::none(),
            })
            .collect();

        let tw: Vec<Vec<ClosedOpenInterval<i64>>> = time_windows
            .into_iter()
            .map(|intervals| {
                intervals
                    .into_iter()
                    .map(|(lo, hi)| ClosedOpenInterval::new(lo, hi))
                    .collect()
            })
            .collect();

        Ok(Self {
            inner: Model::new(
                num_vessels,
                num_berths,
                arrivals,
                deadlines,
                weights,
                pts,
                tw,
            ),
        })
    }

    #[getter]
    pub fn num_vessels(&self) -> usize {
        self.inner.num_vessels()
    }

    #[getter]
    pub fn num_berths(&self) -> usize {
        self.inner.num_berths()
    }
}

impl PyModel {
    /// Returns a reference to the inner model (crate-internal).
    #[inline]
    pub fn inner(&self) -> &Model<i64> {
        &self.inner
    }
}
