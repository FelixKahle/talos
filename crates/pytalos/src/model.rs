// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use pyo3::{exceptions::PyValueError, prelude::*};
use talos_core::ClosedOpenInterval;
use talos_model::Model;

/// The DBAP problem model.
#[pyclass(name = "Model", skip_from_py_object)]
#[derive(Clone)]
pub struct PyModel {
    inner: Model<i64>,
}

#[pymethods]
impl PyModel {
    /// Creates a new PyModel from the given parameters.
    ///
    /// # Panics
    ///
    /// Panics if any of the time windows are invalid (lo > hi). Lengths that do not match
    /// `num_vessels` and `num_berths`, and negative processing times, raise a `ValueError`.
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

        let tw: Vec<Vec<ClosedOpenInterval<i64>>> = time_windows
            .into_iter()
            .map(|intervals| {
                intervals
                    .into_iter()
                    .map(|(lo, hi)| ClosedOpenInterval::new(lo, hi))
                    .collect()
            })
            .collect();

        let inner = Model::try_new(arrivals, deadlines, weights, processing_times, tw)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Returns the number of vessels in the model.
    #[getter]
    pub fn num_vessels(&self) -> usize {
        self.inner.num_vessels()
    }

    /// Returns the number of berths in the model.
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
