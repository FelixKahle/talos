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
    /// # Errors
    ///
    /// Raises a `ValueError` if a vessel column does not hold `num_vessels` entries, `time_windows` does not hold
    /// `num_berths` entries, a time window starts after it ends, or the model refuses the input (processing times
    /// not one per pair of a vessel and a berth, a negative processing time, more vessels or berths than ids can
    /// name).
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
        if time_windows.len() != num_berths {
            return Err(PyValueError::new_err(format!(
                "time_windows length {} != num_berths {num_berths}",
                time_windows.len()
            )));
        }

        let tw = opening_intervals(time_windows)?;

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

/// The time windows of each berth as intervals, or a `ValueError` naming the first window that starts after it ends.
fn opening_intervals(
    time_windows: Vec<Vec<(i64, i64)>>,
) -> PyResult<Vec<Vec<ClosedOpenInterval<i64>>>> {
    time_windows
        .into_iter()
        .enumerate()
        .map(|(berth, windows)| {
            windows
                .into_iter()
                .map(|(lo, hi)| {
                    ClosedOpenInterval::try_from(lo..hi).map_err(|error| {
                        PyValueError::new_err(format!("time window of berth {berth}: {error}"))
                    })
                })
                .collect()
        })
        .collect()
}

impl PyModel {
    /// Returns a reference to the inner model (crate-internal).
    #[inline]
    pub fn inner(&self) -> &Model<i64> {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_refuses_a_window_that_starts_after_it_ends() {
        Python::initialize();
        let model = PyModel::new(
            1,
            1,
            vec![0],
            vec![10],
            vec![1],
            vec![Some(1)],
            vec![vec![(5, 3)]],
        );
        assert!(model.is_err());
    }

    #[test]
    fn test_new_lets_the_model_refuse_processing_times_not_one_per_pair() {
        Python::initialize();
        let error = PyModel::new(
            2,
            1,
            vec![0, 0],
            vec![10, 10],
            vec![1, 1],
            vec![Some(1)],
            vec![vec![]],
        )
        .err()
        .expect("one processing time for two pairs should be refused");
        Python::attach(|py| {
            assert_eq!(
                error.value(py).to_string(),
                "1 processing times for 2 vessels and 1 berths, which need one per pair"
            );
        });
    }

    #[test]
    fn test_opening_intervals_keeps_ordered_windows() {
        let intervals = opening_intervals(vec![vec![(0, 10), (20, 30)], vec![]]).unwrap();
        assert_eq!(
            intervals,
            [
                vec![
                    ClosedOpenInterval::new(0, 10),
                    ClosedOpenInterval::new(20, 30)
                ],
                vec![],
            ]
        );
    }

    #[test]
    fn test_opening_intervals_refuses_a_window_that_starts_after_it_ends() {
        Python::initialize();
        let error = opening_intervals(vec![vec![(0, 10)], vec![(5, 3)]]).unwrap_err();
        Python::attach(|py| {
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "time window of berth 1: start 5 lies after end 3"
            );
        });
    }
}
