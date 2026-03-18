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

use crate::solution::PySolution;
use pyo3::prelude::*;
use talos_ls::exec::TerminationReason;

#[pyclass(name = "TerminationReason", eq, eq_int, skip_from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PyTerminationReason {
    TimeLimitReached = 0,
    SolutionLimitReached = 1,
    IterationLimitReached = 2,
    CycleLimitReached = 3,
    MaxNonImprovingIterations = 4,
    MaxNonImprovingCycles = 5,
    MaxNonImprovingTime = 6,
    TargetObjectiveReached = 7,
    NeighborhoodExhausted = 8,
    Interrupted = 9,
    Aborted = 10,
}

impl From<TerminationReason> for PyTerminationReason {
    fn from(reason: TerminationReason) -> Self {
        match reason {
            TerminationReason::TimeLimitReached => PyTerminationReason::TimeLimitReached,
            TerminationReason::SolutionLimitReached => PyTerminationReason::SolutionLimitReached,
            TerminationReason::IterationLimitReached => PyTerminationReason::IterationLimitReached,
            TerminationReason::CycleLimitReached => PyTerminationReason::CycleLimitReached,
            TerminationReason::MaxNonImprovingIterations => {
                PyTerminationReason::MaxNonImprovingIterations
            }
            TerminationReason::MaxNonImprovingCycles => PyTerminationReason::MaxNonImprovingCycles,
            TerminationReason::MaxNonImprovingTime => PyTerminationReason::MaxNonImprovingTime,
            TerminationReason::TargetObjectiveReached => {
                PyTerminationReason::TargetObjectiveReached
            }
            TerminationReason::NeighborhoodExhausted => PyTerminationReason::NeighborhoodExhausted,
            TerminationReason::Interrupted => PyTerminationReason::Interrupted,
            TerminationReason::Aborted => PyTerminationReason::Aborted,
        }
    }
}

#[pyclass(name = "SearchResult", skip_from_py_object)]
#[derive(Clone)]
pub struct PySearchResult {
    #[pyo3(get)]
    pub(crate) solution: PySolution,
    #[pyo3(get)]
    pub(crate) termination_reason: PyTerminationReason,
    #[pyo3(get)]
    pub(crate) iterations: u64,
    #[pyo3(get)]
    pub(crate) accepted_solutions: u64,
    #[pyo3(get)]
    pub(crate) total_solutions: u64,
    #[pyo3(get)]
    pub(crate) infeasible_moves: u64,
    #[pyo3(get)]
    pub(crate) cycles: u64,
    #[pyo3(get)]
    pub(crate) time_total_secs: f64,
}

#[pymethods]
impl PySearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(objective={}, iterations={}, time={:.3}s, reason={:?})",
            self.solution.objective(),
            self.iterations,
            self.time_total_secs,
            self.termination_reason
        )
    }
}

#[inline(always)]
pub fn outcome_to_py(outcome: talos_ls::outcome::LocalSearchOutcome<i64>) -> PySearchResult {
    let (solution, reason, stats) = outcome.into_inner();

    PySearchResult {
        solution: PySolution::from_inner(solution),
        termination_reason: PyTerminationReason::from(reason),
        iterations: stats.iterations,
        accepted_solutions: stats.accepted_solutions,
        total_solutions: stats.total_solutions,
        infeasible_moves: stats.infeasible_moves,
        cycles: stats.cycles,
        time_total_secs: stats.time_total.as_secs_f64(),
    }
}
