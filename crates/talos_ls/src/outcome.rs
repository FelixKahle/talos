// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::exec::TerminationReason;
use crate::stats::LocalSearchStatistics;
use talos_core::SolverNumeric;
use talos_model::Solution;

/// The final outcome of a local search engine run.
///
/// This struct encapsulates the best solution found, the reason the search
/// was terminated, and the statistics collected during the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSearchOutcome<T>
where
    T: SolverNumeric,
{
    /// The best solution found during the search.
    solution: Solution<T>, // Local search starts from a solution, so we can guarantee that this is always populated.
    /// The exact reason the search stopped.
    termination_reason: TerminationReason,
    /// Performance and iteration statistics.
    stats: LocalSearchStatistics,
}

impl<T> LocalSearchOutcome<T>
where
    T: SolverNumeric,
{
    /// Creates a new local search outcome.
    #[inline]
    pub const fn new(
        solution: Solution<T>,
        termination_reason: TerminationReason,
        stats: LocalSearchStatistics,
    ) -> Self {
        Self {
            solution,
            termination_reason,
            stats,
        }
    }

    /// Returns a reference to the best solution found.
    #[inline]
    pub fn solution(&self) -> &Solution<T> {
        &self.solution
    }

    /// Returns the pure enum variant detailing why the search terminated.
    #[inline]
    pub const fn termination_reason(&self) -> TerminationReason {
        self.termination_reason
    }

    /// Returns a reference to the search statistics.
    #[inline]
    pub const fn stats(&self) -> &LocalSearchStatistics {
        &self.stats
    }

    /// Consumes the outcome, returning just the underlying solution.
    ///
    /// Useful when the caller does not care about stats or termination reasons.
    #[inline]
    pub fn into_solution(self) -> Solution<T> {
        self.solution
    }

    /// Consumes the outcome, returning its constituent parts.
    #[inline]
    pub fn into_inner(self) -> (Solution<T>, TerminationReason, LocalSearchStatistics) {
        (self.solution, self.termination_reason, self.stats)
    }
}

impl<T> From<LocalSearchOutcome<T>> for Solution<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn from(outcome: LocalSearchOutcome<T>) -> Self {
        outcome.into_solution()
    }
}
