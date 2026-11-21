// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Solution-count–based termination for local search.
//!
//! This module provides `SolutionLimitMonitor`, a monitor that terminates a local
//! search once a configured number of candidate solutions have been generated and
//! evaluated. It integrates with the `LocalSearchMonitor` trait and issues a
//! `SearchCommand::Terminate` with `TerminationReason::SolutionLimitReached` when
//! the threshold is met.

use crate::{
    exec::{SearchCommand, TerminationReason},
    monitor::lsmonitor::LocalSearchMonitor,
    stats::LocalSearchStatistics,
};
use talos_core::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

/// A monitor that triggers termination once a specific number of candidate solutions
/// have been generated and evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SolutionLimitMonitor {
    limit: u64,
}

impl SolutionLimitMonitor {
    /// Create a new monitor that terminates once `total_solutions >= limit`.
    pub fn new(limit: u64) -> Self {
        Self { limit }
    }

    /// Returns the configured limit.
    pub fn limit(&self) -> u64 {
        self.limit
    }
}

impl<T> LocalSearchMonitor<T> for SolutionLimitMonitor
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "SolutionLimitMonitor"
    }

    #[inline(always)]
    fn on_start(&mut self, _model: &Model<T>, _initial_solution: SolutionView<'_, T>) {}

    #[inline(always)]
    fn on_end(&mut self, _best_solution: SolutionView<'_, T>, _statistics: &LocalSearchStatistics) {
    }

    #[inline(always)]
    fn on_iteration(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    #[inline(always)]
    fn on_candidate_generated(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _candidate_objective: T,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    #[inline(always)]
    fn on_solution_buffered(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: SolutionView<'_, T>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    #[inline(always)]
    fn on_candidate_accepted(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    #[inline(always)]
    fn on_buffered_solution_accepted(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    #[inline(always)]
    fn on_candidate_rejected(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _rejected_objective: T,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    #[inline(always)]
    fn on_neighborhood_exhausted(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    #[inline(always)]
    fn on_best_solution_updated(
        &mut self,
        _previous_best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _new_best_solution: SolutionView<'_, T>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    #[inline(always)]
    fn search_command(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        statistics: &LocalSearchStatistics,
    ) -> SearchCommand {
        if statistics.total_solutions >= self.limit {
            SearchCommand::Terminate(TerminationReason::SolutionLimitReached)
        } else {
            SearchCommand::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talos_model::index::BerthIndex;

    fn dummy_view() -> SolutionView<'static, i64> {
        const BERTHS: [BerthIndex; 1] = [BerthIndex::new(0)];
        const TIMES: [i64; 1] = [0];
        SolutionView::new(&BERTHS, &TIMES, 0)
    }

    #[test]
    fn test_name_returns_expected() {
        let m = SolutionLimitMonitor::new(10);
        assert_eq!(LocalSearchMonitor::<i64>::name(&m), "SolutionLimitMonitor");
    }

    #[test]
    fn test_limit_accessor() {
        let m = SolutionLimitMonitor::new(42);
        assert_eq!(m.limit(), 42);
    }

    #[test]
    fn test_continues_below_limit() {
        let mut m = SolutionLimitMonitor::new(5);
        let sv = dummy_view();
        let stats = LocalSearchStatistics {
            total_solutions: 4,
            ..Default::default()
        };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );
    }

    #[test]
    fn test_terminates_at_limit() {
        let mut m = SolutionLimitMonitor::new(5);
        let sv = dummy_view();
        let stats = LocalSearchStatistics {
            total_solutions: 5,
            ..Default::default()
        };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Terminate(TerminationReason::SolutionLimitReached)
        );
    }

    #[test]
    fn test_terminates_above_limit() {
        let mut m = SolutionLimitMonitor::new(5);
        let sv = dummy_view();
        let stats = LocalSearchStatistics {
            total_solutions: 100,
            ..Default::default()
        };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Terminate(TerminationReason::SolutionLimitReached)
        );
    }
}
