// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cycle-count–based termination for local search.
//!
//! This module provides `CycleLimitMonitor`, a monitor that terminates a local
//! search once a configured number of cycles (full neighborhood traversals) have
//! been performed. It integrates with the `LocalSearchMonitor` trait and issues a
//! `SearchCommand::Terminate` with `TerminationReason::CycleLimitReached` when
//! the threshold is met.

use crate::{
    monitor::lsmonitor::LocalSearchMonitor,
    statistics::LocalSearchStatistics,
    termination::{SearchCommand, TerminationReason},
};
use talos_core::SolverNumeric;
use talos_model::{Model, SolutionView};

/// A monitor that triggers termination once a specific number of cycles
/// (full neighborhood traversals) have been performed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CycleLimitMonitor {
    limit: u64,
}

impl CycleLimitMonitor {
    /// Creates a new monitor that terminates once `cycles >= limit`.
    #[inline(always)]
    pub fn new(limit: u64) -> Self {
        Self { limit }
    }

    /// Returns the configured limit.
    #[inline(always)]
    pub fn limit(&self) -> u64 {
        self.limit
    }
}

impl<T> LocalSearchMonitor<T> for CycleLimitMonitor
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "CycleLimitMonitor"
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
        if statistics.cycles >= self.limit {
            SearchCommand::Terminate(TerminationReason::CycleLimitReached)
        } else {
            SearchCommand::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talos_model::BerthIndex;

    fn dummy_view() -> SolutionView<'static, i64> {
        const BERTHS: [BerthIndex; 1] = [BerthIndex::new(0)];
        const TIMES: [i64; 1] = [0];
        SolutionView::new(&BERTHS, &TIMES, 0)
    }

    #[test]
    fn test_name_returns_expected() {
        let m = CycleLimitMonitor::new(10);
        assert_eq!(LocalSearchMonitor::<i64>::name(&m), "CycleLimitMonitor");
    }

    #[test]
    fn test_limit_accessor() {
        let m = CycleLimitMonitor::new(7);
        assert_eq!(m.limit(), 7);
    }

    #[test]
    fn test_continues_below_limit() {
        let mut m = CycleLimitMonitor::new(5);
        let sv = dummy_view();
        let stats = LocalSearchStatistics {
            cycles: 4,
            ..Default::default()
        };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );
    }

    #[test]
    fn test_terminates_at_limit() {
        let mut m = CycleLimitMonitor::new(5);
        let sv = dummy_view();
        let stats = LocalSearchStatistics {
            cycles: 5,
            ..Default::default()
        };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Terminate(TerminationReason::CycleLimitReached)
        );
    }

    #[test]
    fn test_terminates_above_limit() {
        let mut m = CycleLimitMonitor::new(5);
        let sv = dummy_view();
        let stats = LocalSearchStatistics {
            cycles: 50,
            ..Default::default()
        };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Terminate(TerminationReason::CycleLimitReached)
        );
    }
}
