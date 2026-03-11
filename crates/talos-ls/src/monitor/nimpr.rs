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

//! Non-improving termination for local search.
//!
//! This module provides `NoImprovementMonitor`, a monitor that terminates a local
//! search after a configured stretch without improvement to the global best solution.
//! Three independent patience modes can be active simultaneously:
//!
//! - **Iterations**: terminate after N individual neighbors evaluated without improvement.
//! - **Cycles**: terminate after N full neighborhood traversals without improvement.
//! - **Duration**: terminate after a wall-clock duration without improvement.
//!
//! Any mode that fires first wins. All three are optional; at least one must be set.
//! The counter / timestamp resets every time `on_best_solution_updated` fires.

use crate::{
    exec::{SearchCommand, TerminationReason},
    monitor::lsmonitor::LocalSearchMonitor,
    stats::LocalSearchStatistics,
};
use std::time::{Duration, Instant};
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

/// A monitor that terminates the search after a configurable stretch without
/// improvement to the global best solution.
///
/// Supports up to three simultaneous patience modes (iterations, cycles, duration).
/// Use the builder methods to configure which modes are active.
#[derive(Debug, Clone)]
pub struct NoImprovementMonitor {
    /// Max non-improving iterations (None = disabled).
    iteration_patience: Option<u64>,
    /// Max non-improving cycles (None = disabled).
    cycle_patience: Option<u64>,
    /// Max wall-clock duration without improvement (None = disabled).
    duration_patience: Option<Duration>,

    last_improved_iteration: u64,
    last_improved_cycle: u64,
    last_improved_time: Instant,
}

impl NoImprovementMonitor {
    /// Creates a monitor with iteration-based patience only.
    pub fn with_iteration_patience(patience: u64) -> Self {
        Self {
            iteration_patience: Some(patience),
            cycle_patience: None,
            duration_patience: None,
            last_improved_iteration: 0,
            last_improved_cycle: 0,
            last_improved_time: Instant::now(),
        }
    }

    /// Creates a monitor with cycle-based patience only.
    pub fn with_cycle_patience(patience: u64) -> Self {
        Self {
            iteration_patience: None,
            cycle_patience: Some(patience),
            duration_patience: None,
            last_improved_iteration: 0,
            last_improved_cycle: 0,
            last_improved_time: Instant::now(),
        }
    }

    /// Creates a monitor with duration-based patience only.
    pub fn with_duration_patience(patience: Duration) -> Self {
        Self {
            iteration_patience: None,
            cycle_patience: None,
            duration_patience: Some(patience),
            last_improved_iteration: 0,
            last_improved_cycle: 0,
            last_improved_time: Instant::now(),
        }
    }

    /// Adds iteration-based patience (can be combined with other modes).
    pub fn and_iteration_patience(mut self, patience: u64) -> Self {
        self.iteration_patience = Some(patience);
        self
    }

    /// Adds cycle-based patience (can be combined with other modes).
    pub fn and_cycle_patience(mut self, patience: u64) -> Self {
        self.cycle_patience = Some(patience);
        self
    }

    /// Adds duration-based patience (can be combined with other modes).
    pub fn and_duration_patience(mut self, patience: Duration) -> Self {
        self.duration_patience = Some(patience);
        self
    }
}

impl<T> LocalSearchMonitor<T> for NoImprovementMonitor
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "NoImprovementMonitor"
    }

    #[inline(always)]
    fn on_start(&mut self, _model: &Model<T>, _initial_solution: SolutionView<'_, T>) {
        self.last_improved_iteration = 0;
        self.last_improved_cycle = 0;
        self.last_improved_time = Instant::now();
    }

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
        statistics: &LocalSearchStatistics,
    ) {
        self.last_improved_iteration = statistics.iterations;
        self.last_improved_cycle = statistics.cycles;
        self.last_improved_time = Instant::now();
    }

    #[inline(always)]
    fn search_command(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        statistics: &LocalSearchStatistics,
    ) -> SearchCommand {
        if let Some(patience) = self.iteration_patience
            && statistics.iterations - self.last_improved_iteration >= patience
        {
            return SearchCommand::Terminate(TerminationReason::MaxNonImprovingIterations);
        }
        if let Some(patience) = self.cycle_patience
            && statistics.cycles - self.last_improved_cycle >= patience
        {
            return SearchCommand::Terminate(TerminationReason::MaxNonImprovingIterations);
        }
        if let Some(patience) = self.duration_patience
            && self.last_improved_time.elapsed() >= patience
        {
            return SearchCommand::Terminate(TerminationReason::MaxNonImprovingIterations);
        }
        SearchCommand::Continue
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

    const TERMINATE: SearchCommand =
        SearchCommand::Terminate(TerminationReason::MaxNonImprovingIterations);

    // --- Constructor / builder tests ---

    #[test]
    fn test_name_returns_expected() {
        let m = NoImprovementMonitor::with_iteration_patience(10);
        assert_eq!(LocalSearchMonitor::<i64>::name(&m), "NoImprovementMonitor");
    }

    #[test]
    fn test_with_iteration_patience_sets_only_iterations() {
        let m = NoImprovementMonitor::with_iteration_patience(50);
        assert_eq!(m.iteration_patience, Some(50));
        assert!(m.cycle_patience.is_none());
        assert!(m.duration_patience.is_none());
    }

    #[test]
    fn test_with_cycle_patience_sets_only_cycles() {
        let m = NoImprovementMonitor::with_cycle_patience(10);
        assert!(m.iteration_patience.is_none());
        assert_eq!(m.cycle_patience, Some(10));
        assert!(m.duration_patience.is_none());
    }

    #[test]
    fn test_with_duration_patience_sets_only_duration() {
        let d = Duration::from_secs(5);
        let m = NoImprovementMonitor::with_duration_patience(d);
        assert!(m.iteration_patience.is_none());
        assert!(m.cycle_patience.is_none());
        assert_eq!(m.duration_patience, Some(d));
    }

    #[test]
    fn test_builder_combinators() {
        let d = Duration::from_secs(1);
        let m = NoImprovementMonitor::with_iteration_patience(100)
            .and_cycle_patience(5)
            .and_duration_patience(d);
        assert_eq!(m.iteration_patience, Some(100));
        assert_eq!(m.cycle_patience, Some(5));
        assert_eq!(m.duration_patience, Some(d));
    }

    // --- Iteration patience ---

    #[test]
    fn test_iteration_patience_continues_within() {
        let mut m = NoImprovementMonitor::with_iteration_patience(10);
        let sv = dummy_view();
        let stats = LocalSearchStatistics { iterations: 9, ..Default::default() };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );
    }

    #[test]
    fn test_iteration_patience_terminates_at_limit() {
        let mut m = NoImprovementMonitor::with_iteration_patience(10);
        let sv = dummy_view();
        let stats = LocalSearchStatistics { iterations: 10, ..Default::default() };
        assert_eq!(m.search_command(sv, sv, None, &stats), TERMINATE);
    }

    // --- Cycle patience ---

    #[test]
    fn test_cycle_patience_continues_within() {
        let mut m = NoImprovementMonitor::with_cycle_patience(3);
        let sv = dummy_view();
        let stats = LocalSearchStatistics { cycles: 2, ..Default::default() };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );
    }

    #[test]
    fn test_cycle_patience_terminates_at_limit() {
        let mut m = NoImprovementMonitor::with_cycle_patience(3);
        let sv = dummy_view();
        let stats = LocalSearchStatistics { cycles: 3, ..Default::default() };
        assert_eq!(m.search_command(sv, sv, None, &stats), TERMINATE);
    }

    // --- Duration patience ---

    #[test]
    fn test_duration_patience_continues_within() {
        let mut m = NoImprovementMonitor::with_duration_patience(Duration::from_secs(60));
        let sv = dummy_view();
        let stats = LocalSearchStatistics::default();
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );
    }

    #[test]
    fn test_duration_patience_terminates_expired() {
        let mut m = NoImprovementMonitor::with_duration_patience(Duration::from_millis(1));
        let sv = dummy_view();
        let stats = LocalSearchStatistics::default();
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(m.search_command(sv, sv, None, &stats), TERMINATE);
    }

    // --- Reset on improvement ---

    #[test]
    fn test_on_best_solution_updated_resets_iteration_counter() {
        let mut m = NoImprovementMonitor::with_iteration_patience(10);
        let sv = dummy_view();
        let stats = LocalSearchStatistics { iterations: 15, ..Default::default() };
        // Simulate improvement at iteration 15
        LocalSearchMonitor::<i64>::on_best_solution_updated(&mut m, sv, sv, None, sv, &stats);

        // 5 more iterations → total 20, non-improving = 5, patience = 10 → continue
        let stats = LocalSearchStatistics { iterations: 20, ..Default::default() };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );

        // 10 more → total 25, non-improving = 10 → terminate
        let stats = LocalSearchStatistics { iterations: 25, ..Default::default() };
        assert_eq!(m.search_command(sv, sv, None, &stats), TERMINATE);
    }

    #[test]
    fn test_on_best_solution_updated_resets_cycle_counter() {
        let mut m = NoImprovementMonitor::with_cycle_patience(5);
        let sv = dummy_view();
        let stats = LocalSearchStatistics { cycles: 10, ..Default::default() };
        LocalSearchMonitor::<i64>::on_best_solution_updated(&mut m, sv, sv, None, sv, &stats);

        let stats = LocalSearchStatistics { cycles: 14, ..Default::default() };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );

        let stats = LocalSearchStatistics { cycles: 15, ..Default::default() };
        assert_eq!(m.search_command(sv, sv, None, &stats), TERMINATE);
    }

    #[test]
    fn test_on_start_resets_all() {
        use talos_model::model::{Model, ProcessingTime};

        let mut m = NoImprovementMonitor::with_iteration_patience(5).and_cycle_patience(3);
        let sv = dummy_view();
        // Exhaust iteration patience
        let stats = LocalSearchStatistics { iterations: 100, ..Default::default() };
        assert_eq!(m.search_command(sv, sv, None, &stats), TERMINATE);

        // Reset via on_start
        let model = Model::new(
            1,
            1,
            vec![0i64],
            vec![100],
            vec![1],
            vec![ProcessingTime::from_raw(10)],
            vec![vec![]],
        );
        LocalSearchMonitor::<i64>::on_start(&mut m, &model, sv);

        // Fresh stats → should continue
        let stats = LocalSearchStatistics::default();
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );
    }

    // --- Combined modes: first to fire wins ---

    #[test]
    fn test_combined_iteration_fires_first() {
        let mut m = NoImprovementMonitor::with_iteration_patience(5).and_cycle_patience(100);
        let sv = dummy_view();
        let stats = LocalSearchStatistics { iterations: 5, ..Default::default() };
        assert_eq!(m.search_command(sv, sv, None, &stats), TERMINATE);
    }

    #[test]
    fn test_combined_cycle_fires_first() {
        let mut m = NoImprovementMonitor::with_iteration_patience(100).and_cycle_patience(3);
        let sv = dummy_view();
        let stats = LocalSearchStatistics { iterations: 10, cycles: 3, ..Default::default() };
        assert_eq!(m.search_command(sv, sv, None, &stats), TERMINATE);
    }
}
