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

//! Time-based termination for local search.
//!
//! This module provides `TimeLimitMonitor`, a lightweight monitor that stops a local
//! search after a configurable wall-clock duration. It integrates with the
//! `LocalSearchMonitor` trait and issues a `SearchCommand::Terminate` when the
//! elapsed time exceeds the configured limit.
//!
//! To minimize overhead, clock checks are throttled using a step mask. The mask is
//! applied to the iteration counter and only when the masked value is zero the clock
//! is queried. A mask of `0x1FFF` yields a check roughly every 8192 iterations, which
//! offers a balance between responsiveness and performance. The mask can be customized
//! via `with_mask`, allowing tighter or looser checking based on the problem scale.
//!
//! The monitor resets its start time on `on_start`, ensuring each search run is measured
//! independently. No state is mutated during `search_command` beyond timing checks, and
//! the termination command uses the zero-cost `TerminationReason::TimeLimitReached` variant.

use crate::{
    exec::{SearchCommand, TerminationReason},
    monitor::lsmonitor::LocalSearchMonitor,
    stats::LocalSearchStatistics,
};
use std::time::{Duration, Instant};
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

/// A lightweight wall-clock monitor that terminates a local search after a fixed duration.
///
/// This monitor records the start time at `on_start` and periodically checks the elapsed time
/// during `search_command`. To reduce overhead, time checks are throttled using `clock_check_mask`,
/// which masks the iteration counter and only queries the clock when the masked value is zero.
/// When `elapsed >= time_limit`, it issues a termination command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeLimitMonitor {
    start_time: Instant,
    time_limit: Duration,
    clock_check_mask: u64,
}

impl TimeLimitMonitor {
    /// Default mask for clock checks to avoid excessive time checks.
    /// This mask checks the clock every 8192 steps (0x1FFF).
    const DEFAULT_STEP_CLOCK_CHECK_MASK: u64 = 0x1FFF;

    /// Creates a new `TimeLimitMonitor` with the specified time limit.
    #[inline]
    pub fn new(time_limit: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            time_limit,
            clock_check_mask: Self::DEFAULT_STEP_CLOCK_CHECK_MASK,
        }
    }

    /// Creates a new `TimeLimitMonitor` with a custom step clock check mask.
    /// Lower mask values check more often; higher values check less often.
    #[inline]
    pub fn with_mask(time_limit: Duration, clock_check_mask: u64) -> Self {
        Self {
            start_time: Instant::now(),
            time_limit,
            clock_check_mask,
        }
    }
}

impl<T> LocalSearchMonitor<T> for TimeLimitMonitor
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "TimeLimitMonitor"
    }

    #[inline(always)]
    fn on_start(&mut self, _model: &Model<T>, _initial_solution: SolutionView<'_, T>) {
        self.start_time = Instant::now();
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
        if (statistics.iterations & self.clock_check_mask) == 0
            && self.start_time.elapsed() >= self.time_limit
        {
            return SearchCommand::Terminate(TerminationReason::TimeLimitReached);
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

    #[test]
    fn test_name_returns_expected() {
        let m = TimeLimitMonitor::new(Duration::from_secs(1));
        assert_eq!(LocalSearchMonitor::<i64>::name(&m), "TimeLimitMonitor");
    }

    #[test]
    fn test_new_sets_default_mask() {
        let m = TimeLimitMonitor::new(Duration::from_secs(5));
        assert_eq!(m.time_limit, Duration::from_secs(5));
        assert_eq!(m.clock_check_mask, 0x1FFF);
    }

    #[test]
    fn test_with_mask_sets_custom_mask() {
        let m = TimeLimitMonitor::with_mask(Duration::from_secs(3), 0xFF);
        assert_eq!(m.time_limit, Duration::from_secs(3));
        assert_eq!(m.clock_check_mask, 0xFF);
    }

    #[test]
    fn test_continues_before_time_limit() {
        // mask = 0 so every iteration checks the clock
        let mut m = TimeLimitMonitor::with_mask(Duration::from_secs(60), 0);
        let sv = dummy_view();
        let stats = LocalSearchStatistics::default();
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );
    }

    #[test]
    fn test_terminates_after_time_limit() {
        let mut m = TimeLimitMonitor::with_mask(Duration::from_millis(1), 0);
        let sv = dummy_view();
        let stats = LocalSearchStatistics::default();
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Terminate(TerminationReason::TimeLimitReached)
        );
    }

    #[test]
    fn test_mask_skips_clock_check() {
        // mask = 0x1 means check only when iterations is even
        let mut m = TimeLimitMonitor::with_mask(Duration::from_millis(1), 0x1);
        let sv = dummy_view();
        std::thread::sleep(Duration::from_millis(5));

        let stats = LocalSearchStatistics {
            iterations: 1,
            ..Default::default()
        }; // odd → masked value != 0, clock not checked
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );

        let stats = LocalSearchStatistics {
            iterations: 2,
            ..Default::default()
        }; // even → masked value == 0, clock checked
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Terminate(TerminationReason::TimeLimitReached)
        );
    }

    #[test]
    fn test_on_start_resets_timer() {
        let mut m = TimeLimitMonitor::with_mask(Duration::from_millis(50), 0);
        let sv = dummy_view();
        std::thread::sleep(Duration::from_millis(60));

        // Build a minimal model for on_start
        use talos_model::model::Model;
        let model = Model::new(
            1,
            1,
            vec![0i64],
            vec![100],
            vec![1],
            vec![talos_model::model::ProcessingTime::from_raw(10)],
            vec![vec![]],
        );

        m.on_start(&model, sv);

        // Timer just reset, should continue
        let stats = LocalSearchStatistics::default();
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );
    }
}
