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

//! Time-based termination for portfolio solver runs.
//!
//! Provides [`TimeLimitMonitor`], a lightweight [`PortfolioMonitor`]
//! that issues [`PortfolioCommand::Terminate`] once a configurable
//! wall-clock duration has elapsed. The timer starts when [`on_start`]
//! is called, ensuring each portfolio run is measured independently.

use crate::monitor::psmonitor::{PortfolioCommand, PortfolioMonitor};
use std::time::{Duration, Instant};
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

/// A wall-clock monitor that terminates a portfolio after a fixed duration.
///
/// The start time is recorded in [`on_start`](PortfolioMonitor::on_start)
/// and checked in [`portfolio_command`](PortfolioMonitor::portfolio_command).
/// When `elapsed >= time_limit`, a [`PortfolioCommand::Terminate`] is returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeLimitMonitor {
    start_time: Instant,
    time_limit: Duration,
}

impl TimeLimitMonitor {
    /// Creates a new `TimeLimitMonitor` with the specified time limit.
    #[inline]
    pub fn new(time_limit: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            time_limit,
        }
    }
}

impl<T> PortfolioMonitor<T> for TimeLimitMonitor
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "TimeLimitMonitor"
    }

    #[inline(always)]
    fn on_start(&mut self, _model: &Model<T>) {
        self.start_time = Instant::now();
    }

    #[inline(always)]
    fn on_end(&mut self, _best_solution: SolutionView<'_, T>) {}

    #[inline(always)]
    fn on_solver_started(&mut self, _solver_name: &str) {}

    #[inline(always)]
    fn on_solver_finished(&mut self, _solver_name: &str, _solution: SolutionView<'_, T>) {}

    #[inline(always)]
    fn on_best_solution_updated(
        &mut self,
        _solver_name: &str,
        _previous_best: SolutionView<'_, T>,
        _new_best: SolutionView<'_, T>,
    ) {
    }

    #[inline(always)]
    fn portfolio_command(&mut self) -> PortfolioCommand {
        if self.start_time.elapsed() >= self.time_limit {
            return PortfolioCommand::Terminate;
        }
        PortfolioCommand::Continue
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
    fn test_name() {
        let m = TimeLimitMonitor::new(Duration::from_secs(1));
        assert_eq!(PortfolioMonitor::<i64>::name(&m), "TimeLimitMonitor");
    }

    #[test]
    fn test_continues_before_time_limit() {
        let mut m = TimeLimitMonitor::new(Duration::from_secs(60));
        assert_eq!(
            PortfolioMonitor::<i64>::portfolio_command(&mut m),
            PortfolioCommand::Continue
        );
    }

    #[test]
    fn test_terminates_after_time_limit() {
        let mut m = TimeLimitMonitor::new(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(
            PortfolioMonitor::<i64>::portfolio_command(&mut m),
            PortfolioCommand::Terminate
        );
    }

    #[test]
    fn test_on_start_resets_timer() {
        let mut m = TimeLimitMonitor::new(Duration::from_millis(50));
        std::thread::sleep(Duration::from_millis(60));

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

        PortfolioMonitor::<i64>::on_start(&mut m, &model);

        // Timer just reset, should continue
        assert_eq!(
            PortfolioMonitor::<i64>::portfolio_command(&mut m),
            PortfolioCommand::Continue
        );
    }

    #[test]
    fn test_default_is_continue() {
        let sv = dummy_view();
        let mut m = TimeLimitMonitor::new(Duration::from_secs(60));
        PortfolioMonitor::<i64>::on_solver_started(&mut m, "test");
        PortfolioMonitor::<i64>::on_solver_finished(&mut m, "test", sv);
        PortfolioMonitor::<i64>::on_best_solution_updated(&mut m, "test", sv, sv);
        PortfolioMonitor::<i64>::on_end(&mut m, sv);
        // None of the above should change the command
        assert_eq!(
            PortfolioMonitor::<i64>::portfolio_command(&mut m),
            PortfolioCommand::Continue
        );
    }
}
