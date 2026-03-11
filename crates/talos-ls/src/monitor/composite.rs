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

//! Composite monitor that fans out to multiple `LocalSearchMonitor` implementations.
//!
//! `CompositeLocalSearchMonitor` holds a `Vec` of boxed monitors and forwards every
//! callback to each of them in order. For `search_command`, the first monitor that
//! returns something other than `SearchCommand::Continue` wins — remaining monitors
//! are skipped.

use crate::{
    exec::SearchCommand, monitor::lsmonitor::LocalSearchMonitor, stats::LocalSearchStatistics,
};
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

/// A composite monitor that delegates every callback to a list of inner monitors.
///
/// Each lifecycle event is forwarded to every registered monitor in insertion order.
/// For `search_command`, the first monitor returning anything other than
/// `SearchCommand::Continue` short-circuits and its command is returned immediately.
#[derive(Default)]
pub struct CompositeLocalSearchMonitor<'a, T>
where
    T: SolverNumeric,
{
    monitors: Vec<Box<dyn LocalSearchMonitor<T> + 'a>>,
}

impl<'a, T> CompositeLocalSearchMonitor<'a, T>
where
    T: SolverNumeric,
{
    /// Creates a new composite monitor with no inner monitors.
    #[inline]
    pub fn new() -> Self {
        Self {
            monitors: Vec::new(),
        }
    }

    /// Creates a new composite monitor pre-allocating space for `capacity` monitors.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            monitors: Vec::with_capacity(capacity),
        }
    }

    /// Adds a monitor, boxing it internally.
    #[inline]
    pub fn add_monitor<M>(&mut self, monitor: M)
    where
        M: LocalSearchMonitor<T> + 'a,
    {
        self.monitors.push(Box::new(monitor));
    }

    /// Adds a previously boxed monitor.
    #[inline]
    pub fn add_boxed_monitor(&mut self, monitor: Box<dyn LocalSearchMonitor<T> + 'a>) {
        self.monitors.push(monitor);
    }

    /// Adds all monitors from an iterator of boxed monitors.
    #[inline]
    pub fn add_boxed_monitors<I>(&mut self, monitors: I)
    where
        I: IntoIterator<Item = Box<dyn LocalSearchMonitor<T> + 'a>>,
    {
        self.monitors.extend(monitors);
    }

    /// Returns a slice of all registered monitors.
    #[inline]
    pub fn monitors(&self) -> &[Box<dyn LocalSearchMonitor<T> + 'a>] {
        &self.monitors
    }
}

impl<'a, T> LocalSearchMonitor<T> for CompositeLocalSearchMonitor<'a, T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        "CompositeLocalSearchMonitor"
    }

    fn on_start(&mut self, model: &Model<T>, initial_solution: SolutionView<'_, T>) {
        for monitor in &mut self.monitors {
            monitor.on_start(model, initial_solution);
        }
    }

    fn on_end(&mut self, best_solution: SolutionView<'_, T>, statistics: &LocalSearchStatistics) {
        for monitor in &mut self.monitors {
            monitor.on_end(best_solution, statistics);
        }
    }

    fn on_iteration(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        statistics: &LocalSearchStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_iteration(
                best_solution,
                accepted_solution,
                buffered_solution,
                statistics,
            );
        }
    }

    fn on_candidate_generated(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        candidate_objective: T,
        statistics: &LocalSearchStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_candidate_generated(
                best_solution,
                accepted_solution,
                buffered_solution,
                candidate_objective,
                statistics,
            );
        }
    }

    fn on_solution_buffered(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: SolutionView<'_, T>,
        statistics: &LocalSearchStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_solution_buffered(
                best_solution,
                accepted_solution,
                buffered_solution,
                statistics,
            );
        }
    }

    fn on_candidate_accepted(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        statistics: &LocalSearchStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_candidate_accepted(
                best_solution,
                accepted_solution,
                buffered_solution,
                statistics,
            );
        }
    }

    fn on_buffered_solution_accepted(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        statistics: &LocalSearchStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_buffered_solution_accepted(best_solution, accepted_solution, statistics);
        }
    }

    fn on_candidate_rejected(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        rejected_objective: T,
        statistics: &LocalSearchStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_candidate_rejected(
                best_solution,
                accepted_solution,
                buffered_solution,
                rejected_objective,
                statistics,
            );
        }
    }

    fn on_neighborhood_exhausted(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        statistics: &LocalSearchStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_neighborhood_exhausted(
                best_solution,
                accepted_solution,
                buffered_solution,
                statistics,
            );
        }
    }

    fn on_best_solution_updated(
        &mut self,
        previous_best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        new_best_solution: SolutionView<'_, T>,
        statistics: &LocalSearchStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_best_solution_updated(
                previous_best_solution,
                accepted_solution,
                buffered_solution,
                new_best_solution,
                statistics,
            );
        }
    }

    fn search_command(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        statistics: &LocalSearchStatistics,
    ) -> SearchCommand {
        for monitor in &mut self.monitors {
            let command = monitor.search_command(
                best_solution,
                accepted_solution,
                buffered_solution,
                statistics,
            );
            if command != SearchCommand::Continue {
                return command;
            }
        }
        SearchCommand::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::TerminationReason;
    use crate::monitor::iteration::IterationLimitMonitor;
    use crate::monitor::solution::SolutionLimitMonitor;
    use talos_model::index::BerthIndex;

    fn dummy_view() -> SolutionView<'static, i64> {
        const BERTHS: [BerthIndex; 1] = [BerthIndex::new(0)];
        const TIMES: [i64; 1] = [0];
        SolutionView::new(&BERTHS, &TIMES, 0)
    }

    #[test]
    fn test_name_returns_expected() {
        let m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::new();
        assert_eq!(m.name(), "CompositeLocalSearchMonitor");
    }

    #[test]
    fn test_new_creates_empty() {
        let m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::new();
        assert_eq!(m.monitors().len(), 0);
    }

    #[test]
    fn test_with_capacity_creates_empty() {
        let m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::with_capacity(5);
        assert_eq!(m.monitors().len(), 0);
    }

    #[test]
    fn test_add_monitor_increases_count() {
        let mut m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::new();
        m.add_monitor(IterationLimitMonitor::new(100));
        m.add_monitor(SolutionLimitMonitor::new(50));
        assert_eq!(m.monitors().len(), 2);
    }

    #[test]
    fn test_add_boxed_monitor_increases_count() {
        let mut m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::new();
        m.add_boxed_monitor(Box::new(IterationLimitMonitor::new(100)));
        assert_eq!(m.monitors().len(), 1);
    }

    #[test]
    fn test_add_boxed_monitors_from_iterator() {
        let mut m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::new();
        let monitors: Vec<Box<dyn LocalSearchMonitor<i64>>> = vec![
            Box::new(IterationLimitMonitor::new(10)),
            Box::new(SolutionLimitMonitor::new(20)),
        ];
        m.add_boxed_monitors(monitors);
        assert_eq!(m.monitors().len(), 2);
    }

    #[test]
    fn test_search_command_continues_when_all_continue() {
        let mut m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::new();
        m.add_monitor(IterationLimitMonitor::new(1000));
        m.add_monitor(SolutionLimitMonitor::new(1000));

        let sv = dummy_view();
        let stats = LocalSearchStatistics::default();
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );
    }

    #[test]
    fn test_search_command_continues_when_empty() {
        let mut m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::new();
        let sv = dummy_view();
        let stats = LocalSearchStatistics::default();
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Continue
        );
    }

    #[test]
    fn test_search_command_terminates_on_first_hit() {
        let mut m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::new();
        // iteration limit fires first (limit=5), solution limit does not (limit=1000)
        m.add_monitor(IterationLimitMonitor::new(5));
        m.add_monitor(SolutionLimitMonitor::new(1000));

        let sv = dummy_view();
        let stats = LocalSearchStatistics { iterations: 10, ..Default::default() };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Terminate(TerminationReason::IterationLimitReached)
        );
    }

    #[test]
    fn test_search_command_second_monitor_fires() {
        let mut m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::new();
        m.add_monitor(IterationLimitMonitor::new(1000)); // won't fire
        m.add_monitor(SolutionLimitMonitor::new(5)); // will fire

        let sv = dummy_view();
        let stats = LocalSearchStatistics { total_solutions: 10, ..Default::default() };
        assert_eq!(
            m.search_command(sv, sv, None, &stats),
            SearchCommand::Terminate(TerminationReason::SolutionLimitReached)
        );
    }

    #[test]
    fn test_callbacks_are_forwarded() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        struct CountingMonitor {
            count: Arc<AtomicU32>,
        }
        impl LocalSearchMonitor<i64> for CountingMonitor {
            fn name(&self) -> &str {
                "CountingMonitor"
            }
            fn on_start(&mut self, _: &talos_model::model::Model<i64>, _: SolutionView<'_, i64>) {}
            fn on_end(&mut self, _: SolutionView<'_, i64>, _: &LocalSearchStatistics) {}
            fn on_iteration(
                &mut self,
                _: SolutionView<'_, i64>,
                _: SolutionView<'_, i64>,
                _: Option<SolutionView<'_, i64>>,
                _: &LocalSearchStatistics,
            ) {
                self.count.fetch_add(1, Ordering::Relaxed);
            }
            fn on_candidate_generated(
                &mut self,
                _: SolutionView<'_, i64>,
                _: SolutionView<'_, i64>,
                _: Option<SolutionView<'_, i64>>,
                _: i64,
                _: &LocalSearchStatistics,
            ) {
            }
            fn on_solution_buffered(
                &mut self,
                _: SolutionView<'_, i64>,
                _: SolutionView<'_, i64>,
                _: SolutionView<'_, i64>,
                _: &LocalSearchStatistics,
            ) {
            }
            fn on_candidate_accepted(
                &mut self,
                _: SolutionView<'_, i64>,
                _: SolutionView<'_, i64>,
                _: Option<SolutionView<'_, i64>>,
                _: &LocalSearchStatistics,
            ) {
            }
            fn on_buffered_solution_accepted(
                &mut self,
                _: SolutionView<'_, i64>,
                _: SolutionView<'_, i64>,
                _: &LocalSearchStatistics,
            ) {
            }
            fn on_candidate_rejected(
                &mut self,
                _: SolutionView<'_, i64>,
                _: SolutionView<'_, i64>,
                _: Option<SolutionView<'_, i64>>,
                _: i64,
                _: &LocalSearchStatistics,
            ) {
            }
            fn on_neighborhood_exhausted(
                &mut self,
                _: SolutionView<'_, i64>,
                _: SolutionView<'_, i64>,
                _: Option<SolutionView<'_, i64>>,
                _: &LocalSearchStatistics,
            ) {
            }
            fn on_best_solution_updated(
                &mut self,
                _: SolutionView<'_, i64>,
                _: SolutionView<'_, i64>,
                _: Option<SolutionView<'_, i64>>,
                _: SolutionView<'_, i64>,
                _: &LocalSearchStatistics,
            ) {
            }
        }

        let counter = Arc::new(AtomicU32::new(0));
        let mut m: CompositeLocalSearchMonitor<'_, i64> = CompositeLocalSearchMonitor::new();
        m.add_monitor(CountingMonitor {
            count: Arc::clone(&counter),
        });
        m.add_monitor(CountingMonitor {
            count: Arc::clone(&counter),
        });

        let sv = dummy_view();
        let stats = LocalSearchStatistics::default();
        m.on_iteration(sv, sv, None, &stats);
        m.on_iteration(sv, sv, None, &stats);

        // 2 monitors × 2 calls = 4
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }
}
