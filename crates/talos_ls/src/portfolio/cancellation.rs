// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The flag by which a member of a portfolio that panics stops its siblings: the guard that raises it as the
//! panic unwinds, and the monitor through which each run reads it.

use crate::{
    monitor::{LocalSearchMonitor, SearchProgress},
    termination::{SearchCommand, TerminationReason},
};
use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicBool, Ordering};
use talos_core::SolverNumeric;
use talos_model::Model;

/// Whether the runs of a portfolio are to stop: raised once by a member that panics, and read by every member's
/// run before each scan and at each checkpoint.
///
/// The flag carries no data that a reader must see with it, so a relaxed load and store suffice: a raised flag is
/// seen at a later read, and a run that reads it a little late makes a few more candidates. It stands on a cache
/// line of its own, so that the runs that read it share that line with nothing they write.
#[derive(Debug, Default)]
pub(super) struct Cancellation {
    raised: CachePadded<AtomicBool>,
}

impl Cancellation {
    /// Asks every run that reads the flag to stop.
    fn raise(&self) {
        self.raised.store(true, Ordering::Relaxed);
    }

    /// Whether the runs are to stop.
    pub(super) fn is_raised(&self) -> bool {
        self.raised.load(Ordering::Relaxed)
    }
}

/// The guard of one member's run: dropped while its thread panics, it raises the flag, so that the siblings of
/// the member stop at their next scan or checkpoint while the panic unwinds to the caller.
#[derive(Debug)]
pub(super) struct CancelOnPanic<'a> {
    cancellation: &'a Cancellation,
}

impl<'a> CancelOnPanic<'a> {
    /// The guard that raises `cancellation` if it is dropped by a panic.
    pub(super) fn new(cancellation: &'a Cancellation) -> Self {
        Self { cancellation }
    }
}

impl Drop for CancelOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.cancellation.raise();
        }
    }
}

/// The monitor of one member's run: it tells the member's own monitor of every event and asks it whether the run
/// goes on, but ends the run as [`TerminationReason::Interrupted`] once the flag of the portfolio is raised.
///
/// The flag is read where the run asks its monitor, before each scan and at each checkpoint, so a run that the
/// flag never stops answers what it answers under the member's monitor alone.
// Every method of `LocalSearchMonitor` is forwarded here by hand: a method the trait gains is forwarded too, or the
// member's monitor is never told of it.
pub(super) struct CancellableMonitor<'a, M> {
    monitor: &'a mut M,
    cancellation: &'a Cancellation,
}

impl<'a, M> CancellableMonitor<'a, M> {
    /// The monitor that tells `monitor` of every event and ends the run once `cancellation` is raised.
    pub(super) fn new(monitor: &'a mut M, cancellation: &'a Cancellation) -> Self {
        Self {
            monitor,
            cancellation,
        }
    }
}

impl<M> std::fmt::Debug for CancellableMonitor<'_, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellableMonitor")
            .field("cancellation", &self.cancellation)
            .finish_non_exhaustive()
    }
}

impl<T, M> LocalSearchMonitor<T> for CancellableMonitor<'_, M>
where
    T: SolverNumeric,
    M: LocalSearchMonitor<T>,
{
    fn on_start(&mut self, model: &Model<T>, progress: SearchProgress<'_, T>) {
        self.monitor.on_start(model, progress);
    }

    fn on_end(&mut self, progress: SearchProgress<'_, T>) {
        self.monitor.on_end(progress);
    }

    fn on_candidate_accepted(&mut self, progress: SearchProgress<'_, T>) {
        self.monitor.on_candidate_accepted(progress);
    }

    fn on_buffered_solution_accepted(&mut self, progress: SearchProgress<'_, T>) {
        self.monitor.on_buffered_solution_accepted(progress);
    }

    fn on_neighborhood_exhausted(&mut self, progress: SearchProgress<'_, T>) {
        self.monitor.on_neighborhood_exhausted(progress);
    }

    fn on_best_solution_updated(&mut self, progress: SearchProgress<'_, T>) {
        self.monitor.on_best_solution_updated(progress);
    }

    fn search_command(&mut self, progress: SearchProgress<'_, T>) -> SearchCommand {
        match self.cancellation.is_raised() {
            true => SearchCommand::Terminate(TerminationReason::Interrupted),
            false => self.monitor.search_command(progress),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::LocalSearchStatistics;
    use talos_core::ClosedOpenInterval;
    use talos_model::{BerthIndex, SolutionView};

    /// Every event a monitor can be told of, with the iterations of its progress, which tell the events apart.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Event {
        Start(u64),
        End(u64),
        CandidateAccepted(u64),
        BufferedSolutionAccepted(u64),
        NeighborhoodExhausted(u64),
        BestSolutionUpdated(u64),
        SearchCommand(u64),
    }

    /// A monitor that records every event it is told of, in order, and lets every run go on.
    #[derive(Debug, Default)]
    struct Recording(Vec<Event>);

    impl LocalSearchMonitor<i64> for Recording {
        fn on_start(&mut self, _: &Model<i64>, progress: SearchProgress<'_, i64>) {
            self.0.push(Event::Start(progress.statistics.iterations));
        }
        fn on_end(&mut self, progress: SearchProgress<'_, i64>) {
            self.0.push(Event::End(progress.statistics.iterations));
        }
        fn on_candidate_accepted(&mut self, progress: SearchProgress<'_, i64>) {
            self.0
                .push(Event::CandidateAccepted(progress.statistics.iterations));
        }
        fn on_buffered_solution_accepted(&mut self, progress: SearchProgress<'_, i64>) {
            self.0.push(Event::BufferedSolutionAccepted(
                progress.statistics.iterations,
            ));
        }
        fn on_neighborhood_exhausted(&mut self, progress: SearchProgress<'_, i64>) {
            self.0
                .push(Event::NeighborhoodExhausted(progress.statistics.iterations));
        }
        fn on_best_solution_updated(&mut self, progress: SearchProgress<'_, i64>) {
            self.0
                .push(Event::BestSolutionUpdated(progress.statistics.iterations));
        }
        fn search_command(&mut self, progress: SearchProgress<'_, i64>) -> SearchCommand {
            self.0
                .push(Event::SearchCommand(progress.statistics.iterations));
            SearchCommand::Continue
        }
    }

    /// A monitor that ends every run it is asked about as aborted, and counts the questions.
    #[derive(Debug, Default)]
    struct AbortingMonitor {
        questions: u64,
    }

    impl LocalSearchMonitor<i64> for AbortingMonitor {
        fn search_command(&mut self, _: SearchProgress<'_, i64>) -> SearchCommand {
            self.questions += 1;
            SearchCommand::Terminate(TerminationReason::Aborted)
        }
    }

    #[test]
    fn test_the_guard_raises_the_flag_only_when_dropped_by_a_panic() {
        let cancellation = Cancellation::default();
        drop(CancelOnPanic::new(&cancellation));
        assert!(!cancellation.is_raised());

        let unwound = std::panic::catch_unwind(|| {
            let _guard = CancelOnPanic::new(&cancellation);
            panic!("a member panics");
        });
        assert!(unwound.is_err());
        assert!(cancellation.is_raised());
    }

    #[test]
    fn test_the_monitor_asks_the_member_until_the_flag_is_raised() {
        let statistics = LocalSearchStatistics::default();
        let berths = [BerthIndex::new(0)];
        let view = SolutionView::new(&berths, &[0], 0);
        let progress = SearchProgress {
            best: view,
            accepted: view,
            buffered: None,
            statistics: &statistics,
        };
        let cancellation = Cancellation::default();
        let mut member = AbortingMonitor::default();
        let mut monitor = CancellableMonitor::new(&mut member, &cancellation);
        assert_eq!(
            monitor.search_command(progress),
            SearchCommand::Terminate(TerminationReason::Aborted)
        );
        cancellation.raise();
        assert_eq!(
            monitor.search_command(progress),
            SearchCommand::Terminate(TerminationReason::Interrupted)
        );
        assert_eq!(member.questions, 1);
    }

    #[test]
    fn test_the_monitor_tells_the_member_of_every_event_in_order() {
        let model = Model::new(
            vec![0],
            vec![10],
            vec![1],
            vec![Some(1)],
            vec![vec![ClosedOpenInterval::new(0, 10)]],
        );
        let berths = [BerthIndex::new(0)];
        let view = SolutionView::new(&berths, &[0], 0);
        let statistics: Vec<_> = (0..7)
            .map(|iterations| LocalSearchStatistics {
                iterations,
                ..LocalSearchStatistics::default()
            })
            .collect();
        let progress = |at: usize| SearchProgress {
            best: view,
            accepted: view,
            buffered: None,
            statistics: &statistics[at],
        };
        let cancellation = Cancellation::default();
        let mut member = Recording::default();
        let mut monitor = CancellableMonitor::new(&mut member, &cancellation);
        // Each method once, in an order no run follows, so that a call forwarded to the wrong method shows.
        monitor.on_best_solution_updated(progress(0));
        monitor.on_start(&model, progress(1));
        monitor.on_neighborhood_exhausted(progress(2));
        monitor.on_end(progress(3));
        assert_eq!(monitor.search_command(progress(4)), SearchCommand::Continue);
        monitor.on_buffered_solution_accepted(progress(5));
        monitor.on_candidate_accepted(progress(6));
        assert_eq!(
            member.0,
            [
                Event::BestSolutionUpdated(0),
                Event::Start(1),
                Event::NeighborhoodExhausted(2),
                Event::End(3),
                Event::SearchCommand(4),
                Event::BufferedSolutionAccepted(5),
                Event::CandidateAccepted(6),
            ]
        );
    }
}
