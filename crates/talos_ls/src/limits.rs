// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The budgets and deadlines that end a run.

use crate::{statistics::LocalSearchStatistics, termination::TerminationReason};
use std::time::{Duration, Instant};

/// The candidates between two reads of the clock while a deadline is set, a power of two.
///
/// A candidate costs from about a hundred nanoseconds to a few microseconds, so a deadline is overrun by at most a
/// few milliseconds, while the read of the clock, some tens of nanoseconds, costs well under a thousandth of a
/// candidate.
const CLOCK_CHECK_INTERVAL: u64 = 8192;

/// The budgets and deadlines that end a run; a `None` sets no limit of its kind, and a run with none ends only when
/// its metaheuristic or its monitor ends it.
///
/// The budgets count work, as [`LocalSearchStatistics`] does. A deadline is checked at every
/// 8192nd candidate, the only time the run reads the clock. Where several limits are reached at one step, the
/// first in the order of the fields names the reason the run ends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SearchLimits {
    /// The longest the run may take from its start.
    pub time: Option<Duration>,
    /// The most candidate moves the run may make.
    pub iterations: Option<u64>,
    /// The most feasible candidates the run may decode.
    pub solutions: Option<u64>,
    /// The most scans of the neighbourhood the run may exhaust.
    pub cycles: Option<u64>,
    /// The most candidate moves the run may make since its last new best, or since its start.
    pub non_improving_iterations: Option<u64>,
    /// The most scans the run may exhaust since its last new best, or since its start.
    pub non_improving_cycles: Option<u64>,
    /// The longest the run may go since its last new best, or since its start.
    pub non_improving_time: Option<Duration>,
}

/// The limits of one run, with the marks they are measured from: the start, and the last new best.
#[derive(Debug, Clone)]
pub(super) struct LimitWatch {
    /// The limits measured.
    limits: SearchLimits,
    /// When the run started, read only if it has a time limit.
    started: Option<Instant>,
    /// When the run last found a new best or started, read only if it has a non-improving time limit.
    last_improved_at: Option<Instant>,
    /// The candidate moves made when the run last found a new best.
    last_improved_iteration: u64,
    /// The scans exhausted when the run last found a new best.
    last_improved_cycle: u64,
}

impl LimitWatch {
    /// Starts measuring `limits` now, at the start of a run.
    pub(super) fn start(limits: SearchLimits) -> Self {
        Self {
            limits,
            started: limits.time.map(|_| Instant::now()),
            last_improved_at: limits.non_improving_time.map(|_| Instant::now()),
            last_improved_iteration: 0,
            last_improved_cycle: 0,
        }
    }

    /// Measures the non-improving limits from now, at a new best found after the work `statistics` counts.
    pub(super) fn on_new_best(&mut self, statistics: &LocalSearchStatistics) {
        self.last_improved_iteration = statistics.iterations;
        self.last_improved_cycle = statistics.cycles;
        if self.last_improved_at.is_some() {
            self.last_improved_at = Some(Instant::now());
        }
    }

    /// The first limit reached after the work `statistics` counts, in the order of the fields of
    /// [`SearchLimits`], or `None`; the deadlines are checked only at every 8192nd candidate.
    pub(super) fn reached(&self, statistics: &LocalSearchStatistics) -> Option<TerminationReason> {
        let clock_due = statistics.iterations.is_multiple_of(CLOCK_CHECK_INTERVAL);
        let past = |since: Option<Instant>, limit: Option<Duration>| match (since, limit) {
            (Some(since), Some(limit)) => clock_due && since.elapsed() >= limit,
            _ => false,
        };
        let at_least = |count: u64, limit: Option<u64>| limit.is_some_and(|limit| count >= limit);
        if past(self.started, self.limits.time) {
            Some(TerminationReason::TimeLimitReached)
        } else if at_least(statistics.iterations, self.limits.iterations) {
            Some(TerminationReason::IterationLimitReached)
        } else if at_least(statistics.total_solutions, self.limits.solutions) {
            Some(TerminationReason::SolutionLimitReached)
        } else if at_least(statistics.cycles, self.limits.cycles) {
            Some(TerminationReason::CycleLimitReached)
        } else if at_least(
            statistics.iterations - self.last_improved_iteration,
            self.limits.non_improving_iterations,
        ) {
            Some(TerminationReason::MaxNonImprovingIterations)
        } else if at_least(
            statistics.cycles - self.last_improved_cycle,
            self.limits.non_improving_cycles,
        ) {
            Some(TerminationReason::MaxNonImprovingCycles)
        } else if past(self.last_improved_at, self.limits.non_improving_time) {
            Some(TerminationReason::MaxNonImprovingTime)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work(iterations: u64, total_solutions: u64, cycles: u64) -> LocalSearchStatistics {
        LocalSearchStatistics {
            iterations,
            cycles,
            total_solutions,
            ..LocalSearchStatistics::default()
        }
    }

    #[test]
    fn test_no_limit_is_never_reached() {
        let watch = LimitWatch::start(SearchLimits::default());
        assert_eq!(watch.reached(&work(u64::MAX, u64::MAX, u64::MAX)), None);
        assert_eq!(watch.started, None);
        assert_eq!(watch.last_improved_at, None);
    }

    #[test]
    fn test_budgets_are_reached_at_their_count() {
        let limits = SearchLimits {
            iterations: Some(10),
            solutions: Some(5),
            cycles: Some(2),
            ..SearchLimits::default()
        };
        let watch = LimitWatch::start(limits);
        assert_eq!(watch.reached(&work(9, 4, 1)), None);
        assert_eq!(
            watch.reached(&work(10, 4, 1)),
            Some(TerminationReason::IterationLimitReached)
        );
        assert_eq!(
            watch.reached(&work(9, 5, 1)),
            Some(TerminationReason::SolutionLimitReached)
        );
        assert_eq!(
            watch.reached(&work(9, 4, 2)),
            Some(TerminationReason::CycleLimitReached)
        );
        // At one step, the first field names the reason.
        assert_eq!(
            watch.reached(&work(10, 5, 2)),
            Some(TerminationReason::IterationLimitReached)
        );
    }

    #[test]
    fn test_non_improving_budgets_count_from_the_last_new_best() {
        let limits = SearchLimits {
            non_improving_iterations: Some(10),
            non_improving_cycles: Some(3),
            ..SearchLimits::default()
        };
        let mut watch = LimitWatch::start(limits);
        assert_eq!(
            watch.reached(&work(10, 0, 0)),
            Some(TerminationReason::MaxNonImprovingIterations)
        );
        watch.on_new_best(&work(8, 0, 1));
        assert_eq!(watch.reached(&work(17, 0, 3)), None);
        assert_eq!(
            watch.reached(&work(18, 0, 3)),
            Some(TerminationReason::MaxNonImprovingIterations)
        );
        assert_eq!(
            watch.reached(&work(17, 0, 4)),
            Some(TerminationReason::MaxNonImprovingCycles)
        );
    }

    #[test]
    fn test_deadlines_are_read_only_at_every_interval_of_candidates() {
        let limits = SearchLimits {
            time: Some(Duration::ZERO),
            ..SearchLimits::default()
        };
        let watch = LimitWatch::start(limits);
        assert_eq!(watch.reached(&work(1, 0, 0)), None);
        assert_eq!(watch.reached(&work(CLOCK_CHECK_INTERVAL - 1, 0, 0)), None);
        assert_eq!(
            watch.reached(&work(0, 0, 0)),
            Some(TerminationReason::TimeLimitReached)
        );
        assert_eq!(
            watch.reached(&work(CLOCK_CHECK_INTERVAL, 0, 0)),
            Some(TerminationReason::TimeLimitReached)
        );
    }

    #[test]
    fn test_non_improving_deadline_is_measured_from_the_last_new_best() {
        let expired = LimitWatch::start(SearchLimits {
            non_improving_time: Some(Duration::ZERO),
            ..SearchLimits::default()
        });
        assert_eq!(
            expired.reached(&work(0, 0, 0)),
            Some(TerminationReason::MaxNonImprovingTime)
        );
        let mut watch = LimitWatch::start(SearchLimits {
            non_improving_time: Some(Duration::from_secs(3600)),
            ..SearchLimits::default()
        });
        let started = watch
            .last_improved_at
            .expect("a non-improving deadline should read the clock at the start");
        assert_eq!(watch.reached(&work(0, 0, 0)), None);
        watch.on_new_best(&work(5, 0, 0));
        assert!(watch.last_improved_at.is_some_and(|at| at >= started));
    }
}
