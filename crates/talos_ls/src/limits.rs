// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The budgets and deadlines that end a run, and the watch that measures a run against them.

use crate::{statistics::LocalSearchStatistics, termination::TerminationReason};
use std::time::{Duration, Instant};

/// The candidates between two checkpoints of a run, at which it reads the clock where a deadline is set and asks
/// its monitor whether to go on.
///
/// A read of the monotonic clock, a call into the system, costs no more than the cheapest candidate, a move and the
/// decode of one or two berths, and neither does the question to a monitor that keeps to its contract. A checkpoint
/// every `CHECKPOINT_INTERVAL` candidates so costs at most `1 / CHECKPOINT_INTERVAL` of the run, and a thousand
/// keeps that within a tenth of a percent. Between two checkpoints a run makes at most this many candidates, and
/// the moves its filters skip on the way.
pub(super) const CHECKPOINT_INTERVAL: u64 = 1000;

/// The budgets and deadlines that end a run; a `None` sets no limit of its kind, and a run with none ends only when
/// its metaheuristic or its monitor ends it.
///
/// The budgets count work, as [`LocalSearchStatistics`] does, and each ends the run at the step at which its count
/// reaches it. The deadlines are measured from the start of the search, after the run has loaded its schedule, and
/// are checked at every thousandth candidate and at the end of every scan of the neighbourhood that runs to
/// exhaustion: a run overruns a deadline by at most a thousand candidates and the moves its filters skip on the way.
/// Where several limits are reached at one check, the first in the order of the fields names the reason the run
/// ends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SearchLimits {
    /// The longest the search may take from its start.
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
    /// The longest the search may go since its last new best, or since its start.
    pub non_improving_time: Option<Duration>,
}

/// Whether a check of the limits reads the clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Clock {
    /// The check measures the deadlines too.
    Read,
    /// The check measures the budgets alone.
    Skip,
}

/// The limits of one run, with the marks they are measured from, and the candidate at which the run must next be
/// checked.
///
/// The run counts its candidates and asks [`is_due`](Self::is_due) after each; only a due step checks the limits
/// and is a checkpoint. A step is due at a checkpoint, at the step at which the budget of candidates or of
/// candidates since the last new best runs out, and at a feasible candidate that exhausts the budget of feasible
/// candidates. The scans are checked at their end, where their counts change.
// Invariant: `due` is zero once the feasible candidates exhausted their budget, and otherwise the least of
// `checkpoint` and the counts at which the two budgets of candidates run out.
// `started` holds an instant exactly when the limits set a time, and `last_improved_at` exactly when they set a
// non-improving time.
#[derive(Debug, Clone)]
pub(super) struct LimitWatch {
    /// The limits measured.
    limits: SearchLimits,
    /// When the search started, if it has a deadline.
    started: Option<Instant>,
    /// When the search last found a new best or started, if it has a non-improving deadline.
    last_improved_at: Option<Instant>,
    /// The candidate moves made when the search last found a new best.
    last_improved_iteration: u64,
    /// The scans exhausted when the search last found a new best.
    last_improved_cycle: u64,
    /// The candidate count of the next checkpoint.
    checkpoint: u64,
    /// The candidate count at which the run must next be checked.
    due: u64,
}

impl LimitWatch {
    /// Starts measuring `limits` now, at the start of a search; reads the clock if a deadline is set.
    pub(super) fn start(limits: SearchLimits) -> Self {
        let mut watch = Self {
            limits,
            started: limits.time.map(|_| Instant::now()),
            last_improved_at: limits.non_improving_time.map(|_| Instant::now()),
            last_improved_iteration: 0,
            last_improved_cycle: 0,
            checkpoint: CHECKPOINT_INTERVAL,
            due: 0,
        };
        watch.schedule(&LocalSearchStatistics::default());
        watch
    }

    /// Whether the step that made the `iterations`-th candidate must be checked.
    #[inline(always)]
    pub(super) fn is_due(&self, iterations: u64) -> bool {
        iterations >= self.due
    }

    /// Makes the current step due if `total_solutions` feasible candidates exhaust their budget.
    #[inline(always)]
    pub(super) fn note_solution(&mut self, total_solutions: u64) {
        if self.solutions_reached(total_solutions) {
            self.due = 0;
        }
    }

    /// Measures the non-improving limits from now, at a new best found after the work `statistics` counts; reads
    /// the clock if a non-improving deadline is set.
    pub(super) fn on_new_best(&mut self, statistics: &LocalSearchStatistics) {
        self.last_improved_iteration = statistics.iterations;
        self.last_improved_cycle = statistics.cycles;
        if self.last_improved_at.is_some() {
            self.last_improved_at = Some(Instant::now());
        }
        self.schedule(statistics);
    }

    /// The first limit reached after the work `statistics` counts, in the order of the fields of
    /// [`SearchLimits`], or `None`; measures the deadlines only where `clock` says so.
    pub(super) fn reached(
        &self,
        statistics: &LocalSearchStatistics,
        clock: Clock,
    ) -> Option<TerminationReason> {
        let past = |since: Option<Instant>, limit: Option<Duration>| match (clock, since, limit) {
            (Clock::Read, Some(since), Some(limit)) => since.elapsed() >= limit,
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

    /// Checks a due step after the work `statistics` counts: the first limit reached, measuring the deadlines if
    /// the step is a checkpoint, and whether it is one; schedules the next due step.
    pub(super) fn check_step(&mut self, statistics: &LocalSearchStatistics) -> StepCheck {
        let checkpoint = statistics.iterations >= self.checkpoint;
        let clock = match checkpoint {
            true => Clock::Read,
            false => Clock::Skip,
        };
        let reached = self.reached(statistics, clock);
        if checkpoint {
            self.checkpoint = statistics.iterations + CHECKPOINT_INTERVAL;
        }
        self.schedule(statistics);
        match (reached, checkpoint) {
            (Some(reason), _) => StepCheck::Reached(reason),
            (None, true) => StepCheck::Checkpoint,
            (None, false) => StepCheck::Continue,
        }
    }

    /// Sets the due step after the work `statistics` counts: the current one if the feasible candidates exhausted
    /// their budget, or else the least of the next checkpoint and the steps at which the budgets of candidates run
    /// out.
    fn schedule(&mut self, statistics: &LocalSearchStatistics) {
        self.due = if self.solutions_reached(statistics.total_solutions) {
            0
        } else {
            let iterations = self.limits.iterations.unwrap_or(u64::MAX);
            self.checkpoint
                .min(iterations)
                .min(self.non_improving_due())
        };
    }

    /// Whether `total_solutions` feasible candidates exhaust their budget.
    #[inline(always)]
    fn solutions_reached(&self, total_solutions: u64) -> bool {
        self.limits
            .solutions
            .is_some_and(|limit| total_solutions >= limit)
    }

    /// The candidate count at which the budget of candidates since the last new best runs out, or `u64::MAX`.
    fn non_improving_due(&self) -> u64 {
        self.limits
            .non_improving_iterations
            .map_or(u64::MAX, |limit| {
                self.last_improved_iteration.saturating_add(limit)
            })
    }
}

/// What the check of a due step found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StepCheck {
    /// A limit was reached: the run ends for this reason.
    Reached(TerminationReason),
    /// No limit was reached, and the step is a checkpoint: the run asks its monitor.
    Checkpoint,
    /// No limit was reached, and the run goes on.
    Continue,
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
        assert_eq!(
            watch.reached(&work(u64::MAX, u64::MAX, u64::MAX), Clock::Read),
            None
        );
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
        let reached = |iterations, solutions, cycles| {
            watch.reached(&work(iterations, solutions, cycles), Clock::Skip)
        };
        assert_eq!(reached(9, 4, 1), None);
        assert_eq!(
            reached(10, 4, 1),
            Some(TerminationReason::IterationLimitReached)
        );
        assert_eq!(
            reached(9, 5, 1),
            Some(TerminationReason::SolutionLimitReached)
        );
        assert_eq!(reached(9, 4, 2), Some(TerminationReason::CycleLimitReached));
        // At one check, the first field names the reason.
        assert_eq!(
            reached(10, 5, 2),
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
            watch.reached(&work(10, 0, 0), Clock::Skip),
            Some(TerminationReason::MaxNonImprovingIterations)
        );
        watch.on_new_best(&work(8, 0, 1));
        assert_eq!(watch.reached(&work(17, 0, 3), Clock::Skip), None);
        assert_eq!(
            watch.reached(&work(18, 0, 3), Clock::Skip),
            Some(TerminationReason::MaxNonImprovingIterations)
        );
        assert_eq!(
            watch.reached(&work(17, 0, 4), Clock::Skip),
            Some(TerminationReason::MaxNonImprovingCycles)
        );
    }

    #[test]
    fn test_deadlines_are_measured_only_where_the_clock_is_read() {
        let watch = LimitWatch::start(SearchLimits {
            time: Some(Duration::ZERO),
            non_improving_time: Some(Duration::ZERO),
            ..SearchLimits::default()
        });
        assert_eq!(watch.reached(&work(1, 0, 0), Clock::Skip), None);
        assert_eq!(
            watch.reached(&work(1, 0, 0), Clock::Read),
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
            expired.reached(&work(0, 0, 0), Clock::Read),
            Some(TerminationReason::MaxNonImprovingTime)
        );
        let mut watch = LimitWatch::start(SearchLimits {
            non_improving_time: Some(Duration::from_secs(3600)),
            ..SearchLimits::default()
        });
        let started = watch
            .last_improved_at
            .expect("a non-improving deadline should read the clock at the start");
        assert_eq!(watch.reached(&work(0, 0, 0), Clock::Read), None);
        watch.on_new_best(&work(5, 0, 0));
        assert!(watch.last_improved_at.is_some_and(|at| at >= started));
    }

    #[test]
    fn test_a_step_is_due_at_each_checkpoint_and_where_a_budget_of_candidates_runs_out() {
        let mut watch = LimitWatch::start(SearchLimits::default());
        assert!(!watch.is_due(CHECKPOINT_INTERVAL - 1));
        assert!(watch.is_due(CHECKPOINT_INTERVAL));
        assert_eq!(
            watch.check_step(&work(CHECKPOINT_INTERVAL, 0, 0)),
            StepCheck::Checkpoint
        );
        assert!(!watch.is_due(2 * CHECKPOINT_INTERVAL - 1));
        assert!(watch.is_due(2 * CHECKPOINT_INTERVAL));

        let mut watch = LimitWatch::start(SearchLimits {
            iterations: Some(7),
            non_improving_iterations: Some(5),
            ..SearchLimits::default()
        });
        assert!(!watch.is_due(4) && watch.is_due(5));
        watch.on_new_best(&work(3, 0, 0));
        assert!(!watch.is_due(6) && watch.is_due(7));
        assert_eq!(
            watch.check_step(&work(7, 0, 0)),
            StepCheck::Reached(TerminationReason::IterationLimitReached)
        );
    }

    #[test]
    fn test_a_feasible_candidate_that_exhausts_its_budget_makes_the_step_due() {
        let mut watch = LimitWatch::start(SearchLimits {
            solutions: Some(3),
            ..SearchLimits::default()
        });
        watch.note_solution(2);
        assert!(!watch.is_due(5));
        watch.note_solution(3);
        assert!(watch.is_due(5));
        assert_eq!(
            watch.check_step(&work(5, 3, 0)),
            StepCheck::Reached(TerminationReason::SolutionLimitReached)
        );
    }

    #[test]
    fn test_a_check_between_checkpoints_reads_no_clock() {
        let mut watch = LimitWatch::start(SearchLimits {
            time: Some(Duration::ZERO),
            iterations: Some(5),
            ..SearchLimits::default()
        });
        assert_eq!(watch.check_step(&work(4, 0, 0)), StepCheck::Continue);
        assert_eq!(
            watch.check_step(&work(5, 0, 0)),
            StepCheck::Reached(TerminationReason::IterationLimitReached)
        );
        assert_eq!(
            watch.check_step(&work(CHECKPOINT_INTERVAL, 0, 0)),
            StepCheck::Reached(TerminationReason::TimeLimitReached)
        );
    }
}
