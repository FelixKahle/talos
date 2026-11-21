// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The observer of a run: it is told of every step, and may end the run.

use crate::{statistics::LocalSearchStatistics, termination::SearchCommand};
use talos_core::SolverNumeric;
use talos_model::{Model, SolutionView};

/// Where a run stands: its best and accepted schedules, the candidate kept aside in the current scan, if any, and
/// the work done.
#[derive(Debug, Clone, Copy)]
pub struct SearchProgress<'a, T> {
    /// The best schedule found so far.
    pub best: SolutionView<'a, T>,
    /// The schedule the run moves from.
    pub accepted: SolutionView<'a, T>,
    /// The best candidate of the current scan, kept aside by a best-improvement descent, or `None`.
    pub buffered: Option<SolutionView<'a, T>>,
    /// The work the run has done.
    pub statistics: &'a LocalSearchStatistics,
}

/// The observer of a run, told of every step: the start, each scan, each candidate and its fate, each new best and
/// the end.
///
/// Every method does nothing by default, and an implementation overrides those it needs; the run is generic over
/// the monitor, so a method left as it is costs nothing. A method that runs once per candidate stands on the hot
/// path of the run and should stay cheap. [`search_command`](Self::search_command) is asked after every candidate
/// and at the start of every scan, and may end the run.
pub trait LocalSearchMonitor<T>
where
    T: SolverNumeric,
{
    /// The run starts from its initial schedule, which is both its best and its accepted schedule.
    fn on_start(&mut self, _model: &Model<T>, _progress: SearchProgress<'_, T>) {}

    /// The run ends with the best schedule of `progress`.
    fn on_end(&mut self, _progress: SearchProgress<'_, T>) {}

    /// A scan of the neighbourhood of the accepted schedule starts.
    fn on_scan_started(&mut self, _progress: SearchProgress<'_, T>) {}

    /// A candidate decoded to a feasible schedule of `candidate_objective`, before its fate is decided.
    fn on_candidate_generated(
        &mut self,
        _progress: SearchProgress<'_, T>,
        _candidate_objective: T,
    ) {
    }

    /// A candidate decoded to no feasible schedule and was undone.
    fn on_candidate_infeasible(&mut self, _progress: SearchProgress<'_, T>) {}

    /// A candidate became the accepted schedule of `progress`.
    fn on_candidate_accepted(&mut self, _progress: SearchProgress<'_, T>) {}

    /// A candidate of `candidate_objective` was rejected and undone.
    fn on_candidate_rejected(&mut self, _progress: SearchProgress<'_, T>, _candidate_objective: T) {
    }

    /// A candidate was kept aside, as the buffered schedule of `progress`, and undone.
    fn on_solution_buffered(&mut self, _progress: SearchProgress<'_, T>) {}

    /// The candidate kept aside became the accepted schedule of `progress` at the end of a scan.
    fn on_buffered_solution_accepted(&mut self, _progress: SearchProgress<'_, T>) {}

    /// A scan of the neighbourhood ran to exhaustion.
    fn on_neighborhood_exhausted(&mut self, _progress: SearchProgress<'_, T>) {}

    /// The accepted schedule became the new best schedule of `progress`.
    fn on_best_solution_updated(&mut self, _progress: SearchProgress<'_, T>) {}

    /// Whether the run goes on.
    fn search_command(&mut self, _progress: SearchProgress<'_, T>) -> SearchCommand {
        SearchCommand::Continue
    }
}

/// The monitor that observes nothing and never ends a run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NoMonitor;

impl<T> LocalSearchMonitor<T> for NoMonitor where T: SolverNumeric {}
