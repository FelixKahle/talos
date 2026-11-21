// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The observer of a run: it is told of the coarse events of the search, and may end the run.

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

/// The observer of a run, told of its coarse events: the start and the end of the search, each accepted move,
/// each scan of the neighbourhood that runs to exhaustion, and each new best.
///
/// Every method does nothing by default, and an implementation overrides those it needs; the run is generic over
/// the monitor, so a method left as it is costs nothing. No method is called per candidate: the counts of the
/// candidates, and of their fates, are in the statistics of every [`SearchProgress`].
/// [`search_command`](Self::search_command) is asked before every scan and at every thousandth candidate, and
/// may end the run.
///
/// The search the monitor is told of starts once the run has loaded its schedule and set up its metaheuristic, so
/// that the time from [`on_start`](Self::on_start) to [`on_end`](Self::on_end) is the time of the search alone, the
/// time its deadlines are measured in.
pub trait LocalSearchMonitor<T>
where
    T: SolverNumeric,
{
    /// The search starts from the initial schedule, which is both its best and its accepted schedule.
    fn on_start(&mut self, _model: &Model<T>, _progress: SearchProgress<'_, T>) {}

    /// The search ends with the best schedule of `progress`.
    fn on_end(&mut self, _progress: SearchProgress<'_, T>) {}

    /// A candidate became the accepted schedule of `progress`.
    fn on_candidate_accepted(&mut self, _progress: SearchProgress<'_, T>) {}

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
