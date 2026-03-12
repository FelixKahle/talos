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

//! Monitoring interface for local search runs.
//!
//! This module defines callbacks for observing the lifecycle of the solver,
//! including start/end events, per‑iteration updates, and notifications on
//! solutions found, accepted, or rejected. Implementations can stream logs,
//! collect metrics, or trigger early termination by returning a search
//! command to the engine. The default `search_command` continues execution,
//! allowing monitors to remain lightweight unless an explicit limit or
//! condition is reached.

use crate::{exec::SearchCommand, stats::LocalSearchStatistics};
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

/// A monitor for local search algorithms.
pub trait LocalSearchMonitor<T>
where
    T: SolverNumeric,
{
    /// Returns the name of the monitor.
    fn name(&self) -> &str;

    /// Called at the start of the local search.
    /// `initial_solution` is the starting point before any exploration begins.
    fn on_start(&mut self, model: &Model<T>, initial_solution: SolutionView<'_, T>);

    /// Called at the end of the local search.
    /// `best_solution` is the globally best state found across all iterations.
    fn on_end(&mut self, best_solution: SolutionView<'_, T>, statistics: &LocalSearchStatistics);

    /// Called at the start of a new iteration.
    /// `current_solution` is the base state from which neighbors will be generated.
    fn on_iteration(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        statistics: &LocalSearchStatistics,
    );

    /// Called when a new neighbor is generated and evaluated.
    /// `candidate_solution` is being considered but has not yet been accepted or rejected.
    fn on_candidate_generated(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        candidate_objective: T, // The candidate solution is only ever partially constructed, so we only pass the objective value here.
        statistics: &LocalSearchStatistics,
    );

    /// Called when a new solution is found buffered.
    ///
    /// This might be used by the engine to remember a solution
    /// while traversing the current neighborhood.
    /// The buffered solution is not yet accepted or rejected, the engine
    /// might accept it later.
    fn on_solution_buffered(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: SolutionView<'_, T>,
        statistics: &LocalSearchStatistics,
    );

    /// Called when the algorithm accepts the candidate.
    /// `accepted_solution` now becomes the `current_solution` for the next iteration.
    fn on_candidate_accepted(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        statistics: &LocalSearchStatistics,
    );

    /// Called when the previously buffered solution is promoted to the accepted solution.
    /// At this point `accepted_solution` *is* the former buffered solution.
    fn on_buffered_solution_accepted(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        statistics: &LocalSearchStatistics,
    );

    /// Called when the algorithm rejects the candidate.
    /// The `current_solution` remains unchanged.
    fn on_candidate_rejected(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        rejected_objective: T,
        statistics: &LocalSearchStatistics,
    );

    /// Called when a mutation produced an infeasible candidate whose delta-decode failed.
    /// The engine has already rolled back the topology; this is a pure telemetry hook.
    fn on_candidate_infeasible(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    /// Called when the current neighborhood has been fully explored without finding
    /// an accepted move.
    fn on_neighborhood_exhausted(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        statistics: &LocalSearchStatistics,
    );

    /// Called specifically when a candidate is accepted AND strictly improves upon the global best.
    /// `new_best_solution` replaces the previous global best.
    fn on_best_solution_updated(
        &mut self,
        previous_best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        new_best_solution: SolutionView<'_, T>,
        statistics: &LocalSearchStatistics,
    );

    /// Determines the command for the next step of the local search.
    fn search_command(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _statistics: &LocalSearchStatistics,
    ) -> SearchCommand {
        SearchCommand::Continue
    }
}

impl<T> std::fmt::Debug for dyn LocalSearchMonitor<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalSearchMonitor {{ name: {} }}", self.name())
    }
}

impl<T> std::fmt::Display for dyn LocalSearchMonitor<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalSearchMonitor: {}", self.name())
    }
}
