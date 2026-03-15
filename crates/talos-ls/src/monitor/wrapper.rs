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

//! Adapter that bridges a [`PortfolioMonitor`] into a [`LocalSearchMonitor`].
//!
//! This wrapper allows a portfolio-level monitor to observe local search
//! runs without knowing any engine-specific details. It translates LS
//! lifecycle events (`on_start`, `on_end`, `on_best_solution_updated`)
//! into portfolio-level callbacks and maps [`PortfolioCommand::Terminate`]
//! to [`SearchCommand::Terminate(Interrupted)`].
//!
//! All LS-specific events (iterations, candidates, neighborhoods, etc.)
//! are silently ignored.

use crate::{
    exec::{SearchCommand, TerminationReason},
    monitor::lsmonitor::LocalSearchMonitor,
    stats::LocalSearchStatistics,
};
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};
use talos_search::monitor::psmonitor::{PortfolioCommand, PortfolioMonitor};

/// Wraps a [`PortfolioMonitor`] so it can be used as a [`LocalSearchMonitor`].
///
/// Only portfolio-relevant events are forwarded; all LS-specific callbacks
/// (iterations, candidates, neighborhoods) are no-ops.
pub struct PortfolioMonitorWrapper<T: SolverNumeric, M: PortfolioMonitor<T>> {
    solver_name: String,
    inner: M,
    _marker: std::marker::PhantomData<T>,
}

impl<T: SolverNumeric, M: PortfolioMonitor<T>> PortfolioMonitorWrapper<T, M> {
    /// Creates a new wrapper that forwards LS events to `inner`
    /// under the given `solver_name`.
    pub fn new(solver_name: String, inner: M) -> Self {
        Self {
            solver_name,
            inner,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns a reference to the inner portfolio monitor.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Returns a mutable reference to the inner portfolio monitor.
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    /// Consumes the wrapper and returns the inner portfolio monitor.
    pub fn into_inner(self) -> M {
        self.inner
    }
}

impl<T: SolverNumeric, M: PortfolioMonitor<T>> LocalSearchMonitor<T>
    for PortfolioMonitorWrapper<T, M>
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn on_start(&mut self, _model: &Model<T>, _initial_solution: SolutionView<'_, T>) {
        self.inner.on_solver_started(&self.solver_name);
    }

    fn on_end(&mut self, best_solution: SolutionView<'_, T>, _statistics: &LocalSearchStatistics) {
        self.inner
            .on_solver_finished(&self.solver_name, best_solution);
    }

    fn on_iteration(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_candidate_generated(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _candidate_objective: T,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_solution_buffered(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: SolutionView<'_, T>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_candidate_accepted(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_buffered_solution_accepted(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_candidate_rejected(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _rejected_objective: T,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_neighborhood_exhausted(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_best_solution_updated(
        &mut self,
        previous_best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        new_best_solution: SolutionView<'_, T>,
        _statistics: &LocalSearchStatistics,
    ) {
        self.inner.on_best_solution_updated(
            &self.solver_name,
            previous_best_solution,
            new_best_solution,
        );
    }

    fn search_command(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _statistics: &LocalSearchStatistics,
    ) -> SearchCommand {
        match self.inner.portfolio_command() {
            PortfolioCommand::Continue => SearchCommand::Continue,
            PortfolioCommand::Terminate => SearchCommand::Terminate(TerminationReason::Interrupted),
        }
    }
}
