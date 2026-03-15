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

//! Monitoring interface for portfolio solver runs.
//!
//! This module defines callbacks for observing the lifecycle of a portfolio
//! of solvers, including start/end events, per-solver notifications, and
//! global-best updates. Implementations can stream logs, collect metrics,
//! or trigger early termination by returning a [`PortfolioCommand`].
//! The default `portfolio_command` continues execution, allowing monitors
//! to remain lightweight unless an explicit limit or condition is reached.

use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

/// Command returned by a [`PortfolioMonitor`] to control portfolio execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PortfolioCommand {
    /// The portfolio should keep running.
    #[default]
    Continue,
    /// The portfolio should stop all solvers.
    Terminate,
}

impl std::fmt::Display for PortfolioCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PortfolioCommand::Continue => write!(f, "Continue"),
            PortfolioCommand::Terminate => write!(f, "Terminate"),
        }
    }
}

/// A monitor for portfolio solver runs.
///
/// This trait is intentionally solver-agnostic. It observes the portfolio
/// at a high level: which solvers start and finish, when the global best
/// improves, and whether the portfolio should continue or terminate.
pub trait PortfolioMonitor<T>
where
    T: SolverNumeric,
{
    /// Returns the name of this monitor.
    fn name(&self) -> &str;

    /// Called when the portfolio run begins.
    fn on_start(&mut self, model: &Model<T>);

    /// Called when the portfolio run ends.
    fn on_end(&mut self, best_solution: SolutionView<'_, T>);

    /// Called when a solver in the portfolio starts.
    fn on_solver_started(&mut self, solver_name: &str);

    /// Called when a solver in the portfolio finishes.
    fn on_solver_finished(&mut self, solver_name: &str, solution: SolutionView<'_, T>);

    /// Called when the global best solution is improved.
    fn on_best_solution_updated(
        &mut self,
        solver_name: &str,
        previous_best: SolutionView<'_, T>,
        new_best: SolutionView<'_, T>,
    );

    /// Determines the command for the portfolio.
    fn portfolio_command(&mut self) -> PortfolioCommand {
        PortfolioCommand::Continue
    }
}

impl<T> std::fmt::Debug for dyn PortfolioMonitor<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PortfolioMonitor {{ name: {} }}", self.name())
    }
}

impl<T> std::fmt::Display for dyn PortfolioMonitor<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PortfolioMonitor: {}", self.name())
    }
}
