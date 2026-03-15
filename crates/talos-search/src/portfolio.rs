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

use crate::oracle::GlobalOracle;
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::Solution};

/// A solver that produces a solution for a given model.
///
/// This trait is intentionally engine-agnostic. Implementations may use
/// local search, constructive heuristics, exact methods, or any other
/// strategy. The only contract is: **model in, solution out**.
///
/// A [`GlobalOracle`] is provided so that solvers running in a portfolio
/// can share and receive improving solutions from other threads.
pub trait PortfolioSolver<T: SolverNumeric, G: GlobalOracle<T>> {
    /// Returns the name of this solver (for logging / identification).
    fn name(&self) -> &str;

    /// Solves the given model and returns the best solution found.
    fn solve(
        &mut self,
        model: &Model<T>,
        oracle: &G,
        time_limit: std::time::Duration,
        non_improving_limit: std::time::Duration,
    ) -> Solution<T>;
}
