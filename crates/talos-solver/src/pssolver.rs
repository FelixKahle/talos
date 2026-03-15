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

//! Parallel portfolio solver.
//!
//! [`ParallelPortfolioSolver`] manages a set of [`PortfolioSolver`]s, a
//! shared [`GlobalOracle`], and a [`PortfolioMonitor`]. On [`solve`] it
//! spawns one scoped thread per solver, lets them run concurrently while
//! sharing improving solutions through the oracle, and returns the overall
//! best solution found.

use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::Solution};
use talos_search::{oracle::GlobalOracle, portfolio::PortfolioSolver};

/// A portfolio runner that executes multiple [`PortfolioSolver`]s in
/// parallel using scoped threads.
///
/// Each solver runs in its own thread and shares solutions with the
/// others through a [`GlobalOracle`]. A [`PortfolioMonitor`] is cloned
/// per thread so that each solver can independently query termination.
///
/// # Type Parameters
///
/// * `T` — numeric type (e.g. `i64`)
/// * `G` — shared oracle (`Sync` required for cross-thread access)
/// * `M` — portfolio monitor (`Clone + Send` — cloned once per solver)
pub struct ParallelPortfolioSolver<T, G>
where
    T: SolverNumeric,
    G: GlobalOracle<T>,
{
    solvers: Vec<Box<dyn PortfolioSolver<T, G> + Send>>,
    oracle: G,
}

impl<T, G> ParallelPortfolioSolver<T, G>
where
    T: SolverNumeric,
    G: GlobalOracle<T> + Sync,
{
    /// Creates an empty portfolio backed by the given oracle.
    pub fn new(oracle: G) -> Self {
        Self {
            solvers: Vec::new(),
            oracle,
        }
    }

    /// Creates an empty portfolio with pre-allocated capacity.
    pub fn with_capacity(oracle: G, capacity: usize) -> Self {
        Self {
            solvers: Vec::with_capacity(capacity),
            oracle,
        }
    }

    // ----------------------------------------------------------------
    // Builder / mutator methods
    // ----------------------------------------------------------------

    /// Adds a solver to the portfolio (consuming builder).
    pub fn with_solver(mut self, solver: impl PortfolioSolver<T, G> + Send + 'static) -> Self {
        self.solvers.push(Box::new(solver));
        self
    }

    /// Adds a pre-boxed solver to the portfolio (consuming builder).
    pub fn with_solver_boxed(mut self, solver: Box<dyn PortfolioSolver<T, G> + Send>) -> Self {
        self.solvers.push(solver);
        self
    }

    /// Adds a solver to the portfolio.
    pub fn add_solver(&mut self, solver: impl PortfolioSolver<T, G> + Send + 'static) {
        self.solvers.push(Box::new(solver));
    }

    /// Adds a pre-boxed solver to the portfolio.
    pub fn add_solver_boxed(&mut self, solver: Box<dyn PortfolioSolver<T, G> + Send>) {
        self.solvers.push(solver);
    }

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------

    /// Returns the number of solvers in the portfolio.
    pub fn num_solvers(&self) -> usize {
        self.solvers.len()
    }

    /// Returns a reference to the shared oracle.
    pub fn oracle(&self) -> &G {
        &self.oracle
    }

    // ----------------------------------------------------------------
    // Execution
    // ----------------------------------------------------------------

    /// Runs all solvers in parallel and returns the best solution found.
    ///
    /// The monitor is cloned once per solver so each thread can
    /// independently check for termination. After all threads join,
    /// the oracle is also consulted — it may hold a solution that is
    /// better than any individual solver's return value due to
    /// cross-pollination.
    ///
    /// # Panics
    ///
    /// * Panics if the portfolio contains no solvers.
    /// * Panics if **all** solvers panic (individual panics are ignored
    ///   as long as at least one solver succeeds).
    pub fn solve(
        &mut self,
        model: &Model<T>,
        time_limit: std::time::Duration,
        non_improving: std::time::Duration,
    ) -> Solution<T> {
        assert!(!self.solvers.is_empty(), "portfolio has no solvers");

        let oracle = &self.oracle;

        let mut results: Vec<Solution<T>> = std::thread::scope(|s| {
            let handles: Vec<_> = self
                .solvers
                .iter_mut()
                .enumerate()
                .map(|(i, solver)| {
                    std::thread::Builder::new()
                        .name(format!("solver-{}-{}", i, solver.name()))
                        .spawn_scoped(s, move || {
                            solver.solve(model, oracle, time_limit, non_improving)
                        })
                        .expect("failed to spawn solver thread")
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        // The oracle may hold a better solution from cross-pollination.
        if let Some(oracle_best) = oracle.with_best(|s| s.clone()) {
            results.push(oracle_best);
        }

        results
            .into_iter()
            .min_by_key(|s| s.objective_value())
            .expect("all portfolio solvers panicked")
    }
}
