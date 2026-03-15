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

//! [`PortfolioSolver`] implementation that wraps the local search [`Engine`].
//!
//! [`LocalSearchSolver`] bundles an [`Engine`], a [`Metaheuristic`], a
//! [`LocalSearchOperator`], an evaluator function, and an initial-solution
//! builder into a single unit that satisfies the portfolio contract:
//! **model in, solution out**.

use crate::{
    engine::Engine, meta::metaheuristic::Metaheuristic, monitor::wrapper::PortfolioMonitorWrapper,
    operator::lsoperator::LocalSearchOperator, params::MutableLocalSearchParams,
};
use talos_core::utils::num::SolverNumeric;
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
    solution::Solution,
};
use talos_search::{
    monitor::psmonitor::PortfolioMonitor, oracle::GlobalOracle, portfolio::PortfolioSolver,
};

/// A [`PortfolioSolver`] backed by the local search [`Engine`].
///
/// # Type Parameters
///
/// * `T` — numeric type (e.g. `i64`)
/// * `H` — metaheuristic (SA, GLS, Tabu, …)
/// * `O` — neighborhood operator
/// * `E` — per-vessel cost evaluator (`Fn(&Model<T>, VesselIndex, BerthIndex, T) -> Option<T>`)
/// * `I` — initial-solution builder (`FnMut(&Model<T>) -> Solution<T>`)
pub struct LocalSearchSolver<T, H, O, E, I>
where
    T: SolverNumeric,
{
    name: String,
    engine: Engine<T>,
    metaheuristic: H,
    operator: O,
    evaluator: E,
    initial_solution_builder: I,
}

impl<T, H, O, E, I> LocalSearchSolver<T, H, O, E, I>
where
    T: SolverNumeric,
{
    /// Creates a new local-search solver with all required components.
    pub fn new(
        name: String,
        engine: Engine<T>,
        metaheuristic: H,
        operator: O,
        evaluator: E,
        initial_solution_builder: I,
    ) -> Self {
        Self {
            name,
            engine,
            metaheuristic,
            operator,
            evaluator,
            initial_solution_builder,
        }
    }
}

impl<T, H, O, E, I, G, M> PortfolioSolver<T, G, M> for LocalSearchSolver<T, H, O, E, I>
where
    T: SolverNumeric,
    H: Metaheuristic<T, G>,
    O: LocalSearchOperator<T>,
    E: Fn(&Model<T>, VesselIndex, BerthIndex, T) -> Option<T>,
    I: FnMut(&Model<T>) -> Solution<T>,
    G: GlobalOracle<T>,
    M: PortfolioMonitor<T>,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn solve(&mut self, model: &Model<T>, oracle: &G, monitor: M) -> Solution<T> {
        let initial = (self.initial_solution_builder)(model);
        let wrapper = PortfolioMonitorWrapper::new(self.name.clone(), monitor);

        let params = MutableLocalSearchParams {
            model,
            operator: &mut self.operator,
            metaheuristic: &mut self.metaheuristic,
            monitor: wrapper,
            oracle,
            berths: initial.berths(),
            start_times: initial.start_times(),
            objective_value: initial.objective_value(),
        };

        self.engine
            .run(params, &self.evaluator, |_| {})
            .into_solution()
    }
}
