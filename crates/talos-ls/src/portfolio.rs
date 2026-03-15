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

use crate::{
    engine::Engine,
    meta::metaheuristic::Metaheuristic,
    monitor::{lsmonitor::LocalSearchMonitor, nimpr::NoImprovementMonitor, time::TimeLimitMonitor},
    operator::lsoperator::LocalSearchOperator,
    params::MutableLocalSearchParams,
};
use talos_core::utils::num::SolverNumeric;
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
    solution::Solution,
};
use talos_search::{oracle::GlobalOracle, portfolio::PortfolioSolver};

struct PortfolioMonitor {
    time_limit: TimeLimitMonitor,
    non_improving: NoImprovementMonitor,
}

impl<T> LocalSearchMonitor<T> for PortfolioMonitor
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        "PortfolioMonitor"
    }

    fn on_start(
        &mut self,
        model: &Model<T>,
        initial_solution: talos_model::solution::SolutionView<'_, T>,
    ) {
        self.time_limit.on_start(model, initial_solution);
        self.non_improving.on_start(model, initial_solution);
    }

    fn on_end(
        &mut self,
        best_solution: talos_model::solution::SolutionView<'_, T>,
        statistics: &crate::stats::LocalSearchStatistics,
    ) {
        self.time_limit.on_end(best_solution, statistics);
        self.non_improving.on_end(best_solution, statistics);
    }

    fn on_iteration(
        &mut self,
        best_solution: talos_model::solution::SolutionView<'_, T>,
        accepted_solution: talos_model::solution::SolutionView<'_, T>,
        buffered_solution: Option<talos_model::solution::SolutionView<'_, T>>,
        statistics: &crate::stats::LocalSearchStatistics,
    ) {
        self.time_limit.on_iteration(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
        self.non_improving.on_iteration(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
    }

    fn on_candidate_generated(
        &mut self,
        best_solution: talos_model::solution::SolutionView<'_, T>,
        accepted_solution: talos_model::solution::SolutionView<'_, T>,
        buffered_solution: Option<talos_model::solution::SolutionView<'_, T>>,
        candidate_objective: T, // The candidate solution is only ever partially constructed, so we only pass the objective value here.
        statistics: &crate::stats::LocalSearchStatistics,
    ) {
        self.time_limit.on_candidate_generated(
            best_solution,
            accepted_solution,
            buffered_solution,
            candidate_objective,
            statistics,
        );
        self.non_improving.on_candidate_generated(
            best_solution,
            accepted_solution,
            buffered_solution,
            candidate_objective,
            statistics,
        );
    }

    fn on_solution_buffered(
        &mut self,
        best_solution: talos_model::solution::SolutionView<'_, T>,
        accepted_solution: talos_model::solution::SolutionView<'_, T>,
        buffered_solution: talos_model::solution::SolutionView<'_, T>,
        statistics: &crate::stats::LocalSearchStatistics,
    ) {
        self.time_limit.on_solution_buffered(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
        self.non_improving.on_solution_buffered(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
    }

    fn on_candidate_accepted(
        &mut self,
        best_solution: talos_model::solution::SolutionView<'_, T>,
        accepted_solution: talos_model::solution::SolutionView<'_, T>,
        buffered_solution: Option<talos_model::solution::SolutionView<'_, T>>,
        statistics: &crate::stats::LocalSearchStatistics,
    ) {
        self.time_limit.on_candidate_accepted(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
        self.non_improving.on_candidate_accepted(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
    }

    fn on_buffered_solution_accepted(
        &mut self,
        best_solution: talos_model::solution::SolutionView<'_, T>,
        accepted_solution: talos_model::solution::SolutionView<'_, T>,
        statistics: &crate::stats::LocalSearchStatistics,
    ) {
        self.time_limit
            .on_buffered_solution_accepted(best_solution, accepted_solution, statistics);
        self.non_improving.on_buffered_solution_accepted(
            best_solution,
            accepted_solution,
            statistics,
        );
    }

    fn on_candidate_rejected(
        &mut self,
        best_solution: talos_model::solution::SolutionView<'_, T>,
        accepted_solution: talos_model::solution::SolutionView<'_, T>,
        buffered_solution: Option<talos_model::solution::SolutionView<'_, T>>,
        rejected_objective: T,
        statistics: &crate::stats::LocalSearchStatistics,
    ) {
        self.time_limit.on_candidate_rejected(
            best_solution,
            accepted_solution,
            buffered_solution,
            rejected_objective,
            statistics,
        );
        self.non_improving.on_candidate_rejected(
            best_solution,
            accepted_solution,
            buffered_solution,
            rejected_objective,
            statistics,
        );
    }

    fn on_neighborhood_exhausted(
        &mut self,
        best_solution: talos_model::solution::SolutionView<'_, T>,
        accepted_solution: talos_model::solution::SolutionView<'_, T>,
        buffered_solution: Option<talos_model::solution::SolutionView<'_, T>>,
        statistics: &crate::stats::LocalSearchStatistics,
    ) {
        self.time_limit.on_neighborhood_exhausted(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
        self.non_improving.on_neighborhood_exhausted(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
    }

    fn on_best_solution_updated(
        &mut self,
        previous_best_solution: talos_model::solution::SolutionView<'_, T>,
        accepted_solution: talos_model::solution::SolutionView<'_, T>,
        buffered_solution: Option<talos_model::solution::SolutionView<'_, T>>,
        new_best_solution: talos_model::solution::SolutionView<'_, T>,
        statistics: &crate::stats::LocalSearchStatistics,
    ) {
        self.time_limit.on_best_solution_updated(
            previous_best_solution,
            accepted_solution,
            buffered_solution,
            new_best_solution,
            statistics,
        );
        self.non_improving.on_best_solution_updated(
            previous_best_solution,
            accepted_solution,
            buffered_solution,
            new_best_solution,
            statistics,
        );
    }

    fn on_candidate_infeasible(
        &mut self,
        best_solution: talos_model::solution::SolutionView<'_, T>,
        accepted_solution: talos_model::solution::SolutionView<'_, T>,
        buffered_solution: Option<talos_model::solution::SolutionView<'_, T>>,
        statistics: &crate::stats::LocalSearchStatistics,
    ) {
        self.time_limit.on_candidate_infeasible(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
        self.non_improving.on_candidate_infeasible(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
    }

    fn search_command(
        &mut self,
        best_solution: talos_model::solution::SolutionView<'_, T>,
        accepted_solution: talos_model::solution::SolutionView<'_, T>,
        buffered_solution: Option<talos_model::solution::SolutionView<'_, T>>,
        statistics: &crate::stats::LocalSearchStatistics,
    ) -> crate::exec::SearchCommand {
        let time_cmd = self.time_limit.search_command(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        );
        if time_cmd != crate::exec::SearchCommand::Continue {
            return time_cmd;
        }
        self.non_improving.search_command(
            best_solution,
            accepted_solution,
            buffered_solution,
            statistics,
        )
    }
}

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

impl<T, H, O, E, I, G> PortfolioSolver<T, G> for LocalSearchSolver<T, H, O, E, I>
where
    T: SolverNumeric,
    H: Metaheuristic<T, G>,
    O: LocalSearchOperator<T>,
    E: Fn(&Model<T>, VesselIndex, BerthIndex, T) -> Option<T>,
    I: FnMut(&Model<T>) -> Solution<T>,
    G: GlobalOracle<T>,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn solve(
        &mut self,
        model: &Model<T>,
        oracle: &G,
        time_limit: std::time::Duration,
        non_improving_limit: std::time::Duration,
    ) -> Solution<T> {
        let initial = (self.initial_solution_builder)(model);
        let monitor = PortfolioMonitor {
            time_limit: TimeLimitMonitor::new(time_limit),
            non_improving: NoImprovementMonitor::with_duration_patience(non_improving_limit),
        };

        let params = MutableLocalSearchParams {
            model,
            operator: &mut self.operator,
            metaheuristic: &mut self.metaheuristic,
            monitor,
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
