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

use rand::SeedableRng;
use rand::rngs::StdRng;
use std::ffi::c_void;
use std::time::Duration;
use talos_ls::engine::Engine;
use talos_ls::eval::calculate_weighted_turnaround_time_unchecked;
use talos_ls::exec::{SearchCommand, TerminationReason};
use talos_ls::meta::gd::GreedyDescent;
use talos_ls::meta::gls::{
    FixedLambda, GuidedLocalSearch, MaxUtilityPenalization, PenalizationTrigger, ReactiveLambda,
    UniformCost,
};
use talos_ls::meta::metaheuristic::{
    AcceptanceOutcome, Metaheuristic, NeighborhoodExhaustionOutcome,
};
use talos_ls::meta::sa::{
    CoolingTrigger, GeometricCooling, LinearCooling, MetropolisCriterion, SimulatedAnnealing,
};
use talos_ls::meta::tabu::{FixedTenure, RandomTenure, SelectionStrategy, TabuSearch};
use talos_ls::meta::tie::KeepFirst;
use talos_ls::monitor::composite::CompositeLocalSearchMonitor;
use talos_ls::monitor::cycle::CycleLimitMonitor;
use talos_ls::monitor::iteration::IterationLimitMonitor;
use talos_ls::monitor::nimpr::NoImprovementMonitor;
use talos_ls::monitor::solution::SolutionLimitMonitor;
use talos_ls::monitor::time::TimeLimitMonitor;
use talos_ls::operator::composite::{
    MultiArmedBanditCompoundOperator, RandomCompoundOperator, RoundRobinCompoundOperator,
};
use talos_ls::operator::filter::{
    inter_berth_shift_filter_unchecked, inter_berth_swap_filter_unchecked,
    intra_berth_shift_filter_unchecked, intra_berth_swap_filter_unchecked,
};
use talos_ls::operator::lsoperator::LocalSearchOperator;
use talos_ls::operator::shift::{InterBerthShiftOperator, IntraBerthShiftOperator};
use talos_ls::operator::swap::{InterBerthSwapOperator, IntraBerthSwapOperator};
use talos_ls::params::LocalSearchParams;
use talos_ls::sgraph::{ScheduleGraph, ScheduleGraphDiff};
use talos_ls::stats::LocalSearchStatistics;
use talos_model::index::{BerthIndex, VesselIndex};
use talos_model::model::Model;
use talos_model::solution::{Solution, SolutionView};

// ----------------------------------------------------------------
// C-compatible output types
// ----------------------------------------------------------------

/// C-compatible mirror of `TerminationReason`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiTerminationReason {
    TimeLimitReached = 0,
    SolutionLimitReached = 1,
    IterationLimitReached = 2,
    CycleLimitReached = 3,
    MaxNonImprovingIterations = 4,
    MaxNonImprovingCycles = 5,
    MaxNonImprovingTime = 6,
    TargetObjectiveReached = 7,
    NeighborhoodExhausted = 8,
    Interrupted = 9,
    Aborted = 10,
}

impl From<TerminationReason> for FfiTerminationReason {
    fn from(r: TerminationReason) -> Self {
        match r {
            TerminationReason::TimeLimitReached => Self::TimeLimitReached,
            TerminationReason::SolutionLimitReached => Self::SolutionLimitReached,
            TerminationReason::IterationLimitReached => Self::IterationLimitReached,
            TerminationReason::CycleLimitReached => Self::CycleLimitReached,
            TerminationReason::MaxNonImprovingIterations => Self::MaxNonImprovingIterations,
            TerminationReason::MaxNonImprovingCycles => Self::MaxNonImprovingCycles,
            TerminationReason::MaxNonImprovingTime => Self::MaxNonImprovingTime,
            TerminationReason::TargetObjectiveReached => Self::TargetObjectiveReached,
            TerminationReason::NeighborhoodExhausted => Self::NeighborhoodExhausted,
            TerminationReason::Interrupted => Self::Interrupted,
            TerminationReason::Aborted => Self::Aborted,
        }
    }
}

/// C-compatible mirror of `LocalSearchStatistics`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiLocalSearchStatistics {
    pub iterations: u64,
    pub cycles: u64,
    pub total_solutions: u64,
    pub accepted_solutions: u64,
    pub infeasible_moves: u64,
    /// Total elapsed time in nanoseconds.
    pub time_total_nanos: u64,
}

impl From<&LocalSearchStatistics> for FfiLocalSearchStatistics {
    fn from(s: &LocalSearchStatistics) -> Self {
        Self {
            iterations: s.iterations,
            cycles: s.cycles,
            total_solutions: s.total_solutions,
            accepted_solutions: s.accepted_solutions,
            infeasible_moves: s.infeasible_moves,
            time_total_nanos: s.time_total.as_nanos() as u64,
        }
    }
}

/// C-compatible result of `Engine::run`.
///
/// The `solution` pointer (if non-null) must be freed with `talos_solution_free`.
#[repr(C)]
pub struct FfiLocalSearchOutcome {
    /// Best solution found.
    pub solution: *mut Solution<i64>,
    pub termination_reason: FfiTerminationReason,
    pub stats: FfiLocalSearchStatistics,
}

// ----------------------------------------------------------------
// Metaheuristic configuration
// ----------------------------------------------------------------

/// Internal enum wrapping all supported metaheuristic configurations.
///
/// Implements `Metaheuristic<i64>` by delegating to the active variant.
enum MetaInner {
    TabuFixed(TabuSearch<FixedTenure>),
    TabuRandom(TabuSearch<RandomTenure<StdRng>>),
    SaGeometric(SimulatedAnnealing<StdRng, GeometricCooling, MetropolisCriterion>),
    SaLinear(SimulatedAnnealing<StdRng, LinearCooling, MetropolisCriterion>),
    GlsFixed(GuidedLocalSearch<MaxUtilityPenalization, UniformCost, FixedLambda>),
    GlsReactive(GuidedLocalSearch<MaxUtilityPenalization, UniformCost, ReactiveLambda>),
    GreedyDescent(GreedyDescent<KeepFirst>),
}

macro_rules! delegate_meta {
    ($self:expr, $method:ident ( $($arg:expr),* $(,)? )) => {
        match $self {
            MetaInner::TabuFixed(m) => m.$method($($arg),*),
            MetaInner::TabuRandom(m) => m.$method($($arg),*),
            MetaInner::SaGeometric(m) => m.$method($($arg),*),
            MetaInner::SaLinear(m) => m.$method($($arg),*),
            MetaInner::GlsFixed(m) => m.$method($($arg),*),
            MetaInner::GlsReactive(m) => m.$method($($arg),*),
            MetaInner::GreedyDescent(m) => m.$method($($arg),*),
        }
    };
}

impl Metaheuristic<i64> for MetaInner {
    fn name(&self) -> &str {
        match self {
            MetaInner::TabuFixed(_) => "TabuSearch",
            MetaInner::TabuRandom(_) => "TabuSearch",
            MetaInner::SaGeometric(_) => "SimulatedAnnealing",
            MetaInner::SaLinear(_) => "SimulatedAnnealing",
            MetaInner::GlsFixed(_) => "GuidedLocalSearch",
            MetaInner::GlsReactive(_) => "GuidedLocalSearch",
            MetaInner::GreedyDescent(_) => "GreedyDescent",
        }
    }

    fn on_start(
        &mut self,
        model: &Model<i64>,
        initial_solution: SolutionView<i64>,
        graph: &ScheduleGraph,
    ) {
        delegate_meta!(self, on_start(model, initial_solution, graph))
    }

    fn on_end(
        &mut self,
        model: &Model<i64>,
        final_solution: SolutionView<i64>,
        graph: &ScheduleGraph,
    ) {
        delegate_meta!(self, on_end(model, final_solution, graph))
    }

    fn on_neighbourhood_exhausted(
        &mut self,
        model: &Model<i64>,
        best_solution: SolutionView<'_, i64>,
        accepted_solution: SolutionView<'_, i64>,
        buffered_solution: Option<SolutionView<'_, i64>>,
        graph: &ScheduleGraph,
    ) -> NeighborhoodExhaustionOutcome {
        delegate_meta!(
            self,
            on_neighbourhood_exhausted(
                model,
                best_solution,
                accepted_solution,
                buffered_solution,
                graph,
            )
        )
    }

    fn should_commit_buffered(
        &mut self,
        model: &Model<i64>,
        best_solution: SolutionView<'_, i64>,
        accepted_solution: SolutionView<'_, i64>,
        buffered_solution: Option<SolutionView<'_, i64>>,
        layout: &ScheduleGraph,
        buffer_layout: &ScheduleGraph,
    ) -> bool {
        delegate_meta!(
            self,
            should_commit_buffered(
                model,
                best_solution,
                accepted_solution,
                buffered_solution,
                layout,
                buffer_layout,
            )
        )
    }

    fn search_command(
        &mut self,
        iteration: u64,
        model: &Model<i64>,
        best_solution: SolutionView<'_, i64>,
        accepted_solution: SolutionView<'_, i64>,
        buffered_solution: Option<SolutionView<'_, i64>>,
    ) -> SearchCommand {
        delegate_meta!(
            self,
            search_command(
                iteration,
                model,
                best_solution,
                accepted_solution,
                buffered_solution,
            )
        )
    }

    fn decide_fate(
        &mut self,
        model: &Model<i64>,
        best_solution: SolutionView<'_, i64>,
        accepted_solution: SolutionView<'_, i64>,
        buffered_solution: Option<SolutionView<'_, i64>>,
        candidate_objective: i64,
        graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    ) -> AcceptanceOutcome {
        delegate_meta!(
            self,
            decide_fate(
                model,
                best_solution,
                accepted_solution,
                buffered_solution,
                candidate_objective,
                graph,
                graph_diff,
            )
        )
    }

    fn on_accept(
        &mut self,
        model: &Model<i64>,
        best_solution: SolutionView<'_, i64>,
        new_accepted_solution: SolutionView<'_, i64>,
        buffered_solution: Option<SolutionView<'_, i64>>,
        graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    ) {
        delegate_meta!(
            self,
            on_accept(
                model,
                best_solution,
                new_accepted_solution,
                buffered_solution,
                graph,
                graph_diff,
            )
        )
    }

    fn on_reject(
        &mut self,
        model: &Model<i64>,
        best_solution: SolutionView<'_, i64>,
        new_accepted_solution: SolutionView<'_, i64>,
        buffered_solution: Option<SolutionView<'_, i64>>,
        candidate_objective: i64,
        graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    ) {
        delegate_meta!(
            self,
            on_reject(
                model,
                best_solution,
                new_accepted_solution,
                buffered_solution,
                candidate_objective,
                graph,
                graph_diff,
            )
        )
    }

    fn on_new_best(
        &mut self,
        model: &Model<i64>,
        new_best: SolutionView<i64>,
        graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    ) {
        delegate_meta!(self, on_new_best(model, new_best, graph, graph_diff))
    }

    fn on_iteration(
        &mut self,
        iteration: u64,
        model: &Model<i64>,
        best_solution: SolutionView<'_, i64>,
        new_accepted_solution: SolutionView<'_, i64>,
        buffered_solution: Option<SolutionView<'_, i64>>,
        graph: &ScheduleGraph,
    ) {
        delegate_meta!(
            self,
            on_iteration(
                iteration,
                model,
                best_solution,
                new_accepted_solution,
                buffered_solution,
                graph,
            )
        )
    }
}

/// Which metaheuristic algorithm to use.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiMetaheuristic {
    TabuSearch = 0,
    SimulatedAnnealing = 1,
    GuidedLocalSearch = 2,
    GreedyDescent = 3,
}

/// SA cooling schedule type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiCoolingSchedule {
    Geometric = 0,
    GeometricHeuristic = 1,
    Linear = 2,
}

/// SA cooling trigger.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiCoolingTrigger {
    Iteration = 0,
    Cycle = 1,
    Acceptance = 2,
}

/// GLS penalization trigger.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiPenalizationTrigger {
    OnExhaustion = 0,
    AfterNonImprovements = 1,
    AfterMoves = 2,
}

/// Flat configuration for the metaheuristic, passed by value.
///
/// Set `metaheuristic` to select the algorithm. Only the fields prefixed
/// with the corresponding algorithm name are read:
///
/// * **TabuSearch** — `tabu_tenure`, `tabu_max_tenure`, `tabu_selection`,
///   `num_vessels`, `num_berths`, `seed`.
///   If `tabu_max_tenure > tabu_tenure`, random tenure in
///   `[tabu_tenure, tabu_max_tenure]` is used; otherwise fixed tenure.
///
/// * **SimulatedAnnealing** — `sa_schedule`, `sa_initial_temperature`,
///   `sa_geometric_alpha`, `sa_linear_decrement`, `sa_min_temperature`,
///   `sa_reheat_factor`, `sa_cooling_trigger`, `seed`.
///   With `GeometricHeuristic`: `sa_heuristic_objective`,
///   `sa_heuristic_acceptance_prob`, `sa_heuristic_sensitivity`,
///   `sa_heuristic_cooling_rate` are used instead of `sa_initial_temperature`
///   and `sa_geometric_alpha`.
///
/// * **GuidedLocalSearch** — `gls_lambda`, `gls_reactive`,
///   `gls_penalization_trigger`, `gls_trigger_n`, `num_vessels`, `num_berths`.
///   If `gls_reactive != 0`: `gls_growth_factor`, `gls_decay_factor`,
///   `gls_min_lambda`, `gls_max_lambda` are also used.
///
/// * **GreedyDescent** — `gd_selection`.
///   0 = FirstImprovement (default), 1 = BestImprovement.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMetaheuristicConfig {
    pub metaheuristic: FfiMetaheuristic,
    pub seed: u64,
    pub num_vessels: usize,
    pub num_berths: usize,

    // Tabu Search
    pub tabu_tenure: u64,
    /// If `tabu_max_tenure > tabu_tenure`, random tenure is used.
    pub tabu_max_tenure: u64,
    /// 0 = BestImprovement, 1 = FirstImprovement.
    pub tabu_selection: i32,

    // Simulated Annealing
    pub sa_schedule: FfiCoolingSchedule,
    pub sa_initial_temperature: f64,
    pub sa_geometric_alpha: f64,
    pub sa_linear_decrement: f64,
    pub sa_min_temperature: f64,
    /// Reheat multiplier on neighbourhood exhaustion. `<= 1.0` disables reheating.
    pub sa_reheat_factor: f64,
    pub sa_cooling_trigger: FfiCoolingTrigger,
    // Heuristic sub-params (GeometricHeuristic only)
    pub sa_heuristic_objective: f64,
    pub sa_heuristic_acceptance_prob: f64,
    pub sa_heuristic_sensitivity: f64,
    pub sa_heuristic_cooling_rate: f64,

    // Guided Local Search
    pub gls_lambda: f64,
    /// Non-zero to use reactive lambda.
    pub gls_reactive: i32,
    pub gls_penalization_trigger: FfiPenalizationTrigger,
    pub gls_trigger_n: u64,
    pub gls_growth_factor: f64,
    pub gls_decay_factor: f64,
    pub gls_min_lambda: f64,
    pub gls_max_lambda: f64,

    // Greedy Descent
    /// 0 = FirstImprovement (default), 1 = BestImprovement.
    pub gd_selection: i32,
}

impl FfiMetaheuristicConfig {
    fn build_metaheuristic(&self) -> MetaInner {
        match self.metaheuristic {
            FfiMetaheuristic::TabuSearch => {
                let sel = match self.tabu_selection {
                    1 => SelectionStrategy::FirstImprovement,
                    _ => SelectionStrategy::BestImprovement,
                };
                if self.tabu_max_tenure > self.tabu_tenure {
                    let rng = StdRng::seed_from_u64(self.seed);
                    let ts = TabuSearch::new(
                        RandomTenure::new(self.tabu_tenure, self.tabu_max_tenure, rng),
                        self.num_vessels,
                        self.num_berths,
                    )
                    .with_selection(sel);
                    MetaInner::TabuRandom(ts)
                } else {
                    let ts = TabuSearch::new(
                        FixedTenure::new(self.tabu_tenure),
                        self.num_vessels,
                        self.num_berths,
                    )
                    .with_selection(sel);
                    MetaInner::TabuFixed(ts)
                }
            }
            FfiMetaheuristic::SimulatedAnnealing => {
                let trigger = match self.sa_cooling_trigger {
                    FfiCoolingTrigger::Cycle => CoolingTrigger::Cycle,
                    FfiCoolingTrigger::Acceptance => CoolingTrigger::Acceptance,
                    _ => CoolingTrigger::Iteration,
                };
                let rng = StdRng::seed_from_u64(self.seed);
                match self.sa_schedule {
                    FfiCoolingSchedule::Geometric => {
                        let cooling = GeometricCooling::new(
                            self.sa_initial_temperature,
                            self.sa_geometric_alpha,
                            self.sa_min_temperature,
                        );
                        let mut sa =
                            SimulatedAnnealing::new(cooling, rng).with_cooling_trigger(trigger);
                        if self.sa_reheat_factor > 1.0 {
                            sa = sa.with_reheat(self.sa_reheat_factor);
                        }
                        MetaInner::SaGeometric(sa)
                    }
                    FfiCoolingSchedule::GeometricHeuristic => {
                        let cooling =
                            SimulatedAnnealing::<StdRng, GeometricCooling>::heuristic_geometric_params(
                                self.sa_heuristic_objective,
                                self.sa_heuristic_acceptance_prob,
                                self.sa_heuristic_sensitivity,
                                self.sa_heuristic_cooling_rate,
                            );
                        let mut sa =
                            SimulatedAnnealing::new(cooling, rng).with_cooling_trigger(trigger);
                        if self.sa_reheat_factor > 1.0 {
                            sa = sa.with_reheat(self.sa_reheat_factor);
                        }
                        MetaInner::SaGeometric(sa)
                    }
                    FfiCoolingSchedule::Linear => {
                        let cooling = LinearCooling::new(
                            self.sa_initial_temperature,
                            self.sa_linear_decrement,
                            self.sa_min_temperature,
                        );
                        let mut sa =
                            SimulatedAnnealing::new(cooling, rng).with_cooling_trigger(trigger);
                        if self.sa_reheat_factor > 1.0 {
                            sa = sa.with_reheat(self.sa_reheat_factor);
                        }
                        MetaInner::SaLinear(sa)
                    }
                }
            }
            FfiMetaheuristic::GreedyDescent => {
                let sel = match self.gd_selection {
                    1 => SelectionStrategy::BestImprovement,
                    _ => SelectionStrategy::FirstImprovement,
                };
                MetaInner::GreedyDescent(GreedyDescent::new().with_selection(sel))
            }
            FfiMetaheuristic::GuidedLocalSearch => {
                let trigger = match self.gls_penalization_trigger {
                    FfiPenalizationTrigger::AfterNonImprovements => {
                        PenalizationTrigger::AfterNonImprovements(self.gls_trigger_n)
                    }
                    FfiPenalizationTrigger::AfterMoves => {
                        PenalizationTrigger::AfterMoves(self.gls_trigger_n)
                    }
                    _ => PenalizationTrigger::OnExhaustion,
                };
                if self.gls_reactive != 0 {
                    let reactive = ReactiveLambda::new(
                        self.gls_lambda,
                        self.gls_growth_factor,
                        self.gls_decay_factor,
                        self.gls_min_lambda,
                        self.gls_max_lambda,
                    );
                    let gls =
                        GuidedLocalSearch::new(self.gls_lambda, self.num_vessels, self.num_berths)
                            .with_lambda(reactive)
                            .with_trigger(trigger);
                    MetaInner::GlsReactive(gls)
                } else {
                    let gls =
                        GuidedLocalSearch::new(self.gls_lambda, self.num_vessels, self.num_berths)
                            .with_trigger(trigger);
                    MetaInner::GlsFixed(gls)
                }
            }
        }
    }
}

// ----------------------------------------------------------------
// Operator configuration
// ----------------------------------------------------------------

/// Compound operator selection strategy.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiCompoundStrategy {
    RoundRobin = 0,
    Random = 1,
    MultiArmedBandit = 2,
}

/// Flat configuration for the compound operator, passed by value.
///
/// Set the `use_*` fields to non-zero to enable the corresponding operator.
/// At least one operator must be enabled or the engine run will return an error.
///
/// * `seed` — used by the `Random` compound strategy.
/// * `bandit_memory_coeff` / `bandit_exploration_coeff` — used only with
///   `MultiArmedBandit`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiOperatorConfig {
    pub strategy: FfiCompoundStrategy,
    pub use_intra_berth_swap: i32,
    pub use_inter_berth_swap: i32,
    pub use_intra_berth_shift: i32,
    pub use_inter_berth_shift: i32,
    pub seed: u64,
    pub bandit_memory_coeff: f64,
    pub bandit_exploration_coeff: f64,
}

impl FfiOperatorConfig {
    fn build_operators(&self) -> Vec<Box<dyn LocalSearchOperator<i64>>> {
        let mut ops: Vec<Box<dyn LocalSearchOperator<i64>>> = Vec::new();
        if self.use_intra_berth_swap != 0 {
            ops.push(Box::new(IntraBerthSwapOperator::new(
                |v_a, v_b, sol, graph, model| unsafe {
                    intra_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
                },
            )));
        }
        if self.use_inter_berth_swap != 0 {
            ops.push(Box::new(InterBerthSwapOperator::new(
                |v_a, v_b, sol, graph, model| unsafe {
                    inter_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
                },
            )));
        }
        if self.use_intra_berth_shift != 0 {
            ops.push(Box::new(IntraBerthShiftOperator::new(
                |v, anchor, sol, graph, model| unsafe {
                    intra_berth_shift_filter_unchecked(v, anchor, sol, graph, model)
                },
            )));
        }
        if self.use_inter_berth_shift != 0 {
            ops.push(Box::new(InterBerthShiftOperator::new(
                |v, anchor, sol, graph, model| unsafe {
                    inter_berth_shift_filter_unchecked(v, anchor, sol, graph, model)
                },
            )));
        }
        ops
    }
}

// ================================================================
// Monitor configuration
// ----------------------------------------------------------------

/// Flat configuration for termination conditions, passed by value.
///
/// Set a limit field to a positive value to enable that termination condition.
/// A value of `0` disables the condition. At least one condition should be
/// enabled, otherwise the search will never terminate.
///
/// * `time_limit_millis` — maximum wall-clock time in milliseconds.
/// * `iteration_limit` — maximum number of iterations.
/// * `solution_limit` — maximum number of accepted solutions.
/// * `cycle_limit` — maximum number of cycles.
/// * `no_improvement_iterations` — stop after this many iterations without improvement.
/// * `no_improvement_cycles` — stop after this many cycles without improvement.
/// * `no_improvement_time_millis` — stop after this many milliseconds without improvement.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMonitorConfig {
    pub time_limit_millis: u64,
    pub iteration_limit: u64,
    pub solution_limit: u64,
    pub cycle_limit: u64,
    pub no_improvement_iterations: u64,
    pub no_improvement_cycles: u64,
    pub no_improvement_time_millis: u64,
}

impl FfiMonitorConfig {
    fn build_monitor(&self) -> CompositeLocalSearchMonitor<'static, i64> {
        let mut mon = CompositeLocalSearchMonitor::new();
        if self.time_limit_millis > 0 {
            mon.add_monitor(TimeLimitMonitor::new(Duration::from_millis(
                self.time_limit_millis,
            )));
        }
        if self.iteration_limit > 0 {
            mon.add_monitor(IterationLimitMonitor::new(self.iteration_limit));
        }
        if self.solution_limit > 0 {
            mon.add_monitor(SolutionLimitMonitor::new(self.solution_limit));
        }
        if self.cycle_limit > 0 {
            mon.add_monitor(CycleLimitMonitor::new(self.cycle_limit));
        }
        if self.no_improvement_iterations > 0 {
            mon.add_monitor(NoImprovementMonitor::with_iteration_patience(
                self.no_improvement_iterations,
            ));
        }
        if self.no_improvement_cycles > 0 {
            mon.add_monitor(NoImprovementMonitor::with_cycle_patience(
                self.no_improvement_cycles,
            ));
        }
        if self.no_improvement_time_millis > 0 {
            mon.add_monitor(NoImprovementMonitor::with_duration_patience(
                Duration::from_millis(self.no_improvement_time_millis),
            ));
        }
        mon
    }
}

// ================================================================
// Engine lifecycle
// ================================================================

/// Creates a new search engine pre-allocated for the given problem dimensions.
///
/// # Safety
///
/// The returned pointer must eventually be freed with `talos_engine_free`.
#[unsafe(no_mangle)]
pub extern "C" fn talos_engine_new(num_vessels: usize, num_berths: usize) -> *mut Engine<i64> {
    Box::into_raw(Box::new(Engine::new(num_vessels, num_berths)))
}

/// Frees an engine previously created by `talos_engine_new`.
///
/// # Safety
///
/// * `engine` must be a valid pointer from `talos_engine_new`.
/// * Must not be called while `talos_engine_run` is executing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_engine_free(engine: *mut Engine<i64>) {
    assert!(!engine.is_null(), "engine must not be null");
    drop(unsafe { Box::from_raw(engine) });
}

// ----------------------------------------------------------------
// Callback types
// ----------------------------------------------------------------

/// Optional custom evaluator function pointer (nullable in C).
///
/// Called for every vessel during solution decoding. Return a non-negative
/// cost value, or any negative value to signal infeasibility (the move will
/// be rejected).
///
/// Pass `NULL` to use the built-in weighted-turnaround-time evaluator.
///
/// Arguments: `(model, vessel_index, berth_index, start_time) -> cost`
pub type FfiEvaluatorFn = Option<
    unsafe extern "C" fn(
        model: *const Model<i64>,
        vessel: usize,
        berth: usize,
        start_time: i64,
    ) -> i64,
>;

/// Callback invoked whenever a new global-best solution is found.
///
/// The `solution_view` pointer is only valid for the duration of the callback
/// invocation. Use the `talos_solution_view_*` accessors to read it, or call
/// `talos_solution_view_to_owned` to deep-copy.
///
/// Arguments: `(solution_view, user_data)`
pub type FfiNewBestCallbackFn =
    unsafe extern "C" fn(solution_view: *const SolutionView<'static, i64>, user_data: *mut c_void);

// ----------------------------------------------------------------
// Run
// ----------------------------------------------------------------

/// Runs the local search engine.
///
/// All configuration is passed by value — no opaque handles, no ownership
/// transfer, no pointers to free.
///
/// Pass `NULL` for `evaluator` to use the built-in weighted-turnaround-time
/// evaluator, or provide a custom function pointer.
///
/// On validation failure (e.g. mismatched dimensions or no operators enabled),
/// the function panics.
///
/// # Safety
///
/// * `engine` must be a valid pointer from `talos_engine_new`.
/// * `model` must be a valid pointer from `talos_model_new`.
/// * `initial_solution` must be a valid pointer to a `SolutionView` (e.g.
///   from `talos_solution_view_new`). The view and its backing arrays must
///   remain valid for the duration of the call.
/// * All pointers must remain valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_engine_run(
    engine: *mut Engine<i64>,
    model: *const Model<i64>,
    meta_config: FfiMetaheuristicConfig,
    op_config: FfiOperatorConfig,
    mon_config: FfiMonitorConfig,
    initial_solution: *const SolutionView<'static, i64>,
    evaluator: FfiEvaluatorFn,
    callback: FfiNewBestCallbackFn,
    user_data: *mut c_void,
) -> FfiLocalSearchOutcome {
    assert!(!engine.is_null(), "engine must not be null");
    assert!(!model.is_null(), "model must not be null");
    assert!(
        !initial_solution.is_null(),
        "initial_solution must not be null"
    );

    let engine = unsafe { &mut *engine };
    let model = unsafe { &*model };
    let init_sol = unsafe { &*initial_solution };

    let mut meta_inner = meta_config.build_metaheuristic();

    let operators = op_config.build_operators();
    assert!(
        !operators.is_empty(),
        "at least one operator must be enabled"
    );

    let monitor = mon_config.build_monitor();

    let berths = init_sol.berths();
    let start_times = init_sol.start_times();
    let objective_value = init_sol.objective_value();

    // Evaluator: use the provided C function pointer, or fall back to the
    // built-in weighted turnaround time (direct call, no dynamic dispatch).
    let model_ptr = model as *const Model<i64>;
    let eval = move |m: &Model<i64>, v: VesselIndex, b: BerthIndex, t: i64| -> Option<i64> {
        match evaluator {
            Some(f) => {
                let result = unsafe { f(model_ptr, v.get(), b.get(), t) };
                if result < 0 { None } else { Some(result) }
            }
            None => unsafe { calculate_weighted_turnaround_time_unchecked(m, v, b, t) },
        }
    };

    // Build callback closure.
    let cb = move |view: SolutionView<'_, i64>| unsafe {
        // Transmute lifetime to 'static for the C function pointer signature.
        // Safe: the pointer is only valid for the duration of this call.
        let view_ptr: *const SolutionView<'static, i64> = std::ptr::from_ref(&view).cast();
        callback(view_ptr, user_data);
    };

    // Dispatch on compound strategy, construct the concrete operator,
    // validate params, and run the engine.
    macro_rules! run_with {
        ($op_expr:expr) => {{
            let mut concrete_op = $op_expr;
            match LocalSearchParams::new(
                model,
                &mut concrete_op,
                &mut meta_inner,
                monitor,
                berths,
                start_times,
                objective_value,
            ) {
                Ok(params) => engine.run(params.into(), eval, cb),
                Err(e) => panic!("LocalSearchParams validation failed: {e}"),
            }
        }};
    }

    let outcome = match op_config.strategy {
        FfiCompoundStrategy::Random => {
            run_with!(RandomCompoundOperator::new(
                operators,
                StdRng::seed_from_u64(op_config.seed)
            ))
        }
        FfiCompoundStrategy::RoundRobin => {
            run_with!(RoundRobinCompoundOperator::new(operators))
        }
        FfiCompoundStrategy::MultiArmedBandit => {
            run_with!(MultiArmedBanditCompoundOperator::new(
                operators,
                op_config.bandit_memory_coeff,
                op_config.bandit_exploration_coeff
            ))
        }
    };

    let (solution, reason, stats) = outcome.into_inner();
    FfiLocalSearchOutcome {
        solution: Box::into_raw(Box::new(solution)),
        termination_reason: FfiTerminationReason::from(reason),
        stats: FfiLocalSearchStatistics::from(&stats),
    }
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::model::talos_model_new;
    use crate::solution::{talos_solution_free, talos_solution_view_free, talos_solution_view_new};

    /// Builds a minimal 2-vessel / 2-berth model via the FFI.
    /// V0: arrival=0, deadline=100, weight=1, p(B0)=5, p(B1)=5
    /// V1: arrival=0, deadline=100, weight=1, p(B0)=10, p(B1)=10
    /// Both berths open [0, 200).
    fn build_test_model() -> *mut Model<i64> {
        use crate::model::FfiClosedOpenIntervalI64;

        let arrivals: [i64; 2] = [0, 0];
        let deadlines: [i64; 2] = [100, 100];
        let weights: [i64; 2] = [1, 1];
        // Row-major: [v0b0, v0b1, v1b0, v1b1]
        let processing_times: [i64; 4] = [5, 5, 10, 10];

        let intervals = [
            FfiClosedOpenIntervalI64 {
                start_inclusive: 0,
                end_exclusive: 200,
            },
            FfiClosedOpenIntervalI64 {
                start_inclusive: 0,
                end_exclusive: 200,
            },
        ];
        let b0_ptr = intervals[0..1].as_ptr();
        let b1_ptr = intervals[1..2].as_ptr();
        let interval_ptrs: [*const FfiClosedOpenIntervalI64; 2] = [b0_ptr, b1_ptr];
        let interval_lens: [usize; 2] = [1, 1];

        unsafe {
            talos_model_new(
                2,
                2,
                arrivals.as_ptr(),
                deadlines.as_ptr(),
                weights.as_ptr(),
                processing_times.as_ptr(),
                interval_ptrs.as_ptr(),
                interval_lens.as_ptr(),
            )
        }
    }

    // ---- Engine lifecycle ----

    /// No-op callback for tests.
    unsafe extern "C" fn noop_callback(
        _view: *const SolutionView<'static, i64>,
        _user_data: *mut c_void,
    ) {
    }

    #[test]
    fn test_engine_lifecycle() {
        let e = talos_engine_new(4, 2);
        assert!(!e.is_null());
        unsafe { talos_engine_free(e) };
    }

    // ---- Metaheuristic config ----

    /// Helper: returns a default-zeroed config for a given metaheuristic type.
    fn zeroed_meta_config(meta: FfiMetaheuristic) -> FfiMetaheuristicConfig {
        FfiMetaheuristicConfig {
            metaheuristic: meta,
            seed: 0,
            num_vessels: 4,
            num_berths: 2,
            tabu_tenure: 0,
            tabu_max_tenure: 0,
            tabu_selection: 0,
            sa_schedule: FfiCoolingSchedule::Geometric,
            sa_initial_temperature: 0.0,
            sa_geometric_alpha: 0.0,
            sa_linear_decrement: 0.0,
            sa_min_temperature: 0.0,
            sa_reheat_factor: 0.0,
            sa_cooling_trigger: FfiCoolingTrigger::Iteration,
            sa_heuristic_objective: 0.0,
            sa_heuristic_acceptance_prob: 0.0,
            sa_heuristic_sensitivity: 0.0,
            sa_heuristic_cooling_rate: 0.0,
            gls_lambda: 0.0,
            gls_reactive: 0,
            gls_penalization_trigger: FfiPenalizationTrigger::OnExhaustion,
            gls_trigger_n: 0,
            gls_growth_factor: 0.0,
            gls_decay_factor: 0.0,
            gls_min_lambda: 0.0,
            gls_max_lambda: 0.0,
            gd_selection: 0,
        }
    }

    #[test]
    fn test_meta_config_tabu_fixed() {
        let mut cfg = zeroed_meta_config(FfiMetaheuristic::TabuSearch);
        cfg.tabu_tenure = 10;
        let _m = cfg.build_metaheuristic();
    }

    #[test]
    fn test_meta_config_tabu_random() {
        let mut cfg = zeroed_meta_config(FfiMetaheuristic::TabuSearch);
        cfg.tabu_tenure = 5;
        cfg.tabu_max_tenure = 15;
        cfg.tabu_selection = 1;
        cfg.seed = 42;
        let _m = cfg.build_metaheuristic();
    }

    #[test]
    fn test_meta_config_sa_geometric() {
        let mut cfg = zeroed_meta_config(FfiMetaheuristic::SimulatedAnnealing);
        cfg.sa_schedule = FfiCoolingSchedule::Geometric;
        cfg.sa_initial_temperature = 100.0;
        cfg.sa_geometric_alpha = 0.99;
        cfg.sa_min_temperature = 0.01;
        cfg.seed = 42;
        let _m = cfg.build_metaheuristic();
    }

    #[test]
    fn test_meta_config_sa_linear() {
        let mut cfg = zeroed_meta_config(FfiMetaheuristic::SimulatedAnnealing);
        cfg.sa_schedule = FfiCoolingSchedule::Linear;
        cfg.sa_initial_temperature = 100.0;
        cfg.sa_linear_decrement = 0.1;
        cfg.sa_min_temperature = 0.01;
        cfg.sa_reheat_factor = 2.0;
        cfg.sa_cooling_trigger = FfiCoolingTrigger::Cycle;
        cfg.seed = 42;
        let _m = cfg.build_metaheuristic();
    }

    #[test]
    fn test_meta_config_gls_fixed() {
        let mut cfg = zeroed_meta_config(FfiMetaheuristic::GuidedLocalSearch);
        cfg.gls_lambda = 0.5;
        let _m = cfg.build_metaheuristic();
    }

    #[test]
    fn test_meta_config_gd_first_improvement() {
        let cfg = zeroed_meta_config(FfiMetaheuristic::GreedyDescent);
        let _m = cfg.build_metaheuristic();
    }

    #[test]
    fn test_meta_config_gd_best_improvement() {
        let mut cfg = zeroed_meta_config(FfiMetaheuristic::GreedyDescent);
        cfg.gd_selection = 1;
        let _m = cfg.build_metaheuristic();
    }

    #[test]
    fn test_meta_config_gls_reactive() {
        let mut cfg = zeroed_meta_config(FfiMetaheuristic::GuidedLocalSearch);
        cfg.gls_lambda = 0.5;
        cfg.gls_reactive = 1;
        cfg.gls_growth_factor = 1.2;
        cfg.gls_decay_factor = 0.8;
        cfg.gls_min_lambda = 0.1;
        cfg.gls_max_lambda = 10.0;
        let _m = cfg.build_metaheuristic();
    }

    // ---- Operator config ----

    #[test]
    fn test_operator_config_build_all() {
        let cfg = FfiOperatorConfig {
            strategy: FfiCompoundStrategy::Random,
            use_intra_berth_swap: 1,
            use_inter_berth_swap: 1,
            use_intra_berth_shift: 1,
            use_inter_berth_shift: 1,
            seed: 42,
            bandit_memory_coeff: 0.0,
            bandit_exploration_coeff: 0.0,
        };
        assert_eq!(cfg.build_operators().len(), 4);
    }

    #[test]
    fn test_operator_config_build_none() {
        let cfg = FfiOperatorConfig {
            strategy: FfiCompoundStrategy::RoundRobin,
            use_intra_berth_swap: 0,
            use_inter_berth_swap: 0,
            use_intra_berth_shift: 0,
            use_inter_berth_shift: 0,
            seed: 0,
            bandit_memory_coeff: 0.0,
            bandit_exploration_coeff: 0.0,
        };
        assert_eq!(cfg.build_operators().len(), 0);
    }

    #[test]
    fn test_operator_config_build_partial() {
        let cfg = FfiOperatorConfig {
            strategy: FfiCompoundStrategy::MultiArmedBandit,
            use_intra_berth_swap: 1,
            use_inter_berth_swap: 0,
            use_intra_berth_shift: 0,
            use_inter_berth_shift: 1,
            seed: 0,
            bandit_memory_coeff: 0.2,
            bandit_exploration_coeff: std::f64::consts::SQRT_2,
        };
        assert_eq!(cfg.build_operators().len(), 2);
    }

    // ---- Monitor config ----

    #[test]
    fn test_monitor_config_all_enabled() {
        let cfg = FfiMonitorConfig {
            time_limit_millis: 1000,
            iteration_limit: 5000,
            solution_limit: 100,
            cycle_limit: 50,
            no_improvement_iterations: 1000,
            no_improvement_cycles: 10,
            no_improvement_time_millis: 500,
        };
        // Just ensure building doesn't panic.
        let _mon = cfg.build_monitor();
    }

    #[test]
    fn test_monitor_config_none_enabled() {
        let cfg = FfiMonitorConfig {
            time_limit_millis: 0,
            iteration_limit: 0,
            solution_limit: 0,
            cycle_limit: 0,
            no_improvement_iterations: 0,
            no_improvement_cycles: 0,
            no_improvement_time_millis: 0,
        };
        let _mon = cfg.build_monitor();
    }

    #[test]
    fn test_monitor_config_partial() {
        let cfg = FfiMonitorConfig {
            time_limit_millis: 0,
            iteration_limit: 100,
            solution_limit: 0,
            cycle_limit: 0,
            no_improvement_iterations: 50,
            no_improvement_cycles: 0,
            no_improvement_time_millis: 0,
        };
        let _mon = cfg.build_monitor();
    }

    // ---- End-to-end run ----

    #[test]
    fn test_engine_run_with_iteration_limit() {
        let model = build_test_model();
        assert!(!model.is_null());

        let engine = talos_engine_new(2, 2);
        let mut meta_config = zeroed_meta_config(FfiMetaheuristic::SimulatedAnnealing);
        meta_config.sa_schedule = FfiCoolingSchedule::Geometric;
        meta_config.sa_initial_temperature = 100.0;
        meta_config.sa_geometric_alpha = 0.99;
        meta_config.sa_min_temperature = 0.001;
        meta_config.seed = 42;
        meta_config.num_vessels = 2;
        meta_config.num_berths = 2;
        let op_config = FfiOperatorConfig {
            strategy: FfiCompoundStrategy::Random,
            use_intra_berth_swap: 1,
            use_inter_berth_swap: 1,
            use_intra_berth_shift: 0,
            use_inter_berth_shift: 0,
            seed: 42,
            bandit_memory_coeff: 0.0,
            bandit_exploration_coeff: 0.0,
        };
        let mon_config = FfiMonitorConfig {
            time_limit_millis: 0,
            iteration_limit: 100,
            solution_limit: 0,
            cycle_limit: 0,
            no_improvement_iterations: 0,
            no_improvement_cycles: 0,
            no_improvement_time_millis: 0,
        };

        // Initial solution: V0→B0, V1→B1, start at 0
        let berths: [usize; 2] = [0, 1];
        let starts: [i64; 2] = [0, 0];
        let init_sol =
            unsafe { talos_solution_view_new(2, berths.as_ptr(), starts.as_ptr(), i64::MAX) };
        assert!(!init_sol.is_null());

        let outcome = unsafe {
            talos_engine_run(
                engine,
                model,
                meta_config,
                op_config,
                mon_config,
                init_sol,
                None,
                noop_callback,
                ptr::null_mut(),
            )
        };

        assert!(!outcome.solution.is_null());
        assert!(outcome.stats.iterations > 0);

        // Clean up
        unsafe {
            talos_solution_free(outcome.solution);
            talos_solution_view_free(init_sol);
            talos_engine_free(engine);
            crate::model::talos_model_free(model);
        }
    }

    #[test]
    fn test_engine_run_with_callback() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let mut meta_config = zeroed_meta_config(FfiMetaheuristic::TabuSearch);
        meta_config.tabu_tenure = 5;
        meta_config.num_vessels = 2;
        meta_config.num_berths = 2;
        let op_config = FfiOperatorConfig {
            strategy: FfiCompoundStrategy::RoundRobin,
            use_intra_berth_swap: 1,
            use_inter_berth_swap: 1,
            use_intra_berth_shift: 0,
            use_inter_berth_shift: 0,
            seed: 0,
            bandit_memory_coeff: 0.0,
            bandit_exploration_coeff: 0.0,
        };
        let mon_config = FfiMonitorConfig {
            time_limit_millis: 0,
            iteration_limit: 200,
            solution_limit: 0,
            cycle_limit: 0,
            no_improvement_iterations: 0,
            no_improvement_cycles: 0,
            no_improvement_time_millis: 0,
        };

        static CALL_COUNT: AtomicU64 = AtomicU64::new(0);
        CALL_COUNT.store(0, Ordering::SeqCst);

        unsafe extern "C" fn on_new_best(
            _view: *const SolutionView<'static, i64>,
            _user_data: *mut c_void,
        ) {
            CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        }

        let berths: [usize; 2] = [0, 1];
        let starts: [i64; 2] = [0, 0];
        let init_sol =
            unsafe { talos_solution_view_new(2, berths.as_ptr(), starts.as_ptr(), i64::MAX) };
        assert!(!init_sol.is_null());

        let outcome = unsafe {
            talos_engine_run(
                engine,
                model,
                meta_config,
                op_config,
                mon_config,
                init_sol,
                None,
                on_new_best,
                ptr::null_mut(),
            )
        };

        assert!(!outcome.solution.is_null());

        unsafe {
            talos_solution_free(outcome.solution);
            talos_solution_view_free(init_sol);
            talos_engine_free(engine);
            crate::model::talos_model_free(model);
        }
    }
}
