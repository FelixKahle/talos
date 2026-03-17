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

use pyo3::{exceptions::PyValueError, prelude::*};
use std::time::Duration;
use talos_ls::{
    engine::Engine,
    exec::{SearchCommand, TerminationReason},
    meta::gls::{
        heuristic_lambda, AdditiveDynamicLambda, DynamicLambda, GeometricDecay, GuidedLocalSearch,
        PenalizationTrigger,
    },
    monitor::{
        composite::CompositeLocalSearchMonitor, cycle::CycleLimitMonitor,
        iteration::IterationLimitMonitor, lsmonitor::LocalSearchMonitor,
        nimpr::NoImprovementMonitor, solution::SolutionLimitMonitor, time::TimeLimitMonitor,
    },
    operator::{
        composite::RoundRobinCompoundOperator,
        filter::{
            inter_berth_shift_filter_unchecked, inter_berth_swap_filter_unchecked,
            intra_berth_shift_filter_unchecked, intra_berth_swap_filter_unchecked,
        },
        lsoperator::LocalSearchOperator,
        shift::{InterBerthShiftOperator, IntraBerthShiftOperator},
        swap::{InterBerthSwapOperator, IntraBerthSwapOperator},
    },
    params::MutableLocalSearchParams,
    stats::LocalSearchStatistics,
};
use talos_model::{index::BerthIndex, solution::SolutionView};

use crate::{
    eval::{make_callback, wtt_evaluator},
    ls::{
        engine::{PyLocalSearchConfig, PyOperator},
        gls::{PyDecay, PyGlsConfig, PyLambdaStrategy, PyTrigger},
        outcome::{PySearchResult, PyTerminationReason},
    },
    model::PyModel,
};

// ----------------------------------------------------------------
// TargetObjectiveMonitor
// ----------------------------------------------------------------

/// A lightweight monitor that terminates when the best objective reaches a target.
struct TargetObjectiveMonitor {
    target: i64,
}

impl LocalSearchMonitor<i64> for TargetObjectiveMonitor {
    fn name(&self) -> &str {
        "TargetObjectiveMonitor"
    }

    fn on_start(
        &mut self,
        _model: &talos_model::model::Model<i64>,
        _initial_solution: SolutionView<'_, i64>,
    ) {
    }

    fn on_end(
        &mut self,
        _best_solution: SolutionView<'_, i64>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_iteration(
        &mut self,
        _best_solution: SolutionView<'_, i64>,
        _accepted_solution: SolutionView<'_, i64>,
        _buffered_solution: Option<SolutionView<'_, i64>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_candidate_generated(
        &mut self,
        _best_solution: SolutionView<'_, i64>,
        _accepted_solution: SolutionView<'_, i64>,
        _buffered_solution: Option<SolutionView<'_, i64>>,
        _candidate_objective: i64,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_solution_buffered(
        &mut self,
        _best_solution: SolutionView<'_, i64>,
        _accepted_solution: SolutionView<'_, i64>,
        _buffered_solution: SolutionView<'_, i64>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_candidate_accepted(
        &mut self,
        _best_solution: SolutionView<'_, i64>,
        _accepted_solution: SolutionView<'_, i64>,
        _buffered_solution: Option<SolutionView<'_, i64>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_buffered_solution_accepted(
        &mut self,
        _best_solution: SolutionView<'_, i64>,
        _accepted_solution: SolutionView<'_, i64>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_candidate_rejected(
        &mut self,
        _best_solution: SolutionView<'_, i64>,
        _accepted_solution: SolutionView<'_, i64>,
        _buffered_solution: Option<SolutionView<'_, i64>>,
        _candidate_objective: i64,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_candidate_infeasible(
        &mut self,
        _best_solution: SolutionView<'_, i64>,
        _accepted_solution: SolutionView<'_, i64>,
        _buffered_solution: Option<SolutionView<'_, i64>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_neighborhood_exhausted(
        &mut self,
        _best_solution: SolutionView<'_, i64>,
        _accepted_solution: SolutionView<'_, i64>,
        _buffered_solution: Option<SolutionView<'_, i64>>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_best_solution_updated(
        &mut self,
        _prev_best: SolutionView<'_, i64>,
        _accepted_solution: SolutionView<'_, i64>,
        _buffered_solution: Option<SolutionView<'_, i64>>,
        _new_best: SolutionView<'_, i64>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn search_command(
        &mut self,
        best_solution: SolutionView<'_, i64>,
        _accepted_solution: SolutionView<'_, i64>,
        _buffered_solution: Option<SolutionView<'_, i64>>,
        _statistics: &LocalSearchStatistics,
    ) -> SearchCommand {
        if best_solution.objective_value() <= self.target {
            SearchCommand::Terminate(TerminationReason::TargetObjectiveReached)
        } else {
            SearchCommand::Continue
        }
    }
}

// ----------------------------------------------------------------
// Operator construction
// ----------------------------------------------------------------

fn build_operator(operators: &[PyOperator]) -> RoundRobinCompoundOperator<'static, i64> {
    let ops: Vec<Box<dyn LocalSearchOperator<i64>>> = operators
        .iter()
        .map(|op| -> Box<dyn LocalSearchOperator<i64>> {
            match op {
                PyOperator::IntraSwap => Box::new(IntraBerthSwapOperator::new(
                    |v_a, v_b, sol, graph, model| unsafe {
                        intra_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
                    },
                )),
                PyOperator::InterSwap => Box::new(InterBerthSwapOperator::new(
                    |v_a, v_b, sol, graph, model| unsafe {
                        inter_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
                    },
                )),
                PyOperator::IntraShift => Box::new(IntraBerthShiftOperator::new(
                    |v, anchor, sol, graph, model| unsafe {
                        intra_berth_shift_filter_unchecked(v, anchor, sol, graph, model)
                    },
                )),
                PyOperator::InterShift => Box::new(InterBerthShiftOperator::new(
                    |v, anchor, sol, graph, model| unsafe {
                        inter_berth_shift_filter_unchecked(v, anchor, sol, graph, model)
                    },
                )),
            }
        })
        .collect();

    RoundRobinCompoundOperator::new(ops)
}

// ----------------------------------------------------------------
// Monitor construction
// ----------------------------------------------------------------

fn build_monitor(config: &PyLocalSearchConfig) -> CompositeLocalSearchMonitor<'static, i64> {
    let mut composite = CompositeLocalSearchMonitor::new();

    if let Some(time_limit) = config.time_limit_secs {
        composite.add_monitor(TimeLimitMonitor::new(Duration::from_secs_f64(time_limit)));
    }

    if let Some(max_iterations) = config.max_iterations {
        composite.add_monitor(IterationLimitMonitor::new(max_iterations));
    }

    if let Some(max_solutions) = config.max_solutions {
        composite.add_monitor(SolutionLimitMonitor::new(max_solutions));
    }

    if let Some(max_cycles) = config.max_cycles {
        composite.add_monitor(CycleLimitMonitor::new(max_cycles));
    }

    if let Some(target) = config.target_objective {
        composite.add_monitor(TargetObjectiveMonitor { target });
    }

    // Non-improving termination: combine iteration, cycle and time patience
    let has_ni_iter = config.max_non_improving_iterations.is_some();
    let has_ni_cycles = config.max_non_improving_cycles.is_some();
    let has_ni_time = config.max_non_improving_time_secs.is_some();

    if has_ni_iter || has_ni_cycles || has_ni_time {
        let mut ni = if let Some(patience) = config.max_non_improving_iterations {
            NoImprovementMonitor::with_iteration_patience(patience)
        } else if let Some(patience) = config.max_non_improving_cycles {
            NoImprovementMonitor::with_cycle_patience(patience)
        } else {
            NoImprovementMonitor::with_duration_patience(Duration::from_secs_f64(
                config.max_non_improving_time_secs.unwrap(),
            ))
        };

        // Chain the remaining ones
        if has_ni_iter && config.max_non_improving_iterations.is_some() {
            // Already set as the primary if it was the first
        }
        if has_ni_cycles {
            ni = ni.and_cycle_patience(config.max_non_improving_cycles.unwrap());
        }
        if has_ni_time {
            ni = ni.and_duration_patience(Duration::from_secs_f64(
                config.max_non_improving_time_secs.unwrap(),
            ));
        }
        if has_ni_iter {
            ni = ni.and_iteration_patience(config.max_non_improving_iterations.unwrap());
        }

        composite.add_monitor(ni);
    }

    composite
}

// ----------------------------------------------------------------
// GLS construction
// ----------------------------------------------------------------

fn build_trigger(config: &PyGlsConfig) -> PenalizationTrigger {
    match config.trigger {
        PyTrigger::OnExhaustion => PenalizationTrigger::OnExhaustion,
        PyTrigger::AfterNonImprovements => {
            PenalizationTrigger::AfterNonImprovements(config.trigger_threshold)
        }
        PyTrigger::AfterMoves => PenalizationTrigger::AfterMoves(config.trigger_threshold),
    }
}

// ----------------------------------------------------------------
// Outcome conversion
// ----------------------------------------------------------------

fn outcome_to_py(outcome: talos_ls::outcome::LocalSearchOutcome<i64>) -> PySearchResult {
    let (solution, reason, stats) = outcome.into_inner();

    PySearchResult {
        objective: solution.objective_value(),
        berths: solution.berths().iter().map(|b| b.get()).collect(),
        start_times: solution.start_times().to_vec(),
        termination_reason: PyTerminationReason::from(reason),
        iterations: stats.iterations,
        accepted_solutions: stats.accepted_solutions,
        total_solutions: stats.total_solutions,
        infeasible_moves: stats.infeasible_moves,
        cycles: stats.cycles,
        time_total_secs: stats.time_total.as_secs_f64(),
    }
}

// ----------------------------------------------------------------
// Solve entry point
// ----------------------------------------------------------------

/// Runs GLS-based local search on the given model and initial solution.
///
/// Arguments:
///     model: The DBAP model.
///     config: Local search configuration (operators, termination criteria).
///     gls_config: Optional GLS configuration. If None, uses default GLS settings.
///     berths: Initial berth assignment per vessel (list of berth indices, len = num_vessels).
///     start_times: Initial start time per vessel (len = num_vessels).
///     objective: Objective value of the initial solution.
///     callback: Optional Python callback invoked on each new best solution.
///               Receives (objective: int, berths: list[int], start_times: list[int]).
#[pyfunction]
#[pyo3(signature = (model, config, gls_config, berths, start_times, objective, callback = None))]
pub fn solve(
    model: &PyModel,
    config: &PyLocalSearchConfig,
    gls_config: Option<&PyGlsConfig>,
    berths: Vec<usize>,
    start_times: Vec<i64>,
    objective: i64,
    callback: Option<Py<PyAny>>,
) -> PyResult<PySearchResult> {
    let inner_model = model.inner();
    let num_vessels = inner_model.num_vessels();
    let num_berths = inner_model.num_berths();

    // Validate inputs.
    if berths.len() != num_vessels {
        return Err(PyValueError::new_err(format!(
            "berths length {} != num_vessels {num_vessels}",
            berths.len()
        )));
    }
    if start_times.len() != num_vessels {
        return Err(PyValueError::new_err(format!(
            "start_times length {} != num_vessels {num_vessels}",
            start_times.len()
        )));
    }
    if config.operators.is_empty() {
        return Err(PyValueError::new_err("operators list must not be empty"));
    }

    // Convert berth indices.
    let berth_indices: Vec<BerthIndex> = berths
        .iter()
        .map(|&b| {
            if b >= num_berths {
                Err(PyValueError::new_err(format!(
                    "berth index {b} out of bounds (num_berths = {num_berths})"
                )))
            } else {
                Ok(BerthIndex::new(b))
            }
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut operator = build_operator(&config.operators);
    let monitor = build_monitor(config);
    let mut engine = Engine::<i64>::new(num_vessels, num_berths);
    let mut cb = make_callback(callback);

    // Build GLS metaheuristic with the appropriate lambda strategy and decay.
    // We dispatch on the combination of lambda_strategy and decay to produce
    // the correct concrete GLS type (the engine is generic over these).
    let result = match gls_config {
        None => {
            // Default GLS: static lambda = 1.0, no decay.
            let mut gls = GuidedLocalSearch::new(num_vessels, num_berths);
            let params = MutableLocalSearchParams {
                model: inner_model,
                operator: &mut operator,
                metaheuristic: &mut gls,
                monitor,
                berths: &berth_indices,
                start_times: &start_times,
                objective_value: objective,
            };
            engine.run(params, wtt_evaluator, &mut *cb)
        }
        Some(gls_cfg) => {
            let trigger = build_trigger(gls_cfg);

            match (&gls_cfg.lambda_strategy, &gls_cfg.decay) {
                // Static lambda
                (PyLambdaStrategy::Static, PyDecay::NoDecay) => {
                    let lambda = gls_cfg.lambda_initial.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.3)
                    });
                    let mut gls = GuidedLocalSearch::new(num_vessels, num_berths)
                        .with_lambda(lambda)
                        .with_trigger(trigger);
                    if gls_cfg.reset_on_best {
                        // StaticLambda doesn't have reset_on_best,
                        // but resetting a constant is a no-op.
                    }
                    let params = MutableLocalSearchParams {
                        model: inner_model,
                        operator: &mut operator,
                        metaheuristic: &mut gls,
                        monitor,
                        berths: &berth_indices,
                        start_times: &start_times,
                        objective_value: objective,
                    };
                    engine.run(params, wtt_evaluator, &mut *cb)
                }
                (PyLambdaStrategy::Static, PyDecay::Geometric) => {
                    let lambda = gls_cfg.lambda_initial.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.3)
                    });
                    let decay = GeometricDecay::new(gls_cfg.decay_factor, gls_cfg.decay_period);
                    let mut gls = GuidedLocalSearch::new(num_vessels, num_berths)
                        .with_lambda(lambda)
                        .with_decay(decay)
                        .with_trigger(trigger);
                    let params = MutableLocalSearchParams {
                        model: inner_model,
                        operator: &mut operator,
                        metaheuristic: &mut gls,
                        monitor,
                        berths: &berth_indices,
                        start_times: &start_times,
                        objective_value: objective,
                    };
                    engine.run(params, wtt_evaluator, &mut *cb)
                }

                // Dynamic lambda
                (PyLambdaStrategy::Dynamic, PyDecay::NoDecay) => {
                    let base = gls_cfg.lambda_initial.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.2)
                    });
                    let lo = gls_cfg.lambda_min.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.1)
                    });
                    let hi = gls_cfg.lambda_max.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.3)
                    });
                    let strategy = DynamicLambda::new(
                        base,
                        gls_cfg.lambda_inc_step,
                        gls_cfg.lambda_dec_step,
                        lo,
                        hi,
                    )
                    .with_reset_on_best(gls_cfg.reset_on_best);
                    let mut gls = GuidedLocalSearch::new(num_vessels, num_berths)
                        .with_lambda_strategy(strategy)
                        .with_trigger(trigger);
                    let params = MutableLocalSearchParams {
                        model: inner_model,
                        operator: &mut operator,
                        metaheuristic: &mut gls,
                        monitor,
                        berths: &berth_indices,
                        start_times: &start_times,
                        objective_value: objective,
                    };
                    engine.run(params, wtt_evaluator, &mut *cb)
                }
                (PyLambdaStrategy::Dynamic, PyDecay::Geometric) => {
                    let base = gls_cfg.lambda_initial.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.2)
                    });
                    let lo = gls_cfg.lambda_min.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.1)
                    });
                    let hi = gls_cfg.lambda_max.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.3)
                    });
                    let strategy = DynamicLambda::new(
                        base,
                        gls_cfg.lambda_inc_step,
                        gls_cfg.lambda_dec_step,
                        lo,
                        hi,
                    )
                    .with_reset_on_best(gls_cfg.reset_on_best);
                    let decay = GeometricDecay::new(gls_cfg.decay_factor, gls_cfg.decay_period);
                    let mut gls = GuidedLocalSearch::new(num_vessels, num_berths)
                        .with_lambda_strategy(strategy)
                        .with_decay(decay)
                        .with_trigger(trigger);
                    let params = MutableLocalSearchParams {
                        model: inner_model,
                        operator: &mut operator,
                        metaheuristic: &mut gls,
                        monitor,
                        berths: &berth_indices,
                        start_times: &start_times,
                        objective_value: objective,
                    };
                    engine.run(params, wtt_evaluator, &mut *cb)
                }

                // Additive lambda
                (PyLambdaStrategy::Additive, PyDecay::NoDecay) => {
                    let base = gls_cfg.lambda_initial.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.2)
                    });
                    let lo = gls_cfg.lambda_min.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.1)
                    });
                    let hi = gls_cfg.lambda_max.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.3)
                    });
                    let strategy = AdditiveDynamicLambda::new(
                        base,
                        gls_cfg.lambda_inc_step,
                        gls_cfg.lambda_dec_step,
                        lo,
                        hi,
                    )
                    .with_reset_on_best(gls_cfg.reset_on_best);
                    let mut gls = GuidedLocalSearch::new(num_vessels, num_berths)
                        .with_lambda_strategy(strategy)
                        .with_trigger(trigger);
                    let params = MutableLocalSearchParams {
                        model: inner_model,
                        operator: &mut operator,
                        metaheuristic: &mut gls,
                        monitor,
                        berths: &berth_indices,
                        start_times: &start_times,
                        objective_value: objective,
                    };
                    engine.run(params, wtt_evaluator, &mut *cb)
                }
                (PyLambdaStrategy::Additive, PyDecay::Geometric) => {
                    let base = gls_cfg.lambda_initial.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.2)
                    });
                    let lo = gls_cfg.lambda_min.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.1)
                    });
                    let hi = gls_cfg.lambda_max.unwrap_or_else(|| {
                        heuristic_lambda(objective as f64, num_vessels * num_berths, 0.3)
                    });
                    let strategy = AdditiveDynamicLambda::new(
                        base,
                        gls_cfg.lambda_inc_step,
                        gls_cfg.lambda_dec_step,
                        lo,
                        hi,
                    )
                    .with_reset_on_best(gls_cfg.reset_on_best);
                    let decay = GeometricDecay::new(gls_cfg.decay_factor, gls_cfg.decay_period);
                    let mut gls = GuidedLocalSearch::new(num_vessels, num_berths)
                        .with_lambda_strategy(strategy)
                        .with_decay(decay)
                        .with_trigger(trigger);
                    let params = MutableLocalSearchParams {
                        model: inner_model,
                        operator: &mut operator,
                        metaheuristic: &mut gls,
                        monitor,
                        berths: &berth_indices,
                        start_times: &start_times,
                        objective_value: objective,
                    };
                    engine.run(params, wtt_evaluator, &mut *cb)
                }
            }
        }
    };

    Ok(outcome_to_py(result))
}
