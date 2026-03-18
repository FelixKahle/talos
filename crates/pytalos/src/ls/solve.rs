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
    meta::gls::{
        heuristic_lambda, AdditiveDynamicLambda, DynamicLambda, GeometricDecay, GuidedLocalSearch,
        PenalizationTrigger,
    },
    monitor::{
        composite::CompositeLocalSearchMonitor, cycle::CycleLimitMonitor,
        iteration::IterationLimitMonitor, nimpr::NoImprovementMonitor,
        solution::SolutionLimitMonitor, time::TimeLimitMonitor,
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
};
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};

use crate::{
    callback::make_callback,
    ls::{
        engine::{PyLocalSearchConfig, PyOperator},
        gls::{PyDecay, PyGlsConfig, PyLambdaStrategy, PyTrigger},
        outcome::{outcome_to_py, PySearchResult},
    },
    model::PyModel,
    solution::PySolution,
};

/// Default evaluator using weighted turnaround time.
#[inline(always)]
fn wtt_evaluator(
    model: &Model<i64>,
    vessel: VesselIndex,
    berth: BerthIndex,
    start: i64,
) -> Option<i64> {
    unsafe {
        talos_ls::eval::calculate_weighted_turnaround_time_unchecked(model, vessel, berth, start)
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
// Heuristic lambda calculation
// ----------------------------------------------------------------

/// Heuristic lambda calculation for GLS,
/// based on the initial objective value, problem size, and a scaling factor.
#[pyfunction]
pub fn heuristic_gls_lambda(objective: f64, num_features: usize, scale: f64) -> f64 {
    talos_ls::meta::gls::heuristic_lambda(objective, num_features, scale)
}

// ----------------------------------------------------------------
// Solve entry point
// ----------------------------------------------------------------

/// Runs GLS-based local search on the given model and initial solution.
#[pyfunction]
#[pyo3(signature = (model, config, gls_config, solution, callback = None))]
pub fn solve(
    model: &PyModel,
    config: &PyLocalSearchConfig,
    gls_config: Option<&PyGlsConfig>,
    solution: &PySolution,
    callback: Option<Py<PyAny>>,
) -> PyResult<PySearchResult> {
    let inner_model = model.inner();
    let inner_solution = solution.inner();
    let num_vessels = inner_model.num_vessels();
    let num_berths = inner_model.num_berths();

    // Validate inputs.
    if inner_solution.num_vessels() != num_vessels {
        return Err(PyValueError::new_err(format!(
            "solution has {} vessels but model has {num_vessels}",
            inner_solution.num_vessels()
        )));
    }
    if config.operators.is_empty() {
        return Err(PyValueError::new_err("operators list must not be empty"));
    }

    // Validate berth indices are within bounds.
    for (i, &berth) in inner_solution.berths().iter().enumerate() {
        if berth.get() >= num_berths {
            return Err(PyValueError::new_err(format!(
                "berth index {} for vessel {i} out of bounds (num_berths = {num_berths})",
                berth.get()
            )));
        }
    }

    let berth_indices = inner_solution.berths();
    let start_times = inner_solution.start_times();
    let objective = inner_solution.objective_value();

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
                berths: berth_indices,
                start_times,
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
                        berths: berth_indices,
                        start_times,
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
                        berths: berth_indices,
                        start_times,
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
                        berths: berth_indices,
                        start_times,
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
                        berths: berth_indices,
                        start_times,
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
                        berths: berth_indices,
                        start_times,
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
                        berths: berth_indices,
                        start_times,
                        objective_value: objective,
                    };
                    engine.run(params, wtt_evaluator, &mut *cb)
                }
            }
        }
    };

    Ok(outcome_to_py(result))
}
