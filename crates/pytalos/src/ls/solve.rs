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
    callback::make_callback,
    ls::{
        gls::{PyDecay, PyGlsConfig, PyLambdaStrategy, PyTrigger},
        outcome::{outcome_to_py, PySearchResult},
    },
    model::PyModel,
    solution::PySolution,
};
use itertools::Itertools;
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

#[pyclass(name = "Operator", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PyOperator {
    IntraSwap = 0,
    InterSwap = 1,
    IntraShift = 2,
    InterShift = 3,
}

/// Configuration for the local search engine.
#[pyclass(name = "LocalSearchConfig", from_py_object)]
#[derive(Clone)]
pub struct PyLocalSearchConfig {
    #[pyo3(get)]
    pub operators: Vec<PyOperator>,
    #[pyo3(get)]
    pub max_iterations: Option<u64>,
    #[pyo3(get)]
    pub max_solutions: Option<u64>,
    #[pyo3(get)]
    pub max_cycles: Option<u64>,
    #[pyo3(get)]
    pub max_non_improving_iterations: Option<u64>,
    #[pyo3(get)]
    pub max_non_improving_cycles: Option<u64>,
    #[pyo3(get)]
    pub max_non_improving_time_secs: Option<f64>,
    #[pyo3(get)]
    pub time_limit_secs: Option<f64>,
}

#[pymethods]
impl PyLocalSearchConfig {
    #[new]
    #[pyo3(signature = (
        operators,
        max_iterations = None,
        max_solutions = None,
        max_cycles = None,
        max_non_improving_iterations = None,
        max_non_improving_cycles = None,
        max_non_improving_time_secs = None,
        time_limit_secs = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        operators: Vec<PyOperator>,
        max_iterations: Option<u64>,
        max_solutions: Option<u64>,
        max_cycles: Option<u64>,
        max_non_improving_iterations: Option<u64>,
        max_non_improving_cycles: Option<u64>,
        max_non_improving_time_secs: Option<f64>,
        time_limit_secs: Option<f64>,
    ) -> Self {
        Self {
            operators: operators.into_iter().unique_by(|op| *op as u8).collect(),
            max_iterations,
            max_solutions,
            max_cycles,
            max_non_improving_iterations,
            max_non_improving_cycles,
            max_non_improving_time_secs,
            time_limit_secs,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LocalSearchConfig(operators={:?}, time_limit={:?}s)",
            self.operators, self.time_limit_secs
        )
    }
}

// ----------------------------------------------------------------
// Evaluator
// ----------------------------------------------------------------

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
// GLS trigger construction
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
// Heuristic lambda (standalone utility)
// ----------------------------------------------------------------

/// Heuristic lambda calculation for GLS,
/// based on the initial objective value, problem size, and a scaling factor.
#[pyfunction]
pub fn heuristic_gls_lambda(objective: f64, num_features: usize, scale: f64) -> f64 {
    heuristic_lambda(objective, num_features, scale)
}

// ----------------------------------------------------------------
// GLS dispatch macro
// ----------------------------------------------------------------

/// Builds `MutableLocalSearchParams`, calls `engine.run`, and returns the result.
/// Exists solely to avoid repeating the same 8-line block in every match arm.
macro_rules! run_with_gls {
    ($engine:expr, $gls:expr, $model:expr, $operator:expr, $monitor:expr,
     $berths:expr, $start_times:expr, $objective:expr, $cb:expr) => {{
        let params = MutableLocalSearchParams {
            model: $model,
            operator: $operator,
            metaheuristic: &mut $gls,
            monitor: $monitor,
            berths: $berths,
            start_times: $start_times,
            objective_value: $objective,
        };
        $engine.run(params, wtt_evaluator, $cb)
    }};
}

// ----------------------------------------------------------------
// PySolver
// ----------------------------------------------------------------

/// Reusable solver that keeps its internal `Engine` across calls,
/// avoiding repeated allocation when solving multiple instances.
///
/// ```python
/// solver = Solver()          # or Solver(100, 5) to pre-allocate
/// r1 = solver.solve(model_a, config, gls_config, solution_a)
/// r2 = solver.solve(model_b, config, gls_config, solution_b)
/// ```
#[pyclass(name = "Solver")]
pub struct PySolver {
    engine: Engine<i64>,
}

#[pymethods]
impl PySolver {
    /// Create a new solver, optionally pre-allocating for a given problem size.
    /// The engine grows automatically if a larger problem is passed to `solve`.
    #[new]
    #[pyo3(signature = (num_vessels = 0, num_berths = 0))]
    fn new(num_vessels: usize, num_berths: usize) -> Self {
        Self {
            engine: Engine::new(num_vessels, num_berths),
        }
    }

    /// Run GLS-based local search on the given model and initial solution.
    #[pyo3(signature = (model, config, gls_config, solution, callback = None))]
    fn solve(
        &mut self,
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
        for (i, &berth) in inner_solution.berths().iter().enumerate() {
            if berth.get() >= num_berths {
                return Err(PyValueError::new_err(format!(
                    "berth index {} for vessel {i} out of bounds (num_berths = {num_berths})",
                    berth.get()
                )));
            }
        }

        let berths = inner_solution.berths();
        let start_times = inner_solution.start_times();
        let objective = inner_solution.objective_value();

        let mut operator = build_operator(&config.operators);
        let monitor = build_monitor(config);
        let mut cb = make_callback(callback);

        let result = match gls_config {
            None => {
                let mut gls = GuidedLocalSearch::new(num_vessels, num_berths);
                run_with_gls!(
                    self.engine,
                    gls,
                    inner_model,
                    &mut operator,
                    monitor,
                    berths,
                    start_times,
                    objective,
                    &mut *cb
                )
            }
            Some(gls_cfg) => {
                let trigger = build_trigger(gls_cfg);
                let nv_nb = num_vessels * num_berths;
                let obj_f64 = objective as f64;

                let static_lambda = || {
                    gls_cfg
                        .lambda_initial
                        .unwrap_or_else(|| heuristic_lambda(obj_f64, nv_nb, 0.3))
                };
                let dynamic_params = || {
                    let base = gls_cfg
                        .lambda_initial
                        .unwrap_or_else(|| heuristic_lambda(obj_f64, nv_nb, 0.2));
                    let lo = gls_cfg
                        .lambda_min
                        .unwrap_or_else(|| heuristic_lambda(obj_f64, nv_nb, 0.1));
                    let hi = gls_cfg
                        .lambda_max
                        .unwrap_or_else(|| heuristic_lambda(obj_f64, nv_nb, 0.3));
                    (base, lo, hi)
                };
                let maybe_decay =
                    || GeometricDecay::new(gls_cfg.decay_factor, gls_cfg.decay_period);

                match (&gls_cfg.lambda_strategy, &gls_cfg.decay) {
                    (PyLambdaStrategy::Static, PyDecay::NoDecay) => {
                        let mut gls = GuidedLocalSearch::new(num_vessels, num_berths)
                            .with_lambda(static_lambda())
                            .with_trigger(trigger);
                        run_with_gls!(
                            self.engine,
                            gls,
                            inner_model,
                            &mut operator,
                            monitor,
                            berths,
                            start_times,
                            objective,
                            &mut *cb
                        )
                    }
                    (PyLambdaStrategy::Static, PyDecay::Geometric) => {
                        let mut gls = GuidedLocalSearch::new(num_vessels, num_berths)
                            .with_lambda(static_lambda())
                            .with_decay(maybe_decay())
                            .with_trigger(trigger);
                        run_with_gls!(
                            self.engine,
                            gls,
                            inner_model,
                            &mut operator,
                            monitor,
                            berths,
                            start_times,
                            objective,
                            &mut *cb
                        )
                    }
                    (PyLambdaStrategy::Dynamic, PyDecay::NoDecay) => {
                        let (base, lo, hi) = dynamic_params();
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
                        run_with_gls!(
                            self.engine,
                            gls,
                            inner_model,
                            &mut operator,
                            monitor,
                            berths,
                            start_times,
                            objective,
                            &mut *cb
                        )
                    }
                    (PyLambdaStrategy::Dynamic, PyDecay::Geometric) => {
                        let (base, lo, hi) = dynamic_params();
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
                            .with_decay(maybe_decay())
                            .with_trigger(trigger);
                        run_with_gls!(
                            self.engine,
                            gls,
                            inner_model,
                            &mut operator,
                            monitor,
                            berths,
                            start_times,
                            objective,
                            &mut *cb
                        )
                    }
                    (PyLambdaStrategy::Additive, PyDecay::NoDecay) => {
                        let (base, lo, hi) = dynamic_params();
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
                        run_with_gls!(
                            self.engine,
                            gls,
                            inner_model,
                            &mut operator,
                            monitor,
                            berths,
                            start_times,
                            objective,
                            &mut *cb
                        )
                    }
                    (PyLambdaStrategy::Additive, PyDecay::Geometric) => {
                        let (base, lo, hi) = dynamic_params();
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
                            .with_decay(maybe_decay())
                            .with_trigger(trigger);
                        run_with_gls!(
                            self.engine,
                            gls,
                            inner_model,
                            &mut operator,
                            monitor,
                            berths,
                            start_times,
                            objective,
                            &mut *cb
                        )
                    }
                }
            }
        };

        Ok(outcome_to_py(result))
    }
}
