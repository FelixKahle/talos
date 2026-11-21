// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    callback::PyCallbackMonitor,
    ls::{
        gls::{PyDecay, PyGlsConfig, PyLambdaStrategy, PyTrigger},
        outcome::{PySearchResult, outcome_to_py},
    },
    model::PyModel,
    solution::PySolution,
};
use itertools::Itertools;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::time::{Duration, Instant};
use talos_ls::{
    AdditiveDynamicLambda, DynamicLambda, Engine, GeometricDecay, GuidedLocalSearch, Lambda,
    LocalSearchParameters, Metaheuristic, Neighborhood, PenalizationTrigger, PenaltyDecay,
    SearchLimits, StaticLambda, heuristic_lambda,
};
#[pyclass(name = "Operator", eq, eq_int, from_py_object)]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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
            operators: operators
                .into_iter()
                .unique_by(|op| op.clone() as u8)
                .collect(),
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
// Parameter construction
// ----------------------------------------------------------------

fn build_neighborhoods(operators: &[PyOperator]) -> Vec<Neighborhood> {
    operators
        .iter()
        .map(|operator| match operator {
            PyOperator::IntraSwap => Neighborhood::IntraBerthSwap,
            PyOperator::InterSwap => Neighborhood::InterBerthSwap,
            PyOperator::IntraShift => Neighborhood::IntraBerthShift,
            PyOperator::InterShift => Neighborhood::InterBerthShift,
        })
        .collect()
}

fn build_limits(config: &PyLocalSearchConfig) -> SearchLimits {
    SearchLimits {
        time: config.time_limit_secs.map(Duration::from_secs_f64),
        iterations: config.max_iterations,
        solutions: config.max_solutions,
        cycles: config.max_cycles,
        non_improving_iterations: config.max_non_improving_iterations,
        non_improving_cycles: config.max_non_improving_cycles,
        non_improving_time: config
            .max_non_improving_time_secs
            .map(Duration::from_secs_f64),
    }
}

fn build_trigger(config: &PyGlsConfig) -> PenalizationTrigger {
    match config.trigger {
        PyTrigger::OnExhaustion => PenalizationTrigger::OnExhaustion,
        PyTrigger::AfterNonImprovements => {
            PenalizationTrigger::AfterNonImprovements(config.trigger_threshold)
        }
        PyTrigger::AfterMoves => PenalizationTrigger::AfterMoves(config.trigger_threshold),
    }
}

/// The guided local search `config` asks for, its default weights derived from the initial objective over the
/// `features` vessel-berth pairs.
fn build_guided_local_search(
    config: &PyGlsConfig,
    objective: f64,
    features: usize,
) -> GuidedLocalSearch {
    let base = |alpha| {
        config
            .lambda_initial
            .unwrap_or_else(|| heuristic_lambda(objective, features, alpha))
    };
    let min = config
        .lambda_min
        .unwrap_or_else(|| heuristic_lambda(objective, features, 0.1));
    let max = config
        .lambda_max
        .unwrap_or_else(|| heuristic_lambda(objective, features, 0.3));
    let lambda = match config.lambda_strategy {
        PyLambdaStrategy::Static => Lambda::Static(StaticLambda::new(base(0.3))),
        PyLambdaStrategy::Dynamic => Lambda::Dynamic(
            DynamicLambda::new(
                base(0.2),
                config.lambda_inc_step,
                config.lambda_dec_step,
                min,
                max,
            )
            .with_reset_on_best(config.reset_on_best),
        ),
        PyLambdaStrategy::Additive => Lambda::AdditiveDynamic(
            AdditiveDynamicLambda::new(
                base(0.2),
                config.lambda_inc_step,
                config.lambda_dec_step,
                min,
                max,
            )
            .with_reset_on_best(config.reset_on_best),
        ),
    };
    let decay = match config.decay {
        PyDecay::NoDecay => PenaltyDecay::None,
        PyDecay::Geometric => PenaltyDecay::Geometric(GeometricDecay::new(
            config.decay_factor,
            config.decay_period,
        )),
    };
    GuidedLocalSearch {
        lambda,
        decay,
        trigger: build_trigger(config),
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
// PySolver
// ----------------------------------------------------------------

/// Reusable solver that keeps its internal `Engine` across calls,
/// avoiding repeated allocation when solving multiple instances.
///
/// ```python
/// solver = Solver()
/// r1 = solver.solve(model_a, config, gls_config, solution_a)
/// r2 = solver.solve(model_b, config, gls_config, solution_b)
/// ```
#[pyclass(name = "Solver")]
pub struct PySolver {
    engine: Engine<i64>,
}

#[pymethods]
impl PySolver {
    /// Create a new solver. The engine sizes itself for each problem passed to `solve`; the sizes are accepted for
    /// compatibility and not used.
    #[new]
    #[pyo3(signature = (num_vessels = 0, num_berths = 0))]
    fn new(num_vessels: usize, num_berths: usize) -> Self {
        let _ = (num_vessels, num_berths);
        Self {
            engine: Engine::new(),
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
        for (i, &berth) in inner_solution.assigned_berths().iter().enumerate() {
            if berth.index() >= num_berths {
                return Err(PyValueError::new_err(format!(
                    "berth index {} for vessel {i} out of bounds (num_berths = {num_berths})",
                    berth.index()
                )));
            }
        }

        let guided_local_search = match gls_config {
            None => GuidedLocalSearch::default(),
            Some(gls_config) => build_guided_local_search(
                gls_config,
                inner_solution.objective_value() as f64,
                num_vessels * num_berths,
            ),
        };
        let parameters = LocalSearchParameters {
            neighborhoods: build_neighborhoods(&config.operators),
            metaheuristic: Metaheuristic::GuidedLocalSearch(guided_local_search),
            limits: build_limits(config),
            ..LocalSearchParameters::default()
        };
        let mut monitor = PyCallbackMonitor::new(callback);

        let started = Instant::now();
        let outcome = self
            .engine
            .run(
                inner_model,
                inner_solution.as_view(),
                &parameters,
                &mut monitor,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(outcome_to_py(outcome, started.elapsed()))
    }
}
