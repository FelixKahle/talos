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

//! C-compatible FFI layer for the Talos local search engine.
//!
//! This module exposes the full configuration space of every metaheuristic
//! (Tabu Search, Simulated Annealing, Guided Local Search, Greedy Descent),
//! operator, and monitor through flat `#[repr(C)]` structs and free
//! functions that can be called from any language with a C FFI.
//!
//! # Architecture
//!
//! Each metaheuristic has its own configuration struct
//! (`FfiTabuSearchConfig`, `FfiSimulatedAnnealingConfig`,
//! `FfiGuidedLocalSearchConfig`, `FfiGreedyDescentConfig`) that maps 1:1
//! to the Rust-side builder API.
//!
//! Internally, three `#[inline(always)]` generic helpers form a
//! zero-cost dispatch chain that avoids both macros and dynamic dispatch
//! for the metaheuristic, evaluator function, and compound operator:
//!
//! 1. **`execute`** — fully monomorphised over `H` (metaheuristic),
//!    `O` (compound operator), and `F` (evaluator closure). Constructs
//!    `LocalSearchParams`, runs the engine, and packs the result.
//! 2. **`dispatch_operator`** — generic over `H` and `F`, matches on
//!    the `FfiCompoundStrategy` discriminant to instantiate a concrete
//!    compound operator and forward to `execute`.
//! 3. **`run_inner`** — generic over `H`, matches on the evaluator
//!    function pointer (custom vs. built-in) and forwards to
//!    `dispatch_operator`.
//!
//! Each per-algorithm exported function (`talos_engine_run_tabu`,
//! `talos_engine_run_sa`, `talos_engine_run_gls`, `talos_engine_run_gd`)
//! constructs the concrete metaheuristic type via exhaustive `match`
//! arms and delegates to `run_inner`.
//!
//! A unified `talos_engine_run` entry point is also provided, dispatching
//! on an `FfiMetaheuristic` discriminant for callers that prefer a single
//! function.

use rand::SeedableRng;
use rand::rngs::StdRng;
use std::ffi::c_void;
use std::time::Duration;
use talos_ls::engine::Engine;
use talos_ls::eval::calculate_weighted_turnaround_time_unchecked;
use talos_ls::meta::gd::GreedyDescent;
use talos_ls::meta::gls::{GuidedLocalSearch, PenalizationTrigger, ReactiveLambda};
use talos_ls::meta::metaheuristic::Metaheuristic;
use talos_ls::meta::sa::{
    GeometricCooling, LinearCooling, LogarithmicCooling, MetropolisCriterion, SimulatedAnnealing,
};
use talos_ls::meta::tabu::{FixedTenure, RandomTenure, TabuSearch};
use talos_ls::meta::tie::{KeepFirst, KeepLast, RandomTieBreak};
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
use talos_model::index::{BerthIndex, VesselIndex};
use talos_model::model::Model;
use talos_model::solution::SolutionView;

use crate::ffils::exec::FfiTerminationReason;
use crate::ffils::outcome::FfiLocalSearchOutcome;
use crate::ffils::params::{
    FfiCompoundStrategy, FfiCoolingSchedule, FfiCoolingTrigger, FfiExhaustionOutcome,
    FfiGreedyDescentConfig, FfiGuidedLocalSearchConfig, FfiMetaheuristic, FfiMonitorConfig,
    FfiOperatorConfig, FfiPenalizationTrigger, FfiSelectionStrategy, FfiSimulatedAnnealingConfig,
    FfiTabuSearchConfig, FfiTieBreaking,
};
use crate::ffils::stats::FfiLocalSearchStatistics;

// ----------------------------------------------------------------
// Callback types
// ----------------------------------------------------------------

/// Optional custom evaluator function pointer (nullable in C).
///
/// Called for every vessel during solution decoding. Return a
/// non-negative cost value, or any negative value to signal
/// infeasibility (the move will be rejected).
///
/// Pass `NULL` to use the built-in weighted-turnaround-time evaluator.
///
/// Signature: `(model, vessel_index, berth_index, start_time) -> cost`
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
/// The `solution_view` pointer is only valid for the duration of the
/// callback invocation. Use the `talos_solution_view_*` accessors to
/// read it, or call `talos_solution_view_to_owned` to deep-copy.
///
/// Signature: `(solution_view, user_data)`
pub type FfiNewBestCallbackFn =
    unsafe extern "C" fn(solution_view: *const SolutionView<'static, i64>, user_data: *mut c_void);

// ----------------------------------------------------------------
// Engine lifecycle
// ----------------------------------------------------------------

/// Creates a new search engine pre-allocated for the given problem
/// dimensions.
///
/// # Safety
///
/// The returned pointer must eventually be freed with
/// `talos_engine_free`.
#[unsafe(no_mangle)]
pub extern "C" fn talos_engine_new(num_vessels: usize, num_berths: usize) -> *mut Engine<i64> {
    Box::into_raw(Box::new(Engine::new(num_vessels, num_berths)))
}

/// Frees an engine previously created by `talos_engine_new`.
///
/// # Safety
///
/// * `engine` must be a valid pointer from `talos_engine_new`.
/// * Must not be called while a run is executing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_engine_free(engine: *mut Engine<i64>) {
    assert!(
        !engine.is_null(),
        "called `talos_engine_free` with `engine` as `null`"
    );
    drop(unsafe { Box::from_raw(engine) });
}

// ----------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------

/// Builds the vector of enabled operators from the config.
fn build_operators(cfg: &FfiOperatorConfig) -> Vec<Box<dyn LocalSearchOperator<i64>>> {
    let mut ops: Vec<Box<dyn LocalSearchOperator<i64>>> = Vec::with_capacity(4);
    if cfg.use_intra_berth_swap != 0 {
        ops.push(Box::new(IntraBerthSwapOperator::new(
            |v_a, v_b, sol, graph, model| unsafe {
                intra_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
            },
        )));
    }
    if cfg.use_inter_berth_swap != 0 {
        ops.push(Box::new(InterBerthSwapOperator::new(
            |v_a, v_b, sol, graph, model| unsafe {
                inter_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
            },
        )));
    }
    if cfg.use_intra_berth_shift != 0 {
        ops.push(Box::new(IntraBerthShiftOperator::new(
            |v, anchor, sol, graph, model| unsafe {
                intra_berth_shift_filter_unchecked(v, anchor, sol, graph, model)
            },
        )));
    }
    if cfg.use_inter_berth_shift != 0 {
        ops.push(Box::new(InterBerthShiftOperator::new(
            |v, anchor, sol, graph, model| unsafe {
                inter_berth_shift_filter_unchecked(v, anchor, sol, graph, model)
            },
        )));
    }
    ops
}

/// Builds the composite monitor from the config.
fn build_monitor(cfg: &FfiMonitorConfig) -> CompositeLocalSearchMonitor<'static, i64> {
    let mut mon = CompositeLocalSearchMonitor::new();
    if cfg.time_limit_millis > 0 {
        mon.add_monitor(TimeLimitMonitor::new(Duration::from_millis(
            cfg.time_limit_millis,
        )));
    }
    if cfg.iteration_limit > 0 {
        mon.add_monitor(IterationLimitMonitor::new(cfg.iteration_limit));
    }
    if cfg.solution_limit > 0 {
        mon.add_monitor(SolutionLimitMonitor::new(cfg.solution_limit));
    }
    if cfg.cycle_limit > 0 {
        mon.add_monitor(CycleLimitMonitor::new(cfg.cycle_limit));
    }
    if cfg.no_improvement_iterations > 0 {
        mon.add_monitor(NoImprovementMonitor::with_iteration_patience(
            cfg.no_improvement_iterations,
        ));
    }
    if cfg.no_improvement_cycles > 0 {
        mon.add_monitor(NoImprovementMonitor::with_cycle_patience(
            cfg.no_improvement_cycles,
        ));
    }
    if cfg.no_improvement_time_millis > 0 {
        mon.add_monitor(NoImprovementMonitor::with_duration_patience(
            Duration::from_millis(cfg.no_improvement_time_millis),
        ));
    }
    mon
}

/// Generic run helper. Constructs the evaluator closure, callback
/// wrapper, compound operator, validates parameters, and launches the
/// engine. Every exported `talos_engine_run_*` function builds the
/// concrete metaheuristic and then delegates here.
///
/// Dispatches on `evaluator` (custom vs. built-in) and
/// `op_config.strategy` (Round-Robin / Random / Multi-Armed Bandit)
/// via `dispatch_operator` and `execute`, producing fully
/// monomorphised, branchless code paths for each combination.
///
/// # Safety
///
/// All pointer arguments must be valid and live for the duration of the
/// call. See the per-function safety docs.
#[allow(clippy::too_many_arguments)]
unsafe fn run_inner<H: Metaheuristic<i64>>(
    engine: *mut Engine<i64>,
    model: *const Model<i64>,
    meta: &mut H,
    op_config: &FfiOperatorConfig,
    mon_config: &FfiMonitorConfig,
    initial_solution: *const SolutionView<'static, i64>,
    evaluator: FfiEvaluatorFn,
    callback: FfiNewBestCallbackFn,
    user_data: *mut c_void,
) -> FfiLocalSearchOutcome {
    assert!(
        !engine.is_null(),
        "called `run_inner` with `engine` as `null`"
    );
    assert!(
        !model.is_null(),
        "called `run_inner` with `model` as `null`"
    );
    assert!(
        !initial_solution.is_null(),
        "called `run_inner` with `initial_solution` as `null`"
    );

    let engine = unsafe { &mut *engine };
    let model = unsafe { &*model };
    let init_sol = unsafe { &*initial_solution };

    let operators = build_operators(op_config);
    assert!(
        !operators.is_empty(),
        "called `run_inner` with no operators enabled in `op_config`"
    );

    let monitor = build_monitor(mon_config);

    let berths = init_sol.berths();
    let start_times = init_sol.start_times();
    let objective_value = init_sol.objective_value();

    // Branch on evaluator once, outside the hot loop, to produce two
    // fully monomorphised, branchless code-paths.
    let model_ptr = model as *const Model<i64>;
    match evaluator {
        Some(f) => {
            let eval =
                move |_m: &Model<i64>, v: VesselIndex, b: BerthIndex, t: i64| -> Option<i64> {
                    let result = unsafe { f(model_ptr, v.get(), b.get(), t) };
                    if result < 0 { None } else { Some(result) }
                };
            dispatch_operator(
                engine,
                model,
                meta,
                monitor,
                operators,
                op_config,
                berths,
                start_times,
                objective_value,
                eval,
                callback,
                user_data,
            )
        }
        None => {
            let eval =
                move |m: &Model<i64>, v: VesselIndex, b: BerthIndex, t: i64| -> Option<i64> {
                    unsafe { calculate_weighted_turnaround_time_unchecked(m, v, b, t) }
                };
            dispatch_operator(
                engine,
                model,
                meta,
                monitor,
                operators,
                op_config,
                berths,
                start_times,
                objective_value,
                eval,
                callback,
                user_data,
            )
        }
    }
}

/// Dispatches on the compound operator strategy, instantiating the
/// concrete compound operator type and forwarding to [`execute`].
///
/// Generic over `H` (metaheuristic) and `F` (evaluator) so that each
/// call site is fully monomorphised — no trait-object indirection for
/// the compound operator or the evaluator.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dispatch_operator<H, F>(
    engine: &mut Engine<i64>,
    model: &Model<i64>,
    meta: &mut H,
    monitor: CompositeLocalSearchMonitor<'static, i64>,
    operators: Vec<Box<dyn LocalSearchOperator<i64>>>,
    op_config: &FfiOperatorConfig,
    berths: &[BerthIndex],
    start_times: &[i64],
    objective_value: i64,
    evaluator: F,
    callback: FfiNewBestCallbackFn,
    user_data: *mut c_void,
) -> FfiLocalSearchOutcome
where
    H: Metaheuristic<i64>,
    F: Fn(&Model<i64>, VesselIndex, BerthIndex, i64) -> Option<i64>,
{
    match op_config.strategy {
        FfiCompoundStrategy::RoundRobin => {
            let mut op = RoundRobinCompoundOperator::new(operators);
            execute(
                engine,
                model,
                meta,
                monitor,
                &mut op,
                berths,
                start_times,
                objective_value,
                evaluator,
                callback,
                user_data,
            )
        }
        FfiCompoundStrategy::Random => {
            let mut op =
                RandomCompoundOperator::new(operators, StdRng::seed_from_u64(op_config.seed));
            execute(
                engine,
                model,
                meta,
                monitor,
                &mut op,
                berths,
                start_times,
                objective_value,
                evaluator,
                callback,
                user_data,
            )
        }
        FfiCompoundStrategy::MultiArmedBandit => {
            let mut op = MultiArmedBanditCompoundOperator::new(
                operators,
                op_config.bandit_memory_coeff,
                op_config.bandit_exploration_coeff,
            );
            execute(
                engine,
                model,
                meta,
                monitor,
                &mut op,
                berths,
                start_times,
                objective_value,
                evaluator,
                callback,
                user_data,
            )
        }
    }
}

/// Fully-monomorphised engine execution core.
///
/// All type parameters (`H`, `O`, `F`) are concrete at each call site,
/// so the compiler generates a dedicated, branchless code-path for every
/// (metaheuristic × compound-operator × evaluator) combination.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn execute<H, O, F>(
    engine: &mut Engine<i64>,
    model: &Model<i64>,
    meta: &mut H,
    monitor: CompositeLocalSearchMonitor<'static, i64>,
    operator: &mut O,
    berths: &[BerthIndex],
    start_times: &[i64],
    objective_value: i64,
    evaluator: F,
    callback: FfiNewBestCallbackFn,
    user_data: *mut c_void,
) -> FfiLocalSearchOutcome
where
    H: Metaheuristic<i64>,
    O: LocalSearchOperator<i64>,
    F: Fn(&Model<i64>, VesselIndex, BerthIndex, i64) -> Option<i64>,
{
    let cb = move |view: SolutionView<'_, i64>| unsafe {
        let view_ptr: *const SolutionView<'static, i64> = std::ptr::from_ref(&view).cast();
        callback(view_ptr, user_data);
    };

    match LocalSearchParams::new(
        model,
        operator,
        meta,
        monitor,
        berths,
        start_times,
        objective_value,
    ) {
        Ok(params) => {
            let outcome = engine.run(params.into(), evaluator, cb);
            let (solution, reason, stats) = outcome.into_inner();
            FfiLocalSearchOutcome {
                solution: Box::into_raw(Box::new(solution)),
                termination_reason: FfiTerminationReason::from(reason),
                stats: FfiLocalSearchStatistics::from(&stats),
            }
        }
        Err(e) => panic!("called `run_inner`: LocalSearchParams validation failed: {e}"),
    }
}

// ----------------------------------------------------------------
// Tabu Search
// ----------------------------------------------------------------

/// Runs the engine with Tabu Search.
///
/// # Safety
///
/// * `engine` must be a valid pointer from `talos_engine_new`.
/// * `model` must be a valid pointer from `talos_model_new`.
/// * `initial_solution` must be a valid `SolutionView` pointer.
/// * All pointers must remain valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_engine_run_tabu(
    engine: *mut Engine<i64>,
    model: *const Model<i64>,
    config: FfiTabuSearchConfig,
    op_config: FfiOperatorConfig,
    mon_config: FfiMonitorConfig,
    initial_solution: *const SolutionView<'static, i64>,
    evaluator: FfiEvaluatorFn,
    callback: FfiNewBestCallbackFn,
    user_data: *mut c_void,
) -> FfiLocalSearchOutcome {
    let selection = config.selection.into_inner();
    let exhaustion = config.exhaustion_outcome.into_inner();

    let random_tenure = config.tenure_max > config.tenure_min;

    match (random_tenure, config.tie_breaking) {
        (false, FfiTieBreaking::KeepFirst) => {
            let mut ts = TabuSearch::new(
                FixedTenure::new(config.tenure_min),
                config.num_vessels,
                config.num_berths,
            )
            .with_tie_breaking(KeepFirst)
            .with_selection(selection)
            .with_exhaustion_outcome(exhaustion);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut ts,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        (false, FfiTieBreaking::KeepLast) => {
            let mut ts = TabuSearch::new(
                FixedTenure::new(config.tenure_min),
                config.num_vessels,
                config.num_berths,
            )
            .with_tie_breaking(KeepLast)
            .with_selection(selection)
            .with_exhaustion_outcome(exhaustion);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut ts,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        (false, FfiTieBreaking::Random) => {
            let mut ts = TabuSearch::new(
                FixedTenure::new(config.tenure_min),
                config.num_vessels,
                config.num_berths,
            )
            .with_tie_breaking(RandomTieBreak::new(StdRng::seed_from_u64(config.seed)))
            .with_selection(selection)
            .with_exhaustion_outcome(exhaustion);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut ts,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        (true, FfiTieBreaking::KeepFirst) => {
            let mut ts = TabuSearch::new(
                RandomTenure::new(
                    config.tenure_min,
                    config.tenure_max,
                    StdRng::seed_from_u64(config.seed),
                ),
                config.num_vessels,
                config.num_berths,
            )
            .with_tie_breaking(KeepFirst)
            .with_selection(selection)
            .with_exhaustion_outcome(exhaustion);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut ts,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        (true, FfiTieBreaking::KeepLast) => {
            let mut ts = TabuSearch::new(
                RandomTenure::new(
                    config.tenure_min,
                    config.tenure_max,
                    StdRng::seed_from_u64(config.seed),
                ),
                config.num_vessels,
                config.num_berths,
            )
            .with_tie_breaking(KeepLast)
            .with_selection(selection)
            .with_exhaustion_outcome(exhaustion);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut ts,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        (true, FfiTieBreaking::Random) => {
            // Use different seeds to avoid correlation between tenure RNG
            // and tie-breaking RNG.
            let mut ts = TabuSearch::new(
                RandomTenure::new(
                    config.tenure_min,
                    config.tenure_max,
                    StdRng::seed_from_u64(config.seed),
                ),
                config.num_vessels,
                config.num_berths,
            )
            .with_tie_breaking(RandomTieBreak::new(StdRng::seed_from_u64(
                config.seed.wrapping_add(1),
            )))
            .with_selection(selection)
            .with_exhaustion_outcome(exhaustion);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut ts,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
    }
}

// ----------------------------------------------------------------
// Simulated Annealing
// ----------------------------------------------------------------

/// Applies the common SA builder options (trigger, reheat, frozen
/// threshold) to a freshly constructed `SimulatedAnnealing`.
fn apply_sa_options<R, C>(
    sa: SimulatedAnnealing<R, C, MetropolisCriterion>,
    config: &FfiSimulatedAnnealingConfig,
) -> SimulatedAnnealing<R, C, MetropolisCriterion>
where
    R: rand::Rng,
    C: talos_ls::meta::sa::CoolingSchedule,
{
    let mut sa = sa.with_cooling_trigger(config.cooling_trigger.into_inner());
    if config.reheat_factor > 1.0 {
        sa = sa.with_reheat(config.reheat_factor);
    }
    if config.frozen_threshold > 0.0 {
        sa = sa.with_frozen_threshold(config.frozen_threshold);
    }
    sa
}

/// Runs the engine with Simulated Annealing.
///
/// # Safety
///
/// * `engine` must be a valid pointer from `talos_engine_new`.
/// * `model` must be a valid pointer from `talos_model_new`.
/// * `initial_solution` must be a valid `SolutionView` pointer.
/// * All pointers must remain valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_engine_run_sa(
    engine: *mut Engine<i64>,
    model: *const Model<i64>,
    config: FfiSimulatedAnnealingConfig,
    op_config: FfiOperatorConfig,
    mon_config: FfiMonitorConfig,
    initial_solution: *const SolutionView<'static, i64>,
    evaluator: FfiEvaluatorFn,
    callback: FfiNewBestCallbackFn,
    user_data: *mut c_void,
) -> FfiLocalSearchOutcome {
    let rng = StdRng::seed_from_u64(config.seed);

    match config.schedule {
        FfiCoolingSchedule::Geometric => {
            let cooling = GeometricCooling::new(
                config.initial_temperature,
                config.alpha,
                config.min_temperature,
            );
            let mut sa = apply_sa_options(SimulatedAnnealing::new(cooling, rng), &config);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut sa,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        FfiCoolingSchedule::GeometricHeuristic => {
            let cooling =
                SimulatedAnnealing::<StdRng, GeometricCooling>::heuristic_geometric_params(
                    config.heuristic_objective,
                    config.heuristic_acceptance_prob,
                    config.heuristic_sensitivity,
                    config.heuristic_cooling_rate,
                );
            let mut sa = apply_sa_options(SimulatedAnnealing::new(cooling, rng), &config);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut sa,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        FfiCoolingSchedule::Linear => {
            let cooling = LinearCooling::new(
                config.initial_temperature,
                config.linear_decrement,
                config.min_temperature,
            );
            let mut sa = apply_sa_options(SimulatedAnnealing::new(cooling, rng), &config);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut sa,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        FfiCoolingSchedule::LinearBudget => {
            let cooling = LinearCooling::from_budget(
                config.initial_temperature,
                config.min_temperature,
                config.linear_budget_iterations,
            );
            let mut sa = apply_sa_options(SimulatedAnnealing::new(cooling, rng), &config);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut sa,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        FfiCoolingSchedule::Logarithmic => {
            let cooling = LogarithmicCooling::new(
                config.log_constant,
                config.log_k_scale,
                config.min_temperature,
            );
            let mut sa = apply_sa_options(SimulatedAnnealing::new(cooling, rng), &config);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut sa,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
    }
}

// ----------------------------------------------------------------
// Guided Local Search
// ----------------------------------------------------------------

/// Runs the engine with Guided Local Search.
///
/// # Safety
///
/// * `engine` must be a valid pointer from `talos_engine_new`.
/// * `model` must be a valid pointer from `talos_model_new`.
/// * `initial_solution` must be a valid `SolutionView` pointer.
/// * All pointers must remain valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_engine_run_gls(
    engine: *mut Engine<i64>,
    model: *const Model<i64>,
    config: FfiGuidedLocalSearchConfig,
    op_config: FfiOperatorConfig,
    mon_config: FfiMonitorConfig,
    initial_solution: *const SolutionView<'static, i64>,
    evaluator: FfiEvaluatorFn,
    callback: FfiNewBestCallbackFn,
    user_data: *mut c_void,
) -> FfiLocalSearchOutcome {
    let trigger = match config.penalization_trigger {
        FfiPenalizationTrigger::AfterNonImprovements => {
            PenalizationTrigger::AfterNonImprovements(config.trigger_n)
        }
        FfiPenalizationTrigger::AfterMoves => PenalizationTrigger::AfterMoves(config.trigger_n),
        _ => PenalizationTrigger::OnExhaustion,
    };

    if config.reactive != 0 {
        let reactive = ReactiveLambda::new(
            config.lambda,
            config.growth_factor,
            config.decay_factor,
            config.min_lambda,
            config.max_lambda,
        );
        let mut gls = GuidedLocalSearch::new(config.lambda, config.num_vessels, config.num_berths)
            .with_lambda(reactive)
            .with_trigger(trigger);
        unsafe {
            run_inner(
                engine,
                model,
                &mut gls,
                &op_config,
                &mon_config,
                initial_solution,
                evaluator,
                callback,
                user_data,
            )
        }
    } else {
        let mut gls = GuidedLocalSearch::new(config.lambda, config.num_vessels, config.num_berths)
            .with_trigger(trigger);
        unsafe {
            run_inner(
                engine,
                model,
                &mut gls,
                &op_config,
                &mon_config,
                initial_solution,
                evaluator,
                callback,
                user_data,
            )
        }
    }
}

// ----------------------------------------------------------------
// Greedy Descent
// ----------------------------------------------------------------

/// Runs the engine with Greedy Descent.
///
/// # Safety
///
/// * `engine` must be a valid pointer from `talos_engine_new`.
/// * `model` must be a valid pointer from `talos_model_new`.
/// * `initial_solution` must be a valid `SolutionView` pointer.
/// * All pointers must remain valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_engine_run_gd(
    engine: *mut Engine<i64>,
    model: *const Model<i64>,
    config: FfiGreedyDescentConfig,
    op_config: FfiOperatorConfig,
    mon_config: FfiMonitorConfig,
    initial_solution: *const SolutionView<'static, i64>,
    evaluator: FfiEvaluatorFn,
    callback: FfiNewBestCallbackFn,
    user_data: *mut c_void,
) -> FfiLocalSearchOutcome {
    let selection = config.selection.into_inner();

    match config.tie_breaking {
        FfiTieBreaking::KeepFirst => {
            let mut gd = GreedyDescent::new()
                .with_selection(selection)
                .with_tie_breaking(KeepFirst);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut gd,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        FfiTieBreaking::KeepLast => {
            let mut gd = GreedyDescent::new()
                .with_selection(selection)
                .with_tie_breaking(KeepLast);
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut gd,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        FfiTieBreaking::Random => {
            let mut gd = GreedyDescent::new()
                .with_selection(selection)
                .with_tie_breaking(RandomTieBreak::new(StdRng::seed_from_u64(config.seed)));
            unsafe {
                run_inner(
                    engine,
                    model,
                    &mut gd,
                    &op_config,
                    &mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
    }
}

// ----------------------------------------------------------------
// Unified entry point
// ----------------------------------------------------------------

/// Flat configuration for the unified `talos_engine_run` entry point.
///
/// Set `metaheuristic` to select the algorithm. Only the fields for the
/// selected algorithm are read. This is a convenience wrapper — prefer
/// the typed `talos_engine_run_tabu` / `_sa` / `_gls` / `_gd` functions
/// for full parameter coverage.
///
/// ## Tabu Search fields
///
/// `tabu_tenure_min`, `tabu_tenure_max`, `tabu_selection`,
/// `tabu_tie_breaking`, `tabu_exhaustion_outcome`, `num_vessels`,
/// `num_berths`, `seed`.
///
/// ## Simulated Annealing fields
///
/// `sa_schedule`, `sa_initial_temperature`, `sa_alpha`,
/// `sa_linear_decrement`, `sa_linear_budget_iterations`,
/// `sa_min_temperature`, `sa_log_constant`, `sa_log_k_scale`,
/// `sa_heuristic_objective`, `sa_heuristic_acceptance_prob`,
/// `sa_heuristic_sensitivity`, `sa_heuristic_cooling_rate`,
/// `sa_reheat_factor`, `sa_frozen_threshold`, `sa_cooling_trigger`,
/// `seed`.
///
/// ## Guided Local Search fields
///
/// `gls_lambda`, `gls_reactive`, `gls_penalization_trigger`,
/// `gls_trigger_n`, `gls_growth_factor`, `gls_decay_factor`,
/// `gls_min_lambda`, `gls_max_lambda`, `num_vessels`, `num_berths`.
///
/// ## Greedy Descent fields
///
/// `gd_selection`, `gd_tie_breaking`, `seed`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMetaheuristicConfig {
    pub metaheuristic: FfiMetaheuristic,
    pub seed: u64,
    pub num_vessels: usize,
    pub num_berths: usize,

    // Tabu Search
    pub tabu_tenure_min: u64,
    pub tabu_tenure_max: u64,
    pub tabu_selection: FfiSelectionStrategy,
    pub tabu_tie_breaking: FfiTieBreaking,
    pub tabu_exhaustion_outcome: FfiExhaustionOutcome,

    // Simulated Annealing
    pub sa_schedule: FfiCoolingSchedule,
    pub sa_initial_temperature: f64,
    pub sa_alpha: f64,
    pub sa_linear_decrement: f64,
    pub sa_linear_budget_iterations: u64,
    pub sa_min_temperature: f64,
    pub sa_log_constant: f64,
    pub sa_log_k_scale: f64,
    pub sa_reheat_factor: f64,
    pub sa_frozen_threshold: f64,
    pub sa_cooling_trigger: FfiCoolingTrigger,
    pub sa_heuristic_objective: f64,
    pub sa_heuristic_acceptance_prob: f64,
    pub sa_heuristic_sensitivity: f64,
    pub sa_heuristic_cooling_rate: f64,

    // Guided Local Search
    pub gls_lambda: f64,
    pub gls_reactive: i32,
    pub gls_penalization_trigger: FfiPenalizationTrigger,
    pub gls_trigger_n: u64,
    pub gls_growth_factor: f64,
    pub gls_decay_factor: f64,
    pub gls_min_lambda: f64,
    pub gls_max_lambda: f64,

    // Greedy Descent
    pub gd_selection: FfiSelectionStrategy,
    pub gd_tie_breaking: FfiTieBreaking,
}

/// Runs the local search engine with the unified configuration struct.
///
/// Dispatches to the appropriate per-algorithm entry point based on
/// `config.metaheuristic`.
///
/// # Safety
///
/// * `engine` must be a valid pointer from `talos_engine_new`.
/// * `model` must be a valid pointer from `talos_model_new`.
/// * `initial_solution` must be a valid `SolutionView` pointer.
/// * All pointers must remain valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_engine_run(
    engine: *mut Engine<i64>,
    model: *const Model<i64>,
    config: FfiMetaheuristicConfig,
    op_config: FfiOperatorConfig,
    mon_config: FfiMonitorConfig,
    initial_solution: *const SolutionView<'static, i64>,
    evaluator: FfiEvaluatorFn,
    callback: FfiNewBestCallbackFn,
    user_data: *mut c_void,
) -> FfiLocalSearchOutcome {
    match config.metaheuristic {
        FfiMetaheuristic::TabuSearch => {
            let tabu_config = FfiTabuSearchConfig {
                num_vessels: config.num_vessels,
                num_berths: config.num_berths,
                tenure_min: config.tabu_tenure_min,
                tenure_max: config.tabu_tenure_max,
                selection: config.tabu_selection,
                tie_breaking: config.tabu_tie_breaking,
                exhaustion_outcome: config.tabu_exhaustion_outcome,
                seed: config.seed,
            };
            unsafe {
                talos_engine_run_tabu(
                    engine,
                    model,
                    tabu_config,
                    op_config,
                    mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        FfiMetaheuristic::SimulatedAnnealing => {
            let sa_config = FfiSimulatedAnnealingConfig {
                schedule: config.sa_schedule,
                cooling_trigger: config.sa_cooling_trigger,
                seed: config.seed,
                initial_temperature: config.sa_initial_temperature,
                alpha: config.sa_alpha,
                linear_decrement: config.sa_linear_decrement,
                linear_budget_iterations: config.sa_linear_budget_iterations,
                min_temperature: config.sa_min_temperature,
                log_constant: config.sa_log_constant,
                log_k_scale: config.sa_log_k_scale,
                heuristic_objective: config.sa_heuristic_objective,
                heuristic_acceptance_prob: config.sa_heuristic_acceptance_prob,
                heuristic_sensitivity: config.sa_heuristic_sensitivity,
                heuristic_cooling_rate: config.sa_heuristic_cooling_rate,
                reheat_factor: config.sa_reheat_factor,
                frozen_threshold: config.sa_frozen_threshold,
            };
            unsafe {
                talos_engine_run_sa(
                    engine,
                    model,
                    sa_config,
                    op_config,
                    mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        FfiMetaheuristic::GuidedLocalSearch => {
            let gls_config = FfiGuidedLocalSearchConfig {
                num_vessels: config.num_vessels,
                num_berths: config.num_berths,
                lambda: config.gls_lambda,
                reactive: config.gls_reactive,
                penalization_trigger: config.gls_penalization_trigger,
                trigger_n: config.gls_trigger_n,
                growth_factor: config.gls_growth_factor,
                decay_factor: config.gls_decay_factor,
                min_lambda: config.gls_min_lambda,
                max_lambda: config.gls_max_lambda,
            };
            unsafe {
                talos_engine_run_gls(
                    engine,
                    model,
                    gls_config,
                    op_config,
                    mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
        FfiMetaheuristic::GreedyDescent => {
            let gd_config = FfiGreedyDescentConfig {
                selection: config.gd_selection,
                tie_breaking: config.gd_tie_breaking,
                seed: config.seed,
            };
            unsafe {
                talos_engine_run_gd(
                    engine,
                    model,
                    gd_config,
                    op_config,
                    mon_config,
                    initial_solution,
                    evaluator,
                    callback,
                    user_data,
                )
            }
        }
    }
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffimodel::{
        model::{FfiClosedOpenIntervalI64, talos_model_free, talos_model_new},
        solution::{talos_solution_free, talos_solution_view_free, talos_solution_view_new},
    };
    use std::ptr;

    /// Builds a minimal 2-vessel / 2-berth model via the FFI.
    /// V0: arrival=0, deadline=100, weight=1, p(B0)=5, p(B1)=5
    /// V1: arrival=0, deadline=100, weight=1, p(B0)=10, p(B1)=10
    /// Both berths open [0, 200).
    fn build_test_model() -> *mut Model<i64> {
        let arrivals: [i64; 2] = [0, 0];
        let deadlines: [i64; 2] = [100, 100];
        let weights: [i64; 2] = [1, 1];
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

    /// No-op callback for tests.
    unsafe extern "C" fn noop_callback(
        _view: *const SolutionView<'static, i64>,
        _user_data: *mut c_void,
    ) {
    }

    fn default_op_config() -> FfiOperatorConfig {
        FfiOperatorConfig {
            strategy: FfiCompoundStrategy::Random,
            use_intra_berth_swap: 1,
            use_inter_berth_swap: 1,
            use_intra_berth_shift: 0,
            use_inter_berth_shift: 0,
            seed: 42,
            bandit_memory_coeff: 0.0,
            bandit_exploration_coeff: 0.0,
        }
    }

    fn default_mon_config() -> FfiMonitorConfig {
        FfiMonitorConfig {
            time_limit_millis: 0,
            iteration_limit: 100,
            solution_limit: 0,
            cycle_limit: 0,
            no_improvement_iterations: 0,
            no_improvement_cycles: 0,
            no_improvement_time_millis: 0,
        }
    }

    /// Helper: create an initial solution view, run the callback, clean up.
    unsafe fn run_and_cleanup(
        engine: *mut Engine<i64>,
        model: *mut Model<i64>,
        run_fn: impl FnOnce(
            *mut Engine<i64>,
            *const Model<i64>,
            *const SolutionView<'static, i64>,
        ) -> FfiLocalSearchOutcome,
    ) {
        let berths: [usize; 2] = [0, 1];
        let starts: [i64; 2] = [0, 0];
        let init_sol =
            unsafe { talos_solution_view_new(2, berths.as_ptr(), starts.as_ptr(), i64::MAX) };
        assert!(!init_sol.is_null());

        let outcome = run_fn(engine, model as *const _, init_sol);
        assert!(!outcome.solution.is_null());
        assert!(outcome.stats.iterations > 0);

        unsafe {
            talos_solution_free(outcome.solution);
            talos_solution_view_free(init_sol);
            talos_engine_free(engine);
            talos_model_free(model);
        }
    }

    // ── Engine lifecycle ─────────────────────────────────────

    #[test]
    fn test_engine_lifecycle() {
        let e = talos_engine_new(4, 2);
        assert!(!e.is_null());
        unsafe { talos_engine_free(e) };
    }

    // ── Operator config ──────────────────────────────────────

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
        assert_eq!(build_operators(&cfg).len(), 4);
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
        assert_eq!(build_operators(&cfg).len(), 0);
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
        assert_eq!(build_operators(&cfg).len(), 2);
    }

    // ── Monitor config ───────────────────────────────────────

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
        let _mon = build_monitor(&cfg);
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
        let _mon = build_monitor(&cfg);
    }

    // ── Tabu Search ──────────────────────────────────────────

    #[test]
    fn test_run_tabu_fixed_keep_first() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiTabuSearchConfig {
            num_vessels: 2,
            num_berths: 2,
            tenure_min: 5,
            tenure_max: 5,
            selection: FfiSelectionStrategy::BestImprovement,
            tie_breaking: FfiTieBreaking::KeepFirst,
            exhaustion_outcome: FfiExhaustionOutcome::Restart,
            seed: 42,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_tabu(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_tabu_random_keep_last() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiTabuSearchConfig {
            num_vessels: 2,
            num_berths: 2,
            tenure_min: 3,
            tenure_max: 10,
            selection: FfiSelectionStrategy::FirstImprovement,
            tie_breaking: FfiTieBreaking::KeepLast,
            exhaustion_outcome: FfiExhaustionOutcome::Terminate,
            seed: 42,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_tabu(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_tabu_random_tie_break() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiTabuSearchConfig {
            num_vessels: 2,
            num_berths: 2,
            tenure_min: 5,
            tenure_max: 15,
            selection: FfiSelectionStrategy::BestImprovement,
            tie_breaking: FfiTieBreaking::Random,
            exhaustion_outcome: FfiExhaustionOutcome::Restart,
            seed: 123,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_tabu(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    // ── Simulated Annealing ──────────────────────────────────

    #[test]
    fn test_run_sa_geometric() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiSimulatedAnnealingConfig {
            schedule: FfiCoolingSchedule::Geometric,
            cooling_trigger: FfiCoolingTrigger::Iteration,
            seed: 42,
            initial_temperature: 100.0,
            alpha: 0.99,
            linear_decrement: 0.0,
            linear_budget_iterations: 0,
            min_temperature: 0.001,
            log_constant: 0.0,
            log_k_scale: 0.0,
            heuristic_objective: 0.0,
            heuristic_acceptance_prob: 0.0,
            heuristic_sensitivity: 0.0,
            heuristic_cooling_rate: 0.0,
            reheat_factor: 0.0,
            frozen_threshold: 0.0,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_sa(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_sa_linear_with_reheat() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiSimulatedAnnealingConfig {
            schedule: FfiCoolingSchedule::Linear,
            cooling_trigger: FfiCoolingTrigger::Cycle,
            seed: 42,
            initial_temperature: 100.0,
            alpha: 0.0,
            linear_decrement: 0.5,
            linear_budget_iterations: 0,
            min_temperature: 0.01,
            log_constant: 0.0,
            log_k_scale: 0.0,
            heuristic_objective: 0.0,
            heuristic_acceptance_prob: 0.0,
            heuristic_sensitivity: 0.0,
            heuristic_cooling_rate: 0.0,
            reheat_factor: 2.0,
            frozen_threshold: 0.0,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_sa(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_sa_linear_budget() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiSimulatedAnnealingConfig {
            schedule: FfiCoolingSchedule::LinearBudget,
            cooling_trigger: FfiCoolingTrigger::Iteration,
            seed: 42,
            initial_temperature: 100.0,
            alpha: 0.0,
            linear_decrement: 0.0,
            linear_budget_iterations: 200,
            min_temperature: 0.01,
            log_constant: 0.0,
            log_k_scale: 0.0,
            heuristic_objective: 0.0,
            heuristic_acceptance_prob: 0.0,
            heuristic_sensitivity: 0.0,
            heuristic_cooling_rate: 0.0,
            reheat_factor: 0.0,
            frozen_threshold: 0.0,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_sa(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_sa_logarithmic() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiSimulatedAnnealingConfig {
            schedule: FfiCoolingSchedule::Logarithmic,
            cooling_trigger: FfiCoolingTrigger::Iteration,
            seed: 42,
            initial_temperature: 0.0,
            alpha: 0.0,
            linear_decrement: 0.0,
            linear_budget_iterations: 0,
            min_temperature: 0.01,
            log_constant: 100.0,
            log_k_scale: 1.0,
            heuristic_objective: 0.0,
            heuristic_acceptance_prob: 0.0,
            heuristic_sensitivity: 0.0,
            heuristic_cooling_rate: 0.0,
            reheat_factor: 0.0,
            frozen_threshold: 0.0,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_sa(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_sa_heuristic_geometric() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiSimulatedAnnealingConfig {
            schedule: FfiCoolingSchedule::GeometricHeuristic,
            cooling_trigger: FfiCoolingTrigger::Iteration,
            seed: 42,
            initial_temperature: 0.0,
            alpha: 0.0,
            linear_decrement: 0.0,
            linear_budget_iterations: 0,
            min_temperature: 0.0,
            log_constant: 0.0,
            log_k_scale: 0.0,
            heuristic_objective: 1000.0,
            heuristic_acceptance_prob: 0.5,
            heuristic_sensitivity: 0.01,
            heuristic_cooling_rate: 0.999,
            reheat_factor: 0.0,
            frozen_threshold: 0.0,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_sa(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_sa_with_frozen_threshold() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiSimulatedAnnealingConfig {
            schedule: FfiCoolingSchedule::Geometric,
            cooling_trigger: FfiCoolingTrigger::Acceptance,
            seed: 42,
            initial_temperature: 50.0,
            alpha: 0.95,
            linear_decrement: 0.0,
            linear_budget_iterations: 0,
            min_temperature: 0.001,
            log_constant: 0.0,
            log_k_scale: 0.0,
            heuristic_objective: 0.0,
            heuristic_acceptance_prob: 0.0,
            heuristic_sensitivity: 0.0,
            heuristic_cooling_rate: 0.0,
            reheat_factor: 0.0,
            frozen_threshold: 0.5,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_sa(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    // ── Guided Local Search ──────────────────────────────────

    #[test]
    fn test_run_gls_fixed() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiGuidedLocalSearchConfig {
            num_vessels: 2,
            num_berths: 2,
            lambda: 0.5,
            reactive: 0,
            penalization_trigger: FfiPenalizationTrigger::OnExhaustion,
            trigger_n: 0,
            growth_factor: 0.0,
            decay_factor: 0.0,
            min_lambda: 0.0,
            max_lambda: 0.0,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_gls(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_gls_reactive() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiGuidedLocalSearchConfig {
            num_vessels: 2,
            num_berths: 2,
            lambda: 0.5,
            reactive: 1,
            penalization_trigger: FfiPenalizationTrigger::AfterMoves,
            trigger_n: 10,
            growth_factor: 1.2,
            decay_factor: 0.8,
            min_lambda: 0.1,
            max_lambda: 10.0,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_gls(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_gls_after_non_improvements() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiGuidedLocalSearchConfig {
            num_vessels: 2,
            num_berths: 2,
            lambda: 1.0,
            reactive: 0,
            penalization_trigger: FfiPenalizationTrigger::AfterNonImprovements,
            trigger_n: 20,
            growth_factor: 0.0,
            decay_factor: 0.0,
            min_lambda: 0.0,
            max_lambda: 0.0,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_gls(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    // ── Greedy Descent ───────────────────────────────────────

    #[test]
    fn test_run_gd_first_improvement() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiGreedyDescentConfig {
            selection: FfiSelectionStrategy::FirstImprovement,
            tie_breaking: FfiTieBreaking::KeepFirst,
            seed: 0,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_gd(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_gd_best_improvement_keep_last() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiGreedyDescentConfig {
            selection: FfiSelectionStrategy::BestImprovement,
            tie_breaking: FfiTieBreaking::KeepLast,
            seed: 0,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_gd(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_run_gd_random_tie_break() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let config = FfiGreedyDescentConfig {
            selection: FfiSelectionStrategy::BestImprovement,
            tie_breaking: FfiTieBreaking::Random,
            seed: 99,
        };
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run_gd(
                    e,
                    m,
                    config,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    // ── Unified entry point ──────────────────────────────────

    fn zeroed_meta_config(meta: FfiMetaheuristic) -> FfiMetaheuristicConfig {
        FfiMetaheuristicConfig {
            metaheuristic: meta,
            seed: 0,
            num_vessels: 2,
            num_berths: 2,
            tabu_tenure_min: 0,
            tabu_tenure_max: 0,
            tabu_selection: FfiSelectionStrategy::BestImprovement,
            tabu_tie_breaking: FfiTieBreaking::KeepFirst,
            tabu_exhaustion_outcome: FfiExhaustionOutcome::Restart,
            sa_schedule: FfiCoolingSchedule::Geometric,
            sa_initial_temperature: 0.0,
            sa_alpha: 0.0,
            sa_linear_decrement: 0.0,
            sa_linear_budget_iterations: 0,
            sa_min_temperature: 0.0,
            sa_log_constant: 0.0,
            sa_log_k_scale: 0.0,
            sa_reheat_factor: 0.0,
            sa_frozen_threshold: 0.0,
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
            gd_selection: FfiSelectionStrategy::FirstImprovement,
            gd_tie_breaking: FfiTieBreaking::KeepFirst,
        }
    }

    #[test]
    fn test_unified_tabu() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let mut cfg = zeroed_meta_config(FfiMetaheuristic::TabuSearch);
        cfg.tabu_tenure_min = 5;
        cfg.tabu_tenure_max = 5;
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run(
                    e,
                    m,
                    cfg,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_unified_sa() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let mut cfg = zeroed_meta_config(FfiMetaheuristic::SimulatedAnnealing);
        cfg.sa_initial_temperature = 100.0;
        cfg.sa_alpha = 0.99;
        cfg.sa_min_temperature = 0.001;
        cfg.seed = 42;
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run(
                    e,
                    m,
                    cfg,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_unified_gls() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let mut cfg = zeroed_meta_config(FfiMetaheuristic::GuidedLocalSearch);
        cfg.gls_lambda = 0.5;
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run(
                    e,
                    m,
                    cfg,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    #[test]
    fn test_unified_gd() {
        let model = build_test_model();
        let engine = talos_engine_new(2, 2);
        let cfg = zeroed_meta_config(FfiMetaheuristic::GreedyDescent);
        unsafe {
            run_and_cleanup(engine, model, |e, m, s| {
                talos_engine_run(
                    e,
                    m,
                    cfg,
                    default_op_config(),
                    default_mon_config(),
                    s,
                    None,
                    noop_callback,
                    ptr::null_mut(),
                )
            });
        }
    }

    // ── Callback test ────────────────────────────────────────

    #[test]
    fn test_engine_run_with_callback() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let model = build_test_model();
        let engine = talos_engine_new(2, 2);

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

        let mut cfg = zeroed_meta_config(FfiMetaheuristic::TabuSearch);
        cfg.tabu_tenure_min = 5;
        cfg.tabu_tenure_max = 5;

        let outcome = unsafe {
            talos_engine_run(
                engine,
                model as *const _,
                cfg,
                default_op_config(),
                FfiMonitorConfig {
                    iteration_limit: 200,
                    ..default_mon_config()
                },
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
            talos_model_free(model);
        }
    }
}
